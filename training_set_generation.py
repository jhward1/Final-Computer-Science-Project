import asyncio
import json
import time
import tiktoken
import pandas as pd
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
from tqdm.asyncio import tqdm

load_dotenv()


class RateLimiter:
    """Sliding-window rate limiter: allows at most max_calls per period seconds."""

    def __init__(self, max_calls: int, period: float = 60.0):
        self.max_calls = max_calls
        self.period = period
        self._timestamps: list[float] = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self.period]
            if len(self._timestamps) >= self.max_calls:
                sleep_for = self.period - (now - self._timestamps[0])
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                now = time.monotonic()
                self._timestamps = [t for t in self._timestamps if now - t < self.period]
            self._timestamps.append(time.monotonic())


_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_message_tokens(messages: list[dict]) -> int:
    """Count tokens in a list of chat messages (4 overhead tokens per message + 2 reply priming)."""
    return sum(len(_ENCODING.encode(m.get("content", ""))) + 4 for m in messages) + 2


class TokenRateLimiter:
    """
    Sliding-window token rate limiter.

    acquire(input_tokens) reserves capacity before a request.
    release(input_tokens, actual_total) records the real token count afterward,
    replacing the reservation so future windows use accurate data.
    """

    def __init__(self, max_tokens: int, period: float = 60.0, completion_buffer: int = 600):
        self.max_tokens = max_tokens
        self.period = period
        self.completion_buffer = completion_buffer
        self._completed: list[tuple[float, int]] = []  # (timestamp, actual_total)
        self._inflight_input: int = 0
        self._lock = asyncio.Lock()

    async def acquire(self, input_tokens: int, completion_buffer: int | None = None) -> None:
        buffer = completion_buffer if completion_buffer is not None else self.completion_buffer
        async with self._lock:
            while True:
                now = time.monotonic()
                self._completed = [(t, n) for t, n in self._completed if now - t < self.period]
                used = sum(n for _, n in self._completed)
                projected = used + self._inflight_input + input_tokens + buffer
                if projected <= self.max_tokens:
                    self._inflight_input += input_tokens
                    return
                wait = (self.period - (now - self._completed[0][0]) + 0.5) if self._completed else 5.0
                await asyncio.sleep(max(wait, 1.0))

    def release(self, input_tokens: int, actual_total: int) -> None:
        self._inflight_input -= input_tokens
        self._completed.append((time.monotonic(), actual_total))


RATE_LIMITER = RateLimiter(max_calls=28)       # stay under 30 req/min
TOKEN_LIMITER = TokenRateLimiter(max_tokens=5500)  # stay under 6000 TPM

CLIENT = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=3,
)
MODEL = "llama-3.1-8b-instant"

OUTPUT_FILE = "training_set.csv"
N_QUESTIONS = 50

# ── Profile distribution ───────────────────────────────────────────────────────
# 50 responses balanced across: certainty 1-5 (10 each), 3 frameworks (~17 each),
# elite networks mentioned (25 yes / 25 no).

FRAMEWORKS = ["Geopolitical", "Sociological", "Economic Protectionism"]
SECONDARY_FRAMEWORKS = [None] + FRAMEWORKS  # Allow for null secondary framework 

def build_profiles(n=N_QUESTIONS):
    return [
        {
            "certainty_score": (i % 5) + 1,      # cycles 1-2-3-4-5
            "framework": FRAMEWORKS[i % 3],       # cycles across 3 frameworks
            "secondary": SECONDARY_FRAMEWORKS[i % 4],       # cycles across 3 frameworks + null
            "elite_networks": i % 2 == 0,         # alternates True/False
        }
        for i in range(n)
    ]

# ── Steering instructions (mirror the rubric in llm_judge.py exactly) ─────────

CERTAINTY_INSTRUCTIONS = {
    1: (
        "Express significant uncertainty throughout. Use hedging language like "
        "'it is possible that', 'some argue', 'remains unclear', 'there is debate about'. "
        "Acknowledge that the situation is contested and that you are genuinely unsure."
    ),
    2: (
        "Express mild uncertainty. Lean toward one explanation but acknowledge other "
        "possibilities. Use occasional hedging such as 'likely', 'tends to', 'may reflect'."
    ),
    3: (
        "Express moderate confidence. Present a main argument clearly while briefly "
        "noting that alternative explanations exist."
    ),
    4: (
        "Express fairly high confidence. State clear conclusions with only minor "
        "qualifications. Avoid heavy hedging."
    ),
    5: (
        "Express very high certainty. Use assertive, definitive language. "
        "Present your explanation as the established consensus without qualification."
    ),
}

FRAMEWORK_INSTRUCTIONS = {
    "Geopolitical": (
        "Include in your answer references to state-level national interests, security concerns, "
        "great power competition, and geoeconomic strategy. Do not focus on trade barriers "
        "as an end in themselves or on informal elite dynamics."
    ),
    "Sociological": (
        "Include in your answer references to elite networks, rent-seeking behavior, personalist "
        "power structures, and informal influence. Focus on who benefits and why, not on "
        "formal state strategy or trade mechanics."
    ),
    "Economic Protectionism": (
        "Include in your answer references to trade barriers, tariffs, protection of domestic "
        "industries, and narrow economic self-interest. Focus on the economic mechanics, "
        "not on security rationales or elite networks."
    ),
}

SECONDARY_FRAMEWORK_INSTRUCTIONS = {
    "Geopolitical": (
        "You may also include some discussion of economic protectionism or elite networks/rent-seeking behavior, but this should not be the main focus."
    ),
    "Sociological": (
        "You may also include some discussion of geopolitical strategy or economic protectionism, but this should not be the main focus."
    ),
    "Economic Protectionism": (
        "You may also include some discussion of geopolitical strategy or elite networks, but this should not be the main focus."
    ),
}

ELITE_NETWORK_INSTRUCTIONS = {
    True: (
        "Explicitly mention oligarchs, ruling-class interests, or "
        "personalist power structures somewhere in your response."
        "But, do not explicitly use the phrase 'elite networks' in your answer. Instead, use more natural language to describe these dynamics."
    ),
    False: "Do not mention elite networks, oligarchs, or personalist power structures.",
}

def build_system_prompt(profile):
    return (
        "You are a scholar of international affairs writing an analytical response. "
        "Follow ALL of the instructions below precisely.\n\n"
        f"FRAMING: {FRAMEWORK_INSTRUCTIONS[profile['framework']]}\n\n"
        f"CERTAINTY: {CERTAINTY_INSTRUCTIONS[profile['certainty_score']]}\n\n"
        f"ELITE NETWORKS: {ELITE_NETWORK_INSTRUCTIONS[profile['elite_networks']]}\n\n"
        f"SECONDARY FRAMEWORK: {SECONDARY_FRAMEWORK_INSTRUCTIONS[profile['secondary']] if profile['secondary'] else 'None'}\n\n"
        "Write 2-3 substantive paragraphs. Do not explain your framing choices."
    )

# ── Generation ─────────────────────────────────────────────────────────────────

async def generate_questions():
    prompt = (
        f"Generate exactly {N_QUESTIONS} diverse, substantive questions about "
        "international affairs, geopolitics, and international economic law.\n\n"
        "Cover a wide range of topics: trade policy, sanctions, security alliances, "
        "great power competition, regional conflicts, multilateral institutions, "
        "technology competition, energy geopolitics, and sovereign debt.\n\n"
        f"Return ONLY a JSON array of exactly {N_QUESTIONS} question strings. "
        "No preamble, no explanation, no markdown — just the array."
    )
    messages = [{"role": "user", "content": prompt}]
    input_tokens = count_message_tokens(messages)
    await RATE_LIMITER.acquire()
    await TOKEN_LIMITER.acquire(input_tokens, completion_buffer=1500)  # questions response is large
    response = await CLIENT.chat.completions.create(model=MODEL, messages=messages)
    actual_total = response.usage.total_tokens if response.usage else input_tokens + 1500
    TOKEN_LIMITER.release(input_tokens, actual_total)
    content = response.choices[0].message.content
    start = content.index('[')
    end = content.rindex(']') + 1
    return json.loads(content[start:end])


async def generate_response(question, profile):
    messages = [
        {"role": "system", "content": build_system_prompt(profile)},
        {"role": "user", "content": question},
    ]
    input_tokens = count_message_tokens(messages)
    await RATE_LIMITER.acquire()
    await TOKEN_LIMITER.acquire(input_tokens)
    try:
        response = await CLIENT.chat.completions.create(model=MODEL, messages=messages)
        actual_total = response.usage.total_tokens if response.usage else input_tokens + 600
        TOKEN_LIMITER.release(input_tokens, actual_total)
        return response.choices[0].message.content
    except Exception as e:
        TOKEN_LIMITER.release(input_tokens, input_tokens)  # free the reservation on error
        return f"ERROR: {str(e)}"


async def main():
    print("Step 1/2 — Generating questions...")
    questions = await generate_questions()
    print(f"  {len(questions)} questions generated.")

    profiles = build_profiles(len(questions))

    print("Step 2/2 — Generating responses...")
    semaphore = asyncio.Semaphore(5)

    async def sem_task(question, profile):
        async with semaphore:
            answer = await generate_response(question, profile)
            return {
                "original_prompt": question,
                "model": MODEL,
                "answer": answer,
                # Target columns kept for post-grading validation; not shown in grading UI
                "target_certainty_score": profile["certainty_score"],
                "target_framework": profile["framework"],
                "target_secondary_framework": profile["secondary"],
                "target_elite_networks": profile["elite_networks"],
            }

    results = await tqdm.gather(*[sem_task(q, p) for q, p in zip(questions, profiles)])

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved {len(df)} rows to {OUTPUT_FILE}")
    print("\nTarget distribution:")
    print(df[["target_certainty_score", "target_framework", "target_secondary_framework", "target_elite_networks"]].value_counts().sort_index())


if __name__ == "__main__":
    asyncio.run(main())
