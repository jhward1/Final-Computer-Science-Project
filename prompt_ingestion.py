from openai import AsyncOpenAI
import argparse
import asyncio
import json
import sys
import time
import tiktoken
import pandas as pd
import requests
from dataclasses import dataclass, field
from dotenv import load_dotenv
import os
from tqdm.asyncio import tqdm # progress bar

# This line bypasses the need for the "useEnvFile" setting
load_dotenv()

FILE_PATH = 'prompts.csv' # This file should have a column named "Prompt" with the prompts to test


# ── Rate limiters ──────────────────────────────────────────────────────────────

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


# ── Model configuration ────────────────────────────────────────────────────────

@dataclass
class ModelConfig:
    """
    Bundles a model's identity and rate limits.

    requests_per_minute: max API calls per minute for this model.
    tokens_per_minute:   max tokens per minute; set to None to disable token limiting
                         (e.g. OpenRouter free tier does not enforce token limits).
    """
    provider: str
    model: str
    requests_per_minute: int
    tokens_per_minute: int | None = None
    # Per-instance rate limiters — created automatically, not passed in
    req_limiter: RateLimiter = field(init=False, repr=False)
    tok_limiter: "TokenRateLimiter | None" = field(init=False, repr=False)

    def __post_init__(self):
        self.req_limiter = RateLimiter(max_calls=self.requests_per_minute)
        self.tok_limiter = TokenRateLimiter(max_tokens=self.tokens_per_minute) if self.tokens_per_minute else None


_CLIENTS: dict[str, AsyncOpenAI] | None = None


def get_clients() -> dict[str, AsyncOpenAI]:
    global _CLIENTS
    if _CLIENTS is None:
        _CLIENTS = {
            "openrouter": AsyncOpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=os.environ["OPENROUTER_API_KEY"],
                max_retries=3,
            ),
            "groq": AsyncOpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=os.environ["GROQ_API_KEY"],
                max_retries=3,
            ),
            "google": AsyncOpenAI(
                base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
                api_key=os.environ["GEMINI_API_KEY"],
                max_retries=3,
            ),
        }
    return _CLIENTS

# Each ModelConfig specifies provider, model ID, and its own rate limits.
# tokens_per_minute=None disables token limiting for that model.
research_models: list[ModelConfig] = [
    ModelConfig("openrouter", "stepfun/step-3.5-flash:free",                   requests_per_minute=20),
    ModelConfig("openrouter", "mistralai/mistral-small-3.1-24b-instruct:free", requests_per_minute=20),
    ModelConfig("openrouter", "qwen/qwen3.6-plus-preview:free",                requests_per_minute=20),
    ModelConfig("google",     "gemini-2.5-flash",                              requests_per_minute=15, tokens_per_minute=1_000_000),
    ModelConfig("groq",       "llama-3.3-70b-versatile",                       requests_per_minute=30, tokens_per_minute=6_000),
]


async def query_answer(config: ModelConfig, prompt: str):
    messages = [{"role": "user", "content": prompt}]
    input_tokens = count_message_tokens(messages)
    await config.req_limiter.acquire()
    if config.tok_limiter:
        await config.tok_limiter.acquire(input_tokens)
    try:
        response = await get_clients()[config.provider].chat.completions.create(model=config.model, messages=messages)
        actual_total = response.usage.total_tokens if response.usage else input_tokens + 600
        if config.tok_limiter:
            config.tok_limiter.release(input_tokens, actual_total)
        return response.choices[0].message.content
    except Exception as e:
        if config.tok_limiter:
            config.tok_limiter.release(input_tokens, input_tokens)  # free reservation on error
        return f"ERROR: {str(e)}"


async def process_prompts(file_path, models: list[ModelConfig] | None = None):
    if models is None:
        models = research_models

    prompts = pd.read_csv(file_path)
    prompts.rename(columns={c: c.strip().title() for c in prompts.columns if c.strip().lower() in ('prompt', 'topic')}, inplace=True)
    prompts['Prompt'] = prompts['Prompt'].str.strip()
    prompt_rows = prompts[['Prompt', 'Topic']].to_dict('records')

    # Load existing responses, dropping any ERROR rows so they get retried
    responses_path = 'model_responses.csv'
    if os.path.exists(responses_path):
        existing = pd.read_csv(responses_path)
        error_mask = existing['answer'].astype(str).str.startswith('ERROR')
        dropped = error_mask.sum()
        if dropped:
            print(f"Dropping {dropped} ERROR row(s) from existing results — will retry.")
            existing = existing[~error_mask]
        completed = set(zip(existing['model'], existing['original_prompt']))
    else:
        existing = pd.DataFrame()
        completed = set()

    tasks = []
    skipped = 0
    for config in models:
        for row in prompt_rows:
            if (config.model, row['Prompt']) in completed:
                skipped += 1
            else:
                tasks.append((config, row['Prompt'], row['Topic']))

    if skipped:
        print(f"Skipping {skipped} already-completed prompt(s).")
    print(f"Starting audit for {len(tasks)} total requests...")

    # Using a semaphore to limit concurrency
    semaphore = asyncio.Semaphore(10) # Process 10 at a time

    async def sem_task(config, prompt, topic):
        async with semaphore:
            answer = await query_answer(config, prompt)
            return {"topic": topic, "original_prompt": prompt, "provider": config.provider, "model": config.model, "answer": answer}

    # Execute all tasks
    if not tasks:
        print("All prompts already have responses. Nothing to do.")
        return

    final_data = await tqdm.gather(*(sem_task(cfg, p, t) for cfg, p, t in tasks))

    # Convert to DataFrame and save (append to any existing results)
    new_df = pd.DataFrame(final_data)
    output_df = pd.concat([existing, new_df], ignore_index=True) if not existing.empty else new_df
    output_df.to_csv(responses_path, index=False)
    print(f"Success! {len(new_df)} new result(s) saved to {responses_path}")
    print(output_df.head())

if __name__ == "__main__":
    def load_models_from_config(path: str) -> dict[str, ModelConfig]:
        with open(path) as f:
            entries = json.load(f)
        return {
            e["name"]: ModelConfig(
                e["provider"],
                e["model"],
                requests_per_minute=e["requests_per_minute"],
                tokens_per_minute=e.get("tokens_per_minute"),
            )
            for e in entries
        }

    parser = argparse.ArgumentParser(description="Run prompt ingestion against selected models.")
    parser.add_argument("csv", nargs="?", help="Path to the CSV file with Prompt and Topic columns.")
    parser.add_argument("--models", nargs="+", metavar="NAME", help="Names of models to run (from config). Runs all if omitted.")
    parser.add_argument("--config", default="cli_models_config.json", metavar="PATH", help="Path to models config JSON (default: cli_models_config.json).")
    parser.add_argument("--list-models", action="store_true", help="List available models from the config and exit.")
    args = parser.parse_args()

    try:
        all_models = load_models_from_config(args.config)
    except FileNotFoundError:
        print(f"Config file not found: {args.config}")
        sys.exit(1)

    if args.list_models:
        print(f"Available models in {args.config}:")
        for name, cfg in all_models.items():
            print(f"  - {name}  ({cfg.provider} / {cfg.model})")
        sys.exit(0)

    if not args.csv:
        parser.error("csv file is required unless --list-models is specified.")

    if args.models:
        unknown = [m for m in args.models if m not in all_models]
        if unknown:
            print(f"Unknown model(s): {', '.join(unknown)}")
            print("Run with --list-models to see available options.")
            sys.exit(1)
        selected = [all_models[m] for m in args.models]
    else:
        selected = list(all_models.values())

    prompts_df = pd.read_csv(args.csv)
    num_prompts = prompts_df['Prompt'].dropna().nunique()
    num_models = len(selected)
    total_responses = num_prompts * num_models

    print(f"\nModels selected ({num_models}):")
    for m in selected:
        print(f"  - {m.provider} / {m.model}")
    print(f"\nThis will run {num_prompts} prompt(s) across {num_models} model(s), generating {total_responses} total responses.")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    asyncio.run(process_prompts(args.csv, models=selected))
