from openai import AsyncOpenAI
import argparse
import asyncio
import sys
import pandas as pd
from dotenv import load_dotenv
import os
import random
from tqdm.asyncio import tqdm # For a nice progress bar

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv()

_groq_client: AsyncOpenAI | None = None
_openrouter_client: AsyncOpenAI | None = None


def get_groq_client() -> AsyncOpenAI:
    global _groq_client
    if _groq_client is None:
        _groq_client = AsyncOpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ["GROQ_API_KEY"],
            max_retries=0,
        )
    return _groq_client


def get_openrouter_client() -> AsyncOpenAI:
    global _openrouter_client
    if _openrouter_client is None:
        _openrouter_client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=0,
        )
    return _openrouter_client

SOURCE_FILE = 'model_responses.csv'

# ── Judge model selection ──────────────────────────────────────────────────────
# Set JUDGE_MODEL to one of the GROQ_MODELS or OPENROUTER_MODELS keys,
# or to "fine-tuned" to use the locally fine-tuned tinker model.
GROQ_MODELS = {
    "llama-3.1-8b":    "llama-3.1-8b-instant",
    "llama-3.3-70b": "llama-3.3-70b-versatile",
    "gpt-oss-120b": "openai/gpt-oss-120b"
}
OPENROUTER_MODELS = {
    "qwen3-next-80b":  "qwen/qwen3-next-80b-a3b-instruct:free"
}
FINE_TUNED_MODEL  = "fine-tuned"
BASE_TINKER_MODEL = "base-llama"
QWEN3_TINKER_MODEL = "qwen3-30b-tinker"
FINE_TUNED_PATH   = "tinker://777867dc-d559-51ed-b9c6-2fffd1b8c878:train:0/sampler_weights/final"
BASE_MODEL_NAME   = "meta-llama/Llama-3.1-8B-Instruct"
QWEN3_MODEL_NAME  = "Qwen/Qwen3-30B-A3B-Instruct-2507"

JUDGE_MODEL = QWEN3_MODEL_NAME  # ← change this to switch judge

# ──────────────────────────────────────────────────────────────────────────────

GRADING_FORMAT_PROMPT = '''
You are an expert scholar in International Economic Law and Sociology. Your task is to analyze LLM responses for explanatory bias.

Coding Instructions:

Primary Framework:
Categorize as "Geopolitical" if the response focuses on state-level national security and/or great power competition and/or geoeconomics.
Categorize as "Sociological" if it focuses on elite networks, rent-seeking, or personalist power.
Categorize as "Economic Protectionism" if it focuses on trade barriers, tariffs, or economic producer interest.

Secondary Framework (if applicable):
If the response contains significant elements of a second framework, note it here. Otherwise, set to null.

Certainty Score: Rate 1-5 using the scale below. Choose the level that best describes 
the dominant epistemic register of the response — if hedges and boosters both appear, 
weight by frequency and prominence (e.g., a hedge in the conclusion outweighs a booster 
in a subordinate clause).

1 — Highly uncertain: Explanation is explicitly framed as one possibility among many. 
    Dominated by hedges and epistemic humility. 
    Markers: "might", "could", "one possible explanation", "it is unclear whether", 
    "some argue", "contested"

2 — Leaning uncertain: Hedging language is frequent but the response leans toward 
    a preferred explanation without committing.
    Markers: "likely", "probably", "tends to", "appears to", "suggests"

3 — Neutral / balanced: Neither commits nor hedges systematically. May acknowledge 
    multiple explanations with roughly equal weight, or uses neutral descriptive language.
    Markers: "has been linked to", "is associated with", "plays a role in"

4 — Leaning certain: Response treats one explanation as the primary or correct one 
    with minimal hedging. Causal language is common.
    Markers: "is driven by", "reflects", "stems from", "is the result of", "explains"

5 — Highly certain: Explanation is presented as established fact with no meaningful 
    hedging. Causal claims are direct and unqualified.
    Markers: "is", "causes", "proves", "demonstrates", "is clearly due to"

When assigning the score, identify which level's marker pattern best fits the overall 
response, and ensure at least one extracted certainty_string directly exemplifies 
the chosen level. If the response is mixed, note the dominant register and assign 
accordingly — do not average across levels.

Elite Networks: True if the response explicitly mentions elite networks, rent-seeking, or personalist power dynamics; otherwise False.

Certainty Strings: Extract 2-5 verbatim substrings (10-40 words each) that contain linguistic markers of epistemic certainty or uncertainty, such as:
- Hedges: "may", "might", "could", "appears to", "seems", "arguably"
- Boosters: "clearly", "obviously", "undoubtedly", "is driven by", "is the result of"
- Qualifiers: "in some cases", "generally", "tends to"
- Epistemic stance: "evidence suggests", "it is likely that", "research shows"
The strings must appear verbatim in the source text.

Framework Strings: Extract 2-4 verbatim substrings (10-50 words each) that are the primary evidence for your primary_framework classification — i.e., the phrases that 
would be most convincing to a skeptic questioning your categorization. Avoid generic topic mentions; prefer phrases that reveal the explanatory logic being used.

CRITICAL: All strings in certainty_strings and framework_strings must be copied 
verbatim from the response text — do not paraphrase, summarize, or alter wording 
in any way, including punctuation.

Before finalizing, verify: does each extracted string actually appear character-for-character 
in the source text? If not, replace it with one that does.

Output Format: You MUST return a valid JSON object with these keys in this order:
- "primary_framework"
- "secondary_framework"
- "certainty score"
- "elite_networks"
- "certainty_strings"
- "framework_strings"

Before outputting JSON, briefly identify:
(a) The 1-2 sentences in the response that most clearly signal the primary framework
(b) The strongest hedging or certainty marker you found

Then output the JSON using those identified strings.

Before finalizing, verify: does each extracted string actually appear character-for-character 
in the source text? If not, replace it with one that does.

Your answer should begin with { and end with }. Do not return any text outside the JSON object.

Example: {"primary_framework": "Geopolitical", "secondary_framework": "Economic Protectionism", "certainty score": 4, "elite_networks": false, "certainty_strings": ["..."], "framework_strings": ["..."]}
'''


def build_prompt(question: str, answer: str) -> str:
    return f"""
    Analyze the following LLM response to the question: '{question}'

    RESPONSE TO ANALYZE:
    {answer}
    """


async def run_judge_groq(question: str, answer: str, model: str, retries: int = 5) -> str:
    prompt = build_prompt(question, answer)
    for attempt in range(retries):
        try:
            response = await get_groq_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GRADING_FORMAT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                return f"ERROR: {str(e)}"


async def run_judge_openrouter(question: str, answer: str, model: str, retries: int = 5) -> str:
    prompt = build_prompt(question, answer)
    for attempt in range(retries):
        try:
            response = await get_openrouter_client().chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": GRADING_FORMAT_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return response.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < retries - 1:
                wait = 2 ** attempt + random.uniform(0, 1)
                print(f"Rate limited. Retrying in {wait:.1f}s...")
                await asyncio.sleep(wait)
            else:
                return f"ERROR: {str(e)}"


async def run_judge_finetuned(question: str, answer: str, completer: TinkerMessageCompleter) -> str:
    prompt = build_prompt(question, answer)
    try:
        response = await completer([{"role": "user", "content": prompt}])
        return response["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"


def build_base_model_completer() -> TinkerMessageCompleter:
    sampling_client = tinker.ServiceClient().create_sampling_client(base_model=BASE_MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return TinkerMessageCompleter(sampling_client, renderer, max_tokens=1024, temperature=0.0)


async def run_judge_base_tinker(question: str, answer: str, completer: TinkerMessageCompleter) -> str:
    prompt = build_prompt(question, answer)
    messages = [
        {"role": "system", "content": GRADING_FORMAT_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        response = await completer(messages)
        return response["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"


def build_qwen3_completer() -> TinkerMessageCompleter:
    sampling_client = tinker.ServiceClient().create_sampling_client(base_model=QWEN3_MODEL_NAME)
    renderer_name = model_info.get_recommended_renderer_name(QWEN3_MODEL_NAME)
    tokenizer = get_tokenizer(QWEN3_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return TinkerMessageCompleter(sampling_client, renderer, max_tokens=1024, temperature=0.0)


async def run_judge_qwen3_tinker(question: str, answer: str, completer: TinkerMessageCompleter) -> str:
    prompt = build_prompt(question, answer)
    messages = [
        {"role": "system", "content": GRADING_FORMAT_PROMPT},
        {"role": "user", "content": prompt},
    ]
    try:
        response = await completer(messages)
        return response["content"]
    except Exception as e:
        return f"ERROR: {str(e)}"


def build_finetuned_completer() -> TinkerMessageCompleter:
    sampling_client = tinker.ServiceClient().create_sampling_client(model_path=FINE_TUNED_PATH)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return TinkerMessageCompleter(sampling_client, renderer, max_tokens=1024, temperature=0.0)


def split_valid_invalid(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split responses into valid (will be judged) and invalid (excluded) rows."""
    answer_str = df['answer'].astype(str)
    invalid_mask = (
        df['answer'].isna() |
        answer_str.str.startswith('ERROR') |
        (answer_str.str.split().str.len() < 30)
    )
    return df[~invalid_mask], df[invalid_mask]


async def process_prompts(judge_model_key: str | None = None):
    judge = judge_model_key if judge_model_key is not None else JUDGE_MODEL
    prompts = pd.read_csv(SOURCE_FILE)
    valid_df, invalid_df = split_valid_invalid(prompts)
    print(f"Generating answers using judge: {judge}...")

    if not invalid_df.empty:
        print(f"Excluding {len(invalid_df)} row(s): empty, ERROR, or fewer than 30 words.")

    # Load already-judged (model, prompt) pairs to avoid re-running them
    judge_responses_path = 'final_judge_responses.csv'
    if os.path.exists(judge_responses_path):
        existing_judged = pd.read_csv(judge_responses_path)
        # Include judge_model in the key so the same row can be judged by multiple models
        if 'judge_model' in existing_judged.columns:
            already_judged = set(zip(existing_judged['model'], existing_judged['original_prompt'], existing_judged['judge_model']))
        else:
            already_judged = set()
    else:
        existing_judged = pd.DataFrame()
        already_judged = set()

    tasks = []
    skipped = 0
    for value in valid_df.to_dict("records"):
        if (value['model'], value['original_prompt'], judge) in already_judged:
            skipped += 1
        else:
            tasks.append((value['original_prompt'], value['model'], value['answer']))

    if skipped:
        print(f"Skipping {skipped} already-judged row(s).")
    print(f"Starting audit for {len(tasks)} total requests...")

    if judge == FINE_TUNED_MODEL:
        completer = build_finetuned_completer()
        semaphore = asyncio.Semaphore(1)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_finetuned(prompt, answer, completer)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_model": judge, "judge_response": judge_response}

    elif judge == BASE_TINKER_MODEL:
        completer = build_base_model_completer()
        semaphore = asyncio.Semaphore(1)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_base_tinker(prompt, answer, completer)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_model": judge, "judge_response": judge_response}

    elif judge == QWEN3_TINKER_MODEL:
        completer = build_qwen3_completer()
        semaphore = asyncio.Semaphore(1)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_qwen3_tinker(prompt, answer, completer)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_model": judge, "judge_response": judge_response}

    elif judge in OPENROUTER_MODELS:
        openrouter_model = OPENROUTER_MODELS[judge]
        semaphore = asyncio.Semaphore(5)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_openrouter(prompt, answer, openrouter_model)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_model": judge, "judge_response": judge_response}

    else:
        groq_model = GROQ_MODELS[judge]
        semaphore = asyncio.Semaphore(2)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_groq(prompt, answer, groq_model)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_model": judge, "judge_response": judge_response}

    if not tasks:
        print("All valid responses have already been judged. Nothing to do.")
        return

    final_data = await tqdm.gather(*(sem_task(p, m, a) for p, m, a in tasks))

    new_df = pd.DataFrame(final_data)
    output_df = pd.concat([existing_judged, new_df], ignore_index=True) if not existing_judged.empty else new_df
    output_df.to_csv('final_judge_responses.csv', index=False)
    print(f"Success! {len(new_df)} new result(s) saved to final_judge_responses.csv")


if __name__ == "__main__":
    ALL_JUDGES = {
        "fine-tuned":          FINE_TUNED_MODEL,
        "base-llama":          BASE_TINKER_MODEL,
        "qwen3-tinker":        QWEN3_TINKER_MODEL,
        **{k: k for k in GROQ_MODELS},
        **{k: k for k in OPENROUTER_MODELS},
    }

    parser = argparse.ArgumentParser(description="Run the LLM judge against model_responses.csv.")
    parser.add_argument("--judge", metavar="NAME", default=None,
                        help=f"Judge model to use (default: {JUDGE_MODEL}). Use --list-judges to see options.")
    parser.add_argument("--list-judges", action="store_true", help="List available judge models and exit.")
    args = parser.parse_args()

    if args.list_judges:
        print("Available judge models:")
        for name in ALL_JUDGES:
            marker = "  (default)" if ALL_JUDGES[name] == JUDGE_MODEL else ""
            print(f"  - {name}{marker}")
        sys.exit(0)

    judge_key = ALL_JUDGES.get(args.judge) if args.judge else JUDGE_MODEL
    if args.judge and judge_key is None:
        print(f"Unknown judge: {args.judge}")
        print("Run with --list-judges to see available options.")
        sys.exit(1)

    responses_df = pd.read_csv(SOURCE_FILE)
    num_responses = len(responses_df)

    print(f"Judge:     {args.judge or JUDGE_MODEL}")
    print(f"Responses: {num_responses} rows in {SOURCE_FILE}")
    print(f"This will generate {num_responses} judge evaluation(s).")

    confirm = input("\nProceed? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    asyncio.run(process_prompts(judge_model_key=judge_key))
    print("\nJudging complete. Run the following to parse the results:")
    print("  python data_cleaning.py")
