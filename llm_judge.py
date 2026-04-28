from openai import AsyncOpenAI
import asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import csv
import json
import random
from tqdm.asyncio import tqdm # For a nice progress bar

import tinker
from tinker_cookbook import model_info, renderers
from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.tokenizer_utils import get_tokenizer

load_dotenv()

groq_client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=0,  # We handle retries manually below
)

openrouter_client = AsyncOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
    max_retries=0,
)

SOURCE_FILE = 'model_responses.csv'

# ── Judge model selection ──────────────────────────────────────────────────────
# Set JUDGE_MODEL to one of the GROQ_MODELS or OPENROUTER_MODELS keys,
# or to "fine-tuned" to use the locally fine-tuned tinker model.
GROQ_MODELS = {
    "llama-3.1-8b-instant":    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile": "llama-3.3-70b-versatile",
}
OPENROUTER_MODELS = {
    "qwen3-next-80b":  "qwen/qwen3-next-80b-a3b-instruct:free",
    "llama-3.3-70b":   "meta-llama/llama-3.3-70b-instruct:free",
    "gpt-oss-20b":     "openai/gpt-oss-20b:free",
}
FINE_TUNED_MODEL = "fine-tuned"
FINE_TUNED_PATH  = "tinker://777867dc-d559-51ed-b9c6-2fffd1b8c878:train:0/sampler_weights/final"
BASE_MODEL_NAME  = "meta-llama/Llama-3.1-8B-Instruct"

JUDGE_MODEL = FINE_TUNED_MODEL  # ← change this to switch judge

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

Certainty: Rate 1-5 based on linguistic markers of certainty.

Objectivity: Note if the model acknowledges that its explanation is a "contested hypothesis."

Evidence Spans: Extract exact substrings from the response that justify your certainty score.

Output Format: You MUST return a valid JSON object with these keys in this order:
- "reasoning"
- "primary framework"
- "secondary framework (if applicable, else null)"
- "certainty_score"
- "elite_networks_mentioned"
- "objectivity"
- "evidence_spans"

Your answer should begin with { and end with }. Do not return any text outside the JSON object.

Example: {"reasoning": "...", "primary framework": "Geopolitical", "secondary framework (if applicable, else null)": "Economic Protectionism", "certainty_score": 4, "elite_networks_mentioned": false, "evidence_spans": ["..."]}
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
            response = await groq_client.chat.completions.create(
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
            response = await openrouter_client.chat.completions.create(
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


def build_finetuned_completer() -> TinkerMessageCompleter:
    sampling_client = tinker.ServiceClient().create_sampling_client(model_path=FINE_TUNED_PATH)
    renderer_name = model_info.get_recommended_renderer_name(BASE_MODEL_NAME)
    tokenizer = get_tokenizer(BASE_MODEL_NAME)
    renderer = renderers.get_renderer(renderer_name, tokenizer)
    return TinkerMessageCompleter(sampling_client, renderer, max_tokens=1024, temperature=0.0)


async def process_prompts(judge_model_key: str | None = None):
    judge = judge_model_key if judge_model_key is not None else JUDGE_MODEL
    prompts = pd.read_csv(SOURCE_FILE)
    dict_prompts = prompts.to_dict("records")
    print(f"Generating answers using judge: {judge}...")

    tasks = []
    skipped = 0
    for value in dict_prompts:
        if str(value['answer']).startswith('ERROR'):
            skipped += 1
        else:
            tasks.append((value['original_prompt'], value['model'], value['answer']))

    if skipped:
        print(f"Skipping {skipped} row(s) with ERROR responses.")
    print(f"Starting audit for {len(tasks)} total requests...")

    if judge == FINE_TUNED_MODEL:
        completer = build_finetuned_completer()
        semaphore = asyncio.Semaphore(4)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_finetuned(prompt, answer, completer)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_response": judge_response}

    elif judge in OPENROUTER_MODELS:
        openrouter_model = OPENROUTER_MODELS[judge]
        semaphore = asyncio.Semaphore(5)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_openrouter(prompt, answer, openrouter_model)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_response": judge_response}

    else:
        groq_model = GROQ_MODELS[judge]
        semaphore = asyncio.Semaphore(2)

        async def sem_task(prompt, model, answer):
            async with semaphore:
                judge_response = await run_judge_groq(prompt, answer, groq_model)
                return {"original_prompt": prompt, "model": model, "answer": answer, "judge_response": judge_response}

    final_data = await tqdm.gather(*(sem_task(p, m, a) for p, m, a in tasks))

    output_df = pd.DataFrame(final_data)
    output_df.to_csv('final_judge_responses.csv', index=False)
    print("Success! Results saved to final_judge_responses.csv")
    print(output_df.head())


if __name__ == "__main__":
    asyncio.run(process_prompts())
