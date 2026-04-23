from openai import AsyncOpenAI
import asyncio
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import csv
import json
from tqdm.asyncio import tqdm # For a nice progress bar

load_dotenv()

import random

client = AsyncOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ["GROQ_API_KEY"],
    max_retries=0,  # We handle retries manually below
)

SOURCE_FILE = 'model_responses.csv' #This file has columns: question, model, response
JUDGE_MODEL = "llama-3.1-8b-instant"  # Use a strong model for judging to ensure high-quality evaluations. Adjust as needed based on your API access and budget.
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

async def run_judge(question, answer, judge_model, retries=5):
    prompt = f"""
    Analyze the following LLM response to the question: '{question}'

    RESPONSE TO ANALYZE:
    {answer}
    """
    for attempt in range(retries):
        try:
            response = await client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": GRADING_FORMAT_PROMPT},
                    {"role": "user", "content": prompt}
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

async def process_prompts():
    prompts = pd.read_csv(SOURCE_FILE)    
    dict_prompts = prompts.to_dict("records")
    print(f"Generating answers for model: {JUDGE_MODEL}...")
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
    
    # Using a semaphore to limit concurrency (prevents overwhelming your API keys)
    semaphore = asyncio.Semaphore(2) # Process 2 at a time to stay within Groq rate limits

    async def sem_task(prompt, model, answer):
        async with semaphore:
            judge_response = await run_judge(prompt, answer, JUDGE_MODEL)
            return {"original_prompt": prompt, "model": model, "answer": answer, "judge_response": judge_response}

    # Execute all tasks
    final_data = await tqdm.gather(*(sem_task(p, m, a) for p, m, a in tasks))

    # Convert to DataFrame and save
    output_df = pd.DataFrame(final_data)

    output_df.to_csv('final_judge_responses.csv', index=False)    
    print(f"Success! Results saved to data frame")
    print(output_df.head())


if __name__ == "__main__":
    asyncio.run(process_prompts())


