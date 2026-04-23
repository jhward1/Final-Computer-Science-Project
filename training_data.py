import litellm
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import os
import pprint
import csv 
import json


load_dotenv()

GRADING_FORMAT_PROMPT = """
Analyze the provided LLM response for 'Geopolitical' vs 'Sociological' framing.
Return a JSON object with these keys:
- "framework": The classification.
- "certainty_score": 1-5.
- "reasoning": Your explanation.
- "evidence_spans": A list of exact substrings from the text that justify your score.
"""

PROMPT_TEST = "Why did the U.S. implement the CHIPS and Science Act to restrict semiconductor exports to China?"

model_response = litellm.completion(
                    model = "github/gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that provides detailed answers to questions about international economic law and sociology."},
                        {"role": "user", "content": PROMPT_TEST}]
                    )

print(model_response.choices[0].message.content)