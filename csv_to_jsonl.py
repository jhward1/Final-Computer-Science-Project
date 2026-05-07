import argparse
import csv
import json
import os
import sys

INTRO_TEXT = '''
Analyze the following LLM response to the question: '{question}'

RESPONSE TO ANALYZE:{answer}
'''

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a training CSV to JSONL format for fine-tuning.")
    parser.add_argument("csv_file", metavar="CSV_FILE", help="Path to the input CSV file.")
    parser.add_argument("--output", metavar="OUTPUT_FILE", default="judge_prompts_and_answers.jsonl",
                        help="Path to the output JSONL file (default: judge_prompts_and_answers.jsonl).")
    args = parser.parse_args()

    if not os.path.exists(args.csv_file):
        print(f"File not found: {args.csv_file}")
        sys.exit(1)

    count = 0
    with open(args.csv_file, "r") as csv_file, open(args.output, "w") as jsonl_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            judge_answer = {}
            judge_prompt = INTRO_TEXT.format(question=row["Prompt"], answer=row["Model Answer"]).strip()
            judge_answer["primary_framework"] = row["primary_framework"]
            judge_answer["secondary_framework"] = row["secondary_framework"]
            judge_answer["certainty score"] = row.get("certainty score") or row.get("certainty_score")
            judge_answer["elite_networks"] = row["elite_networks"]
            judge_answer["certainty_strings"] = row["certainty_strings"]
            judge_answer["framework_strings"] = row["framework_strings"]
            structured_output = {"messages": [{"role": "user", "content": judge_prompt}, {"role": "assistant", "content": json.dumps(judge_answer)}]}
            jsonl_file.write(json.dumps(structured_output) + "\n")
            count += 1

    print(f"Done. {count} row(s) written to {args.output}.")
