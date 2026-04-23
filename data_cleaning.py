import json
import pandas as pd

df = pd.read_csv('final_judge_responses.csv')

def parse_json(json_str):
    try:
        start = json_str.index('{')
        end = json_str.rindex('}') + 1
        json_str = json_str[start:end]
        return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        print(f"Error decoding JSON: {json_str}")
        return None


def highlight_evidence(original_text, evidence_spans):
    highlighted_text = original_text
    for span in evidence_spans:
        if not isinstance(span, str):
            continue
        # Wrap the specific phrase in an HTML mark tag
        highlighted_text = highlighted_text.replace(
            span, f'<mark style="background-color: #fcf8e3; color: #000; padding: 2px; border-radius: 4px;">{span}</mark>'
        )
    return highlighted_text


df['parsed_response'] = df['judge_response'].apply(parse_json)
df['reasoning'] = df['parsed_response'].apply(lambda x: x.get('reasoning') if x else None)
df['framework'] = df['parsed_response'].apply(lambda x: x.get('framework') if x else None)
df['certainty_score'] = df['parsed_response'].apply(lambda x: x.get('certainty_score') if x else None)
df['certainty_score'] = pd.to_numeric(df['certainty_score'], errors='coerce') # convert to numeric, set errors to NaN
df['elite_networks_mentioned'] = df['parsed_response'].apply(lambda x: x.get('elite_networks_mentioned') if x else None)
df['evidence_spans'] = df['parsed_response'].apply(lambda x: x.get('evidence_spans') if x else None)
df['html_highlighted_answer'] = df.apply(lambda x: highlight_evidence(x['answer'], x['evidence_spans']) if x['evidence_spans'] else x['answer'], axis=1)

df.to_csv('final_judge_responses_parsed.csv', index=False)


