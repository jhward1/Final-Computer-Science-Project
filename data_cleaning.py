import ast
import json
import re
import unicodedata
import pandas as pd


# Simple fields whose values never contain apostrophes — safe for regex extraction
_SIMPLE_FIELDS = ['certainty score', 'primary_framework', 'secondary_framework', 'elite_networks']


def parse_json(json_str):
    # Try full JSON parse first
    try:
        start = json_str.index('{')
        end = json_str.rindex('}') + 1
        return json.loads(json_str[start:end])
    except (ValueError, json.JSONDecodeError):
        pass

    # Fallback: regex-extract each simple field individually
    result = {}
    for key in _SIMPLE_FIELDS:
        match = re.search(rf'"{re.escape(key)}":\s*"([^"]*)"', json_str)
        if match:
            result[key] = match.group(1)

    if result:
        print(f"Partial parse (certainty_strings skipped): {result}")
        return result

    print(f"Could not parse: {json_str[:200]}")
    return None


CERTAINTY_COLOR = "#fff3cd"  # yellow
FRAMEWORK_COLOR = "#cce5ff"  # blue

_UNICODE_REPLACEMENTS = [
    # Various dashes → hyphen
    ('‑', '-'), ('‒', '-'), ('–', '-'), ('—', '-'), ('―', '-'),
    # Curly quotes → straight quotes
    ('‘', "'"), ('’', "'"), ('“', '"'), ('”', '"'),
    # Non-breaking space → regular space
    (' ', ' '),
]


def _normalize(text: str) -> str:
    text = unicodedata.normalize('NFKC', text)
    for orig, replacement in _UNICODE_REPLACEMENTS:
        text = text.replace(orig, replacement)
    return text


def highlight_evidence(original_text: str, evidence_spans: list, color: str = CERTAINTY_COLOR) -> str:
    mark = f'<mark style="background-color: {color}; color: #000; padding: 2px; border-radius: 4px;">'
    highlighted_text = _normalize(original_text)
    for span in evidence_spans:
        if not isinstance(span, str) or not span.strip():
            continue
        normalized_span = _normalize(span)
        try:
            highlighted_text = re.sub(
                re.escape(normalized_span),
                f'{mark}{normalized_span}</mark>',
                highlighted_text,
                flags=re.IGNORECASE,
            )
        except re.error:
            continue
    return highlighted_text


def parse_responses(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['parsed_response'] = df['judge_response'].apply(parse_json)
    df['framework'] = df['parsed_response'].apply(lambda x: x.get('primary_framework') if x else None)
    df['secondary_framework'] = df['parsed_response'].apply(lambda x: x.get('secondary_framework') if x else None)
    df['certainty_score'] = df['parsed_response'].apply(
        lambda x: x.get('certainty score') or x.get('certainty_score') if x else None
    )
    df['certainty_score'] = pd.to_numeric(df['certainty_score'], errors='coerce')
    df['elite_networks_mentioned'] = df['parsed_response'].apply(lambda x: x.get('elite_networks') if x else None)
    def parse_spans(row):
        parsed = row['parsed_response']
        raw_response = str(row['judge_response'])

        # Get raw certainty_strings value from parsed dict if available
        raw = parsed.get('certainty_strings') if parsed else None

        # If not in parsed dict, try to extract the raw value directly from the
        # judge response string — handles cases where JSON parsing failed
        if not raw:
            m = re.search(r'"certainty_strings":\s*"(\[.*)', raw_response, re.DOTALL)
            if m:
                raw = m.group(1)
                # Trim trailing JSON closing characters
                raw = re.sub(r'["\s}]+$', '', raw).strip()
                if not raw.endswith(']'):
                    raw += ']'

        if not raw:
            return None
        if isinstance(raw, list):
            return raw

        # Strategy 1: ast.literal_eval (works when no unescaped apostrophes)
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return [s for s in result if isinstance(s, str)]
        except (ValueError, SyntaxError):
            pass

        # Strategy 2: regex split on list item boundaries
        # Handles apostrophes in span text that break ast.literal_eval
        items = re.split(r"',\s*'", raw.strip().strip("[]'\""))
        items = [item.strip().strip("'\"") for item in items if len(item.strip()) > 10]
        return items if items else None

    def _extract_spans(parsed, raw_response, key):
        raw = parsed.get(key) if parsed else None
        if not raw:
            m = re.search(rf'"{re.escape(key)}":\s*"(\[.*)', raw_response, re.DOTALL)
            if m:
                raw = m.group(1)
                raw = re.sub(r'["\s}]+$', '', raw).strip()
                if not raw.endswith(']'):
                    raw += ']'
        if not raw:
            return None
        if isinstance(raw, list):
            return raw
        try:
            result = ast.literal_eval(raw)
            if isinstance(result, list):
                return [s for s in result if isinstance(s, str)]
        except (ValueError, SyntaxError):
            pass
        items = re.split(r"',\s*'", raw.strip().strip("[]'\""))
        items = [item.strip().strip("'\"") for item in items if len(item.strip()) > 10]
        return items if items else None

    def parse_spans(row):
        return _extract_spans(row['parsed_response'], str(row['judge_response']), 'certainty_strings')

    def parse_framework_spans(row):
        return _extract_spans(row['parsed_response'], str(row['judge_response']), 'framework_strings')

    df['evidence_spans'] = df.apply(parse_spans, axis=1)
    df['framework_spans'] = df.apply(parse_framework_spans, axis=1)

    def build_combined_html(row):
        # Apply certainty spans (yellow) first, then framework spans (blue) on top
        text = highlight_evidence(row['answer'], row['evidence_spans'], CERTAINTY_COLOR) if row['evidence_spans'] else _normalize(row['answer'])
        text = highlight_evidence(text, row['framework_spans'], FRAMEWORK_COLOR) if row['framework_spans'] else text
        return text

    df['html_highlighted_answer'] = df.apply(build_combined_html, axis=1)
    return df


if __name__ == "__main__":
    df = pd.read_csv('final_judge_responses.csv')
    parsed = parse_responses(df)
    parsed.to_csv('final_judge_responses_parsed.csv', index=False)
