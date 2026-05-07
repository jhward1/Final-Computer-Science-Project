import requests
import os
from datetime import datetime, date, timezone
from dotenv import load_dotenv

load_dotenv()

PROVIDER_LABELS = {
    "openai":        "OpenAI",
    "anthropic":     "Anthropic",
    "google":        "Google",
    "meta-llama":    "Meta",
    "mistralai":     "Mistral",
    "qwen":          "Qwen (Alibaba)",
    "x-ai":          "xAI",
    "cohere":        "Cohere",
    "nvidia":        "NVIDIA",
    "microsoft":     "Microsoft",
    "deepseek":      "DeepSeek",
    "tencent":       "Tencent",
    "perplexity":    "Perplexity",
    "nousresearch":  "Nous Research",
    "01-ai":         "01.AI",
}


def _to_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (ValueError, TypeError):
        return None


def fetch_openrouter_models() -> list[dict]:
    """Fetch all models from the OpenRouter API and return a normalized list."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    response = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    response.raise_for_status()

    result = []
    for m in response.json().get("data", []):
        model_id: str = m.get("id", "")
        clean_id = model_id.lstrip("~")
        prefix = clean_id.split("/")[0] if "/" in clean_id else clean_id
        created_ts = m.get("created")
        created_date: date | None = (
            datetime.fromtimestamp(created_ts, tz=timezone.utc).date() if created_ts else None
        )
        result.append({
            "id":               model_id,
            "name":             m.get("name") or model_id,
            "provider_slug":    prefix,
            "provider_label":   PROVIDER_LABELS.get(prefix, prefix),
            "is_free":          model_id.endswith(":free"),
            "created":          created_date,
            "context_length":   m.get("context_length"),
            "knowledge_cutoff": m.get("knowledge_cutoff"),
            "input_price":      _to_float(m.get("pricing", {}).get("prompt")),
            "output_price":     _to_float(m.get("pricing", {}).get("completion")),
        })

    return result
