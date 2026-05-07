import argparse
import json
import os
import sys

import requests
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = "cli_models_config.json"


def fetch_openrouter_model(model_id: str) -> dict | None:
    api_key = os.getenv("OPENROUTER_API_KEY")
    r = requests.get(
        "https://openrouter.ai/api/v1/models",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=15,
    )
    r.raise_for_status()
    for m in r.json().get("data", []):
        if m.get("id") == model_id:
            return m
    return None


def load_config() -> list[dict]:
    try:
        with open(CONFIG_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_config(config: list[dict]) -> None:
    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _price(val) -> str:
    try:
        return f"${float(val) * 1_000_000:.4f} / M tokens"
    except (TypeError, ValueError):
        return "N/A"


def cmd_add(model_id: str) -> None:
    print(f"Fetching model details for '{model_id}' from OpenRouter...")
    try:
        m = fetch_openrouter_model(model_id)
    except Exception as e:
        print(f"Error fetching from OpenRouter: {e}")
        sys.exit(1)

    if not m:
        print(f"Model '{model_id}' not found on OpenRouter.")
        sys.exit(1)

    pricing = m.get("pricing", {})
    print(f"\nModel found:")
    print(f"  Name:             {m.get('name', model_id)}")
    print(f"  Model ID:         {m.get('id')}")
    print(f"  Free:             {'Yes' if model_id.endswith(':free') else 'No'}")
    print(f"  Context length:   {m.get('context_length', 'N/A')}")
    print(f"  Knowledge cutoff: {m.get('knowledge_cutoff', 'N/A')}")
    print(f"  Input price:      {_price(pricing.get('prompt'))}")
    print(f"  Output price:     {_price(pricing.get('completion'))}")

    default_name = m.get("name") or model_id
    display_name = input(f"\nDisplay name (press Enter to use '{default_name}', or type a new name): ").strip() or default_name
    rpm_input = input("Requests per minute [20]: ").strip()
    rpm = int(rpm_input) if rpm_input.isdigit() else 20

    confirm = input(f"\nAdd '{display_name}' to {CONFIG_PATH}? (y/n): ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    config = load_config()
    if any(e["model"] == model_id for e in config):
        print(f"'{model_id}' is already in {CONFIG_PATH}.")
        sys.exit(0)

    config.append({
        "name": display_name,
        "provider": "openrouter",
        "model": model_id,
        "requests_per_minute": rpm,
        "tokens_per_minute": None,
    })
    save_config(config)
    print(f"Added '{display_name}' to {CONFIG_PATH}.")


def cmd_list() -> None:
    config = load_config()
    if not config:
        print(f"No models in {CONFIG_PATH}.")
        return
    print(f"Models in {CONFIG_PATH}:")
    for e in config:
        print(f"  - {e['name']}  ({e['provider']} / {e['model']})")


def cmd_remove(display_name: str) -> None:
    config = load_config()
    updated = [e for e in config if e["name"] != display_name]
    if len(updated) == len(config):
        print(f"No model named '{display_name}' found in {CONFIG_PATH}.")
        sys.exit(1)
    save_config(updated)
    print(f"Removed '{display_name}' from {CONFIG_PATH}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=f"Manage {CONFIG_PATH}.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="Add a model by OpenRouter ID.")
    add_parser.add_argument("model_id", metavar="MODEL_ID", help="OpenRouter model ID (e.g. qwen/qwen3-30b:free).")

    subparsers.add_parser("list", help="List models currently in the config.")

    remove_parser = subparsers.add_parser("remove", help="Remove a model by display name.")
    remove_parser.add_argument("name", metavar="NAME", help="Display name of the model to remove.")

    args = parser.parse_args()

    if args.command == "add":
        cmd_add(args.model_id)
    elif args.command == "list":
        cmd_list()
    elif args.command == "remove":
        cmd_remove(args.name)
