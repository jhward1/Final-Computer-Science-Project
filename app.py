import streamlit as st
import asyncio
import tempfile
import os
import io
import json
import pandas as pd
import nest_asyncio

nest_asyncio.apply()

from prompt_ingestion import process_prompts, ModelConfig
from llm_judge import process_prompts as run_judge, GROQ_MODELS, OPENROUTER_MODELS, FINE_TUNED_MODEL, BASE_TINKER_MODEL, QWEN3_TINKER_MODEL, split_valid_invalid
from data_cleaning import parse_responses
from data_viz import render_dashboard
from model_selection import fetch_openrouter_models
from github_storage import sync_from_github, push, is_configured

st.set_page_config(page_title="LLM Bias Analysis Tool", layout="wide")
st.title("LLM Bias Analysis Tool")

sync_from_github()

tab1, tab2, tab3, tab4 = st.tabs(["1. Prompt Ingestion", "2. LLM Judge", "3. Analysis Dashboard", "4. Model Config"])

# ── Tab 1: Prompt Ingestion ────────────────────────────────────────────────────

def load_models(path: str = "models_config.json") -> dict[str, ModelConfig]:
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

if "available_models" not in st.session_state:
    st.session_state["available_models"] = load_models()

AVAILABLE_MODELS = st.session_state["available_models"]

with tab1:
    st.subheader("Upload Prompts CSV")
    st.write("Upload a CSV with **Prompt** and **Topic** columns to query the selected models.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        st.session_state["uploaded_csv"] = uploaded_file.read()

    csv_bytes = st.session_state.get("uploaded_csv")

    if csv_bytes is not None:
        df_preview = pd.read_csv(io.BytesIO(csv_bytes))
        st.subheader("Preview")
        st.dataframe(
            df_preview.head(10),
            width="stretch",
            column_config={c: st.column_config.TextColumn(c, width="large") for c in ["Prompt", "Topic"]},
        )

        missing = [c for c in ["Prompt", "Topic"] if c not in df_preview.columns]
        if missing:
            st.error(f"CSV is missing required column(s): {', '.join(missing)}")
        else:
            st.subheader("Select Models")
            models_df = pd.DataFrame({
                "Selected":     [True] * len(AVAILABLE_MODELS),
                "Model":        list(AVAILABLE_MODELS.keys()),
                "Provider":     [c.provider for c in AVAILABLE_MODELS.values()],
                "Model ID":     [c.model for c in AVAILABLE_MODELS.values()],
                "Req / min":    [c.requests_per_minute for c in AVAILABLE_MODELS.values()],
                "Tokens / min": [c.tokens_per_minute if c.tokens_per_minute else "—" for c in AVAILABLE_MODELS.values()],
            })
            edited_df = st.data_editor(
                models_df,
                column_config={"Selected": st.column_config.CheckboxColumn("Selected")},
                disabled=["Model", "Provider", "Model ID", "Req / min", "Tokens / min"],
                hide_index=True,
                width="stretch",
            )

            selected_names = edited_df.loc[edited_df["Selected"], "Model"].tolist()

            if not selected_names:
                st.warning("Select at least one model to continue.")
            elif st.button("Run Prompt Ingestion"):
                selected_models = [AVAILABLE_MODELS[name] for name in selected_names]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
                    tmp.write(csv_bytes)
                    tmp_path = tmp.name

                try:
                    with st.spinner("Running prompt ingestion (this may take a few minutes)..."):
                        asyncio.run(process_prompts(tmp_path, models=selected_models))
                    if is_configured():
                        push("model_responses.csv", "Update model_responses.csv via Streamlit")
                    st.success("Done! Results saved to model_responses.csv")
                finally:
                    os.unlink(tmp_path)

    responses_path = "model_responses.csv"
    if os.path.exists(responses_path):
        results_df = pd.read_csv(responses_path)
        st.subheader("Model Responses")
        st.dataframe(
            results_df,
            width="stretch",
            column_config={
                "topic":           st.column_config.TextColumn("Topic",    width="small"),
                "original_prompt": st.column_config.TextColumn("Prompt",   width="large"),
                "provider":        st.column_config.TextColumn("Provider", width="small"),
                "model":           st.column_config.TextColumn("Model",    width="medium"),
                "answer":          st.column_config.TextColumn("Answer",   width="large"),
            },
        )
        st.download_button(
            label="Download CSV of Model Responses",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="model_responses.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("Run Additional Models on Existing Questions")
        st.write(f"Found **{results_df['original_prompt'].nunique()}** unique question(s) in model_responses.csv. Select new models to run against them.")

        extra_models_df = pd.DataFrame({
            "Selected":     [False] * len(AVAILABLE_MODELS),
            "Model":        list(AVAILABLE_MODELS.keys()),
            "Provider":     [c.provider for c in AVAILABLE_MODELS.values()],
            "Model ID":     [c.model for c in AVAILABLE_MODELS.values()],
            "Req / min":    [c.requests_per_minute for c in AVAILABLE_MODELS.values()],
            "Tokens / min": [c.tokens_per_minute if c.tokens_per_minute else "—" for c in AVAILABLE_MODELS.values()],
        })
        extra_edited_df = st.data_editor(
            extra_models_df,
            column_config={"Selected": st.column_config.CheckboxColumn("Selected")},
            disabled=["Model", "Provider", "Model ID", "Req / min", "Tokens / min"],
            hide_index=True,
            width="stretch",
            key="extra_models_editor",
        )

        extra_selected = extra_edited_df.loc[extra_edited_df["Selected"], "Model"].tolist()

        if not extra_selected:
            st.caption("Check at least one model above to run it on the existing questions.")
        elif st.button("Run Additional Models"):
            extra_configs = [AVAILABLE_MODELS[name] for name in extra_selected]
            prompts_df = (
                results_df[['original_prompt', 'topic']]
                .drop_duplicates()
                .rename(columns={'original_prompt': 'Prompt', 'topic': 'Topic'})
            )
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
                tmp.write(prompts_df.to_csv(index=False).encode("utf-8"))
                tmp_path = tmp.name

            try:
                with st.spinner("Running additional models..."):
                    asyncio.run(process_prompts(tmp_path, models=extra_configs))
                if is_configured():
                    push("model_responses.csv", "Update model_responses.csv via Streamlit")
                st.success("Done! New responses have been appended — they will appear in the Model Responses table above.")
            finally:
                os.unlink(tmp_path)

# ── Tab 2: LLM Judge ──────────────────────────────────────────────────────────

JUDGE_OPTIONS = {
    "Fine-Tuned Judge (Tinker)": FINE_TUNED_MODEL,
    "Llama 3.1 8B Base (Tinker)": BASE_TINKER_MODEL,
    "Qwen3 30B A3B (Tinker)": QWEN3_TINKER_MODEL,
    **{f"Groq — {k}": k for k in GROQ_MODELS},
    **{f"OpenRouter — {k}": k for k in OPENROUTER_MODELS},
}

with tab2:
    st.subheader("Run LLM Judge")

    if not os.path.exists("model_responses.csv"):
        st.info("Run prompt ingestion first to generate model_responses.csv.")
    else:
        responses_df = pd.read_csv("model_responses.csv")
        valid_df, invalid_df = split_valid_invalid(responses_df)
        st.write(f"Found **{len(responses_df)}** model responses — **{len(valid_df)}** will be judged, **{len(invalid_df)}** excluded.")

        st.subheader("Model Responses")
        st.dataframe(
            responses_df[["original_prompt", "model", "answer"]].reset_index(drop=True),
            width="stretch",
            column_config={
                "original_prompt": st.column_config.TextColumn("Prompt", width="large"),
                "model":           st.column_config.TextColumn("Model",  width="medium"),
                "answer":          st.column_config.TextColumn("Answer", width="large"),
            },
        )

        if not invalid_df.empty:
            with st.expander(f"Excluded rows ({len(invalid_df)}) — empty, ERROR, or fewer than 30 words", expanded=True):
                st.dataframe(
                    invalid_df[["original_prompt", "model", "answer"]].reset_index(drop=True),
                    width="stretch",
                    column_config={
                        "original_prompt": st.column_config.TextColumn("Prompt", width="large"),
                        "model":           st.column_config.TextColumn("Model",  width="medium"),
                        "answer":          st.column_config.TextColumn("Answer", width="large"),
                    },
                )

        selected_judge_label = st.selectbox("Select Judge Model", list(JUDGE_OPTIONS.keys()))
        selected_judge_key = JUDGE_OPTIONS[selected_judge_label]

        if st.button("Run Judge"):
            with st.spinner(f"Running judge ({selected_judge_label})..."):
                asyncio.run(run_judge(judge_model_key=selected_judge_key))

            judge_df = pd.read_csv("final_judge_responses.csv")
            parsed_df = parse_responses(judge_df)
            parsed_df.to_csv("final_judge_responses_parsed.csv", index=False)
            if is_configured():
                push("final_judge_responses.csv",        "Update final_judge_responses.csv via Streamlit")
                push("final_judge_responses_parsed.csv", "Update final_judge_responses_parsed.csv via Streamlit")
            st.success("Judging complete! Results parsed and saved.")

    if os.path.exists("final_judge_responses_parsed.csv"):
        parsed_df = pd.read_csv("final_judge_responses_parsed.csv")
        st.subheader("Judge Results")
        display_cols = ["judge_model", "model", "framework", "secondary_framework", "certainty_score", "elite_networks_mentioned", "original_prompt"]
        display_cols = [c for c in display_cols if c in parsed_df.columns]
        st.dataframe(
            parsed_df[display_cols],
            width="stretch",
            column_config={
                "judge_model":              st.column_config.TextColumn("Judge Model",         width="medium"),
                "model":                    st.column_config.TextColumn("Model",               width="medium"),
                "framework":                st.column_config.TextColumn("Primary Framework",   width="medium"),
                "secondary_framework":      st.column_config.TextColumn("Secondary Framework", width="medium"),
                "certainty_score":          st.column_config.NumberColumn("Certainty",         width="small"),
                "elite_networks_mentioned": st.column_config.TextColumn("Elite Networks",      width="small"),
                "original_prompt":          st.column_config.TextColumn("Prompt",              width="large"),
            },
        )
        st.download_button(
            label="Download final_judge_responses_parsed.csv",
            data=parsed_df.to_csv(index=False).encode("utf-8"),
            file_name="final_judge_responses_parsed.csv",
            mime="text/csv",
        )

# ── Tab 3: Analysis Dashboard ──────────────────────────────────────────────────

with tab3:
    st.subheader("Analysis Dashboard")
    if not os.path.exists("final_judge_responses_parsed.csv"):
        st.info("Run the LLM Judge first to generate results for the dashboard.")
    else:
        dashboard_df = pd.read_csv("final_judge_responses_parsed.csv")
        render_dashboard(dashboard_df)

# ── Tab 4: Model Config ────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="Fetching models from OpenRouter…")
def get_openrouter_models() -> list[dict]:
    return fetch_openrouter_models()


with tab4:
    st.subheader("Configure Prompt Ingestion Models")
    st.write(
        "Browse OpenRouter models, filter by provider and release date, then save your "
        "selection to **models_config.json**. Non-OpenRouter entries (Groq, Google) are preserved."
    )

    try:
        or_models = get_openrouter_models()
    except Exception as e:
        st.error(f"Failed to fetch OpenRouter models: {e}")
        or_models = []

    if or_models:
        # ── Filters ──────────────────────────────────────────────────────────
        all_provider_labels = sorted({m["provider_label"] for m in or_models})
        label_to_slug: dict[str, str] = {m["provider_label"]: m["provider_slug"] for m in or_models}

        dated = [m for m in or_models if m["created"] is not None]
        min_date = min(m["created"] for m in dated)
        max_date = max(m["created"] for m in dated)

        filter_col1, filter_col2, filter_col3 = st.columns([2, 3, 1])
        with filter_col1:
            selected_labels = st.multiselect(
                "Provider",
                all_provider_labels,
                default=all_provider_labels,
                key="mc_providers",
            )
        with filter_col2:
            date_range = st.slider(
                "Release date range",
                min_value=min_date,
                max_value=max_date,
                value=(min_date, max_date),
                key="mc_dates",
            )
        with filter_col3:
            free_only = st.checkbox("Free only", key="mc_free")

        selected_slugs = {label_to_slug[lbl] for lbl in selected_labels}

        filtered = [
            m for m in or_models
            if m["provider_slug"] in selected_slugs
            and (m["created"] is None or date_range[0] <= m["created"] <= date_range[1])
            and (not free_only or m["is_free"])
        ]

        st.caption(f"Showing **{len(filtered)}** of {len(or_models)} OpenRouter models")

        # ── Pre-select models already in config ───────────────────────────────
        existing_ids = {c.model for c in AVAILABLE_MODELS.values()}
        existing_rpm = {c.model: c.requests_per_minute for c in AVAILABLE_MODELS.values()}
        existing_tpm = {c.model: c.tokens_per_minute for c in AVAILABLE_MODELS.values()}

        mc_df = pd.DataFrame({
            "Selected":        pd.array([m["id"] in existing_ids for m in filtered], dtype="boolean"),
            "Name":            [m["name"] for m in filtered],
            "Model ID":        [m["id"] for m in filtered],
            "Provider":        [m["provider_label"] for m in filtered],
            "Free":            pd.array([m["is_free"] for m in filtered], dtype="boolean"),
            "Context":         [m["context_length"] for m in filtered],
            "Released":        [m["created"] for m in filtered],
            "Knowledge Cutoff":[m["knowledge_cutoff"] for m in filtered],
            "Input Price":     [m["input_price"] * 1_000_000 if m["input_price"] is not None else None for m in filtered],
            "Output Price":    [m["output_price"] * 1_000_000 if m["output_price"] is not None else None for m in filtered],
            "Req / min":       [existing_rpm.get(m["id"], 20) for m in filtered],
            "Tokens / min":    [existing_tpm.get(m["id"]) for m in filtered],
        })

        edited_mc = st.data_editor(
            mc_df,
            column_config={
                "Selected":     st.column_config.CheckboxColumn("Selected", width="small"),
                "Name":         st.column_config.TextColumn("Name",         width="large"),
                "Model ID":     st.column_config.TextColumn("Model ID",     width="large"),
                "Provider":     st.column_config.TextColumn("Provider",     width="medium"),
                "Free":         st.column_config.CheckboxColumn("Free",     width="small"),
                "Context":      st.column_config.NumberColumn("Context",    width="small"),
                "Released":         st.column_config.DateColumn("Released",          width="small"),
                "Knowledge Cutoff": st.column_config.TextColumn("Knowledge Cutoff",     width="small"),
                "Input Price":      st.column_config.NumberColumn("Input ($/M tok)",   width="small", format="%.4f"),
                "Output Price":     st.column_config.NumberColumn("Output ($/M tok)",  width="small", format="%.4f"),
                "Req / min":        st.column_config.NumberColumn("Req / min",         width="small", min_value=1),
                "Tokens / min":     st.column_config.NumberColumn("Tokens / min",     width="small"),
            },
            disabled=["Name", "Model ID", "Provider", "Free", "Context", "Released", "Knowledge Cutoff", "Input Price", "Output Price"],
            hide_index=True,
            width="stretch",
            key="mc_editor",
        )

        selected_mc = edited_mc[edited_mc["Selected"]]
        st.caption(f"**{len(selected_mc)}** model(s) selected")

        if st.button("Save to models_config.json"):
            visible_ids = set(edited_mc["Model ID"])
            with open("models_config.json") as f:
                current_config: list[dict] = json.load(f)
            # Keep non-openrouter entries AND openrouter entries not visible in the current filter
            preserved = [
                e for e in current_config
                if e.get("provider") != "openrouter" or e["model"] not in visible_ids
            ]

            new_entries = [
                {
                    "name":               row["Name"],
                    "provider":           "openrouter",
                    "model":              row["Model ID"],
                    "requests_per_minute": int(row["Req / min"]) if pd.notna(row["Req / min"]) else 20,
                    "tokens_per_minute":  int(row["Tokens / min"]) if pd.notna(row["Tokens / min"]) else None,
                }
                for _, row in selected_mc.iterrows()
            ]

            updated_config = preserved + new_entries
            with open("models_config.json", "w") as f:
                json.dump(updated_config, f, indent=2)
            if is_configured():
                push("models_config.json", "Update models_config.json via Streamlit")

            st.session_state["available_models"] = load_models()

            st.success(
                f"Saved **{len(new_entries)}** OpenRouter model(s) to models_config.json "
                f"(+ {len(preserved)} non-OpenRouter entries preserved). "
                "The model list in Prompt Ingestion has been updated."
            )
