import streamlit as st
import asyncio
import tempfile
import os
import pandas as pd

from prompt_ingestion import process_prompts, ModelConfig
from llm_judge import process_prompts as run_judge, GROQ_MODELS, OPENROUTER_MODELS, FINE_TUNED_MODEL
from data_cleaning import parse_responses
from data_viz import render_dashboard

st.set_page_config(page_title="LLM Bias Analysis Tool", layout="wide")
st.title("LLM Bias Analysis Tool")

tab1, tab2, tab3 = st.tabs(["1. Prompt Ingestion", "2. LLM Judge", "3. Analysis Dashboard"])

# ── Tab 1: Prompt Ingestion ────────────────────────────────────────────────────

AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "Qwen 3 Next 80 B":  ModelConfig("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free",      requests_per_minute=20),
    "Llama 3.3 70B":     ModelConfig("openrouter", "meta-llama/llama-3.3-70b-instruct:free",     requests_per_minute=20),
    "GPT OSS 20B":       ModelConfig("openrouter", "openai/gpt-oss-20b:free",                    requests_per_minute=20),
    "Groq Llama 3.1 8B": ModelConfig("groq",       "llama-3.1-8b-instant", requests_per_minute=30, tokens_per_minute=6_000),
    "Gemini 2.5 Flash":  ModelConfig("google",     "gemini-2.5-flash",     requests_per_minute=5, tokens_per_minute=1_000_000),
}

with tab1:
    st.subheader("Upload Prompts CSV")
    st.write("Upload a CSV with **Prompt** and **Topic** columns to query the selected models.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df_preview = pd.read_csv(uploaded_file)
        st.subheader("Preview")
        st.dataframe(
            df_preview.head(10),
            use_container_width=True,
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
                use_container_width=True,
            )

            selected_names = edited_df.loc[edited_df["Selected"], "Model"].tolist()

            if not selected_names:
                st.warning("Select at least one model to continue.")
            elif st.button("Run Prompt Ingestion"):
                selected_models = [AVAILABLE_MODELS[name] for name in selected_names]

                with tempfile.NamedTemporaryFile(delete=False, suffix=".csv", mode="wb") as tmp:
                    uploaded_file.seek(0)
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    with st.spinner("Running prompt ingestion (this may take a few minutes)..."):
                        asyncio.run(process_prompts(tmp_path, models=selected_models))
                    st.success("Done! Results saved to model_responses.csv")
                finally:
                    os.unlink(tmp_path)

    responses_path = "model_responses.csv"
    if os.path.exists(responses_path):
        results_df = pd.read_csv(responses_path)
        st.subheader("Model Responses")
        st.dataframe(
            results_df,
            use_container_width=True,
            column_config={
                "topic":           st.column_config.TextColumn("Topic",    width="small"),
                "original_prompt": st.column_config.TextColumn("Prompt",   width="large"),
                "provider":        st.column_config.TextColumn("Provider", width="small"),
                "model":           st.column_config.TextColumn("Model",    width="medium"),
                "answer":          st.column_config.TextColumn("Answer",   width="large"),
            },
        )
        st.download_button(
            label="Download model_responses.csv",
            data=results_df.to_csv(index=False).encode("utf-8"),
            file_name="model_responses.csv",
            mime="text/csv",
        )

# ── Tab 2: LLM Judge ──────────────────────────────────────────────────────────

JUDGE_OPTIONS = {
    "Fine-Tuned Judge": FINE_TUNED_MODEL,
    **{f"Groq — {k}": k for k in GROQ_MODELS},
    **{f"OpenRouter — {k}": k for k in OPENROUTER_MODELS},
}

with tab2:
    st.subheader("Run LLM Judge")

    if not os.path.exists("model_responses.csv"):
        st.info("Run prompt ingestion first to generate model_responses.csv.")
    else:
        responses_df = pd.read_csv("model_responses.csv")
        st.write(f"Found **{len(responses_df)}** model responses to judge.")
        st.dataframe(
            responses_df[["original_prompt", "model", "answer"]].head(5),
            use_container_width=True,
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
            st.success("Judging complete! Results parsed and saved.")

    if os.path.exists("final_judge_responses_parsed.csv"):
        parsed_df = pd.read_csv("final_judge_responses_parsed.csv")
        st.subheader("Judge Results")
        st.dataframe(
            parsed_df[["model", "framework", "certainty_score", "elite_networks_mentioned", "original_prompt"]],
            use_container_width=True,
            column_config={
                "model":                    st.column_config.TextColumn("Model",         width="medium"),
                "framework":                st.column_config.TextColumn("Framework",     width="medium"),
                "certainty_score":          st.column_config.NumberColumn("Certainty",   width="small"),
                "elite_networks_mentioned": st.column_config.TextColumn("Elite Networks",width="small"),
                "original_prompt":          st.column_config.TextColumn("Prompt",        width="large"),
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
