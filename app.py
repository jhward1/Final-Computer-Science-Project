import streamlit as st
import asyncio
import tempfile
import os
import pandas as pd

from prompt_ingestion import process_prompts, ModelConfig

# Dummy model options — replace with real models and their actual rate limits later.
# tokens_per_minute=None means no token limiting (e.g. OpenRouter free tier).
AVAILABLE_MODELS: dict[str, ModelConfig] = {
    "Qwen 3 Next 80 B":      ModelConfig("openrouter", "qwen/qwen3-next-80b-a3b-instruct:free",      requests_per_minute=20),
    "Llama 3.3 70B":      ModelConfig("openrouter", "meta-llama/llama-3.3-70b-instruct:free",       requests_per_minute=20),
    "GPT OSS 20B":     ModelConfig("openrouter", "openai/gpt-oss-20b:free",      requests_per_minute=20),
    "Groq Llama 3.1 8B":     ModelConfig("groq",       "llama-3.1-8b-instant",      requests_per_minute=30, tokens_per_minute=6_000),
    "Gemini 2.5 Flash": ModelConfig("google",     "gemini-2.5-flash",  requests_per_minute=5, tokens_per_minute=1_000_000),
}

st.title("Prompt Ingestion Tool")
st.write("Upload a CSV with **Prompt** and **Topic** columns to query the selected models.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df_preview = pd.read_csv(uploaded_file)
    st.subheader("Preview")
    st.dataframe(df_preview.head(10), width='stretch')

    missing = [c for c in ["Prompt", "Topic"] if c not in df_preview.columns]
    if missing:
        st.error(f"CSV is missing required column(s): {', '.join(missing)}")
    else:
        st.subheader("Select Models")
        models_df = pd.DataFrame({
            "Selected":          [True] * len(AVAILABLE_MODELS),
            "Model":             list(AVAILABLE_MODELS.keys()),
            "Provider":          [c.provider for c in AVAILABLE_MODELS.values()],
            "Model ID":          [c.model for c in AVAILABLE_MODELS.values()],
            "Req / min":         [c.requests_per_minute for c in AVAILABLE_MODELS.values()],
            "Tokens / min":      [c.tokens_per_minute if c.tokens_per_minute else "—" for c in AVAILABLE_MODELS.values()],
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

                responses_path = "model_responses.csv"
                if os.path.exists(responses_path):
                    results_df = pd.read_csv(responses_path)
                    st.subheader("Results")
                    st.dataframe(results_df)

                    csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="Download model_responses.csv",
                        data=csv_bytes,
                        file_name="model_responses.csv",
                        mime="text/csv",
                    )
            finally:
                os.unlink(tmp_path)
