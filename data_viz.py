import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SOURCE_FILE = 'final_judge_responses_parsed.csv'

FRAMEWORK_PALETTE = {
    "Geopolitical":          "#4C72B0",
    "Sociological":          "#DD8452",
    "Economic Protectionism":"#55A868",
    "None":                  "#C0C0C0",
}

def _build_palette(values) -> dict:
    """Return a palette covering all values, falling back to grey for unknowns."""
    return {v: FRAMEWORK_PALETTE.get(v, "#AAAAAA") for v in values}


def render_dashboard(df: pd.DataFrame):
    df['certainty_score'] = pd.to_numeric(df['certainty_score'], errors='coerce')
    df['elite_networks_mentioned'] = df['elite_networks_mentioned'].astype(str)

    dash_tab1, dash_tab2 = st.tabs(["Analysis", "Model Comparison"])

    # ── Analysis tab ──────────────────────────────────────────────────────────
    with dash_tab1:
        st.markdown("Use the filters below to explore the model responses and judge evaluations.")

        all_judge_models = df['judge_model'].dropna().unique().tolist() if 'judge_model' in df.columns else []
        selected_judge_model = st.selectbox("Judge Model", all_judge_models if all_judge_models else [None])

        with st.expander("Filter Options", expanded=True):
            f_col1, f_col2, f_col3 = st.columns(3)

            all_models = df['model'].unique().tolist()
            selected_models = f_col1.multiselect("Model(s)", all_models, default=all_models)

            all_frameworks = df['framework'].dropna().unique().tolist()
            selected_frameworks = f_col2.multiselect("Framework(s)", all_frameworks, default=all_frameworks)

            elite_options = df['elite_networks_mentioned'].unique().tolist()
            selected_elite = f_col3.multiselect("Elite Networks Mentioned", elite_options, default=elite_options)

            min_score = float(df['certainty_score'].min()) if not df['certainty_score'].isnull().all() else 0.0
            max_score = float(df['certainty_score'].max()) if not df['certainty_score'].isnull().all() else 5.0
            if max_score <= min_score:
                max_score = min_score + 1.0
            score_range = st.slider("Certainty Score Range", min_score, max_score, (min_score, max_score))

            evidence_search = st.text_input("Search Evidence Spans (Keyword)")

        filtered_df = df[
            (df['model'].isin(selected_models)) &
            (df['elite_networks_mentioned'].isin(selected_elite)) &
            (df['certainty_score'] >= score_range[0]) &
            (df['certainty_score'] <= score_range[1])
        ]
        if selected_frameworks:
            filtered_df = filtered_df[filtered_df['framework'].isin(selected_frameworks)]
        if 'judge_model' in df.columns and selected_judge_model:
            filtered_df = filtered_df[filtered_df['judge_model'] == selected_judge_model]
        if evidence_search:
            filtered_df = filtered_df[filtered_df['evidence_spans'].str.contains(evidence_search, case=False, na=False)]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Rows", len(df))
        col2.metric("Filtered Rows", len(filtered_df))
        col3.metric("Avg Certainty", round(filtered_df['certainty_score'].mean(), 2) if not filtered_df.empty else 0)

        st.divider()
        st.subheader("Visual Analysis")
        graph_col1, graph_col2, graph_col3, graph_col4 = st.columns(4)

        with graph_col1:
            st.markdown("**Avg Certainty Score per Model**")
            if not filtered_df.empty:
                avg_certainty = filtered_df.groupby('model')['certainty_score'].mean().reset_index()
                avg_certainty.columns = ['model', 'avg_certainty']
                fig1, ax1 = plt.subplots()
                sns.barplot(data=avg_certainty, y='model', x='avg_certainty', ax=ax1, hue='model', palette='Blues_d', legend=False)
                ax1.set_xlabel("Avg Certainty Score")
                ax1.set_ylabel("Model")
                ax1.set_xlim(0, max_score)
                fig1.tight_layout()
                st.pyplot(fig1)
                plt.close(fig1)
            else:
                st.info("No data to display.")

        with graph_col2:
            st.markdown("**Framework Count per Model**")
            if not filtered_df.empty:
                framework_counts = (
                    filtered_df.dropna(subset=['framework'])
                    .groupby(['model', 'framework'])
                    .size()
                    .reset_index(name='count')
                )
                fig2, ax2 = plt.subplots()
                sns.barplot(data=framework_counts, x='model', y='count', hue='framework', ax=ax2, palette=_build_palette(framework_counts['framework'].unique()))
                ax2.set_xlabel("Model")
                ax2.set_ylabel("Count")
                ax2.tick_params(axis='x', rotation=30)
                ax2.legend(fontsize=7, loc='upper right', title='Framework', title_fontsize=7)
                fig2.tight_layout()
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.info("No data to display.")

        with graph_col3:
            st.markdown("**Secondary Framework Count per Model**")
            if not filtered_df.empty:
                sec_df = filtered_df.copy()
                sec_df['secondary_framework'] = (
                    sec_df['secondary_framework']
                    .fillna('None')
                    .apply(lambda x: 'None' if str(x).strip().lower() in ('null', 'none', '') else x)
                )
                sec_counts = (
                    sec_df.groupby(['model', 'secondary_framework'])
                    .size()
                    .reset_index(name='count')
                )
                fig3, ax3 = plt.subplots()
                sns.barplot(data=sec_counts, x='model', y='count', hue='secondary_framework', ax=ax3, palette=_build_palette(sec_counts['secondary_framework'].unique()))
                ax3.set_xlabel("Model")
                ax3.set_ylabel("Count")
                ax3.tick_params(axis='x', rotation=30)
                ax3.legend(fontsize=7, loc='upper right', title='Secondary Framework', title_fontsize=7)
                fig3.tight_layout()
                st.pyplot(fig3)
                plt.close(fig3)
            else:
                st.info("No data to display.")

        with graph_col4:
            st.markdown("**Elite Networks Mentioned per Model**")
            if not filtered_df.empty:
                elite_counts = (
                    filtered_df.groupby(['model', 'elite_networks_mentioned'])
                    .size()
                    .reset_index(name='count')
                )
                fig4, ax4 = plt.subplots()
                sns.barplot(data=elite_counts, x='model', y='count', hue='elite_networks_mentioned', ax=ax4)
                ax4.set_xlabel("Model")
                ax4.set_ylabel("Count")
                ax4.tick_params(axis='x', rotation=30)
                ax4.legend(fontsize=7, loc='upper right', title='Elite Networks', title_fontsize=7)
                fig4.tight_layout()
                st.pyplot(fig4)
                plt.close(fig4)
            else:
                st.info("No data to display.")

        st.subheader("Filtered Results")
        st.caption("Click a row to see full details below.")
        display_cols = ['model', 'framework', 'secondary_framework', 'certainty_score', 'elite_networks_mentioned', 'original_prompt']
        selection = st.dataframe(
            filtered_df[display_cols].reset_index(drop=True),
            width="stretch",
            selection_mode="single-row",
            on_select="rerun",
            column_config={
                "model":                    st.column_config.TextColumn("Model",                width="medium"),
                "framework":                st.column_config.TextColumn("Primary Framework",    width="medium"),
                "secondary_framework":      st.column_config.TextColumn("Secondary Framework",  width="medium"),
                "certainty_score":          st.column_config.NumberColumn("Certainty",          width="small"),
                "elite_networks_mentioned": st.column_config.TextColumn("Elite Networks",       width="small"),
                "original_prompt":          st.column_config.TextColumn("Prompt",               width="large"),
            },
        )

        selected_rows = selection.selection.rows if not filtered_df.empty else []
        if selected_rows:
            st.divider()
            st.subheader("Detailed Response View")
            row = filtered_df.iloc[selected_rows[0]]

            detail_tab1, detail_tab2 = st.tabs(["Original Content", "Judge Analysis"])
            with detail_tab1:
                st.write("**Prompt:**", row['original_prompt'])
                st.write("**Model Answer:**")
                st.info(row['answer'])
            with detail_tab2:
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Primary Framework", row['framework'])
                m_col2.metric("Secondary Framework", row['secondary_framework'])
                m_col3.metric("Certainty Score", row['certainty_score'])
                st.markdown(
                    '<span style="background-color:#fff3cd;padding:2px 6px;border-radius:4px;">■</span> Certainty spans &nbsp;&nbsp;'
                    '<span style="background-color:#cce5ff;padding:2px 6px;border-radius:4px;">■</span> Framework spans',
                    unsafe_allow_html=True,
                )
                st.markdown(f'<div style="line-height: 1.6; margin-top: 8px;">{row["html_highlighted_answer"]}</div>', unsafe_allow_html=True)

    # ── Model Comparison tab ───────────────────────────────────────────────────
    with dash_tab2:
        st.markdown("Select a question and models to compare answers side by side.")

        all_questions = df['original_prompt'].unique().tolist()
        selected_question = st.selectbox("Select a question", all_questions)

        all_models = df['model'].unique().tolist()
        compare_models = st.multiselect("Select models to compare (max 3)", all_models, default=all_models[:3], max_selections=3)

        if not compare_models:
            st.warning("Select at least one model to compare.")
        else:
            question_df = df[
                (df['original_prompt'] == selected_question) &
                (df['model'].isin(compare_models))
            ]

            if question_df.empty:
                st.info("No responses found for this question and model selection.")
            else:
                cols = st.columns(len(compare_models))
                for col, model in zip(cols, compare_models):
                    row = question_df[question_df['model'] == model]
                    with col:
                        st.markdown(f"**{model}**")
                        if row.empty:
                            st.warning("No response.")
                        else:
                            r = row.iloc[0]
                            st.info(r['answer'])
                            st.caption(f"Primary: {r['framework']} | Secondary: {r['secondary_framework']} | Certainty: {r['certainty_score']} | Elite Networks: {r['elite_networks_mentioned']}")


if __name__ == "__main__":
    st.set_page_config(page_title="LLM Bias Analysis Dashboard", layout="wide")
    st.title("LLM Bias Analysis & Scoring Dashboard")
    df = pd.read_csv(SOURCE_FILE)
    render_dashboard(df)
