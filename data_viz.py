import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SOURCE_FILE = 'final_judge_responses_parsed.csv'


def render_dashboard(df: pd.DataFrame):
    st.markdown("Use the filters below to explore the model responses and judge evaluations.")

    df['certainty_score'] = pd.to_numeric(df['certainty_score'], errors='coerce')
    df['elite_networks_mentioned'] = df['elite_networks_mentioned'].astype(str)

    # --- INLINE FILTERS ---
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
        score_range = st.slider("Certainty Score Range", min_score, max_score, (min_score, max_score))

        evidence_search = st.text_input("Search Evidence Spans (Keyword)")

    # --- APPLY FILTERS ---
    filtered_df = df[
        (df['model'].isin(selected_models)) &
        (df['elite_networks_mentioned'].isin(selected_elite)) &
        (df['certainty_score'] >= score_range[0]) &
        (df['certainty_score'] <= score_range[1])
    ]
    if selected_frameworks:
        filtered_df = filtered_df[filtered_df['framework'].isin(selected_frameworks)]
    if evidence_search:
        filtered_df = filtered_df[filtered_df['evidence_spans'].str.contains(evidence_search, case=False, na=False)]

    # --- METRICS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", len(df))
    col2.metric("Filtered Rows", len(filtered_df))
    col3.metric("Avg Certainty", round(filtered_df['certainty_score'].mean(), 2) if not filtered_df.empty else 0)

    # --- GRAPHS ---
    st.divider()
    st.subheader("Visual Analysis")
    graph_col1, graph_col2, graph_col3 = st.columns(3)

    with graph_col1:
        st.markdown("**Avg Certainty Score per Model**")
        if not filtered_df.empty:
            avg_certainty = filtered_df.groupby('model')['certainty_score'].mean().reset_index()
            avg_certainty.columns = ['model', 'avg_certainty']
            fig1, ax1 = plt.subplots()
            sns.barplot(data=avg_certainty, y='model', x='avg_certainty', ax=ax1, palette='Blues_d')
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
            sns.barplot(data=framework_counts, x='model', y='count', hue='framework', ax=ax2)
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
        st.markdown("**Elite Networks Mentioned per Model**")
        if not filtered_df.empty:
            elite_counts = (
                filtered_df.groupby(['model', 'elite_networks_mentioned'])
                .size()
                .reset_index(name='count')
            )
            fig3, ax3 = plt.subplots()
            sns.barplot(data=elite_counts, x='model', y='count', hue='elite_networks_mentioned', ax=ax3)
            ax3.set_xlabel("Model")
            ax3.set_ylabel("Count")
            ax3.tick_params(axis='x', rotation=30)
            ax3.legend(fontsize=7, loc='upper right', title='Elite Networks', title_fontsize=7)
            fig3.tight_layout()
            st.pyplot(fig3)
            plt.close(fig3)
        else:
            st.info("No data to display.")

    # --- TABLE ---
    st.subheader("Filtered Results")
    st.caption("Click a row to see full details below.")
    display_cols = ['model', 'framework', 'certainty_score', 'elite_networks_mentioned', 'original_prompt']
    selection = st.dataframe(
        filtered_df[display_cols].reset_index(drop=True),
        use_container_width=True,
        selection_mode="single-row",
        on_select="rerun",
        column_config={
            "model":                   st.column_config.TextColumn("Model",            width="medium"),
            "framework":               st.column_config.TextColumn("Framework",        width="medium"),
            "certainty_score":         st.column_config.NumberColumn("Certainty",      width="small"),
            "elite_networks_mentioned":st.column_config.TextColumn("Elite Networks",   width="small"),
            "original_prompt":         st.column_config.TextColumn("Prompt",           width="large"),
        },
    )

    # --- DETAIL VIEW ---
    selected_rows = selection.selection.rows if not filtered_df.empty else []
    if selected_rows:
        st.divider()
        st.subheader("Detailed Response View")
        row = filtered_df.iloc[selected_rows[0]]

        tab1, tab2, tab3 = st.tabs(["Original Content", "Judge Analysis", "Evidence Spans"])
        with tab1:
            st.write("**Prompt:**", row['original_prompt'])
            st.write("**Model Answer:**")
            st.info(row['answer'])
        with tab2:
            st.write("**Framework:**", row['framework'])
            st.write("**Certainty Score:**", row['certainty_score'])
            st.write("**Reasoning:**")
            st.write(row['reasoning'])
        with tab3:
            st.write("**Evidence Spans extracted by Judge:**")
            st.markdown(f'<div style="line-height: 1.6;">{row["html_highlighted_answer"]}</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    st.set_page_config(page_title="LLM Bias Analysis Dashboard", layout="wide")
    st.title("LLM Bias Analysis & Scoring Dashboard")
    df = pd.read_csv(SOURCE_FILE)
    render_dashboard(df)
