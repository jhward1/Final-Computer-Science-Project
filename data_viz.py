import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json


SOURCE_FILE = 'final_judge_responses_parsed.csv' # This file should have columns: model, framework, certainty_score, elite_networks_mentioned, original_prompt, answer, reasoning, evidence_spans, html_highlighted_answer

# Set page configuration
st.set_page_config(page_title="LLM Bias Analysis Dashboard", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv(SOURCE_FILE)
    # Convert certainty_score to numeric, handling potential errors
    df['certainty_score'] = pd.to_numeric(df['certainty_score'], errors='coerce')
    # Ensure elite_networks_mentioned is treated consistently
    df['elite_networks_mentioned'] = df['elite_networks_mentioned'].astype(str)
    return df

df = load_data()

st.title("LLM Bias Analysis & Scoring Dashboard")
st.markdown("Use the filters on the left to explore the model responses and judge evaluations.")

# --- SIDEBAR FILTERS ---
st.sidebar.header("Filter Options")

# 1. Model Filter
all_models = df['model'].unique().tolist()
selected_models = st.sidebar.multiselect("Select Model(s)", all_models, default=all_models)

# 2. Framework Filter
all_frameworks = df['framework'].dropna().unique().tolist()
selected_frameworks = st.sidebar.multiselect("Select Framework(s)", all_frameworks, default=all_frameworks)

# 3. Certainty Score Filter
min_score = float(df['certainty_score'].min()) if not df['certainty_score'].isnull().all() else 0.0
max_score = float(df['certainty_score'].max()) if not df['certainty_score'].isnull().all() else 5.0
score_range = st.sidebar.slider("Certainty Score Range", min_score, max_score, (min_score, max_score))

# 4. Elite Networks Mentioned
elite_options = df['elite_networks_mentioned'].unique().tolist()
selected_elite = st.sidebar.multiselect("Elite Networks Mentioned", elite_options, default=elite_options)

# 5. Evidence Spans Keyword Search
evidence_search = st.sidebar.text_input("Search Evidence Spans (Keyword)")

# --- APPLY FILTERS ---
filtered_df = df[
    (df['model'].isin(selected_models)) &
    (df['elite_networks_mentioned'].isin(selected_elite)) &
    (df['certainty_score'] >= score_range[0]) &
    (df['certainty_score'] <= score_range[1])
]

# Optional filter for frameworks (handling NaNs)
if selected_frameworks:
    filtered_df = filtered_df[filtered_df['framework'].isin(selected_frameworks)]

# Filter for evidence spans keyword
if evidence_search:
    filtered_df = filtered_df[filtered_df['evidence_spans'].str.contains(evidence_search, case=False, na=False)]

# --- MAIN DISPLAY ---

# Metrics Overview
col1, col2, col3 = st.columns(3)
col1.metric("Total Rows", len(df))
col2.metric("Filtered Rows", len(filtered_df))
col3.metric("Avg Certainty", round(filtered_df['certainty_score'].mean(), 2) if not filtered_df.empty else 0)

# --- GRAPHS ---
st.divider()
st.subheader("Visual Analysis")

graph_col1, graph_col2, graph_col3 = st.columns(3)

# Graph 1: Average certainty score per model
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

# Graph 2: Total count of frameworks per model
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

# Graph 3: Elite networks mentioned per model
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

# Data Table
st.subheader("Filtered Results")
st.dataframe(filtered_df[['model', 'framework', 'certainty_score', 'elite_networks_mentioned', 'original_prompt']], use_container_width=True)

# Detailed View Section
st.divider()
st.subheader("Detailed Response View")
if not filtered_df.empty:
    selected_index = st.selectbox("Select a row to see full details:", filtered_df.index)
    row = filtered_df.loc[selected_index]
    
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
        evidence = row['html_highlighted_answer']
        st.markdown(f'<div style="line-height: 1.6;">{evidence}</div>', unsafe_allow_html=True)
        # Try to parse as list if it's a string representation
        #try:
        #    if isinstance(evidence, str) and evidence.startswith('['):
        #        evidence_list = eval(evidence)
        #        for item in evidence_list:
        #            st.success(f"- {item}")
        #    else:
        #        st.write(evidence)
        #except:
        #    st.write(evidence)
else:
    st.warning("No data matches the current filters.")

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("LLM Bias Analysis Tool v1.0")