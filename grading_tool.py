import streamlit as st
import pandas as pd
import json
import os

OUTPUT_FILE = "human_grades.json"
SOURCE_FILE = "training_set.csv"

st.set_page_config(page_title="Response Grader", layout="centered")


def load_grades():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            return json.load(f)
    return {}


def save_grade(idx, grade):
    grades = load_grades()
    grades[str(idx)] = grade
    with open(OUTPUT_FILE, "w") as f:
        json.dump(grades, f, indent=2)


@st.cache_data
def load_responses():
    return pd.read_csv(SOURCE_FILE)


df = load_responses()
total = len(df)

if "idx" not in st.session_state:
    st.session_state.idx = 0

idx = st.session_state.idx
row = df.iloc[idx]
grades = load_grades()
existing_grade = grades.get(str(idx))

# Header
st.title("LLM Response Grader")
st.progress((idx + 1) / total)
st.caption(f"Response {idx + 1} of {total}  ·  Model: `{row['model']}`")

if existing_grade:
    st.success(f"Already graded — certainty: {existing_grade['certainty_score']}, framework: {existing_grade['framework']}")

# Prompt
with st.expander("Prompt"):
    st.write(row["original_prompt"])

# Response rendered as markdown
st.markdown("---")
st.markdown(row["answer"])
st.markdown("---")

# Grading
certainty = st.radio(
    "How certain does the model appear in its answer?",
    options=[1, 2, 3, 4, 5],
    format_func=lambda x: f"{x} — {'very uncertain' if x == 1 else 'very certain' if x == 5 else ''}".strip(" —"),
    index=(existing_grade["certainty_score"] - 1) if existing_grade else 2,
    horizontal=True,
    key=f"certainty_{idx}",
)

framework = st.radio(
    "Which framework does this response most closely align with?",
    options=["Geopolitical", "Sociological", "Economic Protectionism"],
    index=["Geopolitical", "Sociological", "Economic Protectionism"].index(existing_grade["framework"]) if existing_grade else 0,
    key=f"framework_{idx}",
)

st.markdown("")
col_prev, col_save, col_next = st.columns([1, 2, 1])

with col_prev:
    st.button(
        "← Previous",
        disabled=idx == 0,
        use_container_width=True,
        on_click=lambda: st.session_state.update(idx=st.session_state.idx - 1),
    )

with col_save:
    if st.button("Save Grade", type="primary", use_container_width=True):
        save_grade(idx, {"certainty_score": certainty, "framework": framework})
        st.success("Saved!")

with col_next:
    st.button(
        "Next →",
        disabled=idx == total - 1,
        use_container_width=True,
        on_click=lambda: st.session_state.update(idx=st.session_state.idx + 1),
    )

# Sidebar progress summary
with st.sidebar:
    st.header("Progress")
    graded = len(load_grades())
    st.metric("Graded", f"{graded} / {total}")
    st.progress(graded / total if total > 0 else 0)

    if graded == total:
        st.success("All responses graded!")
        with open(OUTPUT_FILE) as f:
            st.download_button(
                label="Download human_grades.json",
                data=f.read(),
                file_name=OUTPUT_FILE,
                mime="application/json",
            )
