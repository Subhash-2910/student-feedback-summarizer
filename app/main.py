"""Streamlit entry point: `streamlit run app/main.py`"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.storage import Store  # noqa: E402
from app.summarize import MERGED_DIR, list_adapters, load_model, split_feedback, summarize, tally  # noqa: E402

st.set_page_config(page_title="Student Feedback Summarizer", page_icon="📝")
st.title("Student Feedback Summarizer")
st.caption("A GPT-2 model fine-tuned with LoRA turns each student comment into a one-line summary: "
           "overall sentiment, what students praise and what concerns them.")

EXAMPLE = """The lectures were clear and well organised. But the course moved way too fast and grading felt arbitrary.

Honestly, the hands-on labs were my favourite part! Office hours were always packed though.

I took this course as an elective. The workload was manageable and the instructor was enthusiastic and approachable."""


@st.cache_resource(show_spinner="Loading model...")
def get_model(adapter, use_merged):
    return load_model(adapter, merged=use_merged)


adapters = list_adapters()
with st.sidebar:
    st.header("Model")
    if not adapters:
        st.warning("No adapter found. Train one with `python -m app.train`.")
    choice = st.selectbox("LoRA adapter (one per course or dataset)", adapters + ["(base GPT-2, no adapter)"])
    adapter = None if choice.startswith("(") else choice
    use_merged = st.checkbox("Use merged model if available", value=True,
                             help="Run `python -m app.merge` to build it.",
                             disabled=adapter is None or not (MERGED_DIR / (adapter or "")).exists())
    st.header("Generation")
    max_tokens = st.slider("Max summary length (tokens)", 10, 80, 40)
    temperature = st.slider("Temperature (0 = deterministic)", 0.0, 1.5, 0.0, 0.1)

tab_run, tab_history = st.tabs(["Summarize", "History"])

with tab_run:
    upload = st.file_uploader("Upload feedback (.txt: one comment per line or paragraph, "
                              ".csv: a 'feedback' column)", type=["txt", "csv"])
    pasted = st.text_area("...or paste comments", value="" if upload else EXAMPLE, height=200)
    comments = split_feedback(upload.getvalue(), upload.name) if upload else split_feedback(pasted)
    st.write(f"**{len(comments)}** comment(s) found.")

    if st.button("Summarize", type="primary", disabled=not comments):
        model, tok = get_model(adapter, use_merged)
        progress, rows, start = st.progress(0.0), [], time.time()
        for i, c in enumerate(comments):
            rows.append({"feedback": c, "summary": summarize(model, tok, c, max_tokens, temperature)})
            progress.progress((i + 1) / len(comments))
        st.success(f"Summarized {len(rows)} comment(s) in {time.time() - start:.1f}s with "
                   f"{adapter or 'base GPT-2'}.")

        overview = tally([r["summary"] for r in rows])
        if overview:
            st.subheader("Across all comments")
            st.bar_chart(pd.DataFrame(overview).set_index("aspect"), color=["#2e7d32", "#c62828"])
        st.subheader("Per comment")
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.download_button("Download CSV", pd.DataFrame(rows).to_csv(index=False), "summaries.csv")
        Store().put(time.strftime("%Y-%m-%d %H:%M:%S"), {"adapter": adapter, "results": rows})

with tab_history:
    history = Store().all()
    if not history:
        st.info("Summaries you run are saved here.")
    for when, run in reversed(list(history.items())):
        with st.expander(f"{when} · {run['adapter'] or 'base'} · {len(run['results'])} comment(s)"):
            st.dataframe(pd.DataFrame(run["results"]), hide_index=True)
