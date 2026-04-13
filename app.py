"""
Streamlit web interface for the sentence-relevance evaluation toolkit.

Run with:
    streamlit run app.py
"""
import contextlib
import io
import os
import re
import tempfile
import time

import pandas as pd
import streamlit as st

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Evaluation Toolkit",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global background ─────────────────────────────────────────────────── */
.stApp { background-color: #e6f4f1; }

/* ── Tabs: wider spacing and legible text ──────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background-color: #b6c3f0;
    border-radius: 10px;
    padding: 6px 8px;
    gap: 12px;
}
.stTabs [data-baseweb="tab"] {
    background-color: transparent;
    color: #3f6f9c;
    border-radius: 7px;
    font-weight: 600;
    font-size: 1rem;
    padding: 0.55rem 2rem;
    min-width: 160px;
    justify-content: center;
}
.stTabs [data-baseweb="tab"]:hover {
    background-color: #c3e2e2;
    color: #3f6f9c;
}
.stTabs [aria-selected="true"] {
    background-color: #3f6f9c !important;
    color: white !important;
}

/* ── Buttons ───────────────────────────────────────────────────────────── */
.stButton > button {
    background-color: #3f6f9c;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover { background-color: #5498c4; color: white; }

/* ── Download buttons ──────────────────────────────────────────────────── */
.stDownloadButton > button {
    background-color: #5498c4;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
.stDownloadButton > button:hover { background-color: #3f6f9c; color: white; }

/* ── Metric cards ──────────────────────────────────────────────────────── */
[data-testid="metric-container"] {
    background-color: white;
    border: 1px solid #b6c3f0;
    border-radius: 8px;
    padding: 0.75rem 1rem;
}

/* ── Expanders ─────────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
    background-color: #c3e2e2 !important;
    color: #3f6f9c !important;
    border-radius: 6px;
    font-weight: 600;
}

/* ── Section headers ───────────────────────────────────────────────────── */
h1 { color: #3f6f9c; }
h2, h3 { color: #5498c4; }

/* ── Alert / info / success banners ────────────────────────────────────── */
div[data-testid="stAlert"] { border-radius: 8px; }

/* ── Suppress all red — replace with blue-teal tones ───────────────────── */
/* Error text */
.stException, [data-testid="stException"] { color: #3f6f9c !important; }
/* Required-field asterisks and validation borders */
.stTextInput [data-baseweb="input"] { border-color: #7b9ee8 !important; }
/* Progress bar fill */
[data-testid="stProgressBar"] > div > div { background-color: #5498c4 !important; }
/* Spinner */
[data-testid="stSpinner"] svg { stroke: #5498c4 !important; }
</style>
""", unsafe_allow_html=True)


# ── Session-state defaults ────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "config": None,
        "input_tmp": None,
        "output_tmp": None,
        "query_done": False,
        "_last_upload": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Configure
# ─────────────────────────────────────────────────────────────────────────────
def tab_configure():
    st.header("Configure")

    uploaded = st.file_uploader("Upload input CSV", type="csv")
    if uploaded is None:
        st.info("Upload a CSV to get started.")
        return

    # Save to a temp file so pandas can read it; re-save only when file changes
    if st.session_state["_last_upload"] != uploaded.name:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded.read())
        tmp.flush()
        st.session_state.input_tmp = tmp.name
        st.session_state["_last_upload"] = uploaded.name
        st.session_state.query_done = False

    df_preview = pd.read_csv(st.session_state.input_tmp, nrows=3)
    cols = df_preview.columns.tolist()

    with st.expander("Preview — first 3 rows", expanded=False):
        st.dataframe(df_preview, use_container_width=True)

    # ── Dataset columns ───────────────────────────────────────────────────────
    st.subheader("Dataset columns")
    c1, c2 = st.columns(2)
    with c1:
        id_col       = st.selectbox("ID column (for resume support)", ["(none)"] + cols)
        sentences_col = st.selectbox("Sentences column", cols)
        options_col  = st.selectbox("Question + answer options column", cols)
    with c2:
        groupby_col  = st.selectbox("Group-by column (optional)", ["(none)"] + cols)

    # ── Correct answer ────────────────────────────────────────────────────────
    correct_answer_col = st.selectbox("Correct answer column (optional)", ["(none)"] + cols)

    # ── Model settings ────────────────────────────────────────────────────────
    st.subheader("Model settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        model_name = st.text_input("Model name", value="gpt-4o")
    with c2:
        temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    with c3:
        max_sentences = int(st.number_input("Max sentences to label", min_value=1,
                                            max_value=100, value=30))

    api_key = st.text_input(
        "OpenAI API key", type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Stored only in memory for this session.",
    )

    custom_prompt = st.text_area(
        "Custom prompt template (optional)",
        height=140,
        help="Use {sentences} and {options} as placeholders. Leave blank for the default prompt.",
    )

    # ── Save ──────────────────────────────────────────────────────────────────
    if st.button("Save configuration"):
        from toolkit.config import ToolkitConfig

        out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        out_tmp.close()
        st.session_state.output_tmp = out_tmp.name

        cfg = ToolkitConfig(
            input_path=st.session_state.input_tmp,
            output_path=st.session_state.output_tmp,
            id_col=id_col if id_col != "(none)" else None,
            sentences_col=sentences_col,
            options_col=options_col,
            groupby_col=groupby_col if groupby_col != "(none)" else None,
            mode="single",
            correct_answer_col=(correct_answer_col
                                if correct_answer_col and correct_answer_col != "(none)"
                                else None),
            model_name=model_name,
            temperature=temperature,
            max_sentences=max_sentences,
        )

        if custom_prompt.strip():
            pt = tempfile.NamedTemporaryFile(delete=False, suffix=".txt", mode="w")
            pt.write(custom_prompt.strip())
            pt.flush()
            cfg.custom_prompt_path = pt.name

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key

        st.session_state.config = cfg
        st.session_state.query_done = False
        st.success("Configuration saved — head to the **Query** tab.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Query
# ─────────────────────────────────────────────────────────────────────────────
def tab_query():
    st.header("Query")

    cfg = st.session_state.config
    if cfg is None:
        st.info("Complete the **Configure** tab first.")
        return

    with st.expander("Current configuration", expanded=True):
        st.json({
            "model": cfg.model_name,
            "temperature": cfg.temperature,
            "mode": cfg.mode,
            "sentences_col": cfg.sentences_col,
            "options_col": cfg.options_col,
            "max_sentences": cfg.max_sentences,
        })

    if st.session_state.query_done:
        st.success("Query already completed. You can re-run it or go to the **Analyze** tab.")
        with open(cfg.output_path, "rb") as f:
            st.download_button("Download results CSV", f,
                               file_name="query_output.csv", mime="text/csv",
                               key="dl_results_existing")

    if st.button("Run Query"):
        _run_query_with_progress(cfg)


def _run_query_with_progress(cfg):
    """Mirrors toolkit/query.py run_query() with a Streamlit progress bar."""
    from toolkit.query import _get_client, _call_model, _parse_single

    df = pd.read_csv(cfg.input_path)
    prompt_template = cfg.get_prompt_template()
    label_cols = [f"label_{i + 1}" for i in range(cfg.max_sentences)]
    extra_cols = ["Raw_Response", "LLM_answer"] + label_cols

    # Resume: collect already-processed IDs
    processed_ids: set = set()
    if not os.path.exists(cfg.output_path) or os.path.getsize(cfg.output_path) == 0:
        header = df.columns.tolist() + extra_cols
        pd.DataFrame(columns=header).to_csv(cfg.output_path, index=False)
    else:
        try:
            existing = pd.read_csv(cfg.output_path)
            if cfg.id_col and cfg.id_col in existing.columns:
                for v in existing[cfg.id_col].dropna().unique():
                    processed_ids.add(str(v))
        except Exception:
            pass

    try:
        client = _get_client(cfg)
    except EnvironmentError as e:
        st.error(str(e))
        return

    n = len(df)
    progress_bar = st.progress(0, text="Starting…")
    errors = []

    for i in range(n):
        row = df.iloc[i]

        if cfg.id_col:
            row_id = str(row[cfg.id_col])
            if row_id in processed_ids:
                progress_bar.progress(i / n, text=f"Skipping row {i+1}/{n} (already done)")
                continue

        progress_bar.progress(i / n, text=f"Processing row {i+1} of {n}…")

        try:
            sentences = str(row[cfg.sentences_col]).strip()
            options   = str(row[cfg.options_col]).strip()
            n_sentences = len(re.findall(r"(?:\bS\d+\s*:|\b\d+\.)", sentences))

            prompt  = prompt_template.format(sentences=sentences, options=options)
            content = _call_model(client, cfg, prompt)

            llm_answer, sentence_labels = _parse_single(content, n_sentences, cfg.max_sentences)
            row_extra = [content, llm_answer] + sentence_labels

        except Exception as e:
            errors.append(f"Row {i}: {e}")
            row_extra = ["", None] + [None] * cfg.max_sentences

        out_row = pd.concat(
            [df.iloc[[i]].reset_index(drop=True),
             pd.DataFrame([row_extra], columns=extra_cols)],
            axis=1,
        )
        out_row.to_csv(cfg.output_path, mode="a", index=False, header=False)
        time.sleep(0.05)

    progress_bar.progress(1.0, text="Done!")
    st.session_state.query_done = True

    processed = n - len(errors)
    st.success(f"Query complete — {processed} rows processed, {len(errors)} errors.")
    if errors:
        with st.expander("Errors"):
            for e in errors:
                st.text(e)

    with open(cfg.output_path, "rb") as f:
        st.download_button("Download results CSV", f,
                           file_name="query_output.csv", mime="text/csv",
                           key="dl_results_new")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Analyze
# ─────────────────────────────────────────────────────────────────────────────
def tab_analyze():
    st.header("Analyze")

    cfg = st.session_state.config
    if cfg is None:
        st.info("Complete the **Configure** tab first.")
        return

    st.markdown("Use the output from the Query step, or upload a previous results file:")
    uploaded_results = st.file_uploader("Upload results CSV (optional)",
                                        type="csv", key="results_upload")

    if uploaded_results is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        tmp.write(uploaded_results.read())
        tmp.flush()
        results_path = tmp.name
    elif st.session_state.query_done and st.session_state.output_tmp:
        results_path = st.session_state.output_tmp
        st.info("Using results from the Query step.")
    else:
        st.info("Run the **Query** step first, or upload an existing results CSV.")
        return

    if st.button("Run Analysis"):
        _run_analysis_display(cfg, results_path)


def _run_analysis_display(cfg, results_path: str):
    from toolkit.analyze import (
        VALID_ANSWERS,
        accuracy_section,
        refusal_rate_section,
        relevance_accuracy_section,
        relevance_distribution_section,
        run_analysis,
    )

    df = pd.read_csv(results_path)

    llm_col = "LLM_answer"
    correct_col = cfg.correct_answer_col or ""

    # ── Metrics ───────────────────────────────────────────────────────────────
    if llm_col not in df.columns:
        st.warning(f"Column '{llm_col}' not found in results CSV.")
        return

    with st.expander("Results", expanded=True):
            valid_mask   = df[llm_col].astype(str).str.strip().isin(VALID_ANSWERS)
            refusal_rate = (~valid_mask).mean()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total rows", len(df))
            m2.metric("Refusal rate", f"{refusal_rate:.1%}")

            if correct_col and correct_col in df.columns:
                valid_df = df[valid_mask].copy()
                correct_mask = (
                    valid_df[llm_col].astype(str).str.strip() ==
                    valid_df[correct_col].astype(str).str.strip()
                )
                accuracy = correct_mask.mean() if len(valid_df) else float("nan")
                m3.metric("Accuracy", f"{accuracy:.1%}" if not pd.isna(accuracy) else "N/A")
            else:
                m3.metric("Accuracy", "N/A")

            st.markdown("**Refusal rate detail**")
            st.code(refusal_rate_section(df, llm_col, cfg.groupby_col))

            if correct_col and correct_col in df.columns:
                st.markdown("**Accuracy detail**")
                st.code(accuracy_section(df, llm_col, correct_col, cfg.groupby_col))
                st.markdown("**Accuracy by majority sentence-relevance label**")
                st.code(relevance_accuracy_section(df, llm_col, correct_col, cfg.max_sentences))

    # ── Relevance distribution ────────────────────────────────────────────────
    st.subheader("Sentence relevance distribution")
    st.code(relevance_distribution_section(df, cfg.max_sentences, cfg.groupby_col))

    # ── Summary CSV via run_analysis ──────────────────────────────────────────
    out_dir = tempfile.mkdtemp()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_analysis(cfg, results_path, out_dir)

    summary_path = os.path.join(out_dir, "summary.csv")
    if os.path.exists(summary_path):
        summary_df = pd.read_csv(summary_path)
        st.subheader("Summary table")
        st.dataframe(summary_df, use_container_width=True)
        with open(summary_path, "rb") as f:
            st.download_button("Download summary.csv", f,
                               file_name="summary.csv", mime="text/csv")

    report_path = os.path.join(out_dir, "analysis_report.txt")
    if os.path.exists(report_path):
        with open(report_path) as f:
            report_text = f.read()
        with st.expander("Full text report"):
            st.text(report_text)
        st.download_button("Download analysis_report.txt", report_text,
                           file_name="analysis_report.txt")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    _init_state()

    st.title("Sentence Relevance Evaluation Toolkit")
    st.caption("Upload a CSV, configure your experiment, query a GPT model, and analyze results.")

    tab1, tab2, tab3 = st.tabs(["1 · Configure", "2 · Query", "3 · Analyze"])
    with tab1:
        tab_configure()
    with tab2:
        tab_query()
    with tab3:
        tab_analyze()


if __name__ == "__main__":
    main()
