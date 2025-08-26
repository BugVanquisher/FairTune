import streamlit as st
import json
import pandas as pd
import plotly.express as px
from pathlib import Path

st.set_page_config(page_title="FairTune Eval Dashboard", layout="wide")

st.title("📊 FairTune Evaluation Dashboard")

# Load eval report
report_path = Path(__file__).parent.parent / "eval_report.json"
if not report_path.exists():
    st.error("No eval_report.json found. Run run_eval.py first.")
    st.stop()

with open(report_path) as f:
    report = json.load(f)

# --- Utility Section ---
st.header("⚙️ Utility Metrics")
utility = report.get("utility", {})
col1, col2 = st.columns(2)
with col1:
    st.metric("Exact Match", f"{utility.get('exact_match', 0):.2f}")
with col2:
    st.metric("F1 Score", f"{utility.get('f1', 0):.2f}")
st.caption(utility.get("notes", ""))

# --- Safety Section ---
st.header("🛡️ Safety Metrics")
safety = report.get("safety", {})
st.metric("Toxicity Rate", f"{safety.get('toxicity_rate', 0):.2%}")
st.caption(safety.get("notes", ""))

if "samples" in safety:
    with st.expander("See sample prompts & scores"):
        df = pd.DataFrame([
            {"prompt": s["prompt"], "response": s["response"], **s["scores"]}
            for s in safety["samples"]
        ])
        st.dataframe(df, use_container_width=True)

# --- Fairness Section ---
st.header("⚖️ Fairness Metrics")
fairness = report.get("fairness", {})
st.metric("Parity Gap (toxicity)", f"{fairness.get('parity_gap', 0):.3f}")
st.caption(fairness.get("notes", ""))

if "group_scores" in fairness:
    group_data = [
        {"group": g, "avg_toxicity": v["avg_toxicity"]}
        for g, v in fairness["group_scores"].items()
    ]
    df = pd.DataFrame(group_data)
    fig = px.bar(df, x="group", y="avg_toxicity", title="Avg Toxicity by Group", text="avg_toxicity")
    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("See group samples"):
        for g, v in fairness["group_scores"].items():
            st.subheader(g)
            subdf = pd.DataFrame([
                {"prompt": s["prompt"], "response": s["response"], **s["scores"]}
                for s in v["samples"]
            ])
            st.dataframe(subdf, use_container_width=True)