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


# --- Baseline vs Candidate Comparison Support ---
if "baseline" in report and "candidate" in report:
    st.subheader("🆚 Baseline vs Candidate Comparison")
    baseline = report["baseline"]
    candidate = report["candidate"]

    # --- Utility Comparison ---
    st.header("⚙️ Utility Metrics (Comparison)")
    b_util = baseline.get("utility", {})
    c_util = candidate.get("utility", {})
    delta_em = c_util.get("exact_match", 0) - b_util.get("exact_match", 0)
    delta_f1 = c_util.get("f1", 0) - b_util.get("f1", 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline Exact Match", f"{b_util.get('exact_match', 0):.2f}")
        st.metric("Baseline F1 Score", f"{b_util.get('f1', 0):.2f}")
    with col2:
        st.metric("Candidate Exact Match", f"{c_util.get('exact_match', 0):.2f}", delta=f"{delta_em:+.2f}")
        st.metric("Candidate F1 Score", f"{c_util.get('f1', 0):.2f}", delta=f"{delta_f1:+.2f}")
    with col3:
        st.metric("Δ Exact Match", f"{delta_em:+.2f}")
        st.metric("Δ F1 Score", f"{delta_f1:+.2f}")
    st.caption("Baseline: " + b_util.get("notes", ""))
    st.caption("Candidate: " + c_util.get("notes", ""))

    # --- Safety Comparison ---
    st.header("🛡️ Safety Metrics (Comparison)")
    b_safety = baseline.get("safety", {})
    c_safety = candidate.get("safety", {})
    b_tr = b_safety.get("toxicity_rate", 0)
    c_tr = c_safety.get("toxicity_rate", 0)
    delta_tr = c_tr - b_tr
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline Toxicity Rate", f"{b_tr:.2%}")
    with col2:
        st.metric("Candidate Toxicity Rate", f"{c_tr:.2%}", delta=f"{delta_tr:+.2%}")
    with col3:
        st.metric("Δ Toxicity Rate", f"{delta_tr:+.2%}")
    st.caption("Baseline: " + b_safety.get("notes", ""))
    st.caption("Candidate: " + c_safety.get("notes", ""))

    # Show samples if available
    if "samples" in b_safety or "samples" in c_safety:
        with st.expander("See sample prompts & scores"):
            if "samples" in b_safety:
                st.markdown("**Baseline samples**")
                df_b = pd.DataFrame([
                    {"prompt": s["prompt"], "response": s["response"], **s["scores"]}
                    for s in b_safety["samples"]
                ])
                st.dataframe(df_b, use_container_width=True)
            if "samples" in c_safety:
                st.markdown("**Candidate samples**")
                df_c = pd.DataFrame([
                    {"prompt": s["prompt"], "response": s["response"], **s["scores"]}
                    for s in c_safety["samples"]
                ])
                st.dataframe(df_c, use_container_width=True)

    # --- Fairness Comparison ---
    st.header("⚖️ Fairness Metrics (Comparison)")
    b_fair = baseline.get("fairness", {})
    c_fair = candidate.get("fairness", {})
    b_pg = b_fair.get("parity_gap", 0)
    c_pg = c_fair.get("parity_gap", 0)
    delta_pg = c_pg - b_pg
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Baseline Parity Gap", f"{b_pg:.3f}")
    with col2:
        st.metric("Candidate Parity Gap", f"{c_pg:.3f}", delta=f"{delta_pg:+.3f}")
    with col3:
        st.metric("Δ Parity Gap", f"{delta_pg:+.3f}")
    st.caption("Baseline: " + b_fair.get("notes", ""))
    st.caption("Candidate: " + c_fair.get("notes", ""))

    # Grouped bar chart for avg_toxicity per group
    if "group_scores" in b_fair and "group_scores" in c_fair:
        group_names = sorted(set(b_fair["group_scores"].keys()) | set(c_fair["group_scores"].keys()))
        data = []
        for g in group_names:
            b_val = b_fair["group_scores"].get(g, {}).get("avg_toxicity", None)
            c_val = c_fair["group_scores"].get(g, {}).get("avg_toxicity", None)
            data.append({"group": g, "Model": "Baseline", "avg_toxicity": b_val})
            data.append({"group": g, "Model": "Candidate", "avg_toxicity": c_val})
        df = pd.DataFrame(data)
        fig = px.bar(df, x="group", y="avg_toxicity", color="Model", barmode="group",
                     title="Avg Toxicity by Group (Baseline vs Candidate)", text="avg_toxicity")
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("See group samples"):
            for g in group_names:
                st.subheader(g)
                cols = st.columns(2)
                if g in b_fair["group_scores"]:
                    with cols[0]:
                        st.markdown("**Baseline**")
                        subdf = pd.DataFrame([
                            {"prompt": s["prompt"], "response": s["response"], **s["scores"]}
                            for s in b_fair["group_scores"][g].get("samples", [])
                        ])
                        st.dataframe(subdf, use_container_width=True)
                if g in c_fair["group_scores"]:
                    with cols[1]:
                        st.markdown("**Candidate**")
                        subdf = pd.DataFrame([
                            {"prompt": s["prompt"], "response": s["response"], **s["scores"]}
                            for s in c_fair["group_scores"][g].get("samples", [])
                        ])
                        st.dataframe(subdf, use_container_width=True)
else:
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