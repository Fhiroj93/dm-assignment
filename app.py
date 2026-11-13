"""Streamlit dashboard for the graph anomaly detection demonstration."""

from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent
EXPLORATION_DIR = PROJECT_ROOT / "results" / "exploration"
RESULTS_DIR = PROJECT_ROOT / "results"

st.set_page_config(page_title="Graph Anomaly Detection", page_icon="🔎", layout="wide")
st.title("🔎 Graph Anomaly Detection")
st.caption("Educational demonstration inspired by ARC and ARCzero")

st.sidebar.header("Detection method")
method = st.sidebar.selectbox("Choose a method", ["Baseline", "Few-shot", "Zero-shot"])
top_k = st.sidebar.slider("Number of nodes to display", min_value=5, max_value=20, value=10)

method_files = {
    "Baseline": (RESULTS_DIR / "baseline" / "baseline_scores.csv", "anomaly_score"),
    "Few-shot": (RESULTS_DIR / "few_shot" / "few_shot_scores.csv", "few_shot_score"),
    "Zero-shot": (RESULTS_DIR / "zero_shot" / "zero_shot_scores.csv", "zero_shot_score"),
}

score_file, score_column = method_files[method]
if not score_file.exists():
    st.error("Results are missing. Run the project scripts before starting the dashboard.")
    st.stop()

scores = pd.read_csv(score_file).sort_values(score_column, ascending=False)
comparison_file = RESULTS_DIR / "comparison" / "model_comparison.csv"
comparison = pd.read_csv(comparison_file) if comparison_file.exists() else None

metric_1, metric_2, metric_3 = st.columns(3)
metric_1.metric("Total nodes", len(scores))
metric_2.metric("Known anomalies", int(scores["label"].sum()))
metric_3.metric("Selected method", method)

st.subheader("Graph overview")
graph_image = EXPLORATION_DIR / "graph_overview.png"
if graph_image.exists():
    st.image(str(graph_image), use_container_width=True)

st.subheader(f"Top {top_k} predicted anomalies - {method}")
display_columns = ["node_id", score_column, "degree", "label"]
st.dataframe(scores.head(top_k)[display_columns], use_container_width=True, hide_index=True)

st.subheader("Anomaly score distribution")
st.bar_chart(scores.head(top_k).set_index("node_id")[score_column])

if comparison is not None:
    st.subheader("Method comparison")
    st.dataframe(comparison, use_container_width=True, hide_index=True)
    chart = comparison.set_index("model")[["precision", "recall", "f1_score"]]
    st.bar_chart(chart)

st.info(
    "Interpretation: a higher anomaly score means that the node differs more "
    "from the normal or pseudo-normal context profile."
)
