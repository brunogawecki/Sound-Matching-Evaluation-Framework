"""Browse eval results: the metric x (corpus x model) benchmark table + drill-down."""
from env import bootstrap

bootstrap()

import numpy as np
import pandas as pd
import streamlit as st

import discovery

st.set_page_config(page_title="Results", layout="wide")
st.title("Results")

runs = discovery.list_result_runs()
if not runs:
    st.info("No eval results yet. Run the **Evaluate** page to produce some.")
    st.stop()

# --- Benchmark table: rows = metric, columns = corpus / model, values = mean ----
st.subheader("Benchmark table (per-metric mean)")
columns = {}
higher_better = {}
for run in runs:
    summary = discovery.load_summary(run.summary_path)
    per_metric = summary.get("per_metric", {})
    column_label = f"{run.corpus} / {run.model}"
    columns[column_label] = {name: stats.get("mean") for name, stats in per_metric.items()}
    for name, stats in per_metric.items():
        higher_better.setdefault(name, stats.get("higher_is_better"))

benchmark = pd.DataFrame(columns)
if not benchmark.empty:
    benchmark.insert(0, "higher_is_better", [higher_better.get(metric) for metric in benchmark.index])
    benchmark.index.name = "metric"
    st.dataframe(benchmark, width="stretch")
    st.caption(
        f"{len(runs)} run(s). Fills in a column per model as families land; today it is a "
        "table-of-one against the baseline."
    )
else:
    st.warning("Result summaries had no per-metric block.")

# --- Drill-down: one run's per-sample distribution -----------------------------
st.subheader("Per-sample drill-down")
run = st.selectbox("Run", runs, format_func=lambda r: f"{r.corpus} / {r.model}")
if not run.per_sample_path.exists():
    st.warning(f"per_sample.csv missing for {run.corpus} / {run.model}.")
    st.stop()

per_sample = pd.read_csv(run.per_sample_path)
numeric_columns = [c for c in per_sample.columns if pd.api.types.is_numeric_dtype(per_sample[c])]

metric = st.selectbox("Metric", numeric_columns) if numeric_columns else None
if metric:
    series = per_sample[metric].dropna()
    left, right = st.columns([1, 2])
    with left:
        st.metric("mean", f"{series.mean():.4g}" if len(series) else "—")
        st.metric("std", f"{series.std():.4g}" if len(series) else "—")
        st.metric("valid", f"{len(series)} / {len(per_sample)}")
    with right:
        if len(series):
            counts, edges = np.histogram(series, bins=min(20, max(1, len(series))))
            histogram = pd.DataFrame(
                {"count": counts},
                index=[f"{edges[i]:.3g}" for i in range(len(counts))],
            )
            histogram.index.name = metric
            st.bar_chart(histogram)

st.caption("Full per-sample matrix (sortable):")
st.dataframe(per_sample, width="stretch", hide_index=True)
