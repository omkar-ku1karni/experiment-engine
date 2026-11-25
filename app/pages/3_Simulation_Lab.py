"""Simulation lab: benchmark results plus live, seeded paired-stream simulations. All results are SIMULATED."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import ROOT, SERIES

from expengine import benchmarks

st.set_page_config(page_title="Simulation lab", page_icon="🧪", layout="wide")
st.title("Simulation lab")
st.caption("Everything on this page is simulated with seeded generators; nothing here replays archive data.")

RES = ROOT / "simulations" / "results"
POL = ["fixed", "naive_peeking", "group_sequential_obf", "group_sequential_pocock"]
LAB = {"fixed": "Fixed horizon", "naive_peeking": "Naive peeking", "group_sequential_obf": "Group sequential (OBF)",
       "group_sequential_pocock": "Group sequential (Pocock)"}
COLOR = dict(zip(POL, SERIES))

tab1, tab2, tab3 = st.tabs(["Fixed vs sequential", "CUPED", "Run your own"])

with tab1:
    if not (RES / "sequential.csv").exists():
        st.info("Run `make simulate` to produce the benchmark tables.")
    else:
        seq = pd.read_csv(RES / "sequential.csv")
        c1, c2 = st.columns(2)
        p0 = c1.selectbox("Baseline", sorted(seq["p0"].unique()), index=1, format_func=lambda v: f"{v:.0%}")
        sh = c2.selectbox("Treatment share", sorted(seq["share"].unique()), format_func=lambda v: f"{v:.0%}")
        s = seq[(seq["p0"] == p0) & (seq["share"] == sh)]
        fig = go.Figure()
        for p in POL:
            g = s[s["policy"] == p].sort_values("effect_multiple")
            fig.add_bar(x=g["effect_multiple"].map(lambda m: f"{m:g}× MDE"), y=g["rejection_rate"], name=LAB[p],
                        marker_color=COLOR[p], error_y=dict(type="data", array=1.96 * g["rejection_se"], color="#52514e"),
                        customdata=np.c_[g["mean_stop_fraction"], g["coverage"]],
                        hovertemplate=LAB[p] + "<br>rejects %{y:.2%}<br>mean sample %{customdata[0]:.0%} of plan"
                                                  "<br>coverage %{customdata[1]:.1%}<extra></extra>")
        fig.add_hline(y=0.05, line_dash="dash", line_color="#52514e", annotation_text="5%")
        fig.update_layout(barmode="group", yaxis_tickformat=".0%", height=380, margin=dict(l=10, r=10, t=10, b=10),
                          yaxis_title="share of runs declaring an effect", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.caption("At 0× MDE the bar is the false-positive rate; elsewhere it is power. Same simulated streams for every policy.")
        st.dataframe(s[["effect_multiple", "policy", "rejection_rate", "rejection_se", "mean_stop_fraction", "coverage", "bias", "bias_if_rejected"]],
                     hide_index=True, width="stretch")

with tab2:
    if (RES / "cuped.csv").exists():
        cu = pd.read_csv(RES / "cuped.csv")
        outcome = st.radio("Outcome", ["continuous", "binary"], horizontal=True)
        c = cu[(cu["outcome"] == outcome) & cu["effect"].isna()]
        fig = go.Figure()
        for i, m in enumerate(sorted(c["missing"].unique())):
            g = c[c["missing"] == m]
            fig.add_scatter(x=g["rho"], y=g["variance_ratio"], mode="lines+markers", name=f"{m:.0%} covariates missing",
                            line=dict(color=SERIES[i], width=2), marker=dict(size=9),
                            error_y=dict(type="data", array=1.96 * g["variance_ratio_se"]),
                            customdata=np.c_[g["realized_correlation"], g["cuped_coverage"], g["cuped_bias"]],
                            hovertemplate="ρ=%{x}: ratio %{y:.3f}<br>realized corr %{customdata[0]:.2f}"
                                          "<br>coverage %{customdata[1]:.1%} · bias %{customdata[2]:+.4f}<extra></extra>")
        if outcome == "continuous":
            r = np.linspace(0, 0.85, 50)
            fig.add_scatter(x=r, y=1 - r**2, mode="lines", name="theory 1 − ρ²", line=dict(color="#52514e", dash="dash"))
        fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), xaxis_title="ρ",
                          yaxis_title="Var(CUPED) / Var(raw)", legend=dict(orientation="h"))
        st.plotly_chart(fig, width="stretch")
        st.dataframe(cu[cu["outcome"] == outcome], hide_index=True, width="stretch")

with tab3:
    st.markdown("Run a small seeded benchmark now. Larger runs belong in `make simulate` (10,000 replications).")
    a, b, c, d = st.columns(4)
    p0 = a.selectbox("Baseline rate", [0.01, 0.05, 0.2], index=1)
    mult = b.selectbox("True effect (× MDE)", [0.0, 0.5, 1.0, 2.0, -1.0], index=0)
    looks = c.slider("Looks", 2, 20, 10)
    reps = d.select_slider("Replications", [500, 1000, 2000, 5000], value=2000)
    seed = st.number_input("Seed", 0, 10**9, 7)
    if st.button("Run sequential simulation", type="primary"):
        cfg = {"alpha": 0.05, "power": 0.8, "looks": looks, "relative_mde": 0.10, "baselines": [p0],
               "treatment_shares": [0.5], "effect_multiples": [mult], "spending": ["obf", "pocock"]}
        out = benchmarks.sequential(cfg, reps, int(seed))
        out["policy"] = out["policy"].map(LAB)
        st.dataframe(out[["policy", "rejection_rate", "rejection_se", "mean_stop_fraction", "coverage", "bias"]], hide_index=True)
    st.divider()
    e, f, g = st.columns(3)
    rho = e.slider("Covariate correlation ρ", 0.0, 0.95, 0.58, 0.01)
    miss = f.slider("Missing covariate share", 0.0, 0.8, 0.0, 0.05)
    n = g.select_slider("Units per arm", [200, 500, 1000, 2000], value=1000)
    if st.button("Run CUPED simulation"):
        cfg = {"n_per_arm": n, "chunk": 500, "outcomes": ["continuous"], "binary_baseline": 0.2,
               "correlations": [rho], "missing_rates": [miss], "effects": [None, "mde"]}
        out = benchmarks.cuped(cfg, 1000, int(seed))
        st.dataframe(out[["effect", "variance_ratio", "variance_ratio_se", "theory_ratio_linear", "equivalent_sample_reduction",
                          "cuped_bias", "cuped_coverage", "raw_rejection", "cuped_rejection"]], hide_index=True)
