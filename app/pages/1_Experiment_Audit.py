"""Experiment audit: allocation, data validation, effect with interval, evidence status, reason codes."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from common import SERIES, archive_tables, badge, need_data, pick_experiment, synthetic_results

from expengine import decision as dec
from expengine.inference import wilson_interval
from expengine.sequential import boundaries

st.set_page_config(page_title="Experiment audit", page_icon="🧪", layout="wide")
need_data()
st.title("Experiment audit")
choice = pick_experiment()
if not choice:
    st.stop()
src, eid = choice


def rules_table(records):
    df = pd.DataFrame(records)[["rule_id", "severity", "status", "observed", "threshold", "explanation"]]
    icon = {"pass": "✅ pass", "fail": "❌ fail", "not_applicable": "– n/a"}
    df["status"] = [("⚠️ fail" if s == "fail" and sev != "critical" else icon[s]) for s, sev in zip(df["status"], df["severity"])]
    st.dataframe(df, hide_index=True, width="stretch")


def reasons(codes, trace):
    for c in codes:
        st.markdown(f"`{c}` — {dec.describe(c)}")
    with st.expander("Decision trace"):
        st.dataframe(pd.DataFrame(trace), hide_index=True, width="stretch")


if src == "synthetic":
    r = next(x for x in synthetic_results() if x["experiment_id"] == eid)
    st.markdown(f"### {r['title']} &nbsp; {badge(r['decision']['recommendation'])} "
                "<span style='color:#52514e;font-size:.8em'>SYNTHETIC SCENARIO</span>", unsafe_allow_html=True)
    st.write(r["hypothesis"])
    a, b = st.columns(2)
    with a:
        st.subheader("Allocation")
        arms = pd.DataFrame(r["arm_summary"])
        total = arms["units"].sum()
        fig = go.Figure()
        fig.add_bar(x=arms["arm_id"], y=arms["units"] / total, name="observed", marker_color=SERIES[0],
                    hovertemplate="%{x}: %{y:.2%} of units<extra>observed</extra>")
        fig.add_scatter(x=arms["arm_id"], y=r["allocation"], mode="markers", name="designed",
                        marker=dict(color=SERIES[1], size=14, symbol="line-ew-open", line=dict(width=3)),
                        hovertemplate="%{x}: %{y:.0%} designed<extra></extra>")
        fig.update_layout(yaxis_tickformat=".0%", height=320, margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=1.15, x=0))
        st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("Sequential monitoring")
        looks = pd.DataFrame(r["looks"])
        K = r["planned_looks"]
        t_all = np.linspace(1 / K, 1, K)
        c_all = boundaries(t_all, 0.05, "obf")
        fig = go.Figure()
        for sign, show in ((1, True), (-1, False)):
            fig.add_scatter(x=t_all, y=sign * c_all, name="OBF boundary (planned)", mode="lines+markers", showlegend=show,
                            line=dict(color="#52514e", dash="dash"), marker=dict(size=5, color="#52514e"),
                            hovertemplate="look at %{x:.0%}: |z| ≥ %{customdata:.2f}<extra></extra>", customdata=c_all)
        fig.add_scatter(x=looks["information_fraction"], y=looks["z"], name="observed z", mode="lines+markers",
                        line=dict(color=SERIES[0], width=2), marker=dict(size=9),
                        customdata=looks[["n", "action"]], hovertemplate="info %{x:.0%} · n=%{customdata[0]:,}<br>z=%{y:.2f} · %{customdata[1]}<extra></extra>")
        stop = looks.iloc[-1]
        fig.add_annotation(x=stop["information_fraction"], y=stop["z"], text=f"stop: look {int(stop['look'])}", showarrow=True,
                           arrowhead=0, ax=0, ay=-28, font=dict(size=11, color="#0b0b0b"))
        fig.update_layout(height=320, margin=dict(l=10, r=10, t=40, b=10), xaxis_tickformat=".0%",
                          xaxis_title="information fraction", xaxis_range=[0, 1.05], yaxis_title="z",
                          legend=dict(orientation="h", y=1.15, x=0))
        st.plotly_chart(fig, width="stretch")
    st.subheader("Data validation and integrity")
    rules_table(r["audit"])
    if r["decision"]["recommendation"] != "blocked":
        p = r["primary"]
        delta = r["policy"].get("delta")
        st.subheader(f"Primary metric: {r['primary_metric']}")
        fig = go.Figure()
        fig.add_scatter(x=[p["ci_low"], p["ci_high"]], y=[0, 0], mode="lines", line=dict(color="#0b0b0b", width=3), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=[p["estimate"]], y=[0], mode="markers", marker=dict(color="#0b0b0b", size=12), showlegend=False,
                        hovertemplate=f"effect %{{x:.4f}}<br>repeated 95% CI [{p['ci_low']:.4f}, {p['ci_high']:.4f}]<extra></extra>")
        fig.add_vline(x=0, line_dash="dot", line_color="#9aa3ad", annotation_text="0")
        if delta is not None:
            fig.add_vline(x=delta, line_color="#1d7a4f", annotation_text=f"δ = {delta}")
        fig.update_layout(height=170, margin=dict(l=10, r=10, t=30, b=10), yaxis_visible=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"{p.get('method', '')} · repeated CI at look {r['stop_look']} of {r['planned_looks']}"
                   + (f" · CUPED variance ratio {p['variance_ratio']:.3f}" if "variance_ratio" in p else ""))
        st.subheader("Guardrails")
        st.dataframe(pd.DataFrame(r["guardrails"]), hide_index=True, width="stretch")
    st.subheader("Recommendation and reason codes")
    reasons(r["decision"]["reason_codes"], r["decision"]["trace"])

else:
    t = archive_tables()
    e = t["experiments"].set_index("experiment_id").loc[eid]
    d = t["decisions"].set_index("experiment_id").loc[eid]
    arms = t["arms"][(t["arms"]["experiment_id"] == eid)].copy()
    st.markdown(f"### Upworthy test `{eid}` &nbsp; {badge(d['recommendation'])} "
                "<span style='color:#52514e;font-size:.8em'>HISTORICAL ARCHIVE · non-actionable</span>", unsafe_allow_html=True)
    st.caption(f"split {e['split']} · created {e['experiment_date']} · {int(e['n_arms_source'])} arms · "
               f"allocation {e['expected_allocation']} · reference arm {e['reference_arm_id']} (earliest-created, predeclared)")
    use = arms[arms["arm_included"]].sort_values("created_at")
    a, b = st.columns(2)
    with a:
        st.subheader("Allocation vs equal split")
        if not use.empty:
            n = use["impressions"].astype(float)
            fig = go.Figure()
            fig.add_bar(x=use["arm_id"], y=n / n.sum(), marker_color=SERIES[0], name="observed share",
                        customdata=np.c_[use["impressions"], use["headline"].fillna("").str[:80]],
                        hovertemplate="%{x}<br>%{customdata[0]:,} impressions (%{y:.2%})<br>%{customdata[1]}<extra></extra>")
            fig.add_hline(y=1 / len(use), line_dash="dash", line_color="#52514e", annotation_text="equal share")
            fig.update_layout(yaxis_tickformat=".0%", height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
    with b:
        st.subheader("CTR by arm (Wilson 95%)")
        if not use.empty:
            lo, hi = wilson_interval(use["clicks"].astype(float), use["impressions"].astype(float))
            ctr = use["clicks"].astype(float) / use["impressions"].astype(float)
            fig = go.Figure(go.Scatter(x=ctr, y=use["arm_id"], mode="markers", marker=dict(color=SERIES[0], size=10),
                                       error_x=dict(type="data", symmetric=False, array=hi - ctr, arrayminus=ctr - lo, color="#0b0b0b"),
                                       customdata=np.c_[use["clicks"], use["impressions"]],
                                       hovertemplate="%{y}: CTR %{x:.3%}<br>%{customdata[0]:,} / %{customdata[1]:,}<extra></extra>"))
            fig.update_layout(xaxis_tickformat=".1%", height=300, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig, width="stretch")
    st.subheader("Data validation and integrity")
    rules_table(t["audit"][t["audit"]["experiment_id"] == eid].to_dict("records"))
    with st.expander("Source rows (including any quarantined or excluded packages)"):
        st.dataframe(arms, hide_index=True, width="stretch")
    st.subheader("Comparisons vs reference arm (exploratory)")
    c = t["comparisons"][t["comparisons"]["experiment_id"] == eid]
    st.dataframe(c[["arm_id", "reference_arm_id", "ctr_arm", "ctr_reference", "diff", "ci_low", "ci_high",
                    "ci_low_bonferroni", "ci_high_bonferroni", "rel_lift", "test", "p_value", "p_holm"]],
                 hide_index=True, width="stretch")
    st.caption(f"Omnibus homogeneity p = {d['homogeneity_p']:.3g} · sensitivity: relative MDE at 80% power ≈ {d['relative_mde_80']:.0%}")
    st.subheader("Recommendation and reason codes")
    reasons([x for x in d["reason_codes"].split(",") if x], json.loads(d["trace"]))
