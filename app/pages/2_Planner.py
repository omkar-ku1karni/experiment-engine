"""Planner: sample size, MDE and duration; optional group-sequential design; exportable config."""

import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from common import SERIES

from expengine import power
from expengine.sequential import boundaries

st.set_page_config(page_title="Planner", page_icon="🧪", layout="wide")
st.title("Experiment planner")
st.caption("Power is assessed against a predeclared minimum detectable effect, never the observed effect.")

c1, c2, c3 = st.columns(3)
with c1:
    p0 = st.number_input("Baseline conversion rate", 0.0005, 0.95, 0.05, step=0.005, format="%.4f")
    mde_type = st.radio("MDE type", ["relative", "absolute"], horizontal=True)
    mde = st.number_input("Minimum detectable effect", 0.0001, 5.0, 0.10 if mde_type == "relative" else 0.005,
                          step=0.01 if mde_type == "relative" else 0.001, format="%.4f")
with c2:
    alpha = st.select_slider("Alpha (two-sided)", [0.01, 0.025, 0.05, 0.1], value=0.05)
    pw = st.select_slider("Power", [0.7, 0.8, 0.85, 0.9, 0.95], value=0.8)
    share = st.slider("Treatment share of traffic", 0.1, 0.9, 0.5, 0.05)
with c3:
    traffic = st.number_input("Eligible daily traffic (units)", 100, 10_000_000, 20_000, step=1000)
    looks = st.slider("Planned interim looks (1 = fixed horizon)", 1, 10, 5)
    spending = st.selectbox("Alpha spending", ["obf", "pocock"], disabled=looks == 1)

try:
    plan = power.plan(p0, mde, relative=mde_type == "relative", alpha=alpha, power=pw, treatment_share=share, daily_traffic=traffic)
except ValueError as e:
    st.error(str(e))
    st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Control units", f"{plan.n_control:,}")
m2.metric("Treatment units", f"{plan.n_treatment:,}")
m3.metric("Total units", f"{plan.n_total:,}")
m4.metric("Expected duration", f"{plan.expected_duration_days:.1f} days")
st.caption(f"Treatment rate under the MDE: {plan.treatment_rate:.4%}. statsmodels cross-check (Cohen's h, arcsine): "
           f"{plan.statsmodels_n_control:,.0f} control units ({plan.statsmodels_n_control / plan.n_control - 1:+.1%}).")

design = plan.to_dict()
if looks > 1:
    t = np.linspace(1 / looks, 1, looks)
    c = boundaries(t, alpha, spending)
    st.subheader("Group-sequential boundaries")
    st.dataframe(pd.DataFrame({"look": range(1, looks + 1), "information": [f"{x:.0%}" for x in t],
                               "cumulative units": (t * plan.n_total).round().astype(int),
                               "day (approx.)": np.ceil(t * plan.n_total / traffic).astype(int),
                               "reject if |z| ≥": c.round(3), "nominal p": (2 * (1 - __import__("scipy").stats.norm.cdf(c))).round(5)}),
                 hide_index=True, width="stretch")
    st.caption("Same maximum sample as the fixed design; expect slightly lower power and a smaller average sample "
               "when the effect is real (see Simulation lab).")
    design["sequential"] = {"looks": looks, "spending": spending, "boundaries": c.round(4).tolist()}

st.subheader("Sensitivity at the planned sample size")
curve = pd.DataFrame(power.sensitivity_curve(p0, plan.n_control, plan.n_treatment, alpha, np.linspace(0.005, max(0.5, 3 * (mde if mde_type == 'relative' else mde / p0)), 80)))
fig = go.Figure(go.Scatter(x=curve["relative_effect"], y=curve["power"], mode="lines", line=dict(color=SERIES[0], width=2),
                           hovertemplate="relative effect %{x:.1%}: power %{y:.1%}<extra></extra>"))
fig.add_hline(y=pw, line_dash="dash", line_color="#52514e", annotation_text=f"target power {pw:.0%}")
fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", height=320, margin=dict(l=10, r=10, t=10, b=10),
                  xaxis_title="true relative effect", yaxis_title="power")
st.plotly_chart(fig, width="stretch")

d1, d2 = st.columns(2)
d1.download_button("Download design (YAML)", yaml.safe_dump(design, sort_keys=False), "design.yaml", "text/yaml")
d2.download_button("Download design (JSON)", json.dumps(design, indent=2), "design.json", "application/json")
