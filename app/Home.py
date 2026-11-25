"""Registry: every audited experiment, archive and synthetic, with filters and run history."""

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from common import ICON, STATUS, need_data, registry, runs

st.set_page_config(page_title="Experiment Decision Engine", page_icon="🧪", layout="wide")
need_data()
st.title("Experiment registry")
st.caption("Health first, then evidence, then value. Archive records are observed; synthetic scenarios are simulated "
           "and labelled as such everywhere.")

reg = registry()
c1, c2, c3, c4 = st.columns([1.2, 1.4, 1.4, 1])
with c1:
    src = st.multiselect("Source", sorted(reg["source"].unique()), default=sorted(reg["source"].unique()))
with c2:
    rec = st.multiselect("Recommendation", list(STATUS), default=list(STATUS))
with c3:
    integ = st.multiselect("Integrity", ["pass", "warning", "blocked"], default=["pass", "warning", "blocked"])
with c4:
    pw = st.selectbox("Problem window", ["any", "inside", "outside"])
c5, c6 = st.columns([2, 1])
with c5:
    d0, d1 = reg["date"].min().date(), reg["date"].max().date()
    dr = st.slider("Created", min_value=d0, max_value=d1, value=(d0, d1))
with c6:
    amax = int(reg["arms"].max())
    arms = st.slider("Arms", 1, amax, (1, amax))

f = reg[reg["source"].isin(src) & reg["recommendation"].isin(rec) & reg["integrity"].isin(integ)
        & reg["date"].dt.date.between(*dr) & reg["arms"].between(*arms)]
if pw != "any":
    f = f[f["problem_window"] == (pw == "inside")]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Experiments", f"{len(f):,}")
for col, key in zip((m2, m3, m4, m5), STATUS):
    col.metric(f"{ICON[key]} {key.replace('_', ' ')}", f"{(f['recommendation'] == key).sum():,}")

st.dataframe(
    f.assign(recommendation=f["recommendation"].map(lambda r: f"{ICON.get(r, '')} {r}")).sort_values("date"),
    width="stretch", hide_index=True, height=360,
    column_config={"units": st.column_config.NumberColumn("units", format="%d"),
                   "date": st.column_config.DateColumn("created")},
)

sel = st.selectbox("Open an experiment in the audit view", [""] + f["experiment_id"].head(2000).tolist())
if sel:
    st.session_state["experiment_id"] = sel
    st.session_state["source"] = f.loc[f["experiment_id"] == sel, "source"].iloc[0]
    st.switch_page("pages/1_Experiment_Audit.py")

arch = f[f["source"] == "upworthy archive"]
if not arch.empty:
    st.subheader("Blocked share by month (archive)")
    by = arch.assign(month=arch["date"].dt.to_period("M").dt.to_timestamp()).groupby("month").agg(
        tests=("experiment_id", "size"), blocked=("recommendation", lambda s: (s == "blocked").mean())).reset_index()
    fig = px.bar(by, x="month", y="blocked", hover_data={"tests": True, "blocked": ":.1%"},
                 color_discrete_sequence=["#2a78d6"], labels={"blocked": "share blocked", "month": ""})
    fig.update_layout(yaxis_tickformat=".0%", height=280, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("Run history")
r = runs().copy()
r["summary"] = r["summary"].map(lambda s: ", ".join(f"{k}={v}" for k, v in json.loads(s).get("recommendations", {}).items()))
st.dataframe(r.iloc[::-1], hide_index=True, width="stretch")
