"""Shared data access for the Streamlit app (read-only over persisted run outputs)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from expengine.config import DATA  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
STATUS = {"ship": "#1d7a4f", "no_ship": "#b23b2e", "inconclusive": "#8a6d1a", "blocked": "#4b4f58"}
LABEL = {"ship": "SHIP", "no_ship": "NO SHIP", "inconclusive": "INCONCLUSIVE", "blocked": "BLOCKED"}
ICON = {"ship": "✅", "no_ship": "⛔", "inconclusive": "❔", "blocked": "🚫"}


def need_data() -> bool:
    if not (DATA / "runs" / "runs.parquet").exists():
        st.error("No runs found. Build them first: `make all` (or `make audit scenarios`).")
        st.stop()
    return True


@st.cache_data
def runs() -> pd.DataFrame:
    return pd.read_parquet(DATA / "runs" / "runs.parquet")


def latest(kind: str) -> Path | None:
    r = runs()
    r = r[r["kind"] == kind]
    return None if r.empty else DATA / "runs" / r.iloc[-1]["run_id"]


@st.cache_data
def archive_tables() -> dict[str, pd.DataFrame]:
    run = latest("archive_audit")
    if run is None:
        return {}
    exp = pd.read_parquet(DATA / "processed" / "experiments.parquet")
    dec = pd.read_parquet(run / "decisions.parquet")
    return {
        "experiments": exp,
        "decisions": dec,
        "audit": pd.read_parquet(run / "audit_results.parquet"),
        "comparisons": pd.read_parquet(run / "comparisons.parquet"),
        "srm": pd.read_parquet(run / "srm.parquet"),
        "arms": pd.read_parquet(DATA / "processed" / "arms.parquet",
                                columns=["arm_id", "experiment_id", "split", "impressions", "clicks", "created_at",
                                         "headline", "arm_included", "arm_exclusion", "row_issues"]),
    }


@st.cache_data
def synthetic_results() -> list[dict]:
    run = latest("synthetic_product")
    return [] if run is None else json.loads((run / "results.json").read_text())


@st.cache_data
def registry() -> pd.DataFrame:
    rows = []
    t = archive_tables()
    if t:
        e = t["experiments"]
        e = e[e["split"] != "undeployed"].merge(t["decisions"][["experiment_id", "recommendation", "integrity_status", "reason_codes", "run_id"]],
                                                on="experiment_id", how="left")
        rows.append(pd.DataFrame({
            "experiment_id": e["experiment_id"], "source": "upworthy archive", "split": e["split"],
            "date": pd.to_datetime(e["experiment_date"]), "arms": e["n_arms_source"].astype(int),
            "units": e["impressions"].astype("float"), "problem_window": e["problem_reconstructed"],
            "integrity": e["integrity_status"], "recommendation": e["recommendation"],
            "reason_codes": e["reason_codes"], "run_id": e["run_id"]}))
    for r in synthetic_results():
        rows.append(pd.DataFrame([{
            "experiment_id": r["experiment_id"], "source": "synthetic", "split": "simulated",
            "date": pd.Timestamp("2026-01-05"), "arms": len(r["arms"]), "units": float(r["planned_units"]),
            "problem_window": False, "integrity": r["integrity_status"], "recommendation": r["decision"]["recommendation"],
            "reason_codes": ",".join(r["decision"]["reason_codes"]), "run_id": r["run_id"]}]))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def pick_experiment(label="Experiment") -> tuple[str, str] | None:
    """Selected (source, experiment_id); remembers the registry selection via session state."""
    reg = registry()
    sources = ["synthetic", "upworthy archive"]
    default_src = st.session_state.get("source", "synthetic")
    src = st.radio("Source", sources, index=sources.index(default_src), horizontal=True)
    ids = reg.loc[reg["source"] == src, "experiment_id"].tolist()
    if not ids:
        st.info("No experiments for this source yet.")
        return None
    current = st.session_state.get("experiment_id")
    idx = ids.index(current) if current in ids else 0
    eid = st.selectbox(label, ids, index=idx)
    st.session_state["source"], st.session_state["experiment_id"] = src, eid
    return src, eid


def badge(rec: str) -> str:
    return (f"<span style='background:{STATUS[rec]};color:#fff;padding:3px 10px;border-radius:3px;"
            f"font-weight:700;letter-spacing:.04em'>{LABEL[rec]}</span>")
