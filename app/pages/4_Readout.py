"""Readout / memo: one-page HTML (and PDF when Chromium is available) with provenance and rationale."""

from datetime import date

import streamlit as st
import streamlit.components.v1 as components

from common import latest, need_data, pick_experiment, synthetic_results

from expengine.memo import archive_context, product_context, render, to_pdf

st.set_page_config(page_title="Readout", page_icon="🧪", layout="wide")
need_data()
st.title("Readout memo")
choice = pick_experiment()
if not choice:
    st.stop()
src, eid = choice
as_of = st.date_input("Memo date", date.today()).isoformat()
if src == "synthetic":
    ctx = product_context(next(r for r in synthetic_results() if r["experiment_id"] == eid), as_of)
else:
    ctx = archive_context(latest("archive_audit"), eid, as_of)
html = render(ctx)
c1, c2 = st.columns(2)
c1.download_button("Download HTML", html, f"{eid}.html", "text/html", type="primary")
if c2.button("Build PDF"):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / f"{eid}.html"
        p.write_text(html)
        try:
            pdf = to_pdf([p])[0]
            c2.download_button("Download PDF", pdf.read_bytes(), f"{eid}.pdf", "application/pdf")
        except Exception as e:  # Chromium not installed
            c2.warning(f"PDF needs Playwright Chromium (`make browsers`): {e.__class__.__name__}")
components.html(html, height=1250, scrolling=True)
