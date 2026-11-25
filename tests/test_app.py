"""Headless smoke test of every Streamlit screen (needs built runs: `make all`)."""

from pathlib import Path

import pytest

from expengine.config import DATA

APP = Path(__file__).resolve().parents[1] / "app"
pytestmark = pytest.mark.skipif(not (DATA / "runs" / "runs.parquet").exists(), reason="no runs built")


@pytest.mark.parametrize("page", ["Home.py", "pages/1_Experiment_Audit.py", "pages/2_Planner.py",
                                  "pages/3_Simulation_Lab.py", "pages/4_Readout.py"])
@pytest.mark.parametrize("source", ["synthetic", "upworthy archive"])
def test_page_renders_without_exceptions(page, source):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP / page), default_timeout=60)
    at.session_state["source"] = source
    at.run()
    assert not at.exception, at.exception
