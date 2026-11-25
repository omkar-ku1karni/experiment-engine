import json
from pathlib import Path

import pandas as pd
import pytest

from expengine import audit, ingest
from expengine.config import load_yaml

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def fixture_run(tmp_path_factory):
    """Ingest + audit the tiny archive fixtures into a temporary workspace."""
    root = tmp_path_factory.mktemp("ws")
    cfg_path = FIX / "archive_fixture.yaml"
    cfg = load_yaml(cfg_path)
    processed, runs = root / "processed", root / "runs"
    manifest = ingest.run(cfg, FIX / "archive", processed)
    summary = audit.run(processed, runs, str(cfg_path))
    run_dir = runs / summary["run_id"]
    return {
        "cfg": cfg, "manifest": manifest, "summary": summary, "run_dir": run_dir,
        "experiments": pd.read_parquet(processed / "experiments.parquet").set_index("experiment_id"),
        "arms": pd.read_parquet(processed / "arms.parquet"),
        "decisions": pd.read_parquet(run_dir / "decisions.parquet").set_index("experiment_id"),
        "audit": pd.read_parquet(run_dir / "audit_results.parquet"),
        "srm": pd.read_parquet(run_dir / "srm.parquet").set_index("experiment_id"),
        "comparisons": pd.read_parquet(run_dir / "comparisons.parquet"),
        "processed": processed, "runs": runs, "cfg_path": cfg_path,
    }
