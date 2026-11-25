"""Download, verify and normalise the Upworthy Research Archive.

    data/raw/*.csv  (immutable, SHA-256 verified)
      -> data/processed/arms.parquet         one row per package (arm), all splits, with validity flags
      -> data/processed/experiments.parquet  one row per clickability_test_id, with inclusion decision
      -> data/processed/quarantine.parquet   malformed rows, raw values kept, with reason codes
      -> data/processed/manifest.json        source URLs, checksums, schema, counts, ingestion time

Nothing is silently dropped: every source row lands in `arms` (and, if malformed, in
`quarantine` too) and every test id lands in `experiments` with an explicit inclusion reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from expengine.config import DATA, load_yaml, sha256_file, stable_hash

SCHEMA_VERSION = 1
TEXT_COLUMNS = ["headline", "excerpt", "eyecatcher_id", "slug"]


# --------------------------------------------------------------------------- download
def download(cfg: dict, raw_dir: Path, force: bool = False) -> None:
    raw_dir.mkdir(parents=True, exist_ok=True)
    for f in cfg["files"]:
        path = raw_dir / f["file"]
        if path.exists() and not force and sha256_file(path) == f["sha256"]:
            print(f"ok        {f['file']}")
            continue
        print(f"download  {f['url']} -> {path.name}")
        req = urllib.request.Request(f["url"], headers={"User-Agent": "experiment-engine/0.1"})
        tmp = path.with_suffix(".part")
        with urllib.request.urlopen(req, timeout=600) as r, tmp.open("wb") as out:
            while chunk := r.read(1 << 20):
                out.write(chunk)
        digest = sha256_file(tmp)
        if digest != f["sha256"]:
            tmp.unlink()
            raise SystemExit(f"Checksum mismatch for {f['file']}: got {digest}, expected {f['sha256']}")
        tmp.rename(path)


# --------------------------------------------------------------------------- read
def count_csv_records(path: Path) -> int:
    """Independent record count with the stdlib parser (handles quoted newlines)."""
    csv.field_size_limit(1 << 30)
    with path.open(newline="", encoding="utf-8") as f:
        return sum(1 for _ in csv.reader(f)) - 1


def read_split(path: Path, split: str) -> pd.DataFrame:
    # pandas' C parser silently drops 57 multi-line records in the undeployed export,
    # so the files are read with DuckDB and checked against the stdlib csv parser.
    con = duckdb.connect()
    df = con.execute(
        "select * from read_csv(?, header = true, all_varchar = true, strict_mode = true)", [str(path)]
    ).df()
    con.close()
    expected = count_csv_records(path)
    if len(df) != expected:
        raise SystemExit(f"{path.name}: read {len(df)} records, stdlib csv parser sees {expected}")
    df.columns = [c if c.strip() else "_index" for c in pd.read_csv(path, nrows=0).columns]
    first = df.columns[0]
    if first == "_id":                       # undeployed export uses Mongo ids
        arm_id = df["_id"]
    else:                                    # deployed exports carry a global package index
        arm_id = "pkg-" + df[first].astype(str)
    out = pd.DataFrame(
        {
            "arm_id": arm_id,
            "experiment_id": df["clickability_test_id"],
            "split": split,
            "created_at_raw": df["created_at"],
            "impressions_raw": df["impressions"],
            "clicks_raw": df["clicks"],
            "vendor_first_place": df.get("first_place"),
            "vendor_winner": df.get("winner"),
            # Upworthy's own dashboard field; the archive notes it is NOT a significance estimate.
            "vendor_significance": df.get("significance"),
            "test_week": df.get("test_week"),
            "source_file": path.name,
            "source_row": np.arange(1, len(df) + 1),
        }
    )
    for col in TEXT_COLUMNS:
        out[col] = df.get(col)
    return out


# --------------------------------------------------------------------------- validate
def _parse_count(raw: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    num = pd.to_numeric(raw, errors="coerce")
    missing = raw.isna()
    nonnumeric = raw.notna() & num.isna()
    nonint = num.notna() & (num != np.floor(num))
    return num, missing, nonnumeric | nonint


def validate_arms(arms: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Row-level schema rules. Adds typed columns plus `row_issues` (list of reason codes)."""
    a = arms.copy()
    imp, imp_missing, imp_bad = _parse_count(a["impressions_raw"])
    clk, clk_missing, clk_bad = _parse_count(a["clicks_raw"])
    created = pd.to_datetime(a["created_at_raw"], errors="coerce", format="mixed")

    checks = {
        "MISSING_TEST_ID": a["experiment_id"].isna(),
        "MISSING_COUNTS": imp_missing | clk_missing,
        "NONINTEGER_COUNTS": imp_bad | clk_bad,
        "NEGATIVE_COUNTS": (imp < 0) | (clk < 0),
        "CLICKS_EXCEED_IMPRESSIONS": clk > imp,
        "INVALID_TIMESTAMP": created.isna(),
        "DUPLICATE_ARM_KEY": a.duplicated("arm_id", keep=False),
    }
    issues = pd.DataFrame(checks)
    a["row_issues"] = issues.apply(lambda r: [k for k, v in r.items() if bool(v)], axis=1)
    a["row_valid"] = ~issues.any(axis=1)
    a["impressions"] = imp.where(a["row_valid"]).astype("Int64")
    a["clicks"] = clk.where(a["row_valid"]).astype("Int64")
    a["created_at"] = created

    start = pd.Timestamp(cfg["problem_window"]["start"])
    end = pd.Timestamp(cfg["problem_window"]["end"]) + pd.Timedelta(days=1)
    a["_in_window"] = (created >= start) & (created < end)
    return a


def build_experiments(arms: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per test id with an explicit inclusion decision. Also marks arm inclusion."""
    deployed = set(cfg["deployed_splits"])
    orphans = arms[arms["experiment_id"].isna()].copy()   # kept in `arms`, never analysed
    a = arms[arms["experiment_id"].notna()].copy()
    a["is_deployed_split"] = a["split"].isin(deployed)

    splits = a.groupby("experiment_id")["split"].agg(lambda s: sorted(set(s)))
    has_deployed = a.groupby("experiment_id")["is_deployed_split"].any()

    # An id present in a deployed split is analysed from its deployed packages only;
    # undeployed drafts sharing that id are kept in `arms` but excluded.
    a["arm_included"] = a["is_deployed_split"] & a["row_valid"]
    a.loc[~a["is_deployed_split"] & a["experiment_id"].map(has_deployed), "arm_exclusion"] = (
        "undeployed_package_of_deployed_test"
    )
    a.loc[~a["is_deployed_split"] & ~a["experiment_id"].map(has_deployed), "arm_exclusion"] = "undeployed_split"
    a.loc[a["is_deployed_split"] & ~a["row_valid"], "arm_exclusion"] = "quarantined_row"

    scope = a[a["is_deployed_split"] | ~a["experiment_id"].map(has_deployed)]
    g = scope.groupby("experiment_id")
    exp = pd.DataFrame(
        {
            "n_arms_source": g.size(),
            "n_arms_valid": g["row_valid"].sum(),
            "first_created_at": g["created_at"].min(),
            "last_created_at": g["created_at"].max(),
            "impressions": g["impressions"].sum(min_count=1),
            "clicks": g["clicks"].sum(min_count=1),
            "any_in_problem_window": g["_in_window"].any(),
        }
    )
    exp["split"] = scope.groupby("experiment_id")["split"].agg(lambda s: sorted(set(s))[0])
    exp["all_splits"] = splits.reindex(exp.index).map(lambda s: ",".join(s))
    exp["in_multiple_splits"] = splits.reindex(exp.index).map(len) > 1
    exp["source"] = cfg["source"]
    exp["experiment_date"] = exp["first_created_at"].dt.date
    # Documented-rule reconstruction of the archive's randomization-problem indicator,
    # assigned by the test's first package creation time.
    start = pd.Timestamp(cfg["problem_window"]["start"])
    end = pd.Timestamp(cfg["problem_window"]["end"]) + pd.Timedelta(days=1)
    exp["problem_reconstructed"] = (exp["first_created_at"] >= start) & (exp["first_created_at"] < end)
    exp["period"] = np.select(
        [exp["first_created_at"] < start, exp["problem_reconstructed"]],
        ["before_window", "problem_window"],
        "after_window",
    )

    # Design metadata the archive does not record stays null (never invented).
    exp["expected_allocation"] = "equal (assumed; not recorded in source)"
    exp["control_definition"] = None
    exp["primary_metric"] = "ctr"
    exp["alpha"] = None
    exp["planned_horizon"] = None
    exp["mde"] = None
    exp["analysis_mode"] = "historical_exploratory"

    # Predeclared exploratory reference arm: earliest-created valid package (ties: arm_id).
    valid = scope[scope["row_valid"]].sort_values(["experiment_id", "created_at", "arm_id"])
    exp["reference_arm_id"] = valid.groupby("experiment_id")["arm_id"].first().reindex(exp.index)

    reason = pd.Series(None, index=exp.index, dtype=object)
    is_deployed = exp["split"].isin(deployed)
    reason[~is_deployed] = "undeployed_split"
    reason[is_deployed & (exp["n_arms_valid"] < exp["n_arms_source"])] = "quarantined_arm"
    reason[is_deployed & reason.isna() & (exp["n_arms_source"] < 2)] = "fewer_than_two_arms"
    reason[is_deployed & reason.isna() & (exp["impressions"].fillna(0) <= 0)] = "zero_impressions"
    exp["exclusion_reason"] = reason
    exp["included"] = reason.isna()

    exp = exp.reset_index().rename(columns={"index": "experiment_id"})
    a = a.drop(columns=["_in_window"])
    a["problem_reconstructed"] = a["experiment_id"].map(exp.set_index("experiment_id")["problem_reconstructed"])
    a.loc[a["arm_included"] & ~a["experiment_id"].map(exp.set_index("experiment_id")["included"]), "arm_exclusion"] = (
        "experiment_excluded"
    )
    a["arm_included"] = a["arm_included"] & a["experiment_id"].map(exp.set_index("experiment_id")["included"])

    orphans = orphans.drop(columns=["_in_window"])
    orphans["is_deployed_split"] = orphans["split"].isin(deployed)
    orphans["arm_included"] = False
    orphans["arm_exclusion"] = "missing_test_id"
    orphans["problem_reconstructed"] = pd.NA
    return exp, pd.concat([a, orphans], ignore_index=True)


# --------------------------------------------------------------------------- run
def run(cfg: dict, raw_dir: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames, files = [], []
    for f in cfg["files"]:
        path = raw_dir / f["file"]
        if not path.exists():
            raise SystemExit(f"Missing {path}. Run `make download` first.")
        digest = sha256_file(path)
        df = read_split(path, f["split"])
        frames.append(df)
        files.append(
            {
                "split": f["split"],
                "file": f["file"],
                "url": f.get("url"),
                "sha256": digest,
                "sha256_matches_config": digest == f.get("sha256"),
                "bytes": path.stat().st_size,
                "rows": len(df),
                "distinct_test_ids": int(df["experiment_id"].nunique()),
                "distinct_arm_ids": int(df["arm_id"].nunique()),
                "columns": list(pd.read_csv(path, nrows=0).columns),
            }
        )
    arms = validate_arms(pd.concat(frames, ignore_index=True), cfg)
    experiments, arms = build_experiments(arms, cfg)

    quarantine = arms[~arms["row_valid"]].copy()
    quarantine["row_issues"] = quarantine["row_issues"].map(",".join)
    arms_out = arms.copy()
    arms_out["row_issues"] = arms_out["row_issues"].map(",".join)

    arms_out.to_parquet(out_dir / "arms.parquet", index=False)
    experiments.to_parquet(out_dir / "experiments.parquet", index=False)
    quarantine.to_parquet(out_dir / "quarantine.parquet", index=False)

    deployed = experiments[experiments["split"].isin(cfg["deployed_splits"])]
    summary = {
        "source_rows": int(len(arms)),
        "rows_missing_test_id": int(arms["experiment_id"].isna().sum()),
        "distinct_test_ids_all_files": int(experiments["experiment_id"].nunique()),
        "deployed_tests": int(len(deployed)),
        "published_benchmark_tests": cfg.get("published_benchmark_tests"),
        "deployed_matches_benchmark": int(len(deployed)) == cfg.get("published_benchmark_tests"),
        "deployed_tests_by_split": deployed["split"].value_counts().sort_index().to_dict(),
        "ids_in_multiple_splits": int(experiments["in_multiple_splits"].sum()),
        "included_tests": int(experiments["included"].sum()),
        "excluded_by_reason": experiments["exclusion_reason"].value_counts().to_dict(),
        "quarantined_rows_by_issue": quarantine["row_issues"].str.split(",").explode().value_counts().to_dict(),
        "problem_reconstructed_deployed": int(deployed["problem_reconstructed"].sum()),
        "problem_reconstructed_share_deployed": round(float(deployed["problem_reconstructed"].mean()), 4),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source": cfg["source"],
        "release": cfg["release"],
        "ingested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config_hash": stable_hash(cfg),
        "input_hash": stable_hash([f["sha256"] for f in files]),
        "problem_flag": {
            "column": "problem_reconstructed",
            "rule": f"first package created between {cfg['problem_window']['start']} and "
            f"{cfg['problem_window']['end']} (inclusive)",
            "provenance": "Reconstructed from archive documentation; the OSF files predate the 2024 column.",
            "source_url": cfg["problem_window"]["source_url"],
        },
        "files": files,
        "summary": summary,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["download", "ingest"])
    ap.add_argument("--config", default="archive.yaml")
    ap.add_argument("--raw-dir", type=Path, default=DATA / "raw")
    ap.add_argument("--out-dir", type=Path, default=DATA / "processed")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.command == "download":
        download(cfg, args.raw_dir, args.force)
        return 0
    manifest = run(cfg, args.raw_dir, args.out_dir)
    print(json.dumps(manifest["summary"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
