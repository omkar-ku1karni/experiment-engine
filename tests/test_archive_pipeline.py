"""End-to-end ingest + audit on tiny synthetic archive fixtures."""

import json

import pandas as pd


def test_every_source_row_is_accounted_for(fixture_run):
    m = fixture_run["manifest"]
    rows_in = sum(f["rows"] for f in m["files"])
    assert rows_in == len(fixture_run["arms"]) == m["summary"]["source_rows"] == 27
    assert m["summary"]["rows_missing_test_id"] == 1
    assert m["summary"]["deployed_tests"] == 11 and m["summary"]["deployed_matches_benchmark"]
    assert all(f["sha256_matches_config"] for f in m["files"])


def test_quarantine_and_exclusions(fixture_run):
    e = fixture_run["experiments"]
    assert e.loc["t_single", "exclusion_reason"] == "fewer_than_two_arms"
    assert e.loc["t_badrow", "exclusion_reason"] == "quarantined_arm"
    assert e.loc["t_badtime", "exclusion_reason"] == "quarantined_arm"
    assert e.loc["t_undeployed", "exclusion_reason"] == "undeployed_split"
    assert e.loc["t_shared", "included"] and e.loc["t_shared", "in_multiple_splits"]
    a = fixture_run["arms"]
    issues = a.loc[~a["row_valid"], "row_issues"].str.split(",").explode()
    assert {"CLICKS_EXCEED_IMPRESSIONS", "INVALID_TIMESTAMP", "MISSING_COUNTS", "MISSING_TEST_ID"} <= set(issues)
    shared = a[(a["experiment_id"] == "t_shared") & (a["split"] == "undeployed")]
    assert (shared["arm_exclusion"] == "undeployed_package_of_deployed_test").all()


def test_srm_blocks_and_problem_window_blocks(fixture_run):
    d = fixture_run["decisions"]
    assert d.loc["t_srm", "recommendation"] == "blocked" and "BLK_SRM" in d.loc["t_srm", "reason_codes"]
    assert "BLK_DOCUMENTED_RANDOMIZATION_PROBLEM" in d.loc["t_window", "reason_codes"]
    assert d.loc["t_clean", "recommendation"] == "inconclusive" and not d.loc["t_clean", "actionable"]
    assert fixture_run["srm"].loc["t_tiny", "srm_method"] == "monte_carlo_multinomial"


def test_no_archive_record_is_ever_actionable(fixture_run):
    d = fixture_run["decisions"]
    assert not d["actionable"].any() and "ship" not in set(d["recommendation"])


def test_sparse_and_multi_arm_comparisons(fixture_run):
    c = fixture_run["comparisons"]
    z = c[c["experiment_id"] == "t_zero"].iloc[0]
    assert z["test"] == "fisher_exact" and z["lift_undefined"]
    multi = c[c["experiment_id"] == "t_multi"]
    assert len(multi) == 3 and (multi["p_holm"] >= multi["p_value"]).all()
    audit = fixture_run["audit"]
    dup = audit[(audit["experiment_id"] == "t_multi") & (audit["rule_id"] == "R007_DUPLICATE_STIMULUS")]
    assert dup["status"].iloc[0] == "fail"


def test_audit_is_reproducible(fixture_run, tmp_path):
    from expengine import audit
    s2 = audit.run(fixture_run["processed"], tmp_path, str(fixture_run["cfg_path"]))
    assert s2["run_id"] == fixture_run["summary"]["run_id"]
    a = pd.read_parquet(tmp_path / s2["run_id"] / "decisions.parquet")
    b = fixture_run["decisions"].reset_index()[a.columns]
    pd.testing.assert_frame_equal(a.sort_values("experiment_id").reset_index(drop=True),
                                  b.sort_values("experiment_id").reset_index(drop=True))
