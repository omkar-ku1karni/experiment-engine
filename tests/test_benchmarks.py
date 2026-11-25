"""Smoke-run every benchmark at low replication and check its qualitative claims."""

import pandas as pd

from expengine import benchmarks


def test_benchmarks_smoke(tmp_path):
    benchmarks.run(reps=1500, out=tmp_path)
    seq = pd.read_csv(tmp_path / "sequential.csv")
    null = seq[seq["effect_multiple"] == 0].groupby("policy")["rejection_rate"].mean()
    assert null["naive_peeking"] > 0.12 > null["group_sequential_obf"]
    assert abs(null["fixed"] - 0.05) < 0.02
    alt = seq[seq["effect_multiple"] == 1].groupby("policy")["mean_stop_fraction"].mean()
    assert alt["group_sequential_obf"] < 0.9 < alt["fixed"] + 0.01
    cu = pd.read_csv(tmp_path / "cuped.csv")
    row = cu[(cu.outcome == "continuous") & (cu.rho == 0.8) & (cu.missing == 0) & cu.effect.isna()].iloc[0]
    assert abs(row["variance_ratio"] - 0.36) < 0.05
    srm = pd.read_csv(tmp_path / "srm.csv")
    assert srm[srm.case == "60/40 under 50/50"]["srm_flag_rate"].min() > 0.99
