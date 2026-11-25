"""Write tiny synthetic CSVs in the Upworthy archive format for tests and CI.

Each test id encodes one edge case; none of this is real archive data.
Run: uv run python scripts/make_fixtures.py
"""

import csv
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "archive"
HEADER = ["", "created_at", "updated_at", "clickability_test_id", "excerpt", "headline", "lede", "slug",
          "eyecatcher_id", "impressions", "clicks", "significance", "first_place", "winner", "share_text", "square", "test_week"]
UNDEPLOYED_HEADER = ["_id"] + HEADER[1:-1]

idx = 0


def pkg(test, created, imp, clk, headline, eye="img1"):
    global idx
    idx += 1
    return [idx, created, "2016-04-02 16:00:00", test, "", headline, "<p>lede</p>", f"slug-{idx}", eye,
            imp, clk, "50.0", "False", "False", "", "", "201445"]


exploratory = [
    # T-CLEAN: balanced two-arm test
    pkg("t_clean", "2014-11-20 10:00:00", 5000, 60, "Headline A"),
    pkg("t_clean", "2014-11-20 10:05:00", 5012, 85, "Headline B"),
    # T-SRM: 60/40 impressions under the equal-allocation assumption
    pkg("t_srm", "2014-11-21 10:00:00", 6000, 70, "SRM A"),
    pkg("t_srm", "2014-11-21 10:01:00", 4000, 50, "SRM B"),
    # T-SINGLE: only one arm
    pkg("t_single", "2014-11-22 09:00:00", 3000, 30, "Lonely"),
    # T-ZERO: a zero-click reference arm (sparse; lift undefined)
    pkg("t_zero", "2014-11-23 09:00:00", 400, 0, "Zero A"),
    pkg("t_zero", "2014-11-23 09:10:00", 410, 3, "Zero B"),
]
confirmatory = [
    # T-BADROW: clicks exceed impressions in one arm -> quarantined, experiment excluded
    pkg("t_badrow", "2014-12-01 09:00:00", 1000, 10, "Bad A"),
    pkg("t_badrow", "2014-12-01 09:01:00", 1000, 2000, "Bad B"),
    # T-BADTIME: unparseable timestamp in one arm
    pkg("t_badtime", "not-a-date", 1000, 10, "Time A"),
    pkg("t_badtime", "2014-12-02 09:00:00", 1000, 12, "Time B"),
    # T-WINDOW: inside the documented randomization-problem window
    pkg("t_window", "2013-08-01 09:00:00", 5000, 60, "Window A"),
    pkg("t_window", "2013-08-01 09:02:00", 5000, 61, "Window B"),
    # T-MULTI: balanced four-arm test, one duplicated stimulus
    pkg("t_multi", "2014-12-03 09:00:00", 2500, 25, "Multi A"),
    pkg("t_multi", "2014-12-03 09:01:00", 2490, 40, "Multi B"),
    pkg("t_multi", "2014-12-03 09:02:00", 2510, 26, "Multi C"),
    pkg("t_multi", "2014-12-03 09:03:00", 2505, 26, "Multi C"),
    # T-TINY: expected counts below 5 -> Monte Carlo SRM
    pkg("t_tiny", "2014-12-04 09:00:00", 3, 1, "Tiny A"),
    pkg("t_tiny", "2014-12-04 09:01:00", 4, 0, "Tiny B"),
    # T-SHARED: also has an undeployed draft package
    pkg("t_shared", "2014-12-05 09:00:00", 3000, 30, "Shared A"),
    pkg("t_shared", "2014-12-05 09:01:00", 3010, 33, "Shared B"),
]
holdout = [
    pkg("t_hold", "2015-01-10 09:00:00", 2000, 20, "Hold A"),
    pkg("t_hold", "2015-01-10 09:01:00", 2003, 22, "Hold B"),
]
undeployed = [
    ["u1", "2015-02-01 09:00:00", "2016-04-02", "t_undeployed", "", "Draft A", "", "s", "e", "", "", "", "", "", "", ""],
    ["u2", "2015-02-01 09:01:00", "2016-04-02", "t_undeployed", "", "Draft B", "", "s", "e", "", "", "", "", "", "", ""],
    ["u3", "2015-02-02 09:00:00", "2016-04-02", "", "", "Orphan", "", "s", "e", "", "", "", "", "", "", ""],
    ["u4", "2014-12-05 09:05:00", "2016-04-02", "t_shared", "", "Shared draft", "", "s", "e", "0", "0", "", "", "", "", ""],
]


def write(name, header, rows):
    with (OUT / name).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


OUT.mkdir(parents=True, exist_ok=True)
write("exploratory.csv", HEADER, exploratory)
write("confirmatory.csv", HEADER, confirmatory)
write("holdout.csv", HEADER, holdout)
write("undeployed.csv", UNDEPLOYED_HEADER, undeployed)
print(f"wrote fixtures to {OUT}")
