"""Study planning: checked against R ``presize`` output, hand values and invariants."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from slicepoint_core import planning as P
from slicepoint_core import stats as S

# --------------------------------------------------------------------------- AUC precision

# Output of R presize::prec_auc(auc, prev = 12078/21136, n = 500), conf.level 0.95: columns
# auc, lwr, upr, conf.width. Recorded from the AI-Assess prototype (reprCohortSampleSize.csv).
PREV = 12078 / 21136
PRESIZE_AUC = [
    (0.75, 0.708083, 0.791917, 0.08383391),
    (0.80, 0.7620785, 0.8379215, 0.07584303),
    (0.85, 0.8169995, 0.8830005, 0.06600094),
    (0.90, 0.8731242, 0.9268758, 0.05375164),
    (0.95, 0.93121, 0.96879, 0.03757992),
]


@pytest.mark.parametrize("auc,lwr,upr,width", PRESIZE_AUC)
def test_auc_precision_matches_presize(auc: float, lwr: float, upr: float, width: float) -> None:
    r = P.auc_precision(auc, 500, PREV)
    assert (r.lower, r.upper, r.width) == pytest.approx((lwr, upr, width), abs=1e-6)
    assert r.n_pos == pytest.approx(285.721, abs=1e-3) and r.n_neg == pytest.approx(214.279, abs=1e-3)


def test_auc_precision_matches_presize_documented_example() -> None:
    # presize documentation example: prec_auc(auc = 0.65, n = 500, prev = 0.1)
    r = P.auc_precision(0.65, 500, 0.1)
    assert (r.n_pos, r.n_neg) == pytest.approx((50, 450))
    assert (r.lower, r.upper, r.width) == pytest.approx((0.5639623, 0.7360377, 0.1720755), abs=1e-6)


def test_auc_standard_error_hand_value() -> None:
    # A = 0.8, 10 positives, 10 negatives: Q1 = 2/3, Q2 = 1.28/1.8
    q1, q2 = 0.8 / 1.2, 1.28 / 1.8
    expected = math.sqrt((0.16 + 9 * (q1 - 0.64) + 9 * (q2 - 0.64)) / 100)
    assert P.auc_standard_error(0.8, 10, 10) == pytest.approx(expected, abs=1e-15)


def test_auc_precision_from_counts_agrees_and_narrows_with_n() -> None:
    a = P.auc_precision(0.85, 400, 0.25)
    b = P.auc_precision_from_counts(0.85, 100, 300)
    assert (a.lower, a.upper, a.prevalence) == pytest.approx((b.lower, b.upper, b.prevalence))
    widths = [P.auc_precision(0.85, n, 0.25).width for n in (50, 100, 400, 1600)]
    assert widths == sorted(widths, reverse=True)
    # a small, imbalanced subgroup is far less precise than the plan as a whole
    assert P.auc_precision_from_counts(0.85, 8, 20).width > 2 * a.width


@pytest.mark.parametrize("auc,prev,width", [(0.85, PREV, 0.066), (0.9, 0.1, 0.10), (0.7, 0.5, 0.05), (0.95, 0.02, 0.08)])
def test_auc_sample_size_is_the_smallest_n(auc: float, prev: float, width: float) -> None:
    r = P.auc_sample_size(auc, prev, width)
    assert r.n == int(r.n) and r.width <= width
    assert P.auc_precision(auc, r.n - 1, prev).width > width


def test_auc_sample_size_round_trips_presize() -> None:
    # 500 cases gave a width of 0.06600094 at AUC 0.85; asking for that width returns 500.
    assert P.auc_sample_size(0.85, PREV, 0.066001).n == 500


def test_conf_level_changes_the_width() -> None:
    w90, w95, w99 = (P.auc_precision(0.8, 300, 0.3, conf_level=c).width for c in (0.90, 0.95, 0.99))
    assert w90 < w95 < w99
    assert w95 / w90 == pytest.approx(1.959963984540054 / 1.6448536269514722, rel=1e-9)


@pytest.mark.parametrize("kwargs", [
    {"auc": 1.0, "n": 100, "prevalence": 0.5},
    {"auc": 0.8, "n": 0, "prevalence": 0.5},
    {"auc": 0.8, "n": 100, "prevalence": 0.0},
    {"auc": 0.8, "n": 100, "prevalence": 0.5, "conf_level": 1.0},
])
def test_auc_precision_rejects_bad_input(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        P.auc_precision(**kwargs)


def test_sample_size_rejects_impossible_width() -> None:
    with pytest.raises(ValueError):
        P.auc_sample_size(0.8, 0.5, 0.0)
    with pytest.raises(ValueError):
        P.auc_sample_size(0.8, 0.5, 1e-9)


# --------------------------------------------------------------------------- proportions


def test_proportion_precision_matches_presize_documented_example() -> None:
    # presize prec_sens(sens = 0.6, n = 50): sensadj 0.5928652, lwr 0.4618144, upr 0.7239161,
    # conf.width 0.2621017
    r = P.proportion_precision(0.6, 50)
    assert (r.center, r.lower, r.upper, r.width) == pytest.approx(
        (0.5928652, 0.4618144, 0.7239161, 0.2621017), abs=1e-6)


@pytest.mark.parametrize("k,n", [(0, 10), (10, 10), (5, 10), (18, 20), (69, 73), (1, 100)])
def test_proportion_precision_agrees_with_observed_wilson(k: int, n: int) -> None:
    lo, hi = S.wilson_interval(k, n) or (math.nan, math.nan)
    r = P.proportion_precision(k / n, n)
    assert (r.lower, r.upper) == pytest.approx((lo, hi), abs=1e-12)


def test_sensitivity_and_specificity_use_their_own_class() -> None:
    sens = P.sensitivity_precision(0.9, 400, 0.25)
    spec = P.specificity_precision(0.9, 400, 0.25)
    assert (sens.n, spec.n, sens.n_total, spec.prevalence) == pytest.approx((100, 300, 400, 0.25))
    assert sens.width == pytest.approx(P.proportion_precision(0.9, 100).width)
    assert spec.width < sens.width  # three times as many negatives


@pytest.mark.parametrize("p,width", [(0.9, 0.10), (0.5, 0.10), (0.99, 0.02), (0.7, 0.25)])
def test_proportion_sample_size_is_the_smallest_n(p: float, width: float) -> None:
    r = P.proportion_sample_size(p, width)
    assert r.n == int(r.n) and r.width <= width
    assert P.proportion_precision(p, r.n - 1).width > width


# --------------------------------------------------------------------------- allocation


def _population() -> pd.DataFrame:
    # 1000 people; positives are more common among B than A, vendors are independent of both
    rows = (
        [("A", "v1", "positive")] * 120 + [("A", "v2", "positive")] * 80
        + [("B", "v1", "positive")] * 90 + [("B", "v2", "positive")] * 60
        + [("A", "v1", "negative")] * 240 + [("A", "v2", "negative")] * 160
        + [("B", "v1", "negative")] * 150 + [("B", "v2", "negative")] * 100
    )
    frame: pd.DataFrame = pd.DataFrame(rows, columns=["group", "vendor", "finding"])
    return frame


def test_stratum_shares_are_joint_and_sum_to_one() -> None:
    s = P.stratum_shares(_population(), ["group", "vendor"])
    assert s.sum() == pytest.approx(1.0)
    assert s[("A", "v1")] == pytest.approx(0.36) and s[("B", "v2")] == pytest.approx(0.16)
    assert P.stratum_shares(_population(), ["group"])["A"] == pytest.approx(0.6)


def test_stratum_shares_reject_missing_values_and_columns() -> None:
    pop = _population()
    with pytest.raises(ValueError):
        P.stratum_shares(pop, ["scanner"])
    with pytest.raises(ValueError):
        P.stratum_shares(pop.assign(group=pop["group"].where(pop.index > 0)), ["group"])
    with pytest.raises(ValueError):
        P.stratum_shares(pop, [])


def test_allocate_sums_exactly_where_rounding_does_not() -> None:
    shares = {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
    assert sum(round(v * 100) for v in shares.values()) == 99  # per-stratum rounding loses a case
    out = P.allocate(shares, 100)
    assert int(out.sum()) == 100 and out.tolist() == [34, 33, 33]


def test_allocate_largest_remainder_hand_value() -> None:
    # quotas 4.7, 3.2, 2.1 -> floors 4, 3, 2 and the one case left goes to the .7
    assert P.allocate({"x": 0.47, "y": 0.32, "z": 0.21}, 10).to_dict() == {"x": 5, "y": 3, "z": 2}
    assert P.allocate({"x": 47, "y": 32, "z": 21}, 10).to_dict() == {"x": 5, "y": 3, "z": 2}  # normalized
    assert P.allocate({"x": 1.0, "y": 0.0}, 7).to_dict() == {"x": 7, "y": 0}
    assert int(P.allocate({"x": 0.5, "y": 0.5}, 0).sum()) == 0


@pytest.mark.parametrize("n", [1, 7, 97, 500, 1234])
def test_allocate_is_within_one_of_the_exact_quota(n: int) -> None:
    rng = np.random.default_rng(0)
    shares = pd.Series(rng.dirichlet(np.ones(11)), index=[f"s{i}" for i in range(11)])
    out = P.allocate(shares, n)
    assert int(out.sum()) == n
    assert bool((np.abs(out.to_numpy() - shares.to_numpy() * n) < 1.0).all())


def test_allocate_rejects_bad_shares() -> None:
    for bad in ({}, {"a": -0.1, "b": 1.1}, {"a": 0.0}):
        with pytest.raises(ValueError):
            P.allocate(bad, 10)
    with pytest.raises(ValueError):
        P.allocate({"a": 1.0}, -1)


def test_allocation_table_follows_each_class() -> None:
    t = P.allocation_table(_population(), ["group", "vendor"], {"positive": 70, "negative": 130}, outcome="finding")
    assert t.groupby("finding")["target_n"].sum().to_dict() == {"negative": 130, "positive": 70}
    pos = t[t["finding"] == "positive"].set_index(["group", "vendor"])["target_n"]
    # positives are 120/80/90/60 of 350 -> quotas 24, 16, 18, 12
    assert pos.to_dict() == {("A", "v1"): 24, ("A", "v2"): 16, ("B", "v1"): 18, ("B", "v2"): 12}
    assert t.groupby("finding")["share"].sum().tolist() == pytest.approx([1.0, 1.0])


def test_allocation_table_rejects_unknown_class_and_outcome_in_by() -> None:
    with pytest.raises(ValueError):
        P.allocation_table(_population(), ["group"], {"indeterminate": 10}, outcome="finding")
    with pytest.raises(ValueError):
        P.allocation_table(_population(), ["group", "finding"], {"positive": 10}, outcome="finding")


# --------------------------------------------------------------------------- plan vs. available


def test_compare_to_available_marks_met_short_and_empty() -> None:
    plan = P.allocation_table(_population(), ["group", "vendor"], {"positive": 70}, outcome="finding")
    available = pd.DataFrame({
        "finding": ["positive", "positive", "positive", "negative"],
        "group": ["A", "A", "B", "A"],
        "vendor": ["v1", "v2", "v1", "v1"],
        "available_n": [40, 4, 18, 999],
    })
    out = P.compare_to_available(plan, available, on=["finding", "group", "vendor"])
    got = out.set_index(["group", "vendor"])[["target_n", "available_n", "shortfall", "status"]]
    assert got.loc[("A", "v1")].tolist() == [24, 40, 0, "met"]
    assert got.loc[("A", "v2")].tolist() == [16, 4, 12, "short"]
    assert got.loc[("B", "v1")].tolist() == [18, 18, 0, "met"]
    assert got.loc[("B", "v2")].tolist() == [12, 0, 12, "empty"]
    assert out.set_index(["group", "vendor"]).loc[("A", "v2"), "fill"] == pytest.approx(0.25)
    assert len(out) == len(plan)  # the unplanned negative stratum is left out


def test_compare_to_available_rejects_duplicates_and_missing_columns() -> None:
    plan = pd.DataFrame({"g": ["a"], "target_n": [5]})
    with pytest.raises(ValueError):
        P.compare_to_available(plan, pd.DataFrame({"g": ["a", "a"], "available_n": [1, 2]}), on=["g"])
    with pytest.raises(ValueError):
        P.compare_to_available(plan, pd.DataFrame({"g": ["a"], "count": [1]}), on=["g"])


def test_pool_levels_offsets_a_gap_with_a_surplus() -> None:
    table = pd.DataFrame({
        "finding": ["positive"] * 4,
        "ethnicity": ["White", "Black", "Hispanic", "Asian"],
        "target_n": [75, 24, 24, 24],
        "available_n": [163, 40, 69, 4],
    })
    table["shortfall"] = (table["target_n"] - table["available_n"]).clip(lower=0)
    assert int(table["shortfall"].sum()) == 20  # Asian is 20 short on its own
    out = P.pool_levels(table, "ethnicity", {"Non-White": ["Black", "Hispanic", "Asian"]}, within=["finding"])
    got = out.set_index("ethnicity")
    assert got.loc["Non-White", ["target_n", "available_n", "shortfall"]].tolist() == [72, 113, 0]
    assert got.loc["Non-White", "pooled_from"] == ("Black", "Hispanic", "Asian")
    assert got.loc["White", ["target_n", "available_n"]].tolist() == [75, 163]


def test_pool_levels_rejects_a_level_in_two_pools() -> None:
    table = pd.DataFrame({"e": ["a", "b"], "target_n": [1, 1]})
    with pytest.raises(ValueError):
        P.pool_levels(table, "e", {"x": ["a"], "y": ["a", "b"]})
    with pytest.raises(ValueError):
        P.pool_levels(pd.DataFrame({"e": ["a"]}), "e", {"x": ["a"]})


def test_end_to_end_plan() -> None:
    """Size the set for an AUC interval, spread it over strata, compare with what exists."""
    size = P.auc_sample_size(0.85, 0.35, 0.08)
    targets = {"positive": round(size.n_pos), "negative": round(size.n_neg)}
    plan = P.allocation_table(_population(), ["group"], targets, outcome="finding")
    assert int(plan["target_n"].sum()) == sum(targets.values())
    available = pd.DataFrame({"finding": ["positive", "positive", "negative", "negative"],
                              "group": ["A", "B", "A", "B"], "available_n": [500, 9, 500, 500]})
    gap = P.compare_to_available(plan, available, on=["finding", "group"])
    short = gap[gap["status"] == "short"]
    assert short[["finding", "group"]].to_numpy().tolist() == [["positive", "B"]]
    # what group B can still support: far wider than the plan's 0.08
    b = P.auc_precision_from_counts(0.85, 9, 500)
    assert b.width > 2 * 0.08
