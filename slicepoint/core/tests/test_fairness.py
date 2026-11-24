"""Subgroup metrics and fairness gaps: checked against hand counts, the functions they are
built from (``stats.wilson_interval``, ``roc``), a published Newcombe example, and simulation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from slicepoint_core import fairness as F
from slicepoint_core import roc as R
from slicepoint_core.stats import fisher_exact_p, wilson_interval


def _table(seed: int = 0, n: int = 900, *, weak: str | None = None, drop: float = 0.0) -> pd.DataFrame:
    """Synthetic cases: sex, vendor (some missing), a score, and a 0.5 threshold call.

    ``weak`` names a vendor whose positives score lower by ``drop``."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.35
    vendor = rng.choice(np.array(["A", "B", "C", None], dtype=object), n, p=[0.5, 0.3, 0.15, 0.05])
    score = rng.normal(0.0, 1.0, n) + 1.3 * y
    if weak is not None:
        score = score - drop * (y & (vendor == weak))
    table: pd.DataFrame = pd.DataFrame({
        "y": y, "sex": rng.choice(["F", "M"], n), "vendor": vendor, "score": score,
        "call": score >= 0.5, "person": np.arange(n),
    })
    return table


def _row(table: pd.DataFrame, **where: object) -> pd.Series:
    mask = np.ones(len(table), dtype=bool)
    for column, value in where.items():
        mask &= (table[column] == value).to_numpy()
    assert mask.sum() == 1, f"{where} matched {mask.sum()} rows"
    row: pd.Series = table[mask].iloc[0]
    return row


# --------------------------------------------------------------------------- building blocks


def test_newcombe_interval_published_example() -> None:
    # Newcombe (1998), Statistics in Medicine 17:873-890, Table II (a): 56/70 against 48/80.
    lower, upper = F.newcombe_interval(56, 70, 48, 80)
    assert (lower, upper) == pytest.approx((0.0524, 0.3339), abs=5e-5)


def test_newcombe_interval_is_antisymmetric_and_bounded() -> None:
    lo, hi = F.newcombe_interval(0, 12, 9, 9)
    assert -1.0 <= lo <= -1.0 + 0.4 and hi < 0
    back = F.newcombe_interval(9, 9, 0, 12)
    assert back == pytest.approx((-hi, -lo))


def test_holm_hand_example_and_missing() -> None:
    assert F.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])
    assert F.holm([0.5, None, 0.5]) == [1.0, None, 1.0]
    assert F.holm([None]) == [None]
    assert F.holm([]) == []


def test_holm_matches_scipy_free_definition() -> None:
    rng = np.random.default_rng(4)
    p = rng.random(9).tolist()
    order = np.argsort(p)
    expected = np.empty(9)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (9 - rank) * p[i]))
        expected[i] = running
    assert F.holm(p) == pytest.approx(expected.tolist())


# --------------------------------------------------------------------------- subgroup_metrics


def test_metrics_match_hand_counts_and_wilson() -> None:
    df = _table()
    out = F.subgroup_metrics(df, "sex", y_true="y", y_pred="call")
    assert set(out["metric"]) == set(F.METRICS[1:])
    f = df[df.sex == "F"]
    tp, fn = int((f.call & f.y).sum()), int((~f.call & f.y).sum())
    fp, tn = int((f.call & ~f.y).sum()), int((~f.call & ~f.y).sum())
    expected = {"sensitivity": (tp, tp + fn), "specificity": (tn, tn + fp), "ppv": (tp, tp + fp),
                "npv": (tn, tn + fn), "positive_rate": (tp + fp, len(f))}
    for metric, (k, n) in expected.items():
        row = _row(out, sex="F", metric=metric)
        assert (row["numerator"], row["denominator"], row["support"]) == (k, n, n)
        assert row["value"] == pytest.approx(k / n)
        assert (row["lower"], row["upper"]) == pytest.approx(wilson_interval(k, n))
        assert (row["method"], row["status"]) == ("wilson", "ok")
        assert (row["n"], row["n_units"], row["n_pos"], row["n_neg"]) == (len(f), len(f), tp + fn, tn + fp)


def test_auc_rows_match_delong() -> None:
    df = _table()
    out = F.subgroup_metrics(df, "vendor", y_true="y", y_score="score", conf_level=0.9)
    assert list(out["metric"].unique()) == ["auc"]
    sub = df[df.vendor == "B"]
    want = R.auc_delong(sub.y.to_numpy(), sub.score.to_numpy(), conf_level=0.9)
    row = _row(out, vendor="B")
    assert (row["value"], row["lower"], row["upper"]) == pytest.approx((want.auc, want.lower, want.upper))
    assert row["support"] == min(sub.y.sum(), (~sub.y).sum())
    assert row["method"] == "delong"


def test_threshold_equals_explicit_calls_and_default_metrics() -> None:
    df = _table()
    a = F.subgroup_metrics(df, "sex", y_true="y", y_score="score", threshold=0.5)
    assert list(a["metric"].unique()) == list(F.METRICS)
    b = F.subgroup_metrics(df, "sex", y_true="y", y_pred="call")
    pd.testing.assert_frame_equal(a[a.metric != "auc"].reset_index(drop=True), b, check_dtype=False)


def test_missing_factor_values_are_their_own_subgroup() -> None:
    df = _table()
    out = F.subgroup_metrics(df, "vendor", y_true="y", y_pred="call", metrics=["sensitivity"])
    assert set(out["vendor"]) == {"A", "B", "C", F.MISSING}
    assert out["n"].sum() == len(df)
    assert _row(out, vendor=F.MISSING)["n"] == df.vendor.isna().sum()


def test_joint_levels() -> None:
    df = _table()
    out = F.subgroup_metrics(df, ["sex", "vendor"], y_true="y", y_pred="call", metrics=["positive_rate"])
    assert len(out) == 8 and out["n"].sum() == len(df)
    row = _row(out, sex="M", vendor="C")
    assert row["n"] == ((df.sex == "M") & (df.vendor == "C")).sum()


def test_small_groups_are_not_estimated() -> None:
    df = _table(n=120)
    out = F.subgroup_metrics(df, "vendor", y_true="y", y_score="score", threshold=0.5, min_count=10)
    small = out[out.support < 10]
    assert len(small) > 0
    assert (small["status"] == "too_small").all() and small["value"].isna().all() and small["lower"].isna().all()
    assert (out[out.support >= 10]["status"] == "ok").all()
    assert (small["denominator"] >= small["support"]).all()  # counts are still reported


def test_integer_levels_and_categoricals() -> None:
    df = _table()
    df["site"] = np.arange(len(df)) % 3
    df["sex_cat"] = df["sex"].astype("category")
    out = F.subgroup_metrics(df, ["site", "sex_cat"], y_true="y", y_pred="call", metrics=["ppv"])
    assert sorted(out["site"].unique()) == [0, 1, 2] and len(out) == 6


# --------------------------------------------------------------------------- people with several cases


def _repeated(df: pd.DataFrame, copies: int) -> pd.DataFrame:
    """Every case ``copies`` times under the same person: no new information."""
    return pd.concat([df] * copies, ignore_index=True)


def test_repeated_cases_do_not_narrow_the_interval() -> None:
    df = _table(n=600)
    kwargs = {"y_true": "y", "y_score": "score", "threshold": 0.5, "metrics": ["auc", "sensitivity"]}
    single = F.subgroup_metrics(df, "sex", **kwargs)  # type: ignore[arg-type]
    naive = F.subgroup_metrics(_repeated(df, 4), "sex", **kwargs)  # type: ignore[arg-type]
    clustered = F.subgroup_metrics(_repeated(df, 4), "sex", unit="person", **kwargs)  # type: ignore[arg-type]
    assert (clustered["method"] == "cluster_bootstrap_percentile").all()
    assert clustered["value"].to_numpy() == pytest.approx(single["value"].to_numpy())
    assert (clustered["support"] == single["support"]).all() and (clustered["n"] == 4 * single["n"]).all()

    def width(t: pd.DataFrame) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return np.asarray(t["upper"] - t["lower"], dtype=np.float64)

    assert (width(naive) < 0.6 * width(single)).all()  # treating copies as new cases halves the width
    assert width(clustered) == pytest.approx(width(single), rel=0.15)


def test_a_person_column_without_repeats_uses_the_formulas() -> None:
    df = _table()
    with_unit = F.subgroup_gaps(df, "sex", y_true="y", y_score="score", threshold=0.5, unit="person")
    without = F.subgroup_gaps(df, "sex", y_true="y", y_score="score", threshold=0.5)
    pd.testing.assert_frame_equal(with_unit, without)
    assert set(without["method"]) == {"delong_unpaired", "newcombe_fisher"}


def test_clustered_gaps_have_intervals_but_no_p_values() -> None:
    df = _repeated(_table(n=500), 3)
    out = F.subgroup_gaps(df, "sex", y_true="y", y_score="score", threshold=0.5, unit="person",
                          metrics=["auc", "specificity"], margin=0.5)
    assert (out["status"] == "ok").all() and out["p_value"].isna().all() and out["p_holm"].isna().all()
    assert ((out["lower"] < out["difference"]) & (out["difference"] < out["upper"])).all()
    assert (out["verdict"] == "within_margin").all()


def test_small_clustered_groups_are_unstable_or_too_small_not_wrong() -> None:
    rng = np.random.default_rng(7)
    df = pd.DataFrame({"y": [True] * 3 + [False] * 9, "g": ["a"] * 12, "person": [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
    df["score"] = rng.normal(size=12)
    out = F.subgroup_metrics(df, "g", y_true="y", y_score="score", unit="person", min_count=2)
    assert out.loc[0, "status"] in {"ok", "unstable"} and out.loc[0, "support"] == 2
    assert F.subgroup_metrics(df, "g", y_true="y", y_score="score", unit="person", min_count=3).loc[0, "status"] == "too_small"


def test_person_level_intervals_cover_when_cases_are_correlated() -> None:
    """Three correlated cases per person and no true gap. The 95% interval for the gap
    should contain zero about 95% of the time with the person column, and clearly less
    often when the repeat cases are treated as independent."""

    def dataset(seed: int) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        persons, per = 300, 3
        y = np.repeat(rng.random(persons) < 0.5, per)
        score = np.repeat(rng.normal(0.0, 1.2, persons), per) + rng.normal(0.0, 0.6, persons * per) + 1.0 * y
        table: pd.DataFrame = pd.DataFrame({
            "y": y, "g": np.repeat(rng.choice(["a", "b"], persons), per),
            "person": np.repeat(np.arange(persons), per), "score": score,
        })
        return table

    reps = 120
    covered = {"person": 0, "naive": 0}
    for i in range(reps):
        df = dataset(2000 + i)
        for name, unit in (("person", "person"), ("naive", None)):
            out = F.subgroup_gaps(df, "g", y_true="y", y_score="score", threshold=0.5, unit=unit,
                                  metrics=["auc", "sensitivity"], adjust="none", n_resamples=400)
            covered[name] += int(((out["lower"] <= 0) & (out["upper"] >= 0)).sum())
    assert covered["person"] / (2 * reps) > 0.90
    assert covered["naive"] / (2 * reps) < 0.86


# --------------------------------------------------------------------------- subgroup_gaps


def test_gap_rows_match_the_underlying_tests() -> None:
    df = _table()
    out = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", threshold=0.5, reference="A",
                          metrics=["auc", "sensitivity"], adjust="none")
    assert out.attrs["family_size"] == 6 and (out["interval_level"] == 0.95).all() and out["p_holm"].isna().all()
    b, a = df[df.vendor == "B"], df[df.vendor == "A"]
    want = R.compare_unpaired(b.y.to_numpy(), b.score.to_numpy(), a.y.to_numpy(), a.score.to_numpy())
    row = _row(out, vendor="B", metric="auc")
    assert (row["difference"], row["lower"], row["upper"], row["p_value"]) == pytest.approx(
        (want.difference, want.lower, want.upper, want.p_value))
    assert (row["value"], row["reference_value"], row["reference"]) == (pytest.approx(want.auc_a), pytest.approx(want.auc_b), "A")
    k1, n1, k2, n2 = int((b.call & b.y).sum()), int(b.y.sum()), int((a.call & a.y).sum()), int(a.y.sum())
    row = _row(out, vendor="B", metric="sensitivity")
    assert row["difference"] == pytest.approx(k1 / n1 - k2 / n2)
    assert row["ratio"] == pytest.approx((k1 / n1) / (k2 / n2))
    assert (row["lower"], row["upper"]) == pytest.approx(F.newcombe_interval(k1, n1, k2, n2))
    assert row["p_value"] == pytest.approx(fisher_exact_p(k1, n1, k2, n2))
    assert (row["support"], row["reference_support"]) == (n1, n2)


def test_default_reference_is_the_largest_subgroup() -> None:
    df = _table()
    out = F.subgroup_gaps(df, "vendor", y_true="y", y_pred="call", metrics=["sensitivity"])
    assert set(out["reference"]) == {"A"} and "A" not in set(out["vendor"])


def test_bonferroni_level_and_holm() -> None:
    df = _table()
    out = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", threshold=0.5, metrics=["auc", "sensitivity"])
    plain = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", threshold=0.5, metrics=["auc", "sensitivity"], adjust="none")
    k = out.attrs["family_size"]
    assert k == int((out["status"] == "ok").sum()) == 6
    assert out["interval_level"].to_numpy() == pytest.approx(1 - 0.05 / k)
    assert ((out["upper"] - out["lower"]) > (plain["upper"] - plain["lower"])).all()
    assert out["p_value"].to_numpy() == pytest.approx(plain["p_value"].to_numpy())
    assert out["p_holm"].tolist() == pytest.approx(F.holm(out["p_value"].tolist()))
    assert (out["p_holm"] >= out["p_value"]).all()
    b, a = df[df.vendor == "B"], df[df.vendor == "A"]
    want = R.compare_unpaired(b.y.to_numpy(), b.score.to_numpy(), a.y.to_numpy(), a.score.to_numpy(), conf_level=1 - 0.05 / k)
    row = _row(out, vendor="B", metric="auc")
    assert (row["lower"], row["upper"]) == pytest.approx((want.lower, want.upper))


def test_verdicts() -> None:
    assert F._verdict(-0.02, 0.03, 0.05) == "within_margin"
    assert F._verdict(-0.20, -0.06, 0.05) == "exceeds_margin"
    assert F._verdict(0.06, 0.2, 0.05) == "exceeds_margin"
    assert F._verdict(-0.20, -0.04, 0.05) == "inconclusive"  # significant, but may be smaller than the margin
    assert F._verdict(-0.10, 0.10, 0.05) == "inconclusive"
    assert F._verdict(-0.02, 0.03, None) is None and F._verdict(None, None, 0.05) is None


def test_a_real_gap_is_found_and_no_gap_is_cleared_with_enough_data() -> None:
    df = _table(seed=3, n=6000, weak="B", drop=1.3)
    out = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", threshold=0.5, reference="A",
                          metrics=["auc", "sensitivity"], margin={"auc": 0.08, "sensitivity": 0.15})
    assert _row(out, vendor="B", metric="auc")["verdict"] == "exceeds_margin"
    assert _row(out, vendor="B", metric="sensitivity")["verdict"] == "exceeds_margin"
    assert _row(out, vendor="B", metric="auc")["difference"] < 0
    assert _row(out, vendor="C", metric="auc")["verdict"] == "within_margin"
    assert _row(out, vendor="C", metric="sensitivity")["verdict"] == "within_margin"
    assert _row(out, vendor=F.MISSING, metric="sensitivity")["verdict"] == "inconclusive"  # ~100 positives
    assert (out["margin"] == out["metric"].map({"auc": 0.08, "sensitivity": 0.15})).all()


def test_margin_mapping_may_leave_a_metric_out() -> None:
    out = F.subgroup_gaps(_table(), "sex", y_true="y", y_score="score", threshold=0.5,
                          metrics=["auc", "ppv"], margin={"auc": 0.1})
    assert _row(out, metric="auc")["verdict"] in {"within_margin", "inconclusive", "exceeds_margin"}
    assert pd.isna(_row(out, metric="ppv")["verdict"]) and pd.isna(_row(out, metric="ppv")["margin"])
    assert not pd.isna(_row(out, metric="ppv")["lower"])


def test_versus_rest() -> None:
    df = _table()
    out = F.subgroup_gaps(df, "vendor", y_true="y", y_pred="call", metrics=["sensitivity"], versus="rest", adjust="none")
    assert set(out["vendor"]) == {"A", "B", "C", F.MISSING} and set(out["reference"]) == {"(rest)"}
    c, rest = df[df.vendor == "C"], df[df.vendor != "C"]
    row = _row(out, vendor="C")
    assert row["reference_value"] == pytest.approx((rest.call & rest.y).sum() / rest.y.sum())
    assert row["difference"] == pytest.approx((c.call & c.y).sum() / c.y.sum() - row["reference_value"])


def test_too_small_subgroup_and_reference() -> None:
    df = _table(n=150)
    out = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", metrics=["auc"], min_count=10, reference="A")
    small = out[out.support < 10]
    assert len(small) > 0 and (small["status"] == "too_small").all() and small["difference"].isna().all()
    assert out.attrs["family_size"] == int((out["status"] == "ok").sum())
    tiny_ref = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", metrics=["auc"], min_count=10, reference=F.MISSING)
    assert _row(tiny_ref, vendor="A")["status"] == "reference_too_small"
    assert set(tiny_ref["status"]) <= {"reference_too_small", "too_small"} and tiny_ref.attrs["family_size"] == 0


def test_joint_reference_is_a_tuple() -> None:
    df = _table()
    out = F.subgroup_gaps(df, ["sex", "vendor"], y_true="y", y_pred="call", metrics=["specificity"], reference=("F", "A"))
    assert len(out) == 7 and all(r == ("F", "A") for r in out["reference"])
    with pytest.raises(ValueError, match="one level for each"):
        F.subgroup_gaps(df, ["sex", "vendor"], y_true="y", y_pred="call", reference="F")


def test_no_false_alarms_when_there_is_no_gap() -> None:
    """Across repeated datasets with no true gap, 'exceeds_margin' should essentially never
    appear and unadjusted p-values below 0.05 should appear about 5% of the time."""
    exceeds = significant = total = 0
    for seed in range(60):
        out = F.subgroup_gaps(_table(seed=100 + seed, n=700), "vendor", y_true="y", y_score="score", threshold=0.5,
                              reference="A", metrics=["auc", "sensitivity"], margin=0.05, min_count=5)
        ok = out[out.status == "ok"]
        exceeds += int((ok["verdict"] == "exceeds_margin").sum())
        significant += int((ok["p_value"] < 0.05).sum())
        total += len(ok)
    assert total == 360 and exceeds == 0
    assert significant / total < 0.10


# --------------------------------------------------------------------------- scan


def test_scan_order_one_stacks_the_factors_as_one_family() -> None:
    df = _table()
    out = F.scan(df, ["sex", "vendor"], y_true="y", y_score="score", metrics=["auc"], reference={"vendor": "A"})
    assert list(out["factor"].unique()) == ["sex", "vendor"] and len(out) == 4
    assert out.attrs["family_size"] == 4 and out["interval_level"].to_numpy() == pytest.approx(1 - 0.05 / 4)
    alone = F.subgroup_gaps(df, "vendor", y_true="y", y_score="score", metrics=["auc"], reference="A", conf_level=1 - 0.05 / 4, adjust="none")
    got = out[out.factor == "vendor"].reset_index(drop=True)
    assert got["level"].tolist() == alone["vendor"].tolist()
    assert got[["difference", "lower", "upper", "p_value"]].to_numpy() == pytest.approx(
        alone[["difference", "lower", "upper", "p_value"]].to_numpy())
    assert out["p_holm"].tolist() == pytest.approx(F.holm(out["p_value"].tolist()))


def test_scan_order_two_adds_intersections() -> None:
    df = _table(n=1500)
    out = F.scan(df, ["sex", "vendor"], order=2, y_true="y", y_pred="call", metrics=["sensitivity"],
                 reference={"sex": "F", "vendor": "A"}, min_count=5)
    assert list(out["factor"].unique()) == ["sex", "vendor", "sex x vendor"]
    joint = out[out.factor == "sex x vendor"]
    assert len(joint) == 7 and set(joint["reference"]) == {"F / A"} and "M / C" in set(joint["level"])
    cell = df[(df.sex == "M") & (df.vendor == "C")]
    assert _row(out, level="M / C")["value"] == pytest.approx((cell.call & cell.y).sum() / cell.y.sum())
    assert out.attrs["family_size"] == int((out["status"] == "ok").sum())


def test_scan_partial_reference_falls_back_to_largest_joint_group() -> None:
    df = _table(n=1500)
    out = F.scan(df, ["sex", "vendor"], order=2, y_true="y", y_pred="call", metrics=["positive_rate"], reference={"sex": "F"})
    sizes = df.assign(vendor=df.vendor.fillna(F.MISSING)).groupby(["sex", "vendor"]).size()
    sex, vendor = sizes.idxmax()
    assert set(out[out.factor == "sex x vendor"]["reference"]) == {f"{sex} / {vendor}"}
    assert set(out[out.factor == "sex"]["reference"]) == {"F"}


# --------------------------------------------------------------------------- input checks


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"y_true": "y"}, "give y_score"),
        ({"y_true": "y", "threshold": 0.5}, "threshold needs y_score"),
        ({"y_true": "y", "y_pred": "call", "y_score": "score", "threshold": 0.5}, "not both"),
        ({"y_true": "y", "y_pred": "call", "metrics": ["auc"]}, "needs y_score"),
        ({"y_true": "y", "y_score": "score", "metrics": ["ppv"]}, "needs y_pred"),
        ({"y_true": "y", "y_score": "score", "metrics": ["accuracy"]}, "unknown metric"),
        ({"y_true": "y", "y_score": "score", "metrics": []}, "empty"),
        ({"y_true": "nope", "y_score": "score"}, "not in the table"),
        ({"y_true": "y", "y_score": "score", "unit": "nope"}, "not in the table"),
        ({"y_true": "score", "y_score": "score"}, "0 and 1"),
        ({"y_true": "y", "y_pred": "score"}, "y_pred must be boolean"),
        ({"y_true": "y", "y_score": "score", "conf_level": 1.0}, "conf_level"),
        ({"y_true": "y", "y_score": "score", "min_count": 0}, "min_count"),
        ({"y_true": "y", "y_score": "score", "n_resamples": 10}, "n_resamples"),
    ],
)
def test_bad_inputs(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        F.subgroup_metrics(_table(n=100), "sex", **kwargs)  # type: ignore[arg-type]


def test_bad_grouping_and_options() -> None:
    df = _table(n=100)
    with pytest.raises(ValueError, match="no column"):
        F.subgroup_metrics(df, "nope", y_true="y", y_score="score")
    with pytest.raises(ValueError, match="at least one"):
        F.subgroup_metrics(df, [], y_true="y", y_score="score")
    with pytest.raises(ValueError, match="empty"):
        F.subgroup_metrics(df.iloc[:0], "sex", y_true="y", y_score="score")
    with pytest.raises(ValueError, match="not a level"):
        F.subgroup_gaps(df, "sex", y_true="y", y_score="score", reference="X")
    with pytest.raises(ValueError, match="versus"):
        F.subgroup_gaps(df, "sex", y_true="y", y_score="score", versus="all")
    with pytest.raises(ValueError, match="adjust"):
        F.subgroup_gaps(df, "sex", y_true="y", y_score="score", adjust="fdr")
    with pytest.raises(ValueError, match="margin must be positive"):
        F.subgroup_gaps(df, "sex", y_true="y", y_score="score", margin=0.0)
    with pytest.raises(ValueError, match="order"):
        F.scan(df, ["sex"], order=2, y_true="y", y_score="score")
    with pytest.raises(ValueError, match="not being scanned"):
        F.scan(df, ["sex"], y_true="y", y_score="score", reference={"vendor": "A"})
    missing_person = df.assign(person=df.person.astype(object).where(df.person > 3, None))
    with pytest.raises(ValueError, match="person id"):
        F.subgroup_metrics(missing_person, "sex", y_true="y", y_score="score", unit="person")
    missing_score = df.assign(score=df.score.where(df.person > 3))
    with pytest.raises(ValueError, match="missing or non-finite"):
        F.subgroup_metrics(missing_score, "sex", y_true="y", y_score="score")


def test_results_are_reproducible() -> None:
    df = _repeated(_table(n=300), 2)
    a = F.subgroup_gaps(df, "sex", y_true="y", y_score="score", threshold=0.5, unit="person", seed=5)
    b = F.subgroup_gaps(df, "sex", y_true="y", y_score="score", threshold=0.5, unit="person", seed=5)
    pd.testing.assert_frame_equal(a, b)
