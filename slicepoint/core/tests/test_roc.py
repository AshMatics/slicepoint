"""ROC analysis: checked against scikit-learn, scipy, a direct evaluation of the DeLong
definition, hand values and simulation.

Not yet checked against R ``pROC`` (``ci.auc``, ``roc.test``): no R was available when
this was written. ``reference_cases`` builds the fixed datasets those fixtures should use.
"""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
import pytest
from scipy import stats as st
from sklearn.metrics import roc_auc_score

from slicepoint_core import planning as P
from slicepoint_core import roc as R

F = npt.NDArray[np.float64]
B = npt.NDArray[np.bool_]


def _simulate(seed: int, n: int, prevalence: float, shift: float, *, round_to: int | None = None) -> tuple[B, F]:
    """Binormal scores: negatives N(0,1), positives N(shift,1). ``round_to`` creates ties."""
    rng = np.random.default_rng(seed)
    y: B = rng.random(n) < prevalence
    s: F = rng.normal(0.0, 1.0, n) + shift * y
    if round_to is not None:
        s = np.round(s, round_to)
    return y, s


def reference_cases() -> dict[str, tuple[B, F]]:
    """Fixed datasets for cross-checks against other software."""
    return {
        "continuous": _simulate(1, 200, 0.3, 1.2),
        "ties": _simulate(2, 150, 0.4, 0.8, round_to=0),
        "small": _simulate(3, 24, 0.5, 1.5),
    }


# --------------------------------------------------------------------------- direct definitions


def _psi(x: F, y: F) -> F:
    """DeLong kernel: 1 if x > y, 1/2 if equal, 0 otherwise, for every positive-negative pair."""
    d = x[:, None] - y[None, :]
    out: F = (d > 0).astype(float) + 0.5 * (d == 0)
    return out


def _direct(y: B, s: F) -> tuple[float, F, F]:
    k = _psi(s[y], s[~y])
    return float(k.mean()), k.mean(axis=1), k.mean(axis=0)


def _direct_var(y: B, s: F) -> float:
    _, v10, v01 = _direct(y, s)
    return float(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))


# --------------------------------------------------------------------------- AUC


@pytest.mark.parametrize("name", ["continuous", "ties", "small"])
def test_auc_matches_sklearn_scipy_and_the_definition(name: str) -> None:
    y, s = reference_cases()[name]
    a = R.auc(y, s)
    assert a == pytest.approx(roc_auc_score(y, s), abs=1e-12)
    u = st.mannwhitneyu(s[y], s[~y]).statistic
    assert a == pytest.approx(u / (y.sum() * (~y).sum()), abs=1e-12)
    assert a == pytest.approx(_direct(y, s)[0], abs=1e-12)


def test_auc_hand_values() -> None:
    assert R.auc([0, 0, 1, 1], [0.1, 0.2, 0.3, 0.4]) == 1.0
    assert R.auc([1, 1, 0, 0], [0.1, 0.2, 0.3, 0.4]) == 0.0
    assert R.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == 0.5  # all tied
    # positives 3, 1; negatives 2, 0 -> pairs won: (3>2),(3>0),(1>0) = 3 of 4
    assert R.auc([True, True, False, False], [3, 1, 2, 0]) == 0.75
    # one tie among four pairs: (2>1)=1, (2=2)=.5, (1=1)=.5, (1<2)=0 -> 2/4
    assert R.auc([1, 1, 0, 0], [2, 1, 1, 2]) == 0.5


def test_auc_is_invariant_to_monotone_transforms() -> None:
    y, s = reference_cases()["continuous"]
    assert R.auc(y, np.exp(s)) == pytest.approx(R.auc(y, s), abs=1e-12)
    assert R.auc(y, -s) == pytest.approx(1.0 - R.auc(y, s), abs=1e-12)


# --------------------------------------------------------------------------- DeLong


@pytest.mark.parametrize("name", ["continuous", "ties", "small"])
def test_delong_variance_matches_the_definition(name: str) -> None:
    y, s = reference_cases()[name]
    r = R.auc_delong(y, s)
    assert r.se is not None and r.se**2 == pytest.approx(_direct_var(y, s), rel=1e-10)
    assert r.auc == pytest.approx(_direct(y, s)[0], abs=1e-12)
    assert (r.n_pos, r.n_neg, r.n_units) == (int(y.sum()), int((~y).sum()), len(y))
    z = 1.959963984540054
    assert (r.lower, r.upper) == pytest.approx((r.auc - z * r.se, min(1.0, r.auc + z * r.se)), abs=1e-12)


def test_delong_hand_value() -> None:
    # positives 3, 1; negatives 2, 0. V10 = [1, .5], V01 = [.5, 1]; each has sample variance 1/8.
    r = R.auc_delong([1, 1, 0, 0], [3, 1, 2, 0])
    assert r.auc == 0.75 and r.se == pytest.approx(math.sqrt(1 / 8 / 2 + 1 / 8 / 2), abs=1e-15)


def test_delong_interval_is_clipped_and_degenerate_when_separated() -> None:
    r = R.auc_delong([0, 0, 0, 1, 1, 1], [1, 2, 3, 4, 5, 6])
    assert (r.auc, r.se, r.lower, r.upper) == (1.0, 0.0, 1.0, 1.0)
    y, s = _simulate(5, 30, 0.5, 2.5)
    assert R.auc_delong(y, s).upper <= 1.0


def test_delong_se_is_close_to_the_planning_formula() -> None:
    """Hanley-McNeil (planning) and DeLong (analysis) should agree roughly on well-behaved data."""
    y, s = _simulate(11, 4000, 0.3, 1.0)
    r = R.auc_delong(y, s)
    assert r.se == pytest.approx(P.auc_standard_error(r.auc, r.n_pos, r.n_neg), rel=0.10)


def test_delong_interval_covers_the_true_auc() -> None:
    true_auc = float(st.norm.cdf(1.0 / math.sqrt(2.0)))  # binormal, shift 1, equal variances
    hits = 0
    for seed in range(400):
        y, s = _simulate(1000 + seed, 150, 0.4, 1.0)
        r = R.auc_delong(y, s)
        hits += r.lower <= true_auc <= r.upper
    assert 0.91 <= hits / 400 <= 0.98


# --------------------------------------------------------------------------- paired


def _two_models(seed: int, n: int, shift_b: float) -> tuple[B, F, F]:
    rng = np.random.default_rng(seed)
    y: B = rng.random(n) < 0.4
    shared = rng.normal(0, 1, n)
    a: F = shared + rng.normal(0, 0.6, n) + 1.0 * y
    b: F = shared + rng.normal(0, 0.6, n) + shift_b * y
    return y, a, b


def test_paired_variance_matches_the_definition() -> None:
    y, a, b = _two_models(7, 180, 0.6)
    r = R.compare_paired(y, a, b)
    _, a10, a01 = _direct(y, a)
    _, b10, b01 = _direct(y, b)
    m, n = len(a10), len(a01)
    var = ((a10 - b10).var(ddof=1) / m) + ((a01 - b01).var(ddof=1) / n)  # variance of the difference, directly
    assert r.se is not None and r.se**2 == pytest.approx(var, rel=1e-10)
    assert r.difference == pytest.approx(R.auc(y, a) - R.auc(y, b), abs=1e-12)
    assert r.z is not None and r.z == pytest.approx(r.difference / r.se)
    assert r.p_value == pytest.approx(2 * st.norm.sf(abs(r.z)))
    assert r.paired and r.method == "delong"


def test_paired_is_antisymmetric_and_null_for_identical_scores() -> None:
    y, a, b = _two_models(8, 120, 0.5)
    ab, ba = R.compare_paired(y, a, b), R.compare_paired(y, b, a)
    assert ab.difference == pytest.approx(-ba.difference) and ab.p_value == pytest.approx(ba.p_value)
    assert (ab.lower, ab.upper) == pytest.approx((-ba.upper, -ba.lower))
    same = R.compare_paired(y, a, a)
    assert (same.difference, same.se, same.z, same.p_value) == (0.0, 0.0, None, None)


def test_paired_uses_the_correlation() -> None:
    """Correlated models: the paired test is far more precise than treating them as separate sets."""
    y, a, b = _two_models(9, 300, 0.7)
    paired = R.compare_paired(y, a, b)
    unpaired = R.compare_unpaired(y, a, y, b)
    assert paired.se is not None and unpaired.se is not None and paired.se < 0.8 * unpaired.se


def test_paired_test_holds_its_size_under_the_null() -> None:
    rejections = 0
    for seed in range(400):
        y, a, b = _two_models(2000 + seed, 150, 1.0)  # same true AUC
        p = R.compare_paired(y, a, b).p_value
        rejections += p is not None and p < 0.05
    assert 0.02 <= rejections / 400 <= 0.09


# --------------------------------------------------------------------------- unpaired


def test_unpaired_adds_the_two_variances() -> None:
    ya, sa = _simulate(21, 160, 0.35, 1.2)
    yb, sb = _simulate(22, 90, 0.35, 0.6)
    r = R.compare_unpaired(ya, sa, yb, sb)
    ea, eb = R.auc_delong(ya, sa), R.auc_delong(yb, sb)
    assert ea.se is not None and eb.se is not None
    assert r.se == pytest.approx(math.hypot(ea.se, eb.se), rel=1e-12)
    assert (r.auc_a, r.auc_b, r.difference) == pytest.approx((ea.auc, eb.auc, ea.auc - eb.auc))
    assert not r.paired and r.lower < r.difference < r.upper


def test_unpaired_test_holds_its_size_under_the_null() -> None:
    rejections = 0
    for seed in range(400):
        ya, sa = _simulate(3000 + seed, 140, 0.4, 1.0)
        yb, sb = _simulate(7000 + seed, 90, 0.3, 1.0)
        p = R.compare_unpaired(ya, sa, yb, sb).p_value
        rejections += p is not None and p < 0.05
    assert 0.02 <= rejections / 400 <= 0.09


def test_a_small_subgroup_cannot_show_equality() -> None:
    """The reason a p-value is not a fairness verdict: a real gap in a small group is 'not significant'."""
    ya, sa = _simulate(31, 600, 0.4, 1.4)
    yb, sb = _simulate(32, 24, 0.4, 0.7)
    r = R.compare_unpaired(ya, sa, yb, sb)
    assert r.p_value is not None and r.p_value > 0.05 and r.upper - r.lower > 0.3


# --------------------------------------------------------------------------- bootstrap


def test_bootstrap_agrees_with_delong_on_independent_cases() -> None:
    y, s = _simulate(41, 400, 0.35, 1.0)
    d, b = R.auc_delong(y, s), R.bootstrap_auc(y, s, n_resamples=1500, seed=1)
    assert b.auc == pytest.approx(d.auc, abs=1e-12) and b.method == "bootstrap_percentile"
    assert (b.lower, b.upper) == pytest.approx((d.lower, d.upper), abs=0.012)
    assert b.n_units == 400 and b.n_resamples_used == 1500 and b.se is None


def test_bootstrap_is_deterministic_per_seed() -> None:
    y, s = _simulate(42, 120, 0.4, 1.0)
    a, b, c = (R.bootstrap_auc(y, s, n_resamples=300, seed=k) for k in (5, 5, 6))
    assert (a.lower, a.upper) == (b.lower, b.upper) and (a.lower, a.upper) != (c.lower, c.upper)


def test_clustering_widens_the_interval_to_the_honest_one() -> None:
    """Four identical images per person carry one person's information, not four."""
    y, s = _simulate(43, 120, 0.4, 1.0)
    people = np.arange(len(y))
    y4, s4, g4 = np.repeat(y, 4), np.repeat(s, 4), np.repeat(people, 4)

    honest = R.bootstrap_auc(y, s, n_resamples=1500, seed=3)
    clustered = R.bootstrap_auc(y4, s4, groups=g4, n_resamples=1500, seed=3)
    naive = R.auc_delong(y4, s4)

    assert clustered.method == "cluster_bootstrap_percentile" and clustered.n_units == 120
    # same seed, same clusters drawn -> the clustered interval equals the one-image-per-person interval
    assert (clustered.lower, clustered.upper) == pytest.approx((honest.lower, honest.upper), abs=1e-12)
    # treating 480 images as independent halves the width
    assert (naive.upper - naive.lower) == pytest.approx(0.5 * (clustered.upper - clustered.lower), rel=0.15)


def test_bootstrap_accepts_any_hashable_group_ids() -> None:
    y, s = _simulate(44, 60, 0.5, 1.0)
    ids = [f"P{i // 2:03d}" for i in range(60)]
    r = R.bootstrap_auc(y, s, groups=ids, n_resamples=200, seed=0)
    assert r.n_units == 30
    with pytest.raises(ValueError):
        R.bootstrap_auc(y, s, groups=[None] * 60, n_resamples=200)
    with pytest.raises(ValueError):
        R.bootstrap_auc(y, s, groups=ids[:-1], n_resamples=200)


def test_bootstrap_refuses_data_too_small_for_an_interval() -> None:
    y = np.array([True] + [False] * 30)
    s = np.arange(31, dtype=float)
    with pytest.raises(ValueError, match="held both classes"):
        R.bootstrap_auc(y, s, n_resamples=200)  # ~36% of replicates miss the single positive
    y[:4] = True  # four positives: under 2% of replicates miss them all
    used = R.bootstrap_auc(y, s, n_resamples=200).n_resamples_used
    assert used is not None and used >= 190


# --------------------------------------------------------------------------- gap


def _two_groups(seed: int, n_a: int, n_b: int, shift_a: float, shift_b: float) -> tuple[B, F, B, B]:
    ya, sa = _simulate(seed, n_a, 0.4, shift_a)
    yb, sb = _simulate(seed + 1, n_b, 0.4, shift_b)
    y, s = np.concatenate([ya, yb]), np.concatenate([sa, sb])
    in_a = np.arange(n_a + n_b) < n_a
    return y, s, in_a, ~in_a


def test_gap_agrees_with_unpaired_delong_on_independent_cases() -> None:
    y, s, a, b = _two_groups(51, 300, 200, 1.3, 0.7)
    boot = R.bootstrap_gap(y, s, a, b, n_resamples=1500, seed=2)
    delong = R.compare_unpaired(y[a], s[a], y[b], s[b])
    assert boot.difference == pytest.approx(delong.difference, abs=1e-12)
    assert (boot.lower, boot.upper) == pytest.approx((delong.lower, delong.upper), abs=0.02)
    assert boot.p_value is None and boot.se is None and not boot.paired
    assert boot.lower > 0  # a real gap of this size is detected at these sizes


def test_gap_is_antisymmetric() -> None:
    y, s, a, b = _two_groups(52, 120, 100, 1.0, 0.8)
    ab = R.bootstrap_gap(y, s, a, b, n_resamples=400, seed=4)
    ba = R.bootstrap_gap(y, s, b, a, n_resamples=400, seed=4)
    assert (ab.difference, ab.lower, ab.upper) == pytest.approx((-ba.difference, -ba.upper, -ba.lower))


def test_gap_handles_people_with_cases_in_both_subsets() -> None:
    """Two images per person, one on each scanner: resampling people keeps the pair together."""
    rng = np.random.default_rng(53)
    n = 150
    y1: B = rng.random(n) < 0.4
    person = rng.normal(0, 1, n)
    s_x: F = person + rng.normal(0, 0.3, n) + 1.0 * y1
    s_y: F = person + rng.normal(0, 0.3, n) + 1.0 * y1
    y, s = np.concatenate([y1, y1]), np.concatenate([s_x, s_y])
    in_x = np.arange(2 * n) < n
    ids = np.concatenate([np.arange(n), np.arange(n)])
    clustered = R.bootstrap_gap(y, s, in_x, ~in_x, groups=ids, n_resamples=1000, seed=6)
    ignoring = R.bootstrap_gap(y, s, in_x, ~in_x, n_resamples=1000, seed=6)
    # the shared person effect cancels in the gap, so the correct interval is much narrower
    assert (clustered.upper - clustered.lower) < 0.6 * (ignoring.upper - ignoring.lower)
    assert clustered.lower < 0 < clustered.upper


def test_gap_rejects_overlap_and_one_class_subsets() -> None:
    y, s, a, b = _two_groups(54, 40, 40, 1.0, 1.0)
    with pytest.raises(ValueError, match="overlap"):
        R.bootstrap_gap(y, s, a, a, n_resamples=200)
    with pytest.raises(ValueError):
        R.bootstrap_gap(y, s, a, b & ~y, n_resamples=200)  # subset b has no positives
    with pytest.raises(ValueError):
        R.bootstrap_gap(y, s, a[:-1], b[:-1], n_resamples=200)


# --------------------------------------------------------------------------- input checks


@pytest.mark.parametrize("y,s", [
    ([1, 1, 1], [0.1, 0.2, 0.3]),            # one class
    ([0, 1, 2], [0.1, 0.2, 0.3]),            # not binary
    ([0, 1], [0.1, 0.2, 0.3]),               # length mismatch
    ([0, 1, 1], [0.1, float("nan"), 0.3]),   # missing score
    (["a", "b"], [0.1, 0.2]),                # labels of the wrong type
    ([[0, 1]], [[0.1, 0.2]]),                # not one-dimensional
])
def test_auc_rejects_bad_input(y: list[object], s: list[object]) -> None:
    with pytest.raises(ValueError):
        R.auc(y, s)


def test_delong_needs_two_cases_per_class_and_valid_levels() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        R.auc_delong([1, 0, 0, 0], [0.9, 0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        R.auc_delong([1, 1, 0, 0], [3, 1, 2, 0], conf_level=1.5)
    with pytest.raises(ValueError):
        R.bootstrap_auc([1, 1, 0, 0], [3, 1, 2, 0], n_resamples=10)
