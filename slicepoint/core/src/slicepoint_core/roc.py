"""ROC analysis: the area under the curve, its uncertainty, and comparisons of two AUCs.

* :func:`auc` is the empirical AUC (the Mann-Whitney statistic; a tie counts one half).
* :func:`auc_delong` adds the DeLong, DeLong and Clarke-Pearson (1988) standard error and
  a normal-approximation interval.
* :func:`compare_paired` tests two models scored on the *same* cases (correlated AUCs).
* :func:`compare_unpaired` tests one model on two *separate* groups of cases, which is
  the subgroup-gap question.
* :func:`bootstrap_auc` and :func:`bootstrap_gap` give percentile intervals that respect
  clustering: when one person contributes several images, people are resampled, not
  images.

Every function is pure and deterministic (the bootstraps take a ``seed``) and does no IO.
A higher score must mean "more likely positive". Missing scores are an error: decide
whether a case with no answer is dropped or counted as wrong before calling.

Which interval to use
---------------------
The DeLong variance treats cases as independent. Repeat images of one person are not, and
DeLong intervals on such data are too narrow. Pass ``groups`` to the bootstrap functions
in that case. With one case per person the two agree closely.

DeLong variances use the midrank algorithm of Sun and Xu (2014), which is exact and
``O(n log n)``; the tests compare it with a direct evaluation of the definition.
"""

from __future__ import annotations

import math
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as st

__all__ = [
    "AucComparison",
    "AucEstimate",
    "auc",
    "auc_delong",
    "bootstrap_auc",
    "bootstrap_gap",
    "compare_paired",
    "compare_unpaired",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]


@dataclass(frozen=True)
class AucEstimate:
    """An AUC with its interval.

    ``se`` is ``None`` for a bootstrap estimate (the interval is a percentile interval).
    ``n_units`` is the number of independent units the interval rests on: cases for
    DeLong, people (clusters) for a clustered bootstrap. ``n_resamples_used`` counts
    bootstrap replicates that contained both classes.
    """

    auc: float
    n_pos: int
    n_neg: int
    lower: float
    upper: float
    conf_level: float
    method: str
    se: float | None = None
    n_units: int | None = None
    n_resamples_used: int | None = None


@dataclass(frozen=True)
class AucComparison:
    """Two AUCs and their difference ``auc_a - auc_b``.

    ``p_value`` is two-sided for "the two AUCs are equal"; it is ``None`` for bootstrap
    comparisons and when the estimated variance is zero. A p-value above 0.05 does not
    show that two groups perform the same: read the interval against the smallest
    difference that would matter.
    """

    auc_a: float
    auc_b: float
    difference: float
    lower: float
    upper: float
    conf_level: float
    method: str
    paired: bool
    se: float | None = None
    z: float | None = None
    p_value: float | None = None
    n_resamples_used: int | None = None


# --------------------------------------------------------------------------- inputs


def _z(conf_level: float) -> float:
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    return float(st.norm.ppf(0.5 + conf_level / 2.0))


def _labels(y_true: npt.ArrayLike) -> BoolArray:
    y = np.asarray(y_true)
    if y.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if y.dtype == np.bool_:
        return y
    if y.dtype.kind in "iuf":
        yf = y.astype(float)
        if bool(np.isnan(yf).any()) or not bool(np.isin(yf, (0.0, 1.0)).all()):
            raise ValueError("y_true must be boolean or contain only 0 and 1 (no missing values)")
        out: BoolArray = yf == 1.0
        return out
    raise ValueError(f"y_true must be boolean or 0/1, got dtype {y.dtype}")


def _scores(y: BoolArray, y_score: npt.ArrayLike, name: str = "y_score") -> FloatArray:
    s = np.asarray(y_score, dtype=np.float64)
    if s.shape != y.shape:
        raise ValueError(f"{name} has shape {s.shape}, y_true has {y.shape}")
    if not bool(np.isfinite(s).all()):
        raise ValueError(f"{name} has missing or non-finite values; drop those cases or score them explicitly")
    return s


def _need_both(y: BoolArray, minimum: int = 1) -> tuple[int, int]:
    m, n = int(y.sum()), int((~y).sum())
    if m < minimum or n < minimum:
        raise ValueError(f"need at least {minimum} positive and {minimum} negative case(s), got {m} and {n}")
    return m, n


# --------------------------------------------------------------------------- AUC and DeLong


def _auc_of(pos: FloatArray, neg: FloatArray) -> float:
    m, n = len(pos), len(neg)
    ranks = st.rankdata(np.concatenate([pos, neg]))
    return float((ranks[:m].sum() - m * (m + 1) / 2.0) / (m * n))


def auc(y_true: npt.ArrayLike, y_score: npt.ArrayLike) -> float:
    """Empirical AUC: the chance that a positive case scores above a negative one.

    Ties count one half. Equal to ``sklearn.metrics.roc_auc_score`` and to the
    Mann-Whitney U statistic divided by ``n_pos * n_neg``.
    """
    y = _labels(y_true)
    s = _scores(y, y_score)
    _need_both(y)
    return _auc_of(s[y], s[~y])


def _components(pos: FloatArray, neg: FloatArray) -> tuple[float, FloatArray, FloatArray]:
    """AUC and DeLong's structural components ``V10`` (per positive) and ``V01`` (per negative)."""
    m, n = len(pos), len(neg)
    tz = st.rankdata(np.concatenate([pos, neg]))
    tx, ty = st.rankdata(pos), st.rankdata(neg)
    v10: FloatArray = (tz[:m] - tx) / n
    v01: FloatArray = 1.0 - (tz[m:] - ty) / m
    return float(v10.mean()), v10, v01


def _delong_var(v10: FloatArray, v01: FloatArray) -> float:
    return float(v10.var(ddof=1) / len(v10) + v01.var(ddof=1) / len(v01))


def auc_delong(y_true: npt.ArrayLike, y_score: npt.ArrayLike, *, conf_level: float = 0.95) -> AucEstimate:
    """AUC with the DeLong standard error and a normal-approximation interval.

    The interval is clipped to [0, 1]. Needs at least two cases in each class. Assumes
    independent cases; see the module docstring for clustered data.
    """
    y = _labels(y_true)
    s = _scores(y, y_score)
    m, n = _need_both(y, 2)
    a, v10, v01 = _components(s[y], s[~y])
    se = math.sqrt(_delong_var(v10, v01))
    half = _z(conf_level) * se
    return AucEstimate(auc=a, n_pos=m, n_neg=n, lower=max(0.0, a - half), upper=min(1.0, a + half),
                       conf_level=conf_level, method="delong", se=se, n_units=m + n)


def _comparison(a: float, b: float, var: float, conf_level: float, paired: bool) -> AucComparison:
    d = a - b
    se = math.sqrt(max(var, 0.0))
    half = _z(conf_level) * se
    z = p = None
    if se > 0:
        z = d / se
        p = float(2.0 * st.norm.sf(abs(z)))
    return AucComparison(auc_a=a, auc_b=b, difference=d, lower=d - half, upper=d + half, conf_level=conf_level,
                         method="delong", paired=paired, se=se, z=z, p_value=p)


def compare_paired(
    y_true: npt.ArrayLike,
    score_a: npt.ArrayLike,
    score_b: npt.ArrayLike,
    *,
    conf_level: float = 0.95,
) -> AucComparison:
    """DeLong test of two models (or two versions, or two readers) scored on the same cases.

    The two AUCs are correlated because they share cases; the test uses the covariance of
    their structural components. Assumes independent cases.
    """
    y = _labels(y_true)
    sa, sb = _scores(y, score_a, "score_a"), _scores(y, score_b, "score_b")
    m, n = _need_both(y, 2)
    a, a10, a01 = _components(sa[y], sa[~y])
    b, b10, b01 = _components(sb[y], sb[~y])
    s10 = np.cov(np.vstack([a10, b10]), ddof=1)
    s01 = np.cov(np.vstack([a01, b01]), ddof=1)
    cov = s10 / m + s01 / n
    var = float(cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1])
    return _comparison(a, b, var, conf_level, paired=True)


def compare_unpaired(
    y_true_a: npt.ArrayLike,
    y_score_a: npt.ArrayLike,
    y_true_b: npt.ArrayLike,
    y_score_b: npt.ArrayLike,
    *,
    conf_level: float = 0.95,
) -> AucComparison:
    """DeLong test of the AUCs of two separate sets of cases (for example two subgroups).

    The sets share no cases, so the variance of the difference is the sum of the two
    DeLong variances. Assumes independent cases, within and between the sets: a person
    with images in both sets breaks it (use :func:`bootstrap_gap`).
    """
    ya, yb = _labels(y_true_a), _labels(y_true_b)
    sa, sb = _scores(ya, y_score_a, "y_score_a"), _scores(yb, y_score_b, "y_score_b")
    _need_both(ya, 2)
    _need_both(yb, 2)
    a, a10, a01 = _components(sa[ya], sa[~ya])
    b, b10, b01 = _components(sb[yb], sb[~yb])
    return _comparison(a, b, _delong_var(a10, a01) + _delong_var(b10, b01), conf_level, paired=False)


# --------------------------------------------------------------------------- bootstrap

_MIN_USABLE_SHARE = 0.9
"""A bootstrap is refused when fewer than this share of replicates held both classes: an
interval from the replicates that happened to be usable would be too narrow."""


def _clusters(groups: Sequence[Hashable] | npt.NDArray[np.generic] | None, n: int) -> list[IntArray]:
    """Case indices of each cluster (one cluster per case when ``groups`` is None)."""
    if groups is None:
        return [np.array([i], dtype=np.intp) for i in range(n)]
    g = np.asarray(groups, dtype=object)
    if g.shape != (n,):
        raise ValueError(f"groups has shape {g.shape}, expected ({n},)")
    codes = np.asarray(pd.factorize(g)[0])  # clusters numbered in order of first appearance; missing -> -1
    if bool((codes < 0).any()):
        raise ValueError("groups has missing values; give every case a person or cluster id")
    order = np.argsort(codes, kind="stable")
    bounds = np.flatnonzero(np.diff(codes[order])) + 1
    return [part.astype(np.intp) for part in np.split(order, bounds)]


def _percentile(values: list[float], conf_level: float) -> tuple[float, float]:
    alpha = 1.0 - conf_level
    lo, hi = np.quantile(np.asarray(values, dtype=float), [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _check_boot(n_resamples: int, conf_level: float) -> None:
    _z(conf_level)
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be at least 100, got {n_resamples}")


def _mixed(y: BoolArray) -> bool:
    return len(y) > 0 and bool(y.any()) and not bool(y.all())


def bootstrap_auc(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    *,
    groups: Sequence[Hashable] | npt.NDArray[np.generic] | None = None,
    n_resamples: int = 2000,
    conf_level: float = 0.95,
    seed: int = 0,
) -> AucEstimate:
    """AUC with a percentile bootstrap interval, resampling people when ``groups`` is given.

    ``groups`` holds a person (or other cluster) id per case. Whole clusters are drawn
    with replacement, so several images of one person stay together. Without ``groups``
    each case is its own cluster. A replicate without both classes is discarded and
    counted out of ``n_resamples_used``; the call fails if more than one in ten is discarded,
    which means one class is too rare for an interval (roughly, fewer than three cases).
    """
    _check_boot(n_resamples, conf_level)
    y = _labels(y_true)
    s = _scores(y, y_score)
    m, n = _need_both(y)
    members = _clusters(groups, len(y))
    rng = np.random.default_rng(seed)
    k = len(members)
    values: list[float] = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[j] for j in rng.integers(0, k, size=k)])
        yy, ss = y[idx], s[idx]
        if _mixed(yy):
            values.append(_auc_of(ss[yy], ss[~yy]))
    if len(values) < _MIN_USABLE_SHARE * n_resamples:
        raise ValueError(f"only {len(values)} of {n_resamples} bootstrap replicates held both classes")
    lower, upper = _percentile(values, conf_level)
    method = "bootstrap_percentile" if groups is None else "cluster_bootstrap_percentile"
    return AucEstimate(auc=_auc_of(s[y], s[~y]), n_pos=m, n_neg=n, lower=lower, upper=upper,
                       conf_level=conf_level, method=method, n_units=k, n_resamples_used=len(values))


def bootstrap_gap(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    in_a: npt.ArrayLike,
    in_b: npt.ArrayLike,
    *,
    groups: Sequence[Hashable] | npt.NDArray[np.generic] | None = None,
    n_resamples: int = 2000,
    conf_level: float = 0.95,
    seed: int = 0,
) -> AucComparison:
    """AUC gap between two subsets of one dataset, with a percentile bootstrap interval.

    ``in_a`` and ``in_b`` are boolean masks over the cases (for example "scanner vendor is
    X" and "scanner vendor is Y", or a subgroup and everyone else). Clusters from
    ``groups`` are resampled across the whole dataset, so a person with cases in both
    subsets is handled correctly, and the subset sizes vary between replicates as they
    would in a new sample. The masks may not overlap.

    No p-value is returned: judge the interval against the smallest gap that matters.
    """
    _check_boot(n_resamples, conf_level)
    y = _labels(y_true)
    s = _scores(y, y_score)
    a = np.asarray(in_a, dtype=bool)
    b = np.asarray(in_b, dtype=bool)
    if a.shape != y.shape or b.shape != y.shape:
        raise ValueError("in_a and in_b must have one value per case")
    if bool((a & b).any()):
        raise ValueError("in_a and in_b overlap")
    _need_both(y[a])
    _need_both(y[b])
    members = _clusters(groups, len(y))
    rng = np.random.default_rng(seed)
    k = len(members)
    values: list[float] = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[j] for j in rng.integers(0, k, size=k)])
        yy, ss, aa, bb = y[idx], s[idx], a[idx], b[idx]
        ya, yb = yy[aa], yy[bb]
        if _mixed(ya) and _mixed(yb):
            values.append(_auc_of(ss[aa][ya], ss[aa][~ya]) - _auc_of(ss[bb][yb], ss[bb][~yb]))
    if len(values) < _MIN_USABLE_SHARE * n_resamples:
        raise ValueError(f"only {len(values)} of {n_resamples} bootstrap replicates held both classes in both subsets")
    auc_a = _auc_of(s[a][y[a]], s[a][~y[a]])
    auc_b = _auc_of(s[b][y[b]], s[b][~y[b]])
    lower, upper = _percentile(values, conf_level)
    method = "bootstrap_percentile" if groups is None else "cluster_bootstrap_percentile"
    return AucComparison(auc_a=auc_a, auc_b=auc_b, difference=auc_a - auc_b, lower=lower, upper=upper,
                         conf_level=conf_level, method=method, paired=False, n_resamples_used=len(values))
