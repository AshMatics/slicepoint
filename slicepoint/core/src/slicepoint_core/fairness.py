"""Subgroup performance and fairness gaps for a binary reference standard.

Everything here works on one table with one row per case (an image, a lesion, a slide,
a study) and plain column names:

* a reference-standard column (boolean or 0/1),
* a model score column and/or a column of binary calls (or a threshold on the score),
* one or more factor columns to slice by (sex, scanner vendor, site, skin tone, ...),
* optionally a person column, when a person can contribute more than one case.

Three functions:

* :func:`subgroup_metrics` gives each subgroup's metrics with an interval;
* :func:`subgroup_gaps` compares each subgroup with a reference subgroup (or with
  everyone else) and judges each gap against the smallest difference that matters;
* :func:`scan` does the same across several factors and their intersections, treating
  all of the comparisons as one family.

Methods. Without a person column, or when every person has one case: Wilson intervals
for proportions, DeLong for AUC, the Newcombe hybrid-score interval and Fisher's exact
test for a difference of proportions, and the unpaired DeLong test for a difference of
AUCs. When any person has several cases those formulas are too confident, so every
interval is a percentile bootstrap that resamples people, and no p-value is given.

Three safeguards apply throughout:

* **Small groups.** An estimate resting on fewer than ``min_count`` independent units is
  not reported (status ``too_small``); the counts are.
* **Many comparisons.** Intervals are widened (Bonferroni) and p-values adjusted (Holm)
  for the number of comparisons made in one call.
* **A margin, not just a p-value.** With a ``margin`` (the smallest difference that
  matters), each gap is ``within_margin``, ``exceeds_margin`` or ``inconclusive``
  according to where its whole interval lies. A gap that is merely "not significant" is
  inconclusive, not fair.

Rows with a missing factor value form their own subgroup, labelled ``(missing)``:
people whose demographics were not recorded are a group worth looking at.

Pure functions: nothing here reads files or keeps state.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as st

from . import roc
from .stats import fisher_exact_p, wilson_interval

__all__ = [
    "METRICS",
    "MISSING",
    "holm",
    "newcombe_interval",
    "scan",
    "subgroup_gaps",
    "subgroup_metrics",
]

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.intp]

METRICS: tuple[str, ...] = ("auc", "sensitivity", "specificity", "ppv", "npv", "positive_rate")
"""``auc`` needs a score; the others need binary calls. ``positive_rate`` is the share of
cases the model calls positive (the quantity behind "demographic parity")."""

MISSING = "(missing)"
"""Label given to a missing factor value."""

_MIN_USABLE_SHARE = 0.9
_BOOT_CELLS = 2_000_000  # replicates x clusters drawn at once


# --------------------------------------------------------------------------- small pieces


def _z(conf_level: float) -> float:
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    return float(st.norm.ppf(0.5 + conf_level / 2.0))


def _wilson(k: int, n: int, conf_level: float) -> tuple[float, float]:
    out = wilson_interval(k, n, _z(conf_level))
    if out is None:
        raise ValueError("a proportion needs at least one case in its denominator")
    return out


def newcombe_interval(k1: int, n1: int, k2: int, n2: int, *, conf_level: float = 0.95) -> tuple[float, float]:
    """Interval for ``k1/n1 - k2/n2``, two independent proportions (Newcombe 1998, method 10).

    Built from the two Wilson intervals, so it stays inside [-1, 1] and behaves when a
    proportion is 0 or 1, where the usual normal-approximation interval collapses.
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = _wilson(k1, n1, conf_level)
    l2, u2 = _wilson(k2, n2, conf_level)
    d = p1 - p2
    return d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def holm(p_values: Sequence[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values. ``None`` entries stay ``None`` and are not counted."""
    idx = [i for i, p in enumerate(p_values) if p is not None]
    out: list[float | None] = [None] * len(p_values)
    order = sorted(idx, key=lambda i: float(p_values[i]))  # type: ignore[arg-type]
    running = 0.0
    for rank, i in enumerate(order):
        p = p_values[i]
        assert p is not None
        running = max(running, min(1.0, (len(order) - rank) * p))
        out[i] = running
    return out


# --------------------------------------------------------------------------- the case table


@dataclass(frozen=True)
class _Cases:
    y: BoolArray
    score: FloatArray | None
    pred: BoolArray | None
    unit: IntArray  # person index per case (the case's own index when no person column)
    clustered: bool  # some person has more than one case
    metrics: tuple[str, ...]


def _column(data: pd.DataFrame, name: str, role: str) -> npt.NDArray[Any]:
    if name not in data.columns:
        raise ValueError(f"{role} column {name!r} is not in the table")
    return np.asarray(data[name].to_numpy())


def _cases(
    data: pd.DataFrame,
    y_true: str,
    y_score: str | None,
    y_pred: str | None,
    threshold: float | None,
    unit: str | None,
    metrics: Sequence[str] | None,
) -> _Cases:
    if len(data) == 0:
        raise ValueError("the table is empty")
    y = roc._labels(_column(data, y_true, "y_true"))
    score = None if y_score is None else roc._scores(y, _column(data, y_score, "y_score"))
    if y_pred is not None and threshold is not None:
        raise ValueError("give y_pred or threshold, not both")
    pred: BoolArray | None = None
    if y_pred is not None:
        try:
            pred = roc._labels(_column(data, y_pred, "y_pred"))
        except ValueError as exc:
            raise ValueError(str(exc).replace("y_true", "y_pred")) from None
    elif threshold is not None:
        if score is None:
            raise ValueError("threshold needs y_score")
        pred = score >= threshold
    if score is None and pred is None:
        raise ValueError("give y_score, y_pred, or y_score with a threshold")

    available = (("auc",) if score is not None else ()) + (METRICS[1:] if pred is not None else ())
    chosen = available if metrics is None else tuple(metrics)
    if not chosen:
        raise ValueError("metrics is empty")
    for m in chosen:
        if m not in METRICS:
            raise ValueError(f"unknown metric {m!r}; choose from {METRICS}")
        if m not in available:
            raise ValueError(f"metric {m!r} needs {'y_score' if m == 'auc' else 'y_pred or a threshold'}")
    if len(set(chosen)) != len(chosen):
        raise ValueError("metrics has duplicates")

    if unit is None:
        codes = np.arange(len(y), dtype=np.intp)
        clustered = False
    else:
        raw = np.asarray(pd.factorize(pd.Series(_column(data, unit, "unit"), dtype=object))[0])
        if bool((raw < 0).any()):
            raise ValueError(f"unit column {unit!r} has missing values; give every case a person id")
        codes = raw.astype(np.intp)
        clustered = int(np.max(codes)) + 1 < len(codes)
    return _Cases(y=y, score=score, pred=pred, unit=codes, clustered=clustered, metrics=chosen)


def _levels(data: pd.DataFrame, by: Sequence[str]) -> tuple[IntArray, list[tuple[Hashable, ...]]]:
    """Group index per case and the joint levels, missing values relabelled."""
    cols = list(by)
    if not cols:
        raise ValueError("by must name at least one column")
    absent = [c for c in cols if c not in data.columns]
    if absent:
        raise ValueError(f"the table has no column(s) {absent}")
    if len(set(cols)) != len(cols):
        raise ValueError("by has duplicate columns")
    filled = data[cols].astype(object).where(data[cols].notna(), MISSING).reset_index(drop=True)
    try:
        codes = filled.groupby(cols, sort=True, observed=True).ngroup().to_numpy()
    except TypeError:  # levels that cannot be ordered against each other
        codes = filled.groupby(cols, sort=False, observed=True).ngroup().to_numpy()
    first = pd.Series(np.arange(len(codes)), dtype=np.int64).groupby(codes).first().to_numpy()
    levels = [tuple(filled.iloc[int(i)].tolist()) for i in first]
    return codes.astype(np.intp), levels


def _by(by: str | Sequence[str]) -> list[str]:
    return [by] if isinstance(by, str) else list(by)


# --------------------------------------------------------------------------- one estimate


def _parts(metric: str, y: BoolArray, pred: BoolArray) -> tuple[BoolArray, BoolArray]:
    """(numerator, denominator) case masks of a proportion metric."""
    if metric == "sensitivity":
        return pred & y, y
    if metric == "specificity":
        return ~pred & ~y, ~y
    if metric == "ppv":
        return pred & y, pred
    if metric == "npv":
        return ~pred & ~y, ~pred
    return pred, np.ones_like(y)  # positive_rate


def _n_units(c: _Cases, mask: BoolArray) -> int:
    return int(len(np.unique(c.unit[mask])))


def _support(c: _Cases, metric: str, group: BoolArray) -> int:
    """Independent units the estimate rests on: people in the metric's denominator, or
    in the smaller class for AUC."""
    if metric == "auc":
        return min(_n_units(c, group & c.y), _n_units(c, group & ~c.y))
    assert c.pred is not None
    return _n_units(c, group & _parts(metric, c.y, c.pred)[1])


def _floor(metric: str, min_count: int) -> int:
    return max(min_count, 2) if metric == "auc" else min_count


def _draws(n_clusters: int, n_resamples: int, seed: int) -> list[IntArray]:
    """Cluster indices for each replicate, in chunks of rows."""
    rng = np.random.default_rng(seed)
    step = max(1, _BOOT_CELLS // max(n_clusters, 1))
    out: list[IntArray] = []
    done = 0
    while done < n_resamples:
        rows = min(step, n_resamples - done)
        out.append(rng.integers(0, n_clusters, size=(rows, n_clusters)).astype(np.intp))
        done += rows
    return out


def _cluster_sums(unit: IntArray, masks: Sequence[BoolArray]) -> tuple[int, list[FloatArray]]:
    _, compact = np.unique(unit, return_inverse=True)
    k = int(np.max(compact)) + 1
    return k, [np.bincount(compact, weights=m.astype(float), minlength=k) for m in masks]


def _percentile(values: FloatArray, conf_level: float) -> tuple[float, float]:
    alpha = 1.0 - conf_level
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


class _Unstable(ValueError):
    """Too few bootstrap replicates had what the metric needs."""


def _boot_proportion(
    c: _Cases, within: BoolArray, parts: Sequence[tuple[BoolArray, BoolArray]], conf_level: float, n_resamples: int, seed: int
) -> tuple[float, float]:
    """Percentile interval for one ratio, or for the difference of two, resampling people in ``within``."""
    masks = [m[within] for pair in parts for m in pair]
    k, sums = _cluster_sums(c.unit[within], masks)
    values: list[FloatArray] = []
    for idx in _draws(k, n_resamples, seed):
        ratios = []
        ok = np.ones(idx.shape[0], dtype=bool)
        for j in range(0, len(sums), 2):
            num, den = sums[j][idx].sum(axis=1), sums[j + 1][idx].sum(axis=1)
            ok &= den > 0
            ratios.append(num / np.where(den > 0, den, 1.0))
        stat = ratios[0] if len(ratios) == 1 else ratios[0] - ratios[1]
        values.append(stat[ok])
    usable = np.concatenate(values)
    if len(usable) < _MIN_USABLE_SHARE * n_resamples:
        raise _Unstable
    return _percentile(usable, conf_level)


@dataclass(frozen=True)
class _Estimate:
    value: float | None
    lower: float | None
    upper: float | None
    method: str | None
    status: str
    support: int
    numerator: int | None
    denominator: int


def _counts(c: _Cases, metric: str, group: BoolArray) -> tuple[int | None, int]:
    if metric == "auc":
        return None, int(group.sum())
    assert c.pred is not None
    num, den = _parts(metric, c.y, c.pred)
    return int((group & num).sum()), int((group & den).sum())


def _estimate(c: _Cases, metric: str, group: BoolArray, conf_level: float, min_count: int, n_resamples: int, seed: int) -> _Estimate:
    support = _support(c, metric, group)
    numerator, denominator = _counts(c, metric, group)
    if support < _floor(metric, min_count):
        return _Estimate(None, None, None, None, "too_small", support, numerator, denominator)
    if metric == "auc":
        assert c.score is not None
        y, s = c.y[group], c.score[group]
        if not c.clustered:
            e = roc.auc_delong(y, s, conf_level=conf_level)
            return _Estimate(e.auc, e.lower, e.upper, "delong", "ok", support, numerator, denominator)
        value = roc.auc(y, s)
        try:
            e = roc.bootstrap_auc(y, s, groups=c.unit[group], n_resamples=n_resamples, conf_level=conf_level, seed=seed)
        except ValueError:
            return _Estimate(value, None, None, "cluster_bootstrap_percentile", "unstable", support, numerator, denominator)
        return _Estimate(value, e.lower, e.upper, e.method, "ok", support, numerator, denominator)
    assert c.pred is not None and numerator is not None
    value = numerator / denominator
    if not c.clustered:
        lower, upper = _wilson(numerator, denominator, conf_level)
        return _Estimate(value, lower, upper, "wilson", "ok", support, numerator, denominator)
    try:
        lower, upper = _boot_proportion(c, group, [_parts(metric, c.y, c.pred)], conf_level, n_resamples, seed)
    except _Unstable:
        return _Estimate(value, None, None, "cluster_bootstrap_percentile", "unstable", support, numerator, denominator)
    return _Estimate(value, lower, upper, "cluster_bootstrap_percentile", "ok", support, numerator, denominator)


# --------------------------------------------------------------------------- one gap


@dataclass(frozen=True)
class _Gap:
    lower: float | None
    upper: float | None
    p_value: float | None
    method: str
    status: str


def _gap(c: _Cases, metric: str, a: BoolArray, b: BoolArray, conf_level: float, n_resamples: int, seed: int) -> _Gap:
    """Interval (and p-value where one exists) for metric(a) - metric(b); the masks do not overlap."""
    if metric == "auc":
        assert c.score is not None
        if not c.clustered:
            r = roc.compare_unpaired(c.y[a], c.score[a], c.y[b], c.score[b], conf_level=conf_level)
            return _Gap(r.lower, r.upper, r.p_value, "delong_unpaired", "ok")
        both = a | b
        try:
            r = roc.bootstrap_gap(c.y[both], c.score[both], a[both], b[both], groups=c.unit[both],
                                  n_resamples=n_resamples, conf_level=conf_level, seed=seed)
        except ValueError:
            return _Gap(None, None, None, "cluster_bootstrap_percentile", "unstable")
        return _Gap(r.lower, r.upper, None, r.method, "ok")
    assert c.pred is not None
    num, den = _parts(metric, c.y, c.pred)
    if not c.clustered:
        k1, n1, k2, n2 = int((a & num).sum()), int((a & den).sum()), int((b & num).sum()), int((b & den).sum())
        lower, upper = newcombe_interval(k1, n1, k2, n2, conf_level=conf_level)
        return _Gap(lower, upper, fisher_exact_p(k1, n1, k2, n2), "newcombe_fisher", "ok")
    both = a | b
    try:
        lower, upper = _boot_proportion(c, both, [(a & num, a & den), (b & num, b & den)], conf_level, n_resamples, seed)
    except _Unstable:
        return _Gap(None, None, None, "cluster_bootstrap_percentile", "unstable")
    return _Gap(lower, upper, None, "cluster_bootstrap_percentile", "ok")


def _verdict(lower: float | None, upper: float | None, margin: float | None) -> str | None:
    if margin is None or lower is None or upper is None:
        return None
    if -margin < lower and upper < margin:
        return "within_margin"
    if lower > margin or upper < -margin:
        return "exceeds_margin"
    return "inconclusive"


def _margin_for(margin: float | Mapping[str, float] | None, metric: str) -> float | None:
    if margin is None:
        return None
    value = margin.get(metric) if isinstance(margin, Mapping) else margin
    if value is None:
        return None
    if not value > 0:
        raise ValueError(f"margin must be positive, got {value}")
    return float(value)


# --------------------------------------------------------------------------- public: metrics


def _check_common(conf_level: float, min_count: int, n_resamples: int) -> None:
    _z(conf_level)
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if n_resamples < 200:
        raise ValueError("n_resamples must be at least 200")


def subgroup_metrics(
    data: pd.DataFrame,
    by: str | Sequence[str],
    *,
    y_true: str,
    y_score: str | None = None,
    y_pred: str | None = None,
    threshold: float | None = None,
    unit: str | None = None,
    metrics: Sequence[str] | None = None,
    conf_level: float = 0.95,
    min_count: int = 10,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each subgroup's metrics with an interval: one row per subgroup and metric.

    ``by`` names one factor column, or several for their joint levels (``["sex", "vendor"]``
    gives one subgroup per sex-vendor combination that occurs). Binary calls come from
    ``y_pred``, or from ``y_score >= threshold``. ``unit`` names the person column.
    ``metrics`` defaults to everything the inputs allow (see :data:`METRICS`).

    Columns: the ``by`` columns, ``metric``, ``n`` (cases in the subgroup), ``n_units``
    (people), ``n_pos``, ``n_neg``, ``numerator`` and ``denominator`` (cases; ``numerator``
    is empty for AUC), ``support`` (people the estimate rests on: those in the
    denominator, or in the smaller class for AUC), ``value``, ``lower``, ``upper``,
    ``conf_level``, ``method`` and ``status``.

    ``status`` is ``ok``; ``too_small`` when ``support`` is below ``min_count`` (no value
    is given); or ``unstable`` when too few bootstrap replicates held what the metric
    needs (a value but no interval). Intervals here are not adjusted for the number of
    subgroups: they describe each subgroup on its own.
    """
    _check_common(conf_level, min_count, n_resamples)
    cols = _by(by)
    c = _cases(data, y_true, y_score, y_pred, threshold, unit, metrics)
    codes, levels = _levels(data, cols)
    rows: list[dict[str, Any]] = []
    for g, level in enumerate(levels):
        group: BoolArray = codes == g
        base = {**dict(zip(cols, level, strict=True)), "n": int(group.sum()), "n_units": _n_units(c, group),
                "n_pos": int((group & c.y).sum()), "n_neg": int((group & ~c.y).sum())}
        for metric in c.metrics:
            e = _estimate(c, metric, group, conf_level, min_count, n_resamples, seed)
            rows.append({**base, "metric": metric, "numerator": e.numerator, "denominator": e.denominator,
                         "support": e.support, "value": e.value, "lower": e.lower, "upper": e.upper,
                         "conf_level": conf_level, "method": e.method, "status": e.status})
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


# --------------------------------------------------------------------------- public: gaps


@dataclass(frozen=True)
class _Planned:
    labels: dict[str, Any]
    metric: str
    group: BoolArray
    other: BoolArray
    reference_label: Any
    this: _Estimate
    that: _Estimate
    status: str


def _plan(
    c: _Cases,
    codes: IntArray,
    levels: list[tuple[Hashable, ...]],
    reference: tuple[Hashable, ...] | None,
    versus: str,
    labels_of: Any,
    min_count: int,
) -> list[_Planned]:
    """Comparisons for one grouping, with point values and feasibility but no intervals yet."""
    sizes = np.bincount(codes, minlength=len(levels))
    ref_index: int | None = None
    if versus == "reference":
        if reference is None:
            ref_index = int(np.argmax(sizes))
        elif reference in levels:
            ref_index = levels.index(reference)
        else:
            raise ValueError(f"reference {reference!r} is not a level; levels are {levels}")
    out: list[_Planned] = []
    for g, level in enumerate(levels):
        if g == ref_index:
            continue
        group: BoolArray = codes == g
        other: BoolArray = ~group if ref_index is None else codes == ref_index
        if not bool(other.any()):
            continue  # a single level has no "everyone else"
        for metric in c.metrics:
            floor = _floor(metric, min_count)
            ests = []
            for mask in (group, other):
                support = _support(c, metric, mask)
                numerator, denominator = _counts(c, metric, mask)
                value: float | None = None
                if support >= floor:
                    if metric == "auc":
                        assert c.score is not None
                        value = roc.auc(c.y[mask], c.score[mask])
                    else:
                        assert numerator is not None
                        value = numerator / denominator
                ests.append(_Estimate(value, None, None, None, "ok" if value is not None else "too_small",
                                      support, numerator, denominator))
            status = "too_small" if ests[0].value is None else "reference_too_small" if ests[1].value is None else "ok"
            ref_label = "(rest)" if ref_index is None else labels_of(levels[ref_index])["__reference__"]
            out.append(_Planned(labels_of(level), metric, group, other, ref_label, ests[0], ests[1], status))
    return out


def _finish(
    c: _Cases,
    planned: list[_Planned],
    margin: float | Mapping[str, float] | None,
    conf_level: float,
    adjust: str,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    if adjust not in ("bonferroni", "none"):
        raise ValueError(f"adjust must be 'bonferroni' or 'none', got {adjust!r}")
    family = sum(p.status == "ok" for p in planned)
    level = conf_level if adjust == "none" or family <= 1 else 1.0 - (1.0 - conf_level) / family
    rows: list[dict[str, Any]] = []
    for p in planned:
        labels = {k: v for k, v in p.labels.items() if k != "__reference__"}
        m = _margin_for(margin, p.metric)
        row: dict[str, Any] = {
            **labels, "reference": p.reference_label, "metric": p.metric,
            "value": p.this.value, "reference_value": p.that.value, "difference": None, "ratio": None,
            "lower": None, "upper": None, "interval_level": level, "p_value": None, "p_holm": None,
            "margin": m, "verdict": None, "support": p.this.support, "reference_support": p.that.support,
            "method": None, "status": p.status,
        }
        if p.status == "ok":
            assert p.this.value is not None and p.that.value is not None
            g = _gap(c, p.metric, p.group, p.other, level, n_resamples, seed)
            row.update(difference=p.this.value - p.that.value,
                       ratio=p.this.value / p.that.value if p.that.value > 0 else None,
                       lower=g.lower, upper=g.upper, p_value=g.p_value, method=g.method, status=g.status,
                       verdict=_verdict(g.lower, g.upper, m))
        rows.append(row)
    if adjust == "bonferroni":
        for row, adjusted in zip(rows, holm([r["p_value"] for r in rows]), strict=True):
            row["p_holm"] = adjusted
    table: pd.DataFrame = pd.DataFrame(rows)
    table.attrs["family_size"] = family
    return table


def _reference_tuple(reference: Hashable | Sequence[Hashable] | None, n_cols: int) -> tuple[Hashable, ...] | None:
    if reference is None:
        return None
    if n_cols == 1 and not isinstance(reference, tuple):
        return (reference,)
    if not isinstance(reference, (tuple, list)) or len(reference) != n_cols:
        raise ValueError(f"reference must give one level for each of the {n_cols} by columns")
    return tuple(reference)


def _check_versus(versus: str) -> None:
    if versus not in ("reference", "rest"):
        raise ValueError(f"versus must be 'reference' or 'rest', got {versus!r}")


def subgroup_gaps(
    data: pd.DataFrame,
    by: str | Sequence[str],
    *,
    y_true: str,
    y_score: str | None = None,
    y_pred: str | None = None,
    threshold: float | None = None,
    unit: str | None = None,
    metrics: Sequence[str] | None = None,
    reference: Hashable | Sequence[Hashable] | None = None,
    versus: str = "reference",
    margin: float | Mapping[str, float] | None = None,
    conf_level: float = 0.95,
    adjust: str = "bonferroni",
    min_count: int = 10,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """Each subgroup against a reference: one row per subgroup and metric.

    ``versus="reference"`` compares every other subgroup with the ``reference`` level
    (a tuple of levels when ``by`` has several columns); left as ``None`` it is the
    largest subgroup. ``versus="rest"`` compares each subgroup with everyone not in it,
    which needs no reference group.

    ``margin`` is the smallest absolute difference that matters, one number or a mapping
    from metric to number. It should come from clinical judgment recorded before the
    analysis, not from the data. Each gap's ``verdict`` is then

    * ``within_margin``: the whole interval lies inside ``(-margin, margin)``;
    * ``exceeds_margin``: the whole interval lies beyond the margin on one side;
    * ``inconclusive``: anything else, usually a subgroup too small to tell.

    ``adjust="bonferroni"`` (the default) widens every interval to level
    ``1 - (1 - conf_level) / k`` for the ``k`` comparisons that could be made, and fills
    ``p_holm``; ``interval_level`` records the level used and ``table.attrs["family_size"]``
    is ``k``. ``adjust="none"`` leaves both alone.

    Columns: the ``by`` columns, ``reference``, ``metric``, ``value``,
    ``reference_value``, ``difference`` (subgroup minus reference), ``ratio``, ``lower``,
    ``upper``, ``interval_level``, ``p_value``, ``p_holm``, ``margin``, ``verdict``,
    ``support``, ``reference_support``, ``method`` and ``status`` (``ok``, ``too_small``,
    ``reference_too_small`` or ``unstable``). ``ratio`` has no interval; it is there for
    comparison with tools that report a ratio band. P-values are empty when people have
    several cases, because the intervals are then bootstrap intervals.
    """
    _check_common(conf_level, min_count, n_resamples)
    _check_versus(versus)
    cols = _by(by)
    c = _cases(data, y_true, y_score, y_pred, threshold, unit, metrics)
    codes, levels = _levels(data, cols)
    ref = _reference_tuple(reference, len(cols)) if versus == "reference" else None

    def labels_of(level: tuple[Hashable, ...]) -> dict[str, Any]:
        return {**dict(zip(cols, level, strict=True)), "__reference__": level[0] if len(cols) == 1 else level}

    return _finish(c, _plan(c, codes, levels, ref, versus, labels_of, min_count), margin, conf_level, adjust, n_resamples, seed)


def scan(
    data: pd.DataFrame,
    factors: Sequence[str],
    *,
    y_true: str,
    y_score: str | None = None,
    y_pred: str | None = None,
    threshold: float | None = None,
    unit: str | None = None,
    metrics: Sequence[str] | None = None,
    order: int = 1,
    reference: Mapping[str, Hashable] | None = None,
    versus: str = "reference",
    margin: float | Mapping[str, float] | None = None,
    conf_level: float = 0.95,
    adjust: str = "bonferroni",
    min_count: int = 10,
    n_resamples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    """:func:`subgroup_gaps` across several factors and their intersections, as one family.

    ``order=1`` looks at each factor alone; ``order=2`` adds every pair of factors
    (sex within vendor, and so on); higher orders add larger combinations. Intersections
    get small quickly, so expect ``too_small`` and ``inconclusive`` rows, which is the
    honest answer for them.

    ``reference`` maps a factor to its reference level. A combination uses the joint
    reference when every factor in it has one, and otherwise its largest subgroup.

    The multiplicity adjustment covers every comparison in the scan, so adding factors or
    raising ``order`` widens every interval. Decide what to scan before looking.

    Columns are those of :func:`subgroup_gaps`, with ``factor`` (for example
    ``"sex x vendor"``) and ``level`` (``"F / GE"``) in place of the ``by`` columns.
    """
    _check_common(conf_level, min_count, n_resamples)
    _check_versus(versus)
    names = list(factors)
    if not names:
        raise ValueError("factors must name at least one column")
    if len(set(names)) != len(names):
        raise ValueError("factors has duplicates")
    if not 1 <= order <= len(names):
        raise ValueError(f"order must be between 1 and the number of factors ({len(names)})")
    refs = dict(reference or {})
    unknown = [k for k in refs if k not in names]
    if unknown:
        raise ValueError(f"reference names factor(s) not being scanned: {unknown}")
    c = _cases(data, y_true, y_score, y_pred, threshold, unit, metrics)
    planned: list[_Planned] = []
    for size in range(1, order + 1):
        for combo in itertools.combinations(names, size):
            codes, levels = _levels(data, combo)
            ref = tuple(refs[f] for f in combo) if versus == "reference" and all(f in refs for f in combo) else None

            def labels_of(level: tuple[Hashable, ...], combo: tuple[str, ...] = combo) -> dict[str, Any]:
                text = " / ".join(str(v) for v in level)
                return {"factor": " x ".join(combo), "level": text, "__reference__": text}

            planned.extend(_plan(c, codes, levels, ref, versus, labels_of, min_count))
    return _finish(c, planned, margin, conf_level, adjust, n_resamples, seed)
