"""Study planning: how many cases a test set needs, and how to spread them over groups.

This module answers questions asked *before* a model is run:

* How wide will the confidence interval on AUC, sensitivity or specificity be with ``n``
  cases, and how many cases does a target width need? (:func:`auc_precision`,
  :func:`auc_sample_size`, :func:`proportion_precision`, :func:`proportion_sample_size`,
  :func:`sensitivity_precision`, :func:`specificity_precision`)
* Given a target population, how many cases belong in each stratum?
  (:func:`stratum_shares`, :func:`allocate`, :func:`allocation_table`)
* Where does the data actually available fall short of that plan, and what precision
  will a short stratum still give? (:func:`compare_to_available`, :func:`pool_levels`,
  :func:`auc_precision_from_counts`)

Every function is pure and deterministic; none does IO. Nothing here is specific to a
disease, a modality or an imaging discipline: the caller names the outcome and the
stratification factors.

Methods
-------
AUC intervals use the Hanley and McNeil (1982) variance of the area under the ROC curve
and a normal approximation, as R's ``presize::prec_auc`` does; the tests reproduce
``presize`` output. The formula assumes independent cases. For repeat images of one
person, plan with the effective number of cases
(:func:`slicepoint_core.stats.kish_effective_n`), not the image count.

Proportions use the Wilson score interval evaluated at the expected proportion, as
``presize::prec_prop(method="wilson")`` does and as :func:`slicepoint_core.stats.wilson_interval`
does for observed counts.

These are precision (interval-width) calculations. They do not give the power of a test
that two groups differ, and a planned width is an expectation, not a guarantee.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as st

__all__ = [
    "AucPrecision",
    "ProportionPrecision",
    "allocate",
    "allocation_table",
    "auc_precision",
    "auc_precision_from_counts",
    "auc_sample_size",
    "auc_standard_error",
    "compare_to_available",
    "pool_levels",
    "proportion_precision",
    "proportion_sample_size",
    "sensitivity_precision",
    "specificity_precision",
    "stratum_shares",
]

_N_MAX = 1e9
"""Upper bracket of the sample-size searches."""


def _z(conf_level: float) -> float:
    if not 0.0 < conf_level < 1.0:
        raise ValueError(f"conf_level must be in (0, 1), got {conf_level}")
    return float(st.norm.ppf(0.5 + conf_level / 2.0))


def _unit(name: str, value: float, *, open_interval: bool = True) -> None:
    ok = 0.0 < value < 1.0 if open_interval else 0.0 <= value <= 1.0
    if not ok:
        raise ValueError(f"{name} must be in {'(0, 1)' if open_interval else '[0, 1]'}, got {value}")


def _solve_n(width_at: Callable[[float], float], target: float, n_low: float) -> float:
    """Smallest real ``n >= n_low`` with ``width_at(n) <= target`` (width falls as n grows)."""
    if target <= 0:
        raise ValueError(f"conf_width must be positive, got {target}")
    if width_at(n_low) <= target:
        return n_low
    if width_at(_N_MAX) > target:
        raise ValueError(f"conf_width {target} needs more than {_N_MAX:.0e} cases")
    lo, hi = n_low, _N_MAX
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if width_at(mid) > target:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 1e-9 * hi:
            break
    return hi


# --------------------------------------------------------------------------- AUC


def auc_standard_error(auc: float, n_pos: float, n_neg: float) -> float:
    """Hanley-McNeil (1982) standard error of an AUC from ``n_pos`` and ``n_neg`` cases.

    ``Q1 = A / (2 - A)``, ``Q2 = 2 A^2 / (1 + A)`` and
    ``SE^2 = [A(1-A) + (n_pos-1)(Q1-A^2) + (n_neg-1)(Q2-A^2)] / (n_pos n_neg)``.
    Counts may be fractional (an expected count ``n * prevalence``).
    """
    _unit("auc", auc)
    if n_pos <= 0 or n_neg <= 0:
        raise ValueError(f"need positive counts in both classes, got n_pos={n_pos}, n_neg={n_neg}")
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (auc * (1.0 - auc) + (n_pos - 1.0) * (q1 - auc * auc) + (n_neg - 1.0) * (q2 - auc * auc)) / (n_pos * n_neg)
    return math.sqrt(max(var, 0.0))


@dataclass(frozen=True)
class AucPrecision:
    """Expected confidence interval of an AUC for a planned test set.

    ``n_pos`` and ``n_neg`` are expected counts and may be fractional. ``lower`` and
    ``upper`` are the untruncated normal-approximation bounds, so ``upper`` can exceed 1
    for a small set and a high AUC; ``width`` is their difference.
    """

    auc: float
    n: float
    prevalence: float
    n_pos: float
    n_neg: float
    lower: float
    upper: float
    width: float
    conf_level: float


def auc_precision_from_counts(auc: float, n_pos: float, n_neg: float, *, conf_level: float = 0.95) -> AucPrecision:
    """Expected AUC interval when the positive and negative counts are known.

    Use this for a subgroup whose available cases are already counted: it shows the
    precision the subgroup will actually give, whatever the plan asked for.
    """
    n = n_pos + n_neg
    return _auc_interval(auc, n, n_pos / n, n_pos, n_neg, conf_level)


def _auc_interval(auc: float, n: float, prevalence: float, n_pos: float, n_neg: float,
                  conf_level: float) -> AucPrecision:
    half = _z(conf_level) * auc_standard_error(auc, n_pos, n_neg)
    return AucPrecision(auc=auc, n=n, prevalence=prevalence, n_pos=n_pos, n_neg=n_neg, lower=auc - half,
                        upper=auc + half, width=2.0 * half, conf_level=conf_level)


def auc_precision(auc: float, n: float, prevalence: float, *, conf_level: float = 0.95) -> AucPrecision:
    """Expected AUC interval for ``n`` cases of which a share ``prevalence`` is positive.

    ``prevalence`` is the share of cases that are positive *by the reference standard the
    AUC is scored against* in the test set, which is not always disease prevalence.
    """
    _unit("prevalence", prevalence)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    return _auc_interval(auc, float(n), prevalence, n * prevalence, n * (1.0 - prevalence), conf_level)


def auc_sample_size(auc: float, prevalence: float, conf_width: float, *, conf_level: float = 0.95) -> AucPrecision:
    """Smallest whole number of cases whose expected AUC interval is no wider than ``conf_width``.

    The returned interval is evaluated at that whole number, so its ``width`` is at most
    ``conf_width``.
    """
    _unit("auc", auc)
    _unit("prevalence", prevalence)
    z = _z(conf_level)

    def width(n: float) -> float:
        return 2.0 * z * auc_standard_error(auc, n * prevalence, n * (1.0 - prevalence))

    n_low = 1.0 / min(prevalence, 1.0 - prevalence)  # at least one expected case in each class
    n = math.ceil(_solve_n(width, conf_width, n_low) - 1e-9)
    return auc_precision(auc, float(n), prevalence, conf_level=conf_level)


# --------------------------------------------------------------------------- proportions


@dataclass(frozen=True)
class ProportionPrecision:
    """Expected Wilson interval of a proportion (sensitivity, specificity, any rate).

    ``n`` is the number of cases the proportion is computed on: the positives for
    sensitivity, the negatives for specificity. ``n_total`` and ``prevalence`` are set
    when the plan was stated for the whole test set. ``center`` is the Wilson midpoint,
    which is pulled from ``p`` toward 0.5.
    """

    p: float
    n: float
    lower: float
    upper: float
    width: float
    center: float
    conf_level: float
    n_total: float | None = None
    prevalence: float | None = None


def proportion_precision(p: float, n: float, *, conf_level: float = 0.95) -> ProportionPrecision:
    """Expected Wilson score interval for a proportion ``p`` observed on ``n`` cases."""
    _unit("p", p, open_interval=False)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z = _z(conf_level)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lower, upper = max(0.0, center - half), min(1.0, center + half)
    return ProportionPrecision(p=p, n=n, lower=lower, upper=upper, width=upper - lower, center=center,
                               conf_level=conf_level)


def proportion_sample_size(p: float, conf_width: float, *, conf_level: float = 0.95) -> ProportionPrecision:
    """Smallest whole number of cases whose expected Wilson interval is no wider than ``conf_width``."""
    _unit("p", p, open_interval=False)

    def width(n: float) -> float:
        return proportion_precision(p, n, conf_level=conf_level).width

    n = math.ceil(_solve_n(width, conf_width, 1.0) - 1e-9)
    return proportion_precision(p, float(n), conf_level=conf_level)


def _class_precision(p: float, n_total: float, class_share: float, prevalence: float,
                     conf_level: float) -> ProportionPrecision:
    _unit("prevalence", prevalence)
    if n_total <= 0:
        raise ValueError(f"n_total must be positive, got {n_total}")
    r = proportion_precision(p, n_total * class_share, conf_level=conf_level)
    return ProportionPrecision(p=r.p, n=r.n, lower=r.lower, upper=r.upper, width=r.width, center=r.center,
                               conf_level=conf_level, n_total=n_total, prevalence=prevalence)


def sensitivity_precision(sens: float, n_total: float, prevalence: float, *,
                          conf_level: float = 0.95) -> ProportionPrecision:
    """Expected interval of a sensitivity when ``n_total * prevalence`` cases are positive."""
    return _class_precision(sens, n_total, prevalence, prevalence, conf_level)


def specificity_precision(spec: float, n_total: float, prevalence: float, *,
                          conf_level: float = 0.95) -> ProportionPrecision:
    """Expected interval of a specificity when ``n_total * (1 - prevalence)`` cases are negative."""
    return _class_precision(spec, n_total, 1.0 - prevalence, prevalence, conf_level)


# --------------------------------------------------------------------------- allocation


def stratum_shares(population: pd.DataFrame, by: Sequence[str]) -> pd.Series:
    """Share of the target population in each joint level of ``by``.

    ``population`` is one row per person or case from the population the model is meant
    for (a registry extract, a published cohort expanded to rows, or samples drawn from a
    population model). Strata are the *joint* levels, so ``by=["sex", "vendor"]`` gives
    one share per sex-vendor combination that occurs. Rows with a missing value in any
    ``by`` column are an error: decide how to treat them before planning.
    """
    cols = list(by)
    if not cols:
        raise ValueError("by must name at least one column")
    missing = [c for c in cols if c not in population.columns]
    if missing:
        raise ValueError(f"population has no column(s) {missing}")
    if len(population) == 0:
        raise ValueError("population is empty")
    if bool(population[cols].isna().any().any()):
        raise ValueError(f"population has missing values in {cols}")
    shares = population.groupby(cols, observed=True, sort=True).size() / len(population)
    out: pd.Series = shares.rename("share")
    return out


def allocate(shares: Mapping[Any, float] | pd.Series, n: int) -> pd.Series:
    """Split ``n`` cases over strata in proportion to ``shares``, summing to exactly ``n``.

    Uses the largest-remainder method: each stratum gets the floor of its exact quota and
    the cases left over go to the largest fractional parts (ties broken by stratum order).
    Rounding each stratum separately does not preserve the total. ``shares`` need not sum
    to 1; they are normalized.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    s = shares if isinstance(shares, pd.Series) else pd.Series(dict(shares), dtype=float)
    s = s.astype(float)
    if len(s) == 0:
        raise ValueError("no strata")
    if bool((s < 0).any()) or not math.isfinite(float(s.sum())) or float(s.sum()) <= 0:
        raise ValueError("shares must be non-negative with a positive, finite sum")
    quota = s.to_numpy() / float(s.sum()) * n
    base = np.floor(quota + 1e-12).astype(int)
    left = int(n - base.sum())
    if left > 0:
        order = np.argsort(-(quota - base), kind="stable")
        base[order[:left]] += 1
    out: pd.Series = pd.Series(base, index=s.index, name="target_n")
    return out


def allocation_table(
    population: pd.DataFrame,
    by: Sequence[str],
    targets: Mapping[Any, int],
    *,
    outcome: str,
) -> pd.DataFrame:
    """Target number of cases per joint stratum, separately for each outcome class.

    ``targets`` maps each outcome class (the values of the ``outcome`` column, for example
    ``{"positive": 286, "negative": 214}``) to the number of cases planned for it,
    typically from :func:`auc_sample_size`. Within a class, cases are spread over the
    joint levels of ``by`` in the proportions that class has in ``population``, so a
    subgroup that is more often positive gets a larger part of the positives.

    Returns one row per (outcome class, stratum) with ``share`` (within the class) and
    ``target_n``; the ``target_n`` of a class sum to its target exactly.
    """
    if outcome in by:
        raise ValueError("outcome must not be one of the stratification columns")
    if outcome not in population.columns:
        raise ValueError(f"population has no column {outcome!r}")
    parts: list[pd.DataFrame] = []
    for cls, n in targets.items():
        sub = population[population[outcome] == cls]
        if len(sub) == 0:
            raise ValueError(f"population has no rows with {outcome} == {cls!r}")
        shares = stratum_shares(sub, by)
        part = pd.DataFrame({"share": shares, "target_n": allocate(shares, int(n))}).reset_index()
        part.insert(0, outcome, cls)
        parts.append(part)
    if not parts:
        raise ValueError("targets is empty")
    table: pd.DataFrame = pd.concat(parts, ignore_index=True)
    return table


# --------------------------------------------------------------------------- plan vs. available


def compare_to_available(
    plan: pd.DataFrame,
    available: pd.DataFrame,
    *,
    on: Sequence[str],
    available_column: str = "available_n",
) -> pd.DataFrame:
    """Set a plan beside the cases actually available, stratum by stratum.

    ``plan`` has the stratum columns ``on`` and ``target_n`` (the output of
    :func:`allocation_table`); ``available`` has the same stratum columns and a count
    column. A planned stratum absent from ``available`` has zero cases. Adds:

    * ``available_n``: cases on hand;
    * ``shortfall``: ``max(target_n - available_n, 0)``;
    * ``fill``: ``available_n / target_n`` (``NaN`` when the target is zero);
    * ``status``: ``"met"``, ``"short"`` or ``"empty"`` (``"met"`` when the target is zero).

    Strata that are available but not planned are left out: they are outside the target
    population as described.
    """
    keys = list(on)
    for name, frame, need in (("plan", plan, [*keys, "target_n"]), ("available", available, [*keys, available_column])):
        lacking = [c for c in need if c not in frame.columns]
        if lacking:
            raise ValueError(f"{name} has no column(s) {lacking}")
    if bool(available.duplicated(keys).any()):
        raise ValueError(f"available has more than one row for some stratum of {keys}")
    have = available[[*keys, available_column]].rename(columns={available_column: "available_n"})
    out: pd.DataFrame = plan.merge(have, how="left", on=keys)
    out["available_n"] = out["available_n"].fillna(0).astype(int)
    out["shortfall"] = (out["target_n"] - out["available_n"]).clip(lower=0).astype(int)
    out["fill"] = (out["available_n"] / out["target_n"].where(out["target_n"] > 0)).astype(float)
    out["status"] = np.where(out["shortfall"] == 0, "met", np.where(out["available_n"] == 0, "empty", "short"))
    return out


def pool_levels(
    table: pd.DataFrame,
    column: str,
    pools: Mapping[str, Sequence[Hashable]],
    *,
    within: Sequence[str] = (),
) -> pd.DataFrame:
    """Merge the levels of one stratification column into broader "super-strata".

    When single strata cannot be filled, a plan can still be met at a coarser level, for
    example three under-filled ethnic groups reported together. ``pools`` maps each new
    level to the existing levels it replaces; levels not named are kept as they are.
    Which levels may be merged is a clinical judgement, so the caller states it.

    ``within`` lists the other stratum columns to keep (for example the outcome class and
    sex). Counts (``target_n``, ``available_n`` and ``shortfall`` if present) are summed;
    ``shortfall`` is recomputed from the sums, because a surplus in one level offsets a
    gap in another once they are pooled. Other columns (``share``, ``fill``, ``status``)
    are dropped, because they do not add up. A ``pooled_from`` column lists the original
    levels.
    """
    if column not in table.columns:
        raise ValueError(f"table has no column {column!r}")
    seen: dict[Hashable, str] = {}
    for new, olds in pools.items():
        for old in olds:
            if old in seen:
                raise ValueError(f"level {old!r} is in two pools ({seen[old]!r} and {new!r})")
            seen[old] = new
    counts = [c for c in ("target_n", "available_n") if c in table.columns]
    if not counts:
        raise ValueError("table has neither target_n nor available_n")
    t = table.copy()
    t["_orig"] = t[column].astype(object)
    t[column] = t["_orig"].map(lambda v: seen.get(v, v))
    keys = [*within, column]
    grouped = t.groupby(keys, observed=True, sort=False)
    out: pd.DataFrame = grouped[counts].sum().reset_index()
    out["pooled_from"] = grouped["_orig"].agg(lambda s: tuple(dict.fromkeys(s))).to_numpy()
    if "target_n" in out and "available_n" in out:
        out["shortfall"] = (out["target_n"] - out["available_n"]).clip(lower=0).astype(int)
    return out
