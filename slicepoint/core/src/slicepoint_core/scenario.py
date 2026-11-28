"""Build a simulated monitoring scenario from pools of cases.

A scenario is a reference set (what the model was validated on) plus a run of monthly
batches at one or more sites. It is how SlicePoint data is turned into a believable story
for surveillance over time: a site that stays in control, a site whose protocol changes in
month 7, a site whose case mix changes when a new service opens.

Inputs are plain tables, one row per case, in SlicePoint's input format (features,
predictions, labels and embeddings joined on ``case_id``), grouped into named **pools**.
Each pool row also carries a ``source_id``: the identity of the underlying scan, so the
same scan is never used twice (once as reference, once at a site) even when it appears in
two pools in different forms.

A :class:`ScenarioRecipe` says what to draw:

* the reference: how many cases, from which pool, with what positive share;
* for each site, a list of **phases**: a span of months, a pool, cases per month, and
  optionally a positive share, sampling weights by factor value, and a group column (so
  that repeat scans of one person are drawn together).

Rules the builder enforces:

* every draw is without replacement across the whole scenario, by ``source_id``;
* pools named in ``reserved_pools`` are kept for their own phases: their ``source_id``
  values are removed from every other pool before anything is drawn;
* draws happen in recipe order (reference first), and a pool that runs short raises
  :class:`ScenarioError` with the counts, rather than silently giving smaller batches;
* the same recipe and seed give the same scenario.

Case identifiers in the output are new (a hash of the scenario name, seed, pool and source
case), so a batch does not reveal which pool it came from; the provenance table keeps the
link.

Months, dates and site names in a scenario are simulated. Nothing here reads files except
:func:`load_pool` and :func:`write_scenario`.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

TABLES = ("features", "predictions", "labels", "embeddings")
SOURCE_ID = "source_id"


class ScenarioError(ValueError):
    """A recipe that cannot be built from the pools given."""


@dataclass(frozen=True)
class Phase:
    """One span of months at one site, drawn from one pool."""

    first_month: int  # 1-based, inclusive
    last_month: int  # inclusive
    pool: str
    per_month: int
    positive_share: float | None = None  # None: whatever the pool gives
    weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)  # column -> value -> weight
    group_by: str | None = None  # draw whole groups (for example all scans of a person)


@dataclass(frozen=True)
class SiteSpec:
    site_id: str
    label: str
    phases: Sequence[Phase]


@dataclass(frozen=True)
class ReferenceSpec:
    pool: str
    n: int
    positive_share: float | None = None


@dataclass(frozen=True)
class ScenarioRecipe:
    name: str
    seed: int
    start_month: str  # "YYYY-MM"
    n_months: int
    reference: ReferenceSpec
    sites: Sequence[SiteSpec]
    label_column: str = "aaa"
    reserved_pools: Sequence[str] = ()
    notes: str = ""

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ScenarioRecipe:
        sites = [
            SiteSpec(
                site_id=s["site_id"],
                label=s["label"],
                phases=[Phase(**p) for p in s["phases"]],
            )
            for s in raw["sites"]
        ]
        return cls(
            name=raw["name"],
            seed=int(raw["seed"]),
            start_month=raw["start_month"],
            n_months=int(raw["n_months"]),
            reference=ReferenceSpec(**raw["reference"]),
            sites=sites,
            label_column=raw.get("label_column", "aaa"),
            reserved_pools=tuple(raw.get("reserved_pools", ())),
            notes=raw.get("notes", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def months(self) -> list[str]:
        start = pd.Period(self.start_month, freq="M")
        return [str(start + i) for i in range(self.n_months)]


@dataclass
class Scenario:
    recipe: ScenarioRecipe
    reference: pd.DataFrame  # one row per case, all columns
    batches: pd.DataFrame  # one row per case, with site_id, month, scan_date
    provenance: pd.DataFrame  # case_id -> pool, source case_id, source_id

    def counts(self) -> pd.DataFrame:
        """Cases and positives by site and month, with the reference as its own row."""
        label = self.recipe.label_column
        ref = pd.DataFrame(
            {
                "site_id": ["(reference)"],
                "month": [""],
                "cases": [len(self.reference)],
                "positives": [int(self.reference[label].astype(bool).sum())],
            }
        )
        by = (
            self.batches.groupby(["site_id", "month"], sort=True)
            .agg(cases=("case_id", "size"), positives=(label, lambda s: int(s.astype(bool).sum())))
            .reset_index()
        )
        return cast(pd.DataFrame, pd.concat([ref, by], ignore_index=True))


def load_pool(folder: str | Path, source_ids: Mapping[str, str] | None = None) -> pd.DataFrame:
    """Read one SlicePoint input folder into one joined table.

    ``source_ids`` maps ``case_id`` to the identity of the underlying scan; without it,
    ``case_id`` is used as its own source.
    """
    folder = Path(folder)
    frame: pd.DataFrame | None = None
    for table in TABLES:
        path = folder / f"{table}.parquet"
        if not path.exists():
            continue
        part = pd.read_parquet(path)
        frame = part if frame is None else frame.merge(part, on="case_id", how="inner", validate="one_to_one")
    if frame is None:
        raise ScenarioError(f"no SlicePoint tables in {folder}")
    frame[SOURCE_ID] = frame["case_id"].map(source_ids) if source_ids is not None else frame["case_id"]
    if frame[SOURCE_ID].isna().any():
        raise ScenarioError(f"{int(frame[SOURCE_ID].isna().sum())} cases in {folder} have no source id")
    return frame


def build_scenario(recipe: ScenarioRecipe, pools: Mapping[str, pd.DataFrame]) -> Scenario:
    """Draw the reference and every site's monthly batches, as the recipe says."""
    _check_recipe(recipe, pools)
    rng = np.random.default_rng(recipe.seed)
    label = recipe.label_column

    reserved = {p: set(pools[p][SOURCE_ID]) for p in recipe.reserved_pools}
    available: dict[str, pd.DataFrame] = {}
    for name, pool in pools.items():
        blocked: set[Any] = set().union(*(ids for p, ids in reserved.items() if p != name)) if reserved else set()
        available[name] = pool[~pool[SOURCE_ID].isin(blocked)].copy()
    used: set[Any] = set()

    def take(
        pool_name: str,
        n: int,
        share: float | None,
        weights: Mapping[str, Mapping[str, float]],
        group_by: str | None,
        what: str,
    ) -> pd.DataFrame:
        pool = available[pool_name]
        pool = pool[~pool[SOURCE_ID].isin(used)]
        drawn = _draw(pool, n, share, weights, group_by, label, rng, what)
        used.update(drawn[SOURCE_ID])
        return drawn

    ref_spec = recipe.reference
    reference = take(ref_spec.pool, ref_spec.n, ref_spec.positive_share, {}, None, "the reference")
    reference = reference.assign(pool=ref_spec.pool)

    months = recipe.months()
    parts: list[pd.DataFrame] = []
    for site in recipe.sites:
        for phase in site.phases:
            span = phase.last_month - phase.first_month + 1
            what = f"site {site.site_id}, months {phase.first_month}-{phase.last_month} ({phase.pool})"
            drawn = take(phase.pool, phase.per_month * span, phase.positive_share, phase.weights, phase.group_by, what)
            month_of = _spread(drawn, span, phase.per_month, phase.group_by, rng)
            parts.append(
                drawn.assign(
                    site_id=site.site_id, month=[months[phase.first_month - 1 + i] for i in month_of], pool=phase.pool
                )
            )
    batches = pd.concat(parts, ignore_index=True) if parts else reference.iloc[0:0].assign(site_id="", month="")

    batches = batches.assign(scan_date=_scan_dates(batches["month"], rng))
    reference = _renumber(reference, recipe, "reference")
    batches = _renumber(batches, recipe, "site")
    provenance = pd.concat(
        [
            reference[["case_id", "pool", "source_case_id", SOURCE_ID]].assign(role="reference", site_id="", month=""),
            batches[["case_id", "pool", "source_case_id", SOURCE_ID, "site_id", "month"]].assign(role="site"),
        ],
        ignore_index=True,
    )
    batches = batches.sort_values(["site_id", "scan_date", "case_id"], ignore_index=True)
    return Scenario(recipe=recipe, reference=reference, batches=batches, provenance=provenance)


def fit_calibration(
    reference: pd.DataFrame, score: str = "pred_max_diameter_mm", label: str = "aaa"
) -> Callable[[pd.DataFrame], np.ndarray]:
    """Fit a calibrated probability of a positive from one model score, on the reference only.

    Platt scaling: a one-feature logistic regression. It turns a model that reports a
    measurement and a yes/no call into one that also reports a probability, which
    confidence-based performance estimation needs. The returned function maps a table
    with the ``score`` column to probabilities.
    """
    frame = reference[[score, label]].dropna()
    y = frame[label].astype(bool).to_numpy()
    if y.all() or not y.any():
        raise ScenarioError("calibration needs both positives and negatives in the reference")
    model = LogisticRegression(C=1e6, max_iter=1000)
    model.fit(frame[[score]].to_numpy(dtype=float), y)

    def predict(table: pd.DataFrame) -> np.ndarray:
        out = np.full(len(table), np.nan)
        ok = table[score].notna().to_numpy()
        out[ok] = model.predict_proba(table.loc[ok, [score]].to_numpy(dtype=float))[:, 1]
        return out

    return predict


def write_scenario(scenario: Scenario, out_dir: str | Path) -> Path:
    """Write the scenario as SlicePoint input folders plus a manifest.

    Layout::

        reference/{features,predictions,labels,embeddings}.parquet
        sites/<site_id>/<YYYY-MM>/{features,predictions,labels,embeddings}.parquet
        cases.parquet         every site case, without embeddings, with site, month and date
        provenance.parquet    case_id -> pool, source case and source scan
        scenario.json         the recipe, counts, and sha256 of every file
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    columns = _table_columns(scenario)
    files: list[Path] = []
    files += _write_folder(scenario.reference, out / "reference", columns)
    for (site, month), batch in scenario.batches.groupby(["site_id", "month"], sort=True):
        files += _write_folder(batch, out / "sites" / str(site) / str(month), columns)
    emb = set(columns["embeddings"]) - {"case_id"}
    cases = scenario.batches[[c for c in scenario.batches.columns if c not in emb and c != "source_case_id"]]
    cases.to_parquet(out / "cases.parquet", index=False)
    scenario.provenance.to_parquet(out / "provenance.parquet", index=False)
    files += [out / "cases.parquet", out / "provenance.parquet"]
    manifest = {
        "recipe": scenario.recipe.to_dict(),
        "simulated": True,
        "counts": scenario.counts().to_dict(orient="records"),
        "files": [
            {"path": str(p.relative_to(out)), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in sorted(files)
        ],
    }
    (out / "scenario.json").write_text(json.dumps(manifest, indent=1, default=str), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------------------


def _check_recipe(recipe: ScenarioRecipe, pools: Mapping[str, pd.DataFrame]) -> None:
    names = {recipe.reference.pool, *recipe.reserved_pools, *(p.pool for s in recipe.sites for p in s.phases)}
    missing = sorted(n for n in names if n not in pools)
    if missing:
        raise ScenarioError(f"recipe names pools that were not given: {missing}")
    for name, pool in pools.items():
        for col in ("case_id", SOURCE_ID, recipe.label_column):
            if col not in pool.columns:
                raise ScenarioError(f"pool {name!r} has no {col!r} column")
    for site in recipe.sites:
        for p in site.phases:
            if not 1 <= p.first_month <= p.last_month <= recipe.n_months:
                raise ScenarioError(
                    f"site {site.site_id}: months {p.first_month}-{p.last_month} outside 1-{recipe.n_months}"
                )
            if p.per_month < 1:
                raise ScenarioError(f"site {site.site_id}: per_month must be at least 1")
            if p.positive_share is not None and not 0 <= p.positive_share <= 1:
                raise ScenarioError(f"site {site.site_id}: positive_share must be between 0 and 1")


def _draw(
    pool: pd.DataFrame,
    n: int,
    share: float | None,
    weights: Mapping[str, Mapping[str, float]],
    group_by: str | None,
    label: str,
    rng: np.random.Generator,
    what: str,
) -> pd.DataFrame:
    w = _weights(pool, weights)
    if group_by is not None:
        return _draw_groups(pool, n, group_by, w, rng, what)
    if share is None:
        return _weighted(pool, w, n, rng, what, len(pool))
    pos = pool[label].astype(bool).to_numpy()
    k = int(round(n * share))
    a = _weighted(pool[pos], w[pos], k, rng, what + ", positives", int(pos.sum()))
    b = _weighted(pool[~pos], w[~pos], n - k, rng, what + ", negatives", int((~pos).sum()))
    return pd.concat([a, b])


def _weighted(
    frame: pd.DataFrame, w: np.ndarray, n: int, rng: np.random.Generator, what: str, have: int
) -> pd.DataFrame:
    if n == 0:
        return cast(pd.DataFrame, frame.iloc[0:0])
    if have < n:
        raise ScenarioError(f"{what}: needs {n} cases, the pool has {have} left")
    p = w / w.sum()
    idx = rng.choice(len(frame), size=n, replace=False, p=p)
    return cast(pd.DataFrame, frame.iloc[np.sort(idx)])


def _draw_groups(
    pool: pd.DataFrame, n: int, group_by: str, w: np.ndarray, rng: np.random.Generator, what: str
) -> pd.DataFrame:
    """Whole groups, in weighted random order, until at least ``n`` cases (a group is never split).

    A group's weight is the largest weight among its cases."""
    gw = pd.Series(w, index=pool.index).groupby(pool[group_by]).max()
    gw = gw[gw > 0]
    groups = gw.index.to_numpy()[rng.choice(len(gw), size=len(gw), replace=False, p=(gw / gw.sum()).to_numpy())]
    chosen: list[Any] = []
    count = 0
    sizes = pool.groupby(group_by).size()
    for g in groups:
        if count >= n:
            break
        chosen.append(g)
        count += int(sizes[g])
    if count < n:
        raise ScenarioError(f"{what}: needs {n} cases, the pool has {count} left")
    return cast(pd.DataFrame, pool[pool[group_by].isin(chosen)])


def _weights(pool: pd.DataFrame, weights: Mapping[str, Mapping[str, float]]) -> np.ndarray:
    w = np.ones(len(pool))
    for column, by_value in weights.items():
        if column not in pool.columns:
            raise ScenarioError(f"weights name a column the pool does not have: {column!r}")
        w *= pool[column].astype(str).map(lambda v, m=by_value: float(m.get(v, 1.0))).to_numpy()
    if (w < 0).any():
        raise ScenarioError("weights must not be negative")
    return w


def _spread(
    drawn: pd.DataFrame, span: int, per_month: int, group_by: str | None, rng: np.random.Generator
) -> list[int]:
    """Give each drawn case a month offset (0-based), about ``per_month`` cases a month.

    Without groups, the cases are shuffled and dealt out evenly, so a phase's positives
    fall at random across its months instead of being rounded month by month. With
    groups, a group's cases go to consecutive months from a random start (a person
    scanned again and again)."""
    if group_by is None:
        order = rng.permutation(len(drawn))
        out = np.empty(len(drawn), dtype=int)
        out[order] = np.arange(len(drawn)) * span // max(len(drawn), 1)
        return cast(list[int], out.tolist())
    out = np.zeros(len(drawn), dtype=int)
    pos = {k: i for i, k in enumerate(drawn.index)}
    for _, g in drawn.groupby(group_by, sort=False):
        start = int(rng.integers(0, max(span - len(g) + 1, 1)))
        for j, idx in enumerate(g.index):
            out[pos[idx]] = min(start + j, span - 1)
    return cast(list[int], out.tolist())


def _scan_dates(months: pd.Series, rng: np.random.Generator) -> pd.Series:
    out = []
    for m in months:
        period = pd.Period(m, freq="M")
        out.append((period.start_time + pd.Timedelta(days=int(rng.integers(0, period.days_in_month)))).date())
    return cast("pd.Series[Any]", pd.Series(out, index=months.index, dtype="object"))


def _renumber(frame: pd.DataFrame, recipe: ScenarioRecipe, role: str) -> pd.DataFrame:
    frame = frame.rename(columns={"case_id": "source_case_id"}).copy()

    def new_id(row: pd.Series) -> str:
        key = f"{recipe.name}|{recipe.seed}|{role}|{row['pool']}|{row['source_case_id']}"
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    frame.insert(0, "case_id", frame.apply(new_id, axis=1) if len(frame) else pd.Series(dtype=str))
    if frame["case_id"].duplicated().any():
        raise ScenarioError(f"{role}: the same source case was drawn twice")
    return cast(pd.DataFrame, frame.reset_index(drop=True))


def _table_columns(scenario: Scenario) -> dict[str, list[str]]:
    cols = list(scenario.reference.columns)
    emb = [c for c in cols if c.startswith("emb_")]
    pred = [c for c in cols if c.startswith("pred_")]
    lab = [c for c in ("max_aortic_diameter_mm", "aaa", "aaa_size_class", scenario.recipe.label_column) if c in cols]
    lab = list(dict.fromkeys(lab))
    skip = {"case_id", SOURCE_ID, "source_case_id", "pool", "site_id", "month", "scan_date", *emb, *pred, *lab}
    feat = [c for c in cols if c not in skip]
    return {
        "features": ["case_id", *feat],
        "predictions": ["case_id", *pred],
        "labels": ["case_id", *lab],
        "embeddings": ["case_id", *emb],
    }


def _write_folder(frame: pd.DataFrame, folder: Path, columns: Mapping[str, list[str]]) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for table, cols in columns.items():
        if len(cols) <= 1:
            continue
        path = folder / f"{table}.parquet"
        frame[cols].to_parquet(path, index=False)
        written.append(path)
    return written
