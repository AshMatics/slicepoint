from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

from slicepoint_core.scenario import (
    ScenarioError,
    ScenarioRecipe,
    build_scenario,
    fit_calibration,
    load_pool,
    write_scenario,
)


def make_pool(n: int, prefix: str, positives: int, seed: int = 0, source_prefix: str | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    aaa = np.zeros(n, dtype=bool)
    aaa[n - positives :] = True
    frame = pd.DataFrame(
        {
            "case_id": [f"{prefix}{i:03d}" for i in range(n)],
            "patient_id": [f"p{prefix}{i:03d}" for i in range(n)],
            "sex": rng.choice(["F", "M"], n),
            "contrast_phase": rng.choice(["venous", "arterial"], n),
            "slice_thickness_mm": 1.5,
            "pred_max_diameter_mm": np.where(aaa, 40.0, 20.0) + rng.normal(0, 3, n),
            "pred_aaa": aaa,
            "max_aortic_diameter_mm": np.where(aaa, 42.0, 19.0),
            "aaa": aaa,
            "emb_0": rng.normal(size=n),
            "emb_1": rng.normal(size=n),
        }
    )
    frame["source_id"] = [f"{source_prefix or prefix}{i:03d}" for i in range(n)]
    return cast(pd.DataFrame, frame)


def recipe(**over: Any) -> ScenarioRecipe:
    raw: dict[str, Any] = {
        "name": "test",
        "seed": 7,
        "start_month": "2026-01",
        "n_months": 6,
        "reference": {"pool": "clean", "n": 40, "positive_share": 0.1},
        "reserved_pools": ["thick"],
        "sites": [
            {
                "site_id": "a",
                "label": "Site A",
                "phases": [{"first_month": 1, "last_month": 6, "pool": "clean", "per_month": 5}],
            },
            {
                "site_id": "b",
                "label": "Site B",
                "phases": [
                    {"first_month": 1, "last_month": 3, "pool": "clean", "per_month": 4, "positive_share": 0.25},
                    {"first_month": 4, "last_month": 6, "pool": "thick", "per_month": 4},
                ],
            },
        ],
    }
    raw.update(over)
    return ScenarioRecipe.from_dict(raw)


@pytest.fixture
def pools() -> dict[str, pd.DataFrame]:
    clean = make_pool(120, "c", positives=20)
    # the thick pool re-makes the first 30 clean scans: same source ids
    thick = make_pool(30, "t", positives=3, seed=1, source_prefix="c").assign(slice_thickness_mm=5.0)
    return {"clean": clean, "thick": thick}


def test_counts_months_and_reference(pools: dict[str, pd.DataFrame]) -> None:
    sc = build_scenario(recipe(), pools)
    assert len(sc.reference) == 40 and int(sc.reference.aaa.sum()) == 4
    counts = sc.counts().set_index(["site_id", "month"])
    assert counts.loc[("a", "2026-01"), "cases"] == 5
    assert set(sc.batches.month) == {f"2026-0{m}" for m in range(1, 7)}
    assert int(sc.batches[(sc.batches.site_id == "b") & (sc.batches.pool == "clean")].aaa.sum()) == 3


def test_no_scan_is_used_twice_and_reserved_pools_are_kept(pools: dict[str, pd.DataFrame]) -> None:
    sc = build_scenario(recipe(), pools)
    used = pd.concat([sc.reference.source_id, sc.batches.source_id])
    assert not used.duplicated().any()
    reserved = set(pools["thick"].source_id)
    assert not reserved & set(sc.reference.source_id)
    assert not reserved & set(sc.batches[sc.batches.pool == "clean"].source_id)
    assert (sc.batches[sc.batches.pool == "thick"].slice_thickness_mm == 5.0).all()


def test_same_seed_same_scenario(pools: dict[str, pd.DataFrame]) -> None:
    a, b = build_scenario(recipe(), pools), build_scenario(recipe(), pools)
    pd.testing.assert_frame_equal(a.batches, b.batches)
    c = build_scenario(recipe(seed=8), pools)
    assert set(c.batches.source_id) != set(a.batches.source_id)


def test_new_case_ids_hide_the_source(pools: dict[str, pd.DataFrame]) -> None:
    sc = build_scenario(recipe(), pools)
    assert not set(sc.batches.case_id) & set(pools["clean"].case_id)
    link = sc.provenance.set_index("case_id")
    row = sc.batches.iloc[0]
    assert link.loc[row.case_id, "source_case_id"] == row.source_case_id


def test_short_pool_fails_with_counts(pools: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ScenarioError, match="needs 60 cases"):
        build_scenario(
            recipe(
                reference={"pool": "clean", "n": 40},
                sites=[
                    {
                        "site_id": "a",
                        "label": "A",
                        "phases": [{"first_month": 1, "last_month": 6, "pool": "clean", "per_month": 10}],
                    }
                ],
            ),
            {"clean": pools["clean"].iloc[:90], "thick": pools["thick"]},
        )


def test_unknown_pool_and_bad_months_are_refused(pools: dict[str, pd.DataFrame]) -> None:
    with pytest.raises(ScenarioError, match="not given"):
        build_scenario(recipe(reference={"pool": "nope", "n": 5}), pools)
    with pytest.raises(ScenarioError, match="outside"):
        build_scenario(
            recipe(
                sites=[
                    {
                        "site_id": "a",
                        "label": "A",
                        "phases": [{"first_month": 5, "last_month": 9, "pool": "clean", "per_month": 1}],
                    }
                ]
            ),
            pools,
        )


def test_groups_are_drawn_whole(pools: dict[str, pd.DataFrame]) -> None:
    repeats = make_pool(20, "r", positives=4, seed=2, source_prefix="r")
    repeats["person"] = [f"x{i // 2}" for i in range(20)]
    sc = build_scenario(
        recipe(
            sites=[
                {
                    "site_id": "a",
                    "label": "A",
                    "phases": [
                        {
                            "first_month": 1,
                            "last_month": 3,
                            "pool": "rep",
                            "per_month": 2,
                            "group_by": "person",
                            "weights": {"aaa": {"True": 5.0}},
                        }
                    ],
                }
            ]
        ),
        {**pools, "rep": repeats},
    )
    sizes = sc.batches.groupby("person").size()
    assert (sizes == 2).all()
    first = sc.batches.sort_values("scan_date").groupby("person").month.apply(list)
    assert all(m[0] <= m[1] for m in first)


def test_calibration_is_monotone_and_needs_both_classes(pools: dict[str, pd.DataFrame]) -> None:
    proba = fit_calibration(pools["clean"])
    p = proba(pd.DataFrame({"pred_max_diameter_mm": [15.0, 25.0, 35.0, 45.0, np.nan]}))
    assert np.all(np.diff(p[:4]) > 0) and np.isnan(p[4])
    with pytest.raises(ScenarioError, match="both"):
        fit_calibration(pools["clean"].assign(aaa=False))


def test_write_and_reload(pools: dict[str, pd.DataFrame], tmp_path: Path) -> None:
    sc = build_scenario(recipe(), pools)
    out = write_scenario(sc, tmp_path / "s")
    folder = out / "sites" / "a" / "2026-01"
    for table in ("features", "predictions", "labels", "embeddings"):
        assert (folder / f"{table}.parquet").exists()
    assert "source_id" not in pd.read_parquet(folder / "features.parquet").columns
    back = load_pool(folder)
    assert len(back) == 5 and {"aaa", "emb_0", "pred_max_diameter_mm"} <= set(back.columns)
    manifest = json.loads((out / "scenario.json").read_text())
    assert manifest["simulated"] is True and manifest["recipe"]["name"] == "test"
    assert all(len(f["sha256"]) == 64 for f in manifest["files"])
