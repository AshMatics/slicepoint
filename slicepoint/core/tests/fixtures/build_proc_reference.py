"""Rebuild ``proc_reference.json``: R ``pROC`` results for fixed synthetic datasets.

Needs an OpenCPU server with ``pROC`` installed; nothing else in the test suite does.
The server only ever receives the seeded synthetic labels and scores built below.

    OPENCPU_URL=https://host/ocpu python tests/fixtures/build_proc_reference.py

The server may refuse R code in arguments, so every argument is JSON or the session key
of an earlier call. ``direction`` cannot be sent that way; pROC's automatic choice is
recorded and must be ``<`` (higher score means positive) for the fixture to be valid.
"""

from __future__ import annotations

import json
import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

OUT = Path(__file__).with_name("proc_reference.json")


def _simulate(seed: int, n: int, prevalence: float, shift: float, round_to: int | None) -> dict[str, list[float]]:
    """Labels and two correlated scores. Score ``a`` matches ``reference_cases`` in test_roc.py."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < prevalence
    a = rng.normal(0.0, 1.0, n) + shift * y
    b = 0.6 * a + rng.normal(0.0, 1.0, n) + 0.3 * shift * y
    if round_to is not None:
        a, b = np.round(a, round_to), np.round(b, round_to)
    return {"y": [int(v) for v in y], "score_a": [float(v) for v in a], "score_b": [float(v) for v in b]}


DATASETS = {
    "continuous": _simulate(1, 200, 0.3, 1.2, None),
    "ties": _simulate(2, 150, 0.4, 0.8, 0),
    "small": _simulate(3, 24, 0.5, 1.5, None),
}
UNPAIRED = [("continuous", "ties"), ("small", "continuous"), ("ties", "small")]


class OpenCpu:
    def __init__(self, base: str) -> None:
        self.base = base.rstrip("/")

    def _post(self, path: str, data: bytes, content_type: str) -> tuple[str, str]:
        req = urllib.request.Request(f"{self.base}{path}", data=data, headers={"Content-Type": content_type})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.headers.get("X-ocpu-session", ""), resp.read().decode()

    def roc(self, y: list[float], score: list[float]) -> str:
        """Fit a ROC curve and return its session key."""
        body = json.dumps({"response": y, "predictor": score, "levels": [0, 1], "quiet": True}).encode()
        key, _ = self._post("/library/pROC/R/roc", body, "application/json")
        if not re.fullmatch(r"x0[0-9a-f]+", key):
            raise RuntimeError(f"no session key from roc(): {key!r}")
        return key

    def call(self, fn: str, **args: str) -> Any:
        """Call a pROC function whose arguments are session keys or JSON literals."""
        data = urllib.parse.urlencode(args).encode()
        _, text = self._post(f"/library/pROC/R/{fn}/json?digits=15&force=true", data, "application/x-www-form-urlencoded")
        return json.loads(text)

    def text(self, path: str) -> str:
        with urllib.request.urlopen(f"{self.base}{path}", timeout=60) as resp:
            return str(resp.read().decode())


def _test(result: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {
        "method": result["method"][0],
        "auc_a": result["estimate"][0],
        "auc_b": result["estimate"][1],
        "statistic": result["statistic"][0],
        "p_value": result["p.value"][0],
        "direction_a": result["roc1"]["direction"][0],
        "direction_b": result["roc2"]["direction"][0],
    }
    if "conf.int" in result:
        out["conf_int"] = result["conf.int"]
    if "parameter" in result:
        out["df"] = result["parameter"][0]
    return out


def main() -> None:
    ocpu = OpenCpu(os.environ["OPENCPU_URL"])
    info = ocpu.text("/library/pROC/info")
    version = re.search(r"^Version:\s+(\S+)", info, re.M)
    built = re.search(r"^Built:\s+(R [\d.]+)", info, re.M)
    keys = {(name, s): ocpu.roc(d["y"], d[s]) for name, d in DATASETS.items() for s in ("score_a", "score_b")}

    single = {}
    for (name, s), key in keys.items():
        lower, auc, upper = ocpu.call("ci.auc", roc=key, method='"delong"')
        single[f"{name}/{s}"] = {"auc": auc, "lower": lower, "upper": upper, "variance": ocpu.call("var", roc=key, method='"delong"')[0]}

    paired = {
        name: _test(ocpu.call("roc.test", roc1=keys[name, "score_a"], roc2=keys[name, "score_b"], method='"delong"', paired="true"))
        for name in DATASETS
    }
    unpaired = {
        f"{a}|{b}": _test(ocpu.call("roc.test", roc1=keys[a, "score_a"], roc2=keys[b, "score_a"], method='"delong"', paired="false"))
        for a, b in UNPAIRED
    }
    directions = {t["direction_a"] for t in [*paired.values(), *unpaired.values()]} | {t["direction_b"] for t in [*paired.values(), *unpaired.values()]}
    if directions != {"<"}:
        raise RuntimeError(f"pROC chose direction {directions}; the fixture needs '<' throughout")

    fixture = {
        "source": {"package": "pROC", "version": version.group(1) if version else None, "r": built.group(1) if built else None, "conf_level": 0.95},
        "datasets": DATASETS,
        "auc_delong": single,
        "paired": paired,
        "unpaired": unpaired,
    }
    OUT.write_text(json.dumps(fixture, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
