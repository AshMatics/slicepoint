<h1 align="center"><img src=".github/assets/slicepoint-logo.png" alt="SlicePoint: Slice. Spot. Decide." width="600"></h1>

[![Release](https://img.shields.io/github/v/tag/AshMatics/slicepoint?label=release&sort=semver)](CHANGELOG.md)
[![CI](https://github.com/AshMatics/slicepoint/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/AshMatics/slicepoint/actions/workflows/ci.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Dependabot](https://img.shields.io/badge/dependabot-enabled-025E8C?logo=dependabot)](.github/dependabot.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)](CHANGELOG.md#known-issues)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dataset: SlicePoint AAA CT](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-slicepoint--aaa--ct-yellow)](https://huggingface.co/datasets/slicepoint/slicepoint-aaa-ct)
[![Dataset: SlicePoint Hospitals](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-slicepoint--hospitals-yellow)](https://huggingface.co/datasets/slicepoint/slicepoint-hospitals)

SlicePoint checks whether an imaging AI model fits a hospital's patients and scanners
before it is switched on. It compares a sample of the site's scans with a reference (the
data the model was validated on) and returns a **GO / BORDERLINE / NO-GO** verdict, five
health checks with their evidence, AI performance by group, cases to review and, when the
vendor publishes one, a comparison with the vendor's training-data datasheet. The first
model is abdominal aortic aneurysm (AAA) detection on CT; the checks are not specific to AAA.

SlicePoint began as the runner-up project at the 2026 Nucleate Pittsburgh hackathon and is
being rebuilt here as open source by Asher Informatics PBC with the student team.

**Version 0.2.0** (last updated 2026-11-24); unreleased changes since 0.2.0 are listed below and
in [CHANGELOG.md](CHANGELOG.md).

**Start here:** [slicepoint/README.md](slicepoint/README.md) has the quick start (install,
run one assessment, open the report), the package map and how to run the service with
its web UI.

## What is in this repository

| Path | What |
|---|---|
| [`slicepoint/schemas/`](slicepoint/schemas) | `slicepoint-schemas`: the data contract (Pydantic models, JSON Schemas) |
| [`slicepoint/core/`](slicepoint/core) | `slicepoint-core`: the analysis library, the `slicepoint` command line, the HTML report |
| [`slicepoint/api/`](slicepoint/api) | `slicepoint-api`: the FastAPI service, with the web UI bundled |
| [`slicepoint/viewer/`](slicepoint/viewer) | The React result viewer (a single offline file, and the live UI for the API) |
| [`slicepoint/segmentation/`](slicepoint/segmentation) | `slicepoint-model`: the aorta segmentation worker (GPU, TotalSegmentator) |
| [`slicepoint/docs/`](slicepoint/docs) | Architecture decisions, the data contract, a hospital guide, references |
| [`slicepoint/datasets/`](slicepoint/datasets) | The source of the Hugging Face dataset cards |
| [`slicepoint/notebooks/`](slicepoint/notebooks) | The quick-start notebook |
| [`slicepoint/python_notebooks/`](slicepoint/python_notebooks) | The student team's original proof of concept, kept as written |
| [`data/`](data) | Where sample data goes (not in git); [data/README.md](data/README.md) says how to download it |
| [`nucleate-hack-2026/`](nucleate-hack-2026) | The hackathon material: brief, documents, starter and federated harness |

## New since 0.2.0 (unreleased)

Three statistics modules in `slicepoint-core`. Each is a set of pure functions with no new
dependencies. None is used yet by the assessment, the command line or the API.

| Module | What it does |
|---|---|
| `slicepoint_core.planning` | Study planning before a model is run: expected confidence-interval width and sample size for AUC, sensitivity, specificity or any rate; stratified allocation of cases |
| `slicepoint_core.roc` | AUC with DeLong intervals, DeLong tests (two models on the same cases, or two separate groups), and bootstrap intervals that resample people rather than images |
| `slicepoint_core.fairness` | Performance by subgroup and the gaps between subgroups, with small groups flagged and intervals adjusted for multiple comparisons |

### Checking against R

The statistics in `planning` and `roc` reproduce the standard R packages:
[presize](https://cran.r-project.org/package=presize) for planning and
[pROC](https://cran.r-project.org/package=pROC) 1.19.1 for ROC. The R results are stored in
the tests (`slicepoint/core/tests/test_planning.py` and
`slicepoint/core/tests/fixtures/proc_reference.json`), so the tests and CI need no R.

`slicepoint/core/tests/fixtures/build_proc_reference.py` rebuilds the pROC fixture from any
[OpenCPU](https://www.opencpu.org/) server that has pROC installed (set `OPENCPU_URL`). It sends
only seeded synthetic data. Asher Informatics maintainers also cross-check new cases against an
internal OpenCPU service that wraps presize, pROC and kappaSize, before adding them as fixtures. For
details on that service, contact Asher Informatics.

## Datasets

Two public datasets on Hugging Face, both CC BY 4.0 and derived from the TotalSegmentator
dataset (University Hospital Basel). [DATASETS.md](DATASETS.md) describes them in full.

| Dataset | What | Size |
|---|---|---|
| [slicepoint/slicepoint-aaa-ct](https://huggingface.co/datasets/slicepoint/slicepoint-aaa-ct) | 567 abdominal CT scans with ground-truth masks, aortic measurements, AAA labels, a baseline model's predictions and embeddings; cohorts for a vendor/site demo; vendor datasheets | ~17 GB (tables ~20 MB) |
| [slicepoint/slicepoint-hospitals](https://huggingface.co/datasets/slicepoint/slicepoint-hospitals) | The seven simulated hospitals from the hackathon, with planted problems and the answer keys; used by the tests | ~14 GB (tables ~10 MB) |

## License and contributors

The code is released under the [MIT License](LICENSE). The datasets keep their own
license (CC BY 4.0, from TotalSegmentator); see [DATASETS.md](DATASETS.md).

SlicePoint started as a student project. Under the hackathon terms each contributor owns
their work and keeps the copyright in it. Every contributor below has confirmed in writing
(by email, November 2026) that their contributions are licensed under the MIT License, which
closes ADR 0001, open question 1. Each contributor's copyright notice is covered by the
"SlicePoint contributors" line in [LICENSE](LICENSE).

| Contributor | Contributed | MIT affirmed |
|---|---|---|
| Yajushi Khurana (PhD student, Carnegie Mellon University and University of Pittsburgh) | The original proof of concept: notebooks and monitoring scripts in `slicepoint/python_notebooks/` | Yes (email, Nov 2026) |
| Umang Pandey (graduate student, Bioengineering, Carnegie Mellon University) | The aorta segmentation and comparison service in `slicepoint/segmentation/` | Yes (email, Nov 2026) |
| Manya Natakala (University of Pittsburgh) | The hackathon design; led the presentation and pitch | Yes (email, Nov 2026) |
| Evangeline Gulea (University of Pittsburgh) | The hackathon design; led the presentation and pitch | Yes (email, Nov 2026) |
| Asher Informatics PBC (John Kalafut, with AI assistance) | The rebuild: schemas, core, API, viewer, datasets, documentation | Yes (copyright holder) |

The team's hackathon pitch deck (team ScottyPanthers) is in
[`nucleate-hack-2026/docs/`](<nucleate-hack-2026/docs/ScottyPanthers — SlicePoint Medical AI Checker.pdf>).

**Not for clinical use.** SlicePoint supports a decision; it is not a medical device and
its outputs are not medical advice.
