# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

RISSK generates a **Unit Risk Score (URS, 0–100)** for each interview in a [Survey Solutions](https://mysurvey.solutions/) export, flagging interviews likely to contain unwanted interviewer behaviour. It is a Kedro pipeline that ingests Survey Solutions ZIP exports (Main Survey Data + Paradata), engineers features from microdata and paradata, and runs anomaly-detection models to produce the score. See `README.md` for the methodology and `FEATURES_SCORES.md` for per-feature definitions.

## Repository layout — two-package uv workspace

This repo is a single `uv` workspace (`[tool.uv.workspace]` in `pyproject.toml`) with **two installed packages**:

- **`rissk/`** (package `rissk`) — the pure ML / data-processing logic. No Kedro dependency. All modules carry a `*_kedro.py` suffix: `feature_processing_kedro.py`, `item_processing_kedro.py`, `unit_processing_kedro.py`, `detection_algorithms_kedro.py`, plus `utils/` (`import_utils_kedro.py`, `file_process_utils_kedro.py`, `stats_utils_kedro.py`). **This is where the real work lives** — feature engineering, scoring functions, and Survey Solutions import logic.
- **`rissk_kedro/`** (package `rissk-pipeline`, imported as `rissk_kedro`) — the Kedro project: pipeline/node definitions, Data Catalog, configuration, and the GUI.

**Nodes are thin wrappers.** Each `rissk_kedro/.../pipelines/*/nodes.py` imports functions from `rissk.*_kedro` and adapts them to Kedro's input/output contract (logging, empty-data guards). When changing pipeline *behaviour*, edit the function in `rissk/`; edit the node only for wiring, logging, or guards.

> The `_kedro` suffix is historical. A legacy Ploomber/OOP pipeline once lived alongside these files and was removed (see git history); the `*_kedro.py` files are now the **only** implementation. `Kedro_vs_Legacy_Changelog.md` documents every intentional behavioural difference and bug fix versus that removed legacy code — consult it before "fixing" something that looks deliberately changed (e.g. NaN-vs-0 score init, GPS unit conversions, sentinel filtering).

## Commands

All dependency and run commands assume the repo root unless noted.

```bash
# Install (root) — both workspace packages + GUI & viz extras.
# --all-packages is REQUIRED: plain `uv sync` prunes the rissk_kedro member (kedro, pyarrow, …).
uv sync --all-packages --extra gui --extra viz

# Launch the GUI (NiceGUI, opens http://localhost:8080)
bash rissk_kedro/run_gui.sh            # macOS / Linux
rissk_kedro\run_gui.bat                # Windows

# Run the pipeline from the CLI (must cd into the Kedro project)
cd rissk_kedro
kedro run                              # full pipeline (__default__)
kedro run --pipeline data_ingestion    # individual stages:
kedro run --pipeline feature_creation  #   data_ingestion → feature_creation → rissk_scoring
kedro run --pipeline rissk_scoring

# Visualise the pipeline DAG
cd rissk_kedro && kedro viz

# Explore scores interactively (marimo, opens a browser) — needs the viz extra
uv run marimo edit notebooks/viz/unit_scores.py   # also: feature_scores.py, interview_scores.py

# Tests (pytest is configured via the `test` extra; testpaths = rissk_kedro/tests)
uv sync --extra test
cd rissk_kedro && pytest               # run all
cd rissk_kedro && pytest tests/path/to/test_x.py::test_name   # single test
```

`kedro` commands **only work from inside `rissk_kedro/`** (that is where `pyproject.toml` declares `[tool.kedro]`). conda users: the equivalent install is `conda env create -f environment.yml && conda activate rissk_kedro`.

## Architecture

### Pipeline graph and data stages

Three pipelines run in sequence (registered in `rissk_kedro/src/rissk_kedro/pipeline_registry.py`):

```
data_ingestion → feature_creation → rissk_scoring
```

Data flows **only** through the Kedro Data Catalog (`conf/base/catalog.yml`) — nodes are pure functions with no shared state; no node does its own `to_csv`/`open`. Datasets land in numbered stage folders under each questionnaire:

```
<data_root>/<questionnaire.name>/latest/
    10_RAW/        # input ZIPs + extracted folders (PartitionedDataset of PathDataset)
    20_INTERIM/    # raw paradata/microdata, base feature tables
    30_PROCESSED/  # processed paradata/microdata, final feature tables
    40_SCORED/     # item_scores.parquet, unit_rissk_scores.csv  ← final output
```

`rissk_kedro/src/rissk_kedro/datasets/path.py` defines `PathDataset`, a custom dataset that returns a `Path` (used so the unzip node can operate on files/dirs directly rather than loading content).

### Configuration system

Kedro `OmegaConfigLoader` with two envs (`conf/base/` defaults, `conf/local/` overrides). `conf/local/` is **git-ignored** and is what the GUI and the driver notebook write — never commit run-specific config to `base/`.

- **`globals.yml`** — `data_root` (default `"data"`, relative to `rissk_kedro/`, or an absolute path), and the active `questionnaire` (`name`, `VERSION` list, optional `filter_var`). Catalog paths interpolate these via `${globals:...}`.
- **`parameters.yml`** — the `features:` map (each feature has a `use:` flag and optional `parameters.contamination`), and `zip_password`. `contamination: auto` lets the model estimate the outlier threshold; a float (0.01–0.50) fixes it.

**One questionnaire per run.** The active questionnaire is selected by config, not by code. To process a different questionnaire, change `questionnaire.name` in `globals.yml` (or use the GUI) and re-run — all catalog paths resolve to that questionnaire's folder. `VERSION: []` means "all versions found in the folder".

### Feature and score dispatch

Feature creation dispatches through explicit dicts, **not** dynamic `getattr`: `ITEM_FEATURE_MAP` and `UNIT_FEATURE_MAP` in `rissk/feature_processing_kedro.py` map a feature key → a `feat_*` function. A key with no map entry is silently skipped — several features in `parameters.yml` are computed-but-not-scored or feature-only (see `Kedro_vs_Legacy_Changelog.md` §7). Item scores initialise to `np.nan` ("not evaluated"), resolved to `0` only at unit-level aggregation.

Scoring (per `README.md` "Process description"): item/unit features → Type 1/2/3 scores → aggregated via **PCA** (responsible-level) + **Isolation Forest** (unit-level), combined by normalised product, **winsorized**, and rescaled to 0–100.

### GUI

`rissk_kedro/app/main.py` is a NiceGUI app. It reads `conf/base` + `conf/local`, writes user choices to **`conf/local/globals.yml` and `conf/local/parameters.yml`**, and runs the pipeline by shelling out to `python -m kedro run` (`asyncio.create_subprocess_exec`), streaming logs to the browser. `conf/local/` is written by the GUI and by the headless driver `rissk_kedro/src/rissk_kedro/driver.py`. The driver is the scheduled-run path: `notebooks/rissk_readme.ipynb` is a thin shell that sets one `CONFIG_FILE` parameter and calls `rissk_kedro.driver.run(CONFIG_FILE)`. `run` reads a per-survey run-config YAML (`notebooks/configs/*.yaml` — survey, questionnaires, which pipelines, S3/sync flags), and for each questionnaire writes `conf/local` (same keys/structure as the GUI) and executes the selected pipelines **in-process via `KedroSession`** (no CLI). Unlike the GUI, which shells out to `python -m kedro run`, the notebook never invokes the CLI.

Interactive result-exploration notebooks live in `notebooks/viz/` (three [marimo](https://marimo.io) apps: `feature_scores.py`, `unit_scores.py`, `interview_scores.py`), backed by the read-only loaders in `rissk_kedro/src/rissk_kedro/viz.py`. They read the pipeline's output files only and import nothing from `rissk/` — the right home for *new* viz/helper code (treat `rissk/` as legacy-origin: read its outputs, don't extend it).

## Project conventions

- **No Kedro hooks.** `HOOKS = ()` in `settings.py` and it stays that way — hooks caused confusion and data-lineage issues. Keep all logic in nodes (and in the `rissk/` functions they call).
- **Static pipelines only.** Do not generate pipelines or nodes dynamically. The pipeline is a fixed DAG; multi-questionnaire handling is done by re-running with different config, not by programmatic node generation.
- **Graceful empty/missing data.** Ingestion and scoring nodes detect missing exports, header-only files, and corrupt files, logging `WARNING`/`ERROR` and returning empty DataFrames rather than crashing (see changelog §3.1). Preserve this when editing nodes.
- **Behavioural changes go in the changelog.** Any intentional divergence in feature/score values should be recorded in `Kedro_vs_Legacy_Changelog.md` with rationale.
