# RISSK visualisation notebooks (marimo)

Three interactive [marimo](https://marimo.io) notebooks for exploring a scored run.
They read **only** the pipeline outputs under
`<data_root>/<questionnaire>/latest/` (via `rissk_kedro.viz`) — run the pipeline
first (e.g. through `notebooks/rissk_readme.ipynb` or the GUI) so there is data to show.

| Notebook | Level | Shows |
|---|---|---|
| `feature_scores.py` | Feature (item) | Distribution of any `s__*` feature score + per-feature coverage |
| `unit_scores.py` | Unit (interview) | The 0–100 `unit_risk_score`: histogram, riskiest interviews, by interviewer/version, vs a feature |
| `interview_scores.py` | Single interview | One interview's URS, the feature scores that drove it, its riskiest items |

Each notebook opens with a **questionnaire dropdown** that scans the data root
(default `rissk_kedro/data/`, newest scored run preselected); all charts react to it.

## Install & launch

```bash
# one-time: install marimo + altair into the workspace env
uv sync --all-packages --extra viz

# edit interactively (opens a browser)
uv run marimo edit notebooks/viz/unit_scores.py

# or run read-only as an app
uv run marimo run notebooks/viz/interview_scores.py
```

> marimo notebooks are plain `.py` files (not `.ipynb`). Data loading lives in
> `rissk_kedro/src/rissk_kedro/viz.py`; the notebooks import nothing from `rissk/`.
