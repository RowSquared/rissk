# RISSK — Getting Started

RISSK uses machine learning to score interviews from Survey Solutions export files,
flagging individual interviews most likely to contain unwanted interviewer behaviour.

---

## Prerequisites

- **Python 3.13** installed on your machine
- An internet connection for the initial install
- Survey Solutions export files (Main Survey Data + Paradata ZIPs)

Verify your Python version:

```bash
python --version
```

---

## Option A — uv (recommended for new users)

[uv](https://docs.astral.sh/uv/) is a fast, self-contained Python package manager.
You do **not** need to manage virtual environments manually.

### 1. Install uv

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. Get the RISSK code

Clone with Git:
```bash
git clone https://github.com/rowsquared/rissk.git
cd rissk
```

Or download the ZIP from GitHub, unzip it, and navigate to the `rissk/` folder.

### 3. Install dependencies

```bash
uv sync --extra gui
```

### 4. Launch the GUI

From the `rissk_kedro/` directory:

**macOS / Linux:**
```bash
bash rissk_kedro/run_gui.sh
```

**Windows:**
```bat
rissk_kedro\run_gui.bat
```

Your browser will open automatically at **http://localhost:8080**.

---

## Option B — conda (for experienced users)

### 1. Get the code

```bash
git clone https://github.com/rowsquared/rissk.git
cd rissk
```

### 2. Create and activate the conda environment

```bash
conda env create -f environment.yml
conda activate rissk_kedro
```

This installs Python 3.13, all pipeline dependencies, and the RISSK package in one step.

### 3. Launch the GUI

From the `rissk_kedro/` directory:

```bash
bash rissk_kedro/run_gui.sh        # macOS / Linux
rissk_kedro\run_gui.bat            # Windows
```

---

## Using the GUI

### Step 1 — Data folder

Choose where RISSK will read and write survey data.

- **Default (`data`):** keeps everything inside the `rissk_kedro/` project folder.
- **Absolute path:** point to any folder on your machine, e.g. `/Users/jane/surveys`.

The GUI shows you the exact subfolder where ZIP files must be placed, e.g.:

```
/Users/jane/surveys/pmpmd_household/latest/10_RAW/
```

Click **Create folder & Open** to create that folder and open it in your file manager.

### Step 2 — Prepare your Survey Solutions exports

Export from Survey Solutions and place the **unmodified ZIP files** in the folder shown:

1. **Main Survey Data** — choose *Tab separated* or *Stata 14*, tick *Include meta information about questionnaire*.
2. **Paradata** — under *Data Type* select *Paradata*.

> Export both files from the **same questionnaire version** consecutively.
> For multiple compatible versions, export each separately and place all ZIPs in the same folder.

Do **not** rename, modify, or unzip the files.

### Step 3 — Questionnaire configuration

- **Questionnaire name:** the template name exactly as it appears in Survey Solutions (e.g. `pmpmd_household`). This is also used as the data folder name.
- **Versions:** comma-separated list of version numbers to process, e.g. `4, 5, 6`.
- **Consent filter (optional):** score only interviews where a specific paradata variable equals a required value (useful for surveys with a consent question).

To switch to a different questionnaire, update the name in the Setup tab and save — or edit `questionnaire.name` directly in `conf/local/globals.yml`.

### Step 4 — Save & Run

1. Click **Save configuration** on the Setup tab.
2. Switch to the **Run** tab.
3. Choose a pipeline stage (leave as *All* for a full run).
4. Click **Run RISSK** and monitor the live log.

Results are written to:
```
<data_root>/<questionnaire_name>/latest/40_SCORED/unit_rissk_scores.csv
```

### Advanced settings

Access the **Advanced** tab to:
- Set a ZIP password (if your exports are password-protected)
- Toggle automatic contamination estimation
- Enable/disable individual features and adjust contamination thresholds

---

## Running without the GUI (command line)

Experienced users can run Kedro directly from the `rissk_kedro/` directory:

```bash
cd rissk_kedro

# Full pipeline
kedro run

# Individual stages
kedro run --pipeline data_ingestion
kedro run --pipeline feature_creation
kedro run --pipeline rissk_scoring
```

Configuration overrides go in `rissk_kedro/conf/local/globals.yml` and `rissk_kedro/conf/local/parameters.yml`
(these files are ignored by git).

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError: nicegui` | Run `pip install "nicegui>=1.4"` in your active environment |
| Browser does not open | Open http://localhost:8080 manually |
| Pipeline fails with "No data found" | Check that ZIP files are in the correct subfolder (see Setup tab) |
| `kedro: command not found` | Activate your environment first (`conda activate rissk` or `source .venv/bin/activate`) |
| ZIPs not extracted | Make sure filenames are not modified; check the ZIP password setting if exports are protected |
