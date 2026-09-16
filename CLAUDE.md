# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Capstone project forecasting **German power grid load & stability** from SMARD data
(Bundesnetzagentur).
Team project (neuefische bootcamp) with 4 members, work happens on `feature/*` branches
off `main`.

- We use data for installed power plants, their power generation & Germany's power consumption to predict the power Germany needs in it's grid for 1 Day ahead.
- We want to create our own model with a time-series analysis to spot days with risk of intervention measures by the grid operators (=TSOs).
- We want to create a risk flag for this by using `residual load` as a target variable
- Extreme cases of `residual load` are our risk cases
  - high residual load (imports, tight margins)
  - egative residual load (renewable oversupply, negative prices, downward redispatch)
- Since we have public comparison values for `predicted residual load` by SMARD.de we can use this to validate our own models.

## Environment & commands

`uv` manages the environment — never call `pip install`. `uv run` re-syncs the venv against
`uv.lock` before running, so prefer it over activating `.venv`.

```bash
make setup      # uv sync --all-groups  (runtime + dev deps, downloads Python >=3.11 if needed)
make lab        # uv run jupyter lab
make test       # uv run pytest
make format     # uv run black .        (black is the formatter; no linter configured)
make mlflow     # uv run mlflow ui      -> http://127.0.0.1:5000
make train      # uv run python -m modeling.train
make predict    # uv run python -m modeling.predict models/linear data/X_test.csv data/y_test.csv
```

Run a single test: `uv run pytest path/to/test_file.py::test_name`.
There is no test suite yet — `testbook` is available for testing notebook cells.

Dependency changes go through uv so `pyproject.toml` and `uv.lock` stay in sync:
`uv add <pkg>`, `uv add --group dev <pkg>`, `uv remove <pkg>`, `uv lock --upgrade`.
Runtime deps are `[project.dependencies]`; notebook/tooling deps live in the `dev` group.
See [UV_SETUP.md](UV_SETUP.md) for the long form.

## Repository layout and what is real vs. template

This repo started from the neuefische `ds-modeling-template`, and **most of `modeling/` is still
that template**:

- `modeling/train.py`, `modeling/predict.py`, `modeling/feature_engineering.py` operate on the
  **coffee quality dataset**, not on grid data. Treat them as reference scaffolding for the
  MLflow logging pattern, not as project code. Same for `notebooks/EDA-and-modeling.ipynb` and
  the MLflow sections of [README.md](README.md).
- `modeling/config.py` is real infrastructure: it reads the MLflow tracking URI from a local
  `.mlflow_uri` file, falling back to the `MLFLOW_URI` env var (both gitignored/untracked).
  `EXPERIMENT_NAME` is still the template default `0-template-ds-modeling`.
- `__init__.py` at the repo root is an intentional placeholder — the eventual home for
  productionized code extracted out of the notebooks.

Actual project work lives in `notebooks/`.

## Data pipeline

**`data/` is gitignored** (`data/*.csv`, `models/*`), so `data/smard.csv` does not come with a
clone — regenerate it by running [notebooks/API-connection.ipynb](notebooks/API-connection.ipynb)
top to bottom.

That notebook is the single source of the dataset:

- SMARD API (`https://www.smard.de/app/chart_data/...`), no API key. Docs: <https://smard.api.bund.dev/>
- The API serves only **whole weekly packages** — `fetch()` lists available packages from the
  index endpoint, downloads the overlapping ones, and trims to the requested range.
- Series are selected by numeric filter ID in the `FILTERS` dict (grid load, residual load,
  wind on/offshore, solar, and the corresponding SMARD forecasts). Region `DE`, hourly
  resolution, timestamps converted to `Europe/Berlin`.
- Written in **German Excel CSV format**: `sep=";"`, `decimal=","`, `utf-8-sig`. Every consumer
  must therefore read with `pd.read_csv(..., delimiter=";")` and convert the numeric columns
  (`str.replace(",", ".")` → `float`), as [notebooks/EDA-robert.ipynb](notebooks/EDA-robert.ipynb)
  does in its first cells.

Known data characteristics established in EDA: hourly gaps at every **spring DST switch**
(02:00).

## Notebook conventions

All notebooks will be created in `notebooks/` during the development phase.

We are currently working on fetching the data and adding our own EDA.

There are two naming tracks, and which one applies depends on why the notebook exists:

- **Per-member exploration** — `EDA-<name>.ipynb`, e.g. `notebooks/EDA-robert.ipynb`. Personal
  scratch work; the name keeps everyone out of everyone else's file, so merge conflicts do not
  arise in the first place.
- **Spec-driven consolidated notebooks** — named after the spec that defines them, e.g.
  `notebooks/EDA-simple.ipynb` for [specs/01-Simple-EDA.md](specs/01-Simple-EDA.md) and
  `notebooks/EDA-deep.ipynb` for [specs/02-Deep-EDA.md](specs/02-Deep-EDA.md). The filename is
  fixed by the spec regardless of who runs it. These are shared files, so conflicts are real and
  get resolved with `nbdime` (see below) rather than avoided by naming.

`notebooks/EDA-robert.ipynb` defines the plotting/aggregation helpers the EDA relies on; reuse
them rather than re-deriving:

- `period_mean(series, freq)` — calendar-period mean (`"W"`, `"M"`) that **drops the incomplete
  first and last period**, so plot edges are not partial-period artefacts. The data starts
  mid-week and ends mid-month, so a plain `.resample()` produces fake dips.
- `style_timeseries(ax, title, ylabel)` — the shared chart style (no top/right spines, y-grid
  only, year major ticks with quarterly minors, thousands-separated y labels).
- `seasonal_plot(df, y_value, title, ylabel)` — month-on-x, year-as-hue seaborn line plot;
  expects a frame with explicit `year` and `month` columns.

Matplotlib is used directly for the styled plots; seaborn for the seasonal/hue plots.

## Notebook merge conflicts

`nbdime` is in the dev group. Enable it once per clone with `nbdime config-git --enable`, then
resolve conflicts with `nbdime mergetool` instead of editing `.ipynb` JSON by hand.
