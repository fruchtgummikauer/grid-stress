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
  - negative residual load (renewable oversupply, negative prices, downward redispatch)
- Since we have public comparison values for `predicted residual load` by SMARD.de we can use this to validate our own models.

**Scope simplifications:**

- Energy generation features are limited to only Wind + Solar. We have published day-ahead forecasts for those only which makes up our baseline to beat. Solar and Wind are also the largest variable sources
- We are therefore excluding all other generation features (e.g. biomass, coal, water, ...) for the scope of this project

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

This repo started from the neuefische `ds-modeling-template`. **Most of `modeling/` is still that
template** — `train.py`, `predict.py` and `feature_engineering.py` operate on the **coffee quality
dataset**, not on grid data. Treat them as reference scaffolding for the MLflow logging pattern,
not as project code; the same goes for `notebooks/EDA-and-modeling.ipynb` and the MLflow sections
of [README.md](README.md), which documents that workflow in full.

Two pieces are real: `modeling/config.py` reads the MLflow tracking URI from a local `.mlflow_uri`
file, falling back to the `MLFLOW_URI` env var (both gitignored); `EXPERIMENT_NAME` is still the
template default `0-template-ds-modeling`. And `__init__.py` at the repo root is an intentional
placeholder — the eventual home for productionized code extracted out of the notebooks.

Actual project work lives in `notebooks/`:

```
notebooks/
  API-connection.ipynb          # the data pipeline — both raw datasets
  EDA-and-modeling.ipynb        # template leftover (coffee dataset) - ignore
  00_project_management/
    PM-Session1.ipynb           # roadmap, Miro/wiki links, domain terms - ignore, just for team documentation
  01_eda/
    EDA-hari.ipynb              # per-member exploration
    EDA-magc.ipynb              # per-member exploration
    EDA-rebap-magc.ipynb        # per-member exploration of reBAP
    EDA-robert.ipynb            # per-member exploration
    EDA-simple-claude.ipynb     # reference: spec 01 output (edited), not adopted
    team-EDA.ipynb              # adopted: spec 03 output
  02_*                          # reserved for spec 04 (forecast metrics) - not created yet
  03_risk_classification/
    risk-definition.ipynb       # adopted: spec 02 output
```

## Data pipeline

**`data/` is gitignored** (`data/*.csv`, `models/*`), so nothing in it comes with a clone.
[notebooks/API-connection.ipynb](notebooks/API-connection.ipynb) regenerates both raw datasets and
is the single source of each. The risk-label files are derived from `smard.csv` by a separate
notebook (see below).

### `data/smard.csv` — the core dataset (Part 1)

- SMARD API (`https://www.smard.de/app/chart_data/...`), no API key. Docs: <https://smard.api.bund.dev/>
- The API serves only **whole weekly packages** — `fetch()` lists available packages from the
  index endpoint, downloads the overlapping ones, and trims to the requested range.
- Eleven series selected by numeric filter ID in the `FILTERS` dict: grid load, residual load,
  wind on/offshore, solar, three SMARD day-ahead forecasts (`Forecast Wind + Solar`,
  `Forecast Grid Load`, `Forecast Residual Load` — the last is our benchmark to beat), and
  **installed capacity** for the same three sources (`Capacity Wind Offshore`, `Capacity Wind
  Onshore`, `Capacity Solar` → `cap_wind_off`, `cap_wind_on`, `cap_solar`). Region `DE`, hourly
  resolution, timestamps converted to `Europe/Berlin`.
- The capacity columns are a **yearly step value repeated on every hour** (one distinct value per
  year) — MW of installed capacity, not a measurement. They are in `SERIES`, so `.describe()` and
  `.corr()` over `SERIES` now include three near-constant columns.
- Written in **German Excel CSV format**: `sep=";"`, `decimal=","`, `utf-8-sig`. Every consumer
  must therefore read with `pd.read_csv(..., delimiter=";", encoding="utf-8-sig")` and convert the
  numeric columns (`str.replace(",", ".")` → `float`).

Current extent: **67,336 hourly rows, 2019-01-01 00:00 → 2026-09-06 23:00**. Treat this as a
snapshot, not a constant — the fetch has already been widened back to 2019 once (it previously
started at 2022-01-01), and the end moves with every re-fetch, so **no literal calendar year
belongs in notebook code**; derive year lists, anchors and colour maps from the loaded data (see
`YEARS` below). Re-fetching is not neutral for anything computed over the whole record: the 2019
backfill shifted whole-record quantiles of `residual_load`, which is why
[02-Risk-Definition.md](.claude/specs/02-Risk-Definition.md) refuses to define a label against
them.

Known data characteristic established in EDA: **one gap per spring DST switch** — 8 in the current
extent, one per year covered, so the count grows with the record and must never be hardcoded. The
missing local hour is 02:00; the first row after each jump is 03:00, so both descriptions appear in
the specs and mean the same thing. The autumn fold is silently collapsed by SMARD rather than
duplicated, so those days carry 24 rows and no gap marker.

### `data/rebap.csv` — cost data (Part 2)

A **second, unrelated API**: netztransparenz.de reBAP / NrvSaldo (*regelzonenübergreifender
einheitlicher Bilanzausgleichsenergiepreis*, the uniform cross-control-area imbalance price).
Differences from SMARD: it needs **OAuth2 credentials**, returns semicolon-separated CSV text
rather than JSON, and takes UTC timestamps. Credentials come from a gitignored `.env`
(`client_id` / `client_secret`) — never paste them into the notebook. The whole Part 2 section
skips cleanly when credentials are absent.

**Kept out of shared work.** reBAP is reserved for cost calculation *after* modelling: no team or
spec-driven EDA, no model features. Personal `EDA-<name>` exploration is fine
(`EDA-rebap-magc.ipynb`).

### `data/risk_labels_daily.csv` / `data/risk_labels_hourly.csv` — risk labels (derived)

Not from an API: written by
[notebooks/03_risk_classification/risk-definition.ipynb](notebooks/03_risk_classification/risk-definition.ipynb)
from `data/smard.csv`, and regenerated by re-running that notebook.

- **Plain CSV** (`sep=","`, `decimal="."`, UTF-8) — a bare `pd.read_csv` works, unlike the two raw
  files.
- Naming: `{direction}_threshold_{basis}`, `{direction}_risk_{basis}_{rule}`,
  `{direction}_range_{start,end}_{basis}_{rule}`. Directions `high` / `low`; bases `rolling`,
  `zero` (low only), `static`; rules `any`, `3h`. Daily also carries the day's max/min and their
  hours, `observations` and `day_complete`.
- Hourly holds `residual_load` plus the crossing flags per direction × basis only — no threshold
  values and no `3h` column (join the daily file on `date` for those).
- **An empty flag means "not evaluable"** — inside the first 365 days (`rolling` has no full
  trailing year) or a day with fewer than 23/24 observations. It never means "not at risk"; never
  `fillna(False)`.
- Thresholds are exported so spec 04 can apply the identical flag to `fc_residual_load` by joining
  on `date`. Never recompute them against another series — the comparison would then measure the
  thresholds, not the forecast.

## Specs and how we use Claude's output

Specs live in **`.claude/specs/`**, numbered. Spec 03 grew too large for one document and is split into a parent plus seven sub-specs (`03.1`–`03.7`), each a complete spec for one notebook section.

The specs are run by the team members and outputs (Code, Claude's interpreation) are edited after that. Do not overwrite these edits. The specs are meant to be run once or if a team member explicitly tells Claude to overwrite (with additional user confirmation) an existing spec output notebook.

**Every spec is a starting point.** Never re-run a spec, and never edit an output notebook back
towards its spec — cells that differ from the spec are the team's edits, not drift to repair.

| Spec | Status |
|---|---|
| [01-Simple-EDA.md](.claude/specs/01-Simple-EDA.md) | Run → `notebooks/01_eda/EDA-simple-claude.ipynb` (reference). The output was edited after it was run (code & interpreation). Do not overwrite the notebook and always ask before you would attempt any edit. |
| [02-Risk-Definition.md](.claude/specs/02-Risk-Definition.md) | Run → `notebooks/03_risk_classification/risk-definition.ipynb` (adopted). The team condensed, fact-checked and reworded the findings. Do not overwrite the notebook and always ask before you would attempt any edit. See the decision status below. |
| [03-combined-cherry-picked-eda.md](.claude/specs/03-combined-cherry-picked-eda.md) + `03.1`–`03.7` | Run → `notebooks/01_eda/team-EDA.ipynb` (adopted). The team modified some plots and interpretations after the spec was run. Do not overwrite the notebook and always ask before you would attempt any edit. |
| [04-forecast-metrics.md](.claude/specs/04-forecast-metrics.md) | Rough draft (a bullet list, not a full spec). Not yet run. Benchmarks SMARD's day-ahead forecasts; output goes to `notebooks/02_*`. |

Decision status of `risk-definition.ipynb` — more notebooks will follow before these are final:

- **Settled:** high and low residual load are two independently thresholded directions, not one
  signed scale.
- **Likely kept, not final:** both day rules (`any` and `3h`), the `rolling` basis (trailing
  365-day quantile over `[D-365, D-1]`), and the `zero` basis for the low direction.
- **Open:** the percentile level (1 % is a default), one model target or two, ramps as a stress
  mode, capacity normalisation.

**A spec run produces input for the team, not a finished deliverable.** We want Claude's analysis
and reasoning, and we decide ourselves what to keep, adjust or throw away. Three consequences:

1. **The suffix marks status, not authorship.**
   - **`<name>-claude.ipynb` = reference.** Claude's spec output, edited or not, kept as
     inspiration for the team ("want to learn more") and as input for Claude's future specs. It
     is *not* project knowledge, and we do not carry everything from it into the project (time &
     scope). Example: `EDA-simple-claude.ipynb`.
   - **No suffix = adopted.** The team reviewed and edited the output, decided to keep it, and
     removed the suffix; every team member can treat it as part of the project. Examples:
     `team-EDA.ipynb`, `risk-definition.ipynb`.
   - A spec's `Deliverable:` line may therefore name either form.
2. **Conclusions in a `-claude` notebook are proposals.** Do not cite them as project facts. An
   adopted notebook's conclusions are project knowledge, except where the notebook itself marks
   them open (e.g. `risk-definition.ipynb` §8 "Open for the modelling spec").
3. **Cell tags are the review mechanism.** Team members tag plot cells in place with the reason
   they matter — `keep`, `duplicate`, `combine with X`, `optional`, `modelling`, `streamlit`,
   `informational`. Spec 03 then consolidates *only* tagged cells, under a strict **"no tag, no
   plot"** rule applied uniformly (including to Claude's own untagged plots). So never silently
   add or drop a plot during implementation: changing what is included means tagging the source
   cell and revising the relevant sub-spec.

## Notebook conventions

Notebooks live in numbered subdirectories of `notebooks/` (see the layout above). There are two
naming tracks, and which applies depends on why the notebook exists.

### Per-member exploration — `EDA-<name>.ipynb`

`EDA-hari.ipynb`, `EDA-magc.ipynb`, `EDA-robert.ipynb`. These are **personal explorations and
experiments**. Their conclusions are one member's reading of the data, not project facts — treat
them as such and do not quote them as established. **Nobody edits anyone else's EDA file**; the
per-name filenames exist so merge conflicts do not arise in the first place.

Their loading conventions genuinely diverge, so never lift code from one into shared work without
rebuilding it against the shared foundation:

| Notebook | Reads | Notes |
|---|---|---|
| `EDA-hari.ipynb` | its own `data/smard_hourly_2019_2026-09-10.csv`, re-fetched in-notebook | works in **GW**, hardcodes year colours and anchor years, hand-rolls a holiday calendar |
| `EDA-magc.ipynb` | `../../data/smard.csv` | restricted to a 2025-only window |
| `EDA-rebap-magc.ipynb` | `data/rebap_2022-2026.csv` | reBAP, quarter-hourly, EUR/MWh; the filename differs from the pipeline's `data/rebap.csv` |
| `EDA-robert.ipynb` | `../../data/smard.csv` | source of the original plotting helpers |

### Spec-driven notebooks

Named by the spec that defines them — `-claude` while a reference, suffix removed once adopted
(see above). These
are **shared files**, so conflicts are real and get resolved with `nbdime` rather than avoided by
naming.

## Shared analysis foundation

[notebooks/01_eda/team-EDA.ipynb](notebooks/01_eda/team-EDA.ipynb) §1 holds the canonical copies
of the helpers — originally from `EDA-robert.ipynb`, corrected in `EDA-simple-claude.ipynb`, and
adopted by spec 03 as the one convention every consolidated notebook builds on
(`risk-definition.ipynb` reuses them). Reuse these rather than re-deriving them:

- **`time_series`** — the shared hourly DataFrame, never `ts` (the name spec 03 standardises on).
  `SERIES` names the eleven observation columns (eight measured / forecast series plus three
  `cap_*` capacity columns); `DERIVED` names `renewables`, `year`, `month`, `hour`,
  `dow`, `is_weekend`, `date`, `season`, `season_year`, `spans_gap`. The split matters: without it
  `.corr()` and `.describe()` silently pull calendar columns in as though they were measurements.
- **`_complete_periods(index, freq)`** — the edge rule, defined in exactly one place and shared by
  both aggregation helpers.
- **`period_mean(series, freq)`** / **`period_energy(...)`** — calendar-period aggregates that
  **drop periods the data does not fully cover**. Note this is *not* "drop the first and last":
  the record starts on a month boundary, so January 2019 is complete and only the trailing month
  goes. A plain `.resample()` produces fake edge dips.
- **`style_timeseries(ax, title, ylabel)`** — the shared chart style (no top/right spines, y-grid
  only, year major ticks with quarterly minors, thousands-separated y labels). `ylabel` is
  **required**, not defaulted.
- **`seasonal_plot(df, y_value, title, ylabel)`** — month-on-x, year-as-hue seaborn line plot;
  expects explicit `year` and `month` columns. It now **honours** `ylabel` (the original hard-coded
  `"MWh"`).
- **`YEARS`** — computed from the loaded data, never hardcoded. Everything year-dependent derives
  from it.

Conventions that go with them, fixed in [01-Simple-EDA.md](.claude/specs/01-Simple-EDA.md)
(Behaviour 11–21) and inherited by every later spec:

- **Units:** average **MWh** for levels, **MWh/day** for energy, **MW/h** for ramps. An hourly unaggregated reading is a level — label it `MWh`; it is still energy, even for a single hourly observation. Ask the user before you use `"MWh per hour"`. `03-combined-cherry-picked-eda.md`'s Convention 4 relabels these as `MW` inside `team-EDA.ipynb` only — a notebook-local exception kept as-is, not a project-wide correction. New specs default to `MWh`.
- **Weeks:** ISO, Monday start; the label side is stated wherever weeks are binned.
- **Seasons:** meteorological, with `season_year = year + (month == 12)`.
- **Descriptive slices** (tail hours, matched windows, longest runs) are computed inside their own plotting cell and never persisted onto `time_series`.
- **Holidays:** one source of truth — `holidays.country_holidays("DE")`, no `subdiv`.
- **Durations, not row counts:** any rule about a length of time (persistence, day completeness,
  windows) is written as a duration and converted to an observation count from the *measured*
  resolution, so a switch to SMARD's `quarterhour` data does not change its meaning. Introduced in
  `risk-definition.ipynb` §3.1.

Matplotlib is used directly for the styled plots; seaborn for the seasonal/hue plots.

## Notebook merge conflicts

`nbdime` is in the dev group. Enable it once per clone with `nbdime config-git --enable`, then
resolve conflicts with `nbdime mergetool` instead of editing `.ipynb` JSON by hand.
