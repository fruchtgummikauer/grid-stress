# CLAUDE.md

Guidance for Claude Code in this repository. The detail lives in the specs (`.claude/specs/`) and
their output notebooks; this file holds the rules and the gotchas that are easy to get wrong.

## Project

Capstone project forecasting **German power grid load & stability** from SMARD data
(Bundesnetzagentur). Team project (neuefische bootcamp), 4 members, `feature/*` branches off `main`.

- Day-ahead forecast of **`residual_load`** from installed capacity, generation and consumption.
- Risk cases are its extremes: **high** (imports, tight margins) and **negative** (renewable
  oversupply, negative prices, downward redispatch) — days at risk of TSO intervention.
- SMARD publishes its own `Forecast Residual Load`: the benchmark our models must beat.

**Scope:** generation features are Wind + Solar only (the only sources with published day-ahead
forecasts, and the largest variable ones); all other generation (biomass, coal, water, …) is out.

## Environment & commands

`uv` manages the environment — never `pip install`. Prefer `uv run` (re-syncs against `uv.lock`).

```bash
make setup      # uv sync --all-groups
make lab        # uv run jupyter lab
make test       # uv run pytest   (no test suite yet; testbook is available for notebook cells)
make format     # uv run black .  (no linter)
make mlflow     # uv run mlflow ui -> http://127.0.0.1:5000
```

Dependencies: `uv add <pkg>` (runtime), `uv add --group dev <pkg>` (notebook/tooling), `uv remove`,
`uv lock --upgrade`. See [UV_SETUP.md](UV_SETUP.md).

## Repository layout

Started from the neuefische `ds-modeling-template`. **`modeling/train.py`, `predict.py`,
`feature_engineering.py`, `notebooks/EDA-and-modeling.ipynb` and the MLflow parts of README.md are
template code on a coffee dataset** — reference for the MLflow pattern only (`make train` /
`make predict` belong to it). Real: `modeling/config.py` (MLflow URI from gitignored `.mlflow_uri`
or `MLFLOW_URI`; `EXPERIMENT_NAME` still the template default) and the root `__init__.py`
placeholder for future productionized code.

```
notebooks/
  API-connection.ipynb          # data pipeline — both raw datasets
  EDA-and-modeling.ipynb        # template leftover - ignore
  00_project_management/        # team documentation - ignore
  01_eda/
    EDA-{hari,magc,robert}.ipynb, EDA-rebap-magc.ipynb   # per-member exploration
    EDA-simple-claude.ipynb     # reference: spec 01
    team-EDA.ipynb              # adopted: spec 03
  02_forecast_metrics/
    forecast-metrics-claude.ipynb          # reference: spec 04
  03_risk_classification/
    risk-definition.ipynb                  # adopted: spec 02
  04_feature_engineering/
    feature-engineering-magc.ipynb         # spec 05 (origin 00:00 on DAY)
    feature-engineering-cutoff-magc.ipynb  # spec 06.1 (spec 05 features at spec 06's cutoff)
    data-leakage-demo-claude.ipynb         # reference, no spec: why fc_grid_load + fc_gen_wind_solar leak
    feature_library/                       # static HTML docs of the features
    Hari_Gridstress_feature_engineering_baselines_metrics.ipynb  # do not read yet (absolute paths, not cleaned up)
  05_modeling/
    regression-models-claude.ipynb  # reference: spec 06 output, refactored by the team
    regression-models-magc.ipynb    # spec 06.2 output: a copy of the -claude notebook with 06.1's features
    grid-load-models-magc.ipynb     # no spec: the refactored -claude notebook retargeted to grid_load, plus §10 residual load
```

`grid-load-models-magc.ipynb` is a team member's experiment (not adopted). It copies the refactored
`regression-models-claude.ipynb` cell for cell, but forecasts `grid_load` (`TARGET`) and scores it
against SMARD's `fc_grid_load` (`BENCHMARK`), which is **no model's input** (asserted). The
`smard_forecast_grid_load` group and `err_grid_load_recent` are gone, lags and rolling stats are
built from `grid_load` (`gl_*`), and the hybrid's linear stage regresses on the `DAY−2` / `DAY−7`
load lags plus a trend (median-filled for the spring-DST gap). `fc_gen_wind_solar` is still a
switchable input. §10 converts each load forecast into a residual-load forecast
(`load − fc_gen_wind_solar`, the way SMARD builds `fc_residual_load`) and scores it against
`fc_residual_load`. Its §5.5 shared-first-fit check allows 1e-6 MWh, because the hybrid's linear stage
rounds differently for different batch sizes. Its conclusions (§9.2, §10.5) are one member's
experiment, not project facts.

Outside `notebooks/`:

- `streamlit/` — the first Streamlit app (Home / EDA / Model), built from
  [Streamlit-draft.md](.claude/specs/Streamlit-draft.md): `streamlit_app.py` (Home),
  `pages/1_EDA.py`, `pages/2_Model.py`, `data_loading.py` (loading ported from `team-EDA.ipynb` §1,
  which stays the source of truth) and `viz_helpers.py`. Chart styling follows the `chart-style`
  skill. There is no `make` target for it. **Known issue:** the app imports its own modules as
  `streamlit.data_loading` / `streamlit.viz_helpers`, but the installed `streamlit` package
  shadows the folder, so `import streamlit.data_loading` raises `ModuleNotFoundError` (checked
  2026-09-29).
- `images/` — MLflow screenshots for the template sections of [README.md](README.md).
- `models/` — gitignored output folder of the template's `make train`, kept by a `.gitkeep`.

`Hari_Gridstress_feature_engineering_baselines_metrics.ipynb` is a per-member experiment that has not
been cleaned up to the notebook conventions yet: it reads `smard_hourly_2019_2026-09-10.csv` from an
absolute, machine-specific path instead of `data/smard.csv`. Do not read it or draw on it until the
team has cleaned it up.

## Data pipeline

**`data/` is gitignored** (`data/*.csv`, `data/metrics/*.csv`, `data/risk_classification/*.csv`,
`data/models/*.csv`, `data/features/*.csv`, `models/*`), so nothing in it comes with a clone.
`data/*.csv` does not reach into subfolders, which is why `data/metrics/`,
`data/risk_classification/`, `data/models/` and `data/features/` each have their own rule; all five
folders are kept in git by an empty `.gitkeep`.
[notebooks/API-connection.ipynb](notebooks/API-connection.ipynb) regenerates both raw datasets and
is the single source of each. The risk-label files, the SMARD forecast-error files, the feature
files and our own model files are derived from `smard.csv` by separate notebooks (see below).

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

### `data/risk_classification/risk_labels_{daily,hourly}.csv` — risk labels (derived)

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
- Thresholds are exported so the identical flag can be applied to `fc_residual_load` by joining
  on `date` — planned in the parked sub-spec
  [04.3-risk-label-link.md](.claude/specs/04.3-risk-label-link.md), not in spec 04 itself. Never
  recompute them against another series — the comparison would then measure the thresholds, not
  the forecast.

### `data/metrics/smard_*.csv` — SMARD forecast errors (derived)

Not from an API: written by
[notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb](notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb)
(spec 04) from `data/smard.csv`, and regenerated by re-running that notebook. The `smard_` prefix
marks them as the baseline to beat; our own model's files live in `data/models/` (below).

- `smard_forecast_errors_hourly.csv` — one row per hour: actual, forecast and `err_*`
  (`forecast − actual`, positive = over-forecast) for residual load, grid load and wind + solar.
  Missing forecasts stay empty (2020-01-31 in the current snapshot), never interpolated. The
  source of truth: any evaluation window, including the modelling spec's test window, is re-scored
  from this file.
- `smard_forecast_errors_daily.csv` — one row per calendar day: per-pair MAE / RMSE / bias /
  `hour_count`, plus residual load's day max/min values, timestamps, value errors and timing
  offsets. The timestamps are plain local time (like the hourly file), so the two join directly;
  use `offset_max_h` / `offset_min_h` rather than subtracting timestamps, which is 1 h off on DST
  days.
- `smard_benchmark_metrics.csv` — long format (`level`, `period`, `pair`, `metric`, `value`,
  `hour_count`) for the cascade above day level (`full`, `trailing_365`, `year`, `month`) plus
  the `season` slice. Partial years and months are included but not marked — read them through
  `hour_count`. A metric that does not apply has no row: the capacity-normalised error exists only
  for wind + solar at `trailing_365` and `year`.
- Plain CSV (`sep=","`, `decimal="."`, UTF-8), like the risk labels.

### `data/models/model_*.csv` — our own forecasts and scoreboard (derived)

Not from an API: written by
[notebooks/05_modeling/regression-models-claude.ipynb](notebooks/05_modeling/regression-models-claude.ipynb)
(spec 06) from `data/smard.csv` and `data/metrics/smard_forecast_errors_hourly.csv`. The notebook
writes them **only when its `EXPORT_ENABLED` toggle is on** (default off, while the team
experiments); both frames are always built in memory. The folder itself exists and is not created
by the notebook.

- `model_forecast_errors_hourly.csv` — long format, one row per test hour × model ×
  split method, for the registry rows and seasonal naive: `timestamp`, `model`, `split_method`,
  `residual_load`, `forecast`, `lower`, `upper`, `err_residual_load`. SMARD is **not** in this file
  (its hourly values are in `data/metrics/smard_forecast_errors_hourly.csv`); join the two on
  `timestamp` (plain local time). A registry row without a single forecast (a failed static fit)
  still has its test hours here, left empty.
- `model_scoreboard.csv` — long format (`model`, `split_method`, `table`, `metric`, `value`,
  `count`), including the SMARD and seasonal-naive rows (`split_method = "none"`). Every value is
  computed on the common test hours. Metrics per `table`:
  - `accuracy`: `MAE`, `RMSE`, `bias`, `skill_pct` (`count` = hours), `months_beating_smard`
    (`value` = months won, `count` = full calendar months), `fit_seconds` (`count` = fits)
  - `extremes`: `{MAE,bias}_{low_extreme,ordinary,high_extreme}_by_{actual,forecast}` and
    `skill_pct_{low_extreme,ordinary,high_extreme}_by_actual` (`count` = hours in the bin;
    `low_extreme` ≤ P1, `ordinary` P25–P75, `high_extreme` > P99 of the test window's actual),
    `{MAE,bias,skill_pct}_day_{max,min}` (`count` = days)
  - `intervals`: `coverage_pct`, `mean_width`

  A metric that does not apply has no row: SMARD has no skill, months, fit-time or interval rows,
  seasonal naive no fit time, the forecast-binned extremes have no skill (each row fills its bins
  with different hours), and a row without forecasts has no rows at all.
- Plain CSV (`sep=","`, `decimal="."`, UTF-8), like the other derived files.

`model_magc_forecast_errors_hourly.csv` and `model_magc_scoreboard.csv` are the same two exports
from [regression-models-magc.ipynb](notebooks/05_modeling/regression-models-magc.ipynb) (spec
06.2), behind that notebook's own `EXPORT_ENABLED` (default off). It was copied from the
`-claude` notebook **before** the refactor, so its extremes metrics still use the old bin names
`bottom` / `ordinary` / `top`, without the extremes skill metrics. Don't compare its scoreboard
with the `-claude` one by metric name alone.

`model_load_forecast_errors_hourly.csv` and `model_load_scoreboard.csv` are the same two exports
from [grid-load-models-magc.ipynb](notebooks/05_modeling/grid-load-models-magc.ipynb), behind its
own `EXPORT_ENABLED` (default off), for **grid load**: the hourly file has `grid_load` and
`err_grid_load` instead of the residual-load columns, and the scoreboard's SMARD row is
`fc_grid_load`. The §10 residual-load comparison is in memory only, not exported.

### `data/features/residual_load_features{,_cutoff}.csv` — feature tables (derived)

Not from an API: written from `data/smard.csv` (and, for the cutoff file,
`data/metrics/smard_forecast_errors_hourly.csv`) by the two feature-engineering notebooks, which
create `data/features/` if it is missing. Plain CSV (`sep=","`, `decimal="."`, UTF-8), one row per
hour of the record, indexed by `timestamp`.

- `residual_load_features.csv` —
  [feature-engineering-magc.ipynb](notebooks/04_feature_engineering/feature-engineering-magc.ipynb)
  (spec 05): features built for a forecast origin at **00:00 on `DAY`**. Under spec 06's 18:00
  issue time and its availability cutoff, some of them read hours not yet published, so they are
  **not** leakage-safe for the regression models.
- `residual_load_features_cutoff.csv` —
  [feature-engineering-cutoff-magc.ipynb](notebooks/04_feature_engineering/feature-engineering-cutoff-magc.ipynb)
  (spec 06.1): the same feature definitions rebuilt at spec 06's availability cutoff; `timestamp`,
  `cutoff` and 16 feature columns. `regression-models-magc.ipynb` does not read this file: it
  rebuilds the same definitions inside its own feature builder.
    regression-models-claude.ipynb  # reference: spec 06, refactored and extended by the team
    regression-models-magc.ipynb    # spec 06.2: pre-refactor copy of -claude with 06.1's features
    ensemble-claude.ipynb           # reference: spec 08, combines the spec 06 model saves
    visualization-01-regression-best-models.ipynb  # spec 09: top picks per category vs SMARD
    visualization-02-classification-risk-labels.ipynb  # spec 10: spec 02's risk flags on the picks vs SMARD
```

- `streamlit/` — app from [Streamlit-draft.md](.claude/specs/Streamlit-draft.md) (Home / EDA /
  Model); chart styling via the `chart-style` skill; no `make` target. **Known issue:** imports like
  `streamlit.data_loading` are shadowed by the installed `streamlit` package → `ModuleNotFoundError`.
- `images/` — README screenshots. `models/` — gitignored template output.

## Data

**`data/` is gitignored** (each subfolder has its own rule and a `.gitkeep`); nothing comes with a
clone except the model saves in `data/models/<model_key>/` (committed on purpose). Raw files come
from [API-connection.ipynb](notebooks/API-connection.ipynb); derived files are regenerated by
re-running their producer.

| File | Producer | Format |
| --- | --- | --- |
| `data/smard.csv` | API-connection (SMARD API, no key) | **German CSV**: `sep=";"`, `decimal=","`, `utf-8-sig` |
| `data/rebap.csv` | API-connection (netztransparenz, OAuth2 via `.env`) | semicolon CSV |
| `data/risk_classification/risk_labels_{daily,hourly}.csv` | `risk-definition.ipynb` | plain CSV |
| `data/risk_classification/model_risk_labels_{daily,hourly}.csv` | `visualization-02-classification-risk-labels.ipynb` (spec 10), if `EXPORT_ENABLED` (default on) | plain CSV |
| `data/metrics/smard_*.csv` | `forecast-metrics-claude.ipynb` (spec 04) | plain CSV |
| `data/models/model_*.csv` | `regression-models-claude.ipynb` (spec 06), if `EXPORT_ENABLED` (default on) | plain CSV, long format |
| `data/models/<model_key>/{config.json,results.joblib}` | same notebook §5, if `SAVE_MODELS` (default off) | JSON + joblib pickle of `RESULTS[model_key]` (spec 06 Behaviour 34) |
| `data/models/ensemble_*.csv` | `ensemble-claude.ipynb` (spec 08), if `EXPORT_ENABLED` (default on) | plain CSV: hourly forecasts + bands, long scoreboard, weights |
| `data/models/model_magc_*.csv` | `regression-models-magc.ipynb` (spec 06.2), only if `EXPORT_ENABLED` | plain CSV, long format |
| `data/features/residual_load_features{,_cutoff}.csv` | the two feature-engineering notebooks | plain CSV |

Column layouts are defined in the producing spec. Gotchas:

- **`smard.csv`**: read with `delimiter=";", encoding="utf-8-sig"` and convert numerics
  (`str.replace(",", ".")` → `float`). Hourly, `Europe/Berlin`, from 2019-01-01; the end moves with
  every re-fetch, so **never hardcode a calendar year** — derive from `YEARS`. Whole-record
  quantiles shift with re-fetches (why spec 02 avoids them). The `cap_*` columns are yearly step
  values repeated hourly, not measurements.
- **DST:** one gap per spring switch (missing 02:00, next row 03:00) — the count grows with the
  record, never hardcode it. Autumn days have 24 rows, no marker.
- **reBAP stays out of shared work** (no team EDA, no model features) — reserved for cost
  calculation after modelling. Never paste credentials into a notebook.
- **Risk labels: an empty flag means "not evaluable"** (first 365 days, incomplete day), never "not
  at risk" — never `fillna(False)`. Apply the exported thresholds to other series by joining on
  `date`; never recompute them against another series. Same rule for `model_risk_labels_*.csv`,
  where `model` in `high_*` / `low_*` columns is that direction's pick (`high_model` / `low_model`).
- **SMARD errors:** `err_* = forecast − actual` (positive = over-forecast); missing forecasts stay
  empty. `smard_forecast_errors_hourly.csv` is the source of truth for re-scoring any window. Use
  `offset_max_h` / `offset_min_h`, not timestamp subtraction (1 h off on DST days).
- **Model exports:** SMARD's hourly values are not in `model_forecast_errors_hourly.csv` — join
  `data/metrics/` on `timestamp`. The `-magc` scoreboard predates the refactor (bins `bottom` /
  `ordinary` / `top`, no `below_zero`, 24-month window) — don't compare it with the `-claude` one by
  metric name. `visualization-01-regression-best-models.ipynb` reads only the `-claude` exports; its
  self-check stops when they and `data/metrics/` come from different runs.
- **Model saves:** `load_saved` per model: `"results"` loads (same data snapshot only — stops
  after a re-fetch), `"config"` refits the saved tuning on any data. `results.joblib` is a pickle —
  load team saves only.
- **Features:** `residual_load_features.csv` (spec 05) is **not** leakage-safe under spec 06's 18:00
  issue time; the `_cutoff` file (spec 06.1) is.

## Specs and how we use Claude's output

Specs live in `.claude/specs/`. **Every spec is a starting point**: run once, then the team edits
the output (code and interpretation). Never re-run a spec, never overwrite an output notebook
(only on a team member's explicit request plus confirmation), and never edit it back towards its
spec — differences are team edits. **Always ask before editing any spec output notebook.**

| Spec | Status |
| --- | --- |
| 01 Simple EDA | Run → `EDA-simple-claude.ipynb` (reference, edited) |
| 02 Risk definition | Run → `risk-definition.ipynb` (adopted) |
| 03 + 03.1–03.7 Combined EDA | Run → `team-EDA.ipynb` (adopted) |
| 04 Forecast metrics | Run → `forecast-metrics-claude.ipynb` (reference) + `data/metrics/` |
| 04.1 / 04.2 / 04.3, 05.1 | Parked — run only when a team member asks |
| 05 Feature engineering | Run → `feature-engineering-magc.ipynb` (not adopted; file status line still says draft) |
| 06 Regression models | Run → `regression-models-claude.ipynb` (reference) + `data/models/`; spec describes the refactored notebook, later changes under *Additions after the refactor* |
| 06.1 Cutoff features | Run → `feature-engineering-cutoff-magc.ipynb` |
| 06.2 Cutoff features in models | Run → `regression-models-magc.ipynb` (PR #33; file status line still says draft) |
| 07 Hybrid linear stage | Draft, parked; partly overtaken — team kept the linear stage (2026-09-30) |
| 08 Ensemble | Run → `ensemble-claude.ipynb` (reference) + three `data/models/ensemble_*.csv`; reads the spec 06 model saves (no refit), weights / edges / windows chosen on the validation year only, bands from out-of-fold validation forecasts; member diagnostic uses spec 09's pick rule and spec 10's risk flags; about 4 min |
| 09 Best-model plots | Run → `visualization-01-regression-best-models.ipynb` (no suffix: team choice); team changes under *Changes during the run* |
| 10 Model risk labeling | Run → `visualization-02-classification-risk-labels.ipynb` (no suffix: team choice) + two label files; team changes under *Changes during the run* |
| Streamlit-draft | Run → `streamlit/` (no trained model shown yet) |

Spec 06 notebook, operationally: `USE_GPU` (default on) makes XGBoost results machine-dependent
(`False` reproduces CPU); a full default run takes about 30 min on 16 cores, seconds with
`load_saved = "results"`.

`risk-definition.ipynb` decision status — **settled:** high and low are two independently
thresholded directions. **Likely kept:** day rules `any` and `3h`, `rolling` basis (trailing
365-day quantile over `[D-365, D-1]`), `zero` basis for low. **Open:** percentile level (1 %
default), one target or two, ramps, capacity normalisation.

How output status works:

1. **Suffix marks status, not authorship.** `-claude` = reference (inspiration, not project
   knowledge). No suffix = adopted by the team. `-magc` on specs 05 / 06.1 / 06.2 is a team
   member's choice; still a spec output, not adopted.
2. **Conclusions in a `-claude` notebook are proposals** — never cite them as project facts.
   Adopted notebooks are project knowledge except where they mark something open.
3. **Cell tags are the review mechanism** (`keep`, `duplicate`, `combine with X`, `optional`,
   `modelling`, `streamlit`, `informational`). Spec 03 applies **"no tag, no plot"** — never
   silently add or drop a plot; change the tag and the sub-spec instead.

## Notebook conventions

- **Per-member `EDA-<name>.ipynb`**: personal explorations, not project facts. **Nobody edits
  anyone else's EDA file.** Their loading diverges (Hari: own re-fetched CSV, GW, hardcoded years;
  magc: 2025 only; rebap-magc: `data/rebap_2022-2026.csv`) — never lift code into shared work
  without rebuilding it on the shared foundation.
- **Spec-driven notebooks** are shared files: resolve conflicts with `nbdime`
  (`nbdime config-git --enable` once per clone, then `nbdime mergetool`), never by hand-editing JSON.

## Shared analysis foundation

[team-EDA.ipynb](notebooks/01_eda/team-EDA.ipynb) §1 holds the canonical helpers — reuse, don't
re-derive:

- **`time_series`** (never `ts`); **`SERIES`** = the eleven observation columns incl. `cap_*`,
  **`DERIVED`** = calendar columns. Keep them apart so `.corr()` / `.describe()` don't pull in
  calendar columns.
- **`period_mean` / `period_energy`** (via `_complete_periods`) drop periods the data doesn't fully
  cover — plain `.resample()` gives fake edge dips.
- **`style_timeseries(ax, title, ylabel)`** (`ylabel` required), **`seasonal_plot(df, y_value,
  title, ylabel)`**, **`YEARS`** (derived from the data, never hardcoded).

Conventions (spec 01, Behaviour 11–21):

- **Units:** **MWh** for levels (an hourly reading is labelled `MWh`), **MWh/day** for energy,
  **MW/h** for ramps. Ask before using "MWh per hour". `team-EDA.ipynb` uses `MW` as a local
  exception only.
- **Weeks:** ISO, Monday start. **Seasons:** meteorological, `season_year = year + (month == 12)`.
- **Holidays:** `holidays.country_holidays("DE")`, no `subdiv`.
- **Durations, not row counts:** time rules are written as durations and converted via the
  measured resolution.
- Descriptive slices stay inside their plotting cell, never persisted onto `time_series`.
- Matplotlib for styled plots, seaborn for seasonal/hue plots.
