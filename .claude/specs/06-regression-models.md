# 06 — Regression Models: Day-Ahead Residual-Load Forecast vs. SMARD

- Status: run, then refactored by the team section by section (branch `refactor/regression-models`,
  2026-09). This spec was updated afterwards to describe the refactored notebook: names, defaults
  and printed tables follow the notebook, and Behaviour numbers are unchanged, because other specs
  cite them. The decisions taken in review of the team's first draft are listed under Goal.
- Updated again on 2026-09-30 for the additions made after the refactor: the Ridge model, the
  random-forest hybrid, the spec 06.1 feature groups, the GPU switch, the new window and search
  defaults and the `below_zero` bin. They are listed under *Additions after the refactor* and
  described in the Behaviours below.
- Updated once more on 2026-09-30 for saving and loading model results (Behaviour 34) and the
  export toggle switched on by default (Behaviour 28).
- Branch: `feature/*` off `main`
- Deliverable: `notebooks/05_modeling/regression-models-claude.ipynb`
  - (reference, `-claude` suffix until the team adopts it)
  - plus two artifacts, `data/models/model_forecast_errors_hourly.csv` and
    `data/models/model_scoreboard.csv`, written when the export toggle is on, which is the
    default (Behaviour 28)
  - plus one save folder per model, `data/models/<model_key>/`, written on request and committed
    to git (Behaviour 34)
- Depends on:
  - the shared foundation in `notebooks/01_eda/team-EDA.ipynb` §1 (as reused by
    `risk-definition.ipynb` and `forecast-metrics-claude.ipynb`)
  - `data/metrics/smard_forecast_errors_hourly.csv` from
    [04-forecast-metrics.md](04-forecast-metrics.md), used to re-score SMARD on our test hours
- Follow-up (not part of this spec): an MLflow spec that replaces the template setup in
  `modeling/`, drafted once this spec is approved.

## Goal

Build our own **day-ahead forecast of `residual_load`** for every hour of a delivery day `DAY`,
issued at the same point in time as SMARD's public forecast. Compare several model families
against SMARD's `fc_residual_load` and against a naive floor, on **identical hours**, in one
scoreboard.

The models may use SMARD's published component forecasts (`fc_grid_load`, `fc_gen_wind_solar`)
as inputs. Our model is therefore a **post-processor** of SMARD's forecast, not an independent
forecast from scratch. The honest claim, if it wins, is "we reduce SMARD's error by X %", not "we
forecast better than the TSOs". Meteorology calls this approach *Model Output Statistics*. It is
not leakage, because the component forecasts for `DAY` are public before our issue time (see
Forecast setting).

The notebook is meant to stay **short**, so every team member can change a configuration value,
re-run it and read the result. Configuration comes first, helpers stay compact, and the
interpretation is written once, at the end.

Concretely, this spec:

1. Fixes the **forecast setting**: issue time, which actuals are available at that time, and
   which hours are forecast. Every model and every training row obeys the same rule.
2. Fits model variants plus a naive floor, all configured from one registry cell:
   - seasonal naive `DAY−7`
   - `sarimax_fourier`: a regression with ARIMA errors, with Fourier terms for the daily and
     weekly seasons; in the registry but switched off by default (it dominates the runtime)
   - LightGBM, **direct** and as a **hybrid**, where a linear model captures the trend and the
     booster models its residuals
   - XGBoost in the same two architectures
   - a Ridge regression, direct only (`linear_direct`)
   - a random forest as the residual stage of a hybrid, hybrid only (`random_forest_hybrid`)
3. Evaluates each model under two **split methods** on the same test year:
   - **static**: fit once, frozen for the whole test year
   - **rolling origin**: refit at a fixed interval on a sliding window

   The difference between the two is the value of refitting.
4. Gives every model a **95 % prediction interval** built the same way for all models, and
   measures whether the interval actually covers 95 % of hours.
5. Presents a **scoreboard** with these columns:
   - MAE, RMSE, bias, the hour count, skill vs. SMARD and the months in which a model beats SMARD
   - error in the tails and on the day max / min, which the risk label is built on
   - interval coverage
6. Closes with one **"Did we beat SMARD?"** section, including the caveats that go with it.

Success means the team can read one table and say, per model and split method: better or worse
than SMARD, by how much, on the ordinary hours and in the tails, with an interval that is honest
or not.

### Design decisions taken in review of the team's first draft

| Draft | This spec | Why |
|---|---|---|
| Forecast horizon of 24 steps | Target is **delivery day `DAY`**, 00:00–23:00 local, issued at **18:00 on `DAY−1`** | A spring DST day has 23 hours, and at 15-min resolution a day has 96 steps. Our model cannot see `DAY−1` up to 23:00 either: the target hours are 9–32 hourly steps after the last usable actual (10–33 under the first draft's 3 h lag). |
| "Same data as SMARD": wind/solar at 18:00, load at 10:00, residual load at 18:00 | The issue time is 18:00 `DAY−1`, the earliest point at which all SMARD inputs for `DAY` exist. Actuals are usable up to issue time minus a **2 h publication lag**. | The residual-load value "published at 18:00" is SMARD's *forecast*, not the actual. Manual checks on smard.de showed actuals 15 min to 2 h behind real time, varying by day; the lag uses the upper end. (The first draft of this spec used 3 h.) |
| Residual load driven by the grid-load and wind/solar forecasts | SMARD's `fc_grid_load` and `fc_gen_wind_solar` for `DAY` are model inputs. `fc_residual_load` is **not** a feature. | It equals `fc_grid_load − fc_gen_wind_solar` exactly (spec 04), so it adds nothing but collinearity. |
| Holdout: keep the last `h` observations | Test = the **last 365 complete delivery days**, ending with the last fully observed day | One full seasonal cycle. It is the same window as spec 04's `trailing_365` headline, so the SMARD number is directly comparable. |
| Rolling-origin CV with a 24-month window | The split method is an **experimental factor** evaluated on the same test year. Static and rolling both use the same training window: **24 months** at first, **36 months** since 2026-09-29 (*Additions after the refactor*). Hyperparameters are tuned on the validation year before the test. | Different test sets would make the split methods incomparable. With equal window lengths, the only difference between the split methods is refitting. |
| Differencing to handle stationarity | SARIMAX uses a fixed `d = 0`, checked (not decided) by one KPSS test; differencing, if ever needed, happens **inside ARIMA** through `d`. Tree models get no differenced target. | The SMARD forecasts and Fourier terms carry level and seasons, so the remaining error is expected to be stationary. Differencing outside the model complicates back-transformation and intervals. |
| SARIMA with `auto_arima` | One **SARIMAX** variant in statsmodels: ARIMA errors plus Fourier terms for the seasons (Behaviour 15). No `pmdarima`. | SARIMA takes one seasonal period; hourly data has three. `m = 168` does not fit on a desktop. Fourier terms carry the seasons cheaply and stay resolution-independent. `pmdarima` is not installed and has had numpy-2 problems. |
| AIC to compare parameters | **No AIC search**: SARIMAX's order is fixed at `(1, 0, 1)` in the registry. All models are compared out of sample (MAE / RMSE on identical hours). | Keeps SARIMAX simple and fast. AIC could only ever compare ARIMA orders within one model; a gradient booster has no AIC. |
| Holt-Winters | **Dropped** (team decision) | It takes no exogenous inputs, handles one season only, and must be additive because residual load goes negative. It adds little over seasonal naive. |
| LightGBM direct, XGBoost hybrid | A **2 × 2 grid** in the registry: {LightGBM, XGBoost} × {direct, hybrid}. The hybrid is a wrapper around either booster. All four are `enabled: True` by default; `sarimax_fourier` is `enabled: False`. | Otherwise algorithm and architecture are confounded. The spec first switched XGBoost off to save runtime; the team switched it on and SARIMAX off instead, since SARIMAX dominates the runtime. |
| Plot with a 95 % confidence interval | A **95 % prediction interval** built by one empirical method for all models, plus measured coverage | Boosters have no native interval. An interval whose coverage is not measured cannot be checked. |
| MAE and RMSE scoreboard | Adds bias, skill vs. SMARD, monthly wins, tail and day-extreme errors, interval coverage and a seasonal-naive row | These follow from spec 04's approach. The seasonal-naive row pulls one baseline out of parked 04.1; the rest of 04.1 stays parked. |
| Pipeline ready for 15-min data | Every window, lag and period is a **duration**. | Project convention (durations, not row counts). |

### Additions after the refactor

Made by the team in the notebook (2026-09-29/30) and decided on the **validation year** in scratch
comparisons; the test year only confirmed.

| Addition | What | Why |
|---|---|---|
| `linear_direct` | A Ridge regression, direct only, with calendar one-hot encoding, median imputation and scaling inside its pipeline (Behaviour 17) | A linear model extrapolates through every input and is not capped at the training range like a tree. In the default run it has the lowest `ordinary` MAE of all rows (2,230 MWh rolling). |
| `random_forest_hybrid` | A random forest as the hybrid's residual stage, on by default; no direct forest (Behaviour 17) | Best test row (rolling MAE 2,333 MWh, +16.4 % vs SMARD), but behind the four boosters on the validation year (2,241 against 2,138–2,165 MWh); kept for its low tail (170 validation hours below 0 MWh: rolling MAE 2,669 MWh against SMARD's 3,841 MWh). A direct forest was tested and rejected: it is capped in the low tail even harder than the direct boosters (`low_extreme` skill −20.4 % rolling, −63.8 % static). |
| spec 06.1 feature groups | `cyclical`, `renewables_history`, `rolling_stats`, **off by default** (Behaviour 18) | They lowered the validation MAE but raised the test MAE in every scratch run (`lgbm_direct` rolling 2,412 → 2,486 MWh with all three on). |
| Training window 36 months, refit every 7 days | `WINDOWS["train"]`, `WINDOWS["refit_every"]` (Behaviour 3) | Chosen from 12 / 24 / 36 / 48 months × 1 / 7 / 30 days. Negative residual load becomes more common every year (170 h below 0 in the validation year, 563 h in the test year), so a refit has to pick up a new level quickly; 36 months beat 24 for every booster. A daily refit added little for about 7× the fits. |
| Tuning folds every 30 days | `WINDOWS["tune_every"]`, run `validation_search` (Behaviour 11) | Decoupled from the 7-day refit: 7-day folds (53 instead of 13) selected equally good configurations (validation MAE within ±10 MWh) but took 46 instead of 13 min. |
| Wider grids, `n_iter = 30` | LightGBM with `linear_tree` / `linear_lambda` and a fixed `subsample_freq = 1`; XGBoost with the loss as a grid value (Behaviour 11) | Better validation MAE. Without `subsample_freq = 1`, LightGBM ignores `subsample`. |
| `USE_GPU` | XGBoost on an NVIDIA GPU (`device = "cuda"`) when `gpu_found()` detects one, else on the CPU; LightGBM always on the CPU | Faster XGBoost fits. GPU and CPU build slightly different trees, so scoreboards from different machines do not match exactly; `USE_GPU = False` reproduces a CPU run. |
| `below_zero` bin | Actual `< 0 MWh` in Scoreboard B, next to the quantile bins (Behaviour 24) | A fixed threshold, the `zero` basis of `risk-definition.ipynb`: it measures the same physical situation (renewable oversupply) in every year, unlike P1. |
| Publication times, 2 h lag kept | §2 lists SMARD's publication rules (Forecast setting) | A 1 h lag, as SMARD's rule allows, was worse on the validation year (rolling MAE +13 MWh on average over the five registry models then, +65 MWh on hours below 0), and the newest rows may still be SMARD estimates at a 17:00 cutoff. |
| Saving and loading | `SAVE_MODELS` and a per-model `load_saved` switch; saves in `data/models/<model_key>/`, committed to git; `joblib` added as a dependency (Behaviour 34) | A full run takes about 30 min. A committed save lets every team member load the results in seconds, or refit a saved configuration on new data without the search. |
| Export on by default | `EXPORT_ENABLED = True` (Behaviour 28) | Team decision: every default run writes the two exports; the toggle stays for runs that should not overwrite them. |

### Forecast setting

For every delivery day `DAY` in the evaluation windows:

| Item | Rule (defaults, all configurable durations) |
|---|---|
| Issue time | 18:00 local on `DAY−1`. Issue time and availability cutoff are **per-day timestamps**, the `issue_time` and `cutoff` columns of `FORECAST_SETTING` (Behaviour 5), not constants; `DATA_INFO` holds their configured clock time and lag. |
| Availability cutoff | `cutoff = issue_time − actuals_lag`, with `actuals_lag = 2 h`: 16:00 on `DAY−1`. Every use of "the cutoff" or "availability cutoff" in this spec means this per-day timestamp. |
| Actuals available at the issue time | An observation stamped `t` (interval start) is usable if its interval ends at or before the cutoff: `t + resolution ≤ cutoff` (`available(stamps, cutoff)`). Hourly: up to the hour stamped 15:00 on `DAY−1`. |
| SMARD forecasts for `DAY` | Available: `fc_gen_wind_solar` is published by 18:00 on `DAY−1` (it sets the issue time), `fc_grid_load` by 10:00 on `DAY−1` and updated whenever it changes by at least 10 %. The notebook's §2 lists SMARD's publication times, source ENTSO-E. |
| SMARD forecasts for the rest of `DAY−1` | Available: published on `DAY−2`. |
| Target hours | Every local hour of `DAY`. That is 24 hours, or 23 on the spring DST day; the autumn fold is collapsed by SMARD. |
| Training rows | Delivery days whose target hours are all available at the fit's issue time, i.e. up to `DAY−2` for a fit issued on `DAY−1`. Every training row is built **as it would have been at its own issue time**. |

## Inherited from the shared foundation

| Inherited | Established in |
|---|---|
| `time_series` (not `ts`), loading, renaming, dtype asserts, data-directory resolver | `team-EDA.ipynb` §1 / [03.1-setup.md](03.1-setup.md) |
| `SERIES` (eleven columns incl. `cap_*`), `DERIVED`, `YEARS`, `spans_gap` | `team-EDA.ipynb` §1 |
| `style_timeseries` (`ylabel` required), season mapping, `DAY_NAMES` | `team-EDA.ipynb` §1 |
| Durations, not row counts; `DAY_COMPLETENESS = 23 / 24` | `risk-definition.ipynb` §3.1 |
| Holidays: `holidays.country_holidays("DE")`, no `subdiv` | [01-Simple-EDA.md](01-Simple-EDA.md) |
| Sign convention `error = forecast − actual`, positive = over-forecast | [04-forecast-metrics.md](04-forecast-metrics.md) |
| Pairwise-complete hours; RMSE recomputed from hourly errors, never averaged; `hour_count` next to every metric; no MAPE | [04-forecast-metrics.md](04-forecast-metrics.md) |
| Tail analysis binned both by actual and by forecast (regression-to-the-mean check); day max / min value error | [04-forecast-metrics.md](04-forecast-metrics.md) |
| SMARD archive caveats (revised load forecast, final-vintage values) | [04-forecast-metrics.md](04-forecast-metrics.md) |

`forecast-metrics-claude.ipynb` is a **reference** notebook, so its conclusions are proposals, not
project facts. This spec uses them only as **hypotheses to test**, for example that SMARD's
grid-load bias varies by hour, season and year and may be learnable. Where the notebook reports a
number, this spec says so rather than asserting it.

Units follow the project default for new specs: hourly readings, forecasts and errors in `MWh`,
capacity in `MW`, skill and coverage in `%`. The `MW` relabelling in `team-EDA.ipynb` does not
apply.

## Scope

### IN

- One **short** notebook, `notebooks/05_modeling/regression-models-claude.ipynb`, runnable top to
  bottom from a fresh kernel against `data/smard.csv` and
  `data/metrics/smard_forecast_errors_hourly.csv`.
- A compact restatement of the shared setup.
- Configuration cells (a model registry, windows, the data-availability rule, split methods,
  features, intervals, plot selection, export toggle, save and load switches) that the team edits
  the way it edits a colour map.
- The forecast setting and a truncation-based leakage test.
- A validation year for tuning, a test year, and the static and rolling split methods.
- Seasonal naive `DAY−7`, LightGBM and XGBoost, each direct and hybrid, a Ridge regression
  (direct) and a random-forest hybrid, built as scikit-learn pipelines and tuned with
  `GridSearchCV` / `RandomizedSearchCV`; `sarimax_fourier` in the registry, switched off by
  default.
- A stationarity check (one KPSS test) for `sarimax_fourier`.
- A minimal, toggleable feature set for the boosters, the linear model and the random forest,
  plus the spec 06.1 groups, switched off by default.
- A self-check cell at the end of every section, plus the closing self-check.
- Empirical 95 % prediction intervals and their coverage.
- The scoreboard (accuracy, extremes, intervals), the static vs. rolling comparison, and one
  forecast plot with its band for a selectable model.
- Two CSV exports behind a toggle that is on by default, one closing "Did we beat SMARD?"
  section, and a self-check cell.
- Saving each model's results and loading them, or only its selected configuration, in a later
  run (Behaviour 34).

### OUT

- Holt-Winters or any exponential-smoothing model (team decision).
- A seasonal ARIMA with a seasonal period `m`; the seasons are carried by Fourier terms.
- Weather data or any input not in `data/smard.csv`.
- Using `fc_residual_load` as a feature.
- Written interpretation after individual tables or plots; the interpretation lives in the
  closing section only.
- Risk flags on our forecast, flag agreement, precision / recall. Our hourly export makes this
  possible later, in the manner of parked [04.3-risk-label-link.md](04.3-risk-label-link.md).
- The rest of parked [04.1-naive-baseline.md](04.1-naive-baseline.md): daily persistence and
  skill scores for SMARD against naive baselines.
- MLflow logging and any change to `modeling/`.
- Significance tests such as Diebold–Mariano. The monthly win count (Behaviour 23) is the
  robustness view in this spec.
- A rich feature set (ramps, interactions, anything beyond the spec 06.1 groups); that is for the
  team's feature-engineering work.
- Quarter-hour data. The pipeline is ready for it (Behaviour 30), but this spec runs on the
  hourly `data/smard.csv`.
- reBAP / cost calculation.
- New dependencies. scikit-learn and scipy (for random-search distributions) are already
  runtime dependencies; `joblib` was added by the team for saving (Behaviour 34).

### Notebook structure

The notebook has nine sections. Its models part is split in two: section 4 defines the models,
section 5 fits them and runs the leakage test.

| § | Section | Behaviours |
|---|---|---|
| 1 | Setup and configuration (1.1 shared setup, 1.2 resolution and durations, 1.3 configuration, 1.4 SMARD's hourly errors, 1.5 self-check) | 1–4 |
| 2 | Forecast setting (2.1 setting, 2.2 feature availability, 2.3 capacity rule, 2.4 unforecastable days, 2.5 self-check) | 5–7, 9 |
| 3 | Windows and split methods (3.1 windows, 3.2 fit schedule, 3.3 split engine and tuning, 3.4 self-check) | 10–14 |
| 4 | Models (4.1 features and the spec 06.1 feature check, 4.2 seasonal naive, 4.3 `sarimax_fourier`, 4.4 KPSS, 4.5 boosters, linear model and random forest, 4.6 self-check) | 15–19 |
| 5 | Fitting and leakage test (5.1 fit or load, 5.2 selected configurations, 5.3 runtime and failures, 5.4 leakage test, 5.5 self-check) | 8, 11, 21, 32, 34 |
| 6 | Prediction intervals (6.1 self-check) | 20 |
| 7 | Evaluation and scoreboard (7.1 common hours, 7.2–7.4 scoreboards A–C, 7.5 static vs. rolling, 7.6 plot, 7.7 self-check) | 21–27 |
| 8 | Export | 28–29 |
| 9 | Closing checks and conclusion (9.1 closing self-check, 9.2 "Did we beat SMARD?") | 31, 33 |

Every section's self-check asserts that `time_series` still carries exactly `SERIES + DERIVED`.
Markdown cells link to other sections by anchors (`#sec-1-3` etc.).

## Behaviour

### Setup and configuration

1. Repeat `team-EDA.ipynb` §1's loading and preparation: the resolver, `time_series`, `SERIES`,
   `DERIVED`, `YEARS`, `style_timeseries` and the season mapping. One markdown cell states they
   are inherited and points at the source; nothing is re-derived.
2. Measure the resolution from the index (`RESOLUTION`, the most common step). Every duration
   below is converted to an observation count from that measured resolution by `n_obs(duration)`,
   never written as a row count. `describe(value)` prints durations readably (`24h`, `7 days`,
   `24 months`).
3. **Configuration cells** in §1.3, each printed once as a summary. Nothing downstream hardcodes a
   value they hold. Windows and times are durations: `pd.Timedelta`, or `pd.DateOffset` for
   calendar months.
   - `DATA_INFO`: `issue_days_before = 1 day` and `issue_clock = 18:00` (the forecast for `DAY`
     is issued at 18:00 on `DAY−1`), `actuals_lag = 2 h`, and `capacity_published_after = 0`, the
     capacity publication rule "year `Y` is published on 1 January of `Y`, 00:00" (Behaviour 7).
   - `WINDOWS`: `test = 365 days`, `validation = 365 days`,
     `train = 36 months` (calendar months, i.e. `DateOffset(months=36)`, not 1095 days),
     `refit_every = 7 days` (rolling method) and `tune_every = 30 days` (spacing of the tuning
     folds, Behaviour 11).
   - `TRAIN_TEST_SPLIT_METHOD = {"static": True, "rolling": True}`: the split methods, each with
     an on/off switch. With both on (the default) the scoreboard shows them side by side.
   - `SEED = 42`, the seed of every booster and every random draw, and the parameters the
     registry references (Behaviour 11): `LGBM_PARAMS` / `XGB_PARAMS` / `LINEAR_PARAMS` /
     `RF_PARAMS` (fixed) and `LGBM_GRID` / `XGB_GRID` / `LINEAR_GRID` / `RF_GRID` (tuned).
     `LGBM_PARAMS` fixes `subsample_freq = 1`, `RF_PARAMS` `n_jobs = -1`.
   - `USE_GPU = True`: XGBoost runs on an NVIDIA GPU (`XGB_PARAMS["device"] = "cuda"`) when
     `gpu_found()` detects one, else on the CPU; the cell prints which. LightGBM's PyPI build has
     no GPU support, so it always runs on the CPU. GPU and CPU runs build slightly different
     XGBoost trees; `USE_GPU = False` reproduces a CPU run.
   - `MODELS`: one entry per model, keyed `sarimax_fourier`, `lgbm_direct`, `lgbm_hybrid`,
     `xgb_direct`, `xgb_hybrid`, `linear_direct`, `random_forest_hybrid`. Each entry holds:
     - `enabled` (`False` for `sarimax_fourier` by default, `True` for the other six)
     - `load_saved`: `False` (fit, the default for all), `"config"` or `"results"` (Behaviour 34)
     - `family`: `sarimax`, `lightgbm`, `xgboost`, `linear` or `random_forest`; it decides how
       the model is fitted
     - `architecture`: `direct` or `hybrid` (Behaviour 17); the linear model is direct only, the
       random forest hybrid only; `None` for `sarimax_fourier`
     - `params`: fixed parameters
     - `grid`: a small tuning grid (Behaviour 11); empty for `sarimax_fourier`
     - `label`: display label
     - `color` (the colour-map pattern the draft asked for)

     The keys are the values of the `model` column in both exports. The rows outside the registry
     sit in `FIXED`, with fixed labels and colours: `actual`, `smard` (scoreboard export only,
     drawn dashed) and `seasonal_naive`, which also holds the naive lag (`7 days`).
   - `SEARCH`: how a model's grid is searched on the validation walk-forward (Behaviour 11):
     `method` (`"grid"`: every configuration; `"random"`: `n_iter` draws with `SEED`, the
     default), `n_iter = 30` (capped at the grid size for a grid of lists), `n_jobs = -1` (folds in
     parallel) and `verbose = 2` (sklearn's progress output).
   - `USE_FEATURE`: one switch per feature group of the boosters, the linear model and the random
     forest (Behaviour 18), so the team can adjust the set later without touching model code.
     `FEATURE_WINDOWS` holds the durations the groups use (the same-hour lags, the recent-error
     window, the two rolling windows and the year period of the cyclical terms), and
     `FEATURE_GROUPS` the model-input columns each group adds. Both dicts end with a commented
     `my_group` template line.
   - `INTERVAL`: `level = 0.95`.
   - `PLOT_MODEL` and `PLOT_SPLIT_METHOD`: which model and split method the forecast plot shows.
     Defaults: `PLOT_MODEL = None`, which the plot cell resolves to the **registry model** with the
     lowest test MAE (never SMARD or seasonal naive), and `PLOT_SPLIT_METHOD = "rolling"`, which
     falls back to `static` when rolling is switched off. Setting a registry key, e.g.
     `"lgbm_hybrid"`, overrides the default.
   - `EXPORT_ENABLED = True`: the export toggle (Behaviour 28).
   - `SAVE_MODELS = False`: saves every fitted model (Behaviour 34).
4. Load `data/metrics/smard_forecast_errors_hourly.csv` into its own frame, `smard_errors`, never
   merged into `time_series`. Fail with a clear message naming
   `notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` in three cases: the file is
   missing, it lacks a needed column, or its timestamps do not match `time_series.index` exactly
   (a stale export from another snapshot).

   The §1.5 self-check asserts the loaded data's structure and the configuration's consistency
   (registry keys and families, the linear model direct only, a hybrid sharing its direct
   variant's grid where the family has a direct variant, grid values that are
   non-empty lists or, under random search only, `scipy.stats` distributions, the `SEARCH` keys,
   `USE_FEATURE` and `FEATURE_GROUPS` naming the same groups).

### Forecast setting and leakage

5. Implement the forecast setting from the Goal as one table, `FORECAST_SETTING`: per delivery
   day its `issue_time`, `cutoff` and `target_hours`. `ROW_FORECAST_SETTING` holds the same times
   per row, and `available(stamps, cutoff)` applies the actuals rule. Every model, feature and
   training row goes through them.
6. **Feature availability rules.** Actual-derived features may use only observations available
   at the row's issue time (Forecast setting). SMARD forecasts for the target hour are available
   (published by 18:00 on `DAY−1`). Calendar and cyclical features are always available. The
   `renewables_history` columns use the first same-hour lag (`DAY−2`), and the `rolling_stats`
   windows end at the cutoff by definition. The §2.2 cell checks
   every same-hour lag in `FEATURE_WINDOWS` against the rule for every row of the record, prints
   each lag's smallest margin to the cutoff, and stops the notebook if a lag reaches past it (e.g.
   `DAY−1`, whose afternoon and evening come after 16:00).
7. **Capacity look-ahead.** The `cap_*` columns are a yearly value. The project treats year `Y`'s
   value as **published on 1 January of `Y`, 00:00** (team decision). A row uses the latest
   capacity value published at or before its issue time.
   - Most rows therefore use the current calendar year's value.
   - Delivery day 1 January is issued at 18:00 on 31 December, before the new year's value is
     out, so it uses the previous year's value. State this; it is the reason the rule is a
     publication time and not a fixed year offset.
   - Rows with no published value yet (only the record's first day, issued before the first
     publication) have an empty capacity feature. That day is unforecastable anyway
     (Behaviour 9), and the default 36-month windows never reach it.
8. **Leakage test by truncation** (§5.4). For a fixed-seed sample of delivery days drawn from the
   training, validation and test windows (`LEAKAGE_SAMPLE_DAYS = 20`, spread evenly over the three
   windows and drawn with `SEED`), plus the two days where leakage is easiest to get wrong (the
   latest 1 January and the latest spring-DST day):
   - rebuild each day's feature rows with `build_features` from data **cut off at that day's
     availability cutoff** (`cut_at_cutoff`): the `MEASURED_COLUMNS` and SMARD's errors up to the
     cutoff, SMARD forecasts up to the end of `DAY`
   - assert that the rebuilt rows are identical to the rows in `FEATURE_TABLE`
   - when `sarimax_fourier` is switched on: for the sampled validation and test days, assert that
     its forecast from the fitting run equals a dynamic prediction from the same fit applied to
     the series cut off at the cutoff (tolerance `1e-6` MWh)

   Any feature that peeks past the cutoff changes when the future is removed, so this one test
   catches leakage without tracking per-feature timestamps. A new feature built from a raw measured
   column that is new to the record needs that column added to `MEASURED_COLUMNS`, or the test
   cannot mask it. The cell also asserts four fitting rules:
   - Tuning and every validation fit never train on a test day; the KPSS OLS (Behaviour 16) sees
     no validation or test row.
   - The test year is forecast but never used for selection: tuning forecasts validation hours
     only.
   - Every fit trains only on days before its fit day, and the fits run in time order; no
     shuffling happens anywhere.
   - Every search fold (Behaviour 11) trains only on hours before those it scores, and scores
     validation hours only. The folds never reach the fit log, so they are checked directly.
9. **Unforecastable days.** Delivery day `DAY` is unforecastable, for **every** model, as soon as a
   SMARD component forecast is missing for any hour of `DAY` or for the remaining hours of `DAY−1`
   after the cutoff (SARIMAX needs those as exogenous inputs). Such a day has no input rows. It is
   a **data exclusion**, not a model failure (Behaviour 21): excluded from training and scoring,
   counted from the data and never hardcoded. The record's first day is excluded for the same
   reason: its `DAY−1` lies outside the record. `EXCLUSION` holds the reason per day and
   `FORECASTABLE` the result. In the current snapshot this is 2020-01-31 (and the day after it),
   which lie outside the default windows.

### Windows and split methods

10. **Test window**: the last **365 complete delivery days**, ending with the last day whose hours
    all have the actual and all SMARD forecasts observed, derived from the data. When the record
    ends at 23:00 this is exactly spec 04's `trailing_365` window; otherwise the incomplete last
    day is left out (the window table shows the first and last day). **Validation window**: the
    365 delivery days immediately before the test window. The table also counts each window's
    unforecastable days and days without a complete actual.

    **Training rows of a fit** (`training_days(fit_day)`): the delivery days inside the
    `WINDOWS["train"]` window (36 months) ending at the fit's cutoff whose rows are all observed by
    that cutoff, and that are trainable (forecastable, a full clock span, a complete actual). The
    five runs and their fit days sit in `RUNS`: `validation_search` (the search folds, one fit
    every `tune_every`), `validation_rolling` (the selected configuration's walk-forward with the
    test year's `refit_every`; calibrates the rolling band), `validation_static` (calibrates the
    static band), `test_static` and `test_rolling`. The notebook stops if a first fit's training
    window starts before the record.
11. **Tuning**: a **rolling-method walk-forward over the validation year**, the same procedure as
    the rolling method (Behaviour 13) applied to the validation year: the first fit is issued for
    the first validation day on the training window ending at its cutoff, then a refit every
    `tune_every` (30 days: 13 folds), independent of the 7-day `refit_every`.
    - **Boosters, the linear model and the random forest**: `search` runs `SEARCH` over the
      folds of `validation_search`, with sklearn's `GridSearchCV` (every configuration) or
      `RandomizedSearchCV` (`n_iter` draws with `SEED`, the default). `walk_forward_folds` turns
      the fit schedule into the `cv=` folds: one fold per search fit, its training rows and its
      forecast hours with an actual, as positions in `FEATURE_TABLE`. The search runs with
      `refit=False` and `error_score=np.nan`.
    - The selection criterion is residual-load **MAE** on validation hours, **hour-weighted** over
      the folds, i.e. pooled over every validation hour; sklearn's plain fold mean would weigh the
      short last fold like a full one. A configuration with a failed fold gets no MAE.
    - The search returns no forecasts, so the winner runs once more through `run_split` on
      `validation_rolling`, with the test year's `refit_every`. Its validation residuals calibrate
      the rolling band (Behaviour 20).
    - **`sarimax_fourier`** has no grid and no search: `tune_fixed(model_key, config)` runs its
      fixed parameters (Behaviour 15) once through `validation_rolling`, which produces the
      residuals for its rolling band. It does the same for a saved configuration (Behaviour 34). **Seasonal naive** needs no run: its validation-year forecast is computed
      directly.
    - Default grids, searched with 30 random draws each:
      - LightGBM (17,496 configurations): `learning_rate ∈ {0.02, 0.05, 0.1}`, `n_estimators ∈
        {300, 600, 1000, 1500}`, `num_leaves ∈ {7, 15, 31}`, `min_child_samples ∈ {20, 50, 100}`,
        `subsample ∈ {0.7, 0.85, 1.0}`, `colsample_bytree ∈ {0.5, 0.7, 1.0}`, `reg_lambda ∈
        {0, 1, 10}`, `linear_tree ∈ {False, True}` (a linear model in each leaf, which can
        extrapolate) and `linear_lambda ∈ {1, 10, 100}`
      - XGBoost (5,832 configurations): `learning_rate ∈ {0.03, 0.05, 0.1}`, `n_estimators ∈
        {600, 1000, 1500}`, `max_depth ∈ {2, 3, 4, 6}`, `min_child_weight ∈ {1, 5, 20}`,
        `subsample ∈ {0.7, 0.85, 1.0}`, `colsample_bytree ∈ {0.5, 0.7, 1.0}`, `reg_lambda ∈
        {5, 20, 50}` and `objective ∈ {reg:squarederror, reg:absoluteerror}`
      - Ridge: `alpha ∈ {0.01, 0.1, 1, 10}` (4 configurations, searched in full)
      - random forest (288 configurations): `n_estimators ∈ {300, 600}`, `max_depth ∈ {None,
        12, 20}`, `min_samples_leaf ∈ {1, 5, 20, 50}`, `max_features ∈ {0.33, 0.5, 0.7, 1.0}`
        and `max_samples ∈ {0.5, 0.7, None}`; the `absolute_error` split criterion is left out,
        it is far too slow in sklearn's forest
      - the hybrid variant of a family uses the same grid as its direct variant, where one exists
        (asserted in §1.5)
      - under random search a grid value may also be a `scipy.stats` distribution, e.g.
        `stats.loguniform(0.01, 0.2)`
    - §5.2 prints each model's validation MAE and search seconds for the selected configuration,
      and the frozen parameters.
    - **Everything selected is frozen**: hyperparameters are chosen once on the validation
      year, printed, and reused unchanged by both split methods on the test year. Refits
      re-estimate coefficients or re-train on the new window; they never re-run the search.
      This keeps static vs. rolling a clean "refit or not" comparison.
    - After tuning, the test-year fits use training windows that **include the validation year**.
      Tuning only chose the configuration; the validation data is not held back from the final
      fits.
    - `evaluate_model(model_key, config=None)` is the entry point per model: tune once (or take a
      saved `config`, no search), freeze the winner, run every enabled run with it.

    **Split engine** (§3.3). Each model family supplies one fit function, registered in
    `FAMILY_FIT`: `fit(model_key, config, train_days, forecast_days)` returns `predict(day)`, which
    forecasts every target hour of one delivery day. `fit_schedule(run)` assigns each forecastable
    day of a run to the latest fit at or before it; `run_split` calls the fit per scheduled fit,
    forecasts its days, applies the failure rules (Behaviour 21) and logs one row per fit.
12. **Static method**: one fit per model, issued for the first test day, on the training window
    ending at that fit's availability cutoff. The frozen model forecasts every test day; its
    inputs (SMARD forecasts, lags) update daily, its parameters do not. For SARIMAX, the daily
    update comes from the dynamic prediction starting at each day's cutoff (Behaviour 15).
13. **Rolling method**: the first fit is issued for the first test day, so it is **identical to
    the static fit**. After that, a refit every `refit_every` (7 days) on the sliding training
    window ending at that refit's availability cutoff.
    - The two split methods therefore produce the same forecasts for the first `refit_every` of
      the test year. The notebook says so, since that stretch cannot differ between them.
    - Between refits, the latest model forecasts each day with that day's inputs. For SARIMAX,
      each refit's model is applied once to its training window plus its `refit_every` stretch and
      predicts each day dynamically, as in the static method.
14. Both split methods produce forecasts for **identical test hours**, and the scoreboard shows
    them side by side.

### Models

15. **`sarimax_fourier`**: a regression with ARIMA errors on the training window, with everything
    fixed in its registry entry (the team can edit the values by hand). Switched off by default
    (Behaviour 3); it runs when switched on.
    - **Exogenous inputs** (`sarimax_exog`):
      - the `FEATURE_TABLE` columns named in the registry's `exog` list; any feature from
        Behaviour 18 can be named there. The team's default is `["fc_gen_wind_solar", "holiday"]`;
        the first draft of this spec also named `fc_grid_load`.
      - Fourier terms for the periods and orders in the registry's `fourier` dict (period →
        order `K`): a **1-day** and a **1-week** period, as durations, with `K_day = 4` and
        `K_week = 3`, computed on the local clock
    - **Error model**: ARIMA with the fixed `order = (1, 0, 1)` and `trend = "c"` (the
      regression's intercept). There is no order search.
      `d = 0` because the SMARD forecasts and the Fourier terms already carry the level and the
      seasons; what remains is roughly SMARD's own error, which comes in episodes but reverts to
      its mean. With `d = 1` the model would carry the level of the last actual forward over the
      9–32 h to the target hours. The model therefore reads as "regression on SMARD's forecasts
      plus seasons, with AR/MA errors that carry SMARD's recent error forward".
    - **Series**: the model takes the local-time rows of `time_series` as they are, as evenly
      spaced steps. It uses no UTC grid and no `NaN` filling (Edge cases).
    - **Forecast by dynamic prediction**: a fitted model is applied **once** to its training
      window plus the stretch it forecasts (statsmodels `apply` with the fixed parameters, no
      re-estimation); the training window provides the history the model's state starts from.
      For each day `DAY`, a dynamic prediction starts at the **first row not yet available**, the
      hour stamped 16:00 on `DAY−1`, and runs to the end of `DAY`, using the exogenous inputs for
      the remaining hours of `DAY−1` and for `DAY`. Only `DAY`'s hours are kept. A
      dynamic prediction from a start point uses only observations before that point, so no daily
      state updates are needed; the leakage test (Behaviour 8) confirms it.
    - **Excluded days inside the series**: an unforecastable day (Behaviour 9) inside SARIMAX's
      training window or forecast stretch has no SMARD forecast to use as an input. Its rows are
      dropped and the series continues as if they were a gap, like the DST hour (Edge cases). The
      notebook states, for the default windows, how many rows were dropped, how many spring-DST
      hours are skipped as a single step, and the number of exogenous inputs.
16. **Stationarity check**. One printed KPSS test on the residuals of an OLS fit of residual load
    on `sarimax_fourier`'s exogenous inputs, over the first validation fit's training window (the
    `WINDOWS["train"]` before the validation start). It runs whether or not `sarimax_fourier` is
    switched on. It is a **check, not a decision**: `d` stays fixed at 0. If KPSS rejects stationarity, the notebook says so and the
    team can change `d` in the registry. A short markdown note states three points:
    - Differencing, if ever needed, happens inside ARIMA through `d`.
    - Tree models need no stationarity but **cannot extrapolate** beyond the target range they
      were trained on. LightGBM's `linear_tree` (a grid value) fits a linear model in each leaf
      and can extrapolate partly.
    - Box-Cox or log transforms are ruled out because residual load is negative in some hours.
17. **Boosters, the linear model and the random forest**: LightGBM and XGBoost regressors with
    the fixed seed `SEED`, each in two architectures, a Ridge regression (direct only) and a
    random forest (hybrid only), built as scikit-learn pipelines by
    `build_pipeline(model_key, config)` and fitted by `fit_booster`, which forecasts all of a
    fit's days at once. `BOOSTER` maps each family to its estimator; the names (`BOOSTER`,
    `fit_booster`, the pipeline step `booster`) are kept for the linear model and the forest too.
    - **Direct** (`select → booster`): the booster predicts `residual_load` from the enabled
      feature groups (Behaviour 18). `select` is a pass-through `ColumnTransformer` that keeps the
      enabled `USE_FEATURE` columns of `FEATURE_TABLE`, in order.
    - **Hybrid** (`HybridRegressor(residual=select → booster, linear_inputs=LINEAR_INPUTS)`): the
      **linear stage** (stage 1) is an ordinary least-squares model (sklearn `LinearRegression`,
      with intercept) on `fc_grid_load`, `fc_gen_wind_solar` and a linear time trend (days since
      the first training day, learned at fit). It is the "parametric model captures trend" part of
      the draft. The **residual stage** (stage 2) is the `select → booster` pipeline, fitted on
      stage 1's in-sample residuals. The forecast is stage 1 + stage 2.
    - **Stage 1's inputs are fixed** (`LINEAR_INPUTS`). The linear stage reads them from the full
      row before `select`, so the `USE_FEATURE` switches affect only the direct booster and
      stage 2; stage 1 always uses the two SMARD forecasts and the trend, even if the
      `smard_forecast_*` groups are switched off.
    - **Linear** (`select → prepare → Ridge`, `linear_direct`): the same enabled feature groups as
      the direct boosters. `prepare` one-hot encodes `hour`, `dow` and `month`
      (`CALENDAR_CATEGORIES`), fills empty values with the median and scales the other columns.
      It is fitted inside the pipeline, so the medians and scales come from each training window
      only; learned on the whole `FEATURE_TABLE` they would leak the test year. Families that need
      this step are listed in `PREPROCESS` (`{"linear"}`). Without the imputation a rolling refit
      fails on the spring-DST lag gaps. The §1.5 self-check forbids a linear hybrid; the §4.6
      self-check runs `select → prepare` on a real training window and asserts the calendar
      one-hot columns, no empty value left and the expected width.
    - **Random forest** (`random_forest_hybrid`): sklearn's `RandomForestRegressor` as the residual
      stage of a hybrid (`select → booster`). It handles empty values itself, so it needs no
      `prepare` step. `SerialPredictForest` fits it on every core (`n_jobs = -1`) but predicts on
      one: in parallel the tree predictions are summed in a varying order and differ by about
      1e-11 MWh between two calls, which the exact comparison of static and rolling in the §5.5
      self-check would catch. The forest runs inside the parallel search folds without slowing
      them down.
    - **No direct random forest.** It was tested with the same grid and search and rejected: a
      forest forecasts an average of training targets, so it is capped in the low tail even harder
      than the direct boosters (test `low_extreme` skill −20.4 % rolling and −63.8 % static by
      actual; the static run never forecast an hour at or below P1), and it added nothing on
      overall MAE (validation 2,254 MWh against 2,138–2,165 MWh for the boosters). The notebook's
      §4.5 states this with the numbers.
    - A grid parameter reaches the estimator as `booster__<name>` (direct) or
      `residual__booster__<name>` (hybrid), recorded in `PARAM_PREFIX`.
    - **No early stopping.** `n_estimators` is a value in the tuning grid. Early stopping would
      need a third time-ordered split inside each training window, which this spec does not
      define.
    - A short markdown note states the expected difference between the two: the hybrid can
      extrapolate the level through stage 1, while the direct booster is capped at its training
      range. This matters in the low tail, where the test year may reach more negative values
      than the training window. Scoreboard B shows it as a positive bias in `low_extreme` and as
      few forecast hours in that bin (Behaviour 24). With `linear_tree = True` the direct LightGBM
      is only partly capped; the linear model is not capped either, it extrapolates through every
      input.
18. **Minimal feature set** for the boosters, the linear model and the random forest, as
    toggleable groups in `USE_FEATURE`, with their columns in `FEATURE_GROUPS`. The groups are
    marked **provisional**: the team's feature-engineering work may replace them.

    | Group | Columns | Available because |
    |---|---|---|
    | calendar | `hour`, `dow`, `month`, `is_weekend`, `holiday` (local time) | always known |
    | smard_forecast_grid_load | `fc_grid_load` for the target hour | published by 18:00 `DAY−1` |
    | smard_forecast_renewables | `fc_gen_wind_solar` for the target hour | published by 18:00 `DAY−1` |
    | lags | `rl_lag_48h`, `rl_lag_168h`: `residual_load` at the same local hour on `DAY−2` and on `DAY−7`; `rl_last_available`: the last available `residual_load` at the cutoff | before the cutoff |
    | recent_smard_error | `err_grid_load_recent`, `err_renewables_recent`: mean `err_grid_load` and mean `err_renewables` over the 24 h ending at the cutoff, read from `smard_forecast_errors_hourly.csv` | before the cutoff (same 2 h actuals lag); tests the reference notebook's observation that SMARD's error comes in episodes |
    | capacity | `cap_total` = `cap_wind_off + cap_wind_on + cap_solar` under the publication rule (Behaviour 7) | published 1 January of its own year |
    | cyclical (off) | `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos`: sine and cosine of the local time of day, the weekday and the day of year (over `FEATURE_WINDOWS["year_period"]` = 365.25 days); `season_code` (winter 0 .. autumn 3) | always known |
    | renewables_history (off) | `capacity_factor_lag_48h`: wind + solar capacity factor (`renewables / cap_total` at that hour); `err_renewables_lag_48h`: SMARD's wind + solar error; both at the same local hour on `DAY−2` (the first same-hour lag) | before the cutoff; the capacity at the `DAY−2` hour is published by then |
    | rolling_stats (off) | `rl_roll_mean_72h`, `rl_roll_std_168h`: mean of `residual_load` over the 72 h and standard deviation over the 168 h ending at the cutoff | before the cutoff |

    The first six groups are on by default: 13 columns. The last three are spec 06.1's
    cutoff-safe versions of spec 05's features ([06.1-spec05-features.md](06.1-spec05-features.md)),
    **off by default** (*Additions after the refactor*); their names differ from
    `regression-models-magc.ipynb` (`capacity_factor_lag_48h` is `vre_cf_lag_48h` there, and
    `renewables_history` + `rolling_stats` are its single group `fe_history`). `is_weekend` and
    `holiday` belong to `calendar` only. The lag, history and rolling column names derive from the
    durations in `FEATURE_WINDOWS` (e.g. 2 days → `rl_lag_48h`). A check cell after §4.1 asserts
    the value ranges of the spec 06.1 groups (sine / cosine in [−1, 1], valid season codes,
    capacity factor in [0, 1]) and recomputes their columns from spec 06.1's definitions on the
    test hours.

    - **Built before the pipeline, not in it.** `build_features(days, frame, errors)` builds
      **every** group, whatever `USE_FEATURE` says, into `FEATURE_TABLE`: one row per target
      hour, each built as at its own issue time. Lags, recent errors and capacity depend on the
      cutoff and on other rows of the record, which a pipeline step does not see. No feature
      learns from the training data, so building them once is leakage-safe. The data are
      arguments, so the leakage test (Behaviour 8) can rebuild rows from cut data.
    - Lags are durations, looked up **by timestamp** (`hour − lag`), never by row position: the
      record skips the spring-DST hour. A lag whose source hour does not exist, such as local 02:00
      on a spring DST day, is `NaN` in `FEATURE_TABLE`; the boosters and the random forest handle
      missing values natively, the linear model fills them inside its pipeline (Behaviour 17).
      No row is dropped.
    - The rolling windows need every hour of the window: a window that contains a spring-DST
      switch is one hour short, so `rl_roll_mean_72h` stays empty for 3 days and
      `rl_roll_std_168h` for 7 days after each switch. They are handled like the lags.
      `recent_smard_error` accepts a partial window.
    - **Adding a feature group**: a switch in `USE_FEATURE` and its columns in `FEATURE_GROUPS`
      (§1.3), the columns at the end of `build_features` in the same order (§4.1; the §4.6
      self-check compares the order), and, for a raw measured column new to the record, an entry
      in `MEASURED_COLUMNS` (Behaviour 8).
19. **Seasonal naive `DAY−7`**: the actual `residual_load` at the same local hour seven days
    earlier. It needs no fitting and is identical in both split methods, so it appears once in the
    scoreboard and the exports, with `split_method = "none"`.

### Prediction intervals

20. **One empirical method for every model** (split-conformal style), seasonal naive included:
    - **Residual sign**: the band uses `r = actual − forecast`, the opposite sign to the project's
      `error = forecast − actual`. State this next to the formula.
    - **Calibration residuals, per split method**: each split method is calibrated the way it is
      used.
      - rolling: the residuals of the validation-year walk-forward (Behaviour 11) of the
        **selected** configuration
      - static: one extra fit per model at the start of the validation year, with the frozen
        configuration, forecasting the whole validation year without refitting; its residuals
        calibrate the static band. A model frozen for a year makes larger errors than a refitted
        one, so rolling residuals would make static bands too narrow.
      - seasonal naive needs no fit; its validation-year residuals calibrate its single row.

      `CALIBRATED_ON` maps each test run to its calibration run (`test_static` →
      `validation_static`, `test_rolling` → `validation_rolling`).
    - **Quantiles**: take the `(1 − level) / 2` and `(1 + level) / 2` quantiles of `r` **per local
      hour of day**, since error size varies strongly with the hour (`band_quantiles`;
      `level = 0.95` gives the 0.025 and 0.975 quantiles).
    - **Band**: `forecast + [q_low(r), q_high(r)]` for each test hour. If a model is strongly
      biased at some hour, both quantiles share a sign and the band does not contain the forecast
      itself. That is correct behaviour, not an error.
    - The band is calibrated on the validation year and not updated during the test year, so
      drift shows up as lost coverage.
    - SMARD is a point forecast and gets no band.
    - `FORECASTS` holds, per (model, split method), the test-hour forecast with `lower` / `upper`;
      it is the source of the hourly export (Behaviour 28).

### Evaluation and scoreboard

21. **Scoring hours.** Every row is scored on the **common hours**: test hours where the actual,
    SMARD's forecast, seasonal naive's forecast and every enabled registry model × split method
    forecast all exist (a strict intersection). SMARD is **re-scored from `smard_forecast_errors_hourly.csv`** on these hours,
    not recomputed from `data/smard.csv`. A cross-check confirms the two agree.
    - **Failure** means a fit or forecast **raises an error**. A failed forecast is never
      silently replaced by another model's forecast or a fallback.
      - **Failed daily forecast**: that day's forecast stays empty, and its hours drop out of the
        common hours.
      - **Failed static fit**: the model × split method has no forecasts at all. It is reported as
        failed, gets no scoreboard row, and is left out of the common-hours intersection, so it
        cannot empty every other row's evaluation set.
      - **Failed rolling refit**: the days until the next successful refit stay empty. The
        previous model does not keep forecasting, because that would be a fallback.
    - A **convergence warning** (e.g. statsmodels' `ConvergenceWarning`) is not a failure: the
      forecast is kept.
    - §5.3 shows, per model × run, the number of fits, failed fits and failed days and the seconds
      spent fitting and forecasting, and lists each failed fit with its error message. Warnings
      raised inside a fit are not printed; the fit log `FIT_LOG` counts them per fit
      (`convergence_warnings`, `other_warnings`). §7.1 prints how many test hours the common hours
      lost. If one model breaks badly, the team switches it off in the registry and re-runs.
22. **Metric definitions** follow spec 04:
    - `error = forecast − actual`
    - MAE, RMSE (recomputed from hourly errors) and bias; the hour count stands in the header line
      above each table and in the export's `count` column
    - no MAPE
    - `skill = 100 · (1 − MAE_model / MAE_SMARD)` on the same hours, in `%`; positive means better
      than SMARD
23. **Scoreboard A — accuracy** (§7.2): one row per enabled model × split method, plus SMARD and
    seasonal naive. Columns:
    - MAE, RMSE, bias, skill vs. SMARD (the hour count in the header line)
    - **months beating SMARD**: `k`, the number of full calendar months in which the row's MAE is
      below SMARD's on the common hours; `n`, the number of calendar months that lie entirely
      inside the test window, stands in the header line. The window rarely starts on the 1st, so
      `n` is usually 11. This is the robustness view in place of a significance test.
    - `fit_seconds` (summed over the split method's **test-year fits only**)

    In the SMARD row, skill, months beating SMARD and `fit_seconds` are shown as "-": they compare
    a row against SMARD, or it has no fit. Seasonal naive has no `fit_seconds` either.

    Tuning time is reported separately: §5.1 prints each model's total tune-and-run time, §5.2
    the search seconds of the selected configuration. Sorted by MAE; the other scoreboards follow
    this row order.

    Each scoreboard is built as long records (`ACCURACY_RECORDS`, `EXTREMES_RECORDS`,
    `INTERVAL_RECORDS`: model, split method, table, metric, value, count) and displayed as a pivot
    by `scoreboard_table`; the records concatenate into `SCOREBOARD`, the scoreboard export.
    Display headers may be friendlier than the export's metric names (e.g. "skill vs SMARD (%)"
    for `skill_pct`).
24. **Scoreboard B — extremes** (§7.3), on the same rows:
    - **Tail bins.** Edges are the `{1, 25, 75, 99} %` quantiles of the **actual** residual load
      over the test window. They are computed once, printed in `MWh` and shared by every row. The
      bins are named after the two risk cases and the ordinary hours between them:
      `low_extreme` (`≤ P1`), `ordinary` (`P25–P75`) and `high_extreme` (`> P99`). Report MAE
      and bias per bin, both **binned by actual** and **binned by the row's own forecast**. The
      forecast-binned table also shows each row's hours per bin: a direct booster capped at its
      training floor puts few hours into `low_extreme` (tree extrapolation, Behaviour 17).
    - **`below_zero`** (actual `< 0 MWh`, `ZERO`) is reported next to the quantile bins, in both
      views (`REPORTED_BINS`, `bin_masks`). It is a fixed threshold, the `zero` basis of
      `risk-definition.ipynb`: unlike P1 it measures the same physical situation (renewable
      oversupply) in every year. It overlaps `low_extreme`.
    - **Skill vs. SMARD** is reported where a row and SMARD are scored on the same hours or days:
      per bin binned by actual, and for the day maximum and minimum. Binned by forecast, every row
      fills a bin with its own hours, so a skill there would compare different hours; it has none.
    - **The regression-to-the-mean rule applies**: only a tail difference that shows in both
      views counts.
    - **Day maximum and day minimum** of residual load, computed from the **common hours** only.
      A day counts if its common hours cover at least `DAY_COMPLETENESS` (23/24) of its expected
      hours. Report value-error MAE, bias and skill, with the day count in the table title.
    - Export metric names:
      `{MAE,bias}_{low_extreme,below_zero,ordinary,high_extreme}_by_{actual,forecast}`,
      `skill_pct_{low_extreme,below_zero,ordinary,high_extreme}_by_actual` and
      `{MAE,bias,skill_pct}_day_{max,min}`. SMARD's rows have no skill. Exports written before
      2026-09-30 have no `below_zero` metrics.
25. **Scoreboard C — intervals** (§7.4): coverage (the share of common hours whose actual lies
    inside the band) against the nominal 95 %, `coverage − nominal` in percentage points
    (negative = band too narrow), and mean band width in `MWh`, per model × split method.
    SMARD has no band and therefore no row here.
26. **Static vs. rolling** (§7.5): one small table with each model's static and rolling MAE and
    the difference `rolling − static`, in `MWh` and in `%`, from Scoreboard A. Negative means
    refitting helped. With one split method switched off, the cell says both are needed.
27. **Forecast plot with the band**: one figure for the model and split method set in
    `PLOT_MODEL` / `PLOT_SPLIT_METHOD` (defaults and fallback in Behaviour 3; the plot title names
    the model and split method shown), with two panels:
    - the local Monday–Sunday week of the test window containing the **highest** actual
      residual-load hour
    - the week containing the **lowest**

    Both weeks are derived from the data, and the rule and dates are printed. A week that extends
    beyond either end of the test window is clipped at the window's edge. Each panel shows the
    actual (in black), SMARD's forecast (dashed), the model's forecast and its 95 % band. The plot has a
    title and axis labels in `MWh`. Changing `PLOT_MODEL` and re-running the cell shows another
    model.

### Export

28. **Export toggle.** Both files are written only if `EXPORT_ENABLED` is `True`, the default
    since 2026-09-30 (it was `False` while the team experimented with the models).
    - The notebook always builds the two export frames in memory, so the scoreboards and the
      self-check work with the toggle off.
    - With the toggle off, the export cell writes nothing. It prints that the export was skipped
      and names the paths it would write.
    - With the toggle on, the notebook writes both files to `data/models/`. The directory is
      assumed to exist: it is kept in git by an empty `.gitkeep`, and its CSVs are ignored by the
      `data/models/*.csv` rule in `.gitignore`. The notebook does not create it.

    `data/models/model_forecast_errors_hourly.csv`, long format, one row per test hour × model ×
    split method, for the registry models and seasonal naive. **SMARD is not in this file**: its
    hourly values already live in `data/metrics/smard_forecast_errors_hourly.csv`. Columns: `timestamp`, `model`, `split_method`, `residual_load`, `forecast`,
    `lower`, `upper`, `err_residual_load`. Missing values stay empty. Timestamps are plain local
    time, like the `data/metrics/smard_*` files, so the two join on `timestamp`.
29. `data/models/model_scoreboard.csv`, long format. Columns: `model`, `split_method`, `table`
    (`accuracy` / `extremes` / `intervals`), `metric`, `value`, `count` (hours or days). It
    includes the SMARD and seasonal-naive rows, both with `split_method = "none"`. The `model`
    values are the keys from Behaviour 3.
    - Both files are plain CSV (`sep=","`, `decimal="."`, UTF-8), like the other derived exports.

### Resolution independence

30. Every window, lag, refit interval, Fourier period, issue time and publication lag is a
    duration. At a sub-hourly resolution, models are compared with each other at native
    resolution. They are compared with SMARD only after aggregating to hourly means, because
    `smard_forecast_errors_hourly.csv` holds hourly errors. State this rule; do not implement a
    sub-hourly run.

### Conclusions and checks

31. Close with one **"Did we beat SMARD?"** section, the notebook's only interpretation. It is
    written for the **default configuration** and says so at the top, because the numbers go
    stale as soon as the team changes a setting. It states:
    - the best **registry model** × split method by MAE (never SMARD or seasonal naive) and its
      skill vs. SMARD, on the common hours, with the hour count; if it does not beat SMARD, the
      section says so plainly
    - how that model does in the tails and on day max / min compared with SMARD; tail bins in a
      365-day window hold only about 88 hours each, so tail statements stay qualitative
    - whether its interval is calibrated (its coverage against 95 %)
    - whether refitting helped (Behaviour 26)
    - the **post-processing framing**: the models use SMARD's component forecasts, so the result
      is an improvement on SMARD, not an independent forecast; and without a SMARD forecast our
      model cannot run
    - the **archive caveats**:
      - `fc_grid_load` may be a revised value rather than the original 10:00 value (SMARD updates
        it whenever it changes by at least 10 %)
      - the actuals are final values rather than first publications; SMARD partly estimates
        generation when data are incomplete, which also affects the newest feature inputs
        (`rl_last_available`, `recent_smard_error`)
      - both affect SMARD and our model alike and cannot be corrected from `data/smard.csv`
    - that the result rests on **one test year**, and whether the monthly win count supports it
32. **Runtime**: §5.1 prints each model's total tune-and-run time and the total over all enabled
    models (`FITTING_SECONDS`), and a warning if that exceeds one hour ("check search method /
    grid / data resolution"). With the default registry (six models, 30 draws over 13 folds)
    fitting takes about 30 min on a 16-core CPU without GPU: `random_forest_hybrid` alone about
    16 min, `lgbm_direct` about 4 min, `linear_direct` a few seconds. The notebook should run in
    about an hour or less on a desktop. If it clearly exceeds that, raise it with the team rather
    than silently shrinking windows or grids.
33. A **closing self-check cell** (§9.1), in addition to the per-section self-checks (Notebook
    structure). It runs against the in-memory export frames. With `EXPORT_ENABLED` on, it also
    reads both files back with a bare `pd.read_csv` and repeats the export checks
    (`export_checks`) on the files as written. The checks:
    - the leakage test (Behaviour 8) passes
    - test and validation windows do not overlap, and no tuning result uses a test hour
    - `err_residual_load == forecast − actual` in the hourly export
    - SMARD's MAE on the common hours equals a direct recomputation from
      `smard_forecast_errors_hourly.csv`
    - scoreboard MAE and RMSE match a direct recomputation from the hourly export
    - `fc_residual_load` is not among any model's features
    - `time_series` still carries exactly `SERIES + DERIVED`, with the row count and time span
      recorded at loading (`LOADED`)

### Saving and loading

34. **Model saves** (§5 markdown, a helper cell before §5.1, `LOAD_PLAN` in §5.1). A save is one
    folder per model, `data/models/<model_key>/`:
    - `config.json`, readable: the registry settings that change the forecasts
      (`model_settings`), the selected configuration, the device, the library versions, the run
      time, and a data snapshot of what it was tuned on (`tuned_on`) and run on (`run_on`): rows,
      span and one hash per calendar month of `time_series[SERIES]` and of the SMARD errors file
    - `results.joblib`: `RESULTS[model_key]` as fitted (forecasts, tuning table, fit logs),
      compressed; fitted estimators are not saved

    Per model, `load_saved` decides what §5.1 does:

    | `load_saved` | §5.1 | Stops when the save differs in |
    |---|---|---|
    | `False` | fits; saves with `SAVE_MODELS = True` (replaces an old save) | – |
    | `"config"` | refits every run with the saved configuration, no search (warns if the data differ); saves with `SAVE_MODELS = True` | a model setting, the search, the features or `DATA_INFO` |
    | `"results"` | loads everything, no fit; never rewrites the save | any setting, incl. `WINDOWS` and the split methods, or either input file (the message lists the differing months) |

    - `LOAD_PLAN` checks every enabled model's save before the first fit, so a mismatch stops at
      once and lists the differences.
    - A requested save that does not exist: the model is fitted and saved, with a warning, even
      with `SAVE_MODELS` off.
    - A save is written only once a model's runs are complete; nothing is saved if every
      configuration failed.
    - The CPU / GPU device is recorded, never compared. Other library versions only warn.
    - §5.2's `tuning` column shows per model whether its configuration comes from this run, a
      saved configuration or a loaded save.
    - The saves are committed to git (the `data/models/*.csv` rule does not reach them).
      `results.joblib` is a pickle: the §5 markdown says to load only saves from the team.

## Data

Sources:

| File | Used for |
|---|---|
| `data/smard.csv` | actuals, SMARD component forecasts, capacity (German CSV format, inherited loader) |
| `data/metrics/smard_forecast_errors_hourly.csv` | SMARD's hourly errors, used to re-score it on the common hours and for the `recent_smard_error` feature (plain CSV) |

Series used:

| Column | Role |
|---|---|
| `residual_load` | target |
| `fc_grid_load`, `fc_gen_wind_solar` | features of the boosters, the linear model and the random forest, and the hybrid's linear-stage inputs for `DAY`; SARIMAX's exogenous inputs as named in its `exog` entry (for `DAY` and the rest of `DAY−1`); both decide whether a day is forecastable |
| `fc_residual_load` | SMARD benchmark only, never a feature |
| `err_grid_load`, `err_renewables` (from the SMARD errors file) | `recent_smard_error` feature; `err_renewables` also for `renewables_history`; under the availability rule |
| `cap_wind_off`, `cap_wind_on`, `cap_solar` | capacity feature; year `Y`'s value counts as published on 1 January of `Y`; the `renewables_history` capacity factor uses the value at its `DAY−2` hour |
| `renewables` (derived, `wind_on + wind_off + solar`) | the `renewables_history` capacity factor |

Derived in this spec:

| Derived | Definition | Unit | Persisted? |
|---|---|---|---|
| `FEATURE_TABLE` | every feature group's columns, one row per target hour, built at that row's issue time | mixed | no |
| forecasts, bands | per model × split method × test hour | MWh | exported (hourly CSV) |
| `err_residual_load` | forecast − actual | MWh | exported (hourly CSV) |
| scoreboard metrics | MAE, RMSE, bias, skill, monthly wins, tail, day-extreme and interval metrics | MWh / % / count | exported (scoreboard CSV) |
| tail-bin edges | P1 / P25 / P75 / P99 of actual residual load over the test window | MWh | printed only |

Exported artifacts:

| File | Grain | Format |
|---|---|---|
| `data/models/model_forecast_errors_hourly.csv` | test hour × model × split method | `sep=","`, `decimal="."`, UTF-8 |
| `data/models/model_scoreboard.csv` | long: model × split method × table × metric | `sep=","`, `decimal="."`, UTF-8 |

Both are written only with `EXPORT_ENABLED = True` (the default). `data/models/` is tracked
through an empty `.gitkeep`; its CSVs are ignored by the `data/models/*.csv` rule in
`.gitignore`. The existing `data/*.csv` rule does not reach subfolders, which is why the separate
rule exists, as for `data/metrics/` and `data/risk_classification/`. The model saves in
`data/models/<model_key>/` (Behaviour 34) match no rule and are committed.

## Edge cases

- **DST and SARIMAX.** SARIMAX takes the local-time rows as evenly spaced steps. The missing
  spring 02:00 is then one skipped step per year, and the autumn fold, which SMARD collapses into
  one row, needs nothing. The time spacing is therefore off by one hour twice a year, which is
  negligible for hourly load; the notebook states it once. Rows of an unforecastable day inside
  SARIMAX's series are dropped the same way (Behaviour 15). Fourier terms and calendar features
  use local time, because load follows the local clock.
- **Spring DST target day.** It has 23 target hours. Every model forecasts all local hours of `DAY`,
  so nothing is padded.
- **Seasonal naive across DST.** If `DAY−7` is a spring DST day, the naive forecast for local 02:00
  has no source and stays `NaN`. That hour drops out of the common hours.
- **Missing SMARD forecast hours.** A missing component forecast for any hour of `DAY`, or for the
  remaining hours of `DAY−1`, makes `DAY` unforecastable for every model: a data exclusion, not a
  model failure (Behaviour 9). The count comes from the data.
- **Record ending mid-day.** If the last jointly observed hour is not 23:00, the incomplete last
  day is left out of the test window (Behaviour 10).
- **Forecasts ahead of actuals.** After a re-fetch the last rows may carry forecasts without
  actuals. The test window ends with the last day whose hours are all jointly observed
  (Behaviour 10), so those rows never enter it.
- **Archive vintages.** `data/smard.csv` holds the final published values: possibly revised load
  forecasts and corrected actuals. Both favour SMARD and our model equally. The caveat is stated
  in Behaviour 31 and cannot be corrected here.
- **Capacity look-ahead.** Handled by the publication rule (Behaviour 7): year `Y`'s value is
  usable from 1 January of `Y`, 00:00. Delivery day 1 January is issued the evening before, so
  it still uses the previous year's value. Only the record's first day has no published value.
  Within-year fleet growth is invisible in a yearly step value.
- **Model failures.** A failed daily forecast leaves that day empty and removes its hours from
  the common hours of every row. A failed static fit takes that model × split method out of the
  scoreboard and the intersection. A failed rolling refit leaves the days until the next
  successful refit empty. A convergence warning keeps the forecast. Failures are counted and
  printed, warnings counted in the fit log (Behaviour 21).
- **Static and rolling coincide at the start.** Both split methods share their first fit, so they
  are identical for the first `refit_every` of the test year (Behaviour 13).
- **Band without the forecast.** A strongly biased hour gives both band quantiles the same sign,
  so the band does not contain the forecast (Behaviour 20). This is expected.
- **Tree extrapolation.** A direct booster cannot predict below the lowest target value in its
  training window (with `linear_tree`, LightGBM only partly). The test year may carry deeper
  negative residual load than the 36 months before it. Scoreboard B shows it through the
  forecast-binned hours in `low_extreme` (Behaviour 24). A direct random forest is capped even
  harder, which is why there is none (Behaviour 17).
- **Parallel forest predictions.** `RandomForestRegressor.predict` with `n_jobs = -1` sums the
  trees in threads, so two calls differ by about 1e-11 MWh. `SerialPredictForest` predicts on one
  thread, so static and rolling stay exactly identical over their shared first fit (Behaviour 17).
- **Empty inputs of the linear model.** Ridge cannot take `NaN`: `prepare` fills the DST lag gaps
  and the rolling-window gaps with the training window's median inside the pipeline, never on
  the whole `FEATURE_TABLE` (Behaviour 17).
- **Regime shift in grid-load bias.** The reference notebook reports that SMARD's grid-load bias
  changes sign between years. A model tuned on the validation year can inherit a correction that
  no longer applies in the test year. The static vs. rolling comparison (Behaviour 26) is where
  this shows.
- **Interval drift.** The band is calibrated on the validation year only. If the error
  distribution shifts, coverage falls below 95 %. Report it; do not recalibrate on test hours.
- **KPSS rejects stationarity.** `d` stays at 0; the notebook reports the result, and changing
  `d` in the registry is the team's call (Behaviour 16).
- **Negative residual load.** No log or Box-Cox transform anywhere.
- **Small tail samples.** About 88 hours per 1 % bin in a 365-day window. Show the hour count,
  and keep the closing statements about tails qualitative.
- **Unforecastable edge days.** If the validation or test window starts with a day that has no
  input (missing SMARD forecast), it is skipped and counted, not back-filled.
- **Missing data files.** A fresh clone has no `data/smard.csv` and no
  `data/metrics/smard_forecast_errors_hourly.csv`. The setup fails with a message naming
  `notebooks/API-connection.ipynb` and `forecast-metrics-claude.ipynb` respectively.
- **Runtime.** SARIMAX on the training window dominates the runtime, which is why it is off by
  default; of the default models, `random_forest_hybrid` is the slowest (about 16 min). The
  registry's `enabled` switches, the grids and `SEARCH` (random search, parallel folds) are the
  lever; the windows are not shrunk silently (Behaviour 32). A loaded save takes seconds
  (Behaviour 34).
- **Stale save after a re-fetch.** A re-fetch changes `data/smard.csv`, so a `"results"` load
  stops and names the differing months. `"config"` still works: it refits the saved configuration
  on the new data and only warns (Behaviour 34).

## Acceptance criteria

### Setup and forecast setting

- [ ] The notebook is `notebooks/05_modeling/regression-models-claude.ipynb` and runs top to
      bottom from a fresh kernel.
- [ ] The shared setup is repeated from `team-EDA.ipynb` §1 and pointed at in one markdown cell.
- [ ] `DATA_INFO`, `WINDOWS`, `TRAIN_TEST_SPLIT_METHOD`, `SEED`, the model parameters and
      grids, `USE_GPU`, `MODELS`, `FIXED`, `SEARCH`, `USE_FEATURE`, `FEATURE_WINDOWS`, `FEATURE_GROUPS`,
      `INTERVAL`, `PLOT_MODEL`, `PLOT_SPLIT_METHOD`, `EXPORT_ENABLED` and `SAVE_MODELS` are configuration cells,
      printed once; no downstream cell hardcodes a value they hold.
- [ ] Every row is built at its own issue time (18:00 on `DAY−1`, 2 h actuals lag by default), and
      the truncation leakage test passes.
- [ ] Capacity follows the publication rule (year `Y` usable from 1 January of `Y`), and the
      1 January consequence is stated.
- [ ] `fc_residual_load` is not a feature of any model.

### Splits and models

- [ ] The test window is the last 365 complete delivery days, and the validation window is the
      365 delivery days before it; both are derived from the data.
- [ ] The validation walk-forward runs for every enabled registry model; every model except
      SARIMAX is tuned over the `validation_search` folds (one every `tune_every`) with
      `GridSearchCV` / `RandomizedSearchCV` (`SEARCH`) by hour-weighted MAE, and the grids match
      the defaults in Behaviour 11 unless the team changed them.
- [ ] Tuning uses only the validation year; the selected configuration per model is printed and
      frozen for both split methods, and refits never re-run the search.
- [ ] The static and rolling methods forecast identical test hours, with equal training windows
      (36 months by default) in both.
- [ ] `sarimax_fourier` is fitted with statsmodels on the local-time rows, with `order = (1, 0, 1)`
      and the Fourier orders fixed in the registry, and forecasts each day by dynamic prediction
      starting at the row stamped 16:00 on `DAY−1`, after one `apply` to its training window plus
      forecast stretch; no order search and no daily state updates. It is in the registry with
      `enabled: False` and runs when switched on.
- [ ] One KPSS check is printed as a check, not a decision, with the short note on
      differencing.
- [ ] LightGBM and XGBoost run direct and hybrid with a fixed seed, Ridge direct (with its
      `prepare` step) and the random forest as a hybrid (predicting on one thread), all as sklearn
      pipelines built by `build_pipeline`.
- [ ] The minimal feature set is implemented as toggleable groups (`USE_FEATURE`), built once into
      `FEATURE_TABLE`, and marked provisional; the three spec 06.1 groups are off by default and
      pass their check cell after §4.1.
- [ ] Seasonal naive `DAY−7` appears as the floor row.
- [ ] Each split method's band is calibrated on its own validation run (static: one frozen fit
      over the validation year).
- [ ] Failures follow Behaviour 21: a failed day stays empty, a failed static fit leaves the
      scoreboard and the intersection, a failed rolling refit empties the days until the next
      refit; failures and the common hours lost are printed, warnings are counted in `FIT_LOG`;
      all other rows are scored on the strict common-hours intersection.
- [ ] Every registry entry has `family` and, for every model except SARIMAX, `architecture`; the
      hybrid's stage 1 inputs do not depend on the `USE_FEATURE` switches.

### Evaluation

- [ ] SMARD is re-scored from `smard_forecast_errors_hourly.csv` on the common hours, and the
      cross-check passes.
- [ ] Scoreboards A (accuracy, incl. months beating SMARD), B (extremes) and C (intervals) exist,
      with the hour or day count in each table's header line and in the export's `count`.
- [ ] Tail bins (`low_extreme`, `ordinary`, `high_extreme`) use shared, printed test-window edges,
      and `below_zero` the fixed 0 MWh threshold;
      both the binned-by-actual and the binned-by-forecast views are shown, with skill vs. SMARD
      only where the hours are the same (by actual, day max / min).
- [ ] Every model's 95 % band comes from the same empirical method, and its coverage and width
      are reported.
- [ ] The static vs. rolling table exists.
- [ ] One forecast plot with the band shows the model set in `PLOT_MODEL` (default `None` = the
      registry model with the lowest test MAE; `PLOT_SPLIT_METHOD` falls back to `static` when
      rolling is off), for the data-derived highest and lowest weeks.
- [ ] The `model` column uses the registry keys plus `seasonal_naive` (both exports) and `smard`
      (scoreboard export only), and rows without a split method carry `split_method = "none"`.
- [ ] Every table and plot has a title and units; no interpretation appears between them.

### Export and conclusions

- [ ] The export toggle `EXPORT_ENABLED` defaults to `True`. With it off, nothing is written and
      the skip is printed. With it on, both files go to `data/models/` as plain comma/period CSV
      with the columns in Behaviours 28–29. The notebook does not create `data/models/`.
- [ ] `SAVE_MODELS` defaults to `False` and every `load_saved` to `False`. Each save holds
      `config.json` and `results.joblib`; `"results"` and `"config"` follow the table in
      Behaviour 34, and every save is checked before the first fit.
- [ ] The "Did we beat SMARD?" section is marked as written for the default configuration and
      states the best registry model's skill (or that none beats SMARD), its tail and interval results, whether refitting helped, the
      post-processing framing, the archive caveats and the one-test-year limitation. A tail claim
      is made only if it shows in both bin views.
- [ ] Every section's self-check and the closing self-check in Behaviour 33 pass.

### Discipline

- [ ] No literal calendar year or date appears in code; windows and examples are derived from the
      data.
- [ ] Every window, lag and period is a duration, not a row count.
- [ ] No Holt-Winters, no seasonal ARIMA, no MAPE, no risk flags, no MLflow run, no weather data.
- [ ] No new dependency is added beyond the team's `joblib` (Behaviour 34); if one appears
      necessary, it is raised rather than `uv add`-ed.
- [ ] `team-EDA.ipynb`, `risk-definition.ipynb`, `forecast-metrics-claude.ipynb` and all `01_eda/`
      notebooks are unchanged.

## Out of bounds

- Editing `team-EDA.ipynb`, `risk-definition.ipynb`, `forecast-metrics-claude.ipynb` or any
  per-member `EDA-<name>.ipynb`.
- Reading or drawing on `Hari_Gridstress_feature_engineering_baselines_metrics.ipynb` before the
  team has cleaned it up.
- Adding dependencies (`pmdarima`, `statsforecast` or others).
- Holt-Winters, other exponential-smoothing models, and seasonal ARIMA with a period `m`.
- Using `fc_residual_load`, any weather data, or any series outside `data/smard.csv` as a feature.
- Tuning on, recalibrating on, or otherwise looking at test-year hours before the final scoring.
- Interpolating or back-filling missing actuals, forecasts or lag sources.
- Risk labels, risk flags on our forecast, and classification metrics (see 04.3).
- The remaining naive baselines and skill scores of parked 04.1.
- Significance tests between forecasts.
- MLflow logging, `modeling/` changes, and productionizing code into the repo-root package.
  These belong to the follow-up spec.
- Changes to `notebooks/API-connection.ipynb`, the `FILTERS` dict or the fetch range.
- reBAP or any cost calculation.
