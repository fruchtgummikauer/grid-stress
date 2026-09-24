# 05 — Regression Models: Day-Ahead Residual-Load Forecast vs. SMARD

- Status: draft. The decisions taken in review of the team's first draft are listed under Goal.
- Branch: `feature/*` off `main`
- Deliverable: `notebooks/05_modeling/regression-models-claude.ipynb`
  - (reference, `-claude` suffix until the team adopts it)
  - plus two artifacts, `data/models/model_forecast_errors_hourly.csv` and
    `data/models/model_scoreboard.csv`, written only when the export toggle is on. It is **off by
    default**: the team experiments first and switches the export on later (Behaviour 28).
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
     weekly seasons
   - LightGBM, **direct** and as a **hybrid**, where a linear model captures the trend and the
     booster models its residuals
   - XGBoost in the same two architectures, in the registry but switched off by default
3. Evaluates each model under two **split methods** on the same test year:
   - **static**: fit once, frozen for the whole test year
   - **rolling origin**: refit at a fixed interval on a sliding window

   The difference between the two is the value of refitting.
4. Gives every model a **95 % prediction interval** built the same way for all models, and
   measures whether the interval actually covers 95 % of hours.
5. Presents a **scoreboard** with these columns:
   - MAE, RMSE, bias, `hour_count`, skill vs. SMARD and the months in which a model beats SMARD
   - error in the tails and on the day max / min, which the risk label is built on
   - interval coverage
6. Closes with one **"Did we beat SMARD?"** section, including the caveats that go with it.

Success means the team can read one table and say, per model and split method: better or worse
than SMARD, by how much, on the ordinary hours and in the tails, with an interval that is honest
or not.

### Design decisions taken in review of the team's first draft

| Draft | This spec | Why |
|---|---|---|
| Forecast horizon of 24 steps | Target is **delivery day `DAY`**, 00:00–23:00 local, issued at **18:00 on `DAY−1`** | A spring DST day has 23 hours, and at 15-min resolution a day has 96 steps. Our model cannot see `DAY−1` up to 23:00 either: the target hours are 10–33 hourly steps after the last usable actual. |
| "Same data as SMARD": wind/solar at 18:00, load at 10:00, residual load at 18:00 | The issue time is 18:00 `DAY−1`, the earliest point at which all SMARD inputs for `DAY` exist. Actuals are usable up to issue time minus a **3 h publication lag**. | The residual-load value "published at 18:00" is SMARD's *forecast*, not the actual. Two manual checks on smard.de showed actuals about 2 h behind real time; 3 h adds a margin. |
| Residual load driven by the grid-load and wind/solar forecasts | SMARD's `fc_grid_load` and `fc_gen_wind_solar` for `DAY` are model inputs. `fc_residual_load` is **not** a feature. | It equals `fc_grid_load − fc_gen_wind_solar` exactly (spec 04), so it adds nothing but collinearity. |
| Holdout: keep the last `h` observations | Test = the **last 365 complete delivery days**, ending with the last fully observed day | One full seasonal cycle. It is the same window as spec 04's `trailing_365` headline, so the SMARD number is directly comparable. |
| Rolling-origin CV with a 24-month window | The split method is an **experimental factor** evaluated on the same test year. Static and rolling both use a **24-month** training window. Hyperparameters are tuned on the validation year before the test. | Different test sets would make the split methods incomparable. With equal window lengths, the only difference between the split methods is refitting. |
| Differencing to handle stationarity | SARIMAX uses a fixed `d = 0`, checked (not decided) by one KPSS test; differencing, if ever needed, happens **inside ARIMA** through `d`. Tree models get no differenced target. | The SMARD forecasts and Fourier terms carry level and seasons, so the remaining error is expected to be stationary. Differencing outside the model complicates back-transformation and intervals. |
| SARIMA with `auto_arima` | One **SARIMAX** variant in statsmodels: ARIMA errors plus Fourier terms for the seasons (Behaviour 15). No `pmdarima`. | SARIMA takes one seasonal period; hourly data has three. `m = 168` does not fit on a desktop. Fourier terms carry the seasons cheaply and stay resolution-independent. `pmdarima` is not installed and has had numpy-2 problems. |
| AIC to compare parameters | **No AIC search**: SARIMAX's order is fixed at `(1, 0, 1)` in the registry. All models are compared out of sample (MAE / RMSE on identical hours). | Keeps SARIMAX simple and fast. AIC could only ever compare ARIMA orders within one model; a gradient booster has no AIC. |
| Holt-Winters | **Dropped** (team decision) | It takes no exogenous inputs, handles one season only, and must be additive because residual load goes negative. It adds little over seasonal naive. |
| LightGBM direct, XGBoost hybrid | A **2 × 2 grid** in the registry: {LightGBM, XGBoost} × {direct, hybrid}. The hybrid is a wrapper around either booster. XGBoost is `enabled: False` by default. | Otherwise algorithm and architecture are confounded. The two boosters usually behave alike, so XGBoost is off to save runtime until someone wants it. |
| Plot with a 95 % confidence interval | A **95 % prediction interval** built by one empirical method for all models, plus measured coverage | Boosters have no native interval. An interval whose coverage is not measured cannot be checked. |
| MAE and RMSE scoreboard | Adds bias, skill vs. SMARD, monthly wins, tail and day-extreme errors, interval coverage and a seasonal-naive row | These follow from spec 04's approach. The seasonal-naive row pulls one baseline out of parked 04.1; the rest of 04.1 stays parked. |
| Pipeline ready for 15-min data | Every window, lag and period is a **duration**. | Project convention (durations, not row counts). |

### Forecast setting

For every delivery day `DAY` in the evaluation windows:

| Item | Rule (defaults, all configurable durations) |
|---|---|
| Issue time `ISSUE_TIME` | 18:00 local on `DAY−1`. `ISSUE_TIME` and `AVAILABILITY_CUTOFF` are **per-day timestamps** returned by the Behaviour 5 helper, not constants; `DATA_INFO` holds their configured clock time and lag. |
| Availability cutoff `AVAILABILITY_CUTOFF` | `AVAILABILITY_CUTOFF = ISSUE_TIME − lag`, with `lag = 3 h`: 15:00 on `DAY−1`. Every use of "the cutoff" or "availability cutoff" in this spec means `AVAILABILITY_CUTOFF`. |
| Actuals available at `ISSUE_TIME` | An observation stamped `t` (interval start) is usable if its interval ends at or before the cutoff: `t + resolution ≤ AVAILABILITY_CUTOFF`. Hourly: up to the hour stamped 14:00 on `DAY−1`. |
| SMARD forecasts for `DAY` | Available: both components are published by 18:00 on `DAY−1`. |
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
  features, intervals, plot selection, export toggle) that the team edits the way it edits a
  colour map.
- The forecast setting and a truncation-based leakage test.
- A validation year for tuning, a test year, and the static and rolling split methods.
- Seasonal naive `DAY−7`, `sarimax_fourier`, LightGBM direct and hybrid; XGBoost direct and hybrid
  in the registry, switched off by default.
- A stationarity check (one KPSS test) for `sarimax_fourier`.
- A minimal, toggleable feature set for the boosters.
- Empirical 95 % prediction intervals and their coverage.
- The scoreboard (accuracy, extremes, intervals), the static vs. rolling comparison, and one
  forecast plot with its band for a selectable model.
- Two CSV exports behind a toggle that is off by default, one closing "Did we beat SMARD?"
  section, and a self-check cell.

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
- A rich feature set (rolling statistics, ramps, interactions); that is for the team's
  feature-engineering work.
- Quarter-hour data. The pipeline is ready for it (Behaviour 30), but this spec runs on the
  hourly `data/smard.csv`.
- reBAP / cost calculation.
- New dependencies.

## Behaviour

### Setup and configuration

1. Repeat `team-EDA.ipynb` §1's loading and preparation: the resolver, `time_series`, `SERIES`,
   `DERIVED`, `YEARS`, `style_timeseries` and the season mapping. One markdown cell states they
   are inherited and points at the source; nothing is re-derived.
2. Measure the resolution from the index. Every duration below is converted to an observation
   count from that measured resolution, never written as a row count.
3. **Configuration cells**, each printed once as a summary. Nothing downstream hardcodes a value
   they hold.
   - `DATA_INFO`: `issue_time = 18:00` on `DAY−1`, `actuals_lag = 3 h`, and the capacity publication
     rule "year `Y` is published on 1 January of `Y`, 00:00" (Behaviour 7).
   - `WINDOWS`: `test = 365 days`, `validation = 365 days`,
     `train = 24 months` (calendar months, i.e. `DateOffset(months=24)`, not 730 days),
     `refit_every = 30 days`.
   - `TRAIN_TEST_SPLIT_METHOD = {"static": True, "rolling": True}`: the split methods, each with
     an on/off switch. With both on (the default) the scoreboard shows them side by side.
   - `MODELS`: one entry per model, keyed `sarimax_fourier`, `lgbm_direct`, `lgbm_hybrid`,
     `xgb_direct`, `xgb_hybrid`. Each entry holds:
     - `enabled` (`False` for the two XGBoost entries by default, `True` for the rest)
     - `family`: `sarimax`, `lightgbm` or `xgboost`; it decides how the model is fitted
     - `architecture`: `direct` or `hybrid` for the boosters (Behaviour 17); not used for
       `sarimax_fourier`
     - fixed parameters
     - a small tuning grid (Behaviour 11)
     - display label
     - colour (the colour-map pattern the draft asked for)

     The keys are the values of the `model` column in both exports. SMARD (`smard`, scoreboard
     export only), seasonal naive (`seasonal_naive`) and the actual series sit outside the
     registry, with fixed colours.
   - `FEATURES`: one switch per feature group (Behaviour 18), so the team can adjust the set
     later without touching model code.
   - `INTERVAL`: `level = 0.95`.
   - `PLOT_MODEL` and `PLOT_SPLIT_METHOD`: which model and split method the forecast plot shows.
     Defaults: `PLOT_MODEL = None`, which the plot cell resolves to the **registry model** with the
     lowest test MAE (never SMARD or seasonal naive), and `PLOT_SPLIT_METHOD = "rolling"`, which
     falls back to `static` when rolling is switched off. Setting a registry key, e.g.
     `"lgbm_hybrid"`, overrides the default.
   - `EXPORT_ENABLED = False`: the export toggle (Behaviour 28).
4. Load `data/metrics/smard_forecast_errors_hourly.csv`. Fail with a clear message naming
   `notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` in two cases: the file is
   missing, or its timestamps do not match `time_series.index` exactly (a stale export from
   another snapshot).

### Forecast setting and leakage

5. Implement the forecast setting from the Goal as one helper that returns the issue time, the
   availability cutoff and the target hours for a given `DAY`. Every model, feature and training
   row goes through it.
6. **Feature availability rules.** Actual-derived features may use only observations available
   at the row's issue time (Forecast setting). SMARD forecasts for the target hour are available
   (published by 18:00 on `DAY−1`). Calendar features are always available.
7. **Capacity look-ahead.** The `cap_*` columns are a yearly value. The project treats year `Y`'s
   value as **published on 1 January of `Y`, 00:00** (team decision). A row uses the latest
   capacity value published at or before its issue time.
   - Most rows therefore use the current calendar year's value.
   - Delivery day 1 January is issued at 18:00 on 31 December, before the new year's value is
     out, so it uses the previous year's value. State this; it is the reason the rule is a
     publication time and not a fixed year offset.
   - Rows with no published value yet (only the record's first day, issued before the first
     publication) have an empty capacity feature. That day is unforecastable anyway
     (Behaviour 9), and the default 24-month windows never reach it.
8. **Leakage test by truncation.** For a fixed-seed sample of delivery days drawn from the
   training, validation and test windows (default 20 days, a constant in that cell):
   - rebuild each day's design rows from data **cut off at that day's availability cutoff**:
     actuals and SMARD errors up to the cutoff, SMARD forecasts up to the end of `DAY`
   - assert that the rebuilt rows are identical to the rows built from the full data
   - for the sampled validation and test days, assert that `sarimax_fourier`'s dynamic
     prediction for that day is identical whether the model is applied to the whole period or to
     the series cut off at the cutoff

   Any feature that peeks past the cutoff changes when the future is removed, so this one test
   catches leakage without tracking per-feature timestamps. The cell also asserts three fitting
   rules:
   - Linear stages and tuning never see a test row.
   - The test year is forecast but never used for selection.
   - No shuffling happens anywhere; every split is time-ordered.
9. **Unforecastable days.** Delivery day `DAY` is unforecastable, for **every** model, as soon as a
   SMARD component forecast is missing for any hour of `DAY` or for the remaining hours of `DAY−1`
   after the cutoff (SARIMAX needs those as exogenous inputs). Such a day has no input rows. It is
   a **data exclusion**, not a model failure (Behaviour 21): excluded from training and scoring,
   counted from the data and never hardcoded. In the current snapshot this is 2020-01-31 (and the
   day after it), which lie outside the default windows.

### Windows and split methods

10. **Test window**: the last **365 complete delivery days**, ending with the last day whose hours
    all have the actual and all SMARD forecasts observed, derived from the data. When the record
    ends at 23:00 this is exactly spec 04's `trailing_365` window; otherwise the incomplete last
    day is left out and the notebook says so. **Validation window**: the 365 delivery days
    immediately before the test window.
11. **Tuning**: a **rolling-method walk-forward over the validation year** runs for **every**
    model: once per grid configuration, or once for a model without a grid (`sarimax_fourier`,
    seasonal naive). It is the same procedure as the rolling method (Behaviour 13), applied to the
    validation year: the first fit is issued for the first validation day on the 24-month window
    ending at its cutoff, then a refit every `refit_every`. The runs without a grid select
    nothing; they produce the validation residuals the rolling band needs (Behaviour 20).
    - The selection criterion is residual-load **MAE** on validation hours.
    - Grids stay small, four configurations per booster by default, all with
      `learning_rate = 0.05`:
      - LightGBM: `num_leaves ∈ {31, 63}` × `n_estimators ∈ {300, 800}`
      - XGBoost: `max_depth ∈ {4, 6}` × `n_estimators ∈ {300, 800}`
      - the hybrid variant of a booster uses the same grid as its direct variant
    - `sarimax_fourier` has no grid: its order and Fourier orders are fixed in the registry
      (Behaviour 15).
    - **Everything selected is frozen**: booster hyperparameters are chosen once on the validation
      year, printed, and reused unchanged by both split methods on the test year. Refits
      re-estimate coefficients or re-train on the new window; they never re-run the grid search.
      This keeps static vs. rolling a clean "refit or not" comparison.
    - After tuning, the test-year fits use training windows that **include the validation year**.
      Tuning only chose the configuration; the validation data is not held back from the final
      fits.
12. **Static method**: one fit per model, issued for the first test day, on the 24-month window
    ending at that fit's availability cutoff. The frozen model forecasts every test day; its
    inputs (SMARD forecasts, lags) update daily, its parameters do not. For SARIMAX, the daily
    update comes from the dynamic prediction starting at each day's cutoff (Behaviour 15).
13. **Rolling method**: the first fit is issued for the first test day, so it is **identical to
    the static fit**. After that, a refit every `refit_every` on the sliding 24-month window ending
    at that refit's availability cutoff.
    - The two split methods therefore produce the same forecasts for the first `refit_every` of
      the test year. The notebook says so, since that stretch cannot differ between them.
    - Between refits, the latest model forecasts each day with that day's inputs. For SARIMAX,
      each refit's model is applied once to its training window plus its 30-day stretch and
      predicts each day dynamically, as in the static method.
14. Both split methods produce forecasts for **identical test hours**, and the scoreboard shows
    them side by side.

### Models

15. **`sarimax_fourier`**: a regression with ARIMA errors on the 24-month window, with everything
    fixed in its registry entry (the team can edit the values by hand):
    - **Exogenous inputs**:
      - `fc_grid_load` and `fc_gen_wind_solar`
      - Fourier terms for a **1-day** and a **1-week** period, as durations, with orders
        `K_day = 4` and `K_week = 3`
      - the holiday flag
    - **Error model**: ARIMA with the fixed `order = (1, 0, 1)`. There is no order search.
      `d = 0` because the SMARD forecasts and the Fourier terms already carry the level and the
      seasons; what remains is roughly SMARD's own error, which comes in episodes but reverts to
      its mean. With `d = 1` the model would carry the level of the last actual forward over the
      10–33 h to the target hours. The model therefore reads as "regression on SMARD's forecasts
      plus seasons, with AR/MA errors that carry SMARD's recent error forward".
    - **Series**: the model takes the local-time rows of `time_series` as they are, as evenly
      spaced steps. It uses no UTC grid and no `NaN` filling (Edge cases).
    - **Forecast by dynamic prediction**: a fitted model is applied **once** to its training
      window plus the stretch it forecasts (statsmodels `apply` with the fixed parameters, no
      re-estimation); the training window provides the history the model's state starts from.
      For each day `DAY`, a dynamic prediction starts at the **first row not yet available**, the
      hour stamped 15:00 on `DAY−1`, and runs to the end of `DAY`, using the SMARD forecasts for the
      remaining hours of `DAY−1` and for `DAY` as exogenous inputs. Only `DAY`'s hours are kept. A
      dynamic prediction from a start point uses only observations before that point, so no daily
      state updates are needed; the leakage test (Behaviour 8) confirms it.
    - **Excluded days inside the series**: an unforecastable day (Behaviour 9) inside SARIMAX's
      training window or forecast stretch has no SMARD forecast to use as an input. Its rows are
      dropped and the series continues as if they were a gap, like the DST hour (Edge cases). The
      notebook states how many rows were dropped.
16. **Stationarity check**. One printed KPSS test on the residuals of an OLS fit of residual load
    on `sarimax_fourier`'s exogenous inputs, over the 24 months ending at the validation start. It
    is a **check, not
    a decision**: `d` stays fixed at 0. If KPSS rejects stationarity, the notebook says so and the
    team can change `d` in the registry. A short markdown note states three points:
    - Differencing, if ever needed, happens inside ARIMA through `d`.
    - Tree models need no stationarity but **cannot extrapolate** beyond the target range they
      were trained on.
    - Box-Cox or log transforms are ruled out because residual load is negative in some hours.
17. **Boosters**: LightGBM and XGBoost regressors with fixed seeds, each in two architectures.
    XGBoost is switched off by default (Behaviour 3) and runs when switched on.
    - **Direct**: the booster predicts `residual_load` from the feature set (Behaviour 18).
    - **Hybrid**: stage 1 is an ordinary least-squares model on `fc_grid_load`,
      `fc_gen_wind_solar` and a linear time trend (days since the training window's start). It is
      the "parametric model captures trend" part of the draft. Stage 2 is the booster, fitted on
      stage 1's in-sample residuals with the enabled feature set (Behaviour 18). The forecast is
      stage 1 + stage 2.
    - **Stage 1's inputs are fixed.** The `FEATURES` switches affect only the direct booster and
      stage 2; stage 1 always uses the two SMARD forecasts and the trend, even if the
      `smard_forecast` group is switched off.
    - **No early stopping.** `n_estimators` is a value in the tuning grid. Early stopping would
      need a third time-ordered split inside each training window, which this spec does not
      define.
    - A short markdown note states the expected difference between the two: the hybrid can
      extrapolate the level through stage 1, while the direct booster is capped at its training
      range. This matters in the low tail, where the test year may reach more negative values
      than the training window.
18. **Minimal feature set** for the boosters, as toggleable groups in `FEATURES`. The groups are
    marked **provisional**: the team's feature-engineering work may replace them.

    | Group | Features | Available because |
    |---|---|---|
    | calendar | local hour, day of week, month, `is_weekend`, holiday flag | always known |
    | smard_forecast | `fc_grid_load`, `fc_gen_wind_solar` for the target hour | published by 18:00 `DAY−1` |
    | lags | `residual_load` at the same local hour on `DAY−2` and on `DAY−7`; the last available `residual_load` at the cutoff | before the cutoff |
    | recent_smard_error | mean `err_grid_load` and mean `err_renewables` over the 24 h ending at the cutoff, read from `smard_forecast_errors_hourly.csv` | before the cutoff (same 3 h actuals lag); tests the reference notebook's observation that SMARD's error comes in episodes |
    | capacity | `cap_wind_off + cap_wind_on + cap_solar` under the publication rule (Behaviour 7) | published 1 January of its own year |

    Lags are durations. A lag whose source hour does not exist, such as local 02:00 on a spring
    DST day, is `NaN`; boosters handle missing values natively. It is never filled.
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
    - **Quantiles**: take the `(1 − level) / 2` and `(1 + level) / 2` quantiles of `r` **per local
      hour of day**, since error size varies strongly with the hour.
    - **Band**: `forecast + [q_low(r), q_high(r)]` for each test hour. If a model is strongly
      biased at some hour, both quantiles share a sign and the band does not contain the forecast
      itself. That is correct behaviour, not an error.
    - The band is calibrated on the validation year and not updated during the test year, so
      drift shows up as lost coverage.
    - SMARD is a point forecast and gets no band.

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
    - The notebook prints, per model × split method, failed fits, failed days and convergence
      warnings, and how many test hours the common hours lost. If one model breaks badly, the team
      switches it off in the registry and re-runs.
22. **Metric definitions** follow spec 04:
    - `error = forecast − actual`
    - MAE, RMSE (recomputed from hourly errors) and bias, with `hour_count` next to every value
    - no MAPE
    - `skill = 1 − MAE_model / MAE_SMARD` on the same hours, in `%`; positive means better than
      SMARD
23. **Scoreboard A — accuracy**: one row per enabled model × split method, plus SMARD and
    seasonal naive. Columns:
    - MAE, RMSE, bias, `hour_count`, skill vs. SMARD
    - **months beating SMARD**, as `k of n`: `n` is the number of calendar months that lie
      entirely inside the test window, `k` the number of those in which the row's MAE is below
      SMARD's on the common hours. This is the robustness view in place of
      a significance test.
    - fit time (seconds, summed over the split method's **test-year fits only**)

    In the SMARD row, skill, months beating SMARD and fit time are shown as "—": they compare a
    row against SMARD, or it has no fit.

    Tuning time (grid walk-forwards, the static validation run) is reported
    separately per model. Sorted by MAE.
24. **Scoreboard B — extremes**, on the same rows:
    - **Tail bins.** Edges are the `{1, 25, 75, 99} %` quantiles of the **actual** residual load
      over the test window. They are computed once, printed in `MWh` and shared by every row.
      Report MAE and bias in the top bin (`> P99`) and the bottom bin (`≤ P1`), both **binned by
      actual** and **binned by the row's own forecast**. The ordinary bin (`P25–P75`) is reported
      alongside for comparison.
    - **The regression-to-the-mean rule applies**: only a tail difference that shows in both
      views counts.
    - **Day maximum and day minimum** of residual load, computed from the **common hours** only.
      A day counts if its common hours cover at least `DAY_COMPLETENESS` (23/24) of its expected
      hours. Report value-error MAE and bias, with `day_count`.
    - Next to the table, print the test-window minimum of residual load against the minimum in
      the static fit's training window, the 24 months before the test start (tree extrapolation,
      Behaviour 17).
25. **Scoreboard C — intervals**: coverage (the share of common hours whose actual lies inside
    the band) against the nominal 95 %, and mean band width in `MWh`, per model × split method.
    SMARD has no band and therefore no row here.
26. **Static vs. rolling**: one small table with each model's MAE difference `rolling − static`.
27. **Forecast plot with the band**: one figure for the model and split method set in
    `PLOT_MODEL` / `PLOT_SPLIT_METHOD` (defaults and fallback in Behaviour 3; the plot title names
    the model and split method shown), with two panels:
    - the local Monday–Sunday week of the test window containing the **highest** actual
      residual-load hour
    - the week containing the **lowest**

    Both weeks are derived from the data, and the rule and dates are printed. A week that extends
    beyond either end of the test window is clipped at the window's edge. Each panel shows the
    actual (in black), SMARD's forecast, the model's forecast and its 95 % band. The plot has a
    title and axis labels in `MWh`. Changing `PLOT_MODEL` and re-running the cell shows another
    model.

### Export

28. **Export toggle.** Both files are written only if `EXPORT_ENABLED` is `True`. The default is
    `False`, because the team experiments with the models first.
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
      skill vs. SMARD, on the common hours, with `hour_count`; if it does not beat SMARD, the
      section says so plainly
    - how that model does in the tails and on day max / min compared with SMARD; tail bins in a
      365-day window hold only about 88 hours each, so tail statements stay qualitative
    - whether its interval is calibrated (its coverage against 95 %)
    - whether refitting helped (Behaviour 26)
    - the **post-processing framing**: the models use SMARD's component forecasts, so the result
      is an improvement on SMARD, not an independent forecast; and without a SMARD forecast our
      model cannot run
    - the **archive caveats**:
      - `fc_grid_load` may be a revised value rather than the original 10:00 value
      - the actuals are final values rather than first publications
      - both affect SMARD and our model alike and cannot be corrected from `data/smard.csv`
    - that the result rests on **one test year**, and whether the monthly win count supports it
32. **Runtime**: the notebook prints each model's total fit and tuning time. With the default
    registry the whole notebook should run in about an hour or less on a desktop. If it clearly
    exceeds that, raise it with the team rather than silently shrinking windows or grids.
33. A **self-check cell**. It runs against the in-memory export frames. With `EXPORT_ENABLED` on,
    it also reads both files back with a bare `pd.read_csv` and repeats the export checks on the
    files as written. The checks:
    - the leakage test (Behaviour 8) passes
    - test and validation windows do not overlap, and no tuning result uses a test hour
    - `err_residual_load == forecast − actual` in the hourly export
    - SMARD's MAE on the common hours equals a direct recomputation from
      `smard_forecast_errors_hourly.csv`
    - scoreboard MAE and RMSE match a direct recomputation from the hourly export
    - `fc_residual_load` is not among any model's features
    - `time_series` still carries exactly `SERIES + DERIVED`

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
| `fc_grid_load`, `fc_gen_wind_solar` | exogenous inputs for `DAY` (and for the rest of `DAY−1` in SARIMAX) |
| `fc_residual_load` | SMARD benchmark only, never a feature |
| `err_grid_load`, `err_renewables` (from the SMARD errors file) | `recent_smard_error` feature, under the availability rule |
| `cap_wind_off`, `cap_wind_on`, `cap_solar` | capacity feature; year `Y`'s value counts as published on 1 January of `Y` |

Derived in this spec:

| Derived | Definition | Unit | Persisted? |
|---|---|---|---|
| design matrices | one row per target hour, built at that row's issue time | mixed | no |
| forecasts, bands | per model × split method × test hour | MWh | exported (hourly CSV) |
| `err_residual_load` | forecast − actual | MWh | exported (hourly CSV) |
| scoreboard metrics | MAE, RMSE, bias, skill, monthly wins, tail, day-extreme and interval metrics | MWh / % / count | exported (scoreboard CSV) |
| tail-bin edges | P1 / P25 / P75 / P99 of actual residual load over the test window | MWh | printed only |

Exported artifacts:

| File | Grain | Format |
|---|---|---|
| `data/models/model_forecast_errors_hourly.csv` | test hour × model × split method | `sep=","`, `decimal="."`, UTF-8 |
| `data/models/model_scoreboard.csv` | long: model × split method × table × metric | `sep=","`, `decimal="."`, UTF-8 |

Both are written only with `EXPORT_ENABLED = True` (default `False`). `data/models/` is tracked
through an empty `.gitkeep`; its CSVs are ignored by the `data/models/*.csv` rule in
`.gitignore`. The existing `data/*.csv` rule does not reach subfolders, which is why the separate
rule exists, as for `data/metrics/` and `data/risk_classification/`.

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
  successful refit empty. A convergence warning keeps the forecast. All are counted and printed
  (Behaviour 21).
- **Static and rolling coincide at the start.** Both split methods share their first fit, so they
  are identical for the first `refit_every` of the test year (Behaviour 13).
- **Band without the forecast.** A strongly biased hour gives both band quantiles the same sign,
  so the band does not contain the forecast (Behaviour 20). This is expected.
- **Tree extrapolation.** A direct booster cannot predict below the lowest target value in its
  training window. The test year may carry deeper negative residual load than the 24 months
  before it. Behaviour 24 prints the two minima next to scoreboard B.
- **Regime shift in grid-load bias.** The reference notebook reports that SMARD's grid-load bias
  changes sign between years. A model tuned on the validation year can inherit a correction that
  no longer applies in the test year. The static vs. rolling comparison (Behaviour 26) is where
  this shows.
- **Interval drift.** The band is calibrated on the validation year only. If the error
  distribution shifts, coverage falls below 95 %. Report it; do not recalibrate on test hours.
- **KPSS rejects stationarity.** `d` stays at 0; the notebook reports the result, and changing
  `d` in the registry is the team's call (Behaviour 16).
- **Negative residual load.** No log or Box-Cox transform anywhere.
- **Small tail samples.** About 88 hours per 1 % bin in a 365-day window. Show `hour_count`, and
  keep the closing statements about tails qualitative.
- **Unforecastable edge days.** If the validation or test window starts with a day that has no
  input (missing SMARD forecast), it is skipped and counted, not back-filled.
- **Missing data files.** A fresh clone has no `data/smard.csv` and no
  `data/metrics/smard_forecast_errors_hourly.csv`. The setup fails with a message naming
  `notebooks/API-connection.ipynb` and `forecast-metrics-claude.ipynb` respectively.
- **Runtime.** SARIMAX on 24 months dominates the runtime. The registry's `enabled` switches and
  small grids are the lever; the windows are not shrunk silently (Behaviour 32).

## Acceptance criteria

### Setup and forecast setting

- [ ] The notebook is `notebooks/05_modeling/regression-models-claude.ipynb` and runs top to
      bottom from a fresh kernel.
- [ ] The shared setup is repeated from `team-EDA.ipynb` §1 and pointed at in one markdown cell.
- [ ] `DATA_INFO`, `WINDOWS`, `TRAIN_TEST_SPLIT_METHOD`, `MODELS`, `FEATURES`, `INTERVAL`,
      `PLOT_MODEL`, `PLOT_SPLIT_METHOD` and `EXPORT_ENABLED` are configuration cells, printed
      once; no downstream cell hardcodes a value they hold.
- [ ] Every row is built at its own issue time (18:00 on `DAY−1`, 3 h actuals lag by default), and
      the truncation leakage test passes.
- [ ] Capacity follows the publication rule (year `Y` usable from 1 January of `Y`), and the
      1 January consequence is stated.
- [ ] `fc_residual_load` is not a feature of any model.

### Splits and models

- [ ] The test window is the last 365 complete delivery days, and the validation window is the
      365 delivery days before it; both are derived from the data.
- [ ] The validation walk-forward runs for every model, and the booster grids match the defaults
      in Behaviour 11 unless the team changed them in the registry.
- [ ] Tuning uses only the validation year; the selected configuration per model is printed and
      frozen for both split methods, and refits never re-run the grid search.
- [ ] The static and rolling methods forecast identical test hours, with 24-month windows in
      both.
- [ ] `sarimax_fourier` is fitted with statsmodels on the local-time rows, with `order = (1, 0, 1)`
      and the Fourier orders fixed in the registry, and forecasts each day by dynamic prediction
      starting at the row stamped 15:00 on `DAY−1`, after one `apply` to its training window plus
      forecast stretch; no order search and no daily state updates.
- [ ] One KPSS check is printed as a check, not a decision, with the short note on
      differencing.
- [ ] LightGBM runs direct and hybrid with fixed seeds; the XGBoost entries exist in the registry
      with `enabled: False` and run when switched on.
- [ ] The minimal feature set is implemented as toggleable groups and marked provisional.
- [ ] Seasonal naive `DAY−7` appears as the floor row.
- [ ] Each split method's band is calibrated on its own validation run (static: one frozen fit
      over the validation year).
- [ ] Failures follow Behaviour 21: a failed day stays empty, a failed static fit leaves the
      scoreboard and the intersection, a failed rolling refit empties the days until the next
      refit; failures, convergence warnings and the common hours lost are printed; all other rows
      are scored on the strict common-hours intersection.
- [ ] Every registry entry has `family` and, for the boosters, `architecture`; the hybrid's
      stage 1 inputs do not depend on the `FEATURES` switches.

### Evaluation

- [ ] SMARD is re-scored from `smard_forecast_errors_hourly.csv` on the common hours, and the
      cross-check passes.
- [ ] Scoreboards A (accuracy, incl. months beating SMARD), B (extremes) and C (intervals) exist,
      with `hour_count` or `day_count` next to every value.
- [ ] Tail bins use shared, printed test-window edges; both the binned-by-actual and the
      binned-by-forecast views are shown.
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

- [ ] The export toggle `EXPORT_ENABLED` defaults to `False`. With it off, nothing is written and
      the skip is printed. With it on, both files go to `data/models/` as plain comma/period CSV
      with the columns in Behaviours 28–29. The notebook does not create `data/models/`.
- [ ] The "Did we beat SMARD?" section is marked as written for the default configuration and
      states the best registry model's skill (or that none beats SMARD), its tail and interval results, whether refitting helped, the
      post-processing framing, the archive caveats and the one-test-year limitation. A tail claim
      is made only if it shows in both bin views.
- [ ] The self-check in Behaviour 33 passes.

### Discipline

- [ ] No literal calendar year or date appears in code; windows and examples are derived from the
      data.
- [ ] Every window, lag and period is a duration, not a row count.
- [ ] No Holt-Winters, no seasonal ARIMA, no MAPE, no risk flags, no MLflow run, no weather data.
- [ ] No new dependency is added; if one appears necessary, it is raised rather than `uv add`-ed.
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
