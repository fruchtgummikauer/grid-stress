# 08 — Ensemble: Combining the Regression Models

- Status: **draft, not yet run.** Updated 2026-09-30 for spec 06's model saves and new defaults.
- Branch: `feature/*` off `main` (drafted on `feature/Ensemble-Forecast-Combination`)
- Deliverable: `notebooks/05_modeling/ensemble-claude.ipynb` (reference, `-claude` suffix until the
  team adopts it), plus three optional artifacts in `data/models/` (Behaviours 20–21):
  - `ensemble_forecast_errors_hourly.csv`
  - `ensemble_scoreboard.csv`
  - `ensemble_weights.csv`
- **No change to any other notebook.** The members are read from spec 06's committed model saves,
  `data/models/<model_key>/{config.json,results.joblib}` (spec 06 Behaviour 34). They already hold
  every run the ensemble needs, the validation runs included.
- Depends on:
  - [06-regression-models.md](06-regression-models.md): the forecast setting (issue time 18:00 on
    `DAY−1`, 2 h actuals lag, cutoff 16:00), the runs `validation_rolling` / `validation_static` /
    `test_static` / `test_rolling`, the interval method (Behaviour 20), the scoreboard definitions
    (Behaviours 21–25, including the `below_zero` bin) and the saves (Behaviour 34)
  - `data/metrics/smard_forecast_errors_hourly.csv` from
    [04-forecast-metrics.md](04-forecast-metrics.md), to re-score SMARD and use it as a member
- Related: [07-hybrid-linear-stage.md](07-hybrid-linear-stage.md) (parked, partly overtaken). If
  the team changes a model, this notebook simply combines the new forecasts once the save is
  replaced.

## Goal

Spec 06's scoreboards show that **no single model wins everywhere**. From the default run with
the new defaults (36-month training window, refit every 7 days), as recorded in spec 06's
*Additions after the refactor*:

- `random_forest_hybrid` is the best test row (rolling MAE 2,333 MWh, +16.4 % against SMARD),
  but it is **behind the four boosters on the validation year** (2,241 against 2,138–2,165 MWh).
- `linear_direct` has the lowest `ordinary`-bin MAE of all rows (2,230 MWh rolling).
- The hybrids, and the random forest in particular, are strong in the low tail. On the 170
  validation hours below 0 MWh the random forest's rolling MAE is 2,669 MWh against SMARD's
  3,841 MWh. Direct trees are capped at their training floor there (spec 06 Behaviour 17).

A **forecast combination** (ensemble) can take the strengths of several models. This spec builds
a small set of combination methods, from a plain mean to weights that depend on the situation. It
tests them against the **best single model**, not only against SMARD, on identical hours.

The random forest shows why the rules below matter. On the validation year it looks like the
fifth-best model, on the test year like the best. Picking members or weights by the test year
would be hindsight. So **every weight, bin edge and window is chosen on the validation year
only**, and the ensemble is scored on the test year exactly once, like every model in spec 06.

Success means the team can read one table and say: does combining beat the best single model,
by how much, on ordinary hours and in the tails, and does the "who wins where" pattern of the
validation year hold in the test year.

## Inherited

| Inherited | Established in |
|---|---|
| `time_series`, loading, `SERIES` / `DERIVED`, resolver, `style_timeseries` | `team-EDA.ipynb` §1 |
| Forecast setting: issue time, availability cutoff, `available(stamps, cutoff)` | [06](06-regression-models.md) Forecast setting, Behaviour 5 |
| Runs (`RUNS`) and the calibration pairing `CALIBRATED_ON` | [06](06-regression-models.md) Behaviours 10–13, 20 |
| Seasonal naive `DAY−7`, looked up by timestamp | [06](06-regression-models.md) Behaviour 19 |
| Empirical interval: per-local-hour quantiles of `r = actual − forecast` | [06](06-regression-models.md) Behaviour 20 |
| Common hours, failure rules, metric definitions, Scoreboards A–C, bin names incl. `below_zero` | [06](06-regression-models.md) Behaviours 21–25 |
| Model saves: `config.json` with the data snapshot (`tuned_on` / `run_on`, one hash per month), `results.joblib` = `RESULTS[model_key]` | [06](06-regression-models.md) Behaviour 34 |
| Sign convention `error = forecast − actual`; no MAPE; RMSE recomputed, never averaged | [04](04-forecast-metrics.md) |
| Regression-to-the-mean rule: a tail claim needs both bin views | [04](04-forecast-metrics.md) Behaviour 16 |
| Durations, not row counts | `risk-definition.ipynb` §3.1 |

Units as in spec 06: forecasts and errors in `MWh`, weights dimensionless, skill and coverage in `%`.

## Scope

### IN

- One **short** notebook, `notebooks/05_modeling/ensemble-claude.ipynb`, runnable top to bottom
  from a fresh kernel in minutes. It **fits no base model**: it loads the base models' forecasts
  from their saves.
- A member diagnostic: validation MAE and error correlation of the members, and a "who wins where"
  table.
- Five combination methods in one registry, all configurable: `equal_mean`, `weighted`,
  `regime_weighted`, `adaptive` and `stacking` (the last is switched off by default).
- Out-of-fold validation forecasts for the band, and the spec 06 interval method.
- Scoreboards A–C with the ensembles next to their members, SMARD and seasonal naive, plus skill
  against the best single member.
- A forecast plot, three optional exports, one closing section and a self-check.

### OUT

- Refitting, retuning or changing any base model, and writing to any save. The ensemble takes the
  forecasts as they are.
- Any change to `regression-models-claude.ipynb` or another notebook.
- New features, new base models, weather data.
- Weights, bin edges or windows chosen on test hours, or recalibrated during the test year.
- Using any actual of the target day, or any actual after the cutoff, to pick weights.
- `sarimax_fourier` as a default member: it is off in spec 06 and has no save. It joins
  automatically once a team member saves it.
- The `-magc` notebook's models: that notebook has no saves.
- Risk flags on the ensemble forecast, classification metrics (see parked
  [04.3-risk-label-link.md](04.3-risk-label-link.md)).
- Significance tests between forecasts.
- MLflow, `modeling/`, new dependencies. `joblib`, scipy (`linprog`) and scikit-learn (`Ridge`) are
  already runtime dependencies.

## Behaviour

### Setup and inputs

1. Repeat `team-EDA.ipynb` §1's loading (resolver, `time_series`, `SERIES`, `DERIVED`,
   `style_timeseries`). The notebook needs `time_series` for the actuals, the calendar, seasonal
   naive and the snapshot check. Measure the resolution from the index, as spec 06 Behaviour 2
   does.
2. **Load the members from the saves.** For every folder in `data/models/` that holds a
   `config.json` and a `results.joblib`, read both. From `results.joblib`, which is
   `RESULTS[model_key]`, take `runs[run][0]`, the forecast series of each run. Load
   `data/metrics/smard_forecast_errors_hourly.csv` for SMARD.
   - `results.joblib` is a pickle. As in spec 06, the markdown says to load only saves from the
     team.
   - A folder with only one of the two files, or a save without one of the four runs, is skipped
     with a printed reason, never silently.
3. **Snapshot check.** A save's forecasts are only valid on the data they were run on. Stop with a
   clear message, and list what differs, when:
   - a save's `run_on` month hashes differ from `time_series[SERIES]` or the SMARD errors file
     loaded here. The hashes are computed with the **same helper as spec 06 §5**, copied, not
     rebuilt. The message says to re-run `API-connection.ipynb` and `forecast-metrics-claude.ipynb`
     at the saves' resolution and extent, or to refit and re-save the models.
   - the saves disagree with each other in `DATA_INFO`, `WINDOWS` or the split methods. Members
     from different forecast settings or windows cannot be combined.

   The device (CPU / GPU) and library versions are printed, not compared. A GPU save and a CPU save
   of different models can be combined: each member is a fixed forecast series.
4. **Configuration cells**, each printed once. Nothing downstream hardcodes a value they hold.
   - `DATA_INFO`: **taken from the saves** (asserted identical in Behaviour 3), not typed in. Only
     `adaptive` needs it (Behaviour 12).
   - `MEMBERS`: the forecasts the ensembles combine. Default: every loaded save, currently
     `lgbm_direct`, `lgbm_hybrid`, `xgb_direct`, `xgb_hybrid`, `linear_direct` and
     `random_forest_hybrid`, plus `smard`. `seasonal_naive` is available but off by default. A
     member listed but not found is dropped with a printed message.
   - `METHODS`: one entry per combination method, keyed `equal_mean`, `weighted`,
     `regime_weighted`, `adaptive`, `stacking`, each with `enabled`, its parameters, `label` and
     `color`. `stacking` is `enabled: False` by default.
   - `REGIMES`: how `regime_weighted` splits the hours (Behaviour 11). Default: `level` only.
     `season` can be switched on as a second dimension.
   - `SHRINK_HOURS = 100`: the shrinkage constant of `regime_weighted`.
   - `ADAPTIVE_WINDOWS = [7 days, 30 days]`: candidate trailing windows for `adaptive`.
   - `INTERVAL`: `level = 0.95`, as in spec 06.
   - `PLOT_METHOD` / `PLOT_SPLIT_METHOD`: default `None` / `"rolling"`, resolved as in spec 06
     Behaviour 27 (the enabled ensemble with the lowest test MAE).
   - `EXPORT_ENABLED = True`, the same default as spec 06 since 2026-09-30.
5. **One ensemble per split method.** Members are combined within a split method. The rolling
   ensemble combines rolling members and learns on their `validation_rolling` forecasts. The static
   ensemble combines static members and learns on `validation_static`. This mirrors spec 06's
   `CALIBRATED_ON`: a frozen model makes larger errors than a refitted one, so weights learned on
   rolling forecasts would suit the static members badly. SMARD and seasonal naive are the same in
   both ensembles. Seasonal naive is computed here with spec 06 Behaviour 19's rule, because it has
   no save.

### Member diagnostic

6. **Member overview on the validation year**, per split method: MAE and bias per member on the
   members' common validation hours, and the **correlation matrix of their errors**. State the
   reading plainly: a combination can only gain where errors are not near-identical. The four
   boosters share their features, so high correlations between them are expected. `linear_direct`
   and `random_forest_hybrid` are the likelier sources of diversity.
7. **Who wins where.** Validation MAE per member per level regime (Behaviour 11's bins, by the
   members' mean forecast) and per season. Print the same table for the test year, marked
   **"stability check only — used for no choice"**. State per bin whether the validation winner is
   also the test winner.
8. **Best single member**: per split method, the registry member with the lowest **validation**
   MAE. It is the reference every ensemble has to beat (Behaviour 17). It is chosen on validation,
   never on test, and printed together with the member that happens to be best on test. On the
   current saves these are expected to differ (a booster against `random_forest_hybrid`), and the
   notebook says so. SMARD is not a candidate: it has its own column.

### Combination methods

All methods produce one forecast per hour from the members' forecasts for that hour. An hour where
any member's forecast is missing gets **no ensemble forecast**. The ensemble never re-normalises
over the remaining members, because that would be a fallback in the sense of spec 06 Behaviour 21.

9. **`equal_mean`**: the plain mean of the members. It has no parameters, so it needs no fit and
   no cross-fitting. It is the floor every other method has to beat. Averaging with equal weights
   is known to be hard to beat (the "forecast combination puzzle").
10. **`weighted`**: one weight per member, `w ≥ 0` and `Σ w = 1`, chosen to minimise **MAE** on
    the validation year. MAE with these constraints is a linear programme, solved exactly with
    `scipy.optimize.linprog`. The weights are printed per split method. With near-identical
    members the solution is not unique; state that single weights of highly correlated members
    are not interpretable on their own.
11. **`regime_weighted`**: separate `weighted` weights per regime.
    - **Regime by forecast level, not by actual.** An hour's regime comes from the **members' mean
      forecast** for that hour, which is known at issue time. The actual is not.
    - **Edges.** The low side uses the **fixed 0 MWh** edge, as spec 06's `below_zero` bin does.
      Hours below 0 MWh are becoming more common every year: 170 in the validation year, 563 in the
      test year (spec 06). A quantile edge from the validation year would therefore mean something
      different in the test year; 0 MWh means renewable oversupply in both. The other edges are the
      `{25, 75, 99} %` quantiles of the members' mean forecast over the validation year, computed
      once and printed in `MWh`. That gives five regimes: `below_zero`, `low` (0 to P25),
      `ordinary` (P25–P75), `high` (P75–P99) and `high_extreme` (> P99). Test hours use the same
      edges.
    - `fc_residual_load` is not used for the regime, only the members' own forecasts. See Open
      questions.
    - **Shrinkage instead of a minimum size.** A regime with few validation hours has noisy
      weights: `high_extreme` holds about 88, `below_zero` about 170. Each regime's weights are
      therefore pulled toward the global `weighted` weights:
      `w_regime = n / (n + k) · w_fit + k / (n + k) · w_global`, with `n` the regime's validation
      hours and `k = SHRINK_HOURS`. A regime with no validation hours therefore gets the global
      weights. That follows from the formula and is not a fallback.
    - With `season` switched on, regimes are level × season, and the same shrinkage applies.
12. **`adaptive`**: weights that follow recent performance.
    - For delivery day `DAY`, a member's weight is proportional to `1 / MAE`, where the MAE is
      computed over the trailing window of hours whose **error is known at `DAY`'s availability
      cutoff**: `t + resolution ≤ cutoff`, the spec 06 availability rule, with the saves'
      `DATA_INFO`.
    - The window reaches back into the validation year at the start of the test year. Those errors
      are known by then, so this is allowed.
    - The window length is chosen from `ADAPTIVE_WINDOWS` on the validation year, by the MAE of
      the adaptive ensemble there, and then frozen.
    - Using test-year actuals **as they are published** is an online update, not selection. The
      weights for `DAY` never see an actual after `DAY`'s cutoff, and the self-check tests this by
      truncation (Behaviour 23).
13. **`stacking`** (off by default): a `Ridge` meta-model on the member forecasts plus the level
    regime as one-hot columns, fitted on the validation year, with a fixed `alpha` in its registry
    entry. It is off by default because it has the most parameters and overlaps `regime_weighted`.
    With few validation hours per regime it is the most likely to overfit.

### Honest intervals

14. **Out-of-fold validation forecasts.** A method's in-sample validation residuals are too small,
    because the weights were fitted on them, so a band calibrated on them would be too narrow.
    Every method with parameters (`weighted`, `regime_weighted`, `stacking`, the window choice of
    `adaptive`) therefore also produces **out-of-fold** validation forecasts:
    - blocked by calendar month of the validation year
    - for each month: fit on the other months, forecast the held-out month
    - blocks rather than random hours, so neighbouring hours with similar errors do not leak into
      the held-out part

    The out-of-fold forecasts calibrate the band and give the "honest" validation MAE shown next to
    the in-sample one. The **final** parameters, used on the test year, are fitted on the full
    validation year. `equal_mean` has no parameters: its validation forecasts are already
    out-of-sample. `adaptive` is out-of-sample by construction once its window is fixed. Only the
    window choice is cross-fitted.

    This fitting on "other months" uses later months of the validation year, which is fine here.
    It never reaches a test hour, and it only estimates the size of the error.
15. **Band**: spec 06 Behaviour 20's method on the out-of-fold validation residuals of each method ×
    split method: per-local-hour quantiles of `r = actual − forecast`, `forecast + [q_low, q_high]`.
    It is calibrated once and not updated during the test year.

### Evaluation

16. **Common hours**: test hours where the actual, SMARD, seasonal naive, every member of both
    split methods and every enabled ensemble have a forecast (a strict intersection). SMARD is
    re-scored from `smard_forecast_errors_hourly.csv`. The notebook prints how many test hours the
    intersection lost, and why.
17. **Scoreboards A–C** with spec 06's definitions and metric names (Behaviours 22–25), including
    the `below_zero` bin in Scoreboard B. Rows: each enabled ensemble × split method, every member,
    SMARD and seasonal naive. Added to Scoreboard A:
    - `skill_vs_best_pct = 100 · (1 − MAE / MAE_best_member)`, against the split method's best
      member (Behaviour 8); positive = better than the best single model
    - months beating the best member, next to months beating SMARD

    Added to Scoreboard B: the same skill against the best member per bin binned by actual and for
    the day max / min. As in spec 06, binned by forecast there is no skill, because each row fills
    the bins with its own hours.
18. **Weights table**: the final weights per method × split method (× regime), as a heatmap or
    table. It shows which member carries which regime. The figure names the most-weighted member
    per regime and whether that matches the validation winner of Behaviour 7.
19. **Forecast plot with the band**: spec 06 Behaviour 27's two-week figure (highest and lowest
    actual week of the test window), for `PLOT_METHOD` × `PLOT_SPLIT_METHOD`. It adds the best
    member's forecast, so the difference is visible.

### Export

20. `EXPORT_ENABLED` defaults to `True`, like spec 06. With it off, the export cell prints the paths
    it would write and writes nothing. The frames are always built in memory, so the self-check
    works either way. `data/models/` exists already, and its CSVs are gitignored.
21. The files:
    - `ensemble_forecast_errors_hourly.csv`: spec 06's hourly export format (`timestamp`, `model`,
      `split_method`, `residual_load`, `forecast`, `lower`, `upper`, `err_residual_load`), with
      `model = "ensemble_<method>"`. Members and SMARD are not repeated here: they stay in their
      saves and files.
    - `ensemble_scoreboard.csv`: spec 06's long scoreboard format, including the member, SMARD and
      seasonal-naive rows on this notebook's common hours, plus the added metrics of Behaviour 17.
    - `ensemble_weights.csv`: `method`, `split_method`, `regime`, `member`, `weight`. `adaptive`
      gets one row per test day (`regime` = the delivery day).
    - Plain CSV (`sep=","`, `decimal="."`, UTF-8). Nothing is written to `model_*` files or to a
      save.

### Conclusion and checks

22. Close with one **"Does combining help?"** section, the notebook's only interpretation, written
    for the default configuration and saying so. It states:
    - the best ensemble × split method and its skill against the **best member** (chosen on
      validation) and against SMARD, on the common hours; if no ensemble beats the best member, it
      says so plainly
    - how the best ensemble compares with the member that happens to be best on test, stated as
      context and not as the benchmark (Behaviour 8)
    - whether `equal_mean` was already as good as the fitted methods
    - the tails: whether an ensemble keeps the low-tail advantage of the hybrids and the random
      forest without their overshoot, judged in both bin views and in `below_zero`; the extreme
      bins hold about 88 hours, so this stays qualitative
    - whether the "who wins where" pattern held from validation to test (Behaviour 7)
    - interval coverage against 95 %
    - the framing: with `smard` as a member the ensemble is still a post-processor of SMARD's
      forecast, and it rests on one test year
23. A **self-check cell**:
    - no parameter, edge or window was fitted on a test hour: every fitting input's index lies in
      the validation year
    - weights are `≥ 0` and sum to 1 (within `1e-9`) for `weighted`, every regime and every
      adaptive day
    - `equal_mean` equals the row mean of the members; an ensemble is empty exactly where a member
      is empty
    - **truncation test for `adaptive`**: for a fixed-seed sample of test days (`SEED = 42`, 20
      days), recompute the day's weights with every actual after that day's cutoff masked; they must
      be identical
    - regime edges come from validation forecasts only, and test hours are assigned with them
    - the loaded member forecasts match the saves they came from (no member altered in memory), and
      seasonal naive matches spec 06's rule
    - scoreboard MAE and RMSE match a direct recomputation from the in-memory hourly frame, and
      SMARD's MAE matches `smard_forecast_errors_hourly.csv`
    - `time_series` still carries exactly `SERIES + DERIVED`

## Data

Sources:

| File | Produced by | Used for |
|---|---|---|
| `data/smard.csv` | `API-connection.ipynb` | actuals, calendar, seasonal naive, snapshot check |
| `data/models/<model_key>/results.joblib` | `regression-models-claude.ipynb` §5 (`SAVE_MODELS`) | member forecasts: validation and test runs |
| `data/models/<model_key>/config.json` | same | snapshot hashes, `DATA_INFO`, `WINDOWS`, selected configuration |
| `data/metrics/smard_forecast_errors_hourly.csv` | `forecast-metrics-claude.ipynb` | SMARD member, re-scoring, snapshot check |

`data/models/model_forecast_errors_hourly.csv` is **not** needed: the saves carry the same test
forecasts, and the validation runs besides.

Derived in this spec:

| Derived | Definition | Unit | Persisted? |
|---|---|---|---|
| member mean forecast | mean of the members per hour; defines the regime | MWh | no |
| regime edges | 0 MWh plus P25 / P75 / P99 of the member mean forecast, validation year | MWh | printed |
| weights | per method × split method (× regime / × day) | — | `ensemble_weights.csv` |
| ensemble forecasts, bands | per method × split method × test hour | MWh | `ensemble_forecast_errors_hourly.csv` |
| out-of-fold validation forecasts | blocked by month (Behaviour 14) | MWh | no |
| scoreboard metrics | spec 06 metrics plus skill vs. best member | MWh / % / count | `ensemble_scoreboard.csv` |

## Edge cases

- **Local data differ from the saves.** The saves were run on the hourly snapshot
  2019-01-01 00:00 → 2026-09-06 23:00 (67,336 rows). After a re-fetch, a switch to quarter-hour
  data, or a stale SMARD errors file, the month hashes differ and Behaviour 3 stops. The fix is
  outside this notebook: regenerate the inputs at the saves' resolution and extent, or refit and
  re-save the models in the spec 06 notebook.
- **A model changed but its save did not.** The ensemble combines whatever is saved. After a change
  in the spec 06 notebook, the saves must be replaced (`SAVE_MODELS = True`) before the ensemble
  sees it. The notebook prints each save's `saved_at` and selected configuration, so an old save is
  visible.
- **Mixed devices.** Saves may come from GPU and CPU runs. That is allowed (Behaviour 3). Re-saving
  an XGBoost model on another machine changes its forecasts slightly, and with them the ensemble.
- **The static hybrids' level shift.** Under static, the hybrids carried a large level shift over
  the test year in the earlier 24-month run (spec 07). If the validation year shows the same,
  `weighted` gives them little weight in the static ensemble. That is the method working. No
  member is excluded by hand.
- **Near-identical members.** Highly correlated members make the LP's weights non-unique. Only the
  combined forecast is interpreted, never one weight in isolation.
- **Small regimes.** `high_extreme` holds about 88 validation hours, `below_zero` about 170.
  Shrinkage (Behaviour 11) keeps their weights from being pure noise. `SHRINK_HOURS` is a choice
  the notebook states, not a derived value.
- **Regime by forecast, risk by actual.** A regime is assigned from the forecast, so an unforeseen
  extreme hour lands in an ordinary regime and gets ordinary weights. The ensemble cannot fix a
  miss that all members share.
- **Missing member forecasts.** A failed day of one member empties that day for every ensemble
  (Behaviour 16). The notebook counts the hours lost per member.
- **Adaptive at the test start.** The first test days' trailing windows lie in the validation year,
  using the validation forecasts of the same split method. This is the only place where validation
  forecasts enter a test-year forecast, and it is legal: their errors are published long before.
- **Pickle.** `results.joblib` can run code when loaded. Load only the team's committed saves.
- **DST.** Per-local-hour band quantiles and hour-of-day tables have one fewer observation at local
  02:00 per year, as in spec 06.
- **Resolution.** Saves, `smard.csv` and the SMARD errors file must share one resolution; the
  snapshot check enforces it. Every window in this notebook is a duration.

## Acceptance criteria

### Setup and inputs

- [ ] `notebooks/05_modeling/ensemble-claude.ipynb` runs top to bottom from a fresh kernel, fits
      no base model and changes no other notebook or save.
- [ ] Members are loaded from `data/models/<model_key>/`; incomplete saves are skipped with a
      reason.
- [ ] Data that differ from the saves' snapshot, or saves that disagree in `DATA_INFO`, `WINDOWS`
      or split methods, stop the notebook with a message listing the differences.
- [ ] `MEMBERS`, `METHODS`, `REGIMES`, `SHRINK_HOURS`, `ADAPTIVE_WINDOWS`, `INTERVAL`,
      `PLOT_METHOD`, `PLOT_SPLIT_METHOD` and `EXPORT_ENABLED` are configuration cells, printed
      once; `DATA_INFO` comes from the saves.
- [ ] Rolling and static ensembles are built separately, each learning on its own validation run.

### Methods

- [ ] Member overview, error correlation and the "who wins where" table exist; the test-year table
      is marked as stability check only.
- [ ] The best single member is chosen by validation MAE and printed next to the test-best member.
- [ ] `equal_mean`, `weighted`, `regime_weighted` and `adaptive` run; `stacking` exists and runs
      when switched on.
- [ ] `weighted` is solved as an LP with `w ≥ 0`, `Σ w = 1`; regimes are assigned by forecast with
      the 0 MWh edge and validation quantile edges, and shrunk toward the global weights.
- [ ] `adaptive` uses only errors known at each day's cutoff; its window is chosen on validation.
- [ ] No ensemble forecast exists for an hour with a missing member.

### Evaluation and export

- [ ] Bands come from out-of-fold validation residuals; coverage and width are reported.
- [ ] Scoreboards A–C use spec 06's definitions, including `below_zero`, and add skill and months
      against the best member.
- [ ] The weights table and the forecast plot exist.
- [ ] `EXPORT_ENABLED` defaults to `True`; the three files are written in the stated format, and
      nothing else in `data/models/` changes.
- [ ] The "Does combining help?" section states the result against the best member and against
      SMARD, whichever way it comes out, with tail claims only where both bin views agree.
- [ ] The self-check in Behaviour 23 passes, including the adaptive truncation test.

### Discipline

- [ ] No weight, edge, window or calibration uses a test hour; nothing is recalibrated in the test
      year.
- [ ] No literal calendar year or date in code; every window is a duration.
- [ ] No new dependency; `regression-models-claude.ipynb`, `regression-models-magc.ipynb`,
      `team-EDA.ipynb`, `risk-definition.ipynb`, `forecast-metrics-claude.ipynb`, the `01_eda/`
      notebooks and every save are unchanged.

## Out of bounds

- Changing the base models, their features, grids, windows or split setup; that is spec 06 / 07
  territory.
- Editing any other notebook, or writing, replacing or deleting a model save.
- Choosing members, weights or methods by test-year results, or dropping a member by hand because
  of them.
- Weather data, reBAP, MLflow, `modeling/`, new dependencies.
- Risk flags and classification metrics on the ensemble.

## Open questions

- **SMARD as a member.** Spec 06 forbids `fc_residual_load` as a *feature*. As an ensemble
  *member* it is the benchmark itself. Mixing it in is legitimate and often helps, but the claim
  becomes "a weighted blend of SMARD and our corrections". Default here: on. Should it be?
- **Regime by SMARD's level.** `fc_grid_load − fc_gen_wind_solar` would be a regime variable
  independent of our models, but it equals `fc_residual_load`. This spec uses the members' mean
  forecast instead. Is that the right choice?
- **`SHRINK_HOURS = 100`**: a reasonable default, or should it be chosen on the validation year
  (cross-fitted, like the adaptive window)?
- **One ensemble per risk direction?** Spec 07 asks whether the model choice should be made per
  risk direction. `regime_weighted` does this implicitly. Is a separate low-tail ensemble for the
  low risk flag worth a follow-up?
- **More members.** Should `sarimax_fourier` be saved once so it can join, and should the `-magc`
  notebook get saves too?
