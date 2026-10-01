# 09 — Best-Model Plots: the Top Picks per Category vs. SMARD

- Status: run 2026-09-30. Numbered 09 on purpose (team choice).
- Branch: `feature/selected-best-model-plots`
- Deliverable: `notebooks/05_modeling/visualization-01-regression-best-models.ipynb` (renamed from
  `regression-best-models.ipynb`, then `regression-visualization-best-models.ipynb`). It writes no files.
- Depends on: the exports of [06-regression-models.md](06-regression-models.md) (Behaviour 28), and
  `data/metrics/smard_forecast_errors_hourly.csv` from [04-forecast-metrics.md](04-forecast-metrics.md),
  and the ensemble exports of [08-ensemble.md](08-ensemble.md).
- Built one section at a time, with a team review of each section (`spec-run-section-loop` skill).
- The body is the spec as run; the notebook follows the changes below.

## Changes during the run

| Change | Why |
|---|---|
| `forecast share` replaces `reach`; the picks table adds SMARD's share | "Reach" needed an explanation first |
| `bin_errors` draws range bars (middle 80 %, bias diamond) instead of boxes | Readable without box-plot vocabulary |
| Flagging view: plain-words title, count labels on each quadrant's quietest spot | Easier to read for an audience |
| Overall adds §4.3 (skill by hour of day) and §4.4 (last full week, `WEEK_START`) | Shows at which hours the picks gain or lose |
| §9.1 monthly SMARD bias table (also reads `err_grid_load`); Why / To try bullets | Every explanation backed by a number |
| Minor: settings cell first, every candidate listed, rows check in the self-check, months column in the picks table | Clarity |
| Dropped: hour-of-day MAE per tail | Too few hours per point; §4.3 covers it |
| One ensemble pick (`ENSEMBLE_PICK`): the best `CANDIDATE_SPLIT` ensemble of spec 08 by the `overall` ranking, added to every category whether it qualifies or not (status printed in §2); rank `E` and `gap to best single` in the picks table; reads `data/models/ensemble_*.csv` and stops if their model rows differ from `model_scoreboard.csv` | Shows a one-size-fits-all candidate next to the specialists (team decision 2026-10-01) |

## Goal

There is no single best model. The rows of `regression-models-claude.ipynb` rank differently on
average accuracy (Scoreboard A) and in the tails (Scoreboard B). This notebook turns the exports
into a **short presentation notebook**:

1. For each of five categories, pick the top `N_PICKS` models that **beat SMARD** there.
2. Show the scoreboard of the picked rows.
3. Show two plots per category.
4. End with an insight summary.

The notebook is **minimal**: no setup explanation, no refitting, and nothing repeated from the
regression notebook beyond what the plots need. Method, leakage and caveats are linked, not restated.

| Category | Hours (by the actual) | Ranked by |
|---|---|---|
| `overall` | all common hours | months beating SMARD (↓), then test MAE (↑) |
| `ordinary` | `P25 ≤ actual ≤ P75` | MAE binned by the actual (↑) |
| `below_zero` | `actual < 0 MWh` | MAE binned by the actual (↑) |
| `low_extreme` | `actual ≤ P1` | MAE binned by the actual (↑) |
| `high_extreme` | `actual > P99` | MAE binned by the actual (↑) |

These are Scoreboard B's bins and edges (§7.3 of the regression notebook). `low_extreme` lies
inside `below_zero` whenever `P1 < 0`. A model can be picked in several categories.

## Scope

### IN

Loading and a reproduction self-check (Behaviour 1–5), the pick rules (6–10), the overview tables
(11–13), ten plots (14–19) and an insight summary (20).

### OUT

- Scoreboard C (intervals) and prediction bands: covered in §6, §7.4 and §7.6 of the regression
  notebook.
- The day maximum and minimum tables of Scoreboard B.
- Risk classification of our forecasts (spec 02's thresholds, day rules, flags): a later, separate
  notebook. The flagging view (Behaviour 17) only points in that direction.
- Significance tests, ensembles, and any metric beyond Scoreboards A and B.

## Behaviour

### Setup and loading

1. **Title cell** (at most four lines): what the notebook shows, which exports it reads, a link to
   `regression-models-claude.ipynb` for method and caveats, and one sentence on the pick rule.
2. **Loading cell:**
   - Finds `data/` by walking up from the working directory.
   - Reads the three CSVs (see Data) with a bare `pd.read_csv` and parses `timestamp` with
     `format="%Y-%m-%d %H:%M:%S"`.
   - A missing file stops the notebook with a `FileNotFoundError` naming the producing notebook
     (for the model exports, also `EXPORT_ENABLED`).
   - Prints the test window, the rows found, and the time span of the SMARD file.
3. **Rebuilt quantities:**
   - **Test hours:** every timestamp of the hourly export.
   - **Common hours:** the test hours that have an actual, SMARD's `fc_residual_load`, and a
     forecast from every `(model, split_method)` row. A row whose forecast is entirely empty (a
     failed fit) is left out of the intersection, as in §7.1.
   - **Edges:** `P1, P25, P75, P99` are the `{0.01, 0.25, 0.75, 0.99}` quantiles of the actual over
     the test hours; the zero edge is `0 MWh`. Inequalities as in §7.3's `tail_bin`.
   - **Full months:** the calendar months whose first and last day both lie in the test window
     (§7.2). Monthly MAE is taken over the common hours.
   - **SMARD:** `err_residual_load` and `fc_residual_load` from the metrics file, joined on
     `timestamp`.
4. **Configuration cell**, every setting in one place:

   ```python
   N_PICKS = 3                 # top picks per category
   MIN_BIN_SHARE = 0.5         # by-forecast hours / by-actual hours, tail categories
   CANDIDATE_SPLIT = "rolling" # only these rows compete
   ```

   - `STYLE`: label and colour per row, copied from `MODELS` and `FIXED` in
     `regression-models-claude.ipynb` (all seven registry models, `actual`, `smard`), because the
     exports carry no colours. SMARD is drawn dashed.
   - `TAIL_COLOR`:

     | Bin | Colour |
     | --- | --- |
     | `low_extreme` | `#17BECF` (cyan) |
     | `below_zero` | `#9EDAE5` (light cyan) |
     | `high_extreme` | `#E6B800` (gold) |
     | `ordinary` | `#E3E8EF` (light grey) |

     These colours only mark bin regions and edges on a residual-load axis. They are never a
     model's colour, and they don't clash with any registry colour.
   - `style_timeseries(ax, title, ylabel)`, copied from team-EDA §1. Plots without a time axis get
     the same look: no top or right spine, a light y grid, thousands separators.
5. **Self-check:** it stops on any failure, naming the file to regenerate.
   - For every exported row, the rebuilt MAE and hour count on the common hours equal the
     scoreboard's `accuracy / MAE` (tolerance `1e-6`).
   - For every bin, the by-actual hours equal the `count` of `MAE_<bin>_by_actual`.
   - The rebuilt months beating SMARD equal `months_beating_smard`.
   - The export's `residual_load` equals the SMARD file's `residual_load` on the common hours.

   On success it prints one line:
   `self-check passed: exports reproduce the scoreboard (<n> rows, <h> common hours)`.

### Picks

6. **Candidates:** the registry rows with `split_method == CANDIDATE_SPLIT`. SMARD and seasonal naive
   never compete.
7. **`overall`:** a row qualifies when its skill vs SMARD is above 0, ranked as in the table in Goal.
8. **Bin categories:** a row qualifies when all three of these hold (§7.3's "both views" rule).
   Qualifying rows are ranked by `MAE_<bin>_by_actual`.
   - **By actual:** `MAE_<bin>_by_actual` is below SMARD's.
   - **By forecast:** `MAE_<bin>_by_forecast` is below SMARD's. The two views count different
     hours, so this compares levels, as §9.2 does.
   - **Reach:** its by-forecast hours are at least `MIN_BIN_SHARE` × its by-actual hours. This stops
     a model that rarely forecasts a tail value from winning on a few confident hours.
9. **Top picks:** the first `N_PICKS` qualifying rows, fewer if fewer qualify. Ties left after
   the ranking keys are broken by model key.
10. **Rejected rows:** under each category, one printed line per rejected candidate naming the rule
    it failed (`by actual`, `by forecast`, `reach 0.17 < 0.5`, `skill ≤ 0`).

### Overview tables

11. **Picks table:** one row per category and rank. Columns:
    - the model label
    - the ranking value (months as `x / n`, or MAE in MWh)
    - SMARD's value, and the skill vs SMARD (%)
    - for the bin categories only: the by-forecast MAE against SMARD's, and the reach
12. **Scoreboard A** (§7.2 layout, without `fit_seconds`) for the union of the picks plus SMARD,
    ordered by MAE, with an extra `picked in` column listing each row's categories.
13. **Scoreboard B**, the same rows and order, as two tables in the §7.3 layout. Day maximum and
    minimum are left out.
    - **By actual:** MAE, bias and skill for the four bins.
    - **By forecast:** MAE, bias, hours, and `reach` (by-forecast hours / by-actual hours).

    Formats: MAE `{:,.0f}`, bias `{:+,.0f}`, skill `{:+.1f}`, reach `{:.2f}`.

### Plots

Every plot:

- shows SMARD plus the category's picks, in rank order and in their `STYLE` colours
- has a title, units on its axes, and a one-line markdown caption above it saying what to read
  (not what it concludes)

Code that several categories share lives in two helpers, `tail_parity(category)` and
`bin_errors(category)`, each defined in the first section that uses it.

14. **Overall, plot 1: monthly skill vs SMARD.**
    - x: full months; y: `100 · (1 − MAE_row / MAE_SMARD)` in `%`.
    - One line with markers per pick, plus a labelled zero line for SMARD.
    - Each legend entry shows `x / n` months beating SMARD.
15. **Overall, plot 2: cumulative advantage over SMARD.**
    - The cumulative sum of `|e_SMARD| − |e_row|` over the common hours in time order, in `MWh`.
    - A rising line means the model is gaining on SMARD. The plot shows whether the gain is steady
      or comes from a few episodes.
    - `style_timeseries`, then month ticks, as in §7.6.
16. **Ordinary, plot 1: MAE by hour of day** on the ordinary hours.
    - x: local hour 0–23; y: MAE in `MWh`.
    - Shows where in the day the models gain. Plot 2 is `bin_errors("ordinary")`.
17. **`tail_parity(category)`: the flagging view** (plot 1 of `below_zero`, `low_extreme` and
    `high_extreme`).
    - **Panels:** one per row (SMARD first), with shared axes. The points are the common hours where
      the actual **or** the row's forecast lies in the bin. x: actual; y: forecast (`MWh`).
    - **Drawing:** the diagonal, the bin edge on both axes in `TAIL_COLOR[category]`, and a tint on
      the quadrant where both lie in the bin.
    - **Annotated counts:**
      - hits: both in the bin
      - misses: only the actual in the bin
      - false alarms: only the forecast in the bin
    - **Panel title:** the reach, `(hits + false alarms) / (hits + misses)`, and whether it passes
      `MIN_BIN_SHARE`.
    - **Consistency:** hits + misses must equal the scoreboard's by-actual hours, and hits + false
      alarms its by-forecast hours.
18. **`bin_errors(category)`: the error distribution** (plot 2 of every bin category, `ordinary`
    included).
    - One box per row (SMARD first) of `forecast − actual` on the by-actual bin hours, in the row's
      colour.
    - A diamond marks the mean (the bias), and a zero line is drawn.
    - y label: `error (MWh), forecast − actual`. The title gives the bin rule and the hour count.
19. **Empty category:** if no row qualifies, the section prints
    `no <CANDIDATE_SPLIT> row beats SMARD in <category>` and draws nothing.

### Insight summary

20. **Closing section**, written with the `interpretation-style` skill.
    - **Opening blockquote:** the summary is written for the exports of the run date; the tables
      above always show the current exports.
    - **Per category, 1–3 bullets:** who wins, by how much against SMARD, and what the two plots
      show.
    - **Three closing bullets:**
      - the best all-rounder, and where the specialists differ from it
      - what the flagging view hints at for the planned classification notebook (misses against
        false alarms in the tails)
      - that the extreme bins hold about 88 hours each, so tail claims stay qualitative
    - **Other caveats** (one test year, post-processor of SMARD, archive values): linked to §9.2 of
      the regression notebook, not repeated.

## Data

| File | Used columns | Role |
|---|---|---|
| `data/models/model_scoreboard.csv` | `model`, `split_method`, `table`, `metric`, `value`, `count` | ranking values and every number in the overview tables |
| `data/models/model_forecast_errors_hourly.csv` | `timestamp`, `model`, `split_method`, `residual_load`, `forecast`, `err_residual_load` | hourly errors for the plots; common hours; edges |
| `data/metrics/smard_forecast_errors_hourly.csv` | `timestamp`, `residual_load`, `fc_residual_load`, `err_residual_load` | SMARD's hourly values (not in the model export) |

- **Format:** plain CSV. Timestamps are naive local time (`Europe/Berlin`), and the files join on
  `timestamp`. `err_* = forecast − actual`.
- **Not used:** the `lower` / `upper` columns of the hourly export.
- **Not read:** the `-magc` exports (their bins and window differ, see CLAUDE.md).
- **Nothing hardcoded:** no date, year, model name or edge. All of them come from the files.

For review only (**not** hardcoded): with the exports of 2026-09-30 and the defaults, the picks
would be:

| Category | Picks (rolling) |
|---|---|
| overall | `lgbm_direct` (10/11), `linear_direct` (10/11), `random_forest_hybrid` (9/11, lower MAE than `xgb_hybrid`) |
| ordinary | `linear_direct`, `random_forest_hybrid`, `lgbm_hybrid` |
| below_zero | `xgb_hybrid`, `random_forest_hybrid`, `lgbm_hybrid` |
| low_extreme | `random_forest_hybrid`, `linear_direct`, `xgb_hybrid` (`xgb_direct` and `lgbm_direct` fail by actual) |
| high_extreme | `xgb_hybrid`, `lgbm_direct`, `xgb_direct` (the other hybrids and Ridge fail by forecast) |

The union covers all six rolling rows, so Scoreboards A and B show every rolling row. The picks
table is what separates them.

## Edge cases

- **Exports from different runs.** The hourly file and the scoreboard come from different runs, or
  the SMARD metrics file was regenerated after a re-fetch that changed actuals. Result: the
  self-check stops and names the file to regenerate. Regenerate in this order:
  `forecast-metrics-claude`, then `regression-models-claude` with `EXPORT_ENABLED`.
- **No `CANDIDATE_SPLIT` rows** (rolling was off in the last run). Result: stop, suggesting
  `CANDIDATE_SPLIT = "static"`.
- **A model missing from `STYLE`** (a new registry model). Result: stop, naming the key to add.
  Never fall back to a default colour. A changed registry colour has to be copied into `STYLE` by
  hand.
- **Fewer than `N_PICKS` qualify.** Result: those are shown, and the rejected lines explain the rest.
  If none qualify, Behaviour 19 applies. `N_PICKS` above the number of candidates is no error.
- **An empty bin** (for example no hour below zero in a test year). Result: nothing qualifies, and
  Behaviour 19 applies.
- **A row with an all-empty forecast.** Result: it is left out of the common hours and is no
  candidate, because it has no scoreboard row.
- **DST.**
  - The hour-of-day plot groups by local hour. The spring day lacks 02:00 and the autumn day has
    24 rows, so no correction is needed.
  - Only the cumulative plot uses timestamps, as its x axis, and it never subtracts them.
- **`P1 ≥ 0`.** Result: `low_extreme` is no longer inside `below_zero`. Nothing breaks: the parity
  plot draws the edge where `P1` is.

## Acceptance criteria

- [ ] The notebook exists and runs top to bottom on a fresh kernel in well under a minute, reading
      only the three CSVs in Data.
- [ ] It writes no files and edits no other notebook or export.
- [ ] Changing the three settings of Behaviour 4 and re-running changes the picks, tables and plots
      with no other edit.
- [ ] The self-check (Behaviour 5) passes on consistent exports and stops on a mismatch.
- [ ] Every pick satisfies Behaviour 7–8, and every rejected candidate is listed with its rule.
- [ ] The picks table, Scoreboard A (with `picked in`) and both Scoreboard B views come before the
      first plot.
- [ ] Every category has exactly two plots, or none per Behaviour 19, each with a title, units and a
      caption.
- [ ] The parity counts match the scoreboard hours (Behaviour 17).
- [ ] Models use registry colours, tail colours mark only bin regions and edges, and SMARD is dashed.
- [ ] Nothing is hardcoded (see Data).
- [ ] The insight summary follows Behaviour 20.

## Out of bounds

- Editing `regression-models-claude.ipynb`, `regression-models-magc.ipynb`, their specs, or any file
  in `data/`.
- Fitting, tuning or loading models (`results.joblib`, `config.json`); changing the scoring, the bins
  or the common-hours rule.
- Reading `data/smard.csv`, `data/rebap.csv`, the `-magc` exports or the risk labels.
- Streamlit changes, including `streamlit/viz_helpers.py`.
- Long setup explanations, or method text repeated from the regression notebook.
