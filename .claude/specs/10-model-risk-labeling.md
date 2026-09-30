# 10 — Model Risk Labeling: Spec 02's Risk Flags on Our Best Forecasts

- Status: **draft** 2026-09-30, not run. Branch: `feature/risk-classification-visualization`.
- Deliverable: `notebooks/03_risk_classification/classification-visualization-risk-labels.ipynb`
  (no suffix: team choice, named to match `regression-visualization-best-models.ipynb`). It writes
  two label files (Behaviour 16).
- Depends on: the risk labels of [02-Risk-Definition.md](02-Risk-Definition.md)
  (`data/risk_classification/`), the exports of [06-regression-models.md](06-regression-models.md)
  (Behaviour 28), `data/metrics/smard_forecast_errors_hourly.csv` from
  [04-forecast-metrics.md](04-forecast-metrics.md), and the picks of
  [09-best-model-plots.md](09-best-model-plots.md), copied into the settings cell.
- Built one section at a time, with a team review per section (`spec-run-section-loop` skill).
- The parked [04.3](04.3-risk-label-link.md) stays parked (whole record, error on risk hours). This
  spec benchmarks SMARD's flags **on the test window only**.

## Goal

`risk-definition.ipynb` flags a day, and the hours within it, as at risk when the **actual**
`residual_load` crosses a threshold, in two independent directions. Spec 09 found no model that
beats SMARD in both tails. So this notebook applies the risk definition to **our forecasts**, using
spec 09's rank-1 model for each tail:

| Direction | Model (spec 09 category) | Bases | Rules |
|---|---|---|---|
| high | `xgb_hybrid` (`high_extreme`) | `rolling` (P99, trailing 365 days) | `any`, `3h` |
| low | `random_forest_hybrid` (`low_extreme`) | `rolling` (P1, trailing 365 days) as **the label**; `zero` (< 0 MWh) as a second, physical flag | `any`, `3h` |

1. Flag every test day and hour on the model's forecast, on SMARD's forecast and on the actual. Use
   the **exported thresholds** and both day rules: `any` ("did the day touch the threshold?") and
   `3h` ("did it sustain it?").
2. Score the forecast flags against the actual flags, with SMARD as the benchmark.
3. Draw three plots per extreme, reusable in Streamlit and the presentation.
4. Bundle the findings in one closing section.

The notebook is **concise**, like `regression-visualization-best-models.ipynb`: it explains no
setup, fits no model and restates no method. The threshold method is in `risk-definition.ipynb`,
the forecast method in `regression-models-claude.ipynb`, and the pick rule in spec 09. All three are
linked, not repeated.

## Scope

**IN:**
- setup and checks (Behaviour 1–5)
- the flags of three sources, with a check that reproduces the actual labels (6–9)
- the scores table (10)
- three plots per extreme, six in all (11–14)
- the export of two label files (15–17)
- the closing findings (18)

**OUT** (things left out of the notebook; forbidden actions are under Out of bounds):
- The `static` basis (demoted in `risk-definition.ipynb` §3.4).
- Choosing a model: the picks come from spec 09. Also out: ensembles, and a model per basis (for
  example spec 09's `below_zero` pick for `zero`). Team decision (2026-09-30): one low model serves
  `rolling` and `zero`, as a starting point given the time left.
- Flagging on a shifted edge (spec 09 §9.3 "to try": a margin chosen by the cost of a miss against a
  false alarm). This is for a later spec.
- Prediction bands (`lower` / `upper`), probabilistic flags and flag probabilities.
- Ramps, capacity normalisation, and one target or two (open points in `risk-definition.ipynb` §8).
- Plain accuracy as a headline. With few risk days, a forecast that never flags scores near 100 %
  (04.3's rule).
- reBAP and cost calculation.
- Streamlit code. The plots are only built to be portable (Behaviour 11).

## Behaviour

### Setup and checks

1. **Title cell** (at most 5 lines):
   - what the notebook does, and the files it reads and writes
   - links to `risk-definition.ipynb` (thresholds, rules, limits of the label),
     `regression-visualization-best-models.ipynb` (picks) and `regression-models-claude.ipynb`
     (forecast method, caveats)
   - one sentence: *the label is a national-balance proxy, not a validated intervention record*
2. **Settings cell**, holding every setting:

   ```python
   PICKS = {"high": "xgb_hybrid", "low": "random_forest_hybrid"}  # spec 09 rank 1: high_extreme / low_extreme
   PICK_CATEGORIES = {"high": ["high_extreme"], "low": ["low_extreme", "below_zero"]}  # checked in Behaviour 4
   SPLIT = "rolling"            # split method of the picks (spec 09's CANDIDATE_SPLIT)
   BASES = {"high": ["rolling"], "low": ["rolling", "zero"]}
   RULES = ["any", "3h"]
   MIN_BIN_SHARE = 0.5          # spec 09's forecast-share floor, used by the pick check
   ZOOM_WEEKS = {"high": [None, None], "low": [None, None]}  # a Monday "YYYY-MM-DD" overrides a zoom rule (Behaviour 12)
   EXPORT_ENABLED = True
   ```

   - `STYLE`, `style_timeseries` and `style_plain` are copied from spec 09's notebook. `STYLE`
     covers all seven registry models, `actual` and `smard`, with SMARD drawn dashed.
   - Colours. None of them is a registry colour. `TAIL_COLOR` marks thresholds, flagged hours and
     hit cells, never a model.

     | Key | Colour |
     |---|---|
     | `TAIL_COLOR` high × `rolling` | `#E6B800` (gold, spec 09's `high_extreme`) |
     | `TAIL_COLOR` low × `rolling` | `#17BECF` (cyan, `low_extreme`) |
     | `TAIL_COLOR` low × `zero` | `#9EDAE5` (light cyan, `below_zero`) |
     | `OUTCOME_COLOR` hit / miss / false alarm | the direction's `TAIL_COLOR` / `#1C1C1C` / `#AEB8C5`, edged in `TAIL_COLOR` |
     | `OUTCOME_COLOR` quiet / not evaluable or outside the test window | `#F2F4F7` / white |

   - Rule constants, written as **durations** as in `risk-definition.ipynb` §3.1:
     `PERSISTENCE = 3h`, `DAY_COMPLETENESS = 23/24`, and `RESOLUTION` measured from the data.
3. **Loading:**
   - Finds `data/` by walking up from the working directory.
   - Reads the five CSVs listed under Data with a bare `pd.read_csv`. It parses `timestamp` with
     `format="%Y-%m-%d %H:%M:%S"` and `date` as a date.
   - A missing file raises `FileNotFoundError`, naming the notebook that produces it.
   - Label columns stay **nullable booleans**: an empty cell means not evaluable. `fillna(False)` is
     never applied to a label.
   - **Test days** are the local dates of the model export. **Common hours** are the test hours with
     an actual, SMARD's `fc_residual_load` and a forecast from both picks.
   - Prints the test window, the picks and the number of common hours.
4. **Pick check.** Each pick must meet two conditions:
   - It has a `SPLIT` row in both model exports.
   - It passes spec 09's bin rule in every `PICK_CATEGORIES` category, read from
     `model_scoreboard.csv`: `MAE_<bin>_by_actual` and `MAE_<bin>_by_forecast` are both below
     SMARD's, and its by-forecast hours are at least `MIN_BIN_SHARE` × its by-actual hours.

   If a pick fails (a re-run changed spec 09's ranking, or the model was switched off), the notebook
   stops and prints `PICKS` and the failed rule. It never falls back to another model silently and
   never re-ranks. After spec 09 is re-run with new picks, the team edits `PICKS`.
5. **Snapshot self-check.**
   - `residual_load` agrees on the common hours (tolerance `1e-6`) across three files: the model
     export, `smard_forecast_errors_hourly.csv` and `risk_labels_hourly.csv`.
   - Every test day is a row of `risk_labels_daily.csv`, and every test hour a row of
     `risk_labels_hourly.csv`.

   On a failure (for example, labels built from an older `smard.csv`), the notebook stops and names
   the regeneration order `API-connection` → `risk-definition` → `forecast-metrics-claude` →
   `regression-models-claude`. It never joins on a partial overlap. On success it prints
   `self-check passed: labels, model exports and SMARD file share one snapshot (<d> days, <h> common hours)`.

### Labels

6. **Rule functions**, copied from `risk-definition.ipynb`: §3.2 `hourly_crossings` and
   `day_rules`, §5.1 `flagged_range`.
   - They are copied because a notebook cannot import another. Behaviour 8 proves the copy
     behaves the same.
   - They run on the test days only.
   - They take any hourly series and the **exported** per-day threshold, joined on `date`. No
     threshold is computed in this notebook.
   - The comparisons: high `value >= threshold`, low `value <= threshold`. The zero basis uses
     `low_threshold_zero`, which is `0`.
7. **Sources and evaluable days.**
   - The three sources:
     - `actual`: flags and ranges **read from `risk_labels_*.csv`**, not recomputed
     - `model`: `PICKS[direction]`, the `SPLIT` rows of the model export
     - `smard`: `fc_residual_load`, from the metrics file
   - For each source × direction × basis, the notebook builds:
     - hourly flags, on the common hours
     - day flags under `any` and `3h`
     - the flagged span (start, end) per day and rule, as in `risk-definition.ipynb` §5.1
   - A day is **evaluable** when its actual label is not empty and its common hours cover at least
     `DAY_COMPLETENESS` of its hours.
   - On any other day, every flag is empty for every source. The day is excluded from the scores and
     counted in the header line, and it is never scored as quiet.
8. **Rule reproduction check.** On the test days whose hours are all common, the rule functions
   applied to the actual `residual_load` must reproduce the file's `any` / `3h` flags and ranges
   exactly, for each direction × basis. Otherwise the notebook stops.
   - The check uses only fully common days. On a day that lost a common hour, the file's flag may
     rest on an hour the forecasts don't have.
   - The header line counts such days.
9. **Structural checks**, on every source:
   - `3h` implies `any`.
   - A range exists exactly where the day is flagged.
   - The day `any` flag equals "at least one flagged hour".

   The notebook also prints the number of days flagged in **both** directions per source (expected
   0). Such days are allowed and reported, not prevented, as in `risk-definition.ipynb` §4.2. A
   non-zero count hints at a join error.

### Scores

10. **Scores table**, shown before the first plot.
    - **Outcome** of `model` and `smard` against `actual`, per evaluable day and per common hour:
      - **hit:** both are flagged
      - **miss:** only the actual is flagged
      - **false alarm:** only the forecast is flagged
      - **quiet:** neither is flagged
    - **Rows:** `high rolling any`, `high rolling 3h`, `low rolling any`, `low rolling 3h`,
      `low zero any`, `low zero 3h`.
    - **Column groups:** `model` and `SMARD`, each with:
      - days: actual flagged, hits, misses, false alarms
      - `recall = hits / (hits + misses)` and `precision = hits / (hits + false alarms)`, in %
      - hours: hits, misses, false alarms. Hours have no `3h` rule, so the `3h` rows repeat the
        hour counts of their `any` row.
    - A zero denominator gives an empty cell, never 0 or 100. There is no accuracy column and no
      correct-negative column: quiet is the remainder.
    - **Header:** the test window, the evaluable and excluded days, and the picks.
    - **Formats:** counts `{:,.0f}`, recall and precision `{:.0f}`.

### Plots

- **Helpers:** there are three, `risk_calendar(direction)`, `event_zoom(direction)` and
  `day_margin(direction)` (plots A–C). They are defined in the high section and called again in the
  low section.
- **Every plot:**
  - has a title, axis units, and a one-line markdown caption saying **what to read**. Conclusions go
    only in Behaviour 18.
  - reads only the export frames (Behaviour 16), so Streamlit can port the helper
  - draws `model` and `SMARD` in their `STYLE` colours, and the actual in black
- **Holidays and weekends** get no plot of their own. The calendar markers and the low zoom rules
  carry them.

11. **Plot A, `risk_calendar`: the test year.**
    - **Layout:** a day grid with ISO weeks as columns (Monday start) and weekdays as rows, month
      labels on top. Two stacked panels on shared axes: **model** above **SMARD**.
    - **Fill:** the day's outcome under `any` on the `rolling` basis (`OUTCOME_COLOR`). Only
      `rolling` is shown here; `zero` appears in plots B and C.
    - **Markers:**
      - a white dot: the actual is also flagged `3h`
      - a bold border: a public holiday (`holidays.country_holidays("DE")`, no `subdiv`, the years of
        the test window)
    - **Legend:** the outcome colours with day counts per panel, plus both markers.
    - Days outside the window, in the partial first and last ISO weeks, are white, not quiet.
12. **Plot B, `event_zoom`: hourly zoom-ins.** Two stacked Monday–Sunday test weeks, picked by the
    zoom rules below or set by `ZOOM_WEEKS`.
    - **Main axes:**
      - the actual (black), SMARD (dashed) and the model, hourly, in `MWh`
      - the day's threshold as a `TAIL_COLOR` step line. For low: `rolling`, plus the dotted 0 MWh
        line of `zero`.
    - **Flag strip** below the main axes, with three rows (actual, model, SMARD) and one cell per
      hour:
      - full `TAIL_COLOR` for an hour in a run that qualifies under `3h`
      - a light tint for an hour flagged under `any` only
      - For low, the strip shows the `rolling` flags.
    - **x axis:** one tick per day (`%a %d %b`), with a holiday's name as a second line.
    - **Printed per panel:** the week, why it was picked, and per flagged day the Definition 2
      sentence, e.g. `Wed 03 Dec: actual 16:00–20:00, model 17:00–19:00, SMARD —`, built from the
      ranges of Behaviour 7.
    - **Zoom rules**, taken from the data, never from dates. Seasons are meteorological (spec 01).

      | | Panel 1 | Panel 2 |
      |---|---|---|
      | high | the week with the most actual `high rolling any` days (ties: the highest actual day maximum) | the week with the most misses and false alarms of model and SMARD together, excluding panel 1's week (ties: the earliest) |
      | low | the week of the flagged (`low rolling any`) public holiday in Mar–Aug with the lowest actual day minimum | the week of the flagged `low rolling any` day in Dec–Feb with the lowest actual day minimum |

      If no week meets a rule, that panel prints `no week meets: <rule>` and stays empty. The other
      panel still draws.
13. **Plot C, `day_margin`: the day-level flagging view**, the day version of spec 09's
    `tail_parity`.
    - **Dots:** one per evaluable day, in `MWh`. x is the actual day extreme minus the threshold, and
      y is the same for the forecast. The extreme is the maximum for high and the minimum for low,
      taken over the common hours.
    - **Panels:** SMARD, then model, on shared square axes. For low there is one row per basis
      (`rolling`, then `zero`).
    - **Drawing:**
      - both zero lines and the diagonal
      - a `TAIL_COLOR` tint on the hits quadrant
      - the counts (hits, misses, false alarms), placed with spec 09's `quiet_spot`
    - **Markers:** filled when the actual day is `3h`, hollow when it is `any` only. This shows which
      misses are single-hour excursions.
    - **Axis range:** clipped to the days with a margin within ± `MARGIN_WINDOW` (a module constant,
      default 10,000 MWh) of 0, so quiet days don't squash the interesting corner. The number of
      days left outside is printed.
    - **Consistency:** the quadrant counts must equal the `any` rows of the scores table, or the
      notebook stops.
14. **Empty direction.** If the actual flags no day in a direction (for example a mild winter for
    high), the calendar still draws, `recall` is empty, and plots B and C print why they show
    nothing.

### Export

15. **Toggle:** the export frames are always built in memory. The files are written only when
    `EXPORT_ENABLED` is on (default on). `data/risk_classification/` exists (it holds a `.gitkeep`),
    and the existing `data/risk_classification/*.csv` rule gitignores the new files.
16. **Two files**, plain CSV (`sep=","`, `decimal="."`, UTF-8), like `risk_labels_*.csv`:

    | File | Grain | Columns |
    |---|---|---|
    | `model_risk_labels_daily.csv` | one row per test day | `date`, `evaluable`, `high_model`, `low_model` (model keys); `high_threshold_rolling`, `low_threshold_rolling`, `low_threshold_zero` (copied); `actual_max`, `actual_min`, `model_max` (high model), `model_min` (low model), `smard_max`, `smard_min` (common hours); flags `{source}_{direction}_risk_{basis}_{rule}`; ranges `{source}_{direction}_range_{start,end}_{basis}_{rule}` |
    | `model_risk_labels_hourly.csv` | one row per common test hour | `timestamp`, `residual_load`, `high_model_forecast`, `low_model_forecast`, `fc_residual_load`; flags `{source}_{direction}_risk_hour_{basis}` |

    - `source` ∈ {`actual`, `model`, `smard`}. In `high_*` columns `model` means `high_model`, and
      in `low_*` columns `low_model`.
    - The naming is that of `risk_labels_*.csv` with a source prefix, so `smard_low_risk_zero_3h`
      reads without the notebook.
    - An empty flag means **not evaluable**, never "not at risk".
    - Timestamps are written as `%Y-%m-%d %H:%M:%S` (naive local time), dates as `%Y-%m-%d`.
17. **Round-trip check**, with the export on: both files are read back with a bare `pd.read_csv`.
    - The row counts equal the test days and the common hours.
    - The flags reload as three states.
    - The Behaviour 9 checks pass on the files as written.

### Findings

18. **Closing section**, written with the `interpretation-style` skill. It holds all of the
    notebook's interpretation.
    - **Opening blockquote:** the findings are written for the exports of the run date. The tables
      and plots above always show the current exports.
    - **Per extreme, 2–4 bullets:**
      - model against SMARD on days and on hours: hits, misses and false alarms, `any` against `3h`
      - for low, `rolling` against `zero`
      - what the calendar shows: season, weekday against weekend, holidays
      - what the zoom weeks show: the timing of the flagged ranges, near misses
    - **Closing bullets:**
      - whether spec 09's hourly finding (fewer misses, not fewer false alarms) holds at the day
        level
      - how much of the flagging difference `3h` removes
      - small counts: a test year holds about a dozen high days, so high-direction claims stay
        qualitative
      - the pick mismatch: the models were picked on spec 09's fixed test-year `P1` / `P99` bins,
        but the label uses the moving `rolling` threshold
    - **Other caveats**, linked, not repeated:
      - the limits of the label (`risk-definition.ipynb` §2.1 / §8)
      - one test year, and the models as post-processors of SMARD (`regression-models-claude.ipynb`
        §9.2)
      - the threshold timing caveat (Edge cases)

## Data

| File | Used columns | Role |
|---|---|---|
| `data/risk_classification/risk_labels_daily.csv` | `date`, `high_threshold_rolling`, `low_threshold_rolling`, `low_threshold_zero`, `{high,low}_risk_rolling_{any,3h}`, `low_risk_zero_{any,3h}`, the matching `*_range_*` columns | thresholds and actual labels |
| `data/risk_classification/risk_labels_hourly.csv` | `timestamp`, `residual_load`, `high_risk_hour_rolling`, `low_risk_hour_rolling`, `low_risk_hour_zero` | actual hour flags; snapshot check |
| `data/models/model_forecast_errors_hourly.csv` | `timestamp`, `model`, `split_method`, `residual_load`, `forecast` | the picks' forecasts; test window |
| `data/models/model_scoreboard.csv` | `model`, `split_method`, `metric`, `value`, `count` | pick check (Behaviour 4) only |
| `data/metrics/smard_forecast_errors_hourly.csv` | `timestamp`, `residual_load`, `fc_residual_load` | SMARD's forecast (not in the model export) |

- **Joins:** the hourly files join on `timestamp`. A threshold reaches an hour through the hour's
  local date.
- **Not read:** `data/smard.csv` (holidays come from the `holidays` package), `data/rebap.csv`, the
  `-magc` exports, the `static` columns, and the `lower` / `upper` bands.
- **Nothing hardcoded:** no date, year, threshold or count. The only exception is the two model keys
  in `PICKS`, by design (Behaviour 2 and 4).

For review only, **not** hardcoded. With the files of 2026-09-30 (test window 2025-09-07 ..
2026-09-06), the actual labels flag:

| Direction × basis | `any` days | `3h` days | Notes |
|---|---|---|---|
| high × rolling | 12 | 7 | all weekdays, Nov–Mar, no holiday |
| low × rolling | 57 | 42 | 37 on weekends; 6 holidays (e.g. 1 May, Pentecost Monday, 1 Jan) |
| low × zero | 111 | 93 | Apr–Sep dominate |

Expected zoom weeks with the default rules:

- **High:** panel 1 is 1–7 Dec 2025, with 3 flagged days.
- **Low:** panel 1 is 25–31 May 2026, the week of Pentecost Monday (the record low of
  −15,562 MWh). Panel 2 is 29 Dec 2025 – 4 Jan 2026 (New Year's Day, a flagged winter low).
- Christmas (24–26 Dec) is flagged in neither direction in this test year.

## Edge cases

- **Handled in Behaviour:**
  - files from different snapshots (5)
  - a pick that no longer qualifies, or is missing (4)
  - not evaluable days (7)
  - a day that lost a common hour (8)
  - a day flagged in both directions (9)
  - zero denominators (10)
  - the edges of the test window (11)
  - an empty direction (14)
- **The threshold is not quite known at issue time.**
  - Day `D`'s rolling threshold uses actuals up to the end of `D−1`. The forecast is issued at
    18:00 on `D−1`, when actuals are known only up to 16:00.
  - The exported threshold is used unchanged (CLAUDE.md: never recompute). At most 8 of the 8,760
    window hours are affected.
  - This is stated as a caveat, not corrected.
- **DST.**
  - `3h` counts consecutive rows, as `risk-definition.ipynb` does (its documented approximation on
    the spring day).
  - The spring day has 23 rows and passes `DAY_COMPLETENESS`; the autumn day has 24.
  - The zoom strips and the calendar use local dates and hours, and never subtract timestamps.
- **`P1 ≥ 0` in a future test year**, which makes the rolling low threshold positive. Nothing
  breaks. The `zero` basis is then the only physical reading, and the findings should say so.
- **Small counts.** Every count is shown with its denominator. No significance test is run.

## Acceptance criteria

- [ ] The notebook exists at the deliverable path and runs top to bottom on a fresh kernel in well
      under a minute. It reads only the five files under Data, fits and loads no model, and edits no
      other notebook, spec output or input file.
- [ ] The checks pass on consistent files and stop on a mismatch:
      - the pick check (Behaviour 4), with the default `PICKS`
      - the snapshot self-check (5)
      - the rule reproduction check (8), on every fully common test day
- [ ] No label column is ever filled with `False`. Not-evaluable days are empty in the flags and in
      the exports, and they are counted.
- [ ] The scores table (Behaviour 10) comes before the first plot, with six rows, `model` and
      `SMARD` column groups, and no accuracy column.
- [ ] Each extreme has exactly three plots (A calendar, B zoom, C day margin), each with a title,
      units and a caption.
- [ ] The quadrant counts of plot C match the scores table.
- [ ] The helpers read only the export frames.
- [ ] Models use their registry colours, `TAIL_COLOR` marks only thresholds, flags and hits, and
      SMARD is dashed.
- [ ] With `EXPORT_ENABLED` on, both files are written in the Behaviour 16 layout and pass the
      round-trip check. With it off, nothing is written.
- [ ] Changing `PICKS` (to a qualifying model), `ZOOM_WEEKS` or `EXPORT_ENABLED` and re-running
      needs no other edit.
- [ ] Nothing is hardcoded (see Data), and all interpretation sits in the closing section
      (Behaviour 18).

## Out of bounds

- Editing any existing file in `data/` (in particular `risk_labels_*.csv`), or any of these:
  - `risk-definition.ipynb`
  - `regression-models-claude.ipynb`
  - `regression-visualization-best-models.ipynb`
  - their specs
- Recomputing, re-tuning or re-leveling thresholds, changing the percentile (it stays 1 %) or the day
  rules, or adding a basis.
- Fitting, tuning or loading models (`results.joblib`, `config.json`), or changing the picks'
  scoring.
- Reading files other than the five under Data (e.g. `data/rebap.csv`, the `-magc` exports).
- Streamlit changes, including `streamlit/viz_helpers.py`. Porting the plots is a later step.
- Long setup explanations, or method text repeated from the three linked notebooks.
- Un-parking spec 04.3.
