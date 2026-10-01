# 10 — Model Risk Labeling: Spec 02's Risk Flags on Our Best Forecasts

- Status: run 2026-09-30. Branch: `feature/risk-classification-visualization`.
- Deliverable: `notebooks/05_modeling/visualization-02-classification-risk-labels.ipynb`
  (no suffix: team choice, numbered after `visualization-01-regression-best-models.ipynb`; moved
  from `03_risk_classification/classification-visualization-risk-labels.ipynb`). It writes
  two label files (Behaviour 16).
- Depends on: the risk labels of [02-Risk-Definition.md](02-Risk-Definition.md)
  (`data/risk_classification/`), the exports of [06-regression-models.md](06-regression-models.md)
  (Behaviour 28), `data/metrics/smard_forecast_errors_hourly.csv` from
  [04-forecast-metrics.md](04-forecast-metrics.md), and the picks of
  [09-best-model-plots.md](09-best-model-plots.md), copied into the settings cell.
- Built one section at a time, with a team review per section (`spec-run-section-loop` skill).
- The parked [04.3](04.3-risk-label-link.md) stays parked (whole record, error on risk hours). This
  spec benchmarks SMARD's flags **on the test window only**.
- The body is the spec as run; the notebook follows the changes below.

## Changes during the run

| Change | Why |
|---|---|
| High `TAIL_COLOR` is crimson-pink `#E0436B`, not gold; `OUTCOME_COLOR` keeps only quiet / not evaluable | Gold and black / grey were too similar; the red stays apart from both XGBoost reds |
| Outcomes by texture: hit = fill, miss = outline, false alarm = hatch, all in `TAIL_COLOR` (plots A and C) | Readable without colour, and the same reading in both plots |
| Holidays: dashed olive border `HOLIDAY_COLOR = #8A9A2B` at lower opacity | A colour no model or tail uses |
| Plot A: quiet, not evaluable (cross-hatched, only if such days exist, with its count), the `3h` dot and holidays in one legend under the plot; first and last test date under the grid | Per-panel legends keep only the outcomes that differ |
| Plot B: threshold as a dash-dot line with a direct label (high: above the line at Monday 01:00; low: outside the right edge), not in the legend | The label reads without the legend; low's P1 and 0 MWh lines sit too close for labels on the plot |
| Plot B: full bar for a `3h` run, low faint bar for `any` only; white row and time grid in the strip; dotted 06 / 12 / 18 guides; legend split into series and risk label; title "Risk: High Extreme (two test weeks)", y label "Residual Load [MWh]" | Easier to tell sustained from brief flags |
| Plot C: count labels at fixed spots (same height in every panel) as a big number on a light tile over the word, recall under hits, an "ordinary" label; false-alarm points drawn solid with a white ring; y label only on the left panel; `quiet_spot` dropped | For an audience; the hatch hid the false alarms |
| `is_set()` (2.2) turns a flag into a plain mask; used by the checks, `outcome()` and the plots | Empty flags crashed `outcome()` and slipped through the structural checks |
| Export frames built in 2.2; everything below reads only them | The checks and plots share one source |
| Extra checks: hour flags in the reproduction check (8), "empty exactly where not evaluable" in the structural checks (9), every value compared in the round trip (17) | Guards the files the scores rely on |

## Goal

`risk-definition.ipynb` flags a day, and its hours, as at risk when the **actual** `residual_load`
crosses a threshold, in two independent directions. Spec 09 found no model that beats SMARD in both
tails, so this notebook applies the risk definition to **our forecasts**, with spec 09's rank-1
model per tail:

| Direction | Model (spec 09 category) | Bases | Rules |
|---|---|---|---|
| high | `xgb_hybrid` (`high_extreme`) | `rolling` (P99, trailing 365 days) | `any`, `3h` |
| low | `random_forest_hybrid` (`low_extreme`) | `rolling` (P1) as **the label**; `zero` (< 0 MWh) as a second, physical flag | `any`, `3h` |

It flags every test day and hour on the model, SMARD and the actual with the **exported
thresholds**, scores the forecast flags against the actual's (SMARD as benchmark), draws three
portable plots per extreme and closes with the findings. It is **concise**: no setup text, no model
fitting, no method restated; `risk-definition.ipynb`, `regression-models-claude.ipynb` and spec 09
are linked instead.

## Scope

**IN:** setup and checks (Behaviour 1–5), the flags of three sources (6–9), the scores table (10),
three plots per extreme (11–14), two label files (15–17), the findings (18).

**OUT** (forbidden actions are under Out of bounds):

- the `static` basis (demoted in `risk-definition.ipynb` §3.4)
- choosing a model, ensembles, or a model per basis. Team decision (2026-09-30): one low model
  serves `rolling` and `zero`, as a starting point given the time left.
- flagging on a shifted edge (spec 09 §9.3 "to try"), prediction bands and flag probabilities
- ramps, capacity normalisation, one target or two (open in `risk-definition.ipynb` §8)
- plain accuracy as a headline: with few risk days, a forecast that never flags scores near 100 %
- reBAP, cost calculation and Streamlit code (the plots are only built to be portable)

## Behaviour

### Setup and checks

1. **Title cell** (at most 5 lines): what the notebook does and which files it reads and writes;
   links to `risk-definition.ipynb`, `visualization-01-regression-best-models.ipynb` and
   `regression-models-claude.ipynb`; the sentence *the label is a national-balance proxy, not a
   validated intervention record*.
2. **Settings cell**, holding every setting:

   ```python
   PICKS = {"high": "xgb_hybrid", "low": "random_forest_hybrid"}  # spec 09 rank 1: high_extreme / low_extreme
   PICK_CATEGORIES = {"high": ["high_extreme"], "low": ["low_extreme", "below_zero"]}  # checked in Behaviour 4
   SPLIT = "rolling"            # split method of the picks (spec 09's CANDIDATE_SPLIT)
   BASES = {"high": ["rolling"], "low": ["rolling", "zero"]}
   RULES = ["any", "3h"]
   MIN_BIN_SHARE = 0.5          # spec 09's forecast-share floor, used by the pick check
   ZOOM_WEEKS = {"high": [None, None], "low": [None, None]}  # a Monday "YYYY-MM-DD" overrides a zoom rule
   EXPORT_ENABLED = True
   MARGIN_WINDOW = 10_000       # MWh: axis range of plot C around 0
   ```

   Plus `STYLE`, `style_timeseries` and `style_plain` copied from spec 09's notebook (seven registry
   models, `actual`, `smard` dashed); `TAIL_COLOR` per direction × basis (low `rolling` cyan
   `#17BECF`, low `zero` light cyan `#9EDAE5`; high see Changes), which marks thresholds, flags and
   hits, never a model; and the rule durations `PERSISTENCE = 3h`, `DAY_COMPLETENESS = 23/24`
   (`RESOLUTION` is measured).
3. **Loading:** finds `data/` by walking up; reads the five files under Data with a bare
   `pd.read_csv`, parsing `timestamp` (`%Y-%m-%d %H:%M:%S`) and `date`; a missing file raises
   `FileNotFoundError` naming its producer. Label columns stay **nullable booleans** (empty = not
   evaluable, never `fillna(False)`). **Test days** are the local dates of the model export;
   **common hours** the test hours with an actual, SMARD's `fc_residual_load` and both picks'
   forecasts. Prints the window, the picks and the common hours.
4. **Pick check:** each pick needs a `SPLIT` row in both model exports and must pass spec 09's bin
   rule in every `PICK_CATEGORIES` category (`MAE_<bin>_by_actual` and `_by_forecast` below SMARD's,
   by-forecast hours ≥ `MIN_BIN_SHARE` × by-actual hours). On a failure the notebook stops with
   `PICKS` and the failed rule; it never falls back or re-ranks. After spec 09 is re-run, the team
   edits `PICKS`.
5. **Snapshot self-check:** `residual_load` agrees (tolerance `1e-6`) on the common hours across the
   model export, the SMARD file and `risk_labels_hourly.csv`, and every test day and hour has a
   label row. Otherwise it stops, naming the regeneration order `API-connection` →
   `risk-definition` → `forecast-metrics-claude` → `regression-models-claude`; it never joins on a
   partial overlap. On success: `self-check passed: labels, model exports and SMARD file share one
   snapshot (<d> days, <h> common hours)`.

### Labels

6. **Rule functions** `hourly_crossings`, `day_rules` (§3.2) and `flagged_range` (§5.1), copied from
   `risk-definition.ipynb` (a notebook cannot import another). They run on the test days, against
   the **exported** per-day threshold joined on `date` (high `>=`, low `<=`, zero basis
   `low_threshold_zero = 0`); no threshold is computed here.
7. **Sources and evaluable days:** `actual` flags and ranges are **read from `risk_labels_*.csv`**;
   `model` is `PICKS[direction]` (`SPLIT` rows); `smard` is `fc_residual_load`. For each source ×
   direction × basis: hour flags on the common hours, day flags under `any` and `3h`, and the
   flagged span per day and rule. A day is **evaluable** when its actual label is set and its common
   hours cover `DAY_COMPLETENESS`; on any other day every flag of every source is empty, and the day
   is counted, excluded from the scores, never scored as quiet.
8. **Rule reproduction check:** on the test days whose hours are all common, the rule functions on
   the actual must reproduce the files' flags and ranges exactly, or the notebook stops. Days that
   lost a common hour are counted, not checked.
9. **Structural checks** on every source: `3h` implies `any`; a range exists exactly where the day
   is flagged; the day `any` flag equals "at least one flagged hour". Days flagged in both
   directions are printed per source (expected 0; allowed, as in `risk-definition.ipynb` §4.2).

### Scores

10. **Scores table**, before the first plot. Outcome of `model` and `smard` against `actual` per
    evaluable day and common hour: **hit** (both flagged), **miss** (only the actual), **false
    alarm** (only the forecast), **quiet** (neither, the remainder). Rows: `high rolling`,
    `low rolling`, `low zero`, each × `any` / `3h`. Per column group (`model`, `SMARD`): actual
    days, hit / missed / false-alarm days, `recall = hits / (hits + misses)` and
    `precision = hits / (hits + false alarms)` in %, and hit / missed / false-alarm hours (the `3h`
    rows repeat their `any` row's hours). A zero denominator gives an empty cell; no accuracy
    column. Header: window, evaluable and excluded days, picks. Formats `{:,.0f}` / `{:.0f}`.

### Plots

Three helpers, `risk_calendar`, `event_zoom` and `day_margin` (plots A–C), are defined in the high
section and called again for low. Every plot has a title, units and a one-line caption saying
**what to read** (conclusions only in Behaviour 18), reads only the export frames (Behaviour 16),
and draws `model` and SMARD in their `STYLE` colours, the actual in black. Styling follows the
Changes table.

11. **Plot A, `risk_calendar`:** the test year as a day grid (ISO weeks as columns, Monday start;
    weekdays as rows; month labels on top), model above SMARD. Each day shows its outcome under
    `any` on `rolling`; a dot marks an actual `3h` day, a border a public holiday
    (`holidays.country_holidays("DE")`, no `subdiv`). Per-panel day counts; days outside the window
    stay blank, not quiet.
12. **Plot B, `event_zoom`:** two Monday–Sunday test weeks. Main axes: the actual, SMARD and the
    model, hourly, with the day's threshold as a step line (low adds the 0 MWh line of `zero`). A
    flag strip below (actual, model, SMARD) marks each flagged hour, distinguishing `3h` runs from
    `any` only; for low it shows `rolling`. One x tick per day (`%a %d %b`), with a holiday's name
    underneath. Printed per panel: the week, why it was picked, and per flagged day the Definition 2
    sentence, e.g. `Wed 03 Dec: actual 07:00–19:00, model 07:00–19:00, SMARD 08:00–18:00`.
    **Zoom rules**, from the data (seasons meteorological), overridden by `ZOOM_WEEKS`:

    | | Panel 1 | Panel 2 |
    |---|---|---|
    | high | most actual `high rolling any` days (ties: highest day maximum) | most misses and false alarms of model and SMARD, panel 1 excluded (ties: earliest) |
    | low | flagged `low rolling any` public holiday in Mar–Aug with the lowest day minimum | flagged `low rolling any` day in Dec–Feb with the lowest day minimum |

    A panel no week meets prints `no week meets: <rule>` and stays empty.
13. **Plot C, `day_margin`:** one dot per evaluable day; x = actual day extreme − threshold, y = the
    same for the forecast (maximum for high, minimum for low, over the common hours). SMARD, then
    model, on shared square axes; for low one row per basis. Zero lines, the diagonal, the four
    quadrants and their counts; filled dots for actual `3h` days, hollow for `any` only. The axes are
    clipped to ± `MARGIN_WINDOW`, with the days left outside printed. The quadrant counts must equal
    the scores table's `any` rows, or the notebook stops.
14. **Empty direction:** if the actual flags no day in a direction, the calendar still draws,
    `recall` is empty, and plots B and C print why they show nothing.

### Export

15. **Toggle:** the export frames are always built; the files are written only when
    `EXPORT_ENABLED` is on (default). The existing `data/risk_classification/*.csv` rule gitignores
    them.
16. **Two files**, plain CSV like `risk_labels_*.csv`:

    | File | Grain | Columns |
    |---|---|---|
    | `model_risk_labels_daily.csv` | one row per test day | `date`, `evaluable`, `high_model`, `low_model`; `high_threshold_rolling`, `low_threshold_rolling`, `low_threshold_zero`; `actual_max`, `actual_min`, `model_max` (high model), `model_min` (low model), `smard_max`, `smard_min`; flags `{source}_{direction}_risk_{basis}_{rule}`; ranges `{source}_{direction}_range_{start,end}_{basis}_{rule}` |
    | `model_risk_labels_hourly.csv` | one row per common test hour | `timestamp`, `residual_load`, `high_model_forecast`, `low_model_forecast`, `fc_residual_load`; flags `{source}_{direction}_risk_hour_{basis}` |

    `source` ∈ {`actual`, `model`, `smard`}; in `high_*` columns `model` is the high pick, in
    `low_*` columns the low pick. An empty flag means **not evaluable**. Dates `%Y-%m-%d`,
    timestamps and ranges `%Y-%m-%d %H:%M:%S` (naive local time).
17. **Round-trip check** (export on): both files read back with a bare `pd.read_csv` have the test
    days and common hours as rows, three-state flags, and pass Behaviour 9.

### Findings

18. **Closing section** (`interpretation-style` skill), holding all interpretation: a dated opening
    blockquote; per extreme 2–4 bullets (model against SMARD on days and hours, `any` against `3h`;
    for low `rolling` against `zero`; the calendar's season, weekday / weekend and holidays; the
    zoom weeks' timing and near misses); closing bullets on whether spec 09's hourly finding (fewer
    misses, not fewer false alarms) holds per day, what `3h` removes, the small counts, and the pick
    mismatch (fixed `P1` / `P99` bins against the moving `rolling` threshold). Other caveats are
    linked: `risk-definition.ipynb` §2.1 / §8, `regression-models-claude.ipynb` §9.2, and the
    threshold timing (Edge cases).

## Data

| File | Used columns | Role |
|---|---|---|
| `data/risk_classification/risk_labels_daily.csv` | `date`, the three thresholds, `{high,low}_risk_rolling_{any,3h}`, `low_risk_zero_{any,3h}`, their `*_range_*` columns | thresholds and actual labels |
| `data/risk_classification/risk_labels_hourly.csv` | `timestamp`, `residual_load`, `high_risk_hour_rolling`, `low_risk_hour_rolling`, `low_risk_hour_zero` | actual hour flags; snapshot check |
| `data/models/model_forecast_errors_hourly.csv` | `timestamp`, `model`, `split_method`, `residual_load`, `forecast` | the picks' forecasts; test window |
| `data/models/model_scoreboard.csv` | `model`, `split_method`, `metric`, `value`, `count` | pick check only |
| `data/metrics/smard_forecast_errors_hourly.csv` | `timestamp`, `residual_load`, `fc_residual_load` | SMARD's forecast |

- Hourly files join on `timestamp`; a threshold reaches an hour through its local date.
- Not read: `data/smard.csv` (holidays come from the `holidays` package), `data/rebap.csv`, the
  `-magc` exports, the `static` columns, the `lower` / `upper` bands.
- Nothing hardcoded (no date, year, threshold or count) except the two model keys in `PICKS`.
- Run of 2026-09-30 (window 2025-09-07 .. 2026-09-06), for checking a re-run only: the actual flags
  high `rolling` 12 / 7 days (`any` / `3h`), low `rolling` 57 / 42, low `zero` 111 / 93; zoom
  weeks 1–7 Dec 2025, 5–11 Jan 2026, 25–31 May 2026 and 29 Dec 2025 – 4 Jan 2026.

## Edge cases

- **Handled in Behaviour:** different snapshots (5), a failing or missing pick (4), not evaluable
  days (7), a day that lost a common hour (8), a day flagged in both directions (9), zero
  denominators (10), the window's partial weeks (11), an empty direction (14).
- **Threshold timing:** day D's `rolling` threshold uses actuals up to the end of D−1, but the
  forecast is issued at 18:00 on D−1 with actuals known up to 16:00. The exported threshold is used
  unchanged (CLAUDE.md: never recompute); at most 8 of the 8,760 window hours are affected. Stated
  as a caveat, not corrected.
- **DST:** `3h` counts consecutive rows, as `risk-definition.ipynb` does; the spring day (23 rows)
  passes `DAY_COMPLETENESS`. Plots use local dates and hours and never subtract timestamps.
- **`P1 ≥ 0` in a future test year:** nothing breaks; `zero` is then the only physical reading, and
  the findings should say so.
- **Small counts:** every count comes with its denominator; no significance test.

## Acceptance criteria

- [ ] Runs top to bottom on a fresh kernel in well under a minute, reading only the five files under
      Data; fits or loads no model; edits no other notebook, spec output or input file.
- [ ] The pick check (4), the snapshot self-check (5) and the rule reproduction check (8) pass on
      consistent files and stop on a mismatch.
- [ ] No label column is ever filled with `False`; not-evaluable days are empty in flags and
      exports, and counted.
- [ ] The scores table (10) comes before the first plot: six rows, `model` and `SMARD` groups, no
      accuracy column.
- [ ] Each extreme has plots A–C, each with a title, units and a caption; plot C's counts match the
      scores table; the helpers read only the export frames.
- [ ] Models use registry colours, `TAIL_COLOR` marks only thresholds, flags and hits, SMARD is
      dashed.
- [ ] With `EXPORT_ENABLED` on, both files follow Behaviour 16 and pass the round trip; off, nothing
      is written.
- [ ] Changing `PICKS` (to a qualifying model), `ZOOM_WEEKS` or `EXPORT_ENABLED` needs no other edit.
- [ ] Nothing hardcoded; all interpretation in Behaviour 18.

## Out of bounds

- Editing an existing file in `data/` (in particular `risk_labels_*.csv`), `risk-definition.ipynb`,
  `regression-models-claude.ipynb`, `visualization-01-regression-best-models.ipynb`, or their specs.
- Recomputing, re-tuning or re-leveling thresholds, changing the percentile (1 %) or the day rules,
  or adding a basis.
- Fitting, tuning or loading models (`results.joblib`, `config.json`), or changing the picks'
  scoring.
- Reading files other than the five under Data.
- Streamlit changes, including `streamlit/viz_helpers.py`.
- Long setup explanations or method text repeated from the linked notebooks.
- Un-parking spec 04.3.
