# 11 — reBAP Cost: What Our Picks Save or Cost Against SMARD

- Status: draft 2026-10-02. Branch: `feature/rebap-visualization`.
- Deliverable: `notebooks/05_modeling/visualization-03-rebap-cost.ipynb` (no suffix). It writes one
  cost file (Behaviour 15).
- Depends on: the exports of [06](06-regression-models.md) (Behaviour 28) and [08](08-ensemble.md),
  `data/metrics/smard_forecast_errors_hourly.csv` ([04](04-forecast-metrics.md)), the picks of
  [09](09-best-model-plots.md) (copied into the settings cell), and `data/rebap.csv`
  (`API-connection.ipynb` Part 2).
- This is the cost calculation CLAUDE.md reserves reBAP for. reBAP never becomes a model feature.
- Built one section at a time with a team review (`spec-run-section-loop` skill).

## Decisions taken before drafting (2026-10-02)

| Decision | Choice | Why |
|---|---|---|
| Resolution | Hourly forecasts, quarter-hourly prices (Behaviour 6) | The 15-min refetch made training slower and the models worse; pricing each quarter-hour keeps the spikes |
| Cost formula | `Δ error × mean\|reBAP_q\|`, absolute price | With a signed price, being more accurate in a negative-price hour counts as a loss (below zero: −32 M€ instead of +50 M€) |
| Day-ahead spread | Not used | `smard.csv` has no day-ahead price; without it a signed settlement rewards over-forecast bias |
| Picks | Spec 09 rank 1, set by hand, with spec 10's ensemble rule; plus a money table of every candidate | Euro and MAE rankings differ (Data); shown, not used to re-pick. Ensembles accepted for now; excluding them later is a team edit |
| Tail hours | By actual (spec 09's bins) | Same hours as the scoreboard, so euro and MWh numbers line up |
| Autumn DST hour | Keep reBAP's second occurrence | SMARD's fetch keeps the later (CET) hour (Behaviour 6), so both files price the same real hour |
| Export | On by default | For a later Streamlit page |

## Goal

Spec 09 shows that the picks beat SMARD in MWh. This notebook prices that advantage with the
**reBAP**, the uniform German imbalance price (EUR/MWh, quarter-hourly). For each category it asks:
*if balancing a MWh of forecast error costs the reBAP of that quarter-hour, what does the pick save
or cost against SMARD, and do price spikes decide it?*

| Category | Hours (by the actual, spec 09) | Model |
|---|---|---|
| `overall` (monthly view) | all common hours | `PICKS["overall"]` |
| `below_zero` | `actual < 0 MWh` | `PICKS["below_zero"]` |
| `low_extreme` | `actual ≤ P1` | `PICKS["low_extreme"]` |
| `high_extreme` | `actual > P99` | `PICKS["high_extreme"]` |

The notebook is concise: no setup text, no fitting, no restated method. It links spec 09's notebook
and `regression-models-claude.ipynb` instead.

**The money is a proxy.** It values *national* forecast error at the imbalance price, and the reBAP
itself responds to the system imbalance. So it measures "the better accuracy, worth this much at the
going price", not "what the TSOs would have paid".

## Scope

**IN:**

| Part | Behaviour |
|---|---|
| Setup and checks | 1–5 |
| Price and cost rules | 6–8 |
| Two overview tables | 9–10 |
| Three plots per category | 11–14 |
| Cost file | 15–16 |
| Findings | 17 |

**OUT** (forbidden actions are under Out of bounds):

- spec 09's `ordinary` category
- signed or day-ahead-spread costs
- 15-min forecasts or actuals, and the error's shape within the hour
- picking by money, or more than one model per category
- the risk labels of specs 02 and 10
- prediction bands and significance tests
- reBAP outside the test window

## Behaviour

### Setup and checks

1. **Title cell** (at most 5 lines):
   - what the notebook shows
   - the files it reads and writes
   - links to `visualization-01-regression-best-models.ipynb` and `regression-models-claude.ipynb`
   - the sentence *the money is a proxy: national forecast error valued at the imbalance price,
     not a settlement*
2. **Settings cell**, holding every setting:

   ```python
   PICKS = {  # by hand: spec 09 rank 1; an ensemble key only if it beats that pick (spec 10 rule)
       "overall": "ensemble_regime_weighted",
       "below_zero": "ensemble_regime_weighted",
       "low_extreme": "random_forest_hybrid",
       "high_extreme": "xgb_hybrid",
   }
   ENSEMBLE_PICK = "ensemble_regime_weighted"  # spec 09's ensemble pick, a money-table row
   SPLIT = "rolling"               # spec 09's CANDIDATE_SPLIT
   MIN_BIN_SHARE = 0.5             # spec 09's forecast-share floor, used by the pick check
   PRICE_BANDS = [0.5, 0.9, 0.99]  # quantiles of the hourly |reBAP| over the common hours
   TOP_HOURS = 10                  # hours in the top-hours plot
   EXPORT_ENABLED = True
   ```

   - Copied from spec 09's notebook: `STYLE` (seven registry models, four ensembles, `actual`,
     `smard` dashed), `TAIL_COLOR` (bin regions only), `style_timeseries` and `style_plain`.
   - `PRICE_COLOR` is a sequential ramp for the price bands. It never repeats a `STYLE` or
     `TAIL_COLOR` colour.
3. **Loading.** Finds `data/` by walking up from the working directory. A missing file raises
   `FileNotFoundError` naming its producer; for `rebap.csv` that is `API-connection.ipynb` Part 2,
   which needs `.env`.
   - **Model, ensemble and SMARD files:** read as in spec 09. Ensemble rows are added only if their
     shared model rows equal `model_scoreboard.csv`; otherwise the notebook stops.
   - **`rebap.csv`:** a German CSV (`delimiter=";"`, `encoding="utf-8-sig"`), prices via
     `str.replace(",", ".")` → `float`, `timestamp` (`%Y-%m-%d %H:%M`, local interval start).
   - **Common hours:** the test hours that have an actual, `fc_residual_load`, and a forecast from
     every pick and money-table row.
   - Prints the window, the picks, the common hours and the reBAP span.
4. **Pick check** (spec 10 Behaviour 4, extended to four categories). Every `PICKS` value and
   `ENSEMBLE_PICK` needs a `SPLIT` row. Each pick must qualify in its category: skill vs SMARD > 0
   for `overall`, spec 09's bin rule for the bins (by actual, by forecast, forecast share
   ≥ `MIN_BIN_SHARE`). On a failure the notebook stops, naming the key and the failed rule. It never
   falls back or re-ranks.
5. **Self-check.** It stops on any failure, naming what to regenerate. On success it prints
   `self-check passed: one snapshot, every common hour priced (<h> hours, <q> quarter-hours)`.
   - **Snapshot:** `residual_load` agrees between the model export and the SMARD file on the common
     hours (tolerance `1e-6`).
   - **Scoreboard:** the rebuilt MAE and hour count of every pick and of SMARD equal
     `accuracy / MAE`.
   - **reBAP file:** `reBAP unterdeckt == reBAP ueberdeckt` on every row, and `Einheit` is
     `EUR/MWh`. The `Datentyp` counts in the window are printed.
   - **Coverage:** after the fold rule (6), every common hour has exactly 4 quarter-hours. A missing
     quarter stops the notebook ("re-run `API-connection.ipynb` Part 2"). Subsets are never priced.

### Price and cost

6. **Hourly price.**
   - **Fold rule:** on an autumn switch day, `rebap.csv` lists 02:00–02:45 twice, first CEST, then
     CET. SMARD's fetch keys its values by local time, so the CET hour overwrites the CEST one. Its
     single 02:00 row is therefore the CET hour (01:00–02:00 UTC). Keep the matching quarter-hours
     with `drop_duplicates("timestamp", keep="last")`. Spring needs no rule: neither file has a
     local 02:00.
   - Each quarter-hour's local start is floored to the hour, SMARD's naive key.
   - `rebap_abs_h = mean(|reBAP_q|)` over the hour is the price of every cost. `rebap_h =
     mean(reBAP_q)` goes into the export only.
   - **Why the mean of absolute values:** spread the hourly error `e_h` (MWh) evenly over the
     quarters. Then `Σ_q (|e_h| / 4) · |reBAP_q| = |e_h| · rebap_abs_h`. A one-quarter spike counts
     at a quarter of the hour's error, which is all an hourly forecast can claim. A sign flip inside
     the hour is not netted.
7. **Cost and saving**, with `e = forecast − actual` (spec 06):
   - `cost_h = |e_h| · rebap_abs_h` (EUR), for SMARD and every model row.
   - `saved_h = cost_SMARD,h − cost_pick,h = (|e_SMARD,h| − |e_pick,h|) · rebap_abs_h`. This is
     spec 09's hourly advantage (§4.4), priced; positive means the pick is cheaper.
8. **Aggregates over a category's hours:**
   - `saved` = Σ `saved_h`
   - `saved %` = `saved` / SMARD cost
   - `MWh skill %` = `100 · (1 − Σ|e_pick| / Σ|e_SMARD|)`
   - `top-band share` = share of `saved` from hours in the top price band

   Formats: money in M€ (`{:+,.1f}`), prices in EUR/MWh (`{:,.0f}`), MWh as in spec 09.

### Overview tables

Both tables come before the first plot. First, print:

- the bin edges (spec 09's `P1` / `P99` of the actual over the test hours, plus the zero edge)
- the price-band edges (the `PRICE_BANDS` quantiles of `rebap_abs_h` over the common hours; one set
  for all categories, so a tail shows how many of its hours are expensive)
- the hours and mean `rebap_abs_h` per bin

9. **Picks table**, one row per category:
   - category, pick, hours, mean `|reBAP|`
   - SMARD cost, pick cost, saved (M€), saved %
   - MWh skill %, top-band share

   A header line gives the window, the common hours and the proxy sentence.
10. **Money table.**
    - Rows: every `SPLIT` registry row (spec 09's candidates) plus `ENSEMBLE_PICK`.
    - Per category group: saved (M€), saved %, money rank, and MAE rank. The MAE rank uses spec 09's
      key: months, then MAE, for `overall`; `MAE_<bin>_by_actual` for the bins. It shows `–` where a
      row fails spec 09's bin rule.
    - Each group's pick is marked. The table never changes `PICKS`.

### Plots

Every plot shows SMARD and the category's pick in their `STYLE` colours. SMARD is dashed, or grey
where the pick loses. Each plot has a title, units, and a one-line caption saying **what to read**;
conclusions go only in Behaviour 17.

Styling follows spec 10's audience style:

- direct labels on reference lines
- `#F5F7FA` count tiles (big number over a small word)
- shared legends under the figure
- no model colour used for another meaning

Helpers: `price_bands` (all four categories), `price_scatter` and `top_hours` (the three tails).
Each is defined where it is first used.

11. **Overall, plot 1: money saved per month.**
    - One bar per full month (spec 09's rule), showing `saved` in M€.
    - Pick colour above 0, SMARD grey below; M€ label on each bar.
    - The title gives the monthly average and the total.
12. **Overall, plot 2: cumulative money saved.**
    - Cumulative `saved_h` in time order (M€), with month ticks as in spec 09 §4.2.
    - The `TOP_HOURS` hours with the largest `|saved_h|` are marked as dots. The three largest are
      labelled with date, hour and `rebap_abs_h`.
    - Shows whether the money comes steadily or in jumps.
13. **`price_bands(category)`: where the money comes from** (plot 3 of `overall`, plot 2 of each
    tail).
    - One bar per price band (by default bottom 50 %, 50–90 %, 90–99 %, top 1 %), showing that band's
      `saved`.
    - Each band is labelled with its EUR/MWh range.
    - Each bar has a tile with the band's hours and its share of hours, next to its share of `saved`.
    - A band with no hours gets `0 h` and no bar.
14. **Tails** (`below_zero`, `low_extreme`, `high_extreme`):
    - **Plot 1, `price_scatter(category)`:**
      - One dot per bin hour: x = `rebap_abs_h` (EUR/MWh, symlog), y = `|e_SMARD| − |e_pick|` (MWh),
        area ∝ `|saved_h|`.
      - Pick colour where the pick was closer, SMARD grey otherwise.
      - A zero line, and the top-band edge as a labelled dashed line.
      - Tiles `pick closer: <n> h, +<x> M€` and `SMARD closer: <n> h, −<y> M€`.
    - **Plot 2:** `price_bands(category)`.
    - **Plot 3, `top_hours(category)`:**
      - Horizontal bars for the `TOP_HOURS` bin hours with the largest `|saved_h|`, sorted by size.
        Pick colour if saved, SMARD grey if lost.
      - Each bar is labelled `Sat 05 Apr 13:00 · 1,234 EUR/MWh · SMARD −4,100 / pick −900 MWh`.
      - The title gives these hours' share of the category's `|saved_h|`.
    - **An empty bin** prints `no hour with the actual <rule> in the test window` and draws nothing.

### Export

15. **`data/models/model_rebap_cost_hourly.csv`.**
    - The frame is always built; the file is written only with `EXPORT_ENABLED` (default on). It is
      gitignored by the `data/models/*.csv` rule.
    - Plain CSV, long format: one row per common hour × row (SMARD plus every money-table row).

    | Column | Content |
    |---|---|
    | `timestamp` | naive local time, `%Y-%m-%d %H:%M:%S` |
    | `model`, `split_method` | SMARD as `smard`, `none` |
    | `residual_load`, `forecast`, `err_residual_load` | from the exports |
    | `rebap_mean`, `rebap_abs_mean` | Behaviour 6 |
    | `cost_eur`, `saved_eur` | Behaviour 7; `saved_eur` empty on SMARD rows |
    | `in_below_zero`, `in_low_extreme`, `in_high_extreme` | the hour's by-actual bin |

16. **Round-trip check** (export on). Read the file back with a bare `pd.read_csv` and check:
    - its rows are the common hours × rows
    - the per-category sums of `saved_eur` reproduce the picks table (tolerance 1 EUR)
    - the quarter-hour rebuild reproduces `cost_eur` (tolerance `1e-6` relative): broadcast
      `|e_h| / 4` onto the hour's quarters, multiply by `|reBAP_q|` and sum

### Findings

17. **Closing section** (`interpretation-style` skill), holding all interpretation:
    - **Opening blockquote:** the date; the tables above always show the current files.
    - **Per category, 1–3 bullets:**
      - what the pick saves (M€, %) against its MWh skill
      - whether the money comes steadily or in jumps
      - which price band carries it, and the decisive hours
    - **Closing bullets:**
      - where the money ranking departs from the MAE ranking
      - the top band's share in the tails against `overall`
      - that the euro figure for `low_extreme` is small because its hours are cheap
      - the proxy caveat
      - one test year, and a calm one for prices. Link `EDA-rebap-magc.ipynb` for the 2022
        contrast; it is a personal exploration, not a project fact.

## Data

| File | Used columns | Role |
|---|---|---|
| `data/rebap.csv` | `timestamp`, `Einheit`, `Datentyp`, `reBAP unterdeckt`, `reBAP ueberdeckt` | quarter-hourly price |
| `data/models/model_forecast_errors_hourly.csv` | `timestamp`, `model`, `split_method`, `residual_load`, `forecast`, `err_residual_load` | errors, test window, edges |
| `data/models/model_scoreboard.csv` | `model`, `split_method`, `table`, `metric`, `value`, `count` | pick check, MAE ranks, self-check |
| `data/models/ensemble_{forecast_errors_hourly,scoreboard}.csv` | as the two above | ensemble rows |
| `data/metrics/smard_forecast_errors_hourly.csv` | `timestamp`, `residual_load`, `fc_residual_load`, `err_residual_load` | SMARD's hourly values |

- **Not read:** `data/smard.csv`, the risk labels, the `-magc` exports, `lower` / `upper`, and
  `bis_utc` (the fold rule needs only the row order).
- **Nothing hardcoded** (no date, year, edge, band edge or count) except the keys in `PICKS` and
  `ENSEMBLE_PICK`.
- **For review only:** a scratch run on the files of 2026-10-02. reBAP covers 2019-01-01 ..
  2026-09-06; the test window is 2025-09-07 .. 2026-09-06; all 8,759 common hours are priced.

  | Category | Hours | Mean \|reBAP\| | SMARD cost | Pick | Saved | MWh skill |
  |---|---|---|---|---|---|---|
  | overall | 8,759 | 113 | 2,667 M€ | `ensemble_regime_weighted` (10/11 months like `lgbm_direct`, MAE 2,277 < 2,363) | +375 M€ (+14.1 %) | +18.4 % |
  | below_zero | 563 | 46 | 172 M€ | `ensemble_regime_weighted` (MAE 2,853 < `xgb_hybrid`'s 2,979) | +75 M€ (+43.8 %) | +30.9 % |
  | low_extreme | 88 | 23 | 5.5 M€ | `random_forest_hybrid` | +1.1 M€ (+20.1 %) | +24.8 % |
  | high_extreme | 88 | 234 | 67 M€ | `xgb_hybrid` | +27 M€ (+40.3 %) | +33.2 % |

  - The top 1 % of price hours carry 11 % of SMARD's cost overall, 56 % below zero and 46 % in
    `high_extreme`.
  - Below zero, `xgb_direct` (MAE rank 4) saves the most, +113 M€. `linear_direct` saves only
    +12.5 M€ despite +21 % MWh skill.
  - These figures price the autumn hour at the mean of all 8 quarters, not with the fold rule.
    That changes one hour.

## Edge cases

- **Different snapshots** (a re-fetch changed the actuals): the self-check stops. Re-run
  `forecast-metrics-claude`, then `regression-models-claude` and `ensemble-claude` (each with
  `EXPORT_ENABLED`).
- **Missing quarter-hours, or reBAP ending early:** the coverage check stops (Behaviour 5).
  Re-run `API-connection.ipynb` Part 2. reBAP beyond the window is ignored.
- **DST:**
  - The fold rule (6) handles autumn; spring needs nothing.
  - If a future fetch changes SMARD's fold handling, the snapshot check does not catch it. The
    rule rests on `fetch()` in `API-connection.ipynb`, so recheck that function after edits.
  - Plots never subtract timestamps.
- **Price exactly 0:** no cost. Sits at 0 on the symlog axis.
- **`Datentyp` other than quality-assured** (a fresh fetch can end on preliminary values): counted
  and used. The findings mention it.
- **Negative `saved`:** not an error; drawn in SMARD grey.
- **Empty bin, or `P1 ≥ 0`:** as in spec 09.
- **Small counts:** the extreme bins hold about 88 hours, and the top band about 1 % of hours.
  Every money figure carries its hour count; no significance test.

## Acceptance criteria

- [ ] Runs top to bottom on a fresh kernel in well under a minute, reading only the files under
      Data. It fits or loads no model and edits no other notebook, spec output or input file.
- [ ] The pick check (4) and the self-check (5) pass on consistent files, and stop on a mismatch or
      a missing quarter-hour.
- [ ] Every cost is `|e_h| · mean|reBAP_q|` after the fold rule. No signed price enters a cost or a
      plot.
- [ ] The picks and money tables come before the first plot. The money table has every `SPLIT`
      candidate plus `ENSEMBLE_PICK`, with money and MAE ranks.
- [ ] `overall` has plots 11–13. Each tail has `price_scatter`, `price_bands` and `top_hours`. Every
      plot has a title, units and a caption.
- [ ] Models use registry colours, SMARD is dashed or grey, and the price bands repeat no model
      colour.
- [ ] Export on: the file follows Behaviour 15 and passes 16. Export off: nothing is written.
- [ ] Changing `PICKS` (to a qualifying row), `PRICE_BANDS`, `TOP_HOURS` or `EXPORT_ENABLED` needs
      no other edit.
- [ ] Nothing is hardcoded (Data), and all interpretation is in Behaviour 17.

## Out of bounds

- Editing `data/rebap.csv`, `data/smard.csv`, any existing export, `API-connection.ipynb`, the
  regression, ensemble or visualization notebooks, or their specs.
- Fitting, tuning or loading models; 15-min forecasting; changing the scoring, bins or common-hours
  rule.
- reBAP as a feature, or in any other notebook.
- Fetching a day-ahead price, or adding a signed or spread cost.
- Streamlit changes, including `streamlit/viz_helpers.py`.
- Long setup text, or method text repeated from the linked notebooks.
