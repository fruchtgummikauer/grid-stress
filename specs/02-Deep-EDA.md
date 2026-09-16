# 02 — Deep EDA of `data/smard.csv`

Status: draft
Branch: `feature/*` off `main`
Deliverable: `notebooks/EDA-deep.ipynb`
Depends on: [01-Simple-EDA.md](01-Simple-EDA.md), which must be complete and merged first.

## Goal

Spec 01 establishes what the data contains. This spec asks the four questions that only become
answerable once that groundwork exists, and that bear directly on the project's purpose —
spotting days at risk of TSO intervention.

1. How much of the multi-year change in the renewable series is **capacity growth** rather than
   behaviour, and does that make the early years unusable as training data?
2. How has the shape of the day changed — does the **duck curve** deepen, and does its belly
   reach below zero?
3. How large are the hour-to-hour **ramps** in residual load, when do they happen, and are they
   getting more frequent? Ramp magnitude is a plausible second stress mode alongside extreme
   level.
4. How good is the public **SMARD day-ahead forecast**, and specifically — does it degrade in
   exactly the tails we care about? That number is the bar our own model has to beat.

Success means the modeling spec can be written against concrete evidence: a decision on capacity
normalisation and start date, a characterised second stress mode, and a benchmark in MW.

## Inherited from spec 01

This spec does not re-derive the conventions. It restates each in one line and moves on. The
notebook repeats spec 01's loading and derivation block (~12 cells) rather than importing from it
— repeating a short, tested block is cheaper than one oversized shared notebook.

| Inherited | Established in 01 |
|---|---|
| Loading, renaming, dtype asserts | Behaviour 1–3 |
| `spans_gap` mask, 5 spring gaps at local 03:00 | Behaviour 5 |
| The residual load identity holds to rounding | Behaviour 9 |
| ISO weeks, Monday start; label side stated | Behaviour 11–12 |
| `period_mean`, the per-day helper, the edge rule | Behaviour 14–17 |
| Reporting units: average MW for levels, MWh/day for energy; `ylabel` required | Behaviour 18 |
| Meteorological seasons, `season_year = year + (month == 12)` | Behaviour 19 |
| The descriptive-slice policy | Behaviour 20 |
| `SERIES` constant and the derived columns | Behaviour 21 |
| `seasonal_plot` with the `ylabel` fix | Behaviour 41 |

Two of these carry real weight here. **Behaviour 9** means this spec does not have to treat
`fc_gen_wind_solar`'s counterpart as an open assumption — spec 01 proved it is exactly the summed
actual generation. **Behaviour 20** is what licenses the tail slices and largest-N selections
below without reopening the no-threshold rule.

## Scope

### IN

- One notebook, `notebooks/EDA-deep.ipynb`, runnable top to bottom from a fresh kernel against
  `data/smard.csv`.
- A compact restatement of spec 01's loading, derived columns and conventions.
- Capacity drift: annual mean and annual maximum wind and solar generation.
- Duck curve evolution: mean hour-of-day residual load profile per year, per season.
- Ramp rates: definition, DST masking, distribution, timing, episodes, and forecast ramp skill.
- Forecast benchmark: error metrics for all three `fc_*` columns against their actuals, on levels
  and on ramps, sliced by time, hour and load regime.
- Written findings: every plot is followed by a short prose statement of what it shows.
- A closing summary carrying the four decisions this spec exists to inform.

### OUT

- Everything spec 01 covers: the data quality audit, the aggregation correctness test, univariate
  description, basic seasonality, the correlation matrix, and the residual load level
  distribution. Restate a number from 01 where a section needs it; do not re-derive it.
- Defining the risk flag, or any threshold — on residual load level **or** on ramp magnitude.
  Deferred to the modeling spec.
- Producing a labelled or flagged dataset artifact, or a ranked risk-day list.
- Implementing capacity normalisation. This spec produces the evidence for that decision; the
  decision itself is the modeling spec's.
- Changing the dataset start date. The recommendation may be made here; acting on it means
  editing `notebooks/API-connection.ipynb`, which is out of bounds.
- Building a forecast of our own. Measuring SMARD's published forecast is in scope; competing
  with it is not.
- Modifying `notebooks/EDA-robert.ipynb` or `notebooks/EDA-simple.ipynb`.
- Any code under `modeling/`.

## Behaviour

### Setup

1. Repeat spec 01's loading and preparation: read the German-format CSV, convert the numeric
   columns, rename to snake_case, build the sorted `ts` frame, assert dtypes.
2. Rebuild the derived columns spec 01 defines — `renewables`, the calendar columns, `season`,
   `season_year`, `spans_gap` — and the `SERIES` constant. Copy `period_mean`,
   `style_timeseries`, `seasonal_plot` (with the `ylabel` fix) and the per-day helper.
3. Restate the inherited conventions in one markdown cell: the week convention, the season
   mapping and December rule, the reporting units, and the descriptive-slice policy. One line
   each, pointing at spec 01 for the reasoning. Do not re-derive or re-argue them.
4. Define the ramp series and the forecast ramp series here, before any section uses them:
   `ramp = residual_load.diff().mask(spans_gap)` and `fc_ramp = fc_res.diff().mask(spans_gap)`,
   plus the three error series `err_load`, `err_res`, `err_gen` as **forecast − actual** (so a
   positive value means over-forecast — state that sign convention once, here).

### Capacity drift

5. Plot annual **mean** and annual **maximum** generation for `wind_on`, `wind_off` and `solar`
   as three separate series — they have very different capacity trajectories and combining them
   hides that. Mean and maximum go in separate panels, or with a clearly separated second axis;
   they are different quantities and must not share one unlabelled scale.
6. Label 2026 as partial (2026-01-01 to 2026-09-09) in every one of these plots, and state the
   bias direction **per series**, because it is not the same for all three: 2026 is missing the
   Oct–Dec quarter, so the wind mean and wind maximum are both biased low, while solar's peak
   season is fully contained and its maximum is essentially unaffected.
7. Do not assert that Oct–Dec is the windiest quarter — **compute it**. A monthly wind
   climatology over the complete years 2022–2025 establishes the claim in one plot, and this is a
   section about evidence rather than assumption.
8. Interpret the annual maximum as a *proxy* for installed capacity and say plainly why it is
   only a proxy: peak observed infeed depends on weather and is reduced by curtailment, so it is
   a lower bound on capacity, not a measurement of it.
9. Close the section with the two decisions this evidence feeds, stated as findings rather than
   as implemented choices: whether the renewable series need capacity normalisation before the
   years are comparable, and whether the 2022 start date is still appropriate or whether the
   early period is too different in capacity terms to be useful training data.

### Duck curve evolution

10. Plot the mean hour-of-day `residual_load` profile with **one line per year**, as four panels
    — one per season, year as hue — so the year-on-year change is visible separately in each
    season instead of being averaged across them. This complements spec 01's daily profile, which
    covers `grid_load`.
11. Mark which year-lines rest on partial data: 2026 has no Oct–Nov, so its autumn line is
    incomplete, and under the inherited December rule the 2022 winter line has no preceding
    December. Restate the December rule in one line; it is spec 01's to define.
12. Describe what the panels show: whether the midday belly deepens year on year, whether it
    reaches below zero and in which seasons, and whether the evening rise steepens. Quantify the
    evening rise from the mean profile itself — the slope between roughly 16:00 and 20:00, in
    MW/h — so it is a number rather than an impression. This is the hand-off to the ramp section,
    which looks at the same phenomenon in the raw hourly data.

### Ramp rates

13. Report the ramp definition and its units: `residual_load.diff()` is the hour-on-hour change
    in residual load. Because the underlying values are hourly MWh, the difference is a change in
    average MW over one hour — label it MW/h and say so once.
14. Report the DST masking and its measured effect, rather than assuming it. Masking is correct
    on method: at the spring switch the neighbouring rows are two hours apart, so their difference
    is a two-hour change masquerading as a one-hour ramp. But **print the five masked values and
    compare them against the ramp percentiles** — in this record they are small relative to the
    distribution, nowhere near the extreme tail, and "the artefact is real but smaller than
    feared" is a more honest finding than repeating an unchecked assumption. Report the masked
    count (expected: 5, plus the leading `NaN` from `.diff()`).
15. Describe the distribution: histogram, percentiles including the 0.1/1/99/99.9 levels, and the
    largest absolute up-ramps and down-ramps with their timestamps. Treat up and down ramps
    separately — they are different operational problems and there is no reason to assume the
    distribution is symmetric.
16. Show when large ramps occur: the hour-of-day, month, season and year signature of the largest
    ramps, separately for up and down, selected by rank under the inherited slice policy. State
    whether the frequency of large ramps is increasing over the record, and relate it back to the
    evening-rise number from Behaviour 12.
17. Plot two or three of the largest ramp episodes at hourly resolution with the concurrent wind,
    solar and grid load, so the reader sees whether the driver was a solar sunset, a wind lull, a
    load step, or several at once. Select them by a stated rule — the days containing the top-3
    `|ramp|` values, each plotted with ±36 h of context — not by eye.
18. Compare the ramp series against `fc_ramp`, masked the same way: overlay their distributions,
    scatter forecast ramp against actual ramp, and report how often the forecast under-states a
    large ramp and by how much. This is the question of whether the published forecast sees the
    stress mode coming at all.
19. Note that ramp magnitude is a plausible **second** stress definition alongside extreme
    residual load level, and record it as a recommendation for the modeling spec. Do **not**
    threshold it, rank days by it as a risk list, or emit a flag — the same rule that governs the
    level tails in spec 01.

### Forecast benchmark

20. Establish the three comparison pairs: `fc_grid_load` vs. `grid_load`, `fc_res` vs.
    `residual_load`, and `fc_gen_wind_solar` vs. `renewables`. The third pair has no single actual
    counterpart column in the file — cite spec 01's identity check, which established that the
    summed generation *is* the counterpart, rather than restating it as an open assumption.
21. Report overall error for each pair: MAE, RMSE and mean bias (signed, per the convention fixed
    in Behaviour 4, so systematic over- or under-forecasting is visible), together with the count
    of hours compared.
22. Report a normalised error so the three pairs are comparable: MAE relative to the mean level of
    the corresponding actual. Do **not** use MAPE on `residual_load` or `fc_res` — the series
    crosses zero, and percentage error explodes there. If MAPE appears at all it is for
    `grid_load` only, which stays well away from zero, and the reason is stated.
23. Slice the error by time, each as its own view: per month over the record, by hour of day, by
    season, and by year. State whether the forecast is getting better or worse over the record and
    where its weak hours are.
24. Slice the error by **regime** — error against the actual level, in bins across the
    `residual_load` range. This is the slice that matters most for the project: if the forecast is
    accurate on ordinary hours but degrades in the tails, then the extreme cases we care about are
    exactly the ones the public benchmark handles worst, and that is the gap our model is trying
    to fill. Report it either way, whichever way it comes out. The bins are a descriptive device
    under the inherited slice policy — document the edges, create no column.
25. State the benchmark plainly at the end of the section: the error numbers our own model has to
    beat, in the same units and over the same period, so the modeling spec can be written against
    a concrete target rather than a vague ambition.

### Presentation

26. Use matplotlib directly for the styled time series plots and seaborn for the seasonal/hue
    plots, matching the existing notebooks.
27. Every plot carries a title, axis labels and explicit units — MWh, average MW, MWh/day or MW/h
    as applicable, never an unlabelled number. Every plot is followed by one to three sentences
    saying what it shows.
28. Close the notebook with a "Findings" section carrying the four decisions this spec exists to
    inform: the capacity normalisation and start-date recommendation, the duck curve conclusion,
    the ramp characterisation and its status as a candidate second stress mode, and the SMARD
    benchmark numbers. Add a self-check cell asserting the invariants — row count, index bounds,
    `spans_gap.sum() == 5`, `ramp.isna().sum() == 6`, and that `ts` carries no columns beyond
    `SERIES` plus the declared derived ones.

## Data

Source file, format and column meanings are as documented in
[01-Simple-EDA.md](01-Simple-EDA.md) — `data/smard.csv`, `sep=";"`, `decimal=","`, `utf-8-sig`,
41 107 hourly rows from `2022-01-01 00:00` to `2026-09-09 23:00`, local `Europe/Berlin`
wall-clock time, values in MWh per hourly interval.

Series this spec derives on top of that:

| Derived        | Definition                                  | Unit |
| -------------- | ------------------------------------------- | ---- |
| `renewables`   | `wind_on + wind_off + solar`                 | MWh  |
| `ramp`         | `residual_load.diff()`, DST steps masked     | MW/h |
| `fc_ramp`      | `fc_res.diff()`, DST steps masked            | MW/h |
| `err_load`     | `fc_grid_load − grid_load`                   | MWh  |
| `err_res`      | `fc_res − residual_load`                     | MWh  |
| `err_gen`      | `fc_gen_wind_solar − renewables`             | MWh  |

All error series follow **forecast − actual**: positive means the forecast was too high.

Characteristics this spec relies on, established in spec 01:

- Five missing hours, one per spring DST switch, all at local 03:00. The autumn fold is already
  collapsed by SMARD.
- The residual load identity holds to within decimal rounding, for both actuals and forecasts.
- `residual_load` and `fc_res` take negative values, and the negative share grows steeply across
  the record.
- Renewable capacity grows over the record, so the years are not drawn from one stationary
  distribution — which is what the capacity drift section measures.

## Edge cases

- **DST gap in the ramp series.** This is where the gap does real damage rather than adding
  noise: the 01:00 → 03:00 difference is a two-hour change presented as a one-hour ramp. It must
  be masked and the count reported. Do **not** assert in prose that these values are extreme —
  measure them and report what they actually are.
- **The leading `NaN`.** `.diff()` produces one, so the masked ramp series has 5 + 1 missing
  values. Any count assertion must expect 6, not 5.
- **Partial year in annual maxima.** An annual maximum is far more sensitive to a missing quarter
  than an annual mean, and the bias is series-specific — 2026 misses the windiest quarter, so
  wind maxima are biased low while solar maxima are essentially unaffected. State the direction
  per series rather than applying one blanket caveat.
- **Annual maximum is not installed capacity.** It is a weather- and curtailment-dependent lower
  bound. Any capacity-normalisation argument built on it must say so.
- **Season definition across the year boundary.** Meteorological winter spans two calendar years.
  The December rule is spec 01's to define; restate it, do not redefine it, and note that 2022
  and 2026 have incomplete winters.
- **Negative residual load is valid data.** It must not be clipped or excluded from aggregates.
  Log scales and log transforms are unusable for this series and for the ramp series, which is
  symmetric about zero.
- **Percentage error on a zero-crossing series.** MAE-relative-to-mean is fine; MAPE on
  `residual_load` or `fc_res` is not, because hours near zero produce unbounded percentage errors
  that dominate the average and make the benchmark meaningless.
- **Forecast alignment.** The `fc_*` columns are day-ahead forecasts already aligned to the hour
  they describe, not to the hour they were issued. The benchmark compares them row-wise on that
  basis, which spec 01's identity check corroborates.
- **Error sign.** Mean bias is meaningless without a stated sign convention, and MAE hides it
  entirely. Fix the convention once (Behaviour 4) and use it everywhere.
- **Regime bins are not thresholds.** Behaviour 24 bins the actual range to expose where the
  forecast fails. Under the inherited slice policy that is description; it becomes a threshold the
  moment a bin is named a risk case or written to a column.
- **Ramps are a candidate, not a definition.** Behaviour 19 records ramp magnitude as a possible
  second stress mode. Selecting the largest-N ramps to plot them is description; assigning a
  cut-off is the modeling spec's job.
- **Missing data file.** A fresh clone has no `data/smard.csv`. The first cell states where the
  file comes from and fails with a clear message if it is absent.

## Acceptance criteria

### Setup

- [ ] The notebook is `notebooks/EDA-deep.ipynb` and runs top to bottom from a fresh kernel with
      no manual intervention.
- [ ] Loading, derived columns, `SERIES` and the copied helpers reproduce spec 01's definitions.
- [ ] The inherited conventions are restated in one markdown cell, each pointing at spec 01, with
      no re-derivation.
- [ ] `ramp`, `fc_ramp` and the three `err_*` series are defined once, before first use, with the
      forecast − actual sign convention stated.
- [ ] `notebooks/EDA-robert.ipynb` and `notebooks/EDA-simple.ipynb` are unchanged.

### Capacity drift

- [ ] Annual mean and annual maximum are plotted for `wind_on`, `wind_off` and `solar` as three
      separate series, with mean and maximum not sharing an unlabelled scale.
- [ ] 2026 is labelled partial in every capacity plot, with the bias direction stated per series
      rather than as one blanket caveat.
- [ ] The "Oct–Dec is the windiest quarter" claim is computed from a monthly wind climatology over
      2022–2025, not asserted.
- [ ] The annual maximum is described as a lower-bound proxy for installed capacity, with weather
      and curtailment named as the reasons.
- [ ] The section closes with findings on capacity normalisation and on the 2022 start date.

### Duck curve

- [ ] Mean hour-of-day `residual_load` is plotted as four season panels with one line per year.
- [ ] The December rule is restated and partial year-lines are marked.
- [ ] The panels are interpreted: belly depth over the years, whether and when it goes negative,
      and the evening rise quantified in MW/h from the mean profile.

### Ramp rates

- [ ] The ramp series is defined as `residual_load.diff()`, masked on `spans_gap`, and labelled
      MW/h.
- [ ] The five masked values are printed and compared against the ramp percentiles, with the
      measured result reported rather than an assumed one.
- [ ] The masked count is reported and accounts for the leading `NaN` (6 total missing).
- [ ] The ramp distribution is reported with percentiles including 0.1/1/99/99.9, up and down
      treated separately.
- [ ] The largest up-ramps and down-ramps are listed with timestamps.
- [ ] The hour-of-day, month, season and year signature of large ramps is shown, with a statement
      on whether large ramps are becoming more frequent, related back to the duck curve number.
- [ ] At least two large ramp episodes are plotted hourly with wind, solar and grid load, selected
      by the stated top-3 `|ramp|` rule with ±36 h context, and the driver identified.
- [ ] `fc_ramp` is compared against the actual ramp series — distribution overlay, scatter, and
      how often the forecast under-states a large ramp.
- [ ] Ramps are recorded as a candidate second stress definition, with no threshold applied.

### Forecast benchmark

- [ ] The three comparison pairs are defined, with the `fc_gen_wind_solar` counterpart resolved by
      citing spec 01's identity check rather than restated as an assumption.
- [ ] MAE, RMSE, mean bias and the compared-hour count are reported for each pair.
- [ ] A normalised error (MAE relative to the mean actual level) is reported so the pairs are
      comparable.
- [ ] MAPE is not applied to `residual_load` or `fc_res`; if used at all it is on `grid_load` only,
      with the zero-crossing reason stated.
- [ ] Error is sliced per month over the record, by hour of day, by season and by year, with a
      statement on whether the forecast is improving.
- [ ] Error is sliced by `residual_load` regime across bins of the actual range, with an explicit
      statement of whether the forecast degrades in the tails, and the bin edges documented.
- [ ] The section ends with the concrete benchmark numbers our model has to beat.

### Discipline and presentation

- [ ] No threshold, cut-off, risk flag or labelled column appears anywhere — neither on residual
      load level nor on ramp magnitude.
- [ ] No rank-based slice or regime bin is written to `ts`; all stay local to their cell.
- [ ] No model of our own is fitted, and no train/test split is made.
- [ ] No capacity normalisation is implemented, and the fetch range is unchanged.
- [ ] Every plot has a title, axis labels and explicit units (MWh, MW, MWh/day or MW/h), and is
      followed by a written interpretation.
- [ ] A closing "Findings" section carries all four decisions this spec informs.
- [ ] A self-check cell asserts row count, index bounds, `spans_gap.sum() == 5`,
      `ramp.isna().sum() == 6`, and that `ts` carries no unexpected columns.
- [ ] The notebook is committed with cleared or reproducible outputs, on a `feature/*` branch, and
      `nbdime` is enabled for the clone.

## Out of bounds

The following are explicitly not part of this spec:

- Threshold or flag definition for extreme residual load or ramp magnitude, including "just a
  first guess" quantile cut-offs.
- Any labelled dataset, flag column, ranked risk-day list, or exported artifact derived from the
  extremes or the ramps.
- Re-doing spec 01's work: the data quality audit, the `period_mean` correctness test, univariate
  description, basic seasonality, or the correlation matrix.
- Fitting any model of our own, including trivial baselines, `statsmodels` decompositions used as
  forecasts, or scikit-learn estimators.
- Implementing capacity normalisation, or rebasing the series onto installed capacity.
- Changing the dataset start date, which would mean editing `notebooks/API-connection.ipynb`.
- Feature matrix construction, train/validation/test splitting, or cross-validation design.
- MLflow runs, experiment logging, or changes to `modeling/config.py`.
- Changes to `notebooks/API-connection.ipynb`, the `FILTERS` dict, or the fetch range.
- Changes to `notebooks/EDA-robert.ipynb` or `notebooks/EDA-simple.ipynb`.
- Adding data sources beyond `data/smard.csv` — no weather, price, redispatch, curtailment or
  cross-border data, even where the capacity and ramp sections would benefit. Installed-capacity
  figures would make Behaviour 8 unnecessary; getting them is a separate spec.
- Adding dependencies. If this spec appears to need a package that is not already in
  `pyproject.toml`, raise it rather than running `uv add`.
- Editing the other team members' EDA notebooks.
