# 01 — Exploratory Data Analysis of `data/smard.csv`

Status: draft
Branch: `feature/basic-eda-robert`
Deliverable: a notebook in `notebooks/`, named `EDA-<member>.ipynb` per the team's
per-member naming convention (for this branch: `notebooks/EDA-robert.ipynb`).

## Goal

Understand the SMARD dataset well enough to make informed modeling decisions for the
1-day-ahead residual load forecast, and to describe — descriptively, without yet defining
thresholds — where the extreme residual load cases that motivate the project actually sit.

The EDA must answer:

1. Is the dataset complete, correctly typed, and continuous enough to be treated as an
   hourly time series?
2. Are our aggregation helpers correct, and what exactly do our calendar conventions mean?
3. What are the trend, seasonal and calendar structures in each series, and how much of the
   multi-year change is renewable capacity growth rather than behaviour?
4. How do the series relate to each other, and which of those relations are candidate
   features for the forecast?
5. What does the `residual_load` distribution look like, how do its high and negative tails
   behave in time, and how large are its hour-to-hour ramps?
6. How good is the public SMARD day-ahead forecast? That is the bar our own model has to
   beat, so its error has to be measured here.

Success means: a teammate who has never opened the data can read the notebook top to bottom
and know what the data contains, what is broken about it, which features are worth
engineering, and how accurate the benchmark already is — without re-running any exploration
themselves.

## Scope

### IN

- One notebook in `notebooks/`, runnable top to bottom from a fresh kernel against
  `data/smard.csv`.
- Loading and type conversion of the German Excel CSV format.
- Data quality audit: coverage, gaps, duplicates, missing values, implausible values, and
  verification of the residual load identity.
- Explicit calendar conventions: ISO weeks (Monday start) and the season definition.
- A one-off correctness test of `period_mean` against a plain `.resample()`.
- A per-day normalisation helper and a comparison of mean vs. normalised aggregates.
- Univariate description of all nine series, including the three SMARD forecast columns.
- Time structure: trend, annual/weekly/daily seasonality, calendar effects, autocorrelation.
- Capacity drift: annual mean and annual maximum wind and solar generation.
- Duck curve evolution: mean hour-of-day residual load profile per year, per season.
- Multivariate structure: correlations, lagged relations, renewables vs. residual load.
- Distribution and tail behaviour of `residual_load`, described but not thresholded.
- Ramp rates: the distribution, timing and forecast behaviour of `residual_load.diff()`.
- Forecast benchmark: error metrics for all three `fc_*` columns against their actuals, on
  levels and on ramps, sliced by time, hour and load regime.
- Written findings: every plot is followed by a short prose statement of what it shows.
- A closing summary listing the modeling-relevant conclusions.

### OUT

- Defining the risk flag or any threshold, on residual load level or on ramp magnitude.
  Deferred to the modeling spec.
- Producing a labelled or flagged dataset artifact.
- Any model fitting, feature-matrix construction, or train/test splitting. The forecast
  benchmark measures SMARD's published forecast; it does not build one of our own.
- Imputing, resampling away, or otherwise repairing the DST gaps.
- Capacity normalisation itself — the capacity drift section produces the evidence for that
  decision, it does not implement it.
- Changing `notebooks/API-connection.ipynb` or the fetch logic.
- Extracting helpers into a Python module. Helpers stay in the notebook for now.
- Any code under `modeling/`.

## Behaviour

### Loading and preparation

1. Read `data/smard.csv` with `pd.read_csv(..., delimiter=";")`. Convert every numeric
   column from the German decimal format (`str.replace(",", ".")` → `float`), and parse
   `timestamp` to datetime.
2. Rename the columns to the project's snake_case names: `wind_off`, `wind_on`, `solar`,
   `grid_load`, `residual_load`, `fc_gen_wind_solar`, `fc_grid_load`, `fc_res`. Set
   `timestamp` as a sorted `DatetimeIndex`.
3. Print the frame's shape, dtypes, index start/end, and `describe()` so the reader sees the
   raw shape of the data before any plot.

### Data quality audit

4. Report coverage: first and last timestamp, total row count, and the expected row count
   for a complete hourly index over that span. State the difference explicitly.
5. Detect and list every gap in the hourly index — the timestamps that a complete hourly
   index would contain but the data does not. Group them and name the spring DST switch as
   the cause where that is what they are.
6. Check for duplicate timestamps and report the count, expected to be zero: SMARD already
   collapses the autumn fold rather than emitting the repeated hour twice.
7. Report missing values per column (count and share). If there are none, say so — an
   explicit "zero missing" is a finding.
8. Sanity-check value ranges: generation series (`wind_off`, `wind_on`, `solar`) must be
   non-negative; `solar` must be at or near zero at night; `grid_load` must be strictly
   positive. Report any violation with its timestamps rather than silently dropping it.
9. Verify the residual load identity empirically:
   `residual_load` vs. `grid_load − (wind_on + wind_off + solar)`, and `fc_res` vs.
   `fc_grid_load − fc_gen_wind_solar`. Report the distribution of the difference, not just
   whether it is zero. A non-zero remainder means SMARD's residual load subtracts more than
   these three series, which changes what our target actually is — say which it turns out
   to be.
10. Do **not** fill, interpolate or reindex away the gaps. Document them and carry on. The
    handling decision belongs to the modeling spec.

### Aggregation conventions and correctness

11. Fix the week convention explicitly: **ISO weeks, Monday start, Sunday end**. In pandas,
    `"W"` is an alias for `"W-SUN"`, which bins by week *ending* Sunday and is therefore
    already ISO-aligned — but the notebook must assert this rather than trust it, by
    checking that the first timestamp falling into each weekly bin is a Monday.
12. State which edge of the week the bin label refers to. `resample("W")` labels each bin
    with its right edge (the Sunday), so a weekly plot's x value is the week *end*. Either
    keep that and label the axis accordingly, or shift to the Monday and say so — but the
    choice is written down, not left implicit.
13. Where weeks are used as a grouping key rather than a resample frequency, use
    `.dt.isocalendar()` and take `year` and `week` from it together. Never pair
    `isocalendar().week` with `.dt.year`: around New Year the ISO year and the calendar year
    disagree, and a year can have 53 ISO weeks.
14. Run a one-off correctness test of `period_mean` against a plain `.resample().mean()` on
    the same series and frequency, for both `"W"` and `"M"`:
    - Assert that on the periods both produce, the values are equal within floating-point
      tolerance (`pd.testing.assert_series_equal` with a tolerance, or `np.allclose`).
    - Assert that `period_mean`'s index is exactly the `.resample()` index minus its first
      and last entry.
    - Print the two dropped edge periods with their `.resample()` values and the neighbouring
      complete period, so the size of the artefact being avoided is visible in MW, not just
      asserted away.
15. Follow the test with a markdown cell explaining it in prose: that `period_mean` is a
    plain calendar-period mean plus one rule — drop the first and last period — that the
    test confirms the arithmetic is identical everywhere else, and that the printed edge
    values show why the rule exists (the data starts mid-week and ends mid-month, so those
    two periods average fewer hours and read as dips that are not in the data). This test is
    run once as a correctness check; it is not repeated for every later aggregation.
16. Add a separate per-day normalisation helper alongside `period_mean`, so month length
    stops contaminating comparisons. It aggregates a period's **sum** and divides by the
    period's day count, and it returns both units:
    - **MWh per day** — the energy view, for later plots about energy volume.
    - **average MW** — the power view, computed as the period sum divided by the number of
      hours actually present in that period, not as `MWh_per_day / 24`.
    It drops incomplete first and last periods on the same rule as `period_mean`.
17. Compare the aggregation variants in one table for monthly aggregates: hourly mean,
    average MW, and MWh per day, side by side with the month length and the hour count.
    State the conclusion in prose, including the parts that turn out to be uninteresting:
    an hourly series' mean and its average MW are the same number except in months where
    hours are missing, so for *means* month length genuinely does cancel; MWh per day is
    that number rescaled by the days-per-month factor, and it is only when we aggregate
    **sums** that ignoring it distorts month-over-month comparison. Name which months in
    this dataset actually differ, and by how much.
18. Fix the reporting convention for the rest of the notebook: **average MW** for load and
    generation levels, **MWh per day** only where the quantity is genuinely energy volume.
    Every plot states which one it is showing.

### Univariate description

19. For each of the nine series, report mean, std, min, max and the 1/5/25/50/75/95/99
    percentiles in one comparison table.
20. Plot each series over the full period using the shared `style_timeseries` helper, at a
    resolution that stays readable (weekly or monthly `period_mean`, not raw hourly).
21. Plot the distribution of each series (histogram, and boxplot where the tails matter).

### Time structure

22. Aggregate with `period_mean(series, freq)` for weekly and monthly views so that the
    incomplete first and last calendar periods are dropped — the data starts mid-week and
    ends mid-month, and a plain `.resample()` produces fake edge dips.
23. Annual seasonality: use `seasonal_plot(df, y_value, title, ylabel)` (month on x, year as
    hue) for `grid_load`, `residual_load`, and the renewables. State whether the years
    differ in level, in shape, or both.
24. Weekly seasonality: mean profile by day of week, and the weekday/weekend contrast.
25. Daily seasonality: mean hour-of-day profile, split by season and by weekday/weekend, so
    the summer solar dent and the winter evening peak are both visible.
26. Calendar effects: characterise the Christmas/New Year drop and check public holidays
    against comparable weekdays.
27. Trend: state whether a multi-year level change is visible in `grid_load` and
    `residual_load` once seasonality is aggregated out, and be explicit that the final
    period is a partial year (data ends 2026-09-09) and must not be read as a full-year
    trend point.
28. Autocorrelation: ACF/PACF of `grid_load` and `residual_load` over lags covering at least
    one week (≥ 168 h), calling out the 24 h and 168 h peaks. Note in the notebook that the
    DST gaps make the lag axis approximate, since ACF treats rows as evenly spaced.

### Capacity drift

29. Plot annual **mean** and annual **maximum** generation for `wind_on`, `wind_off` and
    `solar` as three separate series — they have very different capacity trajectories and
    combining them hides that. Mean and maximum go in separate panels or with a clearly
    separated second axis; they are different quantities and must not share one unlabelled
    scale.
30. Label 2026 as partial (2026-01-01 to 2026-09-09) in every one of these plots, and state
    the bias direction, which is not the same for all three series: the missing Oct–Dec
    quarter is the windiest part of the year, so the 2026 wind mean and wind maximum are
    both biased low, while solar's peak season is fully contained and its maximum is not
    biased in the same way.
31. Interpret the annual maximum as a *proxy* for installed capacity and say plainly why it
    is only a proxy: peak observed infeed depends on weather and is reduced by curtailment,
    so it is a lower bound on capacity, not a measurement of it.
32. Close the section with the two decisions this evidence feeds, stated as findings rather
    than as implemented choices: whether the renewable series need capacity normalisation
    before the years are comparable, and whether the 2022 start date is still appropriate or
    whether the early period is too different in capacity terms to be useful training data.

### Duck curve evolution

33. Plot the mean hour-of-day `residual_load` profile with **one line per year**, as four
    panels — one per season, year as hue — so the year-on-year change in the profile is
    visible separately in each season instead of being averaged across them.
34. Define the season mapping explicitly in the notebook: meteorological seasons
    (winter = Dec/Jan/Feb, spring = Mar/Apr/May, summer = Jun/Jul/Aug, autumn = Sep/Oct/Nov).
    State how December is assigned — a December belongs to the winter of the *following*
    calendar year or of its own, and whichever rule is chosen must be written down, because
    it decides which year's line each December hour lands on.
35. Mark which year-lines rest on partial data: 2026 has no Oct–Nov, so its autumn line is
    incomplete, and the 2022 and 2026 winter lines are partial under either December rule.
36. Describe what the panels show: whether the midday belly deepens year on year, whether it
    reaches below zero and in which seasons, and whether the evening ramp gets steeper — the
    last point being the link to the ramp section.

### Multivariate structure

37. Correlation matrix over all nine series, plotted as a heatmap, with the near-definitional
    relation between `grid_load`, the renewables and `residual_load` called out rather than
    presented as a discovery.
38. Show how `residual_load` responds to renewable infeed: `residual_load` against
    `wind_on + wind_off + solar`, and separately against wind and solar, with the direction
    and strength stated.
39. Lagged relations: correlation of `residual_load` with its own lags at 1 h, 24 h, 48 h and
    168 h, as the empirical basis for candidate lag features.
40. Conclude with a short list of the features this suggests engineering (calendar features,
    lags, rolling means, renewable aggregates, capacity-normalised renewables), as a
    recommendation for the modeling spec — not as implemented code.

### Residual load and its extremes

41. Describe the `residual_load` distribution in detail: histogram, percentiles, skew,
    and how far the tails reach in both directions.
42. Quantify how much of the record sits below zero: count and share of negative hours, and
    how that share develops per year and per month.
43. Show *when* the extremes occur — the tail hours located on the calendar (year, month,
    hour of day, day of week) — so the seasonal and daily signature of both tails is visible.
44. Show a handful of representative episodes as raw hourly plots: at least one sustained
    negative-residual-load stretch and one high-residual-load stretch, with the concurrent
    wind, solar and grid load in the same figure.
45. State the tail behaviour in words and stop there. Do **not** propose, compute or apply a
    threshold, and do not emit a flag column.

### Ramp rates

46. Define the ramp series as `residual_load.diff()` — the hour-on-hour change in residual
    load. Because the underlying values are hourly MWh, the difference is a change in
    average MW over one hour; label it as MW/h and say so once.
47. Mask every difference that spans a DST gap. At the spring switch the neighbouring rows
    are two hours apart, so their difference is a two-hour ramp masquerading as a one-hour
    one and would land straight in the extreme tail. Set those to NaN, and report how many
    were masked so the exclusion is visible rather than silent.
48. Describe the distribution: histogram, percentiles including the 0.1/1/99/99.9 levels,
    and the largest absolute up-ramps and down-ramps with their timestamps. Treat up and
    down ramps separately — they are different operational problems and there is no reason
    to assume the distribution is symmetric.
49. Show when large ramps occur: the hour-of-day, month, season and year signature of the
    largest ramps, separately for up and down. State whether the frequency of large ramps is
    increasing over the record, and relate it to what the duck curve panels showed about the
    evening ramp.
50. Plot two or three of the largest ramp episodes at hourly resolution with the concurrent
    wind, solar and grid load, so the reader sees whether the driver was a solar sunset, a
    wind lull, a load step, or several at once.
51. Compare the ramp series against the forecast ramp series `fc_res.diff()`, masked the same
    way: overlay their distributions, scatter forecast ramp against actual ramp, and report
    how often the forecast under-states a large ramp and by how much. This is the question
    of whether the published forecast sees the stress mode coming at all.
52. Note that ramp magnitude is a plausible *second* stress definition alongside extreme
    residual load level, and record it as a recommendation. Do **not** threshold it, rank
    days by it, or emit a flag — the same rule as for the level tails.

### Forecast benchmark

53. Establish the three comparison pairs and state the one assumption they rest on:
    `fc_grid_load` vs. `grid_load`, `fc_res` vs. `residual_load`, and `fc_gen_wind_solar`
    vs. the summed actual `wind_on + wind_off + solar` — the last having no single actual
    counterpart column in the file, so the sum stands in for it. Say this explicitly.
54. Report overall error for each pair: MAE, RMSE and mean bias (signed, so systematic
    over- or under-forecasting is visible), together with the count of hours compared.
55. Report a normalised error too, so the three pairs are comparable: MAE relative to the
    mean level of the corresponding actual. Do **not** use MAPE on `residual_load` or
    `fc_res` — the series crosses zero, and percentage error explodes there. If MAPE is
    shown at all it is for `grid_load` only, which stays well away from zero.
56. Slice the error by time and condition, each as its own view: per month over the record,
    by hour of day, by season, and by year. State whether the forecast is getting better or
    worse over the record and where its weak hours are.
57. Slice the error by regime — error against the actual level, in bins across the
    `residual_load` range. This is the slice that matters most for the project: if the
    forecast is accurate on ordinary hours but degrades in the tails, then the extreme cases
    we care about are exactly the ones the public benchmark handles worst, and that is the
    gap our model is trying to fill. Report it either way, whichever way it comes out.
58. State the benchmark plainly at the end of the section: the error numbers our own model
    has to beat, in the same units and over the same period, so the modeling spec can be
    written against a concrete target rather than a vague ambition.

### Presentation

59. Reuse `period_mean`, `style_timeseries` and `seasonal_plot` rather than re-deriving
    equivalents. New helpers required by this spec — the per-day normalisation function and
    anything needed for the ramp and benchmark sections — are defined once near the top of
    the notebook alongside the existing ones.
60. Use matplotlib directly for the styled time series plots and seaborn for the
    seasonal/hue plots, matching the existing notebooks.
61. Every plot carries a title, axis labels and explicit units — MWh, average MW, MWh/day or
    MW/h as applicable, never an unlabelled number. Every plot is followed by one to three
    sentences saying what it shows.
62. Close the notebook with a "Findings" section: the data quality issues, the calendar and
    aggregation conventions decided here, the confirmed seasonal structure, the capacity
    drift conclusion, the residual load tail and ramp description, the SMARD benchmark
    numbers, and the recommended features.

## Data

Source file: `data/smard.csv`, produced by `notebooks/API-connection.ipynb` from the SMARD
API. `data/` is gitignored, so the file must be regenerated after a fresh clone.

Format: `sep=";"`, `decimal=","`, `utf-8-sig`, one header row.

Observed extent at the time of writing: `2022-01-01 00:00` to `2026-09-09 23:00`,
41 107 data rows, hourly, region DE, timestamps in local `Europe/Berlin` wall-clock time
(naive, no offset in the file).

| CSV column              | snake_case          | Meaning                                | Unit |
| ----------------------- | ------------------- | -------------------------------------- | ---- |
| `timestamp`             | `timestamp`         | Local hour start, Europe/Berlin         | —    |
| `Wind Offshore`         | `wind_off`          | Offshore wind generation                | MWh   |
| `Wind Onshore`          | `wind_on`           | Onshore wind generation                 | MWh   |
| `Solar`                 | `solar`             | Solar generation                        | MWh   |
| `Grid Load`             | `grid_load`         | Total grid load                         | MWh   |
| `Residual Load`         | `residual_load`     | Load minus wind and solar infeed        | MWh   |
| `Forecast Wind + Solar` | `fc_gen_wind_solar` | SMARD day-ahead wind + solar forecast   | MWh   |
| `Forecast Grid load`    | `fc_grid_load`      | SMARD day-ahead grid load forecast      | MWh   |
| `Forecast Residual Load`| `fc_res`            | SMARD day-ahead residual load forecast  | MWh   |

Values are energy per hourly interval (MWh). Over an hourly interval the numeric value is
the same as the average power in MW, which is why the hourly mean of a series and its
average MW are the same number — a point the aggregation section makes explicit rather than
leaving the reader to infer.

Derived series used in this spec, none of which are persisted:

| Derived                 | Definition                                | Unit |
| ----------------------- | ----------------------------------------- | ---- |
| renewable infeed        | `wind_on + wind_off + solar`               | MWh  |
| ramp                    | `residual_load.diff()`, DST steps masked   | MW/h |
| forecast ramp           | `fc_res.diff()`, DST steps masked          | MW/h |
| per-day energy          | period sum ÷ days in period                | MWh/day |
| average power           | period sum ÷ hours present in period       | MW   |

Known characteristics to expect and confirm:

- One missing local hour at each spring DST switch. The file skips the local **02:00** row —
  e.g. `2025-03-30` runs 01:00 → 03:00. The autumn switch is already handled by SMARD: the
  repeated hour is collapsed rather than duplicated, so the file contains one 02:00 row and
  that hour is likewise absent.
- Strong daily, weekly and annual cycles; winter peak; pronounced Christmas/New Year drop.
- `residual_load` and `fc_res` take negative values during renewable oversupply.
- Renewable capacity grows over the record, so the years are not drawn from one stationary
  distribution — this is what the capacity drift section measures.
- Missing values are assumed to be absent; the audit verifies rather than assumes this.

## Edge cases

- **Spring DST gap.** The hourly index is not complete. Any `.diff()`, `.shift()`,
  rolling window or ACF over the raw index silently treats the jump as a single step.
  Name this in the notebook wherever it applies; do not repair it.
- **DST gap in the ramp series specifically.** This is the case where the gap does real
  damage rather than just adding noise: the 01:00 → 03:00 difference is a two-hour change
  presented as a one-hour ramp, and it is large enough to land in the extreme tail that the
  whole ramp section is about. It must be masked, and the mask count reported.
- **Autumn DST fold.** The repeated 02:00 hour is not in the file. This is a second, less
  obvious source of hour loss and must be reported alongside the spring gap. The autumn
  switch is already handled by SMARD.de.
- **Week convention.** pandas `"W"` means `"W-SUN"` — weeks *ending* Sunday, i.e. ISO
  Monday-start weeks, which is what we want, but the alias reads as if it meant the
  opposite. `"W-MON"` would give weeks ending Monday, which is not ISO. Assert the bin start
  is a Monday rather than relying on the alias reading correctly.
- **Week label side.** A resampled weekly index is labelled with the week's right edge, so
  an unlabelled weekly x axis silently shows week *ends*. State which edge is plotted.
- **ISO year boundary.** `isocalendar().week` must be paired with `isocalendar().year`, never
  with `.dt.year`. Early January can be ISO week 52 or 53 of the previous ISO year, and
  mixing the two scatters those hours into a phantom week at the wrong end of the axis.
- **Partial first and last calendar periods.** Data starts mid-week and ends mid-month, so
  every weekly/monthly aggregation goes through `period_mean` or the per-day helper, never a
  bare `.resample()` — with the single deliberate exception of the correctness test, whose
  whole purpose is to call `.resample()` and compare.
- **Month length.** Month-over-month comparison of *sums* is distorted by day count; for
  *means* it cancels. The per-day helper exists so the distinction is explicit rather than
  assumed, and the comparison table is expected to show that the mean-based views were fine
  all along — a negative result that is still worth writing down.
- **Hours per period ≠ 24 × days.** A month containing the spring switch has one hour fewer
  in the data. Average MW must divide by the hours actually present, not by `24 × days`, or
  that month reads slightly low for no physical reason.
- **Partial final year.** 2026 stops on 09-09. Never compare a 2026 annual aggregate against
  a full year; restrict year-on-year comparisons to matching date ranges or label the
  partial year clearly.
- **Partial year in annual maxima.** An annual maximum is far more sensitive to a missing
  quarter than an annual mean, and the bias is series-specific: 2026 misses Oct–Dec, the
  windiest quarter, so wind maxima are biased low while solar maxima are essentially
  unaffected. State the direction per series rather than applying one blanket caveat.
- **Annual maximum is not installed capacity.** It is a weather- and curtailment-dependent
  lower bound. Any capacity-normalisation argument built on it must say so.
- **Season definition across the year boundary.** Meteorological winter spans two calendar
  years, so the December assignment rule decides which year's line December hours join in
  the duck curve panels. Pick one, write it down, and note that both 2022 and 2026 have an
  incomplete winter under either rule.
- **Negative residual load is valid data.** It must not be clipped, flagged as an error, or
  excluded from aggregates. Log-scale plots and log transforms are therefore unusable for
  this series.
- **Percentage error on a zero-crossing series.** MAE-relative-to-mean is fine; MAPE on
  `residual_load` or `fc_res` is not, because hours near zero produce unbounded percentage
  errors that dominate the average and make the benchmark meaningless.
- **`fc_gen_wind_solar` has no single actual counterpart.** Its comparison partner is the sum
  of three actual series, which is an assumption about what SMARD forecast, not a given.
  State it where the pair is defined.
- **Forecast alignment.** The `fc_*` columns are day-ahead forecasts already aligned to the
  hour they describe, not to the hour they were issued. The benchmark compares them
  row-wise on that basis; if the identity check in the audit suggests otherwise, resolve it
  before reporting any error metric.
- **Solar is zero for most night hours.** Its distribution is zero-inflated; percentile and
  mean summaries of `solar` are misleading without saying so.
- **German decimal format.** A silent failure mode is columns landing as `object` dtype and
  every aggregate looking plausible but wrong. Assert dtypes after conversion.
- **Definitional correlation.** `residual_load ≈ grid_load − (wind_on + wind_off + solar)`.
  A near-perfect correlation between these is arithmetic, not a finding, and must not be
  presented as a feature discovery.
- **Missing data file.** A fresh clone has no `data/smard.csv`. The notebook's first cell
  states where the file comes from and fails with a clear message if it is absent.

## Acceptance criteria

### Structure and loading

- [ ] The notebook lives in `notebooks/`, is named after its author per the team convention,
      and runs top to bottom from a fresh kernel with no manual intervention.
- [ ] The first cell documents the data source and how to regenerate `data/smard.csv`.
- [ ] All numeric columns are `float` after loading; the notebook asserts this.
- [ ] Columns are renamed to the project's snake_case names.

### Data quality

- [ ] Coverage is reported: actual row count vs. the expected count for a complete hourly
      index, with the difference stated.
- [ ] Every index gap is listed, and the spring DST switches are named as the cause.
- [ ] The autumn DST fold is reported as a separate, explicitly named effect.
- [ ] Duplicate timestamps and per-column missing values are reported, including a zero
      result where that is the outcome.
- [ ] Range sanity checks for non-negative generation, night-time solar and positive grid
      load are run, with any violations listed by timestamp.
- [ ] The residual load identity is checked for both actuals and forecasts, with the
      distribution of the difference reported and interpreted.
- [ ] No gap is filled, interpolated or reindexed away anywhere in the notebook.

### Conventions and correctness

- [ ] The week convention is stated as ISO / Monday-start, and asserted by checking that
      each weekly bin starts on a Monday.
- [ ] The weekly bin label side (week start vs. week end) is stated and reflected in axis
      labels.
- [ ] Any week-of-year grouping uses `isocalendar()` year and week together.
- [ ] `period_mean` is tested against `.resample().mean()` for both `"W"` and `"M"`: equality
      asserted on the shared periods, index difference asserted to be exactly the first and
      last period.
- [ ] The test passes, and the two dropped edge periods are printed with their values next to
      a complete neighbouring period.
- [ ] A markdown cell explains the test in prose: what `period_mean` does differently, that
      the arithmetic matches elsewhere, and why the edge rule exists.
- [ ] A per-day normalisation helper exists, returning both MWh/day (period sum ÷ days) and
      average MW (period sum ÷ hours actually present), dropping incomplete edge periods.
- [ ] A comparison table shows hourly mean, average MW and MWh/day for monthly aggregates
      alongside month length and hour count.
- [ ] The comparison is interpreted in prose, including that month length cancels for means
      and matters only for sums, naming which months differ and by how much.
- [ ] The reporting convention — average MW for levels, MWh/day for energy volume — is stated
      and followed.

### Description and time structure

- [ ] A summary statistics table covers all nine series.
- [ ] Each of the nine series has a full-period time series plot and a distribution plot.
- [ ] Weekly and monthly aggregates use `period_mean` or the per-day helper; the only bare
      `.resample()` in the notebook is the one inside the correctness test.
- [ ] Time series plots use `style_timeseries`; seasonal plots use `seasonal_plot`.
- [ ] Annual, weekly and daily seasonality are each shown and described, with the daily
      profile split by season and by weekday/weekend.
- [ ] The Christmas/New Year drop and public holiday behaviour are quantified.
- [ ] Multi-year trend is addressed, with the partial 2026 explicitly excluded or labelled.
- [ ] ACF/PACF for `grid_load` and `residual_load` cover at least 168 lags, with the 24 h and
      168 h peaks called out and the DST caveat noted.

### Capacity drift

- [ ] Annual mean and annual maximum are plotted for `wind_on`, `wind_off` and `solar` as
      three separate series, with mean and maximum not sharing an unlabelled scale.
- [ ] 2026 is labelled partial in every capacity plot, with the bias direction stated
      per series rather than as one blanket caveat.
- [ ] The annual maximum is described as a lower-bound proxy for installed capacity, with
      weather and curtailment named as the reasons.
- [ ] The section closes with findings on capacity normalisation and on the 2022 start date.

### Duck curve

- [ ] Mean hour-of-day `residual_load` is plotted as four season panels with one line per
      year.
- [ ] The meteorological season mapping and the December assignment rule are stated in the
      notebook.
- [ ] Partial year-lines (2026 autumn, and the 2022/2026 winters) are marked as such.
- [ ] The panels are interpreted: midday belly depth over the years, whether and when it goes
      negative, and whether the evening ramp steepens.

### Multivariate and extremes

- [ ] A correlation heatmap over all nine series is present, with the definitional
      `grid_load`/renewables/`residual_load` relation called out as definitional.
- [ ] The response of `residual_load` to renewable infeed is shown and described.
- [ ] Lag correlations for `residual_load` at 1 h, 24 h, 48 h and 168 h are reported.
- [ ] The `residual_load` distribution is described including both tails.
- [ ] The count and share of negative `residual_load` hours is reported overall, per year and
      per month.
- [ ] The calendar signature of both tails (year, month, hour of day, day of week) is shown.
- [ ] At least one negative and one high residual load episode are plotted at hourly
      resolution together with wind, solar and grid load.

### Ramp rates

- [ ] The ramp series is defined as `residual_load.diff()` and labelled MW/h.
- [ ] Differences spanning a DST gap are masked, and the number masked is reported.
- [ ] The ramp distribution is reported with percentiles including 0.1/1/99/99.9, and up and
      down ramps are treated separately.
- [ ] The largest up-ramps and down-ramps are listed with timestamps.
- [ ] The hour-of-day, month, season and year signature of large ramps is shown, with a
      statement on whether large ramps are becoming more frequent.
- [ ] At least two large ramp episodes are plotted hourly with wind, solar and grid load, with
      the driver identified.
- [ ] `fc_res.diff()` is compared against the actual ramp series — distribution overlay,
      scatter, and how often the forecast under-states a large ramp.
- [ ] Ramps are recorded as a candidate second stress definition, with no threshold applied.

### Forecast benchmark

- [ ] The three comparison pairs are defined, with the `fc_gen_wind_solar` counterpart stated
      as an assumption about the summed actual generation series.
- [ ] MAE, RMSE, mean bias and the compared-hour count are reported for each pair.
- [ ] A normalised error (MAE relative to the mean actual level) is reported so the pairs are
      comparable.
- [ ] MAPE is not applied to `residual_load` or `fc_res`; if used at all it is on `grid_load`
      only, with the zero-crossing reason stated.
- [ ] Error is sliced per month over the record, by hour of day, by season and by year, with a
      statement on whether the forecast is improving.
- [ ] Error is sliced by `residual_load` regime across bins of the actual range, with an
      explicit statement of whether the forecast degrades in the tails.
- [ ] The section ends with the concrete benchmark numbers our model has to beat.

### Discipline and presentation

- [ ] No threshold, cut-off, risk flag or labelled column appears anywhere in the notebook —
      neither on residual load level nor on ramp magnitude.
- [ ] No model of our own is fitted, and no train/test split is made.
- [ ] Every plot has a title, axis labels and explicit units (MWh, MW, MWh/day or MW/h), and
      is followed by a written interpretation.
- [ ] A closing "Findings" section lists data quality issues, the conventions decided here,
      seasonal structure, capacity drift, residual load tail and ramp behaviour, the SMARD
      benchmark numbers, and recommended features.
- [ ] The notebook is committed with cleared or reproducible outputs, on a `feature/*`
      branch, and `nbdime` is enabled for the clone.

## Out of bounds

The following are explicitly not part of this spec and must not appear in the delivered
notebook. They belong to later specs:

- Threshold or flag definition for extreme residual load or for ramp magnitude, including
  "just a first guess" quantile cut-offs.
- Any labelled dataset, flag column, ranked risk-day list, or exported artifact derived from
  the extremes or the ramps.
- Fitting any model of our own, including trivial baselines, `statsmodels` decompositions
  used as forecasts, or scikit-learn estimators. Measuring SMARD's published forecast is in
  scope; producing a competing one is not.
- Implementing capacity normalisation, or rebasing the series onto installed capacity. The
  capacity drift section produces the evidence; the decision is the modeling spec's.
- Changing the dataset start date. The recommendation may be made here; acting on it means
  changing `notebooks/API-connection.ipynb`, which is out of bounds.
- Feature matrix construction, train/validation/test splitting, or cross-validation design.
- MLflow runs, experiment logging, or changes to `modeling/config.py`.
- Changes to `notebooks/API-connection.ipynb`, the `FILTERS` dict, or the fetch range.
- Adding data sources beyond `data/smard.csv` — no weather, price, redispatch, curtailment or
  cross-border data in this spec, even where the capacity and ramp sections would benefit.
- Extracting notebook helpers into a Python module, or populating the root `__init__.py`.
- Adding dependencies. If the EDA appears to need a package that is not already in
  `pyproject.toml`, raise it rather than running `uv add` under this spec.
- Editing the other team members' EDA notebooks.
