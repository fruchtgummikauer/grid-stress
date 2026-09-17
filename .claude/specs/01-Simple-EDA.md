# 01 — Simple EDA of `data/smard.csv`

Status: draft
Branch: `feature/*` off `main`
Deliverable: `notebooks/EDA-simple.ipynb`
Follow-up: [02-Deep-EDA.md](02-Deep-EDA.md) covers capacity drift, duck curve evolution, ramp
rates and the SMARD forecast benchmark, and depends on the conventions this spec fixes.

## Prerequisites

The public holiday analysis (Behaviour 29) needs the `holidays` package. This was an open
prerequisite when the spec was written; it has since been satisfied — `holidays>=0.104` is a
runtime dependency in `pyproject.toml` (added in `6c999a3`), so Behaviour 29 simply runs. This
spec still does not authorise adding dependencies of its own; see Out of bounds.

## Goal

Understand the SMARD dataset well enough to make informed modeling decisions for the
1-day-ahead residual load forecast, and to describe — descriptively, without yet defining
thresholds — where the extreme residual load cases that motivate the project actually sit.

This spec must answer:

1. Is the dataset complete, correctly typed, and continuous enough to be treated as an
   hourly time series?
2. Are our aggregation helpers correct, and what exactly do our calendar conventions mean?
3. What are the trend, seasonal and calendar structures in each series?
4. How do the series relate to each other, and which of those relations are candidate
   features for the forecast?
5. What does the `residual_load` distribution look like, and how do its high and negative
   tails behave in time?

Success means: a teammate who has never opened the data can read the notebook top to bottom
and know what the data contains, what is broken about it, and which features are worth
engineering — without re-running any exploration themselves.

This spec also fixes the calendar and aggregation conventions for the whole project. Spec 02
inherits them rather than re-deriving them, so the conventions section is a contract, not a
private choice.

## Scope

### IN

- One notebook, `notebooks/EDA-simple.ipynb`, runnable top to bottom from a fresh kernel
  against `data/smard.csv`.
- Loading and type conversion of the German Excel CSV format.
- Data quality audit: coverage, gaps, duplicates, missing values, implausible values, and
  verification of the residual load identity.
- Project-wide conventions: ISO weeks, the season definition, the reporting units, and the
  descriptive-slice policy.
- A one-off correctness test of `period_mean` against a plain `.resample()`.
- A per-day normalisation helper and a comparison of mean vs. normalised aggregates.
- Univariate description of all nine series, including the three SMARD forecast columns
  treated as ordinary series.
- Time structure: trend, annual/weekly/daily seasonality, calendar effects, autocorrelation.
- Multivariate structure: correlations, lagged relations, renewables vs. residual load.
- Distribution and tail behaviour of `residual_load`, described but not thresholded.
- Written findings: every plot is followed by a short prose statement of what it shows.
- A closing summary listing the modeling-relevant conclusions.

### OUT

- Defining the risk flag or any residual load threshold. Deferred to the modeling spec.
- Producing a labelled or flagged dataset artifact.
- Capacity drift, duck curve evolution, ramp rates and the SMARD forecast benchmark — all
  four are spec 02. The `fc_*` columns appear here only as ordinary series in the
  descriptive and correlation views; no error metric against their actuals is computed.
- Any model fitting, feature-matrix construction, or train/test splitting.
- Imputing, resampling away, or otherwise repairing the DST gaps.
- Changing `notebooks/API-connection.ipynb` or the fetch logic.
- Modifying `notebooks/EDA-robert.ipynb`. Its helper code is copied into the new notebook;
  the original is left exactly as it is.
- Extracting helpers into a Python module. Helpers stay in the notebook for now.
- Any code under `modeling/`.

## Behaviour

### Loading and preparation

1. Read `data/smard.csv` with `pd.read_csv(..., delimiter=";")`. Convert every numeric
   column from the German decimal format (`str.replace(",", ".")` → `float`), and parse
   `timestamp` to datetime.
2. Rename the columns to the project's snake_case names: `wind_off`, `wind_on`, `solar`,
   `grid_load`, `residual_load`, `fc_gen_wind_solar`, `fc_grid_load`, `fc_res`. Set
   `timestamp` as a sorted `DatetimeIndex` on a frame named `ts`. The flat, RangeIndexed
   frame exists only inside the loading cells — everything downstream uses `ts`, so two
   frames cannot drift apart.
3. Print the frame's shape, dtypes, index start/end, and `describe()` so the reader sees the
   raw shape of the data before any plot. Assert that every numeric column is `float`.

### Data quality audit

4. Report coverage: first and last timestamp, total row count, and the expected row count
   for a complete hourly index over that span. State the difference explicitly.
5. Detect and list every gap in the hourly index — the timestamps that a complete hourly
   index would contain but the data does not. Group them and name the spring DST switch as
   the cause where that is what they are. Keep the boolean mask as `spans_gap`; spec 02
   reuses it.
6. Check for duplicate timestamps and report the count, expected to be zero: SMARD already
   collapses the autumn fold rather than emitting the repeated hour twice.
7. Report missing values per column (count and share). If there are none, say so — an
   explicit "zero missing" is a finding.
8. Sanity-check value ranges with an explicit tolerance, because a naive check drowns in
   false positives: generation series (`wind_off`, `wind_on`, `solar`) must be non-negative;
   `grid_load` must be strictly positive; night-time `solar` must be **below 0.5 % of the
   series maximum**, not exactly zero. Most night hours carry a few MWh of reported solar,
   so a strict-zero test would report thousands of "violations" that are simply measurement
   noise. Report any real violation with its timestamps rather than silently dropping it.
9. Verify the residual load identity empirically:
   `residual_load` vs. `grid_load − (wind_on + wind_off + solar)`, and `fc_res` vs.
   `fc_grid_load − fc_gen_wind_solar`. Report the **distribution** of the difference, not
   just whether it is zero, and judge it against decimal rounding: differences of a few
   hundredths of an MWh are the CSV's two-decimal format, not a discrepancy. If the check
   comes out within rounding, state plainly that the target is exactly
   `grid_load − (wind_on + wind_off + solar)` and that `fc_gen_wind_solar`'s actual
   counterpart is exactly the summed generation — spec 02 cites this finding rather than
   re-opening it as an assumption.
10. Do **not** fill, interpolate or reindex away the gaps. Document them and carry on. The
    handling decision belongs to the modeling spec.

### Conventions and correctness

This section fixes conventions for the whole project, spec 02 included.

11. Fix the week convention explicitly: **ISO weeks, Monday start, Sunday end**. In pandas,
    `"W"` is an alias for `"W-SUN"`, which bins by week *ending* Sunday and is therefore
    already ISO-aligned. Assert it rather than trusting the alias — but assert the right
    thing: `to_period("W").start_time` is a Monday **by construction**, so testing that
    proves nothing. The meaningful check is that the first *data* timestamp falling into
    each complete weekly bin is a Monday. The first bin is deliberately excluded, because
    the record opens on Saturday 2022-01-01.
12. State which edge of the week the bin label refers to. `resample("W")` labels each bin
    with its right edge (the Sunday), while `period_mean` labels with the period start (the
    Monday). A weekly plot's x value is therefore not the same date under the two routes —
    say which one is plotted and label the axis accordingly.
13. Where weeks are used as a grouping key rather than a resample frequency, use
    `.dt.isocalendar()` and take `year` and `week` from it together. Never pair
    `isocalendar().week` with `.dt.year`: `2022-01-01` is ISO week 52 of ISO year **2021**
    in this very dataset, and mixing the two scatters those hours into a phantom week at the
    wrong end of the axis. Demonstrate it on that date — it makes the rule concrete. (This
    record contains no 53-week ISO year, so that half of the caveat stays theoretical.)
14. Run a one-off correctness test of `period_mean` against a plain `.resample().mean()` on
    the same series, for both weekly and monthly frequencies:
    - Use `.resample("ME")` for the monthly case. The project runs **pandas 3**, where `"M"`
      has been removed as an offset alias and `.resample("M")` raises `ValueError`.
      `to_period("M")` is unaffected, so `period_mean(s, "M")` itself still works — only the
      comparison side needs the new spelling.
    - Assert that the values agree within floating-point tolerance on the periods both
      produce.
    - Assert that `period_mean` drops exactly the **incomplete** periods. Do not assert
      "the first and last": the data starts at exactly `2022-01-01 00:00`, so January 2022
      is complete and only the last month is dropped. Expect **56 vs. 57** monthly periods
      and **244 vs. 246** weekly ones. Compute the expected set rather than hard-coding it.
    - Relabel before comparing indexes — `period_mean` labels period start, `.resample`
      labels the right edge.
    - Print the dropped edge periods with their `.resample()` values next to a complete
      neighbouring period, so the size of the artefact being avoided is visible in MW.
15. Follow the test with a markdown cell explaining it in prose: that `period_mean` is a
    plain calendar-period mean plus one rule — drop periods the data does not fully cover —
    that the test confirms the arithmetic is identical everywhere else, and that the printed
    edge values show why the rule exists. Note that the rule bites asymmetrically here: both
    edges go for weeks, only the trailing edge for months. This test runs once as a
    correctness check; it is not repeated for every later aggregation.
16. Add a per-day normalisation helper alongside `period_mean`, so month length stops
    contaminating comparisons. It aggregates a period's **sum** and returns both units:
    - **MWh per day** — period sum ÷ days in period, the energy view.
    - **average MW** — period sum ÷ the number of hours **actually present** in that period,
      not `MWh_per_day / 24`.
    It drops incomplete periods on the same rule as `period_mean`. Factor that shared rule
    into one private helper so the edge logic exists in exactly one place.
17. Compare the aggregation variants in one table for monthly aggregates: hourly mean,
    average MW, MWh per day, and naive `sum / (24 × days)`, alongside the month length and
    the true hour count. Then state the conclusion honestly, including the part that turns
    out to be a negative result:
    - The hourly mean and average MW are **the same number, always** — both divide by hours
      present. There is no month in this record where they differ.
    - MWh per day is that number rescaled by days-per-month; for *means* month length
      genuinely cancels.
    - The distortion the helper guards against appears only under the naive
      `sum / (24 × days)` denominator, and only where hours are missing: the five March
      months hold 743 hours instead of 744 and read about **0.13 % low**; September 2026
      holds 216 hours and reads catastrophically low, but the edge rule drops it anyway.
    - It is when we aggregate **sums** that ignoring month length distorts month-over-month
      comparison. Name the affected months and the magnitude.
18. Fix the reporting convention for the rest of the project: **average MW** for load and
    generation levels, **MWh per day** only where the quantity is genuinely energy volume.
    Every plot states which one it is showing. Both `style_timeseries` and `seasonal_plot`
    currently default `ylabel="(MWh)"`, which silently violates this — make `ylabel` a
    required argument in the copies used here.
19. Define the season mapping once, here rather than where it is first plotted, because
    three later sections and all of spec 02 consume it: meteorological seasons
    (winter = Dec/Jan/Feb, spring = Mar/Apr/May, summer = Jun/Jul/Aug, autumn = Sep/Oct/Nov),
    with **December assigned to the following year's winter** (`season_year = year + (month
    == 12)`). Record why: under the alternative rule, winter 2022 would be Jan + Feb + Dec
    2022 and appear complete when it is not, and the record ends before December 2026 so no
    orphan 2027 group arises.
20. State the **descriptive-slice policy** once, so later sections stop colliding with the
    no-threshold rule. Showing "the tail hours" requires selecting them, which reads like
    thresholding but is not: rank-based slices (top/bottom 1 % by rank, largest-N by
    magnitude) are used **for description only**. No boolean column is created, nothing is
    persisted, and no slice is presented as a risk definition. Slices are computed inside
    the plotting cell and never added to `ts`.
21. Define the shared derived columns in one place, immediately after loading, and list them
    against a `SERIES` constant naming the eight data columns. The `renewables` sum and
    `hour` are needed by the audit itself, so they cannot wait for the section that first
    plots them. Without the `SERIES` constant, `ts.corr()` at Behaviour 32 would silently
    pull the derived columns into the "nine series" heatmap.

### Univariate description

22. For each of the nine series, report mean, std, min, max and the 1/5/25/50/75/95/99
    percentiles in one comparison table.
23. Plot each series over the full period using the shared `style_timeseries` helper, at a
    resolution that stays readable (weekly or monthly `period_mean`, not raw hourly).
24. Plot the distribution of each series (histogram, and boxplot where the tails matter).

### Time structure

25. Aggregate with `period_mean(series, freq)` for weekly and monthly views so that periods
    the data does not fully cover are dropped — the data starts mid-week and ends mid-month,
    and a plain `.resample()` produces fake edge dips.
26. Annual seasonality: use `seasonal_plot(df, y_value, title, ylabel)` (month on x, year as
    hue) for `grid_load`, `residual_load`, and the renewables. State whether the years
    differ in level, in shape, or both.
27. Weekly seasonality: mean profile by day of week, and the weekday/weekend contrast.
28. Daily seasonality: the mean hour-of-day profile for **`grid_load`**, as four season
    panels with a weekday and a weekend line in each. Keeping this on `grid_load` leaves the
    `residual_load` hour-of-day profile to spec 02's duck curve, so the two are
    complementary rather than duplicates. The summer solar dent and the winter evening peak
    should both be visible.
29. Calendar effects: characterise the Christmas/New Year drop, and check German **federal**
    public holidays against comparable weekdays — `holidays.country_holidays("DE")` with no
    `subdiv`, since grid load is national and state-specific holidays would make the
    comparison irreproducible.
30. Trend: state whether a multi-year level change is visible in `grid_load` and
    `residual_load` once seasonality is aggregated out. Four complete years plus a stub will
    not carry a trend claim from annual means alone, so add the comparison that does:
    matching **1 January – 9 September** windows year over year. That neutralises the
    partial final year instead of merely labelling it. Where an annual aggregate is shown
    anyway, label 2026 as partial.
31. Autocorrelation: ACF/PACF of `grid_load` and `residual_load` over lags covering at least
    one week (≥ 168 h), calling out the 24 h and 168 h peaks. Pass `method="ywm"` to the
    PACF explicitly, and add a zoomed 0–48 lag panel — 168 lags on a single axis is
    unreadable and collides with the legibility requirement. Note that the DST gaps make the
    lag axis approximate, since ACF treats rows as evenly spaced.

### Multivariate structure

32. Correlation matrix over the nine series — using the `SERIES` constant, not every column
    of `ts` — plotted as a heatmap, with the near-definitional relation between `grid_load`,
    the renewables and `residual_load` called out rather than presented as a discovery.
33. Show how `residual_load` responds to renewable infeed: `residual_load` against
    `wind_on + wind_off + solar`, and separately against wind and solar, with the direction
    and strength stated.
34. Lagged relations: correlation of `residual_load` with its own lags at 1 h, 24 h, 48 h and
    168 h, as the empirical basis for candidate lag features.
35. Conclude with a short list of the features this suggests engineering (calendar features,
    lags, rolling means, renewable aggregates), as a recommendation for the modeling spec —
    not as implemented code.

### Residual load and its extremes

36. Describe the `residual_load` distribution in detail: histogram, percentiles, skew,
    and how far the tails reach in both directions.
37. Quantify how much of the record sits below zero: count and share of negative hours
    overall, per year and per month. Plot the per-month view as a **year × month share
    heatmap, not a line** — the negative share is structurally zero across all of 2022 and a
    line chart renders that as a flat baseline that dominates the axis.
38. Show *when* the extremes occur — the tail hours located on the calendar (year, month,
    hour of day, day of week) — so the seasonal and daily signature of both tails is
    visible. This is a descriptive slice under Behaviour 20.
39. Show representative episodes as raw hourly plots, selected by a stated, reproducible,
    rank-based rule rather than by eye: the **longest run of consecutive negative
    `residual_load` hours**, and the **highest 24-hour rolling mean**, each plotted with
    ±36 h of context and with the concurrent wind, solar and grid load in the same figure.
40. State the tail behaviour in words and stop there. Do **not** propose, compute or apply a
    threshold, and do not emit a flag column.

### Presentation

41. Reuse `period_mean`, `style_timeseries` and `seasonal_plot` from
    `notebooks/EDA-robert.ipynb` rather than re-deriving equivalents — copied across, since
    that notebook is not modified by this spec. Two fixes apply to the copies:
    `seasonal_plot` currently accepts a `ylabel` argument and then discards it, hard-coding
    `"MWh"`; and both helpers default `ylabel="(MWh)"`, which contradicts Behaviour 18. Fix
    the discarded parameter and make `ylabel` required. New helpers required by this spec
    are defined once alongside them near the top.
42. Use matplotlib directly for the styled time series plots and seaborn for the
    seasonal/hue plots, matching the existing notebooks.
43. Every plot carries a title, axis labels and explicit units — MWh, average MW or MWh/day
    as applicable, never an unlabelled number. Every plot is followed by one to three
    sentences saying what it shows.
44. Close the notebook with a "Findings" section: the data quality issues, the calendar and
    aggregation conventions decided here, the confirmed seasonal structure, the residual
    load tail description, and the recommended features. Add a self-check cell asserting the
    invariants an incrementally built notebook is most likely to break — row count, index
    bounds, gap count, zero NaNs across `SERIES`, and that `ts` carries no columns beyond
    `SERIES` plus the declared derived ones. That last assertion is the mechanical proof
    that no flag column crept in.

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
average MW are the same number — a point Behaviour 17 makes explicit rather than leaving the
reader to infer.

Derived series, none of which are persisted:

| Derived                 | Definition                                | Unit |
| ----------------------- | ----------------------------------------- | ---- |
| `renewables`            | `wind_on + wind_off + solar`               | MWh  |
| `year`, `month`, `hour`, `dow`, `is_weekend`, `date` | calendar attributes of the index | — |
| `season`, `season_year` | per Behaviour 19                           | —    |
| `spans_gap`             | index difference > 1 h                     | —    |
| per-day energy          | period sum ÷ days in period                | MWh/day |
| average power           | period sum ÷ hours present in period       | MW   |

Known characteristics to expect and confirm:

- One missing local hour at each spring DST switch — **five in this record**, all at local
  02:00: that is the wall-clock hour the switch skips, so the file has no row carrying that
  label and `2025-03-30` runs 01:00 → 03:00. The autumn
  switch is already handled by SMARD: the repeated hour is collapsed rather than duplicated,
  so the file contains one 02:00 row and that hour is likewise absent.
- Strong daily, weekly and annual cycles; winter peak; pronounced Christmas/New Year drop.
- `residual_load` and `fc_res` take negative values during renewable oversupply, and the
  share grows sharply over the record.
- Missing values are assumed to be absent; the audit verifies rather than assumes this.

## Edge cases

- **Spring DST gap.** The hourly index is not complete. Any `.diff()`, `.shift()`,
  rolling window or ACF over the raw index silently treats the jump as a single step.
  Name this in the notebook wherever it applies; do not repair it.
- **Autumn DST fold.** The repeated 02:00 hour is not in the file. This is a second, less
  obvious source of hour loss and must be reported alongside the spring gap. The autumn
  switch is already handled by SMARD.de.
- **Week convention.** pandas `"W"` means `"W-SUN"` — weeks *ending* Sunday, i.e. ISO
  Monday-start weeks, which is what we want, but the alias reads as if it meant the
  opposite. `"W-MON"` would give weeks ending Monday, which is not ISO.
- **A vacuous assertion.** `to_period("W").start_time` is a Monday for every bin by
  construction, including partial ones. Asserting that passes without testing anything. The
  real check is on the first data timestamp per complete bin.
- **Week label side.** `period_mean` labels the period start, `.resample("W")` the right
  edge. The same week is two different dates depending on the route, so an unlabelled weekly
  axis is ambiguous.
- **ISO year boundary.** `isocalendar().week` must be paired with `isocalendar().year`, never
  with `.dt.year`. `2022-01-01` in this dataset is ISO 2021-W52.
- **pandas 3 removed `"M"`.** `.resample("M")` raises `ValueError`; use `"ME"`.
  `to_period("M")` is unaffected. This bites exactly once, inside the correctness test.
- **"Drop the first and last period" is not the rule.** The rule is "drop periods the data
  does not fully cover". Because the record starts exactly on a month boundary, that is both
  edges weekly but only the trailing edge monthly.
- **Partial first and last calendar periods.** Every weekly/monthly aggregation goes through
  `period_mean` or the per-day helper, never a bare `.resample()` — with the single
  deliberate exception of the correctness test, whose whole purpose is to call `.resample()`
  and compare.
- **Month length.** Month-over-month comparison of *sums* is distorted by day count; for
  *means* it cancels completely. The per-day helper exists so the distinction is explicit
  rather than assumed, and the comparison table is expected to confirm the mean-based views
  were fine all along — a negative result worth writing down.
- **Hours per period ≠ 24 × days.** A month containing the spring switch has 743 hours, not
  744. Average MW must divide by hours actually present; the naive denominator makes those
  months read ~0.13 % low for no physical reason.
- **Partial final year.** 2026 stops on 09-09. Never compare a 2026 annual aggregate against
  a full year; prefer the matching Jan–Sep windows of Behaviour 30, or label it clearly.
- **Negative residual load is valid data.** It must not be clipped, flagged as an error, or
  excluded from aggregates. Log-scale plots and log transforms are therefore unusable for
  this series.
- **Empty groups in the negative-share view.** The negative share is exactly 0 % for all of
  2022, so a per-month line chart shows a flat baseline that swamps the axis. Use a
  year × month heatmap.
- **Solar is near-zero, not zero, at night.** Most night hours carry a few MWh. Exact zeros
  are a small minority of rows, so a strict `== 0` night check produces thousands of false
  violations and a zero-inflation claim would overstate the case. Use the tolerance in
  Behaviour 8.
- **Rounding is not a discrepancy.** The CSV carries two decimals, so identity checks land
  within a few hundredths of an MWh. That is the file format, not a modelling finding.
- **Descriptive slices are not thresholds.** Behaviour 38 and 39 select hours by rank. Under
  Behaviour 20 that is description; it becomes a threshold the moment a boolean column is
  created or a slice is named a risk case. Keep slices local to the plotting cell.
- **German decimal format.** A silent failure mode is columns landing as `object` dtype and
  every aggregate looking plausible but wrong. Assert dtypes after conversion.
- **Definitional correlation.** `residual_load = grid_load − (wind_on + wind_off + solar)`.
  A near-perfect correlation between these is arithmetic, not a finding, and must not be
  presented as a feature discovery.
- **`holidays` may be absent.** See Prerequisites. Behaviour 29 fails at import rather than
  degrading silently, which is intended.
- **Missing data file.** A fresh clone has no `data/smard.csv`. The notebook's first cell
  states where the file comes from and fails with a clear message if it is absent.

## Acceptance criteria

### Structure and loading

- [ ] The notebook is `notebooks/EDA-simple.ipynb` and runs top to bottom from a fresh kernel
      with no manual intervention.
- [ ] The first cell documents the data source and how to regenerate `data/smard.csv`.
- [ ] All numeric columns are `float` after loading; the notebook asserts this.
- [ ] Columns are renamed to the project's snake_case names.
- [ ] Only `ts` is used downstream; the flat frame does not outlive the loading cells.
- [ ] `notebooks/EDA-robert.ipynb` is unchanged.

### Data quality

- [ ] Coverage is reported: actual row count vs. the expected count for a complete hourly
      index, with the difference stated.
- [ ] Every index gap is listed, and the spring DST switches are named as the cause.
- [ ] The autumn DST fold is reported as a separate, explicitly named effect.
- [ ] Duplicate timestamps and per-column missing values are reported, including a zero
      result where that is the outcome.
- [ ] Range checks run with the stated night-solar tolerance, and any real violation is
      listed by timestamp.
- [ ] The residual load identity is checked for both actuals and forecasts, with the
      distribution of the difference reported and judged against decimal rounding.
- [ ] No gap is filled, interpolated or reindexed away anywhere in the notebook.

### Conventions and correctness

- [ ] The week convention is stated as ISO / Monday-start and asserted on the first data
      timestamp of each complete bin, not on `start_time`.
- [ ] The weekly bin label side is stated for both routes and reflected in axis labels.
- [ ] Any week-of-year grouping uses `isocalendar()` year and week together, demonstrated on
      `2022-01-01`.
- [ ] `period_mean` is tested against `.resample()` weekly and monthly, using `"ME"` for the
      monthly case.
- [ ] The test asserts value equality on shared periods and that exactly the incomplete
      periods are dropped, with the expected set computed rather than hard-coded.
- [ ] Indexes are relabelled before comparison.
- [ ] The dropped edge periods are printed with their values next to a complete neighbour.
- [ ] A markdown cell explains the test in prose, including that the rule drops both edges
      weekly but only the trailing edge monthly.
- [ ] A per-day helper exists returning MWh/day and average MW, sharing one edge-rule helper
      with `period_mean`.
- [ ] The comparison table has all four columns — hourly mean, average MW, MWh/day and naive
      `sum/(24×days)` — alongside month length and true hour count.
- [ ] The interpretation states that hourly mean and average MW are always identical, that
      month length cancels for means and matters for sums, and names the affected months with
      magnitudes.
- [ ] The reporting convention is stated, and `ylabel` is required in both helpers.
- [ ] The season mapping and the December rule are defined in this section, with the reason.
- [ ] The descriptive-slice policy is stated before any section that slices by rank.
- [ ] Derived columns are defined after loading, with a `SERIES` constant.

### Description and time structure

- [ ] A summary statistics table covers all nine series.
- [ ] Each of the nine series has a full-period time series plot and a distribution plot.
- [ ] Weekly and monthly aggregates use `period_mean` or the per-day helper; the only bare
      `.resample()` calls in the notebook are the two inside the correctness test.
- [ ] Time series plots use `style_timeseries`; seasonal plots use `seasonal_plot`.
- [ ] Annual and weekly seasonality are shown and described.
- [ ] The daily profile is four season panels × weekday/weekend for `grid_load`.
- [ ] The Christmas/New Year drop is quantified, and federal German holidays are compared
      against comparable weekdays.
- [ ] Trend is addressed via matching 1 Jan – 9 Sep windows year over year, with any annual
      aggregate labelling 2026 partial.
- [ ] ACF/PACF for `grid_load` and `residual_load` cover at least 168 lags with `method="ywm"`
      and a zoomed 0–48 panel, with the 24 h and 168 h peaks and the DST caveat noted.

### Multivariate and extremes

- [ ] The correlation heatmap is built from `SERIES`, not from every column of `ts`, and the
      definitional relation is called out as definitional.
- [ ] The response of `residual_load` to renewable infeed is shown and described.
- [ ] Lag correlations for `residual_load` at 1 h, 24 h, 48 h and 168 h are reported.
- [ ] The `residual_load` distribution is described including both tails.
- [ ] Negative-hour count and share are reported overall, per year, and per month as a
      year × month heatmap.
- [ ] The calendar signature of both tails is shown.
- [ ] Episodes are selected by the stated rank-based rule — longest negative run and highest
      24 h rolling mean — and plotted with ±36 h of context alongside wind, solar and grid
      load.

### Discipline and presentation

- [ ] No threshold, cut-off, risk flag or labelled column appears anywhere in the notebook.
- [ ] No slice is added to `ts`; slices stay local to their plotting cell.
- [ ] No error metric comparing `fc_*` to actuals appears — that is spec 02.
- [ ] No model is fitted and no train/test split is made.
- [ ] `seasonal_plot` honours its `ylabel` argument, and `ylabel` is required in both helpers.
- [ ] Every plot has a title, axis labels and explicit units, and is followed by a written
      interpretation.
- [ ] A closing "Findings" section covers data quality, conventions, seasonal structure,
      residual load tails and recommended features.
- [ ] A self-check cell asserts row count, index bounds, gap count, zero NaNs across `SERIES`,
      and that `ts` carries no unexpected columns.
- [ ] The notebook is committed with cleared or reproducible outputs, on a `feature/*`
      branch, and `nbdime` is enabled for the clone.

## Out of bounds

The following are explicitly not part of this spec:

- Threshold or flag definition for extreme residual load, including "just a first guess"
  quantile cut-offs.
- Any labelled dataset, flag column, ranked risk-day list, or exported artifact.
- **Capacity drift, duck curve evolution, ramp rates, and the SMARD forecast benchmark.**
  All four are [02-Deep-EDA.md](02-Deep-EDA.md). No `residual_load.diff()` analysis and no
  forecast error metric belongs here.
- Model fitting of any kind, including trivial baselines, `statsmodels` decompositions used
  as forecasts, or scikit-learn estimators.
- Feature matrix construction, train/validation/test splitting, or cross-validation design.
- MLflow runs, experiment logging, or changes to `modeling/config.py`.
- Changes to `notebooks/API-connection.ipynb`, the `FILTERS` dict, or the fetch range.
- Changes to `notebooks/EDA-robert.ipynb`, including fixing the `seasonal_plot` bug there.
  The fix lives in this spec's copy only.
- Adding data sources beyond `data/smard.csv` — no weather, price, redispatch or
  cross-border data.
- Extracting notebook helpers into a Python module, or populating the root `__init__.py`.
- **Adding dependencies.** `holidays` is a prerequisite to be installed separately before
  this spec runs; this spec does not run `uv add`, and nothing else may be added either.
- Editing the other team members' EDA notebooks.
