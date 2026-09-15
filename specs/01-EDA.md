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
2. What are the trend, seasonal and calendar structures in each series?
3. How do the series relate to each other, and which of those relations are candidate
   features for the forecast?
4. What does the `residual_load` distribution look like, and how do its high and negative
   tails behave in time?

Success means: a teammate who has never opened the data can read the notebook top to bottom
and know what the data contains, what is broken about it, and which features are worth
engineering — without re-running any exploration themselves.

## Scope

### IN

- One notebook in `notebooks/`, runnable top to bottom from a fresh kernel against
  `data/smard.csv`.
- Loading and type conversion of the German Excel CSV format.
- Data quality audit: coverage, gaps, duplicates, missing values, implausible values.
- Univariate description of all nine series, including the three SMARD forecast columns
  treated as ordinary series.
- Time structure: trend, annual/weekly/daily seasonality, calendar effects, autocorrelation.
- Multivariate structure: correlations, lagged relations, renewables vs. residual load.
- Distribution and tail behaviour of `residual_load`, described but not thresholded.
- Written findings: every plot is followed by a short prose statement of what it shows.
- A closing summary listing the modeling-relevant conclusions.

### OUT

- Defining the risk flag or any high/negative residual load thresholds. Deferred to the
  modeling spec.
- Producing a labelled or flagged dataset artifact.
- Benchmarking the SMARD forecast columns against the actuals with error metrics
  (MAE/RMSE/bias). Deferred to its own spec; here `fc_*` columns are just columns.
- Any model fitting, feature-matrix construction, or train/test splitting.
- Imputing, resampling away, or otherwise repairing the DST gaps.
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
6. Check for duplicate timestamps and report the count, expected to be zero given the
   autumn switch collapses rather than duplicates the folded hour.
7. Report missing values per column (count and share). If there are none, say so — an
   explicit "zero missing" is a finding.
8. Sanity-check value ranges: generation series (`wind_off`, `wind_on`, `solar`) must be
   non-negative; `solar` must be at or near zero at night; `grid_load` must be strictly
   positive. Report any violation with its timestamps rather than silently dropping it.
9. Do **not** fill, interpolate or reindex away the gaps. Document them and carry on. The
   handling decision belongs to the modeling spec.

### Univariate description

10. For each of the nine series, report mean, std, min, max and the 1/5/25/50/75/95/99
    percentiles in one comparison table.
11. Plot each series over the full period using the shared `style_timeseries` helper, at a
    resolution that stays readable (weekly or monthly `period_mean`, not raw hourly).
12. Plot the distribution of each series (histogram, and boxplot where the tails matter).

### Time structure

13. Aggregate with `period_mean(series, freq)` for weekly and monthly views so that the
    incomplete first and last calendar periods are dropped — the data starts mid-week and
    ends mid-month, and a plain `.resample()` produces fake edge dips.
14. Annual seasonality: use `seasonal_plot(df, y_value, title, ylabel)` (month on x, year as
    hue) for `grid_load`, `residual_load`, and the renewables. State whether the years
    differ in level, in shape, or both.
15. Weekly seasonality: mean profile by day of week, and the weekday/weekend contrast.
16. Daily seasonality: mean hour-of-day profile, split by season and by weekday/weekend, so
    the summer solar dent and the winter evening peak are both visible.
17. Calendar effects: characterise the Christmas/New Year drop and check public holidays
    against comparable weekdays.
18. Trend: state whether a multi-year level change is visible in `grid_load` and
    `residual_load` once seasonality is aggregated out, and be explicit that the final
    period is a partial year (data ends 2026-09-09) and must not be read as a full-year
    trend point.
19. Autocorrelation: ACF/PACF of `grid_load` and `residual_load` over lags covering at least
    one week (≥ 168 h), calling out the 24 h and 168 h peaks. Note in the notebook that the
    DST gaps make the lag axis approximate, since ACF treats rows as evenly spaced.

### Multivariate structure

20. Correlation matrix over all nine series, plotted as a heatmap, with the near-definitional
    relation between `grid_load`, the renewables and `residual_load` called out rather than
    presented as a discovery.
21. Show how `residual_load` responds to renewable infeed: `residual_load` against
    `wind_on + wind_off + solar`, and separately against wind and solar, with the direction
    and strength stated.
22. Lagged relations: correlation of `residual_load` with its own lags at 1 h, 24 h, 48 h and
    168 h, as the empirical basis for candidate lag features.
23. Conclude with a short list of the features this suggests engineering (calendar features,
    lags, rolling means, renewable aggregates), as a recommendation for the modeling spec —
    not as implemented code.

### Residual load and its extremes

24. Describe the `residual_load` distribution in detail: histogram, percentiles, skew,
    and how far the tails reach in both directions.
25. Quantify how much of the record sits below zero: count and share of negative hours, and
    how that share develops per year and per month.
26. Show *when* the extremes occur — the tail hours located on the calendar (year, month,
    hour of day, day of week) — so the seasonal and daily signature of both tails is visible.
27. Show a handful of representative episodes as raw hourly plots: at least one sustained
    negative-residual-load stretch and one high-residual-load stretch, with the concurrent
    wind, solar and grid load in the same figure.
28. State the tail behaviour in words and stop there. Do **not** propose, compute or apply a
    threshold, and do not emit a flag column.

### Presentation

29. Reuse `period_mean`, `style_timeseries` and `seasonal_plot` rather than re-deriving
    equivalents. If a new helper is genuinely needed, define it once near the top of the
    notebook alongside the existing ones.
30. Use matplotlib directly for the styled time series plots and seaborn for the
    seasonal/hue plots, matching the existing notebooks.
31. Every plot carries a title, axis labels and units (MW). Every plot is followed by one to
    three sentences saying what it shows.
32. Close the notebook with a "Findings" section: the data quality issues, the confirmed
    seasonal structure, the residual load tail description, and the recommended features.

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
| `Wind Offshore`         | `wind_off`          | Offshore wind generation                | MW   |
| `Wind Onshore`          | `wind_on`           | Onshore wind generation                 | MW   |
| `Solar`                 | `solar`             | Solar generation                        | MW   |
| `Grid Load`             | `grid_load`         | Total grid load                         | MW   |
| `Residual Load`         | `residual_load`     | Load minus wind and solar infeed        | MW   |
| `Forecast Wind + Solar` | `fc_gen_wind_solar` | SMARD day-ahead wind + solar forecast   | MW   |
| `Forecast Grid load`    | `fc_grid_load`      | SMARD day-ahead grid load forecast      | MW   |
| `Forecast Residual Load`| `fc_res`            | SMARD day-ahead residual load forecast  | MW   |

Known characteristics to expect and confirm:

- One missing local hour at each spring DST switch. The file skips the local **02:00** row —
  e.g. `2025-03-30` runs 01:00 → 03:00. Autumn switches produce a single 02:00 row rather
  than a duplicated one, so the repeated hour is absent from the file too.
- Strong daily, weekly and annual cycles; winter peak; pronounced Christmas/New Year drop.
- `residual_load` and `fc_res` take negative values during renewable oversupply.
- Missing values are assumed to be absent; the audit verifies rather than assumes this.

## Edge cases

- **Spring DST gap.** The hourly index is not complete. Any `.diff()`, `.shift()`,
  rolling window or ACF over the raw index silently treats the jump as a single step.
  Name this in the notebook wherever it applies; do not repair it.
- **Autumn DST fold.** The repeated 02:00 hour is not in the file. This is a second, less
  obvious source of hour loss and must be reported alongside the spring gap.
- **Partial first and last calendar periods.** Data starts mid-week and ends mid-month, so
  every weekly/monthly aggregation goes through `period_mean`, never a bare `.resample()`.
- **Partial final year.** 2026 stops on 09-09. Never compare a 2026 annual aggregate against
  a full year; restrict year-on-year comparisons to matching date ranges or label the
  partial year clearly.
- **Negative residual load is valid data.** It must not be clipped, flagged as an error, or
  excluded from aggregates. Log-scale plots and log transforms are therefore unusable for
  this series.
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

- [ ] The notebook lives in `notebooks/`, is named after its author per the team convention,
      and runs top to bottom from a fresh kernel with no manual intervention.
- [ ] The first cell documents the data source and how to regenerate `data/smard.csv`.
- [ ] All numeric columns are `float` after loading; the notebook asserts this.
- [ ] Columns are renamed to the project's snake_case names.
- [ ] Coverage is reported: actual row count vs. the expected count for a complete hourly
      index, with the difference stated.
- [ ] Every index gap is listed, and the spring DST switches are named as the cause.
- [ ] The autumn DST fold is reported as a separate, explicitly named effect.
- [ ] Duplicate timestamps and per-column missing values are reported, including a zero
      result where that is the outcome.
- [ ] Range sanity checks for non-negative generation, night-time solar and positive grid
      load are run, with any violations listed by timestamp.
- [ ] No gap is filled, interpolated or reindexed away anywhere in the notebook.
- [ ] A summary statistics table covers all nine series.
- [ ] Each of the nine series has a full-period time series plot and a distribution plot.
- [ ] Weekly and monthly aggregates use `period_mean`; no bare `.resample()` feeds a plot.
- [ ] Time series plots use `style_timeseries`; seasonal plots use `seasonal_plot`.
- [ ] Annual, weekly and daily seasonality are each shown and described, with the daily
      profile split by season and by weekday/weekend.
- [ ] The Christmas/New Year drop and public holiday behaviour are quantified.
- [ ] Multi-year trend is addressed, with the partial 2026 explicitly excluded or labelled.
- [ ] ACF/PACF for `grid_load` and `residual_load` cover at least 168 lags, with the 24 h and
      168 h peaks called out and the DST caveat noted.
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
- [ ] No threshold, cut-off, risk flag or labelled column appears anywhere in the notebook.
- [ ] No error metric (MAE/RMSE/bias) comparing `fc_*` columns to actuals appears.
- [ ] Every plot has a title, axis labels and MW units, and is followed by a written
      interpretation.
- [ ] A closing "Findings" section lists data quality issues, seasonal structure, residual
      load tail behaviour and recommended features.
- [ ] The notebook is committed with cleared or reproducible outputs, on a `feature/*`
      branch, and `nbdime` is enabled for the clone.

## Out of bounds

The following are explicitly not part of this spec and must not appear in the delivered
notebook. They belong to later specs:

- Threshold or flag definition for extreme residual load, including "just a first guess"
  quantile cut-offs.
- Any labelled dataset, flag column, or exported artifact derived from the extremes.
- Error metrics benchmarking the SMARD `fc_*` forecasts against the actuals.
- Model fitting of any kind, including trivial baselines, `statsmodels` decompositions used
  as forecasts, or scikit-learn estimators.
- Feature matrix construction, train/validation/test splitting, or cross-validation design.
- MLflow runs, experiment logging, or changes to `modeling/config.py`.
- Changes to `notebooks/API-connection.ipynb`, the `FILTERS` dict, or the fetch range.
- Adding data sources beyond `data/smard.csv` — no weather, price, redispatch or
  cross-border data in this spec.
- Extracting notebook helpers into a Python module, or populating the root `__init__.py`.
- Adding dependencies. If the EDA appears to need a package that is not already in
  `pyproject.toml`, raise it rather than running `uv add` under this spec.
- Editing the other team members' EDA notebooks.
