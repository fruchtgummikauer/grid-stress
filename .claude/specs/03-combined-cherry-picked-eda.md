# 03 — Combined Cherry-Picked EDA

Status: draft
Branch: `feature/*` off `main`
Deliverable: `notebooks/01_eda/team-EDA.ipynb`, assembled from the seven sub-specs below, in the
order listed.
Sources: `notebooks/01_eda/EDA-hari.ipynb`, `notebooks/01_eda/EDA-magc.ipynb`,
`notebooks/01_eda/EDA-robert.ipynb`, `notebooks/01_eda/EDA-simple-claude.ipynb`. None of the four
source notebooks are modified by this spec — every plot is copied or rebuilt into the new
notebook, originals are left exactly as they are.

## Goal

The four team members each ran their own exploration of `data/smard.csv` and tagged the plot
cells they consider worth keeping, with the tag text recording *why* ("keep", "duplicate",
"combine with X", "optional", "modelling", "streamlit", ...). This spec (and its seven sub-specs)
consolidates exactly the tagged material into one notebook that runs top to bottom against a
single, shared loading and naming convention, with every "this duplicates that" and "combine with
X" tag resolved into a concrete decision instead of left as an open note.

Every inclusion, exclusion, merge and placement decision was made together with the team during
spec review, cell by cell — this is not an automated tag sweep. Where two notebooks tagged
overlapping plots, the resolution and its reasoning are written out in the relevant sub-spec so
the decision is not lost the next time this notebook is touched.

This top-level spec is too large to keep as one document, so it is split into one sub-spec per
notebook section (03.1–03.7). This document fixes what all seven share — the foundation, the
naming convention, the dynamic-year rule, and the global inclusion/exclusion policy — and each
sub-spec inherits it in one line rather than re-deriving it, the same relationship
[02-Deep-EDA.md](02-Deep-EDA.md) has to [01-Simple-EDA.md](01-Simple-EDA.md).

Success means: `notebooks/01_eda/team-EDA.ipynb` runs from a fresh kernel against
`data/smard.csv` alone, contains no two plots showing the same thing, contains no plot cell that
was not tagged in its source notebook (with the two explicitly agreed exceptions in
[03.2-sanity-check.md](03.2-sanity-check.md)), and every ported plot uses one shared
`time_series` DataFrame, naming convention and unit system rather than four different ones.

## Sub-specs

The notebook is built in this order; each sub-spec is a complete spec in its own right (Goal,
Scope, Behaviour, Data, Edge cases, Acceptance criteria, Out of bounds) covering only its own
section.

| # | Sub-spec | Section | Depends on |
|---|---|---|---|
| 1 | [03.1-setup.md](03.1-setup.md) | Loading, helpers, `YEARS`, `LOADED` — no plots | this document |
| 2 | [03.2-sanity-check.md](03.2-sanity-check.md) | Data-quality checks, one-week plausibility plot | 03.1 |
| 3 | [03.3-univariate-and-time-structure.md](03.3-univariate-and-time-structure.md) | Per-series plots, seasonality, holidays, extreme episodes | 03.1 |
| 4 | [03.4-two-series-comparison-views.md](03.4-two-series-comparison-views.md) | `grid_load` vs. `residual_load`, wind vs. solar overlays | 03.1, 03.3 |
| 5 | [03.5-calendar-structure-heatmaps.md](03.5-calendar-structure-heatmaps.md) | Merged month/weekday/season × hour heatmap, single-event plots | 03.1, 03.3 |
| 6 | [03.6-temporal-dependence-and-extremes.md](03.6-temporal-dependence-and-extremes.md) | ACF/PACF merge, STL, residual-load distribution, ramps | 03.1 |
| 7 | [03.7-context-appendix.md](03.7-context-appendix.md) | Completeness/correlation/load-duration/calendar-share plots, Findings, final self-check | 03.1, all of 03.2–03.6 |

## Inherited foundation

`EDA-simple-claude.ipynb` already implements this project's documented conventions
(CLAUDE.md, [01-Simple-EDA.md](01-Simple-EDA.md)): the `ts` DataFrame, the `SERIES`/`DERIVED`
column split, `period_mean`, `period_energy`, `style_timeseries`, `seasonal_plot`.
[03.1-setup.md](03.1-setup.md) adopts that exact foundation — **renamed to `time_series`** — as
the one convention every other sub-spec builds on. Every plot ported from `EDA-hari.ipynb`,
`EDA-magc.ipynb` and `EDA-robert.ipynb` is rebuilt against `time_series`; their own frame and
variable names (`analysis`, `df`, `valid`, `daily`, `matched`, ...) do not survive into
`team-EDA.ipynb`, and neither does simple-claude's own `ts` — every sub-spec, including quoted
descriptions of simple-claude's source cells, refers to it as `time_series`.

| Reused as `time_series`-based | From | Implemented in |
|---|---|---|
| `time_series` (was `ts`), `SERIES`, `DERIVED`, loading/renaming/dtype asserts | EDA-simple-claude.ipynb | [03.1](03.1-setup.md) |
| `period_mean`, `period_energy`, `_complete_periods` | EDA-simple-claude.ipynb | [03.1](03.1-setup.md) |
| `style_timeseries`, `seasonal_plot` (with the `ylabel` fix) | EDA-simple-claude.ipynb, originally EDA-robert.ipynb | [03.1](03.1-setup.md) |
| `DAY_NAMES = ["Mon",...,"Sun"]` | EDA-simple-claude.ipynb | [03.1](03.1-setup.md) |
| Season mapping, `season_year` December rule | EDA-simple-claude.ipynb | [03.1](03.1-setup.md) |

## Conventions every sub-spec inherits

These are stated once, here, and restated in one line (not re-derived) by each sub-spec that
needs them:

1. **Naming.** The shared DataFrame is `time_series`, never `ts`. No source notebook's own frame
   or variable names survive the port.
2. **No tag, no plot.** A plot cell is ported only if its source cell in `EDA-hari.ipynb`,
   `EDA-magc.ipynb`, `EDA-robert.ipynb` or `EDA-simple-claude.ipynb` carries at least one tag.
   This is applied uniformly, including to `EDA-simple-claude.ipynb`'s own untagged plots — see
   Edge cases for the full list of what this excludes. Non-plot cells a kept plot depends on
   (e.g. rule-selection code) are ported regardless of their own tag.
3. **Dynamic year range.** `data/smard.csv`'s extent is not assumed fixed — the team intends to
   widen the fetch back to 2019. No sub-spec's behaviour may hardcode a specific year, year list,
   anchor-year selection, or colour-per-year mapping; every one is computed from `YEARS`
   ([03.1](03.1-setup.md)) or `time_series.index` at run time.
4. **Units.** Everything ported from `EDA-hari.ipynb` was in GW in the source notebook (divided
   by 1000); every ported value is converted to the project's MW (levels) or MW/h (ramps)
   convention ([01-Simple-EDA.md](01-Simple-EDA.md) Behaviour 18), with axes relabelled
   accordingly. Separately: an **hourly, unaggregated** value of any `SERIES` column is a level
   reading, and `MWh` for that single hour is the same number as `MW` — label such plots `MW`,
   never `"MWh per hour"` (a real redundancy, not just verbose) or bare `MWh`. This corrected an
   existing label in `EDA-simple-claude.ipynb` itself (cells 46, 73, 81, and one markdown cell) —
   see [03.3](03.3-univariate-and-time-structure.md) and
   [03.6](03.6-temporal-dependence-and-extremes.md).
5. **One holiday source of truth.** Wherever German federal holidays are needed
   (`EDA-simple-claude.ipynb`'s existing usage and `EDA-hari.ipynb`'s hand-rolled Easter-based
   calculation), use `holidays.country_holidays("DE")` with no `subdiv`. This avoids a second,
   divergent holiday list and a new `dateutil` dependency.
6. **Descriptive slices stay local.** Any rank- or date-based selection (tail hours, matched
   windows, longest runs) is computed inside its own plotting cell and never persisted onto
   `time_series`, per [01-Simple-EDA.md](01-Simple-EDA.md) Behaviour 20.
7. **Source notebooks are read-only.** Content is copied or rebuilt out of `EDA-hari.ipynb`,
   `EDA-magc.ipynb`, `EDA-robert.ipynb` and `EDA-simple-claude.ipynb`; none of the four is edited.

## Scope

### IN

- One notebook, `notebooks/01_eda/team-EDA.ipynb`, assembling
  [03.1](03.1-setup.md)–[03.7](03.7-context-appendix.md) in order, runnable top to bottom from a
  fresh kernel against `data/smard.csv` alone.
- Everything each sub-spec's own Scope/IN lists.

### OUT

- Everything each sub-spec's own Scope/OUT lists.
- **Any plot cell whose source cell carries no tag**, with the two exceptions
  [03.2-sanity-check.md](03.2-sanity-check.md) names explicitly (both agreed with the team, not
  inferred). See Edge cases for the specific list of otherwise-central plots this excludes.
- Editing `EDA-hari.ipynb`, `EDA-magc.ipynb`, `EDA-robert.ipynb` or `EDA-simple-claude.ipynb`.
- Extending the fetch range in `notebooks/API-connection.ipynb`. This spec consumes whatever span
  `data/smard.csv` already has; it does not fetch further back itself.
- Defining a risk flag or threshold on residual load level or on ramp magnitude. Deferred to the
  modeling spec, same rule as [01-Simple-EDA.md](01-Simple-EDA.md) and
  [02-Deep-EDA.md](02-Deep-EDA.md).
- Model fitting of any kind, feature-matrix construction, or train/test splitting.
- Capacity-drift and forecast-benchmark work (spec 02's territory) — the specific hari items
  cherry-picked into [03.5](03.5-calendar-structure-heatmaps.md) and
  [03.6](03.6-temporal-dependence-and-extremes.md) (harmonic fit, anchor-year day-shape
  comparison, ramp-by-year quantiles) are deliberate exceptions the team chose to include here;
  they do not substitute for spec 02's dedicated treatment.
- Adding dependencies. `holidays` is already a runtime dependency; this spec adds none of its
  own — see Convention 5 on why hari's hand-rolled holiday calculation is replaced rather than
  ported with its `dateutil` import.

## Data

Source file: `data/smard.csv`, produced by `notebooks/API-connection.ipynb`. Format: `sep=";"`,
`decimal=","`, `utf-8-sig`, one header row.

Extent: **not assumed fixed.** At the time this spec was written the tracked extent was
2022-01-01 through 2026-09-09 ([01-Simple-EDA.md](01-Simple-EDA.md)'s snapshot), but the team
intends to widen the fetch back to 2019 — see Convention 3.

| CSV column | snake_case | Unit |
|---|---|---|
| `timestamp` | `timestamp` | — |
| `Wind Offshore` | `wind_off` | MWh |
| `Wind Onshore` | `wind_on` | MWh |
| `Solar` | `solar` | MWh |
| `Grid Load` | `grid_load` | MWh |
| `Residual Load` | `residual_load` | MWh |
| `Forecast Wind + Solar` | `fc_gen_wind_solar` | MWh |
| `Forecast Grid load` | `fc_grid_load` | MWh |
| `Forecast Residual Load` | `fc_res` | MWh |

`SERIES` is these eight columns; `DERIVED` is `renewables`, `year`, `month`, `hour`, `dow`,
`is_weekend`, `date`, `season`, `season_year`, `spans_gap` — see
[03.1-setup.md](03.1-setup.md). Section-specific derived series (`ramp`, `wind_total`, anchor
years, the longest gap-free segment) are documented in the sub-spec that defines them.

## Edge cases

- **Untagged plots are excluded, including central ones — the full list.** Because "no tag, no
  plot" was applied uniformly, several plots that closely match
  [01-Simple-EDA.md](01-Simple-EDA.md)'s required content do **not** appear anywhere in
  `team-EDA.ipynb`:

  | Source | Cell | What it is | Would belong in |
  |---|---|---|---|
  | EDA-simple-claude.ipynb | 27 | Residual-load identity-check histogram | [03.2](03.2-sanity-check.md) |
  | EDA-simple-claude.ipynb | 47 | Boxplot of spread/tails across `SERIES` | [03.3](03.3-univariate-and-time-structure.md) |
  | EDA-simple-claude.ipynb | 61 | Jan–Sep matched year-over-year trend bars | [03.6](03.6-temporal-dependence-and-extremes.md) |
  | EDA-simple-claude.ipynb | 66 | Pearson `SERIES` correlation heatmap | [03.7](03.7-context-appendix.md) |
  | EDA-simple-claude.ipynb | 68 | Residual-load-vs-renewables scatter | [03.3](03.3-univariate-and-time-structure.md) |
  | EDA-simple-claude.ipynb | 76 | Negative-share year × month heatmap | [03.6](03.6-temporal-dependence-and-extremes.md) |
  | EDA-robert.ipynb | 8 | Raw post-load `grid_load` lineplot ("Example Plot") | [03.2](03.2-sanity-check.md) |
  | EDA-robert.ipynb | 18 | Styled DST-gap timeline | [03.2](03.2-sanity-check.md) |
  | EDA-hari.ipynb | 14 | One-week physical-plausibility plot | [03.2](03.2-sanity-check.md) (rebuilt fresh, see there) |
  | EDA-hari.ipynb | 44 | Ramp-magnitude heatmap by season × hour | [03.5](03.5-calendar-structure-heatmaps.md) / [03.6](03.6-temporal-dependence-and-extremes.md) |
  | EDA-hari.ipynb | 56 | Calendar-standardised era comparison | [03.6](03.6-temporal-dependence-and-extremes.md) |
  | EDA-hari.ipynb | 62 | Wind/solar variability & support summary | [03.6](03.6-temporal-dependence-and-extremes.md) |

  If the team wants any of these back, tag the source cell and revise the relevant sub-spec — do
  not silently add them during implementation.
- **Dynamic year range vs. hardcoded anchors.** `EDA-hari.ipynb` hardcodes `YEAR_COLORS` for
  2019–2026, `holiday_years = range(2019, 2026)`, a coverage grid over `range(2019, 2027)`, and
  anchor years `[2019, 2022, 2025, 2026]`. All are rebuilt from `YEARS`
  ([03.1](03.1-setup.md)) in the relevant sub-spec; none of the literals are carried over.
- **`grid_load` and `residual_load` cannot share one colour scale.** `residual_load` crosses
  zero and needs a diverging, zero-centred colormap; `grid_load` is strictly positive and on a
  different level — see [03.5](03.5-calendar-structure-heatmaps.md) for where this matters.
- **Deep-EDA overlap is intentional.** Several behaviours in
  [03.5](03.5-calendar-structure-heatmaps.md) and
  [03.6](03.6-temporal-dependence-and-extremes.md) sit in territory
  [02-Deep-EDA.md](02-Deep-EDA.md) will also cover. They are included because the team tagged and
  confirmed them; their presence here does not satisfy spec 02's acceptance criteria when that
  spec is written.
- **Missing data file.** A fresh clone has no `data/smard.csv`; see
  [03.1-setup.md](03.1-setup.md).

## Acceptance criteria

- [ ] Every sub-spec's own Acceptance criteria are met.
- [ ] `notebooks/01_eda/team-EDA.ipynb` runs top to bottom from a fresh kernel against
      `data/smard.csv` alone, with the seven sections in the order listed above.
- [ ] The DataFrame is named `time_series` everywhere in the notebook; `ts` does not appear.
- [ ] No plot cell in the notebook corresponds to an untagged source cell, except the two
      exceptions [03.2-sanity-check.md](03.2-sanity-check.md) names.
- [ ] No literal calendar year appears anywhere in the notebook's code.
- [ ] No value ported from `EDA-hari.ipynb` is left in GW.
- [ ] No axis label anywhere in the notebook reads `"MWh per hour"`; hourly level readings are
      labelled `MW`, ramps `MW/h`.
- [ ] No `dateutil` import appears anywhere in the notebook.
- [ ] None of `EDA-hari.ipynb`, `EDA-magc.ipynb`, `EDA-robert.ipynb`, `EDA-simple-claude.ipynb`
      are modified.
- [ ] The notebook is committed with cleared or reproducible outputs, on a `feature/*` branch,
      with `nbdime` enabled for the clone.

## Out of bounds

- Everything each sub-spec's own Out of bounds section rules out.
- Including any plot whose source cell has no tag, beyond the two explicit exceptions in
  [03.2-sanity-check.md](03.2-sanity-check.md).
- Editing `EDA-hari.ipynb`, `EDA-magc.ipynb`, `EDA-robert.ipynb` or `EDA-simple-claude.ipynb`.
- Extending or otherwise changing the fetch range in `notebooks/API-connection.ipynb`, the
  `FILTERS` dict, or re-running it as part of this spec.
- Threshold or flag definition for extreme residual load or ramp magnitude, including "just a
  first guess" quantile cut-offs.
- Any labelled dataset, flag column, ranked risk-day list, or exported artifact.
- Model fitting of any kind, including trivial baselines or scikit-learn estimators.
- Feature matrix construction, train/validation/test splitting, or cross-validation design.
- MLflow runs, experiment logging, or changes to `modeling/config.py`.
- Capacity drift and forecast-benchmark analysis in the sense of
  [02-Deep-EDA.md](02-Deep-EDA.md) — the specific hari items pulled into
  [03.5](03.5-calendar-structure-heatmaps.md) and
  [03.6](03.6-temporal-dependence-and-extremes.md) are exceptions the team named explicitly, not
  a general licence to pull in more deep-EDA content without going back to spec 02.
- Adding dependencies, including `dateutil` for the holiday calculation this spec deliberately
  replaces.
- Re-deriving conventions `EDA-simple-claude.ipynb` already fixes (season mapping, week
  convention, reporting units) — restate them only where a ported plot's prose needs the
  reminder, per the pattern [02-Deep-EDA.md](02-Deep-EDA.md) already uses.
