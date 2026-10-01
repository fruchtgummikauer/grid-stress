# Feature Spec: Initial Streamlit Application

- Status: version 1 (§1–§16) run → `streamlit/` (PR #28): Home, EDA, Model. The body of §1–§16 is
  the spec as run; where its "known current state" is out of date, a *Superseded* note points to
  §17.
- Version 2 (§17, planned 2026-10-01): stakeholder model pages built from the spec 09 / spec 10
  notebooks, interactive with Plotly. Branch: `feature/streamlit-model-dashboard`.
- Built one step at a time, with a review and an approved commit per step.

## Changes after the first run

| Change | Why |
|---|---|
| Imports `from data_loading …` / `from viz_helpers …` instead of `from streamlit.…` (step 1 of §17) | The installed `streamlit` package shadowed the `streamlit/` folder, so every page failed with `ModuleNotFoundError`; Streamlit puts the main script's folder on `sys.path` |
| Plotly moved from the dev group to the runtime dependencies; `viz_helpers.py` gains the model, bin and risk colours of the visualization notebooks plus `style_plotly` / `model_line` / `model_label`; the `chart-style` skill allows Plotly for the model pages (step 2 of §17) | Stakeholders need zoom, hover and series toggles; §11's approval rule was followed (team decision, 2026-10-01) |

## 1. Goal

Create the first Streamlit application for the project.

The application should provide a simple interface for presenting the project and exploring work already available in the repository.

The first version must contain three pages:

1. Home
2. EDA
3. Model

This initial application is intended to showcase the current state of the project rather than introduce new analysis or modeling.

---

## 2. Core Principle

The repository is the source of truth.

Do not invent:

* project findings
* datasets
* column names
* file paths
* model results
* evaluation metrics
* thresholds
* charts
* statistics
* conclusions
* model names
* feature names
* data sources

Only use information, code, outputs, files, and results that already exist in the repository.

If required information does not exist in the repository, explicitly identify it as missing instead of creating it.

---

## 3. Working Method

Work incrementally.

Before making changes at each major step:

1. Inspect the relevant repository files.
2. Summarize what was found.
3. Explain the proposed next action.
4. Ask the user for approval.
5. Proceed only after approval.

Do not implement multiple major steps at once.

Do not make assumptions when repository evidence is unavailable.

---

## 4. Initial Repository Inspection

Before writing Streamlit code, inspect the repository to understand its current structure.

Identify, where available:

* existing Python modules
* notebooks
* datasets
* processed datasets
* plots
* EDA outputs
* feature-engineering code
* target-variable definitions
* risk-label definitions
* risk-label analysis
* evaluation-metric analysis
* trained models
* saved model artifacts
* evaluation results
* requirements or dependency files
* existing Streamlit files
* project documentation

**No notebook currently implements a "Streamlit view."** `.claude/specs/04.2-streamlit-views.md`
describes candidate presentation-layer content (an ISO-week cascade level, a single-window zoom
demo, a tolerance-share metric), but its own status line marks it **parked — not run**, and it is
only ever added to `notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` if a team member
explicitly un-parks it. As of this inspection that has not happened: do not treat 04.2 as an
existing notebook, do not look for one, and do not reproduce its *proposed* content (ISO-week
level, zoom demo, tolerance share) on any page unless a team member says it has since been
un-parked and implemented. The same applies to `04.1-naive-baseline.md` (naive/persistence
baselines, skill scores) and `04.3-risk-label-link.md` (SMARD-forecast-vs-risk-label error,
flag agreement, confusion matrix, precision/recall) — both are parked, not run, and nothing in the
repository currently implements them.

> *Superseded (2026-10-01):* 04.1 / 04.2 / 04.3 are still parked, but spec 09
> (`visualization-01-regression-best-models.ipynb`) now compares our forecasts with SMARD, and
> spec 10 (`visualization-02-classification-risk-labels.ipynb`) scores hits, misses, false alarms,
> recall and precision of the risk flags **on the test window**. Version 2 (§17) presents those;
> nothing from the parked specs.

What *does* exist and is safe to reuse:

* `notebooks/03_risk_classification/risk-definition.ipynb` — the risk-label definitions,
  thresholds and findings from `.claude/specs/02-Risk-Definition.md`.
* `notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` — the SMARD forecast-benchmark
  metrics from `.claude/specs/04-forecast-metrics.md` (MAE/RMSE/bias/nMAE, tail behaviour, the
  "benchmark to beat"). This is a benchmark of SMARD's own forecast, not a model of ours, and
  contains no naive baseline, no risk-label linkage, and no Streamlit-oriented views (all parked,
  per above).

Note also that both notebooks' exported CSV artifacts
(`data/risk_classification/risk_labels_daily.csv` / `_hourly.csv` and
`data/metrics/smard_forecast_errors_*.csv` / `smard_benchmark_metrics.csv`) are gitignored and, at
the time of this inspection, **not present on disk** — only their `.gitkeep` placeholders are.
They must be regenerated locally by running the two notebooks before the app can load them; check
for their presence again immediately before Phase 3, since the team may generate them in the
meantime.

Do not modify any files during this inspection step.

After inspection, provide a short repository summary and propose the Streamlit architecture.

Wait for user approval before implementation.

---

## 5. Application Structure

The application must contain three main pages.

### Page 1 — Home

Purpose:

Introduce the project using information already documented in the repository.

Potential content may include, only if supported by repository evidence:

* project title
* project objective
* problem statement
* residual-load definition
* explanation of the project's forecasting or grid-stress context
* overview of the available data
* overview of the application pages

Known project concept:

`Residual Load = Grid Load - Wind Generation - Solar Generation`

Do not add claims about system behavior, findings, model performance, or data coverage unless they are supported by repository content.

---

### Page 2 — EDA

Purpose:

Allow users to explore the exploratory data analysis already performed in the project.

The exact contents of this page must be determined from the repository.

Possible sections should only be implemented if corresponding data or analysis exists.

Examples include:

* dataset overview
* time-series plots
* descriptive statistics
* missing-data analysis
* load patterns
* wind-generation patterns
* solar-generation patterns
* residual-load patterns
* distributions
* seasonality
* correlations
* extreme residual-load observations
* ACF analysis
* PACF analysis
* Fourier or spectral analysis

Do not generate new findings merely to populate the page.

Existing EDA code or repository outputs should be reused where practical.

If interactive filters are appropriate for the existing data, propose them after inspecting the repository.

There is no separate Streamlit-view notebook (§4). **Use `notebooks/01_eda/team-EDA.ipynb` as the
primary source for this page.** Per CLAUDE.md, it is spec 03's consolidated output — it already
applied the "no tag, no plot" rule to `EDA-simple-claude.ipynb`'s tagged cells (including the
`streamlit` / `streamlit for extreme cases` tags) and the team has since edited its plots and
interpretations directly. Treat `team-EDA.ipynb`'s current content, as it stands in the
repository now, as the settled EDA material to draw from — do not go back to `EDA-simple-claude.ipynb`'s
raw tags as a separate source, since `team-EDA.ipynb` already is that consolidation, possibly
revised since. `team-EDA.ipynb` is a shared, team-edited file (CLAUDE.md) — read from it only,
never edit it as part of this work.

---

### Page 3 — Model

Purpose:

Present the current modeling, risk-label, and evaluation work available in the repository.

For this first version, the Model page should also serve as the main place for communicating the project's current risk-analysis findings.

The exact contents must be derived from repository evidence.

> *Superseded (2026-10-01):* six models are saved in `data/models/<model_key>/`, and specs 09 and
> 10 score them against SMARD. The model content moves to the version 2 pages (§17); this page
> becomes "Method: risk definition".

**Known current state (verify again at Phase 1, since it may change before implementation):**
there is no trained model of our own and no saved model artifact anywhere in the repository —
`modeling/*.py` is unrelated coffee-dataset template code per CLAUDE.md, and `models/` holds only
a `.gitkeep`. Real, reusable content instead comes from two run notebooks: `risk-definition.ipynb`
(the risk-label definitions and thresholds, `.claude/specs/02-Risk-Definition.md`) and
`forecast-metrics-claude.ipynb` (SMARD's own forecast-benchmark metrics —
`.claude/specs/04-forecast-metrics.md`, i.e. the target our future model has to beat, not a model
of ours). Naive baselines, flag-agreement/classification metrics against `fc_residual_load`, and
Streamlit-oriented views are all parked specs (`04.1`, `04.3`, `04.2`) with nothing implemented —
see §4. So for this first version, "Model" page content is realistically: risk-label findings +
SMARD-benchmark findings + an explicit "no model trained yet" status, not classification results
of our own.

#### Risk-label section

Locate and use the existing risk-label work in the repository.

Where available, present:

* the current conceptual definition of risk
* the current risk-label formulation
* thresholds or labeling logic
* distinctions between high and low residual-load events
* persistence or duration logic
* class distribution
* identified edge cases
* important findings from the risk-label analysis
* relevant risk-label plots

Do not redefine or modify the risk label unless explicitly requested.

Do not introduce new thresholds.

Use the current repository implementation and documented reasoning as the source of truth.

#### Metrics section

Locate the existing work on evaluation metrics — this is `forecast-metrics-claude.ipynb`
(`.claude/specs/04-forecast-metrics.md`), which benchmarks **SMARD's own day-ahead forecast**
against actuals. There is currently no evaluation of a model of ours to present alongside it.

Where available, present:

* the regression metrics actually calculated for SMARD's forecast (MAE, RMSE, bias, nMAE,
  `hour_count`) and why each is relevant
* limitations or caveats already identified (e.g. the information-set caveat, the
  regression-to-the-mean check, why nMAE is withheld at some levels)
* the "benchmark to beat" framing from that notebook's closing section

Do not invent metric results, and do not present classification metrics (precision, recall,
confusion counts, ROC/PR) — per §4, the spec that would produce those
(`.claude/specs/04.3-risk-label-link.md`) is parked and not implemented.

Clearly distinguish between:

* metrics actually calculated (SMARD's forecast benchmark)
* metrics that are only proposed/parked and not yet computed (naive-baseline skill scores,
  flag-agreement classification metrics) — name them as future work, not results

#### Main findings

Extract the most relevant findings already documented in:

* risk-label notebooks
* EDA notebooks
* metric-analysis notebooks
* model notebooks
* project documentation

Present only findings supported by actual repository evidence.

Each finding should be associated with the relevant chart, statistic, table, or analysis where possible.

Avoid vague claims such as:

* "the model performs well"
* "there is strong seasonality"
* "risk events are rare"

unless those claims are demonstrably supported by repository outputs.

#### Relevant graphs

Identify the most informative existing graphs for communicating the current modeling and risk-analysis work.

Potential examples, only if already available, include:

* residual load with risk periods highlighted
* high-risk versus low-risk events
* risk-label distribution
* duration of risk events
* residual-load distributions by class
* threshold visualizations (rolling / zero / static bases)
* SMARD forecast-benchmark plots: error by residual-load level, day-max/min error, MAE/bias over
  time

Not yet available — do not include, per §4 (04.1/04.2/04.3 are parked, nothing implemented):

* model predictions versus actual values, confusion matrix, ROC or precision-recall plots,
  feature importance, or any metric-comparison chart between our model and SMARD's — none of
  this exists because no model of ours exists yet
* any tolerance-share, ISO-week-level, or single-window-zoom chart — those are 04.2's proposed
  content, not implemented anywhere

Prefer a small number of informative plots over reproducing every notebook figure. Base the
choice of which plots are presentation-ready on what these two notebooks already contain, since
there is no separate reference notebook for this (§4).

#### Modeling section

Where actual model work exists, present:

* modeling objective
* target variable
* feature groups
* train/test methodology
* models evaluated
* evaluation results
* predictions versus actual values
* feature importance
* error analysis

If no completed model currently exists, do not fabricate model results.

The page may still include the existing risk-label, metric, and modeling-design work.

---

## 6. Visualization and Storytelling

The application should not simply reproduce notebooks.

Use repository-supported findings to create a coherent presentation flow.

There is no separate "Streamlit view" reference notebook (§4) — the primary references for
structure and content are `team-EDA.ipynb` (per §5 Page 2), `risk-definition.ipynb`, and
`forecast-metrics-claude.ipynb`.

During repository inspection, determine:

1. which of `team-EDA.ipynb`'s figures and which risk-label / forecast-metrics figures are
   based on real repository data and outputs
2. which layouts work well for the Streamlit application
3. which graphs best support the project's main findings
4. which graphs are redundant or unsuitable for the app

Prefer a presentation sequence such as:

`context → evidence → finding → implication`

but only where the repository supports that interpretation.

Do not invent explanatory conclusions to create a narrative.

---

## 7. Architecture

Do not decide the final file structure until the repository has been inspected.

Prefer a simple Streamlit structure that integrates with the existing project rather than creating unnecessary parallel architecture.

A possible structure, subject to repository inspection, could be:

```text
streamlit_app.py

pages/
    1_EDA.py
    2_Model.py
```

or an equivalent structure compatible with the repository.

Do not create this structure automatically if an existing application structure or project convention is already present.

---

## 8. Reuse Existing Project Code

Avoid duplicating existing logic.

Where possible, reuse existing repository functions for:

* loading data
* preprocessing
* residual-load calculation
* risk-label creation
* feature engineering
* model loading
* prediction
* evaluation
* plotting

Do not copy substantial notebook logic directly into Streamlit if reusable project functions already exist.

If notebook-only logic must be reused, first identify it and propose how it should be refactored.

Wait for approval before performing significant refactoring.

---

## 9. Data Handling

The Streamlit application must use project data already available through the repository.

Requirements:

* do not change raw data
* do not create fake or placeholder datasets
* do not silently impute values
* do not silently recalculate project results differently from the existing analysis
* use existing processed data where available
* clearly handle missing files or unavailable artifacts

Any data transformations performed by the application should match existing project logic.

> *Superseded (2026-10-01):* `data/metrics/` and `risk_labels_*.csv` are on disk now; the version 2
> exports are not (see §17, Data).

**Known gap to re-check at Phase 1:** as of this spec's last inspection, `data/risk_classification/`
and `data/metrics/` contained only their `.gitkeep` placeholders — the CSV exports from
`risk-definition.ipynb` and `forecast-metrics-claude.ipynb` had not been regenerated locally. If
still true when implementation starts, the Model page must show a clear "not yet generated — run
`risk-definition.ipynb` / `forecast-metrics-claude.ipynb` to produce this data" message rather than
fabricate figures or silently skip the section (see §12).

---

## 10. User Interface

Keep the first version simple and suitable for a data-science capstone presentation.

Prioritize:

* clarity
* readability
* intuitive navigation
* consistent layout
* useful visual hierarchy
* clear separation between EDA, risk analysis, and model results

Avoid unnecessary styling or complex custom HTML/CSS unless the project already contains a design system that should be reused.

The application should prioritize communicating the analysis over decorative UI.

---

## 11. Dependencies

Before adding new dependencies:

1. inspect the current dependency configuration
2. determine whether the required package already exists
3. explain why a new dependency is needed
4. request approval before adding it

Do not modify dependency files without approval.

---

## 12. Error Handling

The application should fail clearly when required project assets are unavailable.

Examples:

* missing dataset
* missing model file
* missing expected column
* invalid path
* unavailable risk-label artifact (`data/risk_classification/risk_labels_daily.csv` /
  `_hourly.csv` — gitignored, regenerated by `risk-definition.ipynb`)
* unavailable metric results (`data/metrics/smard_forecast_errors_*.csv` /
  `smard_benchmark_metrics.csv` — gitignored, regenerated by `forecast-metrics-claude.ipynb`)

Prefer informative Streamlit messages over unhandled exceptions where practical.

Do not silently substitute missing content.

---

## 13. Scope

### In scope

* initial Streamlit application
* Home page
* EDA page
* Model page
* repository-aware data loading
* visualization of existing project analysis
* presentation of current risk-label analysis
* presentation of current metric work
* presentation of existing model results
* reuse of `team-EDA.ipynb` and existing risk-label/forecast-metrics figures for layout and
  visualization guidance (no separate Streamlit-view notebook exists, see §4)
* basic navigation
* basic error handling

### Out of scope for the initial version

Unless separately approved:

* new machine-learning models
* redefining risk labels
* changing risk thresholds
* retraining models
* major data-pipeline changes
* new datasets
* external APIs
* deployment configuration
* authentication
* complex custom styling
* invented placeholder results

---

## 14. Implementation Phases

Implementation should be divided into approval checkpoints.

### Phase 1 — Repository audit

Inspect the repository and report:

* relevant files
* data assets
* notebooks
* `notebooks/01_eda/team-EDA.ipynb` (no separate Streamlit-view notebook exists, see §4)
* EDA outputs
* risk-label analysis
* metric-analysis material
* modeling artifacts
* existing reusable functions
* candidate plots for the app
* likely Streamlit integration points

Stop and request approval.

### Phase 2 — Application content map

Based on the repository audit, propose:

* Home page content
* EDA page sections
* Model page sections
* key findings to include
* plots to include
* plots to omit
* relationship between findings and visual evidence

Stop and request approval.

### Phase 3 — Application architecture

Propose:

* file structure
* shared utilities
* data-loading strategy
* plotting strategy
* caching strategy if relevant

Do not create files yet.

Stop and request approval.

### Phase 4 — Home page

Implement the Home page using repository-supported project information.

Show the result and request approval.

### Phase 5 — EDA page

Implement the EDA page using existing analysis and data.

Show the result and request approval.

### Phase 6 — Model page

Implement the Model page using existing:

* risk-label definition
* risk-label findings
* evaluation-metric work
* relevant graphs
* model results
* modeling status

Show the result and request approval.

### Phase 7 — Integration and cleanup

Check:

* navigation
* imports
* paths
* dependencies
* error handling
* consistency between pages
* consistency between written findings and displayed figures

Request approval before final cleanup changes.

---

## 15. Acceptance Criteria

The first Streamlit version is complete when:

* [ ] The repository has been inspected before implementation.
* [ ] `team-EDA.ipynb`, `risk-definition.ipynb` and `forecast-metrics-claude.ipynb` have been
      reviewed.
* [ ] No unsupported project information has been invented, and nothing from the parked specs
      (`04.1`, `04.2`, `04.3`) is presented as implemented.
* [ ] The app launches successfully.
* [ ] A Home page exists.
* [ ] An EDA page exists.
* [ ] A Model page exists.
* [ ] Home-page content comes from repository-supported information.
* [ ] EDA visualizations use repository data or outputs.
* [ ] The Model page includes current risk-label work.
* [ ] The Model page includes current metric-analysis work.
* [ ] Main findings are supported by repository evidence.
* [ ] Relevant graphs are linked clearly to those findings.
* [ ] Existing model results are presented accurately.
* [ ] Missing artifacts are handled clearly.
* [ ] Existing reusable project code is reused where practical.
* [ ] No raw project data is modified.
* [ ] No unnecessary dependencies are introduced.
* [ ] Each major implementation phase has been approved by the user.

---

## 16. First Required Action

Do not begin implementation.

First inspect the repository.

Specifically locate and review:

1. project documentation
2. EDA notebooks
3. risk-label notebooks or scripts
4. metric-analysis notebooks or documentation
5. model notebooks and artifacts
6. `notebooks/01_eda/team-EDA.ipynb`, the primary EDA-page source (no separate Streamlit-view
   notebook exists, see §4)
7. existing figures and plotting code
8. reusable project modules

Return:

1. a concise repository map
2. relevant files for the Streamlit application
3. existing EDA material
4. current risk-label definition and available findings
5. current metric-analysis material
6. existing modeling material
7. candidate graphs for the app
8. relevant structure or visualization ideas from `team-EDA.ipynb` and the risk-label /
   forecast-metrics notebooks
9. reusable functions or modules
10. missing information or artifacts
11. a proposed content map for Home, EDA, and Model

Then stop and wait for user approval.

---

## 17. Version 2 — Stakeholder model pages

### Goal

Show stakeholders (non-technical, data-literate) **where our day-ahead residual-load forecast beats
SMARD's** and **whether it catches risk days better**. Both answers already exist as notebooks:

- [09-best-model-plots.md](09-best-model-plots.md) →
  `notebooks/05_modeling/visualization-01-regression-best-models.ipynb`: accuracy picks vs SMARD.
- [10-model-risk-labeling.md](10-model-risk-labeling.md) →
  `notebooks/05_modeling/visualization-02-classification-risk-labels.ipynb`: spec 02's risk flags
  on the picks vs SMARD.

Version 2 ports their charts and ideas to two new **interactive Plotly** pages. It does not add
analysis: every number comes from the exports below, and text states numbers through f-strings,
never copied from a notebook's dated findings.

### Decisions (team, 2026-10-01)

| Decision | What |
|---|---|
| Chart library | Plotly for the model pages; the EDA and method pages stay matplotlib (`chart-style` skill) |
| Models shown | The notebook picks by default: the `N_PICKS` overall picks for accuracy, `high_model` / `low_model` for risk. A "compare all models" selector below each headline chart adds any registry row |
| Pages | Two new pages, not four: accuracy, by-situation and explorer merged into one page; risk days the other |
| Scope | Plan, then build step by step on `feature/streamlit-model-dashboard` |
| CLAUDE.md | Not edited in this version until the team says so |

### Data

Read only; the app computes no new results and changes no file.

| File | Producer | Use |
|---|---|---|
| `data/models/model_scoreboard.csv` | `regression-models-claude.ipynb` (spec 06), `EXPORT_ENABLED` | Scoreboard values, pick rules, self-check |
| `data/models/model_forecast_errors_hourly.csv` | same | Test hours, forecasts, errors |
| `data/metrics/smard_forecast_errors_hourly.csv` | `forecast-metrics-claude.ipynb` (spec 04) | SMARD's forecast and error |
| `data/risk_classification/model_risk_labels_{daily,hourly}.csv` | `visualization-02-…ipynb` (spec 10), `EXPORT_ENABLED` | Thresholds, flags and ranges per source (`actual`, `model`, `smard`), `high_model` / `low_model` |

- The risk page reads flags from the spec 10 export, so **no rule function is ported** and no
  threshold is recomputed (CLAUDE.md rule).
- Flags stay three-state (`boolean`); an empty flag is "not evaluable", never `fillna(False)`.
- A missing file shows `st.error` naming the producing notebook (and the toggle), as the notebooks'
  `load()` does; the page then stops.
- A light snapshot self-check (spec 09 Behaviour 5) compares the rebuilt MAE and hour counts with
  the scoreboard; on a mismatch, `st.error` names the file to regenerate.
- **Prerequisite:** the `data/models/model_*.csv` and `model_risk_labels_*.csv` files are not on
  disk (2026-10-01). Re-run `regression-models-claude.ipynb` with `load_saved = "results"` and
  `EXPORT_ENABLED` on, then `visualization-02-…ipynb`. Both are spec outputs: a team member runs
  them, or Claude after explicit confirmation.

### Notebook → app mapping

| Notebook chart / idea | Source | App |
|---|---|---|
| Picks table, Scoreboards A / B | spec 09 §3 | KPI cards and a simplified table |
| Monthly skill vs SMARD | spec 09 §4.1 | Headline chart (page 1) |
| Cumulative advantage | spec 09 §4.2 | Page 1, ending value as a KPI |
| Skill by hour of day | spec 09 §4.3 | Page 1 |
| Week view + hourly "who was closer" bars | spec 09 §4.4 | Page 1, week selector |
| Monthly MAE / bias / persistence, worst month | spec 09 §9.1 | Page 1 expander and the "where we lose" note |
| MAE by hour, ordinary hours | spec 09 §5.1 | Page 1, accuracy by situation |
| Error-range bars (middle 80 %, bias diamond) | spec 09 §5.2 / 6.2 / 7.2 / 8.2 | Page 1, one tab per bin |
| Flagging scatter (hits / misses / false alarms) | spec 09 §6.1 / 7.1 / 8.1 | Page 1, same tabs |
| Day scores: hits, misses, false alarms, recall, precision | spec 10 §3 | Page 2 scorecard |
| Risk calendar | spec 10 §4.1 / 5.1 | Page 2 |
| Zoom weeks with threshold and flag strip | spec 10 §4.2 / 5.2 | Page 2, week picker |
| Day margin | spec 10 §4.3 / 5.3 | Page 2 |
| Findings (weekend lead, Nov–Mar weekdays, small counts) | spec 10 §7 | Page 2 text, computed live |

### Behaviour

**Shared code**

1. **`streamlit/model_results.py`** holds cached loaders and derived quantities, ported (not
   imported) from spec 09 §1.2–§2 and spec 10 §1.2:
   - `load_exports()`, with the missing-file handling above. The `data/` folder is found through
     `data_loading._find_data_dir`.
   - Common hours, the `P1 / P25 / P75 / P99` edges, `bin_masks`, full months and monthly MAE.
   - Spec 09's pick rules (`forecast_share`, `failed_rules`, `rank_key`) and settings (`N_PICKS`,
     `MIN_BIN_SHARE`, `CANDIDATE_SPLIT`), so the picks are computed from the scoreboard and never
     hardcoded. Risk picks come from `high_model` / `low_model`.
2. **Charts** use `viz_helpers.style_plotly`, `model_line` and `model_label`, and the colours
   `MODEL_STYLE`, `BIN_COLOR`, `RISK_COLOR`, `OUTCOME_COLOR` and `HOLIDAY_COLOR` (step 2, done). Each
   axis title states its unit (`MWh`, `%`).

**Page 1 — `pages/1_Where_we_beat_SMARD.py`**

3. KPI cards: test MAE against SMARD's, skill vs SMARD (%), months won `x / n`, cumulative MWh
   saved, each for the first overall pick, with the others in the help text.
4. Monthly skill chart: one line or bar group per overall pick, 0 = SMARD. A shaded band marks the
   **longest run of consecutive full months that every overall pick wins**, derived from the data;
   annotations mark the best and the worst month.
5. Cumulative advantage `Σ(|e SMARD| − |e pick|)`, skill by hour of day, and the week view (actual,
   SMARD, first pick; the hourly advantage bars below on a shared x axis). The week selector lists
   every full Monday–Sunday test week, and the default is the last one.
6. Expander "The numbers behind it": spec 09 §9.1's monthly table and the worst month's day count.
7. Under each headline chart, a "compare all models" multiselect, defaulting to the picks.
8. Section "Accuracy by situation", with tabs `ordinary`, `below_zero`, `low_extreme` and
   `high_extreme` (stakeholder labels from `BIN_LABEL`). Each tab shows:
   - the picks of that bin as cards (MAE by actual against SMARD's, forecast share)
   - error-range bars
   - the flagging scatter (one subplot per row, counts per quadrant)
   - for `ordinary` only, MAE by hour
   - a category with no picks says so, as spec 09's `has_picks` does
9. Expander "Forecast explorer": the whole test window (actual, SMARD, chosen models) with a range
   slider and an error chart below on the same x axis. Presets: the last full week, spec 10's
   default zoom weeks, the best and the worst month.

**Page 2 — `pages/2_Risk_days.py`**

10. Tabs High and Low. Low has a `rolling` / `zero` basis toggle, and a `3h` toggle switches the
    day rule from `any`.
11. A plain-words scorecard per tab, e.g. "caught *h* of *n* risk days (SMARD *h′*), *f* false
    alarms (SMARD *f′*)", with recall and precision, built on the evaluable days as in spec 10 §3.
    The denominator always shows, and so does the small-count caveat when *n* is small.
12. Risk calendar as a Plotly grid of ISO weeks × weekdays, model above SMARD:
    - the outcome categories in `RISK_COLOR` / `OUTCOME_COLOR`
    - holidays outlined in `HOLIDAY_COLOR`, a dot for days the actual sustains for 3 h
    - hover text with the date, the outcome and each source's flagged hours
13. Zoom week: the three series against the day's threshold, with the flag strip below.
    - The week picker defaults to spec 10's zoom rules: high, the week with the most actual high
      days; low, the flagged holiday with the lowest day minimum.
14. Day-margin scatter: day extreme minus threshold, actual against forecast, quadrants as in spec
    10 §4.3 / 5.3.
15. Short findings text computed live: the weekday / weekend split of hits and misses, and the
    months of the high days.

**Restructure**

16. The existing pages move behind the new ones and are renamed:
    - `pages/1_EDA.py` → `pages/3_Background.py`
    - `pages/2_Model.py` → `pages/4_Method.py`, retitled "Method: risk definition", without the "no
      model of our own" banner and the hardcoded `(2019–2026)`
17. Home gets a one-sentence answer and a KPI row from `model_results.py`, plus a "Limits" section:
    - one test year
    - small high-risk counts
    - more false alarms than SMARD where the export shows it
    - the losing months
    - national data only
18. All findings text follows the `interpretation-style` skill: the number first, then why, then the
    caveat.

### Steps

| Step | Content | Status |
|---|---|---|
| 1 | Import fix | Done, committed (`3713c29`) |
| 2 | Plotly as a runtime dependency, colours and style helpers, `chart-style` skill | Done, not committed |
| 3 | `model_results.py` (Behaviour 1) | Written and checked on synthetic exports, not committed; the check against the real exports waits on Data's prerequisite |
| 4 | Page 1 (Behaviour 3–9) | Written and checked on synthetic exports, not committed; the explorer's spec 10 zoom-week presets were added in step 5 |
| 5 | Page 2 (Behaviour 10–15) | Written and checked on synthetic exports, not committed. The zoom rules, `persistent_hours`, `flagged_span` and the holidays live in `model_results.py`, shared with page 1's explorer. The day margin always uses the `any` rule, as in spec 10 |
| 6 | Restructure and text (Behaviour 16–18) | |

### Files

- New (3): `streamlit/model_results.py`, `streamlit/pages/1_Where_we_beat_SMARD.py`,
  `streamlit/pages/2_Risk_days.py`.
- Changed (8): `streamlit/streamlit_app.py`, `pages/1_EDA.py` and `pages/2_Model.py` (renamed),
  `streamlit/viz_helpers.py`, `pyproject.toml`, `uv.lock`, `.claude/skills/chart-style/SKILL.md`,
  this spec.
- Untouched: both visualization notebooks, every other spec output notebook, `CLAUDE.md` (until the
  team says so).

### Acceptance criteria

- [ ] Every page runs through `streamlit.testing.v1.AppTest` with no exception, including the
      "exports missing" path (a clear message, no traceback).
- [ ] `uv run streamlit run streamlit/streamlit_app.py`: zoom, hover, legend toggle, selectors and
      tabs work.
- [ ] On the 2026-09-30 snapshot, the app reproduces the notebooks:
  - Random forest hybrid MAE 2,333 against SMARD's 2,790 MWh (+16.4 %)
  - months won 9 / 10 / 10 of 11
  - high `rolling any`: 8 of 12 days against SMARD's 7
  - low `rolling any`: 35 of 57 against 30
  - 8,758 common hours (spec 09) and 8,759 (spec 10)
- [ ] No number, date, year, threshold or model key is hardcoded in the app; all come from the
      exports.
- [ ] No threshold or flag is recomputed; nothing from the parked specs 04.1 / 04.2 / 04.3 appears.

### Out of bounds

- Refitting models, new metrics or significance tests, ensembles.
- Prediction intervals (spec 06 §6 / §7.4 / §7.6), and reBAP costs: a later decision for the team.
- Editing the visualization notebooks or re-running any spec without a team member's confirmation.
