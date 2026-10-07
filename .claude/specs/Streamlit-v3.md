# Streamlit v3 — App Structure and UX

- Status: **draft, reviewed** (2026-10-06), not run. Umbrella spec for the app restructure; each
  page gets its own sub-spec (`Streamlit-v3.<n>-<page>.md`, §4).
- Builds on [Streamlit-draft.md](Streamlit-draft.md) (v1 §1–§16, PR #28; v2 §17, PR #43), which
  stays the record of what was built. Where they disagree, this spec wins for v3 work.
- Audience: **bootcamp demo day** (instructors, peers, recruiters; mostly no energy background).
  A **5 min** live walkthrough, story first, detail on demand.
- Decisions (team review, 2026-10-06):

  | Topic | Decision |
  |---|---|
  | Pages | **5 pages**; no Explore page; Risk days removed (`risk_days.py` kept, off the navigation); reBAP and prediction bands are sections of "Where we beat SMARD" |
  | Background | stays the EDA page, with a short context intro for non-experts on top |
  | Method | tabs, no EDA: risk days, forecast setting, metrics, models, ensemble |
  | Model sets | **spec 06 only**; the spec 06.2 `-magc` set is ignored (no switch, no `model_magc_*.csv`) |
  | Navigation | `st.navigation` + `st.Page`, files in `streamlit/app_pages/` |
  | Shared extras | demo-mode toggle, data check on Home, theme file; **no per-chart source captions** |
  | Results shown | the current runs (spec 06, 08, viz-01/02/03), **labelled once** on Method → Models as "the reference runs of <date>" (`-claude` conclusions are proposals, CLAUDE.md) |
  | Units | MWh as everywhere (CLAUDE.md), no GW exception |

## 1. Pages

| # | Page | File | Takeaway (shown by `header()`) |
|---|---|---|---|
| 1 | **Home** | `home.py` | "We forecast Germany's residual load a day ahead and beat the official forecast by X %." |
| 2 | **Background** | `background.py` | "Wind and sun now drive the grid's swings, in a national rhythm; the extremes are where the grid is at risk." |
| 3 | **Method** | `method.py` | "A fair forecast setting and a strict benchmark." |
| 4 | **Where we beat SMARD** | `beat_smard.py` | "Better on average, best where SMARD is weakest, and worth € X M at the imbalance price." |
| 5 | **Team / About** | `about.py` | "Who we are, what we used, where the code is." |

- **Order:** the sidebar lists the pages in the table's order, with no section labels; every page
  ends with a "Next →" link along it. The demo walks the same order, story first.

### 1.1 Home

1. Title + subtitle: "Day-ahead forecast of Germany's residual load, against the official SMARD
   forecast."
2. **KPI row** (`st.metric` ×3, computed): our MAE vs SMARD's with skill %; months beating SMARD;
   skill on hours below 0 MWh.
3. "What is residual load?": three lines + `st.latex`, one typical week (grid load vs residual load).
4. **Tour:** `st.page_link` per page with icon and takeaway.
5. **Data check** (collapsed expander, §2.3). 6. Footer: data source (SMARD / Bundesnetzagentur)
   [ASSUMPTION: licence line to be checked], data range from `smard.csv`.

### 1.2 Background

1. **Context intro (new, Plotly):** residual load = grid load − wind − solar as one week; two cards,
   **high** (imports, Dunkelflaute) vs **negative** (oversupply, negative prices), each with a
   computed KPI (hours/year above P99 / below 0 MWh); the question in an `st.info`.
2. **EDA (existing matplotlib figures, all kept)**, at most 3 finding bullets each; correlation
   and autocorrelation go into an expander.
3. The negative-hour share is **computed once** (replaces the hardcoded 1.26 % and "about 2 %").

Data: `smard.csv` only; findings from `team-EDA.ipynb` (adopted).

### 1.3 Method

`st.tabs`, each with a 3-bullet summary on top and detail in expanders.

| Tab | Content | Source |
|---|---|---|
| Risk days | two directions, trailing 365-day 1 % quantile, `zero` basis for low, rules `any` / `3h`; today's threshold chart | `risk-definition.ipynb` (day rule open) |
| Forecast setting | timeline: issue 18:00 DAY−1, cutoff 16:00, capacity from 1 Jan; the leakage test in one sentence | spec 06 |
| Metrics | MAE, RMSE, bias, skill = 1 − MAE / MAE_SMARD, tail bins, why no MAPE, SMARD's own error level; reBAP pricing as text only (\|error\| × \|reBAP\|) | spec 04, spec 11 |
| Models | model cards (`streamlit/content/models.yaml`): name, one-line idea, inputs, direct vs hybrid; static vs rolling refit; tuning on the validation year; the reference-run line | `regression-models-claude` |
| Ensemble | members, four combination methods, `regime_weighted` weights (bar), viz-01's compute cost per pick | `ensemble-claude`, viz-01 |

### 1.4 Where we beat SMARD

The core result and the longest page; in demo mode blocks 5–8 collapse into expanders.

1. Headline + KPI row: best pick, the ensemble, SMARD. Below it, **risk days caught vs SMARD**
   (high `rolling` + low `zero`, rule `any`; viz-02's labels; `get_or_info` while they are missing).
2. **When:** monthly skill + cumulative lead (existing). 3. **Where in the day:** skill by hour
   (existing). 4. **At the extremes:** tabs `ordinary` / `below_zero` / `low_extreme` /
   `high_extreme` (existing).
5. **How sure are we (new):** one week with the 95 % band, coverage vs nominal, one sentence on
   the gap [ASSUMPTION: the hourly export carries `lower` / `upper`, otherwise spec 06 §8 adds them].
6. **What it's worth (new, viz-03):** tabs Overall / Below 0 / Low extreme / High extreme, mirroring
   block 4 (MWh, then €). Per tab: € saved and % vs SMARD, saved per price band, top 10 decisive
   hours; Overall also saved per month and cumulative. Scatter and money-vs-MAE-rank in an expander.
   Picks per tab are viz-03's by hand (`ensemble_regime_weighted` for overall and below 0,
   `random_forest_hybrid` low, `xgb_hybrid` high), one constant in `model_results.py`. Caveat on
   top: "a proxy: the reBAP prices the error, it is not what the TSOs paid." Reads
   `model_rebap_cost_hourly.csv` only, never raw `rebap.csv`.
7. **Explore (existing):** week and forecast explorers; presets best week, worst week, Easter 2026,
   longest negative run (computed); the week lives in `explore_week`.
8. **Ensemble vs single models (new, viz-01):** grouped skill bars per category, picks table below.

- The ensemble pick is one more line (`ENSEMBLE_COLOR`) in blocks 2–4, removable in the picker.

### 1.5 Risk days (removed)

Taken off the navigation on 2026-10-06 (team decision); `app_pages/risk_days.py` stays in the repo.
viz-02's risk labels still feed the zoom weeks on "Where we beat SMARD".

### 1.6 Team / About

Names and roles (placeholders for now), repo link, tools, data source, "what we'd do next".

### 1.7 Viz notebooks in the app

Every figure of viz-01/02/03 has one place ("exists" = in the app since v2):

| Notebook | Figures → page / block |
|---|---|
| viz-01 | §3.1 picks → beat SMARD 8; §3.2–3.3 scoreboards → "numbers behind it"; §3.4 compute cost → Method / Ensemble; §4.1–4.3, §5–8, §9.1 → beat SMARD 2–4 (exist, + ensemble line); §4.4 → explorer (exists) |
| viz-02 | §3 scores → beat SMARD 1 (risk days caught) and the zoom weeks; the rest not shown (§1.5) |
| viz-03 | §2.2 / §3.x / §4–6 → beat SMARD 6; §8.3 worked hour **left out** (needs raw `rebap.csv`) |

- **Producing the inputs:** `ensemble-claude` → viz-02 → viz-03 in that order, each executed as a
  copy outside the repo (`nbconvert --output-dir`), so only `data/` changes.
- **Snapshot rule:** `ensemble-claude` stops when the spec 06 saves don't match the local data. On
  2026-10-06 the saves (PR #38) were refitted locally; committing them is a team decision.

## 2. Architecture

### 2.1 Files

```text
.streamlit/config.toml   # theme, at the repo root (read from the directory the command runs in)
streamlit/
  streamlit_app.py       # entry: set_page_config, sidebar(), st.navigation(...).run()
  app_pages/             # page scripts ("pages/" would trigger the old folder navigation)
  components/            # layout.py (TOUR, sidebar, header, next_page), data_check.py; kpis.py, charts.py later
  data_loading.py, model_results.py, viz_helpers.py
  content/models.yaml    # model cards
```

Run `uv run streamlit run streamlit/streamlit_app.py` from the repo root; bare imports stay.

### 2.2 State

| Key | Set by | Default |
|---|---|---|
| `demo_mode` | sidebar toggle | `False` |
| `compare_models` | "compare all models" multiselects (beat SMARD) | the spec 09 picks |
| `explore_week` | the beat SMARD explorer | last full week |

### 2.3 Shared components

- **Sidebar:** demo-mode toggle, "data as of" (record end of `smard.csv`), repo link.
- **Demo mode:** hides detail expanders, collapses beat SMARD's blocks 5–8, hides Plotly toolbars.
- **`header(title, takeaway)`**, **`next_page(page)`**, and **`data_check()`** on Home (every file
  the app reads: found / missing, modified, producing notebook; from `model_results.SOURCES`).

### 2.4 Caching and missing files

- `@st.cache_data` keyed on the file's modification time, so a re-export shows without "Clear cache".
- Cache derived tables (`rolling_threshold`, Background pivots, `risk_scores`, per-bin quantiles),
  not figures; the pages only plot. `usecols=` for the 14 MB hourly export. No `cache_resource`.
- A missing export stops **only its section** (`get_or_info` → `st.info("Run <notebook> ...")`);
  `get_or_stop` only for files a whole page needs (`smard.csv`).

### 2.5 Style and content rules

- Colours only from `viz_helpers.py` (PowerRangers palette, `chart-style` skill); one series = one
  colour everywhere; SMARD dashed slate; high risk warm, low risk cool. Theme: `base = "light"`,
  navy primary, no custom CSS.
- Plotly on the model pages and the context intro (`style_plotly`); matplotlib stays on the EDA.
- Every number computed (f-strings over the data); a unit on every axis; at most 3 bullets per
  block (`interpretation-style`); notebook names stay out of the UI; units per CLAUDE.md.

## 3. Next steps

| Step | Work | Sub-spec | Needs |
|---|---|---|---|
| 0 | Produce the missing exports: `ensemble_*.csv`, `model_risk_labels_*.csv`, `model_rebap_cost_hourly.csv` (§1.7) | — | about 1 h compute |
| 1 | Skeleton: navigation, layout, sidebar, theme, data check, fallbacks, mtime caching. **Built 2026-10-06, not committed** | — | — |
| 2 | Home (KPI row, tour, computed text) | `Streamlit-v3.2-home.md` | 1 |
| 3 | Where we beat SMARD (bands, reBAP, ensemble) | `Streamlit-v3.3-beat-smard.md` | 0, 1 |
| 5 | Background (intro, findings, computed shares) | `Streamlit-v3.5-background.md` | 1 |
| 6 | Method (tabs, model cards, weights) | `Streamlit-v3.6-method.md` | 1 |
| 7 | Team / About | `Streamlit-v3.7-about.md` | names and roles |
| 8 | Rehearsal + `streamlit/DEMO.md` (run order, presets, a fallback screenshot per page) | — | all |

- Steps 2–3 first: Home and the core result. Each sub-spec runs with `spec-run-section-loop`.
  Branch per step [ASSUMPTION: `feature/streamlit-v3-<step>`].
- **Step 1 as built:** `layout.py` holds `TOUR`, `sidebar()`, `header()`, `next_page()`; a
  model-set radio was built and removed again when `-magc` was dropped; the risk calendar still shows
  the Plotly toolbar in demo mode; derived-table caching comes with each page's rewrite.
- **Sub-spec sections:** Goal (takeaway verbatim) · Content blocks (component, data, computed values)
  · Interaction and state · Demo mode · Missing data (per block) · Text (headings, bullets,
  f-string placeholders) · Performance · Acceptance criteria (renders with and without each export,
  demo mode, no hardcoded numbers, units, colours from `viz_helpers.py`, "Next →") · Out of bounds.
- **Follow-ups for the team:** CLAUDE.md's "no trained model shown yet" and the import bug are out
  of date; the `chart-style` skill places `viz_helpers.py` at the repo root (it is in `streamlit/`).
