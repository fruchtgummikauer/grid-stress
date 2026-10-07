# Streamlit v3 — App Structure and UX

- Status: **Run** (2026-10-07, PR #51, merged). Planned as an umbrella spec with one sub-spec per
  page; instead every page was rebuilt directly after a page-by-page UX review for a public
  audience. §1 is the plan as reviewed, §2 what was built, §3 the changes during the run.
- Builds on [Streamlit-draft.md](Streamlit-draft.md) (v1 §1–§16, PR #28; v2 §17, PR #43), which
  stays the record of what was built before. Where they disagree, this spec wins for v3 work.
- Audience: **bootcamp demo day** (instructors, peers, recruiters; mostly no energy background).
  A **5 min** live walkthrough, story first, detail on demand.

## 1. Decisions (team review, 2026-10-06)

| Topic | Decision |
|---|---|
| Pages | **5 pages**; no Explore page; Risk days removed (`risk_days.py` kept, off the navigation) |
| Background | stays the EDA page, with a context intro for non-experts |
| Method | tabs, no EDA: risk days, forecast setting, metrics, models, ensemble |
| Model sets | **spec 06 only**; the spec 06.2 `-magc` set is ignored |
| Navigation | `st.navigation` + `st.Page`, files in `streamlit/app_pages/` |
| Shared extras | demo-mode toggle, data check, theme file; no per-chart source captions |
| Results shown | the current runs (spec 06, 08, viz-01/02/03); `-claude` conclusions are proposals |
| Units | MWh as everywhere (CLAUDE.md), no GW exception |

## 2. As built

Every page: interactive **Plotly** charts, every number in the text computed from the data or the
exports, takeaway first, technical detail in expanders, a "Next →" link along `TOUR`.

| # | Page | File | Takeaway (`header()`) |
|---|---|---|---|
| 1 | Home | `home.py` | "Forecasting tomorrow's pressure on Germany's power grid" (subtitle, no `header()`) |
| 2 | Background | `background.py` | "Before forecasting, we studied every hour since <first year>." |
| 3 | Method | `method.py` | "We don't build a forecast from scratch — we learn where the official one is systematically wrong." |
| 4 | Where we beat SMARD | `beat_smard.py` | "Our forecast misses <skill> % less than the official one." |
| 5 | Team / About | `about.py` | "Who we are, what we used, and where the code is." |

### 2.1 Home

Hook paragraph → fact tiles (years, hourly readings, data up to) → **week chart**: wind + solar
stacked under electricity use, residual load below with the below-zero part shaded; a radio picks
any Mon–Sun week touching the record's last 30 days (default: the last complete one) → formula →
two risk cards → **risk-days chart** from `risk_labels_daily.csv` (daily range band, `rolling` /
`any` flags as dots; empty flags = no dot) → "What we did" with a link → tour from `TOUR` →
glossary and "For the technically curious" expanders.

### 2.2 Background

Four chapters, each with a bold takeaway, a chart and at most three bullets:

1. **A typical day:** hourly profile with a season radio (all year / four seasons); holidays.
2. **The year:** residual load month × hour heatmap (`residual_colorscale`); the other five
   views (series × month / weekday / season) behind radios in an expander.
3. **The change:** negative hours per year and residual load by year (median, P10–P90), both on
   the matched window 1 Jan → record end.
4. **The two extremes:** mirrored bars of the lowest / highest 1 % of hours by month and by hour
   (year and weekday in an expander).

Then "Three things to remember" (team-EDA's findings, plain words) and "For the technically
curious": distribution with shape table, one-hour ramps by year (median and P99), wind+solar share
vs residual load density, Spearman matrix, load duration curves, missing values, the open question.

### 2.3 Method

Intro, then `st.tabs`; a "What this can't tell us" `st.info` below the tabs on every tab.

| Tab | Content | Source |
|---|---|---|
| ① What we forecast | timeline (actuals known to `CUTOFF_HOUR`, issue 18:00, the 24 forecast hours); input flow (`st.graphviz_chart`, spec 06 Behaviour 18 default groups); validation / test years | spec 06; windows via `model_windows()` |
| ② The official forecast | SMARD on the **test year**: one-day chart with date picker (default SMARD's worst day), nMAE per component, miss by hour / month; full tables incl. whole record in an expander | `smard_forecast_errors_hourly.csv` |
| ③ How we judge "better" | MAE and skill in words, a skill example (labelled as illustration), test-year histogram with the scoring bins; RMSE, bias, no MAPE, reBAP proxy in an expander | `load_accuracy()` |
| ④ Risk days | radios for day rule (`any` / `3h`) and low basis (`rolling` / `zero`); daily range + thresholds + flagged days; share of evaluable days flagged per year; `static` in an expander | `risk_labels_daily.csv`, never recomputed |

- `CUTOFF_HOUR = 16` (2 h actuals lag, the team's current trial) is the page's only cutoff value.
- Models and Ensemble tabs: **not built** (§4).

### 2.4 Where we beat SMARD

1. **Headline and proof:** intro (we improve SMARD, not rival it), link to Method; hero bars
   (average miss, SMARD vs ours); four tiles; ensemble placeholder; **risk days caught / missed /
   false alarms** vs SMARD (hidden while `model_risk_labels_*.csv` is missing).
2. **Page-wide switch:** "Compare models" toggle; off = "Our forecast" (spec 09's overall rank 1)
   vs SMARD; on = multiselect of the `rolling` rows, `static` variants behind a checkbox.
   "Which models?" expander: one line per family, and why the headline pick isn't the lowest MAE.
3. **"Is it luck?"** monthly skill (won / lost bars; lines when comparing); **"Where does the lead
   come from?"** running total with the biggest-gain month marked; **"At what time of day?"**.
4. **"When it matters most":** overview bars by situation, then tabs Normal hours / Green-power
   surplus / Most extreme surplus / Most extreme shortage: one-line summary, "saw it coming /
   missed it / false alarm" counts (normal hours: miss by hour); scatter and spread in expanders;
   the pick rule in its own expander.
5. **"Explore the test year":** one explorer; presets a week of your choice, best / worst week,
   best / worst month, spec 10's risk weeks, the whole year; "who was closer" bars for one model.
6. **"Coming next":** ensemble placeholders "How sure are we?" and "One model or a team of
   models?"; monthly numbers for analysts; closing "What this means" (computed).

### 2.5 Team / About

GitHub button; team cards (Robert, Marco, Hari, Monica in one row, Claude below; name, title,
one-liner in the team's words); project in numbers (computed); recap with links; project timeline
(`st.graphviz_chart`, ensemble and bands dashed); data and tools; disclaimer; next steps; developer
expander; data check (hidden in demo mode); footer with attribution and licence links.

- SMARD data: **CC BY 4.0**, credited "Bundesnetzagentur | SMARD.de" (smard.de/en/datennutzung),
  with a note that derived values are ours. Code: link to the repo's `LICENSE` (MIT).

## 3. Architecture

```text
.streamlit/config.toml   # theme, at the repo root
streamlit/
  streamlit_app.py       # entry: set_page_config, navigation from TOUR, sidebar(), run()
  app_pages/             # home, background, method, beat_smard, about (+ risk_days, off the tour)
  components/            # layout.py (TOUR, sidebar, header, next_page, demo_mode), data_check.py
  data_loading.py        # load_smard, get_years, load_risk_labels_daily
  model_results.py       # SOURCES, load_accuracy, load_risk_labels, risk_scores,
                         # model_windows (config.json), ensemble_ready
  viz_helpers.py         # palette, style_plotly, Plotly scales (DIV_SCALE, GRID_LOAD_SCALE,
                         # SEQ_SCALE, residual_colorscale)
```

- Run `uv run streamlit run streamlit/streamlit_app.py` from the repo root.
- **State:** `demo_mode` (sidebar), `rows` (beat SMARD's compare multiselect), `smard_miss_by`
  (Method ②). Other widgets keep Streamlit's default keys.
- **Demo mode:** hides Plotly toolbars and the data check.
- **Caching:** `@st.cache_data` keyed on the file's modification time. A missing export hides or
  explains only its section; `get_or_stop` only where a page can't work without the file.
- **Data check** (About): every file in `SOURCES` plus `smard.csv` and `risk_labels_daily.csv`,
  including the three `ensemble_*.csv`.
- **Style:** colours from `viz_helpers.py` (`chart-style` skill); SMARD dashed slate; high risk
  warm, low risk cool; a unit on every axis; notebook names only in captions and expanders.

## 4. Changes during the run

1. **No sub-specs.** The pages were rebuilt directly from a UX review (comprehension,
   visualisations, text), one page at a time, each confirmed by the team before applying.
2. **Plotly on every page**, matplotlib on none (the plan kept matplotlib on Background).
3. **Home:** no KPI row of model results (they live on "Where we beat SMARD"); fact tiles and the
   week picker instead. The data check moved to Team / About.
4. **Background:** not "all figures kept": restructured into chapters, technical figures in an
   expander; new negative-hours bar; mirrored bars replace the 2 × 2 grouped bars; ramps by year
   replace the ramp-percentile curve. Wording: "the typical hour falls year on year" became
   "has drifted down over the record" (the matched-window medians are not monotonic;
   team-EDA keeps the old wording).
5. **Method:** risk days read the export instead of recomputing thresholds (CLAUDE.md rule);
   SMARD scored on the test year, not the whole record; windows from the model saves.
6. **Where we beat SMARD:** one page-wide switch replaces five "Compare models" multiselects;
   situation tabs renamed; counts replace the scatter as the default view; the week chart and the
   explorer merged. The headline keeps spec 09's pick rule and says so.
7. **Ensemble:** placeholders only (`ensemble_slot`); `ensemble_ready()` changes their wording once
   `ensemble_forecast_errors_hourly.csv`, `ensemble_scoreboard.csv`, `ensemble_weights.csv` exist.
8. **Not built:** bands with coverage (block 5), reBAP section (block 6), Method's Models and
   Ensemble tabs, `content/models.yaml` (model one-liners live in `MODEL_IDEA`, `beat_smard.py`).
9. **Team / About** built with the team's names, titles and one-liners; SMARD attribution and
   licence links added after checking SMARD's terms.

## 5. Next steps

| Step | Work | Needs |
|---|---|---|
| 1 | Produce the missing exports: `model_risk_labels_*.csv` (viz-02), `ensemble_*.csv` (spec 08), `model_rebap_cost_hourly.csv` (viz-03) | about 1 h compute |
| 2 | Ensemble blocks: hero bar, situation bar, bands and coverage, ensemble vs single models | step 1 |
| 3 | reBAP section on "Where we beat SMARD" (viz-03, `model_rebap_cost_hourly.csv` only) | step 1 |
| 4 | Method: Models tab (model cards) and Ensemble tab (members, weights) | step 1 for weights |
| 5 | Docs: `chart-style` skill (Plotly on every page, the Plotly scales); CLAUDE.md's Streamlit lines ("no trained model shown yet", import issue) | — |
| 6 | Rehearsal + `streamlit/DEMO.md` (run order, presets, a fallback screenshot per page) | all |

- **Follow-ups found in review:** three region tints are hex strings in pages (`home.py`
  `GAP_TINT`, `method.py` `KNOWN_TINT` / `TARGET_TINT`, one in `background.py`); they belong in
  `viz_helpers.py` per the `chart-style` skill.
- **Team checks:** each card on Team / About; Method ① wording ("refitted every week", "learns
  from the three years before"); `CUTOFF_HOUR` once the actuals lag is settled; `LICENSE` still
  names neuefische GmbH (template); whether to credit Mayank (initial commit, uv setup).
