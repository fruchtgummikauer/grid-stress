# Streamlit v3 — App Structure and UX

- Status: **Run** (2026-10-07, PR #51, merged; final review, dark mode, reBAP compare 2026-10-08, §4 14–16).
  Planned with one sub-spec per page; instead every page was rebuilt directly after a page-by-page
  UX review for a public audience. §1 is the plan as reviewed, §2 what was built, §4 the changes.
- Builds on [Streamlit-draft.md](Streamlit-draft.md) (v1 §1–§16, PR #28; v2 §17, PR #43), which
  stays the record of what was built before. Where they disagree, this spec wins for v3 work.
- Audience: **bootcamp demo day** (instructors, peers, recruiters; mostly no energy background).
  A **5 min** live walkthrough, story first, detail on demand.

## 1. Decisions (team review, 2026-10-06)

| Topic | Decision |
|---|---|
| Pages | **6 pages + Home**, titles as short questions; Risk days a section of Do we beat SMARD?, its tools on Try it yourself (2026-10-07) |
| Background | stays the EDA page, with a context intro for non-experts |
| Method | tabs, no EDA: risk days, forecast setting, metrics, models, ensemble |
| Model sets | **spec 06 only**; the spec 06.2 `-magc` set is ignored |
| Navigation | `st.navigation` + `st.Page`, files in `streamlit/app_pages/` |
| Shared extras | demo-mode toggle, data check, light + dark theme; no per-chart source captions |
| Results shown | the current runs (spec 06, 08, viz-01/02/03); `-claude` conclusions are proposals |
| Units | MWh as everywhere (CLAUDE.md), no GW exception |

## 2. As built

Every page: interactive **Plotly** charts, every number in the text computed from the data or the
exports, takeaway first, technical detail in expanders, a "Next →" link along `TOUR`.

| # | Page | File | Takeaway (`header()`) |
|---|---|---|---|
| 1 | Home | `home.py` | "Forecasting tomorrow's pressure on Germany's power grid" (subtitle, no `header()`) |
| 2 | When is the grid under pressure? | `background.py` | "Before forecasting, we studied every hour since <first year>." |
| 3 | How do we forecast? | `method.py` | "We don't build a forecast from scratch — we learn where the official one is systematically wrong." |
| 4 | Do we beat SMARD? | `beat_smard.py` | "Our best model overall is <skill> % smarter than SMARD … Combining our models brings this to <ensemble> %." |
| 5 | Try it yourself | `try_it.py` | "Pick the forecasters, the day and the week: the whole test year is here." |
| 6 | What is it worth? | `rebap.py` | "A better forecast, priced in euros." |
| 7 | Who are we? | `about.py` | "Who we are, what we used, and where the code is." |

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
| ⑤ Combining models | the four methods in words (SMARD is a member); `rolling` weights: regime-weighted heatmap (regime × member), weighted bars, adaptive stacked over the test days | `ensemble_weights.csv` |
| ⑥ Models | stakeholder table: type, MAE, skill, months, coverage, best use (from the pick rules); "More columns" (RMSE, bias, per situation, width, vs best); static + naive behind a toggle; CSV download; **no speed column** | `load_accuracy()` |

- `CUTOFF_HOUR = 16` (2 h actuals lag, the team's current trial) is the page's only cutoff value.

### 2.4 Do we beat SMARD? and Try it yourself (`components/lineup.py` shared)

1. **The claim:** intro, link to How do we forecast?; **page-wide switch** "Compare models" (off =
   the three cases: best overall = lowest MAE of the `rolling` singles that beat SMARD, `LEAD`;
   best at too little / too much green power = spec 09 tail rank 1; on = multiselect of the
   `rolling` rows incl. the four ensembles, `static` behind a checkbox); "Which models?"; hero bars
   (MWh, "+x % smarter"); four tiles. Legends and menus: "Random forest hybrid (best overall)".
2. **Is it real?** "Is it luck?" monthly skill (won / lost bars; lines when comparing); "Where does
   the lead come from?" running total, biggest-gain month marked; "At what time of day?".
3. **When it matters most:** bars by situation, then tabs Normal hours / below zero / … most
   extreme (summary, "saw it coming / missed it / false alarm" counts, scatter and spread in
   expanders); **Our best at each extreme** (tiles); **risk days** (`risk_view.py`; hidden while
   `model_risk_labels_*.csv` is missing, an error when not from the model exports' run): tabs
   high / low, line, rule, scorecard, calendar; the pick rule. Hours: all 8,759 labelled ones.
4. **How sure are we?** `LEAD`'s band coverage (tiles; by situation and month vs the promised
   95 %; "How the range is built"); **One model or a team?** each `rolling` ensemble vs our best
   model per situation (computed, not `skill_vs_best_pct*`); Method ⑤.
5. **Wrap-up:** "What this means" (computed: euros and coverage included), the teaser box to
   "What is it worth?", monthly numbers for analysts.
6. **Try it yourself:** "Pick two forecasters" (A / B over SMARD + `rolling` rows; tiles, share of
   hours closer, a week with both bands, default largest gap; miss by situation); "The evening
   before" (suggested day or a date, "Reveal" draws the actual); "Explore the test year" with its
   own switch (presets: a week, best / worst week and month, spec 10's risk weeks, the year).
   `reference_toggles` (on): actual + SMARD (explorer), actual (A / B week), SMARD (evening before).

### 2.5 What is it worth? (spec 11)

"What is the reBAP?" (the German name, at most four sentences), then the simplifications box (one
price, amount not direction, hourly, yardstick not bill). Default: **our ensemble and our best model
overall**; "Compare all priced models" opens a multiselect of every priced row, all on
(`model_rebap_cost_hourly.csv`; never `rebap.csv`). "Calculate" reveals tiles (up to three
forecasts), saved per forecast, running total (three largest hours of the first), saved by
situation, an expander (formula, top-10 hours, every priced model). CSS-only **💸 rise** on the
click (3 s, off under reduced motion); `rebap_revealed` keeps the numbers. Teaser, summary: ensemble.

### 2.6 Who are we?

GitHub button; team cards (Robert, Marco, Hari, Monica in one row, Claude below; name, title,
one-liner in the team's words); project in numbers (computed); recap with links; project timeline
(`st.graphviz_chart`); data and tools; disclaimer; next steps; developer expander; data check
(hidden in demo mode); footer with attribution and licence links.

- SMARD data: **CC BY 4.0**, credited "Bundesnetzagentur | SMARD.de" (smard.de/en/datennutzung),
  with a note that derived values are ours. Code: link to the repo's `LICENSE` (MIT).

## 3. Architecture

```text
.streamlit/config.toml   # [theme.light] + [theme.dark], at the repo root
streamlit/
  streamlit_app.py       # entry: set_page_config, navigation from TOUR, sidebar(), run()
  app_pages/             # home, background, method, beat_smard, try_it, rebap, about
  components/            # layout.py (TOUR, title_of, sidebar, header, next_page), data_check.py,
                         # naming.py (SITUATION, label, label_in_text, colour, money), risk_view.py,
                         # lineup.py (Lineup, model_switch, reference_toggles, row_name)
  data_loading.py        # load_smard, get_years, load_risk_labels_daily
  model_results.py       # SOURCES, load_accuracy, load_risk_labels, risk_scores,
                         # load_rebap_cost, load_ensemble_weights, coverage, is_ensemble,
                         # model_windows (config.json), ensemble_ready
  viz_helpers.py         # palette, style_plotly, Plotly scales, DARK + themed(fig) / tone()
```

- Run `uv run streamlit run streamlit/streamlit_app.py` from the repo root.
- **State:** `demo_mode`; `{beat,explore}_{compare,ensemble,static,rows}`, `{h2h,eve,explore}_{actual,smard}`;
  `h2h_*`, `eve_*` / `eve_revealed`, `rebap_revealed` / `rebap_compare` / `rebap_rows`, `models_*`, `smard_miss_by`; risk view `rule_*` / `basis_*` / `zoom_*`.
- **Demo mode:** hides Plotly toolbars and the data check.
- **Words:** too little (high) / too much (low) green power, also in labels ("best with too much
  green power"), no coined terms; X.x % smarter than SMARD; saw it coming / missed it; "line"; million €; no P-notation,
  recall or precision. **Numbers:** percentages one decimal, MWh and counts whole. Page names: `title_of`.
- **Caching:** `@st.cache_data` keyed on the file's modification time. A missing export hides or
  explains only its section; `get_or_stop` only where a page can't work without the file.
- **Data check** (About): every file in `SOURCES` plus `smard.csv` and `risk_labels_daily.csv`,
  including the three `ensemble_*.csv` and `model_rebap_cost_hourly.csv`.
- **Style:** colours from `viz_helpers.py` (`chart-style` skill); SMARD dashed slate; high risk
  warm, low risk cool; a unit on every axis; notebook names only in captions and expanders.
- **Dark mode:** follows the system setting, ⋮ → Settings switches. Every chart is
  `plotly_chart(themed(fig))`; Graphviz uses `tone()`. Charts follow a switch at the next rerun.

## 4. Changes during the run

1. **No sub-specs.** Pages rebuilt one at a time from a UX review, each confirmed by the team.
2. **Plotly on every page**, matplotlib on none (the plan kept matplotlib on Background).
3. **Home:** fact tiles and the week picker instead of model KPIs; data check on Who are we?.
4. **Background:** chapters, technical figures in an expander; new negative-hours bar; mirrored
   bars and ramps by year replace the grouped bars and the ramp-percentile curve. "Falls year on
   year" became "has drifted down over the record" (matched-window medians are not monotonic).
5. **Method:** risk days read from the export (CLAUDE.md); SMARD on the test year; saved windows.
6. **Do we beat SMARD?:** one page-wide switch replaces five multiselects; counts replace the
   scatter as the default view; the week chart and the explorer merged; spec 09's pick rule.
7. **Ensemble** (after PR #51): see item 11; the placeholders (`ensemble_slot`) are gone.
8. **Not built:** `content/models.yaml` (one-liners live in `components/naming.py`).
9. **Who are we?** team names, titles, one-liners; SMARD attribution and licence after its terms.
10. **reBAP** (after PR #51): a section, then its own page (item 12); `load_rebap_cost` stops when
    the export is not from the model exports' run (hours, actuals, misses, cost = |miss| × price).
11. **Ensemble** (team, 2026-10-07): out of the headline; all four methods in the compare view.
    `load_accuracy` adds spec 08's rows (never candidates) only if their member rows equal
    `model_scoreboard.csv`. One maroon, told apart by `ENSEMBLE_MARK`; About: no longer dashed.
12. **Public round** (team, 2026-10-07): head-to-head and "the evening before" added; reBAP page
    (§2.5); Method ⑥ without a speed column (timings not comparable across singles and ensembles).
13. **Split** (2026-10-07, evening): Do we beat SMARD? = claim → is it real → when it matters →
    how sure → wrap-up; Try it yourself = head-to-head, evening before, explorer. Urls unchanged.
14. **Final review** (2026-10-08): labels model first; ensemble's skill in the headline, its weak
    spot in "What this means"; one decimal, plain words (§3 Words); `title_of`; risk view names as
    elsewhere; reference toggles; "Models compared" = 6 (no seasonal naive, as on Method).
15. **Dark mode** (2026-10-08): `[theme.dark]` (navy `#0F1B2D`, amber primary); `DARK` maps every
    light token (dataviz-validated on navy); heatmap scales stay light; navy text on amber nodes.
16. **reBAP compare** (team, 2026-10-08): ensemble + best model by default, a switch for every
    priced model (reverses item 12's ensemble only); the reBAP explained; labels without coined terms.

## 5. Next steps

| Step | Work | Needs |
|---|---|---|
| 1–4 | ~~Exports; ensemble blocks; reBAP page; Method ⑤ / ⑥~~ done (§4 items 10–12) | — |
| 5 | Docs: `chart-style` skill still says matplotlib on the EDA / Method pages (dark mode added); CLAUDE.md's Streamlit lines ("no trained model shown yet", import issue) | — |
| 6 | Rehearsal + `streamlit/DEMO.md` (run order, presets, a fallback screenshot per page) | all |

- **Follow-ups found in review:** region tints as hex strings in pages (`home.py` `GAP_TINT`,
  `method.py` `KNOWN_TINT` / `TARGET_TINT`, `background.py`) belong in `viz_helpers.py` (`DARK` maps them).
- **Team checks:** each card on Who are we?; Method ① wording ("refitted every week", "learns
  from the three years before"); `CUTOFF_HOUR` once the actuals lag is settled; `LICENSE` still
  names neuefische GmbH (template); whether to credit Mayank (initial commit, uv setup).
