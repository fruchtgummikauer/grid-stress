---
name: chart-style
description: This repo's chart color theme (the PowerRangers palette) and styling conventions for
  the Streamlit app. Use whenever creating or editing a chart in streamlit/streamlit_app.py,
  streamlit/pages/*.py, streamlit/app_pages/*.py or streamlit/viz_helpers.py, or when choosing
  colors for a new chart or the app theme.
---

Every chart in the Streamlit app shares one palette and one set of styling helpers, so the app
reads as a single, coherent document rather than a collection of ad hoc plots. The single source
of truth is `streamlit/viz_helpers.py`; the app theme in `.streamlit/config.toml` (repo root)
repeats its hexes. Never redefine a colour or a style rule inline in a page.

## Colour theme: the PowerRangers palette

Taken from the team logo (navy, amber, teal, periwinkle, slate) on 2026-10-06 and expanded with
deeper steps of the same hues, so lines and bars reach 3:1 on white. Every categorical set below
was checked with the dataviz skill's validator (`scripts/validate_palette.py`): lightness band,
chroma floor, colour-blind (protan / deutan) and normal-vision separation, contrast. Re-run it for
any new combination; don't judge colour-blind safety by eye.

**Text and surfaces**

| Token | Hex | Use |
|---|---|---|
| `INK` | `#1B3C65` | navy: text, headings, primary buttons, the actual / residual-load line (11.2:1) |
| `MUTED` | `#5C6275` | captions, axis labels, ticks (6.1:1) |
| `SURFACE` / `SURFACE_2` | `#FFFFFF` / `#F0F1F5` | page and chart background / cards, sidebar, diverging midpoint |
| `ACCENT` | `#E7A118` | logo amber: KPI highlights, amber text on navy (5.1:1). **Never text on white** (2.2:1) |
| `COLORS["grid"]` | `#E4E6EE` | gridlines, borders |
| link text | `#0B7F81` | dark teal (4.8:1), set in `config.toml` |

**Chart series: `PALETTE`, in this order**

| Slot | Hex | Hue |
|---|---|---|
| 1 | `#12989A` | teal |
| 2 | `#2E64B0` | blue |
| 3 | `#C27C00` | deep amber |
| 4 | `#9A5FA8` | plum |
| 5 | `#C8553D` | vermillion |
| 6 | `#5566C2` | indigo |
| (ensemble) | `#7A2A06` | maroon, `ENSEMBLE_COLOR`: reserved for spec 08's ensembles, never a single model |

- The **order is the safety mechanism**: neighbouring slots pass the colour-blind check (worst
  ΔE 16.0, target 8) and the normal-vision floor (worst 16.5, floor 15). Assign in order, never
  cycle, never generate a 7th hue: fold extra series into "Other" or small multiples.
- **Lines and bars** are checked on neighbouring pairs. **Scatter plots and small multiples**,
  where any two marks can meet, are safe for **slots 1–3 only**: cap them at three series.
- Blue (2) and indigo (6) are close relatives (normal-vision ΔE 5.3): that is why plum, not
  indigo, sits in slot 4. Never put blue and indigo next to each other in a legend order.
- **Documented exception:** teal (1) and plum (4) are ΔE 5.5 under deuteranopia (floor 6), though
  clearly apart for normal vision (18.4). Where only those two models share a chart (the Risk days
  page: `random_forest_hybrid` + `xgb_hybrid`), give `xgb_hybrid` a second cue: markers on its line
  and a direct label at the line end.
- **The ensemble** keeps a colour of its own, far from both XGBoost colours (colour-blind ΔE > 20)
  and from navy and slate.

**Series roles (EDA and method pages)**

| Role | Hex | Note |
|---|---|---|
| `residual_load` | `#1B3C65` navy | the darkest series, as in team-EDA |
| `grid_load` | `#C8553D` vermillion | the "hot" series |
| `wind_on` | `#12989A` teal | |
| `wind_off` | `#2E64B0` blue | |
| `solar` | `#C27C00` deep amber | |
| `renewables` | `#5566C2` indigo | not on the same axes as `wind_off` |
| weekday / weekend | blue / deep amber | the pair passes all-pairs |
| high risk / high tail | `#C8553D` vermillion | warm |
| low risk / low tail | `#2E64B0` blue | cool |
| reference lines, SMARD | `#6B6E80` slate | SMARD always dashed |

The rhythm chart's order (grid load, residual load, wind on, wind off, solar) passes the
neighbouring-pair check. The `COLORS` keys keep their old hue names (`"black"` is now navy,
`"sky_blue"` indigo, ...), so the pages need no edit; read the hex, not the key name.

**Scales**
- Sequential, one hue: `SEQUENTIAL` periwinkle → navy (`#99A8DE` … `#1B3C65`), `SEQ_CMAP`; the grid
  load heatmap uses the one-hue amber `GRID_LOAD_CMAP`.
- Diverging (data crossing zero): blue ↔ grey midpoint ↔ vermillion, `DIV_CMAP`; residual-load
  surfaces use `RESIDUAL_CMAP` via `residual_scale`, with zero white.

**Notebooks.** The notebooks keep their own colours (team-EDA, viz-01/02/03). They are shared,
team-edited files (CLAUDE.md): don't edit them to match. They take the palette only when the team
re-runs or edits them. Until then a model may look different in a notebook and in the app; what
must stay the same is the role each series plays and its warm/cool reading.

## Rules

- **Import colours and styles from `viz_helpers.py`, never hardcode a hex or a named colour
  (`"grey"`, `"black"`) in a page.** If a chart needs a colour the module doesn't have yet, add it
  to `viz_helpers.py` first (and to `config.toml` if it is a theme colour), validate any new
  categorical combination, then import it.
- **One series, one colour, everywhere it appears.** `grid_load` is always vermillion, on every
  page and every chart type. Reuse `series_style(col)` for line/bar plots so this is automatic.
- **Text wears text tokens** (`INK`, `MUTED`), never a series colour: values, labels and legends
  stay navy or slate; a coloured mark beside them carries the identity.
- **Time-indexed plots use `style_timeseries(ax, title, ylabel)`.** No-box, y-grid-only,
  year-major/quarter-minor ticks, thousands-separated labels.
- **Every chart states its unit in the axis label**, per this project's convention: `MWh` for
  levels, `MWh/day` for energy, `MW/h` for ramps, `%` for normalised metrics — ask before using
  any other phrasing (see CLAUDE.md's unit convention).
- **Diverging data (crosses zero) gets a diverging, zero-centred scale; one-sided data does not.**
  `residual_scale(vmin, vmax)` decides this automatically for `residual_load` surfaces.
- **A colour per year is sampled from a continuous colormap (`year_colors`), never a fixed
  year→colour table.** The record's span changes on every re-fetch.
- **Three libraries, each with its own pages:** matplotlib for the EDA and method pages, seaborn
  only for hue-faceted line plots there, and **Plotly for the model pages**. Don't introduce a
  fourth library for a page-specific chart.
- **Figures are closed after rendering** (`plt.close(fig)` after `st.pyplot(fig)`).

## Model pages (Plotly)

The pages that compare our forecasts with SMARD are interactive (zoom, pan, hover, legend
toggles), so they use Plotly (team decision, 2026-10-01).

- **`MODEL_STYLE`** gives each model key one colour from `PALETTE`, the same in both model sets
  (the `-magc` models use the same keys):

  | Model | Slot |
  |---|---|
  | `random_forest_hybrid` | 1 teal |
  | `lgbm_direct` | 2 blue |
  | `linear_direct` | 3 deep amber |
  | `xgb_hybrid` | 4 plum |
  | `lgbm_hybrid` | 5 vermillion |
  | `xgb_direct` | 6 indigo |
  | `ensemble_*` (all four methods) | maroon `ENSEMBLE_COLOR`, told apart by `ENSEMBLE_MARK` (bar pattern via `model_pattern`, marker via `model_symbol`) |
  | `actual` | navy `INK` |
  | `smard` | slate, dashed |
  | `seasonal_naive` | light slate `#A3A7B6` |

  The three overall picks (spec 09) take slots 1–3, which pass all-pairs; the risk picks
  (`xgb_hybrid`, `random_forest_hybrid`) are plum + teal, with the second cue above. A new model
  takes the next free slot, never a new hue; there is no free slot left, so a seventh model is a
  team decision (fold, facet or drop one).
- **`BIN_COLOR`** (residual-load bins) and **`RISK_COLOR` / `OUTCOME_COLOR` / `HOLIDAY_COLOR`**
  (risk labels) mark regions, thresholds and flags — never a model. Cool = low (light periwinkle,
  lavender), warm = high (amber bins, light vermillion `#E8B3A7` for the high-risk threshold). The
  risk tints are below 3:1 on white: label a threshold line, use them as fills.
- **Every Plotly figure goes through `style_plotly(fig, title, ylabel, xlabel=None)`**: white
  background, no box, light y grid, thousands separators, unified hover, legend below. `ylabel` is
  required (units as above). `viz_helpers.py` also registers the `powerrangers` Plotly template as
  the default, so a figure that skips `style_plotly` still gets the palette.
- **Lines use `model_line(key)` and `model_label(key)`**, so colour, dash and label come from one
  table. Colour follows the model, never its rank: a filter that drops models must not repaint
  the others.
- Render with `st.plotly_chart(themed(fig))`; its default `"streamlit"` theme takes the chart
  colours from `config.toml`, which equal `PALETTE`. No `plt.close` is needed.

## Dark mode

`config.toml` has `[theme.light]` and `[theme.dark]`: the app follows the visitor's system
setting, the ⋮ menu → Settings switches. Pages keep using the **light** tokens; `viz_helpers.DARK`
maps each one to its dark counterpart (navy `#0F1B2D` surface, `#E6EAF2` text, validated series).
- **Wrap every figure: `st.plotly_chart(themed(fig))`** (also `column.plotly_chart(...)`). An
  unwrapped chart keeps light colours on a dark page.
- Colours outside Plotly (Graphviz, HTML) go through `tone(token)`. Navy text on amber stays navy.
- Heatmap colour scales are not swapped (they carry their own legend).
- Streamlit reports a theme switch with the next rerun: charts follow at the next click.

## Adding a new colour or helper

1. Check whether `viz_helpers.py` already has it (a token, a `PALETTE` slot, a ramp step).
2. A new **series** takes the next free `PALETTE` slot. A new **region, threshold or flag** colour
   comes from the tints (`SURFACE_2`, lavender `#CCD3EB`, sage `#A1C8CA`) or the ramps, never a
   series hue.
3. Validate any new combination that shares a chart:
   `python3 <dataviz skill>/scripts/validate_palette.py "<hex,hex,...>" --mode light --surface "#FFFFFF"`
   (add `--pairs all` for scatter plots and small multiples).
4. Add it to `viz_helpers.py` (and `config.toml` for theme colours), not to the page that first
   needs it, plus its dark counterpart to `DARK` (validate with `--mode dark --surface "#0F1B2D"`).
