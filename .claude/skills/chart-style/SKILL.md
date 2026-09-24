---
name: chart-style
description: This repo's chart color theme and styling conventions for the Streamlit app.
  Use whenever creating or editing a chart in streamlit_app.py, pages/*.py, or viz_helpers.py.
---

Every chart in the Streamlit app shares one palette and one set of styling helpers, so the app
reads as a single, coherent document rather than a collection of ad hoc plots. The single source
of truth is `viz_helpers.py` at the repo root — never redefine a colour or a style rule inline in
a page.

## Colour theme

An Okabe-Ito-derived, colourblind-safe palette. It keeps the same **semantic roles** as
`team-EDA.ipynb` §1.2's `COLORS`/`SERIES_COLOR` (grid load is the "hot" colour, residual load is
near-black, wind/solar/renewables each get their own hue, high-risk is hot, low-risk is cool), but
swaps the exact hexes — most importantly the red/green pair, the most common colourblind
confusion — for ones that hold up under deuteranopia/protanopia and still contrast on white.

| Role | Colour | Hex |
|---|---|---|
| `grid_load` | vermillion | `#D55E00` |
| `residual_load` | near-black | `#1C1C1C` |
| `wind_on` | bluish-green | `#009E73` |
| `wind_off` | blue | `#0072B2` |
| `solar` | orange | `#E69F00` |
| `renewables` | sky blue | `#56B4E9` |
| high risk / high tail | vermillion | `#D55E00` |
| low risk / low tail | blue | `#0072B2` |
| muted / reference lines | grey | `#707B8C` |

This diverges intentionally from `team-EDA.ipynb`'s exact hexes. That notebook is a shared,
team-edited file (per CLAUDE.md) and is out of scope for this change — do not edit it to match.
The app and the notebooks are allowed to use different exact colours for the same role; what must
stay identical is which role each series plays and that role's relative warm/cool reading.

## Rules

- **Import colours and styles from `viz_helpers.py`, never hardcode a hex in a page.** If a chart
  needs a colour the module doesn't have yet, add it to `viz_helpers.py`'s `COLORS`/`SERIES_COLOR`
  first, then import it — don't inline `"#xxxxxx"` in `pages/*.py` or `streamlit_app.py`.
- **One series, one colour, everywhere it appears.** `grid_load` is always vermillion, on every
  page and every chart type (line, bar, heatmap). Reuse `series_style(col)` for line/bar plots so
  this is automatic.
- **Time-indexed plots use `style_timeseries(ax, title, ylabel)`.** No-box, y-grid-only,
  year-major/quarter-minor ticks, thousands-separated labels — never a custom grid/spine setup for
  a plot whose x-axis is a date.
- **Every chart states its unit in the axis label**, per this project's convention: `MWh` for
  levels, `MWh/day` for energy, `MW/h` for ramps, `%` for normalised metrics — ask before using
  any other phrasing (see CLAUDE.md's unit convention).
- **Diverging data (crosses zero) gets a diverging, zero-centred scale; one-sided data does not.**
  `residual_scale(vmin, vmax)` decides this automatically for `residual_load` surfaces — reuse it
  rather than picking a colormap by eye.
- **A colour per year is sampled from a continuous colormap (`year_colors`), never a fixed
  year→colour table.** The record's span changes on every re-fetch; a hardcoded mapping breaks
  silently when a year is added.
- **Matplotlib for styled/axis-precise charts, seaborn only for hue-faceted line plots**
  (`seasonal_plot`-style month-on-x, year-as-hue). Don't introduce a third plotting library for a
  page-specific chart — check `viz_helpers.py` and the source notebook for an existing helper
  first.
- **Figures are closed after rendering** (`plt.close(fig)` after `st.pyplot(fig)`), so repeated
  Streamlit reruns don't leak matplotlib figures.

## Adding a new colour or helper

1. Check whether the source notebook (`team-EDA.ipynb`, `risk-definition.ipynb`,
   `forecast-metrics-claude.ipynb`) already defines it — port the value, don't invent one.
2. Map it onto this palette's role table above if it's a new series/category; if the notebook's
   hex and this file's hex differ for the same role, use this file's (the colourblind-safe one)
   and note the substitution in the docstring, the way `viz_helpers.py` already does.
3. Add it to `viz_helpers.py`, not to the page that first needs it.
