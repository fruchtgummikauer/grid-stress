"""Home page — project introduction for a general audience.

Content is drawn from CLAUDE.md's project description and `risk-definition.ipynb` §1.7 (the
verified residual-load identity). The two Plotly charts are descriptive (a week of SMARD data the
visitor picks from the record's last month, and spec 02's exported risk days); no model findings, metrics or conclusions are stated here — those
live on the model pages, sourced from the notebooks that produced them.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.layout import TOUR, next_page
from data_loading import get_years, load_risk_labels_daily, load_smard
from model_results import is_set
from viz_helpers import (
    COLORS,
    INK,
    SERIES_COLOR,
    SERIES_LABEL,
    TAIL_COLOR,
    style_plotly,
    themed,
)

st.set_page_config(page_title="Grid Stress", page_icon="⚡", layout="wide")

# One line per page on the tour, keyed by its TOUR path (the list itself follows TOUR's order)
PAGE_BLURB = {
    "app_pages/background.py": "How electricity use, wind and solar change over the day, week and year.",
    "app_pages/method.py": "How we turn the official forecast into a better one, step by step.",
    "app_pages/beat_smard.py": "Where our forecast is more accurate than the official one — and where it isn't.",
    "app_pages/try_it.py": "Put any two forecasts head to head, or pick a day or a week yourself.",
    "app_pages/rebap.py": "What the smaller misses are worth, priced at the imbalance price.",
    "app_pages/about.py": "Who we are, the tools we used, and where to find the code.",
}

GAP_TINT = "#CCD3EB"  # lavender region tint (chart-style: regions take tints, never a series hue)

st.title("⚡ Grid Stress")
st.markdown("### Forecasting tomorrow's pressure on Germany's power grid")

st.markdown("""
Germany's power grid has to balance supply and demand every second. As wind and solar grow, that
balance swings harder: some hours there is too little green power and the country must import,
other hours there is so much that prices turn negative. We forecast these swings **one day ahead**
and spot the days when the grid is at risk.
""")

try:
    time_series = load_smard()
except (FileNotFoundError, RuntimeError) as exc:
    st.warning(f"SMARD data not available: {exc}")
    time_series = None

# --- Fact tiles -------------------------------------------------------------------------------
if time_series is not None:
    years = get_years(time_series)
    tiles = st.columns(3)
    tiles[0].metric(
        "Years of data",
        f"{years[0]} – {years[-1]}",
        help=f"Every hour since {time_series.index.min():%d %B %Y}, updated whenever we re-download "
        "the data.",
    )
    tiles[1].metric(
        "Hourly readings",
        f"{len(time_series):,}",
        help="One reading per hour of electricity use, wind and solar generation.",
    )
    tiles[2].metric(
        "Data up to",
        f"{time_series.index.max():%d %b %Y}",
        help="Source: SMARD, the public electricity-market data platform of Germany's Federal "
        "Network Agency (Bundesnetzagentur).",
    )

# --- Hero chart: one week of residual load ----------------------------------------------------
st.subheader("Residual load in one picture")

if time_series is not None:
    # The Mon–Sun weeks touching the record's last 30 days; the last one may still be partial.
    # Derived from the data, so the choice moves with every re-fetch.
    record_end = time_series.index.max()
    month_start = record_end - pd.Timedelta(days=30)
    first_monday = month_start.normalize() - pd.Timedelta(days=month_start.dayofweek)
    week_starts = list(pd.date_range(first_monday, record_end, freq="7D"))
    complete = [
        start
        for start in week_starts
        if start + pd.Timedelta(days=7, hours=-1) <= record_end
    ]

    def week_label(start):
        end = min(start + pd.Timedelta(days=6), record_end.normalize())
        partial = "" if start in complete else " (so far)"
        return f"{start:%d %b} – {end:%d %b}{partial}"

    week_start = st.radio(
        "Pick a week from the last month of data",
        week_starts,
        index=week_starts.index(complete[-1] if complete else week_starts[-1]),
        format_func=week_label,
        horizontal=True,
    )
    week = time_series.loc[week_start : week_start + pd.Timedelta(days=7, hours=-1)]

    stack_cols = ["wind_off", "wind_on", "solar"]
    residual = week["residual_load"]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.1,
        subplot_titles=(
            "What Germany used, and what wind and solar supplied",
            "The gap: residual load",
        ),
    )
    for col in stack_cols:
        fig.add_trace(
            go.Scatter(
                x=week.index,
                y=week[col],
                name=SERIES_LABEL[col],
                stackgroup="supply",
                line={"width": 0.5, "color": SERIES_COLOR[col]},
                fillcolor=SERIES_COLOR[col],
                hovertemplate="%{y:,.0f} MWh",
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=week.index,
            y=week["grid_load"],
            name="Electricity use (grid load)",
            line={"width": 2.4, "color": SERIES_COLOR["grid_load"]},
            hovertemplate="%{y:,.0f} MWh",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=week.index,
            y=residual.clip(upper=0),
            name="Below zero: more green power than the country uses",
            fill="tozeroy",
            fillcolor=GAP_TINT,
            line={"width": 0},
            hoverinfo="skip",
            showlegend=bool((residual < 0).any()),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=week.index,
            y=residual,
            name="Residual load",
            line={"width": 2.4, "color": SERIES_COLOR["residual_load"]},
            hovertemplate="%{y:,.0f} MWh",
        ),
        row=2,
        col=1,
    )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1}, row=2, col=1)
    style_plotly(fig, "", "MWh", height=600)
    fig.update_annotations(font={"color": INK, "size": 15})
    fig.update_xaxes(
        tickformat="%a %d %b", dtick=24 * 3600 * 1000, hoverformat="%a %d %b, %H:%M"
    )
    st.plotly_chart(themed(fig))

    st.caption(
        "Top: the red line is how much electricity Germany used; the coloured areas are what wind "
        "and solar supplied. The space between them is the **residual load** — the electricity "
        "that has to come from other power plants or from imports. Bottom: that gap on its own. "
        "When it goes below zero, there is more green power than the country can use. "
        "Hover for the exact numbers; drag to zoom, double-click to reset."
    )

st.latex(r"\text{Residual load} = \text{Electricity use} - \text{Wind} - \text{Solar}")

# --- The two risk cases -----------------------------------------------------------------------
st.subheader("Two ways the grid gets stressed")
high_card, low_card = st.columns(2)
with high_card.container(border=True):
    st.markdown("#### 🔺 Too little green power (high)")
    st.markdown(
        "Residual load **peaks**. Germany relies on imports and backup power plants, with little "
        "room for error."
    )
with low_card.container(border=True):
    st.markdown("#### 🔻 Too much green power (low)")
    st.markdown(
        "Residual load **drops below zero**. Prices can turn negative, and the grid operators "
        "have to tell plants to produce less at short notice."
    )
st.caption(
    "We judge both against Germany as a whole and against the past twelve months. A risk day is a "
    "warning sign, not proof that the grid operators had to step in."
)

try:
    labels = load_risk_labels_daily()
except FileNotFoundError as exc:
    st.info(f"Risk days not available yet: {exc}")
    labels = None

if labels is not None:
    high = is_set(labels["high_risk_rolling_any"])
    low = is_set(labels["low_risk_rolling_any"])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=labels.index,
            y=labels["residual_max"],
            name="Highest hour",
            line={"width": 0},
            showlegend=False,
            hovertemplate="Highest hour: %{y:,.0f} MWh<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=labels.index,
            y=labels["residual_min"],
            name="Daily range of residual load",
            fill="tonexty",
            fillcolor=GAP_TINT,
            line={"width": 0},
            hovertemplate="Lowest hour: %{y:,.0f} MWh<extra></extra>",
        )
    )
    for direction, flag, extreme in [
        ("high", high, "residual_max"),
        ("low", low, "residual_min"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=labels.index[flag],
                y=labels.loc[flag, extreme],
                mode="markers",
                name=f"Risk day: very {direction} ({int(flag.sum())})",
                marker={"size": 6, "color": TAIL_COLOR[direction]},
                hovertemplate=f"Risk day: very {direction}<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
    style_plotly(
        fig,
        f"Every day since {labels.index.min().year}: its range, and the days at risk",
        "MWh",
        height=480,
    )
    fig.update_xaxes(
        hoverformat="%a %d %b %Y",
        rangeselector={
            "buttons": [
                {
                    "count": 3,
                    "label": "3 months",
                    "step": "month",
                    "stepmode": "backward",
                },
                {"count": 1, "label": "1 year", "step": "year", "stepmode": "backward"},
                {"label": "All", "step": "all"},
            ],
            "x": 0,
            "y": 1.02,
        },
    )
    st.plotly_chart(themed(fig))
    st.caption(
        "The shaded band runs from each day's lowest to its highest hour. A dot marks a day with "
        "at least one hour among the most extreme of the previous twelve months (currently the "
        "top and bottom 1 %) — red for very high, blue for very low. The first year has no dots: it "
        "has no previous year to compare against. Use the buttons to zoom to the last months."
    )

# --- Benchmark --------------------------------------------------------------------------------
st.subheader("What we did")
st.markdown(
    "SMARD, the data platform of Germany's Federal Network Agency, publishes an official forecast "
    "for tomorrow. We don't start from scratch: we build on that official forecast and try to make "
    "it more accurate."
)
st.page_link(
    "app_pages/beat_smard.py", label="See how much more accurate: Do we beat SMARD? →"
)

# --- Tour -------------------------------------------------------------------------------------
st.subheader("Take the tour")
for path, title, icon, _ in TOUR[1:]:
    st.page_link(path, label=f"**{title}** — {PAGE_BLURB.get(path, '')}", icon=icon)

# --- Glossary and technical notes -------------------------------------------------------------
with st.expander("Words used in this app"):
    st.markdown("""
- **Residual load** — electricity use minus wind and solar generation: what other power plants
  or imports have to cover.
- **Grid operators (TSOs)** — the four companies that run Germany's high-voltage grid
  (50Hertz, Amprion, TenneT, TransnetBW).
- **SMARD** — the public electricity-market data platform of the Federal Network Agency
  (Bundesnetzagentur). It publishes both the measurements and the official forecast we compare
  against.
- **Day-ahead forecast** — a forecast made the day before, for every hour of the next day.
- **Redispatch** — grid operators telling power plants to produce more or less at short notice,
  to keep the grid stable.
- **Green shortage / green surplus** — short for *too little* / *too much green power*: the hours
  with the highest residual load, and the hours with the lowest (often below zero). A model labelled
  "best in green surplus" has the smallest misses in those hours.
- **Ensemble** — a forecast that combines several models (and SMARD's own forecast) as a weighted
  average.
""")

with st.expander("For the technically curious"):
    st.markdown("""
- The residual-load identity was verified against the loaded data in `risk-definition.ipynb`
  §1.7 (`residual_load == grid_load - wind_off - wind_on - solar`, to within rounding).
- **Scope:** we use wind and solar only. They swing the most, and they are the only sources with
  a public day-ahead forecast. Other generation (biomass, coal, hydro, …) is out of scope.
- **Risk days** come from `risk_labels_daily.csv` (`risk-definition.ipynb`): `rolling` basis
  (trailing 365-day quantile), `any` day rule, at the notebook's percentile level.
- **Data:** SMARD API, `data/smard.csv`, regenerated by `notebooks/API-connection.ipynb`. The
  data check (which files the app found) is on the Who are we? page.
""")

st.caption("A capstone project of the neuefische Data Science bootcamp.")

next_page("app_pages/home.py")
