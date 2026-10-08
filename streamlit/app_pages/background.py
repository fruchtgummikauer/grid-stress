"""Background page (the EDA) — sourced from `notebooks/01_eda/team-EDA.ipynb`.

`team-EDA.ipynb` is the primary, team-reviewed source for this page (Streamlit-draft.md §5 Page 2)
and stays read-only. The page retells its sections as four chapters for a general audience, with
interactive Plotly charts:

1. A typical day — §5.2 (daily rhythm, now per season) and §5.3 (holidays)
2. The year — §5.1 (calendar heatmap; residual load month × hour up front, the rest on demand)
3. The change — §6.5 (level by year) and the yearly count of negative hours in the same matched
   calendar window
4. The two extremes — §3.7 (calendar signature, as mirrored bars of hour counts)

§6.3 (distribution and ramps), §7.2 (correlation), §7.3 (load duration curves) and §2.1 (missing
values) sit in the closing "For the technically curious" expander. Conclusions keep team-EDA's
meaning, in plainer words; every number in the text is computed from the loaded data. §3.8
(worked extreme-episode examples) is not yet ported.
"""

import calendar

import holidays
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.layout import header, next_page
from data_loading import get_years, load_smard
from viz_helpers import (
    COLORS,
    DAY_TYPE_COLOR,
    DIV_SCALE,
    GRID_LOAD_SCALE,
    INK,
    RAMP_COLOR,
    SEASON_ORDER,
    SEQ_SCALE,
    SERIES_COLOR,
    TAIL_COLOR,
    residual_colorscale,
    style_plotly,
    themed,
)

st.set_page_config(
    page_title="When is the grid under pressure? — Grid Stress",
    page_icon="📊",
    layout="wide",
)

try:
    time_series = load_smard()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(f"SMARD data not available: {exc}")
    st.stop()

years = get_years(time_series)
residual = time_series["residual_load"]
record_end = time_series.index.max()

# Plain-language names for this page (the shared SERIES_LABEL says "Grid load")
PLAIN = {
    "grid_load": "Electricity use",
    "residual_load": "Residual load",
    "wind_on": "Onshore wind",
    "wind_off": "Offshore wind",
    "solar": "Solar",
}
MONTH_NAMES = list(calendar.month_abbr)[1:]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
HOUR_LABELS = [f"{h:02d}:00" for h in range(24)]

# Same calendar window in every year (1 Jan to the record's latest day), so the in-progress final
# year is compared like for like (team-EDA §6.5)
matched = time_series[
    (time_series.index.month < record_end.month)
    | (
        (time_series.index.month == record_end.month)
        & (time_series.index.day <= record_end.day)
    )
]
WINDOW = f"1 Jan – {record_end:%d %b} of each year"


def hover_closest(fig):
    """Per-point hover, for charts where one x position holds one value (heatmaps, bars)."""
    fig.update_layout(hovermode="closest")
    return fig


header(
    "When is the grid under pressure?",
    "Before forecasting, we studied every hour since " f"{time_series.index.min():%Y}.",
)
st.markdown(
    "Four things stand out: a daily rhythm set by work and sunlight, a summer *solar hole* at "
    "midday, more and more hours where wind and solar exceed demand, and two extremes with completely different "
    "causes. Every chart is interactive — hover for the numbers, drag to zoom, double-click to "
    "reset."
)

# =================================================================================================
# 1. A typical day (§5.2, §5.3)
# =================================================================================================
st.divider()
st.header("1 · What does a typical day look like?")
st.markdown(
    "**Electricity use follows the working day; solar carves a dip into residual load at midday.**"
)

season = st.radio(
    "Show the average day for",
    ["All year", *[s.capitalize() for s in SEASON_ORDER]],
    horizontal=True,
    help="Seasons are meteorological: winter = Dec–Feb, spring = Mar–May, and so on.",
)
in_season = (
    time_series
    if season == "All year"
    else time_series[time_series["season"] == season.lower()]
)
RHYTHM_COLS = ["grid_load", "residual_load", "wind_on", "wind_off", "solar"]
hourly_profile = in_season.groupby("hour")[RHYTHM_COLS].mean()

fig = go.Figure()
for col in RHYTHM_COLS:
    fig.add_trace(
        go.Scatter(
            x=HOUR_LABELS,
            y=hourly_profile[col],
            name=PLAIN[col],
            line={
                "color": SERIES_COLOR[col],
                "width": 2.6 if col in ("grid_load", "residual_load") else 1.8,
            },
            hovertemplate="%{y:,.0f} MWh",
        )
    )
fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
style_plotly(
    fig,
    f"The average day — {season.lower()}",
    "MWh (average hour)",
    xlabel="hour of day",
)
st.plotly_chart(themed(fig))
st.caption(
    "Each line is the average of that hour over every day in the selection. Switch between "
    "winter and summer: in summer, solar pushes midday residual load far down; in winter it "
    "barely dents it."
)
st.markdown("""
- Demand climbs from its overnight low to a daytime plateau.
- Solar rises and falls inside that plateau; wind is nearly flat across the day.
- Residual load is what's left over — the demand other power plants have to cover.
""")

st.subheader("Holidays behave like an extra Sunday")
de_holidays = holidays.country_holidays("DE", years=range(years[0], years[-1] + 1))
is_holiday = pd.Series(time_series.index.date, index=time_series.index).isin(
    set(de_holidays)
)
holiday_profile = (
    time_series.groupby([is_holiday.rename("is_holiday"), "hour"])["grid_load"]
    .mean()
    .unstack(0)
    .rename(columns={False: "Ordinary day", True: "National holiday"})
)
fig = go.Figure()
for name, color in [
    ("Ordinary day", DAY_TYPE_COLOR["weekday"]),
    ("National holiday", DAY_TYPE_COLOR["weekend"]),
]:
    fig.add_trace(
        go.Scatter(
            x=HOUR_LABELS,
            y=holiday_profile[name],
            name=name,
            line={"color": color, "width": 2.8},
            hovertemplate="%{y:,.0f} MWh",
        )
    )
style_plotly(
    fig,
    "Electricity use on national holidays",
    "MWh (average hour)",
    xlabel="hour of day",
    height=400,
)
st.plotly_chart(themed(fig))

peak_gap = holiday_profile["Ordinary day"] - holiday_profile["National holiday"]
holiday_count = int(is_holiday.groupby(time_series.index.date).any().sum())
st.markdown(f"""
- On holidays, daytime use is **lower and flatter**; the night-time level hardly changes (hospitals,
  telecoms and other base load don't take holidays).
- The biggest difference is **{peak_gap.max():,.0f} MWh at {peak_gap.idxmax():02d}:00**, across
  {holiday_count} federal holidays in the data.
- **Why it matters:** a forecast that doesn't know about holidays expects a normal workday's demand.
""")
st.caption("Federal holidays only (no regional ones).")

# =================================================================================================
# 2. The year (§5.1)
# =================================================================================================
st.divider()
st.header("2 · When in the year is the grid under pressure?")
st.markdown("**Sunlight, not the working day, shapes residual load.**")

HEATMAP_DIMENSIONS = {
    "Month": ("month", range(1, 13), MONTH_NAMES),
    "Weekday": ("dow", range(7), DAY_NAMES),
    "Season": ("season", SEASON_ORDER, [s.capitalize() for s in SEASON_ORDER]),
}


def calendar_heatmap(col, dimension, height=480):
    """Mean of `col` per (dimension, hour), coloured as in team-EDA §5.1."""
    key, order, labels = HEATMAP_DIMENSIONS[dimension]
    table = time_series.pivot_table(
        index=key, columns="hour", values=col, aggfunc="mean", observed=True
    )
    table = table.reindex(list(order))
    z = table.to_numpy()
    if col == "residual_load":
        scale = residual_colorscale(np.nanmin(z), np.nanmax(z))
    else:
        scale = {"colorscale": GRID_LOAD_SCALE}
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=HOUR_LABELS,
            y=labels,
            colorbar={"title": {"text": "MWh"}},
            hovertemplate="%{y}, %{x}: %{z:,.0f} MWh<extra></extra>",
            **scale,
        )
    )
    style_plotly(
        fig,
        f"{PLAIN[col]} — average by {dimension.lower()} and hour",
        dimension,
        xlabel="hour of day",
        height=height,
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    return hover_closest(fig)


st.plotly_chart(themed(calendar_heatmap("residual_load", "Month")))
st.caption(
    "Each square is the average residual load for that month and hour. Blue = little left for "
    "other power plants (much green power), red = a lot left to cover. Look for the pale-blue "
    "midday block from spring to late summer: that's solar."
)
st.markdown("""
- Electricity use is organised by the working day: high from 08:00 to 20:00 in every month and on
  every weekday, lower at weekends and in summer.
- Residual load is organised by **sunlight**: solar pushes it towards zero from March to September,
  with the working day only a secondary effect. That is why residual load, not electricity use, is
  what we forecast.
""")
with st.expander("More calendar views"):
    pick_series, pick_dimension = st.columns(2)
    heat_col = pick_series.radio(
        "Series", ["residual_load", "grid_load"], format_func=PLAIN.get, horizontal=True
    )
    heat_dim = pick_dimension.radio("By", list(HEATMAP_DIMENSIONS), horizontal=True)
    st.plotly_chart(themed(calendar_heatmap(heat_col, heat_dim, height=420)))

# =================================================================================================
# 3. The change (§6.5)
# =================================================================================================
st.divider()
st.header("3 · How is this changing?")
st.markdown(
    "**Hours with more green power than demand are multiplying — and the low end keeps stretching down.**"
)

negative_hours = (matched["residual_load"] < 0).groupby(matched.index.year).sum()
fig = go.Figure(
    go.Bar(
        x=negative_hours.index.astype(str),
        y=negative_hours,
        marker_color=TAIL_COLOR["low"],
        text=negative_hours,
        texttemplate="%{text:,}",
        textposition="outside",
        hovertemplate="%{x}: %{y:,} hours<extra></extra>",
    )
)
style_plotly(
    fig, f"Hours with more green power than demand ({WINDOW})", "hours", height=420
)
fig.update_yaxes(range=[0, negative_hours.max() * 1.15 if negative_hours.max() else 1])
st.plotly_chart(themed(hover_closest(fig)))
st.caption(
    f"Hours in which residual load was below zero. Every year is counted over the same window "
    f"({WINDOW}), so {years[-1]} — still in progress — is compared fairly."
)

annual_level = (
    matched["residual_load"]
    .groupby(matched.index.year)
    .agg(
        median="median",
        p10=lambda s: s.quantile(0.10),
        p90=lambda s: s.quantile(0.90),
        n="count",
    )
)
fig = go.Figure(
    go.Scatter(
        x=annual_level.index.astype(str),
        y=annual_level["median"],
        mode="markers",
        marker={"size": 13, "color": INK},
        name="Typical hour (median)",
        error_y={
            "type": "data",
            "symmetric": False,
            "thickness": 6,
            "width": 0,
            "color": "#A6B2E3",
            "array": annual_level["p90"] - annual_level["median"],
            "arrayminus": annual_level["median"] - annual_level["p10"],
        },
        customdata=annual_level[["p10", "p90", "n"]].to_numpy(),
        hovertemplate=(
            "%{x}<br>typical hour: %{y:,.0f} MWh<br>middle 80 %: %{customdata[0]:,.0f} to "
            "%{customdata[1]:,.0f} MWh<br>%{customdata[2]:,} hours<extra></extra>"
        ),
    )
)
fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
style_plotly(fig, f"Residual load by year ({WINDOW})", "MWh", height=440)
st.plotly_chart(themed(hover_closest(fig)))
st.caption(
    "The dot is a typical hour that year (the median); the bar covers the middle 80 % of hours."
)
st.markdown("""
- The typical hour has drifted down over the record, but the **low end falls much faster than the
  middle**: the
  distribution isn't sliding down as a block, it is **stretching downward**.
- Each year's low end reaches further towards, and eventually past, zero — while the high end
  barely moves.
- In short: hours with too much green power are growing; hours with too little stay where they were.
""")

# =================================================================================================
# 4. The two extremes (§3.7)
# =================================================================================================
st.divider()
st.header("4 · The two extremes are different problems")
st.markdown(
    "**Too much green power and too little happen at opposite times — almost a mirror image.**"
)
st.caption(
    "We took the 1 % of hours with the highest and the 1 % with the lowest residual load over the "
    "whole record and asked: when do they happen? (A simple look at the data, not how we define "
    "risk days.)"
)

n_tail = round(0.01 * len(time_series))
low_tail = residual.nsmallest(n_tail).index
high_tail = residual.nlargest(n_tail).index
LOW_NAME = "Lowest 1 % — too much green power"
HIGH_NAME = "Highest 1 % — too little green power"


def _nice_step(value):
    """A 1-2-5 step of roughly `value`."""
    magnitude = 10 ** np.floor(np.log10(value))
    return next(m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= value)


def mirrored_bars(extract, order, labels, title, axis_label, height):
    """Lowest-tail hours to the left, highest-tail hours to the right, per calendar slot."""
    low = pd.Series(extract(low_tail)).value_counts().reindex(list(order), fill_value=0)
    high = (
        pd.Series(extract(high_tail)).value_counts().reindex(list(order), fill_value=0)
    )
    fig = go.Figure()
    fig.add_bar(
        y=labels,
        x=-low,
        orientation="h",
        name=LOW_NAME,
        marker_color=TAIL_COLOR["low"],
        customdata=low,
        hovertemplate="%{y}: %{customdata:,} hours<extra>lowest 1 %</extra>",
    )
    fig.add_bar(
        y=labels,
        x=high,
        orientation="h",
        name=HIGH_NAME,
        marker_color=TAIL_COLOR["high"],
        hovertemplate="%{y}: %{x:,} hours<extra>highest 1 %</extra>",
    )
    style_plotly(fig, title, axis_label, xlabel="hours", height=height)
    step = _nice_step(max(low.max(), high.max()) / 2)
    ticks = np.arange(-2, 3) * step
    fig.update_layout(barmode="relative", bargap=0.15)
    fig.update_xaxes(
        tickvals=ticks,
        ticktext=[f"{abs(t):,.0f}" for t in ticks],
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=True,
        zerolinecolor=COLORS["muted"],
    )
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    return hover_closest(fig)


by_month, by_hour = st.columns(2)
by_month.plotly_chart(
    themed(
        mirrored_bars(
            lambda idx: idx.month, range(1, 13), MONTH_NAMES, "By month", "month", 520
        )
    )
)
by_hour.plotly_chart(
    themed(
        mirrored_bars(
            lambda idx: idx.hour, range(24), HOUR_LABELS, "By hour of day", "hour", 520
        )
    )
)
st.markdown("""
- **Too much green power** (blue, left): overwhelmingly recent, concentrated in the sunny months,
  around midday and at weekends.
- **Too little green power** (red, right): weekdays only, spread evenly across the years, in winter,
  peaking in the evening with a smaller morning peak.
- Too much green power is new and growing with the renewable build-out; too little green power
  is the classic, stable winter stress case.
""")
with st.expander("By year and by weekday"):
    by_year, by_weekday = st.columns(2)
    by_year.plotly_chart(
        themed(
            mirrored_bars(
                lambda idx: idx.year,
                years,
                [str(y) for y in years],
                "By year",
                "year",
                380,
            )
        )
    )
    by_weekday.plotly_chart(
        themed(
            mirrored_bars(
                lambda idx: idx.dayofweek,
                range(7),
                DAY_NAMES,
                "By weekday",
                "weekday",
                380,
            )
        )
    )

# =================================================================================================
# Three things to remember (team-EDA "Findings")
# =================================================================================================
st.divider()
negative_share = 100 * (residual < 0).mean()
with st.container(border=True):
    st.subheader("Three things to remember")
    st.markdown(f"""
1. **Two extremes, two problems.** Too much and too little green power differ in season, time of
   day, day type, trend and cause — they are not one phenomenon.
2. **Too much green power is growing.** Residual load's low end has fallen sharply while its high end
   has barely moved; it is already below zero in {negative_share:.1f} % of all hours.
3. **Sunlight shapes residual load.** Electricity use follows the working day; residual load follows
   the sun — and its hour-to-hour swings are getting bigger.
""")

# =================================================================================================
# For the technically curious (§6.3, §7.2, §7.3, §2.1)
# =================================================================================================
with st.expander("For the technically curious"):
    # --- §6.3 Distribution ---
    st.subheader("Distribution of hourly residual load")
    counts, edges = np.histogram(residual.dropna(), bins=120)
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(
        go.Bar(
            x=centers,
            y=counts,
            width=np.diff(edges),
            marker_color=[TAIL_COLOR["low"] if c < 0 else INK for c in centers],
            hovertemplate="around %{x:,.0f} MWh: %{y:,} hours<extra></extra>",
        )
    )
    for q in (0.01, 0.99):
        fig.add_vline(
            x=residual.quantile(q),
            line={"color": COLORS["muted"], "dash": "dash", "width": 1},
            annotation_text=f"{q:.0%}",
            annotation_font_color=COLORS["muted"],
        )
    fig.add_vline(
        x=residual.median(),
        line={"color": RAMP_COLOR, "width": 1.6},
        annotation_text="median",
        annotation_font_color=INK,
    )
    fig.add_vline(
        x=0,
        line={"color": COLORS["muted"], "width": 1.6},
        annotation_text="zero",
        annotation_font_color=INK,
    )
    style_plotly(
        fig,
        "How often each level of residual load occurs",
        "hours",
        xlabel="residual load (MWh)",
        height=420,
    )
    fig.update_layout(bargap=0)
    fig.update_xaxes(tickformat=",.0f")
    st.plotly_chart(themed(hover_closest(fig)))

    shape = pd.Series(
        {
            "mean (MWh)": residual.mean(),
            "median (MWh)": residual.median(),
            "standard deviation (MWh)": residual.std(),
            "skew": residual.skew(),
            "excess kurtosis": residual.kurt(),
            "minimum (MWh)": residual.min(),
            "maximum (MWh)": residual.max(),
            "range (MWh)": residual.max() - residual.min(),
            "hours below zero": int((residual < 0).sum()),
            "share of hours below zero (%)": negative_share,
        }
    )
    st.dataframe(shape.round(2).to_frame("residual load"))
    st.markdown(f"""
- Close to symmetric with a mild negative skew, spread over a range of
  **{(residual.max() - residual.min()):,.0f} MWh**; one peak, no sharp cut-off.
- The minimum sits further below the median than the maximum sits above it — the low tail is the
  longer one.
""")

    # --- §6.3 Ramps, by year ---
    st.subheader("One-hour swings (ramps), by year")
    ramp = residual.diff().mask(time_series["spans_gap"]).abs()
    matched_ramp = ramp.loc[matched.index].dropna()
    ramp_by_year = matched_ramp.groupby(matched_ramp.index.year).agg(
        median="median", p99=lambda s: s.quantile(0.99)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ramp_by_year.index.astype(str),
            y=ramp_by_year["p99"],
            name="Largest 1 % of swings",
            mode="lines+markers",
            line={"color": RAMP_COLOR, "width": 2.5},
            hovertemplate="%{y:,.0f} MW/h",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ramp_by_year.index.astype(str),
            y=ramp_by_year["median"],
            name="Typical swing (median)",
            mode="lines+markers",
            line={"color": COLORS["muted"], "width": 2, "dash": "dash"},
            hovertemplate="%{y:,.0f} MW/h",
        )
    )
    style_plotly(
        fig, f"Size of one-hour changes in residual load ({WINDOW})", "MW/h", height=400
    )
    fig.update_yaxes(rangemode="tozero")
    st.plotly_chart(themed(fig))
    st.caption(
        "A ramp is the change in residual load from one hour to the next, regardless of direction."
    )

    # --- §7.2 Correlation ---
    st.subheader("Which quantities move together?")
    correlation_frame = time_series[
        ["grid_load", "wind_on", "wind_off", "solar", "residual_load"]
    ].copy()
    correlation_frame["ws_share"] = time_series["renewables"] / time_series["grid_load"]

    fig = go.Figure(
        go.Histogram2d(
            x=100 * correlation_frame["ws_share"],
            y=residual,
            nbinsx=80,
            nbinsy=80,
            colorscale=SEQ_SCALE,
            colorbar={"title": {"text": "hours"}},
            hovertemplate="share %{x}<br>residual %{y}<br>%{z:,} hours<extra></extra>",
        )
    )
    style_plotly(
        fig,
        "The more of demand wind and solar cover, the lower residual load",
        "residual load (MWh)",
        xlabel="wind + solar as a share of electricity use (%)",
        height=460,
    )
    st.plotly_chart(themed(hover_closest(fig)))

    CORRELATION_LABELS = [
        "Electricity use",
        "Onshore wind",
        "Offshore wind",
        "Solar",
        "Residual load",
        "Wind+solar share",
    ]
    correlation = correlation_frame.corr(method="spearman")
    fig = go.Figure(
        go.Heatmap(
            z=correlation.to_numpy(),
            x=CORRELATION_LABELS,
            y=CORRELATION_LABELS,
            zmin=-1,
            zmax=1,
            colorscale=DIV_SCALE,
            text=correlation.to_numpy(),
            texttemplate="%{text:.2f}",
            colorbar={"title": {"text": "Spearman r"}},
            hovertemplate="%{y} vs %{x}: %{z:.2f}<extra></extra>",
        )
    )
    style_plotly(fig, "Rank correlation (Spearman)", "", height=480)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    st.plotly_chart(themed(hover_closest(fig)))
    st.markdown("""
- Residual load rises strongly with electricity use and falls strongly with the wind+solar share.
- Solar and wind are close to uncorrelated — independent inputs.
""")

    # --- §7.3 Load duration curves ---
    st.subheader("Load duration curves")
    positions = np.linspace(0, len(time_series) - 1, 1001).astype(int)
    share_of_hours = 100 * positions / (len(time_series) - 1)
    fig = go.Figure()
    for col in ["grid_load", "residual_load"]:
        ordered = time_series[col].sort_values(ascending=False).to_numpy()
        fig.add_trace(
            go.Scatter(
                x=share_of_hours,
                y=ordered[positions],
                name=PLAIN[col],
                line={"color": SERIES_COLOR[col], "width": 2.4},
                hovertemplate="%{y:,.0f} MWh",
            )
        )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
    fig.add_vline(
        x=100 - negative_share,
        line={"color": TAIL_COLOR["low"], "dash": "dot", "width": 1},
        annotation_text=f"below zero for {negative_share:.1f} % of hours",
        annotation_font_color=INK,
        annotation_position="bottom left",
    )
    style_plotly(
        fig,
        "Load duration curves over the full record",
        "MWh",
        xlabel="% of hours at or above this level",
        height=420,
    )
    fig.update_xaxes(hoverformat=".1f", ticksuffix=" %")
    st.plotly_chart(themed(fig))
    st.markdown("""
- The curves differ in **shape, not just level**: electricity use is flat through the middle and
  steep only at its extremes; residual load falls steadily across its whole range and dives through
  zero at the right end.
- Sorted over the whole multi-year record, so "% of hours" mixes several years' seasons together.
""")

    # --- §2.1 Missing values ---
    st.subheader("Data quality")
    series_cols = [
        "wind_off",
        "wind_on",
        "solar",
        "grid_load",
        "residual_load",
        "fc_gen_wind_solar",
        "fc_grid_load",
        "fc_residual_load",
        "cap_wind_off",
        "cap_wind_on",
        "cap_solar",
    ]
    missing = pd.DataFrame(
        {
            "missing values": time_series[series_cols].isna().sum(),
            "share (%)": (time_series[series_cols].isna().mean() * 100).round(3),
        }
    )
    st.dataframe(missing)
    st.caption(
        f"{int(missing['missing values'].sum())} missing values across {len(series_cols)} series, "
        f"{int(time_series.index.duplicated().sum())} duplicate timestamps."
    )

    st.markdown(
        "**Open question for the model:** whether the risk flag should treat the two extremes as one "
        "target or two, and whether the high extreme should be described by ramp rate rather than level."
    )

st.caption(
    "Source: `notebooks/01_eda/team-EDA.ipynb` (team-reviewed) — §2.1, §3.7, §5.1–5.3, §6.3, §6.5, §7.2, §7.3."
)

st.markdown(
    "**Next:** how we turn these patterns into a forecast for tomorrow, and how we test it fairly."
)
next_page("app_pages/background.py")
