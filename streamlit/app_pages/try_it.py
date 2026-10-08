"""Try it yourself — the interactive half of "Do we beat SMARD?" (split off 2026-10-07).

Put any two forecasters head to head, see what we would have said the evening before, and explore
any week of the test year (spec 09 §4.4 and the explorer). Every number is computed from the
exports through `model_results.py`; the line-up and the "Compare models" switch live in
`components/lineup.py`, shared with "Do we beat SMARD?".
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.layout import header, next_page
from components.lineup import (
    DIRECTION_NAME,
    OURS_NAME,
    line,
    load_lineup,
    model_switch,
    reference_toggles,
)
from components.lineup import full_weeks as test_weeks
from components.lineup import row_name as lineup_row_name
from components.lineup import win_loss_bars as lineup_win_loss_bars
from components.naming import SITUATION, SMARD_NAME, bar_style, color, label
from model_results import (
    BIN_CATEGORIES,
    SMARD_ROW,
    coverage,
    default_zoom_weeks,
    load_risk_labels,
)
from viz_helpers import model_line, style_plotly, themed

st.set_page_config(
    page_title="Try it yourself — Grid Stress", page_icon="🎛️", layout="wide"
)

LINEUP = load_lineup()
acc = LINEUP.acc
LEAD, SMARD_ABS, PUBLIC_ROWS = LINEUP.lead, LINEUP.smard_abs, LINEUP.public_rows
public_label, band_trace = LINEUP.public_label, LINEUP.band_trace
months = len(acc.full_months)
comparing = False  # until the explorer's own switch


def row_name(row):
    """Legend name: model and role tag of a headline row in the default view, else the label."""
    return lineup_row_name(LINEUP, row, comparing)


def win_loss_bars(x, values, row, hover, showlegend=False):
    """Bars in the row's colour where it beats SMARD, slate where SMARD is closer."""
    return lineup_win_loss_bars(x, values, row, row_name(row), hover, showlegend)


header(
    "Try it yourself",
    "Pick the forecasters, the day and the week: the whole test year is here.",
)
st.markdown(
    f"Everything below uses the test year ({LINEUP.test_label}): hours our models never saw while "
    "being built, each compared with SMARD's official day-ahead forecast."
)

try:
    risk_labels = load_risk_labels()
except FileNotFoundError:
    risk_labels = None  # only the explorer's risk-week presets need it


# ============================================================================
# Pick two forecasters (head-to-head, next to the "Compare models" switch; team 2026-10-07)
# ============================================================================
def forecaster_tiles(column, row, other):
    """Four tiles for one side of the head-to-head."""
    column.markdown(f"**{public_label(row)}**")
    mae = acc.value.loc[row, "MAE"]
    column.metric(
        "Average miss per hour",
        f"{mae:,.0f} MWh",
        f"{mae - acc.value.loc[other, 'MAE']:+,.0f} MWh vs the other",
        delta_color="inverse",
    )
    if row == SMARD_ROW:
        column.metric("Smarter than SMARD", "the benchmark")
        column.metric("95 % range holds", "no range published")
    else:
        column.metric(
            "Smarter than SMARD",
            f"{acc.value.loc[row, 'skill_pct']:+.1f} %",
            f"ahead in {acc.value.loc[row, 'months_beating_smard']:.0f} of {months} months",
            delta_color="off",
        )
        column.metric("95 % range holds", f"{coverage(acc, row):.1f} % of hours")


def head_to_head(a, b):
    """Tiles, who-was-closer shares, a week chart and the situations for two rows."""
    abs_a, abs_b = acc.errors[a].abs(), acc.errors[b].abs()
    a_wins, b_wins = float((abs_a < abs_b).mean()), float((abs_b < abs_a).mean())
    side_a, middle, side_b = st.columns([2, 1.4, 2])
    forecaster_tiles(side_a, a, b)
    forecaster_tiles(side_b, b, a)
    with middle:
        st.markdown("**Who was closer?**")
        st.metric("Hours A was closer", f"{100 * a_wins:.1f} %")
        st.metric("Hours B was closer", f"{100 * b_wins:.1f} %")
        st.caption(
            f"of {len(acc.common):,} test hours (ties: {100 * (1 - a_wins - b_wins):.1f} %)"
        )

    weeks = test_weeks(acc)
    gap = {
        monday: float(
            (abs_a - abs_b)[monday : monday + pd.Timedelta(days=7, hours=-1)]
            .abs()
            .sum()
        )
        for monday in weeks
    }
    monday = st.select_slider(
        "Week (default: the week where the two differ most)",
        options=weeks,
        value=max(gap, key=gap.get),
        format_func=lambda d: f"{d:%d %b %Y}",
        key="h2h_week",
    )
    hours = acc.common[
        (acc.common >= monday) & (acc.common < monday + pd.Timedelta(days=7))
    ]
    # SMARD can be A or B here, so only the actual line gets a switch
    show_actual, _ = reference_toggles("h2h", smard=False)
    styles = {a: line(a), b: line(b)}
    if color(a) == color(b):
        styles[b] = line(b, dash="dashdot")
    fig = go.Figure()
    for row, side in [(a, "A"), (b, "B")]:
        if row != SMARD_ROW:
            fig.add_trace(
                band_trace(row, hours, name=f"95 % range {side}", legendgroup=side)
            )
    if show_actual:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=acc.actual[hours],
                name="What really happened",
                line=model_line("actual"),
            )
        )
    for row, side in [(a, "A"), (b, "B")]:
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=acc.forecasts[row][hours],
                name=f"{side}: {label(row)}",
                line=styles[row],
                legendgroup=side,
            )
        )
    style_plotly(
        fig, f"A vs B, week of {monday:%d %b %Y}", "residual load (MWh)", height=460
    )
    fig.update_xaxes(dtick=24 * 3600 * 1000, tickformat="%a %d %b")
    st.plotly_chart(themed(fig), key="h2h_week_chart")

    categories = ["overall", *BIN_CATEGORIES]
    metric = {c: "MAE" if c == "overall" else f"MAE_{c}_by_actual" for c in categories}
    fig = go.Figure()
    for row, side in [(a, "A"), (b, "B")]:
        fig.add_trace(
            go.Bar(
                x=[SITUATION[c] for c in categories],
                y=[acc.value.loc[row, metric[c]] for c in categories],
                name=f"{side}: {label(row)}",
                marker=bar_style(row),
                hovertemplate="%{y:,.0f} MWh",
            )
        )
    style_plotly(
        fig, "Average miss per hour, by situation (lower is better)", "MWh", height=420
    )
    fig.update_layout(barmode="group")
    st.plotly_chart(themed(fig), key="h2h_situations")


st.header("Pick two forecasters")
st.markdown(
    "Put any two forecasts head to head on the test year — two of ours, or one of ours against "
    "SMARD's official forecast."
)
pick_a, pick_b = st.columns(2)
row_a = pick_a.selectbox(
    "Forecaster A",
    PUBLIC_ROWS,
    index=PUBLIC_ROWS.index(LEAD),
    format_func=public_label,
    key="h2h_a",
)
row_b = pick_b.selectbox(
    "Forecaster B",
    PUBLIC_ROWS,
    index=PUBLIC_ROWS.index(SMARD_ROW),
    format_func=public_label,
    key="h2h_b",
)
if row_a == row_b:
    st.info("Pick two different forecasters.")
else:
    head_to_head(row_a, row_b)


# ============================================================================
# What would we have said the evening before? (spec 06's setting: issued 18:00 on DAY−1)
# ============================================================================
st.header("What would we have said the evening before?")
st.markdown(
    "Pick a day of the test year. You see the forecast as it was due at **18:00 the evening "
    "before** — then reveal what really happened."
)
day_of = acc.common.normalize()
lead_gain = (SMARD_ABS - acc.errors[LEAD].abs()).groupby(day_of).sum()
actual_day = acc.actual[acc.common].groupby(day_of)
SUGGESTED = {
    "SMARD's worst day": SMARD_ABS.groupby(day_of).mean().idxmax(),
    f"{OURS_NAME}'s biggest win": lead_gain.idxmax(),
    f"{OURS_NAME}'s worst day against SMARD": lead_gain.idxmin(),
    "Most green power (lowest residual load)": actual_day.min().idxmin(),
    "Least green power (highest residual load)": actual_day.max().idxmax(),
}
eve_row_col, eve_day_col, eve_date_col = st.columns(3)
eve_row = eve_row_col.selectbox(
    "Forecaster",
    PUBLIC_ROWS[1:],
    index=PUBLIC_ROWS[1:].index(LEAD),
    format_func=public_label,
    key="eve_row",
)
suggestion = eve_day_col.selectbox(
    "Day",
    [*SUGGESTED, "Pick a date"],
    format_func=lambda k: k if k == "Pick a date" else f"{k} ({SUGGESTED[k]:%d %b %Y})",
    key="eve_suggestion",
)
if suggestion == "Pick a date":
    eve_day = pd.Timestamp(
        eve_date_col.date_input(
            "Date",
            value=acc.test_days[-1].date(),
            min_value=acc.test_days[0].date(),
            max_value=acc.test_days[-1].date(),
            key="eve_date",
        )
    )
else:
    eve_day = SUGGESTED[suggestion]
day_hours = acc.common[day_of == eve_day]
reveal_key = (eve_row, f"{eve_day:%Y-%m-%d}")
if st.button("👀 Reveal what happened", key="eve_reveal"):
    st.session_state["eve_revealed"] = reveal_key
revealed = st.session_state.get("eve_revealed") == reveal_key

if len(day_hours) == 0:
    st.info("No scored hours on that day.")
else:
    # The actual line has its own switch here: the reveal button
    _, show_smard = reference_toggles("eve", actual=False)
    fig = go.Figure()
    fig.add_trace(band_trace(eve_row, day_hours, name="95 % range"))
    if show_smard:
        fig.add_trace(
            go.Scatter(
                x=day_hours,
                y=acc.forecasts[SMARD_ROW][day_hours],
                name=SMARD_NAME,
                line=line(SMARD_ROW),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=day_hours,
            y=acc.forecasts[eve_row][day_hours],
            name=public_label(eve_row),
            line=line(eve_row, width=2.6),
        )
    )
    if revealed:
        fig.add_trace(
            go.Scatter(
                x=day_hours,
                y=acc.actual[day_hours],
                name="What really happened",
                line=model_line("actual"),
            )
        )
    style_plotly(
        fig,
        f"{eve_day:%A %d %b %Y}: the forecast from 18:00 the evening before",
        "residual load (MWh)",
        height=440,
    )
    fig.update_xaxes(tickformat="%H:00", dtick=3 * 3600 * 1000)
    st.plotly_chart(themed(fig), key="eve_chart")
    if revealed:
        band = acc.bands[eve_row].loc[day_hours]
        actual = acc.actual[day_hours]
        inside = int(((actual >= band["lower"]) & (actual <= band["upper"])).sum())
        ours_abs = acc.errors[eve_row][day_hours].abs()
        closer = int((ours_abs < SMARD_ABS[day_hours]).sum())
        st.success(
            f"Inside the 95 % range in **{inside} of {len(day_hours)}** hours; closer than SMARD in "
            f"**{closer} of {len(day_hours)}**. Average miss {ours_abs.mean():,.0f} MWh against "
            f"SMARD's {SMARD_ABS[day_hours].mean():,.0f} MWh."
        )
    else:
        st.caption(
            "The actual line stays hidden until you press **Reveal what happened**."
        )

st.divider()

# ============================================================================
# Explore any week (§4.4 and the explorer, merged)
# ============================================================================
st.header("Explore the test year")
comparing, ROWS = model_switch(
    LINEUP,
    "explore",
    "Off: our forecasts against SMARD. On: pick any of the models we trained.",
)
full_weeks = test_weeks(acc)
weekly_gain = {
    monday: float(
        (SMARD_ABS - acc.errors[LEAD].abs())[
            monday : monday + pd.Timedelta(days=7, hours=-1)
        ].sum()
    )
    for monday in full_weeks
}
best_week, worst_week = max(weekly_gain, key=weekly_gain.get), min(
    weekly_gain, key=weekly_gain.get
)
lead_skill = LINEUP.monthly_skill(LEAD)
best_month, worst_month = lead_skill.idxmax(), lead_skill.idxmin()
presets = {
    "A week of your choice": None,
    f"Our best week ({best_week:%d %b %Y})": (
        best_week,
        best_week + pd.Timedelta(days=7),
    ),
    f"Our worst week ({worst_week:%d %b %Y})": (
        worst_week,
        worst_week + pd.Timedelta(days=7),
    ),
    f"Best month ({best_month.strftime('%b %Y')})": (
        best_month.start_time,
        best_month.end_time,
    ),
    f"Worst month ({worst_month.strftime('%b %Y')})": (
        worst_month.start_time,
        worst_month.end_time,
    ),
}
if risk_labels is not None:
    for direction, weeks in default_zoom_weeks(risk_labels).items():
        for monday, reason in weeks:
            if monday is not None:
                presets[
                    f"{DIRECTION_NAME[direction]}: {reason} ({monday:%d %b %Y})"
                ] = (monday, monday + pd.Timedelta(days=7))
presets["The whole test year"] = (acc.common[0], acc.common[-1])

controls = st.columns(2)
preset = controls[0].selectbox("Show", list(presets))
if presets[preset] is None:
    # Options as ISO strings: plain values keep the widget state simple
    monday = pd.Timestamp(
        controls[1].selectbox(
            "Week",
            [f"{d:%Y-%m-%d}" for d in full_weeks],
            index=len(full_weeks) - 1,
            format_func=lambda d: f"{pd.Timestamp(d):%d %b %Y} (Mon) – "
            f"{pd.Timestamp(d) + pd.Timedelta(days=6):%d %b %Y}",
        )
    )
    start, end = monday, monday + pd.Timedelta(days=7)
else:
    start, end = presets[preset]
window = acc.common[(acc.common >= start) & (acc.common < end)]
show_actual, show_smard = reference_toggles("explore")
references = [SMARD_ROW] if show_smard else []

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08
)
band_row = LEAD if LEAD in ROWS else ROWS[0]
band = acc.bands[band_row].loc[window]
fig.add_trace(
    go.Scatter(
        x=np.concatenate([window, window[::-1]]),
        y=np.concatenate([band["upper"], band["lower"][::-1]]),
        fill="toself",
        fillcolor=color(band_row),
        opacity=0.15,
        line={"width": 0},
        name=f"95 % range ({label(band_row)})",
        hoverinfo="skip",
    ),
    row=1,
    col=1,
)
if show_actual:
    fig.add_trace(
        go.Scatter(
            x=window,
            y=acc.actual[window],
            name="What really happened",
            line=model_line("actual"),
        ),
        row=1,
        col=1,
    )
for row in [*references, *ROWS]:
    name = row_name(row)
    fig.add_trace(
        go.Scatter(
            x=window,
            y=acc.forecasts[row][window],
            name=name,
            line=line(row),
            legendgroup=name,
        ),
        row=1,
        col=1,
    )
if not comparing:
    advantage = SMARD_ABS[window] - acc.errors[LEAD][window].abs()
    fig.add_trace(
        win_loss_bars(
            window,
            advantage.to_numpy(),
            LEAD,
            "%{y:+,.0f} MWh<extra>who was closer</extra>",
        ),
        row=2,
        col=1,
    )
    bottom_label = f"{OURS_NAME.lower()} vs SMARD (MWh)"
else:
    for row in [*references, *ROWS]:
        fig.add_trace(
            go.Scatter(
                x=window,
                y=acc.errors[row][window],
                name=label(row),
                line=line(row),
                legendgroup=label(row),
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    bottom_label = "miss (MWh), forecast − actual"
# The title names only the lines on the chart (the reference toggles can hide two)
shown = [
    *(["Actual"] if show_actual else []),
    *(["SMARD"] if show_smard else []),
    "our forecasts" if not comparing else "our models",
]
explorer_title = (
    ", ".join(shown[:-1]) + " and " + shown[-1] if len(shown) > 1 else shown[0]
)
explorer_title = f"{explorer_title[0].upper()}{explorer_title[1:]}: {preset}"
style_plotly(
    fig,
    explorer_title,
    "residual load (MWh)",
    height=640,
)
fig.update_yaxes(title_text=bottom_label, row=2, col=1)
if (end - start) <= pd.Timedelta(days=8):
    fig.update_xaxes(dtick=24 * 3600 * 1000, tickformat="%a %d %b")
st.plotly_chart(themed(fig), key="explorer_chart")
inside = coverage(acc, band_row, window)
if not comparing:
    closer = int((advantage > 0).sum())
    st.caption(
        f"Bottom: bars above 0 are hours where {OURS_NAME.lower()} was closer, below 0 (grey) where "
        f"SMARD was. Here: ours closer in {closer} of {len(window)} hours, average miss "
        f"{acc.errors[LEAD][window].abs().mean():,.0f} MWh against SMARD's {SMARD_ABS[window].mean():,.0f} MWh; "
        f"reality inside the 95 % range in {inside:.1f} % of the hours. "
        "One week is an example, not a verdict. Drag to zoom, double-click to reset."
    )
else:
    st.caption(
        f"Shaded: the 95 % range of {label(band_row)}; reality inside it in {inside:.1f} % of the hours shown."
    )


next_page("app_pages/try_it.py")
