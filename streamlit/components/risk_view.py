"""Risk days on "Do we beat SMARD?" — does our forecast flag the days the grid is at risk?

Ported from `notebooks/05_modeling/visualization-02-classification-risk-labels.ipynb` (spec 10)
§3–§5 and §7; it was the separate Risk days page until the team folded it into "Where we beat
SMARD" (2026-10-07). Flags, thresholds and ranges come from spec 10's export through
`model_results.py` — nothing is recomputed, and an empty flag stays "not evaluable", never
"not at risk". The page calls `risk_days_section(labels)`; every chart function takes the labels.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.layout import title_of
from components.naming import SMARD_NAME
from model_results import (
    BASES,
    default_zoom_weeks,
    flagged_span,
    german_holidays,
    is_set,
    outcome,
    persistent_hours,
    risk_scores,
    week_of,
)
from viz_helpers import (
    COLORS,
    HOLIDAY_COLOR,
    MODEL_STYLE,
    OUTCOME_COLOR,
    RISK_COLOR,
    model_label,
    model_line,
    style_plotly,
    themed,
)

MARGIN_WINDOW = (
    10_000  # MWh: axis range of the day-margin chart around 0 (spec 10 settings)
)
DEFAULT_BASIS = {"high": "rolling", "low": "zero"}  # as in the page's closing summary

DIRECTION_TEXT = {
    "high": (
        "Too little green power (high)",
        "residual load higher than in all but 1 % of the hours of the past twelve months: little wind and sun for the demand",
    ),
    "low": (
        "Too much green power (low)",
        "residual load lower than in all but 1 % of the hours of the past twelve months, or below zero: wind and solar exceed demand",
    ),
}
BASIS_TEXT = {
    "rolling": "lowest 1 % of the past twelve months",
    "zero": "below zero",
}
RULE_TEXT = {"any": "at least one hour", "3h": "at least 3 h in a row"}
# The same names as on the other pages
SOURCE_NAME = {"actual": "What really happened", "smard": SMARD_NAME}


def source_label(labels, source, direction):
    """Actual, SMARD, or the direction's model."""
    return (
        model_label(labels.picks[direction])
        if source == "model"
        else SOURCE_NAME[source]
    )


def source_color(labels, source, direction):
    key = labels.picks[direction] if source == "model" else source
    return MODEL_STYLE[key]["color"]


# ============================================================================
# Scorecard (spec 10 §3)
# ============================================================================
def scorecard(labels, direction, basis, rule):
    scores = risk_scores(labels, direction, basis, rule)
    ours, smard = scores["model"], scores["smard"]
    n = ours["actual days"]
    model = source_label(labels, "model", direction)
    if n == 0:
        st.info(f"The actual has no risk day of this kind in the test window.")
        return scores
    st.markdown(
        f"Of the **{n} actual risk days**, **{model}** saw **{ours['hit days']}** coming and SMARD "
        f"**{smard['hit days']}**; {model} raised **{ours['false-alarm days']} false alarms**, "
        f"SMARD **{smard['false-alarm days']}**."
    )
    cards = st.columns(4)
    cards[0].metric(
        "Risk days it saw coming",
        f"{ours['hit days']} of {n}",
        f"{ours['hit days'] - smard['hit days']:+d} days vs SMARD",
        help=f"Caught {ours['recall %']:.1f} % of the risk days (SMARD {smard['recall %']:.1f} %).",
    )
    cards[1].metric(
        "Risk days it missed",
        f"{ours['missed days']}",
        f"{ours['missed days'] - smard['missed days']:+d} days vs SMARD",
        delta_color="inverse",
    )
    precision = "—" if np.isnan(ours["precision %"]) else f"{ours['precision %']:.1f} %"
    cards[2].metric(
        "False alarms",
        f"{ours['false-alarm days']}",
        f"{ours['false-alarm days'] - smard['false-alarm days']:+d} days vs SMARD",
        delta_color="inverse",
        help=f"{precision} of its warnings were real risk days.",
    )
    cards[3].metric(
        "Risk hours it missed",
        f"{ours['missed hours']}",
        f"{ours['missed hours'] - smard['missed hours']:+d} hours vs SMARD",
        delta_color="inverse",
        help=(
            f"Hours: {ours['hit hours']} seen coming, {ours['false-alarm hours']} false alarms "
            f"(SMARD {smard['hit hours']} / {smard['false-alarm hours']}). Hours have no 3 h rule. "
            f"Counted over every hour with a risk label ({len(labels.hourly):,}); the hour charts "
            "above use only hours where every model has a forecast, so their counts can differ by one or two."
        ),
    )
    if n < 20:
        st.caption(
            f"Small numbers: {n} risk days in one test year, so one day moves the share seen coming by "
            f"{100 / n:.1f} percentage points."
        )
    return scores


# ============================================================================
# Risk calendar (spec 10 §4.1 / §5.1)
# ============================================================================
# Display names of the outcomes (the export keeps hit / miss / false alarm / quiet)
OUTCOME_NAME = {
    "hit": "saw it coming",
    "miss": "missed it",
    "false alarm": "false alarm",
    "quiet": "ordinary day",
    "not evaluable": "can't be scored",
}
OUTCOME_MARKER = {  # hit = filled, miss = outline, false alarm = crossed outline (readable without colour)
    "hit": {"symbol": "square"},
    "miss": {"symbol": "square-open"},
    "false alarm": {"symbol": "square-x-open"},
    "quiet": {"symbol": "square"},
    "not evaluable": {"symbol": "square-open"},
}


def risk_calendar(labels, direction, basis, rule):
    days = labels.daily.index
    days_off = german_holidays(days)
    tail = RISK_COLOR[(direction, basis)]
    first_monday = week_of(days[:1])[0]
    column = np.asarray((days - first_monday).days // 7)
    weekday = np.asarray(days.weekday)
    actual = labels.flag("actual", direction, basis, rule)
    sustained = is_set(labels.flag("actual", direction, basis, "3h")).to_numpy()
    holiday = np.array([day in days_off for day in days])
    color = {
        "hit": tail,
        "miss": tail,
        "false alarm": tail,
        "quiet": OUTCOME_COLOR["quiet"],
        "not evaluable": COLORS["muted"],
    }

    sources = ["model", "smard"]
    results = {
        s: outcome(labels.flag(s, direction, basis, rule), actual).fillna(
            "not evaluable"
        )
        for s in sources
    }
    titles = []
    for s in sources:
        counts = results[s].value_counts()
        titles.append(
            f"{source_label(labels, s, direction)}: saw {counts.get('hit', 0)} coming, missed {counts.get('miss', 0)}, "
            f"{counts.get('false alarm', 0)} false alarms"
        )
    fig = make_subplots(rows=2, cols=1, subplot_titles=titles, vertical_spacing=0.14)
    for r, s in enumerate(sources, start=1):
        spans = [
            "<br>".join(
                f"{source_label(labels, src, direction)}: {flagged_span(labels, src, direction, basis, day, rule)}"
                for src in ("actual", "model", "smard")
            )
            for day in days
        ]
        for name in OUTCOME_MARKER:
            mask = (results[s] == name).to_numpy()
            if not mask.any():
                continue
            fig.add_trace(
                go.Scatter(
                    x=column[mask],
                    y=weekday[mask],
                    mode="markers",
                    name=OUTCOME_NAME[name],
                    legendgroup=name,
                    showlegend=r == 1,
                    marker={
                        **OUTCOME_MARKER[name],
                        "size": 13,
                        "color": color[name],
                        "line": {"width": 2, "color": color[name]},
                    },
                    customdata=np.column_stack(
                        [
                            days[mask].strftime("%a %d %b %Y"),
                            np.array(spans, dtype=object)[mask],
                        ]
                    ),
                    hovertemplate=f"%{{customdata[0]}}<br><b>{OUTCOME_NAME[name]}</b><br>flagged hours:<br>%{{customdata[1]}}<extra></extra>",
                ),
                row=r,
                col=1,
            )
        fig.add_trace(
            go.Scatter(
                x=column[holiday],
                y=weekday[holiday],
                mode="markers",
                name="public holiday",
                legendgroup="holiday",
                showlegend=r == 1,
                hoverinfo="skip",
                marker={
                    "symbol": "square-open",
                    "size": 19,
                    "color": HOLIDAY_COLOR,
                    "line": {"width": 1.5},
                },
            ),
            row=r,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=column[sustained],
                y=weekday[sustained],
                mode="markers",
                name="actual: ≥ 3 h in a row",
                legendgroup="sustained",
                showlegend=r == 1,
                hoverinfo="skip",
                marker={
                    "symbol": "circle",
                    "size": 4,
                    "color": MODEL_STYLE["actual"]["color"],
                },
            ),
            row=r,
            col=1,
        )
    month_starts = pd.date_range(days[0], days[-1], freq="MS")
    fig.update_xaxes(
        tickvals=[(m - first_monday).days // 7 for m in month_starts],
        ticktext=[f"{m:%b %y}" for m in month_starts],
        showgrid=False,
        zeroline=False,
        ticks="",
    )
    fig.update_yaxes(
        tickvals=list(range(7)),
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        autorange="reversed",
        showgrid=False,
        zeroline=False,
        ticks="",
    )
    fig.update_layout(
        title={
            "text": "Every test day: did the forecast flag it?",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17},
        },
        height=520,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.08,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 50, "r": 20, "t": 80, "b": 40},
    )
    st.plotly_chart(themed(fig), key=f"calendar_{direction}_{basis}_{rule}")
    st.caption(
        "Filled square: saw it coming. Open square: missed it. Crossed square: false alarm. "
        "Grey: an ordinary day, correctly not flagged. Olive frame: public holiday. Dot: the actual "
        "stayed past the threshold for at least 3 h."
    )


# ============================================================================
# Zoom week (spec 10 §4.2 / §5.2)
# ============================================================================
def zoom_week(labels, direction, basis):
    daily, hourly = labels.daily, labels.hourly
    days = daily.index
    days_off = german_holidays(days)
    zooms = default_zoom_weeks(labels)
    tail = RISK_COLOR[(direction, basis)]
    mondays = sorted(set(week_of(days)))
    mondays = [
        m for m in mondays if m + pd.Timedelta(days=6) <= days[-1] and m >= days[0]
    ]
    suggested = {
        monday: reason for monday, reason in zooms[direction] if monday is not None
    }
    actual_days = is_set(labels.flag("actual", direction, basis, "any"))
    per_week = actual_days.groupby(week_of(days)).sum()

    def week_text(monday):
        text = f"{monday:%d %b %Y}"
        if per_week.get(monday, 0):
            text += (
                f" — {per_week[monday]} risk day{'s' if per_week[monday] != 1 else ''}"
            )
        if monday in suggested:
            text += f" ★ {suggested[monday]}"
        return text

    default = next(iter(suggested), mondays[-1])
    monday = st.selectbox(
        "Week",
        mondays,
        index=mondays.index(default) if default in mondays else len(mondays) - 1,
        format_func=week_text,
        key=f"zoom_{direction}_{basis}",
    )
    end = monday + pd.Timedelta(days=7)
    week = hourly[(hourly.index >= monday) & (hourly.index < end)]
    series = {
        "actual": week["residual_load"],
        "model": week[f"{direction}_model_forecast"],
        "smard": week["fc_residual_load"],
    }
    sources = list(series)

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.05,
    )
    for source in sources:
        key = labels.picks[direction] if source == "model" else source
        fig.add_trace(
            go.Scatter(
                x=week.index,
                y=series[source],
                name=source_label(labels, source, direction),
                line=model_line(key),
            ),
            row=1,
            col=1,
        )
    threshold = (
        daily[f"{direction}_threshold_{basis}"]
        .reindex(week.index.normalize())
        .to_numpy()
    )
    fig.add_trace(
        go.Scatter(
            x=week.index,
            y=threshold,
            name="risk threshold",
            line={"color": tail, "width": 2, "dash": "dashdot", "shape": "hv"},
        ),
        row=1,
        col=1,
    )
    if direction == "low" and basis != "zero":
        fig.add_hline(
            y=0,
            line={"color": RISK_COLOR[("low", "zero")], "dash": "dot", "width": 2},
            row=1,
            col=1,
        )

    in_legend = set()
    for position, source in enumerate(sources):
        flag = week[f"{source}_{direction}_risk_hour_{basis}"]
        long = persistent_hours(flag)
        short = is_set(flag) & ~long
        for hours, height, opacity, name in [
            (long, 0.8, 1.0, "flagged, part of ≥ 3 h"),
            (short, 0.3, 0.45, "flagged, shorter"),
        ]:
            if hours.any():
                fig.add_trace(
                    go.Bar(
                        x=week.index[hours.to_numpy()],
                        y=[height] * int(hours.sum()),
                        base=position - height / 2,
                        width=3_600_000,
                        marker={
                            "color": tail,
                            "opacity": opacity,
                            "line": {"width": 0},
                        },
                        name=name,
                        legendgroup=name,
                        showlegend=name not in in_legend,
                        hovertemplate=f"{source_label(labels, source, direction)} flagged %{{x|%a %H:%M}}<extra></extra>",
                    ),
                    row=2,
                    col=1,
                )
                in_legend.add(name)
    style_plotly(
        fig,
        f"{DIRECTION_TEXT[direction][0]}: week of {monday:%d %b %Y}",
        "residual load (MWh)",
        height=600,
    )
    fig.update_yaxes(
        tickvals=list(range(len(sources))),
        ticktext=[source_label(labels, s, direction) for s in sources],
        range=[len(sources) - 0.5, -0.5],
        showgrid=False,
        tickformat="",
        title_text="",
        row=2,
        col=1,
    )
    day_ticks = pd.date_range(monday, periods=7, freq="D")
    fig.update_xaxes(
        tickvals=day_ticks + pd.Timedelta(hours=12),
        ticktext=[
            f"{d:%a %d %b}" + (f"<br>{days_off.get(d)}" if d in days_off else "")
            for d in day_ticks
        ],
        range=[monday, end],
    )
    for day in day_ticks[1:]:
        fig.add_vline(x=day, line={"color": COLORS["grid"], "width": 1})
    st.plotly_chart(themed(fig), key=f"zoom_chart_{direction}_{basis}")

    flag_columns = [f"{s}_{direction}_risk_{basis}_any" for s in sources]
    flagged = [
        day
        for day in day_ticks
        if day in daily.index and daily.loc[day, flag_columns].eq(True).any()
    ]
    if flagged:
        st.dataframe(
            pd.DataFrame(
                {
                    source_label(labels, s, direction): [
                        flagged_span(labels, s, direction, basis, day)
                        for day in flagged
                    ]
                    for s in sources
                },
                index=[f"{day:%a %d %b}" for day in flagged],
            ),
        )
        st.caption("Flagged hours per day (first–last flagged hour).")
    else:
        st.caption("No source flags a day in this week.")


# ============================================================================
# Day margin (spec 10 §4.3 / §5.3)
# ============================================================================
def day_margin(labels, direction, basis):
    daily = labels.daily
    evaluable = daily["evaluable"]
    tail = RISK_COLOR[(direction, basis)]
    scored = daily[evaluable]
    extreme = "max" if direction == "high" else "min"
    sign = 1 if direction == "high" else -1
    threshold = scored[f"{direction}_threshold_{basis}"]
    x = scored[f"actual_{extreme}"] - threshold
    sustained = is_set(scored[f"actual_{direction}_risk_{basis}_3h"])
    flagged = is_set(scored[f"actual_{direction}_risk_{basis}_any"])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[source_label(labels, s, direction) for s in ("smard", "model")],
        shared_yaxes=True,
    )
    lim = MARGIN_WINDOW
    for col, source in enumerate(["smard", "model"], start=1):
        y = scored[f"{source}_{extreme}"] - threshold
        color = source_color(labels, source, direction)
        shown = (x.abs() <= lim) & (y.abs() <= lim)
        hit_x, hit_y = sign * x >= 0, sign * y >= 0
        fig.add_shape(
            type="rect",
            x0=0,
            x1=sign * lim,
            y0=0,
            y1=sign * lim,
            fillcolor=tail,
            opacity=0.25,
            line_width=0,
            layer="below",
            row=1,
            col=col,
        )
        fig.add_shape(
            type="rect",
            x0=0,
            x1=sign * lim,
            y0=0,
            y1=-sign * lim,
            line={"color": tail, "width": 2},
            layer="below",
            row=1,
            col=col,
        )
        fig.add_shape(
            type="rect",
            x0=0,
            x1=-sign * lim,
            y0=0,
            y1=sign * lim,
            line={"color": tail, "width": 1.5, "dash": "dot"},
            layer="below",
            row=1,
            col=col,
        )
        fig.add_shape(
            type="line",
            x0=-lim,
            x1=lim,
            y0=-lim,
            y1=lim,
            line={"color": COLORS["grid"], "dash": "dash"},
            row=1,
            col=col,
        )
        for mask, name, marker in [
            (
                shown & ~flagged,
                "not a risk day",
                {"size": 5, "color": color, "opacity": 0.3},
            ),
            (
                shown & flagged & ~sustained,
                "risk day, < 3 h",
                {"size": 9, "color": "white", "line": {"color": color, "width": 1.5}},
            ),
            (shown & sustained, "risk day, ≥ 3 h", {"size": 9, "color": color}),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=x[mask],
                    y=y[mask],
                    mode="markers",
                    marker=marker,
                    name=name,
                    legendgroup=name,
                    showlegend=col == 1,
                    customdata=scored.index[mask].strftime("%a %d %b %Y"),
                    hovertemplate="%{customdata}<br>actual %{x:+,.0f} MWh past the threshold<br>forecast %{y:+,.0f} MWh<extra></extra>",
                ),
                row=1,
                col=col,
            )
        for name, (u, v), n in [
            ("saw it coming", (0.5, 0.85), int((hit_x & hit_y).sum())),
            ("missed it", (0.5, -0.85), int((hit_x & ~hit_y).sum())),
            ("false alarms", (-0.5, 0.85), int((~hit_x & hit_y).sum())),
        ]:
            fig.add_annotation(
                x=sign * u * lim,
                y=sign * v * lim,
                text=f"<b>{n}</b> {name}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.85)",
                row=1,
                col=col,
            )
    style_plotly(
        fig,
        f"How close was it? Day {extreme} minus the threshold",
        "forecast past the threshold (MWh)",
        "actual past the threshold (MWh)",
        height=480,
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(
        range=[-lim, lim],
        tickformat="+,.0f",
        zeroline=True,
        zerolinecolor=COLORS["muted"],
    )
    fig.update_yaxes(
        range=[-lim, lim],
        tickformat="+,.0f",
        zeroline=True,
        zerolinecolor=COLORS["muted"],
    )
    st.plotly_chart(themed(fig), key=f"margin_{direction}_{basis}")
    outside = int(
        (
            ~((x.abs() <= lim) & (scored[f"model_{extreme}"] - threshold).abs().le(lim))
        ).sum()
    )
    st.caption(
        f"One dot per scored day: how far the day's {extreme} lay past the threshold, actual (x) against "
        f"forecast (y). Tinted: saw it coming, outlined: missed it, dotted: false alarms. Days further than "
        f"{lim:,} MWh from the threshold are off the chart ({outside} for {source_label(labels, 'model', direction)}). "
        "Uses the 'at least one hour' rule."
    )


# ============================================================================
# Findings, computed live (spec 10 §7)
# ============================================================================
def findings(labels, direction, basis, rule):
    days = labels.daily.index
    days_off = german_holidays(days)
    actual = labels.flag("actual", direction, basis, rule)
    risk_days = days[is_set(actual).to_numpy()]
    if len(risk_days) == 0:
        return
    weekend = risk_days.dayofweek >= 5
    months = pd.Series(risk_days.month).value_counts().sort_index()
    misses = {
        s: days[
            (
                outcome(labels.flag(s, direction, basis, rule), actual) == "miss"
            ).to_numpy()
        ]
        for s in ("model", "smard")
    }
    on_holiday = [d for d in risk_days if d in days_off]
    model = source_label(labels, "model", direction)
    lines = [
        f"- **{int(weekend.sum())} of the {len(risk_days)} risk days fall on a weekend.**\n"
        f"  - Missed on weekends: {model} {int((misses['model'].dayofweek >= 5).sum())}, SMARD "
        f"{int((misses['smard'].dayofweek >= 5).sum())}. On weekdays: {model} "
        f"{int((misses['model'].dayofweek < 5).sum())}, SMARD {int((misses['smard'].dayofweek < 5).sum())}.",
        "- **Risk days by month:** "
        + ", ".join(f"{pd.Timestamp(2000, m, 1):%b} {n}" for m, n in months.items())
        + ".",
    ]
    if on_holiday:
        lines.append(
            f"- **{len(on_holiday)} risk day{'s' if len(on_holiday) != 1 else ''} on a public holiday:** "
            + ", ".join(f"{d:%d %b} ({days_off.get(d)})" for d in on_holiday)
            + "."
        )
    st.markdown("\n\n".join(lines))


# ============================================================================
# Section
# ============================================================================
def risk_days_section(labels):
    """Intro, then one tab per side: scorecard, findings, calendar; week and near misses folded."""
    days, evaluable = labels.daily.index, labels.daily["evaluable"]
    st.markdown(
        f"A day is a **risk day** when the actual residual load crosses the project's risk line "
        f"(see '{title_of('app_pages/method.py')}'). Here the same lines are applied to our forecast and to SMARD's, "
        f"issued the day before, over the test window **{days[0]:%d %b %Y} – {days[-1]:%d %b %Y}** "
        f"({int(evaluable.sum())} days that can be scored). A forecast **saw it coming** when it "
        "flags the risk day too; it **missed it** when it does not, and raises a **false alarm** when it flags a "
        "day the actual does not. A risk day is a warning sign for Germany as a whole, not proof "
        "that the grid operators had to step in."
    )
    if (~evaluable).any():
        st.caption(
            f"{int((~evaluable).sum())} days cannot be scored (incomplete data) and are left out."
        )

    tabs = st.tabs([DIRECTION_TEXT[d][0] for d in BASES])
    for tab, direction in zip(tabs, BASES):
        with tab:
            st.markdown(
                f"**{DIRECTION_TEXT[direction][0]}:** {DIRECTION_TEXT[direction][1]}. Forecast: "
                f"**{source_label(labels, 'model', direction)}**, the best model for this side."
            )
            controls = st.columns(2)
            if len(BASES[direction]) > 1:
                basis = controls[0].radio(
                    "Threshold",
                    BASES[direction],
                    index=BASES[direction].index(DEFAULT_BASIS[direction]),
                    format_func=BASIS_TEXT.get,
                    horizontal=True,
                    key=f"basis_{direction}",
                )
            else:
                basis = BASES[direction][0]
            rule = controls[1].radio(
                "A risk day needs",
                list(RULE_TEXT),
                format_func=RULE_TEXT.get,
                horizontal=True,
                key=f"rule_{direction}",
            )

            scorecard(labels, direction, basis, rule)
            findings(labels, direction, basis, rule)
            risk_calendar(labels, direction, basis, rule)
            with st.expander("A week up close"):
                zoom_week(labels, direction, basis)
            with st.expander("Near misses: how close was it?"):
                day_margin(labels, direction, basis)
