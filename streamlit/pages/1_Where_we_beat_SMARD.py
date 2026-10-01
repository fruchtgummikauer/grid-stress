"""Where we beat SMARD — our day-ahead residual-load forecast against SMARD's.

Ported from `notebooks/05_modeling/visualization-01-regression-best-models.ipynb` (spec 09) §3–§9.1,
per `.claude/specs/Streamlit-draft.md` §17 (Behaviour 3–9). Every number is computed from the
exports through `model_results.py`; nothing is copied from the notebook's dated findings.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from model_results import (
    BIN_CATEGORIES,
    CANDIDATE_SPLIT,
    MIN_BIN_SHARE,
    SMARD_ROW,
    ZERO,
    default_zoom_weeks,
    get_or_stop,
    load_accuracy,
    load_export,
    load_risk_labels,
)
from viz_helpers import (
    BIN_COLOR,
    BIN_LABEL,
    COLORS,
    MODEL_STYLE,
    model_label,
    model_line,
    style_plotly,
)

st.set_page_config(
    page_title="Where we beat SMARD — Grid Stress", page_icon="🏁", layout="wide"
)
st.title("Where our forecast beats SMARD")

acc = get_or_stop(load_accuracy)
if acc.problems:
    st.error(
        "The model exports do not reproduce their own scoreboard, so this page is not shown:\n\n"
        + "\n".join(f"- {problem}" for problem in acc.problems)
    )
    st.stop()

PICKS = acc.picks["overall"]
if not PICKS:
    st.warning(
        f"No `{CANDIDATE_SPLIT}` model beats SMARD on average over the test window "
        f"({acc.test_days[0]:%d %b %Y} – {acc.test_days[-1]:%d %b %Y})."
    )
    st.stop()
FIRST = PICKS[0]
ALL_ROWS = [row for row in acc.errors if row != SMARD_ROW]
SMARD_ABS = acc.errors[SMARD_ROW].abs()


def label(row):
    """Display label of a (model, split_method) row; the split only when it is not the default."""
    base = model_label(row[0])
    return base if row[1] in (CANDIDATE_SPLIT, "none") else f"{base} ({row[1]})"


def line(row, **overrides):
    """Plotly line of a row: the model's colour, dotted for a non-default split."""
    style = model_line(row[0])
    if row[1] not in (CANDIDATE_SPLIT, "none"):
        style["dash"] = "dot"
    return style | overrides


def compare(key, default):
    """'Compare models' multiselect over every row with forecasts, defaulting to `default`."""
    return st.multiselect(
        "Compare models",
        ALL_ROWS,
        default=default,
        format_func=label,
        key=key,
        help="The defaults are the top picks of this category (spec 09's pick rule).",
    )


def monthly_skill(row):
    """Skill vs SMARD (%) per full month."""
    return 100 * (1 - acc.monthly_mae[row] / acc.monthly_mae[SMARD_ROW])


def longest_winning_run():
    """The longest run of consecutive full months that every overall pick wins, or None."""
    wins = pd.concat([monthly_skill(row) > 0 for row in PICKS], axis=1).all(axis=1)
    best, start = None, None
    for i, won in enumerate(wins.to_numpy()):
        if won and start is None:
            start = i
        if start is not None and (not won or i == len(wins) - 1):
            end = i if won else i - 1
            if best is None or end - start > best[1] - best[0]:
                best = (start, end)
            start = None
    return None if best is None else (wins.index[best[0]], wins.index[best[1]])


# ============================================================================
# Headline
# ============================================================================
mae, smard_mae = acc.value.loc[FIRST, "MAE"], acc.value.loc[SMARD_ROW, "MAE"]
skill = acc.value.loc[FIRST, "skill_pct"]
won, months = acc.value.loc[FIRST, "months_beating_smard"], len(acc.full_months)
saved = float((SMARD_ABS - acc.errors[FIRST].abs()).sum())

st.markdown(
    f"Over the test year **{acc.test_days[0]:%d %b %Y} – {acc.test_days[-1]:%d %b %Y}**, our best "
    f"day-ahead forecast (**{label(FIRST)}**) misses the actual residual load by "
    f"**{mae:,.0f} MWh** in an average hour, against **{smard_mae:,.0f} MWh** for SMARD's public "
    f"forecast — **{skill:.1f} % closer**, and better in **{won:.0f} of {months}** full months."
)

others = ", ".join(
    f"{label(row)} {acc.value.loc[row, 'MAE']:,.0f} MWh ({acc.value.loc[row, 'skill_pct']:+.1f} %)"
    for row in PICKS[1:]
)
cards = st.columns(4)
cards[0].metric(
    "Average hourly error",
    f"{mae:,.0f} MWh",
    f"{mae - smard_mae:+,.0f} MWh vs SMARD",
    delta_color="inverse",
    help=f"Mean absolute error over {len(acc.common):,} test hours. Other picks: {others or '—'}.",
)
cards[1].metric(
    "Closer than SMARD",
    f"{skill:+.1f} %",
    help="Skill vs SMARD: 1 − our MAE / SMARD's MAE, on the same hours.",
)
cards[2].metric(
    "Months won",
    f"{won:.0f} / {months}",
    help="Full calendar months in which our MAE was lower than SMARD's.",
)
cards[3].metric(
    "Error saved over the year",
    f"{saved:,.0f} MWh",
    help="Sum over all test hours of |SMARD error| − |our error|.",
)

st.divider()

# ============================================================================
# Monthly skill (spec 09 §4.1) with the winning stretch
# ============================================================================
st.header("When we beat SMARD")
chart = st.container()
rows = compare("monthly", PICKS)

fig = go.Figure()
month_x = acc.full_months.to_timestamp()
fig.add_trace(
    go.Scatter(
        x=month_x,
        y=np.zeros(len(month_x)),
        name=f"{model_label('smard')} (0 %)",
        line=model_line("smard"),
        hoverinfo="skip",
    )
)
for row in rows:
    fig.add_trace(
        go.Scatter(
            x=month_x,
            y=monthly_skill(row).to_numpy(),
            mode="lines+markers",
            name=label(row),
            line=line(row),
            hovertemplate="%{y:+.1f} %",
        )
    )

run = longest_winning_run()
if run is not None:
    n_run = (run[1] - run[0]).n + 1
    fig.add_vrect(
        x0=run[0].start_time - pd.Timedelta(days=15),
        x1=run[1].start_time + pd.Timedelta(days=15),
        fillcolor=COLORS["grid"],
        opacity=0.6,
        line_width=0,
        layer="below",
        annotation_text=f"{n_run} months in a row: every pick ahead of SMARD",
        annotation_position="top left",
    )
first_skill = monthly_skill(FIRST)
for month, text in [(first_skill.idxmax(), "best"), (first_skill.idxmin(), "worst")]:
    fig.add_annotation(
        x=month.to_timestamp(),
        y=first_skill[month],
        text=f"{text}: {month.strftime('%b %Y')}, {first_skill[month]:+.1f} %",
        showarrow=True,
        arrowhead=0,
        ay=-35 if text == "best" else 35,
        font={"color": COLORS["muted"]},
    )
style_plotly(
    fig, f"Skill vs SMARD per full month ({months} months)", "skill vs SMARD (%)"
)
fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%b %Y")
fig.update_yaxes(tickformat="+.0f")
chart.plotly_chart(fig, key="monthly_chart")

loss_months = first_skill[first_skill < 0]
bullets = [
    f"- **{label(FIRST)} is ahead of SMARD in {won:.0f} of {months} full months.**\n"
    f"  - Best month {first_skill.idxmax().strftime('%b %Y')} ({first_skill.max():+.1f} %), worst "
    f"{first_skill.idxmin().strftime('%b %Y')} ({first_skill.min():+.1f} %)."
]
if run is not None and n_run == months:
    bullets.append(f"- **Every top pick wins every one of the {months} full months.**")
elif run is not None:
    bullets.append(
        f"- **The lead is concentrated:** from {run[0].strftime('%b %Y')} to {run[1].strftime('%b %Y')}, every top pick "
        f"wins every month ({n_run} of {months} months in a row)."
    )
if len(loss_months):
    bullets.append(
        "- **SMARD is closer in "
        + ", ".join(
            f"{m.strftime('%b %Y')} ({v:+.1f} %)" for m, v in loss_months.items()
        )
        + "** — see the expander below for SMARD's own monthly error."
    )
st.markdown("\n\n".join(bullets))

# ============================================================================
# Cumulative advantage (§4.2) and skill by hour (§4.3)
# ============================================================================
st.header("How the lead builds up")
left, right = st.columns(2)
with left:
    chart = st.container()
    rows = compare("cumulative", PICKS)
    fig = go.Figure()
    for row in rows:
        advantage = (SMARD_ABS - acc.errors[row].abs()).cumsum()
        fig.add_trace(
            go.Scatter(
                x=advantage.index,
                y=advantage.to_numpy(),
                name=f"{label(row)}: {advantage.iloc[-1]:+,.0f} MWh",
                line=line(row),
                hovertemplate="%{y:+,.0f} MWh",
            )
        )
    style_plotly(
        fig, "Error saved against SMARD, running total", "Σ |e SMARD| − |e ours| (MWh)"
    )
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")
    chart.plotly_chart(fig, key="cumulative_chart")
    st.caption(
        "A steady slope means a steady gain; steps mean a few episodes carry it."
    )

with right:
    chart = st.container()
    rows = compare("by_hour", PICKS)
    hour_of = acc.common.hour
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(24)),
            y=np.zeros(24),
            name=f"{model_label('smard')} (0 %)",
            line=model_line("smard"),
            hoverinfo="skip",
        )
    )
    for row in rows:
        by_hour = acc.skill(row, hour_of)
        fig.add_trace(
            go.Scatter(
                x=by_hour.index,
                y=by_hour.to_numpy(),
                mode="lines+markers",
                name=label(row),
                line=line(row),
                hovertemplate="%{y:+.1f} %",
            )
        )
    style_plotly(
        fig,
        "Skill vs SMARD by hour of day",
        "skill vs SMARD (%)",
        "hour of day (local time)",
    )
    fig.update_xaxes(dtick=1)
    fig.update_yaxes(tickformat="+.0f")
    chart.plotly_chart(fig, key="hour_chart")
    first_by_hour = acc.skill(FIRST, hour_of)
    st.caption(
        f"{label(FIRST)} is ahead of SMARD in {int((first_by_hour > 0).sum())} of 24 hours, most "
        f"at {first_by_hour.idxmax():02d}:00 ({first_by_hour.max():+.1f} %), least at "
        f"{first_by_hour.idxmin():02d}:00 ({first_by_hour.min():+.1f} %)."
    )

# ============================================================================
# One week, hour by hour (§4.4)
# ============================================================================
st.header("One week, hour by hour")
mondays = acc.test_days[acc.test_days.dayofweek == 0]
full_weeks = [day for day in mondays if day + pd.Timedelta(days=6) in acc.test_days]
pick_col, week_col = st.columns([1, 1])
monday = week_col.selectbox(
    "Week",
    full_weeks,
    index=len(full_weeks) - 1,
    format_func=lambda d: f"{d:%d %b %Y} (Mon) – {d + pd.Timedelta(days=6):%d %b %Y}",
)
week_row = pick_col.selectbox(
    "Model", ALL_ROWS, index=ALL_ROWS.index(FIRST), format_func=label
)
week = acc.common[(acc.common >= monday) & (acc.common < monday + pd.Timedelta(days=7))]
advantage = SMARD_ABS[week] - acc.errors[week_row][week].abs()

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08
)
fig.add_trace(
    go.Scatter(
        x=week,
        y=acc.actual[week],
        name=model_label("actual"),
        line=model_line("actual"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=week,
        y=acc.forecasts[SMARD_ROW][week],
        name=model_label("smard"),
        line=model_line("smard"),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Scatter(
        x=week,
        y=acc.forecasts[week_row][week],
        name=label(week_row),
        line=line(week_row),
    ),
    row=1,
    col=1,
)
fig.add_trace(
    go.Bar(
        x=week,
        y=advantage.to_numpy(),
        name="who was closer",
        showlegend=False,
        marker_color=np.where(
            advantage >= 0,
            MODEL_STYLE[week_row[0]]["color"],
            MODEL_STYLE["smard"]["color"],
        ),
        hovertemplate="%{y:+,.0f} MWh",
    ),
    row=2,
    col=1,
)
style_plotly(
    fig,
    f"{label(week_row)} vs SMARD, week of {monday:%d %b %Y}",
    "residual load (MWh)",
    height=620,
)
fig.update_yaxes(title_text="|e SMARD| − |e ours| (MWh)", row=2, col=1)
fig.update_xaxes(dtick=24 * 3600 * 1000, tickformat="%a %d %b")
st.plotly_chart(fig, key="week_chart")
closer = int((advantage > 0).sum())
st.caption(
    f"Bars above 0 ({label(week_row)}'s colour): hours where our forecast was closer; below 0 (grey): "
    f"SMARD was closer. This week: ours closer in {closer} of {len(week)} hours, week MAE "
    f"{acc.errors[week_row][week].abs().mean():,.0f} MWh against SMARD's {SMARD_ABS[week].mean():,.0f} MWh. "
    "One week is an example, not a verdict."
)

# ============================================================================
# The numbers behind it (§9.1)
# ============================================================================
with st.expander("The numbers behind it: monthly error of SMARD and the top picks"):
    smard = load_export("smard").set_index("timestamp")
    smard_error = acc.errors[SMARD_ROW]
    month_of = acc.common.to_period("M")
    smard_daily = smard_error.groupby(acc.common.normalize()).mean()
    columns = {
        ("SMARD", "MAE"): smard_error.abs().groupby(month_of).mean(),
        ("SMARD", "bias"): smard_error.groupby(month_of).mean(),
        ("SMARD", "grid-load bias"): smard.loc[acc.common, "err_grid_load"]
        .groupby(month_of)
        .mean(),
        ("SMARD", "MAE without bias"): (
            smard_error - smard_error.groupby(month_of).transform("mean")
        )
        .abs()
        .groupby(month_of)
        .mean(),
        ("SMARD", "persistence"): smard_daily.groupby(
            smard_daily.index.to_period("M")
        ).apply(lambda days: days.autocorr(1)),
    }
    for row in PICKS:
        columns[(label(row), "MAE")] = acc.errors[row].abs().groupby(month_of).mean()
        columns[(label(row), "bias")] = acc.errors[row].groupby(month_of).mean()
    monthly = pd.concat(columns, axis=1).loc[acc.full_months]
    monthly.index = monthly.index.strftime("%b %Y")
    formats = {
        c: {
            "bias": "{:+,.0f}",
            "grid-load bias": "{:+,.0f}",
            "persistence": "{:.2f}",
        }.get(c[1], "{:,.0f}")
        for c in monthly.columns
    }
    st.dataframe(monthly.style.format(formats))
    st.caption(
        "MAE and bias in MWh on the common test hours (bias = forecast − actual, positive = too "
        "high). *MAE without bias*: SMARD's error after removing its own monthly bias. "
        "*Persistence*: how closely SMARD's daily mean error follows the day before (1 = identical, "
        "0 = unrelated) — our models see SMARD's error of the last 24 h."
    )
    worst = first_skill.idxmin()
    in_month = acc.common[month_of == worst]
    daily_mae = pd.DataFrame(
        {
            name: acc.errors[row][in_month].abs().groupby(in_month.normalize()).mean()
            for name, row in [("ours", FIRST), ("smard", SMARD_ROW)]
        }
    )
    st.markdown(
        f"Worst full month of {label(FIRST)}: **{worst.strftime('%b %Y')}** ({first_skill[worst]:+.1f} %), worse "
        f"than SMARD on {int((daily_mae['ours'] > daily_mae['smard']).sum())} of {len(daily_mae)} days."
    )

st.divider()

# ============================================================================
# Accuracy by situation (§5–§8)
# ============================================================================
st.header("Accuracy by situation")
st.markdown(
    "The average hides the hours that matter most for the grid. Each tab looks at one kind of "
    "hour, chosen by the **actual** residual load of the test year. A model is a pick there only "
    f"if it beats SMARD both on the hours the actual falls in the band and on the hours it "
    f"*forecasts* the band, and forecasts the band at least {MIN_BIN_SHARE:.0%} as often as it occurs."
)

TAIL_EDGE = {
    "below_zero": (ZERO, "low"),
    "low_extreme": (acc.edges["P1"], "low"),
    "high_extreme": (acc.edges["P99"], "high"),
}
RANGE_LEVEL = 0.8  # share of the hours inside each error bar


def error_ranges(category, rows):
    """Spec 09 §5.2: middle-80 % error bar and bias diamond per row on the category's hours."""
    in_bin = acc.bin_hours[category]
    tail = (1 - RANGE_LEVEL) / 2
    fig = go.Figure()
    for row in rows:
        err = acc.errors[row][in_bin]
        low, high = np.quantile(err, [tail, 1 - tail])
        style = MODEL_STYLE[row[0]]
        fig.add_trace(
            go.Bar(
                y=[label(row)],
                x=[high - low],
                base=[low],
                orientation="h",
                showlegend=False,
                marker={
                    "color": style["color"],
                    "opacity": 0.3,
                    "line": {"color": style["color"], "width": 1.5},
                },
                hovertemplate=f"{label(row)}<br>middle {RANGE_LEVEL:.0%}: {low:+,.0f} .. {high:+,.0f} MWh<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                y=[label(row)],
                x=[err.mean()],
                mode="markers",
                showlegend=False,
                marker={
                    "symbol": "diamond",
                    "size": 12,
                    "color": "white",
                    "line": {"color": style["color"], "width": 2},
                },
                hovertemplate=f"{label(row)}<br>bias {err.mean():+,.0f} MWh<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
    style_plotly(
        fig,
        f"Error range, {BIN_LABEL[category].lower()} ({in_bin.sum():,} h)",
        "",
        f"error (MWh), forecast − actual · ← too low | too high →",
        height=160 + 60 * len(rows),
    )
    fig.update_layout(hovermode="closest", bargap=0.5)
    fig.update_xaxes(tickformat="+,.0f", showgrid=True, gridcolor=COLORS["grid"])
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def flagging_view(category, rows):
    """Spec 09 §6.1: forecast against actual on the hours where either lies in the band."""
    edge, side = TAIL_EDGE[category]
    actual_in = acc.bin_hours[category]
    actual_common = acc.actual[acc.common]
    forecast_in = {
        row: acc.bin_masks(acc.forecasts[row][acc.common])[category] for row in rows
    }
    shown = {row: actual_in | forecast_in[row] for row in rows}
    values = pd.concat(
        [
            pd.concat([actual_common[shown[r]], acc.forecasts[r][acc.common][shown[r]]])
            for r in rows
        ]
    )
    pad = 0.05 * (values.max() - values.min())
    lo, hi = values.min() - pad, values.max() + pad
    inside, outside = (
        ((lo, edge), (edge, hi)) if side == "low" else ((edge, hi), (lo, edge))
    )

    fig = make_subplots(
        rows=1,
        cols=len(rows),
        subplot_titles=[label(r) for r in rows],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )
    for col, row in enumerate(rows, start=1):
        is_actual, is_forecast = actual_in[shown[row]], forecast_in[row][shown[row]]
        counts = {
            "hits": (inside, inside, int((is_actual & is_forecast).sum())),
            "misses": (inside, outside, int((is_actual & ~is_forecast).sum())),
            "false alarms": (outside, inside, int((~is_actual & is_forecast).sum())),
        }
        corner = inside
        fig.add_shape(
            type="rect",
            x0=corner[0],
            x1=corner[1],
            y0=corner[0],
            y1=corner[1],
            fillcolor=BIN_COLOR[category],
            opacity=0.3,
            line_width=0,
            layer="below",
            row=1,
            col=col,
        )
        fig.add_shape(
            type="line",
            x0=lo,
            x1=hi,
            y0=lo,
            y1=hi,
            line={"color": COLORS["muted"], "width": 1},
            row=1,
            col=col,
        )
        fig.add_vline(
            x=edge, line_color=BIN_COLOR[category], line_width=2, row=1, col=col
        )
        fig.add_hline(
            y=edge, line_color=BIN_COLOR[category], line_width=2, row=1, col=col
        )
        fig.add_trace(
            go.Scattergl(
                x=actual_common[shown[row]],
                y=acc.forecasts[row][acc.common][shown[row]],
                mode="markers",
                marker={
                    "size": 4,
                    "color": MODEL_STYLE[row[0]]["color"],
                    "opacity": 0.5,
                },
                showlegend=False,
                hovertemplate="actual %{x:,.0f} MWh<br>forecast %{y:,.0f} MWh<extra></extra>",
            ),
            row=1,
            col=col,
        )
        for name, (xs, ys, n) in counts.items():
            fig.add_annotation(
                x=sum(xs) / 2,
                y=sum(ys) / 2,
                text=f"{name}<br><b>{n:,}</b>",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                row=1,
                col=col,
            )
    style_plotly(
        fig,
        f"Did the forecast see it coming? {BIN_LABEL[category]}",
        "forecast (MWh)",
        "actual (MWh)",
        height=430,
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[lo, hi], tickformat=",.0f")
    fig.update_yaxes(range=[lo, hi])
    return fig


tabs = st.tabs([BIN_LABEL[c] for c in BIN_CATEGORIES])
for tab, category in zip(tabs, BIN_CATEGORIES):
    with tab:
        picks = acc.picks[category]
        in_bin = acc.bin_hours[category]
        st.caption(
            f"Actual residual load {acc.bin_rule(category)}: {in_bin.sum():,} of {len(acc.common):,} test hours."
        )
        if not picks:
            st.info(
                f"No `{CANDIDATE_SPLIT}` model beats SMARD here. Why each one fails:\n\n"
                + "\n".join(
                    f"- {label(row)}: {', '.join(rules)}"
                    for row, rules in acc.rejected[category].items()
                )
            )
        else:
            smard_bin = acc.value.loc[SMARD_ROW, f"MAE_{category}_by_actual"]
            for card, row in zip(st.columns(len(picks)), picks):
                bin_mae = acc.value.loc[row, f"MAE_{category}_by_actual"]
                card.metric(
                    label(row),
                    f"{bin_mae:,.0f} MWh",
                    f"{bin_mae - smard_bin:+,.0f} MWh vs SMARD",
                    delta_color="inverse",
                    help=(
                        f"Average error on these hours (SMARD {smard_bin:,.0f} MWh). Forecast share "
                        f"{acc.forecast_share(row, category):.2f} (SMARD {acc.forecast_share(SMARD_ROW, category):.2f}): "
                        "how often the model forecasts this band relative to how often it occurs; 1 = as often."
                    ),
                )
        charts = st.container()
        rows = compare(f"bin_{category}", picks)
        shown_rows = [SMARD_ROW, *rows]
        with charts:
            if category == "ordinary":
                hour = pd.Series(acc.common.hour, index=acc.common)[in_bin]
                fig = go.Figure()
                for row in shown_rows:
                    by_hour = acc.errors[row][in_bin].abs().groupby(hour).mean()
                    fig.add_trace(
                        go.Scatter(
                            x=by_hour.index,
                            y=by_hour.to_numpy(),
                            mode="lines+markers",
                            name=label(row),
                            line=line(row),
                            hovertemplate="%{y:,.0f} MWh",
                        )
                    )
                style_plotly(
                    fig,
                    "Average error by hour of day, ordinary hours",
                    "MAE (MWh)",
                    "hour of day (local time)",
                )
                fig.update_xaxes(dtick=1)
                st.plotly_chart(fig, key=f"ordinary_hour_{category}")
            else:
                st.plotly_chart(
                    flagging_view(category, shown_rows), key=f"flagging_{category}"
                )
                st.caption(
                    "One dot per hour where the actual **or** the forecast is in the band. Tinted corner: "
                    "both in the band (hits). Actual in the band only: misses. Forecast only: false "
                    "alarms. Dots above the diagonal are over-forecasts."
                )
            st.plotly_chart(
                error_ranges(category, shown_rows), key=f"ranges_{category}"
            )
            st.caption(
                f"Each bar holds the error on the middle {RANGE_LEVEL:.0%} of these hours; the diamond is "
                "the average error (bias). A shorter bar is a tighter forecast."
            )

st.divider()

# ============================================================================
# Forecast explorer
# ============================================================================
with st.expander("Forecast explorer: the whole test year, hour by hour"):
    controls = st.columns([1, 2])
    best, worst = first_skill.idxmax(), first_skill.idxmin()
    presets = {
        "Whole test year": (acc.common[0], acc.common[-1]),
        f"Last full week ({full_weeks[-1]:%d %b %Y})": (
            full_weeks[-1],
            full_weeks[-1] + pd.Timedelta(days=7),
        ),
        f"Best month ({best.strftime('%b %Y')})": (best.start_time, best.end_time),
        f"Worst month ({worst.strftime('%b %Y')})": (worst.start_time, worst.end_time),
    }
    # Spec 10's zoom weeks, when the risk-label export exists (the Risk days page shows them too)
    try:
        zooms = default_zoom_weeks(load_risk_labels())
    except FileNotFoundError:
        zooms = {}
    for direction, weeks in zooms.items():
        for monday, reason in weeks:
            if monday is not None:
                presets[
                    f"{direction.capitalize()} risk: {reason} ({monday:%d %b %Y})"
                ] = (
                    monday,
                    monday + pd.Timedelta(days=7),
                )
    preset = controls[0].selectbox("Window", list(presets))
    rows = controls[1].multiselect(
        "Models", ALL_ROWS, default=[FIRST], format_func=label, key="explorer"
    )
    start, end = presets[preset]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.06,
    )
    hours = acc.common
    fig.add_trace(
        go.Scatter(
            x=hours,
            y=acc.actual[hours],
            name=model_label("actual"),
            line=model_line("actual", width=1.6),
        ),
        row=1,
        col=1,
    )
    for row in [SMARD_ROW, *rows]:
        name = model_label("smard") if row == SMARD_ROW else label(row)
        style = model_line("smard") if row == SMARD_ROW else line(row)
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=acc.forecasts[row][hours],
                name=name,
                line=style,
                legendgroup=name,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=hours,
                y=acc.errors[row],
                name=name,
                line=style,
                legendgroup=name,
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    style_plotly(
        fig, "Actual, SMARD and our forecast", "residual load (MWh)", height=640
    )
    fig.update_yaxes(title_text="error (MWh), forecast − actual", row=2, col=1)
    fig.update_xaxes(range=[start, end])
    fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.06}, row=2, col=1)
    st.plotly_chart(fig, key="explorer_chart")
    st.caption(
        "Drag on the chart to zoom, double-click to reset, click a legend entry to hide a series."
    )
