"""Do we beat SMARD? — our day-ahead residual-load forecast against SMARD's.

Ported from `notebooks/05_modeling/visualization-01-regression-best-models.ipynb` (spec 09) §3–§9.1,
viz-02's risk scores (spec 10 §3) and viz-03's cost export (spec 11), retold for a general audience
in five blocks (team decision 2026-10-07): the claim; is it real (month, running total, hour of
day); when it matters most (the situations, our best at each extreme, the risk days); how sure are
we (the 95 % range, one model or a team); and a wrap-up. The interactive tools (head-to-head, the
evening before, the week explorer) are on "Try it yourself". Every number is computed from the
exports through `model_results.py`; nothing is copied from the notebooks' dated findings.

The line-up (our best overall and at each extreme) and the "Compare models" switch live in
`components/lineup.py`, shared with "Try it yourself".
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.layout import header, next_page, title_of
from components.lineup import (
    DIRECTION_NAME,
    OURS_NAME,
    SIDE_CATEGORY,
    SIDE_HOURS,
    line,
    load_lineup,
    model_switch,
)
from components.lineup import row_name as lineup_row_name
from components.lineup import win_loss_bars as lineup_win_loss_bars
from components.naming import (
    ENSEMBLE_IDEA,
    MODEL_IDEA,
    SITUATION,
    bar_style,
    color,
    label,
    label_in_text,
    marker,
    money,
)
from components.risk_view import risk_days_section
from model_results import (
    BIN_CATEGORIES,
    CANDIDATE_SPLIT,
    MIN_BIN_SHARE,
    SMARD_ROW,
    ZERO,
    coverage,
    load_export,
    load_rebap_cost,
    load_risk_labels,
    risk_scores,
)
from viz_helpers import BIN_COLOR, COLORS, INK, model_pattern, style_plotly, themed

st.set_page_config(
    page_title="Do we beat SMARD? — Grid Stress", page_icon="🏁", layout="wide"
)

LINEUP = load_lineup()
acc = LINEUP.acc
PICKS, FIRST, ENSEMBLE, LEAD = LINEUP.picks, LINEUP.first, LINEUP.ensemble, LINEUP.lead
ALL_ROWS, SIDE_BEST, DEFAULT_ROWS = (
    LINEUP.all_rows,
    LINEUP.side_best,
    LINEUP.default_rows,
)
SMARD_ABS, TEST_LABEL = LINEUP.smard_abs, LINEUP.test_label
monthly_skill = LINEUP.monthly_skill


def row_name(row, sep=" "):
    """Legend name: model and role tag of a headline row in the default view, else the label."""
    return lineup_row_name(LINEUP, row, comparing, sep)


def win_loss_bars(x, values, row, hover, showlegend=False):
    """Bars in the row's colour where it beats SMARD, slate where SMARD is closer."""
    return lineup_win_loss_bars(x, values, row, row_name(row), hover, showlegend)


def longest_winning_run(rows):
    """The longest run of consecutive full months that every row in `rows` wins, or None."""
    wins = pd.concat([monthly_skill(row) > 0 for row in rows], axis=1).all(axis=1)
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
# Headline and proof
# ============================================================================
smard_mae = acc.value.loc[SMARD_ROW, "MAE"]
skill = acc.value.loc[LEAD, "skill_pct"]
won, months = acc.value.loc[LEAD, "months_beating_smard"], len(acc.full_months)

header(
    "Do we beat SMARD?",
    f"{OURS_NAME} is {skill:.1f} % smarter than SMARD: its misses are {skill:.1f} % smaller."
    + (
        f" Combining our models brings this to {acc.value.loc[ENSEMBLE, 'skill_pct']:.1f} %."
        if ENSEMBLE
        else ""
    ),
)
st.markdown(
    f"We forecast every hour of a full year our models never saw while being built ({TEST_LABEL}) "
    "and compared each hour with SMARD's official day-ahead forecast. Our models use SMARD's own "
    "forecast ingredients as inputs, so this measures **how much we improve the official "
    "forecast** — not a rival forecast built from scratch."
    + " We show our best models in three roles: the best overall, and the best at each of the two "
    "extremes (one model can hold two roles)."
)
st.page_link(
    "app_pages/method.py",
    label=f"How the test works: {title_of('app_pages/method.py')} →",
)


# --- One switch for the whole page ------------------------------------------------------------
comparing, ROWS = model_switch(
    LINEUP,
    "beat",
    "Off: our forecast against SMARD. On: pick any of the models we trained.",
)

with st.expander("Which models?"):
    st.markdown(
        f"**{OURS_NAME}** is **{label(FIRST)}**: the lowest average miss over all test hours "
        f"({acc.value.loc[FIRST, 'MAE']:,.0f} MWh) among the single models that beat SMARD."
    )
    if SIDE_BEST:
        st.markdown(
            " ".join(
                f"**Our best at {DIRECTION_NAME[side].lower()}** is **{label(row)}**: the lowest "
                f"average miss on the {SIDE_HOURS[side]} of hours, among the models that beat SMARD there."
                for side, row in SIDE_BEST.items()
            )
        )
    st.markdown(
        "\n".join(
            f"- {MODEL_IDEA[key]}"
            for key in dict.fromkeys(row[0] for row in ALL_ROWS)
            if key in MODEL_IDEA
        )
    )
    st.caption(
        "*Hybrid* models split the job: a straight line captures the trend, trees learn the rest."
    )

# --- Hero chart ------------------------------------------------------------------------------
bars = [SMARD_ROW, *ROWS]
fig = go.Figure(
    go.Bar(
        y=[row_name(row, "<br>") for row in bars],
        x=[acc.value.loc[row, "MAE"] for row in bars],
        orientation="h",
        marker={
            "color": [color(row) for row in bars],
            "pattern": {"shape": [model_pattern(row[0]) for row in bars]},
        },
        text=[
            f"{acc.value.loc[row, 'MAE']:,.0f} MWh"
            + (
                ""
                if row == SMARD_ROW
                else f" · {acc.value.loc[row, 'skill_pct']:+.1f} % smarter"
            )
            for row in bars
        ],
        textposition="outside",
        hovertemplate="%{y}: %{x:,.0f} MWh<extra></extra>",
    )
)
style_plotly(
    fig,
    "Average miss per hour, test year",
    "",
    xlabel="MWh",
    height=170 + 55 * len(bars),
)
fig.update_xaxes(range=[0, smard_mae * 1.45])
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_layout(hovermode="closest", showlegend=False)
st.plotly_chart(themed(fig), key="hero")


def headline_tiles(row, name):
    """One row of four tiles for a headline row, labelled with its case name."""
    mae = acc.value.loc[row, "MAE"]
    saved = float((SMARD_ABS - acc.errors[row].abs()).sum())
    cells = st.columns([1.3, 2, 2, 2, 2])
    cells[0].markdown(f"**{name}**  \n{label(row)}")
    cells = cells[1:]
    cells[0].metric(
        "Average miss per hour",
        f"{mae:,.0f} MWh",
        f"{mae - smard_mae:+,.0f} MWh vs SMARD",
        delta_color="inverse",
        help=f"Average miss (mean absolute error) of {label(row)} over {len(acc.common):,} test hours.",
    )
    cells[1].metric(
        "Smarter than SMARD",
        f"{acc.value.loc[row, 'skill_pct']:+.1f} %",
        help="How much of SMARD's miss we remove. 15 % smarter: where SMARD misses by 100, "
        f"we miss by 85. More on '{title_of('app_pages/method.py')}'.",
    )
    cells[2].metric(
        "Months ahead of SMARD",
        f"{acc.value.loc[row, 'months_beating_smard']:.0f} of {months}",
        help="Full calendar months in which our average miss was smaller than SMARD's.",
    )
    cells[3].metric(
        "Error saved over the year",
        f"{saved / 1e6:,.1f} TWh",
        help=f"{saved:,.0f} MWh: the sum over all test hours of SMARD's miss minus ours — as much "
        f"as SMARD's miss in {saved / smard_mae:,.0f} average hours.",
    )


headline_tiles(FIRST, OURS_NAME)

try:
    risk_labels = load_risk_labels()
except FileNotFoundError:
    risk_labels = None  # the data check on Team / About lists the missing export
risk_ready = risk_labels is not None and not risk_labels.problems
st.page_link(
    "app_pages/try_it.py",
    label="Put any two forecasters head to head, or pick a day or a week: Try it yourself →",
)

st.divider()

# ============================================================================
# Is it luck? Month by month (spec 09 §4.1)
# ============================================================================
st.header("Is it luck? Month by month")
month_x = acc.full_months.to_timestamp()
lead_skill = monthly_skill(LEAD)
fig = go.Figure()
if not comparing:
    for row in ROWS:
        fig.add_trace(
            win_loss_bars(
                month_x,
                monthly_skill(row).to_numpy(),
                row,
                f"{row_name(row)}: %{{y:+.1f}} %<extra></extra>",
                showlegend=len(ROWS) > 1,
            )
        )
    fig.update_layout(barmode="group")
else:
    for row in ROWS:
        fig.add_trace(
            go.Scatter(
                x=month_x,
                y=monthly_skill(row).to_numpy(),
                mode="lines+markers",
                name=label(row),
                line=line(row),
                marker=marker(row),
                hovertemplate="%{y:+.1f} %",
            )
        )
fig.add_hline(
    y=0,
    line={"color": COLORS["muted"], "width": 1},
    annotation_text="SMARD",
    annotation_font_color=INK,
)
run = longest_winning_run(ROWS)
if run is not None:
    n_run = (run[1] - run[0]).n + 1
    fig.add_vrect(
        x0=run[0].start_time - pd.Timedelta(days=15),
        x1=run[1].start_time + pd.Timedelta(days=15),
        fillcolor=COLORS["grid"],
        opacity=0.6,
        line_width=0,
        layer="below",
        annotation_text=f"{n_run} months in a row ahead",
        annotation_position="top left",
        annotation_font_color=INK,
    )
style_plotly(
    fig,
    f"How much smarter than SMARD, per month ({months} full months)",
    "% smarter than SMARD",
)
fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%b %Y")
fig.update_yaxes(tickformat="+.0f")
if not comparing:
    fig.update_layout(hovermode="closest")
st.plotly_chart(themed(fig), key="monthly_chart")

loss_months = lead_skill[lead_skill < 0]
bullets = [
    f"- **{OURS_NAME} is ahead of SMARD in {won:.0f} of {months} full months** — best in "
    f"{lead_skill.idxmax().strftime('%B %Y')} ({lead_skill.max():+.1f} %)."
]
if len(loss_months):
    bullets.append(
        "- **SMARD was closer in "
        + ", ".join(
            f"{m.strftime('%B %Y')} ({v:+.1f} %)" for m, v in loss_months.items()
        )
        + "** — no model wins every month. The monthly numbers at the bottom of the page show "
        "how SMARD's own error behaved then."
    )
st.markdown("\n".join(bullets))
st.caption(
    "Bars above the line: our forecast was closer that month. Below (grey): SMARD was closer."
)

# ============================================================================
# Where does the lead come from? (§4.2) — At what time of day? (§4.3)
# ============================================================================
left, right = st.columns(2)
with left:
    st.subheader("Where does the lead come from?")
    fig = go.Figure()
    for row in ROWS:
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
    monthly_gain = (
        (SMARD_ABS - acc.errors[LEAD].abs()).groupby(acc.common.to_period("M")).sum()
    )
    steepest = monthly_gain.idxmax()
    running = (SMARD_ABS - acc.errors[LEAD].abs()).cumsum()
    at = running[running.index.to_period("M") == steepest].index[-1]
    fig.add_annotation(
        x=at,
        y=running[at],
        text=f"{OURS_NAME.lower()}'s biggest gain: {steepest.strftime('%b %Y')}",
        showarrow=True,
        arrowhead=0,
        ay=-40,
        font={"color": INK},
    )
    style_plotly(
        fig, "Error saved against SMARD, running total", "MWh saved (running total)"
    )
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")
    st.plotly_chart(themed(fig), key="cumulative_chart")
    st.caption(
        "A steady slope means a steady gain; steps mean a few episodes carry it."
    )

with right:
    st.subheader("At what time of day?")
    hour_of = acc.common.hour
    hours = [f"{h:02d}:00" for h in range(24)]
    fig = go.Figure()
    if not comparing:
        for row in ROWS:
            fig.add_trace(
                win_loss_bars(
                    hours,
                    acc.skill(row, hour_of).to_numpy(),
                    row,
                    f"{row_name(row)}: %{{y:+.1f}} %<extra></extra>",
                    showlegend=len(ROWS) > 1,
                )
            )
        fig.update_layout(barmode="group")
    else:
        for row in ROWS:
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=acc.skill(row, hour_of).to_numpy(),
                    mode="lines+markers",
                    name=label(row),
                    line=line(row),
                    marker=marker(row),
                    hovertemplate="%{y:+.1f} %",
                )
            )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
    style_plotly(
        fig,
        "How much smarter than SMARD, by hour of day",
        "% smarter than SMARD",
        "hour of day",
    )
    fig.update_yaxes(tickformat="+.0f")
    st.plotly_chart(themed(fig), key="hour_chart")
    first_by_hour = acc.skill(LEAD, hour_of)
    st.caption(
        f"{OURS_NAME} is ahead in {int((first_by_hour > 0).sum())} of 24 hours — most at "
        f"{first_by_hour.idxmax():02d}:00 ({first_by_hour.max():+.1f} %), least at "
        f"{first_by_hour.idxmin():02d}:00 ({first_by_hour.min():+.1f} %)."
    )

st.divider()

# ============================================================================
# When it matters most (§5–§8)
# ============================================================================
st.header("When it matters most")
st.markdown(
    "The average hides the hours that matter most for the grid. Here every test hour is sorted by "
    "what actually happened: normal hours, hours when wind and solar exceed demand, and the most extreme "
    "hours on either side."
)

categories = ["overall", *BIN_CATEGORIES]
metric = {c: "MAE" if c == "overall" else f"MAE_{c}_by_actual" for c in categories}
hour_counts = {
    c: len(acc.common) if c == "overall" else int(acc.bin_hours[c].sum())
    for c in categories
}
x = [f"{SITUATION[c]}<br>({hour_counts[c]:,} h)" for c in categories]
fig = go.Figure()
for row in [SMARD_ROW, *ROWS]:
    fig.add_trace(
        go.Bar(
            x=x,
            y=[acc.value.loc[row, metric[c]] for c in categories],
            name=row_name(row),
            marker=bar_style(row),
            hovertemplate="%{y:,.0f} MWh",
        )
    )
style_plotly(fig, "Average miss per hour, by situation", "MWh", height=460)
fig.update_layout(barmode="group")
st.plotly_chart(themed(fig), key="situations")
st.caption(
    "Lower is better. The extremes are harder for everyone — compare the bars within each group, "
    "not across groups."
)


def tail_counts(category, row):
    """Hits, misses and false alarms of a row in a tail band (spec 09 §6.1)."""
    actual_in = acc.bin_hours[category]
    forecast_in = acc.bin_masks(acc.forecasts[row][acc.common])[category]
    return {
        "saw it coming": int((actual_in & forecast_in).sum()),
        "missed it": int((actual_in & ~forecast_in).sum()),
        "false alarm": int((~actual_in & forecast_in).sum()),
    }


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
        fig.add_trace(
            go.Bar(
                y=[label(row)],
                x=[high - low],
                base=[low],
                orientation="h",
                showlegend=False,
                marker={
                    "color": color(row),
                    "opacity": 0.3,
                    "line": {"color": color(row), "width": 1.5},
                },
                hovertemplate=f"{label(row)}<br>middle {RANGE_LEVEL:.0%}: {low:+,.0f} to {high:+,.0f} MWh<extra></extra>",
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
                    "line": {"color": color(row), "width": 2},
                },
                hovertemplate=f"{label(row)}<br>leans {err.mean():+,.0f} MWh<extra></extra>",
            )
        )
    fig.add_vline(x=0, line_color=COLORS["muted"], line_width=1)
    style_plotly(
        fig,
        f"Spread of the misses, {SITUATION[category].lower()}",
        "",
        "miss (MWh) · ← forecast too low | too high →",
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
            "saw it coming": (inside, inside, int((is_actual & is_forecast).sum())),
            "missed it": (inside, outside, int((is_actual & ~is_forecast).sum())),
            "false alarms": (outside, inside, int((~is_actual & is_forecast).sum())),
        }
        fig.add_shape(
            type="rect",
            x0=inside[0],
            x1=inside[1],
            y0=inside[0],
            y1=inside[1],
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
                marker={"size": 4, "color": color(row), "opacity": 0.5},
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
        f"Each hour as a dot: {SITUATION[category].lower()}",
        "forecast (MWh)",
        "actual (MWh)",
        height=430,
    )
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(range=[lo, hi], tickformat=",.0f")
    fig.update_yaxes(range=[lo, hi])
    return fig


tabs = st.tabs([SITUATION[c] for c in BIN_CATEGORIES])
for tab, category in zip(tabs, BIN_CATEGORIES):
    with tab:
        picks = acc.picks[category]
        in_bin = acc.bin_hours[category]
        smard_bin = acc.value.loc[SMARD_ROW, f"MAE_{category}_by_actual"]
        ours_bin = acc.value.loc[LEAD, f"MAE_{category}_by_actual"]
        st.caption(
            f"Hours with actual residual load {acc.bin_rule(category)}: {in_bin.sum():,} of "
            f"{len(acc.common):,} test hours."
        )
        summary = (
            f"Average miss here: **{OURS_NAME.lower()} {ours_bin:,.0f} MWh**, SMARD {smard_bin:,.0f} MWh "
            f"({100 * (1 - ours_bin / smard_bin):+.1f} %)."
        )
        if picks and picks[0] not in DEFAULT_ROWS:
            best = acc.value.loc[picks[0], f"MAE_{category}_by_actual"]
            summary += f" The best model for this situation is **{label(picks[0])}** ({best:,.0f} MWh)."
        elif not picks:
            summary += " No model beats SMARD reliably here."
        st.markdown(summary)
        rows = [
            SMARD_ROW,
            *(ROWS if comparing else dict.fromkeys([*ROWS, *picks[:1]])),
        ]

        if category == "ordinary":
            hour = pd.Series(acc.common.hour, index=acc.common)[in_bin]
            fig = go.Figure()
            for row in rows:
                by_hour = acc.errors[row][in_bin].abs().groupby(hour).mean()
                fig.add_trace(
                    go.Scatter(
                        x=[f"{h:02d}:00" for h in by_hour.index],
                        y=by_hour.to_numpy(),
                        mode="lines+markers",
                        name=label(row),
                        line=line(row),
                        hovertemplate="%{y:,.0f} MWh",
                    )
                )
            style_plotly(
                fig, "Average miss by hour of day, normal hours", "MWh", "hour of day"
            )
            st.plotly_chart(themed(fig), key=f"ordinary_hour_{category}")
        else:
            fig = go.Figure()
            for row in rows:
                counts = tail_counts(category, row)
                fig.add_trace(
                    go.Bar(
                        x=list(counts),
                        y=list(counts.values()),
                        name=row_name(row),
                        marker=bar_style(row),
                        text=list(counts.values()),
                        textposition="outside",
                        hovertemplate=f"{row_name(row)}: %{{y:,}} hours<extra></extra>",
                    )
                )
            style_plotly(
                fig, "Did the forecast see these hours coming?", "hours", height=380
            )
            fig.update_layout(barmode="group", hovermode="closest")
            st.plotly_chart(themed(fig), key=f"counts_{category}")
            st.caption(
                "*Saw it coming*: the forecast was in this range too. *Missed it*: it happened, the forecast "
                "didn't expect it. *False alarm*: the forecast expected it, it didn't happen."
            )
        with st.expander(
            "More detail: spread of the misses"
            + ("" if category == "ordinary" else ", each hour as a dot")
        ):
            if category != "ordinary":
                st.plotly_chart(
                    themed(flagging_view(category, rows)), key=f"flagging_{category}"
                )
            st.plotly_chart(
                themed(error_ranges(category, rows)), key=f"ranges_{category}"
            )
            st.caption(
                f"Each bar holds the miss on the middle {RANGE_LEVEL:.0%} of these hours; the diamond is the "
                "average miss (does the forecast lean high or low?). A shorter bar is a tighter forecast."
            )


# --- Our best at each extreme and the risk days (spec 10) ----------------------------------------
def side_tiles(side):
    """One row of four tiles for our best model at one extreme, scored on that extreme's hours."""
    row, category = SIDE_BEST[side], SIDE_CATEGORY[side]
    mae = acc.value.loc[row, f"MAE_{category}_by_actual"]
    smard_side = acc.value.loc[SMARD_ROW, f"MAE_{category}_by_actual"]
    cells = st.columns([1.3, 2, 2, 2, 2])
    also = " (also our best overall)" if row == FIRST else ""
    cells[0].markdown(
        f"**Our best at {DIRECTION_NAME[side].lower()}**  \n{label(row)}{also}"
    )
    cells[1].metric(
        f"Average miss, {SIDE_HOURS[side]} of hours",
        f"{mae:,.0f} MWh",
        f"{mae - smard_side:+,.0f} MWh vs SMARD",
        delta_color="inverse",
        help=f"Average miss (mean absolute error) on the {int(acc.bin_hours[category].sum())} test hours with "
        f"actual residual load {acc.bin_rule(category)}.",
    )
    cells[2].metric(
        "Smarter than SMARD there",
        f"{100 * (1 - mae / smard_side):+.1f} %",
        help="How much of SMARD's miss it removes on these hours.",
    )
    cells[3].metric(
        "Smarter than SMARD overall",
        f"{acc.value.loc[row, 'skill_pct']:+.1f} %",
        help="On all test hours, for comparison with our best model overall.",
    )
    # Risk days: spec 10 scores its own picks; shown only when they are this model
    basis = "rolling" if side == "high" else "zero"
    if risk_ready and risk_labels.picks[side] == row[0]:
        scores = risk_scores(risk_labels, side, basis, "any")
        cells[4].metric(
            "Risk days it saw coming",
            f"{scores['model']['hit days']} of {scores['model']['actual days']}",
            f"{scores['model']['hit days'] - scores['smard']['hit days']:+d} days vs SMARD",
            help="Rule: at least one hour past the line"
            + (" (zero)" if basis == "zero" else "")
            + ". Details under 'Did we see the risk days coming?'.",
        )


if SIDE_BEST:
    st.subheader("Our best at each extreme")
    st.markdown("Scored on the hours of that extreme only:")
    for side in SIDE_BEST:
        side_tiles(side)
    if risk_ready and any(
        risk_labels.picks[side] != row[0] for side, row in SIDE_BEST.items()
    ):
        st.caption(
            "The risk-day scores below use other models than these — re-run "
            "visualization-02-classification-risk-labels.ipynb with the current picks."
        )

# --- Risk days (spec 10, the former Risk days page via components/risk_view.py) ---------------
if risk_labels is not None and risk_labels.problems:
    st.warning(
        "The risk-day section is not shown: the risk-label export doesn't match the model "
        "exports.\n\n" + "\n".join(f"- {problem}" for problem in risk_labels.problems)
    )
if risk_ready:
    st.subheader("Did we see the risk days coming?")
    risk_days_section(risk_labels)
    # The closing summary's numbers: high `rolling` + low `zero`, rule `any`
    risk_totals = {
        direction: risk_scores(risk_labels, direction, basis, "any")
        for direction, basis in [("high", "rolling"), ("low", "zero")]
    }

with st.expander("How we pick the best model for each situation"):
    st.markdown(
        f"A model counts as better than SMARD in a situation only if it misses less both on the hours "
        f"that really were in that range **and** on the hours it *forecast* to be in it — and it forecasts "
        f"the range at least {MIN_BIN_SHARE:.0%} as often as it happens. That keeps a model from looking "
        "good by simply never forecasting extremes."
    )

st.divider()

# ============================================================================
# How sure are we? (spec 08 Behaviour 15: the 95 % band and its coverage)
# ============================================================================
BAND_LEVEL = 95  # % of hours the band promises to hold (spec 06 Behaviour 20)
st.header("How sure are we?")
lead_band = acc.bands[LEAD]
actual_common = acc.actual[acc.common]
lead_coverage = coverage(acc, LEAD)
above = 100 * float((actual_common > lead_band["upper"]).mean())
below = 100 * float((actual_common < lead_band["lower"]).mean())
width = float((lead_band["upper"] - lead_band["lower"]).mean())
if lead_coverage < BAND_LEVEL - 1:
    verdict = "a little less often than promised: the range is slightly too narrow"
elif lead_coverage > BAND_LEVEL + 1:
    verdict = "more often than promised: the range is wider than it needs to be"
else:
    verdict = "about as often as promised"
st.markdown(
    f"Every forecast comes with a **range that should hold reality in {BAND_LEVEL} % of hours**. "
    f"For {OURS_NAME.lower()}, reality fell inside it in **{lead_coverage:.1f} %** of the test "
    f"hours — {verdict}."
)
cells = st.columns(4)
cells[0].metric(
    "Reality inside the range",
    f"{lead_coverage:.1f} %",
    f"{lead_coverage - BAND_LEVEL:+.1f} points vs promised",
    delta_color="off",
)
cells[1].metric(
    "Average width",
    f"{width:,.0f} MWh",
    help="Upper minus lower edge, averaged over the test hours.",
)
cells[2].metric(
    "Reality above the range",
    f"{above:.1f} %",
    help=f"Hours where the actual was higher than the upper edge ({(100 - BAND_LEVEL) / 2:.1f} % "
    "if the range were exact).",
)
cells[3].metric(
    "Reality below the range",
    f"{below:.1f} %",
    help=f"Hours where the actual was lower than the lower edge ({(100 - BAND_LEVEL) / 2:.1f} % "
    "if the range were exact).",
)

cov_categories = ["overall", *BIN_CATEGORIES]
masks = {"overall": pd.Series(True, index=acc.common)} | acc.bin_hours
left, right = st.columns(2)
with left:
    fig = go.Figure()
    for row in ROWS:
        values = [
            coverage(acc, row, acc.common[masks[c].to_numpy()]) for c in cov_categories
        ]
        fig.add_trace(
            go.Bar(
                x=[
                    f"{SITUATION[c]}<br>({int(masks[c].sum()):,} h)"
                    for c in cov_categories
                ],
                y=values,
                name=row_name(row),
                marker=bar_style(row),
                hovertemplate="%{y:.1f} %",
            )
        )
    fig.add_hline(
        y=BAND_LEVEL,
        line={"color": COLORS["muted"], "width": 1, "dash": "dash"},
        annotation_text=f"promised {BAND_LEVEL} %",
        annotation_font_color=INK,
    )
    style_plotly(
        fig, "Reality inside the range, by situation", "% of hours", height=460
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[50, 100], tickformat=".0f")
    st.plotly_chart(themed(fig), key="coverage_situations")
    lead_by_situation = {
        c: coverage(acc, LEAD, acc.common[masks[c].to_numpy()]) for c in BIN_CATEGORIES
    }
    weakest = min(lead_by_situation, key=lead_by_situation.get)
    st.caption(
        f"Weakest for {OURS_NAME.lower()} in {SITUATION[weakest].lower()}: "
        f"{lead_by_situation[weakest]:.1f} % — the range's width depends only on the hour of day, "
        "not on the situation, so it holds least where the forecast is hardest."
    )
with right:
    month_of = acc.common.to_period("M")
    fig = go.Figure()
    for row in ROWS:
        by_month = pd.Series(
            {
                month: coverage(acc, row, acc.common[month_of == month])
                for month in acc.full_months
            }
        )
        fig.add_trace(
            go.Scatter(
                x=acc.full_months.to_timestamp(),
                y=by_month.to_numpy(),
                mode="lines+markers",
                name=row_name(row),
                line=line(row),
                marker=marker(row),
                hovertemplate="%{y:.1f} %",
            )
        )
    fig.add_hline(
        y=BAND_LEVEL,
        line={"color": COLORS["muted"], "width": 1, "dash": "dash"},
        annotation_text=f"promised {BAND_LEVEL} %",
        annotation_font_color=INK,
    )
    style_plotly(fig, "Reality inside the range, per month", "% of hours", height=460)
    fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%b %Y")
    fig.update_yaxes(range=[50, 100], tickformat=".0f")
    st.plotly_chart(themed(fig), key="coverage_months")
    st.page_link(
        "app_pages/try_it.py",
        label="See the range around the forecast for any week: Try it yourself →",
    )

with st.expander("How the range is built"):
    st.markdown(f"""
- On the year before the test year, we collect how far reality landed from each forecast, using
  forecasts the combination had not been fitted on (otherwise the misses would look too small).
- For every hour of the day, the range spans the middle {BAND_LEVEL} % of those misses, placed
  around the new forecast. A 3 a.m. forecast gets a 3 a.m. range, an evening forecast an evening one.
- The range is set once, before the test year, and never adjusted during it — so a test year that
  is harder than the year before shows up as a range that holds less often than promised.
- Every model and every ensemble gets its range the same way, so their coverage can be compared.
""")

st.divider()

# ============================================================================
# One model or a team of models? (spec 08 Behaviour 17: skill against the best member)
# ============================================================================
if ENSEMBLE:
    st.header("One model or a team of models?")
    team = [row for row in acc.ensembles if row[1] == CANDIDATE_SPLIT]
    # Against our best single model in each situation (the page's picks), computed here: the
    # export's skill_vs_best_pct uses spec 08's own "best member", not our best model overall
    mae_metric = {
        c: "MAE" if c == "overall" else f"MAE_{c}_by_actual" for c in cov_categories
    }
    best_in = {c: (acc.picks[c] or [FIRST])[0] for c in cov_categories}
    vs = pd.DataFrame(
        {
            c: 100
            * (
                1
                - acc.value.loc[team, mae_metric[c]]
                / acc.value.loc[best_in[c], mae_metric[c]]
            )
            for c in cov_categories
        }
    )
    vs_best = vs["overall"]
    st.markdown(
        f"An ensemble combines the forecasts of our single models — and of SMARD itself — with "
        f"weights learned on the year before. **The best of them, the {label_in_text(ENSEMBLE)}, is "
        f"{vs_best[ENSEMBLE]:.1f} % smarter than our best model overall**"
        + (
            f", and all {len(team)} ways of combining beat it "
            f"({vs_best.min():+.1f} to {vs_best.max():+.1f} %)."
            if (vs_best > 0).all()
            else "."
        )
    )
    st.markdown(
        "\n".join(
            f"- {ENSEMBLE_IDEA[row[0]]}" for row in team if row[0] in ENSEMBLE_IDEA
        )
    )
    fig = go.Figure()
    for row in sorted(team, key=lambda r: -vs_best[r]):
        fig.add_trace(
            go.Bar(
                x=[SITUATION[c] for c in cov_categories],
                y=[vs.loc[row, c] for c in cov_categories],
                customdata=[label(best_in[c]) for c in cov_categories],
                name=label(row),
                marker=bar_style(row),
                hovertemplate="%{y:+.1f} % vs %{customdata}",
            )
        )
    fig.add_hline(
        y=0,
        line={"color": COLORS["muted"], "width": 1},
        annotation_text="our best model there",
        annotation_font_color=INK,
    )
    style_plotly(
        fig,
        "How much smarter than our best model, by situation",
        "% smarter than our best model there",
        height=460,
    )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat="+.0f")
    st.plotly_chart(themed(fig), key="team_vs_best")
    lead_vs = {c: vs.loc[ENSEMBLE, c] for c in BIN_CATEGORIES}
    worst = min(lead_vs, key=lead_vs.get)
    st.caption(
        "Above zero: the ensemble beats our best single model in that situation (overall: our best "
        "model overall; at the extremes: our best at that side; hover for the model). "
        f"The {label_in_text(ENSEMBLE)}'s weakest spot: "
        f"{SITUATION[worst].lower()} ({lead_vs[worst]:+.1f} %). Same colour, different patterns: "
        "the four ensembles."
    )
    st.page_link(
        "app_pages/method.py",
        label=f"How each ensemble weighs its members: {title_of('app_pages/method.py')} → Combining models",
    )
    st.divider()

# ============================================================================
# Wrap-up: what this means, the reBAP teaser (spec 11), the monthly numbers (§9.1)
# ============================================================================
try:
    rebap = load_rebap_cost()
except FileNotFoundError:
    rebap = None  # the data check on Who are we? lists the missing export
if rebap is not None and (rebap.problems or ENSEMBLE not in rebap.saved):
    rebap = None

gains = {
    c: 100 * (1 - acc.value.loc[LEAD, metric[c]] / acc.value.loc[SMARD_ROW, metric[c]])
    for c in BIN_CATEGORIES
}
largest, smallest = max(gains, key=gains.get), min(gains, key=gains.get)
points = [
    f"On an average hour {OURS_NAME.lower()} is **{skill:.1f} % smarter than SMARD** (its misses are "
    f"{skill:.1f} % smaller), and is ahead in **{won:.0f} of {months}** months.",
    f"The gain is largest in **{SITUATION[largest].lower()}** ({gains[largest]:+.1f} %) and smallest in "
    f"**{SITUATION[smallest].lower()}** ({gains[smallest]:+.1f} %).",
]
if SIDE_BEST:
    points.append(
        "At the extremes: "
        + "; ".join(
            f"**our best at {DIRECTION_NAME[side].lower()}** ({label(row)}) is "
            f"**{100 * (1 - acc.value.loc[row, f'MAE_{SIDE_CATEGORY[side]}_by_actual'] / acc.value.loc[SMARD_ROW, f'MAE_{SIDE_CATEGORY[side]}_by_actual']):.1f} % "
            f"smarter than SMARD** on the {SIDE_HOURS[side]} of hours"
            for side, row in SIDE_BEST.items()
        )
        + "."
    )
if ENSEMBLE:
    points.append(
        f"Combining our models helps a little more: the {label_in_text(ENSEMBLE)} is "
        f"{100 * (1 - acc.value.loc[ENSEMBLE, 'MAE'] / acc.value.loc[FIRST, 'MAE']):.1f} % smarter than our best model overall"
        + (
            f" — but not everywhere: in {SITUATION[worst].lower()} it is {-lead_vs[worst]:.1f} % "
            "worse than our best model there."
            if lead_vs[worst] < 0
            else "."
        )
    )
if risk_labels is not None and not risk_labels.problems:
    high, low = risk_totals["high"], risk_totals["low"]
    points.append(
        f"We saw **{high['model']['hit days']} of {high['model']['actual days']}** risk days with too little "
        f"green power coming (SMARD {high['smard']['hit days']}) and **{low['model']['hit days']} of "
        f"{low['model']['actual days']}** with too much, below zero (SMARD {low['smard']['hit days']}), with "
        f"{high['model']['false-alarm days'] + low['model']['false-alarm days']} false alarms "
        f"(SMARD {high['smard']['false-alarm days'] + low['smard']['false-alarm days']})."
    )
if rebap is not None and ENSEMBLE in rebap.saved:
    points.append(
        f"Priced at the imbalance price, our ensemble's misses cost **{money(rebap.saved[ENSEMBLE].sum(), False)}** less "
        f"than SMARD's misses over the year ({100 * rebap.saved[ENSEMBLE].sum() / rebap.cost[SMARD_ROW].sum():.1f} %) "
        "— a yardstick, not a bill."
    )
points.append(
    f"Its {BAND_LEVEL} % range held reality in **{lead_coverage:.1f} %** of the test hours "
    f"({lead_coverage - BAND_LEVEL:+.1f} points)."
)
with st.container(border=True):
    st.subheader("What this means")
    st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1)))

if rebap is not None:
    with st.container(border=True):
        st.markdown(
            f"💶 **What is that worth?** Priced at the imbalance price, our ensemble's smaller "
            f"misses are worth about **{money(rebap.saved[ENSEMBLE].sum(), False)}** over the "
            "test year — a yardstick, not a bill."
        )
        st.page_link(
            "app_pages/rebap.py",
            label="Calculate it: What is it worth? →",
        )

# ============================================================================
# Monthly numbers (§9.1)
# ============================================================================
worst_month = lead_skill.idxmin()
with st.expander("Monthly numbers (for analysts)"):
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
    for row in dict.fromkeys([*DEFAULT_ROWS, *PICKS]):
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
        "MAE and bias in MWh on the common test hours (bias = forecast − actual, positive = too high). "
        "*MAE without bias*: SMARD's error after removing its own monthly bias. *Persistence*: how closely "
        "SMARD's daily mean error follows the day before (1 = identical, 0 = unrelated) — our models see "
        "SMARD's error of the last 24 h."
    )
    in_month = acc.common[month_of == worst_month]
    daily_mae = pd.DataFrame(
        {
            name: acc.errors[row][in_month].abs().groupby(in_month.normalize()).mean()
            for name, row in [("ours", LEAD), ("smard", SMARD_ROW)]
        }
    )
    st.markdown(
        f"Worst full month of {label(LEAD)}: **{worst_month.strftime('%b %Y')}** ({lead_skill[worst_month]:+.1f} %), "
        f"worse than SMARD on {int((daily_mae['ours'] > daily_mae['smard']).sum())} of {len(daily_mae)} days."
    )

next_page("app_pages/beat_smard.py")
