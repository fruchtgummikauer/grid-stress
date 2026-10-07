"""Where we beat SMARD — our day-ahead residual-load forecast against SMARD's.

Ported from `notebooks/05_modeling/visualization-01-regression-best-models.ipynb` (spec 09) §3–§9.1
and viz-02's risk scores (spec 10 §3), retold for a general audience: one headline claim and its
proof first, then month / running total / hour of day, the situations, an explorer, and a closing
summary. Every number is computed from the exports through `model_results.py`; nothing is copied
from the notebooks' dated findings.

The default view shows one model, "our forecast" = spec 09's overall rank 1 (ranked by months won,
then MAE). A page-wide switch opens the comparison of every model. Spec 08's ensemble is planned
(Streamlit-v3.md §1.4 blocks 5 and 8): its blocks are placeholders until `ensemble_*.csv` exist.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from components.layout import header, next_page
from model_results import (
    BIN_CATEGORIES,
    CANDIDATE_SPLIT,
    MIN_BIN_SHARE,
    SMARD_ROW,
    ZERO,
    default_zoom_weeks,
    ensemble_ready,
    get_or_stop,
    load_accuracy,
    load_risk_labels,
    load_export,
    risk_scores,
)
from viz_helpers import (
    BIN_COLOR,
    COLORS,
    INK,
    MODEL_STYLE,
    model_label,
    model_line,
    style_plotly,
)

st.set_page_config(
    page_title="Where we beat SMARD — Grid Stress", page_icon="🏁", layout="wide"
)

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
TEST_LABEL = f"{acc.test_days[0]:%d %b %Y} – {acc.test_days[-1]:%d %b %Y}"
SMARD_NAME = "SMARD (official)"
OURS_NAME = "Our forecast"

# Plain-language situation names for spec 09's bins (the shared BIN_LABEL keeps the statistics)
SITUATION = {
    "overall": "All hours",
    "ordinary": "Normal hours",
    "below_zero": "Green-power surplus (below zero)",
    "low_extreme": "Most extreme surplus (lowest 1 %)",
    "high_extreme": "Most extreme shortage (highest 1 %)",
}
# One line per model family, for the "Which models?" expander
MODEL_IDEA = {
    "linear_direct": "Ridge — a straight-line formula over all inputs.",
    "lgbm_direct": "LightGBM — many small decision trees, each correcting the ones before.",
    "xgb_direct": "XGBoost — the same idea as LightGBM, a different implementation.",
    "lgbm_hybrid": "LightGBM hybrid — a straight-line trend first, trees for what is left.",
    "xgb_hybrid": "XGBoost hybrid — a straight-line trend first, trees for what is left.",
    "random_forest_hybrid": "Random forest hybrid — a straight-line trend first, then many independent trees averaged.",
    "seasonal_naive": "Seasonal naive — simply repeats the same hour one week earlier (a sanity check).",
    "sarimax_fourier": "SARIMAX — a classic statistical time-series model.",
}


def label(row):
    """Display label of a (model, split_method) row; the split only when it is not the default."""
    if row == SMARD_ROW:
        return SMARD_NAME
    base = model_label(row[0])
    return base if row[1] in (CANDIDATE_SPLIT, "none") else f"{base} ({row[1]})"


def line(row, **overrides):
    """Plotly line of a row: the model's colour, dotted for a non-default split."""
    style = model_line("smard" if row == SMARD_ROW else row[0])
    if row != SMARD_ROW and row[1] not in (CANDIDATE_SPLIT, "none"):
        style["dash"] = "dot"
    return style | overrides


def color(row):
    """The row's colour (SMARD slate)."""
    return MODEL_STYLE["smard" if row == SMARD_ROW else row[0]]["color"]


def monthly_skill(row):
    """Skill vs SMARD (%) per full month."""
    return 100 * (1 - acc.monthly_mae[row] / acc.monthly_mae[SMARD_ROW])


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


def ensemble_slot(title, what):
    """A placeholder for a spec 08 ensemble block (Streamlit-v3.md §1.4)."""
    with st.container(border=True):
        st.markdown(f"**🔜 {title}**")
        st.markdown(what)
        if ensemble_ready():
            st.caption("The ensemble results are in — this block is next on our list.")
        else:
            st.caption("Coming soon: the ensemble results are being finalised.")


def win_loss_bars(x, values, row, hover):
    """Bars in the row's colour where it beats SMARD, slate where SMARD is closer."""
    return go.Bar(
        x=x,
        y=values,
        marker_color=np.where(values >= 0, color(row), COLORS["muted"]),
        name=label(row),
        hovertemplate=hover,
        showlegend=False,
    )


# ============================================================================
# Headline and proof
# ============================================================================
mae, smard_mae = acc.value.loc[FIRST, "MAE"], acc.value.loc[SMARD_ROW, "MAE"]
skill = acc.value.loc[FIRST, "skill_pct"]
won, months = acc.value.loc[FIRST, "months_beating_smard"], len(acc.full_months)
saved = float((SMARD_ABS - acc.errors[FIRST].abs()).sum())

header(
    "Where we beat SMARD",
    f"Our forecast misses {skill:.0f} % less than the official one.",
)
st.markdown(
    f"We forecast every hour of a full year our models never saw while being built ({TEST_LABEL}) "
    "and compared each hour with SMARD's official day-ahead forecast. Our models use SMARD's own "
    "forecast ingredients as inputs, so this measures **how much we improve the official "
    "forecast** — not a rival forecast built from scratch."
)
st.page_link("app_pages/method.py", label="How the test works: Method →")

# --- One switch for the whole page ------------------------------------------------------------
switch, picker = st.columns([1, 2])
comparing = switch.toggle(
    "Compare models",
    help="Off: our forecast against SMARD. On: pick any of the models we trained.",
)
if comparing:
    with picker:
        show_static = st.checkbox(
            "Include the 'static' variants",
            help="Static = trained once before the test year instead of being refitted every week.",
        )
        options = [
            row
            for row in ALL_ROWS
            if show_static or row[1] in (CANDIDATE_SPLIT, "none")
        ]
        ROWS = st.multiselect(
            "Models shown on every chart",
            options,
            default=PICKS,
            format_func=label,
            key="rows",
        ) or [FIRST]
else:
    ROWS = [FIRST]

with st.expander("Which models?"):
    st.markdown(
        f"**{OURS_NAME}** is **{label(FIRST)}**: of all models it is ahead of SMARD in the most months "
        f"({won:.0f} of {months}), the team's rule for the headline pick. "
        + (
            f"**{label(min(PICKS, key=lambda r: acc.value.loc[r, 'MAE']))}** has the lowest average miss "
            f"({acc.value.loc[PICKS, 'MAE'].min():,.0f} MWh) but wins fewer months."
            if min(PICKS, key=lambda r: acc.value.loc[r, "MAE"]) != FIRST
            else ""
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
        y=[
            OURS_NAME if (row == FIRST and not comparing) else label(row)
            for row in bars
        ],
        x=[acc.value.loc[row, "MAE"] for row in bars],
        orientation="h",
        marker_color=[color(row) for row in bars],
        text=[
            f"{acc.value.loc[row, 'MAE']:,.0f} MWh"
            + (
                ""
                if row == SMARD_ROW
                else f"  ({-acc.value.loc[row, 'skill_pct']:+.0f} %)"
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
    height=170 + 45 * len(bars),
)
fig.update_xaxes(range=[0, smard_mae * 1.3])
fig.update_yaxes(autorange="reversed", showgrid=False)
fig.update_layout(hovermode="closest", showlegend=False)
st.plotly_chart(fig, key="hero")

cards = st.columns(4)
cards[0].metric(
    "Average miss per hour",
    f"{mae:,.0f} MWh",
    f"{mae - smard_mae:+,.0f} MWh vs SMARD",
    delta_color="inverse",
    help=f"Mean absolute error of {label(FIRST)} over {len(acc.common):,} test hours.",
)
cards[1].metric(
    "Closer than SMARD",
    f"{skill:+.1f} %",
    help="Skill = how much of SMARD's miss we remove. 15 % skill: where SMARD misses by 100, we miss "
    "by 85. More on the Method page.",
)
cards[2].metric(
    "Months ahead of SMARD",
    f"{won:.0f} of {months}",
    help="Full calendar months in which our average miss was smaller than SMARD's.",
)
cards[3].metric(
    "Error saved over the year",
    f"{saved / 1e6:,.2f} TWh",
    help=f"{saved:,.0f} MWh: the sum over all test hours of SMARD's miss minus ours — as much as "
    f"SMARD's miss in {saved / smard_mae:,.0f} average hours.",
)

ensemble_slot(
    "The ensemble: combining our models",
    "A combination of several models usually misses less than any one of them. Its bar will join "
    "the chart above.",
)

# --- Risk days caught (spec 10 §3): high `rolling` + low `zero`, rule `any` --------------------
try:
    risk_labels = load_risk_labels()
except FileNotFoundError:
    risk_labels = None  # the data check on Team / About lists the missing export
if risk_labels is not None and not risk_labels.problems:
    st.subheader("Did we see the risk days coming?")
    outcomes = ["hit days", "missed days", "false-alarm days"]
    outcome_names = ["caught", "missed", "false alarm"]
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Too little green power (high side)",
            "Too much green power (below zero)",
        ],
        shared_yaxes=True,
    )
    risk_totals = {}
    for col, (direction, basis) in enumerate(
        [("high", "rolling"), ("low", "zero")], start=1
    ):
        scores = risk_scores(risk_labels, direction, basis, "any")
        risk_totals[direction] = scores
        pick = model_label(risk_labels.picks[direction])
        for source, name, bar_color in [
            ("smard", SMARD_NAME, COLORS["muted"]),
            (
                "model",
                f"Ours ({pick})",
                MODEL_STYLE[risk_labels.picks[direction]]["color"],
            ),
        ]:
            fig.add_trace(
                go.Bar(
                    x=outcome_names,
                    y=[scores[source][o] for o in outcomes],
                    name=name,
                    marker_color=bar_color,
                    text=[scores[source][o] for o in outcomes],
                    textposition="outside",
                    hovertemplate=f"{name}: %{{y}} days<extra></extra>",
                    legendgroup=source,
                    showlegend=col == 1 or source == "model",
                ),
                row=1,
                col=col,
            )
    style_plotly(
        fig,
        "Risk days in the test year: caught, missed, false alarms",
        "days",
        height=400,
    )
    fig.update_layout(barmode="group", hovermode="closest")
    fig.update_annotations(font={"color": INK})
    st.plotly_chart(fig, key="risk_days")
    st.caption(
        "A risk day is *caught* when the forecast, made the day before, also crosses the line. A "
        "*false alarm* is a forecast crossing on a day that turned out fine. Each side uses the model "
        "that does best there. Risk days are defined on the Method page."
    )

st.divider()

# ============================================================================
# Is it luck? Month by month (spec 09 §4.1)
# ============================================================================
st.header("Is it luck? Month by month")
month_x = acc.full_months.to_timestamp()
first_skill = monthly_skill(FIRST)
fig = go.Figure()
if not comparing:
    fig.add_trace(
        win_loss_bars(
            month_x,
            first_skill.to_numpy(),
            FIRST,
            "%{x|%b %Y}: %{y:+.1f} %<extra></extra>",
        )
    )
else:
    for row in ROWS:
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
    f"How much closer than SMARD, per month ({months} full months)",
    "% closer than SMARD",
)
fig.update_xaxes(dtick="M1", tickformat="%b %Y", hoverformat="%b %Y")
fig.update_yaxes(tickformat="+.0f")
if not comparing:
    fig.update_layout(hovermode="closest")
st.plotly_chart(fig, key="monthly_chart")

loss_months = first_skill[first_skill < 0]
bullets = [
    f"- **{OURS_NAME} is ahead of SMARD in {won:.0f} of {months} full months** — best in "
    f"{first_skill.idxmax().strftime('%B %Y')} ({first_skill.max():+.1f} %)."
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
        (SMARD_ABS - acc.errors[FIRST].abs()).groupby(acc.common.to_period("M")).sum()
    )
    steepest = monthly_gain.idxmax()
    running = (SMARD_ABS - acc.errors[FIRST].abs()).cumsum()
    at = running[running.index.to_period("M") == steepest].index[-1]
    fig.add_annotation(
        x=at,
        y=running[at],
        text=f"biggest gain: {steepest.strftime('%b %Y')}",
        showarrow=True,
        arrowhead=0,
        ay=-40,
        font={"color": INK},
    )
    style_plotly(
        fig, "Error saved against SMARD, running total", "MWh saved (running total)"
    )
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")
    st.plotly_chart(fig, key="cumulative_chart")
    st.caption(
        "A steady slope means a steady gain; steps mean a few episodes carry it."
    )

with right:
    st.subheader("At what time of day?")
    hour_of = acc.common.hour
    hours = [f"{h:02d}:00" for h in range(24)]
    fig = go.Figure()
    if not comparing:
        fig.add_trace(
            win_loss_bars(
                hours,
                acc.skill(FIRST, hour_of).to_numpy(),
                FIRST,
                "%{x}: %{y:+.1f} %<extra></extra>",
            )
        )
        fig.update_layout(hovermode="closest")
    else:
        for row in ROWS:
            fig.add_trace(
                go.Scatter(
                    x=hours,
                    y=acc.skill(row, hour_of).to_numpy(),
                    mode="lines+markers",
                    name=label(row),
                    line=line(row),
                    hovertemplate="%{y:+.1f} %",
                )
            )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
    style_plotly(
        fig,
        "How much closer than SMARD, by hour of day",
        "% closer than SMARD",
        "hour of day",
    )
    fig.update_yaxes(tickformat="+.0f")
    st.plotly_chart(fig, key="hour_chart")
    first_by_hour = acc.skill(FIRST, hour_of)
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
    "what actually happened: normal hours, hours with a green-power surplus, and the most extreme "
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
            name=OURS_NAME if (row == FIRST and not comparing) else label(row),
            marker_color=color(row),
            hovertemplate="%{y:,.0f} MWh",
        )
    )
style_plotly(fig, "Average miss per hour, by situation", "MWh", height=460)
fig.update_layout(barmode="group")
st.plotly_chart(fig, key="situations")
st.caption(
    "Lower is better. The extremes are harder for everyone — compare the bars within each group, "
    "not across groups."
)
ensemble_slot(
    "The ensemble in each situation",
    "One more bar per group, for the combined forecast.",
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
            "hits": (inside, inside, int((is_actual & is_forecast).sum())),
            "misses": (inside, outside, int((is_actual & ~is_forecast).sum())),
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
        ours_bin = acc.value.loc[FIRST, f"MAE_{category}_by_actual"]
        st.caption(
            f"Hours with actual residual load {acc.bin_rule(category)}: {in_bin.sum():,} of "
            f"{len(acc.common):,} test hours."
        )
        summary = (
            f"Average miss here: **{OURS_NAME.lower()} {ours_bin:,.0f} MWh**, SMARD {smard_bin:,.0f} MWh "
            f"({100 * (1 - ours_bin / smard_bin):+.1f} %)."
        )
        if picks and picks[0] != FIRST:
            best = acc.value.loc[picks[0], f"MAE_{category}_by_actual"]
            summary += f" The best model for this situation is **{label(picks[0])}** ({best:,.0f} MWh)."
        elif not picks:
            summary += " No model beats SMARD reliably here."
        st.markdown(summary)
        rows = [SMARD_ROW, *(ROWS if comparing else dict.fromkeys([FIRST, *picks[:1]]))]

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
            st.plotly_chart(fig, key=f"ordinary_hour_{category}")
        else:
            fig = go.Figure()
            for row in rows:
                counts = tail_counts(category, row)
                fig.add_trace(
                    go.Bar(
                        x=list(counts),
                        y=list(counts.values()),
                        name=label(row),
                        marker_color=color(row),
                        text=list(counts.values()),
                        textposition="outside",
                        hovertemplate=f"{label(row)}: %{{y:,}} hours<extra></extra>",
                    )
                )
            style_plotly(
                fig, "Did the forecast see these hours coming?", "hours", height=380
            )
            fig.update_layout(barmode="group", hovermode="closest")
            st.plotly_chart(fig, key=f"counts_{category}")
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
                    flagging_view(category, rows), key=f"flagging_{category}"
                )
            st.plotly_chart(error_ranges(category, rows), key=f"ranges_{category}")
            st.caption(
                f"Each bar holds the miss on the middle {RANGE_LEVEL:.0%} of these hours; the diamond is the "
                "average miss (does the forecast lean high or low?). A shorter bar is a tighter forecast."
            )

with st.expander("How we pick the best model for each situation"):
    st.markdown(
        f"A model counts as better than SMARD in a situation only if it misses less both on the hours "
        f"that really were in that range **and** on the hours it *forecast* to be in it — and it forecasts "
        f"the range at least {MIN_BIN_SHARE:.0%} as often as it happens. That keeps a model from looking "
        "good by simply never forecasting extremes."
    )

st.divider()

# ============================================================================
# Explore any week (§4.4 and the explorer, merged)
# ============================================================================
st.header("Explore the test year")
mondays = acc.test_days[acc.test_days.dayofweek == 0]
full_weeks = [day for day in mondays if day + pd.Timedelta(days=6) in acc.test_days]
weekly_gain = {
    monday: float(
        (SMARD_ABS - acc.errors[FIRST].abs())[
            monday : monday + pd.Timedelta(days=7, hours=-1)
        ].sum()
    )
    for monday in full_weeks
}
best_week, worst_week = max(weekly_gain, key=weekly_gain.get), min(
    weekly_gain, key=weekly_gain.get
)
best_month, worst_month = first_skill.idxmax(), first_skill.idxmin()
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
                    f"{direction.capitalize()}-side risk: {reason} ({monday:%d %b %Y})"
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

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38], vertical_spacing=0.08
)
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
for row in [SMARD_ROW, *ROWS]:
    name = OURS_NAME if (row == FIRST and not comparing) else label(row)
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
if len(ROWS) == 1:
    advantage = SMARD_ABS[window] - acc.errors[ROWS[0]][window].abs()
    fig.add_trace(
        win_loss_bars(
            window,
            advantage.to_numpy(),
            ROWS[0],
            "%{y:+,.0f} MWh<extra>who was closer</extra>",
        ),
        row=2,
        col=1,
    )
    bottom_label = "who was closer (MWh)"
else:
    for row in [SMARD_ROW, *ROWS]:
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
style_plotly(
    fig,
    f"Actual, SMARD and {'our forecast' if len(ROWS) == 1 else 'our models'}: {preset}",
    "residual load (MWh)",
    height=640,
)
fig.update_yaxes(title_text=bottom_label, row=2, col=1)
if (end - start) <= pd.Timedelta(days=8):
    fig.update_xaxes(dtick=24 * 3600 * 1000, tickformat="%a %d %b")
st.plotly_chart(fig, key="explorer_chart")
if len(ROWS) == 1:
    closer = int((advantage > 0).sum())
    st.caption(
        f"Bottom: bars above 0 are hours where our forecast was closer, below 0 (grey) where SMARD was. "
        f"Here: ours closer in {closer} of {len(window)} hours, average miss "
        f"{acc.errors[ROWS[0]][window].abs().mean():,.0f} MWh against SMARD's {SMARD_ABS[window].mean():,.0f} MWh. "
        "One week is an example, not a verdict. Drag to zoom, double-click to reset."
    )

st.divider()

# ============================================================================
# Ensemble blocks (spec 08; Streamlit-v3.md §1.4 blocks 5 and 8)
# ============================================================================
st.header("Coming next")
left, right = st.columns(2)
with left:
    ensemble_slot(
        "How sure are we?",
        "Each forecast with a 95 % range around it — and a check of how often reality really falls "
        "inside that range.",
    )
with right:
    ensemble_slot(
        "One model or a team of models?",
        "The ensemble against every single model, situation by situation, and how much weight each "
        "member gets.",
    )

# ============================================================================
# Monthly numbers (§9.1)
# ============================================================================
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
        "MAE and bias in MWh on the common test hours (bias = forecast − actual, positive = too high). "
        "*MAE without bias*: SMARD's error after removing its own monthly bias. *Persistence*: how closely "
        "SMARD's daily mean error follows the day before (1 = identical, 0 = unrelated) — our models see "
        "SMARD's error of the last 24 h."
    )
    in_month = acc.common[month_of == worst_month]
    daily_mae = pd.DataFrame(
        {
            name: acc.errors[row][in_month].abs().groupby(in_month.normalize()).mean()
            for name, row in [("ours", FIRST), ("smard", SMARD_ROW)]
        }
    )
    st.markdown(
        f"Worst full month of {label(FIRST)}: **{worst_month.strftime('%b %Y')}** ({first_skill[worst_month]:+.1f} %), "
        f"worse than SMARD on {int((daily_mae['ours'] > daily_mae['smard']).sum())} of {len(daily_mae)} days."
    )

# ============================================================================
# What this means
# ============================================================================
gains = {
    c: 100 * (1 - acc.value.loc[FIRST, metric[c]] / acc.value.loc[SMARD_ROW, metric[c]])
    for c in BIN_CATEGORIES
}
largest, smallest = max(gains, key=gains.get), min(gains, key=gains.get)
points = [
    f"On an average hour we miss **{skill:.0f} % less** than the official forecast, and we're ahead in "
    f"**{won:.0f} of {months}** months.",
    f"The gain is largest in **{SITUATION[largest].lower()}** ({gains[largest]:+.0f} %) and smallest in "
    f"**{SITUATION[smallest].lower()}** ({gains[smallest]:+.0f} %).",
]
if risk_labels is not None and not risk_labels.problems:
    high, low = risk_totals["high"], risk_totals["low"]
    points.append(
        f"On risk days we catch **{high['model']['hit days']} of {high['model']['actual days']}** on the high "
        f"side (SMARD {high['smard']['hit days']}) and **{low['model']['hit days']} of "
        f"{low['model']['actual days']}** below zero (SMARD {low['smard']['hit days']}), with "
        f"{high['model']['false-alarm days'] + low['model']['false-alarm days']} false alarms "
        f"(SMARD {high['smard']['false-alarm days'] + low['smard']['false-alarm days']})."
    )
with st.container(border=True):
    st.subheader("What this means")
    st.markdown("\n".join(f"{i}. {p}" for i, p in enumerate(points, start=1)))

next_page("app_pages/beat_smard.py")
