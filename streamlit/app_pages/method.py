"""Method page — how the forecast is set up, scored and turned into risk days.

Streamlit-v3.md §1.3, first three of its five tabs (Models and Ensemble follow later), retold for a
general audience with interactive Plotly charts:

1. **What we forecast** — the forecast setting of spec 06 (issue 18:00 on DAY−1, actuals cutoff
   `CUTOFF_HOUR`), the inputs (spec 06 Behaviour 18, default groups) and the validation / test
   windows, read from the model saves' `config.json`.
2. **The official forecast** — SMARD's day-ahead errors from `data/metrics/
   smard_forecast_errors_hourly.csv` (spec 04, the source of truth for re-scoring), on the test year.
3. **How we judge "better"** — MAE, skill, bias, RMSE and the residual-load bins of spec 09, from
   `model_results.load_accuracy` when the model exports exist.
4. **Risk days** — spec 02's exported thresholds and flags (`risk_labels_daily.csv`), never
   recomputed (CLAUDE.md). An empty flag is "not evaluable" and is left out of every share.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.layout import header, next_page
from data_loading import load_risk_labels_daily
from model_results import (
    SMARD_ROW,
    get_or_info,
    is_set,
    load_accuracy,
    load_export,
    model_windows,
)
from viz_helpers import (
    ACCENT,
    BIN_COLOR,
    BIN_LABEL,
    COLORS,
    INK,
    MUTED,
    SERIES_COLOR,
    SURFACE_2,
    TAIL_COLOR,
    model_line,
    style_plotly,
)

st.set_page_config(page_title="Method — Grid Stress", page_icon="🧭", layout="wide")

# Forecast setting (spec 06 §2.1 DATA_INFO). The actuals lag is still under experimentation: change
# the cutoff here only, every text and chart on this page follows.
ISSUE_HOUR = 18
CUTOFF_HOUR = 16  # = ISSUE_HOUR − actuals_lag (2 h, the team's current trial)
LAG_HOURS = ISSUE_HOUR - CUTOFF_HOUR

KNOWN_TINT = "#A1C8CA"  # sage region tint (chart-style: regions take tints)
TARGET_TINT = "#CCD3EB"  # lavender
GAP_TINT = COLORS["grid"]


def no_y_axis(fig):
    """Hide the y axis of a schematic chart."""
    fig.update_yaxes(visible=False, showgrid=False)
    return fig


header(
    "Method: how we forecast and how we check",
    "We don't build a forecast from scratch — we learn where the official one is systematically wrong.",
)
st.markdown(f"""
Every evening at {ISSUE_HOUR}:00, Germany's grid regulator publishes a forecast for each hour of the
next day. We take the ingredients of that forecast — expected electricity use and expected wind and
solar — and train models that learn where they are systematically wrong. Then we check, on a full
year the models have never seen, whether our corrected forecast misses less often and by less.
""")

windows = get_or_info(model_windows, "The model's validation and test years")
smard = get_or_info(
    lambda: load_export("smard").set_index("timestamp"), "SMARD's forecast errors"
)
if windows is not None:
    test_start, test_end = windows["test"]
    test_end = test_end + pd.Timedelta(hours=23)
    TEST_LABEL = f"{test_start:%d %b %Y} – {test_end:%d %b %Y}"

tab_setting, tab_smard, tab_metrics, tab_risk = st.tabs(
    [
        "① What we forecast",
        "② The official forecast",
        "③ How we judge “better”",
        "④ Risk days",
    ]
)

# =================================================================================================
# ① What we forecast
# =================================================================================================
with tab_setting:
    st.markdown(f"""
- **What:** residual load for each of the 24 hours of tomorrow.
- **When:** the forecast is due at **{ISSUE_HOUR}:00 the day before** — using only what is known by then.
- **Tested on:** a full year the models never saw while being built.
""")

    st.subheader("A forecast may only use what is known in time")
    fig = go.Figure()
    regions = [
        (
            -30,
            CUTOFF_HOUR - 24,
            KNOWN_TINT,
            f"Actual data known<br>(up to {CUTOFF_HOUR}:00)",
        ),
        (
            CUTOFF_HOUR - 24,
            ISSUE_HOUR - 24,
            GAP_TINT,
            f"{LAG_HOURS} h<br>reporting<br>delay",
        ),
        (0, 24, TARGET_TINT, "The 24 hours we forecast"),
    ]
    for x0, x1, color, text in regions:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=0.25,
            y1=0.75,
            fillcolor=color,
            line={"width": 0},
            layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=0.5,
            text=text,
            showarrow=False,
            font={"color": INK, "size": 13},
        )
    fig.add_shape(
        type="line",
        x0=ISSUE_HOUR - 24,
        x1=ISSUE_HOUR - 24,
        y0=0.1,
        y1=0.95,
        line={"color": INK, "width": 3},
    )
    fig.add_annotation(
        x=ISSUE_HOUR - 24,
        y=0.98,
        text=f"<b>Forecast issued, {ISSUE_HOUR}:00</b>",
        showarrow=False,
        yanchor="bottom",
        font={"color": INK},
    )
    fig.add_annotation(
        x=-12,
        y=0.05,
        text="<b>the day before</b>",
        showarrow=False,
        font={"color": MUTED},
    )
    fig.add_annotation(
        x=12,
        y=0.05,
        text="<b>the forecast day</b>",
        showarrow=False,
        font={"color": MUTED},
    )
    fig.add_trace(
        go.Scatter(
            x=[-30, 26],
            y=[0, 1],
            mode="markers",
            marker={"opacity": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    style_plotly(fig, "", "", height=260)
    ticks = [-24, -12, CUTOFF_HOUR - 24, ISSUE_HOUR - 24, 0, 12, 24]
    fig.update_xaxes(
        tickvals=ticks,
        ticktext=[f"{h % 24:02d}:00" for h in ticks],
        range=[-30, 26],
        showgrid=False,
    )
    fig.update_layout(hovermode=False, margin={"t": 30, "b": 30})
    st.plotly_chart(no_y_axis(fig), config={"staticPlot": True})
    st.caption(
        f"Measured data reach SMARD with a delay, so at {ISSUE_HOUR}:00 we only know actual values up to "
        f"{CUTOFF_HOUR}:00. Every input our models use respects this line — otherwise a model could "
        '"peek" at the answer and look better than it really is. This year\'s installed wind and '
        "solar capacity counts as known from 1 January."
    )

    st.subheader("From the official ingredients to our forecast")
    st.graphviz_chart(f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="{SURFACE_2}", color="{COLORS['grid']}",
        fontname="sans-serif", fontcolor="{INK}", fontsize=12, margin="0.18,0.1"];
  edge [color="{MUTED}"];
  fc_load  [label="SMARD forecast:\\nelectricity use"];
  fc_vre   [label="SMARD forecast:\\nwind + solar"];
  calendar [label="Calendar:\\nhour, weekday, holidays"];
  recent   [label="Recent actual residual load\\n(2 days and 1 week ago,\\nlatest before {CUTOFF_HOUR}:00)"];
  misses   [label="SMARD's recent misses\\n(last 24 h before {CUTOFF_HOUR}:00)"];
  capacity [label="Installed wind +\\nsolar capacity"];
  model    [label="Our model\\nlearns SMARD's\\ntypical misses", fillcolor="{ACCENT}", color="{ACCENT}"];
  ours     [label="Our residual-load\\nforecast", fillcolor="{TARGET_TINT}"];
  official [label="SMARD's own residual-\\nload forecast", style="rounded,dashed"];
  compare  [label="Compared hour by hour\\non the test year", fillcolor="white"];
  {{fc_load fc_vre calendar recent misses capacity}} -> model;
  model -> ours -> compare;
  official -> compare [style=dashed];
}}
""")
    st.caption(
        "This is called *post-processing* (Model Output Statistics): the official forecast's own "
        'ingredients are inputs, so our claim is "we reduce SMARD\'s error by X %", not "we forecast '
        'better than the grid operators".'
    )

    st.subheader("Practise on one year, get graded on the next")
    if windows is not None:
        validation_start, validation_end = windows["validation"]
        record_start = (
            smard.index.min()
            if smard is not None
            else validation_start - pd.DateOffset(years=3)
        )
        periods = [
            (
                "Earlier years — models learn from these",
                record_start,
                validation_start,
                SURFACE_2,
            ),
            (
                "Validation year — choose settings",
                validation_start,
                validation_end + pd.Timedelta(days=1),
                KNOWN_TINT,
            ),
            (
                "Test year — final grade",
                test_start,
                test_end + pd.Timedelta(hours=1),
                TARGET_TINT,
            ),
        ]
        fig = go.Figure()
        for name, start, end, color in periods:
            fig.add_trace(
                go.Bar(
                    y=[""],
                    x=[(end - start).total_seconds() * 1000],
                    base=[start],
                    orientation="h",
                    name=name,
                    marker={"color": color, "line": {"color": MUTED, "width": 0.5}},
                    hovertemplate=f"{name}<br>{start:%d %b %Y} – {end - pd.Timedelta(hours=1):%d %b %Y}<extra></extra>",
                )
            )
        style_plotly(fig, "", "", height=200)
        fig.update_layout(barmode="overlay", hovermode="closest", margin={"t": 10})
        fig.update_xaxes(type="date", showgrid=False)
        st.plotly_chart(no_y_axis(fig))
        st.markdown(f"""
We tuned every model on the **validation year** ({validation_start:%d %b %Y} – {validation_end:%d %b %Y})
and only then looked at the **test year** ({TEST_LABEL}) — like a student who practises on old exams
and is graded on a new one. Each model learns from the three years before the day it forecasts, and
is refitted every week as new data arrives.
""")

# =================================================================================================
# ② The official forecast
# =================================================================================================
with tab_smard:
    if smard is None or windows is None:
        st.info(
            "SMARD's errors on the test year need the SMARD and model exports (see the data check on Team / About)."
        )
    else:
        test = smard.loc[test_start:test_end]
        PAIRS = {
            "residual_load": ("Residual load", SERIES_COLOR["residual_load"]),
            "grid_load": ("Electricity use", SERIES_COLOR["grid_load"]),
            "renewables": ("Wind + solar", SERIES_COLOR["renewables"]),
        }

        def scores(frame):
            rows = {}
            for col, (label, _) in PAIRS.items():
                err = frame[f"err_{col}"].dropna()
                rows[label] = {
                    "average miss (MAE, MWh)": err.abs().mean(),
                    "big misses weighted more (RMSE, MWh)": np.sqrt((err**2).mean()),
                    "leans high (+) or low (−) (bias, MWh)": err.mean(),
                    "miss as % of typical level (nMAE)": 100
                    * err.abs().mean()
                    / frame.loc[err.index, col].mean(),
                    "hours": len(err),
                }
            return pd.DataFrame(rows).T

        test_scores = scores(test)
        mae = test_scores.loc["Residual load", "average miss (MAE, MWh)"]
        nmae = test_scores.loc["Residual load", "miss as % of typical level (nMAE)"]
        st.markdown(f"""
- On an average hour of the test year, **SMARD's residual-load forecast misses by about {mae:,.0f} MWh**
  (roughly {nmae:.0f} % of the typical level).
- **Electricity use is easy to forecast; wind and solar are not** — weather is harder to predict than
  how much power Germany will use.
- The misses are not random: they depend on the time of day and the season. That is the room our
  models work in.
""")

        st.subheader("One day: forecast versus reality")
        daily_miss = (
            test["err_residual_load"].abs().groupby(test.index.normalize()).mean()
        )
        day = st.date_input(
            "Pick a day from the test year",
            value=daily_miss.idxmax().date(),
            min_value=test_start.date(),
            max_value=test_end.date(),
            help="Opens on the day SMARD missed most on average.",
        )
        one_day = test.loc[str(day)]
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=one_day.index,
                y=one_day["residual_load"],
                name="What really happened",
                line=model_line("actual"),
                hovertemplate="%{y:,.0f} MWh",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=one_day.index,
                y=one_day["fc_residual_load"],
                name="SMARD's forecast (day before)",
                line=model_line("smard"),
                fill="tonexty",
                fillcolor=GAP_TINT,
                customdata=one_day["err_residual_load"],
                hovertemplate="%{y:,.0f} MWh (miss %{customdata:+,.0f})",
            )
        )
        style_plotly(
            fig, f"Residual load on {pd.Timestamp(day):%A, %d %B %Y}", "MWh", height=420
        )
        fig.update_xaxes(tickformat="%H:%M", hoverformat="%H:%M")
        st.plotly_chart(fig)
        st.caption(
            f"The grey area is SMARD's miss. That day it averaged {daily_miss.loc[pd.Timestamp(day)]:,.0f} MWh "
            f"per hour, against {mae:,.0f} MWh on a typical test-year day."
        )

        left, right = st.columns([1, 1.4])
        with left:
            st.subheader("Which part is hard?")
            fig = go.Figure(
                go.Bar(
                    y=test_scores.index,
                    x=test_scores["miss as % of typical level (nMAE)"],
                    orientation="h",
                    marker_color=[color for _, color in PAIRS.values()],
                    text=test_scores["miss as % of typical level (nMAE)"],
                    texttemplate="%{text:.1f} %",
                    textposition="outside",
                    hovertemplate="%{y}: %{x:.1f} %<extra></extra>",
                )
            )
            style_plotly(
                fig,
                "SMARD's average miss, % of the typical level",
                "",
                xlabel="%",
                height=320,
            )
            fig.update_xaxes(
                range=[0, test_scores["miss as % of typical level (nMAE)"].max() * 1.3]
            )
            fig.update_yaxes(autorange="reversed", showgrid=False)
            fig.update_layout(hovermode="closest", showlegend=False)
            st.plotly_chart(fig)
        with right:
            st.subheader("When does SMARD miss most?")
            by = st.radio(
                "By", ["hour of day", "month"], horizontal=True, key="smard_miss_by"
            )
            err = test["err_residual_load"].abs()
            if by == "hour of day":
                grouped = err.groupby(err.index.hour).mean()
                x = [f"{h:02d}:00" for h in grouped.index]
            else:
                grouped = err.groupby(err.index.to_period("M")).mean()
                x = [p.strftime("%b %Y") for p in grouped.index]
            fig = go.Figure(
                go.Bar(
                    x=x,
                    y=grouped,
                    marker_color=COLORS["muted"],
                    hovertemplate="%{x}: %{y:,.0f} MWh<extra></extra>",
                )
            )
            fig.add_hline(
                y=mae,
                line={"color": INK, "dash": "dot", "width": 1},
                annotation_text="test-year average",
                annotation_font_color=INK,
            )
            style_plotly(
                fig, f"SMARD's average residual-load miss by {by}", "MWh", height=320
            )
            fig.update_layout(hovermode="closest")
            st.plotly_chart(fig)

        with st.expander("All the numbers"):
            st.markdown(f"**Test year** ({TEST_LABEL})")
            st.dataframe(
                test_scores.style.format("{:,.1f}").format("{:,.0f}", subset=["hours"])
            )
            st.markdown(
                f"**Whole record** ({smard.index.min():%d %b %Y} – {smard.index.max():%d %b %Y}), for context"
            )
            st.dataframe(
                scores(smard)
                .style.format("{:,.1f}")
                .format("{:,.0f}", subset=["hours"])
            )
            st.caption(
                "Miss = forecast − actual: positive means SMARD forecast too high. Hours without a SMARD forecast are left out."
            )

# =================================================================================================
# ③ How we judge "better"
# =================================================================================================
with tab_metrics:
    st.markdown("""
- **Average miss (MAE):** on a typical hour, how far is the forecast from what really happened?
- **Skill:** the share of SMARD's miss that we remove — 10 % skill means our misses are 10 % smaller.
- **The extremes count separately:** a forecast can be good on ordinary hours and poor exactly when
  it matters, so we also score the highest and lowest hours on their own.
""")
    accuracy = get_or_info(load_accuracy, "The test-year scoring")
    if accuracy is not None:
        smard_mae = accuracy.errors[SMARD_ROW].abs().mean()
        example = 0.9 * smard_mae
        st.subheader("Skill in one example")
        fig = go.Figure(
            go.Bar(
                y=["SMARD", "A forecast with 10 % skill"],
                x=[smard_mae, example],
                orientation="h",
                marker_color=[COLORS["muted"], INK],
                text=[smard_mae, example],
                texttemplate="%{text:,.0f} MWh",
                textposition="outside",
                hovertemplate="%{y}: %{x:,.0f} MWh<extra></extra>",
            )
        )
        style_plotly(
            fig, "Average miss per hour on the test year", "", xlabel="MWh", height=240
        )
        fig.update_xaxes(range=[0, smard_mae * 1.25])
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_layout(hovermode="closest", showlegend=False)
        st.plotly_chart(fig)
        st.caption(
            f"SMARD's real average miss on the test year is {smard_mae:,.0f} MWh. The second bar is an "
            "illustration, not a result: our models' actual skill is on the **Where we beat SMARD** page."
        )

        st.subheader("Ordinary hours and extremes")
        actual = accuracy.actual[accuracy.common]
        e = accuracy.edges
        fig = go.Figure(
            go.Histogram(
                x=actual,
                nbinsx=100,
                marker_color=INK,
                hovertemplate="%{x} MWh: %{y:,} hours<extra></extra>",
            )
        )
        regions = [
            ("low_extreme", actual.min(), e["P1"]),
            ("ordinary", e["P25"], e["P75"]),
            ("high_extreme", e["P99"], actual.max()),
        ]
        for name, x0, x1 in regions:
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=BIN_COLOR[name],
                opacity=0.55,
                layer="below",
                line_width=0,
                annotation_text=BIN_LABEL[name],
                annotation_position="top left",
                annotation_font_color=INK,
            )
        fig.add_vline(
            x=0,
            line={"color": TAIL_COLOR["low"], "width": 1.5, "dash": "dot"},
            annotation_text="← below zero",
            annotation_position="bottom left",
            annotation_font_color=INK,
        )
        style_plotly(
            fig,
            "Test-year hours by residual load, with the scoring groups",
            "hours",
            xlabel="residual load (MWh)",
            height=420,
        )
        fig.update_layout(hovermode="closest", bargap=0.02)
        fig.update_xaxes(tickformat=",.0f")
        st.plotly_chart(fig)
        st.caption(
            f"Besides the average over all hours, we score four groups of test-year hours: the lowest 1 % "
            f"({accuracy.bin_rule('low_extreme')}), all hours below zero, the ordinary middle half "
            f"({accuracy.bin_rule('ordinary')}) and the highest 1 % ({accuracy.bin_rule('high_extreme')})."
        )

    with st.expander("More on the scores"):
        st.markdown("""
- **RMSE** squares each miss before averaging, so a few big misses weigh more than many small ones.
- **Bias** is the average signed miss: does the forecast lean high or low?
- **Why no percentage error (MAPE)?** Residual load crosses zero, and dividing by numbers close to
  zero makes percentage errors explode.
- **What a miss costs:** the "Where we beat SMARD" page also prices each miss with the imbalance price
  (reBAP) — |miss| × |price|. It is a proxy: the reBAP prices the error, it is not what the grid
  operators actually paid.
""")

# =================================================================================================
# ④ Risk days
# =================================================================================================
with tab_risk:
    st.markdown("""
- A day counts as a **risk day** if residual load pokes through a line set by the **previous twelve
  months** — at the top (too little green power) or the bottom (too much).
- On the low side there is a second, simpler line: **zero**.
- Low-side risk days are far more common today than in 2019 — that's why we always compare models on
  the same recent year.
""")
    labels = get_or_info(load_risk_labels_daily, "Risk days")
    if labels is not None:
        rule_col, basis_col = st.columns(2)
        rule = rule_col.radio(
            "A day counts if",
            ["any", "3h"],
            format_func={
                "any": "at least one hour crosses",
                "3h": "at least 3 hours in a row cross",
            }.get,
            horizontal=True,
        )
        low_basis = basis_col.radio(
            "Low-side line",
            ["rolling", "zero"],
            format_func={"rolling": "lowest 1 % of the past year", "zero": "zero"}.get,
            horizontal=True,
        )
        flags = {
            "high": labels[f"high_risk_rolling_{rule}"],
            "low": labels[f"low_risk_{low_basis}_{rule}"],
        }
        flagged = {d: is_set(f).to_numpy() for d, f in flags.items()}

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=labels["residual_max"],
                line={"width": 0},
                showlegend=False,
                hovertemplate="highest hour: %{y:,.0f} MWh<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=labels["residual_min"],
                name="Daily range of residual load",
                fill="tonexty",
                fillcolor=TARGET_TINT,
                line={"width": 0},
                hovertemplate="lowest hour: %{y:,.0f} MWh<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=labels["high_threshold_rolling"],
                name="High line (top 1 % of the past year)",
                line={"color": TAIL_COLOR["high"], "width": 1.6},
                hovertemplate="high line: %{y:,.0f} MWh<extra></extra>",
            )
        )
        low_line = labels[f"low_threshold_{low_basis}"]
        fig.add_trace(
            go.Scatter(
                x=labels.index,
                y=low_line,
                name=(
                    "Low line (bottom 1 % of the past year)"
                    if low_basis == "rolling"
                    else "Low line (zero)"
                ),
                line={"color": TAIL_COLOR["low"], "width": 1.6},
                hovertemplate="low line: %{y:,.0f} MWh<extra></extra>",
            )
        )
        for direction, extreme in [("high", "residual_max"), ("low", "residual_min")]:
            mask = flagged[direction]
            fig.add_trace(
                go.Scatter(
                    x=labels.index[mask],
                    y=labels.loc[mask, extreme],
                    mode="markers",
                    name=f"Risk day, {direction} side ({int(mask.sum())})",
                    marker={
                        "size": 6,
                        "color": TAIL_COLOR[direction],
                        "line": {"color": "white", "width": 0.5},
                    },
                    hovertemplate=f"risk day ({direction})<extra></extra>",
                )
            )
        style_plotly(
            fig,
            "Every day's range of residual load, the two lines, and the risk days",
            "MWh",
            height=520,
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
                    {
                        "count": 1,
                        "label": "1 year",
                        "step": "year",
                        "stepmode": "backward",
                    },
                    {"label": "All", "step": "all"},
                ],
                "x": 0,
                "y": 1.02,
            },
        )
        st.plotly_chart(fig)
        st.caption(
            "The lines are recomputed every day from the previous twelve months only, so they keep up as "
            "the grid changes and never use information from the future. The first year has no lines: "
            "it has no previous year to compare with. Use the buttons to zoom in."
        )

        st.subheader("How many days are flagged, per year?")
        year = labels.index.year
        evaluable_days = {d: f.notna().groupby(year).sum() for d, f in flags.items()}
        rates = pd.DataFrame(
            {
                d: 100
                * is_set(f).groupby(year).sum()
                / evaluable_days[d].replace(0, np.nan)
                for d, f in flags.items()
            }
        )
        days_in_data = labels.groupby(year).size()
        x = [f"{y} (so far)" if days_in_data[y] < 365 else str(y) for y in rates.index]
        fig = go.Figure()
        for direction, name in [("high", "High side"), ("low", "Low side")]:
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=rates[direction],
                    name=name,
                    marker_color=TAIL_COLOR[direction],
                    customdata=evaluable_days[direction],
                    hovertemplate=f"{name}: %{{y:.1f}} % of %{{customdata:,}} days<extra></extra>",
                )
            )
        style_plotly(fig, "Share of days flagged as risk days", "% of days", height=400)
        fig.update_layout(barmode="group", hovermode="closest")
        st.plotly_chart(fig)
        st.caption(
            "Only days that can be judged count: a complete day with a line to compare against. Years "
            "without bars have no such days on that side."
        )

        with st.expander("Reference: the whole-record line"):
            static = pd.DataFrame(
                {
                    "high side (whole-record top 1 %)": labels[
                        f"high_risk_static_{rule}"
                    ],
                    "low side (whole-record bottom 1 %)": labels[
                        f"low_risk_static_{rule}"
                    ],
                }
            )
            per_year = pd.DataFrame(
                {
                    name: 100
                    * is_set(f).groupby(year).sum()
                    / f.notna().groupby(year).sum().replace(0, np.nan)
                    for name, f in static.items()
                }
            )
            st.dataframe(per_year.style.format("{:.1f} %", na_rep="—"))
            st.caption(
                "A single line over the whole record shifts with every data update and uses future data, "
                "so it is only a reference, not how we define risk days."
            )

st.info(
    """
**What this can't tell us**
- **No safety margin:** a "high" day is extreme compared with the past year, not proof the grid ran out of capacity.
- **No regional detail:** this is Germany as a whole; local grid bottlenecks — the biggest real cause of interventions — are invisible here.
- **No record of interventions:** the data don't say whether the grid operators actually stepped in.
""",
    icon="ℹ️",
)

st.caption(
    "Sources: `notebooks/03_risk_classification/risk-definition.ipynb` (risk labels), "
    "`notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` (SMARD errors), "
    "`notebooks/05_modeling/regression-models-claude.ipynb` (forecast setting, windows, scoring)."
)

next_page("app_pages/method.py")
