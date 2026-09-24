"""Model page — risk-label definition and the SMARD forecast benchmark.

Per `.claude/specs/Streamlit-draft.md` §5 Page 3: there is no trained model of our own yet
(`modeling/*.py` is unrelated coffee-dataset template code, `models/` is empty). Real content
comes from two read-only, team-reviewed notebooks:

- `notebooks/03_risk_classification/risk-definition.ipynb` — the risk-label definition and
  threshold construction.
- `notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` — SMARD's own day-ahead forecast
  benchmarked against actuals (the target our future model has to beat).

Both notebooks' exported CSVs (`data/risk_classification/`, `data/metrics/`) are not needed here:
every number below is computed directly from `data/smard.csv`, the same way the source notebooks
do it, so this page works even before those exports are regenerated.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loading import load_smard
from viz_helpers import COLORS, TAIL_COLOR, style_timeseries

st.set_page_config(page_title="Model — Grid Stress", page_icon="🧭", layout="wide")
st.title("Risk Label & Forecast Benchmark")
st.caption(
    "Sources: `notebooks/03_risk_classification/risk-definition.ipynb` and "
    "`notebooks/02_forecast_metrics/forecast-metrics-claude.ipynb` (both read-only)."
)

st.warning(
    "**No model of our own exists yet.** `modeling/*.py` is unrelated template code, and "
    "`models/` holds no saved artifact. This page presents the risk-label definition and the "
    "public SMARD forecast benchmark — the target a future model of ours will need to beat."
)

try:
    time_series = load_smard()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(f"SMARD data not available: {exc}")
    st.stop()

# ============================================================================
# Risk-label definition (risk-definition.ipynb §2)
# ============================================================================
st.header("Risk-label definition")
st.markdown(
    """
Risk is defined as a `residual_load` magnitude extreme enough that TSOs plausibly need
intervention measures (redispatch, reserve activation, cross-border exchange, curtailment) to
keep the grid balanced. It splits into two structurally different mechanisms:

- **High residual load** — least renewable cover relative to demand (the Dunkelflaute-type
  winter stress case).
- **Low / negative residual load** — renewable oversupply, negative prices, downward redispatch.

**What this data cannot establish** (`risk-definition.ipynb` §2.1):
- **No margin** — the high direction is a *relative* extreme in this record, not a demonstrated
  approach to a capacity limit (no installed-capacity/availability/cross-border data here).
- **No regional detail** — the label is a **national-balance proxy**; regional transmission
  congestion (the largest real driver of German redispatch) is invisible in region-`DE` data.
- **No intervention record** — nothing in the data records whether a TSO actually intervened.
"""
)

st.divider()

# ============================================================================
# Threshold construction (risk-definition.ipynb §3)
# ============================================================================
st.header("Threshold construction")
st.caption(
    "Rolling (365-day trailing quantile, causal) and zero-crossing (low direction only) are the "
    "recommended label bases; static (whole-record quantile) is demoted to a reference line — "
    "it is unstable under re-fetch and not causal."
)

residual = time_series["residual_load"]
P_DEFAULT = 0.01
WINDOW = pd.Timedelta(days=365)

DAY = pd.Series(residual.index.normalize(), index=residual.index)
DAYS = pd.DatetimeIndex(sorted(DAY.unique()))


def rolling_threshold(series: pd.Series, q: float, window: pd.Timedelta = WINDOW) -> pd.Series:
    """Trailing-window quantile per calendar day, using only days strictly before it
    (`risk-definition.ipynb` §3.2). `D`'s own data never contributes to its own threshold.
    """
    roll = series.rolling(window).quantile(q)
    end_of_day = roll.groupby(series.index.normalize()).last()
    end_of_day.index = end_of_day.index + pd.Timedelta(days=1)
    thr = end_of_day.reindex(DAYS)
    return thr.where(thr.index >= series.index.min().normalize() + window)


THRESHOLDS = {
    ("high", "rolling"): rolling_threshold(residual, 1 - P_DEFAULT),
    ("high", "static"): pd.Series(float(residual.quantile(1 - P_DEFAULT)), index=DAYS),
    ("low", "rolling"): rolling_threshold(residual, P_DEFAULT),
    ("low", "zero"): pd.Series(0.0, index=DAYS),
    ("low", "static"): pd.Series(float(residual.quantile(P_DEFAULT)), index=DAYS),
}
DIRECTION_LABEL = {"high": "High residual load", "low": "Low / negative residual load"}
BASIS_STYLE = {
    "rolling": {"linestyle": "-", "linewidth": 2.0},
    "static": {"linestyle": "--", "linewidth": 1.4},
    "zero": {"linestyle": ":", "linewidth": 1.4},
}

fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
for ax, direction in zip(axes, ("high", "low")):
    roll = THRESHOLDS[(direction, "rolling")]
    ax.plot(roll.index, roll.to_numpy(), color=TAIL_COLOR[direction],
             **BASIS_STYLE["rolling"], label=f"rolling ({WINDOW.days}-day trailing)")
    ax.axhline(THRESHOLDS[(direction, "static")].iloc[0], color=COLORS["muted"],
                **BASIS_STYLE["static"], label="static (whole record)")
    if direction == "low":
        ax.axhline(0, color=COLORS["black"], **BASIS_STYLE["zero"],
                    label="zero (physical oversupply boundary)")
    style_timeseries(ax, f"{DIRECTION_LABEL[direction]} — threshold at p = {P_DEFAULT:.1%}",
                      "Residual load threshold (MWh)")
    ax.legend(frameon=False, loc="best")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- The **low threshold drifts several times more than the high** one: the high line wanders in a
  relatively narrow band, while the low line marches steadily downward and crosses zero partway
  through the record.
- Before the crossing point, the `zero` basis flags nothing at all; after it, the `static`
  reference line is a poor summary of the current rolling threshold.
"""
)

st.divider()

# ============================================================================
# Positive rate by year (risk-definition.ipynb §3.6)
# ============================================================================
st.header("Positive rate by year")
st.caption("Share of days flagged per calendar year, `any` day rule, p = 1 %.")

RESOLUTION = time_series.index.to_series().diff().mode().iloc[0]
MIN_RUN = int(pd.Timedelta("3h") / RESOLUTION)
EXPECTED_OBS_PER_DAY = int(pd.Timedelta("1D") / RESOLUTION)
MIN_OBS_PER_DAY = int(np.ceil((23 / 24) * EXPECTED_OBS_PER_DAY))
obs_per_day = residual.groupby(DAY).size().reindex(DAYS, fill_value=0)
day_complete = obs_per_day >= MIN_OBS_PER_DAY


def hourly_crossings(series, threshold_by_day, direction):
    thr = pd.Series(threshold_by_day.reindex(series.index.normalize()).to_numpy(), index=series.index)
    crossed = (series >= thr) if direction == "high" else (series <= thr)
    return crossed.astype("boolean").mask(thr.isna())


def day_any_rule(hourly_flag, threshold_by_day):
    filled = hourly_flag.fillna(False).astype(bool)
    any_rule = filled.groupby(DAY).any().reindex(DAYS, fill_value=False)
    valid = (threshold_by_day.notna() & day_complete).reindex(DAYS, fill_value=False)
    return any_rule.astype("boolean").mask(~valid)


positive_rate = pd.DataFrame({
    f"{d} ({b})": day_any_rule(hourly_crossings(residual, thr, d), thr)
    for (d, b), thr in THRESHOLDS.items()
})
by_year = positive_rate.groupby(positive_rate.index.year).mean()
by_year.index.name = "year"
st.dataframe(by_year.style.format("{:.1%}", na_rep="—"))

days_per_year = positive_rate.groupby(positive_rate.index.year).size()
partial = days_per_year[days_per_year < 365]
if len(partial):
    st.caption(f"Partial year(s): {partial.to_dict()} — rates for these are not directly comparable to full years.")

st.markdown(
    """
- The **low extreme's base rate is strongly non-stationary**: under `static`/`zero` it is empty
  for the record's early years and climbs steeply, ending an order of magnitude higher.
- The **rolling basis is more stable**, but not stationary either — it is the only basis under
  which the early and late record are remotely comparable.
- **Consequence for modelling:** a chronological train/test split has non-exchangeable base rates
  for the low-extreme label — any classifier metric computed across it is not directly
  comparable.
"""
)

st.divider()

# ============================================================================
# SMARD forecast benchmark (forecast-metrics-claude.ipynb §2-3)
# ============================================================================
st.header("How good is today's public forecast?")

PAIRS = {
    "residual_load": {"forecast": "fc_residual_load", "label": "Residual load"},
    "grid_load": {"forecast": "fc_grid_load", "label": "Grid load"},
    "renewables": {"forecast": "fc_gen_wind_solar", "label": "Wind + solar"},
}

rows = []
for actual, spec in PAIRS.items():
    err = (time_series[spec["forecast"]] - time_series[actual]).dropna()
    actual_mean = time_series.loc[err.index, actual].mean()
    rows.append({
        "pair": spec["label"],
        "MAE (MWh)": round(err.abs().mean()),
        "RMSE (MWh)": round(np.sqrt((err ** 2).mean())),
        "bias (MWh)": round(err.mean()),
        "nMAE (%)": round(100 * err.abs().mean() / actual_mean, 1),
        "hour_count": len(err),
    })
headline = pd.DataFrame(rows).set_index("pair")
residual_row = headline.loc["Residual load"]

st.markdown(
    f"""
SMARD (the German grid regulator) already publishes a day-ahead forecast of residual load — the
demand left over after wind and solar. This is the public benchmark our own forecast will need
to beat, not a model we built.

- **On an average hour, SMARD's forecast is off by about {residual_row['MAE (MWh)']:,.0f} MWh** —
  roughly {residual_row['nMAE (%)']:.0f} % of typical residual load — measured over
  {residual_row['hour_count']:,} hours (2019–2026).

- **Grid demand is easier to forecast than the renewable side.** SMARD's grid-load forecast is
  off by only {headline.loc['Grid load', 'nMAE (%)']:.0f} % on average, versus
  {headline.loc['Wind + solar', 'nMAE (%)']:.0f} % for wind + solar — weather is harder to predict
  than how much electricity Germany will use.
"""
)

with st.expander("See the full numbers (MAE, RMSE, bias, hour count)"):
    st.dataframe(headline)
    st.caption(
        "These are provisional, full-record numbers. "
        "`forecast-metrics-claude.ipynb`'s own closing section re-scores SMARD on the "
        "modelling spec's eventual test window instead."
    )
