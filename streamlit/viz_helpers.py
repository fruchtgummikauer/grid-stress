"""Shared plotting helpers and colour theme for the Streamlit app.

Colour roles are ported from `notebooks/01_eda/team-EDA.ipynb` §1.2 (same series get the same
warm/cool reading — grid load hot, residual load near-black, low risk cool, high risk hot), but
the exact hexes are an Okabe-Ito-derived, colourblind-safe substitution — see
`.claude/skills/chart-style/SKILL.md` for the full palette table and the rules for using it.
`team-EDA.ipynb` itself is a shared, team-edited file and is intentionally left unchanged.

The model pages (Plotly) use the model, bin and risk colours of the two visualization notebooks
in `notebooks/05_modeling/` instead, so a model keeps its notebook colour; see the section
"Model pages (Plotly)" below and the chart-style skill.
"""

import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm

# --- Colour configuration (Okabe-Ito-derived, colourblind-safe — .claude/skills/chart-style) ---
COLORS = {
    "vermillion": "#D55E00",   # grid_load, high risk / high tail
    "black": "#1C1C1C",        # residual_load
    "bluish_green": "#009E73",  # wind_on
    "blue": "#0072B2",         # wind_off, low risk / low tail
    "orange": "#E69F00",       # solar
    "sky_blue": "#56B4E9",     # renewables
    "reddish_purple": "#CC79A7",  # day-type: weekday (distinct from every series role)
    "muted": "#707B8C",        # reference lines, secondary text
    "grid": "#E3E8EF",         # plot gridlines
}

DAY_TYPE_COLOR = {"weekday": COLORS["reddish_purple"], "weekend": COLORS["blue"]}

SERIES_COLOR = {
    "grid_load": COLORS["vermillion"],
    "residual_load": COLORS["black"],
    "wind_on": COLORS["bluish_green"],
    "wind_off": COLORS["blue"],
    "solar": COLORS["orange"],
    "renewables": COLORS["sky_blue"],
}

SERIES_LABEL = {
    "grid_load": "Grid load",
    "residual_load": "Residual load",
    "wind_on": "Onshore wind",
    "wind_off": "Offshore wind",
    "solar": "Solar",
    "renewables": "Wind + solar",
}

TAIL_COLOR = {"low": COLORS["blue"], "high": COLORS["vermillion"]}
RAMP_COLOR = COLORS["orange"]

DIV_CMAP = LinearSegmentedColormap.from_list(
    "gridstress_diverging",
    [COLORS["bluish_green"], "#DCEEEF", "#FFFFFF", "#F8D1C5", COLORS["vermillion"]],
)
GRID_LOAD_CMAP = LinearSegmentedColormap.from_list(
    "gridstress_grid_load",
    [COLORS["sky_blue"], "#B9DCC7", "#FDBA67", COLORS["orange"], COLORS["vermillion"]],
)
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "gridstress_residual",
    [COLORS["blue"], "#778ACC", "#FFFFFF", "goldenrod", COLORS["vermillion"]],
)

HEADLINE_COLS = ["grid_load", "residual_load"]
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]


def series_style(col: str, **overrides):
    """Line kwargs for `col`: colour and width, from the configuration above."""
    style = {
        "color": SERIES_COLOR[col],
        "linewidth": 2.4 if col in HEADLINE_COLS else 1.8,
        "label": SERIES_LABEL[col],
    }
    style.update(overrides)
    return style


def style_timeseries(ax, title, ylabel):
    """Custom grid, no box, year ticks. `ylabel` is required (team-EDA.ipynb §1.1)."""
    ax.set_title(title, loc="center", fontsize=15, pad=12)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, color="grey")
    ax.grid(axis="y", color="0.9", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors="black", length=0)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")


def residual_scale(vmin, vmax):
    """Colormap and norm for a residual-load surface — zero is white in every case
    (team-EDA.ipynb §1.2).
    """
    if vmin < 0 < vmax:
        return RESIDUAL_CMAP, TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    if vmin >= 0:
        return RESIDUAL_CMAP, Normalize(vmin=0, vmax=vmax)
    half = ListedColormap(RESIDUAL_CMAP(np.linspace(0.0, 0.5, 256)))
    return half, Normalize(vmin=vmin, vmax=0)


def year_colors(years, cmap="viridis"):
    """A colour per year, sampled from `cmap` — never a fixed year-to-colour table
    (team-EDA.ipynb §1.2).
    """
    years = list(years)
    sampled = plt.get_cmap(cmap)(np.linspace(0.05, 0.95, len(years)))
    return dict(zip(years, sampled))

# --- Model pages (Plotly) ---------------------------------------------------------------------
# Label and colour per row, copied from `STYLE` in visualization-01-regression-best-models.ipynb
# and visualization-02-classification-risk-labels.ipynb (themselves copied from MODELS / FIXED in
# regression-models-claude.ipynb; `seasonal_naive` straight from FIXED), so a model has the same
# colour in the notebooks and the app.
MODEL_STYLE = {
    "sarimax_fourier": {"label": "SARIMAX + Fourier", "color": "#D9A53A"},
    "lgbm_direct": {"label": "LightGBM direct", "color": "#2C6EBA"},
    "lgbm_hybrid": {"label": "LightGBM hybrid", "color": "#2F8F5B"},
    "xgb_direct": {"label": "XGBoost direct", "color": "#E95D0F"},
    "xgb_hybrid": {"label": "XGBoost hybrid", "color": "#B10F0F"},
    "linear_direct": {"label": "Ridge direct", "color": "#7A4FA3"},
    "random_forest_hybrid": {"label": "Random forest hybrid", "color": "#8C564B"},
    "seasonal_naive": {"label": "Seasonal naive (DAY−7)", "color": "#9098A2"},
    "actual": {"label": "Actual residual load", "color": "#1C1C1C"},
    "smard": {"label": "SMARD day-ahead", "color": "#48505A", "dash": "dash"},
}

# Residual-load bins of the accuracy scoreboard (visualization-01 `TAIL_COLOR`): they mark bin
# regions and edges only, never a model.
BIN_COLOR = {
    "low_extreme": "#17BECF",  # cyan
    "below_zero": "#9EDAE5",  # light cyan
    "ordinary": "#E3E8EF",  # light grey
    "high_extreme": "#E6B800",  # gold
}
BIN_LABEL = {
    "low_extreme": "Lowest 1 % of hours",
    "below_zero": "Below zero",
    "ordinary": "Ordinary hours (middle 50 %)",
    "high_extreme": "Highest 1 % of hours",
}

# Risk-label thresholds, flagged hours and outcome cells per (direction, basis)
# (visualization-02 `TAIL_COLOR`, `OUTCOME_COLOR`, `HOLIDAY_COLOR`), never a model's colour.
RISK_COLOR = {
    ("high", "rolling"): "#E0436B",  # crimson-pink, apart from both XGBoost reds
    ("low", "rolling"): "#17BECF",  # cyan
    ("low", "zero"): "#9EDAE5",  # light cyan
}
OUTCOME_COLOR = {"quiet": "#F2F4F7", "not evaluable": "#FFFFFF"}
HOLIDAY_COLOR = "#8A9A2B"  # olive


def model_label(key):
    """Display label of a model key (or `actual` / `smard`)."""
    return MODEL_STYLE[key]["label"]


def model_line(key, width=None):
    """Plotly `line` dict of a model key: its colour, SMARD dashed, headline rows thicker."""
    style = MODEL_STYLE[key]
    return {
        "color": style["color"],
        "dash": style.get("dash", "solid"),
        "width": width or (2.4 if key == "actual" else 1.8),
    }


def style_plotly(fig, title, ylabel, xlabel=None, height=460):
    """The app's chart look for a Plotly figure — the counterpart of `style_timeseries`.

    No box, light y grid only, thousands separators, one hover box per x position, legend below.
    `ylabel` is required: every chart states its unit.
    """
    fig.update_layout(
        title={"text": title, "x": 0.5, "xanchor": "center", "font": {"size": 17}},
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -0.15,
            "xanchor": "center",
            "x": 0.5,
        },
        margin={"l": 60, "r": 20, "t": 60, "b": 40},
    )
    fig.update_xaxes(
        title={"text": xlabel or "", "font": {"color": "grey"}},
        showgrid=False,
        showline=False,
        zeroline=False,
        ticks="",
    )
    fig.update_yaxes(
        title={"text": ylabel, "font": {"color": "grey"}},
        showgrid=True,
        gridcolor=COLORS["grid"],
        showline=False,
        zeroline=False,
        ticks="",
        tickformat=",.0f",
    )
    return fig
