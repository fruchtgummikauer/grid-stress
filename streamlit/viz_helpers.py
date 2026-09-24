"""Shared plotting helpers and colour theme for the Streamlit app.

Colour roles are ported from `notebooks/01_eda/team-EDA.ipynb` §1.2 (same series get the same
warm/cool reading — grid load hot, residual load near-black, low risk cool, high risk hot), but
the exact hexes are an Okabe-Ito-derived, colourblind-safe substitution — see
`.claude/skills/chart-style/SKILL.md` for the full palette table and the rules for using it.
`team-EDA.ipynb` itself is a shared, team-edited file and is intentionally left unchanged.
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
