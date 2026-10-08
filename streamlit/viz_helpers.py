"""Shared plotting helpers and colour theme for the Streamlit app.

The colours are the PowerRangers palette (team logo, 2026-10-06), validated with the dataviz
skill's checker on white: lightness band, chroma floor, colour-blind and normal-vision
separation, contrast. See `.claude/skills/chart-style/SKILL.md` for the full table and the rules.
The series roles of `notebooks/01_eda/team-EDA.ipynb` §1.2 are kept (grid load hot, residual
load darkest, low risk cool, high risk hot); only the hexes differ. The notebooks keep their own
colours until the team re-runs or edits them.
"""

import matplotlib.dates as mdates
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    Normalize,
    TwoSlopeNorm,
)

# --- Colour configuration (PowerRangers palette — .claude/skills/chart-style) -----------------
# Text and surfaces
INK = "#1B3C65"  # navy: text, headings, the actual / residual-load line
MUTED = "#5C6275"  # captions, axis labels, ticks
SURFACE = "#FFFFFF"  # page and chart background
SURFACE_2 = "#F0F1F5"  # cards, sidebar, neutral midpoint of diverging scales
ACCENT = "#E7A118"  # logo amber: KPI highlights, never text on white (2.2:1)

# Chart series, in validated order (adjacent pairs pass for lines and bars; the first three
# also pass all-pairs, for scatter plots and small multiples). Plum sits in slot 4 so that the
# models read apart at a glance (plum vs blue ΔE 15.2; indigo vs blue only 5.3).
PALETTE = ["#12989A", "#2E64B0", "#C27C00", "#9A5FA8", "#C8553D", "#5566C2"]
# The ensemble's colour: a seventh, dark colour reserved for the model combination, far from both
# XGBoost colours (colour-blind ΔE > 20) and from navy / slate. Never a single model.
ENSEMBLE_COLOR = "#7A2A06"  # maroon

# The keys keep their old names so the pages need no edit; the hue names are approximate.
COLORS = {
    "vermillion": "#C8553D",  # grid_load, high risk / high tail
    "black": INK,  # residual_load (navy, the darkest series)
    "bluish_green": "#12989A",  # wind_on (teal)
    "blue": "#2E64B0",  # wind_off, low risk / low tail
    "orange": "#C27C00",  # solar (deep amber)
    "sky_blue": "#5566C2",  # renewables (indigo)
    "reddish_purple": "#9A5FA8",  # plum, spare categorical hue
    "muted": "#6B6E80",  # slate: reference lines, SMARD
    "grid": "#E4E6EE",  # plot gridlines
}

# Weekday vs weekend share a chart: blue + deep amber pass all-pairs (plum + blue does not)
DAY_TYPE_COLOR = {"weekday": COLORS["blue"], "weekend": COLORS["orange"]}

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

# Diverging: two poles (blue / vermillion) and a neutral grey midpoint, never a hue at zero
DIV_COLORS = [COLORS["blue"], "#A9B8E0", SURFACE_2, "#E8B3A7", COLORS["vermillion"]]
DIV_CMAP = LinearSegmentedColormap.from_list("gridstress_diverging", DIV_COLORS)
# Sequential, one hue (amber), light -> dark
GRID_LOAD_COLORS = ["#FBEFD3", "#EFC56A", "#D99A1A", "#B57400", "#8A5800"]
GRID_LOAD_CMAP = LinearSegmentedColormap.from_list(
    "gridstress_grid_load", GRID_LOAD_COLORS
)
# Residual load: zero is white in every case (team-EDA.ipynb §1.2)
RESIDUAL_COLORS = [
    COLORS["blue"],
    "#99A8DE",
    "#FFFFFF",
    "#E8B3A7",
    COLORS["vermillion"],
]
RESIDUAL_CMAP = LinearSegmentedColormap.from_list(
    "gridstress_residual", RESIDUAL_COLORS
)
# Sequential, one hue (periwinkle -> navy), for magnitudes without a warm/cool meaning
SEQUENTIAL = ["#99A8DE", "#7385CB", "#4E62AE", "#2F4A86", INK]
SEQ_CMAP = LinearSegmentedColormap.from_list("gridstress_sequential", SEQUENTIAL)

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
    ax.set_title(title, loc="center", fontsize=15, pad=12, color=INK)
    ax.set_xlabel("")
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, length=0)
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


def _plotly_scale(colors):
    """A Plotly colorscale with `colors` evenly spaced from 0 to 1."""
    return [[i / (len(colors) - 1), color] for i, color in enumerate(colors)]


# Plotly counterparts of the colormaps above (same anchors), for the interactive pages
DIV_SCALE = _plotly_scale(DIV_COLORS)
GRID_LOAD_SCALE = _plotly_scale(GRID_LOAD_COLORS)
SEQ_SCALE = _plotly_scale(["#FFFFFF", *SEQUENTIAL])


def residual_colorscale(vmin, vmax):
    """Plotly Heatmap kwargs for a residual-load surface — the counterpart of `residual_scale`:
    zero is white in every case, diverging when the surface crosses zero, else the relevant half.
    """
    if vmin < 0 < vmax:
        return {"colorscale": _plotly_scale(RESIDUAL_COLORS), "zmid": 0}
    if vmin >= 0:
        return {
            "colorscale": _plotly_scale(RESIDUAL_COLORS[2:]),
            "zmin": 0,
            "zmax": vmax,
        }
    return {"colorscale": _plotly_scale(RESIDUAL_COLORS[:3]), "zmin": vmin, "zmax": 0}


def year_colors(years, cmap="viridis"):
    """A colour per year, sampled from `cmap` — never a fixed year-to-colour table
    (team-EDA.ipynb §1.2).
    """
    years = list(years)
    sampled = plt.get_cmap(cmap)(np.linspace(0.05, 0.95, len(years)))
    return dict(zip(years, sampled))


# --- Model pages (Plotly) ---------------------------------------------------------------------
# Label, colour and line style per model key. A model keeps its colour on every page. Colours:
# option B of the 2026-10-07 review, chosen so any two models can share a chart (weakest pair
# OKLab ΔE 13.5 normal / 6.5 deuteranopia, all-pairs, against 5.3 / 3.2 before). Teal and blue
# stay from PALETTE; the other four are model-only colours, so indigo (slot 6) is "Wind + solar"
# only. Direct models are drawn dotted, hybrids solid: the architecture never rests on colour
# alone. The viz notebooks keep their older colours.
MODEL_COLOR = {
    "random_forest_hybrid": PALETTE[0],  # teal
    "lgbm_direct": PALETTE[1],  # blue
    "linear_direct": "#349F3A",  # green
    "xgb_hybrid": "#CD74B9",  # pink
    "lgbm_hybrid": "#A3394E",  # wine
    "xgb_direct": "#9257E0",  # violet
}
DIRECT_DASH = "dot"
MODEL_STYLE = {
    "random_forest_hybrid": {
        "label": "Random forest hybrid",
        "color": MODEL_COLOR["random_forest_hybrid"],
    },
    "lgbm_direct": {
        "label": "LightGBM direct",
        "color": MODEL_COLOR["lgbm_direct"],
        "dash": DIRECT_DASH,
    },
    "linear_direct": {
        "label": "Ridge direct",
        "color": MODEL_COLOR["linear_direct"],
        "dash": DIRECT_DASH,
    },
    "xgb_hybrid": {"label": "XGBoost hybrid", "color": MODEL_COLOR["xgb_hybrid"]},
    "lgbm_hybrid": {"label": "LightGBM hybrid", "color": MODEL_COLOR["lgbm_hybrid"]},
    "xgb_direct": {
        "label": "XGBoost direct",
        "color": MODEL_COLOR["xgb_direct"],
        "dash": DIRECT_DASH,
    },
    "sarimax_fourier": {
        "label": "SARIMAX + Fourier",
        "color": "#7385CB",
    },  # periwinkle (off)
    "seasonal_naive": {"label": "Seasonal naive (DAY−7)", "color": "#A3A7B6"},
    # Spec 08's ensembles share one colour: a family, not four series. ENSEMBLE_MARK tells them
    # apart (bar pattern, marker symbol), always next to a label
    "ensemble_regime_weighted": {
        "label": "Ensemble (regime-weighted)",
        "color": ENSEMBLE_COLOR,
    },
    "ensemble_weighted": {"label": "Ensemble (weighted)", "color": ENSEMBLE_COLOR},
    "ensemble_equal_mean": {"label": "Ensemble (equal mean)", "color": ENSEMBLE_COLOR},
    "ensemble_adaptive": {"label": "Ensemble (adaptive)", "color": ENSEMBLE_COLOR},
    "actual": {"label": "Actual residual load", "color": INK},
    "smard": {"label": "SMARD day-ahead", "color": COLORS["muted"], "dash": "dash"},
}

# Residual-load bins of the accuracy scoreboard: they mark bin regions and edges only, never a
# model (cool = low, warm = high).
BIN_COLOR = {
    "low_extreme": "#7385CB",  # periwinkle
    "below_zero": "#CCD3EB",  # lavender
    "ordinary": "#E4E6EE",  # light grey
    "high_extreme": ACCENT,  # logo amber
}
BIN_LABEL = {
    "low_extreme": "Lowest 1 % of hours",
    "below_zero": "Below zero",
    "ordinary": "Ordinary hours (middle 50 %)",
    "high_extreme": "Highest 1 % of hours",
}

# Risk-label thresholds, flagged hours and outcome cells per (direction, basis), never a model's
# colour: light tints (warm = high, cool = low), each at least colour-blind ΔE 8.9 from every
# model line incl. the ensemble. They are below 3:1 on white, so a threshold line needs a label.
RISK_COLOR = {
    ("high", "rolling"): "#E8B3A7",  # light vermillion
    ("low", "rolling"): "#A6B2E3",  # light periwinkle
    ("low", "zero"): "#CCD3EB",  # lavender
}
OUTCOME_COLOR = {"quiet": SURFACE_2, "not evaluable": SURFACE}
HOLIDAY_COLOR = ACCENT  # logo amber


ENSEMBLE_MARK = {
    "ensemble_regime_weighted": {"pattern": "", "symbol": "circle"},
    "ensemble_weighted": {"pattern": "/", "symbol": "square"},
    "ensemble_equal_mean": {"pattern": ".", "symbol": "diamond"},
    "ensemble_adaptive": {"pattern": "x", "symbol": "triangle-up"},
}


def model_pattern(key):
    """Plotly bar `pattern_shape` of a model key: a pattern per ensemble, none for the rest."""
    return ENSEMBLE_MARK.get(key, {}).get("pattern", "")


def model_symbol(key):
    """Plotly marker symbol of a model key: a symbol per ensemble, a circle for the rest."""
    return ENSEMBLE_MARK.get(key, {}).get("symbol", "circle")


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
        title={
            "text": title,
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 17, "color": INK},
        },
        height=height,
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
        font={"color": INK},
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
        title={"text": xlabel or "", "font": {"color": MUTED}},
        showgrid=False,
        showline=False,
        zeroline=False,
        ticks="",
    )
    fig.update_yaxes(
        title={"text": ylabel, "font": {"color": MUTED}},
        showgrid=True,
        gridcolor=COLORS["grid"],
        showline=False,
        zeroline=False,
        ticks="",
        tickformat=",.0f",
    )
    if _demo_mode():
        # Projector view: the toolbar is drawn transparent (Plotly's show/hide is a chart config,
        # not a layout setting; this keeps every page's st.plotly_chart call unchanged)
        hidden = "rgba(0,0,0,0)"
        fig.update_layout(
            modebar={"bgcolor": hidden, "color": hidden, "activecolor": hidden}
        )
    return fig


# --- Dark mode ------------------------------------------------------------------------------------
# Streamlit shows the light or the dark theme of .streamlit/config.toml (the visitor's system
# setting, or the ⋮ menu → Settings). Pages keep using the light tokens above; `themed(fig)` swaps
# each one for its dark counterpart at render time, so no page needs dark-specific code. Every dark
# colour was checked with the dataviz validator on the dark surface (--mode dark --surface
# #0F1B2D): the series and model sets separate at least as well as their light originals.
# Heatmap colour scales are left as they are: they carry their own legend.
DARK = {
    # Text and surfaces
    INK: "#E6EAF2",  # text, the actual / residual-load line
    MUTED: "#A7AEC2",  # captions, axis labels
    SURFACE: "#0F1B2D",  # deep navy page and chart background
    SURFACE_2: "#1A2A42",
    COLORS["grid"]: "#2A3B57",  # gridlines, also the "ordinary hours" bin
    COLORS[
        "muted"
    ]: "#766259",  # SMARD (still dashed): warm slate, apart from teal and blue
    "#A3A7B6": "#7C8196",  # seasonal naive
    # Series (EDA order; teal and blue are also the first two models)
    PALETTE[0]: "#2EA5A9",  # teal
    PALETTE[1]: "#4461C3",  # blue
    PALETTE[2]: "#A45300",  # deep amber
    PALETTE[3]: "#966298",  # plum
    PALETTE[4]: "#F25E3D",  # vermillion
    PALETTE[5]: "#7386FF",  # indigo
    # Models (teal and blue above)
    MODEL_COLOR["linear_direct"]: "#4F802C",  # green
    MODEL_COLOR["xgb_hybrid"]: "#944E92",  # pink -> orchid
    MODEL_COLOR["lgbm_hybrid"]: "#D5616D",  # wine -> rose red
    MODEL_COLOR["xgb_direct"]: "#8783EA",  # violet
    ENSEMBLE_COLOR: "#E9B6C9",  # the lightest series in dark, as maroon is the darkest in light
    # Region, threshold and flag tints: darker, so a fill doesn't glare on navy
    "#CCD3EB": "#3A4870",  # lavender
    "#A6B2E3": "#5466A8",  # light periwinkle
    "#E8B3A7": "#9A5446",  # light vermillion
    "#99A8DE": "#5A6DB0",
    "#A1C8CA": "#3F7476",  # sage
    "#A9B8E0": "#4E6098",
    "WHITE": "#0F1B2D",
    "RGBA(255,255,255,0.8)": "rgba(15,27,45,0.8)",
    "RGBA(255,255,255,0.85)": "rgba(15,27,45,0.85)",
}
DARK = {light.upper(): dark for light, dark in DARK.items()}


def is_dark():
    """True when the visitor sees the dark theme. Streamlit reports a theme switch with the next
    rerun (any click), and on the very first load it may not know yet: then light."""
    import streamlit as st

    try:
        return st.context.theme.type == "dark"
    except Exception:
        return False


def tone(color):
    """A light-theme token in the active theme, for colours outside Plotly (Graphviz, HTML)."""
    return DARK.get(color.upper(), color) if is_dark() else color


def _recolor(value):
    """`value` with every light token swapped for its dark one; colour scales untouched."""
    if isinstance(value, dict):
        return {k: v if k == "colorscale" else _recolor(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_recolor(v) for v in value]
    if isinstance(value, np.ndarray) and value.dtype.kind in "OU":
        return np.array([_recolor(v) for v in value.ravel()], dtype=object).reshape(
            value.shape
        )
    if isinstance(value, str) and len(value) <= 24:
        return DARK.get(value.upper(), value)
    return value


def themed(fig):
    """The figure in the active theme: unchanged in light, dark tokens in dark. Wrap every
    figure passed to `plotly_chart`."""
    return go.Figure(_recolor(fig.to_plotly_json())) if is_dark() else fig


def _demo_mode():
    """True while the sidebar's demo-mode toggle is on (False outside a Streamlit run)."""
    import streamlit as st

    try:
        return bool(st.session_state.get("demo_mode", False))
    except Exception:
        return False


# Plotly default template: every figure gets the palette, even one that skips style_plotly.
# Model lines still take their colour from MODEL_STYLE (model_line), never from the colorway order.
pio.templates["powerrangers"] = go.layout.Template(
    layout={
        "colorway": PALETTE,
        "font": {"color": INK},
        "paper_bgcolor": SURFACE,
        "plot_bgcolor": SURFACE,
        "xaxis": {"gridcolor": COLORS["grid"]},
        "yaxis": {"gridcolor": COLORS["grid"]},
    }
)
pio.templates.default = "plotly_white+powerrangers"
