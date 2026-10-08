"""Shared display names and styles of the model pages ("Do we beat SMARD?", "What is it worth?",
"How do we forecast?" Models tab): situation names, model one-liners, row labels, colours and money format.

A (model, split_method) pair is a "row", as in `model_results.py`.
"""

from model_results import CANDIDATE_SPLIT, SMARD_ROW
from viz_helpers import MODEL_STYLE, model_label, model_pattern, model_symbol

SMARD_NAME = "SMARD (official)"

# Plain-language situation names for spec 09's bins (the shared BIN_LABEL keeps the statistics)
SITUATION = {
    "overall": "All hours",
    "ordinary": "Normal hours",
    "below_zero": "Too much green power (below zero)",
    "low_extreme": "Too much green power, most extreme (lowest 1 %)",
    "high_extreme": "Too little green power, most extreme (highest 1 %)",
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
# One line per ensemble method (spec 08 Behaviours 9–12)
ENSEMBLE_IDEA = {
    "ensemble_equal_mean": "Equal mean — every member counts the same.",
    "ensemble_weighted": "Weighted — fixed weights, learned on the year before the test year.",
    "ensemble_regime_weighted": "Regime-weighted — separate weights for low, normal and high "
    "forecast levels, so each situation leans on the members that handle it best.",
    "ensemble_adaptive": "Adaptive — weights that follow which members did best in recent weeks.",
}


def label(row):
    """Display label of a row; the split only when it is not the default."""
    if row == SMARD_ROW:
        return SMARD_NAME
    base = model_label(row[0])
    return base if row[1] in (CANDIDATE_SPLIT, "none") else f"{base} ({row[1]})"


def label_in_text(row):
    """The label inside a sentence: "regime-weighted ensemble" for "Ensemble (regime-weighted)";
    model names keep their capitals (never `.lower()` a label: "XGBoost" → "xgboost").
    """
    if row == SMARD_ROW or not row[0].startswith("ensemble_"):
        return label(row)
    base = model_label(row[0])
    return label(row).replace(
        base, base.removeprefix("Ensemble (").removesuffix(")") + " ensemble"
    )


def color(row):
    """The row's colour (SMARD slate)."""
    return MODEL_STYLE["smard" if row == SMARD_ROW else row[0]]["color"]


def bar_style(row):
    """Plotly bar `marker` of a row: its colour, and a pattern per ensemble."""
    return {"color": color(row), "pattern": {"shape": model_pattern(row[0])}}


def marker(row):
    """Plotly `marker` for a line with markers: a symbol per ensemble."""
    return {"symbol": model_symbol(row[0]), "size": 7}


def money(eur, signed=True):
    """EUR in millions, e.g. `+293.9 million €`."""
    return f"{eur / 1e6:{'+' if signed else ''},.1f} million €"
