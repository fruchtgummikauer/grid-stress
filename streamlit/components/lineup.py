"""The line-up shared by "Do we beat SMARD?" and "Try it yourself": which models the pages speak
about, their names, the page-wide model switch and the chart helpers both pages draw with.

The default view shows three cases (team decision 2026-10-07): "Our best model overall" (lowest
average miss among the `rolling` single models that beat SMARD; it leads the text) and "Our best at
too little / too much green power" (spec 09's rank 1 in the highest / lowest 1 % of hours). One
model can hold two cases. The "Compare models" switch opens any of the models we trained.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.naming import SITUATION, color, label, label_in_text
from components.risk_view import DIRECTION_TEXT
from model_results import (
    CANDIDATE_SPLIT,
    SMARD_ROW,
    Accuracy,
    get_or_stop,
    load_accuracy,
)
from viz_helpers import COLORS, model_line, model_pattern

OURS_NAME = "Our best model overall"
DIRECTION_NAME = {d: text[0] for d, text in DIRECTION_TEXT.items()}
SIDE_CATEGORY = {"high": "high_extreme", "low": "low_extreme"}
# Role per extreme for legends and menus, e.g. "XGBoost hybrid (best with too little green power)"
SHORT_SIDE = {"high": "with too little green power", "low": "with too much green power"}
# "highest 1 %" / "lowest 1 %": the bracket of the situation name
SIDE_HOURS = {
    side: SITUATION[c].split("(")[1].rstrip(")") for side, c in SIDE_CATEGORY.items()
}


@dataclass
class Lineup:
    """The rows a page speaks about, derived from `load_accuracy()`."""

    acc: Accuracy
    picks: list  # spec 09's `overall` picks, best first
    first: tuple  # our best model overall
    ensemble: (
        tuple | None
    )  # spec 09's ensemble pick, for "One model or a team of models?" only
    side_best: dict  # side -> our best at that extreme
    cases: dict  # case name -> row; one row can hold several
    default_rows: list  # the rows of the default view
    ours: dict  # headline row -> its case name(s)
    all_rows: list  # every row but SMARD
    public_rows: (
        list  # SMARD, then the `rolling` rows by average miss (the forecaster menus)
    )
    compare_default: list  # "Compare models" opens on these

    @property
    def lead(self):
        """The row the text speaks about."""
        return self.first

    @property
    def smard_abs(self):
        """SMARD's absolute miss per common hour."""
        return self.acc.errors[SMARD_ROW].abs()

    @property
    def test_label(self):
        return f"{self.acc.test_days[0]:%d %b %Y} – {self.acc.test_days[-1]:%d %b %Y}"

    def public_label(self, row):
        """Menu label: "Random forest hybrid (best overall)" for the headline rows."""
        return f"{label(row)} ({self.ours[row]})" if row in self.ours else label(row)

    def monthly_skill(self, row):
        """Skill vs SMARD (%) per full month."""
        return 100 * (1 - self.acc.monthly_mae[row] / self.acc.monthly_mae[SMARD_ROW])

    def band_trace(self, row, hours, **kwargs):
        """The row's 95 % range on `hours` as a shaded area (no hover)."""
        band = self.acc.bands[row].loc[hours]
        return go.Scatter(
            x=np.concatenate([hours, hours[::-1]]),
            y=np.concatenate([band["upper"], band["lower"][::-1]]),
            fill="toself",
            fillcolor=color(row),
            opacity=0.13,
            line={"width": 0},
            hoverinfo="skip",
            **kwargs,
        )


def load_lineup() -> Lineup:
    """The page's line-up; stops the page when the exports can't carry it."""
    acc = get_or_stop(load_accuracy)
    if acc.problems:
        st.error(
            "The model exports do not reproduce their own scoreboard, so this page is not shown:\n\n"
            + "\n".join(f"- {problem}" for problem in acc.problems)
        )
        st.stop()

    picks = acc.picks["overall"]
    if not picks:
        st.warning(
            f"No `{CANDIDATE_SPLIT}` model beats SMARD on average over the test window "
            f"({acc.test_days[0]:%d %b %Y} – {acc.test_days[-1]:%d %b %Y})."
        )
        st.stop()
    first = picks[
        0
    ]  # our best model overall: lowest average miss among the single models
    # Our best at each extreme: spec 09's rank 1 by actual in that bin
    side_best = {
        side: acc.picks[category][0]
        for side, category in SIDE_CATEGORY.items()
        if acc.picks[category]
    }
    cases = {OURS_NAME: first} | {
        f"Our best at {DIRECTION_NAME[side].lower()}": row
        for side, row in side_best.items()
    }
    default_rows = list(dict.fromkeys(cases.values()))
    # Short role tags for legends and menus; the text keeps the full case names
    roles = {first: ["overall"]}
    for side, row in side_best.items():
        roles.setdefault(row, []).append(SHORT_SIDE[side])
    ours = {
        row: "best " + " & ".join(r) for row, r in roles.items()
    }  # e.g. "best overall & with too much green power"
    all_rows = [row for row in acc.errors if row != SMARD_ROW]
    # "Compare models" opens on our top 3 by average miss plus Ridge, the simplest model, as a yardstick
    ridge = ("linear_direct", CANDIDATE_SPLIT)
    return Lineup(
        acc=acc,
        picks=picks,
        first=first,
        ensemble=acc.ensemble_pick,
        side_best=side_best,
        cases=cases,
        default_rows=default_rows,
        ours=ours,
        all_rows=all_rows,
        public_rows=[SMARD_ROW]
        + sorted(
            (row for row in all_rows if row[1] == CANDIDATE_SPLIT),
            key=lambda row: acc.value.loc[row, "MAE"],
        ),
        compare_default=list(
            dict.fromkeys([*picks[:3], *([ridge] if ridge in acc.errors else [])])
        ),
    )


def model_switch(lineup: Lineup, key: str, help_text: str):
    """The "Compare models" and "Show the ensemble" switches: (comparing, rows shown).

    Off: the default view's rows. The ensemble switch adds spec 08's ensemble pick to either view
    (off by default: the headline shows single models, team decision 2026-10-07).
    """
    switch, picker = st.columns([1, 2])
    comparing = switch.toggle("Compare models", help=help_text, key=f"{key}_compare")
    show_ensemble = lineup.ensemble is not None and switch.toggle(
        "Show the ensemble",
        help=f"Add the {label_in_text(lineup.ensemble) if lineup.ensemble else 'ensemble'}: "
        "our single models and SMARD combined, with weights learned on the year before"
        + (
            f" ({lineup.acc.value.loc[lineup.ensemble, 'skill_pct']:.1f} % smarter than SMARD)."
            if lineup.ensemble
            else "."
        ),
        key=f"{key}_ensemble",
    )
    extra = [lineup.ensemble] if show_ensemble else []
    if not comparing:
        return False, list(dict.fromkeys([*lineup.default_rows, *extra]))
    with picker:
        show_static = st.checkbox(
            "Also show models trained only once",
            help="Trained once before the test year instead of being refitted every week ('static').",
            key=f"{key}_static",
        )
        options = [
            row
            for row in lineup.all_rows
            if show_static or row[1] in (CANDIDATE_SPLIT, "none")
        ]
        rows = st.multiselect(
            "Models shown on every chart",
            options,
            default=lineup.compare_default,
            format_func=label,
            key=f"{key}_rows",
        )
    return True, list(dict.fromkeys([*(rows or [lineup.lead]), *extra]))


def reference_toggles(key, actual=True, smard=True):
    """Switches for the reference lines, on by default for comparison: (show actual, show SMARD).
    A line without a switch (`actual=False` / `smard=False`) counts as shown."""
    left, right = st.columns(2)
    return (
        not actual
        or left.toggle("Show what really happened", value=True, key=f"{key}_actual"),
        not smard or right.toggle("Show SMARD", value=True, key=f"{key}_smard"),
    )


def row_name(lineup: Lineup, row, comparing, sep=" "):
    """Legend name: model and role tag of a headline row in the default view, e.g.
    "Random forest hybrid (best overall)"; else the model's label."""
    if comparing or row not in lineup.ours:
        return label(row)
    return f"{label(row)}{sep}({lineup.ours[row]})"


def line(row, **overrides):
    """Plotly line of a row: the model's colour and dash (direct dotted), dash-dot for a
    non-default split."""
    style = model_line("smard" if row == SMARD_ROW else row[0])
    if row != SMARD_ROW and row[1] not in (CANDIDATE_SPLIT, "none"):
        style["dash"] = "dashdot"
    return style | overrides


def win_loss_bars(x, values, row, name, hover, showlegend=False):
    """Bars in the row's colour where it beats SMARD, slate where SMARD is closer."""
    return go.Bar(
        x=x,
        y=values,
        marker={
            "color": np.where(values >= 0, color(row), COLORS["muted"]),
            "pattern": {"shape": model_pattern(row[0])},
        },
        name=name,
        hovertemplate=hover,
        showlegend=showlegend,
    )


def full_weeks(acc: Accuracy):
    """Mondays of the test weeks that lie fully in the test window."""
    mondays = acc.test_days[acc.test_days.dayofweek == 0]
    return [day for day in mondays if day + pd.Timedelta(days=6) in acc.test_days]
