"""Who are we? (Streamlit-v3.md §1.6) — who made this, what it rests on, what comes next.

Team names, titles and one-liners as the team gave them (2026-10-07), each with a one-line joke in the team's
own words. The project numbers are computed from the data and the
model exports, never typed in. Licences: the code's as in the repository's LICENSE (MIT); SMARD's
data under CC BY 4.0 with the attribution "Bundesnetzagentur | SMARD.de" (smard.de/en/datennutzung).
The data check sits after the developer section and is hidden in demo mode.
"""

import streamlit as st

from components.data_check import data_check
from components.layout import REPO_URL, TOUR, demo_mode, header, next_page
from data_loading import get_years, load_smard
from model_results import SMARD_ROW, is_ensemble, load_accuracy
from viz_helpers import ACCENT, COLORS, INK, MUTED, SURFACE_2, tone

st.set_page_config(
    page_title="Who are we? — Grid Stress", page_icon="👥", layout="wide"
)

REPO_LICENSE = f"{REPO_URL}/blob/main/LICENSE"
SMARD_TERMS = "https://www.smard.de/en/datennutzung"
CC_BY = "https://creativecommons.org/licenses/by/4.0/"

# Name, emoji, title and a one-line joke (all in the team's own words).
# The humans share one row in this order; Claude, the intern, gets the row below.
TEAM = [
    (
        "Robert",
        "🧠",
        "The Caffeinated Oracle",
        "Already has the answer, probably pushed the notebook too — just don't ask when he last slept.",
    ),
    (
        "Marco",
        "🔌",
        "The Omnipresent API Whisperer",
        "Keeps the APIs talking and the models connected — somehow still online when the internet gives up.",
    ),
    (
        "Hari",
        "📈",
        "The Chart Wizard",
        "Turns every dataset into a feature supermarket and every chart into a show — static is apparently not an option.",
    ),
    (
        "Monica",
        "🗺️",
        "The Meme-Powered Explorer",
        "Has too much on her plate, turns side quests into notebooks — double-check the route, but expect a meme on the way back.",
    ),
    (
        "Claude",
        "🤖",
        "The Caffeineless Intern",
        "Works around the clock without coffee, produces suspicious amounts of code — still best to keep a human in the loop.",
    ),
]

header("Who are we?", "Who we are, what we used, and where the code is.")
st.markdown(
    "We are a team of data-science students at the neuefische bootcamp, plus one very diligent "
    "intern. We built this project end to end: from downloading years of German grid data to "
    "forecasting tomorrow's residual load and spotting the days when the grid is at risk."
)
st.link_button("View the code on GitHub", REPO_URL, icon="💻")

# --- Who we are -------------------------------------------------------------------------------
st.subheader("Who we are")


def member_card(column, name, emoji, title, joke):
    """One bordered card: emoji and name, title in bold, the one-liner."""
    with column.container(border=True, height="stretch"):
        st.markdown(f"#### {emoji} {name}")
        st.markdown(f"**{title}**")
        st.markdown(joke)


*humans, intern = TEAM
for column, member in zip(st.columns(len(humans)), humans):
    member_card(column, *member)
member_card(st.columns(len(humans))[0], *intern)

# --- The project in numbers -------------------------------------------------------------------
st.subheader("The project in numbers")
tiles = []
try:
    time_series = load_smard()
    years = get_years(time_series)
    tiles += [
        (
            "Hours of data analysed",
            f"{len(time_series):,}",
            "Every hour of grid data we downloaded.",
        ),
        ("Years covered", f"{years[0]} – {years[-1]}", None),
    ]
except (FileNotFoundError, RuntimeError):
    pass
try:
    accuracy = load_accuracy()
    # The headline row of "Do we beat SMARD?": our best model overall (lowest average miss)
    first = accuracy.picks["overall"][0] if accuracy.picks["overall"] else None
    models = {
        row[0]
        for row in accuracy.errors
        if row != SMARD_ROW and not is_ensemble(row) and row[0] != "seasonal_naive"
    }  # the seasonal-naive sanity check is no model of ours (as on the Method page)
    tiles += [
        (
            "Models compared",
            f"{len(models)}",
            "Single-model families scored against SMARD on the test year (not counting a simple sanity check)"
            + (
                f", plus {len({row[0] for row in accuracy.ensembles})} ways of combining them."
                if accuracy.ensembles
                else "."
            ),
        ),
        (
            "Test hours scored",
            f"{len(accuracy.common):,}",
            "Hours of a year the models never saw while being built.",
        ),
    ]
    if first is not None:
        tiles.append(
            (
                "Smarter than SMARD",
                f"{accuracy.value.loc[first, 'skill_pct']:+.1f} %",
                "How much of the official forecast's miss our forecast removes.",
            )
        )
except FileNotFoundError:
    accuracy, first = None, None
if tiles:
    for column, (label, value, help_text) in zip(st.columns(len(tiles)), tiles):
        column.metric(label, value, help=help_text)

# --- What we built ----------------------------------------------------------------------------
st.subheader("What we built")
skill_text = (
    f"made a forecast **{accuracy.value.loc[first, 'skill_pct']:.1f} % smarter than SMARD** (smaller misses) on a "
    "year our models never saw"
    if first is not None
    else "compared our forecasts with the official day-ahead forecast on a year our models never saw"
)
st.markdown(
    f"We showed how the grid's pressure behaves, defined risk days and set up a fair test, "
    f"{skill_text}, and put a price on that gain with the imbalance price (reBAP)."
)
LINKED = ["background", "method", "beat-smard", "try-it", "worth"]
links = st.columns(len(LINKED))
for column, (path, title, icon, _) in zip(
    links, [entry for entry in TOUR if entry[3] in LINKED]
):
    column.page_link(path, label=title, icon=icon)

st.graphviz_chart(f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="{tone(SURFACE_2)}", color="{tone(COLORS['grid'])}",
        fontname="sans-serif", fontcolor="{tone(INK)}", fontsize=12, margin="0.18,0.1"];
  edge [color="{tone(MUTED)}"];
  data [label="Data pipeline"]; eda [label="Exploration"]; risk [label="Risk definition"];
  features [label="Features"]; models [label="Models"];
  ensemble [label="Ensemble"];
  bands [label="Uncertainty ranges"];
  cost [label="Price of a miss\\n(reBAP)"];
  app [label="This app", fillcolor="{ACCENT}", color="{ACCENT}", fontcolor="{INK}"];
  data -> eda -> risk -> features -> models -> ensemble -> bands -> app;
  ensemble -> cost -> app;
}}
""")
st.caption("How the project grew, from raw data to this app.")

# --- Data and tools ---------------------------------------------------------------------------
st.subheader("Data and tools")
data_col, tools_col = st.columns(2)
with data_col:
    st.markdown(f"""
**Data**
- **Grid load, wind and solar generation, forecasts and installed capacity:**
  Bundesnetzagentur | SMARD.de, hourly, for Germany as a whole — licensed under
  [CC BY 4.0]({CC_BY}) ([terms of use]({SMARD_TERMS})). We processed the data and derived new
  values from it (forecasts, risk labels, scores); these are our own, not SMARD's.
- **Imbalance prices (reBAP):** netztransparenz.de — used only to price forecast misses.
""")
with tools_col:
    st.markdown("""
**Tools**
- **Data:** pandas, holidays, the SMARD and netztransparenz APIs
- **Models:** scikit-learn, LightGBM, XGBoost, statsmodels
- **Charts and app:** Plotly, matplotlib, seaborn, Streamlit
- **Environment:** uv, Jupyter
""")

st.info(
    "This is an independent student project. It is not affiliated with the Bundesnetzagentur, SMARD "
    "or the transmission system operators, and its forecasts are not intended for operational use.",
    icon="ℹ️",
)

# --- What we'd do next ------------------------------------------------------------------------
st.subheader("What we'd do next")
st.markdown("""
1. **Calibrate the uncertainty range** so it holds reality as often as it promises (95 % of hours).
2. **Settle the open risk-day settings:** how extreme counts as "extreme", and whether the two sides
   need one model or two.
3. **Add regional data:** local grid bottlenecks are the main real cause of interventions, and a
   national view can't see them.
4. **Run the forecast every day**, live, instead of on a fixed test year.
""")

# --- For developers ---------------------------------------------------------------------------
with st.expander("For developers"):
    st.markdown(f"""
- **Code:** [{REPO_URL.removeprefix("https://")}]({REPO_URL}) — [MIT License]({REPO_LICENSE})
- **Run it yourself:** clone the repository, run `make setup`, run
  `notebooks/API-connection.ipynb` once to download the data, then from the repository root:
  `uv run streamlit run streamlit/streamlit_app.py`.
- `data/` is not in the repository: every file is created by the notebook listed in the data check.
""")
if not demo_mode():
    data_check()  # its own collapsed expander (expanders can't nest)

st.caption(
    f"Data: Bundesnetzagentur | SMARD.de, [CC BY 4.0]({CC_BY}) · Imbalance prices: "
    f"netztransparenz.de · Code: [GitHub]({REPO_URL}), [MIT License]({REPO_LICENSE})"
)

next_page("app_pages/about.py")
