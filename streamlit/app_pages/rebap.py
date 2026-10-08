"""What is it worth? — our smaller forecast misses priced at the imbalance price (reBAP).

Reads only viz-03's cost export (`notebooks/05_modeling/visualization-03-rebap-cost.ipynb`, spec 11)
through `model_results.load_rebap_cost`, never `rebap.csv` (CLAUDE.md). The page explains the
simplifications first; the numbers appear only after the visitor presses "Calculate" (team
decisions 2026-10-07: only our ensemble — spec 09's ensemble pick — is offered, 💸 rising on the click).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.layout import header, next_page
from components.naming import SITUATION, bar_style, label, label_in_text, money
from model_results import (
    REBAP_CATEGORIES,
    SMARD_ROW,
    get_or_stop,
    load_accuracy,
    load_rebap_cost,
)
from viz_helpers import COLORS, INK, model_line, style_plotly, themed

st.set_page_config(
    page_title="What is it worth? — Grid Stress", page_icon="💶", layout="wide"
)

PRICE_TOP = 0.99  # the dearest 1 % of hours, viz-03's top price band
TOP_HOURS = 10  # rows in the dearest-hours table
RAIN_DROPS = 40  # 💸 per click
RAIN_SECONDS = 3.0

acc = get_or_stop(load_accuracy)
rebap = get_or_stop(load_rebap_cost)
if acc.problems or rebap.problems:
    st.error(
        "The cost export doesn't match the model exports, so this page is not shown:\n\n"
        + "\n".join(f"- {problem}" for problem in acc.problems + rebap.problems)
    )
    st.stop()

header("What is it worth?", "A better forecast, priced in euros.")

with st.container(border=True):
    st.markdown("""
**How we put a price on a forecast miss — and what we simplified**

- **One national price.** Every miss is priced at Germany's imbalance price (reBAP) of that hour —
  the average of its four quarter-hours.
- **Amount, not direction.** We price the size of the miss and ignore whether it was too high or
  too low, so being more accurate never counts as a loss.
- **Hourly forecasts.** A price spike in one quarter-hour counts for a quarter of the hour's miss.
- **A yardstick, not a bill.** Nobody paid these sums: market parties settle their own, much
  smaller imbalances, and the price itself reacts to the imbalance.
""")

# --- Model and button -------------------------------------------------------------------------
# Only our ensemble is offered (team decision 2026-10-07)
ROW = acc.ensemble_pick
if ROW is None or ROW not in rebap.saved:
    st.warning(
        "Our ensemble is not in the cost export, so there is nothing to price. Re-run "
        "ensemble-claude.ipynb, then visualization-03-rebap-cost.ipynb."
    )
    st.stop()


def option_label(row):
    """Display name of the priced forecast."""
    return f"Our {label_in_text(row)}"


picker, button = st.columns([3, 1], vertical_alignment="bottom")
picker.markdown(
    f"**Forecast priced: {option_label(ROW)}** — it combines our single models and SMARD's own "
    "forecast."
)
clicked = button.button(
    "💸 Calculate the savings", type="primary", use_container_width=True
)
if clicked:
    st.session_state["rebap_revealed"] = ROW


def money_rain():
    """💸 rising from the bottom of the screen to the top, once (CSS only, no clicks blocked)."""
    rng = np.random.default_rng()
    drops = "".join(
        f'<span style="left:{rng.uniform(0, 96):.1f}vw;'
        f"animation-delay:{rng.uniform(0, RAIN_SECONDS * 0.5):.2f}s;"
        f'font-size:{rng.uniform(1.6, 3.0):.1f}rem">💸</span>'
        for _ in range(RAIN_DROPS)
    )
    st.markdown(
        f"""
<style>
.money-rain {{position:fixed; inset:0; pointer-events:none; z-index:999999; overflow:hidden}}
.money-rain span {{position:absolute; bottom:-4rem; opacity:0;
  animation:money-rise {RAIN_SECONDS}s ease-out forwards}}
@keyframes money-rise {{
  0% {{transform:translateY(0) rotate(-10deg); opacity:0}}
  10% {{opacity:1}}
  100% {{transform:translateY(-115vh) rotate(15deg); opacity:0}}
}}
@media (prefers-reduced-motion: reduce) {{.money-rain {{display:none}}}}
</style>
<div class="money-rain">{drops}</div>
""",
        unsafe_allow_html=True,
    )


if st.session_state.get("rebap_revealed") != ROW:
    st.info("Press **Calculate** to see what our ensemble saves against SMARD.")
    next_page("app_pages/rebap.py")
    st.stop()

if clicked:
    money_rain()

# --- Results ----------------------------------------------------------------------------------
smard_cost = rebap.cost[SMARD_ROW].sum()
saved = rebap.saved[ROW]
saved_pct = 100 * saved.sum() / smard_cost
skill = acc.value.loc[ROW, "skill_pct"]
top = rebap.price >= rebap.price.quantile(PRICE_TOP)
top_share = saved[top].sum() / saved.sum()
name = option_label(ROW)

if abs(saved_pct - skill) < 1:
    versus = (
        "about the same share as in MWh: the lead holds when balancing is expensive"
    )
elif saved_pct > skill:
    versus = "more than in MWh: the lead is larger when balancing is expensive"
else:
    versus = (
        "less than in MWh: the lead is smaller in the hours when balancing is expensive"
    )
st.markdown(
    f"Priced at each hour's imbalance price, SMARD's misses over the test year add up to "
    f"**{money(smard_cost, False)}**. **{name}** misses **{money(saved.sum(), False)} less** "
    f"({saved_pct:.1f} %, against {skill:.1f} % fewer MWh) — {versus}."
)

cards = st.columns(4)
cards[0].metric(
    "SMARD's misses, priced",
    money(smard_cost, False),
    help=f"Sum over {len(rebap.cost):,} test hours of |miss| × the hour's average |reBAP|.",
)
cards[1].metric(
    "This forecast, priced",
    money(rebap.cost[ROW].sum(), False),
    f"{money(-saved.sum())} vs SMARD",
    delta_color="inverse",
)
cards[2].metric(
    "Saved over the year",
    money(saved.sum()),
    f"{saved_pct:+.1f} % of SMARD's",
    help="SMARD's priced misses minus this forecast's. Positive: this forecast was cheaper.",
)
cards[3].metric(
    f"From the most expensive {100 * (1 - PRICE_TOP):.0f} % of hours",
    f"{100 * top_share:.1f} %",
    help=f"Share of the saving earned in the {int(top.sum())} hours with an average |reBAP| of "
    f"at least {rebap.price[top].min():,.0f} EUR/MWh. A large share means a few price spikes "
    "carry the result.",
)

left, right = st.columns(2)
with left:
    running = saved.cumsum() / 1e6
    fig = go.Figure(
        go.Scatter(
            x=running.index,
            y=running.to_numpy(),
            name=label(ROW),
            line=model_line(ROW[0]),
            hovertemplate="%{y:+,.1f} million €<extra></extra>",
        )
    )
    for hour in saved.abs().nlargest(3).index:
        fig.add_annotation(
            x=hour,
            y=running[hour],
            text=f"{hour:%d %b %H:00}<br>{money(saved[hour])}",
            showarrow=True,
            arrowhead=0,
            ay=-45,
            font={"color": INK, "size": 11},
        )
    style_plotly(
        fig,
        "Money saved against SMARD, running total",
        "million € saved (running total)",
    )
    fig.update_xaxes(dtick="M2", tickformat="%b %Y")
    fig.update_layout(showlegend=False)
    st.plotly_chart(themed(fig), key="rebap_running")
    st.caption(
        "Steps are single hours with a high price and a large difference in miss — the three "
        "largest are labelled."
    )

with right:
    masks = {"overall": pd.Series(True, index=saved.index)} | {
        c: acc.bin_hours[c] for c in REBAP_CATEGORIES[1:]
    }
    values = [saved[masks[c]].sum() / 1e6 for c in REBAP_CATEGORIES]
    fig = go.Figure(
        go.Bar(
            x=[
                f"{SITUATION[c]}<br>({int(masks[c].sum()):,} h)"
                for c in REBAP_CATEGORIES
            ],
            y=values,
            marker=bar_style(ROW),
            text=[f"{v:+,.1f}" for v in values],
            textposition="outside",
            hovertemplate="%{y:+,.1f} million €<extra></extra>",
        )
    )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
    style_plotly(
        fig, "Money saved against SMARD, by situation", "million € saved", height=460
    )
    fig.update_layout(hovermode="closest", showlegend=False)
    st.plotly_chart(themed(fig), key="rebap_situations")
    tails = REBAP_CATEGORIES[1:]
    share = {c: saved[masks[c]].sum() / saved.sum() for c in tails}
    biggest = max(tails, key=share.get)
    st.caption(
        f"Above zero: this forecast was cheaper. {SITUATION[biggest]} is "
        f"{100 * masks[biggest].mean():.1f} % of the hours but brings {100 * share[biggest]:.1f} % "
        "of the saving. The groups overlap (the lowest 1 % lie below zero), so they don't add up."
    )

with st.expander("How the money is calculated"):
    st.markdown(f"""
- **The price of an hour** is the average of its four quarter-hour imbalance prices, each taken
  as an amount (|reBAP|), in EUR/MWh. Over the test year it averages
  {rebap.price.mean():,.0f} EUR/MWh; the most expensive hour reaches {rebap.price.max():,.0f} EUR/MWh.
- **The cost of a miss** is |forecast − actual| (MWh) × that price.
- **Why the amount, not the signed price?** The reBAP turns negative when there is too much power.
  With the signed price, being *more* accurate in a negative-price hour would count as a loss.
- **Why only a yardstick?** The reBAP itself reacts to the system's imbalance, and the national
  forecast miss is not what any one market party settles.
""")
    dear = saved.abs().nlargest(TOP_HOURS).index
    table = pd.DataFrame(
        {
            "hour": dear.strftime("%a %d %b %Y, %H:00"),
            "price (EUR/MWh)": rebap.price[dear].to_numpy(),
            "SMARD miss (MWh)": acc.errors[SMARD_ROW][dear].to_numpy(),
            "this forecast's miss (MWh)": acc.errors[ROW][dear].to_numpy(),
            "saved": saved[dear].to_numpy() / 1e6,
        }
    )
    st.markdown(f"**The {TOP_HOURS} hours that moved the result most** ({label(ROW)})")
    st.dataframe(
        table.style.format(
            {
                "price (EUR/MWh)": "{:,.0f}",
                "SMARD miss (MWh)": "{:+,.0f}",
                "this forecast's miss (MWh)": "{:+,.0f}",
                "saved": "{:+,.1f} million €",
            }
        ),
        hide_index=True,
    )

    euro = rebap.saved.sum()
    every = pd.DataFrame(
        {
            "model": [label(row) + (" ◀" if row == ROW else "") for row in rebap.saved],
            "saved": [euro[row] / 1e6 for row in rebap.saved],
            "% of SMARD's": [100 * euro[row] / smard_cost for row in rebap.saved],
            "rank in euros": euro.rank(ascending=False).astype(int).to_numpy(),
            "average miss (MWh)": [acc.value.loc[row, "MAE"] for row in rebap.saved],
        }
    ).sort_values("rank in euros")
    st.markdown("**Every priced model in euros** (◀ = selected)")
    st.dataframe(
        every.style.format(
            {
                "saved": "{:+,.1f} million €",
                "% of SMARD's": "{:+.1f} %",
                "average miss (MWh)": "{:,.0f}",
            }
        ),
        hide_index=True,
    )
    st.caption(
        "For comparison only: the page prices our ensemble. Not every model was priced."
    )

next_page("app_pages/rebap.py")
