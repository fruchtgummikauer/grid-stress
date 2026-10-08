"""What is it worth? — our smaller forecast misses priced at the imbalance price (reBAP).

Reads only viz-03's cost export (`notebooks/05_modeling/visualization-03-rebap-cost.ipynb`, spec 11)
through `model_results.load_rebap_cost`, never `rebap.csv` (CLAUDE.md). The page explains the
simplifications first; the numbers appear only after the visitor presses "Calculate" (💸 rising on
the click, team 2026-10-07). Default: our ensemble (spec 09's ensemble pick) and our best model
overall; a switch compares every priced model (team 2026-10-08).
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.layout import header, next_page
from components.lineup import load_lineup
from components.naming import SITUATION, bar_style, label, money
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
LINEUP = load_lineup()
rebap = get_or_stop(load_rebap_cost)
if acc.problems or rebap.problems:
    st.error(
        "The cost export doesn't match the model exports, so this page is not shown:\n\n"
        + "\n".join(f"- {problem}" for problem in acc.problems + rebap.problems)
    )
    st.stop()

header("What is it worth?", "A better forecast, priced in euros.")

st.markdown(
    "**What is the reBAP?** The reBAP (*regelzonenübergreifender einheitlicher "
    "Bilanzausgleichsenergiepreis*) is Germany's single imbalance price, set for every quarter-hour. "
    "Companies that feed in or use more or less electricity than they announced pay or receive it "
    "for the difference. It rises when the grid needs a lot of balancing and can turn negative when "
    "there is too much power, which makes it a fair yardstick for what a forecast miss is worth."
)

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

# --- Forecasts and button ---------------------------------------------------------------------
# Default: our ensemble and our best model overall; the switch opens every priced model
PRICED = sorted(rebap.saved.columns, key=lambda row: -rebap.saved[row].sum())
DEFAULT = [row for row in (acc.ensemble_pick, LINEUP.first) if row in PRICED]
if not DEFAULT:
    st.warning(
        "Neither our ensemble nor our best model is in the cost export, so there is nothing to "
        "price. Re-run visualization-03-rebap-cost.ipynb."
    )
    st.stop()


def name(row):
    """Display name: "Random forest hybrid (best overall)" for a headline row, else the label."""
    return LINEUP.public_label(row)


switch, picker, button = st.columns([1.2, 2.3, 1], vertical_alignment="bottom")
comparing = switch.toggle(
    "Compare all priced models",
    key="rebap_compare",
    help="Off: our ensemble and our best single model. On: every model in the cost export; "
    "remove or add models in the list.",
)
if comparing:
    ROWS = (
        picker.multiselect(
            "Forecasts priced",
            PRICED,
            default=PRICED,
            format_func=name,
            key="rebap_rows",
        )
        or DEFAULT[:1]
    )
else:
    ROWS = DEFAULT
    picker.markdown(
        "**Forecasts priced:** "
        + " and ".join(f"**{name(row)}**" for row in ROWS)
        + " — the ensemble combines our single models and SMARD's own forecast."
    )
clicked = button.button(
    "💸 Calculate the savings", type="primary", use_container_width=True
)
if clicked:
    st.session_state["rebap_revealed"] = True


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


if not st.session_state.get("rebap_revealed"):
    st.info("Press **Calculate** to see what our forecasts save against SMARD.")
    next_page("app_pages/rebap.py")
    st.stop()

if clicked:
    money_rain()

# --- Results ----------------------------------------------------------------------------------
LEAD = ROWS[0]  # the first forecast shown: its largest hours are labelled and listed
smard_cost = rebap.cost[SMARD_ROW].sum()
top = rebap.price >= rebap.price.quantile(PRICE_TOP)
saved = {row: rebap.saved[row] for row in ROWS}
euro = {row: saved[row].sum() for row in ROWS}
saved_pct = {row: 100 * euro[row] / smard_cost for row in ROWS}
skill = {row: acc.value.loc[row, "skill_pct"] for row in ROWS}
top_share = {row: saved[row][top].sum() / euro[row] for row in ROWS}


def versus(row):
    """The euro share against the MWh share, in words."""
    if abs(saved_pct[row] - skill[row]) < 1:
        return (
            "about the same share as in MWh: the lead holds when balancing is expensive"
        )
    if saved_pct[row] > skill[row]:
        return "more than in MWh: the lead is larger when balancing is expensive"
    return (
        "less than in MWh: the lead is smaller in the hours when balancing is expensive"
    )


st.markdown(
    f"Priced at each hour's imbalance price, SMARD's misses over the test year add up to "
    f"**{money(smard_cost, False)}**."
)
st.markdown(
    "\n".join(
        f"- **{name(row)}** misses **{money(euro[row], False)} less** ({saved_pct[row]:.1f} %, "
        f"against {skill[row]:.1f} % fewer MWh) — {versus(row)}."
        for row in ROWS
    )
)

if len(ROWS) <= 3:
    cards = st.columns(1 + len(ROWS))
    cards[0].metric(
        "SMARD's misses, priced",
        money(smard_cost, False),
        help=f"Sum over {len(rebap.cost):,} test hours of |miss| × the hour's average |reBAP|.",
    )
    for card, row in zip(cards[1:], ROWS):
        card.metric(
            f"Saved: {label(row)}",
            money(euro[row]),
            f"{saved_pct[row]:+.1f} % of SMARD's",
            help=f"Priced misses {money(rebap.cost[row].sum(), False)}. "
            f"{100 * top_share[row]:.1f} % of the saving comes from the most expensive "
            f"{100 * (1 - PRICE_TOP):.0f} % of hours ({int(top.sum())} hours with at least "
            f"{rebap.price[top].min():,.0f} EUR/MWh).",
        )
else:
    st.metric(
        "SMARD's misses, priced",
        money(smard_cost, False),
        help=f"Sum over {len(rebap.cost):,} test hours of |miss| × the hour's average |reBAP|.",
    )

# Saved per forecast, largest first
order = sorted(ROWS, key=lambda row: euro[row])
fig = go.Figure(
    go.Bar(
        y=[name(row) for row in order],
        x=[euro[row] / 1e6 for row in order],
        orientation="h",
        marker={
            "color": [bar_style(row)["color"] for row in order],
            "pattern": {"shape": [bar_style(row)["pattern"]["shape"] for row in order]},
        },
        text=[f"{money(euro[row])} · {saved_pct[row]:.1f} %" for row in order],
        textposition="outside",
        customdata=[100 * top_share[row] for row in order],
        hovertemplate="%{y}: %{x:+,.1f} million €<br>"
        "%{customdata:.1f} % of it from the most expensive 1 % of hours<extra></extra>",
    )
)
style_plotly(
    fig,
    "Saved against SMARD over the test year",
    "",
    xlabel="million € saved",
    height=150 + 45 * len(ROWS),
)
fig.update_xaxes(range=[0, max(euro.values()) / 1e6 * 1.35])
fig.update_yaxes(showgrid=False)
fig.update_layout(hovermode="closest", showlegend=False)
st.plotly_chart(themed(fig), key="rebap_by_model")

left, right = st.columns(2)
with left:
    fig = go.Figure()
    for row in ROWS:
        running = saved[row].cumsum() / 1e6
        fig.add_trace(
            go.Scatter(
                x=running.index,
                y=running.to_numpy(),
                name=label(row),
                line=model_line(row[0]),
                hovertemplate=f"{label(row)}: %{{y:+,.1f}} million €<extra></extra>",
            )
        )
    lead_running = saved[LEAD].cumsum() / 1e6
    for hour in saved[LEAD].abs().nlargest(3).index:
        fig.add_annotation(
            x=hour,
            y=lead_running[hour],
            text=f"{hour:%d %b %H:00}<br>{money(saved[LEAD][hour])}",
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
    fig.update_layout(showlegend=len(ROWS) > 1)
    st.plotly_chart(themed(fig), key="rebap_running")
    st.caption(
        "Steps are single hours with a high price and a large difference in miss — the three "
        f"largest for {label(LEAD)} are labelled."
    )

with right:
    masks = {"overall": pd.Series(True, index=rebap.saved.index)} | {
        c: acc.bin_hours[c] for c in REBAP_CATEGORIES[1:]
    }
    x = [f"{SITUATION[c]}<br>({int(masks[c].sum()):,} h)" for c in REBAP_CATEGORIES]
    fig = go.Figure()
    for row in ROWS:
        values = [saved[row][masks[c]].sum() / 1e6 for c in REBAP_CATEGORIES]
        fig.add_trace(
            go.Bar(
                x=x,
                y=values,
                name=label(row),
                marker=bar_style(row),
                text=[f"{v:+,.1f}" for v in values] if len(ROWS) <= 2 else None,
                textposition="outside",
                hovertemplate=f"{label(row)}: %{{y:+,.1f}} million €<extra></extra>",
            )
        )
    fig.add_hline(y=0, line={"color": COLORS["muted"], "width": 1})
    style_plotly(
        fig, "Money saved against SMARD, by situation", "million € saved", height=460
    )
    fig.update_layout(hovermode="closest", barmode="group", showlegend=len(ROWS) > 1)
    st.plotly_chart(themed(fig), key="rebap_situations")
    tails = REBAP_CATEGORIES[1:]
    share = {c: saved[LEAD][masks[c]].sum() / euro[LEAD] for c in tails}
    biggest = max(tails, key=share.get)
    st.caption(
        f"Above zero: the forecast was cheaper. For {label(LEAD)}, {SITUATION[biggest].lower()} "
        f"is {100 * masks[biggest].mean():.1f} % of the hours but brings "
        f"{100 * share[biggest]:.1f} % of the saving. The groups overlap (the lowest 1 % lie "
        "below zero), so they don't add up."
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
    dear = saved[LEAD].abs().nlargest(TOP_HOURS).index
    table = pd.DataFrame(
        {
            "hour": dear.strftime("%a %d %b %Y, %H:00"),
            "price (EUR/MWh)": rebap.price[dear].to_numpy(),
            "SMARD miss (MWh)": acc.errors[SMARD_ROW][dear].to_numpy(),
            "this forecast's miss (MWh)": acc.errors[LEAD][dear].to_numpy(),
            "saved": saved[LEAD][dear].to_numpy() / 1e6,
        }
    )
    st.markdown(f"**The {TOP_HOURS} hours that moved the result most** ({label(LEAD)})")
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

    total = rebap.saved.sum()
    every = pd.DataFrame(
        {
            "model": [label(row) + (" ◀" if row in ROWS else "") for row in PRICED],
            "saved": [total[row] / 1e6 for row in PRICED],
            "% of SMARD's": [100 * total[row] / smard_cost for row in PRICED],
            "rank in euros": range(1, len(PRICED) + 1),
            "average miss (MWh)": [acc.value.loc[row, "MAE"] for row in PRICED],
        }
    )
    st.markdown("**Every priced model in euros** (◀ = shown above)")
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

next_page("app_pages/rebap.py")
