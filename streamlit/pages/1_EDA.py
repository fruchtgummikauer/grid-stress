"""EDA page — sourced from `notebooks/01_eda/team-EDA.ipynb`.

Per `.claude/specs/Streamlit-draft.md` §5 Page 2: `team-EDA.ipynb` is the primary, team-reviewed
source for this page. Every plot below ports that notebook's actual plotting code (cited by
section number) rather than re-deriving new analysis. `team-EDA.ipynb` itself is read-only —
nothing here edits it.

This version covers §1 (dataset overview), §3.7 (residual-load extreme calendar signature), §5.1
(calendar heatmap), §5.2 (the national daily rhythm), §5.3 (grid load on national holidays), §6.3
(residual-load distribution and ramps), §6.5 (residual-load level change over the years), §7.2
(correlation) and §7.3 (load duration curves), plus the closing Findings. §3.8 (worked
extreme-episode examples) is not yet ported — left for a later iteration rather than silently
combined into the above.
"""

import calendar

import holidays
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from data_loading import get_years, load_smard
from viz_helpers import (
    COLORS,
    DAY_TYPE_COLOR,
    DIV_CMAP,
    GRID_LOAD_CMAP,
    RAMP_COLOR,
    SEASON_ORDER,
    TAIL_COLOR,
    residual_scale,
    series_style,
    year_colors,
)

st.set_page_config(page_title="EDA — Grid Stress", page_icon="📊", layout="wide")
st.title("Exploratory Data Analysis")
st.caption("Source: `notebooks/01_eda/team-EDA.ipynb` (team-reviewed, read-only).")

try:
    time_series = load_smard()
except (FileNotFoundError, RuntimeError) as exc:
    st.error(f"SMARD data not available: {exc}")
    st.stop()

# --- §1 Dataset overview ---
st.header("Dataset overview")
col1, col2 = st.columns(2)
with col1:
    st.metric("Rows", f"{len(time_series):,}")
    st.metric("Years covered", f"{time_series['year'].nunique()}")
with col2:
    st.metric("Start", f"{time_series.index.min():%Y-%m-%d %H:%M}")
    st.metric("End", f"{time_series.index.max():%Y-%m-%d %H:%M}")

series_cols = ["wind_off", "wind_on", "solar", "grid_load", "residual_load",
               "fc_gen_wind_solar", "fc_grid_load", "fc_residual_load",
               "cap_wind_off", "cap_wind_on", "cap_solar"]
missing = pd.DataFrame({
    "n_missing": time_series[series_cols].isna().sum(),
    "share_%": (time_series[series_cols].isna().mean() * 100).round(3),
})
with st.expander("Missing values per series (§2.1)"):
    st.dataframe(missing)
    st.caption(
        f"{int(missing['n_missing'].sum())} missing values across {len(series_cols)} series, "
        f"{int(time_series.index.duplicated().sum())} duplicate timestamps."
    )

st.divider()

# --- §3.7 Residual-load extremes: calendar signature ---
st.header("When do residual-load extremes occur? (§3.7)")
st.caption("Descriptive slice — the extreme 1 % of hours at each tail, by rank. Not a risk definition.")

n_tail = round(0.01 * len(time_series))
low_tail = time_series["residual_load"].nsmallest(n_tail)
high_tail = time_series["residual_load"].nlargest(n_tail)

DIMENSIONS = [
    ("year", lambda idx: idx.year, None),
    ("month", lambda idx: idx.month, range(1, 13)),
    ("hour of day", lambda idx: idx.hour, range(24)),
    ("day of week", lambda idx: idx.dayofweek, range(7)),
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

fig, axes = plt.subplots(2, 2, figsize=(15, 8))
for ax, (label, extract, full_range) in zip(axes.flat, DIMENSIONS):
    frame = pd.DataFrame({
        "lowest 1 % (renewable oversupply)": pd.Series(extract(low_tail.index)).value_counts(normalize=True),
        "highest 1 % (undersupply / tight margins)": pd.Series(extract(high_tail.index)).value_counts(normalize=True),
    })
    if full_range is not None:
        frame = frame.reindex(list(full_range))
    frame = frame.sort_index() * 100

    x = np.arange(len(frame))
    ax.bar(x - 0.2, frame.iloc[:, 0].to_numpy(), width=0.4, label=frame.columns[0], color=TAIL_COLOR["low"])
    ax.bar(x + 0.2, frame.iloc[:, 1].to_numpy(), width=0.4, label=frame.columns[1], color=TAIL_COLOR["high"])
    ax.set_xticks(x)
    ax.set_xticklabels(DAY_NAMES if label == "day of week" else [str(v) for v in frame.index], fontsize=9)
    ax.set_title(f"by {label}", fontsize=12, pad=8)
    ax.set_ylabel("share of that tail (%)", color="grey")
    ax.set_xlabel(label)
    ax.grid(axis="y", color="0.92", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
axes.flat[0].legend(loc="upper left", bbox_to_anchor=(0, 1.35), ncol=2, frameon=False, fontsize=9)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- The two extremes have **almost mirror-image calendar signatures**.
- **Low/negative (renewable oversupply):** overwhelmingly recent, concentrated in high-solar
  months, midday hours, weekends.
- **High (undersupply / tight reserve margins):** weekday-only, spread evenly across years,
  winter months, evening peak with a secondary morning peak.
- The low extreme is growing and seasonal (renewable build-out); the high extreme is stable and
  structural (the classic winter stress case).
"""
)

st.divider()

# --- §5.1 Calendar context, both series ---
st.header("Calendar heatmap (§5.1)")
st.caption(
    "`residual_load` is coloured by §1.2's `residual_scale`, which keeps zero white in every "
    "case: a diverging scale centred on zero when the surface crosses it, otherwise the "
    "relevant half of that same scale starting at zero."
)

MONTH_NAMES = list(calendar.month_abbr)[1:]
HEATMAP_DIMENSIONS = [
    ("month", MONTH_NAMES, range(1, 13), "Month × hour"),
    ("dow", DAY_NAMES, range(7), "Weekday × hour"),
    ("season", SEASON_ORDER, SEASON_ORDER, "Season × hour"),
]

fig = plt.figure(figsize=(16.5, 10.5), constrained_layout=True)
gs = fig.add_gridspec(2, 4, width_ratios=[1.08, 1.08, 1.00, 0.045])

SERIES_LABEL = {"grid_load": "Grid load", "residual_load": "Residual load"}
for row, col in enumerate(["grid_load", "residual_load"]):
    tables = [
        time_series.pivot_table(index=dim, columns="hour", values=col, aggfunc="mean", observed=True).reindex(order)
        for dim, _, order, _ in HEATMAP_DIMENSIONS
    ]
    vmin = min(np.nanmin(t.to_numpy()) for t in tables)
    vmax = max(np.nanmax(t.to_numpy()) for t in tables)

    if col == "residual_load":
        cmap, norm = residual_scale(vmin, vmax)
    else:
        cmap, norm = GRID_LOAD_CMAP, plt.Normalize(vmin=vmin, vmax=vmax)

    im = None
    for position, (table, (_, labels, _, title)) in enumerate(zip(tables, HEATMAP_DIMENSIONS)):
        ax = fig.add_subplot(gs[row, position])
        im = ax.imshow(table.to_numpy(), aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
        ax.set_title(f"{SERIES_LABEL[col]} — {title}", loc="left", fontsize=12, pad=8)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xticks(range(0, 24, 6), ["00", "06", "12", "18"])
        ax.set_xlabel("local hour", labelpad=6)

    cax = fig.add_subplot(gs[row, 3])
    fig.colorbar(im, cax=cax).set_label(f"mean {SERIES_LABEL[col].lower()} (average MW)", labelpad=10)

fig.suptitle("How do the two loads vary across calendar context?", fontsize=17)
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- `grid_load` is dominated by working days: 08:00–20:00 present in every month and every
  weekday, dimmer at weekends and in summer.
- `residual_load`: solar pushes it towards zero from March to September; Saturday and Sunday are
  lighter than the working week at every hour.
- Calendar structure: `grid_load` is organised by the working day, `residual_load` by
  **sunlight** with the working day as a secondary effect — the whole reason `residual_load`,
  not `grid_load`, is this project's target.
"""
)

st.divider()

# --- §5.2 The national daily rhythm ---
st.header("The national daily rhythm (§5.2)")

RHYTHM_COLS = ["grid_load", "residual_load", "wind_on", "wind_off", "solar"]
hourly_profile = time_series.groupby("hour")[RHYTHM_COLS].mean()

fig, ax = plt.subplots(figsize=(13.7, 6.0))
for col in RHYTHM_COLS:
    ax.plot(hourly_profile.index, hourly_profile[col].to_numpy(), **series_style(col))
ax.axhline(0, color=COLORS["muted"], lw=0.8)
ax.set_title("The national daily rhythm", fontsize=15, pad=12)
ax.set_xlabel("local clock hour")
ax.set_ylabel("average MW", color="grey")
ax.set_xticks(range(0, 24, 3), [f"{h:02d}:00" for h in range(0, 24, 3)])
ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.legend(frameon=False, ncol=3)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- Demand ramps up from its overnight valley to a daytime plateau.
- Solar rises and falls inside that plateau; wind is nearly flat across the day.
- `residual_load` is what's left over — the demand other power plants need to cover.
"""
)

st.divider()

# --- §5.3 Grid load on national holidays ---
st.header("Grid load on national holidays (§5.3)")
st.caption('Holidays from `holidays.country_holidays("DE")`, no `subdiv` — federal holidays only.')

de_holidays = holidays.country_holidays("DE", years=range(min(get_years(time_series)), max(get_years(time_series)) + 1))
holiday_dates = set(de_holidays)
is_holiday = pd.Series(time_series.index.date, index=time_series.index).isin(holiday_dates)

holiday_profile = (
    time_series.groupby([is_holiday.rename("is_holiday"), "hour"])["grid_load"]
    .mean()
    .unstack(0)
    .rename(columns={False: "Ordinary day", True: "National holiday"})
)

fig, ax = plt.subplots(figsize=(11.8, 6.1))
ax.plot(holiday_profile.index, holiday_profile["Ordinary day"].to_numpy(),
        color=DAY_TYPE_COLOR["weekday"], lw=2.8, label="Ordinary day")
ax.plot(holiday_profile.index, holiday_profile["National holiday"].to_numpy(),
        color=DAY_TYPE_COLOR["weekend"], lw=2.8, label="National holiday")
ax.set_title("Grid-load profile on national holidays", fontsize=15, pad=12)
ax.set_xlabel("local clock hour")
ax.set_ylabel("average MW", color="grey")
ax.set_xticks(range(0, 24, 2))
ax.grid(axis="y", color=COLORS["grid"], lw=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.legend(frameon=False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

peak_gap = holiday_profile["Ordinary day"] - holiday_profile["National holiday"]
st.markdown(
    f"""
- Holidays **lower and soften** the daily load; the overnight valley is nearly unchanged
  (base load like hospitals/telecoms doesn't take holidays).
- Holiday count in the dataset: **{int(is_holiday.groupby(time_series.index.date).any().sum())}**.
  Largest gap: **{peak_gap.max():,.0f} MW** at {peak_gap.idxmax():02d}:00. Smallest gap:
  **{peak_gap.min():,.0f} MW** at {peak_gap.idxmin():02d}:00.
- For modelling: a holiday behaves like an extra weekend day rather than a shutdown.
"""
)

st.divider()

# --- §6.3 Residual-load distribution and ramps ---
st.header("Residual-load distribution and ramps (§6.3)")

residual = time_series["residual_load"]
ramp = residual.diff().mask(time_series["spans_gap"])
abs_ramp = ramp.abs().dropna()

shape = pd.Series({
    "mean": residual.mean(), "median": residual.median(), "std": residual.std(),
    "skew": residual.skew(), "excess kurtosis": residual.kurt(),
    "min": residual.min(), "max": residual.max(), "range": residual.max() - residual.min(),
    "negative hours": int((residual < 0).sum()),
    "negative share %": 100 * (residual < 0).mean(),
})
st.dataframe(shape.round(2).to_frame("residual_load (MWh)"))

RAMP_QUANTILES = [0.80, 0.90, 0.95, 0.975, 0.99, 0.995]
ramp_percentiles = abs_ramp.quantile(RAMP_QUANTILES)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.8), gridspec_kw={"width_ratios": [1.15, 1]}, constrained_layout=True)
axes[0].hist(residual, bins=120, color=COLORS["black"])
for q in (0.01, 0.99):
    axes[0].axvline(residual.quantile(q), color=COLORS["muted"], linestyle="--", linewidth=1)
    axes[0].text(residual.quantile(q), axes[0].get_ylim()[1] * 0.94, f" {q:.0%}", color=COLORS["muted"], fontsize=9)
axes[0].axvline(0, color=COLORS["vermillion"], linewidth=1.6, ymin=0, ymax=0.8)
axes[0].text(0, axes[0].get_ylim()[1] * 0.80, "zero", color=COLORS["vermillion"], fontsize=10)
axes[0].axvline(residual.median(), color=COLORS["orange"], linewidth=1.6)
axes[0].text(residual.median(), axes[0].get_ylim()[1] * 0.80, "Median", color=COLORS["orange"], fontsize=10)
axes[0].set_title("Distribution of hourly Residual Load", fontsize=13, pad=8)
axes[0].set_xlabel("Residual Load (MWh)")
axes[0].set_ylabel("hours", color="grey")

axes[1].plot(np.array(RAMP_QUANTILES) * 100, ramp_percentiles.to_numpy(), color=RAMP_COLOR, marker="o", markersize=5.5, lw=2.5)
for q in (0.90, 0.99):
    axes[1].annotate(f"P{q * 100:g} {ramp_percentiles.loc[q]:,.0f}",
                      xy=(q * 100, ramp_percentiles.loc[q]), xytext=(8, -4),
                      textcoords="offset points", fontsize=10, color=COLORS["muted"])
axes[1].margins(x=0.10, y=0.12)
axes[1].set_title("High extreme of one-hour residual ramps", fontsize=13, pad=8)
axes[1].set_xlabel("ramp percentile")
axes[1].set_ylabel("|1-hour residual ramp| (MW/h)", color="grey")

for ax in axes:
    ax.grid(axis="y", color="0.92", linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- Distribution is close to symmetric with a mild negative skew, spread over a range of more
  than 80 GW; no second mode, no sharp cut-off.
- The minimum sits further below the median than the maximum sits above it — the low tail is the
  longer one.
- 1.26 % of all hours are already negative — a small but growing, summer-clustered share.
- Ramp sizes rise steeply in the upper percentiles: the hours that matter operationally are rare
  and large.
"""
)

st.divider()

# --- §6.5 Residual-load level change over the years ---
st.header("Residual-load level change over the years (§6.5)")
st.caption(
    "Matched calendar window (1 Jan to the current record's latest date, applied to every "
    "year), so the in-progress final year is compared like for like."
)

complete_years = [
    year for year in get_years(time_series)
    if time_series.index.min() <= pd.Timestamp(year=year, month=1, day=1)
    and time_series.index.max() >= pd.Timestamp(year=year, month=12, day=31, hour=23)
]
partial_years = [year for year in get_years(time_series) if year not in complete_years]

cutoff = time_series.index.max()
matched = time_series[
    (time_series.index.month < cutoff.month)
    | ((time_series.index.month == cutoff.month) & (time_series.index.day <= cutoff.day))
]

annual_level = matched["residual_load"].groupby(matched.index.year).agg(
    median="median", p10=lambda s: s.quantile(0.10), p90=lambda s: s.quantile(0.90), n="count",
)
st.dataframe(annual_level.round(0))

LEVEL_YEAR_COLOR = year_colors(annual_level.index)
fig, ax = plt.subplots(figsize=(11.8, 6.1), constrained_layout=True)
for year, row in annual_level.iterrows():
    color = LEVEL_YEAR_COLOR[year]
    ax.plot([year, year], [row["p10"], row["p90"]], lw=6, color=color, alpha=0.30, solid_capstyle="round", zorder=1)
    ax.scatter(year, row["median"], s=110, color=color, edgecolor="white", linewidth=0.8, zorder=3)
    ax.text(year, row["p90"] + 0.02 * annual_level["p90"].max(), f"hours={int(row['n']):,}",
            va="bottom", ha="center", fontsize=9.5, color=COLORS["muted"])
ax.set_xticks(list(annual_level.index),
              [f"{year}\n· partial" if year in partial_years else str(year) for year in annual_level.index])
ax.set_title("Residual-load level by year (same calendar window)", fontsize=15, pad=12)
ax.set_xlabel("year")
ax.set_ylabel("Median with P10–P90 range (MWh)", color="grey")
ax.grid(axis="y", color="0.92", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- **Median residual load falls year on year**, but the **P10 end falls faster than the median**:
  the distribution is not sliding down as a block, it is **stretching downward**.
- The low extreme of each year reaches further towards, and eventually past, zero — while the
  P90 (high extreme) barely moves.
- Central finding: residual load has become lower at its low extreme and not lower at its high
  extreme, and faster-moving throughout — the oversupply tail is growing while the tight-margin
  tail stays put.
"""
)

st.divider()

# --- §7.2 Correlation matrix ---
st.header("Correlation matrix (§7.2)")
st.caption("Spearman rank correlation — not the Pearson matrix.")

correlation_frame = time_series[["grid_load", "wind_on", "wind_off", "solar", "residual_load"]].copy()
correlation_frame["ws_share"] = time_series["renewables"] / time_series["grid_load"]
CORRELATION_LABELS = ["Grid load", "Onshore wind", "Offshore wind", "Solar", "Residual load", "Wind+solar share"]
correlation = correlation_frame.corr(method="spearman")

fig, ax = plt.subplots(figsize=(8.2, 7))
im = ax.imshow(correlation.to_numpy(), vmin=-1, vmax=1, cmap=DIV_CMAP)
ax.set_title("Which system variables tend to move together?", fontsize=14, pad=12)
ax.set_xticks(range(len(CORRELATION_LABELS)), CORRELATION_LABELS, rotation=25, ha="right")
ax.set_yticks(range(len(CORRELATION_LABELS)), CORRELATION_LABELS)
for row in range(len(CORRELATION_LABELS)):
    for col in range(len(CORRELATION_LABELS)):
        value = correlation.iloc[row, col]
        ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=9.2,
                color="white" if abs(value) >= 0.58 else COLORS["black"],
                fontweight="bold" if abs(value) >= 0.58 else "normal")
fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04).set_label("Spearman r")
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.markdown(
    """
- `residual_load` correlates strongly and positively with `grid_load`, and strongly and
  negatively with the wind+solar share.
- Solar and wind are close to uncorrelated — independent inputs.
"""
)

st.divider()

# --- §7.3 Load duration curves ---
st.header("Load duration curves (§7.3)")
st.caption("Sorted highest to lowest over the whole multi-year record, so \"% of hours\" mixes several years' seasons together.")

share_of_hours = np.linspace(0, 100, len(time_series))
fig, ax = plt.subplots(figsize=(12, 5))
for col in ["grid_load", "residual_load"]:
    ax.plot(share_of_hours, time_series[col].sort_values(ascending=False).to_numpy(), **series_style(col))
ax.axhline(0, color=COLORS["black"], linewidth=0.8)
ax.set_title("Load duration curves over the full record", fontsize=15, pad=12)
ax.set_xlabel("% of hours at or above this level")
ax.set_ylabel("MWh", color="grey")
ax.grid(axis="y", color="0.92", linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
ax.legend(frameon=False)
plt.tight_layout()
st.pyplot(fig)
plt.close(fig)

negative_share = 100 * (time_series["residual_load"] < 0).mean()
st.markdown(
    f"""
- The two curves have **different shapes, not just different levels**: `grid_load` is flat
  through the middle and steep only at its extremes (a predictable demand curve); `residual_load`
  falls steadily across its whole range and dives through zero at the right end.
- Residual load is negative in **{negative_share:.2f} %** of hours.
"""
)

st.divider()

# --- Findings ---
st.header("Findings")
st.markdown(
    """
**Univariate and time structure.** Solar is the only series with a clear multi-year trend; wind
is dominated by weather variance; grid load is remarkably stable. Residual-load extremes have
mirror-image calendar signatures: the oversupply tail is recent, summer, midday and weekend; the
tight-reserve-margin extreme is winter, weekday, evening, and spread evenly across years.

**Distribution and ramps.** Residual load's distribution is broad, unimodal and mildly
left-skewed, with about 2 % of hours already negative. Its P10 has fallen sharply while its P90
has barely moved, and its ramp P99 has grown much faster than its median.

**The single most consequential observation:** the two extremes of residual load are not one
phenomenon — they differ in season, hour, day type, trend and mechanism. Whether the risk flag
should treat them as one target or two (and whether the high extreme should be engineered by ramp
rate rather than level) is an open modelling decision.

*Source: `team-EDA.ipynb`'s closing "Findings" section, condensed to what this page's plots
support.*
"""
)
