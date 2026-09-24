"""Shared data-loading helpers for the Streamlit app.

Ported from `notebooks/01_eda/team-EDA.ipynb` §1 (Setup) — the loading, dtype conversion and
derived-column logic is kept identical to that notebook, which stays the source of truth. This
module only adds Streamlit-specific caching and "artifact not yet generated" handling on top.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

# The CSV headers exactly as notebooks/API-connection.ipynb writes them.
COLUMNS = {
    "Wind Offshore": "wind_off",
    "Wind Onshore": "wind_on",
    "Solar": "solar",
    "Grid Load": "grid_load",
    "Residual Load": "residual_load",
    "Forecast Wind + Solar": "fc_gen_wind_solar",
    "Forecast Grid Load": "fc_grid_load",
    "Forecast Residual Load": "fc_residual_load",
    "Capacity Wind Offshore": "cap_wind_off",
    "Capacity Wind Onshore": "cap_wind_on",
    "Capacity Solar": "cap_solar",
}

SERIES = [
    "wind_off", "wind_on", "solar", "grid_load", "residual_load",
    "fc_gen_wind_solar", "fc_grid_load", "fc_residual_load",
    "cap_wind_off", "cap_wind_on", "cap_solar",
]

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

SEASON_OF_MONTH = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}
SEASON_ORDER = ["winter", "spring", "summer", "autumn"]

DERIVED = [
    "renewables", "year", "month", "hour", "dow", "is_weekend",
    "date", "season", "season_year", "spans_gap",
]


def _find_data_dir() -> Path:
    """Walk upward from this file's location to the first parent holding a `data/` folder."""
    start = Path(__file__).resolve().parent
    for p in (start, *start.parents):
        if (p / "data").is_dir():
            return p / "data"
    raise RuntimeError(f"no data/ directory found starting from {start} or any parent.")


def _find_smard_csv(data_dir: Path) -> Path:
    """Locate the SMARD export inside `data_dir`.

    CLAUDE.md and `team-EDA.ipynb` name it `data/smard.csv`; this repository's current local
    clone instead holds a date-stamped `smard_*.csv`. Rather than guessing which file to load,
    prefer the canonical name and fall back to a single unambiguous `smard*.csv` match, failing
    clearly otherwise.
    """
    canonical = data_dir / "smard.csv"
    if canonical.exists():
        return canonical
    candidates = sorted(data_dir.glob("smard*.csv"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"no smard.csv (or smard*.csv) found in {data_dir}. data/ is gitignored, so the "
            "file is not in a fresh clone — regenerate it by running "
            "notebooks/API-connection.ipynb top to bottom."
        )
    raise FileNotFoundError(
        f"multiple smard*.csv candidates found in {data_dir}: "
        f"{[c.name for c in candidates]} — expected exactly one; resolve the ambiguity before "
        "loading."
    )


@st.cache_data
def load_smard() -> pd.DataFrame:
    """Load and prepare `time_series`, identically to `team-EDA.ipynb` §1.3-§1.4."""
    data_dir = _find_data_dir()
    path = _find_smard_csv(data_dir)

    raw = pd.read_csv(path, delimiter=";", encoding="utf-8-sig")
    assert set(raw.columns) == {"timestamp"} | set(COLUMNS), (
        f"unexpected CSV header: {sorted(set(raw.columns) ^ ({'timestamp'} | set(COLUMNS)))}"
    )

    raw = raw.rename(columns=COLUMNS)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"], format="%Y-%m-%d %H:%M")
    for col in COLUMNS.values():
        raw[col] = raw[col].str.replace(",", ".").astype(float)

    time_series = raw.set_index("timestamp").sort_index()

    time_series["renewables"] = time_series[["wind_on", "wind_off", "solar"]].sum(axis=1)
    time_series["year"] = time_series.index.year
    time_series["month"] = time_series.index.month
    time_series["hour"] = time_series.index.hour
    time_series["dow"] = time_series.index.dayofweek
    time_series["is_weekend"] = time_series.index.dayofweek >= 5
    time_series["date"] = time_series.index.date
    time_series["season"] = pd.Categorical(
        time_series.index.month.map(SEASON_OF_MONTH), categories=SEASON_ORDER, ordered=True
    )
    time_series["season_year"] = time_series.index.year + (time_series.index.month == 12)
    time_series["spans_gap"] = time_series.index.to_series().diff() > pd.Timedelta("1h")

    assert list(time_series.columns) == SERIES + DERIVED, list(time_series.columns)
    return time_series


def get_years(time_series: pd.DataFrame) -> list[int]:
    """`YEARS`, computed from the loaded data — never hardcoded (`team-EDA.ipynb` §1.4)."""
    return sorted(int(y) for y in time_series["year"].unique())
