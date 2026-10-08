"""Data check for the Who are we? page (Streamlit-v3.md §1.2 block 5, §2.3; moved off Home so
public visitors don't meet developer tooling first).

Lists every file the app reads, found or missing, with its modification date and the notebook
that produces it. `data/` is gitignored, so a fresh clone starts with most of them missing.
"""

from datetime import datetime

import pandas as pd
import streamlit as st

from data_loading import (
    RISK_LABELS_DAILY,
    RISK_LABELS_PRODUCER,
    _find_data_dir,
    _find_smard_csv,
)
from model_results import SOURCES

SMARD_PRODUCER = "notebooks/API-connection.ipynb"

# What each file feeds in the app
USED_BY = {
    "smard": "Do we beat SMARD?, Try it yourself (SMARD's hourly errors)",
    "scoreboard": "Do we beat SMARD?, Try it yourself",
    "hourly": "Do we beat SMARD?, Try it yourself",
    "risk_daily": "Do we beat SMARD? (risk days), Try it yourself (risk weeks)",
    "risk_hourly": "Do we beat SMARD? (risk days)",
    "ensemble_hourly": "Do we beat SMARD? (ensembles)",
    "ensemble_scoreboard": "Do we beat SMARD? (ensembles)",
    "ensemble_weights": "How do we forecast? (ensemble weights)",
    "rebap_cost": "What is it worth?",
}


def _row(relative, producer, used_by):
    """One table row for the file at data/`relative`."""
    path = _find_data_dir() / relative
    found = path.is_file()
    modified = datetime.fromtimestamp(path.stat().st_mtime) if found else None
    return {
        "file": f"data/{relative}",
        "status": "✅ found" if found else "❌ missing",
        "modified": f"{modified:%Y-%m-%d %H:%M}" if modified else "—",
        "used by": used_by,
        "produced by": producer,
    }


def data_check_table():
    """One row per file the app reads, in the order of SOURCES (plus smard.csv and the spec 02 risk labels first)."""
    try:
        smard = (
            _find_smard_csv(_find_data_dir()).relative_to(_find_data_dir()).as_posix()
        )
    except FileNotFoundError:
        smard = "smard.csv"
    rows = [
        _row(smard, SMARD_PRODUCER, "every page"),
        _row(RISK_LABELS_DAILY, RISK_LABELS_PRODUCER, "Home (risk days chart)"),
    ]
    for name, (relative, producer) in SOURCES.items():
        rows.append(_row(relative, producer, USED_BY[name]))
    return pd.DataFrame(rows)


def data_check():
    """A collapsed expander with the data check table and a one-line summary."""
    table = data_check_table()
    missing = int((table["status"] != "✅ found").sum())
    label = (
        "Data check: all files found"
        if not missing
        else f"Data check: {missing} file(s) missing"
    )
    with st.expander(label, expanded=False):
        st.caption(
            "data/ is gitignored: a missing file is created by running the notebook in the "
            "last column. The app picks up a new file on the next reload."
        )
        st.dataframe(table, hide_index=True, width="stretch")
