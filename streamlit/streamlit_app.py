"""Entry point: page config and navigation (Streamlit-v3.md §1.1, §2.1).

Run from the repo root, so `.streamlit/config.toml` (the theme) is found:

    uv run streamlit run streamlit/streamlit_app.py

The pages live in `app_pages/`; their order, titles and sections are in
`components/layout.py` (`TOUR`): Home, Background, Method, Where we beat SMARD, Team / About.
"""

import streamlit as st

from components.layout import navigation_pages, remember_page, sidebar

st.set_page_config(page_title="Grid Stress", page_icon="⚡", layout="wide")

navigation = st.navigation(navigation_pages())  # pages and order: layout.TOUR
sidebar()  # demo mode, data date: on every page
remember_page(navigation)
navigation.run()
