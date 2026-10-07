"""Shared page furniture: sidebar, header and the "Next →" link (Streamlit-v3.md §2.2, §2.3).

`sidebar()` runs once per rerun from `streamlit_app.py`, before the page, so its widgets sit in
the sidebar of every page and keep their state across pages (widget keys = session_state keys).
"""

import streamlit as st

from data_loading import load_smard

REPO_URL = "https://github.com/fruchtgummikauer/grid-stress"

# The tour (§1.1): path, title, icon, url. The order is the sidebar and "Next →" order;
# streamlit_app.py builds the navigation from this table. app_pages/risk_days.py is not on it
# (removed from the app, team decision 2026-10-06).
TOUR = [
    ("app_pages/home.py", "Home", "⚡", "home"),
    ("app_pages/background.py", "Background", "📊", "background"),
    ("app_pages/method.py", "Method", "🧭", "method"),
    ("app_pages/beat_smard.py", "Where we beat SMARD", "🏁", "beat-smard"),
    ("app_pages/about.py", "Team / About", "👥", "about"),
]


def navigation_pages():
    """The st.Page objects in TOUR order, without section labels; the first page is the default."""
    return [
        st.Page(path, title=title, icon=icon, url_path=url, default=position == 0)
        for position, (path, title, icon, url) in enumerate(TOUR)
    ]


def remember_page(page):
    """Store the running page's url (called by streamlit_app.py before `page.run()`)."""
    st.session_state["_current_url"] = page.url_path


def current_page():
    """TOUR path of the running page (the home page if unknown)."""
    url = st.session_state.get("_current_url", TOUR[0][3])
    return next((path for path, _, _, page_url in TOUR if page_url == url), TOUR[0][0])


def sidebar():
    """Demo-mode toggle, data date, repo link."""
    st.session_state.setdefault("demo_mode", False)

    with st.sidebar:
        st.toggle(
            "Demo mode",
            key="demo_mode",
            help="For the projector: hides chart toolbars (more to come per page).",
        )

        try:
            record_end = load_smard().index.max()
            st.caption(f"Data as of {record_end:%Y-%m-%d %H:%M}")
        except (FileNotFoundError, RuntimeError):
            st.caption("Data: `data/smard.csv` not found")
        st.caption(f"[Code on GitHub]({REPO_URL})")


def header(title: str, takeaway: str):
    """Page title plus its one-line takeaway (used by the v3 page rewrites)."""
    st.title(title)
    st.markdown(f"### {takeaway}")


def next_page(current: str):
    """A "Next →" link to the page after `current` (a path from TOUR) on the tour."""
    paths = [entry[0] for entry in TOUR]
    position = paths.index(current)
    if position + 1 < len(TOUR):
        path, title = TOUR[position + 1][:2]
        st.divider()
        st.page_link(path, label=f"Next: {title} →")


def demo_mode() -> bool:
    """True while the sidebar's demo-mode toggle is on."""
    return bool(st.session_state.get("demo_mode", False))
