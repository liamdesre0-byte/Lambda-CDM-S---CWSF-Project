"""Streamlit app launcher using ccomplet2ee interface."""

from __future__ import annotations

import ccomplet2ee as c2


def launch() -> None:
    """Run the Streamlit dashboard frontend."""
    c2.run_streamlit_app()
