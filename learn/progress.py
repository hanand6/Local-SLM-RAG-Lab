# ============================================================
# learn/progress.py — keep Learn progress across restarts
#
# Saves completed chapters and interview-card drafts to
# data/learn_progress.json (gitignored). Lab results are not saved:
# they're cheap to re-run and tied to the current document.
#
# Drafts live in st.session_state["learn_drafts"] (a plain dict), not
# only in the text boxes: Streamlit discards a widget's value when it
# isn't rendered (e.g. you switch chapters), so the dict is the source
# of truth and the boxes are initialised from it.
# ============================================================

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

from config import LEARN_PROGRESS_FILE, PERSIST_INDEX

_FILE = Path(LEARN_PROGRESS_FILE)


def load_once() -> None:
    """Load saved progress into this browser session (first render only)."""
    if st.session_state.get("_learn_loaded"):
        return
    st.session_state._learn_loaded = True
    data = {}
    if PERSIST_INDEX and _FILE.exists():
        try:
            data = json.loads(_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}
    st.session_state.learn_done = set(data.get("done", []))
    st.session_state.learn_drafts = dict(data.get("drafts", {}))


def save() -> None:
    if not PERSIST_INDEX:
        return
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        _FILE.write_text(json.dumps({
            "done": sorted(st.session_state.get("learn_done", set())),
            "drafts": {k: v for k, v in st.session_state.get("learn_drafts", {}).items() if v},
        }, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass   # saving is a convenience; never break the page over it


def drafts() -> dict:
    return st.session_state.setdefault("learn_drafts", {})


def widget_key(key: str) -> str:
    """Widget key for a draft box. Changes after a reset, so the browser
    drops the old text instead of keeping it."""
    return f"{key}_g{st.session_state.get('learn_draft_gen', 0)}"


def on_draft_change(key: str) -> None:
    drafts()[key] = st.session_state.get(widget_key(key), "")
    save()


def reset() -> None:
    st.session_state.learn_done = set()
    st.session_state.learn_drafts = {}
    # Clear the widgets too, or their remembered values would win.
    for k in [k for k in st.session_state if str(k).startswith(("draft_", "done_"))]:
        del st.session_state[k]
    st.session_state.learn_reset_confirm = False
    st.session_state.learn_draft_gen = st.session_state.get("learn_draft_gen", 0) + 1
    save()
