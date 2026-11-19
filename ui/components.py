"""Reusable pieces of the Streamlit UI."""

import textwrap

import streamlit as st

from ui.api_client import ChatResult, Profile, Source
from ui.config import DEMO_ACCOUNTS, DEPARTMENT_COLORS


def department_badge(department: str) -> None:
    st.badge(department.upper(), color=DEPARTMENT_COLORS.get(department, "gray"))  # type: ignore[arg-type]


def render_login_header() -> None:
    st.markdown("## 💬 FinSolve Assistant")
    st.caption(
        "Ask questions about company documents. You only see answers from the data "
        "your role is allowed to access."
    )


def render_demo_accounts() -> None:
    with st.expander("Demo accounts"):
        st.markdown(
            "\n".join(f"- `{username}` · {role}" for username, role in DEMO_ACCOUNTS)
            + "\n\nAll demo accounts use the password set in `SEED_USER_PASSWORD` in `.env`."
        )


def render_sidebar(profile: Profile) -> tuple[bool, bool]:
    """Draw the sidebar. Returns (new_chat_clicked, logout_clicked)."""
    with st.sidebar:
        st.markdown(f"### {profile.full_name}")
        st.caption(f"@{profile.username}")
        st.badge(profile.role_display_name, icon="🔐", color="primary")
        st.markdown(f"_{profile.role_description}_")

        st.markdown("**You can ask about**")
        for name, description in profile.departments:
            department_badge(name)
            st.caption(description)

        st.divider()
        new_chat = st.button("➕ New chat", use_container_width=True)
        st.toggle(
            "Show all retrieved sources",
            key="show_all_sources",
            help="Also list sources that were retrieved but not cited in the answer.",
        )
        logout = st.button("Log out", use_container_width=True)
    return new_chat, logout


def _render_source(source: Source) -> None:
    location = f"{source.title} › {source.section}" if source.section else source.title
    st.markdown(f"**[{source.number}]** {location}")
    cols = st.columns([1, 3])
    with cols[0]:
        department_badge(source.department)
    with cols[1]:
        details = f"`{source.document}`"
        if source.quarter:
            details += f" · {source.quarter}"
        details += f" · relevance {source.score:.2f}"
        st.caption(details)
    st.markdown(
        "> " + textwrap.shorten(source.excerpt, 450, placeholder=" …").replace("\n", "\n> ")
    )


def render_sources(result: ChatResult, show_all: bool) -> None:
    cited = [s for s in result.sources if s.cited]
    others = [s for s in result.sources if not s.cited]
    shown = cited + (others if show_all else [])
    if not shown:
        return
    label = f"📎 Sources ({len(cited)} cited"
    label += f", {len(others)} more retrieved)" if show_all and others else ")"
    with st.expander(label):
        for source in shown:
            _render_source(source)


def render_answer_footer(result: ChatResult, question: str = "") -> None:
    parts = [f"⏱ {result.latency_ms / 1000:.1f}s"]
    if result.search_query and result.search_query.strip() != question.strip():
        # A follow-up was rewritten into a standalone search: show it for transparency.
        parts.insert(0, f"🔎 searched for: “{result.search_query}”")
    if result.route == "sql+docs":
        parts.insert(0, "🧮 computed with SQL")
    if result.model:
        parts.append(result.model)
    st.caption(" · ".join(parts))
