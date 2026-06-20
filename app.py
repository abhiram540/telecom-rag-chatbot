"""Streamlit chat UI for the telecom RAG chatbot (FR-01 – FR-05).

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

from chain import stream_answer
from config import SAMPLE_QUESTIONS

st.set_page_config(page_title="NovaCell Support Assistant", page_icon="📶")


def _init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []  # conversation history (FR-03)
    if "pending" not in st.session_state:
        st.session_state.pending = None  # question queued from a sidebar click


def _render_sidebar() -> None:
    with st.sidebar:
        st.header("Sample questions")
        st.caption("Click to ask instantly.")
        for q in SAMPLE_QUESTIONS:
            if st.button(q, key=f"sample-{q}", use_container_width=True):
                st.session_state.pending = q  # FR-02

        st.divider()
        if st.button("🧹 Clear conversation", use_container_width=True):
            st.session_state.messages = []  # FR-04
            st.session_state.pending = None
            st.rerun()


def _handle_question(question: str) -> None:
    """Append the user turn, stream the grounded answer, and store both."""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(question))  # FR-05
        except Exception as exc:
            answer = (
                "Sorry — something went wrong reaching the assistant. "
                f"Please try again.\n\n`{exc}`"
            )
            st.error(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})


def main() -> None:
    _init_state()
    st.title("📶 NovaCell Support Assistant")
    st.caption(
        "Grounded in NovaCell's FAQ, resolved tickets, and official guides. "
        "For account-specific help, use the MyTelecom app or call 611."
    )

    _render_sidebar()

    # Replay history (FR-03).
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # A question from either the chat box or a sidebar click.
    question = st.chat_input("Ask a telecom support question...")
    if st.session_state.pending:
        question = st.session_state.pending
        st.session_state.pending = None

    if question:
        _handle_question(question)


if __name__ == "__main__":
    main()
