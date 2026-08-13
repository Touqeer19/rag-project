"""
app.py — Streamlit chat UI for the RAG system.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
from query import answer_question, PERSIST_DIR

st.set_page_config(page_title="RAG Q&A", page_icon="📚", layout="centered")

st.title("📚 RAG Q&A")
st.caption("Ask questions grounded in the documents you've ingested.")

if not os.path.isdir(PERSIST_DIR):
    st.warning(
        "No vector store found yet. Add documents to `data/` and run "
        "`python ingest.py` first, then refresh this page."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving and generating..."):
            try:
                answer, sources = answer_question(question)
                reply = answer
                if sources:
                    reply += f"\n\n*Sources: {', '.join(sources)}*"
            except Exception as e:
                reply = f"Error: {e}"
        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
