"""
app.py — Streamlit chat UI for the RAG system.

Run locally with:
    streamlit run app.py

When deployed on Streamlit Community Cloud, configuration comes from
st.secrets (set in the app's Settings > Secrets panel) instead of a local
.env file. Defaults below point at the portfolio bot, since that's the
version meant to be shown to recruiters.
"""

import os
import streamlit as st

# --- Bridge Streamlit secrets into environment variables ---
# Locally, .env (loaded by python-dotenv inside query.py/ingest.py) already
# populates these. On Streamlit Cloud there is no .env file, so anything set
# in st.secrets gets copied into the environment here instead. setdefault
# means a local .env always wins if both happen to be present.
try:
    for key, value in st.secrets.items():
        os.environ.setdefault(key, str(value))
except Exception:
    pass  # No secrets.toml present (e.g. running locally without one) — fine.

# Default to portfolio mode so a fresh deployment (no .env, no secrets set
# yet) still shows something sensible rather than an empty generic bot.
os.environ.setdefault("DATA_DIR", "portfolio_data")
os.environ.setdefault("PERSIST_DIR", "portfolio_chroma_db")
os.environ.setdefault("COLLECTION_NAME", "portfolio")
os.environ.setdefault("PORTFOLIO_MODE", "true")

import ingest
from query import answer_question, PERSIST_DIR, PORTFOLIO_MODE

st.set_page_config(page_title="Touqeer's Portfolio Q&A", page_icon="📚", layout="centered")

st.title("📚 Touqeer's Portfolio Q&A" if PORTFOLIO_MODE else "📚 RAG Q&A")
st.caption(
    "Ask about Touqeer's background, skills, and projects."
    if PORTFOLIO_MODE
    else "Ask questions grounded in the documents you've ingested."
)

# --- Auto-build the vector store on first run ---
# We don't commit the vector store binary to git — it's rebuilt from the
# source documents (which ARE committed) the first time the app starts.
if not os.path.isdir(PERSIST_DIR):
    with st.spinner("Setting up for the first time — indexing documents..."):
        docs = ingest.load_documents()
        if docs:
            chunks = ingest.chunk_documents(docs)
            ingest.build_vector_store(chunks)
        else:
            st.error(
                f"No documents found in `{ingest.DATA_DIR}/`. "
                "Add files there and redeploy."
            )
            st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input(
    "Ask a question, e.g. 'What ML projects has he worked on?'"
    if PORTFOLIO_MODE
    else "Ask a question about your documents..."
)

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
