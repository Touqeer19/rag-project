"""
query.py — Retrieve relevant chunks for a question and generate a grounded
answer using either Claude (Anthropic) or Gemini (Google), chosen via
LLM_PROVIDER in .env.

CLI usage:
    python query.py "What does the document say about X?"
"""

import logging
import os
import sys

# chromadb 0.5.20's bundled telemetry has a known bug with newer posthog
# versions (the call still fires and then errors internally, regardless of
# ANONYMIZED_TELEMETRY). Silencing this specific logger suppresses the noise.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

PERSIST_DIR = os.environ.get("PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")
TOP_K = 8

# Which LLM to use for the generation step: "anthropic" or "gemini"
PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-sonnet-4-6")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

PORTFOLIO_MODE = os.environ.get("PORTFOLIO_MODE", "false").strip().lower() == "true"

GENERIC_SYSTEM_PROMPT = """You are a precise, grounded Q&A assistant.
Answer the user's question using ONLY the context provided below.
If the context doesn't contain enough information to answer, say so clearly —
do not make anything up.
Cite the source filename for each claim, like [source: filename.pdf]."""

PORTFOLIO_SYSTEM_PROMPT = """You are answering questions from a recruiter or
hiring manager about Touqeer Ahmad's background, skills, and projects, based
ONLY on the context provided below.

Speak about Touqeer in the third person, in a professional but warm tone —
like a knowledgeable colleague summarizing his work, not like Touqeer himself
speaking. Be specific: cite concrete details (technologies, metrics, project
names) from the context rather than vague summaries. If a question asks
something the context doesn't cover, say so honestly and suggest the
recruiter ask Touqeer directly — do not invent qualifications.
Cite the source document for each claim, like [source: filename.md]."""

SYSTEM_PROMPT = PORTFOLIO_SYSTEM_PROMPT if PORTFOLIO_MODE else GENERIC_SYSTEM_PROMPT


def get_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=PERSIST_DIR,
    )


def retrieve(question: str, vector_store, k: int = TOP_K):
    results = vector_store.similarity_search(question, k=k)
    return results


def build_context(chunks):
    parts = []
    for c in chunks:
        source = c.metadata.get("source", "unknown")
        parts.append(f"[source: {source}]\n{c.page_content}")
    return "\n\n---\n\n".join(parts)


def _generate_answer_anthropic(question: str, context: str):
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Copy .env.example to .env and add your key."
        )

    client = anthropic.Anthropic(api_key=api_key)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text


def _generate_answer_gemini(question: str, context: str):
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Copy .env.example to .env and add your key."
        )

    client = genai.Client(api_key=api_key)
    user_message = f"Context:\n\n{context}\n\nQuestion: {question}"

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=user_message,
        config={"system_instruction": SYSTEM_PROMPT},
    )
    return response.text


def generate_answer(question: str, context: str):
    if PROVIDER == "gemini":
        return _generate_answer_gemini(question, context)
    elif PROVIDER == "anthropic":
        return _generate_answer_anthropic(question, context)
    else:
        raise RuntimeError(
            f"Unknown LLM_PROVIDER '{PROVIDER}' in .env — use 'anthropic' or 'gemini'."
        )


def answer_question(question: str):
    vector_store = get_vector_store()
    chunks = retrieve(question, vector_store)

    if not chunks:
        return "No documents found. Run `python ingest.py` first.", []

    context = build_context(chunks)
    answer = generate_answer(question, context)
    sources = sorted({c.metadata.get("source", "unknown") for c in chunks})
    return answer, sources


def main():
    if len(sys.argv) < 2:
        print('Usage: python query.py "your question here"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    print(f"\nQuestion: {question}\n")

    answer, sources = answer_question(question)

    print("Answer:")
    print(answer)
    print(f"\nSources used: {', '.join(sources)}")


if __name__ == "__main__":
    main()