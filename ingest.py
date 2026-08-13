"""
ingest.py — Load documents from data/, split into chunks, embed them,
and persist to a local Chroma vector store.

Run this once after adding/changing files in data/:
    python ingest.py
"""

import logging
import os
from pathlib import Path

# chromadb 0.5.20's bundled telemetry has a known bug with newer posthog
# versions (the call still fires and then errors internally, regardless of
# ANONYMIZED_TELEMETRY). Silencing this specific logger suppresses the noise.
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

# Configurable so you can run separate "collections" (e.g. general test data
# vs. portfolio data) without them mixing — set DATA_DIR / PERSIST_DIR in
# .env, or leave unset to use the defaults below.
DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
PERSIST_DIR = os.environ.get("PERSIST_DIR", "chroma_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "documents")

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

LOADERS = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_documents():
    """Load every supported file in DATA_DIR."""
    docs = []
    files = [f for f in DATA_DIR.iterdir() if f.is_file()]

    if not files:
        print(f"No files found in {DATA_DIR}/. Add PDF, TXT, or MD files and re-run.")
        return docs

    for file_path in files:
        suffix = file_path.suffix.lower()
        loader_cls = LOADERS.get(suffix)
        if loader_cls is None:
            print(f"Skipping unsupported file type: {file_path.name}")
            continue

        print(f"Loading {file_path.name} ...")
        loader = loader_cls(str(file_path))
        loaded = loader.load()

        # Tag each chunk's source metadata with the original filename,
        # so we can cite it later when answering questions.
        for d in loaded:
            d.metadata["source"] = file_path.name

        docs.extend(loaded)

    return docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    print(f"Split {len(docs)} document(s) into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks):
    print("Loading local embedding model (all-MiniLM-L6-v2) ...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding chunks and writing to Chroma ...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=PERSIST_DIR,
    )
    print(f"Done. Vector store persisted to ./{PERSIST_DIR}/")
    return vector_store


def main():
    docs = load_documents()
    if not docs:
        return
    chunks = chunk_documents(docs)
    build_vector_store(chunks)


if __name__ == "__main__":
    main()