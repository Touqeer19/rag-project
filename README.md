# RAG Q&A System

A general-purpose Retrieval-Augmented Generation (RAG) pipeline: drop in documents
(PDF, TXT, MD), ask questions in natural language, get answers grounded in your
own data with source citations.

## Architecture

```
 Documents (PDF/TXT/MD)
        │
        ▼
   ingest.py
   ├─ Load documents
   ├─ Split into chunks (RecursiveCharacterTextSplitter)
   ├─ Embed chunks (sentence-transformers, runs locally — no API cost)
   └─ Store vectors in Chroma (local, persistent, on-disk vector DB)
        │
        ▼
   query.py / app.py
   ├─ Embed the user's question
   ├─ Retrieve top-k most similar chunks from Chroma
   ├─ Build a prompt: question + retrieved context
   └─ Send to Claude (Anthropic API) for a grounded answer with citations
```

**Why this stack:**
- **LangChain** — handles document loading, chunking, and the retrieval chain glue.
- **sentence-transformers (`all-MiniLM-L6-v2`)** — free, local embeddings. No API
  key needed for this step, and it keeps ingestion cost at zero.
- **Chroma** — lightweight local vector database, persists to disk, no server to run.
- **Anthropic Claude or Google Gemini** — generates the final answer from
  retrieved context. Choose which one via `LLM_PROVIDER` in `.env` (`anthropic`
  or `gemini`). Gemini has a free tier, which makes it a good choice if you'd
  rather not add a payment method to test this out.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# edit .env: set LLM_PROVIDER to "anthropic" or "gemini", and add the matching key
```

- Anthropic key: https://console.anthropic.com/ — pay-per-token, no free tier.
- Gemini key: https://aistudio.google.com/apikey — free tier available.

## Usage

1. **Add your documents** to the `data/` folder (PDF, TXT, or MD files).

2. **Ingest** — chunk, embed, and store them:
   ```bash
   python ingest.py
   ```
   This builds a local vector store in `chroma_db/`. Re-run this any time you
   add or change documents.

3. **Query** from the command line:
   ```bash
   python query.py "What does this document say about refund policy?"
   ```

4. **Or launch the web UI:**
   ```bash
   streamlit run app.py
   ```

## "Chat with my portfolio" mode

The project includes a ready-to-use portfolio setup in `portfolio_data/` —
markdown write-ups of real projects and background, suitable for a
recruiter-facing Q&A bot. This runs as a **separate vector store and
collection** from your general test data, so the two never mix.

To activate it:

```bash
cp .env.portfolio.example .env
# edit .env: add your real ANTHROPIC_API_KEY or GEMINI_API_KEY
python ingest.py
python query.py "What data science projects has Touqeer worked on?"
```

Or launch the web UI (`streamlit run app.py`) for a chat interface a
recruiter can actually use — deploy it (e.g. Streamlit Community Cloud) and
share the link.

To switch back to your general test data, just restore your previous `.env`
(or delete `DATA_DIR` / `PERSIST_DIR` / `PORTFOLIO_MODE` from it).

### Adding more content

Drop more markdown or PDF files into `portfolio_data/` (resume, more project
write-ups, certifications) and re-run `python ingest.py` to include them.

- Swap Chroma for a hosted vector DB (Pinecone, Weaviate, pgvector) to show you
  can work with production-scale infra.
- Add hybrid search (keyword + vector) for better recall on exact terms.
- Add re-ranking (e.g. Cohere rerank or a cross-encoder) before generation.
- Wrap `query.py` in a FastAPI endpoint and deploy it — pairs well with your
  AWS experience.
- Point it at a real dataset: e.g. index your own resume + project docs and
  turn it into "ask my portfolio a question" for recruiters.

## Files

| File              | Purpose                                      |
|-------------------|-----------------------------------------------|
| `ingest.py`       | Load → chunk → embed → store documents        |
| `query.py`        | CLI: retrieve + generate an answer             |
| `app.py`          | Streamlit chat UI                              |
| `requirements.txt`| Dependencies                                   |
| `.env.example`    | Template for API key config                    |
| `data/`           | Put your source documents here                 |
