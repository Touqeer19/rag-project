# Sample Document: RAG Project Notes

This is a placeholder document so the pipeline works immediately after setup.
Replace this with your own PDFs, text files, or markdown notes.

## What is RAG?

Retrieval-Augmented Generation combines a retrieval step (finding relevant
information from a knowledge base) with a generation step (an LLM writing an
answer using that retrieved information). This grounds the model's answers in
real source material instead of relying purely on what it memorized during
training, and lets you cite where each answer came from.

## Try it

After running `python ingest.py`, ask:

    python query.py "What is RAG?"

You should get an answer that cites sample.md as the source.
