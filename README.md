# EvidenceRAG

A local-first, open-source retrieval-augmented generation (RAG)
application for answering questions over public, appropriately
licensed technical or scientific documents.

## Goals

- Run the core application locally without a paid inference API.
- Retrieve relevant evidence using keyword and semantic search.
- Generate answers grounded in retrieved passages.
- Provide traceable citations and abstain when evidence is insufficient.
- Evaluate retrieval and answer quality using a reproducible benchmark.
- Expose the pipeline through an API and a lightweight user interface.

## Status

Under development.

## Planned technology

- Python
- Hugging Face sentence-transformers
- FAISS and BM25
- LangChain
- Ollama
- FastAPI and Pydantic
- Streamlit
- pytest and Ragas

## Data and licensing

The corpus will use public documents whose licenses permit the
intended use. Document provenance and licensing will be recorded.
Downloaded corpora, indexes, model weights, and secrets will not
be committed by default.

## Development

Environment setup and installation instructions will be documented
as the project evolves.
