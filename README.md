# RAG Document Assistant

A **Retrieval-Augmented Generation (RAG)** question-answering system over a
knowledge base. It embeds documents into a vector index, retrieves the most
relevant passages for a question via semantic similarity, and returns a
**grounded answer with its sources** — the core pattern behind modern AI
assistants and chatbots.

## Why this project
LLMs hallucinate when they answer from memory alone. RAG fixes this by
retrieving real source passages and grounding the answer in them. This project
implements the full retriever → context → generator pipeline and serves it
through an API.

## What it does
- **Chunks** a knowledge base into passages.
- **Embeds** passages into a vector index. The embedding backend is pluggable:
  - `sentence-transformers` (all-MiniLM-L6-v2) for semantic neural embeddings — *recommended, free, CPU-only*.
  - A **TF-IDF fallback** so the project runs anywhere with no heavy dependencies.
- **Retrieves** the top-k most similar passages for a query using cosine similarity.
- **Generates** a grounded answer:
  - With `OPENAI_API_KEY` set → sends the retrieved context to an LLM (gpt-4o-mini) for a natural-language answer (full RAG).
  - Without a key → returns an **extractive** answer from the top passage, so it works out of the box.
- Returns the **sources and similarity scores** behind every answer (explainability).
- Serves it all via a **FastAPI** `/ask` endpoint.

## Example (extractive mode, no API key)
```
Q: Why use PR-AUC instead of accuracy for fraud detection?
A: Class imbalance occurs when one class vastly outnumbers another, such as
   fraud making up under 2% of transactions. Techniques include class weighting,
   oversampling (SMOTE), undersampling, and evaluating with PR-AUC instead of accuracy.
   (retrieved by semantic similarity — the question never used the word "imbalance")
```

## Tech stack
Python · sentence-transformers / scikit-learn (TF-IDF) · NumPy · FastAPI · (optional) OpenAI

## Project structure
```
rag-document-assistant/
├── data/
│   └── knowledge_base.md   # sample ML knowledge base
├── src/
│   ├── rag.py              # RAG engine: chunk, embed, retrieve, answer
│   ├── demo.py             # runnable CLI demo
│   └── api.py              # FastAPI /ask endpoint
├── requirements.txt
└── README.md
```

## How to run
```bash
pip install -r requirements.txt

# 1) CLI demo (answers sample questions with sources)
cd src
python demo.py

# 2) Serve the Q&A API
uvicorn api:app --reload
# curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
#   -d '{"question":"How does RAG reduce hallucination?"}'

# 3) (Optional) Enable LLM-generated answers:
export OPENAI_API_KEY=sk-...        # then run as above
```
Point it at your own documents by replacing `data/knowledge_base.md`.

## Skills demonstrated
- Building a RAG pipeline: chunking, embeddings, vector retrieval, grounding
- Semantic search with cosine similarity over a vector index
- Prompt construction for grounded, source-cited LLM answers
- Clean, pluggable, production-style Python and a REST API
- Designing for graceful fallback (runs with or without GPUs/API keys)

## Author
Naveen Kumar Kovi — M.S. Computer Science, NJIT
