"""RAG Q&A API (FastAPI).

Exposes /ask which retrieves relevant context from the knowledge base and
returns a grounded answer with its sources.

Run:  uvicorn api:app --reload   (from the src/ directory)
"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from rag import RAGEngine

KB = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.md")

app = FastAPI(title="RAG Document Assistant", version="1.0")
_engine = None


def engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine(KB)
    return _engine


class Query(BaseModel):
    question: str
    k: int = 3


@app.get("/")
def health():
    return {"status": "ok", "service": "rag-document-assistant"}


@app.post("/ask")
def ask(q: Query):
    return engine().answer(q.question, q.k)
