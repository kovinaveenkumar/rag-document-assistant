"""A lightweight Retrieval-Augmented Generation (RAG) engine.

Pipeline:
  1. Chunk a knowledge base into passages.
  2. Embed passages into vectors.
  3. On a query, embed it and retrieve the top-k most similar passages via
     cosine similarity (a vector index).
  4. Compose a grounded answer from the retrieved context.

Embeddings are pluggable:
  * If `sentence-transformers` is installed -> semantic neural embeddings
    (model: all-MiniLM-L6-v2). Recommended; free and runs on CPU.
  * Otherwise -> a TF-IDF vectorizer fallback, so the project runs anywhere
    with zero heavy dependencies.

Generation is pluggable too:
  * If OPENAI_API_KEY is set -> the retrieved context is sent to an LLM for a
    natural-language answer (the full RAG pattern).
  * Otherwise -> an extractive answer from the top passage (no API key needed).
"""
import os
import re
from typing import List, Tuple

import numpy as np


def chunk_text(text: str) -> List[str]:
    blocks = re.split(r"\n##\s+", text)
    return [b.strip().replace("\n", " ") for b in blocks if len(b.strip()) > 40]


class _STEmbedder:
    """Neural sentence embeddings (preferred)."""
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.name = "sentence-transformers/all-MiniLM-L6-v2"

    def fit_transform(self, docs):
        return self.model.encode(docs, normalize_embeddings=True)

    def transform(self, queries):
        return self.model.encode(queries, normalize_embeddings=True)


class _TfidfEmbedder:
    """TF-IDF fallback (no heavy deps)."""
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english")
        self.name = "tfidf"

    def _norm(self, m):
        m = np.asarray(m, dtype=float)
        n = np.linalg.norm(m, axis=1, keepdims=True)
        n[n == 0] = 1.0
        return m / n

    def fit_transform(self, docs):
        return self._norm(self.vec.fit_transform(docs).toarray())

    def transform(self, queries):
        return self._norm(self.vec.transform(queries).toarray())


def _make_embedder():
    try:
        return _STEmbedder()
    except Exception:
        return _TfidfEmbedder()


class RAGEngine:
    def __init__(self, kb_path: str):
        self.embedder = _make_embedder()
        with open(kb_path) as f:
            self.chunks = chunk_text(f.read())
        self.embeddings = self.embedder.fit_transform(self.chunks)
        self.backend = self.embedder.name

    def retrieve(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        q = self.embedder.transform([query])[0]
        scores = self.embeddings @ q
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def answer(self, query: str, k: int = 3) -> dict:
        retrieved = self.retrieve(query, k)
        context = "\n\n".join(c for c, _ in retrieved)
        if os.getenv("OPENAI_API_KEY"):
            answer_text, mode = self._llm_answer(query, context), "llm-generated"
        else:
            answer_text, mode = retrieved[0][0], "extractive (no API key)"
        return {
            "query": query,
            "answer": answer_text,
            "mode": mode,
            "embedding_backend": self.backend,
            "sources": [{"text": c[:160] + "...", "score": round(s, 3)} for c, s in retrieved],
        }

    def _llm_answer(self, query: str, context: str) -> str:
        from openai import OpenAI
        client = OpenAI()
        prompt = ("Answer the question using ONLY the context below. If the answer "
                  f"isn't in the context, say so.\n\nContext:\n{context}\n\n"
                  f"Question: {query}\nAnswer:")
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}], temperature=0.2)
        return resp.choices[0].message.content.strip()
