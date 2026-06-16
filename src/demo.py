"""Runnable demo: builds the index and answers sample questions with sources."""
import os

from rag import RAGEngine

KB = os.path.join(os.path.dirname(__file__), "..", "data", "knowledge_base.md")

QUESTIONS = [
    "What is overfitting and how do I prevent it?",
    "Why use PR-AUC instead of accuracy for fraud detection?",
    "How does retrieval-augmented generation reduce hallucination?",
]


def main():
    print("Building vector index from knowledge base...")
    rag = RAGEngine(KB)
    print(f"Indexed {len(rag.chunks)} passages.\n")
    for q in QUESTIONS:
        res = rag.answer(q, k=2)
        print(f"Q: {q}")
        print(f"   [mode: {res['mode']}]")
        print(f"A: {res['answer'][:280]}")
        print(f"   top source score: {res['sources'][0]['score']}\n")


if __name__ == "__main__":
    main()
