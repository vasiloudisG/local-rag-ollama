"""Phase B - Query: embed question -> retrieve top-k chunks -> augment prompt -> generate."""

import sys

import chromadb
import ollama

import config
from ingest import embed

SYSTEM_PROMPT = """You answer questions using ONLY the provided context.
If the answer is not in the context, say "I don't know based on the documents."
Cite the sources you used as [source, page X]."""


def retrieve(question, k=config.TOP_K):
    collection = chromadb.PersistentClient(path=config.DB_DIR).get_collection(config.COLLECTION)
    results = collection.query(query_embeddings=embed([question], "search_query"), n_results=k)
    # Chroma returns one list per query; we sent a single question.
    return list(zip(results["documents"][0], results["metadatas"][0], results["distances"][0]))


def build_prompt(question, hits):
    context = "\n\n".join(f"[{meta['source']}, page {meta['page']}]\n{doc}" for doc, meta, _ in hits)
    return f"Context:\n{context}\n\nQuestion: {question}"


def answer(question, show_sources=True):
    hits = retrieve(question)

    if show_sources:
        print("\n--- Retrieved chunks ---")
        for doc, meta, dist in hits:
            print(f"page {meta['page']:>2} | similarity {1 - dist:.2f} | {doc[:90]}...")
        print("------------------------\n")

    stream = ollama.chat(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(question, hits)},
        ],
        options={"temperature": 0},  # deterministic, factual answers
        stream=True,
    )
    for part in stream:
        print(part["message"]["content"], end="", flush=True)
    print()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        answer(" ".join(sys.argv[1:]))
    else:
        print("Ask a question about your documents (empty line to quit).")
        while question := input("\n> ").strip():
            answer(question)
