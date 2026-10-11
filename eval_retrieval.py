"""Measure retrieval quality for different chunk sizes, using an in-memory index.

For each test question we know a phrase the right chunk must contain. We report
the rank of the first chunk containing it (lower is better) and hit@k.
"""

from pathlib import Path

import chromadb

import config
from ingest import chunk_text, embed, load_pdf

TEST_SET = [
    ("how many lakes were used to get chl-a maps from?", "15 lakes"),
    ("Which lakes were studied?", "15 lakes"),
    ("What evaluation metrics were used?", "Root Mean Square Error"),
    ("Which atmospheric correction algorithm was applied?", "C2RCC"),
    ("How many images were in the training data set?", "1,080 images"),
]
CONFIGS = [(250, 50), (150, 30), (100, 25)]  # (chunk size, overlap) in words


def build_index(size, overlap):
    chunks = []
    for pdf in Path(config.PDF_DIR).glob("*.pdf"):
        for _, text in load_pdf(pdf):
            chunks.extend(chunk_text(text, size, overlap))
    collection = chromadb.EphemeralClient().get_or_create_collection(
        f"eval-{size}-{overlap}", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=[str(i) for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks, "search_document"),
    )
    return collection


def main():
    for size, overlap in CONFIGS:
        collection = build_index(size, overlap)
        print(f"\nchunk={size} overlap={overlap} ({collection.count()} chunks)")
        ranks = []
        for question, expected in TEST_SET:
            docs = collection.query(
                query_embeddings=embed([question], "search_query"), n_results=collection.count()
            )["documents"][0]
            rank = next(i + 1 for i, d in enumerate(docs) if expected in d)
            ranks.append(rank)
            print(f"  rank {rank:>2} | {question}")
        for k in (4, 6):
            print(f"  hit@{k}: {sum(r <= k for r in ranks)}/{len(ranks)}")


if __name__ == "__main__":
    main()
