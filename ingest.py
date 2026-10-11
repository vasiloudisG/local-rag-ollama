"""Phase A - Indexing: load PDFs -> chunk -> embed -> store in a vector DB."""

from pathlib import Path

import chromadb
import ollama
import pymupdf

import config


def load_pdf(path):
    """Return a list of (page_number, text), one entry per page, without the reference list."""
    pages = []
    with pymupdf.open(path) as doc:
        for i, page in enumerate(doc):
            text = page.get_text()
            # Citations are noise for retrieval, and long DOI/URL runs overflow the embedder.
            lines = text.splitlines()
            if "References" in (line.strip() for line in lines):
                cut = [line.strip() for line in lines].index("References")
                pages.append((i + 1, "\n".join(lines[:cut])))
                break
            pages.append((i + 1, text))
    return pages


def normalize(text):
    """Unify symbol variants so 'chl-a' in a question matches 'Chl‐α' in the paper."""
    for variant in ("‐", "‑", "–", "−"):
        text = text.replace(variant, "-")
    return text.replace("α", "a")


def chunk_text(text, size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP):
    """Split text into overlapping windows of `size` words."""
    words = text.split()
    step = size - overlap
    return [" ".join(words[i:i + size]) for i in range(0, max(len(words) - overlap, 1), step)]


def embed(texts, prefix, batch_size=8):
    # nomic-embed-text expects a task prefix: "search_document" for stored text,
    # "search_query" for questions. Same model must be used for both.
    # Small batches keep the local embedding server from running out of memory.
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = [f"{prefix}: {normalize(t)}" for t in texts[i:i + batch_size]]
        vectors.extend(ollama.embed(model=config.EMBED_MODEL, input=batch)["embeddings"])
    return vectors


def main():
    client = chromadb.PersistentClient(path=config.DB_DIR)
    # Rebuild from scratch so re-running never duplicates chunks.
    if config.COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(config.COLLECTION)
    collection = client.create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})

    for pdf in sorted(Path(config.PDF_DIR).glob("*.pdf")):
        ids, chunks, metadatas = [], [], []
        for page, text in load_pdf(pdf):
            for n, chunk in enumerate(chunk_text(text)):
                ids.append(f"{pdf.stem}-p{page}-c{n}")
                chunks.append(chunk)
                metadatas.append({"source": pdf.name, "page": page})

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embed(chunks, "search_document"),
            metadatas=metadatas,
        )
        print(f"Indexed {pdf.name}: {len(chunks)} chunks")

    print(f"Done. Collection '{config.COLLECTION}' has {collection.count()} chunks.")


if __name__ == "__main__":
    main()
