"""Shared settings. Swap models here without touching the pipeline code."""

PDF_DIR = "docs"            # put any PDFs you want to index in this folder
DB_DIR = "chroma_db"        # where the vector database is persisted
COLLECTION = "papers"

EMBED_MODEL = "nomic-embed-text"   # text -> 768-dim vector
LLM_MODEL = "llama3.2:3b"          # generates the final answer

CHUNK_SIZE = 150     # words per chunk
CHUNK_OVERLAP = 30   # words shared between consecutive chunks
TOP_K = 4            # how many chunks to retrieve per question
