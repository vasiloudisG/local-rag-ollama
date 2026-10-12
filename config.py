"""Shared settings. Swap models here without touching the pipeline code."""

from pathlib import Path

# Absolute paths, so scripts work no matter which folder they are launched from
# (an MCP client such as Claude Desktop starts the server from its own directory).
ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / "docs"          # put any PDFs you want to index in this folder
DB_DIR = str(ROOT / "chroma_db") # where the vector database is persisted
COLLECTION = "papers"

EMBED_MODEL = "nomic-embed-text"   # text -> 768-dim vector
LLM_MODEL = "llama3.2:3b"          # generates the final answer

CHUNK_SIZE = 150     # words per chunk
CHUNK_OVERLAP = 30   # words shared between consecutive chunks
TOP_K = 4            # how many chunks to retrieve per question
