# Local RAG over research papers

A minimal Retrieval-Augmented Generation (RAG) pipeline that answers questions about PDF documents, with citations. It runs fully locally and for free, using open-source models through [Ollama](https://ollama.com).

I tested it on my own peer-reviewed paper, *Forecasting Lakes' Chlorophyll Concentrations Using Satellite Images and Generative Adversarial Networks* (Water Resources Research, [doi:10.1029/2024WR037138](https://doi.org/10.1029/2024WR037138)). The PDF is not included in the repo: download it from the link, or use any PDFs you like.

```
> how many lakes were used to get chl-a maps from?

--- Retrieved chunks ---
page  1 | similarity 0.77 | operation of the model lays on the fact that typically Chl‐α increases ...
page  2 | similarity 0.76 | as follows. In Section 2, we discuss the proposed methodology ...
...
According to [Water Resources Research.pdf, page 3], a data set of 15 lakes distributed in Europe was created.
```

## How it works

**Indexing** ([ingest.py](ingest.py)): load PDF → strip the reference list → split into overlapping chunks → embed → store in a vector database.

**Querying** ([ask.py](ask.py)): embed the question → retrieve the top-k most similar chunks → add them to the prompt → the LLM answers using only that context and cites pages.

| Component | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF | `pypdf` dropped the spaces between words on this paper |
| Embeddings | `nomic-embed-text` (768-dim) | Small (274 MB), local, strong for its size |
| Vector DB | ChromaDB (cosine, HNSW) | Embedded, no server, persisted to disk |
| LLM | `llama3.2:3b` | Fits a 16 GB laptop; swappable in [config.py](config.py) |

The system prompt grounds the model: answer only from the context, otherwise say *"I don't know based on the documents."*, and always cite sources.

## Evaluating retrieval

[eval_retrieval.py](eval_retrieval.py) runs a small test set of questions, each paired with a phrase the correct chunk must contain, and reports that chunk's rank and hit@k for several chunk sizes:

| Chunk size / overlap (words) | Rank for *"how many lakes…?"* | hit@4 |
|---|---|---|
| 250 / 50 | 7 | 3/5 |
| **150 / 30** (chosen) | **1** | **4/5** |
| 100 / 25 | 1 | 4/5, other questions drop |

Large chunks blur several topics into one embedding, so a single key sentence gets lost. Very small chunks lose surrounding context. Normalizing symbol variants (`Chl‐α` → `Chl-a`) also helps match how users type.

## Run it

Requires Python 3.10+ and [Ollama](https://ollama.com).

```bash
ollama pull nomic-embed-text
ollama pull llama3.2:3b

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Put PDFs in docs/, then:
python ingest.py
python ask.py                      # interactive, empty line quits
python ask.py "Which atmospheric correction algorithm was used?"
python eval_retrieval.py
```

## Next steps

- Hybrid search (BM25 + vectors) and a reranker for higher precision
- Sentence-aware chunking instead of fixed word windows
- Answer-level evaluation (faithfulness, relevance) with RAGAS
- A multilingual embedding model (e.g. `bge-m3`) for Greek documents
- Expose retrieval as an MCP server so any MCP client can query the papers
