"""MCP server: exposes the RAG retrieval step as tools any MCP client (e.g. Claude Desktop) can call.

The client's own LLM decides when to search and writes the final answer;
this server only does retrieval, locally, over the existing Chroma index.
"""

from pathlib import Path

from mcp.server.mcpserver import MCPServer

import config
from ask import retrieve

server = MCPServer(
    "research-papers",
    instructions="Search the user's indexed research papers. Cite the source and page of every fact you use.",
)


@server.tool()
def search_papers(query: str, k: int = config.TOP_K) -> str:
    """Semantic search over the user's indexed research papers (satellite imagery, chlorophyll
    forecasting, GANs). Returns the k most relevant text chunks with source file and page number.
    Call it again with a rephrased query if the results don't answer the question."""
    hits = retrieve(query, k)
    return "\n\n".join(
        f"[{meta['source']}, page {meta['page']}, similarity {1 - dist:.2f}]\n{doc}"
        for doc, meta, dist in hits
    )


@server.tool()
def list_papers() -> list[str]:
    """List the PDF files that are indexed and searchable."""
    return sorted(p.name for p in Path(config.PDF_DIR).glob("*.pdf"))


if __name__ == "__main__":
    server.run()  # stdio transport: the client launches this script and talks over stdin/stdout
