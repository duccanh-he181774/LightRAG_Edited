"""
Demo: LightRAG with Claude CLI as LLM backend.

No Anthropic API key required — uses your active Claude Code session.
Requires an embedding model. Two options (pick one):
  Option A: Ollama  -> run `ollama pull nomic-embed-text` first
  Option B: OpenAI  -> set OPENAI_API_KEY env var
"""
import asyncio
import os
import sys

# ── Add LightRAG package to path ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "LightRAG"))

from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from llm_claude_cli import claude_cli_complete

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION — edit this block
# ══════════════════════════════════════════════════════════════════════════════
WORKING_DIR = "./rag_storage"

# Choose embedding backend: "ollama" or "openai"
EMBEDDING_BACKEND = "ollama"

# Ollama settings (used when EMBEDDING_BACKEND == "ollama")
OLLAMA_EMBED_MODEL = "bge-m3:latest"      # pull with: ollama pull bge-m3:latest
OLLAMA_HOST = "http://localhost:11434"

# OpenAI settings (used when EMBEDDING_BACKEND == "openai")
OPENAI_EMBED_MODEL = "text-embedding-3-small"
# ══════════════════════════════════════════════════════════════════════════════


def get_embedding_func():
    """Return an EmbeddingFunc dataclass instance for LightRAG."""
    if EMBEDDING_BACKEND == "ollama":
        from lightrag.llm.ollama import ollama_embed
        from functools import partial

        # Use ollama_embed.func (unwrapped) to avoid double-wrapping inside EmbeddingFunc
        return EmbeddingFunc(
            embedding_dim=1024,        # bge-m3 output dimension
            max_token_size=8192,
            model_name=OLLAMA_EMBED_MODEL,
            func=partial(ollama_embed.func, embed_model=OLLAMA_EMBED_MODEL, host=OLLAMA_HOST),
        )

    elif EMBEDDING_BACKEND == "openai":
        from lightrag.llm.openai import openai_embed
        from functools import partial

        return EmbeddingFunc(
            embedding_dim=1536,        # text-embedding-3-small default
            max_token_size=8191,
            model_name=OPENAI_EMBED_MODEL,
            func=partial(openai_embed.func, model=OPENAI_EMBED_MODEL),
        )

    else:
        raise ValueError(f"Unknown EMBEDDING_BACKEND: {EMBEDDING_BACKEND!r}")


async def main():
    os.makedirs(WORKING_DIR, exist_ok=True)

    embedding_func = get_embedding_func()

    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=claude_cli_complete,
        embedding_func=embedding_func,
    )
    await rag.initialize_storages()

    # ── Insert a document ─────────────────────────────────────────────────────
    sample_text = """
    Albert Einstein was a theoretical physicist born in Ulm, Germany in 1879.
    He developed the theory of relativity, one of the two pillars of modern physics.
    Einstein received the Nobel Prize in Physics in 1921 for his discovery of the
    law of the photoelectric effect. He moved to the United States in 1933 and
    became a professor at the Institute for Advanced Study in Princeton, New Jersey.
    Einstein collaborated with many scientists including Niels Bohr and Max Planck.
    """

    print("Inserting document and extracting entities...")
    await rag.ainsert(sample_text)
    print("Done inserting.\n")

    # ── Query ─────────────────────────────────────────────────────────────────
    query = "Who is Albert Einstein and what are his contributions?"

    for mode in ("naive", "local", "global", "hybrid"):
        print(f"=== Query mode: {mode} ===")
        result = await rag.aquery(query, param=QueryParam(mode=mode))
        print(result)
        print()

    await rag.finalize_storages()


if __name__ == "__main__":
    asyncio.run(main())
