"""Local embeddings via Ollama (nomic-embed-text).

nomic-embed-text expects a task prefix on every input, so we add one:
  search_document: -> text being stored
  search_query:    -> text used to look things up
  clustering:      -> short entity labels compared with each other
"""
import ollama

from . import config

_client = ollama.Client(host=config.OLLAMA_HOST)


def _embed(inputs: list[str]) -> list[list[float]]:
    return _client.embed(model=config.EMBED_MODEL, input=inputs)["embeddings"]


def embed_documents(texts: list[str]) -> list[list[float]]:
    return _embed([f"search_document: {t}" for t in texts])


def embed_query(text: str) -> list[float]:
    return _embed([f"search_query: {text}"])[0]


def embed_label(text: str) -> list[float]:
    return _embed([f"clustering: {text}"])[0]
