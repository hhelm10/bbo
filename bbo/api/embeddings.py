"""Unified embedding interface for local and API-based models."""

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

# Model name -> (provider, model_id, embed_dim)
EMBEDDING_REGISTRY = {
    "nomic-embed-text-v1.5": ("local", "nomic-ai/nomic-embed-text-v1.5", 768),
    "all-MiniLM-L6-v2": ("local", "sentence-transformers/all-MiniLM-L6-v2", 384),
    "text-embedding-3-small": ("openai", "text-embedding-3-small", 1536),
    "voyage-3-lite": ("voyage", "voyage-3-lite", 1024),
}


def _embed_local(model_id: str, texts: List[str], batch_size: int = 64) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_id, trust_remote_code=True)

    # nomic requires document prefix
    if "nomic" in model_id:
        texts = [f"search_document: {t}" for t in texts]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return embeddings


def _embed_openai(model_id: str, texts: List[str], batch_size: int = 2000) -> np.ndarray:
    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embeddings.create(model=model_id, input=batch)
        batch_emb = [item.embedding for item in resp.data]
        all_embeddings.extend(batch_emb)

    return np.array(all_embeddings, dtype=np.float32)


def _embed_voyage(model_id: str, texts: List[str], batch_size: int = 128) -> np.ndarray:
    import voyageai
    client = voyageai.Client(api_key=os.environ["VOYAGE_API_KEY"])

    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        resp = client.embed(batch, model=model_id)
        all_embeddings.extend(resp.embeddings)

    return np.array(all_embeddings, dtype=np.float32)


_PROVIDERS = {
    "local": _embed_local,
    "openai": _embed_openai,
    "voyage": _embed_voyage,
}


def embed_texts(texts: List[str], model: str, batch_size: int = None) -> np.ndarray:
    """Embed texts using the specified model.

    Parameters
    ----------
    texts : list of str
    model : str
        One of the keys in EMBEDDING_REGISTRY
    batch_size : int, optional
        Override default batch size

    Returns
    -------
    np.ndarray of shape (len(texts), embed_dim)
    """
    if model not in EMBEDDING_REGISTRY:
        raise ValueError(f"Unknown embedding model: {model}. "
                         f"Available: {list(EMBEDDING_REGISTRY.keys())}")

    provider, model_id, _ = EMBEDDING_REGISTRY[model]
    call_fn = _PROVIDERS[provider]

    kwargs = {"model_id": model_id, "texts": texts}
    if batch_size is not None:
        kwargs["batch_size"] = batch_size

    return call_fn(**kwargs)
