from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings


@lru_cache(maxsize=1)
def get_embedding_model(model_name: str) -> Embeddings:
    """
    使用 BAAI/bge-small-zh-v1.5 作为本地 Embedding 模型（通过 sentence-transformers 下载到本地缓存）。
    """

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception:  # pragma: no cover
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore

    return HuggingFaceEmbeddings(
        model_name=model_name,
        encode_kwargs={"normalize_embeddings": True},
    )

