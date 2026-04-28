from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, List, Optional

from langchain_core.documents import Document


@dataclass
class RetrievalResult:
    docs: List[Document]
    elapsed_ms: float
    debug_text: Optional[str] = None


def retrieve_docs(
    *,
    vector_db: Any,
    query: str,
    top_k: int,
    return_debug_text: bool = False,
) -> RetrievalResult:
    """
    用户问题 -> 向量检索 Top-k 文本块。
    """

    t0 = time.perf_counter()
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    docs: List[Document] = retriever.invoke(query)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    debug_text = None
    if return_debug_text:
        parts = []
        for i, d in enumerate(docs, 1):
            src = d.metadata.get("file_name") or d.metadata.get("source", "unknown")
            snippet = " ".join(d.page_content.split())
            snippet = snippet[:260] + ("…" if len(snippet) > 260 else "")
            parts.append(f"[{i}] {src}\n{snippet}")
        debug_text = "\n\n".join(parts) if parts else "(无检索结果)"

    return RetrievalResult(docs=docs, elapsed_ms=elapsed_ms, debug_text=debug_text)

