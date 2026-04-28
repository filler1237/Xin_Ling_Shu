from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


def _import_chroma():
    try:
        from langchain_chroma import Chroma

        return Chroma
    except Exception:  # pragma: no cover
        from langchain_community.vectorstores import Chroma  # type: ignore

        return Chroma


def delete_collection(*, persist_dir: Path, collection_name: str) -> bool:
    persist_dir.mkdir(parents=True, exist_ok=True)
    import chromadb

    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        client.delete_collection(collection_name)
        return True
    except Exception:
        return False


def wipe_persist_dir(persist_dir: Path) -> bool:
    if not persist_dir.exists():
        return True
    ok = True
    for p in list(persist_dir.iterdir()):
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=False)
            else:
                p.unlink()
        except Exception:
            ok = False
    return ok


def close_vector_db(vector_db: Any) -> None:
    """
    释放 Chroma/SQLite 文件句柄，避免 Windows 下重建/删除时被锁住。
    """

    client = getattr(vector_db, "_client", None)
    system = getattr(client, "_system", None)
    stop = getattr(system, "stop", None)
    if callable(stop):
        try:
            stop()
        except Exception:
            pass


def build_vector_db(
    *,
    chunks: List[Document],
    embedding: Embeddings,
    persist_dir: Path,
    collection_name: str,
):
    """
    构建并持久化 Chroma 向量库。
    """

    persist_dir.mkdir(parents=True, exist_ok=True)
    Chroma = _import_chroma()
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=str(persist_dir),
        collection_name=collection_name,
    )
    try:
        vs.persist()
    except Exception:
        pass
    return vs


def load_vector_db(
    *,
    embedding: Embeddings,
    persist_dir: Path,
    collection_name: str,
):
    """
    直接加载已持久化的 Chroma（避免重复 embedding）。
    """

    if not persist_dir.exists():
        raise FileNotFoundError(f"向量库目录不存在：{persist_dir}")
    Chroma = _import_chroma()
    return Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embedding,
        collection_name=collection_name,
    )
