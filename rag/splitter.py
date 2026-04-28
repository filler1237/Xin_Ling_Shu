from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    docs: List[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    使用 RecursiveCharacterTextSplitter 切分文档。
    - chunk_size / chunk_overlap 作为实验参数可调
    """

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    for i, c in enumerate(chunks):
        c.metadata = dict(c.metadata or {})
        c.metadata.setdefault("chunk_id", i)
    return chunks

