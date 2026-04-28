from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from langchain_core.documents import Document

try:
    from langchain_community.document_loaders import PyMuPDFLoader
except Exception as e:  # pragma: no cover
    PyMuPDFLoader = None  # type: ignore[assignment]
    _pymupdf_import_error = e


def _load_text_file(path: Path) -> List[Document]:
    """
    将 .md/.txt 当作纯文本加载（Markdown 不做渲染，保持原始文本，方便 RAG 检索）。
    """

    raw = None
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            raw = path.read_text(encoding=enc)
            break
        except Exception:
            continue
    if raw is None:
        raw = path.read_text(errors="ignore")

    return [
        Document(
            page_content=raw,
            metadata={
                "source": str(path.as_posix()),
                "file_name": path.name,
                "file_ext": path.suffix.lower(),
            },
        )
    ]


def _load_pdf_file(path: Path) -> List[Document]:
    if PyMuPDFLoader is None:  # pragma: no cover
        raise RuntimeError(
            "未安装/无法导入 PyMuPDFLoader。请安装依赖 pymupdf。原始错误："
            + repr(_pymupdf_import_error)
        )
    loader = PyMuPDFLoader(str(path))
    docs = loader.load()
    for d in docs:
        d.metadata.setdefault("source", str(path.as_posix()))
        d.metadata.setdefault("file_name", path.name)
        d.metadata.setdefault("file_ext", path.suffix.lower())
    return docs


def iter_supported_files(docs_dir: Path) -> Iterable[Path]:
    exts = {".md", ".txt", ".pdf"}
    for p in docs_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("~$"):
            yield p


def load_documents(docs_dir: Path) -> List[Document]:
    """
    自动读取目录下所有 md/txt/pdf。
    """

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs_dir 不存在：{docs_dir}")

    docs: List[Document] = []
    for path in iter_supported_files(docs_dir):
        if path.suffix.lower() in (".md", ".txt"):
            docs.extend(_load_text_file(path))
        elif path.suffix.lower() == ".pdf":
            docs.extend(_load_pdf_file(path))
    return docs


def load_path(path: Path) -> List[Document]:
    """
    加载单个文件或目录。
    - file: 返回该文件解析出的 Documents
    - dir: 递归加载目录下所有支持格式
    """

    if not path.exists():
        raise FileNotFoundError(f"path 不存在：{path}")
    if path.is_dir():
        return load_documents(path)
    ext = path.suffix.lower()
    if ext in (".md", ".txt"):
        return _load_text_file(path)
    if ext == ".pdf":
        return _load_pdf_file(path)
    return []
