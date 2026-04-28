from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from rag.embedding import get_embedding_model
from rag.loader import load_documents
from rag.splitter import split_documents
from rag.vector_store import build_vector_db, load_vector_db

from psych_support.config import PsychConfig


@dataclass
class MultiKB:
    symptom_db: object
    method_db: object
    strategy_db: object


def _collection_name(base: str, suffix: str) -> str:
    return f"{base}_{suffix}"


def _strategy_item_to_text(item: dict) -> str:
    name = str(item.get("strategy_name", "")).strip()
    target = str(item.get("target_state", "")).strip()
    sev = item.get("severity_level", [])
    sev_text = "、".join([str(x) for x in sev]) if isinstance(sev, list) else str(sev)
    desc = str(item.get("strategy_description", "")).strip()
    escalation = str(item.get("escalation", "")).strip()
    phases = item.get("phases", [])
    phase_lines = []
    if isinstance(phases, list):
        for p in phases:
            if not isinstance(p, dict):
                continue
            phase_lines.append(
                f"- {p.get('phase_name','')}: 目标={p.get('goal','')} 时长={p.get('duration','')} 方法={p.get('recommended_methods',[])}"
            )
    conditions = item.get("conditions", [])
    cond_lines = []
    if isinstance(conditions, list):
        for c in conditions:
            if not isinstance(c, dict):
                continue
            cond_lines.append(f"- if {c.get('if','')} then {c.get('then','')}")
    raw_json = json.dumps(item, ensure_ascii=False)
    return (
        f"策略名称：{name}\n"
        f"目标状态：{target}\n"
        f"适用严重度：{sev_text}\n"
        f"策略描述：{desc}\n\n"
        f"阶段：\n" + ("\n".join(phase_lines) if phase_lines else "- （无）") + "\n\n"
        f"条件逻辑：\n" + ("\n".join(cond_lines) if cond_lines else "- （无）") + "\n\n"
        f"升级路径：{escalation}\n\n"
        f"原始JSON：{raw_json}"
    )


def load_strategy_kb_documents(strategy_dir: Path) -> List[Document]:
    """
    读取 strategy_kb.json 等策略知识库 JSON（数组），并转换为可向量化的 Documents。
    - 会跳过 strategies.json（调度映射文件）
    """

    docs: List[Document] = []
    if not strategy_dir.exists():
        return docs
    for p in strategy_dir.glob("*.json"):
        if p.name == "strategies.json":
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            continue
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            text = _strategy_item_to_text(item)
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(p.as_posix()),
                        "file_name": p.name,
                        "file_ext": ".json",
                        "strategy_name": item.get("strategy_name", ""),
                        "target_state": item.get("target_state", ""),
                        "severity_level": item.get("severity_level", []),
                        "row": i,
                    },
                )
            )
    return docs


def build_multi_kb(cfg: PsychConfig, root: Path, *, base_collection: str = "psych_support") -> MultiKB:
    emb = get_embedding_model(cfg.embedding_model)
    db_path = cfg.db_path(root)
    symptom_docs = load_documents(cfg.symptom_docs_path(root))
    method_docs = load_documents(cfg.method_docs_path(root))
    strategy_docs = load_strategy_kb_documents(cfg.strategy_kb_path(root))

    symptom_chunks = split_documents(
        symptom_docs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    method_chunks = split_documents(
        method_docs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )
    strategy_chunks = split_documents(
        strategy_docs,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
    )

    symptom_db = build_vector_db(
        chunks=symptom_chunks,
        embedding=emb,
        persist_dir=db_path,
        collection_name=_collection_name(base_collection, "symptom"),
    )
    method_db = build_vector_db(
        chunks=method_chunks,
        embedding=emb,
        persist_dir=db_path,
        collection_name=_collection_name(base_collection, "method"),
    )
    strategy_db = build_vector_db(
        chunks=strategy_chunks,
        embedding=emb,
        persist_dir=db_path,
        collection_name=_collection_name(base_collection, "strategy"),
    )
    return MultiKB(symptom_db=symptom_db, method_db=method_db, strategy_db=strategy_db)


def load_multi_kb(cfg: PsychConfig, root: Path, *, base_collection: str = "psych_support") -> MultiKB:
    emb = get_embedding_model(cfg.embedding_model)
    db_path = cfg.db_path(root)
    symptom_db = load_vector_db(
        embedding=emb,
        persist_dir=db_path,
        collection_name=_collection_name(base_collection, "symptom"),
    )
    method_db = load_vector_db(
        embedding=emb,
        persist_dir=db_path,
        collection_name=_collection_name(base_collection, "method"),
    )
    strategy_db = load_vector_db(
        embedding=emb,
        persist_dir=db_path,
        collection_name=_collection_name(base_collection, "strategy"),
    )
    return MultiKB(symptom_db=symptom_db, method_db=method_db, strategy_db=strategy_db)


def retrieve_from_db(vector_db: object, query: str, top_k: int) -> List[Document]:
    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    return retriever.invoke(query)
