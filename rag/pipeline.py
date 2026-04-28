from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from rag.config import RAGConfig
from rag.embedding import get_embedding_model
from rag.generator import AnswerResult, create_llm, generate_answer
from rag.loader import load_documents, load_path
from rag.retriever import RetrievalResult, retrieve_docs
from rag.splitter import split_documents
from rag.vector_store import build_vector_db, close_vector_db, delete_collection, load_vector_db, wipe_persist_dir
from psych_support.analyzer import assess_safety
from psych_support.responder import crisis_response, security_refusal_response


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class AskResult:
    answer: str
    citations_md: str
    retrieved_debug: Optional[str] = None
    retrieval_ms: Optional[float] = None


class RAGAgent:
    """
    完整 RAG Pipeline 封装：
    - build_vector_db()
    - load_vector_db()
    - retrieve_docs()
    - generate_answer()
    - ask()
    """

    def __init__(self, cfg: RAGConfig):
        self.cfg = cfg
        self._root = _project_root()
        self._embedding = get_embedding_model(cfg.embedding_model)
        self._vector_db: Optional[Any] = None
        self._llm = create_llm(model=cfg.ollama_model, base_url=cfg.ollama_base_url)
        self._history: List[Tuple[str, str]] = []

    @property
    def vector_db(self):
        if self._vector_db is None:
            self.load_vector_db()
        return self._vector_db

    def build_vector_db(self, *, force_rebuild: bool = False):
        docs_dir = self.cfg.docs_path(self._root)
        persist_dir = self.cfg.db_path(self._root)

        if force_rebuild:
            if self._vector_db is not None:
                close_vector_db(self._vector_db)
            self._vector_db = None
            deleted = delete_collection(persist_dir=persist_dir, collection_name=self.cfg.collection_name)
            if not deleted:
                wipe_persist_dir(persist_dir)

        docs = load_documents(docs_dir)
        if not docs:
            raise RuntimeError(f"未加载到任何文档：{docs_dir}")

        chunks = split_documents(
            docs,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        try:
            self._vector_db = build_vector_db(
                chunks=chunks,
                embedding=self._embedding,
                persist_dir=persist_dir,
                collection_name=self.cfg.collection_name,
            )
        except ValueError as e:
            msg = str(e)
            if "default_tenant" in msg or "tenant" in msg:
                wipe_persist_dir(persist_dir)
                self._vector_db = build_vector_db(
                    chunks=chunks,
                    embedding=self._embedding,
                    persist_dir=persist_dir,
                    collection_name=self.cfg.collection_name,
                )
            else:
                raise
        return self._vector_db

    def load_vector_db(self):
        persist_dir = self.cfg.db_path(self._root)
        self._vector_db = load_vector_db(
            embedding=self._embedding,
            persist_dir=persist_dir,
            collection_name=self.cfg.collection_name,
        )
        return self._vector_db

    def retrieve_docs(self, query: str, *, top_k: Optional[int] = None) -> RetrievalResult:
        k = top_k if top_k is not None else self.cfg.top_k
        return retrieve_docs(
            vector_db=self.vector_db,
            query=query,
            top_k=k,
            return_debug_text=self.cfg.debug_return_retrieved,
        )

    def generate_answer(self, query: str, retrieved_docs) -> AnswerResult:
        return generate_answer(
            llm=self._llm,
            query=query,
            retrieved_docs=retrieved_docs,
            chat_history=self._history,
            max_history_turns=self.cfg.max_history_turns,
            max_context_chars=self.cfg.max_context_chars,
        )

    def ask(self, query: str, *, top_k: Optional[int] = None) -> AskResult:
        history_user = [q for q, _ in self._history[-8:]] if self._history else []
        analysis = assess_safety(query, history_user_texts=history_user)
        if analysis.crisis:
            safe = crisis_response()
            self._history.append((query, safe.reply))
            return AskResult(
                answer=safe.reply,
                citations_md=safe.references,
                retrieved_debug="### crisis\n- 触发危机分支，已跳过检索。",
            )
        if analysis.blocked:
            safe = security_refusal_response(category=analysis.block_category, reason=analysis.block_reason)
            self._history.append((query, safe.reply))
            return AskResult(
                answer=safe.reply,
                citations_md=safe.references,
                retrieved_debug=f"### security_guard\n- {analysis.block_category}: {analysis.block_reason}",
            )
        rr = self.retrieve_docs(query, top_k=top_k)
        ar = self.generate_answer(query, rr.docs)
        self._history.append((query, ar.answer))
        return AskResult(
            answer=ar.answer,
            citations_md=ar.citations_md,
            retrieved_debug=rr.debug_text,
            retrieval_ms=rr.elapsed_ms,
        )

    def clear_history(self) -> None:
        self._history.clear()

    def add_documents_and_update_db(self, file_paths: Sequence[Path]) -> Dict[str, Any]:
        """
        加分项：上传新文档后增量更新向量库。
        """
        if not file_paths:
            return {"status": "empty", "added_chunks": 0, "added_docs": 0}

        docs: List[Any] = []
        for p in file_paths:
            docs.extend(load_path(p))

        if not docs:
            return {"status": "empty", "added_chunks": 0, "added_docs": 0}

        chunks = split_documents(
            docs,
            chunk_size=self.cfg.chunk_size,
            chunk_overlap=self.cfg.chunk_overlap,
        )

        vs = self.vector_db
        ids = []
        for i, c in enumerate(chunks):
            src = c.metadata.get("source", "")
            cid = f"{src}::chunk::{i}"
            ids.append(cid)
        vs.add_documents(chunks, ids=ids)
        try:
            vs.persist()
        except Exception:
            pass

        return {"status": "ok", "added_chunks": len(chunks), "added_docs": len(docs)}

    def run_chunk_experiments(self) -> List[Dict[str, Any]]:
        """
        参数实验模块：
        - chunk_size: [200, 500, 1000]
        - overlap: [50, 100]
        输出：
        - 文本块数量
        - 检索耗时（ms，平均）
        - 简单命中率（关键词匹配）
        """

        root = self._root
        docs_dir = self.cfg.docs_path(root)
        docs = load_documents(docs_dir)
        if not docs:
            raise RuntimeError(f"未加载到任何文档：{docs_dir}")

        test_queries = [
            "学分是什么？",
            "绩点是怎么换算的？",
            "深技大校园公交什么时候试运行？",
            "E-0 食堂超市什么时候开业？",
            "开学报到前要带哪些证件？",
        ]

        def keywords(q: str) -> List[str]:
            q = q.strip()
            ks = []
            for w in ["学分", "绩点", "换算", "公交", "试运行", "E-0", "超市", "开业", "报到", "证件"]:
                if w in q:
                    ks.append(w)
            if not ks:
                ks = [x for x in q.replace("？", "").replace("?", "").split() if len(x) >= 2]
            return ks

        chunk_sizes = [200, 500, 1000]
        overlaps = [50, 100]

        results: List[Dict[str, Any]] = []
        for cs in chunk_sizes:
            for ov in overlaps:
                chunks = split_documents(docs, chunk_size=cs, chunk_overlap=ov)

                exp_dir = root / "data" / "db" / f"exp_cs{cs}_ov{ov}"
                vs = build_vector_db(
                    chunks=chunks,
                    embedding=self._embedding,
                    persist_dir=exp_dir,
                    collection_name=f"{self.cfg.collection_name}_exp_cs{cs}_ov{ov}",
                )

                times: List[float] = []
                hits = 0
                for q in test_queries:
                    rr = retrieve_docs(
                        vector_db=vs,
                        query=q,
                        top_k=self.cfg.top_k,
                        return_debug_text=False,
                    )
                    times.append(rr.elapsed_ms)
                    kset = keywords(q)
                    joined = "\n".join([d.page_content for d in rr.docs])
                    if any(k in joined for k in kset):
                        hits += 1

                avg_ms = sum(times) / max(len(times), 1)
                hit_rate = hits / max(len(test_queries), 1)
                results.append(
                    {
                        "chunk_size": cs,
                        "overlap": ov,
                        "chunks": len(chunks),
                        "avg_retrieval_ms": round(avg_ms, 2),
                        "hit_rate": round(hit_rate, 3),
                    }
                )

        self._print_table(results)
        return results

    @staticmethod
    def _print_table(rows: List[Dict[str, Any]]) -> None:
        if not rows:
            print("(no results)")
            return
        headers = list(rows[0].keys())
        colw = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}

        def line(vals):
            return " | ".join(str(v).ljust(colw[h]) for h, v in zip(headers, vals))

        print(line(headers))
        print("-+-".join("-" * colw[h] for h in headers))
        for r in rows:
            print(line([r.get(h, "") for h in headers]))
