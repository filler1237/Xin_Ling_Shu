from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from psych_support.analyzer import assess_safety
from psych_support.config import PsychConfig
from psych_support.knowledge import MultiKB, build_multi_kb, load_multi_kb, retrieve_from_db
from psych_support.responder import SupportReply, crisis_response, generate_support_reply, security_refusal_response
from psych_support.strategy import load_strategy_map, select_strategy


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class ChatResult:
    analysis: Dict[str, object]
    strategy_pipeline: List[str]
    reply: str
    references: str


class PsychSupportAgent:
    def __init__(self, cfg: Optional[PsychConfig] = None):
        self.cfg = cfg or PsychConfig()
        self.root = _project_root()
        self.kb: Optional[MultiKB] = None
        self.strategy_map = load_strategy_map(self.cfg.strategy_path(self.root))
        self.history: List[Tuple[str, str]] = []

    def build_kb(self) -> None:
        self.kb = build_multi_kb(self.cfg, self.root)

    def load_kb(self) -> None:
        self.kb = load_multi_kb(self.cfg, self.root)

    def ensure_kb(self) -> None:
        if self.kb is not None:
            return
        try:
            self.load_kb()
        except Exception:
            self.build_kb()

    def chat(self, user_text: str) -> ChatResult:
        history_user = [q for q, _ in self.history[-8:]] if self.history else []
        analysis_obj = assess_safety(user_text, history_user_texts=history_user)
        analysis = analysis_obj.to_dict()

        if analysis_obj.crisis:
            safe = crisis_response()
            self.history.append((user_text, safe.reply))
            return ChatResult(
                analysis=analysis,
                strategy_pipeline=["safety_protocol"],
                reply=safe.reply,
                references=safe.references,
            )

        if analysis_obj.blocked:
            safe = security_refusal_response(category=analysis_obj.block_category, reason=analysis_obj.block_reason)
            self.history.append((user_text, safe.reply))
            return ChatResult(
                analysis=analysis,
                strategy_pipeline=["security_guard"],
                reply=safe.reply,
                references=safe.references,
            )

        self.ensure_kb()
        assert self.kb is not None
        symptom_docs = retrieve_from_db(self.kb.symptom_db, user_text, self.cfg.top_k_symptom)
        method_docs = retrieve_from_db(self.kb.method_db, user_text, self.cfg.top_k_method)
        strategy_query = f"{analysis_obj.intent} {analysis_obj.emotion} {user_text}"
        strategy_docs = retrieve_from_db(self.kb.strategy_db, strategy_query, self.cfg.top_k_strategy)

        pipeline = select_strategy(str(analysis_obj.intent), self.strategy_map)
        out: SupportReply = generate_support_reply(
            cfg=self.cfg,
            user_text=user_text,
            analysis=analysis,
            strategy_pipeline=pipeline,
            symptom_docs=symptom_docs,
            method_docs=method_docs,
            strategy_docs=strategy_docs,
            chat_history=self.history,
        )
        self.history.append((user_text, out.reply))
        return ChatResult(
            analysis=analysis,
            strategy_pipeline=pipeline,
            reply=out.reply,
            references=out.references,
        )
