from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def sessions_dir(root: Optional[Path] = None) -> Path:
    r = root or _project_root()
    d = (r / "data" / "sessions").resolve()
    d.mkdir(parents=True, exist_ok=True)
    return d


def new_session_id() -> str:
    return str(uuid.uuid4())


@dataclass
class SessionData:
    session_id: str
    title: str
    created_at: float
    updated_at: float
    summary: str
    history: List[Tuple[str, str]]
    last_query: str = ""
    last_agent_a: str = ""
    last_agent_b: str = ""
    last_arbiter: str = ""
    last_citations_md: str = ""
    last_retrieved_debug: str = ""
    last_status_md: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "summary": self.summary,
            "history": list(self.history),
            "last_query": self.last_query,
            "last_agent_a": self.last_agent_a,
            "last_agent_b": self.last_agent_b,
            "last_arbiter": self.last_arbiter,
            "last_citations_md": self.last_citations_md,
            "last_retrieved_debug": self.last_retrieved_debug,
            "last_status_md": self.last_status_md,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "SessionData":
        sid = str(d.get("session_id") or "").strip() or new_session_id()
        title = str(d.get("title") or "").strip()
        created_at = float(d.get("created_at") or time.time())
        updated_at = float(d.get("updated_at") or created_at)
        summary = str(d.get("summary") or "")
        last_query = str(d.get("last_query") or "")
        last_agent_a = str(d.get("last_agent_a") or "")
        last_agent_b = str(d.get("last_agent_b") or "")
        last_arbiter = str(d.get("last_arbiter") or "")
        last_citations_md = str(d.get("last_citations_md") or "")
        last_retrieved_debug = str(d.get("last_retrieved_debug") or "")
        last_status_md = str(d.get("last_status_md") or "")
        raw_history = d.get("history") or []
        history: List[Tuple[str, str]] = []
        if isinstance(raw_history, list):
            for item in raw_history:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    history.append((str(item[0]), str(item[1])))
        if not title:
            title = "新会话"
            if history:
                title = (history[0][0] or "新会话").strip()[:24] or "新会话"
        return SessionData(
            session_id=sid,
            title=title,
            created_at=created_at,
            updated_at=updated_at,
            summary=summary,
            history=history,
            last_query=last_query,
            last_agent_a=last_agent_a,
            last_agent_b=last_agent_b,
            last_arbiter=last_arbiter,
            last_citations_md=last_citations_md,
            last_retrieved_debug=last_retrieved_debug,
            last_status_md=last_status_md,
        )


def _session_path(session_id: str, root: Optional[Path] = None) -> Path:
    sid = (session_id or "").strip()
    if not sid:
        sid = new_session_id()
    return sessions_dir(root) / f"{sid}.json"


def save_session(data: SessionData, root: Optional[Path] = None) -> Path:
    now = time.time()
    if not data.created_at:
        data.created_at = now
    data.updated_at = now

    p = _session_path(data.session_id, root)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(p)
    return p


def load_session(session_id: str, root: Optional[Path] = None) -> SessionData:
    p = _session_path(session_id, root)
    d = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(d, dict):
        raise ValueError("Invalid session json")
    return SessionData.from_dict(d)


def delete_session(session_id: str, root: Optional[Path] = None) -> bool:
    p = _session_path(session_id, root)
    try:
        if p.exists():
            p.unlink()
        return True
    except Exception:
        return False


def list_sessions(root: Optional[Path] = None, *, limit: int = 200) -> List[SessionData]:
    d = sessions_dir(root)
    items: List[SessionData] = []
    for p in d.glob("*.json"):
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                continue
            items.append(SessionData.from_dict(raw))
        except Exception:
            continue
    items.sort(key=lambda x: x.updated_at, reverse=True)
    return items[: max(1, int(limit))]
