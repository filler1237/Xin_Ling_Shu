from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StructuredA:
    main_issue: str = ""
    risk_level: str = ""
    watch_metrics: List[str] = None  # type: ignore[assignment]
    seek_help_when: List[str] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "main_issue": self.main_issue,
            "risk_level": self.risk_level,
            "watch_metrics": list(self.watch_metrics or []),
            "seek_help_when": list(self.seek_help_when or []),
        }


@dataclass
class StructuredB:
    top2_actions: List[Dict[str, str]] = None  # type: ignore[assignment]
    optional_actions: List[Dict[str, str]] = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "top2_actions": list(self.top2_actions or []),
            "optional_actions": list(self.optional_actions or []),
        }


_JSON_BLOCK_RE = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.IGNORECASE)


def _extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON_BLOCK_RE.search(text or "")
    if not m:
        return None
    raw = m.group(1).strip()
    try:
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _normalize_lines(text: str) -> List[str]:
    lines: List[str] = []
    for ln in (text or "").splitlines():
        s = ln.strip()
        if not s:
            continue
        s = s.lstrip("-•*").strip()
        if s:
            lines.append(s)
    return lines


def parse_structured_a(text: str) -> StructuredA:
    data = _extract_json_block(text) or {}
    if data.get("type") == "agent_a":
        main_issue = str(data.get("main_issue") or "").strip()
        risk_level = str(data.get("risk_level") or "").strip()
        watch = data.get("watch_metrics") or []
        seek = data.get("seek_help_when") or []
        watch_metrics = [str(x).strip() for x in watch if str(x).strip()][:3]
        seek_help_when = [str(x).strip() for x in seek if str(x).strip()][:3]
        return StructuredA(main_issue=main_issue, risk_level=risk_level, watch_metrics=watch_metrics, seek_help_when=seek_help_when)

    lines = _normalize_lines(text)
    watch_metrics: List[str] = []
    seek_help_when: List[str] = []
    for ln in lines:
        if any(k in ln for k in ["观察", "指标", "频率", "强度", "持续", "影响"]):
            watch_metrics.append(ln)
        if any(k in ln for k in ["就医", "升级", "尽快", "急诊", "自伤", "轻生", "危机"]):
            seek_help_when.append(ln)
    watch_metrics = watch_metrics[:3]
    seek_help_when = seek_help_when[:3]
    main_issue = ""
    if lines:
        main_issue = lines[0][:80]
    return StructuredA(main_issue=main_issue, risk_level="", watch_metrics=watch_metrics, seek_help_when=seek_help_when)


def parse_structured_b(text: str) -> StructuredB:
    data = _extract_json_block(text) or {}
    if data.get("type") == "agent_b":
        top2 = data.get("top2_actions") or []
        opt = data.get("optional_actions") or []

        def norm_actions(items) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            if not isinstance(items, list):
                return out
            for it in items:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title") or "").strip()
                duration = str(it.get("duration") or "").strip()
                trigger = str(it.get("trigger") or "").strip()
                metric = str(it.get("metric") or "").strip()
                if not title:
                    continue
                out.append({"title": title, "duration": duration, "trigger": trigger, "metric": metric})
            return out

        return StructuredB(top2_actions=norm_actions(top2)[:2], optional_actions=norm_actions(opt)[:2])

    lines = _normalize_lines(text)
    top2_actions: List[Dict[str, str]] = []
    for ln in lines:
        if len(top2_actions) >= 2:
            break
        if any(k in ln for k in ["分钟", "每天", "每周", "次数", "时长", "触发", "验收"]):
            top2_actions.append({"title": ln[:80], "duration": "", "trigger": "", "metric": ""})
    return StructuredB(top2_actions=top2_actions, optional_actions=[])

