from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


DEFAULT_PIPELINE = ["empathy", "reflection", "open_question", "suggestion"]


def load_strategy_map(strategy_file: Path) -> Dict[str, List[str]]:
    if not strategy_file.exists():
        return {}
    raw = json.loads(strategy_file.read_text(encoding="utf-8"))
    out: Dict[str, List[str]] = {}
    for k, v in raw.items():
        if isinstance(v, list):
            out[k] = [str(x) for x in v]
    return out


def select_strategy(intent: str, strategy_map: Dict[str, List[str]]) -> List[str]:
    if intent in strategy_map:
        return strategy_map[intent]
    return strategy_map.get("默认", DEFAULT_PIPELINE)

