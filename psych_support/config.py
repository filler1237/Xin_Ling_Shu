from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class PsychConfig:
    symptom_docs_dir: str = "data/psych_docs/symptoms"
    method_docs_dir: str = "data/psych_docs/methods"
    strategy_file: str = "data/psych_docs/strategy/strategies.json"
    strategy_kb_dir: str = "data/psych_docs/strategy"
    db_dir: str = "data/psych_db"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    ollama_model: str = "qwen2.5:7b"
    ollama_base_url: str = "http://localhost:11434"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k_symptom: int = 3
    top_k_method: int = 3
    top_k_strategy: int = 2

    def symptom_docs_path(self, root: Path) -> Path:
        return (root / self.symptom_docs_dir).resolve()

    def method_docs_path(self, root: Path) -> Path:
        return (root / self.method_docs_dir).resolve()

    def strategy_path(self, root: Path) -> Path:
        return (root / self.strategy_file).resolve()

    def strategy_kb_path(self, root: Path) -> Path:
        return (root / self.strategy_kb_dir).resolve()

    def db_path(self, root: Path) -> Path:
        return (root / self.db_dir).resolve()
