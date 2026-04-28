from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RAGConfig:
    """
    所有路径字段都使用“相对项目根目录”的相对路径字符串，便于迁移与本地部署。
    """

    docs_dir: str
    db_dir: str
    collection_name: str

    embedding_model: str
    ollama_model: str
    ollama_base_url: str

    chunk_size: int
    chunk_overlap: int
    top_k: int

    debug_return_retrieved: bool
    max_context_chars: int
    max_history_turns: int

    @staticmethod
    def from_project_root(project_root: Path) -> "RAGConfig":
        """
        默认 docs_dir 指向当前仓库里已生成的 5 个 md 文件所在目录：
        ./新建文件夹

        若你希望换成 ./data/docs，将 md/txt/pdf 放进去并在启动时传 --docs_dir data/docs
        """

        _ = project_root
        return RAGConfig(
            docs_dir="新建文件夹",
            db_dir="data/db",
            collection_name="local_rag",
            embedding_model="BAAI/bge-small-zh-v1.5",
            ollama_model="qwen2.5:7b",
            ollama_base_url="http://localhost:11434",
            chunk_size=500,
            chunk_overlap=100,
            top_k=4,
            debug_return_retrieved=True,
            max_context_chars=12000,
            max_history_turns=6,
        )

    def docs_path(self, project_root: Path) -> Path:
        return (project_root / self.docs_dir).resolve()

    def db_path(self, project_root: Path) -> Path:
        return (project_root / self.db_dir).resolve()
