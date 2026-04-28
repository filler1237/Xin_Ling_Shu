from __future__ import annotations

import argparse
from pathlib import Path

from psych_support.agent import PsychSupportAgent
from psych_support.config import PsychConfig
from rag.config import RAGConfig
from rag.pipeline import RAGAgent
from unified_agent.agent import UnifiedAgent


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RAG 知识问答系统（LangChain + Chroma + BGE + Ollama + Gradio）")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("build", help="构建/更新向量库（若已存在则增量写入/覆盖由实现决定）")
    sub.add_parser("ui", help="启动 Gradio 可视化界面")
    sub.add_parser("exp", help="运行 chunk 参数实验，对比结果输出表格")
    sub.add_parser("psy-build", help="构建心理支持多知识库（symptom_db + method_db）")
    sub.add_parser("psy-chat", help="心理支持智能体命令行对话演示")
    sub.add_parser("unified-build", help="构建统一智能体四库（rag_docs + symptom_db + method_db + strategy_db）")
    sub.add_parser("unified-chat", help="统一智能体命令行对话演示（四库）")

    parser.add_argument("--docs_dir", default=None, help="文档目录（默认使用配置中的 docs_dir）")
    parser.add_argument("--db_dir", default=None, help="向量库持久化目录（默认使用配置中的 db_dir）")
    parser.add_argument("--ollama_model", default=None, help="Ollama 模型名（如 qwen2.5:7b）")
    parser.add_argument("--top_k", type=int, default=None, help="检索 Top-k（默认使用配置中的 top_k）")
    parser.add_argument("--chunk_size", type=int, default=None, help="切分 chunk_size（默认使用配置中的 chunk_size）")
    parser.add_argument("--chunk_overlap", type=int, default=None, help="切分 chunk_overlap（默认使用配置中的 chunk_overlap）")
    args = parser.parse_args()

    root = _project_root()
    cfg = RAGConfig.from_project_root(root)

    if args.docs_dir:
        cfg.docs_dir = args.docs_dir
    if args.db_dir:
        cfg.db_dir = args.db_dir
    if args.ollama_model:
        cfg.ollama_model = args.ollama_model
    if args.top_k is not None:
        cfg.top_k = args.top_k
    if args.chunk_size is not None:
        cfg.chunk_size = args.chunk_size
    if args.chunk_overlap is not None:
        cfg.chunk_overlap = args.chunk_overlap

    agent = RAGAgent(cfg)

    if args.cmd == "build":
        agent.build_vector_db(force_rebuild=False)
        print("✅ 向量库构建完成")
        return

    if args.cmd == "exp":
        agent.run_chunk_experiments()
        return

    if args.cmd == "ui":
        from ui.app import run_gradio

        run_gradio(cfg)
        return

    if args.cmd == "psy-build":
        pcfg = PsychConfig()
        if args.ollama_model:
            pcfg.ollama_model = args.ollama_model
        if args.db_dir:
            pcfg.db_dir = args.db_dir
        if args.chunk_size is not None:
            pcfg.chunk_size = args.chunk_size
        if args.chunk_overlap is not None:
            pcfg.chunk_overlap = args.chunk_overlap

        pa = PsychSupportAgent(pcfg)
        pa.build_kb()
        print("✅ 心理支持多知识库构建完成")
        return

    if args.cmd == "psy-chat":
        pcfg = PsychConfig()
        if args.ollama_model:
            pcfg.ollama_model = args.ollama_model
        if args.db_dir:
            pcfg.db_dir = args.db_dir
        if args.chunk_size is not None:
            pcfg.chunk_size = args.chunk_size
        if args.chunk_overlap is not None:
            pcfg.chunk_overlap = args.chunk_overlap

        pa = PsychSupportAgent(pcfg)
        pa.ensure_kb()
        print("心理支持智能体（非医疗用途）已启动，输入 q 退出。")
        while True:
            user_text = input("\n你：").strip()
            if user_text.lower() in {"q", "quit", "exit"}:
                print("已退出。")
                break
            result = pa.chat(user_text)
            print("\n[识别结果]", result.analysis)
            print("[策略流水线]", " -> ".join(result.strategy_pipeline))
            print("\n助手：", result.reply)
            print("\n[参考资料]\n", result.references)
        return

    if args.cmd == "unified-build":
        ua = UnifiedAgent(rag_cfg=cfg)
        ua.rebuild_all(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
        print("✅ 统一智能体四库构建完成")
        return

    if args.cmd == "unified-chat":
        ua = UnifiedAgent(rag_cfg=cfg)
        print("统一智能体（四库，非医疗用途）已启动，输入 q 退出。")
        while True:
            user_text = input("\n你：").strip()
            if user_text.lower() in {"q", "quit", "exit"}:
                print("已退出。")
                break
            res = ua.ask(user_text, top_k=(cfg.top_k or 4))
            print("\n[分析]", res.analysis)
            print("\n[智能体A]", res.agent_a_answer)
            print("\n[智能体B]", res.agent_b_answer)
            print("\n[仲裁智能体]", res.arbiter_answer)
            print("\n[最终输出]", res.answer)
            print("\n[仲裁信息]", f"selected={res.selected_agent}; reason={res.arbiter_reason}; conflict={res.conflict_report}")
            print("\n[参考资料]\n", res.citations_md)
        return


if __name__ == "__main__":
    main()
