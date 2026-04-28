from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

from rag.config import RAGConfig
from unified_agent.agent import UnifiedAgent
from unified_agent.session_store import SessionData, delete_session, list_sessions, load_session, new_session_id, save_session


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_unified_agent(cfg: RAGConfig, agent: Optional[UnifiedAgent]) -> UnifiedAgent:
    if agent is not None:
        return agent
    return UnifiedAgent(rag_cfg=cfg)


def _normalize_upload_path(file_obj: Any) -> Path:
    raw = getattr(file_obj, "name", file_obj)
    return Path(str(raw))


def _thinking_markup() -> str:
    return (
        '<div class="thinking-indicator">'
        '<span class="thinking-dot"></span>'
        '<span class="thinking-dot"></span>'
        '<span class="thinking-dot"></span>'
        "</div>"
    )


def _status_text(cfg: RAGConfig, agent: UnifiedAgent, res) -> str:
    return (
        "### 当前状态\n"
        f"- 模式：统一智能体（四库）\n"
        f"- docs_dir：./{cfg.docs_dir}\n"
        f"- rag_db_dir：./{cfg.db_dir}\n"
        f"- psych_db_dir：./{agent.psych_cfg.db_dir}\n"
        f"- analysis：{res.analysis}\n"
        f"- selected_agent：{res.selected_agent}\n"
        f"- arbiter_reason：{res.arbiter_reason}\n"
        f"- conflict_report：{res.conflict_report}"
    )


def _empty_chat_history() -> List[Dict[str, str]]:
    return []


def _to_chat_history(turns: List[tuple[str, str]]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for q, a in turns:
        out.append({"role": "user", "content": str(q)})
        out.append({"role": "assistant", "content": str(a)})
    return out


def _session_choices(root: Path) -> List[tuple[str, str]]:
    items = list_sessions(root)
    out: List[tuple[str, str]] = [("（未选择）", "")]
    for s in items:
        label = f"{s.title} | {s.session_id[:8]}"
        out.append((label, s.session_id))
    return out


def _ensure_session(agent: UnifiedAgent, root: Path, session_id: Optional[str]) -> str:
    sid = (session_id or "").strip()
    if sid:
        agent.session_id = sid
        return sid
    sid = new_session_id()
    agent.session_id = sid
    save_session(
        SessionData(
            session_id=sid,
            title="新会话",
            created_at=0.0,
            updated_at=0.0,
            summary="",
            history=[],
        ),
        root,
    )
    return sid


def run_gradio(cfg: RAGConfig) -> None:
    root = _project_root()

    css = """
    /* 全局极简浅色背景 */
    .gradio-container {
        background: #F9FAFB !important;
    }
    .app-shell {
        max-width: 1440px;
        margin: 0 auto;
    }
    /* 卡片悬浮极简风格 */
    .hero-card, .sidebar-card, .chat-card {
        border: none !important;
        border-radius: 16px;
        background: #FFFFFF !important;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.03) !important;
    }
    .hero-card {
        padding: 16px 20px;
        margin-bottom: 16px;
    }
    .chat-card {
        padding: 16px;
    }
    .sidebar-card {
        padding: 16px;
    }
    .section-title {
        font-weight: 600;
        color: #374151;
        margin-bottom: 8px;
    }
    .subtle-text {
        color: #9CA3AF;
        font-size: 0.96rem;
    }
    /* 聊天气泡样式定制 (类似 ChatGPT) */
    .message.user {
        background: #ECFDF5 !important;
        border: 1px solid #D1FAE5 !important;
        border-radius: 12px 12px 0 12px !important;
        color: #1F2937 !important;
    }
    .message.bot {
        background: #FFFFFF !important;
        border: 1px solid #F3F4F6 !important;
        border-radius: 12px 12px 12px 0 !important;
        color: #374151 !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02) !important;
    }
    /* 思考动画 */
    .thinking-indicator {
        display: flex;
        align-items: center;
        gap: 6px;
        padding: 12px 16px;
        border-radius: 12px 12px 12px 0;
        background: #FFFFFF;
        border: 1px solid #F3F4F6;
        width: fit-content;
        margin-top: 4px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.02);
    }
    .thinking-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #9CA3AF;
        animation: pulse-dot 1.4s infinite ease-in-out both;
    }
    .thinking-dot:nth-child(1) { animation-delay: -0.32s; }
    .thinking-dot:nth-child(2) { animation-delay: -0.16s; }
    .thinking-dot:nth-child(3) { animation-delay: 0s; }
    
    @keyframes pulse-dot {
        0%, 80%, 100% { transform: scale(0); opacity: 0.3; }
        40% { transform: scale(1); opacity: 1; }
    }
    """

    with gr.Blocks(
        title="心灵树",
        fill_height=False,
    ) as demo:
        state_agent = gr.State(value=None)
        chat_state = gr.State(value=_empty_chat_history())
        session_state = gr.State(value="")

        with gr.Column(elem_classes="app-shell"):
            with gr.Column(elem_classes="hero-card", scale=0):
                gr.Markdown(
                    "## 心灵树\n"
                    "<div class='subtle-text'>"
                    "以聊天为中心的轻量界面，支持智能体 A / B / 仲裁智能体联动问答，"
                    "并保留知识库维护与调试能力。"
                    "</div>"
                )

            with gr.Tabs():
                with gr.Tab("💬 主对话区"):
                    with gr.Row(equal_height=True):
                        with gr.Column(scale=7, min_width=780):
                            with gr.Column(elem_classes="chat-card"):
                                chatbot = gr.Chatbot(
                                    label="对话区",
                                    height=720,
                                    show_label=False,
                                    avatar_images=(None, None),
                                    layout="bubble",
                                    placeholder="开始一段新的对话吧",
                                )
                                thinking = gr.HTML(value="", visible=False)
                                question = gr.Textbox(
                                    label="消息输入",
                                    show_label=False,
                                    placeholder="输入你的问题，按 Enter 或点击发送…",
                                    lines=3,
                                )
                                with gr.Row():
                                    btn_ask = gr.Button("发送", variant="primary")
                                    btn_clear = gr.Button("清空对话")

                        with gr.Column(scale=3, min_width=320):
                            with gr.Column(elem_classes="sidebar-card"):
                                with gr.Accordion("⚙️ 参数设置", open=True):
                                    top_k = gr.Slider(1, 10, value=cfg.top_k, step=1, label="Top-k")
                                    chunk_size = gr.Dropdown([200, 500, 1000], value=cfg.chunk_size, label="chunk_size")
                                    chunk_overlap = gr.Dropdown([50, 100], value=cfg.chunk_overlap, label="chunk_overlap")
                                    btn_build = gr.Button("重建向量库（按当前参数）")

                                with gr.Accordion("💾 会话", open=False):
                                    session_select = gr.Dropdown(
                                        choices=_session_choices(root),
                                        value="",
                                        label="选择会话",
                                        interactive=True,
                                    )
                                    with gr.Row():
                                        btn_new_session = gr.Button("新建会话")
                                        btn_delete_session = gr.Button("删除会话")

                                with gr.Accordion("📚 知识库维护", open=False):
                                    upload = gr.Files(label="上传文档（md/txt/pdf）", file_count="multiple")
                                    kb_target = gr.Dropdown(
                                        choices=["rag_docs", "symptom_db", "method_db", "strategy_db"],
                                        value="rag_docs",
                                        label="上传目标库",
                                    )
                                    btn_update = gr.Button("上传并更新向量库")

                with gr.Tab("🧠 智能体思维与调试"):
                    with gr.Column(elem_classes="sidebar-card"):
                        gr.Markdown("### 🤖 A：事实归纳", elem_classes="section-title")
                        agent_a_answer = gr.Markdown(value="等待提问…")
                    with gr.Column(elem_classes="sidebar-card"):
                        gr.Markdown("### 🤖 B：策略建议", elem_classes="section-title")
                        agent_b_answer = gr.Markdown(value="等待提问…")
                    with gr.Column(elem_classes="sidebar-card"):
                        gr.Markdown("### ⚖️ 仲裁最终版", elem_classes="section-title")
                        arbiter_answer = gr.Markdown(value="等待提问…")
                        
                    with gr.Accordion("🔍 证据与运行状态", open=False):
                        with gr.Tabs():
                            with gr.Tab("参考资料"):
                                citations = gr.Markdown(value="暂无内容。")
                            with gr.Tab("检索片段"):
                                retrieved = gr.Markdown(value="暂无内容。")
                            with gr.Tab("运行状态"):
                                status = gr.Markdown(value="等待提问…")

        def do_build(cs: int, ov: int, agent: Optional[UnifiedAgent]):
            cfg2 = cfg
            cfg2.chunk_size = int(cs)
            cfg2.chunk_overlap = int(ov)
            a = _ensure_unified_agent(cfg2, agent)
            yield a, "### 当前状态\n- 正在重建全部知识库（四库）…（首次可能较慢）"
            try:
                a.rebuild_all(chunk_size=int(cs), chunk_overlap=int(ov))
                yield a, f"### 当前状态\n- 已重建四库：chunk_size={cs}, overlap={ov}"
            except Exception as e:
                yield a, f"### 当前状态\n- 重建失败：{type(e).__name__}: {e}"

        def do_ask(message: str, history: List[Dict[str, str]], k: int, agent: Optional[UnifiedAgent], session_id: str):
            text = (message or "").strip()
            if not text:
                return (
                    "",
                    history or [],
                    history or [],
                    agent,
                    session_id or "",
                    gr.update(visible=False, value=""),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    gr.update(),
                    "### 当前状态\n- 请输入问题。",
                    gr.update(choices=_session_choices(root), value=(session_id or "").strip() or ""),
                )

            current_history = list(history or [])
            current_history.append({"role": "user", "content": text})
            current_history.append({"role": "assistant", "content": "正在思考中..."})
            yield (
                "",
                current_history,
                current_history,
                agent,
                session_id or "",
                gr.update(visible=True, value=_thinking_markup()),
                "生成中…",
                "生成中…",
                "生成中…",
                "生成中…",
                "生成中…",
                "### 当前状态\n- AI 正在思考中…",
                gr.update(choices=_session_choices(root), value=(session_id or "").strip() or ""),
            )

            a = _ensure_unified_agent(cfg, agent)
            sid = _ensure_session(a, root, session_id)
            try:
                res = a.ask(text, top_k=int(k))
                if sid:
                    existing_title = ""
                    if a.history:
                        existing_title = (a.history[0][0] or "").strip()[:24]
                    title = existing_title or "新会话"
                    status_md = _status_text(cfg, a, res)
                    save_session(
                        SessionData(
                            session_id=sid,
                            title=title,
                            created_at=0.0,
                            updated_at=0.0,
                            summary=getattr(a, "_summary", ""),
                            history=list(a.history),
                            last_query=text,
                            last_agent_a=res.agent_a_answer,
                            last_agent_b=res.agent_b_answer,
                            last_arbiter=res.arbiter_answer,
                            last_citations_md=res.citations_md,
                            last_retrieved_debug=res.retrieved_debug,
                            last_status_md=status_md,
                        ),
                        root,
                    )
                current_history[-1] = {"role": "assistant", "content": res.answer}
                yield (
                    "",
                    current_history,
                    current_history,
                    a,
                    sid,
                    gr.update(visible=False, value=""),
                    res.agent_a_answer,
                    res.agent_b_answer,
                    res.arbiter_answer,
                    res.citations_md,
                    res.retrieved_debug,
                    status_md,
                    gr.update(choices=_session_choices(root), value=sid),
                )
            except Exception as e:
                current_history[-1] = {"role": "assistant", "content": f"抱歉，这次处理失败了：{type(e).__name__}: {e}"}
                yield (
                    "",
                    current_history,
                    current_history,
                    a,
                    sid,
                    gr.update(visible=False, value=""),
                    "本次请求失败，未生成。",
                    "本次请求失败，未生成。",
                    "本次请求失败，未生成。",
                    "暂无内容。",
                    "暂无内容。",
                    f"### 当前状态\n- 请求失败：{type(e).__name__}: {e}",
                    gr.update(choices=_session_choices(root), value=sid),
                )

        def do_clear(agent: Optional[UnifiedAgent], session_id: str):
            sid = (session_id or "").strip()
            if agent is not None:
                agent.clear_history()
                if sid:
                    agent.session_id = sid
                    save_session(
                        SessionData(
                            session_id=sid,
                            title="新会话",
                            created_at=0.0,
                            updated_at=0.0,
                            summary="",
                            history=[],
                        ),
                        root,
                    )
            return (
                "",
                [],
                [],
                agent,
                sid,
                gr.update(visible=False, value=""),
                "等待提问…",
                "等待提问…",
                "等待提问…",
                "暂无内容。",
                "暂无内容。",
                "### 当前状态\n- 已清空对话历史。",
                gr.update(choices=_session_choices(root), value=sid),
            )

        def do_new_session(agent: Optional[UnifiedAgent]):
            a = _ensure_unified_agent(cfg, agent)
            a.clear_history()
            sid = new_session_id()
            a.session_id = sid
            save_session(
                SessionData(
                    session_id=sid,
                    title="新会话",
                    created_at=0.0,
                    updated_at=0.0,
                    summary="",
                    history=[],
                    last_query="",
                    last_agent_a="",
                    last_agent_b="",
                    last_arbiter="",
                    last_citations_md="",
                    last_retrieved_debug="",
                    last_status_md="",
                ),
                root,
            )
            return (
                "",
                [],
                [],
                a,
                sid,
                gr.update(visible=False, value=""),
                "等待提问…",
                "等待提问…",
                "等待提问…",
                "暂无内容。",
                "暂无内容。",
                "### 当前状态\n- 已新建会话。",
                gr.update(choices=_session_choices(root), value=sid),
            )

        def do_load_session(selected: str, agent: Optional[UnifiedAgent]):
            sid = (selected or "").strip()
            a = _ensure_unified_agent(cfg, agent)
            if not sid:
                return (
                    "",
                    [],
                    [],
                    a,
                    "",
                    gr.update(visible=False, value=""),
                    "等待提问…",
                    "等待提问…",
                    "等待提问…",
                    "暂无内容。",
                    "暂无内容。",
                    "### 当前状态\n- 请选择要加载的会话。",
                    gr.update(choices=_session_choices(root), value=""),
                )
            data = load_session(sid, root)
            a.session_id = data.session_id
            a.import_session_state({"summary": data.summary, "history": data.history})
            ch = _to_chat_history(list(a.history))
            a_md = data.last_agent_a or "等待提问…"
            b_md = data.last_agent_b or "等待提问…"
            c_md = data.last_arbiter or "等待提问…"
            cit_md = data.last_citations_md or "暂无内容。"
            ret_md = data.last_retrieved_debug or "暂无内容。"
            st_md = data.last_status_md or "### 当前状态\n- 已加载会话。"
            return (
                "",
                ch,
                ch,
                a,
                sid,
                gr.update(visible=False, value=""),
                a_md,
                b_md,
                c_md,
                cit_md,
                ret_md,
                st_md,
                gr.update(choices=_session_choices(root), value=sid),
            )

        def do_delete_session(selected: str, agent: Optional[UnifiedAgent]):
            sid = (selected or "").strip()
            if sid:
                delete_session(sid, root)
            a = _ensure_unified_agent(cfg, agent)
            if getattr(a, "session_id", "") == sid:
                a.clear_history()
                a.session_id = ""
            return (
                "",
                [],
                [],
                a,
                "",
                gr.update(visible=False, value=""),
                "等待提问…",
                "等待提问…",
                "等待提问…",
                "暂无内容。",
                "暂无内容。",
                "### 当前状态\n- 已删除会话。",
                gr.update(choices=_session_choices(root), value=""),
            )

        def do_refresh_sessions():
            return gr.update(choices=_session_choices(root), value=""), ""

        def do_update(target: str, files, agent: Optional[UnifiedAgent]):
            if not files:
                return agent, "### 当前状态\n- 请先选择文件。"
            a = _ensure_unified_agent(cfg, agent)
            saved = []
            if target == "rag_docs":
                ra = a.rag
                docs_dir = root / cfg.docs_dir
                docs_dir.mkdir(parents=True, exist_ok=True)
                upload_dir = docs_dir / "uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                for f in files:
                    src = _normalize_upload_path(f)
                    dst = upload_dir / src.name
                    dst.write_bytes(src.read_bytes())
                    saved.append(dst)
                yield a, f"### 当前状态\n- 已保存 {len(saved)} 个文件到 rag_docs，正在更新向量库…"
                try:
                    info = ra.add_documents_and_update_db(saved)
                    yield a, (
                        "### 当前状态\n"
                        f"- rag_docs 已更新：added_docs={info.get('added_docs')}, "
                        f"added_chunks={info.get('added_chunks')}（文件保存在 ./{cfg.docs_dir}/uploads）"
                    )
                except Exception as e:
                    yield a, f"### 当前状态\n- rag_docs 更新失败：{type(e).__name__}: {e}"
                return

            if target == "symptom_db":
                base_dir = root / a.psych_cfg.symptom_docs_dir
            elif target == "method_db":
                base_dir = root / a.psych_cfg.method_docs_dir
            else:
                base_dir = root / a.psych_cfg.strategy_kb_dir

            base_dir.mkdir(parents=True, exist_ok=True)
            upload_dir = base_dir / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                src = _normalize_upload_path(f)
                dst = upload_dir / src.name
                dst.write_bytes(src.read_bytes())
                saved.append(dst)
            yield a, f"### 当前状态\n- 已保存 {len(saved)} 个文件到 {target}，正在重建心理支持三库…"
            try:
                a.psy.build_kb()
                yield a, f"### 当前状态\n- 心理支持三库已更新：新增文件已写入 {target} 并完成重建"
            except Exception as e:
                yield a, f"### 当前状态\n- 心理支持三库更新失败：{type(e).__name__}: {e}"

        ask_outputs = [
            question,
            chatbot,
            chat_state,
            state_agent,
            session_state,
            thinking,
            agent_a_answer,
            agent_b_answer,
            arbiter_answer,
            citations,
            retrieved,
            status,
            session_select,
        ]

        btn_build.click(
            do_build,
            inputs=[chunk_size, chunk_overlap, state_agent],
            outputs=[state_agent, status],
            show_progress=True,
        )
        btn_ask.click(
            do_ask,
            inputs=[question, chat_state, top_k, state_agent, session_state],
            outputs=ask_outputs,
            show_progress=True,
        )
        question.submit(
            do_ask,
            inputs=[question, chat_state, top_k, state_agent, session_state],
            outputs=ask_outputs,
            show_progress=True,
        )
        btn_clear.click(
            do_clear,
            inputs=[state_agent, session_state],
            outputs=ask_outputs,
        )

        btn_new_session.click(
            do_new_session,
            inputs=[state_agent],
            outputs=ask_outputs,
        )
        session_select.change(
            do_load_session,
            inputs=[session_select, state_agent],
            outputs=ask_outputs,
        )
        btn_delete_session.click(
            do_delete_session,
            inputs=[session_select, state_agent],
            outputs=ask_outputs,
        )
        btn_update.click(
            do_update,
            inputs=[kb_target, upload, state_agent],
            outputs=[state_agent, status],
            show_progress=True,
        )

        demo.load(do_refresh_sessions, inputs=[], outputs=[session_select, session_state])
    demo.queue(default_concurrency_limit=8)
    demo.launch(theme=gr.themes.Base(primary_hue="emerald", neutral_hue="gray"), css=css)
