from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Dict, List, Sequence, Tuple


@dataclass
class AgentDraft:
    agent_name: str
    answer: str
    citations_md: str
    evidence_count: int
    evidence_kbs: Sequence[str]
    evidence_sources: Sequence[str]
    structured: Dict[str, object]


@dataclass
class ArbiterResult:
    selected_agent: str
    final_answer: str
    reason: str
    evidence_score_a: int
    evidence_score_b: int
    conflict_detected: bool
    conflict_report: str


def _safe_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def _has_all(text: str, parts: Sequence[str]) -> bool:
    t = text or ""
    return all(p in t for p in parts)


def _contains_any(text: str, parts: Sequence[str]) -> bool:
    t = text or ""
    return any(p in t for p in parts)


def _pick_variant(query: str, variants: Sequence[str]) -> str:
    if not variants:
        return ""
    q = (query or "").encode("utf-8", errors="ignore")
    h = hashlib.md5(q).digest()
    idx = int.from_bytes(h[:2], "big") % len(variants)
    return str(variants[idx])


def _is_sleep_related(query: str, analysis: Dict[str, object]) -> bool:
    q = _safe_text(query)
    if not q:
        return False
    intent = str((analysis or {}).get("intent") or "")
    emotion = str((analysis or {}).get("emotion") or "")
    if intent == "睡眠问题" or emotion == "睡眠问题":
        return True
    sleep_signals = [
        "失眠",
        "睡不着",
        "睡不好",
        "入睡",
        "上床",
        "夜醒",
        "早醒",
        "半夜醒",
        "多梦",
        "睡眠",
        "睡前",
        "睡觉",
    ]
    return any(s in q for s in sleep_signals)


def _is_campus_query(query: str) -> bool:
    q = _safe_text(query)
    if not q:
        return False
    triggers = [
        "深圳技术大学",
        "深技大",
        "学校",
        "校内",
        "心理咨询",
        "心理中心",
        "心理辅导",
        "校医院",
        "辅导员",
        "预约",
        "挂号",
        "地点",
        "地址",
        "时间",
        "开放时间",
        "工作时间",
        "联系方式",
        "电话",
        "邮箱",
        "流程",
        "怎么联系",
        "怎么预约",
    ]
    return any(t in q for t in triggers)


def _validate_final_answer(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return False
    forbidden = [
        "候选A",
        "候选B",
        "证据充分性",
        "评分",
        "结合候选",
        "冲突检测",
        "依据来源",
        ".txt",
        ".md",
        ".json",
    ]
    if _contains_any(t, ["【", "】"]):
        return False
    if _contains_any(t, forbidden):
        return False
    return True


def _risk_label(analysis: Dict[str, object]) -> str:
    lvl = int(analysis.get("risk_level") or 0) if isinstance(analysis, dict) else 0
    if lvl >= 4:
        return "偏高"
    if lvl >= 3:
        return "中等"
    return "偏低"


def _pick_lines(text: str, *, max_lines: int) -> List[str]:
    out: List[str] = []
    for raw in (text or "").splitlines():
        s = raw.strip()
        if not s:
            continue
        if len(s) > 200:
            s = s[:200] + "…"
        out.append(s)
        if len(out) >= max_lines:
            break
    return out


def _fallback_answer(*, query: str, analysis: Dict[str, object], draft_a: AgentDraft, draft_b: AgentDraft) -> str:
    emotion = str(analysis.get("emotion") or "").strip()
    intent = str(analysis.get("intent") or "").strip()
    risk = _risk_label(analysis)
    campus_q = _is_campus_query(query)
    sleep_q = _is_sleep_related(query, analysis)

    openers = [
        "我能理解这段时间你可能挺不好受的。",
        "你现在这样扛着确实不容易。",
        "我听懂你的担心了，这种状态很消耗。",
        "这种感觉放在任何人身上都挺折磨的。",
    ]
    opener = _pick_variant(query, openers)
    topic = intent or "困扰"
    if emotion:
        topic = f"**{emotion}**相关的困扰"
    empathy = f"{opener} 你提到的核心是{topic}。"

    why_1 = "当紧张、压力或情绪波动让身体处于**高唤醒**状态时，短时的**呼吸/放松训练**往往能帮助唤醒回落，让你更容易重新获得一点掌控感。"
    why_2_sleep = "如果你在床上越躺越清醒，使用**20分钟规则**能减少“床=清醒”的绑定，把床重新关联到入睡。"
    why_2_energy = "当你白天**疲惫/没动力**时，把任务缩到“能开始的大小”，比逼自己硬撑更容易恢复行动感，也更不容易陷入自责循环。"

    step_1 = "今晚可以先做一个**短时放松**：3分钟4-6呼吸（吸4拍、呼6拍，做3轮）+ 2分钟肩颈/身体扫描。你可以用“焦虑0–10分”做验收，目标是睡前比开始时下降**至少2分**。"
    step_2_sleep = "如果上床后**20分钟仍清醒**：先起身离开床，做低刺激活动（轻音乐/翻纸质书），困意回来再上床。验收标准用“入睡时间”和“夜醒次数”观察，先以**3–7天**为一个周期。"
    step_2_energy = (
        "给自己一个**10分钟最低行动量**：只选一件最小任务（例如“打开文档”“写3行”“收拾桌面一角”），"
        "计时10分钟就停。验收标准不是效率，而是“**我开始了**”。如果连续2–3天都能做到，再把时长增加到15–25分钟。"
    )

    campus_text = ""
    if campus_q:
        campus_variants = [
            "如果你是**深圳技术大学**的学生，我建议你优先用**校内支持资源**：学校心理咨询中心、学院辅导员、校医院。若有持续干呕等身体症状，先做一次医学排查会更稳妥。",
            "如果你在**深圳技术大学**，可以先从**校内渠道**获得支持：心理咨询中心/辅导员/校医院。身体症状明显时，优先把躯体原因排除掉，会更安心。",
        ]
        campus_text = _pick_variant(query, campus_variants)

    parts: List[str] = []
    parts.append(f"{empathy} 你现在最关心的是：**{query.strip()}**。")
    if campus_text:
        parts.append(campus_text)
    risk_variants = [
        f"从安全边界上看，我会把风险放在**{risk}**（仅基于你的描述，非诊断）。如果症状明显恶化、开始影响基本生活功能，或你感觉难以自控，优先考虑线下专业帮助。",
        f"就风险边界来说，我倾向于评估为**{risk}**（仅基于你的描述，非诊断）。若你开始明显影响学习/睡眠/进食，或出现强烈失控感，建议尽快寻求线下支持。",
    ]
    parts.append(_pick_variant(query, risk_variants))
    bridge_variants = [
        f"我先把逻辑说清楚：{why_1} 同时，{why_2_sleep if sleep_q else why_2_energy}",
        f"为什么会建议这两条？一方面，{why_1} 另一方面，{why_2_sleep if sleep_q else why_2_energy}",
    ]
    parts.append(_pick_variant(query, bridge_variants))
    parts.append(step_1)
    parts.append(step_2_sleep if sleep_q else step_2_energy)
    parts.append(
        "如果出现这些情况，建议**升级处理**：连续3–7天仍无改善且白天功能明显下降；身体不适明显加重（持续呕吐、胸闷心悸、惊恐发作等）或睡眠显著恶化；出现自伤/轻生想法或强烈绝望感。"
    )
    return "\n\n".join(parts)


def _is_action_query(query: str) -> bool:
    q = _safe_text(query)
    if not q:
        return False
    keywords = ["怎么", "如何", "步骤", "方案", "计划", "建议", "做法", "流程", "操作", "实现", "解决"]
    return any(k in q for k in keywords)


def _is_physical_signal(query: str) -> bool:
    q = _safe_text(query)
    if not q:
        return False
    signals = [
        "干呕",
        "呕吐",
        "恶心",
        "腹痛",
        "头痛",
        "发烧",
        "胸闷",
        "心悸",
        "呼吸困难",
        "眩晕",
        "疼",
        "痛",
    ]
    return any(s in q for s in signals)


def _needs_followup(query: str, analysis: Dict[str, object], points: Dict[str, object], draft_a: AgentDraft, draft_b: AgentDraft) -> bool:
    q = _safe_text(query)
    if not q:
        return False
    try:
        lvl = int((analysis or {}).get("risk_level") or 0)
    except Exception:
        lvl = 0
    if lvl >= 4:
        return False
    top2 = points.get("top2_actions") or []
    if not isinstance(top2, list):
        top2 = []
    if len(top2) < 1:
        return True
    if (draft_a.evidence_count + draft_b.evidence_count) <= 1:
        return True
    vague_triggers = ["怎么办", "怎么做", "怎么缓解", "怎么改善", "帮帮我", "我很难受", "我不舒服"]
    if any(t in q for t in vague_triggers) and len(q) <= 18:
        return True
    return False


def _build_followup_questions(query: str, analysis: Dict[str, object], points: Dict[str, object]) -> List[str]:
    q = _safe_text(query)
    intent = str((analysis or {}).get("intent") or "").strip()
    emotion = str((analysis or {}).get("emotion") or "").strip()
    sleep_q = _is_sleep_related(q, analysis)
    campus_q = _is_campus_query(q)
    physical_q = _is_physical_signal(q)

    openers = [
        "如果你愿意，我想先确认两点，这样我给你的建议会更贴合你。",
        "为了把建议做得更贴近你的情况，我想先问你两个小问题。",
        "我先问两个关键点，确认后我会把方案写得更具体、更好执行。",
    ]
    preface = _pick_variant(q, openers)

    qs: List[str] = []
    if campus_q:
        qs.append("你更希望我帮你搞清楚的是**预约心理咨询的流程**，还是**联系方式/地点/开放时间**这些信息？")
        qs.append("你现在的情况更偏**想找人聊一聊**，还是已经影响到**学习/作息/进食**需要更强的支持？")
    elif sleep_q:
        qs.append("你的睡眠问题大概持续了**多久**？更像是**入睡困难**、**夜醒**还是**早醒**？")
        qs.append("白天的影响主要在**精力**、**情绪**还是**注意力/效率**上？你更想先改善哪一个？")
    else:
        if physical_q:
            qs.append("你说的身体不适（比如**干呕/恶心**）通常在什么情况下更明显：吃饭前后、紧张时、早晨起床后，还是随机出现？")
        qs.append("这种状态大概持续了**多久**？最近是否已经影响到**学习/工作/进食/睡眠**中的某一项？")
        if not qs or len(qs) < 2:
            topic = intent or (emotion or "困扰")
            qs.append(f"你更希望我现在优先帮你的是：把{topic}背后的**可能原因**讲清楚，还是给一个**今天就能执行的两步方案**？")

    out = [preface]
    for s in qs[:2]:
        out.append(s)
    return out


def _compute_evidence_score(draft: AgentDraft) -> int:
    score = 0
    score += min(4, (int(draft.evidence_count) + 1) // 2)
    score += min(3, len(set([s for s in draft.evidence_sources if s])))
    score += min(2, len(set([k for k in draft.evidence_kbs if k])))
    if len(_safe_text(draft.answer)) >= 80:
        score += 1
    return min(score, 10)


def build_conflict_report(ans_a: str, ans_b: str) -> Tuple[bool, str]:
    a = _safe_text(ans_a)
    b = _safe_text(ans_b)
    if not a or not b:
        return False, "至少一条候选答案为空，无法判断冲突。"

    hard_conflict_pairs: List[Tuple[str, str]] = [
        ("可以", "不可以"),
        ("建议", "不建议"),
        ("需要", "不需要"),
        ("必须", "不必"),
        ("应当", "不应"),
        ("是", "不是"),
    ]
    triggers: List[str] = []
    for pos, neg in hard_conflict_pairs:
        a_has = pos in a and neg in b
        b_has = pos in b and neg in a
        if a_has or b_has:
            triggers.append(f"{pos}/{neg}")

    if triggers:
        msg = "检测到潜在冲突关键词对：" + "，".join(triggers)
        return True, msg

    # 若两条答案核心语句重复度很低，也提示可能存在立场偏差（弱冲突）
    a_words = set([w for w in a if w.strip()])
    b_words = set([w for w in b if w.strip()])
    overlap = len(a_words.intersection(b_words))
    base = max(1, min(len(a_words), len(b_words)))
    ratio = overlap / base
    if ratio < 0.2:
        return True, f"两条答案词面重合度较低（{ratio:.2f}），可能存在立场差异。"

    return False, "未检测到明显冲突。"


def arbitrate_answers(query: str, draft_a: AgentDraft, draft_b: AgentDraft, *, analysis: Dict[str, object], llm=None) -> ArbiterResult:
    evidence_score_a = _compute_evidence_score(draft_a)
    evidence_score_b = _compute_evidence_score(draft_b)
    conflict_detected, conflict_report = build_conflict_report(draft_a.answer, draft_b.answer)

    tie_break = ""
    selected = draft_a
    if evidence_score_b > evidence_score_a:
        selected = draft_b
    elif evidence_score_b == evidence_score_a:
        if _is_action_query(query):
            selected = draft_b
            tie_break = "（同分：问题更偏向步骤/执行，优先B）"
        else:
            tie_break = "（同分：默认优先A）"

    reason = (
        f"证据充分性评分：{draft_a.agent_name}={evidence_score_a}，{draft_b.agent_name}={evidence_score_b}。{tie_break}"
        + (" 检测到冲突，优先选择更保守且证据更充分的一侧。" if conflict_detected else " 未检测到明显冲突。")
    )

    # 若提供 llm，仲裁智能体输出“解释层 + 行动层”融合答案；失败则回退默认候选答案
    def extract_points() -> Dict[str, object]:
        a = dict(draft_a.structured or {})
        b = dict(draft_b.structured or {})
        main_issue = str(a.get("main_issue") or "").strip()
        if not main_issue:
            main_issue = str(analysis.get("intent") or "").strip() or str(analysis.get("emotion") or "").strip()
        risk = str(a.get("risk_level") or "").strip()
        if not risk:
            lvl = int(analysis.get("risk_level") or 0)
            risk = "偏高" if lvl >= 4 else ("中等" if lvl >= 3 else "偏低")

        watch = a.get("watch_metrics") or []
        seek = a.get("seek_help_when") or []
        if not isinstance(watch, list):
            watch = []
        if not isinstance(seek, list):
            seek = []
        watch = [str(x).strip() for x in watch if str(x).strip()][:3]
        seek = [str(x).strip() for x in seek if str(x).strip()][:3]

        top2 = b.get("top2_actions") or []
        opt = b.get("optional_actions") or []
        if not isinstance(top2, list):
            top2 = []
        if not isinstance(opt, list):
            opt = []

        def norm_actions(xs) -> List[Dict[str, str]]:
            out: List[Dict[str, str]] = []
            for it in xs:
                if not isinstance(it, dict):
                    continue
                title = str(it.get("title") or "").strip()
                if not title:
                    continue
                out.append(
                    {
                        "title": title,
                        "duration": str(it.get("duration") or "").strip(),
                        "trigger": str(it.get("trigger") or "").strip(),
                        "metric": str(it.get("metric") or "").strip(),
                    }
                )
            return out

        top2_actions = norm_actions(top2)[:2]
        optional_actions = norm_actions(opt)[:2]
        return {
            "main_issue": main_issue,
            "risk_level": risk,
            "watch_metrics": watch,
            "seek_help_when": seek,
            "top2_actions": top2_actions,
            "optional_actions": optional_actions,
        }

    points = extract_points()
    final_answer = selected.answer
    if llm is not None:
        campus_hint = ""
        if _is_campus_query(query):
            campus_hint = (
                "\n额外硬规则（校园优先）：如果用户的问题涉及校内资源/求助/预约/地点/联系方式/流程，"
                "请优先给出与**深圳技术大学**相关的可操作信息（只允许使用你在输入材料中看到的内容）。"
                "如果输入材料里没有具体联系方式或地点，请不要提“未检索到/没有找到/缺少资料”，"
                "改为用自然语言建议用户优先通过学校官方渠道（学校官网、心理咨询中心公告、学院辅导员、校医院）确认最新信息。"
            )
        system = "\n".join(
            [
                "你是心理支持场景的仲裁智能体（非医疗用途）。任务：把解释与行动整合成一份更像人说话的最终回复。",
                "硬约束：",
                "1) 不得出现“候选A/B”“评分”“冲突检测”等报告语言。",
                "2) 不要输出任何【】标题；改为多段短文，每段不超过4行。",
                "3) 用加粗强调关键字或关键句（少量即可）。",
                "4) 避免固定口头禅与套路句：不要反复使用相同开头与相同句式。",
                "5) 只要用户没有明确提到睡眠/失眠/入睡/夜醒，不要出现“上床/入睡/20分钟规则/夜醒次数”等睡眠策略内容。",
                "6) 不要输出任何依据来源/文件名/后缀（例如 .txt/.md/.json）。",
                campus_hint.strip(),
                "输出建议（尽量满足）：先共情复述，再给两条最关键行动建议，并说明为什么适合当前情况，最后给升级处理边界。",
            ]
        ).strip()
        user = (
            f"【用户问题】\n{query}\n\n"
            f"【分析结果】\n{analysis}\n\n"
            "【结构化要点（请优先使用这些，不要自己编造）】\n"
            f"{points}\n\n"
            "【补充上下文（如需语气与细节，可参考）】\n"
            f"解释信息：{draft_a.answer}\n\n"
            f"行动方案：{draft_b.answer}\n"
        )
        try:
            resp = llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
            candidate = str(getattr(resp, "content", "")).strip()
            final_answer = candidate or selected.answer
        except Exception:
            final_answer = selected.answer

    if _needs_followup(query, analysis, points, draft_a, draft_b):
        follow_parts = _build_followup_questions(query, analysis, points)
        if follow_parts:
            final_answer = (final_answer.strip() + "\n\n" + "\n".join([p.strip() for p in follow_parts if p.strip()])).strip()

    return ArbiterResult(
        selected_agent=selected.agent_name,
        final_answer=final_answer,
        reason=reason,
        evidence_score_a=evidence_score_a,
        evidence_score_b=evidence_score_b,
        conflict_detected=conflict_detected,
        conflict_report=conflict_report,
    )
