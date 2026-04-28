from __future__ import annotations

from dataclasses import dataclass
import base64
import binascii
import re
import string
from typing import Dict, Iterable, List, Optional, Sequence


CRISIS_KEYWORDS = [
    "不想活了",
    "想死",
    "结束生命",
    "自杀",
    "轻生",
    "自伤",
    "伤害自己",
    "想伤害自己",
    "割腕",
    "割手",
    "划伤",
    "划自己",
    "用刀伤害自己",
    "烧伤",
    "烫伤",
    "撞墙",
    "撞头",
    "咬自己",
    "打自己",
    "流血",
    "出血",
    "见血",
    "我不配活着",
    "我该死",
    "想惩罚自己",
    "自我惩罚",
    "只有痛才能缓解",
    "用疼痛麻痹自己",
    "用痛麻痹自己",
    "撑不下去了",
    "想消失",
    "活着没意义",
    "活着没有意义",
    "好累不想继续了",
    "不想继续了",
    "想逃离一切",
    "刀",
    "小刀",
    "刀片",
    "剪刀",
    "玻璃碎片",
    "锋利物品",
]

_CRISIS_IDEATION = [
    "不想活了",
    "想死",
    "结束生命",
    "自杀",
    "轻生",
    "撑不下去了",
    "想消失",
    "活着没意义",
    "活着没有意义",
    "好累不想继续了",
    "不想继续了",
    "想逃离一切",
    "生无可恋",
    "绝望",
]

_CRISIS_SELF_HARM = [
    "自伤",
    "伤害自己",
    "想伤害自己",
    "用刀伤害自己",
    "割腕",
    "割手",
    "划伤",
    "划自己",
    "烧伤",
    "烫伤",
    "撞墙",
    "撞头",
    "咬自己",
    "打自己",
    "流血",
    "出血",
    "见血",
    "我不配活着",
    "我该死",
    "想惩罚自己",
    "自我惩罚",
    "只有痛才能缓解",
    "用疼痛麻痹自己",
    "用痛麻痹自己",
]

_CRISIS_TOOLS = [
    "刀",
    "小刀",
    "刀片",
    "剪刀",
    "玻璃碎片",
    "锋利物品",
]

_THIRD_PARTY_MARKERS = [
    "朋友",
    "同学",
    "室友",
    "家人",
    "父母",
    "妈妈",
    "爸爸",
    "哥哥",
    "姐姐",
    "弟弟",
    "妹妹",
    "男朋友",
    "女朋友",
    "对象",
    "同事",
    "同学",
    "老师",
    "学生",
    "孩子",
    "他",
    "她",
    "TA",
    "他们",
    "她们",
    "别人",
    "有人",
    "网友",
]

_HELPING_VERBS = [
    "安慰",
    "劝",
    "劝解",
    "开导",
    "帮助",
    "陪",
    "怎么办",
    "如何",
    "怎么",
    "该怎么",
    "我该",
    "我应该",
]

_SELF_EXPLICIT_RE = re.compile(
    r"(我自己|本人|我现在|我真的|我一直|我可能).{0,10}"
    r"(不想活了|想死|结束生命|自杀|轻生|自伤|伤害自己|想伤害自己|割腕|划自己|撞墙|撞头|咬自己|打自己)"
)

_SELF_STRONG_RE = re.compile(
    r"(我(?!朋友|同学|室友|家人|父母|妈妈|爸爸|对象|同事|老师|学生|孩子|他|她|TA|他们|她们|别人|有人|网友))"
    r".{0,8}(不想活了|想死|结束生命|自杀|轻生)"
    r"|"
    r"(我(?!朋友|同学|室友|家人|父母|妈妈|爸爸|对象|同事|老师|学生|孩子|他|她|TA|他们|她们|别人|有人|网友))"
    r".{0,6}(想|要|准备|打算|可能会|会).{0,6}(自杀|轻生|自伤|伤害自己|割腕|划自己|撞墙|撞头|咬自己|打自己)"
)

_THIRD_PARTY_RE = re.compile(
    r"(朋友|同学|室友|家人|父母|妈妈|爸爸|对象|同事|老师|学生|孩子|他|她|TA|别人|有人|网友).{0,10}"
    r"(不想活了|想死|结束生命|自杀|轻生|自伤|伤害自己|割腕|划伤|流血|出血|见血)"
)

_HOW_TO_HELP_RE = re.compile(
    r"(怎么|如何|该怎么|我该|我应该).{0,10}(安慰|劝|开导|帮助|陪).{0,20}(自杀|想死|轻生|不想活了|结束生命|自伤)"
)


@dataclass
class AnalysisResult:
    emotion: str
    intent: str
    risk_level: int
    crisis: bool
    blocked: bool = False
    block_category: str = ""
    block_reason: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "emotion": self.emotion,
            "intent": self.intent,
            "risk_level": self.risk_level,
            "crisis": self.crisis,
            "blocked": self.blocked,
            "block_category": self.block_category,
            "block_reason": self.block_reason,
        }


def _contains_any(text: str, words: List[str]) -> bool:
    return any(w in text for w in words)


def _is_third_party_context(t: str) -> bool:
    if _HOW_TO_HELP_RE.search(t):
        return True
    if _THIRD_PARTY_RE.search(t):
        return True
    if any(m in t for m in _THIRD_PARTY_MARKERS) and any(k in t for k in _HELPING_VERBS) and _contains_any(t, CRISIS_KEYWORDS):
        return True
    return False


def _is_self_crisis(t: str) -> bool:
    if not _contains_any(t, CRISIS_KEYWORDS):
        return False
    if _SELF_STRONG_RE.search(t):
        return True
    if _is_third_party_context(t):
        return False
    if _SELF_EXPLICIT_RE.search(t):
        return True
    has_ideation = any(k in t for k in _CRISIS_IDEATION)
    has_harm = any(k in t for k in _CRISIS_SELF_HARM)
    has_tool = any(k in t for k in _CRISIS_TOOLS)
    if has_tool and not has_harm and not has_ideation:
        return False
    if has_ideation or has_harm:
        return True
    return False


_INJECTION_RE = re.compile(
    r"(忽略|无视|跳过).{0,12}(之前|上面|先前).{0,12}(指令|规则|要求|提示)|"
    r"(现在|从现在起).{0,12}(你是|扮演|假装)|"
    r"(泄露|透露|显示|打印|输出|发出来|发给我|给我|提供|说出|贴出|公布).{0,16}"
    r"(系统提示|系统提示词|系统指令|隐藏指令|开发者消息|prompt|system prompt|developer message|提示词)|"
    r"(系统提示|系统提示词|系统指令|隐藏指令|开发者消息|prompt|system prompt|developer message|提示词).{0,16}"
    r"(发出来|发给我|给我|提供|说出|贴出|公布|告诉我|给我看|展示)|"
    r"(jailbreak|越狱|DAN|绕过规则|break the rules|ignore previous instructions)",
    flags=re.IGNORECASE,
)

_TRANSLATION_BYPASS_RE = re.compile(
    r"(翻译|translate|译成|翻成).{0,30}(忽略之前|ignore previous|system prompt|开发者消息|developer message|提示注入|jailbreak|越狱)",
    flags=re.IGNORECASE,
)

_MULTISTEP_PROBE_RE = re.compile(
    r"(一步一步|逐步|step by step|下一步|继续|更具体|详细一点|展开说|细讲|给步骤|给流程|给教程)",
    flags=re.IGNORECASE,
)

_SELF_HARM_EUPHEMISM_RE = re.compile(
    r"(无痛|不痛苦|最轻松|最快|快速|最简单|最有效|100%|必死).{0,12}"
    r"(自杀|轻生|结束生命|自尽|结束自己|离开这个世界|不想活|想死)",
    flags=re.IGNORECASE,
)

_SELF_HARM_METHOD_RE = re.compile(
    r"(怎么|如何|方法|步骤|教程|指南|方案|手法).{0,12}"
    r"(自杀|轻生|结束生命|自尽|结束自己|离开这个世界)",
    flags=re.IGNORECASE,
)

_BASE64_HINT_RE = re.compile(r"(base64|b64|解码|decode)", flags=re.IGNORECASE)


def _is_mostly_printable(s: str) -> bool:
    if not s:
        return False
    printable = set(string.printable) | set("，。；：？！“”‘’（）【】《》—·、\n\t ")
    good = sum(1 for ch in s if ch in printable)
    return good / max(1, len(s)) >= 0.85


def _extract_base64_candidates(text: str) -> List[str]:
    candidates: List[str] = []
    for m in re.finditer(r"([A-Za-z0-9+/=_-]{16,})", text):
        s = m.group(1).strip()
        if len(s) > 4096:
            continue
        if any(c in s for c in [" ", "\n", "\t", "，", "。"]):
            continue
        candidates.append(s)
    return candidates


def _try_b64decode(s: str) -> Optional[str]:
    raw = s.strip()
    raw = raw.replace("-", "+").replace("_", "/")
    pad = (-len(raw)) % 4
    if pad:
        raw = raw + ("=" * pad)
    try:
        out = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError):
        return None
    try:
        txt = out.decode("utf-8", errors="ignore").strip()
    except Exception:
        return None
    if not txt or len(txt) < 6:
        return None
    if not _is_mostly_printable(txt):
        return None
    return txt[:800]


def assess_safety(user_text: str, *, history_user_texts: Sequence[str] = ()) -> AnalysisResult:
    t = (user_text or "").strip()
    analysis = analyze_user_input(t)
    if analysis.crisis:
        return analysis

    if t and _TRANSLATION_BYPASS_RE.search(t):
        analysis.blocked = True
        analysis.block_category = "translation_bypass"
        analysis.block_reason = "检测到翻译型提示注入文本。"
        return analysis

    third_party = _is_third_party_context(t)
    if not third_party and (_SELF_HARM_METHOD_RE.search(t) or _SELF_HARM_EUPHEMISM_RE.search(t)):
        analysis.crisis = True
        analysis.risk_level = max(analysis.risk_level, 5)
        analysis.intent = "危机干预"
        analysis.emotion = "高危"
        return analysis

    decoded_hits: List[str] = []
    if t and (_BASE64_HINT_RE.search(t) or _extract_base64_candidates(t)):
        for c in _extract_base64_candidates(t):
            decoded = _try_b64decode(c)
            if decoded:
                decoded_hits.append(decoded)
        probe_text = "\n".join([t] + decoded_hits)
        if _INJECTION_RE.search(probe_text) or _TRANSLATION_BYPASS_RE.search(probe_text):
            analysis.blocked = True
            analysis.block_category = "base64_bypass"
            analysis.block_reason = "检测到疑似编码后的提示注入/越狱内容。"
            return analysis
        if not third_party and (_SELF_HARM_METHOD_RE.search(probe_text) or _SELF_HARM_EUPHEMISM_RE.search(probe_text)):
            analysis.crisis = True
            analysis.risk_level = max(analysis.risk_level, 5)
            analysis.intent = "危机干预"
            analysis.emotion = "高危"
            return analysis

    if t and _INJECTION_RE.search(t):
        analysis.blocked = True
        analysis.block_category = "prompt_injection"
        analysis.block_reason = "检测到提示注入/越狱/索要系统提示的企图。"
        return analysis

    recent = [x.strip() for x in (history_user_texts or []) if str(x).strip()][-8:]
    if recent and _MULTISTEP_PROBE_RE.search(t):
        hist_blob = "\n".join(recent)
        if _INJECTION_RE.search(hist_blob) or _TRANSLATION_BYPASS_RE.search(hist_blob) or _SELF_HARM_METHOD_RE.search(hist_blob):
            analysis.blocked = True
            analysis.block_category = "multi_step_probe"
            analysis.block_reason = "检测到多步引导式索要攻击手法/系统提示。"
            return analysis

    return analysis


def analyze_user_input(text: str) -> AnalysisResult:
    """
    轻量可运行版本：规则分类（可替换为 LLM 分类）。
    覆盖：焦虑/自我否定/压力/情绪低落，并输出风险等级。
    """

    t = text.strip()
    if not t:
        return AnalysisResult("中性", "倾诉", 1, False)

    if _is_self_crisis(t):
        return AnalysisResult("高危绝望", "危机干预", 5, True)

    sleep_words = [
        "睡不着",
        "失眠",
        "入睡困难",
        "睡眠浅",
        "睡不好",
        "早醒",
        "半夜醒",
        "夜醒",
        "多梦",
        "睡眠质量",
        "越睡越累",
    ]
    anxiety_words = ["焦虑", "紧张", "心慌", "害怕", "恐惧", "担心", "忧虑", "坐立不安", "烦躁", "惊恐"]
    self_deny_words = ["我很差", "没用", "失败", "一无是处", "不配", "讨厌自己", "自卑", "自责", "内疚", "我不行"]
    stress_words = [
        "压力",
        "压力大",
        "扛不住",
        "崩溃",
        "工作好多",
        "学业",
        "考试",
        "deadline",
        "压得喘不过气",
        "喘不过气",
        "累到爆",
        "忙不过来",
        "撑不住",
    ]
    depression_words = ["抑郁", "绝望", "无望", "生无可恋", "对什么都没兴趣", "兴趣下降", "对一切都没意思"]
    low_mood_words = [
        "难过",
        "低落",
        "情绪低落",
        "没有动力",
        "开心不起来",
        "空虚",
        "提不起劲",
        "没精神",
        "疲惫",
        "很累",
        "倦怠",
        "不想做",
        "厌学",
        "厌工",
        "做什么都不想",
        "心情不好",
        "沮丧",
    ]

    has_sleep = _contains_any(t, sleep_words)
    has_anxiety = _contains_any(t, anxiety_words)
    has_stress = _contains_any(t, stress_words)
    has_depression = _contains_any(t, depression_words)
    has_low_mood = _contains_any(t, low_mood_words)

    if _contains_any(t, self_deny_words):
        return AnalysisResult("自我否定", "自我否定", 4, False)
    if has_depression:
        return AnalysisResult("抑郁", "抑郁", 4, False)
    if has_sleep and not (has_anxiety or has_stress or has_low_mood):
        return AnalysisResult("睡眠问题", "睡眠问题", 3, False)
    if has_anxiety and has_sleep:
        return AnalysisResult("焦虑", "睡眠问题", 3, False)
    if has_anxiety:
        return AnalysisResult("焦虑", "焦虑", 3, False)
    if has_stress:
        return AnalysisResult("压力", "压力过大", 3, False)
    if has_low_mood:
        return AnalysisResult("情绪低落", "情绪低落", 3, False)

    return AnalysisResult("中性", "倾诉", 2, False)
