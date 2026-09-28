# ============================================================
# 追问挂起期间的离题判定器（旁路分流第 3 步）
# 设计：docs/design-side-assistant-bypass.md §3.2
#
# 两层结构，控成本也控误判：
#   1. 正则门（确定性、零成本）：典型追问回答不命中 → 不产生任何模型调用
#   2. 小模型裁决（仅对过门消息）：输出 {"intent":"answer"|"other"}，
#      含糊一律 answer（宁可不旁路）
#
# 误判两个方向都安全：
#   误判为旁路 → 多一轮旁路回答 + 追问回放，状态无损，用户重答即可；
#   误判为回答 → 即旁路上线前的行为（离题文本进分诊，白烧一轮）。
#
# 影子模式（SIDE_ASSISTANT_ENABLED=false）：判定照常运行只记日志，
# 不改路由 —— 用真实流量标定误判率，达标后再翻开关
# （Rasa interactive learning 的对应物）。
# ============================================================

import json
import re

from src.core.config import get_settings
from src.core.logger import logger

INTENT_ANSWER = "answer"   # 是挂起追问的回答 → 走原路径（resume 进图）
INTENT_OTHER = "other"     # 与诊断无关的其他请求 → 旁路回答 + 回放追问

# 正则门：命中任一模式才继续到小模型裁决。词类来自 bypass 设计 §3.2 ——
# 「另有所指」的请求词（查数据/查文档/问概念/操作问题单）；
# 典型追问回答（"没有其他现象""是的每次都这样""充电时跳闸"）不命中。
_OFFTOPIC_GATE = re.compile(
    "|".join([
        r"统计|报表|数据|指标|趋势",
        r"规范|标准|参数|文档|知识库|资料",
        r"报告|解读",
        r"影响|变更|单号|建单|结案|工单",
        r"帮我查|查一下|查询|多少",
        r"是什么|什么意思|怎么查",
    ])
)

_JUDGE_SYSTEM = (
    "你是对话意图分类器。用户正在一次车辆故障诊断中，系统刚问了一个诊断追问，"
    "用户发来一条新消息。判断这条消息是「追问的回答」还是「与诊断无关的其他请求」。"
    "规则：消息在回答追问（含\"没有\"\"不清楚\"\"好像是\"\"每次都这样\"这类模糊回答）"
    "判 answer；消息在问别的事（查数据/查文档/问概念/操作问题单等）判 other；"
    "两者都像或说不清判 answer（宁可不旁路）。"
    '只输出 JSON：{"intent":"answer"} 或 {"intent":"other"}，不要输出其他内容。'
)

_JUDGE_USER_TMPL = "诊断追问：{question}\n用户消息：{message}"


def hits_offtopic_regex(message: str) -> bool:
    """正则门：消息是否命中「另有所指」模式（零成本，无模型调用）。"""
    return bool(_OFFTOPIC_GATE.search(message))


def _parse_verdict(raw: str) -> str:
    """防御式解析小模型输出；任何不合规一律 answer。"""
    text = (raw or "").strip()
    # 容错 markdown 代码围栏（小模型偶尔包一层 ```json）
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text)
    try:
        data = json.loads(text)
    except Exception:
        return INTENT_ANSWER
    intent = data.get("intent") if isinstance(data, dict) else None
    return INTENT_OTHER if intent == INTENT_OTHER else INTENT_ANSWER


async def _judge_with_llm(question: str, message: str) -> str:
    """小模型裁决。★ 测试通过 monkeypatch 本函数注入确定性结果。"""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    llm = ChatOpenAI(
        model=settings.SIDE_ASSISTANT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0,
        timeout=5,       # 判定在用户请求路径上：宁可误判为 answer 也不能吊着
        max_tokens=16,
    )
    resp = await llm.ainvoke([
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user", "content": _JUDGE_USER_TMPL.format(
            question=question[:500], message=message[:500],
        )},
    ])
    return _parse_verdict(resp.content)


async def classify_offtopic(question: str, message: str) -> str:
    """挂起追问 + 用户消息 → INTENT_ANSWER / INTENT_OTHER。

    契约（bypass 设计 §3.2 / §5）：
      - 正则门未命中 → 直接 answer，不产生模型调用；
      - 过门后小模型裁决，解析失败/超时/任何异常 → 一律 answer；
      - 影子模式由调用方（chat 层）按 SIDE_ASSISTANT_ENABLED 决定是否改路由，
        本函数只负责判定并留痕 —— 影子期日志就是误判率统计的数据源。
    """
    if not hits_offtopic_regex(message):
        return INTENT_ANSWER
    try:
        verdict = await _judge_with_llm(question, message)
    except Exception as e:
        # 裁决器挂了按 answer 处理 = 现状行为，最坏多烧一轮分诊，无损状态
        logger.warning(f"[SIDE] 裁决器异常，按 answer 处理: {type(e).__name__} {e}")
        verdict = INTENT_ANSWER
    enabled = get_settings().SIDE_ASSISTANT_ENABLED
    logger.info(
        f"[SIDE] hit_regex=True enabled={enabled} verdict={verdict} "
        f"q={question[:60]!r} msg={message[:60]!r}"
    )
    return verdict
