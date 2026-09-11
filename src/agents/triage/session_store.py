# ============================================================
# 分诊会话存储（B2 重构）
#
# 职责边界：
#   - 控制面（某条消息归 Supervisor 还是分诊处理）由 Supervisor
#     checkpointer 的 interrupt 快照决定，不在本模块；
#   - 本模块只管分诊工作数据（TriageState + 待答追问）的存取生命周期，
#     是三个入口（chat 工具 / /api/v1/triage REST）共用的唯一 key 规范出处。
#
# ★ 为什么进度要放外部存储而不是图 checkpoint：
#   LangGraph 的 checkpoint 只在节点完成时落盘，interrupt 暂停期间
#   节点内的局部进度不会持久化；而 resume 会让节点从头重放。
#   call_triage_agent 在每次 interrupt 前把进度写进这里，
#   重放时据此"快进"到断点，避免已完成轮次的 LLM 调用重跑。
# ============================================================

import json
from dataclasses import dataclass

from src.agents.triage.state import TriageState
from src.core.logger import logger

# ★ 必须覆盖"用户隔了很久才回答追问"的场景：挂起的 interrupt 存活在
# checkpointer 里没有 TTL，若 store 先过期，resume 时工具会误判为首轮，
# 重跑诊断且 interrupt 位置错位。24h 内未回复视为放弃（见 worker_tools 的
# 兜底：store 过期时按全新诊断重启，可自愈但会丢上下文）。
TRIAGE_STATE_TTL = 86400


# ── 退出指令（B1 逃生口）──
# 精确匹配短指令，避免把长描述里恰好含"取消"二字的正常回答误判为退出
TRIAGE_EXIT_COMMANDS = {
    "退出诊断", "退出分诊", "结束诊断", "结束分诊", "停止诊断",
    "取消诊断", "取消分诊", "取消", "退出", "不诊了", "重新诊断",
}


def is_triage_exit(message: str) -> bool:
    """判断是否为退出分诊的显式指令（去空白和尾部标点后精确匹配）。"""
    msg = message.strip().strip("。，！？!?,.~")
    return msg in TRIAGE_EXIT_COMMANDS


@dataclass
class TriageProgress:
    """一次暂停前落盘的分诊进度。

    state.round 同时就是重放时需要空转消耗的 interrupt 数量：
    首轮工作发生在任何 interrupt 之前，此后每轮工作恰好由一个 interrupt 门控。
    """

    state: TriageState        # 最后一轮完成后的分诊状态
    reply: str                # 该轮产出的追问原文（恢复时直接展示）


class TriageSessionStore:
    """TriageState 的 Redis 存取。key 规范统一为 triage_state:{thread_id}。"""

    def __init__(self, redis):
        self.redis = redis

    def _key(self, thread_id: str) -> str:
        return f"triage_state:{thread_id}"

    async def save(self, thread_id: str, state: TriageState, reply: str) -> None:
        payload = json.dumps(
            {"state": state.model_dump_json(), "reply": reply},
            ensure_ascii=False,
        )
        await self.redis.set(self._key(thread_id), payload, ex=TRIAGE_STATE_TTL)

    async def load(self, thread_id: str) -> TriageProgress | None:
        raw = await self.redis.get(self._key(thread_id))
        if not raw:
            return None
        data = json.loads(raw)
        return TriageProgress(
            state=TriageState.model_validate_json(data["state"]),
            reply=data.get("reply", ""),
        )

    async def clear(self, thread_id: str) -> None:
        await self.redis.delete(self._key(thread_id))
        logger.info(f"[TRIAGE-STORE] cleared thread={thread_id}")
