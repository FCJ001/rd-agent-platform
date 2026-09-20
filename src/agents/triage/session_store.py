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
from src.core.config import get_settings
from src.core.logger import logger

# ── TTL：进度必须比挂起的 interrupt 活得久 ──
# checkpointer 里的挂起 interrupt 和这里的进度是同一段追问历史的两个副本。
# 进度先过期时，resume 重放会让工具读不到进度、把它误判成首轮重跑 ——
# 后果不是「丢上下文」那么轻：既白烧一整轮模型调用，又把 checkpointer 里
# 记录的历史回答当成本轮新输入重新处理一遍，追问序列与 round 从此错位
# （实测轨迹见 tests/test_session_store_ttl.py）。
#
# 因此取 settings.TRIAGE_STATE_TTL_SECONDS（启动期校验它 > checkpointer TTL，
# 默认 8 天 > 7 天），并在 load() 里读时续期 —— 与 checkpointer 的
# refresh_on_read 对齐：只要 interrupt 还活着（每次被读都续期），进度就一定还在。
# 真正的「放弃」发生在两侧同时过期时：那时 chat 看不到挂起的 interrupt，
# 消息按新对话处理，存量 key 不会被读到。
def _ttl_seconds() -> int:
    return get_settings().TRIAGE_STATE_TTL_SECONDS


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
    """TriageState 的 Redis 存取。key 规范统一为 triage_state:{thread_id}。

    两个 key，职责不同：
      triage_state:{thread_id}    诊断进度本身（丢了就是重开一轮，代价可接受）
      triage_pending:{thread_id}  「本会话有一段追问在等回答」的标记
    ★ 为什么要第二个 key：进度和 checkpointer 里的挂起 interrupt 是同一段
      历史的两个副本，进度有可能先没（Redis 淘汰、清库、TTL 配错）。只有
      进度时无从区分「全新会话」和「resume 撞上进度过期」—— 后者按首轮重跑
      会把历史回答重放一遍。标记就是那个判据（见 worker_tools 的放弃分支）。
    """

    def __init__(self, redis):
        self.redis = redis

    def _key(self, thread_id: str) -> str:
        return f"triage_state:{thread_id}"

    def _pending_key(self, thread_id: str) -> str:
        return f"triage_pending:{thread_id}"

    async def save(self, thread_id: str, state: TriageState, reply: str) -> None:
        """落盘进度，并标记「有一段追问在等回答」。

        两件事绑在一起做：save 的调用点就是「即将挂起等用户回答」，漏打标记
        会让放弃分支失去判据，所以不留给调用方各自记得。
        """
        payload = json.dumps(
            {"state": state.model_dump_json(), "reply": reply},
            ensure_ascii=False,
        )
        await self.redis.set(self._key(thread_id), payload, ex=_ttl_seconds())
        await self.mark_pending(thread_id)

    async def mark_pending(self, thread_id: str) -> None:
        await self.redis.set(self._pending_key(thread_id), "1", ex=_ttl_seconds())

    async def is_pending(self, thread_id: str) -> bool:
        """本会话是否有追问在等回答（进度可能已经过期消失）。"""
        return bool(await self.redis.get(self._pending_key(thread_id)))

    async def _refresh_ttl(self, thread_id: str) -> None:
        """读时续期（对齐 checkpointer 的 refresh_on_read）。

        尽力而为：数据已经读到了，TTL 没续上不该让整轮诊断失败。
        它真正的作用在「用户隔很久才回来回答追问」这条路径上 —— 那次读
        正好把进度续上，从而保证进度不会先于挂起的 interrupt 过期。
        """
        try:
            await self.redis.expire(self._key(thread_id), _ttl_seconds())
        except Exception as e:
            logger.warning(f"[TRIAGE-STORE] 续期失败（数据已读到，不影响本轮）thread={thread_id}: {e}")

    async def load(self, thread_id: str) -> TriageProgress | None:
        raw = await self.redis.get(self._key(thread_id))
        if not raw:
            return None
        try:
            data = json.loads(raw)
            progress = TriageProgress(
                state=TriageState.model_validate_json(data["state"]),
                reply=data.get("reply", ""),
            )
        except Exception as e:
            # Redis 里的数据损坏 / 版本不兼容：按无会话处理并清掉脏数据，
            # 让本轮按全新诊断重启（可自愈），而不是让工具直接崩掉
            logger.warning(f"[TRIAGE-STORE] 数据损坏，重置会话 thread={thread_id}: {e}")
            await self.clear(thread_id)
            return None
        await self._refresh_ttl(thread_id)
        return progress

    async def clear(self, thread_id: str) -> None:
        """清进度与挂起标记。诊断收敛、用户退出、放弃过期对话都走这里。"""
        await self.redis.delete(self._key(thread_id), self._pending_key(thread_id))
        logger.info(f"[TRIAGE-STORE] cleared thread={thread_id}")
