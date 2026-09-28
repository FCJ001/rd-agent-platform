# ============================================================
# 子智能体注册表 —— 多轮 HITL 智能体接入平台的唯一登记处
# 设计：docs/multi-agent-architecture-design.md §6.5（接入契约）
#
# 分诊（triage）是第一个按此形态接入的子智能体样板：
#   子图（session_graph）+ handoff 工具（orchestrator）+ done 节点。
# 以后接入类似智能体（报告评审流/变更审批流…）在此登记一行，
# chat 层的退出判定、旁路工具摘除、闸门口径即自动生效——
# 接入成本是 O(业务逻辑)，基础设施（锁/恢复/旁路/trace）全部共享。
#
# ★ 硬规则：多轮流程只允许以子图接入，禁止「工具内 interrupt + 外挂
#   store」——那是 P1 单图化还掉的债，不能再长出来（架构守护测试锚定）。
# ============================================================

from dataclasses import dataclass
from typing import Callable

from src.agents.triage.session_store import is_triage_exit


@dataclass(frozen=True)
class SubAgentSpec:
    """一个子智能体的接入描述。

    interrupt_type: 图内 interrupt 载荷的 type 字段（父图快照里的身份证）
    exit_checker:   精确匹配的退出指令判定（长描述含"取消"不得误判）
    entry_tool:     对应的 supervisor 入口工具名 —— 旁路助手必须摘除它，
                    防止旁路线程另起同种诊断（与挂起任务抢状态的老问题）
    gated:          追问轮（resume 回合）是否过全局闸门。默认 True
                    （容量保护 fail-closed）；显式 False 留给未来明确
                    不占模型容量的轻量流程
    """

    interrupt_type: str
    exit_checker: Callable[[str], bool]
    entry_tool: str
    gated: bool = True


SUB_AGENTS: dict[str, SubAgentSpec] = {
    "triage_followup": SubAgentSpec(
        interrupt_type="triage_followup",
        exit_checker=is_triage_exit,
        entry_tool="call_triage_agent",
    ),
}


def spec_for(interrupt_type: str) -> SubAgentSpec | None:
    """按 interrupt 载荷的 type 查接入规格；未登记返回 None（fail-safe）。"""
    return SUB_AGENTS.get(interrupt_type)


def is_task_exit(interrupt_type: str, message: str) -> bool:
    """挂起任务类型 + 用户消息 → 是否退出指令。

    未登记的类型一律不是退出（fail-safe：宁可当追问回答走原路径，
    也不能把未知流程的会话误杀）。
    """
    spec = spec_for(interrupt_type)
    return bool(spec and spec.exit_checker(message))


def entry_tool_for(interrupt_type: str) -> str | None:
    """挂起任务类型 → 旁路助手必须摘除的入口工具名；未登记不摘。"""
    spec = spec_for(interrupt_type)
    return spec.entry_tool if spec else None


def is_gated(interrupt_type: str) -> bool:
    """挂起任务类型 → 追问轮是否过全局闸门；未登记默认 True（收紧）。"""
    spec = spec_for(interrupt_type)
    return spec.gated if spec else True


# ── 任务切换（确认式「换台」，docs 最终方案 2026-09-27）──────────────────
# 挂起期间用户显式要求结束当前流程去办别的事。与退出词同层的精确匹配，
# 必须发生在离题判定器之前（"切换"不命中离题正则，漏拦截会被当追问回答）。
# 铁律：切换动作只由这些精确关键词触发，判定器/提示永远无权终止诊断。
SWITCH_COMMANDS = {"切换", "切换流程"}

# chat 层切换分支发给图的规范化退出令牌 —— 必须是 TRIAGE_EXIT_COMMANDS
# 的成员，这样旧路径（工具内退出分支）和新路径（子图 wait_answer）都认，
# 且不用改全局退出词表（影响面收敛在 chat 层）
SWITCH_EXIT_TOKEN = "退出诊断"


def is_task_switch(message: str) -> bool:
    """是否为任务切换指令（去空白和尾部标点后精确匹配，防长句误伤）。"""
    msg = (message or "").strip().strip("。，！？!?,.~")
    return msg in SWITCH_COMMANDS
