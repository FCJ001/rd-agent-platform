# ============================================================
# 业务线（数据作用域）的取值工具
#
# 为什么需要「推导」而不是只靠显式值：分诊的 UserContext.business_line
# 只有 engineer / business 角色有值，而高频用户是 customer —— 他们在
# 诊断发生时往往还没建单，拿不到问题单的权威归属。没有归属，结论就进不了
# 项目维度的知识池（等于白诊断一次）。
#
# 推导依据是数据事实：根因编码带业务线前缀（RC-EV-0012 / PH-IA-001），
# 分诊候选本身也按线过滤过，所以从 primary_cause_code 反推是可靠的。
# 这是全项目第四处「从编码前缀反推归属」，收敛到这里，别再散着写。
# ============================================================

import re

from src.core.config import get_settings

# 形如 RC-EV-0012 / PH-IA-001：前缀段、业务线段（纯字母）、序号（数字开头）
# ★ 要求第三段以数字开头，才能避开 ISS-2025-00001 / CR-2025-0088 这类
#   「第二段是年份」的编号 —— 它们第二段是数字，天然不匹配
_CODE_LINE_RE = re.compile(r"^[A-Za-z]{2,6}-([A-Za-z]{2,8})-\d")


def line_from_code(code: str | None) -> str:
    """从带线前缀的编码取业务线：RC-EV-0012 → ev。取不到返回空串。

    容忍前后空白（编码可能从表单/日志里带进来），但不做大小写以外
    的模糊匹配 —— 匹配不到就是没有归属，不做猜测。
    """
    m = _CODE_LINE_RE.match((code or "").strip())
    return m.group(1).lower() if m else ""


def resolve_business_line(
    explicit: str | None, cause_code: str | None = None,
) -> str:
    """确定一条记录的业务线：显式值优先，缺失时从根因编码前缀推导。

    只接受配置里已知的业务线（settings.BUSINESS_LINES）—— 推导出的值
    不在白名单里就返回空串，宁可无归属也不写入一个没人认识的线。
    """
    allowed = get_settings().business_lines
    if explicit and explicit in allowed:
        return explicit
    derived = line_from_code(cause_code)
    return derived if derived in allowed else ""
