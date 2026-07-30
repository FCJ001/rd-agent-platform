"""Java ALM 平台交互工具。开发期调 Java API 打 logger 占位。"""

from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.core.config import get_settings
from src.core.deps import UserContext
from src.core.logger import logger

settings = get_settings()


@tool
async def call_create_issue(
    title: str,
    description: str,
    severity: str,
    business_line: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """在 ALM 平台创建问题单。
    适用场景：用户描述的故障需要正式跟踪处理时（安全类/硬件类/需采购类）。
    调用时机：诊断收敛后，Agent 判断需要走正式流程。

    Args:
        title: 问题标题（简短描述）
        description: 问题描述（故障现象、DTC码等）
        severity: 严重度（blocker/critical/normal/minor）
        business_line: 业务线（ev=电动化 / ia=智能化）
    """
    ctx = runtime.context
    payload = {
        "title": title,
        "description": description,
        "severity": severity,
        "business_line": business_line,
        "source": ctx.role,
        "reporter_id": ctx.user_id,
        "owner_domain_id": ctx.owner_domain_id,
    }

    logger.info(
        f"[JAVA-API] POST {settings.PLATFORM_ALM_API_URL}/issues "
        f"user={ctx.user_id} role={ctx.role} "
        f"payload={payload}"
    )

    return (
        f"已向 ALM 平台提交问题单创建请求。\n"
        f"平台地址：{settings.PLATFORM_ALM_URL}/issues\n"
        f"问题描述：{title}\n"
        f"严重度：{severity} | 业务线：{business_line}"
    )


@tool
async def call_link_issue(
    issue_no: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """关联已有问题单到当前诊断会话。
    适用场景：用户提到已有问题单号，需要基于此单进行诊断。

    Args:
        issue_no: ALM 平台问题单号，如 ISS-2025-00123
    """
    ctx = runtime.context

    logger.info(
        f"[JAVA-API] GET {settings.PLATFORM_ALM_API_URL}/issues/{issue_no} "
        f"user={ctx.user_id} session={ctx.session_id}"
    )

    return (
        f"已关联问题单 {issue_no}。\n"
        f"查看详情：{settings.PLATFORM_ALM_URL}/issues/{issue_no}\n"
        f"接下来可以对此问题进行诊断分析。"
    )


@tool
async def call_close_issue(
    issue_no: str,
    root_cause: str,
    fix_action: str,
    runtime: ToolRuntime[UserContext],
) -> str:
    """在 ALM 平台结案。
    适用场景：诊断出明确根因且修复方案已实施时。
    注意：Agent 只建议结案，最终由平台侧审批。

    Args:
        issue_no: ALM 平台问题单号
        root_cause: 根因描述
        fix_action: 修复措施
    """
    ctx = runtime.context
    payload = {
        "issue_no": issue_no,
        "root_cause": root_cause,
        "fix_action": fix_action,
        "status": "verified",
        "operator_id": ctx.user_id,
    }

    logger.info(
        f"[JAVA-API] POST {settings.PLATFORM_ALM_API_URL}/issues/{issue_no}/close "
        f"user={ctx.user_id} payload={payload}"
    )

    return (
        f"已提交结案建议。\n"
        f"问题单号：{issue_no}\n"
        f"根因：{root_cause}\n"
        f"修复措施：{fix_action}\n"
        f"平台审批链接：{settings.PLATFORM_ALM_URL}/issues/{issue_no}"
    )


PLATFORM_TOOLS = [call_create_issue, call_link_issue, call_close_issue]
