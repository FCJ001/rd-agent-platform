"""跨服务工具：调项目二 rd-knowledge-svc 的知识库和 BI 查询。"""

import json

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.core.config import get_settings
from src.core.deps import UserContext
from src.core.logger import logger

settings = get_settings()


def _auth_headers(ctx: UserContext) -> dict:
    """项目二的身份走 HTTP Header，不放在 Body 里。
    可选 header 只在有值时发送，避免空字符串被 FastAPI 校验拒绝（422）。
    """
    headers = {
        "X-User-Id": ctx.user_id,
        "X-Session-Id": ctx.session_id,
        "X-User-Role": ctx.role,
        "Content-Type": "application/json",
    }
    if ctx.business_line:
        headers["X-Business-Line"] = ctx.business_line
    if ctx.owner_domain_id:
        headers["X-Owner-Domain-Id"] = str(ctx.owner_domain_id)
    return headers


async def _post(endpoint: str, body: dict, ctx: UserContext, timeout: int = 30) -> dict:
    """统一的跨服务 POST，带身份透传 + 错误降级。"""
    url = f"{settings.KNOWLEDGE_SVC_URL}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, json=body, headers=_auth_headers(ctx))
            resp.raise_for_status()
            return resp.json()
    except httpx.ConnectError:
        logger.warning(f"[REMOTE] 无法连接项目二: {url}")
        return {"error": "知识库服务暂不可用，请稍后重试"}
    except httpx.TimeoutException:
        logger.warning(f"[REMOTE] 请求超时: {url}")
        return {"error": "知识库服务响应超时，请稍后重试"}
    except Exception as e:
        logger.warning(f"[REMOTE] 请求失败: {url} err={e}")
        return {"error": f"知识库服务异常: {str(e)}"}


def _unwrap(data: dict) -> dict:
    """项目二响应统一包在 ResponseSchema 里，实际数据在 data.data。"""
    return data.get("data", data)


@tool
async def call_knowledge_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """搜索研发知识库，查找技术文档、规范标准、历史解决方案。
    适用场景：用户询问技术参数、标准规范、设计文档、竞品信息时。
    示例："800V高压系统绝缘设计标准"、"BMS SOC算法原理"、"竞品电池包能量密度"

    Args:
        message: 搜索问题（原文传递）
    """
    ctx = runtime.context
    logger.info(f"[REMOTE] call_knowledge_agent user={ctx.user_id} q={message[:80]}")

    body = {
        "question": message,
        "channels": ["doc_rag", "graph_rag"],
    }

    resp = await _post("/api/v1/knowledge/search", body, ctx)
    if "error" in resp:
        return resp["error"]

    data = _unwrap(resp)
    answer = data.get("answer", "")
    channels = data.get("channels", [])

    lines = [answer]
    if channels:
        lines.append(f"\n检索通道：{'、'.join(channels)}")

    return "\n".join(lines)


@tool
async def call_operation_agent(message: str, runtime: ToolRuntime[UserContext]) -> str:
    """查询运营/BI 数据，生成统计报表和图表。
    适用场景：用户询问统计数据、趋势分析、质量报表时。
    示例："最近一个月问题单Top5故障现象"、"Q3关闭率趋势"、"各域缺陷密度对比"

    Args:
        message: 查询问题（原文传递）
    """
    ctx = runtime.context
    logger.info(f"[REMOTE] call_operation_agent user={ctx.user_id} q={message[:80]}")

    body = {
        "question": message,
        "session_id": f"{ctx.user_id}:{ctx.session_id}",
        "with_chart": True,
    }

    resp = await _post("/api/v1/bi/query", body, ctx, timeout=60)
    if "error" in resp:
        return resp["error"]

    data = _unwrap(resp)
    summary = data.get("summary", "")
    success = data.get("success", False)
    sql = data.get("sql", "")
    rows = data.get("data", [])
    row_count = data.get("row_count", 0)
    has_chart = data.get("chart") is not None

    lines = [summary]

    if not success:
        return f"查询失败：{summary[:200]}"

    if row_count <= 10 and rows:
        lines.append("\n**数据明细：**")
        for row in rows:
            lines.append(f"- {json.dumps(row, ensure_ascii=False)}")

    if has_chart:
        lines.append(f"\n*图表数据已生成*")

    return "\n".join(lines)


REMOTE_TOOLS = [call_knowledge_agent, call_operation_agent]
