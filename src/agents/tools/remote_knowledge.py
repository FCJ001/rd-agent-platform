"""跨服务工具：调项目二 rd-knowledge-svc 的知识库和 BI 查询。"""

import json

import httpx
from langchain_core.tools import tool
from langgraph.prebuilt import ToolRuntime

from src.core.config import get_settings
from src.core.deps import UserContext
from src.core.logger import logger

settings = get_settings()

# 服务间静态 token，正式环境换成 JWT 透传
SERVICE_TOKEN = "rd-agent-internal"


async def _post(endpoint: str, body: dict, timeout: int = 30) -> dict:
    """统一的跨服务 POST 请求，带错误处理和降级。"""
    url = f"{settings.KNOWLEDGE_SVC_URL}{endpoint}"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                url,
                json=body,
                headers={
                    "X-Service-Token": SERVICE_TOKEN,
                    "Content-Type": "application/json",
                },
            )
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
        "role": ctx.role,
        "business_line": ctx.business_line,
        "owner_domain_id": ctx.owner_domain_id,
        "user_id": ctx.user_id,
    }

    data = await _post("/api/v1/knowledge/search", body)

    if "error" in data:
        return data["error"]

    answer = data.get("answer", "")
    sources = data.get("sources", [])

    lines = [answer]
    if sources:
        lines.append("\n**参考来源：**")
        for s in sources[:5]:
            lines.append(f"- {s.get('title', s.get('name', '未知'))}")

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
        "role": ctx.role,
        "business_line": ctx.business_line,
        "owner_domain_id": ctx.owner_domain_id,
        "user_id": ctx.user_id,
    }

    data = await _post("/api/v1/bi/query", body, timeout=60)

    if "error" in data:
        return data["error"]

    answer = data.get("answer", "")
    rows = data.get("rows", [])
    echarts_option = data.get("echarts_option")

    lines = [answer]

    if rows and len(rows) <= 10:
        lines.append("\n**数据明细：**")
        for row in rows:
            lines.append(f"- {json.dumps(row, ensure_ascii=False)}")

    if echarts_option:
        lines.append(f"\n*图表数据已生成*")

    return "\n".join(lines)


REMOTE_TOOLS = [call_knowledge_agent, call_operation_agent]
