# ============================================================
# 问题单查询接口
#
# 步 1 的验收载体：同一条 SQL、同一个 handler，
# 只因为 X-User-Id 不同，返回的行数就不同。
#
#   curl -H "X-User-Id: 1"  .../api/v1/issues   → engineer eng01  10 条
#   curl -H "X-User-Id: 10" .../api/v1/issues   → business biz_ev 12 条
#   curl -H "X-User-Id: 12" .../api/v1/issues   → aftersales       5 条
#   curl -H "X-User-Id: 13" .../api/v1/issues   → customer cust01  3 条
#   curl -H "X-User-Id: 14" .../api/v1/issues   → admin           24 条
# ============================================================

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, ConfigDict
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_repository import BaseRepository
from src.core.base_schema import PageResult, ResponseSchema
from src.core.deps import UserContext, get_current_user
from src.core.exceptions import ERR_ISSUE_NOT_FOUND, BizException
from src.core.logger import logger
from src.core.permissions import build_issue_filters
from src.infra.db import get_db
from src.modules.alm.model import AlmIssue

router = APIRouter(prefix="/api/v1/issues", tags=["问题单"])


class IssueOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    issue_no: str
    title: str
    source: str
    business_line: str
    severity: str
    status: str
    model_code: str | None = None
    sw_version: str | None = None
    dtc_snapshot: str | None = None
    owner_domain_id: int | None = None
    reporter_id: int | None = None


@router.get("", response_model=ResponseSchema[PageResult[IssueOut]])
async def list_issues(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    keyword: str | None = Query(None, description="标题/描述模糊搜索"),
    ctx: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    repo = BaseRepository(AlmIssue, db)
    filters = build_issue_filters(ctx)

    items, total = await repo.get_page(
        offset=(page - 1) * page_size,
        limit=page_size,
        keyword=keyword,
        search_fields=["title", "description"],
        filters=filters,
    )

    logger.info(f"问题单列表 role={ctx.role} user={ctx.user_id} 命中 {total} 条")

    return ResponseSchema(
        data=PageResult[IssueOut](
            items=[IssueOut.model_validate(i) for i in items],
            total=total,
            page=page,
            page_size=page_size,
        )
    )


@router.get("/{issue_id}", response_model=ResponseSchema[IssueOut])
async def get_issue(
    issue_id: int,
    ctx: UserContext = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    ★ 详情走的是和列表【同一套】filters。
      「列表过滤了、详情没过滤」是越权最常见的入口：
      列表里看不到别人的单，但把 URL 里的 id 改一改照样能拿到。
      看不见的单一律报"不存在"，不报"无权限" —— 后者等于确认了这个 id 存在。
    """
    repo = BaseRepository(AlmIssue, db)
    issue = await repo.get_by_id(issue_id, filters=build_issue_filters(ctx))
    if issue is None:
        raise BizException(f"问题单不存在或无权访问：id={issue_id}", ERR_ISSUE_NOT_FOUND)
    return ResponseSchema(data=IssueOut.model_validate(issue))
