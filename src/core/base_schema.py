# ============================================================
# 统一响应结构
#
# 所有 HTTP 接口返回 ResponseSchema[T]，前端只需要判一个 code。
# ★ 业务异常也走 HTTP 200 + code != 200（见 exceptions.py）——
#   目的是不让浏览器/网关的 4xx/5xx 重试逻辑干扰业务错误的展示。
#   真正的 5xx 只留给"服务挂了"。
#
# 用法：
#   @router.get("/issues", response_model=ResponseSchema[PageResult[IssueOut]])
# ============================================================

from typing import Generic, Optional, TypeVar

from pydantic import BaseModel

T = TypeVar("T")


class ResponseSchema(BaseModel, Generic[T]):
    code: int = 200
    message: str = "success"
    data: Optional[T] = None

    # trace_id 回带给前端，出问题时用户截个图就能定位到日志
    trace_id: Optional[str] = None


class PageResult(BaseModel, Generic[T]):
    """分页结果包装。total 是过滤后的总数 —— 行级过滤生效时它会随角色变化"""
    items: list[T] = []
    total: int = 0
    page: int = 1
    page_size: int = 20
