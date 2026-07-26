from typing import Generic, Sequence, Type, TypeVar

from sqlalchemy import ColumnElement, delete, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.base_model import BaseModel

# T 必须是 BaseModel 的子类
T = TypeVar("T", bound=BaseModel)


class BaseRepository(Generic[T]):
    """
    通用仓储（相当于 MyBatis 的 BaseMapper）

    ★ 与常见模板的唯一区别：查询方法都接受 filters 参数。
      行级过滤条件从外面传进来（由 src/core/permissions.py 按角色生成），
      Repository 只负责把它拼进 WHERE —— 权限规则和数据访问解耦，
      规则改了不用动 Repository，Repository 也不需要认识 UserContext。
    """

    def __init__(self, model: Type[T], db: AsyncSession):
        self.model = model
        self.db = db

    async def get_by_id(self, id: int, filters: Sequence[ColumnElement[bool]] | None = None) -> T | None:
        """
        ★ 带 filters 时不能用 db.get() —— 那是主键直取，绕过 WHERE。
          越权访问最常见的入口就是"列表过滤了，详情没过滤"：
          列表里看不到别人的单，但把 id 改一改照样能拿到详情。
        """
        stmt = select(self.model).where(self.model.id == id)
        if filters:
            stmt = stmt.where(*filters)
        result = await self.db.execute(stmt)
        return result.scalar_one_or_none()

    async def get_all(
        self,
        offset: int = 0,
        limit: int = 100,
        filters: Sequence[ColumnElement[bool]] | None = None,
    ) -> Sequence[T]:
        stmt = select(self.model)
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        return result.scalars().all()

    async def create(self, obj: T) -> T:
        self.db.add(obj)
        await self.db.flush()
        await self.db.refresh(obj)
        return obj

    async def update(self, obj: T) -> T:
        """obj 必须是本会话查出来的对象，不能是 new 出来的"""
        await self.db.flush()
        await self.db.refresh(obj)
        return obj

    async def delete(self, obj: T) -> None:
        await self.db.delete(obj)
        await self.db.flush()

    async def delete_by_id(self, id: int) -> None:
        stmt = delete(self.model).where(self.model.id == id)
        await self.db.execute(stmt)

    async def get_page(
        self,
        offset: int = 0,
        limit: int = 20,
        keyword: str | None = None,
        search_fields: list[str] | None = None,
        filters: Sequence[ColumnElement[bool]] | None = None,
    ) -> tuple[list[T], int]:
        """
        通用分页 + 模糊搜索 + 行级过滤

        参数：
            offset / limit  分页
            keyword         搜索关键词
            search_fields   参与模糊匹配的字段名，如 ["title", "description"]
            filters         行级过滤条件，来自 permissions.build_filters(ctx)

        返回：(数据列表, 总条数)
        """
        stmt = select(self.model)

        # ① 行级过滤：多个条件之间是 AND
        if filters:
            stmt = stmt.where(*filters)

        # ② 关键词：多个字段之间是 OR
        if keyword and search_fields:
            conditions = []
            for field_name in search_fields:
                column = getattr(self.model, field_name, None)
                if column is not None:
                    conditions.append(column.ilike(f"%{keyword}%"))
            if conditions:
                stmt = stmt.where(or_(*conditions))

        # ★ total 必须基于【同一个 stmt】来数。
        #   如果这里另起 select(func.count()).select_from(self.model)，
        #   会出现"列表只给 3 条、total 却报 24"——
        #   行是挡住了，条数却把库里到底有多少泄露了出去。这类越权最隐蔽。
        count_stmt = select(func.count()).select_from(stmt.subquery())
        total = (await self.db.execute(count_stmt)).scalar_one()

        stmt = stmt.order_by(self.model.id.asc()).offset(offset).limit(limit)
        result = await self.db.execute(stmt)
        items = list(result.scalars().all())

        return items, total
