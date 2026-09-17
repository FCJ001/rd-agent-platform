#!/usr/bin/env python
# ============================================================
# 启用 / 停用 PostgreSQL 行级安全（RLS）：按业务线隔离数据行
#
# 与应用的配合（缺一不可）：
#   · 应用侧：src/core/scope.py 的 ContextVar + src/infra/db.py 的
#     after_begin 钩子，每个事务执行 set_config('app.business_line', ?, true)
#   · 数据库侧：本脚本建策略
#
# ★★ 最重要的前置条件：应用连接用的角色【不能是超级用户，也不能是表属主】。
#    超级用户无条件绕过 RLS（连 FORCE ROW LEVEL SECURITY 都拦不住），
#    属主在未加 FORCE 时也豁免。任一条成立时开 RLS 都是摆设 ——
#    本脚本会先检查并拒绝执行，而不是给你一个"看起来开了"的假象。
#
#    官方 postgres 镜像的 POSTGRES_USER 建出来就是超级用户，所以
#    docker-compose 里的 rdagent 目前属此类，必须先做角色拆分：
#      1) 建普通角色（应用用）：CREATE ROLE rd_agent_app LOGIN PASSWORD '...';
#      2) 授权：GRANT USAGE ON SCHEMA public / GRANT SELECT,INSERT,UPDATE,DELETE
#         ON ALL TABLES / GRANT USAGE,SELECT ON ALL SEQUENCES
#      3) 应用 .env 的 DB_USER 改成 rd_agent_app
#      4) 迁移/种子/回填脚本继续用属主角色（它们没有业务线概念，需要全量可见）
#
# 用法：
#   python scripts/enable_rls.py --check              # 只检查前置条件
#   python scripts/enable_rls.py --apply              # 建策略并启用
#   python scripts/enable_rls.py --apply --verify     # 启用后再验证一次实际过滤
#   python scripts/enable_rls.py --disable            # 停用（回滚）
# ============================================================

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import psycopg2  # noqa: E402

from src.core.config import get_settings  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 带 business_line 列、需要按业务线隔离的业务表
# （AI 影子表不在内：它们没有业务线列，隔离靠 user_id/session_id，
#   要纳入得先加列 —— 见 docs 里的待办）
SCOPED_TABLES = [
    "alm_issues",
    "alm_requirements",
    "alm_change_requests",
    "alm_config_items",
    "alm_baselines",
    "owner_domains",
    "root_causes",
    "phenomena",
    "dtc_codes",
    # 分诊结论自 2026-09 起带 business_line，是项目知识复用的检索维度，
    # 一并纳入隔离。注意存量行 business_line 为 NULL —— 策略下查不到，
    # 这是刻意的 fail-closed（不可归属的存量不进共享池）
    "ai_triage_results",
]

POLICY_NAME = "business_line_scope"
SCOPE_SETTING = "app.business_line"


def connect():
    s = get_settings()
    return psycopg2.connect(
        host=s.DB_HOST, port=s.DB_PORT,
        user=s.DB_USER, password=s.DB_PASSWORD, dbname=s.DB_NAME,
    )


def check_connecting_role(cur) -> list[str]:
    """连接角色要能改表：属主或超级用户。它只负责执行 DDL，不是被隔离对象。"""
    cur.execute("SELECT current_user, usesuper FROM pg_user WHERE usename = current_user")
    role, is_super = cur.fetchone()
    logger.info(f"执行 DDL 的连接角色: {role}（superuser={is_super}）")

    cur.execute(
        """SELECT count(*) FROM pg_tables
           WHERE schemaname = 'public' AND tablename = ANY(%s) AND tableowner = current_user""",
        (SCOPED_TABLES,),
    )
    owned = cur.fetchone()[0]
    if not is_super and owned == 0:
        return [f"连接角色 {role} 既不是超级用户也不是目标表属主，无法建策略/改表"]
    return []


def check_app_role(cur, app_role: str) -> list[str]:
    """★ 真正的关键检查：应用连接用的角色必须受 RLS 约束。

    超级用户无条件绕过（FORCE 也拦不住），BYPASSRLS 属性同理，
    表属主在未加 FORCE 时豁免。任一条成立，RLS 就是摆设。
    """
    problems = []

    cur.execute("SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = %s", (app_role,))
    row = cur.fetchone()
    if row is None:
        return [f"应用角色 {app_role} 不存在"]
    is_super, bypass_rls = row
    logger.info(f"应用角色: {app_role}（superuser={is_super}, bypassrls={bypass_rls}）")

    if is_super:
        problems.append(
            f"应用角色 {app_role} 是超级用户 —— 无条件绕过 RLS，开了也不生效。"
            "官方 postgres 镜像的 POSTGRES_USER 建出来就是超级用户，必须另建普通角色"
        )
    if bypass_rls:
        problems.append(f"应用角色 {app_role} 带 BYPASSRLS 属性，RLS 对其无效")

    cur.execute(
        """SELECT tablename FROM pg_tables
           WHERE schemaname = 'public' AND tablename = ANY(%s) AND tableowner = %s""",
        (SCOPED_TABLES, app_role),
    )
    owned = sorted(t for (t,) in cur.fetchall())
    if owned:
        problems.append(
            f"应用角色 {app_role} 是这些表的属主：{owned}。"
            "属主靠 FORCE 才能受约束，但那样维护脚本（也用属主连接）会一起被限制 —— "
            "正确做法是应用用非属主角色，维护脚本用属主角色"
        )
    return problems


def apply_policies(cur) -> int:
    """建策略并启用 RLS。返回处理的表数。"""
    # 未设作用域时 current_setting 返回 NULL 或 ''（PG 对自定义 GUC 的
    # SET LOCAL 结束后会留下空串）—— 两者都匹配不到任何真实业务线值，
    # 因此"忘了设作用域"的路径是查不到数据的（fail-closed），不是查到全部。
    using = f"business_line = current_setting('{SCOPE_SETTING}', true)"
    applied = 0
    for table in SCOPED_TABLES:
        cur.execute(
            f"""DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1 FROM pg_policies
                    WHERE tablename = '{table}' AND policyname = '{POLICY_NAME}'
                ) THEN
                    CREATE POLICY {POLICY_NAME} ON {table}
                      USING ({using}) WITH CHECK ({using});
                END IF;
            END $$;"""
        )
        cur.execute(f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY")
        # FORCE：让表属主也受策略约束。配合「非属主应用角色」使用；
        # 若应用角色就是属主，则必须加 FORCE 才有意义
        cur.execute(f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY")
        applied += 1
        logger.info(f"  {table}: 策略就位 + ENABLE + FORCE")
    return applied


def disable_policies(cur) -> int:
    for table in SCOPED_TABLES:
        cur.execute(f"ALTER TABLE {table} NO FORCE ROW LEVEL SECURITY")
        cur.execute(f"ALTER TABLE {table} DISABLE ROW LEVEL SECURITY")
        cur.execute(f"DROP POLICY IF EXISTS {POLICY_NAME} ON {table}")
    logger.info(f"  {len(SCOPED_TABLES)} 张表已停用 RLS 并删除策略")
    return len(SCOPED_TABLES)


def verify(cur, app_role: str, sample_table: str = "phenomena") -> None:
    """启用后实测：SET ROLE 切到应用角色（→ current_user 变化，RLS 按它判定），
    设了作用域只看到本线，不设作用域什么都看不到。"""
    # 全表行数要用连接角色（不受隔离）数，切到应用角色后就看不全了
    cur.execute(f"SELECT count(*) FROM {sample_table} WHERE business_line IS NOT NULL")
    total = cur.fetchone()[0]
    logger.info(f"验证（{sample_table} 全表 {total} 行；以下以 {app_role} 身份查询）")

    cur.execute("BEGIN")
    try:
        cur.execute(f"SET LOCAL ROLE {app_role}")
    except Exception as e:
        cur.execute("ROLLBACK")
        logger.warning(f"无法 SET ROLE {app_role}（连接角色权限不足），跳过实测: {e}")
        return


    cur.execute(f"SELECT count(*) FROM {sample_table}")
    unscoped = cur.fetchone()[0]
    logger.info(f"  未设作用域可见 {unscoped} 行（应为 0 = fail-closed）")

    cur.execute(f"SET LOCAL {SCOPE_SETTING} = 'ev'")
    cur.execute(f"SELECT count(*) FROM {sample_table}")
    seen = cur.fetchone()[0]
    logger.info(f"  作用域=ev 可见 {seen} 行")
    cur.execute("ROLLBACK")

    if unscoped != 0:
        logger.error(
            "★ 未设作用域仍能看到数据 —— RLS 未真正生效，"
            "应用角色很可能仍是超级用户/属主（见文件头注释）"
        )
        sys.exit(1)
    if total and seen >= total:
        logger.warning("★ 作用域过滤后行数等于全表 —— 该表数据可能只有一条业务线，属正常")


def main():
    ap = argparse.ArgumentParser(description="启用/停用业务线 RLS")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="只检查前置条件")
    g.add_argument("--apply", action="store_true", help="建策略并启用")
    g.add_argument("--disable", action="store_true", help="停用并删除策略")
    ap.add_argument("--verify", action="store_true", help="启用后实测过滤效果")
    ap.add_argument(
        "--app-role", default="",
        help="应用连接用的角色名（用于前置检查与实测）。"
             "DDL 由连接角色执行，但它本身不受隔离约束，必须显式区分",
    )
    args = ap.parse_args()

    if not args.app_role and not args.check:
        logger.error("必须用 --app-role 指定应用连接角色（--check 可省略，仅检查连接角色）")
        sys.exit(2)

    conn = connect()
    conn.autocommit = False
    cur = conn.cursor()
    try:
        problems = check_connecting_role(cur)
        if args.app_role:
            problems += check_app_role(cur, args.app_role)

        if args.check:
            for p in problems:
                logger.error(f"阻断：{p}")
            logger.info("前置条件检查完成：" + ("不通过" if problems else "通过"))
            sys.exit(1 if problems else 0)

        if problems:
            for p in problems:
                logger.error(f"阻断：{p}")
            logger.error("前置条件不满足，拒绝启用 —— 见文件头注释的角色拆分步骤")
            sys.exit(1)

        if args.disable:
            disable_policies(cur)
            conn.commit()
            logger.info("RLS 已停用")
            return

        applied = apply_policies(cur)
        conn.commit()
        logger.info(f"RLS 已启用：{applied} 张表")

        if args.verify:
            verify(cur, args.app_role)
    finally:
        cur.close()
        conn.close()


if __name__ == "__main__":
    main()
