# ============================================================
# 用户种子数据（14 人 / 5 类角色）
#
# ★ 本脚本【不写密码】—— 认证归 ALM 平台（Java）。
#   开发期身份靠 get_current_user() 读 X-User-Id 查 users 表，
#   调试就是 curl -H "X-User-Id: 3"，换个数字就换角色。
#
# 覆盖 4 类业务角色 + admin，保证每条行过滤规则都有「看得到」和「看不到」两组数据。
# 前置：先跑 scripts/seed_domains.py（工程师要挂到责任域上）
#
# 用法: cd rd-agent-platform && python scripts/seed_users.py
# ============================================================

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def seed() -> None:
    import psycopg2

    settings = get_settings()
    conn = psycopg2.connect(
        host=settings.DB_HOST, port=settings.DB_PORT,
        user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
    )
    conn.autocommit = False
    cur = conn.cursor()

    # 每个责任域一个工程师 —— 保证 engineer 的行过滤有正反用例
    cur.execute("SELECT id, name, business_line FROM owner_domains ORDER BY id")
    domains = cur.fetchall()
    if not domains:
        raise SystemExit("owner_domains 为空，先跑 python scripts/seed_domains.py")

    users = [
        (f"eng{i:02d}", f"工程师-{domain_name}", "engineer", line, domain_id)
        for i, (domain_id, domain_name, line) in enumerate(domains, start=1)
    ]
    users += [
        ("biz_ev", "业务-电动化", "business", "ev", None),
        ("biz_ia", "业务-智能化", "business", "ia", None),
        ("service01", "售后专员", "aftersales", None, None),
        ("cust01", "客户张三", "customer", None, None),
        ("admin", "系统管理员", "admin", None, None),
    ]

    for username, real_name, role_type, line, domain_id in users:
        cur.execute(
            """INSERT INTO users (username, real_name, role_type, business_line, owner_domain_id, is_active)
               VALUES (%s,%s,%s,%s,%s,true) ON CONFLICT (username) DO NOTHING""",
            (username, real_name, role_type, line, domain_id),
        )

    conn.commit()

    cur.execute("""SELECT id, username, real_name, role_type,
                          COALESCE(business_line,'-'), COALESCE(owner_domain_id,0)
                   FROM users ORDER BY id""")
    for uid, uname, rname, role, line, dom in cur.fetchall():
        logger.info(f"  [{uid:>2}] {uname:<10} {rname:<16} role={role:<10} line={line:<3} domain={dom}")

    logger.info(f"用户种子完成：{len(users)} 人，覆盖 5 类角色")
    logger.info("调试用法：curl -H 'X-User-Id: 1' http://localhost:8000/api/v1/issues")

    cur.close()
    conn.close()


if __name__ == "__main__":
    seed()
