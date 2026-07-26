# ============================================================
# 责任域种子数据（9 条，手写固定）
#
# ★ 这 9 条是【唯一事实来源】：
#   后面 scripts/gen_alm_kg.py 按域调 LLM 扩写根因、init_neo4j.py 建
#   BELONGS_TO 关系，都从这里 import OWNER_DOMAINS，不要各自再抄一份。
#
# 域的划分按 E/E 架构走，不是按部门 —— 这是汽车研发和医疗科室的关键差异：
# 一个现象往往横跨多个域（「续航缩水」既可能是电池系统域也可能是热管理域），
# 分诊要判的就是"该派给哪个域"，所以域必须正交且可穷举。
#
# 用法: cd rd-agent-platform && python scripts/seed_domains.py
# ============================================================

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 9 个责任域：电动化 5 个 + 智能化 4 个
# ----------------------------------------------------------
OWNER_DOMAINS = [
    {"name": "电池系统域", "business_line": "ev", "description": "电芯、模组、电池包、BMS 软硬件", "owner_name": "域负责人-电池"},
    {"name": "电驱系统域", "business_line": "ev", "description": "电机、电控 MCU、减速器、扭矩控制", "owner_name": "域负责人-电驱"},
    {"name": "充电系统域", "business_line": "ev", "description": "OBC 车载充电机、直流快充、充电协议", "owner_name": "域负责人-充电"},
    {"name": "热管理域", "business_line": "ev", "description": "电池热管理、空调、冷却回路、PTC", "owner_name": "域负责人-热管理"},
    {"name": "整车控制域", "business_line": "ev", "description": "VCU 整车控制器、能量管理、能量回收", "owner_name": "域负责人-整车"},
    {"name": "智能驾驶域", "business_line": "ia", "description": "感知、融合、规控、ADAS 功能安全", "owner_name": "域负责人-智驾"},
    {"name": "智能座舱域", "business_line": "ia", "description": "中控 HMI、仪表、语音、多媒体", "owner_name": "域负责人-座舱"},
    {"name": "车联网域", "business_line": "ia", "description": "T-Box、远程控车、云端服务、账号", "owner_name": "域负责人-车联网"},
    {"name": "OTA升级域", "business_line": "ia", "description": "整车 OTA、差分包、灰度放量、回滚", "owner_name": "域负责人-OTA"},
]


def seed() -> None:
    import psycopg2

    settings = get_settings()
    conn = psycopg2.connect(
        host=settings.DB_HOST, port=settings.DB_PORT,
        user=settings.DB_USER, password=settings.DB_PASSWORD, dbname=settings.DB_NAME,
    )
    conn.autocommit = False
    cur = conn.cursor()

    for d in OWNER_DOMAINS:
        # name 上有 UNIQUE，ON CONFLICT DO NOTHING 让脚本天然幂等（可重复跑）
        cur.execute(
            """INSERT INTO owner_domains (name, business_line, description, owner_name)
               VALUES (%s,%s,%s,%s) ON CONFLICT (name) DO NOTHING""",
            (d["name"], d["business_line"], d["description"], d["owner_name"]),
        )

    conn.commit()

    cur.execute("SELECT id, name, business_line FROM owner_domains ORDER BY id")
    for row in cur.fetchall():
        logger.info(f"  [{row[0]}] {row[1]} ({row[2]})")
    logger.info(f"责任域种子完成：{len(OWNER_DOMAINS)} 个")

    cur.close()
    conn.close()


if __name__ == "__main__":
    seed()
