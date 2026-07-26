# ============================================================
# 问题单夹具数据（24 条）
#
# ★ 这份夹具是为「行级过滤」验收专门设计的，行数不是随手填的：
#
#   | 角色                | 过滤条件                          | 期望行数 |
#   |---------------------|-----------------------------------|---------|
#   | engineer  eng01     | owner_domain_id = 1（电池系统域） | 10      |
#   | business  biz_ev    | business_line = 'ev'              | 12      |
#   | aftersales service01| status IN ('closed','verified')   | 5       |
#   | customer  cust01    | reporter_id = 自己                | 3       |
#   | admin               | 不限制                            | 24      |
#
#   五个数字【两两不等】—— 这是刻意的。如果 engineer 和 business 都返回 12，
#   测试通过也无法证明过滤生效（可能两个分支走到了同一个 WHERE）。
#   另外 eng04（热管理域）/ eng05（整车控制域）期望 0 行，提供「看不到」的反例。
#
# 另一处刻意设计：customer 的过滤是 reporter_id 而【不是】source='customer'。
#   所以 #13 / #19 / #22 虽然 source='customer'，但 reporter_id 为 NULL
#   （模拟 NHTSA 外部导入的投诉），cust01 看不到它们。
#   如果哪天有人把过滤条件写成 source == 'customer'，行数会从 3 变成 6，测试立刻报警。
#
# 前置：seed_domains.py → seed_users.py
# 用法: cd rd-agent-platform && python scripts/seed_issues.py
# ============================================================

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ----------------------------------------------------------
# 字段顺序：
# issue_no, title, source, line, severity, status, model_code, sw_version,
# vin, dtc, reporter(username 或 None), domain(责任域名 或 None), external_ref
# ----------------------------------------------------------
ISSUES = [
    # ---- 电池系统域 × 10（engineer eng01 的可见集）----
    ("ISS-2025-00001", "续航里程比标称少 20%", "customer", "ev", "normal", "open",
     "EV-A01", "2024.32.5", "A1B2C3", None, "cust01", "电池系统域", None),
    ("ISS-2025-00002", "电池单体压差超限告警", "engineer", "ev", "critical", "analyzing",
     "EV-A01", "2024.32.5", None, "P0A0F", "eng01", "电池系统域", None),
    ("ISS-2025-00003", "SOC 跳变，从 40% 直接掉到 15%", "aftersales", "ev", "critical", "fixing",
     "EV-A02", "2024.30.1", "D4E5F6", "P0AFA", "service01", "电池系统域", None),
    ("ISS-2025-00004", "低温环境可用容量骤降", "engineer", "ev", "normal", "analyzing",
     "EV-A01", "2024.32.5", None, None, "eng01", "电池系统域", None),
    ("ISS-2025-00005", "电池包绝缘阻值低于阈值报警", "aftersales", "ev", "blocker", "verified",
     "EV-B01", "2024.28.9", "G7H8I9", "P0A0F,P1AF0", "service01", "电池系统域", None),
    ("ISS-2025-00006", "快充到 80% 后功率降至 30kW", "customer", "ev", "normal", "open",
     "EV-A02", "2024.30.1", "J1K2L3", None, "cust01", "电池系统域", None),
    ("ISS-2025-00007", "静置一夜掉电 5%", "customer", "ev", "minor", "closed",
     "EV-A01", "2024.32.5", None, None, None, "电池系统域", "nhtsa:11512345"),
    ("ISS-2025-00008", "BMS 主动均衡功能不启动", "engineer", "ev", "normal", "analyzing",
     "EV-B01", "2024.28.9", None, None, "eng01", "电池系统域", None),
    ("ISS-2025-00009", "电池包底部轻微磕碰后持续告警", "aftersales", "ev", "critical", "closed",
     "EV-A02", "2024.30.1", "M4N5O6", "P0A0F", "service01", "电池系统域", None),
    ("ISS-2025-00010", "循环 300 次后 SOH 降至 92%，低于目标值", "business", "ev", "normal", "open",
     "EV-A01", "2024.32.5", None, None, "biz_ev", "电池系统域", None),

    # ---- 其它 ev 域 × 2（凑够 business_line='ev' 共 12 条）----
    ("ISS-2025-00011", "起步顿挫，扭矩响应延迟约 300ms", "customer", "ev", "normal", "open",
     "EV-A01", "2024.32.5", "P7Q8R9", None, "cust01", "电驱系统域", None),
    ("ISS-2025-00012", "直流快充枪握手失败，报协议超时", "engineer", "ev", "critical", "analyzing",
     "EV-B01", "2024.28.9", None, "U0155", "eng03", "充电系统域", None),

    # ---- 智能座舱域 × 3 ----
    ("ISS-2025-00013", "中控屏黑屏，重启后恢复", "customer", "ia", "normal", "open",
     "IA-C01", "2025.02.3", None, None, None, "智能座舱域", "nhtsa:11598877"),
    ("ISS-2025-00014", "语音助手唤醒后无响应", "aftersales", "ia", "minor", "closed",
     "IA-C01", "2025.02.3", None, None, "service01", "智能座舱域", None),
    ("ISS-2025-00015", "仪表盘偶发花屏，约 2 秒恢复", "engineer", "ia", "normal", "analyzing",
     "IA-C02", "2025.01.8", None, None, "eng07", "智能座舱域", None),

    # ---- 智能驾驶域 × 3 ----
    ("ISS-2025-00016", "ACC 跟车距离突变，急减速", "engineer", "ia", "critical", "analyzing",
     "IA-C02", "2025.01.8", None, "U0100", "eng06", "智能驾驶域", None),
    ("ISS-2025-00017", "车道保持在大曲率弯道提前退出", "business", "ia", "normal", "open",
     "IA-C01", "2025.02.3", None, None, "biz_ia", "智能驾驶域", None),
    ("ISS-2025-00018", "AEB 误触发（前方无障碍物）", "aftersales", "ia", "blocker", "verified",
     "IA-C02", "2025.01.8", "S1T2U3", "U0100,U0155", "service01", "智能驾驶域", None),

    # ---- 车联网域 × 3 ----
    ("ISS-2025-00019", "远程控车指令超时无响应", "customer", "ia", "normal", "open",
     "IA-C01", "2025.02.3", None, None, None, "车联网域", "nhtsa:11603311"),
    ("ISS-2025-00020", "T-Box 4G 频繁掉网重连", "engineer", "ia", "normal", "fixing",
     "IA-C02", "2025.01.8", None, "U0073", "eng08", "车联网域", None),
    ("ISS-2025-00021", "账号多端登录被互踢", "business", "ia", "minor", "open",
     "IA-C01", "2025.02.3", None, None, "biz_ia", "车联网域", None),

    # ---- OTA升级域 × 3 ----
    ("ISS-2025-00022", "OTA 升级卡在 87% 不动", "customer", "ia", "critical", "fixing",
     "IA-C01", "2025.02.3", None, None, None, "OTA升级域", "nhtsa:11611002"),
    ("ISS-2025-00023", "差分包校验失败自动回滚", "engineer", "ia", "critical", "analyzing",
     "IA-C02", "2025.01.8", None, None, "eng09", "OTA升级域", None),
    ("ISS-2025-00024", "灰度放量后座舱重启率环比上升", "business", "ia", "normal", "open",
     "IA-C01", "2025.02.3", None, None, "biz_ia", "OTA升级域", None),
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

    # 用 username / 域名 反查 id —— 夹具里写名字而不写数字，改顺序也不会错位
    cur.execute("SELECT username, id FROM users")
    user_ids = dict(cur.fetchall())
    cur.execute("SELECT name, id FROM owner_domains")
    domain_ids = dict(cur.fetchall())
    if not user_ids or not domain_ids:
        raise SystemExit("users / owner_domains 为空，先跑 seed_domains.py + seed_users.py")

    for (issue_no, title, source, line, severity, status, model_code,
         sw_version, vin, dtc, reporter, domain, external_ref) in ISSUES:
        cur.execute(
            """INSERT INTO alm_issues
                 (issue_no, title, description, source, business_line, severity, status,
                  model_code, sw_version, vin, dtc_snapshot, reporter_id, owner_domain_id, external_ref)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT (issue_no) DO NOTHING""",
            (issue_no, title, title, source, line, severity, status,
             model_code, sw_version, vin, dtc,
             user_ids.get(reporter) if reporter else None,
             domain_ids.get(domain) if domain else None,
             external_ref),
        )

    conn.commit()

    # ------------------------------------------------------
    # 自校验：这五个数字必须两两不等，否则行过滤测试会假通过
    # ------------------------------------------------------
    checks = [
        ("engineer   eng01 (电池系统域)", "owner_domain_id = %s", (domain_ids["电池系统域"],), 10),
        ("business    biz_ev (ev 线)", "business_line = %s", ("ev",), 12),
        ("aftersales service01", "status IN ('closed','verified')", (), 5),
        ("customer    cust01", "reporter_id = %s", (user_ids["cust01"],), 3),
        ("admin", "TRUE", (), 24),
    ]
    logger.info("—— 行过滤期望值自校验 ——")
    counts, ok = [], True
    for label, where, params, expect in checks:
        cur.execute(f"SELECT count(*) FROM alm_issues WHERE {where}", params)
        got = cur.fetchone()[0]
        flag = "PASS" if got == expect else "FAIL"
        ok = ok and got == expect
        counts.append(got)
        logger.info(f"  [{flag}] {label:<26} 期望 {expect:>2} 实际 {got:>2}")

    if len(set(counts)) != len(counts):
        ok = False
        logger.error(f"  [FAIL] 行数出现重复 {counts} —— 夹具失去区分能力，必须调整")
    else:
        logger.info(f"  [PASS] 五个角色行数两两不等 {counts}")

    # 反例：这两个域没有问题单，工程师应该什么都看不到
    for eng_domain in ("热管理域", "整车控制域"):
        cur.execute("SELECT count(*) FROM alm_issues WHERE owner_domain_id = %s", (domain_ids[eng_domain],))
        got = cur.fetchone()[0]
        flag = "PASS" if got == 0 else "FAIL"
        ok = ok and got == 0
        logger.info(f"  [{flag}] 反例 {eng_domain:<22} 期望  0 实际 {got:>2}")

    cur.close()
    conn.close()

    logger.info(f"问题单夹具完成：{len(ISSUES)} 条")
    if not ok:
        raise SystemExit("夹具自校验未通过")


if __name__ == "__main__":
    seed()
