"""指标标准值 + 代码判阈值（绝不用 LLM 做数值比较）。"""

from enum import Enum
from typing import Literal

from pydantic import BaseModel


class Severity(str, Enum):
    NORMAL = "normal"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricStandard(BaseModel):
    metric_name: str
    business_line: str  # ev | ia
    unit: str | None = None
    compare_mode: Literal["between", "gte", "lte", "eq"] = "between"
    normal_min: float | None = None
    normal_max: float | None = None
    warning_threshold: float | None = None
    critical_threshold: float | None = None
    interpretation_hint: str = ""
    related_cause_codes: list[str] = []
    standard_source: str = ""


# 10 条国标/厂标，Python 常量不建表
METRIC_STANDARDS: list[MetricStandard] = [
    # ── 电动化线 ──
    MetricStandard(
        metric_name="battery_soh", business_line="ev", unit="%",
        compare_mode="gte", normal_min=80.0, critical_threshold=70.0,
        interpretation_hint="电池健康度 SOH 低于 80% 表示容量衰减明显，低于 70% 触发三包更换标准",
        related_cause_codes=["RC-EV-0001"],
        standard_source="GB/T 31484 + 三包规定",
    ),
    MetricStandard(
        metric_name="cell_volt_diff", business_line="ev", unit="mV",
        compare_mode="lte", normal_max=50.0, warning_threshold=50.0, critical_threshold=80.0,
        interpretation_hint="单体压差过大表示电芯一致性恶化，可能触发 BMS 限功率",
        related_cause_codes=["RC-EV-0002"],
        standard_source="维修手册电池诊断章节",
    ),
    MetricStandard(
        metric_name="pack_insulation", business_line="ev", unit="MΩ",
        compare_mode="gte", normal_min=100.0, critical_threshold=1.0,
        interpretation_hint="绝缘电阻低于 1MΩ 属于严重漏电风险，需立即停用并检查",
        related_cause_codes=["RC-EV-0004"],
        standard_source="GB 18384-2020",
    ),
    MetricStandard(
        metric_name="charge_power", business_line="ev", unit="kW",
        compare_mode="gte", normal_min=6.0, warning_threshold=3.0,
        interpretation_hint="充电功率过低可能是充电机、BMS 限功率或热管理策略导致",
        related_cause_codes=["RC-EV-0031"],
        standard_source="充电系统技术条件",
    ),
    MetricStandard(
        metric_name="motor_temp", business_line="ev", unit="°C",
        compare_mode="lte", normal_max=120.0, warning_threshold=120.0, critical_threshold=150.0,
        interpretation_hint="电机温度过高可能导致 IGBT 降功率或保护性断电",
        related_cause_codes=["RC-EV-0011"],
        standard_source="电机控制器规格书",
    ),
    # ── 智能化线 ──
    MetricStandard(
        metric_name="ota_fail_rate", business_line="ia", unit="%",
        compare_mode="lte", normal_max=1.0, critical_threshold=3.0,
        interpretation_hint="OTA 升级失败率超过 3% 需暂停灰度放量并排查失败原因",
        standard_source="OTA 灰度放量规范",
    ),
    MetricStandard(
        metric_name="hmi_cold_boot_ms", business_line="ia", unit="ms",
        compare_mode="lte", normal_max=3500.0, warning_threshold=3500.0, critical_threshold=8000.0,
        interpretation_hint="座舱冷启动超过 8 秒用户可感知延迟，通常为 SoC 内存泄漏或存储碎片化",
        related_cause_codes=["RC-IA-0001"],
        standard_source="座舱性能基线",
    ),
    MetricStandard(
        metric_name="gps_offset", business_line="ia", unit="m",
        compare_mode="lte", normal_max=10.0, warning_threshold=10.0,
        interpretation_hint="定位偏移超过 10m 影响导航精度，可能为天线故障或定位算法参数漂移",
        standard_source="导航系统精度指标",
    ),
    MetricStandard(
        metric_name="voice_wake_rate", business_line="ia", unit="%",
        compare_mode="gte", normal_min=95.0, warning_threshold=90.0,
        interpretation_hint="语音唤醒率低于 90% 用户投诉率高，通常为麦克风阵列或降噪算法问题",
        standard_source="座舱语音交互验收标准",
    ),
    MetricStandard(
        metric_name="adas_false_positive", business_line="ia", unit="次/千公里",
        compare_mode="lte", normal_max=0.5, warning_threshold=1.0, critical_threshold=2.0,
        interpretation_hint="ADAS 误触发频次过高存在安全隐患，需立即排查传感器和控制逻辑",
        related_cause_codes=["RC-IA-0012"],
        standard_source="ADAS 功能安全目标",
    ),
]


def judge(metric_key: str, value: float) -> tuple[bool, Severity]:
    """纯代码判阈值。返回 (是否异常, 严重度)。"""
    standard = None
    for s in METRIC_STANDARDS:
        if s.metric_name == metric_key:
            standard = s
            break
    if standard is None:
        return False, Severity.NORMAL

    mode = standard.compare_mode

    if standard.critical_threshold is not None:
        if mode in ("lte", "eq") and value > standard.critical_threshold:
            return True, Severity.CRITICAL
        if mode == "gte" and value < standard.critical_threshold:
            return True, Severity.CRITICAL

    if standard.warning_threshold is not None:
        if mode in ("lte", "eq") and value > standard.warning_threshold:
            return True, Severity.WARNING
        if mode == "gte" and value < standard.warning_threshold:
            return True, Severity.WARNING

    if standard.normal_min is not None and value < standard.normal_min:
        return True, Severity.WARNING
    if standard.normal_max is not None and value > standard.normal_max:
        return True, Severity.WARNING
    if mode == "between" and standard.normal_min is not None and standard.normal_max is not None:
        if not (standard.normal_min <= value <= standard.normal_max):
            return True, Severity.WARNING

    return False, Severity.NORMAL


def find_standard(metric_key: str) -> MetricStandard | None:
    for s in METRIC_STANDARDS:
        if s.metric_name == metric_key:
            return s
    return None
