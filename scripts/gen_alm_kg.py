# ============================================================
# 诊断图谱生成脚本
# 手写 20 条根因骨架 → LLM 按域扩写 → 校验去重
# 产出：data/raw/alm_kg.json（JSONL，每行一条根因）
# 一份文件同时喂 PostgreSQL + Neo4j + Milvus
#
# 用法: cd rd-agent-platform && python scripts/gen_alm_kg.py --target-causes 60
# 不调 LLM 跑干跑（只写骨架）：python scripts/gen_alm_kg.py --skeleton-only
# ============================================================

import argparse
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.core.config import get_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = RAW_DIR / "alm_kg.json"

# ---- 责任域（从 seed_domains import，单一来源）----
from scripts.seed_domains import OWNER_DOMAINS  # noqa: E402

# ----------------------------------------------------------
# 20 条根因骨架（手写，作为 LLM 的 few-shot 样例）
# ★ 注意样例里的现象是【交叉复用】的：
#   「续航异常衰减」同时出现在 RC-EV-0001（电芯老化）和 RC-EV-0011（热管理保守）
#   这是刻意设计的，用来向 LLM 示范"现象码要复用"
# ----------------------------------------------------------
SEED_CAUSES = [
    {
        "code": "RC-EV-0001", "name": "动力电池电芯容量衰减", "domain": "电池系统域", "business_line": "ev",
        "description": "电芯经历大量循环后活性物质损失，可用容量下降",
        "cause": "循环次数累积 + 高温工况 + 频繁大倍率快充加速 SEI 膜增厚",
        "prevent": "限制单日快充次数，高温环境预冷后充电",
        "fix_way": "整包更换,单模组更换,BMS 容量学习重标定",
        "fix_duration": "3-7个工作日", "fix_success_rate": "92%",
        "easy_hit": "行驶里程 8 万公里以上、长期依赖直流快充的运营车辆",
        "cost_money": "8000-45000元",
        "verify_items": "读取BMS循环次数,测单体容量一致性,导出SOH历史曲线",
        "phenomena": ["续航异常衰减", "SOH数值偏低", "满电续航显示下降"],
        "dtc": ["P0A7F"],
    },
    {
        "code": "RC-EV-0002", "name": "电池单体压差过大", "domain": "电池系统域", "business_line": "ev",
        "description": "模组内单体电压离散度超标，触发 BMS 均衡或限功率",
        "cause": "单体一致性差 + 均衡策略失效 + 采样线束接触电阻异常",
        "prevent": "定期做静置压差检测，出厂加严分选",
        "fix_way": "主动均衡,更换离群单体,采样线束复检",
        "fix_duration": "2-4个工作日", "fix_success_rate": "88%",
        "easy_hit": "低温地区、长期浅充浅放的车辆",
        "cost_money": "3000-15000元",
        "verify_items": "静置2小时后测单体压差,检查采样线束阻值,读均衡记录",
        "phenomena": ["动力输出受限", "SOH数值偏低", "充电中止"],
        "dtc": ["P0AFA", "P0A80"],
    },
    {
        "code": "RC-EV-0003", "name": "BMS SOC 估算偏差", "domain": "电池系统域", "business_line": "ev",
        "description": "SOC 显示值与实际容量偏差过大，导致续航估算不准",
        "cause": "电流积分累积误差 + OCV-SOC 曲线老化偏移 + 温度补偿不足",
        "prevent": "定期做满充满放校准，升级 SOC 算法模型",
        "fix_way": "SOC 重标定,BMS 软件升级,更换采样芯片",
        "fix_duration": "1-3个工作日", "fix_success_rate": "90%",
        "easy_hit": "长期浅充浅放、极少满充的车辆",
        "cost_money": "0-3000元",
        "verify_items": "满充满放记录 SOC 跳变点,对比 OCV 曲线,检查电流传感器零点漂移",
        "phenomena": ["续航异常衰减", "SOC跳变", "满电续航显示下降"],
        "dtc": ["P0A7F", "P0A80"],
    },
    {
        "code": "RC-EV-0004", "name": "电池包绝缘故障", "domain": "电池系统域", "business_line": "ev",
        "description": "电池包对地绝缘电阻低于安全阈值",
        "cause": "电芯漏液/冷凝水积聚/高压接插件密封失效/底部磕碰导致壳体微裂纹",
        "prevent": "定期绝缘检测，涉水后烘干检查，加强底部防护板",
        "fix_way": "排查漏电点并更换故障模组,清洁并烘干高压回路,更换密封件",
        "fix_duration": "5-15个工作日", "fix_success_rate": "75%",
        "easy_hit": "南方多雨地区、有底部磕碰史的车辆",
        "cost_money": "5000-50000元",
        "verify_items": "绝缘摇表逐段排查,拆底护板视觉检查,气密性测试",
        "phenomena": ["绝缘报警灯亮", "无法上高压", "充电中止"],
        "dtc": ["P0A0F", "P1AF0"],
    },
    {
        "code": "RC-EV-0011", "name": "电机控制器 IGBT 过热降功率", "domain": "电驱系统域", "business_line": "ev",
        "description": "持续大电流输出时 IGBT 结温超限，触发降功率保护",
        "cause": "冷却回路流量不足 + IGBT 开关损耗偏大 + 散热器翅片堵塞",
        "prevent": "定期检查冷却液液位和泵流量，优化 PWM 策略",
        "fix_way": "更换IGBT模块,清洗散热器,升级MCU散热策略固件",
        "fix_duration": "3-5个工作日", "fix_success_rate": "85%",
        "easy_hit": "高温环境连续急加速/爬坡的车辆",
        "cost_money": "5000-20000元",
        "verify_items": "读MCU温度日志,检查冷却液流量,红外测IGBT实际结温",
        "phenomena": ["动力输出受限", "仪表功率限制提示", "加速无力"],
        "dtc": ["P0A2D", "P0A3F"],
    },
    {
        "code": "RC-EV-0012", "name": "减速器异响", "domain": "电驱系统域", "business_line": "ev",
        "description": "减速器齿轮啮合产生异常噪声，低速/滑行时明显",
        "cause": "齿轮加工精度不足/轴承预紧力不当/润滑不良/齿面点蚀",
        "prevent": "出厂加严 NVH 台架检测，定期更换减速器油",
        "fix_way": "更换减速器总成,更换输入轴轴承,重新调整预紧力",
        "fix_duration": "2-5个工作日", "fix_success_rate": "90%",
        "easy_hit": "行驶 5 万公里以上车辆",
        "cost_money": "3000-12000元",
        "verify_items": "路试录音频谱分析,举升机空转听诊,放油检查金属屑",
        "phenomena": ["驱动系统异响", "滑行啸叫声", "加速抖动"],
        "dtc": [],
    },
    {
        "code": "RC-EV-0021", "name": "直流快充枪握手失败", "domain": "充电系统域", "business_line": "ev",
        "description": "插入快充枪后无法建立通信，充电桩报协议超时",
        "cause": "CCS 通信协议栈版本不匹配/CP 信号线接触不良/充电座温度传感器故障",
        "prevent": "充电协议兼容性测试覆盖主流桩品牌，CP 信号回路增加自检",
        "fix_way": "升级 OBC 通信固件,更换充电座总成,清洁 CP 端子",
        "fix_duration": "1-3个工作日", "fix_success_rate": "93%",
        "easy_hit": "使用老旧版本快充桩、充电座有腐蚀的车辆",
        "cost_money": "0-3000元",
        "verify_items": "读取充电握手日志,测量 CP 信号波形,跨桩交叉验证",
        "phenomena": ["充电中止", "无法上高压", "充电速度低于预期"],
        "dtc": ["P0D3F", "U0155"],
    },
    {
        "code": "RC-EV-0022", "name": "OBC 车载充电机效率下降", "domain": "充电系统域", "business_line": "ev",
        "description": "交流慢充功率明显低于标称值，充电时间延长 30% 以上",
        "cause": "PFC 电路老化/DC-DC 变压器磁芯退化/散热不良导致降功率",
        "prevent": "充电机进气格栅定期清洁，避免长期满功率充电",
        "fix_way": "更换OBC模块,清洁散热风道,升级充电策略固件",
        "fix_duration": "2-4个工作日", "fix_success_rate": "87%",
        "easy_hit": "长期使用 220V 插座充电、环境粉尘多的车辆",
        "cost_money": "5000-15000元",
        "verify_items": "对比充电功率与标称值,检查 OBC 温度日志,测量输入电流谐波",
        "phenomena": ["充电速度低于预期", "充电中止", "充电口过热"],
        "dtc": ["P0D3F"],
    },
    {
        "code": "RC-EV-0031", "name": "热管理策略过度保守", "domain": "热管理域", "business_line": "ev",
        "description": "低温或高温工况下标定阈值偏保守，提前限功率、提前停充",
        "cause": "标定表边界值取值过于安全，未按实车工况做二次标定",
        "prevent": "分区域气候做标定验证，收集实车温度分布",
        "fix_way": "重标定热管理MAP,OTA下发新标定包",
        "fix_duration": "5-10个工作日", "fix_success_rate": "80%",
        "easy_hit": "北方冬季、南方夏季高温连续快充场景",
        "cost_money": "0元（OTA）",
        "phenomena": ["续航异常衰减", "动力输出受限", "充电速度低于预期"],
        "dtc": ["P0A2D"],
    },
    {
        "code": "RC-EV-0032", "name": "空调压缩机不启动", "domain": "热管理域", "business_line": "ev",
        "description": "高温天气开启空调后压缩机不工作，出风口为常温",
        "cause": "压缩机控制器低压供电异常/高压互锁回路断开/冷媒泄漏触发低压保护",
        "prevent": "定期检查冷媒压力和高压接插件状态",
        "fix_way": "更换压缩机控制器,补充冷媒并查漏,修复高压互锁回路",
        "fix_duration": "2-5个工作日", "fix_success_rate": "82%",
        "easy_hit": "夏季高温暴晒后首次开空调的车辆",
        "cost_money": "2000-8000元",
        "verify_items": "读空调控制器故障码,测量高低压侧压力,检查压缩机供电电压",
        "phenomena": ["空调不制冷", "仪表功率限制提示", "无法上高压"],
        "dtc": ["P0A0F", "P0AFA"],
    },
    {
        "code": "RC-EV-0041", "name": "VCU 能量回收策略异常", "domain": "整车控制域", "business_line": "ev",
        "description": "松油门后回收功率远低于标称值，或无回收",
        "cause": "电池 SOC 过高禁止回收/低温限功率/制动踏板信号误判",
        "prevent": "优化回收策略的 SOC-温度 MAP，增加踏板意图识别冗余",
        "fix_way": "升级VCU固件,标定回收MAP,检查制动踏板传感器",
        "fix_duration": "1-3个工作日", "fix_success_rate": "90%",
        "easy_hit": "满电出发、冬季冷车行驶的车辆",
        "cost_money": "0元（OTA）",
        "verify_items": "读 VCU 回收功率日志,检查 SOC-温度边界条件,路试验证",
        "phenomena": ["续航异常衰减", "加速抖动", "仪表功率限制提示"],
        "dtc": [],
    },
    {
        "code": "RC-IA-0001", "name": "座舱SoC内存泄漏导致HMI卡死", "domain": "智能座舱域", "business_line": "ia",
        "description": "中控应用长时间运行后内存持续增长，最终 OOM 触发看门狗复位",
        "cause": "地图/多媒体进程未释放纹理缓存，长时间不熄火累积",
        "prevent": "灰度阶段加内存水位埋点，超阈值主动重启服务",
        "fix_way": "修复应用内存泄漏,OTA推送补丁,临时方案双拨轮重启",
        "fix_duration": "10-20个工作日", "fix_success_rate": "85%",
        "easy_hit": "长途连续行驶 4 小时以上、常驻导航+音乐的用户",
        "cost_money": "0元（OTA）",
        "verify_items": "导出座舱dmesg,监控内存水位曲线,复现连续运行4小时",
        "phenomena": ["中控屏显示异常", "系统响应迟滞", "语音交互失效"],
        "dtc": [],
    },
    {
        "code": "RC-IA-0002", "name": "中控屏触控失灵", "domain": "智能座舱域", "business_line": "ia",
        "description": "触摸屏局部或全部区域无响应，重启后暂时恢复",
        "cause": "触控 IC 固件静电闩锁/屏幕排线接触不良/触摸屏老化",
        "prevent": "触控 IC 增加看门狗自动复位，排线增加固定胶",
        "fix_way": "升级触控IC固件,更换触摸屏总成,重新插拔排线",
        "fix_duration": "1-3个工作日", "fix_success_rate": "90%",
        "easy_hit": "干燥季节静电多发、屏幕已使用 3 年以上的车辆",
        "cost_money": "1000-5000元",
        "verify_items": "触摸屏自检测试,检查排线连接,复现静电环境",
        "phenomena": ["中控屏显示异常", "系统响应迟滞", "语音交互失效"],
        "dtc": [],
    },
    {
        "code": "RC-IA-0011", "name": "ACC 雷达失准导致跟车距离异常", "domain": "智能驾驶域", "business_line": "ia",
        "description": "自适应巡航跟车距离与设定值偏差过大，过近或过远",
        "cause": "前雷达安装支架变形/雷达标定数据丢失/雷达罩积雪结冰",
        "prevent": "雷达标定纳入保养项目，冬季增加雷达加热功能",
        "fix_way": "重新标定雷达,校正安装支架,升级融合算法降低单传感器权重",
        "fix_duration": "1-2个工作日", "fix_success_rate": "88%",
        "easy_hit": "经历过轻微追尾/前保险杠维修的车辆",
        "cost_money": "500-3000元",
        "verify_items": "诊断仪读取雷达标定偏差角,路试验证跟车距离,检查支架变形量",
        "phenomena": ["ACC跟车异常", "ADAS功能受限提示", "系统响应迟滞"],
        "dtc": ["U0100", "U0155"],
    },
    {
        "code": "RC-IA-0012", "name": "AEB 误触发", "domain": "智能驾驶域", "business_line": "ia",
        "description": "前方无实际障碍物时自动紧急制动意外触发",
        "cause": "毫米波雷达鬼影目标/视觉感知误检/融合算法置信度阈值偏低",
        "prevent": "融合算法提高确认门槛，增加目标生命周期的时域验证",
        "fix_way": "升级感知融合固件,调整AEB触发阈值,OTA推送",
        "fix_duration": "5-15个工作日", "fix_success_rate": "75%",
        "easy_hit": "隧道出入口、金属护栏密集路段",
        "cost_money": "0元（OTA）",
        "verify_items": "导出事件前后 30 秒传感器原始数据,回灌仿真复现,检查目标列表",
        "phenomena": ["ADAS功能受限提示", "系统响应迟滞", "制动异响"],
        "dtc": ["U0100"],
    },
    {
        "code": "RC-IA-0021", "name": "T-Box 4G 模块频繁掉网", "domain": "车联网域", "business_line": "ia",
        "description": "T-Box 蜂窝网络频繁断开重连，远程控车不可用",
        "cause": "4G 模块固件兼容性/基站切换乒乓效应/天线增益不足",
        "prevent": "优化 PLMN 选网策略，增加信号质量过滤",
        "fix_way": "升级4G模块固件,更换天线总成,增加eSIM多运营商切换",
        "fix_duration": "1-5个工作日", "fix_success_rate": "82%",
        "easy_hit": "地库/山区等弱信号区域的车辆",
        "cost_money": "0-2000元",
        "verify_items": "导出T-Box网络日志,测量天线驻波比,统计掉网频率和时长",
        "phenomena": ["远程控车超时", "语音交互失效", "系统响应迟滞"],
        "dtc": ["U0073"],
    },
    {
        "code": "RC-IA-0022", "name": "车联网云端推送延迟", "domain": "车联网域", "business_line": "ia",
        "description": "云端下发的指令（如远程开空调）延迟超过 30 秒或失败",
        "cause": "MQTT 长连接心跳超时/云平台消息队列积压/T-Box 休眠唤醒时序错误",
        "prevent": "MQTT keepalive 调优，增加推送ACK超时重传",
        "fix_way": "升级T-Box通信协议栈,云端调整MQTT QoS等级,优化休眠唤醒策略",
        "fix_duration": "3-10个工作日", "fix_success_rate": "78%",
        "easy_hit": "车辆长时间静置后首次远程控车",
        "cost_money": "0元（OTA）",
        "verify_items": "对比云端/车端消息时间戳,检查MQTT session状态,模拟弱网环境",
        "phenomena": ["远程控车超时", "系统响应迟滞"],
        "dtc": ["U0073"],
    },
    {
        "code": "RC-IA-0031", "name": "OTA 升级包下载中断", "domain": "OTA升级域", "business_line": "ia",
        "description": "升级包下载到中途失败或校验和不匹配，升级流程中断",
        "cause": "CDN 节点故障/车辆 4G 信号波动/T-Box 存储空间不足/断点续传逻辑缺陷",
        "prevent": "增加 CDN 多节点容灾，升级前预检存储空间和信号强度",
        "fix_way": "重试下载,切换CDN节点,清理T-Box缓存,更换T-Box",
        "fix_duration": "1-3个工作日", "fix_success_rate": "95%",
        "easy_hit": "地库/高速移动中的车辆",
        "cost_money": "0元（OTA）",
        "verify_items": "检查T-Box剩余空间,导出下载日志,手动切换CDN节点测试",
        "phenomena": ["OTA升级失败", "系统响应迟滞"],
        "dtc": [],
    },
    {
        "code": "RC-IA-0032", "name": "差分包校验失败回滚", "domain": "OTA升级域", "business_line": "ia",
        "description": "升级包安装后校验不通过，自动回滚到上一个版本",
        "cause": "基线版本不匹配/差分包制作时遗漏依赖/目标ECU存储有坏块",
        "prevent": "升级前全量版本一致性校验，差分包增加冗余校验码",
        "fix_way": "回滚后重新制作差分包,改用全量包升级,更换故障ECU",
        "fix_duration": "1-5个工作日", "fix_success_rate": "90%",
        "easy_hit": "跨多个大版本的升级、有过自行刷写历史的ECU",
        "cost_money": "0元（OTA）",
        "verify_items": "对比差分包MD5与云端记录,检查目标ECU版本号,读回滚日志",
        "phenomena": ["OTA升级失败", "系统响应迟滞", "版本号显示异常"],
        "dtc": [],
    },
    {
        "code": "RC-IA-0041", "name": "网关 CAN 总线负载率过高", "domain": "GW网关", "business_line": "ia",
        "description": "GW 网关 CAN 总线负载率持续超过 80%，导致报文延迟或丢失",
        "cause": "某 ECU 异常发包/周期报文频率配置过高/新增节点未做总线负载评估",
        "prevent": "网络设计阶段预留 30% 负载余量，量产前做 worst-case 总线负载测试",
        "fix_way": "优化异常ECU的报文发送策略,调整周期报文频率,增加CAN FD分段",
        "fix_duration": "5-15个工作日", "fix_success_rate": "75%",
        "easy_hit": "高配车型（ECU 数量多）、ADAS 功能全开的车辆",
        "cost_money": "0元（OTA）",
        "verify_items": "CAN 总线记录仪采集 1 小时,统计各节点负载占比,定位异常报文源",
        "phenomena": ["系统响应迟滞", "中控屏显示异常", "ADAS功能受限提示"],
        "dtc": ["U0073"],
    },
]


# ---- pydantic schema for LLM structured output ----
class PhenomenonMeta(BaseModel):
    name: str
    is_core: bool = False
    weight: float = 1.0


class CauseRecord(BaseModel):
    code: str
    name: str
    domain: str
    business_line: str
    description: str
    cause: str = ""
    prevent: str = ""
    fix_way: str = ""
    fix_duration: str = ""
    fix_success_rate: str = ""
    easy_hit: str = ""
    cost_money: str = ""
    verify_items: str = ""
    phenomena: list[str] = []
    phenomena_meta: list[PhenomenonMeta] = []
    dtc: list[str] = []


class CauseBatch(BaseModel):
    """LLM 一次返回 6-8 条根因"""
    causes: list[CauseRecord] = Field(..., min_length=1, max_length=10)


# ----------------------------------------------------------
# LLM 扩写
# ----------------------------------------------------------
EXPAND_PROMPT = """你是汽车研发领域的资深故障诊断专家，负责构建「现象码 → 根因 → 责任域」知识图谱。

【当前责任域】{domain_name}（{business_line}线）
描述：{domain_desc}

【已存在的根因名称（不要重复）】
{existing_names}

【全局现象池（尽量复用现有现象码，不要每条根因都造新现象）】
{phenomenon_pool}

【参考样例（few-shot）】
{seed_examples}

请为「{domain_name}」补充 {target_count} 条新的根因，要求：
1. 现象码尽量从全局现象池里选，一条根因对应 2-4 个现象
2. 至少 1 个现象是多条根因共享的（交叠是图谱诊断价值的来源）
3. 名称不能与「已存在的根因名称」重复
4. 验证项（verify_items）要具体可操作，如「读取XX日志」「测量XX电压」
5. code 格式：RC-{{line.upper()}}-{{seq:04d}}，序号从 {next_seq} 开始

返回 JSON。"""


def _format_seed_for_prompt(domain: str) -> str:
    matches = [c for c in SEED_CAUSES if c.get("domain") == domain]
    if not matches:
        return "（该域暂无样例，参照其他域的格式）"
    lines = []
    for c in matches:
        lines.append(
            f"  {c.get('code', '?')} {c.get('name', '?')}: {c.get('description', '')}\n"
            f"    现象: {', '.join(c.get('phenomena', []))}\n"
            f"    处置: {c.get('fix_way', '')} | 周期: {c.get('fix_duration', '')} | 成功率: {c.get('fix_success_rate', '')}\n"
            f"    验证: {c.get('verify_items', '')}"
        )
    return "\n".join(lines)


def _normalize_cause(raw: dict, domain_info: dict) -> dict:
    """将 LLM 的各种非标准格式统一为 CauseRecord 能接受的 dict。

    LLM 常见变体：
      1. 字段包在 root_cause 里：{"root_cause": "xxx", "phenomena": [...], ...}
      2. verify_items 是 list 而非 string
      3. 缺少 domain / business_line / description / code / name
    """
    # 1. 如果外层只有一个业务 key（如 root_cause），展开它
    if len(raw) == 1 and isinstance(list(raw.values())[0], dict):
        raw = dict(list(raw.values())[0])
    # 2. 如果有 root_cause key，用它填充 name
    if "root_cause" in raw:
        raw = dict(raw)
        raw.setdefault("name", raw.pop("root_cause"))
    # 3. verify_items: list → comma string
    if isinstance(raw.get("verify_items"), list):
        raw["verify_items"] = ", ".join(raw["verify_items"])
    # 4. phenomena: ensure list
    if not isinstance(raw.get("phenomena"), list):
        raw["phenomena"] = []
    # 5. dtc: ensure list
    if not isinstance(raw.get("dtc"), list):
        raw["dtc"] = []
    # 6. fill missing required fields
    raw.setdefault("code", "")
    raw.setdefault("name", raw.get("root_cause", ""))
    raw.setdefault("domain", domain_info["name"])
    raw.setdefault("business_line", domain_info["business_line"])
    raw.setdefault("description", raw.get("cause", raw.get("name", "")))
    raw.setdefault("fix_way", "")
    raw.setdefault("fix_duration", "")
    raw.setdefault("fix_success_rate", "")
    raw.setdefault("easy_hit", "")
    raw.setdefault("cost_money", "")
    raw.setdefault("verify_items", "")
    raw.setdefault("cause", "")
    raw.setdefault("prevent", "")
    return raw


def expand_with_llm(all_causes: list[dict], target_per_domain: int = 7) -> list[dict]:
    """用 LLM 为每个责任域扩写根因。失败不阻断，返回已有数据。"""
    settings = get_settings()
    if not settings.DASHSCOPE_API_KEY:
        logger.warning("DASHSCOPE_API_KEY 未设置，跳过 LLM 扩写")
        return all_causes

    from langchain_openai import ChatOpenAI

    # ★ DashScope 的 OpenAI 兼容模式不完全支持 with_structured_output。
    #   用 JSON mode（response_format={"type": "json_object"}）替代，
    #   手工解析后用 pydantic 校验，与 structured_output 等价。
    llm = ChatOpenAI(
        model=settings.CHAT_MODEL,
        api_key=settings.DASHSCOPE_API_KEY,
        base_url=settings.BASE_URL_CHAT,
        temperature=0.7,
        timeout=60,
        model_kwargs={"response_format": {"type": "json_object"}},
    )

    existing_names = {c["name"] for c in all_causes}
    all_phenomena = set()
    for c in all_causes:
        all_phenomena.update(c.get("phenomena", []))

    code_seq = len(all_causes) + 1

    for domain_info in OWNER_DOMAINS:
        domain = domain_info["name"]
        domain_existing = [c for c in all_causes if c["domain"] == domain]
        if len(domain_existing) >= target_per_domain:
            logger.info(f"  {domain}: 已有 {len(domain_existing)} ≥ {target_per_domain}，跳过")
            continue

        need = target_per_domain - len(domain_existing)
        logger.info(f"  LLM 扩写 {domain}: 需 {need} 条 ...")

        prompt = EXPAND_PROMPT.format(
            domain_name=domain,
            business_line=domain_info["business_line"],
            domain_desc=domain_info["description"],
            existing_names=", ".join(sorted(c.get("name", "") for c in domain_existing)),
            phenomenon_pool=", ".join(sorted(all_phenomena)) if all_phenomena else "（尚无，请创建 3-5 个现象并复用）",
            seed_examples=_format_seed_for_prompt(domain),
            target_count=need,
            next_seq=code_seq,
        )

        try:
            response = llm.invoke(prompt)
            raw = json.loads(response.content)

            # LLM 可能返回 {"causes": [...]} 或直接返回 [...]
            cause_list = raw.get("causes", raw) if isinstance(raw, dict) else raw
            if isinstance(cause_list, dict):
                # 个别模型返回 {code: {data}}, 取 values
                cause_list = list(cause_list.values())
            if not isinstance(cause_list, list):
                raise ValueError(f"LLM 返回格式异常: {type(cause_list)}")

            # 逐条归一化后再校验
            normalized = [_normalize_cause(item, domain_info) for item in cause_list]
            batch = CauseBatch(causes=normalized)
            for c in batch.causes:
                if c.name not in existing_names:
                    c.code = f"RC-{domain_info['business_line'].upper()}-{code_seq:04d}"
                    code_seq += 1
                    existing_names.add(c.name)
                    all_phenomena.update(c.phenomena)
                    all_causes.append(c.model_dump())
            logger.info(f"    ✓ 新增 {len(batch.causes)} 条")
        except Exception as e:
            logger.error(f"    ✗ LLM 扩写 {domain} 失败: {e}")

    return all_causes


# ----------------------------------------------------------
# 校验 & 计算 weight / is_core
# ----------------------------------------------------------
def validate_and_enrich(causes: list[dict]) -> list[dict]:
    """统计现象复用次数 → 计算 weight → 判定 is_core"""
    # 每条根因覆盖的现象集合
    cause_phenom_map: list[tuple[int, set[str]]] = [
        (i, set(c.get("phenomena", []))) for i, c in enumerate(causes)
    ]

    # 每个现象出现的根因数
    phenom_freq: Counter[str] = Counter()
    for _, phenoms in cause_phenom_map:
        phenom_freq.update(phenoms)

    for idx, phenoms in cause_phenom_map:
        meta = []
        for p in phenoms:
            refs = phenom_freq[p]
            import math
            weight = round(1.0 / math.sqrt(refs), 3)
            # is_core = 该现象只被这条根因引用（真正的"独有现象"）
            is_core = refs == 1
            meta.append({"name": p, "is_core": is_core, "weight": weight})
        causes[idx]["phenomena_meta"] = meta

    return causes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-causes", type=int, default=60, help="目标根因总数")
    parser.add_argument("--skeleton-only", action="store_true", help="只写骨架，不调 LLM")
    args = parser.parse_args()

    causes: list[dict] = [dict(c) for c in SEED_CAUSES]

    if not args.skeleton_only:
        per_domain = max(1, args.target_causes // len(OWNER_DOMAINS))
        causes = expand_with_llm(causes, target_per_domain=per_domain)
    else:
        logger.info("--skeleton-only 模式：仅使用手写 20 条骨架")

    causes = validate_and_enrich(causes)

    # 输出为 JSONL，每行一条根因
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for c in causes:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    logger.info(f"输出: {OUT_PATH} ({len(causes)} 条根因)")

    # 统计
    all_ph = set()
    for c in causes:
        all_ph.update(c.get("phenomena", []))
    core_count = sum(1 for c in causes for p in c.get("phenomena_meta", []) if p.get("is_core"))
    logger.info(f"  现象码总数: {len(all_ph)}（核心 {core_count} 个）")


if __name__ == "__main__":
    main()
