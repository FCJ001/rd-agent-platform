"""DTC 故障码提取。

SAE J2012 标准 5 位 DTC：首字母（P/C/B/U，排除 I/O/Q）+ 4 位十六进制，
如 P0A7F、C1B62、U0155。旧正则 [A-Z]\\d{4,5} 只认数字，会漏掉
第三位是字母的真实故障码（P0A7F 匹配不到），导致 dedup 的 DTC 门槛漏配。
"""

import re

# lookaround 用字母数字定界：中文语境下 \b 永远匹配不上（汉字属于 \w）
DTC_CODE_RE = re.compile(r"(?<![A-Za-z0-9])[A-HJ-NPR-Z][0-9A-Fa-f]{4}(?![A-Za-z0-9])")


def extract_dtc_codes(text: str | None) -> list[str]:
    """从自由文本提取 DTC 码，统一转大写。"""
    if not text:
        return []
    return [m.upper() for m in DTC_CODE_RE.findall(text)]
