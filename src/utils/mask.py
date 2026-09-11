"""字段脱敏工具。行级过滤在 BaseRepository 做，这里做字段级打码。"""

import re


def mask_phone(phone: str | None) -> str | None:
    """手机号脱敏：138****5678"""
    if not phone:
        return phone
    return re.sub(r"(\d{3})\d{4}(\d{4})", r"\1****\2", str(phone))


def mask_vin(vin: str | None) -> str | None:
    """VIN 脱敏：只留后 6 位，前面用 * 替代"""
    if not vin or len(vin) < 6:
        return vin
    return "*" * (len(vin) - 6) + vin[-6:]


def mask_email(email: str | None) -> str | None:
    """邮箱脱敏：u***@example.com"""
    if not email or "@" not in email:
        return email
    local, domain = email.split("@", 1)
    if len(local) <= 1:
        return f"{local}***@{domain}"
    return f"{local[0]}***@{domain}"


ROLE_MASK_RULES = {
    "customer": {"phone": mask_phone, "vin": mask_vin, "email": mask_email},
    "aftersales": {"phone": mask_phone, "vin": mask_vin},
    "business": {"phone": mask_phone},
    "engineer": {},   # 工程师全量可见
}


def apply_mask(data: dict, role: str) -> dict:
    """按角色对敏感字段打码，原地修改并返回。"""
    rules = ROLE_MASK_RULES.get(role, {})
    if not rules:
        return data
    for key in list(data.keys()):
        if key in rules and data[key]:
            data[key] = rules[key](data[key])
    return data


def redact_sensitive_fields(data: dict, role: str) -> dict:
    """对外暴露的统一脱敏入口。返回脱敏后的新 dict。"""
    return apply_mask(dict(data), role)


# 自由文本中的敏感模式：VIN（17 位，排除 I/O/Q）和大陆手机号
# 注意不能用 \b 定界 —— 中文汉字属于 \w，"号LSV..."之间没有词边界，
# 中文语境下 \b 永远匹配不上；用显式字母数字 lookaround 防止截断更长串
_VIN_RE = re.compile(r"(?<![A-Za-z0-9])[A-HJ-NPR-Z0-9]{17}(?![A-Za-z0-9])")
_PHONE_RE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")


def mask_free_text(text: str | None) -> str | None:
    """对自由文本里的 VIN / 手机号打码。

    用户消息（chat 输入）在进 LLM / 入库前过这里，兜住
    「结构化字段已脱敏、但原文里还带着完整 VIN」的漏网情况。
    """
    if not text:
        return text
    text = _VIN_RE.sub(lambda m: mask_vin(m.group(0)), text)
    text = _PHONE_RE.sub(lambda m: mask_phone(m.group(0)), text)
    return text
