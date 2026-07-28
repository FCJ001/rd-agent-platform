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
