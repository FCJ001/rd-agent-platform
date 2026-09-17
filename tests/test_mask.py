# ============================================================
# A1 脱敏工具单元测试 —— 纯函数，CI 可跑
#
# mask.py 曾是"零调用"的合规硬伤：工具齐全但没有一层真正在用。
# 这组测试锁住脱敏函数本身的行为；调用点（写入路径 / 进 LLM 前）
# 由代码评审保证：
#   - 写入：sync.upsert_issue / webhook.consume_alm_event → apply_mask
#   - LLM：chat 入口 mask_free_text + load_issue_context 兜底
# ============================================================

from src.utils.mask import (
    apply_mask, mask_email, mask_free_text, mask_phone, mask_vin,
)


def test_mask_vin_keeps_last_six():
    assert mask_vin("LSVAA1234AB567890") == "*" * 11 + "567890"


def test_mask_vin_short_passthrough():
    assert mask_vin("ABC123") == "ABC123"
    assert mask_vin("") == ""
    assert mask_vin(None) is None


def test_mask_phone():
    assert mask_phone("13812345678") == "138****5678"
    assert mask_phone("12345") == "12345"  # 非手机号格式不动


def test_mask_email():
    assert mask_email("zhangsan@example.com") == "z***@example.com"
    assert mask_email("a@b.com") == "a***@b.com"
    assert mask_email("not-an-email") == "not-an-email"


def test_apply_mask_by_role():
    data = {"vin": "LSVAA1234AB567890", "phone": "13812345678", "title": "黑屏"}
    masked = apply_mask(dict(data), "customer")
    assert masked["vin"].endswith("567890")
    assert masked["phone"] == "138****5678"
    assert masked["title"] == "黑屏"  # 非敏感字段不动

    # engineer 全量可见
    assert apply_mask(dict(data), "engineer") == data


def test_mask_free_text_vin_and_phone():
    text = "我的车架号是LSVAA1234AB567890，电话13812345678，中控屏黑屏"
    out = mask_free_text(text)
    assert "LSVAA1234AB567890" not in out
    assert "13812345678" not in out
    assert "567890" in out and "138****5678" in out
    assert "黑屏" in out


def test_mask_free_text_no_false_positive_on_normal_text():
    text = "软件版本 2024.32.5 升级后偶尔黑屏，故障码 U0155"
    assert mask_free_text(text) == text


def test_mask_free_text_empty():
    assert mask_free_text("") == ""
    assert mask_free_text(None) is None


# ── 绕过向量回归（A1 评审修复后锁定）────────────────────────────────────

def test_apply_mask_unknown_role_fails_closed():
    """未知角色必须按最严格策略脱敏，绝不能 fail-open 返回原文。"""
    data = {"vin": "LSVAA1234AB567890", "phone": "13812345678", "title": "黑屏"}
    masked = apply_mask(dict(data), "unknown_role")
    assert masked["vin"].endswith("567890") and not masked["vin"].startswith("LSV")
    assert masked["phone"] == "138****5678"


def test_mask_free_text_lowercase_vin():
    """VIN 不区分大小写（ISO 3779），小写也要打码。"""
    out = mask_free_text("vin: lsvaa1234ab567890 请查一下")
    assert "lsvaa1234ab567890" not in out
    assert "567890" in out


def test_mask_free_text_phone_with_plus86():
    """+86 前缀的手机号同样命中。"""
    out = mask_free_text("联系 +8613812345678")
    assert "8613812345678" not in out.replace("+86", "")  # 原完整号码不残留
    assert "138****5678" in out


def test_mask_free_text_email():
    """自由文本里的邮箱也要打码。"""
    out = mask_free_text("有问题发邮件到 zhangsan@example.com 反馈")
    assert "zhangsan@example.com" not in out
    assert "z***@example.com" in out


def test_mask_free_text_plain_phone_without_plus_untouched_prefix():
    """无前缀号码行为保持：数字串不被截断。"""
    out = mask_free_text("工单号 2024112300012345 不是手机号")
    assert "2024112300012345" in out  # 16 位长数字不是手机号，不误伤

