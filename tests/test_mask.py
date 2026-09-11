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
