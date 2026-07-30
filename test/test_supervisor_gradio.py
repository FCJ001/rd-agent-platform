"""RD Agent Platform — Supervisor + Workers 测试控制台"""

import asyncio
import json
import os
import sys
import time
import uuid

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr
import httpx

from src.core.config import get_settings

BASE = "http://localhost:8000"
settings = get_settings()

# ═══════════════════════════════════════════════════════════════
# 操作按钮 HTML（注入到对话气泡底部）
# ═══════════════════════════════════════════════════════════════

_ACTION_HTML = """
<div style="margin-top:14px;padding:10px 14px;background:#f0f6ff;
border-radius:8px;border:1px solid #c5d9f0;user-select:none;">
  <div style="font-size:0.8rem;color:#6b7280;margin-bottom:8px;">请选择操作：</div>
  <button onclick="triggerAction('create')" style="
    display:inline-block;margin:0 8px 6px 0;padding:6px 16px;
    background:#2563eb;color:#fff;border:none;border-radius:6px;
    font-size:0.85rem;cursor:pointer;font-weight:500;
  ">📝 创建问题单</button>
  <button onclick="triggerAction('platform')" style="
    display:inline-block;margin:0 8px 6px 0;padding:6px 16px;
    background:#fff;color:#374151;border:1px solid #d1d5db;border-radius:6px;
    font-size:0.85rem;cursor:pointer;font-weight:500;
  ">🔗 跳转平台</button>
  <button onclick="triggerAction('close')" style="
    display:inline-block;margin:0 8px 6px 0;padding:6px 16px;
    background:#fff;color:#374151;border:1px solid #d1d5db;border-radius:6px;
    font-size:0.85rem;cursor:pointer;font-weight:500;
  ">✅ 建议结案</button>
</div>
"""

# JavaScript: 点击按钮 → 写隐藏 textbox → 触发 Gradio 事件
_ACTION_JS = """
<script>
function triggerAction(action) {
    const tb = document.querySelector('#action_trigger textarea, #action_trigger input');
    if (tb) {
        const nativeInputValueSetter = Object.getOwnPropertyDescriptor(
            window.HTMLInputElement.prototype, 'value'
        ).set;
        nativeInputValueSetter.call(tb, action);
        tb.dispatchEvent(new Event('input', { bubbles: true }));
    }
}
</script>
"""


# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════

def call_chat(user_id, session_id, message, role="engineer"):
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{BASE}/api/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": message, "role": role},
        )
        return resp.json()["data"]["reply"]


def call_chat_stream(user_id, session_id, message, role="engineer"):
    async def _stream():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{BASE}/api/v1/chat/stream",
                json={"user_id": user_id, "session_id": session_id, "message": message, "role": role},
            ) as resp:
                buffer = ""
                full = ""
                async for chunk in resp.aiter_bytes():
                    buffer += chunk.decode("utf-8")
                    while "\n\n" in buffer:
                        line, buffer = buffer.split("\n\n", 1)
                        line = line.strip()
                        if not line.startswith("data: "):
                            continue
                        try:
                            d = json.loads(line[6:])
                            if d["type"] == "token":
                                full += d["content"]
                            elif d["type"] == "done":
                                return full
                            elif d["type"] == "error":
                                return f"\u26a0 {d['message']}"
                        except json.JSONDecodeError:
                            pass
                return full
    return asyncio.run(_stream())


# ═══════════════════════════════════════════════════════════════
# 判断是否需要追加操作按钮
# ═══════════════════════════════════════════════════════════════

def _should_show_actions(reply: str) -> bool:
    """检测 Agent 是否主动询问操作选项（唯一关键词，避免与分诊追问冲突）。"""
    return "接下来需要我做什么" in reply


def _maybe_append_actions(reply: str) -> str:
    """如果是分诊收敛结论，在气泡底部追加操作按钮。"""
    if _should_show_actions(reply):
        return reply + _ACTION_HTML
    return reply


# ═══════════════════════════════════════════════════════════════
# Test cases
# ═══════════════════════════════════════════════════════════════

CASES = {
    "triage": "车机黑屏，DTC码U0100，高速上出现过两次",
    "follow": "触控也失灵了，原厂半年了，没有其他异常",
    "impact": "分析变更影响：升级网关MCU固件到v3.2.1，目标基线2025Q3",
    "report": "解读台架测试报告：电池SOH 72%，单体压差 85mV，HMI冷启动 9200ms，绝缘电阻 0.5M\u03a9",
    "memory": "上个月也出现过黑屏问题，帮我查下之前的诊断结果",
    "dedup": "车机中控屏黑屏，系统响应慢，DTC码U0100",
    "knowledge": "电池健康度SOH低于多少需要更换？绝缘电阻标准是多少？",
    "operation": "最近一个月各域的缺陷密度对比",
}


# ═══════════════════════════════════════════════════════════════
# Handlers
# ═══════════════════════════════════════════════════════════════

def on_send(message, uid, sid, history, role):
    if not message.strip():
        return history, sid, "", role
    if not sid:
        sid = f"s_{uuid.uuid4().hex[:6]}"
    history = list(history) if history else []
    history.append({"role": "user", "content": message})

    reply = call_chat_stream(uid, sid, message, role)
    reply = _maybe_append_actions(reply)
    history.append({"role": "assistant", "content": reply})

    return history, sid, "", role


def on_quick_test(key, uid, sid, history, role):
    if key == "follow":
        pass
    else:
        sid = f"s_{uuid.uuid4().hex[:6]}"
    history = list(history) if history else []
    history.append({"role": "user", "content": CASES[key]})

    reply = call_chat_stream(uid, sid, CASES[key], role)
    reply = _maybe_append_actions(reply)
    history.append({"role": "assistant", "content": reply})

    return history, sid, role


def on_action_trigger(action, uid, sid, history, role):
    """隐藏 textbox 触发：用户点了气泡里的按钮。"""
    if not sid or not action:
        return history, sid, role

    history = list(history) if history else []

    if action == "create":
        message = "创建"
    elif action == "close":
        message = "结案"
    elif action == "platform":
        history.append({"role": "user", "content": "跳转"})
        history.append({"role": "assistant", "content": f"ALM 平台地址：{settings.PLATFORM_ALM_URL}/issues\n可直接在平台查看或创建问题单。"})
        return history, sid, role
    else:
        return history, sid, role

    history.append({"role": "user", "content": message})
    reply = call_chat_stream(uid, sid, message, role)
    history.append({"role": "assistant", "content": reply})
    return history, sid, role


def on_full_test(uid, role):
    lines = []
    t_total = time.time()

    sid1 = f"f_{uuid.uuid4().hex[:4]}"

    for i, (label, key) in enumerate([
        ("Dedup 去重检测", "dedup"),
        ("Triage 分诊", "triage"),
        ("Triage 追问 (bypass)", "follow"),
        ("Memory 记忆召回", "memory"),
        ("Impact 变更影响分析", "impact"),
        ("Report 报告解读", "report"),
        ("Knowledge 知识库搜索", "knowledge"),
        ("Operation 运营BI查询", "operation"),
    ]):
        t0 = time.time()
        sid = sid1 if key == "follow" else f"f_{uuid.uuid4().hex[:4]}"
        reply = call_chat(uid, sid, CASES[key], role)
        elapsed = time.time() - t0
        status = "\u2705" if elapsed < 30 else "\u23f3"
        lines.append(f"### {status} R{i+1}: {label}  ({elapsed:.1f}s)\n\n{reply[:400]}")

    lines.append(f"\n\n> 总耗时: **{time.time() - t_total:.0f}s**")
    return "\n\n".join(lines)


# ═══════════════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════════════

CSS = """
body, .gradio-container { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
footer { display: none !important; }

.header {
    padding: 20px 0 8px 0;
    border-bottom: 1px solid #e5e7eb;
    margin-bottom: 16px;
}
.header h2 { margin: 0; font-weight: 600; color: #111827; font-size: 1.25rem; }

.test-panel {
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 16px;
}
.test-panel h4 {
    margin: 0 0 12px 0;
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.test-panel button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    margin-bottom: 6px !important;
}
"""


# ═══════════════════════════════════════════════════════════════
# UI
# ═══════════════════════════════════════════════════════════════

def build_ui():
    with gr.Blocks(title="RD Agent Platform", head=_ACTION_JS) as demo:

        # ── Header ──
        with gr.Row(elem_classes="header"):
            gr.HTML('<h2>RD Agent Platform <span style="font-weight:400;color:#6b7280;">· Test Console</span></h2>')

        # ── Session bar ──
        with gr.Row():
            uid = gr.Textbox(label="User ID", value="test", scale=2, show_label=False, container=False)
            role_dd = gr.Dropdown(
                choices=["engineer", "business", "aftersales", "customer"],
                value="engineer", label="", show_label=False, scale=1, container=False,
            )
            sid = gr.Textbox(label="Session", placeholder="Auto-generated", scale=3, show_label=False, container=False)
            new_btn = gr.Button("新会话", size="sm", scale=1)
            clear_btn = gr.Button("清屏", size="sm", scale=1)

        new_btn.click(lambda: (f"u_{uuid.uuid4().hex[:6]}", f"s_{uuid.uuid4().hex[:6]}"), inputs=[], outputs=[uid, sid])

        # ── Main: chat + panel ──
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="", height=520, show_label=False, sanitize_html=True)
                msg = gr.Textbox(
                    placeholder="输入消息…  例如：车机黑屏，DTC码U0100 / 分析变更影响 / 解读测试报告",
                    show_label=False, container=False,
                )

            with gr.Column(scale=1, elem_classes="test-panel"):
                gr.HTML('<h4>快速测试</h4>')
                btn_triage = gr.Button("🔬 分诊诊断", variant="secondary")
                btn_impact = gr.Button("📊 变更影响分析", variant="secondary")
                btn_report = gr.Button("📋 报告解读", variant="secondary")
                btn_memory = gr.Button("🧠 跨会话记忆", variant="secondary")
                btn_dedup = gr.Button("🔍 问题去重检测", variant="secondary")
                btn_knowledge = gr.Button("📚 知识库搜索", variant="secondary")
                btn_operation = gr.Button("📊 运营BI查询", variant="secondary")
                btn_full = gr.Button("▶ 一键完整流程", variant="primary")

        # ── 隐藏控件：用于 JS → Python 通信 ──
        action_trigger = gr.Textbox(visible=False, elem_id="action_trigger")

        # ── Full test output ──
        with gr.Row():
            full_output = gr.Textbox(
                label="", lines=20, max_lines=40, show_label=False,
                placeholder="点击「一键完整流程」查看 8 轮端到端测试结果…",
                container=False,
            )

        # ── Events ──
        msg.submit(on_send, [msg, uid, sid, chatbot, role_dd], [chatbot, sid, msg, role_dd])

        btn_triage.click(lambda u, s, h, r: on_quick_test("triage", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])
        btn_impact.click(lambda u, s, h, r: on_quick_test("impact", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])
        btn_report.click(lambda u, s, h, r: on_quick_test("report", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])
        btn_memory.click(lambda u, s, h, r: on_quick_test("memory", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])
        btn_dedup.click(lambda u, s, h, r: on_quick_test("dedup", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])
        btn_knowledge.click(lambda u, s, h, r: on_quick_test("knowledge", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])
        btn_operation.click(lambda u, s, h, r: on_quick_test("operation", u, s, h, r), [uid, sid, chatbot, role_dd], [chatbot, sid, role_dd])

        clear_btn.click(lambda: ([], ""), inputs=[], outputs=[chatbot, sid])

        btn_full.click(on_full_test, [uid, role_dd], [full_output])

        # 操作按钮 → 隐藏 textbox → on_action_trigger
        action_trigger.input(
            on_action_trigger,
            [action_trigger, uid, sid, chatbot, role_dd],
            [chatbot, sid, role_dd],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
