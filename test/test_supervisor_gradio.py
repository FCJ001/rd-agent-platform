"""RD Agent Platform — Supervisor + Workers 测试控制台"""

import asyncio
import json
import time
import uuid

import gradio as gr
import httpx

BASE = "http://localhost:8000"


# ═══════════════════════════════════════════════════════════════
# API
# ═══════════════════════════════════════════════════════════════

async def _call_chat(user_id, session_id, message):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{BASE}/api/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": message},
        )
        return resp.json()["data"]["reply"]


def call_chat(user_id, session_id, message):
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{BASE}/api/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": message},
        )
        return resp.json()["data"]["reply"]


def call_chat_stream(user_id, session_id, message):
    async def _stream():
        async with httpx.AsyncClient(timeout=120) as client:
            async with client.stream(
                "POST", f"{BASE}/api/v1/chat/stream",
                json={"user_id": user_id, "session_id": session_id, "message": message},
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
# Test cases
# ═══════════════════════════════════════════════════════════════

CASES = {
    "triage":  "车机黑屏，DTC码U0100，高速上出现过两次",
    "follow":  "触控也失灵了，原厂半年了，没有其他异常",
    "impact":  "分析变更影响：升级网关MCU固件到v3.2.1，目标基线2025Q3",
    "report":  "解读台架测试报告：电池SOH 72%，单体压差 85mV，HMI冷启动 9200ms，绝缘电阻 0.5M\u03A9",
    "memory":  "上个月也出现过黑屏问题，帮我查下之前的诊断结果",
}


# ═══════════════════════════════════════════════════════════════
# Handlers
# ═══════════════════════════════════════════════════════════════

def on_send(message, uid, sid, history):
    if not message.strip():
        return history, sid, None, ""
    if not sid:
        sid = f"s_{uuid.uuid4().hex[:6]}"
    history = list(history) if history else []
    history.append({"role": "user", "content": message})

    reply = call_chat_stream(uid, sid, message)
    history.append({"role": "assistant", "content": reply})

    agent = _detect_agent(reply)
    return history, sid, agent, ""


def on_quick_test(key, uid, sid, history):
    if key == "follow":
        pass
    else:
        sid = f"s_{uuid.uuid4().hex[:6]}"
    history = list(history) if history else []
    history.append({"role": "user", "content": CASES[key]})

    reply = call_chat_stream(uid, sid, CASES[key])
    history.append({"role": "assistant", "content": reply})

    agent = _detect_agent(reply)
    return history, sid, agent


def on_full_test(uid):
    lines = []
    t_total = time.time()

    sid1 = f"f_{uuid.uuid4().hex[:4]}"

    for i, (label, key) in enumerate([
        ("Triage \u5206\u8bca", "triage"),
        ("Triage \u8ffd\u95ee (bypass)", "follow"),
        ("Memory \u8bb0\u5fc6\u53ec\u56de", "memory"),
        ("Impact \u53d8\u66f4\u5f71\u54cd\u5206\u6790", "impact"),
        ("Report \u62a5\u544a\u89e3\u8bfb", "report"),
    ]):
        t0 = time.time()
        sid = sid1 if key == "follow" else f"f_{uuid.uuid4().hex[:4]}"
        reply = call_chat(uid, sid, CASES[key])
        elapsed = time.time() - t0
        status = "\u2705" if elapsed < 30 else "\u23f3"
        lines.append(f"### {status} R{i+1}: {label}  ({elapsed:.1f}s)\n\n{reply[:400]}")

    lines.append(f"\n\n> \u603b\u8017\u65f6: **{time.time() - t_total:.0f}s**")
    return "\n\n".join(lines)


def _detect_agent(reply):
    if "\u7f6e\u4fe1\u5ea6" in reply or ("\u8bca\u65ad" in reply and "\u6839\u56e0" in reply):
        return "\u5206\u8bca Agent"
    if "\u98ce\u9669" in reply and ("\u53d8\u66f4" in reply or "\u57fa\u7ebf" in reply):
        return "\u5f71\u54cd\u5206\u6790 Agent"
    if "\u6307\u6807" in reply and "\u5f02\u5e38" in reply:
        return "\u62a5\u544a\u89e3\u8bfb Agent"
    if "\u5386\u53f2" in reply or "\u8bb0\u5fc6" in reply:
        return "Supervisor (Memory)"
    return "Supervisor"


# ═══════════════════════════════════════════════════════════════
# UI
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
.header span { color: #6b7280; font-size: 0.875rem; }

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

.status-bar {
    padding: 6px 12px;
    background: #f9fafb;
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    font-size: 0.8rem;
    color: #6b7280;
    margin-bottom: 8px;
}
.status-bar .agent-name { color: #111827; font-weight: 500; }

/* Compact buttons in test panel */
.test-panel button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    border-radius: 6px !important;
    margin-bottom: 6px !important;
}
"""


def build_ui():
    with gr.Blocks(title="RD Agent Platform") as demo:

        # ── Header ──
        with gr.Row(elem_classes="header"):
            gr.HTML('<h2>RD Agent Platform <span style="font-weight:400;color:#6b7280;">\u00b7 Test Console</span></h2>')

        # ── Session bar ──
        with gr.Row():
            uid = gr.Textbox(label="User ID", value="test", scale=2, show_label=False, container=False)
            sid = gr.Textbox(label="Session", placeholder="Auto-generated", scale=4, show_label=False, container=False)
            new_btn = gr.Button("\u65b0\u4f1a\u8bdd", size="sm", scale=1)
            clear_btn = gr.Button("\u6e05\u5c4f", size="sm", scale=1)

        new_btn.click(lambda: (f"u_{uuid.uuid4().hex[:6]}", f"s_{uuid.uuid4().hex[:6]}"), inputs=[], outputs=[uid, sid])

        # ── Agent status ──
        agent_label = gr.Textbox(value="\u5c31\u7eea", interactive=False, show_label=False, container=False, elem_classes="status-bar")

        # ── Main: chat + panel ──
        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="", height=500, show_label=False)
                msg = gr.Textbox(
                    placeholder="\u8f93\u5165\u6d88\u606f\u2026  \u4f8b\u5982\uff1a\u8f66\u673a\u9ed1\u5c4f\uff0cDTC\u7801U0100 / \u5206\u6790\u53d8\u66f4\u5f71\u54cd / \u89e3\u8bfb\u6d4b\u8bd5\u62a5\u544a",
                    show_label=False, container=False,
                )

            with gr.Column(scale=1, elem_classes="test-panel"):
                gr.HTML('<h4>\u5feb\u901f\u6d4b\u8bd5</h4>')
                btn_triage = gr.Button("\U0001f52c \u5206\u8bca\u8bca\u65ad", variant="secondary")
                btn_impact = gr.Button("\U0001f4ca \u53d8\u66f4\u5f71\u54cd\u5206\u6790", variant="secondary")
                btn_report = gr.Button("\U0001f4cb \u62a5\u544a\u89e3\u8bfb", variant="secondary")
                btn_memory = gr.Button("\U0001f9e0 \u8de8\u4f1a\u8bdd\u8bb0\u5fc6", variant="secondary")
                btn_full = gr.Button("\u25b6 \u4e00\u952e\u5b8c\u6574\u6d41\u7a0b", variant="primary")

        # ── Full test output ──
        with gr.Row():
            full_output = gr.Textbox(
                label="", lines=20, max_lines=40, show_label=False,
                placeholder="\u70b9\u51fb\u300c\u4e00\u952e\u5b8c\u6574\u6d41\u7a0b\u300d\u67e5\u770b 5 \u8f6e\u7aef\u5230\u7aef\u6d4b\u8bd5\u7ed3\u679c\u2026",
                container=False,
            )

        # ── Events ──
        msg.submit(on_send, [msg, uid, sid, chatbot], [chatbot, sid, agent_label, msg])

        btn_triage.click(lambda u, s, h: on_quick_test("triage", u, s, h), [uid, sid, chatbot], [chatbot, sid, agent_label])
        btn_impact.click(lambda u, s, h: on_quick_test("impact", u, s, h), [uid, sid, chatbot], [chatbot, sid, agent_label])
        btn_report.click(lambda u, s, h: on_quick_test("report", u, s, h), [uid, sid, chatbot], [chatbot, sid, agent_label])
        btn_memory.click(lambda u, s, h: on_quick_test("memory", u, s, h), [uid, sid, chatbot], [chatbot, sid, agent_label])

        clear_btn.click(lambda: ([], "", "\u5c31\u7eea"), inputs=[], outputs=[chatbot, sid, agent_label])

        btn_full.click(on_full_test, [uid], [full_output])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
