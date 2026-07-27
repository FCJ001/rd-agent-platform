"""Supervisor + 3 Workers 手动测试页面。
启动：python test/test_supervisor_gradio.py
依赖：pip install gradio httpx
"""

import asyncio
import json
import uuid

import gradio as gr
import httpx

BASE = "http://localhost:8000"


# ══════════════════════════════════════════════════════════════════════
# 核心调用
# ══════════════════════════════════════════════════════════════════════

async def sync_chat(user_id, session_id, message):
    async with httpx.AsyncClient(timeout=120) as client:
        resp = await client.post(
            f"{BASE}/api/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": message},
        )
        data = resp.json()
        return data["data"]["reply"]


def sync_chat_blocking(user_id, session_id, message):
    """同步版本，供 Gradio handler 使用"""
    with httpx.Client(timeout=120) as client:
        resp = client.post(
            f"{BASE}/api/v1/chat",
            json={"user_id": user_id, "session_id": session_id, "message": message},
        )
        data = resp.json()
        return data["data"]["reply"]


async def _stream_chat(user_id, session_id, message):
    """SSE 流式 → 收集完整回复"""
    async with httpx.AsyncClient(timeout=120) as client:
        async with client.stream(
            "POST", f"{BASE}/api/v1/chat/stream",
            json={"user_id": user_id, "session_id": session_id, "message": message},
        ) as resp:
            buffer = ""
            full_reply = ""
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
                            full_reply += d["content"]
                        elif d["type"] == "done":
                            return full_reply
                        elif d["type"] == "error":
                            return f"[错误] {d['message']}"
                    except json.JSONDecodeError:
                        pass
            return full_reply


def stream_chat(user_id, session_id, message):
    """同步包装器"""
    return asyncio.run(_stream_chat(user_id, session_id, message))


# ══════════════════════════════════════════════════════════════════════
# 测试用例
# ══════════════════════════════════════════════════════════════════════

TEST_CASES = {
    "分诊首轮": "车机黑屏，DTC码U0100，高速上出现过两次",
    "分诊追问": "触控也失灵了，原厂半年了，没有其他异常",
    "变更影响": "分析变更影响：升级网关MCU固件到v3.2.1，目标基线2025Q3",
    "报告解读": "解读台架测试报告：电池SOH 72%，单体压差 85mV，HMI冷启动 9200ms，绝缘电阻 0.5MΩ",
    "记忆召回": "上个月也出现过黑屏问题，帮我查下之前的诊断结果",
}


# ══════════════════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════════════════

CSS = """
.header { text-align: center; margin: 20px 0; }
.header h1 { font-size: 1.6em; margin: 0; }
.header p { color: #888; margin: 4px 0 0; }
.worker-tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
              font-size: 0.8em; margin-right: 6px; color: white; }
.tag-triage { background: #10b981; }
.tag-impact { background: #f59e0b; }
.tag-report { background: #3b82f6; }
.tag-memory { background: #8b5cf6; }
footer { visibility: hidden; }
"""


def detect_workers(reply: str) -> str:
    tags = []
    if "置信度" in reply or ("诊断" in reply and "根因" in reply):
        tags.append(("分诊", "tag-triage"))
    if "风险" in reply and ("变更" in reply or "基线" in reply or "依赖" in reply):
        tags.append(("影响分析", "tag-impact"))
    if "紧急" in reply or ("指标" in reply and "异常" in reply):
        tags.append(("报告解读", "tag-report"))
    if "历史" in reply or "记忆" in reply or "之前" in reply:
        tags.append(("记忆", "tag-memory"))
    if not tags:
        return '<span class="worker-tag" style="background:#9ca3af">Supervisor</span>'
    return "".join(
        f'<span class="worker-tag {c}">{n}</span>' for n, c in tags
    )


# ── 同步 handler（asyncio.run 包装，兼容 Gradio）─────────────────────

def on_send(message, uid_val, sid_val, history):
    """发送消息 → SSE 流式收集 → 更新聊天记录"""
    if not message.strip():
        return history, sid_val, "", ""
    if not sid_val:
        sid_val = f"s_{uuid.uuid4().hex[:6]}"
    history = list(history) if history else []
    history.append({"role": "user", "content": message})

    reply = stream_chat(uid_val, sid_val, message)
    history.append({"role": "assistant", "content": reply})

    tag_html = detect_workers(reply)
    return history, sid_val, tag_html, ""


def quick_test(case_key, uid_val, sid_val, history):
    """快捷测试按钮"""
    if case_key == "分诊追问":
        pass  # 复用当前 session 模拟追问
    else:
        sid_val = f"s_{uuid.uuid4().hex[:6]}"
    history = list(history) if history else []
    history.append({"role": "user", "content": TEST_CASES[case_key]})

    reply = stream_chat(uid_val, sid_val, TEST_CASES[case_key])
    history.append({"role": "assistant", "content": reply})

    tag_html = detect_workers(reply)
    return history, sid_val, tag_html


def full_flow(uid_val):
    """一键完整流程：5 轮端到端测试"""
    lines = []
    sid1 = f"f_{uuid.uuid4().hex[:4]}"

    # Round 1: 分诊诊断
    lines.append("### Round 1: 分诊诊断\n")
    r1 = sync_chat_blocking(uid_val, sid1, TEST_CASES["分诊首轮"])
    lines.append(r1[:400])

    # Round 2: 追问（同 session bypass）
    lines.append("\n### Round 2: 追问（同 session bypass）\n")
    r2 = sync_chat_blocking(uid_val, sid1, TEST_CASES["分诊追问"])
    conv = "收敛" if "置信度" in r2 else "继续追问"
    lines.append(f"**状态：{conv}**\n\n{r2[:300]}")

    # Round 3: 新会话 → 记忆召回
    sid2 = f"f_{uuid.uuid4().hex[:4]}"
    lines.append("\n### Round 3: 新会话 → 记忆召回\n")
    r3 = sync_chat_blocking(uid_val, sid2, TEST_CASES["记忆召回"])
    ok = "成功" if ("U0100" in r3 or "历史" in r3 or "之前" in r3) else "未命中"
    lines.append(f"**召回：{ok}**\n\n{r3[:400]}")

    # Round 4: 影响分析
    sid3 = f"f_{uuid.uuid4().hex[:4]}"
    lines.append("\n### Round 4: 影响分析\n")
    r4 = sync_chat_blocking(uid_val, sid3, TEST_CASES["变更影响"])
    lines.append(r4[:300])

    # Round 5: 报告解读
    sid4 = f"f_{uuid.uuid4().hex[:4]}"
    lines.append("\n### Round 5: 报告解读\n")
    r5 = sync_chat_blocking(uid_val, sid4, TEST_CASES["报告解读"])
    lines.append(r5[:300])

    return "\n".join(lines)


def new_session():
    """生成新用户和会话 ID"""
    return f"u_{uuid.uuid4().hex[:6]}", f"s_{uuid.uuid4().hex[:6]}"


def clear_chat():
    return [], "", ""


def build_ui():
    with gr.Blocks(title="Supervisor + 3 Workers 测试") as demo:

        gr.HTML(
            '<div class="header">'
            '<h1>Supervisor + 3 Workers 测试</h1>'
            '<p>call_triage_agent · call_impact_agent · call_report_agent</p>'
            '</div>'
        )

        # ── 会话 ──
        with gr.Row():
            uid = gr.Textbox(label="User ID", value="test", scale=1)
            sid = gr.Textbox(label="Session ID（留空自动生成）", scale=2)
            new_btn = gr.Button("新会话", scale=1, size="sm")
            clear_btn = gr.Button("清屏", scale=1, size="sm")

        new_btn.click(fn=new_session, outputs=[uid, sid])

        # ── 对话 ──
        worker_tag = gr.HTML("")

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(label="对话记录", height=480)
                msg = gr.Textbox(
                    label="消息",
                    placeholder="试试：车机黑屏，DTC码U0100 / 分析变更影响 / 解读测试报告...",
                )

            with gr.Column(scale=1):
                gr.Markdown("#### 快速测试")
                btn_triage = gr.Button("分诊诊断", variant="secondary")
                btn_impact = gr.Button("变更影响", variant="secondary")
                btn_report = gr.Button("报告解读", variant="secondary")
                gr.Markdown("---")
                btn_memory = gr.Button("跨会话记忆", variant="secondary")
                gr.Markdown("---")
                btn_full = gr.Button("一键完整流程", variant="primary")

        # ── 事件绑定 ──

        # 消息发送
        msg.submit(
            fn=on_send,
            inputs=[msg, uid, sid, chatbot],
            outputs=[chatbot, sid, worker_tag, msg],
        )

        # 快捷按钮（用 lambda 偏应用 case_key）
        btn_triage.click(
            fn=lambda u, s, h: quick_test("分诊首轮", u, s, h),
            inputs=[uid, sid, chatbot],
            outputs=[chatbot, sid, worker_tag],
        )
        btn_impact.click(
            fn=lambda u, s, h: quick_test("变更影响", u, s, h),
            inputs=[uid, sid, chatbot],
            outputs=[chatbot, sid, worker_tag],
        )
        btn_report.click(
            fn=lambda u, s, h: quick_test("报告解读", u, s, h),
            inputs=[uid, sid, chatbot],
            outputs=[chatbot, sid, worker_tag],
        )
        btn_memory.click(
            fn=lambda u, s, h: quick_test("记忆召回", u, s, h),
            inputs=[uid, sid, chatbot],
            outputs=[chatbot, sid, worker_tag],
        )

        clear_btn.click(fn=clear_chat, outputs=[chatbot, sid, worker_tag])

        # ── 一键完整流程 ──
        full_output = gr.Markdown("")

        btn_full.click(fn=full_flow, inputs=[uid], outputs=[full_output])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=CSS)
