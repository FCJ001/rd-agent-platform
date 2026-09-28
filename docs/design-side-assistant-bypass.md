# 分诊追问期间的用户意图分流（旁路回答）设计

> 状态：**已实施并直接开启（2026-09-24）**，`SIDE_ASSISTANT_ENABLED` 默认 true；回退改 false = 影子模式
> 日期：2026-09-23（设计）/ 2026-09-24（实施）
> 范围：`src/api/routers/chat.py`（同步 + SSE）、新增判定模块与旁路助手、config、测试、评测
> 不动：分诊状态机（`src/agents/triage/graph.py`）、`call_triage_agent` 的快进结构、`TriageSessionStore` key 规范

## 实施记录（2026-09-24，与设计的偏差说明）

落地清单 8 项全部完成，两处按实施期发现调整：

1. **L6 用例的落点（§6 第 8 项）**：`run_eval.py --live` 是 `run_triage`
   内核直调、不经过 chat 路由层，「打断分流」行为无法用它断言。调整：
   chat 层分流行为由 `tests/test_offtopic_interruption.py` 覆盖（图未被
   调用/快照仍挂起/store 不变）；`live_cases.json` 的 L6 落为**内核级**
   离题容错用例（离题注入后现象集不污染、补答仍收敛）。
2. **快照与判定在回合锁内执行**：设计说「判定只依赖快照+消息文本，
   发生在任何图调用之前」，实施补一条 —— 快照读取也在 `_chat_turn_lock`
   内（同步/SSE 两路本来就在锁内取快照），保证与并发 resume 的一致性。
   路由逻辑收敛为 `chat._route_turn`（同步/SSE 共用，快照只取一次）。

附带产出：`chat._log_turn_trace` 回合级结构化 trace v1（路由/延迟/token
累计口径），对应 `docs/multi-agent-architecture-design.md` P0③。

---

## 0. 最终方案（速览）

**一句话**：追问挂起期间，用户消息若判定为"问别的智能体"，就**不碰图**——
用无状态旁路助手就地回答，然后把挂起的追问原样回放。诊断状态机零改动。

**分流（四步，判定全部发生在任何图调用之前）**：

| # | 条件 | 动作 | 状态影响 |
|---|---|---|---|
| 1 | 快照无挂起 interrupt | 现状：新消息进 Supervisor | — |
| 2 | 有挂起 + 退出词 | 现状：`Command(resume)` → 工具清 store | 诊断作废（用户显式） |
| 3 | 有挂起 + 判定旁路 | **不调用图**：旁路回答 + 回放追问 | **零变动** |
| 4 | 其余 | 现状：`Command(resume)` 按回答处理 | 正常推进 |

**三个组件**：

1. `src/agents/triage/offtopic.py` — 正则门（零成本，典型回答不命中）
   + 小模型裁决（仅对过门消息，`{"intent":"answer"|"other"}`，含糊一律 answer）
2. `src/agents/workers/side_assistant.py` — 无状态 `create_agent`，无 checkpointer，
   工具 = `WORKER_TOOLS` 去 `call_triage_agent` + `search_past_diagnoses`，
   必须透传 `context=ctx`（业务线作用域）
3. `src/api/routers/chat.py` — 两个分流点（同步 / SSE），快照合一，
   复用 `_pending_question` 回放追问

**本期必做**：上述三件 + config 两个开关 + §7 测试 + 影子模式灰度。
**本期可选（建议同期，前端零协议改动）**：追问期间输入框提示语 +"诊断进行中"横幅
（§9）。**不做**：park/带记忆旁路、连续旁路计数、REST 入口、判定器面板（§9）。

**为何是这个方案**：它是 Rasa / Bot Framework 的经典"打断-回答-重问"模式
在 LangGraph 语义下的正确落点（§2.5 业界对照）；且由 §2 四条实测语义唯一确定——
绕过 interrupt 不可行（[A]）、工具内 raise 不可行（[D]）、park 代价过高（[B]/[F]）。

---

## 1. 问题

分诊多轮追问挂起期间（`call_triage_agent` 内 `interrupt()` 等待回答），用户发来的消息
**全部**被当作"追问的回答"喂回分诊图——包括明显该由其他智能体处理的请求
（"帮我查下这个月 EV 线故障统计"）。唯一逃生口是精确匹配的退出词
（`TRIAGE_EXIT_COMMANDS`，见 `src/agents/triage/session_store.py`）。

现状行为与代价：

| 场景 | 现状行为 | 代价 |
|---|---|---|
| 回答追问 | `Command(resume)` 进图 ✅ | — |
| 退出指令 | resume → 工具清 store、诊断作废 ✅ | — |
| **问其他智能体** | 被当回答进 `node_parse_answer`，抽不出现象 → 继续追问 | 白烧 LLM 调用；噪声拖到 MAX_ROUNDS 强制收敛；**用户的问题被静默丢弃** |

这是 B2 重构"单一控制面 = Supervisor checkpointer"的刻意代价：路由层只看快照里
有没有挂起 interrupt（`chat.py` 的 `_has_pending_triage`），不看消息内容，Supervisor
在追问期间不参与（其 system prompt 明确"期间用户回复由框架转交，不经你手"）。

## 2. 设计约束（实测 langgraph 语义，1.1.6 + checkpoint-redis 0.4.0）

以下四条全部由可复现脚本验证过（将固化为 `tests/test_langgraph_interrupt_contract.py`），
它们直接决定了方案形态：

- **[A] 挂起期间用新输入（非 `Command(resume)`）调用图**：消息被并进 state，
  **同一个 interrupt 被原样重新抛出**，图不前进。
  → "绕过 interrupt 把消息交给 Supervisor"不可行：消息被静默吞掉，还留在历史里。
- **[D] 工具消费了 resume 值之后再抛异常**：该值**已提交**（写入 checkpoint 的
  resume 映射）；用户重答的新值会挂到**下一个** interrupt 上 → 跳问、轮次错位。
  → "工具内识别离题后 raise 交出控制权"不可行。
- **[E] 工具在 `interrupt()` 之前抛异常**（= 生产里闸门/会话锁的真实顺序）：
  值不提交，重试时使用新值。
  → 现有 busy 路径语义正确；同时确立硬约束：**分流判定绝不能发生在工具消费
  resume 值之后**。这把判定推到 chat 入口层。
- **[B]/[F] park（工具返回文本消费 interrupt、进度留外部 store、之后新 task 重问）**：
  技术上可行，但快进次数必须改为"相对 park 点的基线"；朴素沿用 `state.round`
  在 park 发生于 round>0 后的下一次 resume 会停在 `fast-forward` 空载荷上
  → 用户看到兜底文案"请继续描述故障细节。"，且该条真实回答被吞。
  → park 引入第二份需对账的状态（正是 B2 重构消除的东西），降级为 V2 备选。

## 2.5 业界对照（2026-09 调研，含 GitHub 源）

"表单/追问进行中被用户打断"是经典问题，主流框架都有明确解法，
且收敛到同一模式——**每轮先分类，离题就先回答，然后自动重问挂起的问题**：

| 框架 | 机制 | 与本方案的关系 |
|---|---|---|
| **Bot Framework Adaptive Dialogs**（微软，生产级对话框架） | 每个输入提示有 `InputDialog.AllowInterruptions` 属性，可设 true/false/**布尔表达式**（如"仅当识别出高置信其他意图才打断"）。打断时父对话框先处理，随后输入提示**自动 re-prompt** 原问题 | 模式同构：表达式门 ≈ 我们的"正则门 + 保守默认"；自动 re-prompt ≈ 追问回放。见 [property 文档](https://learn.microsoft.com/en-us/dotnet/api/microsoft.bot.builder.dialogs.adaptive.input.inputdialog.allowinterruptions)、[botframework-sdk#5471](https://github.com/microsoft/botframework-sdk/issues/5471) |
| **Rasa Forms**（对话管理老牌标准） | 官方"unhappy path"模式：回答填不进槽位 → `ActionExecutionRejection` → 规则先回离题意图 → **重新激活表单 → 自动重问 `requested_slot`**；放弃流程用 `action_deactivate_loop`（≈ 我们的退出词）。官方强烈建议用 interactive learning 从真实对话补规则（≈ 我们的影子模式） | 与本方案逐条对应。见 [Forms 文档](https://legacy-docs-oss.rasa.com/docs/rasa/forms) |
| **LangGraph**（我们的栈） | 官方**不提供内置方案**：路由决策归属调用方（"Detect interrupts... fetch user input... resume"）；文档仅覆盖"无效回答→图内校验→重问"，不覆盖"离题→先办别的→回来"。社区共识是**先分类再决定 resume** | 与本方案的入口层判定一致。见 [Interrupts 文档](https://docs.langchain.com/oss/python/langgraph/interrupts)、[论坛讨论](https://forum.langchain.com/t/how-to-resume-the-agent-workflow-after-interrupt/12787) |
| **agent-chat-ui**（LangChain 官方生产前端，⭐3.1k） | 另一条互补路线：挂起 interrupt 渲染成**结构化表单卡片**（accept/edit 按钮），用户输入天然作为 resume——用 UI 契约约束输入模态，从源头减少离题 | 启发：前端给追问加"诊断进行中"横幅 + 输入框提示语，降低离题率（配合而非替代服务端旁路）。见 [repo](https://github.com/langchain-ai/agent-chat-ui) |
| **OpenAI Agents SDK**（⭐29.6k，Swarm 嫡系） | 无 interrupt 锁：每轮由同一个顶层 LLM 重新路由（handoff），子流程必须可重入 | 对应本方案 V2 的 park 路线；不中断式锁死用户体验更好但工程代价高。见 [repo](https://github.com/openai/openai-agents-python) |

结论：本方案 = Rasa/Bot Framework 的经典"打断-回答-重问"模式
+ LangGraph 语义下（客户端持有路由权）的正确落点，属于业界标准做法的移植，
不是自创结构。调研同时带来两个低成本补充：前端"诊断进行中"提示（§9）、
影子模式用真实对话标定判定器（Rasa interactive learning 的对应物，§5）。

## 3. 方案：入口层判定 + 旁路回答（zero-state bypass）

**核心性质：旁路路径是纯读取，不碰图。** interrupt 位置、checkpointer、
`triage_state` store 三者零改动，不需要任何新的对账机制。

### 3.1 判定层（chat 入口）

分流点：`chat.py` 同步接口与 SSE 接口各一处（现 `_has_pending_triage` 判断）。
改为先取一次快照（合并现有 `_has_pending_triage` + `_pending_question` 的一次
`aget_state` 往返）：

```
1. 快照无挂起 interrupt            → 现状：新消息进 Supervisor
2. 有挂起且 is_triage_exit(msg)     → 现状：Command(resume)，工具走退出分支
3. 有挂起且判定为旁路请求（3.2）    → 不调用图：旁路助手回答（3.3）+ 回放追问（3.4）
4. 其余                            → 现状：Command(resume)，按追问回答处理
```

判定只依赖快照 + 消息文本，发生在任何图调用之前 → 满足 [E] 的硬约束。

### 3.2 判定器：正则门 + 小模型裁决（`src/agents/triage/offtopic.py`）

两层，控成本也控误判：

1. **正则门（确定性、零成本）**：仅当消息命中"另有所指"模式才继续。命中词类：
   统计/报表/数据/指标/趋势、规范/标准/参数/文档/知识库/资料、报告/解读、
   影响/变更单/单号/建单/结案/工单、帮我查/查一下/查询/多少、是什么/什么意思/怎么查。
   典型追问回答（"没有其他现象""是的每次都这样""充电时跳闸"）不命中 →
   **不产生任何额外模型调用**。
2. **小模型裁决（仅对过门消息）**：输入 = 挂起的追问原文 + 用户消息，
   输出 `{"intent":"answer"|"other"}`；prompt 规则明确
   **"两者都像或说不清 → answer"**（宁可不旁路）。
   模型走独立配置（默认 qwen-flash 级小模型），带超时；
   解析失败/超时/任何异常 → 一律 answer（= 现状行为）。

误判两个方向都安全：
- 误判为旁路 → 多一轮旁路回答 + 追问回放，状态无损，用户重答即可；
- 误判为回答 → 即今天的行为。

### 3.3 旁路助手（`src/agents/workers/side_assistant.py`）

`create_agent` 构建，**不带 checkpointer**（无状态、单轮）：

- **工具集** = `WORKER_TOOLS` 去掉 `call_triage_agent`，加 `search_past_diagnoses`。
  保留 `call_knowledge_agent` / `call_operation_agent` / `call_report_agent` /
  `call_impact_agent` 与平台工具——这就是"其他智能体"本身。
  **必须摘掉 `call_triage_agent`**：否则模型可能在旁路线程另起一次分诊，
  与挂起中的目标诊断并跑（两个 thread 抢同一份 store 进度的老问题）。
- **必须 `context_schema=UserContext` 且 `ainvoke(..., context=ctx)`**：
  BI、去重、知识库、历史结论全部从 `runtime.context.business_line / role`
  取作用域。漏传 = 旁路回答跨业务线取数（本项目已有跨租户泄漏门禁，属红线）。
- **system prompt 要点**："用户正在一次故障诊断中，诊断暂停在追问：<Q>。
  这条消息与诊断无关，只处理它；不要给出诊断结论、不要重复诊断、
  不要调用分诊工具；回答简洁。"
- 不占 `triage_gate` 额度（旁路发生在 gate 之外），不写任何诊断结论/记忆。

### 3.4 回复编排

```
<旁路回答>

---

我们接着刚才的诊断 —— <挂起的追问原文>

（想结束这次诊断，回复「退出诊断」）
```

追问原文复用现有 `_pending_question`（`chat.py`，busy 路径回放用的同一函数）。
SSE 路径：跳过图 token 流，把整段文本作为 token 帧下发后发 `done`。
前端零改动（markdown 渲染）；BUSY_CODES 可重试样式不涉及。

**旁路助手自身失败**（模型超时等）：**不**回落去 resume（会把离题文本喂进
分诊解析、白烧一轮且污染现象集），改为回放追问 + 一句
"刚才那条消息没能处理成功，可以先回答上面的追问，或重发一次"，记 warning。
状态依旧零变动。

## 4. 不变量与失败模式总表

| 场景 | 行为 | 状态影响 |
|---|---|---|
| 回答追问 | resume 进图（现状） | 正常推进 |
| 退出指令 | resume → 清 store（现状） | 诊断作废（用户显式要求） |
| 旁路请求 | 旁路回答 + 回放追问 | **零变动** |
| 判定器异常 | 当回答处理（= 现状） | 正常推进 |
| 旁路助手异常 | 回放追问 + 失败提示 | 零变动 |
| 连续多次旁路 | 每次回放追问并提示退出 | 诊断持续挂起，TTL 由 `load()` 读时续期维持 |

残余风险：用户反复旁路又不退出 → 诊断一直挂着。一期用文案引导；
若需要，二期在 chat 层加"连续 N 次旁路后建议退出"计数，不动状态机。

## 5. 配置与灰度

`src/core/config.py` 新增：

- `SIDE_ASSISTANT_ENABLED`（默认 **false**）
- `SIDE_ASSISTANT_MODEL`（默认 `qwen-flash`）

**影子模式先行**：`SIDE_ASSISTANT_ENABLED=false` 时判定器照常运行但只记日志
（`[SIDE] hit_regex=True verdict=answer q=... msg=...`），不改路由。
用真实流量跑一段时间统计误判率，达标后再翻开关。

> 实施决策（2026-09-24）：**跳过影子期直接开启**。依据：两个方向的误判都安全
> （误判为回答 = 原行为；误判为旁路 = 状态零变动、用户重答即可），且
> `[SIDE]` 日志在开启状态下照常输出，线上误判率可事后统计，出问题关开关即回退。

## 6. 落地清单

| # | 文件 | 动作 |
|---|---|---|
| 1 | `src/agents/triage/offtopic.py` | 新增：正则门 + 小模型裁决 |
| 2 | `src/agents/workers/side_assistant.py` | 新增：无状态旁路助手 |
| 3 | `src/api/routers/chat.py` | 改：两个分流点（同步/SSE），快照合一 |
| 4 | `src/core/config.py` + `.env.example` | 改：两个新配置 |
| 5 | `tests/test_offtopic_interruption.py` | 新增（见 §7） |
| 6 | `tests/test_langgraph_interrupt_contract.py` | 新增：固化 [A]/[D]/[E]/[F] 四条语义 |
| 7 | `tests/test_triage_hitl.py` | 补：按生产真实顺序（interrupt 之前拒）的 busy 用例 |
| 8 | `eval/cases/live_cases.json` | 新增 L6"追问中途被打断"场景 |

## 7. 测试设计

沿用 `test_triage_hitl.py` 的 StubRedis + fake model 骨架，全部 CI 可跑：

1. **典型回答不触发判定**：正则门不命中 → 无小模型调用、正常 resume。
2. **旁路主路径**：命中且裁决 other → 断言图 **未被以 `Command(resume)` 调用**、
   快照仍挂起 interrupt、store 的 round/reply 不变、回复含旁路答案与回放追问。
3. **裁决器异常 → 回落 answer**：正常 resume，行为同现状。
4. **旁路助手异常 → 回放追问**：状态不变、回复含追问原文与失败提示。
5. **chat 回复编排**：stub 旁路助手，断言拼接格式（同步 + SSE 两路）。
6. **契约测试**：[A] 新输入不消费 interrupt；[D] 消费后抛 → 值已提交、新值错位；
   [E] 消费前抛 → 值不提交、重试用新值；[F] park 朴素快进的空追问缺陷。
7. **影子模式**：`ENABLED=false` 时判定只记日志、路由不变。

## 8. 否决的备选

- **park（暂停-恢复，让真 Supervisor 回答旁路问题）**：回答质量最好（带记忆与
  历史结论），但需新增 park 标记 + park 点基线，快进公式改为
  `state.round - round_at_park`，工具内三路分支（继续/新故障/歧义），
  Supervisor prompt 要教"继续"，且 park 状态对路由层不可见——两套状态对账
  正是 B2 重构消除的东西。留作 V2（确需"带记忆的旁路回答"时再上）。
- **图内全工具**：分诊图自己拥有所有工具 = 分诊与调度揉成一个 agent，
  `TriageState` 被非诊断工具污染，不划算。
- **REST `/api/v1/triage` 同步处理**：该入口不走 interrupt，追问文本直接返回，
  调用方看得见问题，影响小；可选后续复用同一个判定门。

## 9. V2 展望（不在本期）

- **前端提示语（低成本，建议同期做）**：追问期间输入框 placeholder 改为
  "回答上面的诊断问题，或直接提新问题（会自动转给对应助手）" + 消息区顶部
  "诊断进行中"横幅。依据 agent-chat-ui 的做法（用 UI 契约约束输入模态），
  能显著降低误判为旁路的比例，零协议改动。
- park 式带记忆旁路；
- 连续旁路计数 → 主动建议退出或自动挂起；
- REST 入口接同一判定门；
- 判定器统计面板（命中率/误判率，接反馈闭环）。
