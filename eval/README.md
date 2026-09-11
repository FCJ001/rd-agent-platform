# 评测集（C3）

分诊质量的两层评测：**评分级门禁**（CI 每次跑）和**端到端实况**（发版前/每日跑）。

```
eval/
├── cases/
│   ├── scoring_cases.json   # 评分级案例 —— 纯函数，CI 回归门禁
│   └── live_cases.json      # L1-L5 实况场景 —— 需要真实 DB/Neo4j/LLM
└── run_eval.py              # 运行器
```

## 分层定义（L1-L5）

| 层 | 能力 | 考察点 |
|----|------|--------|
| L1 | 单现象直达 | 明确描述 → 首轮收敛，根因正确 |
| L2 | 多现象组合 | 多现象 Noisy-OR 合成区分候选 |
| L3 | 追问收敛 | 信息不足时追问，补全后收敛 |
| L4 | 干扰排除 | 否认证据收窄候选（核心否认 ×0.4 / 共享否认 ×0.7 可区分） |
| L5 | 边界 | 安全关键词终止、模糊表述不误收敛、图谱权重衰减生效 |

## 评分级门禁（CI）

```bash
python eval/run_eval.py        # 或 pytest tests/test_eval_gate.py
```

纯函数驱动 `confidence.apply_context_weights` + `check_convergence`，不依赖任何外部服务。
案例用**合成候选 + 控制变量对照**锁行为契约：S2a/S2b 锁证据分级差（hedged vs reported）、
S7/S7b 锁否认阻尼系数、S4 锁 Noisy-OR 单调性、S5 锁反馈回写的权重衰减传导。
任何置信度公式改动导致契约破坏 → 门禁失败。

## 端到端实况

```bash
# 前置：docker compose up -d + seed 脚本已跑 + LLM key 可用
python eval/run_eval.py --live --threshold 0.8
```

逐场景跑完整 `run_triage` 多轮对话，按 L1-L5 输出准确率，总体低于门限或任一场景
异常时返回非零退出码。场景断言支持：现象关键词命中（含口语别名）、根因 any-of、
安全终止、否认非空。

## 目录下的策略产物

`convergence_policy.json`（由 `scripts/fit_convergence.py` 拟合产出）是收敛判定的
数据驱动策略：`check_convergence` 检测到它就按 P(采纳)≥门限收敛，否则回退常数规则。
跑 `--live` 前若刚切换过置信度公式，建议先删掉策略文件用常数口径回归，
确认评分层无回归后再重新拟合 —— 新旧公式的历史样本混拟合会带偏置。

## 加案例的约定

- 评分级：改置信度公式时**必须**新增/调整对照案例，区间断言收紧到能区分新旧行为。
- 实况：`expect_cause` 用数组表示 any-of —— 知识库里现象集完全重合的候选（如
  RC-IA-0001/0002）不强行二选一，否则测的是知识库而不是分诊。
