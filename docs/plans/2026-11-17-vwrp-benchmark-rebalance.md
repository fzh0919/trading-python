# VWRP 基准化 + 对称攻防纪律重构 Implementation Plan (v5.3.2)

> 由 `writing-plans` 生成、`executing-plans` 分批执行；每阶段结束跑一次全量离线回归。

**Goal:** 把系统核心基准从「守住 4%/3.4% 现金利息」重构为「长期跑赢 VWRP.L + 对称攻防纪律 + 以损定仓 & 频率硬约束」。

**Architecture:** Python 确定性层（config / t212_client / analytics / portfolio_rules）先算出基准、穿透暴露、以损定仓额度与频率余量；LLM 层（prompts / schemas / report_agent / pick_agent）只负责解读与表达，不承担算术。

**Tech Stack:** Python 3.12 / pydantic / pandas / httpx / unittest（全离线 Mock）。

---

## Stage 1: 配置底座 + 0-Token 行为统计注入

### Task 1.1 `src/config.py` `StrategySection` 新增 12 字段
默认值：`benchmark_symbol="VWRP.L"`、`min_invest_order_gbp=700.0`、`min_isa_equity_order_gbp=1200.0`、`min_isa_etf_order_gbp=400.0`、`invest_risk_budget_gbp=85.0`、`max_weekly_new_buys=2`、`max_monthly_new_buys=6`、`rebuy_cooldown_days=15`、`isa_cash_deploy_trigger_pct=15.0`、`invest_cash_idle_warn_pct=40.0`、`spread_estimate_us_pct=0.05`、`spread_estimate_uk_eu_pct=0.20`；`benchmark_symbol` 去空格大写。

### Task 1.2 `config.yaml#strategy` 同步同名键 + 中文注释

### Task 1.3 `src/data_providers/t212_client.py`
- 新增 `compute_trade_behavior_stats()`：7/30 天买卖笔数、分币种平均单笔、税费占成交额、清仓冷却清单、`coverage_days/partial`。
- 新增 `_format_trade_behavior_text()` 紧凑单段（<=160 字）。
- `build_trade_context()` 新增 `cooldown_days` 入参，返回 `trade_behavior` + `trade_behavior_text`。

### Task 1.4 `_compute_pending_order_quant_note()` 改挂单限价口径
- BUY 限价单新增 `限价空间 x.x%`（以 `limit_price` 为成本基线，比例法换算避免 GBp/GBP 100 倍错位）。
- 标签：距现价 `-3%~-12%` → `[有效回踩埋伏单]`；限价高于现价 → `[追价单]`。

### Task 1.5 单测 `tests/test_t212_client.py`
7/30 天笔数、分币种平均、税费占比、冷却清单、限价口径不被现价误判、覆盖不足 `partial=True`。

---

## Stage 2: 确定性风控与真实暴露穿透

### Task 2.1 `src/data_providers/analytics.py`
- `ETF_TECH_PENETRATION_PCT` 静态近似表（VWRP/VWRL/SWDA 25%、VUAG/CSP1/VUSA 32%、CNX1/EQQB 50%、ISF/VUKE 1%）。
- `estimate_penetration_tech_exposure()` + `compute_exposure_summary()` 输出「穿透后真实科技暴露(估算)」行。

### Task 2.2 `src/engine/portfolio_rules.py`
- `_round_trip_friction_pct()` / `_friction_hint()` 加入买卖价差估算。
- `evaluate_holding()`：ISA 不再被 4.0% 现金门槛否决（改机会成本置换提醒）；Invest 改 R:R 判定。
- `compute_risk_based_size_gbp()`：`min(£85/|止损%|, 单仓上限剩余额度)` 并夹最小建仓门槛；`HoldingEvaluation` 增 `rr_ratio`/`suggested_size_gbp`/`min_order_gbp`。
- `apply_risk_override_guardrail(..., trade_behavior=...)`：`FREQUENCY_CAP` + `REBUY_COOLDOWN`，只拦 BUY/ADD/SWAP。

### Task 2.3 接线 `report_agent.py` / `pick_agent.py` 传 `trade_behavior`

### Task 2.4 单测 `tests/test_analytics_and_rules.py`

---

## Stage 3: 消除 Prompt「守现金吃息」偏见

### Task 3.1 `src/agents/prompts.py#build_investor_profile` 重写（VWRP 宗旨 / 资金分桶 / 卖出三分类 / 开仓硬纪律）
### Task 3.2 `src/agents/schemas.py` 4 处描述文案（L94 / L128 / L140 / 兜底红线）
### Task 3.3 `src/agents/report_agent.py` analyst_sys + auditor_sys 对称指令
### Task 3.4 `src/agents/pick_agent.py#critic_sys` R:R + 穿透 + 配额；ISA 现金超阈值时 ETF 分批方案
### Task 3.5 单测 `tests/test_db_and_end_to_end.py`（含 1800/1000 上限回归）

---

## Stage 4: 周报/早报 Benchmark 闭环

### Task 4.1 `analytics.compute_window_return_pct()` 纯函数
### Task 4.2 `report_agent.py` 注入 1 行基准对照（≤60 字，不挤占 1800）
### Task 4.3 `scheduler.py` 周报 `benchmark_dashboard` + DS 诊断 prompt + 推送文案
### Task 4.4 单测 `tests/test_sidecar_memory_and_weekly.py`

---

## 收尾
- `src/version.py` + `pyproject.toml` → `5.3.2`
- `CHANGELOG.md` 顶部新增 `v5.3.2`
- `docs/MODULE_CONTRACTS.md` 同步签名
- `.\.venv\Scripts\python.exe -m compileall -q src tests`
- `.\.venv\Scripts\python.exe -m tests.run_tests tests`

## 不可触碰边界
`/report<=1800`、`/pick<=1000`、`/us<=450|600` 字数硬上限；Telegram 纯文本（禁 `#`/`**`）；双币种隔离（账户 GBP vs 标的本地货币）；SQLite 单写入入口与幂等迁移；T212 只读；`_ref/` 不变。
