# Invest GBP+USD 双钱包与双账户币种隔离 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在不破坏 `total_cash` / `total_combined_cash` 的 GBP 口径前提下，引入「Invest 本地 USD 钱包锚 + API 总现金联动推算 GBP 钱包」，并把双钱包的换汇摩擦、现金基准利息与校准告警接到风控规则、Prompt 与 Telegram。

**Architecture:** 新增纯函数模块 `src/engine/wallet_ledger.py`（模型 + 推算 + 解析 + 核对 + 展示，无 I/O），SQLite 只加一张幂等表 `wallet_balances` 存锚点，`portfolio_rules` 通过可选 `wallet` 参数感知钱包状态（默认 `None` 完全走旧逻辑，保证现有 120 个单测不抖），各 Agent 在拼 Prompt 时由 Python 算好金额再注入（不让 LLM 口算汇率）。

**Tech Stack:** Python 3.12 / Pydantic v2 / aiosqlite / unittest（`python -m tests.run_tests tests`）

---

## 0. 施工前必须拍板的 6 个决策（我的反对意见）

按全局规则先说风险，不认同的地方请直接否掉。

### 决策 1（必须改）：核对阈值不能用「或」，要用「取大」

原方案：差额 `> £10` **或** `> 2%` 就告警。
问题：£500 现金的 2% 只有 £10，而市场中间价与 T212 内部换汇价的偏差本身约 `0.15%`，£1,200 的 USD 仓位即约 £1.4 残差；再叠加平台按月计息入账与挂单占用口径差异，日常噪声轻松到 £5~£20。用「或」等于天天报假警，两周后你就会静音它——这套校准机制就废了。

**建议改为 `threshold = max(£10, 总现金 × 2%)`**（等价于「同时超过两者」才告警），差额永远显示在 `/cash` 回执里，只是不打扰你。

### 决策 2（必须承认）：推算出的 GBP 钱包永远是估算值，不得作硬约束

`USD ÷ GBPUSD` 用市场中间价，T212 换汇含约 `0.15%` 点差，所以 `GBP 钱包 = 总现金 − USD/汇率` 自带单向偏差 ≈ USD 名义 × 0.15%（$1,000 ≈ £1.4）。
因此估算值只用于「提示 + 排序参考 + 摩擦说明书」，**严禁**用它卡死下单金额、仓位上限或止损判定。所有 Python 规则只用二元信号（USD 钱包 `>0` / `=0`），金额原样交给模型做定性判断。

### 决策 3（必须补）：锚点会随每笔美股成交立刻过期

API 不返回分币种余额，你今天只要用 USD 钱包买一次 NVDA，`total_cash` 就变、USD 锚不变，推算的 GBP 钱包当场错（负值钳位后彻底失真，还会掩盖真问题）。
**v1 处理（推荐）**：不自动改锚，只在 `/pos`、`/cash`、Prompt 标注「USD 锚自 {日期} 起有 N 笔美股成交未校准，请 /cash 复核」。
**v2（可选，默认关）**：用 `recent_orders` 里 `inst_currency=USD` 的 Invest 成交滚动加减名义金额。**不建议 v1 就上**，因为 T212 是否对 Invest 卖出所得自动换回 GBP 尚未实机验证（见决策 4），自动滚动会把未知错误放大成静默错账。

### 决策 4（必须验证一次）：「卖出美股所得留 USD」是平台设置行为，不是 API 事实

`docs/T212_API_manual.json` 只声明「multi-currency accounts are not currently supported through the API」，没说 Invest 卖出美股后钱落在哪个钱包——这取决于 App 里 Invest 账户的自动换汇开关。
若实际是自动换回 GBP，你方案里「美股卖出 0% 回程换汇、基准 3.4%」的前提整条崩掉。
**落地动作**：上线后手动做一次小额「买入 + 卖出」验证，结果写进 `DEVELOPER_GUIDE.md`。在此之前 Prompt 措辞必须是条件句「若 USD 钱包余额充足且平台未自动换汇」。`/cash` 差额若逐日单调扩大，就是该假设错了的早期信号。

### 决策 5（必须区分）：`USD 3.4% < GBP 4.0%`，持有 USD 现金本身是负 carry

$1,000 存 USD 一年比折成 GBP 少约 `0.6%`，还没算换汇往返 `0.30%`。所以除了「美股买不买」，还多一层判断：**USD 钱包闲置超 N 天且无美股计划 → 应换回 GBP**（0.15% 单边成本，约 33 天回本）。
这恰恰是双钱包机制最有价值的产出，我会写进 Prompt（`usd_idle_days` 超阈值提示）。要不要默认开启请确认（我建议开启）。

### 决策 6（必须守住）：钱包金额禁止进入 Implicit Cache 静态前缀

`chat_agent` 靠静态前缀吃 Gemini 隐式缓存（`/us` 打碎缓存是新近修过的 P0）。钱包金额每天变，**只能进每轮动态上下文**，绝不塞进 `build_investor_profile()` 拼出的 `system_instruction`。
所以 `prompts.py` 只加「利率与摩擦规则」（静态），金额一律通过 `wallet_line` 注入 prompt 正文。

---

## 1. 现状核查结论（已核对代码与 API 文档）

| 核查点 | 结论 | 证据 |
| --- | --- | --- |
| API 能否给出分币种现金 | 不能。`Cash` 仅 3 个字段，均为主币种折算 | `docs/T212_API_manual.json` `AccountSummary`/`Cash` schema；正文 "Multi-currency accounts are not currently supported through the API" |
| API 能否下非主币种订单 | 不能（本项目只读，但不要向用户暗示 AI 能替你换汇） | 同上 "Order execution: Orders can be executed only in the primary account currency" |
| 当前现金字段 | `available_to_trade` / `in_pies_cash` / `reserved_cash` / `total_cash` 全 GBP | `t212_client.py:1110-1113` |
| 当前摩擦规则 | ISA 与 Invest 共用一套：UK 个股 0.5% 印花税、UK ETF 0%、其余 `fx_fee×2` | `portfolio_rules.py:73-96` |
| 当前现金基准 | 全场景硬编码 `cash_interest_rate_pct`（4.0%） | `portfolio_rules.py:461,596`；`prompts.py:24-29` |
| 现金展示位置 | `/pos`(chat_agent:531)、`/report` PM(report_agent:837)、`/us`(report_agent:1175)、`/pick`(pick_agent:571) | 四处需同步 |
| 文档冲突 | `DEVELOPER_GUIDE.md` §一.2 现写「双账户仅保留 GBP 现金」，本方案后**必须改写** | `DEVELOPER_GUIDE.md:10-12` |

---

## 2. 数据与计算口径（唯一真源）

1. **不变式**：`AccountSnapshot.total_cash` / `total_account_value` / `PortfolioData.total_combined_cash` 保持 GBP，权重与集中度分母不动。
2. **新表 `wallet_balances`**：只存锚点与来源，不存推算值。
   `(account, currency) PK, balance, source('user'|'derived'|'default'), reported_at_uk, reported_at_ms`
3. **推算（方案 A）**：`usd_gbp = usd_anchor / gbp_usd`；`gbp_wallet = total_cash_gbp − usd_gbp`；`< 0 → 0.0 + ⚠️ 要求校准`。
4. **核对（方案 B）**：`reported_total = gbp_report + usd_report/gbp_usd`；`diff = reported_total − total_cash_gbp`；
   `threshold = max(tolerance_gbp, total_cash_gbp × tolerance_pct/100)`；`abs(diff) <= threshold → 通过`。
5. **摩擦口径（Invest 非对称）**：

| 场景 | 换汇 | 印花税 | 现金基准 |
| --- | --- | --- | --- |
| ISA 英股 | 0% | 0.5%（买入） | GBP 4.0% |
| ISA LSE 英镑 ETF | 0% | 0% | GBP 4.0% |
| ISA 美股/欧股 | 0.15%/腿（往返 0.30%） | 0 | GBP 4.0% |
| Invest 美股，USD 钱包>0 | 0% | 0 | **USD 3.4%** |
| Invest 美股，USD 钱包=0 | 0.15% 单边 | 0 | GBP 4.0% |
| Invest 英股，GBP 钱包>0 | 0% | 0.5% | GBP 4.0% |
| Invest 英股，GBP 钱包=0（USD 换回） | 0.15% | 0.5% | **USD 3.4%** |
| Invest LSE ETF，GBP 钱包=0 | 0.15% | 0 | **USD 3.4%** |
| Invest 欧股 | 0.15% 单边 | 0 | GBP 4.0% |

6. **禁止 LLM 口算**：金额、差额、阈值全部由 Python 算完写成字符串注入。

---

## 3. 改动清单（影响面）

| 文件 | 类型 | 说明 |
| --- | --- | --- |
| `config.yaml` / `src/config.py` | 改 | `usd_cash_interest_rate_pct: 3.4`、`wallet_reconcile_tolerance_gbp: 10.0`、`wallet_reconcile_tolerance_pct: 2.0`、`usd_wallet_idle_alert_days: 30` |
| `src/engine/wallet_ledger.py` | **新增** | 模型 + 推算/解析/核对/展示纯函数（约 210 行） |
| `src/storage/db.py` | 改 | `wallet_balances` 建表 + `upsert_wallet_balance` / `get_wallet_balances` |
| `src/engine/portfolio_rules.py` | 改 | `_round_trip_friction_pct` / `_friction_hint` / 现金基准接受可选 `wallet`；`evaluate_holding` / `evaluate_portfolio` 加可选 `wallet=None` |
| `src/agents/prompts.py` | 改 | 双钱包利率与摩擦规则（**只写规则不写金额**） |
| `src/agents/chat_agent.py` | 改 | `/pos` 钱包行 + 新增 `handle_wallet_calibration()`（`/cash` 0-Token 实现） |
| `src/agents/report_agent.py` | 改 | PM Prompt 与 `/us` 现金行加钱包拆解（控字） |
| `src/agents/pick_agent.py` | 改 | 现金摘要加钱包拆解 + 资金来源提示 |
| `src/bot/handlers.py` | 改 | `/cash` 路由、`BOT_COMMANDS`、help 文案、`_is_heavy_ai_command` 放行 |
| `tests/test_wallet_ledger.py` | **新增** | 11 组离线单测 |
| `tests/test_db_storage.py` / `test_db_and_end_to_end.py` | 改 | 追加 db 往返与 Prompt 断言 |
| `AGENTS.md` / `DEVELOPER_GUIDE.md` / `docs/MODULE_CONTRACTS.md` / `CHANGELOG.md` / `src/version.py` / `pyproject.toml` | 改 | 模块索引、口径改写、签名表、变更记录、版本 `5.3.0` |

**不改**：`t212_client.py` 抓取链路（零风险）、`analytics.py`、`screener.py`、`market_data.py`。

