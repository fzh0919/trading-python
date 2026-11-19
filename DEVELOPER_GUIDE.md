# Trading 212 × Gemini AI Advisor — 业务契约与数据模型手册 (DEVELOPER_GUIDE)

> **定位说明**：热启动索引与开发铁律已收口至 [`AGENTS.md`](./AGENTS.md)；本手册仅保留业务前提、Agent 模型路由与 SQLite 核心表结构供按需查阅。完整函数签名字典见 [`docs/MODULE_CONTRACTS.md`](./docs/MODULE_CONTRACTS.md)，变更历史见 [`CHANGELOG.md`](./CHANGELOG.md)。

---

## 一、账户实况前提与业务规则

1. **双账户与核心-卫星结构（`v5.4.0`）**：面向英国个人投资者（`Europe/London`），管理 Trading 212 **Stocks ISA**（免税长线：**核心宽基 `VWRP.L` 目标占比 ≥60% 无上限，个股卫星层含挂单 ≤30% 且 ≤8 只**；`<£1,200` 历史散仓享老仓豁免，仅在反弹近回本窗顺势分批归核，不低位强平）+ **Invest**（应税右侧战术仓：无现金超带催买，实行 `RISK_ON 2 / NEUTRAL 1 / RISK_OFF 0` 在险槽位制，挂妥保本止损即释放槽位解锁新仓）；**不使用 Pie，不开启分红自动再投资**。
2. **双币种严格隔离（`GBP` 结算 vs 本地货币报价）**：
   - 账户净值、现金、盈亏与 `walletImpact` 统一 `GBP` 口径；**Invest 另维护本地 `GBP + USD` 双钱包锚点**（API v0 不返回分币种现金，USD 余额走 SQLite `wallet_balances` 账本，`GBP 钱包 ≈ API 总现金 − USD/GBPUSD` 仅作展示与摩擦参考，不回写 `total_combined_cash`）；Stocks ISA 仍为纯 `GBP` 钱包；Telegram 用 `/wallet` 查看对账、`/usd 金额` 上报 USD、`/wallet gbp 金额 [usd 金额]` 单报或双报，每周六 `12:00~22:00 UK` 自动推送一次对账提醒；
   - 标的现价、持仓均价、挂单触发价与技术指标按**标的本地货币**（美股 `USD`、欧股 `EUR`、英股 `GBp` 便士 / `GBP` 英镑）计价，Python 层统一处理换算并校验 `GBp/GBP` 100 倍错位。
3. **基准使命、摩擦核算与零 Quota 豁免**：
   - 双账户合计以长期跑赢 `VWRP.L`（含分红再投资）为唯一业绩基准；闲置及买入限价挂单占用的 `GBP` 现金享 ~`4.0%` 年化利息（挂单未成交前照常计息），Invest `USD` 钱包闲置现金按 `3.4%` 计提（仅作等待期保底收益，严禁把“守住现金吃息”当默认结论）；
   - 非英镑标的买卖各收 `0.15%` 换汇费（往返 `0.30%`），英国个股买入收 `0.50%` 印花税（LSE ETF 免税）；Invest 买卖美股时若 `USD` 钱包余额充足则 0% 换汇，需从 `GBP` 换汇时对换汇部分计 `0.15%` 单边；卖出任何存量仓位/清理边角料及买入核心宽基 ETF 享 **100% 零 Quota 豁免**（不消耗周/月开仓配额与在险槽位）。
4. **N95 本地算力换 Token 与预算定位**：
   - 运行于 **Intel N95 + 16GB RAM** Docker 容器（零公网端口暴露，接入 `homelab-net` `ops-notify`），Gemini Paid Tier 月度预算控制在 **\$10 USD 以内（上限 \$20 USD）**，享每月 5,000 次免费 Google Search Grounding。
   - `/report`、`/pick`、`/us` 将量化指标（CYQ、周线、形态、相关性、CVaR、Disagreement Matrix、`RiskOverride`、CIO `deployment_gate`）全部在 Python 0-Token 算完，组装为 `AnalysisContextPack` 喂给模型。

---

## 二、Agent 模型路由与输出契约（`config.yaml` -> `src/agents/`）

| 业务流 / 节点 | 配置槽位 (`models.*`) | 默认模型 & 思考深度 | 工具 / 运行策略 | 最终输出硬上限 |
| :--- | :--- | :--- | :--- | :--- |
| **`/report` Step 1 联网（核心）** | `report_priority_search_model` / `report_search_model` | 宏观&1v1 `Hydrogen` (`LOW`) / 批量 `3.5-flash-lite` (`LOW`) | `GoogleSearch`（每批 `<=3` 标的，`Semaphore(3)`，3h SQLite 快照，英文搜/中文出，禁切 DeepSeek） | 注入内部上下文 |
| **`/report` Step 2A 机会（核心）** | `report_analyst_model` | `gemini-4-hydrogen` (`HIGH`) *(备援 DeepSeek HIGH)* | 本地全维 `ContextPack` + `MarketToolkit`（`<=2` 轮，禁重搜） | 内部草稿 `<=1200` 字 |
| **`/report` Step 2B 风控（核心）** | `report_auditor_model` | `gemini-4-hydrogen` (`HIGH`) *(备援 DeepSeek HIGH → `3.5-flash-lite` LOW)* | 本地全维 `ContextPack` + `MarketToolkit`（`<=2` 轮，禁重搜） | 内部草稿 `<=1200` 字 |
| **`/report` Step 2C 交叉质询（辩论）** | `report_debate_model` | `deepseek-flash` (`HIGH`) | 首轮草稿 + 共享 FactSheet；只反驳对手、不给最终结论；失败即跳过 | `<=2400` 字 ×2（并发） |
| **`/report` Step 2D 中立索引** | `report_neutral_model` | `deepseek-flash` (`HIGH`) | R1+R2+分歧矩阵；摘要四段 + 争议标的原文引用 / 数字核验只标记 `[无据]` | `<=900` / `<=400` 字（并发） |
| **`/report` Step 3 PM 拍板（核心）**| `report_model` | `gemini-4-neon` (`HIGH`, 单次 `300s`) *(备援 Hydrogen HIGH → DeepSeek HIGH + 自修复)* | `MorningReportDecision`（含 `PickGateDecision` 单向收紧权）+ 自修复 + `RiskOverride` + 护栏后 `deployment_gate` 横幅注入与落库；PM `pick_gate.allow=false` 由 `/pick` 初筛前硬拒绝；只注入 R3 索引，不注入 R2 全文 | **Telegram `<=1800` 字** *(Prompt 1500~1650)* |
| **`/pick` Stage 1 投研员** | `pick_researcher_model` | `gemini-4-hydrogen` (`HIGH`) *(备援 DeepSeek HIGH)* | 四维量化+新闻/评级预聚合 + `MarketToolkit`（`<=2` 轮，带方向 `<=3` 轮） | 内部提案 |
| **`/pick` Stage 2 风控官** | `pick_model` | `gemini-4-neon` (`MEDIUM`) *(备援 Hydrogen HIGH → DeepSeek HIGH+Schema)* | `PickDecision` + 自修复 + Post-Screen 确定性校验（PM 否决、Invest DEFEND/冷静期/单日上限/满槽、ISA `equity_open/etf_open`、ISA 饱和、跨账户个股去重、`ROTATE` 置换源核验、`0.5R` 半仓算仓；核心 ETF 双持豁免） | **Telegram `<=1000` 字** |
| **`/us` 美股增量哨兵（分级门控）** | `us_sentinel_model` / `us_anomaly_model` | 平稳 `deepseek-flash` (`HIGH`) / 异动 `Hydrogen` (`HIGH`) | 本地 Anomaly Gate：平稳免工具核对；异动切 Gemini+搜+工具深挖 | **平稳 `<=450` / 异动 `<=600` 字** |
| **日常自由单聊 / `/deep`** | `chat_model` / `chat_deep_model` | 日常 `deepseek-flash` (`HIGH`) / `/deep` `Neon` (`MEDIUM`) | 本地切片 + 静态前缀缓存 + 按需工具（`<=2` 轮，`/deep <=4`），双厂互备 | **Prompt `<=500` 字（`/deep` `1000~1200`）** |
| **盘后 T+5 复盘微反思** | `chat_model` | `deepseek-flash` (`HIGH`) *(备援 Hydrogen MEDIUM)* | 本地算完 T+N 方向感知收益（`BUY/ADD` 多头、`TRIM/STOP/SWAP` 避损；未触价 `BUY/ADD` 标记 `NOT_FILLED` 并排除胜率）后生成 `2~4` 句（`<80 tokens`）教训 | 存入 SQLite 记忆列 |
| **DeepSeek 侧车（筛选/质检/周报/记忆）** | `ds_helper_model` / `ds_helper_thinking_level` | `deepseek-flash` (`HIGH`)，单次 `ds_helper_timeout_seconds=20s` | `ds_pick_prefilter_enabled`（ID 名单）/ `ds_semantic_audit_enabled`（只读质检）/ 周六诊断 / 偏好抽取；缺 `DEEPSEEK_API_KEY` 时 **0 调用**，绝不回落 Gemini | 名单 JSON / `violations` / `<=450` 字诊断 / 带原话锚点笔记 |
| **盘中异动哨兵** | 复用 `chat_model` 与 `search_model` | 扫描 0-Token（实时价：Finnhub → T212 持仓 → 日线末根）；仅触发标的做 1v1 联网 + DS (`HIGH`) 判推送 | `intraday_sentinel_enabled`，工作日 `08:00~21:00 UK` 每 `30` 分钟、`execution_lock` 被占则跳过；限价/止损挂单均纳入逼近判定 | Telegram `<=250` 字纯文本（仅 `should_push=true` 且发送成功） |

> **调用级韧性（`v5.3.3` / `v5.4.2`）**：`gemini_round_timeout_seconds=120s` 为单轮硬超时；硬超时按 `gemini_timeout_max_retries=0` **直接切备援**（不再空转重试），5xx/429 仍按 `gemini_max_transient_retries=2` 重试。`/report` Step 2 改为分段预算 `report_step2_round1/2/3_seconds=360/180/120s`（总闸 `660s`）：R1 任一侧降级即跳过 R2/R3；R2/R3 失败只发 `report_debate_degraded`，不污染 R1 的 `report_step2_degraded`。PM 单次 `report_pm_timeout_seconds=300s`（`round_timeout_seconds` 只作用首候选，备援仍用 120s）；DS 质询/中立带 `max_output_tokens` 上限（R2 4096 / 摘要 2048 / 核验 1024）。

---

## 三、SQLite 数据库核心表速查（`data/advisor.db`，周期切换点 `07:30 UK`）

1. `portfolio_snapshots(id, cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json)`：双账户完整快照。
2. `morning_reports(cycle_date PK, report_text, conclusion_text, premarket_note, daily_guardrails, context_pack_json, updated_at_ms)`：当日 `/report` 全文、红线、`/us` 笔记及当日 `AnalysisContextPack`（跨指令 0-Token 复用）。
3. `pick_radars(id, created_at_uk, created_at_ms, focus_query, research_text, final_text)`：`/pick` 投研草稿与终审存档。
4. `pick_recommendations(id, ..., account, symbol, action, entry_price, thesis, horizon_days, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, outcome_status, reflection_lesson, t10_return_pct, t10_alpha_pct, t20_return_pct, t20_alpha_pct, mfe_pct, mae_pct, source, fill_status)`：推荐记录与 `T+5/10/20` Alpha、`MFE/MAE` 极值及反思记忆；`source='pick'|'report'`（`/report` 的 ADD/TRIM/STOP 也进入同一 ledger），`fill_status='FILLED'|'NOT_FILLED'|'ASSUMED'|'UNKNOWN'`。
5. `chat_history(id, cycle_date, user_query, bot_reply, created_at_ms)`：单周期对话记忆（自动保留最近 12 轮）。
6. `llm_usage_logs(id, ..., cycle_date, call_label, model, thinking_level, api_rounds, prompt_tokens, cached_tokens, output_tokens, thought_tokens, estimated_cost_usd, latency_ms, success, search_queries)`：Token、缓存与 Search 用量账本。
7. `ops_events(id, created_at_uk, created_at_ms, severity, source, alert_key, message, notified, suppressed_count)`：运维告警审计。
8. `scheduler_runs(job_name, uk_date, fired_at_uk, PK(job_name, uk_date))`：定时任务幂等锁。
9. `ohlcv_cache(symbol, period, last_bar_date, last_close, updated_at_ms, payload_json, PK(symbol, period))`：`2y` 日线增量缓存。
10. `grounding_news_cache(cache_key PK, scope, symbol, summary_text, updated_at_ms)`：Google Search 联网摘要缓存（默认 `3h` TTL）。
11. `t212_instruments_metadata(ticker PK, yahoo_symbol, isin, currency_code, short_name, inst_type, extended_hours, working_schedule_id, isa_eligible, invest_eligible, updated_at_ms)`：T212 标的元数据与可买性（永久保留）。
12. `t212_dividend_archive(id PK, account, ticker, yahoo_symbol, amount_gbp, gross_per_share, paid_on, paid_on_ms)`：T212 历史实收英镑股息归档（永久保留）。
13. `quant_health_audits(audit_date PK, created_at_uk, summary_text, payload_json, ai_diagnosis)`：周末离线候选池与阈值体检档案 + DeepSeek 周度诊断（`<=450` 字，老库经 `_ensure_columns` 幂等补列）。
14. `fundamentals_cache` / `active_universe_pool` / `index_constituents_cache`：多市场 150 只精锐个股底池海选持久化与 21 天基本面缓存。
15. `wallet_balances(account_label, currency, available_amount, reserved_amount, updated_at_uk, updated_at_ms, source, note, PK(account_label, currency))`：Invest `GBP+USD` 双钱包本地校准锚点（默认种入 `USD=0.0`）。
16. `user_preference_notes(id, source_cycle_date, quote_text, rule_summary, expires_on_uk, created_at_ms)`：带**用户原话锚点**与 TTL 的长期偏好记忆（默认 14 天过期，每日 ≤3 条，`quote_text` 必须是原始发言的逐字子串）。
17. `pending_order_snapshots(account_label, order_id, ..., first_seen_ms, last_seen_ms, PK(account_label, order_id))`：挂单活动快照（与历史成交交叉验证推送成交回执，保留 3 天）。
18. `nav_history(uk_date PK, total_value_gbp, cash_gbp, invested_gbp, net_flow_gbp, isa_value_gbp, invest_value_gbp, benchmark_close, updated_at_ms)`：日频 GBP 标量净值锚点（供 `/perf` TWR 剔入金绩效归因，**永不清理**）。
