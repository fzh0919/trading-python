# Trading 212 × Gemini AI Advisor — 开发者与 AI 维护手册 (DEVELOPER_GUIDE)

本文档为紧凑型工程交接手册，供开发者与后续接入的 AI 助手以最少 Token 快速掌握架构边界、核心契约、N95 本地重算升级方案、**按零冲突依赖顺序排列的模块改进追踪表（`IMP-01 ~ IMP-19`）**与历次变更日志。

---

## 一、核心设计铁律与账户实况前提

1. **`/report` 核心基石地位（严禁降级铁律）**：
   - `/report` 是全系统唯一的全景事实底座与风控总纲，是同交易周期（每日 `07:30 UK` 切换）内 `/pick`、`/us` 与自由聊天框的**绝对参照物（Single Source of Truth）**。
   - **任何优化严禁降低 `/report` 的规模、上下文完整性、模型档位（`Argon HIGH` + `Hydrogen HIGH`）、工具调用轮次（`6` 轮）或全市场分域联网检索深度**。
2. **N95 本地重算换 Token 策略（定时任务不怕慢，日常交互极速轻量）**：
   - **硬件算力充沛**：系统从 GCP `e2-medium` 迁移至英国本地常驻主机 **Intel N95 + 16GB RAM + 512GB SSD**，本地 CPU 与内存算力完全充裕。
   - **核心原则**：`/report`、`/pick`、`/us` 为后台定时任务，本地 Python 多算 10~30 秒完全无感。因此必须将**筹码分布（CYQ）、周线共振、K 线突破形态、公司身份与财报/除息日历锚定、本地多空证据冲突表（Disagreement）、T+5/T+10 历史建议 Alpha 结算**全部放在本地 Python 0-Token 算完，组装成高密度 `AnalysisContextPack` 喂给大模型，从而减少大模型盲目调工具的轮数；而**日常自由聊天与平淡交易日的 `/us` 则复用今日快照并默认采用轻量模型（`Hydrogen LOW`）**。
3. **算力配额与月度预算边界（Gemini Paid Tier）**：
   - 运行在 Google AI Pro 订阅下的 **Gemini API Paid Tier**（含每月 \$10 USD Credit + Prepaid 余额，享每月 5,000 次免费 Google Search Grounding）。
   - **预算目标**：月度总消耗控制在 **\$10 USD 以内为完美状态，月度硬上限约 \$20 USD**。
4. **账户实况简化约束（只留 `GBP` 现金 + 标的本地货币报价 + 不用 Pie/分红复投）**：
   - **持仓规模与简化边界**：ISA 与 Invest 每个账户日常各约 **10 只持仓**（双账户合计 ~20 只）；**不使用 Pie，不开启分红自动再投资**（开发时无需考虑 Pie 或股息复投逻辑）。
   - **钱包 `GBP` 结算 vs 标的本地货币（Native Currency）报价**：双账户均**只保留 `GBP` 现金**，因此 T212 返回的账户现金与 `walletImpact`（持仓市值 `currentValue`、成本 `totalCost`、浮盈 `unrealizedProfitLoss`、汇率影响 `fxImpact`、成交净额 `netValue`）**全部统一为 `GBP` 英镑**；而标的现价、持仓均价（`averagePricePaid`）、挂单触发价（`limitPrice`/`stopPrice`）、成交价（`fill.price`）及技术面/目标价均按**标的所属交易所本地货币（美股 `USD`、欧股 `EUR`、英股 `GBp` 便士 / `GBP` 英镑）**计价。Python 层必须严格区分账户结算币种（`GBP`）与标的报价币种（`inst_currency`），严禁交给 LLM 口算。
5. **4.0% 无风险现金锚 + 摩擦门槛 + 反迎合审计**：
   - 闲置 `GBP` 现金享 ~4.0% 年化利息；非英镑标的每次买卖固定发生 `0.15%` 换汇费（往返 `0.30%`）；英国个股买入收 `0.50%` 印花税（LSE ETF 免印花税）。对历史 AI 结论和用户已有活动挂单保持冷酷客观，跑不赢 4% 现金利息直白裁决 `CANCEL` / `MODIFY`。
6. **跨端开发运维与纯文本控字**：
   - 系统时钟锁死 `Europe/London`；开发端为英国本地主力 Windows 台式机（与 N95 同网段局域网直连）+ 出差随身 Windows 笔记本（通过 SSH / Tailscale 异地维护）。
   - 输出采用纯文本倒金字塔排版（**严禁使用 `#` 与 `**`**），硬控字数上限防止 Telegram 截断。

---

## 二、目录结构与三大 `REF` 项目借鉴映射

```text
trading-python/                     # [当前主项目]
├── docs/
│   ├── T212_API_manual.json        # [官方规范] Trading 212 Public API v0 OpenAPI 3.0 手册（含 Position.createdAt 与限流配额）
│   └── finnhub_API_manual.json     # [官方规范] Finnhub API OpenAPI 规范手册（免费 60次/分；/quote、/calendar/earnings、/company-news、/stock/profile2、/stock/recommendation 等参考）
├── _ref/                           # [本地只读参考区，已被 .gitignore 与 .dockerignore 排除]
│   ├── OLD_trading-gs/             # [已停用] 初代 GAS 脚本 (v4.2~v4.4)，仅作历史对照
│   ├── REF_stock-master/           # [参考库 1] 重点借鉴：instock/core/kline/cyq.py (筹码分布算法 CYQCalculator)、
│   │                               #           indicator/calculate_indicator.py (MACD/BOLL/ADX)、strategy/ (平台突破/低ATR形态)
│   ├── REF_TradingAgents-main/     # [参考库 2] 重点借鉴：memory/settlement.py & reflection.py (T+5后验Alpha结算与2~4句微型反思)、
│   │                               #           agents/context.py (确定性标的身份锚定防幻觉)
│   └── REF_daily_stock_analysis-main/ # [参考库 3] 重点借鉴：analysis_context_pack (当日预计算快照复用)、agent/disagreement.py (0-Token多空分歧表)、
│                                      #           agent/risk_override.py (后置风控护栏)、chat_object_decision.py (单聊标的提取)、provider_cache.py (缓存前缀对齐)
├── .env.example / config.yaml      # 密钥模板 / 模型路由、日程、风控阈值与候选股票池
├── deploy/                         # trading-advisor.service (systemd) & upgrade.sh (一键热升级)
├── DEVELOPER_GUIDE.md              # 本文档：架构契约 + IMP 拓扑顺序追踪表 + 复盘台账 + Changelog
    ├── README.md                   # 项目背景与部署使用说明
    ├── src/
    │   ├── main.py                 # 入口：初始化 SQLite/OpsNotifier/各Client，启动调度器与长轮询
    │   ├── version.py              # 版本号单一真源 (APP_VERSION)
    │   ├── config.py               # Pydantic 配置加载 (.env + config.yaml)
    │   ├── ops_notifier.py         # 统一 Telegram 运维告警（WARN 30m 冷却、CRITICAL 防抖、离线队列恢复）
    │   ├── data_providers/
    │   │   ├── t212_client.py      # T212 API v0 只读客户端（summary/orders 1req/5s, positions 1req/1s, history 20req/1m）
    │   │   ├── market_data.py      # Yahoo Finance + Finnhub + FRED 并发抓取、身份日历锚定与 TTL 缓存
    │   │   └── analytics.py        # MA/RSI14/ATR/周线共振、CYQ 筹码分布、去 Beta 残差相关性、跨账户与行业暴露
    │   ├── engine/
    │   │   ├── calendar_utils.py   # 英美夏令时错位周、NYSE/LSE 假日、半日市与交易日推算
    │   │   ├── portfolio_rules.py  # 持仓确定性评估、0-Token 多空冲突摘要 (Disagreement) 与后置风控护栏 (RiskOverride)
    │   │   └── screener.py         # /pick Python 零 Token 四维量化初筛（基本面 + CYQ 筹码 + 突破形态 + 低相关）
    │   ├── storage/
    │   │   └── db.py               # SQLite 核心表异步持久化、07:30 UK 周期管理、ContextPack 快照与 T+5 推荐结算
    │   ├── agents/
    │   │   ├── llm_client.py       # google-genai 封装（重试、备援降级、ThinkingLevel 钳位、工具调用、Token/Search 埋点；已按用户要求移除程序计价）
    │   │   ├── prompts.py          # 共享投资者画像、摩擦规则与 system_instruction 拼装器
    │   │   ├── schemas.py          # MorningReportDecision / PickDecision、JSON 自修复与倒金字塔渲染器
    │   │   ├── tools.py            # MarketToolkit 工具箱 + 分域（英/欧/美/ETF）强制 Google Search
    │   │   ├── report_agent.py     # /report 三步流水线（核心基石） & /us 美股异动门控哨兵
    │   │   ├── pick_agent.py       # /pick & 周五盘后选股雷达（Researcher -> Risk Critic -> Post-Screen 校验）
    │   │   └── chat_agent.py       # /pos 0-Token 瞬时快照 & 标的切片轻量单聊（默认 Hydrogen LOW，支持 /deep）
    │   └── bot/
    │       ├── telegram_client.py  # httpx 长连接复用、长轮询、状态编辑与 3800 字分块发送
    │       ├── handlers.py         # 指令路由、chat_id 白名单、单飞互斥锁、/health、/usage 与 /stats 胜率看板
    │       └── scheduler.py        # 工作日早报（含断网补发）、美股盘前哨兵、周五雷达与每日盘后 T+5 自动结算
    └── tests/                      # 纯离线单元测试套件（Mock HTTP/LLM，零外部请求）
```

---

## 三、核心契约速查（模型路由 / 字数限制 / 数据库 Schema）

### 3.1 Agent 模型路由与输出契约（`config.yaml` -> `src/agents/`）
| 业务流 / 节点 | 配置槽位 (`models.*`) | 默认模型 & 思考深度 | 工具 / 运行策略 | 最终输出硬上限 |
| :--- | :--- | :--- | :--- | :--- |
| **`/report` Step 1 联网（核心）** | `report_search_model` | `gemini-4-hydrogen` (`LOW`) | `GoogleSearch`（每批 `<=3` 标的，`Semaphore(3)`） | 注入内部上下文 |
| **`/report` Step 2A 机会（核心）** | `report_analyst_model` | `gemini-4-hydrogen` (`HIGH`) | 本地全维 `ContextPack` + `MarketToolkit`（`<=6` 轮） | 内部草稿 `<=1200` 字 |
| **`/report` Step 2B 风控（核心）** | `report_model` | `gemini-4-argon` (`HIGH`) | 本地全维 `ContextPack` + `MarketToolkit`（`<=6` 轮） | 内部草稿 `<=1200` 字 |
| **`/report` Step 3 PM 拍板（核心）**| `report_model` | `gemini-4-argon` (`HIGH`) | `MorningReportDecision` + 自修复 + `RiskOverride` | **Telegram `<=1500` 字** |
| **`/pick` Stage 1 投研员** | `pick_researcher_model` | `gemini-4-hydrogen` (`HIGH`) | 本地四维量化+筹码 Top 候选 + `MarketToolkit`（`<=4` 轮）| 内部提案 |
| **`/pick` Stage 2 风控官** | `pick_model` | `gemini-4-argon` (`HIGH`) | `PickDecision` + 自修复 + Post-Screen 确定性校验 | **Telegram `<=1000` 字** |
| **`/us` 美股增量哨兵（分级门控）** | `us_sentinel_model` | 平稳日 `Hydrogen LOW` / 异动日 `Hydrogen MEDIUM` (或 `Argon MEDIUM`) | 本地异动门控（Anomaly Gate）：无异动免强搜极简播报；有异动深挖 | **Telegram `<=450` 字** |
| **日常自由单聊（默认轻量）** | `chat_model` | `gemini-4-hydrogen` (`LOW`)（`/deep` 切高阶） | 本地提取提及标的切片今日快照 + 按需工具（`<=3` 轮） | **Telegram `<=500` 字** |
| **盘后 T+5 复盘微反思** | `search_model` / `chat_model` | `gemini-4-hydrogen` (`LOW`) | 本地 0-Token 算完 T+5 Alpha 后生成 `2~4` 句（`<80 tokens`）教训 | 存入 SQLite 记忆列 |

### 3.2 SQLite 数据库核心表速查（`data/advisor.db`，周期切换点 `07:30 UK`）
1. `portfolio_snapshots(id, cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json)`：双账户完整 JSON 快照。
2. `morning_reports(cycle_date PK, report_text, conclusion_text, premarket_note, daily_guardrails, context_pack_json, updated_at_ms)`：当日 `/report` 全文、`daily_vetoes` JSON、`/us` 笔记及当日预计算 `AnalysisContextPack` 快照（**供 `/pick`、`/us` 与 `chat` 跨指令 0-Token 复用**）。
3. `pick_radars(id, created_at_uk, created_at_ms, focus_query, research_text, final_text)`：`/pick` 投研草稿与终审报告存档。
4. `pick_recommendations(id, created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis, horizon_days, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, outcome_status, reflection_lesson)`：结构化推荐与 **T+5/T+10 后验结算及微型反思档案**（供 `/report`、`/pick` 注入历史教训及 `/stats` 看板读取）。
5. `chat_history(id, cycle_date, user_query, bot_reply, created_at_ms)`：单周期对话记忆（自动裁剪保留最近 `12` 轮）。
6. `llm_usage_logs(id, ..., cycle_date, call_label, model, thinking_level, api_rounds, prompt_tokens, cached_tokens, output_tokens, thought_tokens, estimated_cost_usd, latency_ms, success, search_queries)`：Token、缓存命中与 Google Search 用量账本。
7. `ops_events(id, created_at_uk, created_at_ms, severity, source, alert_key, message, notified, suppressed_count)`：运维告警审计表。
8. `scheduler_runs(job_name, uk_date, fired_at_uk, PK(job_name, uk_date))`：定时任务当日触发幂等锁。

### 3.3 核心模块接口字典（精确签名与字段契约速查，随 `IMP` 滚动更新）

> 💡 下游窗口直接按本字典接线，**不必通读千行上游源码**；后续新增公共函数/字段后，须在对应小节同步补齐 1~2 行契约。

- **`src/config.py`（`IMP-01`）**：
  - `ModelsSection` 新增：`chat_deep_model`, `chat_deep_thinking_level`, `google_search_free_quota_monthly` (`5000`), `google_search_warn_pct` (`85.0`)
  - `StrategySection` 新增：`us_anomaly_move_pct` (`1.8`), `us_anomaly_vix_change_pct` (`4.0`), `settlement_holding_days` (`5`)
  - `AppConfig.reload_pools_if_changed() -> bool`（按 `mtime_ns` 原地热重载 `watchlist` 与 `universe`，YAML 损坏时保留旧配置）
- **`src/engine/calendar_utils.py`（`IMP-02`）**：
  - `infer_market_from_symbol(symbol) -> str`（返回 `US | UK | XETRA | EURONEXT`）
  - `is_trading_day(day, market="US") -> bool` | `next_trading_day(start_date, market="US", inclusive=False) -> date`
  - `add_trading_days(start_date, n_days, market="US") -> date`（跳过休市日精确推算 `T+N`）
  - `get_market_close_utc(day, market="US") -> datetime | None` | `has_market_closed_on_date(target_date, market="US", now_utc=None) -> bool`
  - `is_us_early_close / is_uk_early_close / is_eu_early_close / is_uk_bank_holiday / is_eu_market_holiday` | `get_market_calendar_alerts(now_utc=None) -> list[str]`
  - `UsMarketSchedule` 新增字段：`is_us_early_close: bool`, `close_time_et: str`（已动态调整 `close_time_uk` 与 `minutes_to_close`）
- **`src/storage/db.py`（`IMP-03`）**：
  - 周期/快照：`await db.get_latest_snapshot() -> PortfolioData | None`；`await db.save_morning_report(..., context_pack: dict | str | None = None)`；`await db.get_context_pack(cycle_date=None, *, allow_fallback=True) -> dict | None`；`await db.close()`
  - 结算/反思：`await db.get_pending_settlements(limit=50, *, only_with_entry_price=False)`；`await db.settle_recommendation(rec_id, *, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, outcome_status="SETTLED", reflection_lesson="") -> bool`（仅更新 `PENDING`）；`await db.get_symbol_lessons(symbol, limit=3)`；`await db.get_recent_lessons(limit=5)`；`await db.get_recommendation_stats() -> dict`
  - 对话/告警/用量/清理：`await db.get_previous_cycle_chat_history(cycle_date=None, limit=2)`；`await db.get_unnotified_ops_events(*, hours=24, severities=("CRITICAL", "WARN"), limit=20)`；`await db.mark_ops_events_notified(event_ids)`；`await db.get_usage_summary(days=30, now_utc=None)`（含 `"today"`, `"last_7d"`, `"last_nd"`, `"this_month"`, `"by_label_7d"`）；`await db.prune_old_records(retention_days=90, *, now_utc=None) -> dict[str, int]`
- **`src/data_providers/t212_client.py`（`IMP-04`）**：
  - `T212Client(..., snapshot_loader: Callable[[], Awaitable[PortfolioData | None]] | None = None)`；`set_snapshot_loader(...)`；`await t212.close()`
  - `PositionItem` 新增：`created_at: str | None = None`, `isin: str | None = None`；`PendingOrderItem` 新增：`filled_quantity: float = 0.0`, `status: str = ""`
  - `AccountSnapshot` 新增：`recent_transactions: list[HistoricalTransactionItem]`, `recent_dividends: list[HistoricalDividendItem]`, `pending_orders_error`, `is_fallback`, `fallback_source`, `fallback_reason`, `snapshot_fetched_at_uk`
  - `PortfolioData` 新增：`is_fallback: bool`, `fallback_source: str`, `fallback_note: str`
  - `build_trade_context(...)` 返回字典键：`pending_orders_text`, `recent_trades_text`, `trade_summary_30d_text`, `cash_flow_summary_text`, `traded_tickers`
- **`src/data_providers/market_data.py`（`IMP-05`）**：
  - `await market_data.get_histories(symbols: list[str], period="1y") -> dict[str, pd.DataFrame]`（0 次 `.info`，含 Singleflight 与 LRU）
  - `await market_data.get_bundles(symbols, include_news=True, include_earnings=True, include_info=True, isin_map: dict[str, str] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, TickerMarketBundle]]`
  - `await market_data.get_us_realtime_quote(symbol)`；`await market_data.get_us_deep_intel(symbol) -> dict`（6h 缓存，含 `recommendations`, `earnings_surprises`, `insider_sentiment`）；`await market_data.close()`
  - `TickerMarketBundle` 新增：`long_name`, `exchange`, `isin`, `identity_summary` (`build_identity_anchor(isin_override=None)`), `beta`, `profit_margin_pct`, `revenue_growth_pct`, `roe_pct`, `debt_to_equity`, `analyst_count`, `recommendation_key`, `ex_dividend_date`, `days_to_ex_dividend`；`MacroSnapshot` 新增：`us_2y_yield`, `yield_spread_10y_2y`, `breakeven_10y_pct`, `ecb_deposit_rate`, `is_stale: bool`
- **`src/data_providers/analytics.py`（`IMP-06`）**：
  - `compute_cyq_metrics(hist, cyq_days=210, factor=150) -> CYQMetrics | None`（字段：`benefit_part_pct`, `avg_cost`, `concentration_90_pct`, `cost_90_low`, `cost_90_high`, `concentration_70_pct`, `regime`）
  - `compute_drawdown_since_date(hist, created_at=None, max_lookback_days=60) -> float | None`
  - `compute_technical_snapshot(symbol, hist, position_created_at=None, bundle=None) -> TechnicalSnapshot | None`
  - `TechnicalSnapshot` 新增：`atr14`, `atr_pct`, `volume_ratio_20d`, `weekly_ma10`, `weekly_ma30`, `weekly_bullish`, `weekly_rsi14`, `bb_width_pct`, `bb_squeeze`, `adx14`, `cyq_benefit_part_pct`, `cyq_avg_cost`, `cyq_concentration_90_pct`, `cyq_cost_90_low`, `cyq_cost_90_high`, `cyq_regime`, `days_to_ex_dividend`, `is_ex_div_window` (`-3..3`), `is_ex_div_drop_guard` (`-3..0`)
  - `resolve_market_benchmark(symbol, default_benchmark="SPY", uk_benchmark="ISF.L", eu_benchmark="EXS1.DE") -> list[str]`
  - `compute_correlation_matrix(histories, symbols=None, lookback_days=60, benchmark_symbol="SPY", uk_benchmark_symbol="ISF.L", eu_benchmark_symbol="EXS1.DE") -> pd.DataFrame`
  - `infer_etf_index_family(symbol, bundle=None, name_hint="") -> str | None` | `compute_exposure_summary(portfolio, bundle_map) -> str`
- **`src/engine/portfolio_rules.py`（`IMP-07`）**：
  - `evaluate_holding(..., hist=None, pending_orders=None, account_total_value=None, total_portfolio_value=None) -> HoldingEvaluation`（新增 `native_return_pct`, `effective_stop_pct`, `pending_buy_weight_pct`, `has_pending_stop`, `ex_div_guarded`）；`evaluate_portfolio(..., histories=None)`
  - `build_disagreement_matrix(data, tech_map, bundle_map, strategy, evals=None) -> list[DisagreementItem]` | `format_disagreement_matrix_text(items) -> str`
  - `apply_risk_override_guardrail(decision, *, portfolio=None, macro=None, strategy=None, bundle_map=None, evals=None, high_correlation_symbols=None, daily_vetoes=None, vix_danger_threshold=28.0) -> RiskOverrideResult`（含 `overrides: list[RiskOverrideRecord]`, `notes: list[str]`）
- **`src/agents/llm_client.py`（`IMP-01`）**：
  - `normalize_thinking_level(model: str, thinking_level: str) -> str`（从已移除的 `pricing.py` 迁入；`pricing.py` 已按用户要求删除，仅记录 Token 与 `search_queries`）

### 3.4 模块 ↔ 测试文件 ↔ 共享 Mock 桩映射表（改哪块跑哪个文件）
| 源码模块 (`src/`) | 对应单元测试 (`tests/`) | 共用 Mock / 桩对象（变更上游接口时须同步检查） |
| :--- | :--- | :--- |
| `config.py`, `llm_client.py` | `test_llm_pricing_and_usage.py` | Fake GenAI Client |
| `calendar_utils.py`, `scheduler.py` | `test_calendar_and_scheduler.py` | `_MockT212`, `_MockMarketData`, `_StubLLM`, `_StubTg` |
| `storage/db.py` | `test_db_storage.py` | 临时目录 SQLite 真实建库 |
| `t212_client.py` | `test_t212_client.py` | `httpx.MockTransport` |
| `market_data.py` | `test_market_data.py` | `_DummyOps`, Fake `httpx`/yfinance |
| `analytics.py`, `portfolio_rules.py`, `screener.py` | `test_analytics_and_rules.py` | 纯离线 `pd.DataFrame` 构造器（须含 `Open/High/Low/Close/Volume`） |
| `ops_notifier.py`, `tools.py`, `handlers.py(/health)` | `test_v504_ops_and_grounding.py` | `_StubTelegram`, `_FakeT212Healthy`, `_FakeMarketHealthy` |
| 全链路端到端 (`/pos`/`report`/`pick`/`us`/`stats`) | `test_db_and_end_to_end.py` | `_FakeT212`, `_FakeMarketData`, `_RecordingLLM`, `_RecordingTg` |

> ⚡ 单文件秒测：`python -m unittest tests.<模块名>`；改完收口必跑全量 `python -m tests.run_tests tests`。

### 3.5 待做模块 (`IMP-07 ~ 16`) ↔ `_ref/REF_*` 参考源码精确符号落点（免翻整库）
- **`IMP-07` (`portfolio_rules.py`)**：`_ref/REF_daily_stock_analysis-main/daily_stock_analysis-main/src/agent/disagreement.py`（`build_agent_disagreement_summary`）& `risk_override.py`（`build_risk_override_plan`, `_downgrade_signal`）
- **`IMP-08` (`screener.py`)**：`_ref/REF_stock-master/stock-master/instock/core/strategy/breakthrough_platform.py`（`check`）、`low_atr.py`（`check_low_increase`）、`enter.py`（`check_volume`）
- **`IMP-12` (`report_agent.py`)**：`_ref/REF_daily_stock_analysis-main/daily_stock_analysis-main/src/schemas/analysis_context_pack.py`（`AnalysisContextPack`）& `src/analysis_context_pack_prompt.py`（`format_analysis_context_pack_prompt_section`）
- **`IMP-14` (`chat_agent.py`)**：`_ref/REF_daily_stock_analysis-main/daily_stock_analysis-main/src/agent/chat_object_decision.py`（`decide_chat_object`, `object_mentions`）& `src/llm/provider_cache.py`
- **`IMP-16` (`scheduler.py`)**：`_ref/REF_TradingAgents-main/TradingAgents-main/tradingagents/memory/settlement.py`（`Settlement`, `fetch_returns`）& `reflection.py`（`Reflector`）

---

## 四、开发、测试、大文件读取与跨模块不变量铁律

- **大文件禁止全量 `read_file`（防上下文膨胀）**：
  - 对 **>400 行的源文件/测试文件**（如 `t212_client.py` 1555行、`market_data.py` 1356行、`analytics.py` 1287行、`db.py` 1068行及大型单测），**禁止直接整篇读入上下文**；必须先用 `rg -n "^(class |async def |def |    async def |    def )" <文件>` 定位行号，再按 100~150 行范围分段读取；上游接口签名直接查上文 **3.3 接口字典**。
- **每次修改代码后必须执行并通过**：
  ```bash
  python -m compileall -q src tests
  python -m tests.run_tests tests
  ```
- **5 条系统核心不变量（严禁破坏）**：
  1. **SQLite 写锁单入口**：所有写库必须走 `AdvisorDatabase._connect(write=True)`，严禁私建连接绕过 `self._write_lock` 引发 `database is locked`。
  2. **降级数据不回写**：`portfolio.is_fallback` 或 `macro.is_stale` 为 `True` 时严禁调用 `db.save_snapshot()` 或覆盖有效缓存。
  3. **双币种严格隔离**：账户现金/净值/盈亏固定为 `GBP`，标的现价/均价/止损用本地货币；英股须防 `GBp` 便士与 `GBP` 英镑 100 倍量级错位。
  4. **空行情防毒化 + 完整 K 线桩**：空 DataFrame 或全挂宏观响应不得写入缓存；所有行情测试桩必须返回包含 `Open/High/Low/Close/Volume` 的 DataFrame。
  5. **数据库延续 & 跨端打包边界**：线上 `data/advisor.db` 只做幂等迁移（`CREATE TABLE IF NOT EXISTS` / `_ensure_columns`），严禁删库；`patchX.Y.Z.zip` 内部路径强制正斜杠 `/`，并排除 `.env`、`data/`、`__pycache__/`、`.venv/`，同步更新 `src/version.py` 与 `pyproject.toml`。

---

## 五、模块评估与待改进追踪表（按零冲突拓扑执行顺序排列：`IMP-01` → `IMP-19`）

> [!IMPORTANT]
> **为什么按 `IMP-01` → `IMP-19` 顺序执行？**
> 下表已按**自底向上的严格拓扑依赖顺序（6 个阶段）**重排：
> `Phase 1 配置/日历/数据库底座 (IMP-01~03)` → `Phase 2 外部数据采集 (IMP-04~05)` → `Phase 3 N95本地零Token重计算引擎 (IMP-06~08)` → `Phase 4 LLM网关/Schema/工具箱 (IMP-09~11)` → `Phase 5 三大AI业务编排器 (IMP-12~14)` → `Phase 6 Telegram路由/调度/运维部署收口 (IMP-15~19)`。
> **按此编号从 `IMP-01` 依次做到 `IMP-19`，前序模块的接口和数据结构直接被后续模块消费，彻底杜绝改后面破前面、或反复修改顶层 Agent 的冲突！**

### 5.1 跨对话窗口执行与回填规范（每个新开对话窗口必读！）

当开发者在**干净的新对话窗口**中指派执行某个 `IMP-xx`（例如：“请按 `DEVELOPER_GUIDE.md` 执行 `IMP-07`”）时，负责执行的 AI 助手必须严格遵守以下闭环：
1. **对标实现**：阅读 **3.3 接口字典**、**3.5 `REF_*` 精确落点**、本表中该 `IMP-xx` 条目以及源码内的 `# TODO(IMP-xx)` 注释，必要时按需查阅官方手册（`T212_API_manual.json`、`finnhub_API_manual.json`）；**严禁削弱 `/report` 的规模与分析质量**。
2. **编写并运行单测**：对照 **3.4 测试映射表** 补充/更新离线单测，开发期可用单文件秒测，收尾必须跑通 `python -m compileall -q src tests` 和 `python -m tests.run_tests tests`。
3. **精简打勾并控制文档体积（防膨胀上限）**：
   - 将 5.2 主表对应行的状态改为 `- [x] 已完成 (YYYY-MM-DD)` 并勾选全部 `- [x]` 小项；
   - 最右侧 **「完成改动记录」只保留 3~4 条简练要点（不超过 150 字，仅写：公开契约变化、核心修复与单测总数）**，严禁把私有函数或内部实现细节流水账堆进表格；详细修改历史如确需留存，追加到 [`CHANGELOG.md`](./CHANGELOG.md)。
4. **在 5.3 复盘台账追加 1 行摘要**：只记修改文件、一句核心结论与单测结果，保持主手册长期精简可读。
5. **双向接口交接与字典同步（允许为性能/效率调整签名，但须写明交接备注）**：若因优化需要新增或修改函数参数/数据结构，无需死板禁止改动，但**严禁只把提醒留在自己行内**；必须同步写到**“收件人”位置**并更新契约字典：① 若暴露或修改了公共函数/数据结构，在 **3.3 核心模块接口字典** 同步增补/更新 1~2 行精确签名与字段；② 在 5.2 主表对应**下游 `IMP-yy` 行**的待改进项中追加 `[接自 IMP-xx]` 勾选条目；③ 在下游待改 `.py`/测试文件对应调用行上方补充 `# TODO(IMP-yy): [接自 IMP-xx] ...` 注释（若改动必须联动他模块才不报错，则在表格和 5.3 中写明跨模块连带说明，待全部完成后统一联调验收）。

---

### 5.2 全模块拓扑顺序改进主表（`IMP-01` → `IMP-19`）

| 执行序号 & 优先级 | 完成状态 | 所属阶段 & 模块路径 | 模块职责与现状评分 | 代码逻辑诊断 & 具体待改进项（Actionable Checklist） | 完成改动记录（由执行窗口完成后打勾并回填做了什么） |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **IMP-01**<br>Step #1 (P1) | - [x] 已完成 (2026-11-14) | **Phase 1: 底座契约**<br>`src/config.py` &<br>`config.yaml` *(已按用户要求移除 `pricing.py`)* | **配置加载与纯 Token/Search 用量底座**<br>现状：**9.5 / 10** | - [x] 新增 `us_anomaly_*`、`settlement_holding_days`、`chat_deep_*`、`google_search_free_quota_monthly` 配置与边界校验<br>- [x] 实现 `AppConfig.reload_pools_if_changed()`（`mtime` 热重载+坏 YAML 保护）<br>- [x] 移除 `pricing.py`，迁入 `normalize_thinking_level` 至 `llm_client.py` 并切至纯 Token + Search 用量 | 新增上述配置契约与 `mtime` 热重载；删除 `pricing.py` 并保留库表兼容；新增 4 组单测，全套 `24 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-02**<br>Step #2 (P2) | - [x] 已完成 (2026-11-14) | **Phase 1: 底座契约**<br>`src/engine/calendar_utils.py` | **英美欧交易日历引擎**<br>现状：**9.6 / 10** | - [x] 新增美股 `13:00 ET`、LSE `12:30 UK`、Euronext `14:05 CET` 半日市与欧股休市日识别<br>- [x] 新增 `add_trading_days`、`next_trading_day`、`infer_market_from_symbol`、`has_market_closed_on_date`、`get_market_calendar_alerts` | 补全英美欧半日市/休市及交易日推算与收市判定，支持动态半日市收盘时间（同步至 `3.3` 与 `IMP-12~16`）；全套 `25 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-03**<br>Step #3 (P0) | - [x] 已完成 (2026-11-14) | **Phase 1: 底座契约**<br>`src/storage/db.py` | **SQLite 异步数据仓库**<br>现状：**9.5 / 10** | - [x] 配置 `WAL`/`busy_timeout=5000` + `asyncio.Lock` 进程内写锁<br>- [x] 扩展 `morning_reports.context_pack_json` 与 `get_latest_snapshot()`<br>- [x] 扩展 `pick_recommendations` 结算/反思 6 列与 `get_pending_settlements` / `settle_recommendation` / `get_recommendation_stats`<br>- [x] 新增 `prune_old_records(90)`、`get_unnotified_ops_events`、`this_month` 自然月统计 | 完成写锁串行化、幂等迁移、ContextPack 与 T+5 结算接口、跨周期聊天及 90 天清理（同步至 `3.3` 与 `IMP-09~18`）；新增 5 组单测，全套 `30 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-04**<br>Step #4 (P0) | - [x] 已完成 (2026-11-14) | **Phase 2: 数据采集**<br>`src/data_providers/t212_client.py` | **T212 双账户只读底座**<br>现状：**9.8 / 10** | - [x] 实例级 `httpx.AsyncClient` 连接池复用 + `close()` + `_portfolio_lock`<br>- [x] 解析 `Position.created_at` 与 `Position.isin`；支持限流头精准退避<br>- [x] 新增 `snapshot_loader` + 内存/DB 双级 5xx 回退（不回落 `401/403`）<br>- [x] 补全部分成交挂单剩余量与近 30 天出入金/派息流水 | 补齐连接池复用、限流头避让、开仓日/ISIN 解析、分级容错回退、部分成交及现金流水归因（同步至 `3.3` 与 `IMP-07~18`）；全套 `35 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-05**<br>Step #5 (P0) | - [x] 已完成 (2026-11-14) | **Phase 2: 数据采集**<br>`src/data_providers/market_data.py` | **多源行情与身份/日历聚合**<br>现状：**9.6 / 10** | - [x] 新增 0-info 批量接口 `get_histories(symbols, "1y")` 与 Singleflight 去重<br>- [x] 修复低股息率放大 100 倍 Bug；扩充 `TickerMarketBundle` 身份/质量/除息日与 `get_us_deep_intel`<br>- [x] 实现 `(symbol, period)` LRU 上限、防空值投毒、过期兜底与 FRED 扩展 | 新增批量日 K 线、身份锚定与除息日、质量因子、LRU 防毒与 Finnhub/FRED 扩展及 `close()`（同步至 `3.3` 与 `IMP-08~19`）；新增 8 组单测，全套 `43 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-06**<br>Step #6 (P0) | - [x] 已完成 (2026-11-14) | **Phase 3: N95 本地量化**<br>`src/data_providers/analytics.py` | **本地零 Token 量化与筹码引擎**<br>现状：**9.8 / 10** | - [x] 移植 `CYQMetrics`/`compute_cyq_metrics` 与 `_normalize_adjusted_ohlcv`（拆股/100倍坏点自愈）<br>- [x] 扩展 `TechnicalSnapshot`（`atr14`, `volume_ratio_20d`, 周线共振, `bb_squeeze`, `adx14`, `cyq_*`, 除息守卫）与 `compute_drawdown_since_date`<br>- [x] 分市场去 Beta 基准（`SPY`/`ISF.L`/`EXS1.DE`）、2日滚动时差补偿与克隆 ETF 暴露归并 | 完成全维本地定量与筹码引擎、开仓后回撤、跨市场去 Beta 与除息防误判紧凑渲染（同步至 `3.3` 与 `IMP-07~12`）；新增 4 组单测，全套 `47 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-07**<br>Step #7 (P0) | - [x] 已完成 (2026-11-14) | **Phase 3: N95 本地量化**<br>`src/engine/portfolio_rules.py` | **持仓规则、多空冲突表与风控护栏**<br>现状：**9.7 / 10**<br>*(移植 `REF_daily_stock_analysis` 分歧表与后置护栏)* | - [x] **利用 `Position.createdAt` 修复移动止盈时间盲区**：结合 `IMP-04` 的 `created_at` 与历史日 K，计算波段最高点回撤时仅截取 `max(60日前, 持仓开仓日)` 之后的最高价，彻底杜绝把建仓前历史高点误判为持仓回撤<br>- [x] **引入 ATR 自适应止损 & 联动活动挂单风控**：结合 `IMP-06` 的 `ATR(14)` 替代 Invest 仓一刀切的 `-12%` 固定止损；评估超重/止损时计入同标的已挂出的 `BUY`/`STOP` 挂单<br>- [x] **新增 0-Token 本地多空分歧摘要 `build_disagreement_matrix()`（参考 `REF_daily_stock_analysis/agent/disagreement.py`）**：对存在信号冲突的持仓（例如“基本面/长线趋势看多 vs 短期获利盘过高/逼近财报/行业超重看空”），由本地代码直接生成紧凑的正反方证据对照行，供 `/report` Step 3 PM 直接裁决<br>- [x] **新增后置确定性风控护栏 `apply_risk_override_guardrail()`（参考 `REF_daily_stock_analysis/agent/risk_override.py`）**：当宏观处于高危状态（如 `VIX >= 28`）或单股已触及跨账户 `22%` 集中度红线时，代码级强制将任何激进 `ADD / BUY` 建议降级为 `HOLD / WATCH` 并附注护栏原因<br>- [x] **[接自 IMP-04 & IMP-06] 消费已就绪的本地指标与除息防误判**：在规则中读取 `tech.is_ex_div_drop_guard`（或 `bundle.days_to_ex_dividend`）豁免除息日自然低开导致的假破位/假止损；读取 `tech.atr14`/`atr_pct`、`tech.cyq_regime`/`cyq_benefit_part_pct`、`tech.weekly_bullish`/`adx14`/`volume_ratio_20d` 参与评分与 `build_disagreement_matrix()`；消费 `pos.created_at` 时必须兼容 `None`（老仓或缺失字段回退 60d） | 接入开仓后回撤、ATR 自适应止损、除息防误判、挂单叠加、汇率校准与 4% 门槛；新增 `build_disagreement_matrix` 与 `apply_risk_override_guardrail`（同步至 `3.3` 与 `IMP-10/12/13`）；新增 3 组单测，全套 `50 tests OK`（详见 [CHANGELOG](./CHANGELOG.md)）。 |
| **IMP-08**<br>Step #8 (P1) | - [ ] 待开始 | **Phase 3: N95 本地量化**<br>`src/engine/screener.py` | **`/pick` 本地四维量化选股漏斗**<br>现状：**7.3 / 10**<br>*(移植 `REF_stock-master` 突破形态与基本面入模)* | - [ ] **移植 `REF_stock-master` 经典突破形态检测（`strategy/`）**：在本地初筛中增加 **平台突破（`breakthrough_platform`）、低 ATR 波动率收敛（`low_atr`）、放量站上均线（`enter.py`）** 形态识别加分项<br>- [ ] **四维综合打分入模**：将 **基本面估值空间（`forward_pe`, `dividend_yield_pct`, `target_mean_price` 上涨空间）** + **CYQ 筹码单峰密集度（`IMP-06`）** + **周/日线形态共振** + **与现有 ~20 只持仓的低相关性** 融合进 `score_candidate()`，支持传入已拉取的 `held_histories` 免重复请求<br>- [ ] **移除硬编码 ETF 列表 & 动态扩展观察池**：改为基于 `quote_type` 与配置识别 ETF，并支持自动将近 90 天实盘曾交易过的标的纳入扫描<br>- [ ] **[接自 IMP-04 & IMP-05 & IMP-06] 两段式批量拉取 + 本地基准去 Beta + 同步改测试桩**：① 第一阶段粗筛改调 `market_data.get_histories(fetch_needed, period="1y")`（0 次 `.info` 调用），仅对 Top 候选补调 `get_bundles(..., isin_map=...)`（`pos.isin` 需兼容 `None`）；② `fetch_needed` 一并加入 `"ISF.L"`（及欧股基准若池内有欧股），并在 `compute_correlation_matrix` 按 `IMP-06` 启用分市场基准，在 `compute_technical_snapshot` 传入 `bundle`、在 `score_candidate` 消费 `bb_squeeze`/`volume_ratio_20d`/`weekly_bullish`/`cyq_*`；③ 改用 `get_histories` 时**须同步在 `tests/test_db_and_end_to_end.py` `_FakeMarketData` 与 `tests/test_calendar_and_scheduler.py` `_MockMarketData` 补齐 `get_histories` 桩及完整 OHLCV 列** | *（待执行窗口完成后回填）* |
| **IMP-09**<br>Step #9 (P1) | - [ ] 待开始 | **Phase 4: AI 基础设施**<br>`src/agents/llm_client.py` | **Gemini 异步网关与并发/超时控制**<br>现状：**8.6 / 10**<br>*(参考 `REF_daily_stock_analysis/provider_cache.py`)* | - [ ] **单轮调用增加硬超时（`asyncio.wait_for`）**：防止 `HIGH` 思考或联网搜索偶发无限挂起<br>- [ ] **同轮多工具并发执行**：将 `for call in fn_calls` 串行等待改为 `asyncio.gather` 并发调用，缩短 `/report` 与 `/pick` 工具轮耗时<br>- [ ] **多轮工具返回紧凑裁剪 & Search 免费额度预警**：限制单次工具返回的最大字符数防止多轮循环 Input Token 滚雪球；按用户要求不再做程序 USD 计价，改为当 Google Search 查询量逼近 `google_search_free_quota_monthly`（5,000 次）预警线时通过 `OpsNotifier` 发送提醒<br>- [ ] **[接自 IMP-03] 统一 Search 预警统计口径**：结合 `db.get_usage_summary()` 已新增的自然月口径 `"this_month"`（英国时区当月 1 号起算）与 `IMP-15` 对齐触发自然月额度预警，而非仅按滚动 `last_nd` 30 天计算 | *（待执行窗口完成后回填）* |
| **IMP-10**<br>Step #10 (P1) | - [ ] 待开始 | **Phase 4: AI 基础设施**<br>`src/agents/schemas.py` | **结构化 Schema 与控字渲染器**<br>现状：**8.5 / 10**<br>*(适配后置风控护栏标记与结构化去重)* | - [ ] **`flagged_positions` 结构化升级**：由纯文本 `list[str]` 升级为支持 `symbol` 与 `summary` 的结构化项（兼容旧字符串），彻底消除脆弱的正则切割去重<br>- [ ] **支持渲染后置风控护栏标记（Risk Override Note）**：当 `IMP-07` 的确定性风控护栏拦截或降级了某条激进建议时，在渲染输出中清晰展示 `[🛡️风控护栏修正]`<br>- [ ] 优化 `_fit_max_chars` 尾部裁剪，避免裁剪后在末尾留下孤立的 `【...】` 标题行<br>- [ ] **[接自 IMP-07] 对接 `RiskOverrideResult`**：渲染器可接收并高亮 `apply_risk_override_guardrail()` 输出的 `overrides` / `notes` | *（待执行窗口完成后回填）* |
| **IMP-11**<br>Step #11 (P1) | - [ ] 待开始 | **Phase 4: AI 基础设施**<br>`src/agents/tools.py` | **Agent 工具箱与分域强制联网**<br>现状：**8.1 / 10**<br>*(接入 CYQ/周线/结算记忆，强搜加 Semaphore)* | - [ ] **强制联网并发限流**：在 `run_forced_market_grounding` 中增加 `asyncio.Semaphore(3)` 控制同时飞行的 Google Search 请求数（双账户 ~20 只持仓约 7~8 批），防止瞬间并发触发 `429`<br>- [ ] **`inspect_ticker` 增强**：返回信息中自动包含 `IMP-06` 算出的 CYQ 筹码获利盘/集中度、周线趋势、除息日及 `IMP-03` 中该标的的历史推荐 T+5 反思教训<br>- [ ] 清理过期 `_GROUNDING_CACHE` 条目，防止常驻内存膨胀<br>- [ ] **[接自 IMP-04 & IMP-05 & IMP-06] 对接纯行情接口、美股深度情报与分市场基准**：`check_portfolio_correlation` 改调 `market_data.get_histories()` 并补拉 `"ISF.L"`（传入分市场相关性矩阵）；`inspect_ticker` 调用 `compute_technical_snapshot(sym, hist, bundle=bundle)` 并传递已有持仓 `isin`（兼容 `None`），同时按需接入 `market_data.get_us_deep_intel(sym)`（6h 缓存）与 `bundle.identity_summary` / `ex_dividend_date` | *（待执行窗口完成后回填）* |
| **IMP-12**<br>Step #12 (P0) | - [ ] 待开始 | **Phase 5: AI 业务编排**<br>`src/agents/report_agent.py` | **`/report` 核心基石与 `/us` 门控哨兵**<br>现状：**8.2 / 10**<br>*(集成全维 ContextPack、多空冲突表与 `/us` 异动门控)* | - [ ] **组装并持久化 `AnalysisContextPack`（参考 `REF_daily_stock_analysis`）**：在 `build_portfolio_fact_sheet` 中整合 `bundle.identity_summary`、CYQ 筹码、周线共振、除息/财报锚点、`IMP-07` 本地多空冲突表（Disagreement）及 `IMP-03` 历史反思教训（`TechnicalSnapshot` 可直接 `model_dump()`），并在 `/report` 生成后通过 `db.save_morning_report(..., context_pack=...)` 存入 `context_pack_json`<br>- [ ] **强化 `/report` 多空交锋与后置护栏（严禁缩减 `/report` 规模）**：在 Step 3 PM 拍板 Prompt 中显式注入 Bull/Bear 分歧点与本地 Disagreement 矩阵；PM 输出后调用 `IMP-07` 的 `apply_risk_override_guardrail` 做安全兜底；修正 4h 净值快照回退时的时间跨度标注<br>- [ ] **`/us` 盘中哨兵引入本地异动门控（Anomaly Gate，省 Token 核心）**：开盘前/盘中先用本地 Python 检查美股持仓涨跌幅、挂单距离与 VIX 变动；若无任何持仓异动（均小于 `cfg.strategy.us_anomaly_move_pct` 与 `us_anomaly_vix_change_pct`）且宏观平稳，免除重复强搜并走 `Hydrogen LOW` 输出极简平安报；一旦检测到异动标或宏观急变，自动针对异动股开启强搜 + 深度归因<br>- [ ] **[接自 IMP-02 & IMP-04 & IMP-05 & IMP-06 & IMP-07] 必须接线清单**：① 调用 `compute_technical_snapshot` 时传 `position_created_at=pos.created_at`（兼容 `None`）与 `bundle=bundles.get(sym)`，调 `get_bundles` 传入 `isin_map`；② **当 `portfolio.is_fallback` 为 `True` 时跳过 `await db.save_snapshot(portfolio)`，防降级缓存污染快照表**；③ 在 `_compute_portfolio_diff_text()` 与早报 Prompt 消费 `recent_transactions` / `recent_dividends` / `cash_flow_summary_text` 区分外部出入金/派息与交易盈亏；④ 复用 `get_market_calendar_alerts()`（`IMP-02`）提示欧股休市与英美半日市；⑤ 调 `evaluate_portfolio(..., histories=histories)`、`build_disagreement_matrix()` 与 `apply_risk_override_guardrail()`（取 `res.decision` 渲染） | *（待执行窗口完成后回填）* |
| **IMP-13**<br>Step #13 (P1) | - [ ] 待开始 | **Phase 5: AI 业务编排**<br>`src/agents/pick_agent.py` | **`/pick` 与周五选股雷达编排器**<br>现状：**8.1 / 10**<br>*(复用持仓历史 + 注入复盘记忆 + Post-Screen 强校验)* | - [ ] **消除持仓行情并发重复拉取**：先获取或复用 `held_histories` / 当日 `AnalysisContextPack`，直接作为 `existing_histories` 传给 `IMP-08` 的 `run_quant_screener`<br>- [ ] **注入历史选股 T+5 复盘反思（参考 `REF_TradingAgents`）**：从 `db.py` 的 `get_recent_lessons()` / `get_symbol_lessons()` 读取近期选股结算的经验教训注入 Researcher 与 Risk Critic，防止对近期连续失效的板块/形态重复推荐（推荐落库配合 `cfg.strategy.settlement_holding_days`）<br>- [ ] **增加 Post-Screen 确定性代码强校验**：在 Risk Critic 输出 `PickDecision` 后，用真实本地货币 `bundle.current_price` 自动回填缺失的 `entry_price`，并调用 `IMP-07` 风控护栏硬性拦截超相关性阈值或违反当日 `daily_vetoes` 的推荐<br>- [ ] **[接自 IMP-02 & IMP-04 & IMP-05 & IMP-06 & IMP-07] 传递新建仓/ISIN 上下文与节假日提示**：`get_bundles` 传持仓 `isin_map`，`compute_technical_snapshot` 传 `position_created_at=pos.created_at` 与 `bundle=held_bundles.get(sym)`，注入 `bundle.identity_summary` / `ex_dividend_date`，复用 `get_market_calendar_alerts()`，并向 `evaluate_portfolio(..., histories=held_histories)` 与 `apply_risk_override_guardrail(..., high_correlation_symbols=..., daily_vetoes=...)` 接线 | *（待执行窗口完成后回填）* |
| **IMP-14**<br>Step #14 (P1) | - [ ] 待开始 | **Phase 5: AI 业务编排**<br>`src/agents/chat_agent.py` | **瞬时查仓（`/pos`）与轻量智能单聊**<br>现状：**8.4 / 10**<br>*(参考 `REF_daily_stock_analysis/chat_object_decision.py`)* | - [ ] **0-Token 本地提问标的提取与快照局部切片**：解析 `user_query` 中提及的股票代码/名称，直接从 `db.py` 的 `get_context_pack()` 当日快照中提取对应标的的筹码/技术/风控详情注入 Prompt，无需每次调工具即可秒级精准回答<br>- [ ] **对齐 Gemini Implicit Cache 静态前缀顺序**：将日内不变的 `/report` 原文、风控红线置前，动态持仓与当前问题置后，最大化多轮对话缓存命中率（享 75% 输入折扣）<br>- [ ] **支持 `/deep` 按需升级重模型 & 跨周期记忆平滑**：日常对话默认走 `Hydrogen LOW`（极速省钱）；当用户消息以 `/deep` 开头时自动切换 `cfg.models.chat_deep_model` / `chat_deep_thinking_level` 深挖；跨 `07:30 UK` 周期时通过 `db.get_previous_cycle_chat_history()` 保留上一周期最近 2 条核心对话摘要防断层<br>- [ ] **[接自 IMP-02] 休市/半日市与节假日提示**：复用 `calendar_utils.get_market_calendar_alerts()`（`IMP-02`）替下当前手写的局部银行假日拼接 | *（待执行窗口完成后回填）* |
| **IMP-15**<br>Step #15 (P1) | - [ ] 待开始 | **Phase 6: 交互与调度**<br>`src/bot/handlers.py` | **指令路由、互斥锁与看板中枢**<br>现状：**8.5 / 10**<br>*(防重入互斥锁 + `/usage` Token/Search 看板 + `/stats` 胜率看板)* | - [ ] **手动指令与定时任务共享互斥锁**：防止定时任务正在跑 `/report` 时用户手动点 `/report` 造成双倍并发消耗<br>- [ ] **`/usage` 强化纯 Token 与 Search 免费额度进度展示**：按用户要求不展示程序估算 USD 费用（账单以 AI Studio 为准），进一步增强 `cached_tokens` 命中率、Thinking Token 占比，并将 Search 免费额度统计统一切换为 `db.get_usage_summary()["this_month"]`（英国自然月 1 号起算，与 `IMP-03` / `IMP-09` 对齐）<br>- [ ] **`/stats` 升级为真实后验胜率与 Alpha 看板**：结合 `IMP-03` 的 `get_recommendation_stats()` / `get_recent_lessons()` 与当前本地货币现价，展示历史推荐胜率、平均超额 Alpha、近期经验教训及最近 10 条推荐表现<br>- [ ] **[接自 IMP-03 & IMP-04 & IMP-05] `/health` 补全降级与静默告警体检**：识别 `portfolio.is_fallback`/`acc.is_fallback`（禁止降级缓存显示纯绿灯）、`macro.is_stale`，并利用 `db.get_unnotified_ops_events()` 在体检中汇报尚未推送或曾被静默抑制的运维事件 | *（待执行窗口完成后回填）* |
| **IMP-16**<br>Step #16 (P1) | - [ ] 待开始 | **Phase 6: 交互与调度**<br>`src/bot/scheduler.py` | **时区感知调度与盘后自动复盘结算**<br>现状：**8.5 / 10**<br>*(N95 断网补发 + 挂载 `REF_TradingAgents` 盘后结算)* | - [ ] **新增早报断网/重启补发（Catch-up）与失败重试**：适配本地 N95 家庭宽带重拨场景，若 `07:30~07:35 UK` 因断网错过或返回 `⚠️` 失败，在 `09:30 UK` 前恢复时自动补跑成功后再锁定 `scheduler_runs`<br>- [ ] **挂载每日盘后推荐自动结算与微反思任务（`run_daily_settlement`，参考 `REF_TradingAgents/memory/settlement.py` & `reflection.py`） & 90 天清理**：每日定时调用 `db.get_pending_settlements()`，联合 `IMP-02` 的 `infer_market_from_symbol` + `add_trading_days` + `has_market_closed_on_date` 确认已满 `cfg.strategy.settlement_holding_days`（5 个完整交易日）且当日已正式收盘，用本地行情算出真实收益率与相对基准（`SPY`/`ISF.L`）的 Alpha，调用一次 `Hydrogen LOW` 生成 2~4 句话（`<80 tokens`）复盘教训并经 `db.settle_recommendation()` 幂等写回；定期调用 `db.prune_old_records(retention_days=90)` 瘦身<br>- [ ] **[接自 IMP-02] 尊重半日市动态收盘时间**：`weekly_radar` 已通过 `us_sched.minutes_to_close` 动态切至半日市（如感恩节次日 `18:30 UK`），切勿回退写死 `21:30 UK` | *（待执行窗口完成后回填）* |
| **IMP-17**<br>Step #17 (P2) | - [ ] 待开始 | **Phase 6: 运维与收口**<br>`src/ops_notifier.py` | **统一运维告警中心**<br>现状：**8.9 / 10**<br>*(断电重启后未达告警持久化补推)* | - [ ] **[接自 IMP-03] 断电重启后从 DB 捞取未送达告警**：`flush_pending()` 支持在本地 N95 主机断电/重启后，调用 `db.get_unnotified_ops_events()` 捞取最近 24 小时 `notified=0` 的关键告警合并补推，并通过 `db.mark_ops_events_notified()` 标记已送达（注意区分冷却抑制写入的 `notified=0, suppressed_count>0` 事件，避免冷却降噪记录刷屏） | *（待执行窗口完成后回填）* |
| **IMP-18**<br>Step #18 (P2) | - [ ] 待开始 | **Phase 6: 运维与收口**<br>`src/main.py` &<br>`src/bot/telegram_client.py` | **入口生命周期与 Telegram 通信层**<br>现状：**9.0 / 10**<br>*(家庭宽带瞬断退避 + 停机连接池释放)* | - [ ] `TelegramClient._post` 遇到 Telegram `429`（读取 `retry_after`）或家庭宽带瞬断时增加轻量指数退避重试<br>- [ ] **[接自 IMP-03 & IMP-04 & IMP-05] 注入 DB 冷启动快照兜底并全链路释放连接池**：在 `main.py` 创建 `T212Client` 时注入 `snapshot_loader=db.get_latest_snapshot`；停机 `finally` 中统一调用 `await t212.close()`、`await market_data.close()`、`await telegram.close()`（及 `await db.close()`）优雅释放所有连接资源 | *（待执行窗口完成后回填）* |
| **IMP-19**<br>Step #19 (P3) | - [ ] 待开始 | **Phase 6: 运维与收口**<br>`deploy/upgrade.sh` &<br>`tests/` | **热升级回滚与跨端打包脚本**<br>现状：**9.1 / 10**<br>*(单测失败自动回滚 + Windows 一键打包)* | - [ ] **`deploy/upgrade.sh` 增加单测失败自动回滚（Auto-Rollback）**：当第 5 步单测失败时，自动从备份还原旧代码与数据库并重启旧版服务，确保线上零中断<br>- [ ] **新增 Windows 端一键补丁打包脚本 `deploy/pack_patch.ps1`**：供英国本地台式机（同网段 LAN）或出差 Windows 笔记本一键校验并生成内部路径强制为 `/` 的 `patchX.Y.Z.zip`<br>- [ ] **[接自 IMP-05 & IMP-06] 集成测试与测试桩总复核**：确认 `tests/test_db_and_end_to_end.py`（`_FakeMarketData`）与 `tests/test_calendar_and_scheduler.py`（`_MockMarketData`）已完整实现 `get_histories` 与含 `Open/High/Low/Close/Volume` 的测试 K 线，补齐全链路端到端覆盖并通过最终编译与全量单测 | *（待执行窗口完成后回填）* |

---

### 5.3 改进执行复盘台账（Execution Log — 由各执行窗口完成后在此追加 1 行摘要）

> **填写模板**：`- **[YYYY-MM-DD] IMP-xx (模块名)**：窗口 | 修改：... | 核心变化：一句话结论 | 验证：Ran XX tests, OK`（详细流水已归档至 [`CHANGELOG.md`](./CHANGELOG.md)）

- **`[2026-11-14] IMP-01 (配置与纯 Token 底座)`**：修改 `config.yaml`, `src/config.py`, `src/agents/llm_client.py` 等并移除 `pricing.py` | 补齐热重载与新配置契约、切至纯 Token + Search 用量统计 | `24 tests OK`
- **`[2026-11-14] IMP-02 (英美欧交易日历引擎)`**：修改 `src/engine/calendar_utils.py`, `tests/test_calendar_and_scheduler.py` | 补全英美欧半日市/欧股休市、动态收盘与 `add_trading_days` 系列函数 | `25 tests OK`
- **`[2026-11-14] IMP-03 (SQLite 异步数据仓库)`**：修改 `src/storage/db.py`, `tests/test_db_storage.py` | 实现进程内写锁与 WAL、ContextPack/T+5 结算接口、跨周期聊天及 90 天清理 | `30 tests OK`
- **`[2026-11-14] IMP-04 (T212 双账户只读底座)`**：修改 `src/data_providers/t212_client.py`, `tests/test_t212_client.py` | 实现连接池复用、限流头避让、`created_at`/`isin`、双级 5xx 回退、部分成交与现金流水 | `35 tests OK`
- **`[2026-11-14] IMP-05 (多源行情与身份/日历聚合)`**：修改 `src/data_providers/market_data.py`, `tests/test_market_data.py` | 实现 0-info `get_histories`、Singleflight/LRU 防毒、股息率校准、身份/除息日与深度情报 | `43 tests OK`
- **`[2026-11-14] IMP-06 (本地零 Token 量化与筹码引擎)`**：修改 `src/data_providers/analytics.py`, `tests/test_analytics_and_rules.py` | 完成 `CYQMetrics`、拆股/坏点自愈、多周期指标、开仓后回撤、分市场去 Beta 与除息守卫 | `47 tests OK`
- **`[2026-11-14] IMP-07 (持仓规则、多空冲突表与风控护栏)`**：修改 `src/engine/portfolio_rules.py`, `tests/test_analytics_and_rules.py`（下游加 `# TODO` 标注 `schemas.py`/`report_agent.py`/`pick_agent.py`） | 接入开仓后回撤、ATR 止损、除息豁免、挂单/汇率/现金门槛与 `build_disagreement_matrix`/`apply_risk_override_guardrail` | `50 tests OK`
- *（当前进度：`7 / 19` 已完成，下一步待从 `IMP-08` 开始执行）*

---

## 六、历史修改日志（已冷热分离）

完整版本变更记录（`v5.0.1` ~ `v5.0.5`）及已完成 `IMP-xx` 模块的详细技术实现流水，统一归档于 [`CHANGELOG.md`](./CHANGELOG.md)，新窗口默认无需通读历史归档。
