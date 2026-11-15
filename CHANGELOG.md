# Trading 212 × Gemini AI Advisor — 历史变更与模块执行归档日志 (CHANGELOG)

> 说明：本文件归档历史版本修改日志（`v5.0.1` ~ `v5.0.5`）及已完成 `IMP-xx` 模块的完整实现细节，以保持主开发手册 [`DEVELOPER_GUIDE.md`](./DEVELOPER_GUIDE.md) 精简紧凑、节省 AI 助手上下文 Token。

---

## 一、`IMP-01 ~ IMP-07` 详细完成归档

- **`[2026-11-14] IMP-01 (配置加载与纯 Token/Search 用量底座)`**：修改文件 `config.yaml`, `src/config.py`, `src/agents/pick_agent.py`, `src/agents/llm_client.py`, `src/storage/db.py`, `src/bot/handlers.py`, `tests/test_llm_pricing_and_usage.py`, `tests/test_db_and_end_to_end.py`, `README.md`, `DEVELOPER_GUIDE.md`（删除 `src/agents/pricing.py`）。在 `ModelsSection` 新增 `chat_deep_model/thinking_level`、`google_search_free_quota_monthly/warn_pct`，在 `StrategySection` 新增 `us_anomaly_move_pct`、`us_anomaly_vix_change_pct`、`settlement_holding_days` 及边界校验；新增标的大写去重清洗与 `AppConfig.reload_pools_if_changed()`（`mtime_ns` 热重载+异常保护，并接入 `pick_agent.py`）；移除 `src/agents/pricing.py`，将 `normalize_thinking_level` 迁入 `llm_client.py`，清理 USD 程序计价输出，新增 4 组单测；全套 24 个单测通过。
- **`[2026-11-14] IMP-02 (英美欧交易日历引擎)`**：修改文件 `src/engine/calendar_utils.py`, `tests/test_calendar_and_scheduler.py`, `DEVELOPER_GUIDE.md`。新增 `BERLIN_TZ`、`EU_HOLIDAYS_EXTRA`、`normalize_market` 与 `infer_market_from_symbol`；补全美股（感恩节次日、周一至周四的 `12-24` 与 `07-03` 均按 `13:00 ET`）、英股 LSE（`12-24`/`12-31` 按 `12:30 UK`）及 Euronext（`12-24`/`12-31` 按 `14:05 CET`）半日市判定，并在 `UsMarketSchedule` 动态调整 `close_time_uk`/`minutes_to_close`（使 `scheduler.py` 黑色星期五 `weekly_radar` 自动提前于 `18:30 UK` 触发）；新增 Xetra/Euronext 欧股休市日识别（含 `05-01` 劳动节及 Xetra `12-24`/`12-31` 全休差异）；实现 `is_trading_day`、`add_trading_days`、`next_trading_day`、`get_market_close_utc`、`has_market_closed_on_date` 及 `get_market_calendar_alerts`；全套 25 个单测通过。
- **`[2026-11-14] IMP-03 (SQLite 异步数据仓库)`**：修改文件 `src/storage/db.py`, `tests/test_db_storage.py`, `DEVELOPER_GUIDE.md`。新增 `_connect()`、`PRAGMA busy_timeout=5000`/`synchronous=NORMAL`/`temp_store=MEMORY`、`self._write_lock` 进程内写锁、批量 `executescript` 建表与 `_ensure_columns()` 幂等迁移、`executemany` 批量插入及预留 `close()`；新增索引；幂等扩展 `morning_reports.context_pack_json` 与 `save_morning_report`/`get_context_pack`/`get_latest_snapshot`；幂等扩展 `pick_recommendations` 6 个结算/反思字段（默认 `horizon_days=5`），新增 `get_pending_settlements`、`settle_recommendation`（`PENDING` 幂等保护）、`get_symbol_lessons(limit=3)`、`get_recent_lessons`、`get_recommendation_stats`；修复同毫秒排序与 `get_chat_history` 切片，新增 `get_previous_cycle_chat_history`、`get_unnotified_ops_events`、`mark_ops_events_notified`、`this_month` 用量统计及保留最新快照兜底的 `prune_old_records(retention_days=90)`；全套 30 个单测通过。
- **`[2026-11-14] IMP-04 (T212 双账户只读底座)`**：修改文件 `src/data_providers/t212_client.py`, `tests/test_t212_client.py`, `DEVELOPER_GUIDE.md`。在 `T212Client` 新增 `_get_client()` 懒加载实例级 `httpx.AsyncClient` 连接池复用、`close()` 优雅释放及 `_portfolio_lock` 并发防击穿；对照 `T212_API_manual.json` 在 `PositionItem` 新增 `created_at` 与 `isin`；在 `_request_with_retry` 按 `(account, path)` 桶解析 `x-ratelimit-remaining`/`x-ratelimit-reset`（兼容秒/毫秒时间戳，429 精准退避及 `remaining==0` 翻页主动避让），增加 `408`/`5xx` 短退避重试；引入 `snapshot_loader` 注入与瞬时故障内存/DB 双级回退（非瞬时 `401/403` 不回退，保留原快照时间戳，保护仅 `summary` 5xx 现金归零）；补全 `PendingOrderItem` 部分成交剩余数量及 `pending_orders_error`；接入近 30 天 `/equity/history/transactions` 与 `/equity/history/dividends`（`recent_transactions`/`recent_dividends`）并在 `build_trade_context()` 输出 `cash_flow_summary_text`；全套 35 个单测通过。
- **`[2026-11-14] IMP-05 (多源行情与身份/日历聚合)`**：修改文件 `src/data_providers/market_data.py`, `tests/test_market_data.py`, `DEVELOPER_GUIDE.md`。新增 `get_histories(symbols, period="1y")` 批量接口（`yf.download` + 单标的兜底 + 0 次 `.info` 调用）、Singleflight 并发合并与 `_fetch_yf_sync` 开关透传及已有 `1y` K 线复用；通过 `trailingAnnualDividendYield`/`dividendRate` 交叉校验修复低股息率放大 100 倍 Bug；扩充 `TickerMarketBundle`（`long_name`/`exchange`/`ex_dividend_date`/`days_to_ex_dividend`/`isin`/`identity_summary` 及质量字段 `beta`/`profit_margin_pct`/`revenue_growth_pct`/`roe_pct`/`debt_to_equity`/`analyst_count`/`recommendation_key`），`get_bundles` 新增 `include_info` 与 `isin_map`；实施 `(symbol, period)` 隔离 LRU 容量治理、防空值投毒与 Stale-while-error 过期回退、`YFRateLimitError` 重建 `Ticker` 重试；FRED 增补 `DGS2`/`T10Y2Y`/`T10YIE`/`ECBDFR`，修复 Finnhub 并发 `last_error` 互盖并新增 `get_quote`/`get_profile_by_isin` 及 6h 缓存 `get_us_deep_intel` 与 `await close()`；全套 43 个单测通过。
- **`[2026-11-14] IMP-06 (本地零 Token 量化与筹码引擎)`**：修改文件 `src/data_providers/analytics.py`, `tests/test_analytics_and_rules.py`, `DEVELOPER_GUIDE.md`。新增 `_normalize_adjusted_ohlcv`（`Adj Close` 同比例缩放 OHLC + 单日 `GBp/GBP` 100 倍坏点剔除 + 未复权拆股断层回溯对齐）；移植并向量化 `CYQMetrics`/`compute_cyq_metrics`（相对成交量衰减 + 显式 `turnover` 兼容 + 自适应格栅 + 停牌过滤）；在 `TechnicalSnapshot` 扩展 `atr14`/`atr_pct`、`volume_ratio_20d`、`weekly_ma10`/`weekly_ma30`/`weekly_bullish`/`weekly_rsi14`、`bb_width_pct`/`bb_squeeze`、`adx14`、`cyq_*` 及除息窗口标记 `days_to_ex_dividend`/`is_ex_div_window`/`is_ex_div_drop_guard`；新增 `compute_drawdown_since_date(hist, created_at, max_lookback_days=60)`、`resolve_market_benchmark` + `ffill(limit=1)` 错位假期对齐 + 2 日滚动相关性补偿（美股/跨境 ETF 锚定 `SPY`、英股 `ISF.L`、欧股 `EXS1.DE`/`VERG.L`）、`infer_etf_index_family` 克隆 ETF 底层暴露归并及紧凑 `format_technical_line`；全套 47 个单测通过。
- **`[2026-11-14] IMP-07 (持仓规则、多空冲突表与风控护栏)`**：修改文件 `src/engine/portfolio_rules.py`, `tests/test_analytics_and_rules.py`, `DEVELOPER_GUIDE.md`（在 `src/agents/schemas.py`, `src/agents/report_agent.py`, `src/agents/pick_agent.py` 增补下游 `# TODO` 接线标注）。在 `evaluate_holding` / `evaluate_portfolio` 支持按单仓 `pos.created_at` 与可选 `hist` 独立截取 `compute_drawdown_since_date` 修复建仓前高点移动止盈误触；接入 `ATR(14)` 自适应 Invest 止损（`-clamp(2.8 * atr_pct, 8%, 18%)`）及除息窗口（`is_ex_div_drop_guard` / `days_to_ex_dividend`）假破位/临界止损豁免；联动活动 `BUY`/`STOP` 挂单按 `GBP` 估算在途仓位叠加与保护识别；接入 `weekly_bullish`、`adx14`、`volume_ratio_20d`、`cyq_regime`/`cyq_benefit_part_pct`、本币 vs 英镑汇率失真校准与 `4%` 现金+摩擦门槛校验；新增 `DisagreementItem`、`build_disagreement_matrix()`、`format_disagreement_matrix_text()` 及 `RiskOverrideRecord`/`RiskOverrideResult`/`apply_risk_override_guardrail()`（统一拦截/降级 `MorningReportDecision` 与 `PickDecision` 中触发 `VIX>=28`、跨账户 `22%` 集中度、5 日财报、`daily_vetoes` 或高相关重叠的激进建议并附注 `[🛡️风控护栏修正]`）；新增 3 组覆盖单测，全套 50 个单测通过。

---

## 二、版本修改日志（Changelog `v5.0.1` ~ `v5.0.5`）

### `v5.0.5`（2026-11-14）
Telegram 菜单交互防误触精简、晨报提前至 `07:30 UK`、定时去重持久化、英国 Bank Holiday 标注与 **Token 成本瘦身路由优化**：

1. **移除底部快捷按钮，精简左侧 `/` 菜单（`src/bot/handlers.py`, `src/main.py`, `src/ops_notifier.py`）**
   - 删除容易误触且与左侧栏重复的 `CORE_QUICK_KEYBOARD` 与 `/menu`，在启动通知和 `/help`/`/start` 中下发 `{"remove_keyboard": True}` 自动收起旧底部键盘。
   - 左侧 Telegram `/` 菜单（`BOT_COMMANDS`）聚焦为 6 个核心指令（`/pos`、`/report`、`/pick`、`/us`、`/stats`、`/help`），`/health`/`/usage`/`/clear`/`/cancel` 运维指令保留在 `/help` 中分层说明。
2. **晨报调至 `07:30 UK` 并同步交易周期切换点（`config.yaml`, `src/config.py`, `src/storage/db.py`, `src/main.py`）**
   - 将 `morning_report_hour/minute` 调为 `07:30 UK`（美股盘前保持开盘前 15 分钟、周五雷达保持美股收盘后 30 分钟）。
   - 将数据库交易周期函数升级为 `get_trading_cycle_id(now_utc, reset_hour, reset_minute)` 并由 `main.py` 按 `07:30 UK` 注入 `AdvisorDatabase`，保证早报红线与全天 `/us`/`/pick`/单聊严格处于同一周期。
3. **定时任务触发持久化去重（`src/storage/db.py`, `src/bot/scheduler.py`）**
   - 幂等新增 `scheduler_runs(job_name, uk_date, fired_at_uk)` 表，重启或发版刚好落在 5 分钟触发窗口内也不会重复发送消息。
4. **英国 Bank Holiday 标注而不改时间（`src/engine/calendar_utils.py`, `src/agents/report_agent.py`, `src/agents/pick_agent.py`, `src/agents/chat_agent.py`）**
   - 新增英格兰及威尔士（LSE）法定假日识别 `is_uk_bank_holiday`（含复活节、五月/八月银行假日及圣诞/节礼日顺延）；定时推送时间保持不变，并在 `/report` 顶部（仍限 `<= 1500` 字）及 `/pick`/`chat` 上下文中标注 `LSE 休市`。
5. **全链路成本与模型路由优化（`config.yaml`, `src/config.py`, `src/agents/chat_agent.py`, `src/agents/report_agent.py`, `src/agents/pick_agent.py`, `src/agents/tools.py`）**
   - **分层搜索与 `/report` 质量强化**：新增 `report_search_model` 专属槽位使 `/report` 强制全市场搜索及深挖保持 `gemini-4-hydrogen`（`LOW`），而单聊 / `/pick` / `/us` 联网搜索使用 `search_model: gemini-3.5-flash-lite`（`LOW`）；搜索分批粒度从每批 `5` 只收紧至 `3` 只减少漏检；`/report` Bull/Bear 内部草稿放宽至 `1200 字内`、`report_max_tool_rounds=6`，最终 Telegram 呈现仍严格 `<= 1500 字`。
   - **单聊上下文瘦身与路由调优**：日常单聊改为 `gemini-4-hydrogen` (`LOW`)、`/us` 改为 `gemini-4-hydrogen` (`MEDIUM`)；`answer_user_chat` 移除全量 `build_portfolio_fact_sheet` 注入，改为注入轻量双账户持仓摘要 + 挂单 + 今日早报/哨兵/红线，复杂新标的技术面/新闻按需调用工具查询。
   - **防无意义重复调工具**：引导 `/report`、`/pick` 与 `chat` 避免对已有完整数据的常规持仓无故重复调用工具，同时保留 `/report` 对疑点标的与活动挂单充分深挖的能力。
6. **确定性风控规则与去 Beta 相关性升级（`config.yaml`, `src/config.py`, `src/engine/portfolio_rules.py`, `src/data_providers/analytics.py`, `src/engine/screener.py`, `src/agents/tools.py`）**
   - **跨账户合并个股集中度（ETF 豁免）**：新增 `max_total_portfolio_weight_pct: 22.0` 按发行人跨 ISA+Invest 合并口径限制个股总仓位；宽基/行业 ETF 完全豁免集中度卡控。
   - **利润保护与深套防死扛**：新增 `trailing_profit_activation_pct: 20.0` + `trailing_drawdown_from_peak_pct: -12.0`（基于近 60 日波段高点 `drawdown_60d_pct` 识别移动止盈回撤，避免 52 周高点误报）及 `isa_deep_drawdown_review_pct: -25.0`（ISA 深套基本面复核警示）。
   - **趋势过滤与规则互斥**：新增 `require_uptrend_for_add: true`（严格要求 `Price > MA200` 才可触发 `ADD_CANDIDATE`），同时在触发止损/超重/深套时强制清零加仓分，彻底消除自相矛盾提示。
   - **去市场 Beta 残差相关性**：`compute_correlation_matrix(..., benchmark_symbol="SPY")` 对个股采用去大盘 Beta 的超额收益相关性，避免系统性普跌误报重叠；纯指数克隆 ETF（如 `VUAG.L` 与 `CSP1.L`）仍保留高相关重叠识别。

### `v5.0.4`（2026-11-14）
全链路 Telegram 运维告警机制、Gemini 容错降级、JSON 自修复、英欧美/ETF 强制联网搜索与交互升级：

1. **统一 Telegram 运维告警机制与启动/恢复感知（`src/ops_notifier.py`, `src/storage/db.py`, `src/main.py`, `src/bot/scheduler.py`, `src/data_providers/`）**
   - 区分 `WARN`（30 分钟冷却去重合并）与 `CRITICAL`（防死循环短防抖，必达告警）；在 Telegram 断连时入库缓存离线事件，恢复后自动推送 `[系统恢复]` 汇总。
   - 进程启动自动发送带版本号的 `[系统上线]` 通知；全链路异常主动上报 `OpsNotifier` 并落库 `ops_events`。
2. **Gemini 瞬时重试、备援模型自动降级与全挂阻断告警（`src/agents/llm_client.py`, `config.yaml`, `src/config.py`）**
   - 对 `429`/`503`/超时执行指数退避重试；主模型异常时自动降级至 `fallback_model`（`gemini-4-hydrogen HIGH`）并发 `WARN` 通知；全挂时推送 `CRITICAL` 告警。
3. **结构化 JSON 自动修复重试与风控红线保护（`src/agents/schemas.py`, `src/agents/report_agent.py`, `src/agents/pick_agent.py`）**
   - 初次 JSON 解析失败时通过 `parse_or_repair_structured_json` 发起 1 次快速结构化修复，防止 `daily_vetoes` 静默丢失。
4. **全市场（英/欧/美股 + LSE ETF）强制 Google Search Grounding（`src/agents/tools.py`, `src/agents/report_agent.py`, `src/agents/pick_agent.py`）**
   - 新增 `run_forced_market_grounding` 分市场并发联网检索（带 TTL 缓存），并在 `llm_usage_logs` 中统计 `web_search_queries` 用量。
5. **`/health` 运维体检与 `/help` 分层（`src/bot/handlers.py`, `src/bot/telegram_client.py`）**
   - 新增 `/health` 实时自检指令：汇报版本、运行时长、各外部 API 连通性、最近定时任务结果及最近 5 条运维事件。

### `v5.0.3`（2026-11-14）
Gemini 模型路由优化、Prompt 结构化瘦身与全链路 Token 可观测性升级：

1. **分角色模型路由重配与 Thinking 边界校验（`config.yaml`, `src/config.py`, `src/agents/pricing.py`, `src/agents/llm_client.py`）**
   - 拆分为职责对应的模型槽位，并在 `pricing.normalize_thinking_level` 中拦截无效档位（如 Gemini 4 传入 `MINIMAL` 时钳位为 `LOW`）。
2. **全链路 Token 埋点、成本估算与 `/usage` 看板（`src/agents/llm_client.py`, `src/agents/pricing.py`, `src/storage/db.py`, `src/bot/handlers.py`）**
   - 累加每次调用的输入、缓存、输出与思考 Token，按官方计价估算费用并写入 `llm_usage_logs`，新增 `/usage` 指令。
3. **Prompt 去噪与 `system_instruction` 分离（`src/agents/prompts.py`, `src/engine/portfolio_rules.py`, `src/agents/chat_agent.py`）**
   - 稳定角色设定抽离至 `system_instruction`；常规无异常 `HOLD` 仓位合并为单行摘要；单聊历史早期轮次压缩为单行摘要。
4. **一键升级脚本（`deploy/upgrade.sh`）**
   - 支持优雅 `SIGTERM` 停服 -> 备份最近 5 份 `advisor.db` 与 `config.yaml` -> 解压 -> 编译及单测验收 -> 重启服务。

### `v5.0.2`（2026-11-14）
全面代码审计与 `/us` 美股「增量哨兵（Delta Sentinel）」升级：

1. **`/us` 升级为增量复核哨兵（`src/agents/report_agent.py`, `src/storage/db.py`）**
   - 读取完整 `/report` 原文与 `daily_guardrails`，过滤仅美股持仓与挂单，要求二态输出 `✅ 维持早报计划` 或 `🚨 盘前增量修正`。
2. **打通自由聊天与 `/us` 上下文（`src/agents/chat_agent.py`）**
   - 自由聊天同时读取今日 `/report` 正文 + 最新 `/us` 补充 + `/pick` 结果 + 今日对话记录。
3. **边界 Bug 修复（`src/data_providers/t212_client.py`, `src/agents/schemas.py`）**
   - 修复历史成交 `/equity/history/orders` 中账户 `GBP` 覆盖美股 `USD` 的问题；增强 `render_morning_report` 符号去重解析。

### `v5.0.1`（2026-11-14）
实盘跑通后的日志体检与首轮修复：

1. **修复 T212 活动挂单触发价币种误判 Bug（`src/data_providers/t212_client.py`）**：优先采用 `instrument.currency` 与后缀规则，避免美股止损单被错标为 `GBP`。
2. **消除 `/report` 三重重复并防止尾部财报预警被截断（`src/agents/schemas.py`, `src/agents/report_agent.py`）**：严格隔离 `top_actions`、`pending_order_audits` 与 `flagged_positions`，并将财报预警与操作复盘置前。
3. **优化 `/pick` 精确现金计算与休市感知（`src/agents/pick_agent.py`）**：由 Python 注入精确现金占比与开休市状态。
4. **`/us` 休市聚焦美股并硬控 `<= 450` 字（`src/agents/report_agent.py`）**。
5. **Telegram 长轮询 `httpx.AsyncClient` 连接复用与网络抖动日志降噪（`src/bot/telegram_client.py`, `src/main.py`）**。
