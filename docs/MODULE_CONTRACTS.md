# Trading 212 × Gemini AI Advisor — 模块接口契约与参考映射（冷参考）

> **冷文档说明**：本文件收纳全量模块签名字典与 `_ref/` 算法源映射，供跨模块重构或深度排查时按需查阅；日常开发热启动入口见 [`AGENTS.md`](../AGENTS.md) 与 [`DEVELOPER_GUIDE.md`](../DEVELOPER_GUIDE.md)。

### 3.3 核心模块接口字典（精确签名与字段契约速查，随 `IMP` 滚动更新）

> 💡 下游窗口直接按本字典接线，**不必通读千行上游源码**；后续新增公共函数/字段后，须在对应小节同步补齐 1~2 行契约。

- **`src/config.py`（`IMP-01` & N95 迁移前置 & 混合双厂商路由）**：
  - `EnvSettings` 新增：`ops_notify_url: str = ""`（`OPS_NOTIFY_URL`）、`ops_notify_token: str = ""`（`OPS_NOTIFY_TOKEN`）、`deepseek_api_key: str = ""`（`DEEPSEEK_API_KEY`）、`deepseek_base_url: str = "https://api.deepseek.com"`（`DEEPSEEK_BASE_URL`）
  - `ModelsSection` 新增：`us_anomaly_model` (`gemini-4-hydrogen`), `us_anomaly_thinking_level` (`HIGH`), `report_priority_search_model` (`gemini-4-hydrogen`), `report_priority_search_thinking_level` (`LOW`), `gemini_fallback_model` (`gemini-4-hydrogen`), `gemini_fallback_thinking_level` (`MEDIUM`), `deepseek_fallback_model` (`deepseek-flash`), `deepseek_fallback_thinking_level` (`HIGH`), `chat_deep_model`, `chat_deep_thinking_level`, `google_search_free_quota_monthly` (`5000`), `google_search_warn_pct` (`85.0`)
  - `StrategySection` 新增：`us_anomaly_move_pct` (`1.8`), `us_anomaly_vix_change_pct` (`4.0`), `settlement_holding_days` (`5`)；`UniverseSection` 新增可选 `etf_symbols: list[str] = []`
  - `AppConfig.reload_pools_if_changed() -> bool`（按 `mtime_ns` 原地热重载 `watchlist` 与 `universe`，YAML 损坏时保留旧配置）
- **`src/ops_notifier.py` & `src/healthcheck.py`（N95 迁移前置合规 & `IMP-17`）**：
  - `OpsNotifier(..., ops_notify_url="", ops_notify_token="", service_name="trading-python", transport=None)`；`ops_notify_enabled: bool`；`await ops.close()`
  - `await ops.notify(..., title="", business_blocking=False, bot_message="") -> bool`（`ops_notify_url` 为空时降级走 `TelegramClient`；已配置时技术类仅 POST `ops-notify`）
  - `await ops.notify_business_critical(*, source, alert_key, message, title="", bot_message="") -> bool`（强业务阻塞双发：`ops-notify CRITICAL` + 交易 Bot 极简提示）
  - `await ops.flush_pending() -> bool`（先刷内存缓冲，再调用 `db.get_unnotified_ops_events(hours=24)` 过滤 `suppressed_count>0` 冷却记录与刚刷新的内存同件后按 `ops_notify_url` 双态分流分片补推，成功后再调 `db.mark_ops_events_notified()`）
  - `send_ops_notify_sync(url, ..., service="trading-python", severity="CRITICAL", alert_key="process_fatal_crash", title=..., message, cooldown_sec=0, token="", timeout_s=5.0) -> bool`（`main()` 致命崩溃兜底上报）
  - `src/healthcheck.py`：`write_heartbeat(path=DEFAULT_HEARTBEAT_PATH, now_ts=None) -> Path`；`check_health(heartbeat_path=DEFAULT_HEARTBEAT_PATH, db_path=DEFAULT_DB_PATH, *, max_stale_seconds=180.0, now_ts=None) -> tuple[bool, str]`（默认 `/app/data/advisor.db`，未设 `ADVISOR_DB_PATH` 时自动回退 `data/advisor.db`；只读 SQLite URI `mode=ro` 校验）
- **`src/engine/calendar_utils.py`（`IMP-02`）**：
  - `infer_market_from_symbol(symbol) -> str`（返回 `US | UK | XETRA | EURONEXT`）
  - `is_trading_day(day, market="US") -> bool` | `next_trading_day(start_date, market="US", inclusive=False) -> date`
  - `add_trading_days(start_date, n_days, market="US") -> date`（跳过休市日精确推算 `T+N`）
  - `get_market_close_utc(day, market="US") -> datetime | None` | `has_market_closed_on_date(target_date, market="US", now_utc=None) -> bool`
  - `is_us_early_close / is_uk_early_close / is_eu_early_close / is_uk_bank_holiday / is_eu_market_holiday` | `get_market_calendar_alerts(now_utc=None) -> list[str]`
  - `UsMarketSchedule` 新增字段：`is_us_early_close: bool`, `close_time_et: str`（已动态调整 `close_time_uk` 与 `minutes_to_close`）
- **`src/storage/db.py`（`IMP-03`、`IMP-20` & `IMP-22`）**：
  - 周期/快照：`await db.get_latest_snapshot() -> PortfolioData | None`；`await db.save_morning_report(..., context_pack: dict | str | None = None)`；`await db.get_context_pack(cycle_date=None, *, allow_fallback=True) -> dict | None`；`await db.close()`
  - 结算/反思/体检：`await db.get_pending_settlements(limit=50, *, only_with_entry_price=False)`；`await db.settle_recommendation(rec_id, *, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, outcome_status="SETTLED", reflection_lesson="", mfe_pct=None, mae_pct=None) -> bool`（仅更新 `PENDING`）；`await db.get_settled_recommendations_for_followup(limit=50)`；`await db.update_recommendation_followup(rec_id, *, t10_return_pct=None, t10_alpha_pct=None, t20_return_pct=None, t20_alpha_pct=None, mfe_pct=None, mae_pct=None) -> bool`；`await db.save_quant_audit(audit_date, summary_text, payload=None)` & `await db.get_latest_quant_audit() -> dict | None`；`await db.get_symbol_lessons(symbol, limit=3)`；`await db.get_recent_lessons(limit=5)`；`await db.get_recommendation_stats() -> dict`（含 `avg_t10_alpha_pct`, `avg_t20_alpha_pct`, `avg_mfe_pct`, `avg_mae_pct`）
  - 对话/告警/用量/清理：`await db.get_previous_cycle_chat_history(cycle_date=None, limit=2)`；`await db.get_unnotified_ops_events(*, hours=24, severities=("CRITICAL", "WARN"), limit=20)`；`await db.mark_ops_events_notified(event_ids)`；`await db.get_usage_summary(days=30, now_utc=None)`（含 `"today"`, `"last_7d"`, `"last_nd"`, `"this_month"`, `"by_label_7d"`）；`await db.prune_old_records(retention_days=90, *, now_utc=None) -> dict[str, int]`（清理过期快照/日志/告警及 `ohlcv_cache`，永久保留 T212 元数据与分红归档表）
  - 行情缓存 & T212 归档（`IMP-20`）：`await db.get_cached_ohlcv(symbols, period="2y") -> dict[str, dict]`；`await db.save_cached_ohlcv(entries) -> int`；`await db.upsert_t212_instruments(items) -> int`；`await db.get_t212_instrument_map() -> dict[str, dict]`；`await db.upsert_dividends(items) -> int`；`await db.get_cumulative_dividends_by_symbol() -> dict[tuple[str, str], float]`
- **`src/data_providers/t212_client.py`（`IMP-04`、`IMP-20` & `IMP-21`）**：
  - `T212Client(..., snapshot_loader: Callable[[], Awaitable[PortfolioData | None]] | None = None, db=None)`；`set_snapshot_loader(...)`；`set_db(db)`；`await t212.fetch_instruments_metadata() -> list[dict]`（`/equity/metadata/instruments`，`1 req/50s` 限流桶）；`await t212.close()`
  - `PositionItem` 新增：`created_at: str | None = None`, `isin: str | None = None`, `cumulative_dividends_gbp: float = 0.0`, `total_return_with_div_pct: float | None = None`（`format_positions_text` 展示含息总回报）；`PendingOrderItem` 新增：`filled_quantity: float = 0.0`, `status: str = ""`
  - `AccountSnapshot` 新增：`recent_transactions: list[HistoricalTransactionItem]`, `recent_dividends: list[HistoricalDividendItem]`, `pending_orders_error`, `is_fallback`, `fallback_source`, `fallback_reason`, `snapshot_fetched_at_uk`
  - `PortfolioData` 新增：`is_fallback: bool`, `fallback_source: str`, `fallback_note: str`
  - `build_trade_context(..., tech_map=None, histories=None)` 返回字典键：`pending_orders_text`（传入 `tech_map`/`histories` 时实算币种对齐偏离 `%`、ATR 倍数、20 日经验触价率及筹码/噪音标签）, `recent_trades_text`, `trade_summary_30d_text`, `cash_flow_summary_text`, `traded_tickers`
- **`src/data_providers/market_data.py`（`IMP-05`、`IMP-20` & `IMP-21`）**：
  - `MarketDataService(..., db=None)` & `set_db(db)`；`normalize_history_index(index) -> pd.DatetimeIndex | None`（剥离时区并归整至当地交易日历零点；由 `_sanitize_history_df` 与 `screener._extract_ohlcv_series` 共用）
  - `await market_data.get_histories(symbols: list[str], period="2y") -> dict[str, pd.DataFrame]`（兼容 `"1y"`；0 次 `.info`，含 Singleflight、LRU 与 SQLite `ohlcv_cache` `1mo` 增量拼接 / `>0.5%` 拆股全量重拉）
  - `await market_data.get_bundles(symbols, include_news=True, include_earnings=True, include_info=True, isin_map: dict[str, str] | None = None) -> tuple[dict[str, pd.DataFrame], dict[str, TickerMarketBundle]]`（缺失 `sector`/`short_name` 且有 `isin` 时自动调 `finnhub.get_profile_by_isin` 补全）
  - `await market_data.get_us_realtime_quote(symbol)`；`await market_data.get_us_prepost_quotes(symbols) -> dict[str, dict[str, Any]]`（`5d/15m prepost=True` 美股盘前/盘后价与跳空计算）；`await market_data.get_us_deep_intel(symbol) -> dict` & `await market_data.get_us_deep_intel_batch(symbols, max_concurrency=4) -> dict[str, dict[str, Any]]`（6h 缓存 + `Semaphore` 并发控制）；`FredClient._fetch_series_stats(client, series_id, limit=25)`（返回最新值及 5d/20d 变动，新增 `BAMLH0A0HYM2`）；`await market_data.close()`
  - `TickerMarketBundle` 新增：`long_name`, `exchange`, `isin`, `identity_summary` (`build_identity_anchor(isin_override=None)`), `beta`, `profit_margin_pct`, `revenue_growth_pct`, `roe_pct`, `debt_to_equity`, `analyst_count`, `recommendation_key`, `eps_estimate`, `ex_dividend_date`, `days_to_ex_dividend`；`MacroSnapshot` 新增：`us_10y_change_20d_bps`, `us_2y_yield`, `yield_spread_10y_2y`, `breakeven_10y_pct`, `hy_oas_spread_pct`, `hy_oas_change_20d_bps`, `ecb_deposit_rate`, `is_stale: bool`
- **`src/data_providers/analytics.py`（`IMP-06` & `IMP-21`）**：
  - `compute_cyq_metrics(hist, cyq_days=210, factor=150) -> CYQMetrics | None`（字段：`benefit_part_pct`, `avg_cost`, `concentration_90_pct`, `cost_90_low`, `cost_90_high`, `concentration_70_pct`, `regime`）
  - `compute_drawdown_since_date(hist, created_at=None, max_lookback_days=60) -> float | None`
  - `compute_technical_snapshot(symbol, hist, position_created_at=None, bundle=None) -> TechnicalSnapshot | None`
  - `TechnicalSnapshot` 新增：`atr14`, `atr_pct`, `volume_ratio_20d`, `weekly_ma10`, `weekly_ma30`, `weekly_bullish`, `weekly_rsi14`, `bb_width_pct`, `bb_squeeze`, `adx14`, `cyq_benefit_part_pct`, `cyq_avg_cost`, `cyq_concentration_90_pct`, `cyq_cost_90_low`, `cyq_cost_90_high`, `cyq_regime`, `days_to_ex_dividend`, `is_ex_div_window` (`-3..3`), `is_ex_div_drop_guard` (`-3..0`)
  - `resolve_market_benchmark(symbol, default_benchmark="SPY", uk_benchmark="ISF.L", eu_benchmark="EXS1.DE") -> list[str]`
  - `compute_correlation_matrix(histories, symbols=None, lookback_days=60, benchmark_symbol="SPY", uk_benchmark_symbol="ISF.L", eu_benchmark_symbol="EXS1.DE") -> pd.DataFrame`
  - `compute_portfolio_cvar_and_risk_attribution(portfolio, histories, lookback_days=120) -> dict[str, Any]`（PSD 收缩协方差算 `portfolio_vol_pct` 与成分风险贡献前 3 波动源 + 95% 单日历史 `cvar_95_pct`/`cvar_95_gbp`）
  - `infer_etf_index_family(symbol, bundle=None, name_hint="") -> str | None` | `compute_exposure_summary(portfolio, bundle_map, histories=None) -> str`
- **`src/engine/portfolio_rules.py`（`IMP-07`）**：
  - `evaluate_holding(..., hist=None, pending_orders=None, account_total_value=None, total_portfolio_value=None) -> HoldingEvaluation`（新增 `native_return_pct`, `effective_stop_pct`, `pending_buy_weight_pct`, `has_pending_stop`, `ex_div_guarded`）；`evaluate_portfolio(..., histories=None)`；`format_evaluations_text(..., compact_hold=True)` 合并无高优风险的常规 `HOLD`
  - `build_disagreement_matrix(data, tech_map, bundle_map, strategy, evals=None) -> list[DisagreementItem]` | `format_disagreement_matrix_text(items) -> str`（单行折叠同质 `SUPPORT_VS_WEAK_TREND`）
  - `apply_risk_override_guardrail(decision, *, portfolio=None, macro=None, strategy=None, bundle_map=None, evals=None, high_correlation_symbols=None, daily_vetoes=None, vix_danger_threshold=28.0) -> RiskOverrideResult`（含 `overrides: list[RiskOverrideRecord]`, `notes: list[str]`）
- **`src/engine/screener.py`（`IMP-08`、`IMP-21` & `IMP-22`）**：
  - `ScreenerCandidate` 字段：`is_recent_traded: bool = False`, `rs_20d_pct: float | None = None`, `marginal_vol_delta_pct: float | None = None`, `regime_tag: str = ""`, `dimension_scores: dict[str, float]`, `dimension_pct: dict[str, float]`, `pros: list[str]`, `cons: list[str]`, `pattern_signals: list[str]`
  - 形态/动量/风控函数：`detect_breakthrough_platform(hist, window=60) -> bool` | `detect_low_atr_contraction(hist, tech, min_bars=120) -> bool` | `detect_volume_ma_breakout(hist, tech) -> bool` | `detect_minervini_stage2(hist, tech, rs_60d_pct=None) -> dict` | `detect_vcp_contraction(hist, tech=None, window=120) -> bool` | `compute_relative_strength(hist, benchmark_hist) -> dict` | `compute_vpvr_profile(hist, window=60, bins=24) -> dict | None` | `compute_market_breadth(stage1_tech, universe_syms, histories=None, benchmark_symbols=...) -> dict` | `compute_portfolio_risk_metrics(histories, portfolio, candidates, *, lookback_days=60, assumed_new_weight=0.05) -> dict`（3 日跨时区对数收益相关矩阵 + 单日波动率还原 PSD 收缩协方差）| `run_offline_universe_and_threshold_audit(histories, portfolio, universe, strategy) -> dict[str, Any]`（0-Token 120d 滚动重叠、>=60d Phase 4 僵尸股与突破样本回测体检）
  - `score_candidate(..., deep_intel=None) -> ScreenerCandidate`（双轨非对称权重：`ISA 25/15/35/25` vs `Invest 35/35/15/15` + ETF 60d VPVR + Stage2 豁免 + 美股深度情报 `value_quality` 加减分）；`await run_quant_screener(..., existing_histories=None, top_n_per_track=8)`（Stage 2 美股批量接入 `get_us_deep_intel_batch`）
- **`src/agents/llm_client.py`（`IMP-01` & `IMP-09`）**：
  - `normalize_thinking_level(model: str, thinking_level: str) -> str`（从已移除的 `pricing.py` 迁入；`pricing.py` 已按用户要求删除，仅记录 Token 与 `search_queries`）
  - `GeminiClient(..., deepseek_api_key="", deepseek_base_url="https://api.deepseek.com", gemini_fallback_model="gemini-4-hydrogen", gemini_fallback_thinking_level="MEDIUM", deepseek_fallback_model="deepseek-flash", deepseek_fallback_thinking_level="HIGH", deepseek_round_timeout_seconds=60.0, deepseek_max_retries=1, round_timeout_seconds=120.0, tool_timeout_seconds=60.0, max_concurrent_tools=6, max_tool_result_chars=4000, usage_summary_provider=None, search_free_quota_monthly=5000, search_warn_pct=85.0)`；`await llm.close()`；`set_usage_summary_provider(...)`（支持 Gemini + DeepSeek OpenAI 兼容协议双厂商路由、`reasoning_content` 工具透传、JSON Schema 动态兜底注入、Google Search 禁切 DeepSeek 隔离及 `"this_month"` Search 免费额度 `OpsNotifier` 预警）
- **`src/agents/prompts.py` & `src/agents/schemas.py`（`IMP-10`）**：
  - `build_investor_profile(strategy, *, include_order_audit_rule=True)` & `build_agent_system_instruction(role_instruction, strategy, *, include_order_audit_rule=True)`（`/pick` 传 `False` 隔离存量挂单审计原则）
  - `FlaggedPositionItem(symbol: str, summary: str)`；`MorningReportDecision.flagged_positions: list[FlaggedPositionItem]`（`mode="before"` 校验器兼容旧 `str` / `dict`）
  - `render_morning_report(decision, max_chars=1800, *, risk_override: RiskOverrideResult | None = None) -> str` & `render_pick_report(decision, max_chars=1000, *, risk_override: RiskOverrideResult | None = None) -> str`（统一高亮 `[🛡️风控护栏修正]` 并保留护栏驳回项，`_fit_max_chars` 尾部防孤立 `【...】` 标题行）；`trim_text_at_line_boundary(text, max_chars) -> str`（支持单行句界/分句边界优雅裁剪）
- **`src/agents/tools.py`（`IMP-11` & `IMP-21`）**：
  - `run_forced_market_grounding(..., max_concurrent_searches: int = 3, priority_symbols: list[str] | None = None, priority_search_model: str = "", priority_thinking_level: str = "LOW") -> str`（宏观与 `priority_symbols` 脱离 3 股拼团转 1 对 1 独立深搜并可分层指定 `priority_search_model`，`asyncio.Semaphore` 限流 + `_GROUNDING_CACHE` 过期清理；英文检索 Prompt 对齐 RNS/SEC 一手信源，强制输出紧凑中文要点）
  - `MarketToolkit(market_data, portfolio, *, llm=None, search_model="gemini-4-hydrogen", ops_notifier=None, db=None)`（英文工具声明引导英文 Query；`inspect_ticker` 传入持仓 `isin`/`created_at` 与 `bundle`、按需调 `get_us_deep_intel` 与 `db.get_symbol_lessons`，含 CYQ/周线/除息/身份紧凑裁剪 `<4000` 字符；`check_portfolio_correlation` 改调 `get_histories` 并补拉 `SPY`/`ISF.L`/区域基准）
- **`src/agents/report_agent.py`（`IMP-12` & `IMP-21`）**：
  - `build_analysis_context_pack(..., us_intel_map=None) -> dict` & `build_portfolio_fact_sheet(..., db=None, context_pack_out=None)`（保持 4 元组兼容；接入 `get_us_deep_intel_batch` 与 `compute_exposure_summary(..., histories=histories)`；Pack 含 `pack_version`, `cycle_date`, `calendar_alerts`, `macro`, `exposure_summary`, `cash_flow_summary`, `portfolio_diff`, `disagreements`, `recent_lessons`, `symbols[sym]` -> `identity_summary`/`technical`/`technical_line`/`bundle_summary`/`evaluations`/`disagreements`/`lessons`/`deep_intel`）
  - `generate_morning_report(...)`（自动提取财报 `<=7d`/除息 `-3..3d`/单日涨跌 `>=3%` 持仓为 `priority_symbols` 传入强搜，透传 `tech_map`/`histories` 至 `build_trade_context`，接 `apply_risk_override_guardrail` + `is_fallback` 跳过 `save_snapshot` + `context_pack` 落库）；`generate_us_sentinel(...)`（`PRE` 时段或 Finnhub 涨跌幅为 `0` 时调用 `get_us_prepost_quotes` 补全盘前价 + 本地 Anomaly Gate：平稳 `LOW<=450` 免强搜，异动 `MEDIUM<=600` 强搜）
- **`src/agents/pick_agent.py`（`IMP-13`）**：
  - `generate_pick_radar(...)`（签名不变；两段式加载将 `held_histories` 作为 `existing_histories` 传 `run_quant_screener`，注入 `ContextPack`/`get_recent_lessons`/`get_symbol_lessons`/日历提醒/`MarketToolkit(db=db)`，Post-Screen 自动回填本地 `bundle.current_price` 并经 `apply_risk_override_guardrail` 硬拦截高相关与 `daily_vetoes` 后再以 `settlement_holding_days` 落库）
- **`src/agents/chat_agent.py`（`IMP-14`）**：
  - `ChatObjectMention` & `ChatObjectResolution(decision_kind, target_symbols, denied_symbols)`；`decide_chat_objects(user_query, portfolio, context_pack=None, *, history_turns=None, prev_cycle_turns=None, max_symbols=4) -> ChatObjectResolution`（0-Token 本地代码/名称匹配、否定词排除及跨轮追问沿用）
  - `answer_user_chat(...)`（签名不变；静态早报/红线/`pick[:1050]`/基线/跨周期摘要前置以对齐 Gemini Implicit Cache，动态持仓+`ContextPack` 局部切片+日历提醒+当前问题后置；`/deep` 切 `chat_deep_model`/`chat_deep_thinking_level` 且 `MarketToolkit` 传入 `db=db`）
- **`src/bot/handlers.py` & `src/bot/scheduler.py`（`IMP-15`、`IMP-16` & `IMP-22`）**：
  - `TradingScheduler(..., execution_lock: asyncio.Lock | None = None, morning_retry_cooldown_s: float = 300.0)` 新增 `execution_lock` 与 `current_job_name`；`AdvisorBot(..., execution_lock: asyncio.Lock | None = None)` 优先复用 `scheduler.execution_lock` 对重型 AI 指令互斥防重入
  - `await scheduler.run_prewarm(scope, now_utc=None) -> dict[str, Any]`（工作日早报/美股哨兵/周五 `/pick` 前 10 分钟写入 `scheduler_runs` 锁静默预热持仓+候选池 `2y` 日线、宏观快照，并每日同步 1 次 T212 `instruments` 元数据与分红归档）
  - `await scheduler.run_daily_settlement(now_utc=None) -> dict[str, Any]`（每日 `22:00~23:30 UK` 自动对满 `T+N` 且已收盘的 `pick_recommendations` 算真实收益/Alpha 与初始 MFE/MAE，调用 `Hydrogen LOW` 微反思写回，并 0-Token 追评 `T+10`/`T+20` 延伸 Alpha 与极值 + `prune_old_records(90)`）；`await scheduler.run_weekly_quant_audit(now_utc=None)`（每周六闲时或周中补跑静默落库）；早报 `07:30~09:30 UK` 断网补发与冷却重试（成功才锁 `scheduler_runs`）
  - `/usage` 切 `this_month` 自然月额度与缓存/思考占比；`/stats` 接 `get_recommendation_stats`/`get_recent_lessons`/本地现价与 `GBp/GBP` 100 倍校验、多窗口 Alpha 轨迹（T+5/10/20）、MFE/MAE 极值分布与 `get_latest_quant_audit`；`/health` 报 `is_fallback`/`is_stale`/未送达静默告警与 `ops-notify` 状态
- **`src/bot/telegram_client.py` & `src/main.py`（`IMP-18`、`P0-502`）**：
  - `TelegramClient(..., max_retries=2, base_backoff_s=0.5, max_retry_wait_s=30.0, delivery_max_retries=4)`（`_post` 遇 `429` 按 `parameters.retry_after`/`Retry-After` 等待，遇 `5xx`/网络瞬断指数退避重试，耗时仍耗尽时 `httpx.HTTPError` 照常上抛；`_post(..., max_retries=None)` 可按调用覆盖重试预算，`send_message` 固定使用 `delivery_max_retries`）
  - `await tg.get_updates(offset=None, timeout_s=25) -> list[dict]`：上游 `error_code >= 500` 或缺失 `error_code` 或 `401`/`403` 时抛 `TelegramPollError(httpx.HTTPError)`（携带 `method`/`error_code`/`description`），由 `run_app._poll_loop` 的 `httpx.HTTPError` 分支做连续失败计数 + `record_offline_incident` + 5s 退避；客户端类 4xx 仍返回 `[]`（保持原语义）
  - `AdvisorBot._deliver(chat_id, text, *, remove_keyboard=False) -> bool`：统一回复出口，`send_message` 返回 `False` 或抛异常时写 `ERROR` 日志并 `ops.notify(severity="CRITICAL", source="telegram:delivery", alert_key="telegram_reply_undelivered")`；`_send_reply` 为其薄封装
  - `await run_app(stop_event=None)`（注入 `T212Client(..., snapshot_loader=db.get_latest_snapshot)`，启动就绪后执行 `await ops.flush_pending()`，`finally` 依次容错释放 `t212`/`market_data`/`ops`/`tg`/`db`）

### 3.4 模块 ↔ 测试文件 ↔ 共享 Mock 桩映射表（改哪块跑哪个文件）
| 源码模块 (`src/`) | 对应单元测试 (`tests/`) | 共用 Mock / 桩对象（变更上游接口时须同步检查） |
| :--- | :--- | :--- |
| `config.py`, `llm_client.py` | `test_llm_pricing_and_usage.py` | Fake GenAI Client |
| `calendar_utils.py`, `scheduler.py` | `test_calendar_and_scheduler.py` | `_MockT212`, `_MockMarketData`, `_StubLLM`, `_StubTg` |
| `storage/db.py` | `test_db_storage.py` | 临时目录 SQLite 真实建库 |
| `t212_client.py` | `test_t212_client.py` | `httpx.MockTransport` |
| `market_data.py` | `test_market_data.py` | `_DummyOps`, Fake `httpx`/yfinance |
| `analytics.py`, `portfolio_rules.py`, `screener.py` | `test_analytics_and_rules.py` | 纯离线 `pd.DataFrame` 构造器（须含 `Open/High/Low/Close/Volume`） |
| `ops_notifier.py`, `healthcheck.py`, `main.py`, `telegram_client.py`, `tools.py`, `handlers.py(/health)` | `test_v504_ops_and_grounding.py` | `_StubTelegram`, `_FakeT212Healthy`, `_FakeMarketHealthy` |
| `schemas.py` & 全链路端到端 (`/pos`/`report`/`pick`/`us`/`stats`) | `test_db_and_end_to_end.py` | `_FakeT212`, `_FakeMarketData`, `_RecordingLLM`, `_RecordingTg` |

> ⚡ 单文件秒测：`python -m unittest tests.<模块名>`；改完收口必跑全量 `python -m tests.run_tests tests`。

### 3.5 核心算法模块 (`IMP-06 ~ 16`) ↔ `_ref/REF_*` 参考源码精确符号映射（免翻整库）
- **`IMP-06` (`analytics.py`)**：`_ref/REF_stock-master/stock-master/instock/core/kline/cyq.py`（`CYQCalculator` 筹码分布三角分布与换手衰减模型）& `indicator/calculate_indicator.py`（`MACD/BOLL/ADX`）
- **`IMP-07` (`portfolio_rules.py`)**：`_ref/REF_daily_stock_analysis-main/daily_stock_analysis-main/src/agent/disagreement.py`（`build_agent_disagreement_summary`）& `risk_override.py`（`build_risk_override_plan`, `_downgrade_signal`）
- **`IMP-08` (`screener.py`)**：`_ref/REF_stock-master/.../strategy/`（`breakthrough_platform.py`, `low_atr.py`, `enter.py`）；`_ref/REF_stock-screener-main/stock-screener-main/src/screening/`（`phase_indicators.py`: `validate_minervini_trend_template`/`detect_vcp_pattern`/`calculate_relative_strength`/`classify_phase`, `benchmark.py`, `screener.py`: `calculate_value_score`）；`_ref/REF_PyPortfolioOpt-main/.../risk_models.py`（仅提取纯 NumPy 对角协方差收缩与增量风险贡献思想，不引入 `scipy`/`cvxpy`/`sklearn`）
- **`IMP-12` (`report_agent.py`)**：`_ref/REF_daily_stock_analysis-main/daily_stock_analysis-main/src/schemas/analysis_context_pack.py`（`AnalysisContextPack`）& `src/analysis_context_pack_prompt.py`（`format_analysis_context_pack_prompt_section`）
- **`IMP-14` (`chat_agent.py`)**：`_ref/REF_daily_stock_analysis-main/daily_stock_analysis-main/src/agent/chat_object_decision.py`（`decide_chat_object`, `object_mentions`）& `src/llm/provider_cache.py`
- **`IMP-16` (`scheduler.py`)**：`_ref/REF_TradingAgents-main/TradingAgents-main/tradingagents/memory/settlement.py`（`Settlement`, `fetch_returns`）& `reflection.py`（`Reflector`）
