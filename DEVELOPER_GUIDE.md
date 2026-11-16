# Trading 212 × Gemini AI Advisor — 开发者与 AI 维护手册 (DEVELOPER_GUIDE)

本文档为紧凑型工程交接手册，供开发者与后续接入的 AI 助手以最少 Token 快速掌握架构边界、核心契约、N95 本地重算体系与 **22 个核心模块速查索引（`IMP-01 ~ IMP-22`）**。历史改动流水、模块评分与版本发布日志统一归档于 [`CHANGELOG.md`](./CHANGELOG.md)。

---

## 一、核心设计铁律与账户实况前提

1. **`/report` 核心基石地位（严禁降级铁律）**：
   - `/report` 是全系统唯一的全景事实底座与风控总纲，是同交易周期（每日 `07:30 UK` 切换）内 `/pick`、`/us` 与自由聊天框的**绝对参照物（Single Source of Truth）**。
   - **任何优化严禁降低 `/report` 的规模、上下文完整性与全市场分域联网检索深度**；按用户授权（`v5.2.0`），`/report` 保留三步对话结构，仅将 Bear 挑刺审计（`report_auditor_model`）锁定 `Argon HIGH`，PM 拍板与 `/pick` 终审切换 `Hydrogen HIGH`，全市场联网检索切换 `gemini-3.5-flash-lite` 并通过 SQLite 搜索快照与工具收口减少重复调用。
2. **N95 本地重算换 Token 策略（定时任务不怕慢，日常交互极速轻量）**：
   - **硬件算力与运行定位**：`IMP-01 ~ IMP-22` 已全部完成并通过全链路验收，面向英国本地常驻主机 **Intel N95 + 16GB RAM + 512GB SSD** 的 Docker Compose 容器化运行（GCP 旧版停服后经 `sqlite3 .backup` 统一切割），本地 CPU 与内存算力完全充裕。
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

## 二、目录结构与 `REF` 参考项目借鉴映射

```text
trading-python/                     # [当前主项目]
├── docs/
│   ├── PROJECT_MIGRATION_RULES.md  # [强制规范] N95 Homelab 容器化、SQLite 备份与 ops-notify 统一告警规范 (v1.2)
│   ├── T212_API_manual.json        # [官方规范] Trading 212 Public API v0 OpenAPI 3.0 手册（含 Position.createdAt 与限流配额）
│   └── finnhub_API_manual.json     # [官方规范] Finnhub API OpenAPI 规范手册（免费 60次/分；/quote、/calendar/earnings、/company-news、/stock/profile2、/stock/recommendation 等参考）
├── _ref/                           # [本地只读参考区，已被 .gitignore 与 .dockerignore 排除]
│   ├── OLD_trading-gs/             # [已停用] 初代 GAS 脚本 (v4.2~v4.4)，仅作历史对照
│   ├── REF_stock-master/           # [参考库 1] instock/core/kline/cyq.py (CYQ 筹码分布)、indicator/ (MACD/BOLL/ADX)、strategy/ (平台突破/低ATR/放量)
│   ├── REF_stock-screener-main/    # [参考库 2] Minervini 8 条件模板、VCP 波动收敛、相对强弱 RS 与市场广度
│   ├── REF_PyPortfolioOpt-main/    # [参考库 3] Ledoit-Wolf / PSD 收缩协方差与组合边际风险度量
│   ├── REF_Riskfolio-Lib-master/   # [参考库 4] 组合风险预算与多资产风控参考
│   ├── REF_skfolio-main/           # [参考库 5] 组合优化与风险平价参考
│   ├── REF_TradingAgents-main/     # [参考库 6] memory/settlement.py & reflection.py (T+5 后验 Alpha 结算与微型反思)、agents/context.py (身份锚定)
│   └── REF_daily_stock_analysis-main/ # [参考库 7] analysis_context_pack (当日快照复用)、disagreement.py (多空分歧表)、risk_override.py (后置护栏)、chat_object_decision.py (单聊切片)、provider_cache.py (缓存前缀)
├── Dockerfile / docker-compose.yml # N95 容器化编排（锁定 python:3.12.13-slim-bookworm、mem_limit: 2g、healthcheck、homelab-net）
├── .env.example / config.yaml      # 密钥模板（含 OPS_NOTIFY_*）/ 模型路由、日程、风控阈值与候选股票池
├── DEVELOPER_GUIDE.md              # 本文档：架构契约 + 22 个核心模块速查索引
├── CHANGELOG.md                    # 历史变更归档：IMP-01~22 验收评分、执行日志与各版本发布记录
├── README.md                       # 项目背景与 N95 Git + Docker Compose 部署说明
├── src/
    │   ├── main.py                 # 入口：初始化 SQLite/OpsNotifier/各Client，维护 /tmp 心跳、致命崩溃兜底与长轮询
    │   ├── healthcheck.py          # 纯 Python 容器探活：校验 /tmp/advisor.heartbeat 新鲜度与只读 SQLite 状态
    │   ├── version.py              # 版本号单一真源 (APP_VERSION = "5.2.0")
    │   ├── config.py               # Pydantic 配置加载 (.env + config.yaml)
    │   ├── ops_notifier.py         # 统一运维告警（ops-notify 双态分流、WARN 冷却、CRITICAL 防抖与离线缓冲）
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
| **`/report` Step 1 联网（核心）** | `report_search_model` | `gemini-3.5-flash-lite` (`LOW`) | `GoogleSearch`（每批 `<=3` 标的，`Semaphore(3)`，3h SQLite 持久化快照） | 注入内部上下文 |
| **`/report` Step 2A 机会（核心）** | `report_analyst_model` | `gemini-4-hydrogen` (`HIGH`) | 本地全维 `ContextPack` + `MarketToolkit`（`<=2` 轮，禁重搜） | 内部草稿 `<=1200` 字 |
| **`/report` Step 2B 风控（核心）** | `report_auditor_model` | `gemini-4-argon` (`HIGH`) | 本地全维 `ContextPack` + `MarketToolkit`（`<=2` 轮，禁重搜） | 内部草稿 `<=1200` 字 |
| **`/report` Step 3 PM 拍板（核心）**| `report_model` | `gemini-4-hydrogen` (`HIGH`) | `MorningReportDecision` + 自修复 + `RiskOverride` | **Telegram `<=1800` 字** *(Prompt 1500~1650)* |
| **`/pick` Stage 1 投研员** | `pick_researcher_model` | `gemini-4-hydrogen` (`HIGH`) | 本地四维量化+新闻/评级预聚合 + `MarketToolkit`（`<=2` 轮，带方向 `<=3` 轮）| 内部提案 |
| **`/pick` Stage 2 风控官** | `pick_model` | `gemini-4-hydrogen` (`HIGH`) | `PickDecision` + 自修复 + Post-Screen 确定性校验 | **Telegram `<=1000` 字** |
| **`/us` 美股增量哨兵（分级门控）** | `us_sentinel_model` | 平稳日 `Hydrogen LOW` / 异动日 `Hydrogen MEDIUM` (或 `Argon MEDIUM`) | 本地异动门控（Anomaly Gate）：无异动免强搜极简播报；有异动深挖 | **平稳 `<=450` / 异动 `<=600` 字** |
| **日常自由单聊（默认轻量）** | `chat_model` | `gemini-4-hydrogen` (`LOW`)（`/deep` 切高阶） | 本地切片（含新闻快照与常规持仓折叠）+ 按需工具（`<=2` 轮，`/deep <=4`） | **Prompt `<=500` 字（`/deep` `1000~1200`）** |
| **盘后 T+5 复盘微反思** | `search_model` / `chat_model` | `gemini-4-hydrogen` (`LOW`) | 本地 0-Token 算完 T+5 Alpha 后生成 `2~4` 句（`<80 tokens`）教训 | 存入 SQLite 记忆列 |

### 3.2 SQLite 数据库核心表速查（`data/advisor.db`，周期切换点 `07:30 UK`）
1. `portfolio_snapshots(id, cycle_date, fetched_at_uk, fetched_at_ms, total_combined_value, payload_json)`：双账户完整 JSON 快照。
2. `morning_reports(cycle_date PK, report_text, conclusion_text, premarket_note, daily_guardrails, context_pack_json, updated_at_ms)`：当日 `/report` 全文、`daily_vetoes` JSON、`/us` 笔记及当日预计算 `AnalysisContextPack` 快照（**供 `/pick`、`/us` 与 `chat` 跨指令 0-Token 复用**）。
3. `pick_radars(id, created_at_uk, created_at_ms, focus_query, research_text, final_text)`：`/pick` 投研草稿与终审报告存档。
4. `pick_recommendations(id, created_at_uk, created_at_ms, account, symbol, action, entry_price, thesis, horizon_days, benchmark_symbol, settled_at_uk, raw_return_pct, alpha_pct, outcome_status, reflection_lesson, t10_return_pct, t10_alpha_pct, t20_return_pct, t20_alpha_pct, mfe_pct, mae_pct)`：结构化推荐与 **T+5/T+10/T+20 后验结算、MFE/MAE 极值及微型反思档案**（供 `/report`、`/pick` 注入历史教训及 `/stats` 看板读取）。
5. `chat_history(id, cycle_date, user_query, bot_reply, created_at_ms)`：单周期对话记忆（自动裁剪保留最近 `12` 轮）。
6. `llm_usage_logs(id, ..., cycle_date, call_label, model, thinking_level, api_rounds, prompt_tokens, cached_tokens, output_tokens, thought_tokens, estimated_cost_usd, latency_ms, success, search_queries)`：Token、缓存命中与 Google Search 用量账本。
7. `ops_events(id, created_at_uk, created_at_ms, severity, source, alert_key, message, notified, suppressed_count)`：运维告警审计表。
8. `scheduler_runs(job_name, uk_date, fired_at_uk, PK(job_name, uk_date))`：定时任务当日/当周触发幂等锁。
9. `ohlcv_cache(symbol, period, last_bar_date, last_close, updated_at_ms, payload_json, PK(symbol, period))`：`2y` 日线持久化底座（支持 `1mo` 增量拼接与 `>0.5%` 拆股全量重拉，`prune_old_records` 清理超期行）。
10. `t212_instruments_metadata(ticker PK, yahoo_symbol, isin, currency_code, short_name, inst_type, extended_hours, working_schedule_id, isa_eligible, invest_eligible, updated_at_ms)`：T212 标的元数据与分账户可买性长期归档表（永久保留）。
11. `t212_dividend_archive(id PK, account, ticker, yahoo_symbol, amount_gbp, gross_per_share, paid_on, paid_on_ms)`：T212 历史实收英镑股息长期归档表（永久保留）。
12. `quant_health_audits(audit_date PK, created_at_uk, summary_text, payload_json)`：周末离线候选池与止盈/止损阈值体检档案。
13. `fundamentals_cache(symbol PK, ..., payload_json, updated_at_ms)` / `active_universe_pool(symbol PK, ..., quality_score, isa_eligible, invest_eligible, is_etf, updated_at_uk)` / `index_constituents_cache(index_name PK, symbols_json, updated_at_ms)`：多市场底池海选（S&P500+NASDAQ100+FTSE100+FTSE250+EuroStoxx50+LSE ETF -> 150 只精锐个股）持久化与基本面 21 天防限流缓存。

### 3.3 核心模块接口字典（精确签名与字段契约速查，随 `IMP` 滚动更新）

> 💡 下游窗口直接按本字典接线，**不必通读千行上游源码**；后续新增公共函数/字段后，须在对应小节同步补齐 1~2 行契约。

- **`src/config.py`（`IMP-01` & N95 迁移前置）**：
  - `EnvSettings` 新增：`ops_notify_url: str = ""`（`OPS_NOTIFY_URL`）、`ops_notify_token: str = ""`（`OPS_NOTIFY_TOKEN`）
  - `ModelsSection` 新增：`chat_deep_model`, `chat_deep_thinking_level`, `google_search_free_quota_monthly` (`5000`), `google_search_warn_pct` (`85.0`)
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
  - `GeminiClient(..., round_timeout_seconds=120.0, tool_timeout_seconds=60.0, max_concurrent_tools=6, max_tool_result_chars=4000, usage_summary_provider=None, search_free_quota_monthly=5000, search_warn_pct=85.0)`；`set_usage_summary_provider(...)`（单轮 `asyncio.wait_for` 超时、同轮 `asyncio.gather` 并发+裁剪及 `"this_month"` Search 免费额度 `OpsNotifier` 预警）
- **`src/agents/prompts.py` & `src/agents/schemas.py`（`IMP-10`）**：
  - `build_investor_profile(strategy, *, include_order_audit_rule=True)` & `build_agent_system_instruction(role_instruction, strategy, *, include_order_audit_rule=True)`（`/pick` 传 `False` 隔离存量挂单审计原则）
  - `FlaggedPositionItem(symbol: str, summary: str)`；`MorningReportDecision.flagged_positions: list[FlaggedPositionItem]`（`mode="before"` 校验器兼容旧 `str` / `dict`）
  - `render_morning_report(decision, max_chars=1800, *, risk_override: RiskOverrideResult | None = None) -> str` & `render_pick_report(decision, max_chars=1000, *, risk_override: RiskOverrideResult | None = None) -> str`（统一高亮 `[🛡️风控护栏修正]` 并保留护栏驳回项，`_fit_max_chars` 尾部防孤立 `【...】` 标题行）；`trim_text_at_line_boundary(text, max_chars) -> str`（支持单行句界/分句边界优雅裁剪）
- **`src/agents/tools.py`（`IMP-11` & `IMP-21`）**：
  - `run_forced_market_grounding(..., max_concurrent_searches: int = 3, priority_symbols: list[str] | None = None) -> str`（`priority_symbols` 脱离 3 股拼团转 1 对 1 独立深搜，`asyncio.Semaphore` 限流 + `_GROUNDING_CACHE` 过期清理）
  - `MarketToolkit(market_data, portfolio, *, llm=None, search_model="gemini-4-hydrogen", ops_notifier=None, db=None)`（`inspect_ticker` 传入持仓 `isin`/`created_at` 与 `bundle`、按需调 `get_us_deep_intel` 与 `db.get_symbol_lessons`，含 CYQ/周线/除息/身份紧凑裁剪 `<4000` 字符；`check_portfolio_correlation` 改调 `get_histories` 并补拉 `SPY`/`ISF.L`/区域基准）
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
- **`src/bot/telegram_client.py` & `src/main.py`（`IMP-18`）**：
  - `TelegramClient(..., max_retries=2, base_backoff_s=0.5, max_retry_wait_s=30.0)`（`_post` 遇 `429` 按 `parameters.retry_after`/`Retry-After` 等待，遇 `5xx`/网络瞬断指数退避重试，耗时仍耗尽时 `httpx.HTTPError` 照常上抛）
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

---

## 四、开发、测试、大文件读取与跨模块不变量铁律

- **大文件禁止全量 `read_file`（防上下文膨胀）**：
  - 对 **>400 行的源文件/测试文件**（如 `t212_client.py` 1597行、`screener.py` 1417行、`market_data.py` 1356行、`analytics.py` 1287行、`report_agent.py` 1087行、`db.py` 1068行及大型单测），**禁止直接整篇读入上下文**；必须先用 `rg -n "^(class |async def |def |    async def |    def )" <文件>` 定位行号，再按 100~150 行范围分段读取；上游接口签名直接查上文 **3.3 接口字典**。
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
  5. **数据库延续 & 容器同步边界**：线上 `data/advisor.db` 只做幂等迁移（`CREATE TABLE IF NOT EXISTS` / `_ensure_columns`），严禁删库；宿主机备份或迁移库文件必须通过 `sqlite3 <db> ".backup <clean.db>"` 生成一致快照；代码通过 Git 同步至 `/opt/homelab/stacks/trading-python/` 并由 Docker Compose 重建（`.gitignore` 与 `.dockerignore` 均严格排除 `.env`、`data/`、`_ref/`、`logs/`、`__pycache__/`、`.venv/`），版本发布同步更新 `src/version.py` 与 `pyproject.toml`。

---

## 五、核心模块速查索引（`IMP-01` → `IMP-22` 架构地图）

> 按自底向上的 7 阶段依赖排列，方便后续 Debug 或迭代时**零翻库直接定位源码与测试文件**。所有 `IMP-01 ~ IMP-22` 的历史重构清单、评分（`9.5 ~ 9.9 / 10`）与单测演进记录均已归档至 [`CHANGELOG.md`](./CHANGELOG.md)。

| 模块 & 阶段 | 核心文件（点击直达） | 核心职责与关键逻辑摘要（Debug / 迭代速查） | 对应单测 |
| :--- | :--- | :--- | :--- |
| **IMP-01**<br>Phase 1 配置 | [`src/config.py`](./src/config.py)<br>[`config.yaml`](./config.yaml) | Pydantic 校验 `.env` + `config.yaml`；`AppConfig.reload_pools_if_changed()` 按 `mtime_ns` 热重载自选/选股池（坏 YAML 保护）；管理模型路由、异动/风控阈值与 Search 月额度。 | [`test_llm_pricing_and_usage.py`](./tests/test_llm_pricing_and_usage.py) |
| **IMP-02**<br>Phase 1 日历 | [`src/engine/calendar_utils.py`](./src/engine/calendar_utils.py) | `infer_market_from_symbol` 识别 `US/UK/XETRA/EURONEXT`；覆盖英美欧假日与半日市（美 `13:00 ET`、LSE `12:30 UK`、Euronext `14:05 CET`）、动态收盘 `UsMarketSchedule`、`add_trading_days` 与 `get_market_calendar_alerts`。 | [`test_calendar_and_scheduler.py`](./tests/test_calendar_and_scheduler.py) |
| **IMP-03**<br>Phase 1 存储 | [`src/storage/db.py`](./src/storage/db.py) | `WAL` + `busy_timeout=5000` + `self._write_lock` 写锁串行化；`07:30 UK` 周期管理；`morning_reports.context_pack_json`、`pick_recommendations` `T+5` 结算/反思读写、`this_month` 统计与 `prune_old_records(90)`。 | [`test_db_storage.py`](./tests/test_db_storage.py) |
| **IMP-04**<br>Phase 2 券商 | [`src/data_providers/t212_client.py`](./src/data_providers/t212_client.py) | T212 双账户只读客户端；连接池复用、`_portfolio_lock`、`x-ratelimit-*` 精准退避；解析 `Position.created_at`/`isin`、部分成交挂单、30 天出入金/派息流水及内存/DB 双级 5xx 回退（不回退 `401/403`）。 | [`test_t212_client.py`](./tests/test_t212_client.py) |
| **IMP-05**<br>Phase 2 行情 | [`src/data_providers/market_data.py`](./src/data_providers/market_data.py) | Yahoo + Finnhub + FRED 并发聚合；0-info `get_histories`、Singleflight 去重、`normalize_history_index` 剥离混合时区、`TickerMarketBundle` 身份/除息/质量因子、6h 美股深度情报、LRU 防空投毒与过期兜底。 | [`test_market_data.py`](./tests/test_market_data.py) |
| **IMP-06**<br>Phase 3 量化 | [`src/data_providers/analytics.py`](./src/data_providers/analytics.py) | 本地 0-Token 量化引擎：`_normalize_adjusted_ohlcv` 拆股/100 倍坏点自愈、`CYQMetrics` 筹码分布、周线共振/`ATR14`/`ADX14`/`BB`、建仓后回撤、除息守卫、分市场去 Beta 相关性与克隆 ETF 暴露归并。 | [`test_analytics_and_rules.py`](./tests/test_analytics_and_rules.py) |
| **IMP-07**<br>Phase 3 规则 | [`src/engine/portfolio_rules.py`](./src/engine/portfolio_rules.py) | 持仓规则与后置护栏：结合 `created_at` 算移动止盈、`ATR` 自适应止损、除息防假破位、挂单叠加、`4%` 现金+摩擦门槛；`build_disagreement_matrix` 本地多空分歧表与 `apply_risk_override_guardrail` 硬拦截降级。 | [`test_analytics_and_rules.py`](./tests/test_analytics_and_rules.py) |
| **IMP-08**<br>Phase 3 选股 | [`src/engine/screener.py`](./src/engine/screener.py) | `/pick` 四维双轨量化漏斗：4 桶 Stage 1（含防御低波桶）、Minervini Stage 2 / VCP / 平台突破 / 60d VPVR、双轨权重（`ISA 25/15/35/25` vs `Invest 35/35/15/15`）、双锚广度及 3 日跨时区对数收益 PSD 收缩协方差。 | [`test_analytics_and_rules.py`](./tests/test_analytics_and_rules.py) |
| **IMP-09**<br>Phase 4 网关 | [`src/agents/llm_client.py`](./src/agents/llm_client.py) | `GeminiClient` 异步网关：`normalize_thinking_level` 钳位、单轮 `wait_for` 硬超时、同轮 `asyncio.gather` 工具并发与 `_compact_tool_result` 裁剪、瞬时重试/备援降级及 `"this_month"` Search 免费额度 `OpsNotifier` 预警。 | [`test_llm_pricing_and_usage.py`](./tests/test_llm_pricing_and_usage.py) |
| **IMP-10**<br>Phase 4 渲染 | [`src/agents/schemas.py`](./src/agents/schemas.py) | `MorningReportDecision` / `PickDecision` 结构化 Schema（`FlaggedPositionItem` 去重）与 JSON 自修复；倒金字塔控字渲染、`RiskOverrideResult` `[🛡️风控护栏修正]` 置顶高亮及 `_fit_max_chars` 防孤立 `【...】` 标题裁剪。 | [`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-11**<br>Phase 4 工具 | [`src/agents/tools.py`](./src/agents/tools.py) | `run_forced_market_grounding` 分域 Google 强搜（`Semaphore(3)` 限流 + 过期缓存清理）；`MarketToolkit` 整合 `inspect_ticker`（含 CYQ/周线/除息/深度情报/历史教训，`<4000` 字）与 `check_portfolio_correlation`。 | [`test_v504_ops_and_grounding.py`](./tests/test_v504_ops_and_grounding.py) |
| **IMP-12**<br>Phase 5 晨报 | [`src/agents/report_agent.py`](./src/agents/report_agent.py) | `/report` 三步流水线（Step 1 联网 -> Step 2A/2B 机会/风控 -> Step 3 PM + `RiskOverride` `<=1800` 字），组装并落库 `AnalysisContextPack`（`is_fallback` 不写库）；`/us` 哨兵本地异动门控（平稳 `LOW<=450` / 异动强搜 `MEDIUM<=600`）。 | [`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-13**<br>Phase 5 选股 | [`src/agents/pick_agent.py`](./src/agents/pick_agent.py) | `/pick` 编排器：两段加载复用 `held_histories` 与 `ContextPack`/T+N 历史教训，串联 Researcher -> Risk Critic，经 `_post_screen_validate_decision` 自动回填现价、护栏硬拦截并按 `settlement_holding_days` 落库。 | [`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-14**<br>Phase 5 单聊 | [`src/agents/chat_agent.py`](./src/agents/chat_agent.py) | `/pos` 0-Token 实时查仓；`decide_chat_objects` 0-Token 提取提问标的并从 `ContextPack` 局部切片；静态早报/红线/`pick[:1050]`/跨周期摘要前置对齐 Implicit Cache；默认 `Hydrogen LOW<=500`，`/deep` 升档。 | [`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-15**<br>Phase 6 路由 | [`src/bot/handlers.py`](./src/bot/handlers.py) | `AdvisorBot` 指令路由与白名单鉴权；与调度器共享 `execution_lock` 互斥防重入；`/usage` 月度额度与缓存/思考占比、`/stats` 胜率/Alpha/教训/现价（含 `GBp` 校验）、`/health` 降级与静默告警体检。 | [`test_v504_ops_and_grounding.py`](./tests/test_v504_ops_and_grounding.py)<br>[`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-16**<br>Phase 6 调度 | [`src/bot/scheduler.py`](./src/bot/scheduler.py) | `Europe/London` 调度器：早报 `07:30~09:30 UK` 断网补发与成功才锁库、动态半日市周五雷达、`22:00 UK` `run_daily_settlement`（按英欧美日历计算 `T+N` Alpha + `Hydrogen LOW` 微反思 + 90 天库清理）。 | [`test_calendar_and_scheduler.py`](./tests/test_calendar_and_scheduler.py) |
| **IMP-17**<br>Phase 6 告警 | [`src/ops_notifier.py`](./src/ops_notifier.py) | `OpsNotifier` 统一运维告警：`ops-notify` 双态分流（技术单发 / 强业务阻塞双发）、WARN/CRITICAL 冷却合并、`flush_pending()` 去重恢复近 24h DB 未送达告警及顶层 `send_ops_notify_sync`。 | [`test_v504_ops_and_grounding.py`](./tests/test_v504_ops_and_grounding.py) |
| **IMP-18**<br>Phase 6 入口 | [`src/main.py`](./src/main.py)<br>[`src/bot/telegram_client.py`](./src/bot/telegram_client.py) | `TelegramClient._post` 支持 `429 retry_after` 与 `5xx`/断网指数退避；`run_app` 注入 `T212` `snapshot_loader` 冷启动兜底、启动 `ops.flush_pending()`、`/tmp` 心跳维护、致命崩溃上报与全连接池容错关闭。 | [`test_v504_ops_and_grounding.py`](./tests/test_v504_ops_and_grounding.py) |
| **IMP-19**<br>Phase 6 运维 | [`src/healthcheck.py`](./src/healthcheck.py)<br>[`docker-compose.yml`](./docker-compose.yml) | 纯 Python 容器探活（校验 `/tmp/advisor.heartbeat` 与只读 `advisor.db`，含宿主机相对路径回退）；Docker Compose 锁定 `python:3.12.13-slim-bookworm`、`mem_limit: 2g`、`homelab-net` 与全链路迁移热备合规验收。 | [`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-20**<br>Phase 7 预热与归档 | [`src/storage/db.py`](./src/storage/db.py)<br>[`src/data_providers/market_data.py`](./src/data_providers/market_data.py)<br>[`src/data_providers/t212_client.py`](./src/data_providers/t212_client.py)<br>[`src/bot/scheduler.py`](./src/bot/scheduler.py)<br>[`src/agents/report_agent.py`](./src/agents/report_agent.py) | `ohlcv_cache`（`2y` 日线 `1mo` 增量拼接与 `>0.5%` 拆股重拉）、T212 `instruments` 元数据与分红长期归档（含息总回报）、定时任务前 10 分钟 `run_prewarm` 静默预热、Finnhub `eps_estimate` 保留与 `get_profile_by_isin` 兜底、`/us` `get_us_prepost_quotes` 盘前 15m 真实跳空补全。 | [`test_db_storage.py`](./tests/test_db_storage.py)<br>[`test_market_data.py`](./tests/test_market_data.py)<br>[`test_t212_client.py`](./tests/test_t212_client.py)<br>[`test_calendar_and_scheduler.py`](./tests/test_calendar_and_scheduler.py)<br>[`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-21**<br>Phase 7 深度风控与情报 | [`src/data_providers/market_data.py`](./src/data_providers/market_data.py)<br>[`src/data_providers/analytics.py`](./src/data_providers/analytics.py)<br>[`src/data_providers/t212_client.py`](./src/data_providers/t212_client.py)<br>[`src/engine/screener.py`](./src/engine/screener.py)<br>[`src/agents/tools.py`](./src/agents/tools.py)<br>[`src/agents/report_agent.py`](./src/agents/report_agent.py) | FRED `limit=25` 5d/20d 利率动量与 `BAMLH0A0HYM2` 高收益债利差、`get_us_deep_intel_batch` 批量美股评级/EPS 惊喜/高管 MSPR 接入 `/report` 与 `/pick`、PSD 收缩协方差前 3 波动风险贡献源与 95% 单日历史 CVaR、挂单币种对齐偏离/ATR 倍数/20 日经验触价率预审、重点异动/财报持仓 1 对 1 独立 Google Search。 | [`test_market_data.py`](./tests/test_market_data.py)<br>[`test_analytics_and_rules.py`](./tests/test_analytics_and_rules.py)<br>[`test_t212_client.py`](./tests/test_t212_client.py)<br>[`test_v504_ops_and_grounding.py`](./tests/test_v504_ops_and_grounding.py)<br>[`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |
| **IMP-22**<br>Phase 7 多窗归因与体检 | [`src/storage/db.py`](./src/storage/db.py)<br>[`src/engine/screener.py`](./src/engine/screener.py)<br>[`src/bot/scheduler.py`](./src/bot/scheduler.py)<br>[`src/bot/handlers.py`](./src/bot/handlers.py) | `pick_recommendations` 幂等扩展 `T+10/T+20` 延伸 Alpha 与 `MFE/MAE` 极值分布、盘后 0-Token 自动追评、周六闲时 `run_weekly_quant_audit`（120d 滚动高相关重叠、`>=60d` Phase 4 僵尸股、突破样本止盈止损回测）静默落库及 `/stats` 看板全景展示。 | [`test_db_storage.py`](./tests/test_db_storage.py)<br>[`test_analytics_and_rules.py`](./tests/test_analytics_and_rules.py)<br>[`test_calendar_and_scheduler.py`](./tests/test_calendar_and_scheduler.py)<br>[`test_db_and_end_to_end.py`](./tests/test_db_and_end_to_end.py) |

---

## 六、历史修改日志（已冷热分离）

全部 `IMP-01 ~ IMP-22` 模块详细改动记录、现状评分、测试演进及历史版本修改日志（`v5.0.1 ~ v5.2.0`），统一归档于 [`CHANGELOG.md`](./CHANGELOG.md)，新窗口默认无需通读历史归档。
