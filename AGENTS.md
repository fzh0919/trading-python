# Trading 212 × Gemini AI Advisor — 工作区开发入口 (AGENTS)

> 本文件为自动加载的热启动骨架；详细规则按需查阅 [`DEVELOPER_GUIDE.md`](./DEVELOPER_GUIDE.md)（模型路由 & DB 表）、[`docs/MODULE_CONTRACTS.md`](./docs/MODULE_CONTRACTS.md)（接口签名）或 [`CHANGELOG.md`](./CHANGELOG.md)（`v5.5.0+` 变更）。

## 一、不可破坏的系统铁律

1. **SQLite 写锁单入口**：所有写库必须走 `AdvisorDatabase._connect(write=True)`，严禁私建连接绕过 `self._write_lock`。
2. **降级数据不回写**：`portfolio.is_fallback` 或 `macro.is_stale` 为 `True` 时严禁调用 `db.save_snapshot()` 或覆盖有效缓存；空 DataFrame 或全挂宏观响应不得写入缓存。
3. **双币种严格隔离**：账户现金/净值/盈亏固定 `GBP`，标的现价/均价/止损用本地货币（`USD`/`EUR`/`GBp`/`GBP`），英股须防 `GBp` 与 `GBP` 100 倍量级错位，严禁交给 LLM 口算。
4. **`/report` 基石与纯文本控字**：`07:30 UK` `/report` 是全天 `Single Source of Truth`，严禁降低其深度；Telegram 输出禁用 `#` 与 `**`，受字数硬上限（`/report<=2800`、`/pick<=1000`、`/us<=450|600`）。
5. **英文检索 → 中文输出解耦**：`src/agents/tools.py` 的联网检索指令与 `MarketToolkit.declarations()` 工具声明保持英文（对齐 RNS/SEC 一手信源），检索结果与终端决策保持紧凑简体中文。
6. **数据库延续**：线上 `data/advisor.db` 只做幂等迁移（`CREATE TABLE IF NOT EXISTS` / `_ensure_columns`），严禁删库；发布版本同步 `src/version.py` 与 `pyproject.toml`。

## 二、模块 ↔ 职责 ↔ 单测总索引

> 修改前先用 `rg -n "^(class |async def |def |    async def |    def )" <文件>` 定位行号再按 100~150 行分段读取，禁止整篇读取 >400 行大文件。

| 源码模块 (`src/`) | 核心职责 | 对应单测 (`tests/`) |
| :--- | :--- | :--- |
| `config.py`, `config.yaml`, `version.py` | Pydantic 配置、`.env` 出网校验、池热重载、版本号 | `test_llm_pricing_and_usage.py` |
| `engine/calendar_utils.py` | `US/UK/XETRA/EURONEXT` 交易日历、半日市、夏令时 | `test_calendar_and_scheduler.py` |
| `storage/db/` (`core`, `schema`, `portfolio_repo`, `recommendations_repo`, `ops_and_chat_repo`, `market_cache_repo`, `memory_and_wallet_repo`) | SQLite `WAL` 异步仓（Domain Mixins 共享 `core._connect` 单写锁）、快照/结算/`nav_history`/双钱包/缓存 | `test_db_storage.py`, `test_sidecar_memory_and_weekly.py`, `test_intraday_sentinel.py`, `test_nav_performance.py`, `test_v554_module_facades.py`, `test_v559_m6_observability.py` |
| `data_providers/t212_client/` (`models`, `wallets_and_tickers`, `http_mixin`, `client`, `context_formatting`) | T212 双账户只读、限流退避、挂单量化、分红/频率统计 | `test_t212_client.py` |
| `data_providers/market_data/` (`models_and_parsing`, `external_clients`, `yahoo_mixin`, `service`) | Yahoo+Finnhub+FRED 并发聚合、`2y` 缓存、盘前跳空、基础财务 | `test_market_data.py` |
| `data_providers/analytics/` (`technical`, `exposure`, `correlation`, `performance`, `postmarket_review`) | 复权自愈、CYQ/周线/ATR/ADX、穿透暴露、CVaR、TWR/资金流校准、盘后 0-Token 执行账单与 2x2 象限 | `test_analytics_and_rules.py`, `test_nav_performance.py`, `test_v553_observability_and_attribution.py`, `test_v559_m6_observability.py`, `test_v5510_phase2_postmarket_review.py` |
| `engine/deployment_gate/` (`common`, `core_policy`, `ledgers`, `regime`, `triggers`, `banner`) | 纯确定性 CIO 组合门控、四桶核心政策、Regime/危机通道、横幅、Invest 派息宽基蓄水池准入 | `test_report_pick_gate.py`, `test_v551_dual_philosophy.py`, `test_v559_m5_capital_efficiency.py` |
| `engine/portfolio_rules/` (`models_and_friction`, `dual_philosophy`, `sizing`, `holding_eval`, `risk_override`) | 双哲学持仓/买入规则（ISA 价值门 vs Invest 动态 R:R）、护栏、算仓、蓄水池旁路与清宽基换股配对指令 | `test_analytics_and_rules.py`, `test_report_pick_gate.py`, `test_v551_dual_philosophy.py`, `test_invest_wallet_split.py`, `test_wallet_rules_and_commands.py`, `test_v559_m5_capital_efficiency.py` |
| `engine/screener/` (`patterns`, `scoring`, `runner`, `audit`, `audition`), `engine/universe_seeds.py`, `refresh_pool.py` | 四维双轨量化选股、150 只海选底池、VPVR/Stage2/VCP | `test_analytics_and_rules.py`, `test_universe_audition.py`, `test_v551_dual_philosophy.py`, `test_v559_m6_observability.py` |
| `agents/llm_client.py` | Gemini+DeepSeek 混合路由/双向灾备、Thinking 钳位、工具裁剪 | `test_llm_pricing_and_usage.py` |
| `agents/ds_sidecar.py` | DeepSeek 侧车出口（凭证门控、硬超时静默降级、JSON 抽取） | `test_ds_pick_prefilter_and_auditor.py` |
| `agents/semantic_auditor.py` | `/report` & `/pick` 只读语义质检与事实底座 | `test_ds_pick_prefilter_and_auditor.py`, `test_report_pick_gate.py` |
| `agents/prompts.py`, `agents/schemas.py` | 双哲学画像、结构化 Schema、JSON 自修复与倒金字塔渲染 | `test_db_and_end_to_end.py`, `test_report_pick_gate.py`, `test_v551_dual_philosophy.py`, `test_v559_m5_capital_efficiency.py` |
| `agents/tools.py` | 分域 Google 强搜（英搜中出）与 `MarketToolkit` | `test_v504_ops_and_grounding.py` |
| `agents/report_agent/` (`fact_sheet`, `debate`, `core_deployment`, `morning_report`, `sentinels`) | `/report` 三步辩论流水线、门控接线、`/us` 哨兵、盘中快报 | `test_db_and_end_to_end.py`, `test_intraday_sentinel.py`, `test_report_step2_budget.py`, `test_report_debate_round2.py`, `test_report_pick_gate.py`, `test_v558_us_sentinel.py` |
| `agents/pick_agent.py` | `/pick` DeepSeek 初筛 -> Researcher -> Critic -> 硬拦截落库（含蓄水池票据约束块） | `test_db_and_end_to_end.py`, `test_ds_pick_prefilter_and_auditor.py`, `test_report_pick_gate.py`, `test_v559_m5_capital_efficiency.py`, `test_v559_m6_observability.py` |
| `agents/chat_agent.py` | `/pos` 0-Token 查仓、Implicit Cache、标的切片、`/deep` | `test_db_and_end_to_end.py`, `test_v558_daily_chat.py` |
| `bot/handlers.py`, `bot/scheduler/` (`settlement_math`, `fill_receipts`, `settlement_mixin`, `weekly_audit_mixin`, `intraday_mixin`, `postmarket_mixin`, `core`), `bot/telegram_client.py` | 指令路由、互斥锁、`/perf`/`/wallet`/`/review`、日程、T+5/10/20 结算、盘中回执、盘后账单与 Flash Lite 催化 | `test_calendar_and_scheduler.py`, `test_db_and_end_to_end.py`, `test_sidecar_memory_and_weekly.py`, `test_intraday_sentinel.py`, `test_nav_performance.py`, `test_wallet_rules_and_commands.py`, `test_v559_m6_observability.py`, `test_v5510_phase2_postmarket_review.py` |
| `ops_notifier.py`, `healthcheck.py`, `main.py` | `ops-notify` 双态告警、心跳/只读 DB 探活、生命周期 | `test_v504_ops_and_grounding.py` |
| `redaction.py` | 日志/告警/诊断统一脱敏出口（`token`/`api_key`/`Bearer`/`Basic` → `***`） | `test_v504_ops_and_grounding.py` |

## 三、开发验收命令

```bash
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.run_tests tests
```

- 单文件快测：`.\.venv\Scripts\python.exe -m unittest tests.<模块名>`（全部离线 Mock；行情桩须含 `Open/High/Low/Close/Volume`）。
- 出网与脱敏自查：`.\.venv\Scripts\python.exe scripts\audit_egress.py`（非零退出码需人工复核）。
- 每次迭代完成后将变更追加至 [`CHANGELOG.md`](./CHANGELOG.md) 顶部。

## 四、部署与容器

- 容器名 `t212-ai-advisor`（纯出站无映射端口，登记见根 `DEPLOYMENT_CHEATSHEET.md`）；发版统一走根目录 `scripts\deploy-n95.ps1`，严禁在 Dockge 修改本项目 `docker-compose.yml`。

## 五、安全红线（P1）

- **出网白名单**：`src/config.py` `EnvSettings._guard_egress_endpoints` 强制 `https` 并校验 `DEEPSEEK_BASE_URL`（默认 `api.deepseek.com` 等）与 `T212_BASE_URL`（`live/demo.trading212.com`）。放宽出口**只能改 `.env`**（`LLM_EGRESS_ALLOWED_HOSTS` / `T212_EGRESS_ALLOWED_HOSTS`），严禁修改 `_LLM_EGRESS_HOSTS` / `_T212_EGRESS_HOSTS` 常量。
- **发版红线闸**：`sync-up.ps1` 与 `deploy-n95.ps1` 的红线闸不得移除、放宽或绕过；调整正则须两侧同步并人工确认。
- **交付前 diff 自查 5 条**（任一为「是」须在变更说明中显式列出并等人工确认）：
  1. 是否新增 `http(s)://` 出网字面量或新域名？
  2. 是否新增日志/异常打印 `api_key` / `token` / `secret` / `cfg.env`？
  3. 是否新增读取环境变量（尤其凭据类）？
  4. 是否改动 `base_url` / 超时 / 重试 / 灾备链？
  5. 是否改动风控护栏开关（`deployment_gate_mode` / `pick_rr_guard_mode` / `trade_cadence_guard_mode` / `intraday_fill_receipt_enabled` 等）？
- **Agent 执行禁区**：
  - 禁止执行：`git push`、`scripts\deploy-n95.ps1`、`sync-up.ps1`。
  - 密钥只做布尔探测（`Test-Path` / `[bool]$env:X`），严禁 `Get-Content .env` 或回显密钥片段。
  - 文件读取黑名单：`.env`、`.env.*`、`data/`、`*.db`、`*.db-wal`、`*.db-shm`、`*.sqlite`、`*.sqlite3` 一律不读。
