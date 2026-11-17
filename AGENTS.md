# Trading 212 × Gemini AI Advisor — 工作区开发入口 (AGENTS)

> 本文件为 Antigravity 自动加载的热启动骨架；仅在需要时按需查阅 [`DEVELOPER_GUIDE.md`](./DEVELOPER_GUIDE.md)（模型路由 & DB 表）、[`docs/MODULE_CONTRACTS.md`](./docs/MODULE_CONTRACTS.md)（接口签名字典）或 [`CHANGELOG.md`](./CHANGELOG.md)（近期变更）。

## 一、不可破坏的系统铁律

1. **SQLite 写锁单入口**：所有写库必须走 `AdvisorDatabase._connect(write=True)`，严禁私建连接绕过 `self._write_lock`。
2. **降级数据不回写**：`portfolio.is_fallback` 或 `macro.is_stale` 为 `True` 时严禁调用 `db.save_snapshot()` 或覆盖有效缓存；空 DataFrame 或全挂宏观响应不得写入缓存。
3. **双币种严格隔离**：账户现金/净值/盈亏固定 `GBP`，标的现价/均价/止损用本地货币（`USD`/`EUR`/`GBp`/`GBP`），英股须防 `GBp` 与 `GBP` 100 倍量级错位，严禁交给 LLM 口算。
4. **`/report` 基石与纯文本控字**：`07:30 UK` `/report` 是全天 `Single Source of Truth`，严禁降低其深度；Telegram 输出禁用 `#` 与 `**`，受字数硬上限（`/report<=1800`、`/pick<=1000`、`/us<=450|600`）。
5. **英文检索 → 中文输出解耦**：`src/agents/tools.py` 的联网检索指令与 `MarketToolkit.declarations()` 工具声明保持英文（对齐 RNS/SEC 一手信源），检索结果与终端决策保持紧凑简体中文。
6. **数据库延续**：线上 `data/advisor.db` 只做幂等迁移（`CREATE TABLE IF NOT EXISTS` / `_ensure_columns`），严禁删库；发布版本同步 `src/version.py` 与 `pyproject.toml`。

## 二、模块 ↔ 职责 ↔ 单测总索引

> 修改前先用 `rg -n "^(class |async def |def |    async def |    def )" <文件>` 定位行号再按 100~150 行分段读取，禁止整篇读取 >400 行大文件。

| 源码模块 (`src/`) | 核心职责 | 对应单测 (`tests/`) |
| :--- | :--- | :--- |
| `config.py`, `config.yaml`, `version.py` | Pydantic 配置、`.env`、热重载自选/选股池、版本号 | `test_llm_pricing_and_usage.py` |
| `engine/calendar_utils.py` | `US/UK/XETRA/EURONEXT` 交易日历、半日市、夏令时错位周 | `test_calendar_and_scheduler.py` |
| `storage/db.py` | SQLite `WAL` 异步仓、周期/快照/结算/行情与搜索缓存、`get_recent_context_packs` 门控迟滞只读衔接、`wallet_balances` 双钱包锚点、`quant_health_audits.ai_diagnosis`、`user_preference_notes` 与 `pending_order_snapshots` 挂单成交快照、`nav_history` 日频净值锚点（GBP 标量，**永不清理**）、`get_settled_outcome_series` 原始 Alpha 序列 | `test_db_storage.py`, `test_sidecar_memory_and_weekly.py`, `test_intraday_sentinel.py`, `test_nav_performance.py` |
| `data_providers/t212_client.py` | T212 双账户只读、限流退避、挂单量化（**按挂单限价**核 R:R）与元/分红归档、成交频率/冷却 0-Token 统计 | `test_t212_client.py` |
| `data_providers/market_data.py` | Yahoo+Finnhub+FRED 并发聚合、`2y` 缓存、盘前 15m 跳空、Yahoo v8 chart 兜底（仅 yfinance 抛异常时）、Finnhub Basic Financials（PB/EV-EBITDA/流动比率别名补值） | `test_market_data.py` |
| `data_providers/analytics.py` | 复权自愈、CYQ、周线/ATR/ADX、去 Beta 相关性、CVaR、**TWR 组合绩效（剔入金）／交易归因（分币种）／信号 t 值与衰减／HHI 集中度／波动 2 年分位** | `test_analytics_and_rules.py`, `test_nav_performance.py` |
| `engine/deployment_gate.py` | `v5.4.0` 纯确定性 CIO 组合门控：账本恒等式与挂单三分法、ISA 核心≥60%/卫星≤30%·≤8只闸门、Invest 在险槽位与以盈解锁、`gate_epoch_date` 历史包袱豁免、去汇率失真 `SPY` Regime（进 1d / 退 3d 防死锁）、A/B/C/D 触发器与 `DEFEND` 四步协议、PM 单向收紧钳位、`<=110` 字门控横幅与 2 年离线回放 | `test_report_pick_gate.py` |
| `engine/portfolio_rules.py` | 持仓规则、移动止盈/除息守卫、Disagreement、`RiskOverride`、双钱包摩擦+价差估算、R:R≥2:1、以损定仓、频率/冷却三态护栏（宽基 ETF 与老仓清理 **100% 零 Quota 豁免**）；`v5.3.4` 起 `/pick` 买入侧复算（`RR_BELOW_FLOOR`/`RR_UNVERIFIED`/`EX_DIV_WINDOW`）；`v5.4.0` 起跨双账户单票 `12%` 硬上限、`/pick` `ISA_SATELLITE_SATURATED` 拦截与 `<£1,200` 历史散仓顺风反弹温和归核（单次最多浮出 2 只） | `test_analytics_and_rules.py`, `test_report_pick_gate.py` |
| `engine/screener.py`, `engine/universe_seeds.py`, `refresh_pool.py` | 四维双轨量化选股（维度3 含 PB / EV-EBITDA 估值倍数）、150 只海选底池、VPVR/Stage2/VCP、Invest 钱包摩擦加权、候选行「财报/除息」双日历锚点 | `test_analytics_and_rules.py`, `test_universe_audition.py` |
| `agents/llm_client.py` | `GeminiClient`（Gemini+DeepSeek 混合路由/双向灾备）、Thinking 钳位、工具裁剪、Search 额度预警、硬超时快速灾备（`timeout_max_retries`，默认 0）+ `is_degraded_llm_output` 哨兵识别 | `test_llm_pricing_and_usage.py` |
| `agents/ds_sidecar.py` | DeepSeek 侧车唯一出口：`ds_helper_available` 凭证门控、`call_ds_json` 硬超时 + 静默降级、`extract_json_object`、`sanitize_plain_text` | `test_ds_pick_prefilter_and_auditor.py` |
| `agents/semantic_auditor.py` | `/report` & `/pick` 只读语义质检（幻觉价位/挂单计息/驳回外泄/红线冲突）、`build_fact_baseline`（含净可用占比与总现金占比双口径） | `test_ds_pick_prefilter_and_auditor.py`, `test_report_pick_gate.py` |
| `agents/prompts.py`, `agents/schemas.py` | 投资者画像（VWRP 基准化/ISA 核心-卫星/零 Quota 豁免/Invest 在险槽位以盈解锁）、`PickGateDecision`、JSON 自修复、二段首行 `gate_banner` 与 `[🛡️风控护栏修正]` 倒金字塔渲染 | `test_db_and_end_to_end.py`, `test_report_pick_gate.py` |
| `agents/tools.py` | 分域 Google 强搜（英文搜/中文出）与 `MarketToolkit` | `test_v504_ops_and_grounding.py` |
| `agents/report_agent.py` | `/report` 三步流水线（Step 2 总时长护栏 + 中性降级占位）、`AnalysisContextPack`、护栏后确定性 `deployment_gate` 求值与落库、`/us` 异动门控、语义质检接线、盘中异动快报 | `test_db_and_end_to_end.py`, `test_intraday_sentinel.py`, `test_report_step2_budget.py`, `test_report_pick_gate.py` |
| `agents/pick_agent.py` | `/pick` 宽漏斗 DeepSeek ID 初筛 -> Researcher -> Risk Critic -> Post-Screen 硬拦截落库（联动 `deployment_gate` 饱和分流、跨账户同票去重、`ROTATE` 置换源核验、连损恢复 `0.5R` 半仓算仓） + 偏好注入 | `test_db_and_end_to_end.py`, `test_ds_pick_prefilter_and_auditor.py`, `test_report_pick_gate.py` |
| `agents/chat_agent.py` | `/pos` 0-Token 查仓、Implicit Cache 前缀、标的切片、`/deep` 与带原话偏好注入（dynamic_tail） | `test_db_and_end_to_end.py` |
| `bot/handlers.py`, `bot/scheduler.py`, `bot/telegram_client.py` | 指令路由、`execution_lock` 互斥、`/perf` 0-Token 绩效看板（TWR/回撤/Alpha-Beta-IR/胜率/HHI/t 值）、自动日程、盘后 21:40 NAV 补采、`T+5/10/20` 结算、`/wallet`/`/usd` 双钱包校准、盘中哨兵、挂单成交回执（快照比对 + 历史成交交叉验证）、周六策略周报（含 VWRP 基准闭环）、偏好抽取 | `test_calendar_and_scheduler.py`, `test_db_and_end_to_end.py`, `test_sidecar_memory_and_weekly.py`, `test_intraday_sentinel.py`, `test_nav_performance.py` |
| `ops_notifier.py`, `healthcheck.py`, `main.py` | `ops-notify` 双态告警、心跳/只读 DB 探活、生命周期管理 | `test_v504_ops_and_grounding.py` |
| `redaction.py` | 日志/告警/诊断输出统一脱敏出口（`token=` / `api_key=` / `Bearer` / `Basic` → `***`）；`market_data` 全部 `last_error` 与 `/diag` 渲染均经此函数 | `test_v504_ops_and_grounding.py` |

## 三、开发验收命令

```bash
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.run_tests tests
```

- 单文件快测：`.\.venv\Scripts\python.exe -m unittest tests.<模块名>`（全部离线 Mock，无真实网络请求；行情桩须含 `Open/High/Low/Close/Volume`）。
- 出网与脱敏自查：`.\.venv\Scripts\python.exe scripts\audit_egress.py`（只读扫描；未知出网主机 / 日志求值密钥 / 依赖漂移 → 退出码 1，需人工复核后再提交）。
- 每次功能迭代完成后，将变更追加至 [`CHANGELOG.md`](./CHANGELOG.md) 顶部当前版本下。

## 四、部署与容器

- 容器名 `t212-ai-advisor`；发版统一走根目录 `scripts\deploy-n95.ps1`（见根 `AGENTS.md` 红线第 6 条：自研项目走 Git + 脚本发版，严禁在 Dockge 网页端直接修改本项目 `docker-compose.yml`）。
- 本项目无宿主机映射端口（纯出站：T212 只读 API + Telegram Long Polling），容器登记见根 `DEPLOYMENT_CHEATSHEET.md` 的「核心运维与现有容器入口总览」表。

## 五、Agent / 第三方模型接入边界（P1 安全红线）

> 允许用 DeepSeek / Gemini / Claude（含第三方中转站）辅助开发；但**出网目标与凭据边界不得由代码决定**。

### 5.1 出网白名单（代码强制，非口头约定）

`src/config.py` 的 `EnvSettings._guard_egress_endpoints` 在**配置加载期**校验，不通过直接抛 `ValueError`（进程起不来）：

| 配置项 | 默认允许主机 | 放宽方式（仅 `.env`） |
| :--- | :--- | :--- |
| `DEEPSEEK_BASE_URL` | `api.deepseek.com`（同表登记 `generativelanguage.googleapis.com` / `api.anthropic.com` 备用） | `LLM_EGRESS_ALLOWED_HOSTS=relay.example.com` |
| `T212_BASE_URL` | `live.trading212.com` / `demo.trading212.com` | `T212_EGRESS_ALLOWED_HOSTS=...` |

- 一律强制 `https`，`http://` 直接拒绝（防 API Key 与持仓明文裸奔）。
- **放宽出口只能改 `.env`，不得改 `_LLM_EGRESS_HOSTS` / `_T212_EGRESS_HOSTS` 常量**；任何「顺手把中转域名写进常量」的改动视为红灯。

### 5.2 红线文件（禁止擅自修改，需人工逐行 review）

`.gitignore`、`.dockerignore`、`src/config.py`（白名单/Envs）、`src/agents/llm_client.py`（egress/密钥）、`src/data_providers/t212_client.py`（鉴权段）、`src/ops_notifier.py`、`Dockerfile`、`docker-compose.yml`、`D:\HOMELAB\scripts\*.ps1`、`pyproject.toml` / `requirements.txt`（**新增依赖等同红灯**）。

> `sync-up.ps1` 与 `deploy-n95.ps1` 已内置**敏感路径提交/推送闸**（拦截 `.env`、`data/`、`*.db`、`logs/`、`_ref/`）。该闸**不得移除、放宽或绕过**；如需调整正则，必须两侧脚本同步并经人工确认。

### 5.3 交付前 diff 自查 5 条

1. 是否新增 `http(s)://` 出网字面量或新域名？
2. 是否新增日志/异常打印 `api_key` / `token` / `secret` / `cfg.env`？
3. 是否新增读取环境变量（尤其凭据类）？
4. 是否改动 `base_url` / 超时 / 重试 / 灾备链？
5. 是否改动风控护栏开关（`deployment_gate_mode` / `pick_rr_guard_mode` / `trade_cadence_guard_mode` / `intraday_fill_receipt_enabled` 等）？

任一为「是」→ 必须在该次变更说明中显式列出并等待人工确认。

### 5.4 会话分工

| 档 | 可用第三方模型 / 中转 | 范围 |
| :--- | :--- | :--- |
| 🟢 绿 | ✅ | 读代码、写/跑**离线**单测、文档、注释、非红线文件重构 |
| 🟡 黄 | ✅（需 5.3 自查） | SQL / 指标计算 / 文案 / 日志措辞 |
| 🔴 红 | ❌ 官方端点或人工逐行 review | 5.2 全部红线文件 + 任何新增依赖 |

### 5.5 Agent 执行禁区

- 禁止执行：`git push`、`D:\HOMELAB\scripts\deploy-n95.ps1`、`sync-up.ps1`、`ssh`、`docker exec ... env`、`pip install`、任何 `curl | bash`。
- 密钥只做**布尔探测**（`Test-Path` / `[bool]$env:X`），严禁 `Get-Content .env`，严禁在回复中回显密钥片段。
- 文件读取黑名单（原 `.clineignore` 语义，Cline 下线后由本节承载）：`.env`、`.env.*`、`data/`、`logs/`、`*.db`、`*.db-wal`、`*.db-shm`、`*.sqlite`、`*.sqlite3`、`*.log`、`_ref/` 一律不读。

