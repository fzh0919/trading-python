# Trading 212 × Gemini AI 私人量化投顾 (v5.5.10)

面向英国个人投资者（Trading 212 **Stocks ISA** 纯 `GBP` 核心-卫星仓 + **Invest** `GBP`/`USD` 双钱包右侧机动仓，标的按本地货币报价）的 7×24 全天候私人量化投顾与风控审计系统（只读 API、Telegram Long Polling、零公网端口暴露、Docker 容器化部署于本地 Intel N95）。

---

## 1. 核心能力与技术特点

* **双账户币种隔离与对称摩擦核算**：严格隔离 `GBP` 总账口径、`Invest` 本地 `GBP+USD` 双钱包锚点（`/wallet` / `/usd` 对账）与 `USD`/`EUR`/`GBp`/`GBP` 标的本地报价；将 `4.0%` GBP / `3.4%` USD 闲置现金利息（含未成交挂单占用资金）、跨币种 `0.15%` 单边换汇磨损（Invest 同币种复用免换汇）、买卖价差及 `0.50%` 英国个股印花税纳入全部 R:R 与摩擦评估。
* **本地 Python 0-Token 量化底座与 CIO 组合门控（`v5.5.9` / `v5.5.10`）**：在请求大模型前，本地完成 `2y` SQLite 日线缓存、MA/RSI14/ATR14/ADX14/周线共振、CYQ 筹码分布、移动止盈/除息防假破位、去大盘 Beta 相关性、PSD 收缩协方差前 3 波动源与 95% 单日 CVaR、TWR 绩效归因与出入金流水自愈（`/perf`）、四维双轨选股漏斗（右侧动量双锚拆轨）、多空冲突矩阵（`Disagreement Matrix`）、后置确定性风控护栏（`RiskOverride`）与 **CIO 组合门控（`deployment_gate`）**——支持四态市场状态机（`RISK_ON / NEUTRAL / RECOVERY / RISK_OFF`）与双层市场宽度、A1 回踩 + A2 突破双通道扫描、Invest 派息宽基蓄水池（`VWRL.L`/`ISF.L`/`VERX.L`/`VEUR.L`）与清宽基换股配对、硬拦截 PM 否决/冷静期/单日上限/满槽/ISA 饱和与跨账户单票 12%；ISA 核心宽基 ≥60% 作为 DCA 目标，核心范围含全球/标普 + **FTSE100/欧洲分散器**。
* **以 `/report` 为基石的跨指令复用与盘后闭环**：每日 `07:30 UK` 生成 `/report` 全景晨报并落库 `AnalysisContextPack`（含当日 `deployment_gate`），供同周期 `/us`、`/pick` 与日常单聊 0-Token 复用；`/pick` 对缺失当日基线一律拒绝；盘后 `21:40 UK` 自动采集 NAV 净值锚点并做近 30 天外部出入金流水校准，`22:00~23:30 UK` 自动结算 `T+5`/`T+10`/`T+20` 方向感知 Alpha、`MFE/MAE` 与 2×2 过程象限（未触价 `BUY/ADD` 记 `NOT_FILLED` 并排除胜率），同步生成 0-Token 盘后执行账单（本地股价 vs FX 拆解、Plan vs Actual 纪律扫描、异动 Flash Lite 催化检索，可通过 `/review` 查看）并清理 90 天旧缓存；周末闲时离线体检选股池与阈值。
* **DeepSeek V4.1 Flash 只读侧车（防「传话筒效应」）**：`/pick` 宽漏斗（每轨 15 只）由 DS 仅做**候选 ID 筛选**、`/report` 与 `/pick` 渲染后做**只读语义质检**（只告警不拦截）、盘中哨兵做异动与挂单成交回执推送、周六输出 `<=600` 字**独立策略诊断周报**、每日结算末尾可按开关抽取**带用户原话锚点 + TTL** 的长期偏好（`user_preference_memory_enabled` 默认关闭）；DS 故障或缺 key 时一律确定性静默降级（0 调用）。

---

## 2. 项目代码结构

```text
trading-python/
├── AGENTS.md                       # AI 工作区热启动入口（核心不变量 + 模块/单测索引 + P1 安全红线）
├── DEVELOPER_GUIDE.md              # 业务规则、Agent 模型路由与 SQLite 核心表速查
├── CHANGELOG.md                    # 当前活跃版本变更日志（历史归档见 docs/CHANGELOG_ARCHIVE.md）
├── config.yaml / .env.example      # 模型路由、日程、门控/风控阈值与密钥模板
├── Dockerfile / docker-compose.yml # N95 容器化编排（python:3.12.13-slim-bookworm、homelab-net）
├── docs/
│   ├── MODULE_CONTRACTS.md         # 各模块函数签名字典与 _ref 算法映射（冷参考）
│   ├── CHANGELOG_ARCHIVE.md        # IMP-01~22 详录与早期版本历史（冷归档）
│   ├── plans/                      # 各批次架构设计与实施蓝本归档（冷文档）
│   └── *_API_manual.json           # Trading 212 / Finnhub 官方 OpenAPI 规范
├── src/
│   ├── main.py / config.py / version.py / ops_notifier.py / healthcheck.py / redaction.py / refresh_pool.py
│   ├── data_providers/
│   │   ├── t212_client/            # models, wallets_and_tickers, http_mixin, client, context_formatting
│   │   ├── market_data/            # models_and_parsing, external_clients, yahoo_mixin, service
│   │   └── analytics/              # technical, exposure, correlation, performance, postmarket_review
│   ├── engine/
│   │   ├── calendar_utils.py / universe_seeds.py
│   │   ├── deployment_gate/        # common, core_policy, ledgers, regime, triggers, banner
│   │   ├── portfolio_rules/        # models_and_friction, dual_philosophy, sizing, holding_eval, risk_override
│   │   └── screener/               # patterns, scoring, runner, audit, audition
│   ├── storage/db/                 # core, schema, portfolio_repo, recommendations_repo, ops_and_chat_repo, market_cache_repo, memory_and_wallet_repo
│   ├── agents/
│   │   ├── llm_client.py / ds_sidecar.py / semantic_auditor.py / prompts.py / schemas.py / tools.py / pick_agent.py / chat_agent.py
│   │   └── report_agent/           # fact_sheet, debate, core_deployment, morning_report, sentinels
│   └── bot/
│       ├── telegram_client.py / handlers.py
│       └── scheduler/              # core, settlement_math, fill_receipts, settlement_mixin, weekly_audit_mixin, intraday_mixin, postmarket_mixin
└── tests/                          # 纯离线单元测试套件（run_tests.py + 26 组模块测试，零真实外网请求）
```

* 本地离线验证：`.\.venv\Scripts\python.exe -m compileall -q src tests`、`.\.venv\Scripts\python.exe -m tests.run_tests tests`、`.\.venv\Scripts\python.exe scripts\audit_egress.py`

---

## 3. Telegram 指令与自动日程（`Europe/London`）

**自动日程（自动适配英美冬夏令时错位周、半日市与法定假日）：**
* **任务前 `-10m`**：静默预热持仓与候选池 `2y` 日线、宏观快照，每日同步 1 次 T212 元数据与分红归档。
* **工作日 `07:30 UK`**（支持 `07:30~09:30` 断网补发）：推送 `/report` 全景晨报并切换新交易周期。
* **工作日 `08:00~21:00 UK`**：盘中 0-Token 哨兵（每 30 分钟扫描一次异动与限价/止损挂单逼近，并交叉核对历史成交推送挂单成交回执）。
* **美股开盘前 `15m`**（常规 `14:15` / 错位周 `13:15 UK`）：推送 `/us` 盘前增量哨兵（结合 15m 盘前跳空检测）。
* **周五美股收盘后 `30m`**：`enable_weekly_radar` 默认 `false`（`v5.5.10` 起停用周五自动 `/pick`，手动 `/pick` 不受影响；显式开启时仍按半日市动态触发）。
* **工作日 `21:40` & `22:00~23:30 UK`**：盘后采集 NAV 净值锚点并校准近 30 天外部出入金流水；结算 `T+5`/`T+10`/`T+20` 方向感知 Alpha、`MFE/MAE` 与 2×2 过程象限，生成 0-Token 盘后执行账单（仅纪律偏离或重大公告异动时静默提醒）并清理 90 天旧缓存。
* **周六 `10:00~22:00 UK`**：静默海选多市场 150 只精锐个股底池 + 离线阈值体检 + DeepSeek 策略诊断周报（`<=600` 字）与双钱包对账提醒。

**手动指令（自由单聊支持 `/deep` 开头升级深度推理）：**
* `/pos` / `/wallet` / `/usd <金额>` / `/review`：0-Token 实时仓位与挂单、Invest `GBP+USD` 双钱包对账校准、最近盘后执行账单（股价/FX 拆解与 2×2 复盘）
* `/perf`：0-Token 组合绩效看板（TWR 时间加权收益、最大回撤、Alpha/Beta/IR、分币种胜率、HHI 集中度、信号 t 值）
* `/report`：核心全景晨报（宏观 + CIO 组合门控 + 双账户 + 全市场联网扫描 + 逐仓/挂单审计 + 未持仓核心宽基 `core_deployment`，`<=2800` 字；PM 结构化失败时返回 `⚠️` 且不写入当日基线）
* `/pick [方向]`：四维量化初筛 + 联网催化 + 投研/风控双审 + 组合门控硬拦截（PM 否决/账户通道/ISA 饱和/单票 12%/跨账户去重/蓄水池配对）+ R:R 硬拦截（`pick_rr_guard_mode=hard`；`<=1000` 字）
* `/us`：美股盘前/盘中/收盘增量简报（平稳 `<=450` / 异动 `<=600` 字）
* `/stats`：历史推荐 `T+5/10/20` 方向感知胜率与 Alpha 轨迹、`MFE/MAE`、2×2 过程分布、未成交样本、复盘教训与周末体检看板
* `/refresh_pool` / `/health` / `/usage` / `/clear` / `/cancel` / `/help`：手动刷新海选池 / 全链路健康体检 / Token 与 Search 额度 / 清空单聊记忆 / 中止分析 / 指令说明
