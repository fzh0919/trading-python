# Trading 212 × Gemini AI 私人量化投顾 (v5.2.0)

面向英国个人投资者（Trading 212 **Stocks ISA** + **Invest** 双账户，仅保留 `GBP` 现金、标的按本地货币报价）的 7×24 全天候私人量化投顾与风控审计系统（只读 API、Telegram Long Polling、零公网端口暴露、Docker 容器化部署于本地 Intel N95）。

---

## 1. 核心能力与技术特点

* **双币种与真实摩擦核算**：严格隔离 `GBP` 钱包结算与 `USD`/`EUR`/`GBp`/`GBP` 标的本地报价；将 `~4.0%` 闲置现金利息（含未成交挂单占用资金）、`0.15%` 单边换汇磨损及 `0.50%` 英国个股印花税纳入全部收益评估。
* **本地 Python 0-Token 量化底座**：在请求大模型前，本地完成 `2y` SQLite 日线缓存、MA/RSI14/ATR14/ADX14/周线共振、CYQ 筹码分布、移动止盈/除息防假破位、去大盘 Beta 相关性、PSD 收缩协方差前 3 波动风险贡献源与 95% 单日 CVaR、挂单触价率精算、Finnhub 深度情报、FRED 利率/信用利差、四维双轨选股漏斗（Minervini Stage 2 / VCP / VPVR / 150 只多市场海选），以及多空冲突矩阵（`Disagreement Matrix`）和后置确定性风控护栏（`RiskOverride`）。
* **以 `/report` 为基石的跨指令复用与复盘闭环**：每日 `07:30 UK` 生成 `/report` 全景晨报与风控红线并落库 `AnalysisContextPack`，供同周期 `/us` 盘前哨兵（含 `yfinance 15m` 盘前跳空检测）、`/pick` 雷达与日常单聊 0-Token 复用；盘后自动结算 `T+5`/`T+10`/`T+20` Alpha 与 `MFE/MAE` 极值并生成微反思记忆，周末闲时离线体检选股池与阈值。

---

## 2. 项目代码结构

```text
trading-python/
├── AGENTS.md                       # AI 工作区热启动入口（核心不变量 + 模块/单测索引）
├── DEVELOPER_GUIDE.md              # 业务规则、Agent 模型路由与 SQLite 核心表速查
├── CHANGELOG.md                    # 当前活跃版本变更日志（历史归档见 docs/CHANGELOG_ARCHIVE.md）
├── config.yaml / .env.example      # 模型路由、日程、风控阈值与密钥模板
├── Dockerfile / docker-compose.yml # N95 容器化编排（python:3.12.13-slim-bookworm、homelab-net）
├── docs/
│   ├── PROJECT_MIGRATION_RULES.md  # N95 容器化迁移、SQLite 备份与 ops-notify 规范
│   ├── MODULE_CONTRACTS.md         # 各模块函数签名字典与 _ref 算法映射（冷参考）
│   ├── CHANGELOG_ARCHIVE.md        # IMP-01~22 详录与早期版本历史（冷归档）
│   └── *_API_manual.json           # Trading 212 / Finnhub 官方 OpenAPI 规范
├── src/
│   ├── main.py / config.py / version.py / ops_notifier.py / healthcheck.py / refresh_pool.py
│   ├── data_providers/             # t212_client.py, market_data.py, analytics.py
│   ├── engine/                     # calendar_utils.py, portfolio_rules.py, screener.py, universe_seeds.py
│   ├── storage/                    # db.py (SQLite WAL 异步持久化)
│   ├── agents/                     # llm_client.py, prompts.py, schemas.py, tools.py, report_agent.py, pick_agent.py, chat_agent.py
│   └── bot/                        # telegram_client.py, handlers.py, scheduler.py
└── tests/                          # 纯离线单元测试套件（run_tests.py + 9 组模块测试，零真实外网请求）
```

* 本地离线验证：`.\.venv\Scripts\python.exe -m compileall -q src tests && .\.venv\Scripts\python.exe -m tests.run_tests tests`
* 部署与容器迁移规范详见 [`docs/PROJECT_MIGRATION_RULES.md`](./docs/PROJECT_MIGRATION_RULES.md)。

---

## 3. Telegram 指令与自动日程（`Europe/London`）

**自动日程（自动适配英美冬夏令时错位周、半日市与法定假日）：**
* **任务前 `-10m`**：静默预热持仓与候选池 `2y` 日线、宏观快照，每日同步 1 次 T212 元数据与分红归档。
* **工作日 `07:30 UK`**（支持 `07:30~09:30` 断网补发）：推送 `/report` 全景晨报并切换新交易周期。
* **美股开盘前 `15m`**（常规 `14:15` / 错位周 `13:15 UK`）：推送 `/us` 盘前增量哨兵（结合 15m 盘前跳空检测）。
* **周五美股收盘后 `30m`**（常规 `21:30` / 半日市 `18:30 UK`）：推送 `/pick` 周末选股雷达。
* **工作日 `22:00~23:30 UK`**：盘后静默结算 `T+5`/`T+10`/`T+20` Alpha 与 `MFE/MAE`，写回微反思并清理 90 天旧缓存。
* **周六 `10:00~22:00 UK`**：静默海选多市场 150 只精锐个股底池 + 离线阈值体检。

**手动指令（自由单聊支持 `/deep` 开头升级深度推理）：**
* `/pos`：0-Token 实时仓位、净值、挂单与近 7 天成交
* `/report`：核心全景晨报（宏观 + 双账户 + 全市场联网扫描 + 逐仓/挂单审计，`<=1800` 字）
* `/pick [方向]`：四维量化初筛 + 联网催化 + 投研/风控双审 + 后置硬拦截（`<=1000` 字）
* `/us`：美股盘前/盘中/收盘增量简报（平稳 `<=450` / 异动 `<=600` 字）
* `/stats`：历史推荐 `T+5/10/20` 胜率与 Alpha 轨迹、`MFE/MAE`、复盘教训与周末体检看板
* `/refresh_pool` / `/health` / `/usage` / `/clear` / `/cancel`：手动刷新海选池 / 全链路健康体检 / Token 与 Search 额度 / 清空单聊记忆 / 中止分析
