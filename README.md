# Trading 212 × Gemini AI 私人量化投顾 (v5.4.0)

面向英国个人投资者（Trading 212 **Stocks ISA** 纯 `GBP` 核心-卫星仓 + **Invest** `GBP`/`USD` 双钱包右侧机动仓，标的按本地货币报价）的 7×24 全天候私人量化投顾与风控审计系统（只读 API、Telegram Long Polling、零公网端口暴露、Docker 容器化部署于本地 Intel N95）。

---

## 1. 核心能力与技术特点

* **双账户币种隔离与对称摩擦核算**：严格隔离 `GBP` 总账口径、`Invest` 本地 `GBP+USD` 双钱包锚点（`/wallet` / `/usd` 对账）与 `USD`/`EUR`/`GBp`/`GBP` 标的本地报价；将 `4.0%` GBP / `3.4%` USD 闲置现金利息（含未成交挂单占用资金）、跨币种 `0.15%` 单边换汇磨损（Invest 同币种复用免换汇）、买卖价差及 `0.50%` 英国个股印花税纳入全部 R:R 与摩擦评估。
* **本地 Python 0-Token 量化底座与 CIO 组合门控（`v5.4.0`）**：在请求大模型前，本地完成 `2y` SQLite 日线缓存、MA/RSI14/ATR14/ADX14/周线共振、CYQ 筹码分布、移动止盈/除息防假破位、去大盘 Beta 相关性、PSD 收缩协方差前 3 波动源与 95% 单日 CVaR、TWR 绩效归因（`/perf`）、四维双轨选股漏斗，以及多空冲突矩阵（`Disagreement Matrix`）、后置确定性风控护栏（`RiskOverride`）与 **CIO 组合门控（`deployment_gate`）**——硬拦截 PM 否决、DEFEND/冷静期/单日上限/满槽、ISA 退潮期个股与未达阶梯宽基、ISA 个股卫星 ≤30%·≤8只与跨双账户单票 12%；ISA 核心宽基 ≥60% 作为 DCA 目标（现金超上沿才分批归拢，不强制卖出卫星）；Invest 在险槽位以盈解锁，美元原币种 `SPY` 防死锁市场状态机。
* **以 `/report` 为基石的跨指令复用与复盘闭环**：每日 `07:30 UK` 生成 `/report` 全景晨报与风控红线并落库 `AnalysisContextPack`（含当日 `deployment_gate`），供同周期 `/us` 盘前哨兵、`/pick` 雷达与日常单聊 0-Token 复用；`/pick` 对缺失当日基线/ContextPack 一律拒绝，绝不拿上一交易日旧红线冒充今日裁决；盘后 `21:40 UK` 自动补采 NAV 净值锚点（结算时再兜底补写一次），`22:00~23:30 UK` 自动结算 `T+5`/`T+10`/`T+20` Alpha 与 `MFE/MAE` 极值并生成微反思记忆——**BUY/ADD 记多头收益，TRIM/STOP/SWAP 记避损收益；未触价的 BUY/ADD 标记 `NOT_FILLED` 并排除出胜率**，`/report` 的 ADD/TRIM/STOP 也进入同一 T+N ledger；周末闲时离线体检选股池与阈值。
* **DeepSeek V4.1 Flash 只读侧车（防「传话筒效应」）**：`/pick` 宽漏斗（每轨 15 只）由 DS 仅做**候选 ID 筛选**、`/report` 与 `/pick` 渲染后做**只读语义质检**（幻觉价位/挂单计息/内部驳回外泄/红线冲突，只告警不拦截）、盘中哨兵做异动与挂单成交回执推送、周六输出 `<=450` 字**独立策略诊断周报**、每日结算末尾从原始对话抽取**带用户原话锚点 + TTL** 的长期偏好；DS 故障或缺 key 时一律确定性静默降级（0 调用）。

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
│   ├── data_providers/             # t212_client.py, market_data.py, analytics.py
│   ├── engine/                     # calendar_utils.py, deployment_gate.py, portfolio_rules.py, screener.py, universe_seeds.py
│   ├── storage/                    # db.py (SQLite WAL 异步持久化)
│   ├── agents/                     # llm_client.py, ds_sidecar.py, semantic_auditor.py, prompts.py, schemas.py, tools.py, report_agent.py, pick_agent.py, chat_agent.py
│   └── bot/                        # telegram_client.py, handlers.py, scheduler.py
└── tests/                          # 纯离线单元测试套件（run_tests.py + 14 组模块测试，零真实外网请求）
```

* 本地离线验证：`.\.venv\Scripts\python.exe -m compileall -q src tests`、`.\.venv\Scripts\python.exe -m tests.run_tests tests`、`.\.venv\Scripts\python.exe scripts\audit_egress.py`

---

## 3. Telegram 指令与自动日程（`Europe/London`）

**自动日程（自动适配英美冬夏令时错位周、半日市与法定假日）：**
* **任务前 `-10m`**：静默预热持仓与候选池 `2y` 日线、宏观快照，每日同步 1 次 T212 元数据与分红归档。
* **工作日 `07:30 UK`**（支持 `07:30~09:30` 断网补发）：推送 `/report` 全景晨报并切换新交易周期。
* **美股开盘前 `15m`**（常规 `14:15` / 错位周 `13:15 UK`）：推送 `/us` 盘前增量哨兵（结合 15m 盘前跳空检测）。
* **周五美股收盘后 `30m`**（常规 `21:30` / 半日市 `18:30 UK`）：推送 `/pick` 周末选股雷达。
* **工作日 `22:00~23:30 UK`**：盘后静默结算 `T+5`/`T+10`/`T+20` 方向感知 Alpha 与 `MFE/MAE`（`TRIM/STOP/SWAP` 记避损；未触价 `NOT_FILLED` 排除胜率），写回微反思并清理 90 天旧缓存。
* **周六 `10:00~22:00 UK`**：静默海选多市场 150 只精锐个股底池 + 离线阈值体检。

**手动指令（自由单聊支持 `/deep` 开头升级深度推理）：**
* `/pos` / `/wallet` / `/usd <金额>`：0-Token 实时仓位、挂单与 Invest `GBP+USD` 双钱包对账校准
* `/perf`：0-Token 组合绩效看板（TWR 时间加权收益、最大回撤、Alpha/Beta/IR、分币种胜率、HHI 集中度、信号 t 值）
* `/report`：核心全景晨报（宏观 + CIO 组合门控 + 双账户 + 全市场联网扫描 + 逐仓/挂单审计，`<=1800` 字；PM 结构化失败时返回 `⚠️` 且不写入当日基线）
* `/pick [方向]`：四维量化初筛 + 联网催化 + 投研/风控双审 + 组合门控硬拦截（PM 否决/账户通道/ISA 饱和/单票 12%/跨账户去重）+ R:R 可配置硬拦截（`pick_rr_guard_mode=hard` 时生效，默认 advisory；`<=1000` 字）
* `/us`：美股盘前/盘中/收盘增量简报（平稳 `<=450` / 异动 `<=600` 字）
* `/stats`：历史推荐 `T+5/10/20` 方向感知胜率与 Alpha 轨迹、`MFE/MAE`、未成交样本、复盘教训与周末体检看板
* `/refresh_pool` / `/health` / `/usage` / `/clear` / `/cancel`：手动刷新海选池 / 全链路健康体检 / Token 与 Search 额度 / 清空单聊记忆 / 中止分析
