# Trading 212 × Gemini AI 私人量化投顾 (v5.1.1)

面向英国个人投资者（Trading 212 **Stocks ISA** 免税长线账户 + **Invest** 应税短线账户，每账户约 10 只持仓，双账户均**仅保留 `GBP` 英镑现金**、不使用 Pie、不开启分红自动再投资）的 7×24 全天候私人量化投顾与风控审计系统（只读 API、Telegram Long Polling 常驻、零公网端口暴露、Docker 容器化部署）。

---

## 1. 项目背景与运行环境

### 解决的核心痛点
1. **英镑现金结算 + 标的本地货币报价的摩擦核算**：人在英国（基准时区 `Europe/London`），双账户现金与钱包结算（`walletImpact`）统一为 `GBP` 英镑，而横跨英股（`GBp` 便士 / `GBP` 英镑）、美股（`USD`）与欧股（`EUR`）的个股/ETF 行情、均价与挂单触发价均按**标的所属市场的本地货币**报价；闲置英镑现金享 ~4.0% 年化无风险利息，英国个股买入收 `0.50%` 印花税（LSE ETF 免税），非英镑标的每次买卖固定发生 `0.15%` 即时换汇磨损（往返 `0.30%`）。所有操作建议必须扣费后跑赢 4% 现金锚。
2. **N95 本地 0-Token 量化重算与后置确定性护栏**：仓位权重、本币/汇率盈亏拆解、近 30 日出入金/派息流水归因、MA/Wilder RSI14/ATR14/ADX14/周线多空共振、CYQ 筹码分布（ETF 采用 60d VPVR）、基于建仓日（`Position.createdAt`）的波段高点移动止盈、除息窗口防假破位守卫、分市场去大盘 Beta 残差相关性、四维双轨非对称选股漏斗（Minervini Stage 2 / VCP / 3 日跨时区对数收益收缩协方差）、本地 0-Token 多空冲突矩阵（`Disagreement Matrix`）及后置风控护栏（`RiskOverride` 强制降级高亮 `[🛡️风控护栏修正]`）全部由本地 Python 确定性实算后再喂给 Gemini；对用户已有活动挂单执行零情面独立审计（`KEEP` / `MODIFY` / `CANCEL`）。
3. **以 `/report` 为绝对基石的全天风控对齐与跨指令快照复用**：每日 `07:30 UK` 的 `/report` 全景晨报是全系统的唯一事实底座，其生成的今日风控红线（`daily_guardrails`）、全文基线与预计算 `AnalysisContextPack` 快照落库 SQLite，供同周期 `/us` 盘前异动门控哨兵、`/pick` 选股雷达与日常单聊（0-Token 提问标的局部切片 + Gemini Implicit Cache 静态前缀对齐）零重复计算复用；盘后自动执行 `T+5` 真实 Alpha 结算与微反思记忆闭环。

### 算力与硬件部署环境
- **大模型与预算定位**：基于 Google AI Pro 订阅的 **Gemini API Paid Tier**（享有高并发配额、每月 5,000 次免费 Google Search Grounding 及 API 数据不用于模型训练的隐私保护）。理想月度消耗控制在 **\$10 USD 赠金以内（完美状态）**，月度上限 **\$20 USD**——在**绝对不缩减 `/report` 规模与深度**的前提下，通过优化日常单聊与消除重复工具调用控制成本。
- **服务端硬件与容器架构**：运行于英国本地常驻迷你主机 **Intel N95 + 16GB RAM + 512GB SSD**（`/opt/homelab/stacks/trading-python/`），采用 Docker Compose（`mem_limit: 2g`、`TZ=Europe/London`、纯 Python `healthcheck`），通过外部共享内网 `homelab-net` 接入 `ops-notify` 统一告警网关，**零公网端口暴露**。
- **居家与跨国出差双场景维护**：
  - **交互层**：采用 Telegram Long Polling 架构，无论在英国本地家中还是出国出差（任意时区/酒店网络），手机与笔记本均可零端口映射实时交互；系统内部时钟与日程严格锁死 `Europe/London (UK)` 与交易所本地时间（`ET`）。
  - **开发运维层**：日常在英国本地 Windows 台式机或出差随身 Windows 笔记本（通过 SSH / Tailscale 虚拟局域网）编写代码并运行离线单测，通过私有 Git 仓库同步至 N95 主机并执行 `docker compose up -d --build` 滚动重建。

### 演进脉络与工作区目录说明
- **`docs/PROJECT_MIGRATION_RULES.md`**：N95 Homelab 项目迁移与容器化强制规范（v1.2）。
- **`docs/T212_API_manual.json`**：Trading 212 Public API v0 官方 OpenAPI 3.0 规范文档（含所有接口 Schema 与限流规则）。
- **`docs/finnhub_API_manual.json`**：Finnhub API 官方 OpenAPI 规范文档（美股实时报价、财报日历、公司新闻与深度评级）。
- **`_ref/`（本地隔离参考区，不进 Git / Docker）**：早期 GAS 脚本与 7 套参考开源项目（`REF_stock-master`、`REF_stock-screener-main`、`REF_PyPortfolioOpt-main`、`REF_Riskfolio-Lib-master`、`REF_skfolio-main`、`REF_TradingAgents-main`、`REF_daily_stock_analysis-main`）。

> **开发者与 AI 交接手册**：模块拆解、数据库/模型契约、`IMP-01 ~ IMP-19` 拓扑验收记录见 [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)，详细版本与模块修改流水见 [CHANGELOG.md](CHANGELOG.md)。

---

## 2. 环境配置与 N95 容器化部署（Git + Docker Compose）

**前置条件**：
- 本地开发与离线单测：Python `>= 3.12`（容器基础镜像已锁定 `python:3.12.13-slim-bookworm`）。
- N95 宿主机环境：Docker + Docker Compose v2，且已创建外部共享网络 `homelab-net`（若首次初始化：`docker network create homelab-net`）。

**API 密钥与环境变量准备（`.env`，严禁提交至 Git）**：
- **Trading 212 API Key**（只读）：在 T212 生成 Key 时勾选 `Account Data`、`Portfolio`、`Orders - Read`、`History - Orders`（**勿勾选**下单执行权限与 Pie 权限）。
- **Google Gemini API Key**：Paid Tier Key，用于多智能体推理与 Google Search Grounding 联网检索。
- **Telegram Bot Token & Chat ID**：独立 Bot Token + 单用户 `chat_id` 白名单鉴权。
- **Finnhub / FRED API Key**（免费可选）：增强美股财报日历与美债/利率/失业率官方宏观序列。
- **Ops-Notify 运维网关（N95 必配）**：
  - `OPS_NOTIFY_URL=http://ops-notify:8000/alert`（通过 `homelab-net` 内网推送运维告警，不刷屏交易交互 Bot；未配置时自动降级为直接走 Telegram）。
  - `OPS_NOTIFY_TOKEN=`（若 `ops-notify` 开启鉴权则填写 Bearer Token，否则留空）。

### 本地开发与离线验证（Windows 开发端）

```bash
# 首次初始化 Python 3.12.13 独立虚拟环境（遵循 PROJECT_MIGRATION_RULES v1.2）
uv python install 3.12.13
uv venv --python 3.12.13
uv pip install -r requirements.txt

# 日常编译、全套纯离线单测（84 tests）与 Git 同步
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.run_tests tests
git add -A && git commit -m "<msg>" && git push
```

### N95 首次部署、旧端迁移与日常更新（Git 同步 + Docker Compose）

> [!CAUTION]
> 严禁在旧云主机与 N95 上同时运行同一 `TELEGRAM_BOT_TOKEN` 的 Long Polling 进程，否则会触发 `409 Conflict` 与定时任务重复推送。请严格遵循 `docs/PROJECT_MIGRATION_RULES.md` 第五节 SOP 执行迁移：先彻底停服旧端，再通过 `sqlite3 .backup` 生成一致热备库。

```bash
# 0. 旧云主机安全停服 + 生成一致热备库（迁移场景必做）
# 在旧端停服后执行：
sqlite3 /path/to/old/advisor.db ".backup clean_advisor.db"
# 将 clean_advisor.db 传至 N95 的 /opt/homelab/stacks/trading-python/data/advisor.db

# 1. 拉取最新代码至 N95 标准栈目录（首次部署请先 git clone）
cd /opt/homelab/stacks/trading-python
git pull --ff-only

# 2. 首次部署配置密钥与持久化目录（日常更新跳过此步）
mkdir -p data
cp -n .env.example .env && nano .env
chmod 600 .env

# 3. 确认共享内网存在后构建并启动容器（自动挂载 ./data:/app/data 与 ./config.yaml:ro）
docker network inspect homelab-net >/dev/null 2>&1 || docker network create homelab-net
docker compose up -d --build

# 4. 巡检容器状态与探活（确认 STATUS 为 healthy 且无报错）
docker compose ps
docker compose exec advisor python -m src.healthcheck
docker compose logs -f --tail=100

# 5. 双验收：ops-notify 端到端测试告警 + 宿主机备份验证（首次迁移必做）
docker compose exec advisor python -c "from src.config import load_config; from src.ops_notifier import send_ops_notify_sync; c=load_config(); print(send_ops_notify_sync(c.env.ops_notify_url, severity='INFO', alert_key='n95_migration_test', title='N95 迁移验收测试', message='trading-python 已接入 homelab-net ops-notify', cooldown_sec=0, token=c.env.ops_notify_token))"
# 随后在 N95 执行一次宿主机备份脚本（例如 /opt/homelab/scripts/backup_daily.sh）确认 ./data/ 与 .env 加密归档成功，
# 并在 Telegram 触发一次 /health 校验全链路状态；旧云主机保留关机冷备观察 7 天后再释放。
```

---

## 3. Telegram 指令与自动日程

**自动推送日程（锁死伦敦时间 `Europe/London`，自动适配英美冬夏令时错位周、半日市与法定假日）：**
- **工作日 `07:30 UK`（支持 `07:30~09:30 UK` 断网/重启自动补发）**：自动推送 `/report` 全景晨报，并切换新一天的交易周期与风控红线。
- **美股开盘前 15 分钟（常规 `14:15 UK` / 错位周 `13:15 UK`）**：自动推送 `/us` 盘前增量哨兵（美股法定休市日静音提醒；本地异动门控平稳日仅发极简平安报）。
- **周五美股收盘后 30 分钟（常规 `21:30 UK` / 美股半日市自动提前至 `18:30 UK`）**：自动推送 `/pick` 周末选股雷达。
- **工作日 `22:00~23:30 UK`（盘后静默任务）**：自动执行 `T+5`（按英/欧/美各市场交易日历与收盘判定）历史推荐真实收益/Alpha 结算、微反思记忆写回及 90 天旧库记录清理。

**手动指令（左侧 `/` 菜单 + `/help`，自由单聊支持 `/deep` 开头升级深度推理）：**
- `/pos` — 瞬时拉取双账户实时仓位、净值、挂单与近 7 天成交（0 Token 秒出）
- `/report` — **核心全景晨报**（宏观 + 双账户 + 30日复盘 + 全市场分域 Google 联网扫描 + 逐仓/挂单审计 + 风控护栏，`<=1800` 字）
- `/pick [方向]` — 四维量化初筛 + 联网催化扫描 + 投研员/风控官双审 + 后置硬拦截（`<=1000` 字）
- `/us` — 美股盘前/盘中/收盘/休市自适应增量简报（平稳日 `<=450` 字 / 异动日 `<=600` 字）
- `/stats` — 查看历史选股真实 `T+5` 胜率、超额 Alpha、近期复盘教训与待结算浮盈
- `/health` — 全链路实时体检（T212 降级/行情/Finnhub/FRED/Gemini/DB/未送达告警与 `ops-notify` 网关状态）
- `/usage` — 查看 Token 消耗、缓存命中率、思考占比与本月 Google Search 免费额度进度
- `/clear` / `/cancel` — 清空今日单聊记忆 / 立即中止正在运行的耗时分析
