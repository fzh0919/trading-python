# Trading 212 × Gemini AI 私人量化投顾 (v5.0.5)

面向英国个人投资者（Trading 212 **Stocks ISA** 免税长线账户 + **Invest** 应税短线账户，每账户约 10 只持仓，双账户均**仅保留 `GBP` 英镑现金**、不使用 Pie、不开启分红自动再投资）的 7×24 全天候私人量化投顾与风控审计系统（只读 API、Telegram Long Polling 常驻、零公网端口暴露、Docker 容器化部署）。

---

## 1. 项目背景与运行环境

### 解决的核心痛点
1. **英镑现金结算 + 标的本地货币报价的摩擦核算**：人在英国（基准时区 `Europe/London`），双账户现金与钱包结算（`walletImpact`）统一为 `GBP` 英镑，而横跨英股（`GBp` 便士 / `GBP` 英镑）、美股（`USD`）与欧股（`EUR`）的个股/ETF 行情、均价与挂单触发价均按**标的所属市场的本地货币**报价；闲置英镑现金享 ~4.0% 年化无风险利息，英国个股买入收 `0.50%` 印花税（LSE ETF 免税），非英镑标的每次买卖固定发生 `0.15%` 即时换汇磨损（往返 `0.30%`）。所有操作建议必须扣费后跑赢 4% 现金锚。
2. **杜绝 AI 口算与顺从迎合**：仓位权重、本币/汇率盈亏拆解、MA20/50/200、Wilder RSI14、去大盘 Beta 残差相关性均由 Python 确定性实算后再喂给 Gemini；对用户已有活动挂单执行零情面独立审计（`KEEP` / `MODIFY` / `CANCEL`）。
3. **以 `/report` 为绝对基石的全天风控对齐**：每日 `07:30 UK` 的 `/report` 全景晨报是全系统的核心参照物，其生成的今日风控红线（`daily_guardrails`）与全文基线落库 SQLite，直接作为同周期后续 `/us` 盘前哨兵、`/pick` 选股与日常聊天框的决策基准。

### 算力与硬件部署环境
- **大模型与预算定位**：基于 Google AI Pro 订阅的 **Gemini API Paid Tier**（享有高并发配额、每月 5,000 次免费 Google Search Grounding 及 API 数据不用于模型训练的隐私保护）。理想月度消耗控制在 **\$10 USD 赠金以内（完美状态）**，月度上限 **\$20 USD**——在**绝对不缩减 `/report` 规模与深度**的前提下，通过优化日常单聊与消除重复工具调用控制成本。
- **服务端硬件与容器架构**：运行于英国本地常驻迷你主机 **Intel N95 + 16GB RAM + 512GB SSD**（`/opt/homelab/stacks/trading-python/`），采用 Docker Compose（`mem_limit: 2g`、`TZ=Europe/London`、纯 Python `healthcheck`），通过外部共享内网 `homelab-net` 接入 `ops-notify` 统一告警网关，**零公网端口暴露**。
- **居家与跨国出差双场景维护**：
  - **交互层**：采用 Telegram Long Polling 架构，无论在英国本地家中还是出国出差（任意时区/酒店网络），手机与笔记本均可零端口映射实时交互；系统内部时钟与日程严格锁死 `Europe/London (UK)` 与交易所本地时间（`ET`）。
  - **开发运维层**：日常在英国本地 Windows 台式机或出差随身 Windows 笔记本（通过 SSH / Tailscale 虚拟局域网）编写代码并运行离线单测，通过私有 Git 仓库同步至 N95 主机并执行 `docker compose up -d --build` 滚动重建。

### 演进脉络与工作区目录说明
- **`docs/PROJECT_MIGRATION_RULES.md`**：N95 Homelab 项目迁移与容器化强制规范（v1.1）。
- **`docs/T212_API_manual.json`**：Trading 212 Public API v0 官方 OpenAPI 3.0 规范文档（含所有接口 Schema 与限流规则）。
- **`_ref/`（本地隔离参考区，不进 Git / Docker）**：早期 GAS 脚本与三大参考开源项目（`REF_TradingAgents-main`、`REF_daily_stock_analysis-main`、`REF_stock-master`）。

> **开发者与 AI 交接手册**：模块拆解、数据库/模型契约、待改进追踪表（`IMP-01 ~ IMP-19`）与版本日志见 [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)。

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
python -m compileall -q src tests
python -m tests.run_tests tests
git add -A && git commit -m "<msg>" && git push
```

### N95 首次部署与日常更新（Git 同步 + Docker Compose）

> [!CAUTION]
> 严禁在旧云主机与 N95 上同时运行同一 `TELEGRAM_BOT_TOKEN` 的 Long Polling 进程，否则会触发 `409 Conflict` 与定时任务重复推送。迁移前务必先彻底停服旧端，并用 `sqlite3 advisor.db ".backup clean.db"` 生成热备库后再拷入 `./data/advisor.db`。

```bash
# 1. 拉取最新代码至 N95 标准栈目录
cd /opt/homelab/stacks/trading-python
git pull --ff-only

# 2. 首次部署配置密钥与持久化目录（日常更新跳过此步）
mkdir -p data
cp -n .env.example .env && nano .env
chmod 600 .env

# 3. 确认共享内网存在后构建并启动容器（自动挂载 ./data:/app/data 与 ./config.yaml:ro）
docker network inspect homelab-net >/dev/null 2>&1 || docker network create homelab-net
docker compose up -d --build

# 4. 巡检容器状态（确认 health 为 healthy 且无报错）
docker compose ps
docker compose logs -f --tail=100
```

---

## 3. Telegram 指令与自动日程

**自动推送日程（锁死伦敦时间 `Europe/London`，自动适配英美冬夏令时错位周与法定假日）：**
- **工作日 `07:30 UK`**：自动推送 `/report` 全景晨报，并切换新一天的交易周期与风控红线。
- **美股开盘前 15 分钟（常规 `14:15 UK` / 错位周 `13:15 UK`）**：自动推送 `/us` 盘前增量哨兵（美股法定休市日静音提醒）。
- **周五美股收盘后 30 分钟（常规 `21:30 UK`）**：自动推送 `/pick` 周末选股雷达。

**手动指令（左侧 `/` 菜单 + `/help`）：**
- `/pos` — 瞬时拉取双账户实时仓位、净值、挂单与近 7 天成交（0 Token 秒出）
- `/report` — **核心全景晨报**（宏观 + 双账户 + 30日复盘 + 全市场分域 Google 联网扫描 + 逐仓/挂单审计，`<=1500` 字）
- `/pick [方向]` — 量化初筛 + 联网催化扫描 + 投研员/风控官双审（`<=1000` 字）
- `/us` — 美股盘前/盘中/收盘/休市自适应增量简报（`<=450` 字）
- `/stats` — 查看最近保存的历史选股与调仓记录
- `/health` — 全链路实时体检（T212/行情/Finnhub/FRED/Gemini/DB 状态与近期告警）
- `/usage` — 查看 Token 消耗、缓存命中率与本月 Google Search 免费额度进度
- `/clear` / `/cancel` — 清空今日单聊记忆 / 立即中止正在运行的耗时分析
