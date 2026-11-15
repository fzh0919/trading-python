# N95 Homelab 项目迁移与容器化引入规范（PROJECT_MIGRATION_RULES v1.1）

> **使用方式**：每当需要将云上已有项目或新项目接入 N95 Homelab 时，必须向 AI 助手提供本规范文档。**任何违反以下强制条款的方案（如宿主机直接 `pip install`、裸跑脚本、开放公网端口、明文提交密钥、多项目共用 Long Polling Token）一律禁止执行。**

---

## 一、开发与部署分层铁律

1. **代码在本地开发，运行在 N95 容器**：
   - 代码开发与离线单测在 Windows 本地 `<HOMELAB_ROOT>\projects\<project_name>\`（台式机 `D:\HOMELAB\...` / 笔记本 `F:\HOMELAB\...`）完成，通过私有 Git 仓库同步，再由部署脚本让 N95 拉取代码并重建容器。
   - 严禁在 N95 宿主机直接跑 `pip install`、`npm install -g` 或宿主机 `systemd` 业务进程。
   - `projects\` 下必须保持**扁平一对一**结构；若项目本地开发需参考大型第三方源码，统一放入该项目内的 `_ref/`（先移除内部 `.git`），并在 `.gitignore` 与 `.dockerignore` 中**双重排除 `_ref/`**；小体积 API 手册放 `docs/` 随 Git 同步。
2. **N95 标准部署路径**：所有项目必须部署在 `/opt/homelab/stacks/<project_name>/`：
   ```text
   /opt/homelab/stacks/<project_name>/
   ├── docker-compose.yml    # 必选：服务编排（须含下方 6 项强制配置）
   ├── Dockerfile            # 必选：或明确锁定基础镜像版本（禁止无约束 :latest）
   ├── .dockerignore         # 必选：排除 _ref/、.git/、.venv/、data/、.env*
   ├── .env                  # 必选（如有密钥）：仅存 N95 本地，权限 chmod 600，严禁进 Git
   ├── .env.example          # 必选：脱敏环境变量模板，纳入 Git
   ├── .gitignore            # 必选：必须排除 .env、data/、_ref/、logs/、__pycache__/、.venv/
   ├── GEMINI.md / .agents/  # 推荐：项目专属 AI 规则与开发文档，纳入 Git
   ├── docs/                 # 可选：项目本地 API 参考手册（<1MB）
   ├── config.*              # 按需：非敏感配置文件（建议 :ro 只读挂载）
   └── data/                 # 必选（如有状态）：数据库与持久化状态集中挂载目录
   ```

---

## 二、`docker-compose.yml` 强制配置清单

每个迁移项目的 `docker-compose.yml` **必须**包含以下 6 项，缺一不可：

1. **自动恢复**：`restart: unless-stopped`
2. **时区锁定**：`environment` 中显式注入 `TZ=Europe/London`（除非业务明确要求特定时区）
3. **日志体积硬限制（防 512GB SSD 爆盘）**：
   ```yaml
   logging:
     driver: "json-file"
     options:
       max-size: "10m"
       max-file: "3"
   ```
4. **内存上限分级保护（按业务重要性与计算负载分配）**：必须设置 `mem_limit`，防止低优先级实验项目内存泄漏拖垮 16GB N95 上的核心服务：
   - **P0 核心/量化计算型**（如 `trading-python` 等涉及投资决策与 Pandas/NumPy 矩阵重算）：给宽裕上限 **`1.5g` ~ `2g`**，严防峰值被 OOM 误杀；
   - **P1 常规业务服务**（标准异步 Bot / Web API）：**`512m` ~ `1g`**；
   - **P2 低优先级 / 轻量脚本 / 实验容器**：**`256m` ~ `512m`**。
5. **优雅停机与健康检查**：
   - 含数据库写入的项目必须配置 `stop_grace_period: 30s`（避免重启时被强制杀进程损坏事务）。
   - 配置 `healthcheck`（供 Docker 与 Uptime Kuma 探测）。
6. **端口暴露与共享内网**：禁止绑定 `0.0.0.0:<port>:<port>` 暴露到外部；如项目带有内部 Web 面板，仅绑定 `127.0.0.1` 或接入 `homelab-net` 外部共享网络（通过 Tailscale 访问）。

---

## 三、数据持久化、SQLite 与备份兼容性规则

1. **状态目录统一**：所有数据库与持久文件必须挂载至项目内 `./data:/app/data`，严禁写入容器内部临时目录或匿名卷（Unnamed Volume），否则重建容器即丢数据且无法被备份脚本扫描。
2. **SQLite 专项铁律**：
   - 必须启用 `WAL` 模式与 `busy_timeout`，并确保进程内写锁串行化；**严禁多个容器同时读写同一个 SQLite 文件**。
   - 不得长期持有排他文件锁导致宿主机无法执行 `sqlite3 .backup` 在线热备。

---

## 四、Telegram 通信与错误告警统一规范（全员 Long Polling）

1. **全部统一为 Long Polling（取消 Webhook）**：
   - 原在云端使用 Telegram Webhook 接收消息的项目，迁入 N95 时**必须改写为 Long Polling（长轮询）**，使 N95 只做纯出站连接，路由器零端口映射。
2. **交互 Bot 令牌隔离**：
   - Telegram 规定同一 Bot Token 同一时刻只允许一个连接做 Long Polling，否则触发 `409 Conflict`。
   - **每个需要指令交互的项目，必须向 `@BotFather` 申请独立 Bot Token**。
3. **统一报错与状态告警（`ops-notify` 契约与双发分流原则）**：
   - 所有项目**禁止只在本地打印 Error 日志不通知**！必须接入 `homelab-net` 外部共享网络中的 `ops-notify` 网关，通过环境变量 `OPS_NOTIFY_URL`（N95 线上为 `http://ops-notify:8000/alert`）开关控制；未配置该环境变量时（如本地单测或旧云端过渡期）应自动优雅降级，不得阻塞主流程。
   - **接口契约（`POST http://ops-notify:8000/alert`）**：
     - 请求头：`Content-Type: application/json`，支持可选 `Authorization: Bearer <OPS_NOTIFY_TOKEN>`（未配置 Token 时内网免鉴权）。
     - 请求体 JSON：
       ```json
       {
         "service": "trading-python",
         "severity": "WARN",
         "alert_key": "t212_rate_limit",
         "title": "简短告警标题",
         "message": "详细告警内容",
         "cooldown_sec": 1800
       }
       ```
       *(其中 `service`、`message` 必填；`severity` 取值 `INFO | WARN | CRITICAL`；`alert_key` 与 `cooldown_sec` 可选，用于去重与冷却防刷屏)*
   - **告警分流与双发策略**：
     - **技术/运维类告警**（接口限流退避、单数据源抖动、网络重试、容器异常、备份回执）：**仅发往 `ops-notify`（运维总控 Bot）**，严禁刷屏污染业务交互 Bot 窗口。
     - **强业务阻塞类告警**（如晨报 `/report` 彻底失败、券商 API Key `401/403` 失效导致今日风控失效）：**允许双发**——发往 `ops-notify`（`CRITICAL`）的同时，在项目自身 Bot 补发 1 条极简提示。

---

## 五、项目迁移标准执行 SOP（旧云主机 → N95）

AI 助手协助迁移具体项目时，必须严格按以下顺序执行，严禁跳过第 2 步直接双开：

1. **本地适配改造（在 `<HOMELAB_ROOT>\projects\<name>`）**：
   - 补齐 `Dockerfile`、`docker-compose.yml`、`.gitignore`、`.env.example`；若旧项目用 Telegram Webhook，先改为 Long Polling；接入 `ops-notify` 错误告警；跑通本地编译与离线单测，推至 GitHub 私有库。
2. **旧云主机安全停服（防双开抢占）**：
   - 在旧云主机停止容器/进程并确认彻底退出（避免新旧两台同时对同一 Telegram Bot 做 Long Polling 或重复跑定时任务）。
3. **数据与密钥迁移**：
   - 若旧端有 SQLite，先执行 `sqlite3 <db> ".backup <clean.db>"` 后再拷贝至 N95 `/opt/homelab/stacks/<name>/data/`。
   - 在 N95 拉取 Git 代码，放入 `.env` 并执行 `chmod 600 .env`。
4. **N95 构建启动与日志巡检**：
   - 执行 `docker compose up -d --build`，通过 `docker compose logs -f` 确认无缺包、无挂载目录 UID 权限报错。
5. **双验收（业务 + 备份）**：
   - 触发一次业务健康检查（如 `/health`）与测试告警推送至运维 Bot。
   - 手动跑一次宿主机 `backup_daily.sh`，验证新项目的 `data/` 与 `.env` 已成功加密写入 Google Drive。
6. **冷备观察期**：
   - 旧云主机保留**关机状态观察 7 天**，确认 N95 稳定度过一个完整周期后再释放旧云主机。
