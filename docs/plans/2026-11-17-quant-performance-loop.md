# V5.3.4 实盘绩效闭环（NAV → TWR）+ K 线兜底 实施计划

> 本文为 `writing-plans` 阶段产出，执行阶段按 Task 顺序落地。

**Goal:** 用零新增外部供应商的方式，把「每日净值 → TWR 绩效 → 交易归因 → 信号质检」做成可查询的本地闭环，并消除 Yahoo 单点故障。

**Architecture:** 新增轻量 `nav_history` 表（只存 GBP 标量、永不清理）；`analytics.py` 增加纯函数计算层（输入 DataFrame/list，输出 dict，可单测）；`report_agent` 在写快照时幂等 upsert NAV；Telegram 新增 0-Token `/perf` 指令直出；`market_data` 增加 Yahoo 原生 v8 chart 端点作为 yfinance 失败后的兜底。

**Tech Stack:** Python 3.11+ / pandas / numpy / aiosqlite / httpx / unittest（全离线 Mock）

---

## 调研结论（先于编码，已验证）

| 候选源 | 结论 | 证据 |
|---|---|---|
| Stooq CSV | ❌ 不可用 | 实测返回 JS 工作量证明页（`This site requires JavaScript to verify your browser`），无 CSV |
| Yahoo v8 chart（query1/query2） | ✅ 可用 | 实测 `AZN.L`/`VWRP.L` 均返回 OHLCV + `adjclose`，无需 key |
| Polygon.io 免费档 | ❌ 不采用 | 仅美股、5 次/分，覆盖不到 LSE（本项目主战场） |
| Alpha Vantage 免费档 | ⏸ 暂不采用 | 25 次/日，无法承载 150 只底池；若未来启用，仅作第三层兜底且默认关闭 |

---

### Task 1: `nav_history` 表与仓储

**Files:**
- Modify: `src/storage/db.py`（init 建表段 + 两个方法）

**Step 1** 建表（幂等 `CREATE TABLE IF NOT EXISTS`，不参与 `prune_old_records`）：

```sql
CREATE TABLE IF NOT EXISTS nav_history (
    uk_date TEXT PRIMARY KEY,
    total_value_gbp REAL NOT NULL,
    cash_gbp REAL NOT NULL DEFAULT 0,
    invested_gbp REAL NOT NULL DEFAULT 0,
    net_flow_gbp REAL NOT NULL DEFAULT 0,
    isa_value_gbp REAL NOT NULL DEFAULT 0,
    invest_value_gbp REAL NOT NULL DEFAULT 0,
    benchmark_close REAL,
    cash_rate_pct REAL,
    created_at_ms INTEGER NOT NULL
);
```

**Step 2** `upsert_nav_history(row)`：`INSERT ... ON CONFLICT(uk_date) DO UPDATE`，重跑同日不产生重复行。
**Step 3** `get_nav_history(limit)`：按 `uk_date` 升序返回 `list[dict]`。
**Step 4** 断言 `prune_old_records` 不会删除 nav_history（测试覆盖）。

---

### Task 2: 计算层纯函数（`src/data_providers/analytics.py`）

**Files:** Modify `src/data_providers/analytics.py`

| 函数 | 输入 → 输出 | 关键口径 |
|---|---|---|
| `build_nav_row(portfolio, benchmark_close, uk_date)` | PortfolioData → dict | 只把 DEPOSIT/WITHDRAW/FEE/INTEREST 计为外部现金流；分红**不**计入 |
| `compute_portfolio_performance(nav_rows, benchmark_hist, rf)` | list/dict → dict | `r_t = (NAV_t − CF_t)/NAV_{t−1} − 1`；MaxDD 含起点 1.0；恢复天数 |
| `compute_trade_attribution(portfolio)` | PortfolioData → dict | 按钱包币种分组，**禁止跨币种求和**；胜率/盈亏比/期望/四向归因 |
| `compute_signal_quality(rows)` | list[dict] → dict | T+5/10/20 Alpha 的 t 值、衰减斜率、`significant` 标注 |
| `compute_concentration_metrics(portfolio)` | PortfolioData → dict | HHI、有效标的数、Top1/Top3、现金占比 |

**Step 1** 先写测试（见 Task 6），确认失败。
**Step 2** 实现到测试通过。
**Step 3** `TechnicalSnapshot` 增加 `vol_60d_percentile`（2 年 60 日滚动波动的分位），`format_technical_line` 追加显示。

---

### Task 3: 采集接入（`src/agents/report_agent.py`）

**Files:** Modify `src/agents/report_agent.py:1071-1074`

在 `save_snapshot` 之后，同样以 `not portfolio.is_fallback` 为门禁，upsert 当日 NAV 行；任何异常只记 warning，**绝不允许打断 /report**。基准价取已缓存的 `cfg.strategy.benchmark_symbol` 收盘（0 新增请求）。

---

### Task 4: `/perf` 指令（`src/bot/handlers.py`）

**Files:** Modify `src/bot/handlers.py`（路由 + `_build_perf_message` + `_build_help_text`）

- 路由位置：`/stats` 附近，0-Token 直出，不进入 `_run_query_task`
- 输出：账户层（TWR/波动/Sortino/MaxDD/恢复/相对基准 Alpha-Beta-IR/现金拖累）+ 交易层（胜率/盈亏比/期望/四向归因）+ 信号层（t 值/衰减）；纯文本，禁 `#` 与 `**`
- 样本不足时明确写「样本 n=xx，统计不显著」，不粉饰

---

### Task 5: K 线兜底（`src/data_providers/market_data.py`）

**Files:** Modify `src/data_providers/market_data.py`

- 新增 `_fetch_yahoo_chart_sync(symbol, period)`：`httpx.Client` 直连 `query1/query2` v8 chart，UA 伪装，解析 open/high/low/close/volume + adjclose → 标准 OHLCV DataFrame
- 接入点：`_download_histories_raw()` 在 yfinance 批量 + 单只回退**都失败后**才调用，双 host 顺序尝试
- 约束：不改变既有复权口径（`_normalize_adjusted_ohlcv` 仍负责连续性校验）；失败返回空 DataFrame 由上层降级

---

### Task 6: 测试（`tests/test_nav_performance.py`）

全离线，无网络：

1. `build_nav_row`：当日 DEPOSIT 50 → `net_flow_gbp == 50`；分红不计入
2. `compute_portfolio_performance`：100 → 110 → 入金 50 后 160 → TWR ≈ +10%（不是 +60%）
3. MaxDD：100/120/90/110 → −25%
4. 基准对齐：与基准同收益 → beta≈1、alpha≈0
5. `compute_trade_attribution`：2 盈 2 亏 → 胜率 50%、盈亏比正确
6. `compute_signal_quality`：已知序列 → t 值正确、n<3 时不报显著
7. `nav_history` upsert 幂等（同日两写 → 1 行）+ 不被 prune 删除

**命令：**
```
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m unittest tests.test_nav_performance -v
.\.venv\Scripts\python.exe -m tests.run_tests tests
```

---

### Task 7: 版本与文档

- `src/version.py` / `pyproject.toml` 保持 `5.3.4`（本次并入当前版本，不新增小版本号）
- `CHANGELOG.md` 顶部 5.3.4 追加条目
- `AGENTS.md` 模块索引补 `nav_history` 与 `/perf` 一行

---

## 明确不做（本期）

| 项 | 原因 |
|---|---|
| 本地 Beta/Alpha 注入技术行 | 需把基准 K 线穿透到 4 个调用点并新增取数，违背「零新增请求」，且与波动分位信息重叠 |
| Polygon / Alpha Vantage | 覆盖或额度不满足（见调研表） |
| 因子模型 / 组合优化器 / ML | 样本量不足，属过度设计 |
