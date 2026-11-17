# Pending-Order Fill Receipt Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 挂单成交/部分成交后，用盘中哨兵已有的 0-Token 扫描推一条 Telegram 回执，并提醒校准 Invest 双钱包。

**Architecture:** 新增 `pending_order_snapshots` 表记录每轮存活挂单；每轮哨兵先做「快照 diff」，消失的挂单再与 T212 历史成交（`recent_orders`）交叉验证；只有验证到成交才推送，纯撤单静默，接口 403/历史不可用整轮静默。全程复用 `get_portfolio(use_cache=True)`，0 额外 Token、0 额外 API 调用。

**Tech Stack:** Python 3.12、aiosqlite、asyncio、`unittest.IsolatedAsyncioTestCase`（项目使用 `python -m tests.run_tests tests`）。

---

## 关键约束（实现时不可违反）

1. **T212 历史 `id` 是 fill id，不是 order id**（`src/data_providers/t212_client.py:960`）。匹配只能按「同账户 + 同标的 + 同方向 + `first_seen_ms` 时间窗 + 成交量合计」。
2. **挂单接口 403 不等于全部成交**：`pending_orders_supported=False` 的账户必须整轮排除，且不得删除该账户快照行。
3. **历史接口不可用就不推**（`order_history_supported=False`）：宁可不报，不可误报。
4. 写库只走 `AdvisorDatabase._connect(write=True)`（SQLite 写锁单入口）。
5. Telegram 纯文本：禁 `#` / `**`，回执 ≤450 字。

## Task 1: 快照表与存取（`src/storage/db.py`）

- `init()` 幂等建表 `pending_order_snapshots`，主键 `(account_label, order_id)`，含 `first_seen_ms` / `last_seen_ms` / `filled_quantity`，索引 `idx_pending_order_snapshots_seen(last_seen_ms)`。
- 方法：`upsert_pending_order_snapshots(rows, *, seen_at_ms=0, seen_at_uk="") -> int`（`ON CONFLICT ... DO UPDATE` 只更新 `last_seen_*` 与实时字段，保留 `first_seen_*`）、`get_pending_order_snapshots() -> list[dict]`、`delete_pending_order_snapshots(keys) -> int`。
- `prune_old_records()` 追加 `DELETE ... WHERE last_seen_ms < now-3d` 并返回计数。

## Task 2: 纯函数（`src/bot/scheduler.py`）

- `diff_pending_order_snapshots(previous, live_rows, *, supported_accounts) -> {"partial","vanished","live"}`；partial 分支需回填 `first_seen_ms`（来自 prev 行）。
- `match_fill_history(order_row, recent_orders, *, grace_ms=60_000) -> dict | None`；`is_complete = 成交量合计 >= quantity * (1 - 0.001)`。
- `format_fill_receipt(order_row, match, *, partial=False, cross_day=False, max_chars=450) -> str`。

## Task 3: 接线（`src/bot/scheduler.py`）

- `_collect_order_fill_receipts(now_utc, *, uk_date) -> list[tuple[str, str]]`：开关 → 三方法可选性 → 组合 → 支持账户 → live 行 → 读快照 → diff → 部分成交/消失分支 → 落库与删行。
- `_run_intraday_sentinel_job` 先在锁内调它（独立 try），再跑异动扫描；推送成功后才 `_mark_job_locked(f"fill_receipt:{dedupe_key}", uk_date, ...)`。
- 去重键：完整成交 `{account}:{order_id}`；部分成交 `{account}:{order_id}:p{filled_qty}`。

## Task 4: 开关与单测

- `StrategySection.intraday_fill_receipt_enabled: bool = True` + `config.yaml` 同名项。
- `tests/test_intraday_sentinel.py` 新增 `TestFillReceipts`：首轮静默播种、完整成交只推一次、撤单静默并删行、部分成交去重、403 不误判、纯函数边界与格式。

## Task 5: 文档与版本

- `CHANGELOG.md` 顶部 `v5.3.3`；`docs/MODULE_CONTRACTS.md` 补签名；`AGENTS.md` 模块索引两行；`src/version.py` + `pyproject.toml` → `5.3.3`。

## 验收命令

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -c "import sys,unittest; s=unittest.defaultTestLoader.loadTestsFromName('tests.test_intraday_sentinel'); r=unittest.TextTestRunner(verbosity=1,stream=sys.stdout).run(s); sys.exit(0 if r.wasSuccessful() else 1)"
.\.venv\Scripts\python.exe -m tests.run_tests tests
```

部署（用户手动执行）：Git 提交后走根目录 `scripts\deploy-n95.ps1`，重启后 `/health` 应显示 `v5.3.3`。
