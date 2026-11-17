# Report Step 2 Timeout Budget & Timeout-Aware Failover Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** `/report` Step 2（Bull+Bear 并发）在旗舰模型长时间无响应时不再死等 6~40 分钟：硬超时不再被当成可重试抖动，且 Step 2 有总时长护栏，超时后用中性占位继续 PM 拍板。

**Architecture:** 两层改动。① `GeminiClient._call_single_round()` 把「硬超时」与「5xx/429」分档：硬超时按 `timeout_max_retries`（默认 0）快速失败并立刻切备援，5xx 仍按 `max_transient_retries`（默认 2）重试；三个预算（单轮超时/瞬时重试/超时重试）与 Step 2 总预算全部提到 `config.yaml` 并由 `main.py` 接线。② `report_agent.py` 新增 `_gather_step2_notes()`：`asyncio.wait(timeout=...)` 到点即取消未完成任务，并把「超时」与「模型不可用哨兵文案」统一替换为中性占位，杜绝 `⚠️ AI 模型服务当前全部不可用…` 这段运维告警被当成 Bear 审计意见塞进 PM prompt。

**Tech Stack:** Python 3.12、asyncio、Pydantic v2、`unittest.IsolatedAsyncioTestCase`（`python -m tests.run_tests tests`）。

---

## 关键约束（实现时不可违反）

1. **不改 `generate()` 的对外返回值契约**：全败时仍返回 `⚠️ AI 模型服务…` 字符串供 Telegram 直出；只在 `report_agent` 侧识别并替换，其他调用方零影响。
2. **不降低 5xx/429 韧性**：`max_transient_retries` 默认仍为 2，只有 `TimeoutError`（我方 `asyncio.wait_for` 到点）走快速失败。
3. **超时取消必须回收任务**：`task.cancel()` 后必须 `await asyncio.gather(*pending, return_exceptions=True)`，避免 usage 记录丢尾与 `Task was destroyed` 告警。
4. **PM prompt 禁止出现 `⚠️` 运维文案**：降级只能是中性中文占位（见 `STEP2_*_FALLBACK`）。
5. **默认值保持现状行为可解释**：`gemini_round_timeout_seconds=120`、`gemini_max_transient_retries=2` 与今日代码一致，唯一行为变化是 `gemini_timeout_max_retries=0`。
6. 不触碰 SQLite 写锁、降级数据回写、双币种隔离三条铁律。

---

## Task 1: 超时分档重试 + 降级哨兵识别（`src/agents/llm_client.py`）

**Files:** Modify `src/agents/llm_client.py:137-141`（新增哨兵常量与判定）、`:244-293`（新增 `timeout_max_retries` 参数）、`:507-560`（`_call_single_round` 重试门控）

- 新增模块级常量与判定：`is_degraded_llm_output(text)` 命中 `DEGRADED_LLM_MARKERS`（`⚠️ AI 模型服务当前全部不可用` / `⚠️ AI 模型密钥未配置` / `⚠️ GEMINI_API_KEY 未配置`）。
- `__init__` 新增关键字参数 `timeout_max_retries: int = 0`，钳制到 `[0, max(0, max_transient_retries)]`。
- `_call_single_round()`：`is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))`；可用重试预算 `= timeout_max_retries if is_timeout else max_transient_retries`；超预算时 `logger.warning("Hard timeout on %s after %.0fs; skipping further retries to fail over sooner")` 后 `raise`。

## Task 2: 配置与接线（`src/config.py`, `config.yaml`, `src/main.py`）

**Files:** Modify `src/config.py:103`（`ModelsSection` 新增 4 字段）、`config.yaml:45`（同 4 键）、`src/main.py:81-96`（构造接线）

- `gemini_round_timeout_seconds: float = Field(default=120.0, ge=15.0, le=600.0)`
- `gemini_max_transient_retries: int = Field(default=2, ge=0, le=5)`
- `gemini_timeout_max_retries: int = Field(default=0, ge=0, le=2)`
- `report_step2_timeout_seconds: float = Field(default=240.0, ge=60.0, le=900.0)`
- `main.py` 传入前三项到 `GeminiClient(...)`。

## Task 3: Step 2 总预算与中性降级（`src/agents/report_agent.py`）

**Files:** Modify `src/agents/report_agent.py:15`（import 哨兵判定）、`:679` 前（新增常量与 `_gather_step2_notes`）、`:882-906`（替换裸 `asyncio.gather`）

- `STEP2_ANALYST_FALLBACK` / `STEP2_AUDITOR_FALLBACK` 中性占位文案。
- `_gather_step2_notes(analyst_coro, auditor_coro, *, timeout_s) -> (analyst_notes, auditor_notes, degraded_roles)`：`ensure_future` 两个协程 → `asyncio.wait(timeout=timeout_s)` → 未完成者 `cancel()` 并回收 → 收集文本，空文本或 `is_degraded_llm_output()` 命中则替换占位并记入 `degraded_roles`。
- `generate_morning_report` 用 `cfg.models.report_step2_timeout_seconds` 调用它；`degraded_roles` 非空时 `logger.warning` + `ops.notify(WARN, alert_key="report_step2_degraded")`。

## Task 4: 离线单测

**Files:** Modify `tests/test_llm_pricing_and_usage.py:228-282`；Create `tests/test_report_step2_budget.py`

- 现测 `test_round_timeout_triggers_transient_retry_and_fallback` 改为断言「硬超时不重试」：`attempts == ["gemini-4-argon", "gemini-4-hydrogen"]`，usage 记录 2 条（首条 `success=False`）。
- 新增 `test_timeout_retries_respects_budget`：`max_transient_retries=2, timeout_max_retries=1` → `attempts == ["gemini-4-argon", "gemini-4-argon", "gemini-4-hydrogen"]`。
- 新增 `test_report_step2_budget.py`：正常并发（无降级）、单侧超时（对侧结果保留 + 占位文案）、双侧超时、降级哨兵文案被替换、`is_degraded_llm_output` 边界。

## Task 5: 文档与版本

**Files:** Modify `CHANGELOG.md`（v5.3.3 追加条目）、`docs/MODULE_CONTRACTS.md:85`（`GeminiClient` 新参数）与 `:96`（`generate_morning_report` 预算语义）、`AGENTS.md` 模块索引、`DEVELOPER_GUIDE.md` 模型路由表。

- `src/version.py` / `pyproject.toml` 已是 `5.3.3`，无需再升。

## 验收命令

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m unittest tests.test_report_step2_budget -v
.\.venv\Scripts\python.exe -m unittest tests.test_llm_pricing_and_usage -v
.\.venv\Scripts\python.exe -m tests.run_tests tests
```

全绿后由用户手动 `git commit` + 根目录 `scripts\deploy-n95.ps1` 发版；重启后 `/health` 应显示 `v5.3.3`。
