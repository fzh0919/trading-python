# Telegram 502 静默失败修复（P0）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让 Telegram 上游 502 不再被吞成「轮询成功」，并在回复未送达时抛出 CRITICAL 运维告警，闭环 HOMELAB OPS MONITOR 的可观测性。

**Architecture:** 两处最小侵入改动：① `TelegramClient.get_updates` 在上游 5xx/无 error_code/401/403 时抛出 `TelegramPollError(httpx.HTTPError)`，复用 `src/main.py` 中既有的 `_poll_loop` 异常分支（计数 + `record_offline_incident` + 5s 退避）；② `AdvisorBot` 新增 `_deliver()` 统一发送出口，发送失败时 `ops.notify(CRITICAL, alert_key="telegram_reply_undelivered")` 并通过 `_post(max_retries=self._delivery_max_retries)` 把 5xx 重试预算从 2 次/1.5s 提升到 4 次/7.5s。

**Tech Stack:** Python 3.12、httpx、asyncio、pytest 风格的 `unittest.IsolatedAsyncioTestCase`（项目实际用 `tests/run_tests.py` + `unittest`）。

---

## Task 1: `get_updates` 不再吞掉上游故障

**Files:**
- Modify: `src/bot/telegram_client.py:11`（新增异常类）、`:43-58`（新增 `delivery_max_retries`）、`:95-131`（`_post` 支持按调用覆盖重试预算）、`:159-167`（`get_updates` 抛错）
- Test: `tests/test_v504_ops_and_grounding.py`（新增 `TestTelegram502SilentFailureFix`）

**Step 1: 写失败用例**

```python
async def test_getupdates_502_raises_poll_error_instead_of_empty_list(self) -> None:
    calls = 0

    async def _handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(502, json={"ok": False, "error_code": 502, "description": "Bad Gateway"})

    tg = TelegramClient("tok", transport=httpx.MockTransport(_handler), max_retries=1, base_backoff_s=0.0)
    try:
        with self.assertRaises(TelegramPollError) as ctx:
            await tg.get_updates(offset=5, timeout_s=1)
        self.assertIsInstance(ctx.exception, httpx.HTTPError)
        self.assertEqual(ctx.exception.error_code, 502)
        self.assertEqual(calls, 2)
    finally:
        await tg.close()
```

**Step 2: 跑测试确认 FAIL**

Run: `.\.venv\Scripts\python.exe -m unittest tests.test_v504_ops_and_grounding.TestTelegram502SilentFailureFix -v`
Expected: FAIL（`ImportError: cannot import name 'TelegramPollError'`）

**Step 3: 实现**

```python
class TelegramPollError(httpx.HTTPError):
    """Upstream getUpdates failure (e.g. HTTP 502) surfaced to the polling supervisor."""

    def __init__(self, method: str, error_code: int | None, description: str) -> None:
        self.method = method
        self.error_code = error_code
        self.description = description
        super().__init__(f"{method} upstream failure (error_code={error_code}): {description}")
```

```python
    @staticmethod
    def _is_poll_transport_failure(data: dict[str, Any]) -> bool:
        raw = data.get("error_code")
        if raw is None:
            return True
        try:
            code = int(raw)
        except (TypeError, ValueError):
            return True
        return code >= 500 or code in (401, 403)
```

`get_updates` 的 `if not data.get("ok")` 分支改为先记日志，再按 `_is_poll_transport_failure` 决定抛错或返回 `[]`。

**Step 4: 跑测试确认 PASS**

**Step 5: 提交**（用户环境按需）

---

## Task 2: 回复送达校验 + CRITICAL 告警

**Files:**
- Modify: `src/bot/handlers.py:115-119`（`_send_reply` 改为委托 `_deliver`）、新增 `_deliver`、`:460/:474/:482/:494/:510/:531/:557/:567/:582` 全部改走 `_deliver`
- Test: `tests/test_v504_ops_and_grounding.py`

**Step 1: 写失败用例**

`_StubTelegram(fail_delivery=True)` + 真实 `AdvisorDatabase` + `OpsNotifier(ops_notify_url 为空)`，断言 `await bot._send_reply("999", "答案正文") is False` 且 `db.get_recent_ops_events()` 含 `alert_key == "telegram_reply_undelivered"`。

**Step 2: 跑测试确认 FAIL**（`_send_reply` 返回 `None`）

**Step 3: 实现**

```python
    async def _deliver(self, chat_id: str | int, text: str, *, remove_keyboard: bool = False) -> bool:
        """Send a Telegram reply and raise a CRITICAL ops alert when upstream drops it."""
        reply_markup = REMOVE_KEYBOARD_MARKUP if remove_keyboard else None
        try:
            delivered = await self.tg.send_message(chat_id, text, reply_markup=reply_markup)
        except Exception as exc:
            logger.warning("Telegram reply delivery raised (%s: %s)", type(exc).__name__, exc)
            delivered = False
        if not delivered:
            logger.error("Telegram reply undelivered (chat_id=%s, chars=%d)", chat_id, len(text))
            if self.ops is not None:
                try:
                    await self.ops.notify(
                        severity="CRITICAL",
                        source="telegram:delivery",
                        alert_key="telegram_reply_undelivered",
                        message=(
                            f"回复未送达 chat_id={chat_id}（{len(text)} 字符）：Telegram 上游 5xx 或网络中断。"
                            "请检查 getUpdates 轮询日志，恢复后手动重发该问题。"
                        ),
                    )
                except Exception as alert_exc:
                    logger.warning("Failed raising undelivered-reply alert: %s", alert_exc)
        return delivered
```

**Step 4: 跑测试确认 PASS**

---

## Task 3: 5xx 送达重试预算 4 次

**Files:** `src/bot/telegram_client.py`

`__init__(..., delivery_max_retries: int = 4)` → `self._delivery_max_retries = max(self._max_retries, int(delivery_max_retries))`；`_post(..., max_retries: int | None = None)` 内部 `retry_budget`；`send_message` 传 `max_retries=self._delivery_max_retries`。

**Test:** MockTransport 前 4 次返 502、第 5 次 200 → `send_message` 返回 True 且 `attempts == 5`。

---

## Task 4: 文档与账本

- `CHANGELOG.md`：`v5.2.0` 下追加第 10 条。
- `docs/MODULE_CONTRACTS.md`：更新 `src/bot/telegram_client.py` 契约条目，补 `TelegramPollError`、`delivery_max_retries`、`AdvisorBot._deliver`。

---

## 全量验收

```bash
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.run_tests tests
```
