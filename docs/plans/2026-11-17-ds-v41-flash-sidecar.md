# DeepSeek V4.1 Flash 侧车增强 Implementation Plan

**Goal:** 在不触碰原始真源数值的前提下，用 DeepSeek Flash 侧车提升 `/pick` 候选覆盖面、给 `/report` 与 `/pick` 增加只读语义质检、补上盘中异动哨兵与周六策略周报/带原话锚点的偏好记忆。

**Architecture:** 所有 DS 调用收敛到 `src/agents/ds_sidecar.py`（统一门控 + 硬超时 + 静默降级 + JSON 提取）；DS 只产出「ID 名单 / 只读质检 / 独立快报 / 带原话注解」，FactSheet、持仓与挂单数值、`daily_vetoes` 始终由 Python/SQLite 直出主模型。任何 DS 失败一律确定性回退原逻辑，不阻塞 `/report`、`/pick` 与调度主循环。

**Tech Stack:** Python 3.12、pydantic v2、aiosqlite、asyncio、unittest（全离线 Mock）

## 全局红线（与 AGENTS.md 对齐）

1. SQLite 写库只走 `AdvisorDatabase._connect(write=True)`；迁移只用 `CREATE TABLE IF NOT EXISTS` / `_ensure_columns`。
2. 降级数据（`portfolio.is_fallback` / `macro.is_stale`）不回写缓存与快照。
3. Telegram 纯文本：禁 `#` 与 `**`；`/report<=1800`、`/pick<=1000`。
4. DS 侧车开关关闭或 `DEEPSEEK_API_KEY` 缺失时 **0 调用**，绝不静默回落 Gemini 顶包。
5. 禁止嵌套蒸馏：周报与偏好抽取均单跳直读 `chat_history` / `pick_recommendations` / `quant_health_audits` 原始行。

## Batch 1：Phase 1（宽漏斗初筛）+ Phase 2（语义质检）

### Task 0 共享侧车基础设施

- Create: `src/agents/ds_sidecar.py`
- Modify: `src/config.py`（ModelsSection：`ds_helper_model` / `ds_helper_thinking_level` / `ds_helper_timeout_seconds` / `ds_pick_prefilter_enabled` / `ds_semantic_audit_enabled`；StrategySection：`pick_wide_funnel_top_n` / `pick_prefilter_keep_per_track` / `pick_search_symbols_per_track`）
- Modify: `config.yaml`
- Test: `tests/test_ds_pick_prefilter_and_auditor.py`

```python
def ds_helper_available(llm, cfg) -> bool
def extract_json_object(text: str) -> dict[str, Any] | None
async def call_ds_json(cfg, llm, *, prompt, system_instruction="", timeout_s=None, usage_label) -> dict | None
```

`ds_helper_available` 优先读 `getattr(llm, "deepseek_api_key", None)`，属性缺失时退回 `cfg.env.deepseek_api_key`；为空即不可用（现有测试桩无该属性且 EnvSettings 为空 → 既有用例零影响）。

### Task 1 `/pick` 宽漏斗初筛

- `pick_agent.py:497-509`：`top_n_per_track=8` → `cfg.strategy.pick_wide_funnel_top_n`
- `pick_agent.py:583-585`：检索标的 `[:6]+[:6]` → `[:pick_search_symbols_per_track]`
- 新增 `_resolve_keep_candidates()` / `_prefilter_candidates_with_ds()`

行为：开关关或 DS 不可用 → `isa[:keep]`/`invest[:keep]`；候选总数 <=12 → 不初筛；DS 返回 `{"keep_isa","keep_invest","flags"}`，symbol 归一后只保留真实候选、按原始排名重排、单轨截断 `keep`、返回原始对象；`flags` 只进 DEBUG 日志；解析异常/空集回退；喂 DS 的上下文仅 screener 本地字段（0 网络）。

### Task 2 语义质检侧车

- Create: `src/agents/semantic_auditor.py`
- Modify: `report_agent.py`（save 之后、return 之前）；`pick_agent.py`（`_post_screen_validate_decision` 增补 `llm`/`ops_notifier`）

```python
class SemanticAuditResult(BaseModel):
    passed: bool = True
    violations: list[str] = Field(default_factory=list)
    skipped: bool = False

def build_fact_baseline(*, scope, portfolio, evals=None, trade_ctx=None, guardrails=None, fact_sheet="") -> str
async def audit_generated_output(llm, *, cfg, scope, fact_baseline, rendered_text, strategy,
                                 timeout_s=None, ops_notifier=None, headline="",
                                 selected_symbols=None, known_symbols=None) -> SemanticAuditResult
```

四项红线：幻觉价位 / 违背挂单计息 / `/pick` 驳回外泄 / 与 `daily_vetoes` 冲突。`pick` 先跑 0-Token 确定性预检再交 DS 复核。只告警不拦截：violations 非空 → `ops.notify(WARN)` + `logger.warning`，文本一字不改。

## Batch 2：Phase 4A + Phase 4B

- `quant_health_audits` 幂等补列 `ai_diagnosis TEXT DEFAULT ''`；`save_quant_audit(..., ai_diagnosis="")`；`get_latest_quant_audit()` 返回该列
- `_run_weekly_quant_audit_impl` 单跳读 `get_recommendation_stats()` + `get_recent_lessons(limit=8)` → DS `{"diagnosis"}` ≤450 字去 `#`/`**` → 落库；周六 job 推送，工作日 catch-up 仍静默；`/stats` 展示
- 新表 `user_preference_notes(id, source_cycle_date, quote_text, rule_summary, expires_on_uk, created_at_ms)` + 索引
- 新方法 `get_user_queries_since` / `upsert_preference_notes` / `get_active_preference_notes` / `prune_expired_preference_notes`
- 每日结算末尾 `_extract_preference_notes(base)`：当日 00:00 UK 起的原始原话 → DS `{"notes":[{"quote","rule","ttl_days"}]}`；`quote` 必须逐字子串否则丢弃；`rule<=60` 字、ttl 1~30、每日 ≤3 条
- 注入 `prompts.format_preference_notes_block()` → `[2026-11-17 · 原话: "..."] 规则`；`chat_agent` 放 `dynamic_tail`（保护隐式缓存），`pick_agent` 放 researcher/critic prompt

## Batch 3：Phase 3 盘中异动哨兵

- 配置：`intraday_sentinel_enabled=True`、`intraday_poll_minutes=30`、`intraday_move_alert_pct=3.5`、`intraday_order_near_pct=1.5`
- 工作日 `08:00~21:00 UK` 且 `not execution_lock.locked()` 时按 `intraday_poll_minutes` 节流；**不得复用 `_execute_job`**（锁不可重入会死锁），独立 `_run_intraday_sentinel_impl` + 直接 `self.tg.send_message`
- 0-Token 扫描 `move_pct=(现价/前收-1)*100`、`order_dist_pct=min(|现价-挂单价|/现价*100)`；平静 0 LLM
- 触发后单标 `run_forced_market_grounding(scope="intraday")` + DS `{"should_push","alert_text"<=250字}`；仅 `should_push` 才推送并写 `scheduler_runs` 键 `intraday_alert:{sym}:{uk_date}`

## 验收命令（每批必跑）

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m unittest tests.<新增模块> -v
$env:TEST_WATCHDOG_SECONDS="180"; .\.venv\Scripts\python.exe -m tests.run_tests tests
```

## 版本与文档

三批全绿后 `src/version.py` + `pyproject.toml` → `5.3.0`；同步 `CHANGELOG.md`、`AGENTS.md` 模块表、`docs/MODULE_CONTRACTS.md`、`DEVELOPER_GUIDE.md`、`README.md`。
