# Trading 212 × Gemini AI Advisor — 变更日志 (CHANGELOG)

> 仅维护当前活跃版本（`v5.2.0+`）以便快速追加；`IMP-01 ~ IMP-22` 施工详录与 `v5.0.1 ~ v5.1.2` 历史版本已归档至 [`docs/CHANGELOG_ARCHIVE.md`](./docs/CHANGELOG_ARCHIVE.md)。

### `v5.3.1`（2026-11-17）
盘中哨兵与语义质检的 3 处隐患修复（代码审查发现，均先复现再修）：

1. **盘中哨兵不再吃日线陈旧价（`src/bot/scheduler.py`）**：`get_bundles(include_info=False)` 分支只回落到 `2y` 日线末根 `Close`，15 分钟缓存内会读到旧价、美股开盘初期还可能因当日日线未生成而整段跳过。现改为三层实时价覆盖：**Finnhub `get_us_realtime_quote()`（美股真实时价 + `prev_close`）→ T212 持仓 `cur_price` → 日线末根 Close**，并用 Finnhub `prev_close` 作为涨跌幅基准；`_intraday_move_pct()` 新增 `prev_close` 入参，日线末根为昨日时改用该根 Close 作前收（仅当滞后 >10 天才判定不可用，避免陈旧缓存伪装成日内波动）。
2. **止损挂单纳入逼近判定 + 发送失败不占当日去重（`src/bot/scheduler.py`）**：`_order_distance_pct()` 同时兼容 `limit_price` 与 `stop_price`（取最小触发距离），修掉止损挂单逼近被漏报；推送改为检查 `send_message()` 的布尔返回值，`False` 时 `continue` 不写 `scheduler_runs("intraday_alert:{sym}")`，当天后续扫描可正常补推。
3. **语义质检数值底座放宽截断（`src/agents/semantic_auditor.py`）**：`build_fact_baseline()` 的 FactSheet 上限 `4000 → 12000` 字符，消除 ~20 只持仓时排名靠后标的正常报价被判「幻觉价位」的误报 WARN。
4. **回归单测**：`tests/test_intraday_sentinel.py` 增至 15 条（新增 Finnhub 实时价覆盖昨日日线、`prev_close` 基准优先、止损价逼近、发送失败不占去重额度并可下轮补推、陈旧日线保护）；`tests/test_ds_pick_prefilter_and_auditor.py` 增至 20 条（新增 >4k 字符 FactSheet 尾部数值不丢）。

### `v5.3.0`（2026-11-17）
DeepSeek V4.1 Flash 侧车增强（防「传话筒效应」：FactSheet、持仓与挂单数值、`daily_vetoes` 始终由 Python/SQLite 直出主模型，DS 只做 ID 筛选、只读质检、独立快报与带原话锚点的记忆抽取）：

1. **共享侧车底座（新增 `src/agents/ds_sidecar.py`；`src/config.py`, `config.yaml`）**：新增 `ModelsSection.ds_helper_model="deepseek-flash"`、`ds_helper_thinking_level="HIGH"`、`ds_helper_timeout_seconds=20.0`、`ds_pick_prefilter_enabled`、`ds_semantic_audit_enabled` 与 `StrategySection.pick_wide_funnel_top_n=15` / `pick_prefilter_keep_per_track=8` / `pick_search_symbols_per_track=8`。`ds_helper_available()` 强制校验真实 `DEEPSEEK_API_KEY`，**杜绝缺 key 时静默回落 Gemini 顶包**；`call_ds_json()` 统一硬超时 + 非 JSON/异常静默降级，`extract_json_object()` 容忍 ```json 围栏与前后噪声，`sanitize_plain_text()` 统一去 Markdown 控字。
2. **Phase 1 `/pick` 宽漏斗初筛（`src/agents/pick_agent.py`）**：`run_quant_screener` 由 `top_n_per_track=8` 放宽至可配 15（最多 30 只候选），新增 `_prefilter_candidates_with_ds()` 用 0-Token 本地维度聚合让 DS 输出 `{"keep_isa","keep_invest","flags"}`，Python 侧做幻觉 symbol 丢弃、按原始 screener 排名重排、单轨截断 8 只并**回传未经改动的原始 `ScreenerCandidate` 对象**；`flags` 只进 DEBUG 日志，绝不进入 Prompt/Telegram。解析异常、空集、开关关闭或 DS 不可用一律确定性回退前 8；Google 检索与 Finnhub 深挖标的扩至每轨 8 只。
3. **Phase 2 语义质检侧车（新增 `src/agents/semantic_auditor.py`；`src/agents/report_agent.py`, `src/agents/pick_agent.py`）**：`audit_generated_output()` 在 `/report` 落库后返回前、`/pick` 渲染后执行，逐条核对幻觉价位、违背挂单计息、headline 内部驳回外泄、与 `daily_vetoes` 冲突四项红线；`/pick` 先跑 0-Token 确定性预检（headline 提及未入选且非持仓的候选代码），再交 DS 复核。初期只告警不拦截：违规仅 `ops.notify(WARN, alert_key=semantic_audit_{scope})` + `logger.warning`，渲染文本一字不改；超时/异常静默 `skipped=True`。
4. **离线单测（Batch 1）**：新增 `tests/test_ds_pick_prefilter_and_auditor.py`（19 条：侧车门控/JSON 提取/超时降级、初筛对象恒等与回退、开关与缺 key 零调用、语义质检告警与静默降级）。
5. **Phase 4A 周六策略周报（`src/storage/db.py`, `src/bot/scheduler.py`, `src/bot/handlers.py`）**：`quant_health_audits` 幂等补列 `ai_diagnosis TEXT DEFAULT ''`（`CREATE TABLE` + `_ensure_columns` 双路径，老库不丢数据）；`save_quant_audit(..., ai_diagnosis="")` / `get_latest_quant_audit()` 同步读写。`_run_weekly_quant_audit_impl` 单跳直读 `get_recommendation_stats()`（T+5/10/20、MFE/MAE）与 `get_recent_lessons(limit=8)` 原始行，交 DS 产出 <=450 字、已去 `#`/`**` 的周度诊断落库；周六 job 推送「📊 周六策略体检周报」，**无诊断时返回空串静默**（`_execute_job` 不再发送空消息），工作日 catch-up 保持静默；`/stats` 新增「【AI 策略诊断】」区块。
6. **Phase 4B 带原话锚点与 TTL 的偏好记忆（`src/storage/db.py`, `src/agents/prompts.py`, `src/bot/scheduler.py`, `src/agents/chat_agent.py`, `src/agents/pick_agent.py`）**：幂等新增 `user_preference_notes(id, source_cycle_date, quote_text, rule_summary, expires_on_uk, created_at_ms)` 与过期索引；新增 `get_user_queries_since` / `upsert_preference_notes`（14 天内同规则去重）/ `get_active_preference_notes` / `prune_expired_preference_notes`。每日结算末尾 `_extract_preference_notes()` 从当日 00:00 UK 起的**原始 `user_query`** 单跳抽取，`quote` 必须空白归一后逐字命中某条原话（否则整条丢弃），`rule<=60` 字、TTL 1~30 天且每日 ≤3 条；无对话/缺 key/开关关闭均 0 调用。`format_preference_notes_block()` 输出 `[2026-11-17 · 原话: "..."] 规则（有效至 ...）`，`chat_agent` 注入 **dynamic_tail**（保护 Gemini 隐式缓存前缀），`pick_agent` 注入 researcher/critic prompt。
7. **离线单测（Batch 2）**：新增 `tests/test_sidecar_memory_and_weekly.py`（18 条：老库补列与诊断往返、偏好去重/TTL/剪枝/时间窗查询、原话逐字校验与缺失 quote 丢弃、周报包装与静默回退、无 key 零调用、`/stats` 诊断展示）。
8. **Phase 3 盘中主动盯盘哨兵（`src/config.py`, `config.yaml`, `src/bot/scheduler.py`, `src/agents/report_agent.py`）**：新增 `intraday_sentinel_enabled` / `intraday_poll_minutes=30` / `intraday_move_alert_pct=3.5` / `intraday_order_near_pct=1.5`。工作日 `08:00~21:00 UK` 且 `execution_lock` 空闲时按 30 分钟节流扫描**各自市场正在开盘中**的持仓与挂单（`US` 用 `phase=OPEN`，`UK/EU` 用 08:00-16:30 + 银行假日排除），平静时 0 LLM 调用。触发条件 `|日内涨跌| >= 3.5%` 或 `距最近挂单价 <= 1.5%`，仅对触发标的做 `run_forced_market_grounding(scope="intraday")` 1v1 强搜 + DS `{"should_push", "alert_text"<=250}` 判定，`should_push=true` 才推送 Telegram；**推送成功后才写** `scheduler_runs("intraday_alert:{sym}", uk_date)` 做当日幂等去重，单轮最多 2 条。哨兵走独立 `_run_intraday_sentinel_job` 并拒绝复用 `_execute_job`（`asyncio.Lock` 不可重入会死锁），锁被用户指令占用时直接跳过、绝不排队。
9. **离线单测（Batch 3）**：新增 `tests/test_intraday_sentinel.py`（12 条：涨跌/挂单距离阈值边界、窗口与周末/美股未开盘门控、锁占用跳过、30 分钟节流与当日去重、`should_push=false` 静默且不占用去重额度、缺 key 零推送、`_intraday_move_pct`/`_order_distance_pct` 纯函数）。
10. **全量回归**：`206 tests OK`（`compileall` 无语法错误）。

### `v5.2.1`（2026-11-17）
围绕 Trading 212 双账户币种隔离与 Invest `GBP + USD` 双钱包管理完成 Phase 1~5 全链路改造（ISA 维持纯 GBP 单钱包与全量换汇门槛，Invest 落地 `GBP 4.0% / USD 3.4%` 双钱包对称换汇摩擦、`/wallet` & `/usd` 对账与周六提醒，全套 `157 tests OK`）：

1. **T212 双账户币种隔离与 Invest GBP+USD 双钱包底座（Phase 1~2：`src/config.py`, `config.yaml`, `src/storage/db.py`, `src/data_providers/t212_client.py`, `src/main.py`）**：
   - **配置**：`strategy` 新增 `usd_cash_interest_rate_pct=3.4`、`invest_multi_currency_enabled=true`、`wallet_reconcile_warn_days=7`、`wallet_mismatch_warn_gbp=10.0`（保留 `cash_interest_rate_pct=4.0` / `fx_fee_pct=0.15` / `uk_stamp_duty_pct=0.50`）。
   - **SQLite 锚点**：幂等新增 `wallet_balances(account_label, currency, available_amount, reserved_amount, updated_at_uk, updated_at_ms, source, note)` 复合主键表，并 `INSERT ... WHERE NOT EXISTS` 种入 `('Invest','USD')=0.0`；新增 `get_wallet_balances` / `upsert_wallet_balance`（负值与非有限值钳位 `0.0`）/ `clear_manual_gbp_override`（仅清除 `manual*` GBP 行）。
   - **快照拆解**：`AccountSnapshot` 增补 `multi_currency_enabled`、`usd_*`、`gbp_wallet_*`、`wallet_source`、`wallet_updated_at_uk`、`wallet_reconcile_warning`、`gbp_usd_rate`、双币种利率字段；`PortfolioData` 增补 `usd_cash_interest_rate_pct` / `gbp_usd_rate`。纯函数 `reconcile_invest_wallets()` 按 `GBP 钱包 ≈ API 总现金 − USD/GBPUSD` 推算，负值钳位 `0.0` 并告警；用户手动双报时按 `max(£10, 总现金×2%)` 校验偏差；`T212Client.get_portfolio()` 在有 DB 时自动挂载拆解，`set_gbp_usd_rate()` 由 `/report`、`/pick`、`/health` 的宏观快照注入首个已知汇率并升级 TTL 缓存。
   - **流水与展示修复**：`build_trade_context()` 资金流水/派息改按 `currency` 分币种汇总，USD 流水不再并入 GBP 总额；`format_account_overview()` 在 Invest 启用双钱包时输出「现金钱包拆解: GBP 钱包 ≈ £x (可用 £y, 4.0%) + USD 钱包 $a (可用 $b ≈ £c, 3.4%)」及校准警告，`total_cash` / `total_combined_cash` 仍保持 GBP 口径。
   - 新增 `tests/test_invest_wallet_split.py`（13 组离线单测）与 `tests/test_db_storage.py` 钱包往返用例，全量回归 `135 tests OK`。
2. **Telegram 双钱包对账指令与量化摩擦对称改造（Phase 3~4：`src/bot/handlers.py`, `src/bot/scheduler.py`, `src/engine/portfolio_rules.py`, `src/engine/screener.py`）**：
   - **`/wallet` / `/usd` 免锁指令**：两者均由 `handle_update` 直接应答且 `_is_heavy_ai_command=False`（不抢 `execution_lock`、不打断在跑分析）；`/usd <可用> [挂单]` 上报 Invest USD 锚点并清除手动 GBP（恢复 `GBP=TOTAL-USD` 推算），若 `USD/GBPUSD > 总现金 + max(£10, 2%)` 立即阻断并提示先 `/pos` 复核；`/wallet gbp x` 单报 GBP 时按总现金反推 USD，`/wallet gbp x usd y` 双报即时给出核对差额与告警阈值；只读 `/wallet` 展示实时总现金、GBP/USD、USD 记录与距今校准天数。
   - **周六对账提醒**：`scheduler_runs` 以 `weekly_wallet_reconcile:{ISO周}` 幂等键，在周六 `12:00~22:00 UK` 推送轻量提醒（当前 USD 记录、更新时间、实时 Invest 总现金、推算 GBP 钱包），提示回复 `/usd 金额` 校准。
   - **双钱包非对称摩擦（`portfolio_rules.py`）**：`evaluate_holding` / `evaluate_portfolio` / `build_disagreement_matrix` 接入 Invest `AccountSnapshot`；美股在 `usd_available>0` 时 0% 换汇并以 `3.4%` 为现金基准，`=0` 时 `0.15%` 单边 + `4.0%`；英股/LSE ETF 在 `gbp_available>0` 时仅计印花税，`=0` 时加计 `0.15%` 换回损耗并以 `3.4%` 为基准；`_friction_hint` 明确标注「GBP不足需USD换回(+0.15%换汇)」等资金来源；ISA 逻辑保持不变。
   - **选股赛道摩擦对称（`screener.py`）**：`score_candidate(..., invest_wallet=portfolio.invest)` 让 Invest 赛道按 GBP/USD 可用余额动态加减分（LSE ETF/英股 GBP 不足降权并标注换回成本，美股持 USD 免换汇加权、USD 不足标注单边换汇，欧股必标换汇成本），ISA 赛道保留原有免换汇加分。
   - 新增 `tests/test_wallet_rules_and_commands.py`（11 组离线单测）并同步 `test_calendar_and_scheduler.py` / `test_v504_ops_and_grounding.py` 断言，全量回归 `146 tests OK`。
3. **Agent Prompt 双钱包注入、现金流币种归因与版本归档（Phase 5：`src/agents/prompts.py`, `report_agent.py`, `pick_agent.py`, `chat_agent.py`, `schemas.py`, `src/version.py`, `pyproject.toml`, `README.md`, `DEVELOPER_GUIDE.md`, `tests/`）**：
   - **静态画像（`prompts.py`）**：`build_investor_profile` 改写为「ISA 仅 GBP（4.0%，美/欧每笔换汇）；Invest GBP(4.0%)+USD(3.4%) 双钱包，复用同币种 0% 换汇、跨币种单边 0.15%」，并要求结合实时 GBP/USD 与对应币种现金利息后作判断；画像只含规则不含金额，保护 Gemini 隐式缓存前缀。
   - **动态上下文（新增 `build_wallet_context_text`）**：统一输出实时 GBP/USD、ISA GBP、Invest GBP 钱包 + USD 钱包（可用/挂单/折英镑金额、4.0%/3.4% 利率、过期未校准警告），`compact=True` 供 `/us`、`/pos` 控字；已注入 `/report` Analyst/Auditor 共享上下文与 PM Prompt、`/pick` 精确现金底座与 Critic Prompt、`/us` 现金头、`/pos` 头部、`answer_user_chat` 动态尾部（静态前缀零改动）。
   - **现金流归因（`report_agent._compute_portfolio_diff_text`）**：区间出入金/派息/利息/扣费按币种聚合，USD 流水不再并入 GBP，剔除出入金后净值变动严格取 GBP 口径。
   - **结构化字段引导（`schemas.py`）**：`ReportActionItem.reason`、`PendingOrderAuditItem.reason`、`PickActionItem.thesis` 描述补充 Invest 资金路径示例（如「用 USD 钱包免换汇」「GBP 不足需 USD 换回含 0.15% 损耗」），字符上限不变。
   - **验收与归档**：新增 `TestWalletSnapshotCompat`、`TestDualWalletFrictionMatrix`、`TestWalletCommandParsing` 及端到端双钱包 Prompt/上限断言；版本号同步至 `5.2.1`，全量回归 `157 tests OK`（`compileall` 通过）。

### `v5.2.0`（2026-11-16）
围绕 Intel N95 闲时算力与外部免费接口深度挖掘完成 `IMP-20 ~ IMP-22` 升级及多市场 150 只精锐底池海选引擎（0 新增重依赖、严守 Gemini \$10/月 预算与核心不变量，全套 `114 tests OK`）：

1. **本地 `2y` 日线增量缓存、T212 元数据/分红归档与盘前 `-10m` 静默预热（`IMP-20`）**：
   - `src/storage/db.py` 新增 `ohlcv_cache`、`t212_instruments_metadata`、`t212_dividend_archive`；`MarketDataService.get_histories()` 扩至 `2y` 并支持 `1mo` 增量合并与 `>0.5%` 拆股全量重拉。
   - `T212Client` 接入 `/equity/metadata/instruments`（`1 req/50s`）与累计实收股息汇总，持仓展示含息总回报；`TradingScheduler.run_prewarm()` 在早报、美股哨兵与周五雷达前 10 分钟静默预热并每日同步 1 次 T212 元数据与分红归档。
   - 修复 `/us` 盘前哨兵 Finnhub `/quote` 停留在昨收的盲区，通过 `get_us_prepost_quotes()`（`5d/15m prepost=True`）捕获美股盘前真实跳空；保留 Finnhub `eps_estimate` 并启用 `get_profile_by_isin()` 补全缺失公司资料。
2. **组合真实风险归因/CVaR、挂单触价率精算与 Finnhub/FRED/Search 深度接线（`IMP-21`）**：
   - `analytics.py` 新增 `compute_portfolio_cvar_and_risk_attribution()`，在 `compute_exposure_summary()` 输出 PSD 收缩协方差前 3 波动风险贡献源与 95% 单日历史 CVaR 尾部回撤估算。
   - `t212_client.py` `build_trade_context()` 实算活动挂单的币种对齐偏离 `%`、ATR 倍数、20 日经验触价率及筹码支撑/过窄止损/远端低概率标签。
   - `FredClient` 升级 `limit=25`（0 额外请求算 20d 利率动量 bps）并新增高收益债利差 `BAMLH0A0HYM2`；`get_us_deep_intel_batch()` 接入 `/report` 与 `/pick` Stage 2 打分；`run_forced_market_grounding()` 对临近财报/除息或涨跌 `>=3%` 持仓开启 1 对 1 独立深搜。
3. **多窗口后验追踪（`T+5/10/20` + `MFE/MAE`）与周六离线量化体检（`IMP-22`）**：
   - 每日盘后结算同步记录 `MFE/MAE` 并 0-Token 自动追评已满 `T+10`/`T+20` 的延伸 Alpha；周六 `10:00~22:00 UK` 静默运行 `run_offline_universe_and_threshold_audit()` 并落库 `quant_health_audits`，在 `/stats` 看板完整呈现。
4. **多市场 0-Token 海选引擎（150 只精锐底池）、`N × H` 相关性优化与防限流熔断**：
   - 新增 `src/engine/universe_seeds.py` 与 `config.yaml` 扩展配置：覆盖 **S&P 500 + NASDAQ 100 + FTSE 100 + FTSE 250 + Euro Stoxx 50** 基准池（内置兜底种子 + 30 天缓存依赖零新增 `HTMLParser` 同步），并将 LSE 英镑 UCITS ETF 独立底座扩至 25 只。
   - `src/storage/db.py` 与 `src/data_providers/t212_client.py` 新增 `fundamentals_cache`（21 天 TTL）、`active_universe_pool`、`index_constituents_cache`，并按 52 秒冷却错峰同步 ISA/Invest 双账户元数据校验 `isa_eligible`/`invest_eligible`。
   - `src/data_providers/market_data.py` 将 `max_cache_entries` 升至 `512`，新增带并发控制（`2` 并发 + `0.45s` 节流）、20 条断点落库与连续 3 次 429 自动熔断的 `get_fundamentals_cached()`，并对大批量日线分片增加 `0.35s` 间隔。
   - `src/data_providers/analytics.py` 为 `compute_correlation_matrix()` 新增 `against_symbols` 矩形切片模式（仅算 `候选 × 持仓` 的 `N × H` 相关性，消除 `O(N²)` 性能瓶颈）。
   - `src/engine/screener.py` 新增 `run_universe_audition()`（T212 可买性交集 → `1y` 流动性/趋势预筛 → 盈利/估值/规模/动量打分 + 单行业上限 `28` + 英欧最低 `30` 席位 → 精选 150 只个股），`run_quant_screener()` 接入共享精锐个股池、合规门控及 Stage 1 单行业上限，保留双轨非对称权重分层。
   - 接通 `pick_agent.py`、`scheduler.py`（周六自动海选 + 日常预热动态池），并新增 Telegram `/refresh_pool` 与命令行 `python -m src.refresh_pool`（`src/refresh_pool.py`）手动触发入口。
5. **版本同步与全量回归验收（`src/version.py`, `pyproject.toml`, `README.md`, `DEVELOPER_GUIDE.md`, `CHANGELOG.md`, `tests/`）**：
   - 版本号升级至 `5.2.0`，在 `IMP-20~22` 之上增补 `tests/test_universe_audition.py` 7 组单测。
6. **API 路由成本优化与本地缓存升级（用户授权归入 `v5.2.0`）**：
   - **模型路由与备援防升档（`config.yaml`, `src/config.py`, `src/agents/llm_client.py`）**：新增 `report_auditor_model: "gemini-4-argon"` 专门保留给 `/report` Step 2B Bear 挑刺审计；`report_model`（PM 拍板）与 `pick_model`（`/pick` 终审）统一切换为 `gemini-4-hydrogen` (`HIGH`)；`report_search_model` 切换为 `gemini-3.5-flash-lite`（消除搜索阶段思考 Token 浪费）；`_candidate_chain` 修复 `Hydrogen` 失败时禁止反向升级回 `Argon`。
   - **SQLite 持久化搜索快照与工具收口（`src/storage/db.py`, `src/agents/tools.py`, `src/agents/report_agent.py`）**：新增 `grounding_news_cache`（默认 `3h` TTL，`prune_old_records` 自动清理），打通 `run_forced_market_grounding` 与 `MarketToolkit.search_market_news` 跨指令/跨重启复用；`/report` Step 2 禁用 `search_market_news`、收敛 `report_max_tool_rounds=2`，并在 `FactSheet` 与 `ContextPack` 预注更多 Finnhub 新闻与 `grounding_news` 快照。
   - **`/pick` 与 `chat` 本地切片省流（`src/agents/pick_agent.py`, `src/agents/chat_agent.py`）**：`/pick` 在 Stage 1 前本地预聚合 Top 候选 Finnhub 新闻与深度评级并收敛工具轮次；`chat` 将 `premarket_note` 移出静态前缀以保护全天早报隐式缓存不被 `/us` 打碎，并对常规指定标的问询折叠非焦点平稳持仓（`114 tests OK`）。
7. **联网检索与工具声明英文化、紧凑中文要点输出（用户授权归入 `v5.2.0`）**：
   - **`src/agents/tools.py`**：将 `_single_search_query` 的 `system_instruction`、`run_forced_market_grounding` 全部分域/1对1/宏观/主题检索 Prompt，以及 `MarketToolkit.declarations()` 的 4 项工具声明与 `search_market_news` 调用统一切换为英文，显式锚定 `RNS`、`SEC`、欧股公告、业绩指引修正及机构评级变动等一手英文信源与术语；抽取共享输出契约 `_GROUNDING_OUTPUT_CONTRACT`，强制模型直接输出紧凑简体中文要点（保留段落标题与 `近期无重大新公告：SYM1, SYM2` 合并格式）。
   - **`tests/test_v504_ops_and_grounding.py` & `DEVELOPER_GUIDE.md`**：同步更新单测过滤条件并补充英文 Prompt/工具声明与中文合并格式断言（`114 tests OK`）。
8. **文档冷热分离与 Antigravity 自动发现优化（用户授权归入 `v5.2.0`）**：
   - 新增 [`AGENTS.md`](./AGENTS.md) 作为热启动单一索引；将接口签名字典与 `_ref` 映射剥离至冷文档 [`docs/MODULE_CONTRACTS.md`](./docs/MODULE_CONTRACTS.md)，将 `IMP-01 ~ IMP-22` 详录与 `v5.0.1 ~ v5.1.2` 历史归档至 [`docs/CHANGELOG_ARCHIVE.md`](./docs/CHANGELOG_ARCHIVE.md)；同步精简 [`DEVELOPER_GUIDE.md`](./DEVELOPER_GUIDE.md)、[`README.md`](./README.md) 与本文件。
9. **Gemini + DeepSeek 混合路由、早报分层搜索、`/us` 异动门控分流与双向跨厂商灾备（用户授权归入 `v5.2.0`）**：
   - **配置与环境变量（`.env.example`, `src/config.py`, `config.yaml`）**：新增 `DEEPSEEK_API_KEY` / `DEEPSEEK_BASE_URL`；`report_auditor` 切换为 `gemini-4-hydrogen` (`XHIGH`)，`report_analyst` / `pick_researcher` 调为 `MEDIUM`；`chat_model`、平稳 `/us`（`us_sentinel_model`）与盘后 `T+5` 反思切换至 `deepseek-flash` (`HIGH`)；`/us` 异动门控触发自动切 `us_anomaly_model`（`gemini-4-hydrogen HIGH`）；`/deep` 保留 `gemini-4-hydrogen` (`HIGH`)；早报新增 `report_priority_search_model`（`gemini-4-hydrogen LOW`）负责宏观与重点异动 1 对 1 深搜，常规 3 股批次保留 `gemini-3.5-flash-lite`。
   - **双厂商异步引擎与双向备援（`src/agents/llm_client.py`, `src/main.py`）**：复用 `httpx` 原生支持 DeepSeek `/chat/completions`、`thinking` + `reasoning_effort`、OpenAI 小写 JSON Schema 工具转换、多轮 `reasoning_content` 透传与 `reasoning_tokens`/缓存命中记账；实现 DeepSeek `60s` 快失败回退 `gemini-4-hydrogen (MEDIUM)`、Gemini 推理/结构化终审故障自动跨厂切入 `deepseek-flash (HIGH)`（动态注入 Pydantic Schema + `json_object`），并严格隔离 `use_google_search=True` 禁止切入无联网能力的 DeepSeek；`run_app` 停机自动调用 `llm.close()`。
   - **业务接线与回归测试（`src/agents/tools.py`, `src/agents/report_agent.py`, `src/bot/scheduler.py`, `src/bot/handlers.py`, `tests/`）**：落地早报分层搜索、`/us` 方案 B（平稳 `deepseek-flash HIGH` / 异动 `gemini-4-hydrogen HIGH`）、`T+5` 反思 `HIGH` 思考及 `/usage` 双厂商路由展示；新增跨厂备援链、DeepSeek 工具调用/Schema 兜底及分层搜索单测，全量回归通过。
10. **质量优先微调（用户授权归入 `v5.2.0`）**：
   - 将 `report_analyst_thinking_level` 与 `pick_researcher_thinking_level` 由 `MEDIUM` 升回 `HIGH`；
   - 将 `report_auditor_model`（Bear 挑刺审计官）升级为旗舰 `gemini-4-carbon` (`HIGH`)，并在 `llm_client.py` `_candidate_chain` 同步补齐 `carbon` 同级备援保护（`Carbon -> Hydrogen -> DeepSeek`）。
11. **Telegram 上游 502 静默失败修复（`src/bot/telegram_client.py`, `src/bot/handlers.py`）**：`get_updates` 遇 `5xx`/缺失 `error_code`/`401`/`403` 时抛出新增 `TelegramPollError(httpx.HTTPError)`（原实现直接吞成 `[]`，令 `_poll_loop` 误判成功、`record_offline_incident` 永不触发）；`sendMessage` 改走 `_post(..., max_retries=self._delivery_max_retries)` 把 5xx 重试预算提升至 4 次（约 7.5s 指数退避），并把 `AdvisorBot` 全部回复出口收敛到新增 `_deliver()`，送达失败立即 `ops.notify(CRITICAL, alert_key="telegram_reply_undelivered")`；新增 4 组离线单测（`120 tests OK`），实施计划留档 [`docs/plans/2026-11-16-telegram-502-silent-failure-p0.md`](./docs/plans/2026-11-16-telegram-502-silent-failure-p0.md)。

