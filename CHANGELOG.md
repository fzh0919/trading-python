# Trading 212 × Gemini AI Advisor — 变更日志 (CHANGELOG)

> 仅维护当前活跃版本（`v5.2.0+`）以便快速追加；`IMP-01 ~ IMP-22` 施工详录与 `v5.0.1 ~ v5.1.2` 历史版本已归档至 [`docs/CHANGELOG_ARCHIVE.md`](./docs/CHANGELOG_ARCHIVE.md)。

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
10. **Telegram 上游 502 静默失败修复（`src/bot/telegram_client.py`, `src/bot/handlers.py`）**：`get_updates` 遇 `5xx`/缺失 `error_code`/`401`/`403` 时抛出新增 `TelegramPollError(httpx.HTTPError)`（原实现直接吞成 `[]`，令 `_poll_loop` 误判成功、`record_offline_incident` 永不触发）；`sendMessage` 改走 `_post(..., max_retries=self._delivery_max_retries)` 把 5xx 重试预算提升至 4 次（约 7.5s 指数退避），并把 `AdvisorBot` 全部回复出口收敛到新增 `_deliver()`，送达失败立即 `ops.notify(CRITICAL, alert_key="telegram_reply_undelivered")`；新增 4 组离线单测（`120 tests OK`），实施计划留档 [`docs/plans/2026-11-16-telegram-502-silent-failure-p0.md`](./docs/plans/2026-11-16-telegram-502-silent-failure-p0.md)。

