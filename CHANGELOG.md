# Trading 212 × Gemini AI Advisor — 变更日志 (CHANGELOG)

> 仅维护当前活跃版本（`v5.5.0+`）以便快速追加；`IMP-01 ~ IMP-22` 施工详录与 `v5.0.1 ~ v5.4.2` 历史版本已归档至 [`docs/CHANGELOG_ARCHIVE.md`](./docs/CHANGELOG_ARCHIVE.md)。

### `v5.5.10`（2026-11-21 · 关闭周五自动 /pick + 外部资金流核算修复与周六体检纠偏，`src/version.py` 与 `pyproject.toml` 同步升至 `5.5.10`）

施工计划见 [`docs/plans/2026-11-21-5.5.10-weekly-audit-nav-cashflow-plan.md`](./docs/plans/2026-11-21-5.5.10-weekly-audit-nav-cashflow-plan.md)。本版仅修实盘资金流口径与调度开关，不新增出网域名、不改凭据/`base_url`/灾备链、不改风控护栏开关。

1. **停用周五收盘后自动 `/pick`**（`config.yaml`/`src/config.py`）：`schedule.enable_weekly_radar` 默认改为 `false`，同时关闭 `prewarm_weekly_radar` 与 `weekly_radar`；手动 `/pick` 完全不受影响，显式配置 `true` 时仍按原半日市逻辑触发。
2. **外部资金流符号与时间窗修复**（`src/data_providers/analytics/performance.py`）：
   - 新增 `_collect_external_flows_gbp`：`DEPOSIT` 恒 `+abs`、`WITHDRAW/WITHDRAWAL` 恒 `-abs`，彻底免疫 T212 已负数化的提现；支持 `(after_ms, until_ms]` 窗口，周末/节假日/22:00 后出入金自动滚入下一交易日；
   - `build_nav_row` 接收前序行边界 `prev_snapshot_ms/prev_uk_date`，同日 07:30→21:40 覆写严格幂等，并输出 `created_at_ms` 与 T212 抓取时刻对齐；
   - `compute_account_split_performance` 修复分账户净值 `value + flow` 双计外部流（改为 `value`），消除入金日假暴涨与次日假回撤；
   - 新增 `reconcile_recent_nav_flows`（近 30 天安全窗、按账户校验 `transactions_supported`、分页上限疑似截断即跳过、仅提议三列流量修正）与 `compute_window_nav_performance`（`twr_7d` 优先、Modified Dietz 兜底、流水不可验证时显式 `flows_unverified`）。
3. **持久化与写入接线**（`src/storage/db/portfolio_repo.py`/`morning_report.py`/`settlement_mixin.py`）：`upsert_nav_history` 优先使用行内 `created_at_ms`；`get_nav_history` 透出 `created_at_ms`；新增 `get_previous_nav_row` 与单写锁内 `update_nav_history_flows`（仅改 `net_flow_gbp`/`isa_net_flow_gbp`/`invest_net_flow_gbp`）。`/report` 与盘后 NAV 快照写入前查前序行；盘后快照完成后执行幂等流水校准，异常静默记日志、绝不阻断主流程。
4. **周六策略体检纠偏**（`src/bot/scheduler/weekly_audit_mixin.py`）：体检周报改用剔入金 7 日 TWR/Dietz 对比 VWRP，新增 `account_change_method`/`net_flow_7d_gbp`/`net_pnl_7d_gbp`/`flows_unverified`，摘要展示「账户7日TWR x%（净盈亏 £ / 剔净入金 £）」；流水 403 时明确提示「出入金流水不可用，暂不计算跑赢基准超额」，杜绝假 Alpha。AI 诊断上限 `450→600` 字并强制完整收尾，防末尾半句截断。
5. **单测与验收**：补齐周五雷达默认关闭/显式开启、负数提现、周末与周五 22:30 入金归集、同日覆写幂等、分账户大额入金平滑、近 30 天流水自愈（403 跳过、二次零变更、不动净值列）、周报 TWR 与假 Alpha 拦截、451~600 字诊断完整输出、C 触发器负数提现净额；`compileall`、全量离线单测与 `scripts/audit_egress.py` 均通过。

**Phase 2：盘后 0-Token 执行账单 + T+5/20 过程复盘增强与周六诊断闭环**（施工计划见 [`docs/plans/2026-11-21-5.5.10-phase2-postmarket-review-plan.md`](./docs/plans/2026-11-21-5.5.10-phase2-postmarket-review-plan.md)；与 Phase 1 合并发版，版本维持 `5.5.10`）：

6. **配置与幂等存储扩展**（`config.yaml`/`src/config.py`/`src/storage/db/{schema,recommendations_repo,portfolio_repo,memory_and_wallet_repo,ops_and_chat_repo}.py`）：新增 `postmarket_review_enabled`/`postmarket_push_on_alert_only`/`postmarket_event_search_enabled`/`postmarket_anomaly_min_move_pct`/`postmarket_max_search_symbols` 五个可独立降级开关；`pick_recommendations` 幂等补 `process_quadrant`/`holding_catalyst` 两列；新增 `daily_postmarket_reviews` 表（`uk_date` 主键）并纳入 90 天清理（保留最新 1 条）；新增 `get_snapshot_for_cycle_date`、`upsert_daily_postmarket_review`、`get_latest_postmarket_review`、`get_recent_postmarket_reviews`，`get_recommendation_stats` 增补严格排除 `SHADOW_*` 的 `quadrant_counts`。
7. **纯确定性盘后账单与 2x2 过程分类**（新增 `src/data_providers/analytics/postmarket_review.py` 并导出）：按 `(account, yahoo_symbol)` 对齐早盘/EOD 快照，股数不变仓位拆出本地股价 P&L £ vs FX 影响 £；股数变动仓位统一走 `intraday_traded` 并从当日成交提取明细，绝不伪造未实现盈亏差；内置 `GBp/GBP` 100 倍与币种一致性保护。Plan vs Actual 扫描覆盖 `HIGH_SLIPPAGE`（>+1.0%）、核心宽基豁免的 `UNPLANNED_TRADE`、Invest `MISSING_BREAKEVEN_STOP` 与 `UNPROTECTED_LOSS`；硬门控异动候选（`>=3.5%`、`>=2×ATR%`、相对基准超额 `>=2.5%` 或当日买入后回撤 `<=-3.0%`）平静日为空。`format_postmarket_review_text`（<=900 字）与 `format_postmarket_alert_text`（<=250 字）纯文本输出，自动剔除 `#`/`**`。
8. **结算反思升级与 Flash Lite 催化检索**（`src/bot/scheduler/{settlement_mixin,postmarket_mixin,core}.py`）：T+5/T+20 反思 Prompt 注入 `regime`、`MFE/MAE`、实盘成交滑点、2x2 象限与客观催化，并强制 `GOOD_LOSS` 不得建议放宽止损、`BAD_WIN` 必须警示运气陷阱；新增 `_fetch_catalyst_with_flash_lite` 复用既有 `gemini-3.5-flash-lite` + Google Search + `grounding_news_cache`（3h TTL），单次 `round_timeout_seconds=30.0`、无公告强制原样返回「无明确基本面公告，属市场/板块波动」，整段 `try/except` 静默降级；全晚共享 `postmarket_max_search_symbols` 检索预算（默认 3），异常单仅在 `|raw|>=8%` / `|alpha|>=6%` / `MAE<=-8%` 时消耗配额。修复反思 LLM 临时失败导致结算滞留 `PENDING`：改为 `reflection_lesson=""` 继续数值落库（T+20 仍可补写）。盘后账单在 NAV 兜底之后、90 天清理之前挂入同一 `22:00 UK daily_settlement`，降级/零净值组合直接跳过且不写库；仅纪律偏离或实质公告异动时附带一条 `<=250` 字静默提醒。
9. **周六诊断与下游标签闭链**（`src/bot/scheduler/weekly_audit_mixin.py`/`src/agents/report_agent/fact_sheet.py`/`src/agents/pick_agent.py`/`src/bot/handlers.py`）：周六 `_generate_weekly_diagnosis` 原始输入补齐 `shadow_stats`、`exit_rule_ab`、`adoption_stats + quadrant_counts` 与近 5 日盘后账单汇总，维持 `<=600` 字纯文本上限；`/report` 与 `/pick` 教训标签增补 `regime`、`MFE/MAE`、`process_quadrant`（旧记录平滑省略）；新增 0-Token 无锁 `/review` 指令并在 `_is_heavy_ai_command` 排除、`/help` 补说明，`/stats` 增补 2x2 过程分布行。
10. **单测与验收**：新增 `tests/test_v5510_phase2_postmarket_review.py` 覆盖本地股价/FX 拆解、`intraday_traded` 隔离、币种 100 倍保护、Plan vs Actual 四类告警、核心宽基豁免、异动三重门控与平静日 0 搜索、3 次检索硬上限、开关关闭 0 搜索、2x2 四象限、Prompt 注入、Flash Lite 超时/缓存、反思失败仍结算、旧库幂等补列、账单幂等覆写/90 天清理保留最新、影子隔离、周六诊断断层闭合、`/review` 无锁纯文本与降级快照不写库；`compileall`、全量离线单测与 `scripts/audit_egress.py` 均通过。本阶段零新增出网域名、零新增凭据环境变量、未改动既有风控护栏开关与 `.agents/skills`。
11. **发版前全量复检收口（P1 / P2）**：
    - **P1（周末重复账单与重推告警）**：`settlement_mixin.py` 与 `postmarket_mixin.py` 增补 `weekday() < 5` 工作日门控（与 `nav_snapshot` 同口径），杜绝周六/周日 22:00 `daily_settlement` 回放周五收盘价连写 3 条重复账单、连推 3 晚告警及污染周六 5 日汇总；
    - **P2-1（催化缓存隔离与日期窗）**：`_fetch_catalyst_with_flash_lite` 缓存键按 `uk_date + cache_scope + symbol` 隔离结算窗口与盘后异动窗口，结算检索 `window_label` 明确携带 `[rec_date to exit_date]` 起止日期；
    - **P2-2（无早盘基线防自减）**：`postmarket_mixin.py` 回退 `get_previous_snapshot(6.0)` 后二次校验 `morning.fetched_at_ms < portfolio.fetched_at_ms`，杜绝单快照库自减落伪 £0.00 账单；
    - **P2-3（账单口径对齐）**：`postmarket_review.py` 保本止损改用 `_round_trip_friction_pct` 按量加权校验（与 `ledgers.py` / `models_and_friction.py` 1-to-1 对齐）、保留 `GBp` vs `GBP` 大小写区分、同日往返交易净股数不变亦归入 `intraday_traded`；
    - **P2-4（周六 7 日 TWR 锚点对齐）**：`performance.py::compute_window_nav_performance` 在周末 `start_date` 无行时向前 3 天桥接上周五收盘锚点，与 `VWRP.L` 近 5 交易日基准完整对齐（含周一开盘跳空）。

### `v5.5.9`（2026-11-20 · M1→M6 全部落地，`src/version.py` 与 `pyproject.toml` 同步升至 `5.5.9`）

进攻引擎模块化升级：四态形态切换 + 右侧动量解耦 + 双通道猎物 + 右尾释放 + 资本效率与可观测闭环（施工总计划见 [`docs/plans/2026-11-20-5.5.9-offense-upgrade-plan.md`](./docs/plans/2026-11-20-5.5.9-offense-upgrade-plan.md)）。全版本六大模块（M1→M6）按序合入并严守全部既有防守红线：

**M1：四态市场状态机与零外网宽度融合**（`config.yaml` / `src/config.py` / `deployment_gate/{common,regime,triggers,ledgers,banner}` / `models_and_friction.py` / `morning_report.py` / `pick_agent.py` / `prompts.py`）：

1. **四态演进与滞回防抖**：保留既有 `"RISK_ON" / "NEUTRAL" / "RISK_OFF"` 枚举并新增 `"RECOVERY"` 早春轻兵试探态（1 槽位 / 1.0% 热力 / 0.6R，需 SPY>MA50 且 VIX<24、HY OAS<5.2% 连稳 3 天才进，破 MA50 或 VIX≥26 单日急退 `RISK_OFF`）；`RISK_ON` VIX 门槛放宽至 `<20.0` 并补齐 2 天常规退档滞回（VIX≥22 或宏观过期单日急降）；`RISK_OFF` 1 天极速冻结铁律不动。
2. **双层零外网市场宽度融合**：`compute_regime_breadth`（65% 本地池 + 35% US/UK/EU 多地基准，双层皆空返回 `None` 中性放行、绝不造假 `50.0`）；`morning_report.py` 按 `SPY` 最新 K 线日期做 `max_bar_lag_days=2` 新鲜度校验，旧海选缓存自动回退当日实时 `pool_snaps`；`pool_snaps` 与 `ohlcv_cache` 统一 `price > MA200` 口径。
3. **试探期防死锁与口径同步**：`ledgers.py` 用 `min(loss_risk_scale, regime_risk_scale)` 杜绝连损与 `RECOVERY` 连乘死锁；`models_and_friction.py` 在 `0 < scale < 1` 试探缩放期自动启用 `min_invest_pilot_order_gbp=500` 防 £700 碎单死锁；门控横幅、2 年历史回放、`/pick` 约束块与 `prompts.py` 全面同步。

**M2：右侧动量定价解耦（解除滞后卖方均值锁喉）**（`analytics/technical.py` / `portfolio_rules/{dual_philosophy,holding_eval}` / `screener/{patterns,runner,scoring}` / `pick_agent.py` / `morning_report.py` / `semantic_auditor.py` / `prompts.py`）：

1. **三处共尺与双锚拆轨**：新增唯一真源 `is_strong_right_side_leader`（非 `RISK_OFF` + MA200 上方 + BULL/PULLBACK + 周线不空 + `RS20>0 & RS60>0`，缺 RS 一律 fail-closed）与 `resolve_invest_target_anchors`，将选股/开仓 `llm_ceiling`（含 `entry+4×ATR`）与持仓 `deterministic_target`（仅 `max(mean, high×0.90)`，防合成 ATR 掩盖前瞻 R:R 恶化）拆轨，统一贯通 `scoring.py`、`/pick` 护栏与 `holding_eval.py`。
2. **全链路 RS 基准与 `.L` 校准补盲**：`TechnicalSnapshot` 增补 `rs_20d_pct / rs_60d_pct`；`runner.py` 透出 `out_bench_histories` 并采用完整分区基准循环回退，`pick_agent.py` 持仓评估与手输新票/加仓护栏、`/report` 持仓重算全部打通基准 RS；`_calibrate_target_price` 扩展 `.L` 100 倍校准并同步写入 `semantic_auditor.py` 底座与 `/pick` 事实卡。

**M3：双通道猎物扫描与选股去反转偏置**：

1. **A1 回踩 + A2 突破双通道**（防守红线零改动：A2 只解锁「可运行 /pick」提示，槽位/热力/R:R 硬门全链不动）：
   - `src/engine/deployment_gate/common.py`：新增 `TRIGGER_A2_RSI_EXCLUDE_ABOVE = 82.0`、`TRIGGER_A2_MAX_DIST_MA50_PCT = 14.0`；A1 通道 `72.0 / ±2%` 界逐字节保留。
   - `src/engine/deployment_gate/triggers.py`：`_scan_a_equity_candidates` 新增可选 `regime/benchmark_hist`（`regime=None` 时精确复现旧行为，直调旧单测零破坏）；A2 = Stage2 Minervini +（VCP 收缩 或 平台突破，量能确认内嵌两检测器：VCP 末期 ≤0.9× 缩量 / 平台突破日 ≥1.8× 放量）+ `dist_ma50 ∈ (1.5%, +14%]` + RSI<82 + 财报/除息排除窗与 A1 完全一致；**A2 无 bundle 一律 fail-closed**（堵死「突破股财报雷区失明」红线洞）；形态联动：`RISK_ON/RECOVERY` A2 优先、`NEUTRAL/UNKNOWN` A1 优先且 A2 仅放行 VCP+strong RS（复用 M2 `is_strong_right_side_leader` 同一把尺子，RS 无基准时 fail-closed）、`RECOVERY` A2 须 strong RS（对齐「仅放行最强 RS 领涨股」）、`RISK_OFF` 永不产出 A2；hits 新增 `channel/pattern` 字段，RUN 文案带 `NVDA·A2突破` 紧凑标签，形态排序后再截断 `trigger_max_symbols=3`。
   - `src/agents/report_agent/core_deployment.py` + `morning_report.py`：`_enrich_gate_candidate_bundles` 预筛接收 `histories=gate_histories`（内存数据、零新外网），**A2 名额只给本地 0 网络形态确认通过者**（Minervini Stage-2 + VCP/平台；无 RS 版 minervini 为扫描带 RS 版的严格超集，绝不饿死扫描可录取者）——堵死「牛市 60 只池子普通上涨股吃光 `[:8]` 名额、真突破股拿不到 bundle 被 fail-closed 静默丢弃」的 P1 饿死链路；同时 A1 预筛下界从 `abs()<=tol` 对齐为共享常量 `TRIGGER_A1_MIN_DIST_MA50_PCT=-2.0`（补齐 `[-2.0%,-1.5%)` 子带回踩股的财报/除息体检）；单批上限 `6→8`——**本波次唯一外网增量：同一批量 `get_bundles` 调用最多多带 2 个符号**（无新域名/字面量，`audit_egress.py` CLEAN）。
2. **Invest 选股器去反转偏置**（`src/engine/screener/scoring.py` Dim1，改动仅在 Invest 轨分支；ISA 价值轨 RSI 规则逐字节不动、`strong_rs` 对 ISA 恒 False 双重隔离）：
   - `RSI<=38` 加分 `+4.5 → +1.5`：深度超减反转归还 ISA 价值轨，防两条哲学串味（旧码下 RSI 34 反超 RSI 50 动量健康档 +1.5 分，现已反转为 -1.5 分）；
   - 满足 `Minervini Stage 2 + strong_rs` 的真领涨股取消 RSI 70~81 罚分（旧码 RSI 76 罚 -4.14），仅 `>=82` 极端超买线性轻提示（82→92 映射 -1.0→-3.0）；非领涨股零松绑（70→84 线性 -2.0→-7.0 原样保留，RISK_OFF 防御罚分原样保留）。
3. **`audition.py` 评估后不改**：PE>65 仅 -3.5/100、45~65 区间中性、海选打分无任何 RSI 项、营收增速 ≥22% 有独立 +4.5 对冲；唯一可预期效应是抬高每周底池换血率，收益不可量化——按计划「反过度设计守则」标记为不改。
4. **交付复检修正（P1）与知情注记（P3）**：
   - P1（已修）：预筛形态确认 + A1 带对齐（见第 1 条）——复现链路：`pool_syms≈60` 时仅凭粗形状即可命中 25~35 只普通上涨股，前排占满 8 个 bundle 名额后，后排真突破股 `bundle=None` 被 fail-closed 100% 静默丢弃；修复后普通上涨股零消耗名额（恒量即可证伪 VCP/平台），真突破股必得 bundle。新增 `TestV559M3P1BundlePrefilter` 3 例钉死该链路。
   - P3-1（知情不改，代码注释已载明）：`detect_breakthrough_platform` 回看 60 日，`PLATFORM` 实为「近 60 日放量破 MA60 后仍在 +14% 乖离内的右侧延续结构」，非严格当日首破；建仓仍过 Researcher/Critic/R:R 全链，且 NEUTRAL 只认 VCP。
   - P3-2（顺手统一）：A2 的 RS 基准改按 `resolve_market_benchmark` 逐候选回退（英股 ISF.L / 欧股 EXS1.DE → SPY），与 M2 选股器/持仓分区基准同尺；区域基准历史不在内存时回落 SPY（零额外请求，生产行为不变）。
5. **单测**：新增 `TestV559M3DualChannel`（11 例：双通道命中、RSI/乖离边界算术、NEUTRAL 平台拒绝 vs VCP 放行、RECOVERY strong-RS 门、RISK_OFF 冻结、无 bundle fail-closed、财报/除息窗、形态优先排序与 cap、`regime=None` 旧行为回归、gate 级 RUN·targeted 接线）、`TestV559M3P1BundlePrefilter`（3 例：牛市池 bundle 只给形态确认者 + 端到端不饿死、A1 负子带对齐、无 hist A2 形状零消耗）与 `TestV559M3ScreenerRsiBias`（3 例：leader 豁免 + ≥82 轻提示算术、非领涨旧罚分零松绑、ISA 轨逐字节隔离）；全量 `593/593` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`。注：本机全量墙钟 ~61-65s（M3 自身测试 <0.6s），`tests/run_tests.py` 默认 60s watchdog 在本机偏紧，验收全量时建议 `TEST_WATCHDOG_SECONDS=600`。

**M4：右尾释放与盈利管理**（改动前已按验收门完成离线 A/B 回测；防守红线零改动，硬止损/配额/12% 上限/槽位热力账本逐字节不动）：

1. **改动前离线 A/B 回测（384 样本同池配对 + 10000 次 bootstrap，本地无生产库故经确认改用 yfinance 拉 2y 静态 universe 67 只日线，复刻 `audit.py::_backtest_breakout_thresholds` 入场逻辑）**：
   - 阶梯保本@+15% + 锁盈地板@+25% + 沿 MA20 跟踪（LADDER-B）vs 现行纯 trailing：**盈亏比 1.20→1.28（60d 窗口）/ 1.58→1.89（120d 窗口）**，最大单笔 +57.9%→+87.7% 未被 +25% 催卖封顶；净期望 +0.09pp/+0.28pp（t=0.88/1.61，95% CI 含零）——**如实只主张盈亏比与右尾形态改善，不主张净期望显著提升**；
   - 纯 `+10%` 硬锁（无 MA20 跟踪）在 120d 下劣于纯 trailing（-0.12pp）→ 设计修正：`+10%` 仅作地板，「沿 MA20 跟踪」为主形态（数据支持的唯一变体）；
   - 含 +25% 机械催卖的旧口径上界臂平均收益并不更低，但把最大单笔封死在 +39.4%——证明催卖伤害的是右尾形态而非均值，与 M2/M3 解锁领涨股双锚目标的进攻纲领直接冲突。
2. **阶梯移损取代 `+25%` 机械催卖**（`models_and_friction.py` 新增 `emit_invest_ladder_advisories` 纯函数 + 3 常量 `INVEST_LADDER_BREAKEVEN_ADVISORY_PCT=15.0` / `INVEST_LADDER_LOCK_FLOOR_PCT=10.0` / `INVEST_TRAILING_DD_NEUTRAL_LEADER_PCT=-11.0`，全部 `getattr(strategy,...)` 可覆盖，零 config 改动）：
   - 浮盈 `+15%~+25%` 且未挂保本（对齐价 `≥成本×(1+往返摩擦)`，复用 `ledgers.py` 同一条 Financed 判定线，`_rescale_price_pair` 防 GBp 错位）：输出「推保本解锁槽位」提醒（止损上移至成本+摩擦即转 0 热力/0 槽位）；除息窗附「暂缓一日防除权低开击穿」注记；
   - 浮盈 `≥+25%` 且站稳 MA20 无放量派发且移动止盈未触发：不再加 `trim_score` 催卖，改为「止损上移至 max(成本+10% 锁盈地板, 沿 MA20 跟踪)」advisory（纯百分比表述，绝不输出绝对价防 GBp/GBP 100 倍错位）；跌破 MA20 / 放量长阴 / 移动止盈警戒已触发 → **原样保留旧催卖文案**（弱势股零松绑）；`RISK_OFF` 一律不输出进攻 advisory（冻结态零例外）。
3. **NEUTRAL 结构分层移动止盈**：`is_strong_right_side_leader`（M2 同一把尺子，RS 缺失 fail-closed）且站稳 MA20 的确认领涨仓缓冲 -8.5→**-11**（计划区间 -11~-12 取保守端）；跌破 MA20 或非领涨维持 -8.5；RISK_ON 维持 -12；RECOVERY 按计划文本限定维持 -8.5。最大增量峰值回吐 2.5pp 且被新阶梯止损兜底（+15% 后本金已保本）。
4. **「已保本仓顺势二段加仓」评估后裁剪（过度设计）**：现有 harness 无法验证（需多腿入场模拟+合并会计+配额建模，工程量大于 M4 其余部分之和，违背本模块自己的回测验收门）；单日 ≤1/周 ≤2 硬配额下每笔顺势加仓挤掉全新 A2 突破候选且趋势后段 R:R 结构性更差；现有 `add_score` + `/pick` 全链硬门已允许 LLM 有机提出加仓——确定性金字塔规则只加压力不加信息。待 M6 影子追踪统计有机加仓建议的真实 T+20 表现后再议。
5. **工程落位**：`holding_eval.py` 结构旗标（`holding_above_ma20`/`heavy_down_vol`）与 RS 计算前移（纯计算零副作用，供分层与豁免门使用）；纯函数 `_rescale_price_pair` 从 `dual_philosophy.py` 下沉至 `models_and_friction.py`（依赖方向修正，facade 导入源同步，下游 `ledgers.py` 零改动）；`prompts.py` Invest 条目同步阶梯口径（红线 #5：`semantic_auditor.py` 经核实无止盈类检查，无同步对象）；文件行数守卫触发后把阶梯块下沉为 `models_and_friction.py` 辅助函数（`holding_eval.py` 922L < 1000L 天花板）。
6. **单测**：新增 `TestV559M4RightTail` 12 例（+15% 保本 advisory 与 HOLD bias、已有保本挂单静默、RISK_OFF 冻结、除息窗注记、+25% 领涨移损 advisory vs 旧催卖、跌破 MA20/放量长阴旧催卖回归、NEUTRAL 领涨 -11 边界（-10 静默/-11.5 触发）、非领涨与 RECOVERY 维持 -8.5、RISK_ON 维持 -12、纯百分比无绝对价、profile 同步）；全量 `605/605` 通过（`TEST_WATCHDOG_SECONDS=600`），`compileall` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`；交付前 diff 自查 5 条全部「否」。
7. **交付复检 P2 修复（评审提出，已修）**：
   - P2-1：`has_breakeven_stop` 旧过滤 `side==SELL **or** STOP in type` 会把高位 SELL LIMIT 止盈单与半仓止损误判为已保本而静音 +15% 提醒——收紧为账本同口径 `side==SELL **and** STOP in type`，并逐行复刻 `ledgers.py` Financed 判定（`covered_qty >= 数量×0.99` + 按量加权对齐止损价 `>= 成本×(1+摩擦)`），两条判定线 1-to-1；
   - P2-2：`RISK_OFF/stop_flag` 早退 `return 0` 误把 v5.5.8 的 `+25%` 防守止盈提示一并截断——重构为「进攻 advisory（+15% 解锁槽位/+25% 让利润奔跑豁免）在 RISK_OFF 与止损复核态冻结，防守止盈提示全形态保留」，确保危机态绝不比 v5.5.8 更宽松；
   - 新增 3 例钉死回归（高位 LIMIT 不静音、半仓止损不静音、RISK_OFF@+30% 保留旧催卖文案），全量 `608/608` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`。

**M5：资本效率与双账户解卡**（Invest 派息宽基蓄水池 + 清宽基换股 + ISA 边角料席位解卡；防守红线零改动：硬止损/配额/12% 上限/`isa_satellite_max_total_pct=30%`/既有护栏开关逐字节不动）：

1. **蓄水池准入四锁 + 30% 防抖缓冲带**（`config.yaml`/`src/config.py` 新增 7 字段：`invest_reservoir_enabled` 总开关 / `etf_whitelist`（VWRL.L/ISF.L/VERX.L/VEUR.L 仅 DIST）/ `cash_reserve_pct=30` / `min_idle_cash_pct=50` / `min_ticket_gbp=1000` / `funding_buffer_pct=2` / `funding_min_need_gbp=200`；`VEUR.L` 补入 `universe.etf_symbols` 对齐 `CORE_SLEEVE_EU` 与 `_DISTRIBUTING_CORE_ETFS`——修复「VEUR.L 买单被 `classify_pending_orders` 误分类为 `NEW_EQUITY_BUY` 白占个股槽位」的契约洞；本波次唯一符号级外网增量：每周池刷新/盘中哨兵多观察 1 只同域名 Yahoo ETF）：
   - `ledgers.py`：Invest 账本新增 `reservoir_positions/reservoir_total_gbp/pending_reservoir_buys`（0 槽位 0 热力 0 Quota 本就由 ETF 跳过逻辑结构性成立，零新占用代码）；`triggers.py` 新增 `reservoir_advice` 准入块——锁① regime∈{RISK_ON,NEUTRAL}（RISK_OFF/RECOVERY 不新建）、锁② ISA 核心达标（≥60%）或 ISA 可用现金 ≤15%（免税账户优先）、锁③ idle>50% 严格边界、锁④ 仅 DIST 白名单（ACC 类如 VERG.L 对蓄水池不可见）；票据 = 现金 − 30%×NAV 溢出（<£1,000 不签发防碎 churn），已挂蓄水池买单去重，补仓目标 = 持仓最大者（无持仓默认 VWRL.L）；
   - `banner.py`：WAIT 默认分支 line2 尾部 + ISA RUN 分支 line1（line1 不参与 110 字符截断，防关键数字被吃）追加蓄水池短标签；DEFEND/ROTATE/RUN·targeted 分支零改动（退潮态绝不出现蓄水池字样）；
   - 快照口径回归：当前实盘（ISA core 36.34%<60 且 deployable 21.11%>15）锁②失败 → 蓄水池票据不签发、现金优先补 ISA 核心，「ISA 核心缺口优先」红线的可证伪回归已钉死。
2. **`holding_eval.py` 核心宽基 Beta 旁路（两层）**（`models_and_friction.py` 新增单源判定 `is_invest_reservoir_symbol`（白名单层，管理蓄水池准入/资金源）+ `is_invest_core_beta_symbol` / 下沉函数 `emit_invest_core_beta_bypass_note`（旁路层））：第 1 层 = 蓄水池白名单 4 只；第 2 层（衔接复检补强，开关无关）= curated 静态家族表中的**任意 Invest 核心宽基持仓**（如经 ERI 合规 DIST 通道买入的 VUSA.L/IUSA.L/VUKE.L/IMEU.L 遗留仓）——家族推断刻意 `bundle=None` 走 `_ETF_INDEX_FAMILY_MAP` 静态表，堵死「行业 ETF 名称含 S&P 500（如 IITU.L Info Tech）被名称回退拖入核心特权」的误伤链；两层均旁路 ATR 止损复核、移动止盈（-8.5/-11/-12）、M4 阶梯（+15/+25，`emit_invest_ladder_advisories` 按 `is_invest_core_beta_symbol` 早退）、前瞻 R:R 与 RSI/乖离技术面（`_evaluate_holding_technicals` 整段跳过——跳过与早退返回初始值严格等价）；非 RISK_OFF 输出 Beta 停泊说明（白名单层标注「蓄水池」，家族层标注「核心宽基ETF」），RISK_OFF 输出防御评估注记；核心宽基卖出绝不触发连损冷静期（ETF 卖出在 `evaluate_tactical_quota_guards` 连损链结构性豁免，已钉回归）；表外行业/主题/债 ETF（IITU.L）保持 v5.5.8 全套战术处置、未知符号 fail-closed 保持战术处置、ISA VWRP.L 不受影响；总开关关闭仅停用蓄水池推荐/配对/白名单层，「Invest 核心宽基按 Beta 车辆评估」的分类修正不受开关影响（家族层为分类事实）；准入票据与清宽基资金源严格保持白名单 4 只。行数守卫触发（1030L>1000L）→ 按 M4 先例下沉至 `models_and_friction.py` 辅助函数并收回 M5 多行条件 → **999L ≤ 999 达标**（语义等价经全量复跑确认）。
3. **清宽基换股（ETF-to-Equity Funding）确定性配对指令**（`sizing.py` 新增 `build_etf_to_equity_funding_note` 纯函数 + 共享 helpers）：仅当护栏后存留的 Invest 个股 BUY/ADD 票面 > 可用流动现金且蓄水池在手时触发：`need = ceil((ticket−free)×1.02)`（2% 价差/汇率缓冲），整股向上取整（GBp/GBP 100 倍错位用持仓自身币种标记防御），释放标的 = 蓄水池市值最大可覆盖者；free≥ticket 绝不触碰宽基（零多余换手）、缺口 <£200 走常规换汇路径（噪声地板，卖 ETF+CGT 显劣于 0.15% 换汇）、SWAP/ROTATE 不触发（与 `funding_from` 弱仓置换路径互斥防双资金源）、蓄水池余量不足不编造指令；同标的挂单抵扣与 `_build_deterministic_cash_note` 同口径（`max(0, planned−existing)`）防双扣；文案含零Quota/CGT 部分处置提示/GBP→USD 单边 0.15% 换汇路径。接线：`/pick` 护栏链后注入 `render_pick_report`（`schemas.py` 新增 `funding_note=""` 可选参数，sec2 追加一行，默认空串旧渲染逐字节兼容），`_format_gate_constraint_block` 新增蓄水池票据行（eligible）与零Quota 资金源行（held-only）严禁 LLM 自行选码/口算股数；`/report` 侧 `build_report_size_notes` 新增可选 `strategy` 参数（None 时与 v5.5.8 逐字节一致），仅当 top_action 的 size_hint 可解析出 £ 金额才追加紧凑配对指令（不可解析不编造数字，精确配对留给 /pick——/report 无确定性 sizing_map 的唯一不造假方案）。
4. **ISA 边角料席位解卡**：`committed_satellite_count` 排除「<£1,200 且已挂**全额**（≥0.99×持仓量，复用 Financed 同一容差）卖出限价单」的边角料（`ledgers.py` 独立扫描 SELL LIMIT——补齐 `classify_pending_orders` 只认 BUY/SELL-STOP 的分类盲区），撤销卖单下次运行自动回锁（零持久化、零 DB 迁移）；`committed_satellite_pct` 30% 权重红线一毫米不动（席位解锁后 pct≥30% 仍饱和）；`legacy_crumb_count` 与边角料归拢提示文案不变；解锁后 `isa_equity_open` 危机价值通道恢复。
5. **口径同步（红线 #5）**：`prompts.py` 全量画像「绝不催买」改写为「绝不催买**个股**」+ 蓄水池票据/配对指令/席位解锁三句；`_build_compact_pick_critic_profile` +1 行（配对指令以 Python 输出为准）；`_build_compact_chat_profile` +1 行（Beta 停车场勿按个股短线纪律点评）；`semantic_auditor.py` 复核无现金/止盈类互斥检查，无同步对象。
6. **单测**：新增 `tests/test_v559_m5_capital_efficiency.py` 52 例——A 蓄水池准入 11（四锁逐条、idle 50/50.1 严格边界、小账户票据下限、挂单去重、ACC 永不可见、多池选最大、总开关、今日快照锁②失败回归、**P2 复检：溢出票据按 available 口径不双花挂单现金**）、B 核心宽基旁路 11（-13%/RSI80/+25% 阶梯/+30% 移动止盈全旁路、RISK_OFF 防御注记、衔接复检补强 2 例：遗留 VUSA.L 核心宽基 -13% 旁路 ATR 止损且不被蓄水池资金源误纳、VUSA.L +25% 旁路 M4 阶梯、IITU.L 行业 ETF 保持 REVIEW_STOP、个股止损回归、ISA 不受影响、ETF 卖亏损连损豁免）、C 配对指令 15（need 算术/整股取整/GBp 防错、free≥ticket 零换手、噪声地板、多池选大、余量不足、挂单抵扣 £612 算术、蓄水池自买不自筹、SWAP 排除、/report size_hint 两分支与 `strategy=None` 逐字节回归、render 注入与 1000 字上限、**P1-2 复检：ISA 动作永不拼 Invest 配对指令**、**P3-1 复检：欧股走 GBP→EUR 换汇文案**）、D 席位解卡 9（全额卖限价解锁、部分卖单/SELL STOP/≥£1,200 不解锁、pct 红线仍饱和、equity_open 恢复、**P1-1 复检：跨账户同名（ISA AMZN 2.2269 股边角料 + Invest AMZN 6.3696 股）席位照常解锁且蓄水池侧不受污染、拆单全额卖出（25+15 股）累计解锁**）、E 渲染口径 6（banner 110 字符预算与 DEFEND 零蓄水池字样、ISA RUN line1 标签、约束块票据行、prompts 三画像同步）；总开关用例补强「家族分类层不受开关影响（VWRL.L/VUSA.L 恒旁路、IITU.L 恒战术）」断言；全量 **660/660** 通过（`TEST_WATCHDOG_SECONDS=600`，墙钟 ~66s），`compileall` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`；交付前 diff 自查 5 条全部「否」。
7. **交付复检修正（用户评审提出，已修）**：
   - **P1-1（实盘命中：ISA AMZN 2.2269 股边角料挂全额卖单 + Invest AMZN 6.3696 股同名持仓）**：席位解卡扫描误用全组合 `pos_by_sym`（`all_positions()` 中 Invest 追加在后、同名键覆盖 ISA）→ ISA 边角料的覆盖持仓永远解析到 Invest 行、`2.2269 >= 6.3696×0.99` 恒 False、席位静默永不解锁——改为 **ISA 专属映射** `isa_pos_by_sym`，并按标的**累计**同类 SELL LIMIT 股数（拆单全额卖出如 25+15 股也解锁）；
   - **P1-2（跨税务账户串台）**：`_report_reservoir_funding_suffix` 漏判 `act.account == "INVEST"`，会把「先卖 Invest VWRL.L £1,020」拼到 ISA 买单（如 [ISA] ADD MSFT £2,000）后——加上账户硬校验，并把标的排除从白名单扩为 `is_invest_core_beta_symbol`（核心宽基动作本身是停泊机器，永不支取蓄水池；`build_etf_to_equity_funding_note` 同步统一排除口径，防无 sizing_map 时 VUSA.L ADD 误配对）；
   - **P2（双花挂单占用现金）**：蓄水池溢出票据与 idle 占比改用 **`available_to_trade_gbp`**（对齐 ISA `deployable_pct` 口径）——挂单已占用的现金不再计入溢出，杜绝「买票据后自由现金击穿 30% 流动弹药带」（例：净值 £20k/总现金 £14k/可用 £11k → 票据 £5k 而非 £8k；可用 £9k 时 idle 45% 直接锁③拒绝）；
   - **P3（顺手校准）**：`_reservoir_fx_txt` 欧股（.DE/.PA/.AS/.MC/.MI/.SW）改印「GBP→EUR 单边换汇」（原误印 GBP→USD）；`config.yaml`/`src/config.py` 的 `invest_reservoir_enabled` 注释改为两层语义准确表述（开关只停用蓄水池推荐/配对/白名单层，核心宽基家族分类旁路不受影响）。

**M6：进攻可观测性与离线 A/B 验证闭环（防过拟合安全阀；全部纯离线、0-token、0 新增出网）**——影子拦截追踪 + 退出规则 A/B 固化 + `/perf` 分形态归因，量化「风控挡雷 vs 护栏误杀」：

1. **影子拦截追踪（Shadow Rejections，反事实对照专用车道）**：`/pick` 中被 `RR_BELOW_FLOOR / RR_UNVERIFIED / CRITIC` 硬拦截的 Top 1~2 候选轻量落库（`pick_recommendations` 同表新增 `regime` / `shadow_trigger` / `shadow_reason` 三列，`_ensure_columns` 幂等迁移，零删表零破坏旧记录），`source='shadow'` + `outcome_status='SHADOW_REJECTED'` 双重隔离，T+20 自动结算转 `SHADOW_SETTLED` 统计「护栏正确避雷率（T+20 收益 ≤0） vs 错杀大牛股率（T+20 收益 ≥+10%，`SHADOW_MISSED_WINNER_PCT`），中间带计中性」+ 按形态分桶 `by_regime`；`/stats` 增渲染一行（n<30 强制「仅参考」标注），周审计载荷带 `shadow_stats` 供 DS 诊断参考。
   - **隔离契约四连修补（防影子行反向污染实盘）**：`log_recommendations` 两条 dedupe 查询排除 SHADOW 家族——**堵死「5 天窗口内同票影子行抑制真实推荐」的进攻削弱暗洞**；`get_recommendation_stats` 的 `total_count` 排除影子行并新增 `shadow_count`（`by_account`/`adoption` 经论证天然安全：前者硬编码 `SETTLED`、后者依赖 `actual_filled` 恒 NULL）；`get_recent_recommendations`（`/stats` 最近列表）排除影子行；`/stats` 空账早退守卫扩展「影子已结 >0 也不早退」。
   - **采集侧纯函数化**（`pick_agent._collect_guardrail_shadow_rows / _collect_critic_shadow_rows / _merge_shadow_rows_cap`）：护栏侧从 `override_res.overrides` 过滤 R:R 家族触发器并以 **`final_decision.selected_actions` 差集防 advise 模式双重计数**（存活票据是真实推荐，永不进影子）；Critic 侧 = 提案白名单 ∩ 正式候选池（account 由 `track` 派生，堵死 ISA 候选误归 Invest）− 已选 − 持仓；cap 2 条/次，优先级 RR_BELOW_FLOOR > RR_UNVERIFIED > CRITIC；无正锚价一律跳过；采集失败仅 `logger.warning` 绝不影响 `/pick` 主流程；真实 tactical 行落库同步携带当日 `regime` 标签。
   - **T+20 影子结算（0-token）**：`settlement_mixin` 挂影子队列（`hasattr` 守卫兼容桩 DB），与实盘结算共用同一次 `get_histories` 批（测试断言零额外请求）、复用 `_align_entry_price_gbp_guard` 防 GBp/GBP 量纲错位、行情批失败路径同步清空影子队列（拒绝拿空 OHLC 伪造数据）、无 LLM 复盘无成交对账；`_run_daily_settlement_impl` 返回值新增 `shadow_settled`，作业文案仅 >0 时追加「影子复盘 N 条」。
2. **退出规则离线 A/B 固化（v5.5.8 基线 vs v5.5.9 M4 阶梯）**：`audit.py` 抽出共享入场预计算/判定（`_backtest_entry_series` / `_breakout_entry_ok` / `_atr_stop_pct_or_fallback`，旧 `threshold_backtest` 行为等价、旧断言零改动），新增 `_backtest_exit_rule_ab` **同批入场配对**双臂——基线臂 = 纯移动止盈（`+trailing_profit_activation_pct% / trailing_drawdown_from_peak_pct%`）+ ATR 止损；阶梯臂 = ATR 止损 → 峰值 `+invest_ladder_breakeven_advisory_pct%(15%)` 后止损抬至成本×(1+往返摩擦)保本地板 → 峰值 `+invest_take_profit_review_pct%(25%)` 后 `max(成本×(1+invest_ladder_lock_floor_pct%), MA20)` 收盘跟踪；**两臂同扣往返摩擦表**（美股/欧股 `2×fx+2×0.05=0.40%`、英股 `.L` `stamp+0.10=0.60%`、LSE 英镑 ETF `0.10%`，读现役 `strategy.fx_fee_pct/uk_stamp_duty_pct` 零新造硬编码），输出胜率/盈亏比/净均收益/**最大单笔亏损**/退出原因分布 + `delta` + `friction_model` + 幸存者偏差显式标注（静态现役底池回溯、整体收益高估、仅作规则间相对比较）+ 简化口径声明（未模拟双锚/RECOVERY 缩放/形态门控，防历史 VIX/HY 伪造后视偏差）+ `small_sample`（n<30 摘要强制「仅参考」）；周审计载荷新增 `exit_rule_ab` 键与一行纯文本摘要（无 `#`/`**`）。**只观测不回写**——A/B 结果不自动调参，杜绝「回测→调参→再回测」自我验证回路。
3. **`/perf` 分形态归因（描述性统计）**：`performance.py` 新增 `compute_regime_attribution`——日 TWR 与 `compute_portfolio_performance` 同式（剔入金），按当日 `morning_reports.context_pack_json` 的 `regime` 标签分桶（`RISK_ON/NEUTRAL/RECOVERY/RISK_OFF`），每桶输出天数/桶内复利 TWR/平均仓位/平均现金占比，桶乘积还原全窗口 TWR；无标签日单列「未标记」（晚间 NAV 任务不重算形态——那需要新增 VIX/HY 出网请求，违背 M1 零外网精神；无早报日诚实留空绝不伪造）；`portfolio_repo.get_regime_history` 只读解析 + 坏 JSON 跳过；`/perf` 渲染一行「分形态归因: RISK_ON 42天 TWR+3.1% 仓位69%现金31% | …」。**零数据库迁移**——复用既有 `morning_reports` 表。
4. **测试与验收**：新增 `tests/test_v559_m6_observability.py` 18 个纯离线用例：legacy 库幂等迁移（旧记录零破坏）/ 影子隔离矩阵（stats·最近列表·双队互斥·lessons·outcome_series 全不受污染）/ **同票反抑制回归**（窗口内影子行不挡真实推荐）/ 同日影子去重 / 避雷-错杀-中性三态统计与 `by_regime` / 三个采集纯函数（触发器家族、advise 防双计、候选 `track` 派生、cap=2）/ T+20 影子结算（0-token 断言 LLM 零调用、共用行情批、实盘计数器不动）/ A/B 双臂（暴跌对称 ATR 止损逐位相等、慢阴跌阶梯臂净收益占优、摩擦表三市场、payload 结构与纯文本规则）/ 分形态归因（手算 TWR 分桶、入金流剔除、<2 有标日不出块）/ `get_regime_history` 坏 JSON 跳过 / `/perf`、`/stats` 渲染含新行且无 `#`/`**`。**678/678 全绿**（660→678，`TEST_WATCHDOG_SECONDS=600` 约 70s）、`compileall` 通过、`scripts/audit_egress.py` `RESULT: CLEAN`、交付前 diff 自查 5 条全部「否」（零新增出网域名/零密钥打印/零新增环境变量/零 LLM 链路改动/零风控护栏开关改动）。
5. **施工中顺手排雷 3 处**：影子 INSERT 列/占位符对齐复核（12 占位符 + 3 SQL 字面量）；`log_shadow_rejections` 行内 `regime` 优先于车道兜底值；`/stats` 空账早退在「影子已结 >0」时放行（影子-only 库也展示反事实证据）。
6. **交付复检收口（P2 / P3）**：`portfolio_repo.get_regime_history` 增补 `deployment_gate.regime` 回退（零后视偏差无缝接续 v5.4.0~v5.5.8 存量早报形态）；`recommendations_repo.get_recent_recommendations` 默认无参分支同步排除影子状态族，彻底封死未来调用方误混风险；本日志补齐 M1/M2 摘要形成完整六模块封版记录。

### `v5.5.8`（2026-11-20）

日常聊天全链路瘦身、纠偏与实时校准（普通单聊 DeepSeek + `/deep` Gemini；施工总计划见 [`docs/plans/2026-11-20-5.5.8-daily-chat-plan.md`](./docs/plans/2026-11-20-5.5.8-daily-chat-plan.md)）。零新存储文件、零新增出网域名、风控护栏开关全部不动。

#### 增量合入（2026-11-20）：美股增量哨兵（`/us`）全链路真源校准、去重瘦身与防重搜

> 施工计划见 [`docs/plans/2026-11-20-5.5.8-us-sentinel-plan.md`](./docs/plans/2026-11-20-5.5.8-us-sentinel-plan.md)；同版本增量补齐 `/us` 链路（`src/version.py` 与 `pyproject.toml` 维持 `5.5.8`）。

1. **Wave 1 — 真源校准、真 Delta 门控与 0-Token 挂单对账**：
   - `market_data/yahoo_mixin.py`：`_extract_intraday_15m_quote` 新增 `bar_date_et / quote_time_et / session_tag（PRE|REG|POST）/ prev_close_date_et` 美东时效元数据；`market_data/external_clients.py`：Finnhub `/quote` 读取原生 Unix 秒级 `t` 换算美东 `bar_date_et` 并标 `source="finnhub_quote"`（`t` 缺失时不写，100% 兼容旧单测桩），双源堵死「隔日旧 `dp`/旧 15m K 线误报为今日盘前异动」。
   - `report_agent/sentinels.py`：`_evaluate_us_anomaly_gate` 升级为真 Delta 门控——休市/周末熔断（`WEEKEND/HOLIDAY` 直接平稳并跳过全部实时行情拉取）；`PRE/OPEN` 阶段仅采信 `bar_date_et == ny_date`（或无元数据桩）的新鲜报价，PRE 阶段早报已覆盖的昨收旧跌幅不再二次报警；VIX 改为相对早报基线增量（达 ±4% 或**新突破** 28 才触发，早报已 ≥28 的高位静默不再天天破门控），无早报包时完整保留日线变动 + 静态 28 兜底。
   - 补齐 `t212.refresh_portfolio_fx` 兜底（Invest 汇率缺失时零网络原位刷新）；`append_premarket_note` 增加空壳防护——仅当日确有早报正文/结论才回写，杜绝无早报日插入 `report_text=''` 空壳行抹掉上一周期兜底早报。
   - 注入两大 0-Token 实盘真源块：`【美股盘前/盘中实时快照（唯一实时价真源）】`（实时价、较昨收、时段与美东时刻；无当日成交显式写「沿用昨收」）与 `【早报后美股挂单变动对账】`（`fact_sheet.build_analysis_context_pack` 增补轻量 `pending_orders` 列表；旧包无该字段时回退比对 `symbols` 非持仓标的），显式区分「已撤销/已成交消失（无需再操作）」「新增挂单」「当前仍有效挂单（附距实时价距离）」，终结三套价格口径打架与喊撤空气单。
2. **Wave 2 — 轻量事实组装与上下文深度瘦身（Token 直降 ~55%–65%）**：
   - `prompts.py`：新增 `_build_compact_us_sentinel_profile`（`compact_for_us_sentinel=True`，~450 tokens 替代 ~1,850 全量画像），删除 ISA 四桶票据、现金催配提案、单笔金额/周月配额、连损冷静恢复与「一律按挂单限价核算」重审长段；`_has_complete_embedded_vetoes` 与 `_clean_chat_plain_text` 下沉为通用 `has_complete_embedded_vetoes / clean_telegram_plain_text`（`chat_agent.py` 保留原名别名，严禁 `sentinels.py` 反向导入 `chat_agent.py`）。
   - `sentinels.py`：以专用 `_fetch_us_sentinel_facts_data + build_us_sentinel_facts` 取代全量 `_assemble_fact_sheet_bundle`——只拉美股 `get_bundles(include_news=False, include_earnings=True)`，不再拉 `.L`/欧股行情、Yahoo RSS 新闻、最高 30 次 Finnhub 深度接口与历史教训；宏观块显式加注「HH:MM UTC 已最新拉取，禁止重复调 get_macro_snapshot」；仅对异动 ∪ 活动挂单 ∪ 7 日财报/±3 日除息标的打印紧凑技术锚点，其余合并 1 行；彻底移除跨账户暴露、`🟢 [加仓候选]` 规则引擎、Disagreement Matrix、静态评级/EPS/内部人与身份锚定块；早报已内嵌完整红线时不再重复注入 `=== 今日风控红线 ===`；System 提示改写为「仅 7 天内财报或 3 天内除息才提醒，严禁拿 15 天以上远期财报硬凑静默期」。
3. **Wave 3 — 止血重复搜索、收紧工具集、修字数上限与输出护栏**：
   - `tools.py`：`search_market_news` 缓存修复为事件感知——含 `subpoena/guidance/FDA/downgrade` 等具体事件词的定向检索在任何词数下穿透缓存，泛新闻查询（如 8 词 `"AVGO" OR "Broadcom" stock news November 20 2026`）无视词数命中当日 `sym` 级缓存；`MarketToolkit.declarations(allowed_tools=None)` 新增可选白名单（默认返回全量 4 工具，`/report`/`/pick`/单聊零改动）。
   - `/us` 异动日仅暴露 `{"search_market_news"}` 单工具 + `max_tool_rounds=1`（90% 异动日 1 轮 Hydrogen 直出，保留极端突发逃生通道），工具缓存窗口对齐 `min(grounding_cache_ttl_seconds, 1800)`；前置搜索去掉硬编码 `bypass_cache=True`，改 `ttl_seconds=900 / persist_ttl_seconds=1800`（30 分钟窗口防连敲重烧额度），Top 2 真异动票（涨跌 ≥3% 或挂单逼近 ≤1.5%）1-to-1 提纯深搜、其余 3-in-1 合并批搜、总目标 ≤6 只，活动美股挂单标的自动并入检索目标。
   - 字数契约严格兑现：`600 if 🚨 else 450`（与门控状态解耦，兼容 `test_authorized_length_ceilings`）；输出过 `clean_telegram_plain_text` 轻度纯文本清洗；打 1 行 `us_sentinel completed [phase/gate/reasons/model/prompt_chars/search_targets/reply_chars]` 标准遥测。
   - 单测：修补 `test_imp12` 显式 patch `get_us_market_schedule_uk` 为确定性工作日 `OPEN` 时段（防周六日/假日 CI 误挂）；新增 `tests/test_v558_us_sentinel.py` 10 例覆盖盘前双源时效、VIX 增量门控、休市熔断、挂单对账与检索并入、空壳行防护、事件感知缓存穿透、单工具收紧与 I/O 隔离、红线去重、字数契约与纯文本清洗、精简画像；全量 `560/560` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`。

1. **Wave 1 — 逻辑纠偏、跨周期隔离与 0-Token 盘中实时结构校准**：
   - `storage/db/portfolio_repo.py`：新增向后兼容 `get_morning_context_meta()`（单查询返回 `static_report/premarket_note/conclusion/stale/cycle_date`，替代旧的双次 `get_morning_context`）与 `get_latest_pick_radar_with_ts()`（带 `created_at_ms`）；原方法签名 100% 保持。
   - `chat_agent`：`/pick` 时效改用统一交易周期判定（`created_at_ms → db._cycle_id` 与当前 07:30 UK 周期比对），跨周期旧提案一律跳过；严禁用 `pick_created_at_ms < morning_updated_at_ms` 误杀当日 `/pick`（下午 `/us` 的 `append_premarket_note` 会覆写 `updated_at_ms`）。`stale=True` 时早报标题显式改为「【上一周期陈旧早报（今日晨报尚未生成，仅作参考）】」。
   - 修复 `【今日风控红线】` 100% 重复注入：`_has_complete_embedded_vetoes` 对早报正文与每条 guardrail 做统一标点规范化（去除末尾 `。；.;` 与空白）后再比对，消除「早报 `；` 拼接 vs guardrail 句号」导致的恒 False。
   - 单标的误标修复：`_COMPARE_HINT_RE` 移除口语高频词「还是」，`decision_kind="compare"` 仅在解析出 ≥2 只标的时触发；`_build_symbol_alias_index` 内置轻量高频中文别名表（微软/英伟达/亚马逊/谷歌/博通/联合健康/摩根大通/优步/特斯拉/苹果/台积电/阿斯麦/阿斯利康/联合利华/先锋全球等），仅注册 portfolio ∪ ContextPack 实际存在的标的；「标普/富时/纳指/欧股」泛指大盘词严格不绑定任何单只 ETF。
   - `/deep` 前缀剥离（含可选 `@botname`）与空参保护：`clean_query` 用于标的解析、Prompt 末尾与 `chat_history` 落库；空 `/deep` 直接返回用法提示（0 次 LLM 调用、不写库）。
   - 注入 0-Token `【盘中实时结构校准（HH:MM UK…）】`：严格复用 `deployment_gate/ledgers.py` 的 `classify_pending_orders + compute_account_ledgers`（与 `/report`、`/pick` 同口径），用 `context_pack.symbols + 持仓名` 零网络构造 `lite_bundles`（防 `IBTA.L` 等非核心短债 ETF 误计入 ISA 个股卫星），输出 ISA 核心宽基（含挂单后）/个股卫星/现金 + Invest 在险槽/保本/现金 1–2 行实盘真相；整段包在 try/except 内静默降级。
   - 零网络 `refresh_portfolio_fx` 兜底：仅当 Invest 汇率缺失时，从内存 `_macro_cache` 或 ContextPack macro 提取 `gbp_usd>0` 原位刷新，严禁聊天热路径新增阻塞外网请求；ContextPack 切片标题加注「早报静态快照，现价与挂单状态以实盘为准」，ETF 切片过滤不含标的/指数关键词的雅虎杂讯新闻（如 Kenya oil rig）。
2. **Wave 2 — 上下文瘦身、前缀缓存重排与长回复防失忆**：
   - `prompts.py` 新增 `_build_compact_chat_profile`（`compact_for_chat=True`）：日常聊天专属精简画像，利息/动态 R:R/零 Quota/不催买各收束为唯一 1 句，剔除连损冷静恢复半仓与 4 桶政策票据长句等 `/report`/`/pick` 专属细则（`systemInstruction` ~1,900 → ~1,000 tokens）；「一律按挂单限价核算」反迎合审计原则保留。
   - 消除重复：移除动态区顶部 `build_wallet_context_text(compact=True)` 现金行（双账户块已完整含现金/挂单占用/双钱包拆解），顶部仅补标 `GBP/USD 实时汇率`；`【本周期组合基线】` 在早报正文已含集中度/敞口摘要时不再重复打印底层敞口，并过滤 `[SUPPORT_VS_WEAK_TREND 合并 xN]` 纯占位泛化行（仅保留 STOP_VS_EX_DIV 等具体冲突与 T+N 反思）。
   - 跨周期过渡记忆移入 `dynamic_tail`：仅当今日对话轮数 == 0 时保留最近 1 条「[上一周期]」问答作为桥梁（放【今日对话记录】开头），保缓存前缀逐字节稳定且上午多轮后不再夹带昨晚过期问答。
   - `/deep` 非焦点持仓紧凑化：新增 `_format_compact_positions_table`，焦点标的 + 权重 ≥15% 或浮亏 ≤-10% 异动仓保留完整明细，其余常规仓一行速览（`• CODE [占比% | 浮盈% | 现价 CCY]`），省 ~550–700 tokens；普通单聊维持原「异动保留 + 其余合并 1 行」行为。
   - Prefix Cache 重排：静态前缀顺序改为 早报 → 去重红线 → 本周期基线 → 本周期 `【今日选股雷达】`（置于静态前缀末尾，下午 `/pick` 重跑只击穿该段之后的缓存）；前哨补充维持 dynamic_tail 前沿。
   - 长回复「首尾保留（Head+Tail）」：`_format_chat_history_compact` 最近 1 轮回复 ≤550 字全文保留，超长回复保留前 300 字（结论）+ `…（中略）…` + 后 300 字（执行预案/价位/股数），更早轮次 320 字精简、早期轮次单行摘要。
3. **Wave 3 — 工具边界松绑、纯文本护栏与轻量遥测**：
   - `chat_sys` 工具许可补充：盘中实时大盘/指数涨跌幅可直接调用 `inspect_ticker`（支持 Yahoo 指数代码 `^FTSE`/`^STOXX`/`^GDAXI`/`^GSPC`）或 `search_market_news`；`tools.py` 同步补 `inspect_ticker`/`search_market_news` 描述与参数示例。
   - `MacroSnapshot` 零额外网络请求增补 `spy_1d_pct/qqq_1d_pct/ftse_1d_pct`（直接从已拉取的 SPY/QQQ/ISF.L DataFrame 最近两根 Close 计算）并输出到 `to_text()`。
   - 聊天出口轻量纯文本护栏 `_clean_chat_plain_text`：仅剥离 `**` 与行首 `#{1,6}\s+` 标题符（保留单个 `*` 如 `3*ATR`、换行、`•` 与【】），绝不截断。
   - 每轮聊天后打一行结构化 `logger.info`（mode/model/target/prompt_chars/static_chars/reply_chars），不新增任何磁盘文件或配置项。
   - 单测：新增 `tests/test_v558_daily_chat.py` 共 23 例（红线去重、还是/中文别名/泛指大盘不绑定、跨周期 `/pick` 隔离与当日 `/us` 后存活、陈旧早报标牌、`get_morning_context_meta`、IBTA.L 不误计卫星、无 `refresh_portfolio_fx` 容错、空 `/deep` 零消耗、首尾保留、纯文本清洗、前缀逐字节稳定、1d 零请求）；`test_db_and_end_to_end.py` 顺序断言同步为 `早报 < 红线 < 基线 < 选股雷达 < 时间`；全量 `550/550` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`。

### `v5.5.7`（2026-11-19）

`/pick` 全链路重构（Researcher + Critic + 确定性风控闭环），施工总计划见 [`docs/plans/2026-11-19-5.5.7-pick-pipeline-plan.md`](./docs/plans/2026-11-19-5.5.7-pick-pipeline-plan.md)。

1. **Wave 1 — 确定性真源统一**：
   - `src/config.py` 新增 `invest_vix_tighten_rr_threshold=22.0`；`dual_philosophy.resolve_invest_dynamic_rr_floor` 纳入 VIX（NEUTRAL+VIX≥22 → 2.5），并透传至 `holding_eval`、`sizing`、`risk_override`、`deployment_gate.evaluate_deployment_gate` 与早报 FactSheet，实现早报/Researcher/Critic/Python 四处同一门槛。
   - `pick_agent` 对齐 `/report` 调用 `refresh_portfolio_fx` 后再构建现金底座，根除 `GBP/USD 暂缺`。
   - `deployment_gate/triggers._scan_a_equity_candidates` 新增右侧硬门槛（`dist_ma200_pct>0`、`weekly_bullish is not False`、回踩区间 `[-2%, tol]`），拒绝假右侧猎物。
   - `_format_gate_constraint_block` 同步 `approved_report_core_gbp` 与实时活动买单：不再重复签发核心宽基票据，已挂单标的标注“已挂单执行中，勿重复建仓”。
2. **Wave 2 — 选股漏斗与 DeepSeek 初筛**：
   - `run_quant_screener` 新增通道短路参数（`enable_isa_equities` / `enable_isa_etfs` / `enable_invest_track` / `excluded_symbols` / `out_bundles` / `out_intel` / `out_tech`），ISA 通道关闭即跳过 Stage-2/搜索，15 天冷却股前置剔除。
   - `scoring.py`：Invest 英股取消 `+2.5` 免换汇改为 `-1.0` 印花税摩擦、新增前瞻 R:R 可达性降权（8~15 分）、Invest `dim1` 顶格 24~30 仅留给强右侧领涨；`_prefilter_row` 增补行业/地区、目标空间、ATR14、60日RS、预估R:R、安全边际/FCF。
3. **Wave 3 — Researcher 上下文与 1-to-1 深搜**：
   - `_format_summary_line` 增补 `[行业/地区]`、ATR14、周线/ADX、60日RS、量比、CYQ 均成本、结构支撑位与 `.L` 校准后的卖方区间，删除每行重复 `广度`。
   - `run_forced_market_grounding` 为 `scope="pick"` 新增 30 天反方深搜与独立缓存键 `grounding:{date}:pick_deep:{sym}`（上限 6 只），`scope="report"` 提示词与 `prio` 缓存键一字未动。
   - 早报裁决按实时持仓/挂单差集过滤；ISA 个股通道关闭时折叠绿标加仓持仓；移除 ContextPack 冲突摘要与重复身份段、复用 Stage-2 bundles/deep intel；`/pick` 工具集移除 `get_macro_snapshot`；Researcher 输出收紧至 ≤1,200 字、分轨授权 0 提案、5 要素含 1 句 Pre-Mortem。
4. **Wave 4 — Critic 一手事实卡与去噪**：新增 0-Token 白名单事实卡（严格交集候选/`_inspect_cache`/持仓，上限 5 只，含现价/ATR/RSI/均线/相关性/财报/除息/卖方/摩擦/ISA 价值门预判）；删除重复双钱包块；偏好记忆开关同时门控读取；Critic 画像改为 `compact_for_pick_critic` 精简版，六重护栏压缩去重。
5. **Wave 5 — Schema 洁净化与统一风控数学**：
   - `PickDecision.stop_price/target_price` 改为 `float | None`，新增 `planned_entry_price/planned_amount_gbp`；`planned_entry_price` 经 `.L` 校准并钳位 `[现价×0.92, ×1.03]` 且 `> stop`，入库 `entry_price` 永远锁死市场现价。
   - 净 R:R 与以损定仓统一用 `effective_risk_pct = risk_pct + friction_pct`；`stop_too_tight` 仍用纯价格 `risk_pct` 对比 `max(1.5×ATR, 4%)`。
   - 碎单（`below_min_order`）在 hard 模式硬拦截并区分“止损过宽/单仓余量不足”；Invest 已有未成交买入挂单的同票重复推荐硬拦截。
   - `compute_trade_behavior_stats` 支持 `exempt_symbols` 剔除零 Quota 核心宽基，并按 `(账户, 标的, 日期)` 归并同日拆单；`_cadence_violation` 在 `partial=True` 且 `coverage_days>0` 时仍硬拦 15 天清仓冷却（7/30 天频次保持软提示，纯 `{"partial": True}` 极简桩保持原样）。
   - 新增 `_build_deterministic_cash_note`：Python 接管执行后精确余现，已挂单不重复扣减，拦截前后账目一致。
6. **Wave 6 — 渲染与质检补盲**：`_reconcile_pick_headline` 识别被拦标的的中文宽基简称并重建“风控筛后执行…其余现金待命”；空仓文案移除硬编码 `R:R≥2:1`；`build_fact_baseline(scope="pick", candidate_bundles=...)` 补齐新股现价与币种锚点；版本同步 `5.5.7`。

### `v5.5.6`（2026-11-19）

LLM 工具调用末轮强制收口、`/stats` 账本 5 日跨源去重与存量清理、盘中哨兵静默冷却与侧车思考降档（规划见 [`docs/plans/2026-11-19-5.5.6-hardening-plan.md`](./docs/plans/2026-11-19-5.5.6-hardening-plan.md)）。

1. **Wave 1 — LLM 工具调用末轮强制收口**：
   - `src/agents/llm_client.py`：Gemini `_run_model_once` 在存在函数声明工具且进入末轮（`round_idx == max_tool_rounds` 或无 `tool_executor`）时保留 `tools` 并注入 `ToolConfig(function_calling_config=FunctionCallingConfig(mode=NONE))`（不影响 `GoogleSearch` 联网搜索）；DeepSeek `_run_deepseek_once` / `_call_deepseek_single_round` 末轮同步切换 `tool_choice="none"`；若末轮仍只返回工具调用且无文本，报错明确记录 `末轮仍返回工具调用(...)且无文本`，根治 `gemini-4-hydrogen` 末轮误判空文本浪费 Token 并降级的问题。
2. **Wave 2 — `/stats` 推荐账本 5 日跨源去重、存量清理与口径对齐**：
   - `src/storage/db/recommendations_repo.py` & `src/storage/db/schema.py`：`log_recommendations` 新增 `dedupe_window_days` / `dedupe_cross_source` 参数在 SQLite 单写锁内完成批内 + 跨源原子去重；新增幂等方法 `dedupe_existing_pending_recommendations(window_days=5)` 并在 `AdvisorDatabase.init()` 启动时自动执行，清理存量 5 日内重复的未结算 `PENDING` 战术副本（保留最早条、绝不动已结算历史与 `core_dca`）；`get_recent_recommendations` 支持 `exclude_core_dca` / `only_core_dca`。
   - `src/agents/pick_agent.py` & `src/agents/report_agent/morning_report.py`：战术推荐统一走 `dedupe_window_days=5, dedupe_cross_source=True`（消除早报与 `/pick` 5 分钟连写双份记录）。
   - `src/bot/handlers.py`：`/stats` 顶部样本总览补充 `核心定投归档 N 条`；下方「最近 10 条」严格过滤 `core_dca` 与顶部战术口径对齐并补 `[早报]`/`[选股]` 来源标签；核心宽基定投拆至独立「【近期核心宽基定投归档（不计入战术胜率）】」区块（最多 3 条）。
3. **Wave 3 — Token 用量止血**：
   - `src/config.py` & `config.yaml`：新增 `intraday_quiet_cooldown_minutes: 150` 与 `intraday_quiet_retrigger_delta_pct: 1.5`；`ds_helper_thinking_level` 由 `"HIGH"` 降为 `"MEDIUM"`（仅作用于轻量侧车 JSON 节点，不影响主研判链路）。
   - `src/bot/scheduler/core.py` & `src/bot/scheduler/intraday_mixin.py`：`intraday_alert` 判定 `should_push=false` 后写入内存静默冷却（不占用当日 `intraday_alert:{sym}` 推送锁；冷却期内价格再额外波动 `>=1.5%` 立即破除冷却复评），消除每 30 分钟反复空转。
4. **Wave 4 — `/report` PM 链路真源统一、杂音清理与排版修复**（`report_thinking_level` 维持 `HIGH`，全部风控护栏开关不变）：
   - `src/agents/report_agent/debate.py`：R3 `factcheck` 改为与 Bull/Bear 共用完整 `shared_context or fact_sheet`（双账户现金/市值、频率冷却、挂单、票据、搜索新闻），根治真数据被误标 `[无据]`；`_build_debate_digest_prompt` 删除「关键原文」复述要求，并对裁剪后的 `digest` 清理尾部孤立 Markdown/`【】` 标题行；串行预算 `remaining <= 10ms` 视为耗尽（对齐 Windows 单调时钟精度，消除 `never awaited` 前构造抖动）。
   - `src/data_providers/t212_client/client.py` & `src/agents/report_agent/morning_report.py`：新增纯同步 `T212Client.refresh_portfolio_fx(portfolio, gbp_usd_rate)` 并复用 `reconcile_invest_wallets` 原位回填 Invest 双钱包；`/report` 拿到 `macro.gbp_usd` 后立即刷新局部 `portfolio`（Mock 无该方法时回退 `set_gbp_usd_rate`），消除冷启动「GBP/USD 暂缺 / 待汇率折算」。
   - `src/agents/report_agent/morning_report.py`：将 Python 确定性 `Regime`、判定理由、`resolve_invest_dynamic_rr_floor` 动态 `R:R` 门槛与 ISA/Invest 门控状态前置注入 `shared_context`，硬纪律禁止 Bull/Bear/PM 自行改判 Regime；`analyst_sys` 硬编码 `R:R≥2:1` 对齐为当日动态门槛；PM Prompt 移除不可满足的「2100~2500 字」，改以 Schema 字段上限为准（渲染层控 `<=2800`），并明确 `top_actions` 优先实质调仓、全观望时仅 1 条 `CASH HOLD`（或至多 1 条重点 `HOLD`），其余裁决移入 `flagged_positions`/`hold_summary`。
   - `src/engine/deployment_gate/core_policy.py` & `src/agents/report_agent/core_deployment.py`：核心票据新增 `currency` 字段，`_format_core_policy_ticket_block` 价格/MA50/MA200 显式带币种，`GBp` 额外附注 `(≈£x.xx)` 防 100 倍量级歧义。
   - `src/agents/prompts.py` & `src/agents/schemas.py`：ISA 核心-卫星旧文案「统一归拢 VWRP.L」改为四桶宽基政策口径；`core_deployment` 描述对齐 `0~1 条（仅审批 Python 唯一票据）`；`render_morning_report` 在 ISA 核心部署/动作之后插入 `【活动挂单裁决】` 子标题，`daily_vetoes` 拼接前 `strip().rstrip("。；; ")` 消除 `。；` 连用。
   - `src/data_providers/t212_client/context_formatting.py`：`build_trade_context` 顶部汇总保留 `现金利息` 总额，`merged_events` 明细过滤 `INTEREST_ON_FREE_CASH` / `LENDING_INTEREST`，避免每日小额利息挤占真实出入金/派息。
   - `src/storage/db/portfolio_repo.py`：`get_past_decision_context(limit=5, *, exclude_cycle_date=None)` 支持过滤同日旧晨报（`/report` 重跑不再自我锚定），并对每条结论在 140 字边界优雅截断补 `…`；`generate_morning_report` 安全探测当前 `cycle_id` 并兼容不接受新参数的测试 Mock。
   - 单测：`test_report_debate_round2` / `test_wallet_rules_and_commands` / `test_report_pick_gate` / `test_t212_client` / `test_db_storage` 新增 12 条离线用例（R3 底座统一与回退、摘要去空尾、冷启动汇率回填与 Mock 回退、Regime 真源注入、GBp 币种、ISA 挂单分区与标点、利息明细过滤、同日 Memory Log 过滤与截断）；全量 `499/499` 通过，`scripts/audit_egress.py` `RESULT: CLEAN`。

### `v5.5.5`（2026-11-19）

稳定性加固（yfinance 401 / Finnhub 429 / DeepSeek 高峰期串行）与 Invest Portfolio Heat 门控升级（规划见 [`docs/plans/2026-11-19-5.5.5-stability-and-invest-heat-plan.md`](./docs/plans/2026-11-19-5.5.5-stability-and-invest-heat-plan.md)）。ISA 价值轨 0.5% 风险预算零影响。

1. **Wave 1 — 日志告警修复**：
   - `src/data_providers/market_data/yahoo_mixin.py`：新增 `_reset_yfinance_crumb_cache()` 防御性清理 `yf.data.YfData()` 的 crumb/cookie；`_run_yf_call_with_retry` 识别 `401 / invalid crumb / unauthorized` 并清令牌后短退避重试，保留顶级 `yf.Ticker` / `time.sleep` 的 `mock.patch` 接缝。
   - `src/agents/schemas.py`：`render_pick_report` 将 Python 确定的护栏行从「本轮精选行动」拆到独立「【风控护栏记录】」区块，仍严格 `<=1000` 字且纯文本；`src/agents/semantic_auditor.py` 红线第 3 条明确豁免该区块，消除 `/pick` 误报「内部驳回外泄」。
2. **Wave 2 — Finnhub 限流分流**：
   - `src/data_providers/market_data/external_clients.py`：`FinnhubClient` 新增 `max_concurrency=2`、`min_interval_seconds`（生产 0.35s / Mock 0.0）、`earnings_cache_ttl_seconds=21600`、`retry_429_delay_seconds`；全部 GET 走 `_throttled_get`（信号量 + 最小发包间隔 + 429 解析 `Retry-After` 限幅 `[1,5]s` 重试 1 次，且锁/信号量按 event loop 懒初始化）；`calendar/earnings` 增加 6h 内存缓存与 in-flight 合并，仅 HTTP 200 写缓存（429/异常/降级不回写）。
   - `src/data_providers/market_data/service.py`：`get_us_deep_intel_batch` 默认并发由 `4` 收敛为 `2`。
3. **Wave 3 — `/report` Step 2 高峰期串行**：
   - `src/config.py` / `config.yaml`：新增 `report_debate_sequential: True`；`report_step2_timeout_seconds` 660→900、`round2` 180→300、`round3` 120→210。
   - `src/agents/report_agent/debate.py`：新增 `_run_named_notes_sequential(coro_factories, *, timeout_s, step_ceiling_s=None)` 延迟工厂串行执行（单步预算 `min(ceiling, remaining)`，预算耗尽不构造后续协程，无 `never awaited` 告警），R2/R3 默认串行、`False` 可回滚并发。
4. **Wave 4 — Invest Portfolio Heat 主闸与 0.75% 风险预算**：
   - `src/config.py` / `config.yaml`：新增 `invest_risk_budget_pct: 0.75`、`invest_max_heat_risk_on/neutral/risk_off_pct: 3.0/2.0/0.0`；`invest_slots_risk_on/neutral` 默认 2/1→3/2。
   - `src/engine/portfolio_rules/models_and_friction.py`：`compute_risk_based_size_gbp` 仅 Invest 分支改用 `invest_risk_budget_pct`（保留 £85 地板与 0.5R 复出），ISA 分支严格维持原 `0.5% NAV`。
   - `src/engine/deployment_gate/ledgers.py`：新增 `unit_risk_gbp` / `pending_heat_gbp`（未成交新开个股买单 ×1R）/ `committed_heat_gbp` / `committed_heat_pct`，保留 `heat_gbp_total` 原语义。
   - `src/engine/deployment_gate/triggers.py`：按 regime 写入 `heat_cap_pct` / `heat_headroom_pct`，槽位满或承诺热力触顶统一进入 `ROTATE` 或 `WAIT(heat_full)`；`RISK_OFF`（DEFEND）/ 冷静期 / 单日上限优先级不变。
   - `src/engine/deployment_gate/banner.py` / `src/agents/pick_agent.py`：RUN 横幅与 `/pick` 约束块动态展示 `unit_risk_gbp` 与承诺热力（移除硬编码 £85），`_apply_gate_channel_constraints` 增加 RUN 态热力触顶兜底拦截。
   - `src/agents/prompts.py`：以 `max(£85, Invest净值×0.75%)` 表述单笔风险预算；钱包与核心策略提示词补充「按风险算出的美股金额超出 USD 余额时正常由 GBP 换汇补足，严禁为省换汇费缩仓」。
   - 单测：新增/对齐 `test_market_data` / `test_ds_pick_prefilter_and_auditor` / `test_report_pick_gate` / `test_report_debate_round2` / `test_report_step2_budget` / `test_analytics_and_rules` / `test_wallet_rules_and_commands` 覆盖 401 重试、护栏拆区、429 与缓存、串行单飞、Invest £14k→£105 / £30k→£225 / ISA £58k→£290 隔离、热力触顶与承诺热力；全量 `481/481` 通过。

### `v5.5.4`（2026-11-19）

核心模块结构瘦身（纯结构重构、零业务逻辑变更；规划归档见 [`docs/plans/2026-11-19-5.5.4-module-slimdown-plan.md`](./docs/plans/2026-11-19-5.5.4-module-slimdown-plan.md)）。采用**子包门面模式（Subpackage Facade）**将全部 9 个突破 2,100~2,851 行的核心文件（共 22,820 行，占 `src/` 74%）拆解为 47 个高内聚领域子模块 + 9 个兼容门面，最大子模块降至 929 行，全部外部调用方与原有 459 个单测零改动。

1. **Wave 1 — 纯确定性计算层**：
   - `src/engine/deployment_gate/`（原 2,109L）：拆为 `common` / `core_policy` / `ledgers` / `regime` / `triggers` / `banner`（180~539L），形成严格无环依赖图。
   - `src/data_providers/analytics/`（原 2,482L）：拆为 `technical` / `exposure` / `correlation` / `performance`（339~845L），门面保留 10 个跨模块私用符号。
   - `src/engine/screener/`（原 2,487L）：拆为 `patterns` / `scoring` / `runner` / `audit` / `audition`（385~664L）。
2. **Wave 2 — 存储仓与核心风控层**：
   - `src/storage/db/`（原 2,851L）：以 Domain Mixins 拆为 `core` / `schema` / `portfolio_repo` / `recommendations_repo` / `ops_and_chat_repo` / `market_cache_repo` / `memory_and_wallet_repo`（87~661L）；仅 `core.py` 直接操作 `aiosqlite`，严守 `_connect(write=True)` 单写锁铁律。
   - `src/engine/portfolio_rules/`（原 2,739L）：拆为 `models_and_friction` / `dual_philosophy` / `sizing` / `holding_eval` / `risk_override`（224~929L），并无损提炼 `_evaluate_holding_technicals` 及 `_build_override_portfolio_context` / `_apply_report_overrides` / `_apply_pick_overrides` 阶段函数。
   - `src/data_providers/t212_client/`（原 2,446L）：拆为 `models` / `wallets_and_tickers` / `http_mixin` / `client` / `context_formatting`（175~795L），HTTP/限流/重试逻辑原样平移。
3. **Wave 3 — 高 `mock.patch` 耦合与异步编排层**：
   - `src/data_providers/market_data/`（原 2,293L）：拆为 `models_and_parsing` / `external_clients` / `yahoo_mixin` / `service`（394~743L），完整保护 26 处 `yf`/`time` 补丁接缝。
   - `src/agents/report_agent/`（原 2,601L）：拆为 `fact_sheet` / `debate` / `core_deployment` / `morning_report` / `sentinels`（437~795L），无损提炼 `_build_morning_pm_prompt` 与 `_persist_morning_report_and_audit`，经门面运行时解析保护 4 处外部函数补丁接缝。
   - `src/bot/scheduler/`（原 2,812L）：拆为 `settlement_math` / `fill_receipts` / `settlement_mixin` / `weekly_audit_mixin` / `intraday_mixin` / `core`（344~650L），经门面运行时解析保护 4 处外部函数补丁接缝。
4. **防线单测与版本同步**：新增 `tests/test_v554_module_facades.py`（验证 9 个子包门面导出、`<1000` 行上限、`db` 单写锁铁律与 `mock.patch` 接缝）；`src/version.py` / `pyproject.toml` 升至 `5.5.4`，同步更新 `AGENTS.md` 模块索引；全量回归 **463/463 全绿**，`scripts/audit_egress.py` `RESULT: CLEAN`。

### `v5.5.3`（2026-11-19）

单聊上下文对齐、双哲学盘中哨兵、分账归因与实盘成交反哺、LLM 可靠性与 ISA 价值因子覆盖率可观测性。

1. **单聊早报上下文对齐（`src/agents/chat_agent.py`）**：`answer_user_chat()` 早报切片 `2050 → 2800`，与 `v5.5.2` 的 `/report<=2800` 对齐，避免 Invest 区块尾部与 `[🛡️风控护栏修正]` 被截断。
2. **双哲学差异化盘中哨兵（`src/config.py`, `config.yaml`, `src/bot/scheduler.py`, `src/agents/report_agent.py`）**：新增 `intraday_isa_move_alert_pct=5.0`；哨兵按标的所属账户分流（跨账户双持按更敏感的 Invest 口径），Invest 无 `STOP`/`STOP_LIMIT` 裸露仓位在浮亏跌破自适应止损线（`invest_stop_loss_alert_pct` 与 14 日 ATR 钳位 `-clamp(2.8×atr,8%,18%)` 取更早触发者）或浮盈后自 60 日高点回撤触及移动止盈线时，即使单日未达 3.5% 也预警；ISA 普通个股改用 5.0% 噪音阈值，ISA 核心宽基新增 52 周高点 `core_etf_drawdown_steps_pct` 阶梯通道。`generate_intraday_alert()` 新增 `account_context`/`trigger_reason`（带默认值向后兼容），`morning_slice` `1200 → 1800`，仍限 `<=250` 字纯文本。
3. **Invest USD 成交回执只读对账参考（`src/bot/scheduler.py`）**：`format_fill_receipt()` 新增 `usd_wallet_available`，Invest 且 USD 成交时追加「若走 USD 钱包结算，参考余额约 $xxx，核对后可发送 /usd xxx」，**绝不自动改写 `wallet_balances`**；`_collect_order_fill_receipts()` 安全读取 USD 锚点，异常静默降级为 `None`，总字数仍受 `FILL_RECEIPT_MAX_CHARS=450` 限制。
4. **分账户外部现金流与分账 TWR（`src/storage/db.py`, `src/data_providers/analytics.py`, `src/bot/handlers.py`）**：`nav_history` 幂等补列 `isa_net_flow_gbp` / `invest_net_flow_gbp`（默认 0，旧行自动兼容）；`build_nav_row()` 原 `net_flow_gbp` 不变并分账户累加；新增 `compute_account_split_performance()` 分别复用 `compute_portfolio_performance()` 口径输出 ISA/Invest 双 TWR、最大回撤与样本数。
5. **四桶核心政策归因接线（`src/bot/handlers.py`, `src/bot/scheduler.py`）**：`/perf` 输出「分账户归因」并展示「四桶核心政策（60日）：组合 X% vs VWRP Y%（区域倾斜效应 ±Z% | 跟踪误差 T%）」；`_build_benchmark_dashboard()` 同步写入 `core_policy_benchmark` 与摘要，周六离线体检与 DeepSeek 周度诊断可区分区域倾斜 Beta 与选股 Alpha。
6. **推荐台账实盘成交对齐与反思分层（`src/storage/db.py`, `src/bot/scheduler.py`, `src/bot/handlers.py`）**：`pick_recommendations` 幂等补列 `actual_filled` / `actual_fill_price` / `execution_slippage_pct`；结算时按同账户+同标的+同方向在 `[rec_date, exit_date]` 匹配 T212 真实成交（经 GBp/GBP 100 倍对齐后计算执行滑点），命中置 `fill_status=FILLED`，账户历史可用但未命中记 `actual_filled=0`，历史不可用保持 `NULL` 不误判。ISA `BUY/ADD` 的 T+5 不再生成短期反思（避免 5 日噪音污染巴菲特纪律），改在 T+20 结算时若 `reflection_lesson` 为空再补写；Invest 保持 T+5 即时反思。`get_recommendation_stats()` 新增 `by_account` 与 `adoption_stats`，`/stats` 展示分账户表现与实盘采纳率/平均滑点，旧字段全部保留。
7. **0 改表可观测性聚合（`src/storage/db.py`, `src/bot/handlers.py`）**：新增 `get_llm_reliability_summary()`（近 N 天总体与分模型/分节点 `calls`/`failed_calls`/`success_rate_pct`/P50/P95，Python 端算分位避免 SQLite 扩展）与 `get_ops_quality_summary()`（语义质检 report/pick 分类、`[事实价不符]`/`[建议价方向异常]`/`内部驳回` 标签、degraded/fallback 降级计数）；`/usage` 追加「近 7 天模型可靠性」与「近 7 天质检与降级」两小节，`/health` 追加一行摘要，查询失败一律安全忽略。
8. **ISA 价值因子多市场覆盖率体检（`src/engine/screener.py`, `src/bot/scheduler.py`）**：新增纯函数 `compute_fundamentals_coverage()` 按 US/UK(.L)/EU(.AS/.DE/.PA) 统计非 ETF 标的 `roe_pct`/`debt_to_equity`/`revenue_growth_pct`/`fcf_yield_pct`/`peg_ratio`/目标价覆盖率并列出 `<50%` 风险；`run_offline_universe_and_threshold_audit()` 新增可选 `fundamentals_rows`；周六审计从 `db.get_fundamentals_cache()` 读取并注入 `_generate_weekly_diagnosis()` 原始数字（仍 `<=450` 字且禁止编造）。
9. **文档与版本**：`src/version.py` / `pyproject.toml` 升至 `5.5.3`，`DEVELOPER_GUIDE.md` 对齐 `/report<=2800`、四桶核心宽基、双哲学、Neon 路由、超时 `150s`/`60s` 与 `nav_history`/`pick_recommendations` 新列；新增 `tests/test_v553_observability_and_attribution.py` 离线覆盖。

### `v5.5.2`（2026-11-19）

双钱包挂单冻结、成交回执防误报、早报篇幅放宽与 DeepSeek 高峰期超时治理（[`docs/plans/2026-11-19-5.5.2-wallet-receipt-report-hardening-plan.md`](./docs/plans/2026-11-19-5.5.2-wallet-receipt-report-hardening-plan.md)）。

1. **Invest USD 挂单自动冻结（`src/data_providers/t212_client.py`, `src/bot/handlers.py`）**：`reconcile_invest_wallets()` 在 `/equity/orders` 可用且汇率就绪时，按 `side=BUY & currency=USD` 活动挂单汇总美元名义冻结额（限价 × 剩余数量优先，其次止损价，兜底 `value`），并以 `min(usd_total, usd_pending_buy_notional, reserved_cash × rate)` 三重钳位拆分 USD/GBP 钱包；撤单或成交后挂单消失即自动归零释放，`/equity/orders` 403/异常时安全退回本地静态锚点。`/wallet` 与 `format_account_overview()` 同步展示 USD/GBP 可用与挂单占用，且不改变 GBP 总现金/净值口径。
2. **成交回执防撤单误报（`src/data_providers/t212_client.py`, `src/bot/scheduler.py`）**：`HistoricalOrderItem` 保留原 `order_id`；`match_fill_history()` 建立三道防线——有 `order_id` 时只做精确匹配、绝不降级模糊；旧历史桩才允许时间窗口匹配，且 LIMIT 单强制校验物理边界（BUY `fill.price <= limit × 1.005`、SELL `fill.price >= limit × 0.995`）；`fill_ids` + `claimed_fill_ids` 实现一对一互斥认领，同一成交不会被两笔撤单重复回执。ISA 回执尾部改为仅提示 `/pos`，Invest 才提示 `/usd`/`/wallet` 校准。
3. **`/report` 篇幅放宽（`src/agents/schemas.py`, `src/agents/report_agent.py`, `AGENTS.md`, `src/bot/handlers.py`）**：渲染默认硬上限 `2200 → 2800`、PM Prompt 软目标 `1700~2000 → 2100~2500`、`/us` 继承 `morning_baseline[:2800]`；同步放宽 `headline<=55`、`macro_and_cash<=130`、`account_diagnosis<=180`、`flagged_positions<=60`、`hold_summary<=80`、`trade_review<=100`、`daily_vetoes<=50`、`earnings_watch<=45` 与行动/挂单/核心票据理由至 60 字，仍严格保持单条 Telegram 消息不拆包。
4. **DeepSeek 超时治理与 Semantic Auditor 底座增强（`src/config.py`, `config.yaml`, `src/main.py`, `src/agents/llm_client.py`, `src/agents/report_agent.py`, `src/agents/semantic_auditor.py`）**：新增 `deepseek_round_timeout_seconds=150.0`（`httpx` 读超时与 `asyncio.wait_for` 对齐），`ds_helper_timeout_seconds` 由 `40 → 60`；`httpx.TimeoutException`/`ReadTimeout` 与 `asyncio.TimeoutError` 纳入 `timeout_max_retries=0` 快速故障转移，不再白等第二轮。PM Prompt 明令禁止把「撤单」与「吃息」拼成伪因果。`build_fact_baseline()` 新增 ISA/Invest 双账户全持仓成本-现价表（含跨账户同名标的），语义质检失败日志不再截断至 400 字（仅 ops-notify 保留 400 字）。
5. **版本与测试**：`src/version.py` / `pyproject.toml` 升级至 `5.5.2`；新增/更新双钱包冻结与撤单释放、`order_id` 精确匹配与限价边界、一对一互斥认领、ISA/Invest 回执文案、2800 字渲染边界、DeepSeek 超时快速转移、双账户成本底座等用例；全量回归 **443/443 全绿**，`scripts/audit_egress.py` RESULT: CLEAN。

### `v5.5.1`（2026-11-18）

核心宽基从「全球 + 标普500」扩展为「全球/标普核心 + FTSE100/欧洲核心分散器」，让 ISA 的 60% 目标不再只能靠 VWRP 完成；`/report` 新增未持仓核心宽基的正式部署通道，`/pick` 对 DCA 宽基豁免 R:R 硬拦。

1. **核心宽基范围（`src/engine/deployment_gate.py`, `src/engine/portfolio_rules.py`）**：`_CORE_ETF_INDEX_FAMILIES` 与 `_CORE_WIDE_BASE_FAMILIES` 新增 `FTSE 100 (英国富时100)` 与 `Europe Equity (欧洲宽基)`。`ISF.L / VUKE.L / VERG.L / VEUR.L` 等真正宽基现在计入 ISA 60% 核心目标、享受核心通道与 VIX 豁免；`CNX1/EQGB`（纳指）、`IUKD/VHYL`（高息）、`IGLT/VGOV`（国债）、`SGLN`（商品）、`INRG/IITU/HEAL`（主题）仍保持降级，不虚增核心占比。新增 `core_diversifier_pct`、`core_diversifier_share_of_core_pct` 与软偏好 `prefer_core_diversifier`（核心分散器占核心 <33% 且核心未达标时，门控/Prompt 优先提示 FTSE100/欧洲宽基，**不是硬科技上限**，避免被迫买低增长价值陷阱）。
2. **`/report` 未持仓核心宽基部署（`src/agents/schemas.py`, `src/agents/report_agent.py`）**：`MorningReportDecision` 新增 `core_deployment`（0~2 条，仅 ISA，BUY，`price_zone`/`size_hint`/`reason`）；`top_actions` 仍只做存量 ADD/TRIM/STOP/HOLD，现有契约不破坏。`/report` 会一次性补拉未持仓核心/分散宽基的轻量行情与均线上下文（`include_news=False`，零额外 LLM 调用），PM 可推荐 `ISF.L/VERG.L` 等分散美股科技；确定性护栏只允许门控核心集合、禁止已持仓/挂单重复、最多 2 条、低于 `min_isa_etf_order_gbp` 拦截、`etf_open=false` 时清空。新增条目写入 T+N ledger（`source=report`、`fill_status=ASSUMED`）并纳入 5 日同向去重。
3. **`/pick` DCA 宽基放行（`src/engine/portfolio_rules.py`, `src/agents/pick_agent.py`）**：核心/分散宽基的 BUY/ADD 豁免 `RR_UNVERIFIED` / `RR_BELOW_FLOOR`（结构性 DCA 不适用 2:1），除息窗口降为 `[⚠️纪律提醒]` 提示；仍保留 VIX 豁免、频率零 Quota、跨账户双持豁免。应税 Invest 新增硬拦：禁止买入累积型核心宽基（`VWRP/VUAG/SWDA/CSP1` 等，防离岸 ERI 报税摩擦），仅允许派息型 `VWRL/VUSA/ISF/VUKE/VERG/VEUR` 等。
4. **门控与 Prompt 协同**：`format_deployment_gate_banner()` 在软偏好下将「归拢 VWRP」改为「归拢 ISF/VERG 等低科技宽基」；`_format_gate_constraint_block()` 与 `/report` Bull/PM Prompt 同步提示「核心不足且科技偏高时优先 FTSE100/欧洲宽基，趋势破位/追高不硬凑」；`core_etf_context` 作为只读候选块注入 Bull/Bear/PM 与语义质检事实底座。
5. **离线单测**：新增 `TestV551CoreDeployment`（渲染/入账/非核心/Invest/已持仓/低于最小额/etf_open 关闭拦截、Invest 累积型 vs 派息型）、核心分散器计入 60% 与软偏好、`/pick` 核心 ETF 豁免 R:R 与除息降级；全量回归 **404/404 全绿**，`scripts/audit_egress.py` RESULT: CLEAN。
6. **模型路由升级 `Gemini 4 Neon`（2026-11-19，`config.yaml`, `src/config.py`, `src/agents/llm_client.py`, `DEVELOPER_GUIDE.md`）**：按「拍板升级、大输入/工具/备援留 Hydrogen」分层替换——`/report` Step 3 PM 拍板由 `gemini-4-carbon (HIGH)` 切至 `gemini-4-neon (HIGH)`（规避 2026-12-01 Carbon 涨至 \$4/\$20 且 Vals Finance 表现更优、延迟更稳）；`/pick` Stage 2 风控终审与 `/deep` 深度单聊由 `gemini-4-hydrogen (HIGH)` 切至 `gemini-4-neon (MEDIUM)`；Step 2 Bull/Bear、`/pick` Stage 1、`/us` 异动、搜索及备援继续保留 `Hydrogen` / `3.5-flash-lite` / `deepseek-flash`。`llm_client.py` 同步将 `neon` 纳入 `XHIGH` 白名单及 `_candidate_chain` 同厂降级至 `gemini-4-hydrogen (HIGH)` 兜底分支。
7. **辩论思考预算、空文本诊断与 `/pick` 护栏标题对齐（2026-11-19，`config.yaml`, `src/config.py`, `src/agents/report_agent.py`, `src/agents/llm_client.py`, `src/agents/pick_agent.py`, `src/agents/schemas.py`）**：
   - 修复 `/report` Step 2 R2/R3 在 `HIGH`/`MEDIUM` 思考下被 `max_output_tokens`（4096/2048/1024）吃满导致空文本连续降级的问题：`report_debate_max_output_tokens` 提至 `16384`，R3 `digest`/`factcheck` 放宽至 `8192`/`8192`（正文长度仍由 `trim_text_at_line_boundary` 控制）；`llm_client.py` 空文本失败原因补充 `finish_reason`/`out`/`thought` 诊断字段，多级降级告警串联全部前序失败原因。
   - 修复 `/pick` 护栏拦截标的（如 `VWRP.L(ADD→拦截)`）后 `headline` 仍残留被拦标的引发 `[内部驳回外泄]` 告警的问题：新增 `_reconcile_pick_headline()` 在全部护栏执行后同步修正结论句；同步修复 `render_pick_report()` 用 `risk_override.decision` 无条件覆盖门控后 `final_decision` 的隐患。
   - `ds_helper_timeout_seconds` 由 `20.0s` 放宽至 `40.0s`，避免 `/pick` 30 股宽漏斗 `pick_prefilter`（DeepSeek HIGH）在 20 秒被掐断静默降级。

8. **v5.5.1 修订：Architecture A 确定性四桶核心政策引擎（2026-11-19，`src/data_providers/analytics.py`, `src/engine/deployment_gate.py`, `src/agents/report_agent.py`, `src/agents/schemas.py`, `src/agents/pick_agent.py`, `src/engine/portfolio_rules.py`, `src/storage/db.py`, `config.yaml`）**：
   - **穿透底座修复**：`estimate_penetration_tech_exposure()` 先经 `normalize_issuer_root()` 归一（GOOGL→GOOG 不再漏算），科技因子补入 PANW/CRWD/NOW/SNOW/DDOG/NET/ANET/MRVL/KLAC/LRCX/AMAT/CDNS/SNPS；新增 VERG/VERX/VEUR/VEUA/IMEU/SMEA/CUKX/VUKG 静态科技权重与 `_ETF_REGIONAL_LOOKTHROUGH`，新增 `estimate_portfolio_regional_lookthrough()` 输出美/英/欧陆穿透并显式暴露未映射 ETF（禁止静默吞没）；`infer_etf_index_family()` 增加 2x/3x/inverse/leveraged/covered call 黑名单，杠杆 ETP 不再获核心豁免；新增 `classify_etf_distribution_policy()` 区分 ACC/DIST/UNKNOWN。
   - **四桶政策引擎**：以 `CORE_UPPER_BAND_PCT=65`、锚 42/32、欧洲 8/10、英国 6/8、标普 tilt 4、单日预算 `min(£1,600, 3% ISA)`、单票上限与英国穿透 12%/14% 替代旧 33% 软目标；`compute_core_policy_gaps()` 保留 VUAG 存量计入 60% 分母但冻结增量（超额美股宽基抵扣锚目标至 32% 保底），`compute_core_allocation_plan()` 由 Python 唯一确定子桶代码/金额/趋势守卫（跌破 MA200 逾 5% 未达 -15% 暂缓；RSI≥72 或距 MA50>5% 改 PULLBACK_LIMIT），单次最多 1 张未持仓票据。
   - **`/report` 改为 Python 出票 + PM 单向批/缩/缓**：`CoreDeploymentItem` 新增 `verdict=APPROVE/SHRINK/DEFER` 与 `approved_gbp`；`_validate_core_deployment()` 只允许审批 Python 唯一票据，换代码（VUAG/IUKD 等）硬拦、超额金额向下钳位并重写 `size_hint`、DEFER 不渲染不写账本；获批金额回写 `isa.approved_report_core_gbp` 供同日 `/pick` 扣减；Prompt 删除“PM 自由挑宽基”，改为票据终审。
   - **战略 DCA 台账隔离**：`/report` 核心宽基写入 `source='core_dca'`、`horizon_days=20`，并按 `source` 分组落库；`/pick` 核心宽基同样标记 `core_dca`，提示词明确核心宽基 `stop_price/target_price` 填 `null`，禁止编造止损。
   - **`/pick` 税务与通道收口**：`_DISTRIBUTING_CORE_ETFS` 移除累积型 `VERG/CUKX`，补入派息型 `VERX/IUSA`；Invest 全部 ETF 按 `classify_etf_distribution_policy()` 校验（核心必须 DIST、非核心 ACC 亦硬拦 ERI）；RISK_OFF 未达阶梯与 ISA 仍有现金/核心缺口时禁止 Invest 新建核心宽基；同日 `/report` 已批准则 `/pick` 不再重复放行，单次最多 1 条 ISA 核心动作，`US_TILT` 冻结时硬拦标普加仓。
   - **DCA 确定性护栏**：`portfolio_rules` 新增 `CORE_DCA_GUARD`（hard 模式），在豁免 R:R 前校验子桶上限/标普冻结、ISA 5% 现金地板、英/欧/标普跌破 MA200 与 RSI 追高，彻底替代“无脑空白豁免”。
   - **绩效口径与测试**：`get_recommendation_stats()`/`get_settled_outcome_series()`/`get_recent_lessons()` 全部排除 `source='core_dca'` 并新增 `core_dca_count` 观察位；新增 `compute_core_policy_benchmark_comparison()` 区分区域倾斜 Beta 与选股 Alpha。补齐真实 2026-11-18 账户快照回归（46.53% 核心、VUAG 冻结、VWRP 挂单满足 32% 锚、VERG/ISF 唯一票据、PM 换票/超额/DEFER 拦截）与穿透/杠杆/派息/护栏/DB 用例；全量回归 **416/416 全绿**，`scripts/audit_egress.py` RESULT: CLEAN。

9. **双账户双哲学解耦（2026-11-19，`src/engine/portfolio_rules.py`, `src/engine/screener.py`, `src/engine/deployment_gate.py`, `src/data_providers/market_data.py`, `src/agents/*`, `AGENTS.md`, 施工蓝图 `docs/plans/2026-11-19-5.5.2-dual-philosophy-plan.md`）**：Stocks ISA 改走巴菲特价值流派（质量门 × 安全边际，不再被短期波动洗出），Invest 改走右侧动量流派（Regime 动态 R:R + ATR 噪声地板 + 前瞻 R:R 置换）；`/report` 单消息内按账户分区块排版，字数上限 `1800 → 2200`。本次迭代仍随 `v5.5.1` 发布，不单独 bump 版本号。
   - **数据层（`market_data.py`）**：`TickerMarketBundle` 新增可选 `fcf_yield_pct` / `target_high_price` / `target_low_price` / `peg_ratio`（旧 JSON 缓存向后兼容）；新增 `_compute_safe_fcf_yield_pct()`（`fcf/mcap` 超出 `-50%~+50%` 判为跨币种脏数据丢弃）与 `_pick_peg_ratio()`；Finnhub `/stock/metric` 增 `pfcf` 别名，`P/FCF≥2` 时补算 FCF 收益率。
   - **规则引擎（`portfolio_rules.py`）**：新增 `evaluate_isa_value_gate()`（质量门：亏损/ROE≤0/非金融高杠杆且流动性紧/营收≤-10%/FCF<-2%，金融·地产·REIT·公用事业杠杆豁免；安全边际锚：分析师目标折价 / ERP / FCF 率 / 52 周回撤；高 PE + 低增长或 PEG≥2.8 判泡沫）、`resolve_invest_dynamic_rr_floor()`（RISK_ON 右侧领涨 1.6 / NEUTRAL 2.0 / 弱势 2.5）与 `compute_invest_forward_rr()`；`_resolve_pick_rr()` 增 RISK_ON 双锚目标上沿与 `STOP_TOO_TIGHT` ATR 噪声地板（`max(1.5×ATR, 4%)`）；`evaluate_holding()` 拆分支——ISA 取消 `Price>MA200` 加仓前提（四道锁价值左侧加仓）并取消移动止盈/RSI/乖离/CYQ 波动性减仓（仅保留集中度/极端泡沫/质量恶化/散仓归核），Invest 移动止盈随 Regime 由 `-12%` 收紧至 `-8.5%`、RISK_ON 站稳 MA20 不因 RSI 卖飞、前瞻 R:R `<0.8:1` 提示置换；`apply_risk_override_guardrail()` 增 `tech_map`，ISA 买入改走 `ISA_QUALITY_FAIL` / `ISA_MOS_INSUFFICIENT`，Invest 走动态 R:R 与 `STOP_TOO_TIGHT`；危机日 ISA「质量全过 + ≥20% 深折价」保留分批逆向通道，Invest 仍 100% 冻结。
   - **选股器与门控（`screener.py`, `deployment_gate.py`）**：`ISA_LONG_TERM` 权重改为 趋势 15 / 形态 10 / 价值质量 50 / 组合 25，去掉破 MA200、Phase4、周线偏空的动量重罚，改为「优质股回撤 +5」与 FCF / ERP 加分、金融杠杆豁免、劣质股硬扣分；`INVEST_TACTICAL` 保持原权重并在 RISK_ON 给 Stage2 领涨额外 `+2`；`run_quant_screener()` 增 `macro` / `regime_tag` 并统一 `RISK-ON/NEUTRAL/RISK-OFF` 词表；门控新增 `isa.value_crisis_mode` 与危机价值通道（仍受 12% 单票、30%/8 只卫星、2% 现金地板与 PM 一票否决约束），`DEFEND` 四步协议区分 ISA 价值买单与 Invest 未成交单。
   - **Agent 编排（`report_agent.py`, `pick_agent.py`）**：`/report` 在门控求出真实 Regime 后按该 Regime 重算持仓评估并同步 FactSheet/ContextPack，护栏传入 `tech_map` 与 `deployment_gate`，Bull/Bear Prompt 写入双哲学边界；`/pick` 透传 `macro` / `regime_tag` / `regime`，Post-Screen 为「已有持仓 + 本轮入选标的」统一构建 `all_tech_map` 后再做确定性校验，避免新候选无技术快照导致的价值门/R:R 盲判。
   - **Prompt & Schema（`prompts.py`, `schemas.py`）**：投资者画像明确 ISA 巴菲特价值流派（卖出仅 4 情形）与 Invest 右侧动量流派（动态 R:R + ATR 地板 + 前瞻 R:R）；`stop_price` / `target_price` / `headline` 描述同步双轨口径。
   - **`/report` 单消息分账户排版 + 2200 字上限（`schemas.py`, `report_agent.py`, `bot/handlers.py`, `AGENTS.md`）**：`render_morning_report()` 默认 `max_chars=2200`，`sec1` 在同一条 Telegram 消息内确定性拆出 `🏛️ ISA 价值账户` 与 `⚡ Invest 右侧账户` 两个子区块（动作、核心宽基部署、挂单审计各归其位），空账户仅一行带过；三个顶级标题与 `gate_banner` 置顶不变，`[🛡️风控护栏修正]` 固定压尾；Prompt 目标篇幅 `1700~2000` 字，`/report` 基线切片同步放宽至 `2200`，铁律 4 与 `/help` 文案同步。
   - **离线单测**：新增 `tests/test_v551_dual_philosophy.py` 覆盖 13 组场景（FCF/目标区间币种防御、ISA 质量门与金融豁免、安全边际与泡沫、危机价值通道 vs Invest 冻结、ISA 四道锁左侧加仓、ISA 不被波动洗出、Invest 动态 R:R / ATR 噪声地板 / 牛市目标上沿、牛市让利润奔跑与震荡收紧、前瞻 R:R 恶化、ISA 选股价值偏好、分账户排版与 2200 上限）；迁移 5 处受新语义影响的既有单测；全量回归 **430/430 全绿**，`scripts/audit_egress.py` RESULT: CLEAN。

### `v5.5.0`（2026-11-18）

风控闭环加固与记账纠偏（[`docs/plans/2026-11-17-PLAN_HARDENING_POST_AUDIT.md`](./docs/plans/2026-11-17-PLAN_HARDENING_POST_AUDIT.md)）：把「模型自觉」收紧为「Python 硬拦」，同时解开危机期核心宽基抄底与 `DEFEND` 退出的两个死锁，并摆正 T+N 与 TWR 的记账符号。

**⚠️ 风控开关变更（已由账户持有人授权，需人工最终确认）**：`config.yaml` 中 `trade_cadence_guard_mode: "advisory" → "hard"`、`pick_rr_guard_mode: "advisory" → "hard"`、`user_preference_memory_enabled: true → false`。前两项把频率/冷却与 R:R/除息从「仅提示」切为 Python 硬拦截；第三项关闭散户情绪偏好抽取，避免短期口癖污染长期纪律。

1. **硬护栏与伪宽基清洗（`config.yaml`, `src/engine/deployment_gate.py`）**：`_build_etf_set()` 只保留底层指数家族属于「全球宽基 / 标普500」的真正核心宽基（VWRP.L / VWRL.L / VUAG.L / SWDA.L），IUKD.L / VHYL.L / IGLT.L / SGLN.L / CNX1.L 等主题、高息、债券、商品 ETF 自动退入 `other_etf_gbp`，不再虚增 `core_etf_pct` 或享受核心通道豁免；`_native_risk_to_gbp()` 先保留 `GBp/GBx` 原始大小写，英股便士风险不再被 `upper()` 放大 100 倍；Invest 以盈解锁改为「止损挂单覆盖 ≥99% 股数 + 加权对齐止损价 ≥ 成本+摩擦」双条件，部分覆盖时已保护段按真实价差、裸露段按自适应止损比例合并计入 Portfolio Heat。
2. **`/pick` 四大穿仓口封堵（`src/engine/portfolio_rules.py`, `src/agents/pick_agent.py`）**：Invest 未真实留存 USD（`usd_cash_total≈0` 且近期美股 SELL 回流 GBP）时按双边换汇计摩擦，`usd_avail` 不足以覆盖整笔最小建仓额时不再免单边 FX；`_resolve_pick_rr()` 先用实时行情对齐 `entry_price` 的 GBp/GBP 单位再算 R:R；`_shares_from_gbp()` 删除「缺失币种就把 .L 盲除 100」的致命启发式；`_build_pick_size_note()` 接收 `existing_position_value_gbp`，ADD 算仓按账户真实存量扣减单票余量；Invest 只要 `status not in {RUN, ROTATE}`（含全部 `WAIT` 子模式与 `UNKNOWN`）一律冻结新开个股/非核心 ETF，且单次 `/pick` 按剩余槽位与单日额度递减，杜绝一次连推 2~3 笔击穿上限；跨账户同票去重补充 `normalize_issuer_root()`（GOOGL/GOOG 不再漏网）；Critic 结构化解析失败改为 fail-closed 返回 `⚠️`，不再裸推未经护栏核验的原始文本。
3. **危机期宽基抄底与 `DEFEND` 死锁修补（`src/engine/deployment_gate.py`）**：`crisis_eased` 取消「mom20>0 即算修复」，MA200 下方死猫跳 3 天不再提前解除 `RISK_OFF`（仅 MA200 数据不足时才允许动量兜底）；`RISK_OFF` 退出新增「唯一剩余风险标记是历史 52 周回撤」分支，-25% 见底后重新站稳 MA200 满 3 天可阶梯恢复 `NEUTRAL`，不再被一年前旧高点锁死；`_core_etf_drawdown_step()` 同时计算 VWRP.L 与 SPY 的 52 周回撤并取更深一侧，危机期英镑贬值无法掩盖美元资产真回撤；`RISK_OFF` 下若宽基持续 ≤-10%、可用现金超上沿、核心宽基未达标且尚无接针挂单，ISA 保持 `RUN·ladder` 允许补挂；跨月持仓（卖出距 epoch 已超 `order_history_days`）缺近期买单不再盲目豁免，避免漏计真实连损。
4. **自适应算仓与 Prompt 冲突清理（`src/engine/portfolio_rules.py`, `src/agents/pick_agent.py`, `src/agents/report_agent.py`）**：`compute_risk_based_size_gbp()` 新增 `risk_budget_scale` 与 `0.5% NAV` 终身免调参预算（`max(£85, NAV×0.5%)`）；ISA 只吃 NAV 下限、绝不被 Invest 连损 0.5R 砍半，Invest 在 £30,000 净值时自动扩容至 £150；`apply_risk_override_guardrail(..., risk_budget_scale=...)` 按单笔账户隔离缩放；`MACRO_VIX` 仅冻结个股与非核心 ETF，VIX≥28 时核心宽基定投不再被误杀；`/report` 门控候选在 MA50 回踩粗筛后补拉一次轻量 `get_bundles(include_news=False, include_earnings=True)`（≤6 只），5 日财报与 5 日除息排雷真正生效；删除「Invest 现金 >40% 必须凑单」与「无个股就买累积型 VWRP.L」的催买冲突句，明确 ISA 归拢 VWRP.L、应税 Invest 仅可提示派息型 VWRL.L（防离岸 ERI 报税摩擦），无 R:R≥2:1 时直接空数组。
5. **T+N 结算方向符号与触价起算（`src/bot/scheduler.py`）**：`SWAP` 属于换入新标的多头，方向改为 `+1`；`_direction_adjusted_alpha_pct()` 让方向作用于整段超额——多头仍为 `stock - bench`，`TRIM/STOP/SELL` 为 `bench - stock`（砍弱股跌 -3% 而基准涨 +5% 记 +8%，乱砍抗跌股而基准跌 -10% 记 -9%）；`_entry_fill_status(..., exclude_rec_date=True)` 在推荐生成时该市场已收盘（如周五 21:30 `/pick`）时排除当日已收盘低点，不再马后炮判 `FILLED`；`/report` 同向动作写入 ledger 前增加 5 日 `(account, symbol, action)` 去重，避免同一持仓连写 10 条高自相关 T+20 记录。
6. **TWR 现金利息真实入账（`src/data_providers/analytics.py`, `src/agents/report_agent.py`）**：`_EXTERNAL_FLOW_SIGNS` 仅保留 `DEPOSIT/WITHDRAW/WITHDRAWAL`；`INTEREST_ON_FREE_CASH`、`LENDING_INTEREST` 与平台 `FEE` 属账户内部收益/损耗，不再作为外部现金流抵消净值涨跌；USD 外部出入金按 `gbp_usd_rate` 折算计入 `net_flow_gbp`（无汇率则跳过，绝不臆造）；早报基准行删除「1 日净值变动 − 7 日基准」的异周期错位差值，只客观并列「基准近7天/近30天」与「本账户较上次快照」。
7. **离线单测与验收**：新增 `TestV550PostAuditHardening`（部分覆盖止损不解锁、Invest `WAIT/none` 与非核心 ETF 冻结、主题 ETF 失去核心特权、VIX30 核心宽基豁免、MA200 下方反弹不退 `RISK_OFF` + 旧高点回撤解除、持续深跌补挂阶梯、未持仓候选财报 Bundle 排雷）；`TestV534PickBuyGuards` 新增 GBp 2500/LLM £25 的 100 倍对齐与缺失币种不盲除 100；`TestStage2PenetrationAndSizing` 新增 ISA `0.5% NAV` 与 Invest 半仓隔离；`TestDirectionAwareSettlement` 新增 `SWAP=+1`、STOP +8%/−9% Alpha 符号与盘后排除当日低点；`TestNavRowAndPerformance` 将 FEE/利息改为内部项并新增 USD 入金折算与利息计入 TWR。全量离线回归 **399/399 全绿**，`scripts/audit_egress.py` RESULT: CLEAN。
