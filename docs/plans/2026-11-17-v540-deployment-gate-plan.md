# Trading 212 × Gemini AI Advisor — v5.4.0 施工蓝本（Implementation Blueprint · 已归档冷文档）

> **冷归档说明（2026-11-17）**：本任务（`v5.4.0` CIO 组合门控）已完成施工与旗舰模型验收归档；配套策略设计稿见同目录 [`2026-11-17-report-pick-gate-design.md`](./2026-11-17-report-pick-gate-design.md)。以下保留原始施工蓝本备查。

---

## 一、施工前必读红线与系统铁律（对齐 `AGENTS.md`）

1. **禁止读取黑名单路径**：严禁读取 `.env`、`.env.*`、`data/`、`logs/`、`*.db`、`*.db-wal`、`*.db-shm`、`*.sqlite*`、`*.log`、`_ref/`。所有回放与单测一律使用内存/临时库或离线合成 DataFrame 桩（须含 `Open/High/Low/Close/Volume`）。
2. **大文件分段读取纪律**：修改 `>400` 行源码前，必须先用 `grep` 定位 `class / def / async def` 行号，再按 `100~150` 行分段读取，严禁整篇吞读大文件。
3. **红线文件最小触碰原则（`AGENTS.md` §5.2 & §5.3）**：
   - **`src/config.py` 属红线文件**：仅允许在 `StrategySection` 内新增 v5.4.0 策略参数与对应 `@field_validator`，**严禁触碰 `EnvSettings`、`_LLM_EGRESS_HOSTS`、`_T212_EGRESS_HOSTS` 或任何出网/鉴权逻辑**；交付时必须逐行展示 `src/config.py` 的 diff 供人工复核。
   - **零触碰 `src/data_providers/t212_client.py`**：本版本所需全部原始字段（持仓 `created_at`、挂单 `side/order_type/limit_price/stop_price/currency`、历史成交 `realised_pnl/filled_at_ms`、外部入金 `recent_transactions`）均已在 `PortfolioData` 现成暴露，在 `deployment_gate.py` 与 `portfolio_rules.py` 消费即可，无需改动 `t212_client.py`。
   - **零新增外部依赖、零新增出网域名、零删改线上 DB 表结构**；版本同步更新 `src/version.py` 与 `pyproject.toml` 为 `5.4.0`（修改 `pyproject.toml` 仅限 `version = "5.4.0"` 一行，严禁改动 `dependencies`，完成后同样列出 diff 供人工确认）。
4. **双币种与英股便士铁律**：
   - 英股挂单常以 `GBp`（便士）报价（如实盘 `SHEL.L` 限价 `3480.00 GBp = £34.80`），所有挂单/止损价与现价/均价比较必须经过 `_rescale_price_pair` 对齐量级，严禁发生 100 倍错位。
5. **纯文本与字数硬上限**：
   - Telegram 渲染严禁出现 `#` 与 `**`；`/report <= 1800` 字、`/pick <= 1000` 字；门控块渲染于 `/report` 第二段（`【宏观、账户与今日风控红线】`）标题正下方第一行，控制在 `<= 110` 字。

---

## 二、业务灵魂：施工绝不可走样的 5 条核心逻辑

在写任何一行代码前，施工 Agent 必须吃透以下实盘背景与规则边界（详见 [`docs/plans/2026-11-17-report-pick-gate-design.md`](./2026-11-17-report-pick-gate-design.md)）：

1. **现金恒等式不重复扣减 + 挂单三分法**：
   - `AccountSnapshot.available_to_trade` **本身已经是扣除了 `reserved_cash`（未成交买单锁定额）之后的净可部署现金**（实盘 ISA：总现金 `£20,862.50 [36.4%]` − 挂单 `£8,741.16 [15.2%]` = 可用 `£12,121.34 [21.14%]`）。**严禁再写 `available_to_trade - pending_buy` 的二次扣减代码！**
   - 挂单必须严格分为三类：
     - `PROTECTIVE_SELL_STOP`（`side == "SELL"` 且含 `STOP`）：如 Invest 的 `AMZN @ $243.50`、`CF @ $113.50`。**属于正向风控保护垫，用于抵扣在险敞口（Portfolio Heat），绝不阻塞门控，绝不触发「先审挂单暂停开仓」！**
     - `CORE_ETF_BUY`（`side == "BUY"` 且标的属于核心宽基/ETF）：计入承诺宽基仓位。
     - `NEW_EQUITY_BUY`（`side == "BUY"` 且标的为个股，如 ISA 的 `SHEL.L`、`RKT.L`）：视同「准新增个股」，同时计入 **ISA 个股卫星占比/只数** 与对应账户的**在险槽位**。
2. **ISA 存量散仓「老仓豁免强平 + 循序渐进顺势瘦身」**：
   - 用户 ISA 当前有 13 只历史遗留个股（含 8 只 `<£1,000` 碎单），加上 2 张个股买单达 15 只 / 41.9%。
   - **入口硬封死**：当 ISA 个股（持仓 + 个股 `BUY` 挂单）`> 30%` 或 `> 8` 只时，**锁死 ISA 新开个股通道**（门控提示审撤未成交个股买单、闲钱归拢 `VWRP.L`；`/pick` 的 ISA 赛道过滤掉所有新增个股，仅保留宽基 ETF 候选）。
   - **出口绝不盲目清仓**：存量 `< min_isa_equity_order_gbp`（`£1,200`）的历史边角料享有**老仓豁免**，严禁在大盘下跌或缩量回调时催促一刀切割肉凑比例！仅当边角料处于**顺风反弹窗**（`return_pct >= -3.0%` 且 `RSI >= 60` 或距 `MA50/20日高点` 阻力位近）或**基本面/趋势实质破位**时，温和标记顺势归核，且单次早报最多浮出 **1~2 只**条件最成熟的边角料。
3. **清理边角料与买入宽基「100% 零 Quota 豁免」**：
   - **卖出（`SELL`）任何存量仓位/边角料**，完全不消耗任何开仓配额；
   - **买入核心宽基 ETF（`benchmark_symbol` 及 `cfg.universe.etf_symbols`，如 `VWRP.L`、`VUAG.L`）**，在周/月开仓次数统计（`_cadence_violation`）与在险槽位中**一律豁免（计 `0` 次、`0` 槽位）**。
4. **Invest 独立「在险槽位 + 以盈解锁 + 新纪元起算锚点」**：
   - **彻底取消 Invest 的现金超带 `dca` 催买**：Invest 常拿 80% 英镑吃息完全正常，只有 A 触发器（右侧猎物）命中时才评估 `RUN·targeted` 或 `ROTATE`。
   - **账户物理隔离**：在险槽位（`RISK_ON: 2 / NEUTRAL: 1 / RISK_OFF: 0`）**只算 Invest 账户自身的持仓与未成交个股买单**，ISA 的 13 只历史老仓 100% 隔离，绝不卡死 Invest！
   - **以盈解锁（Financed Risk）**：Invest 持仓若挂有活动 `PROTECTIVE_SELL_STOP`，且经币种/便士对齐后 `stop_price >= avg_price * (1 + round_trip_friction_pct / 100)`（如实盘 `AMZN`：均价 `$238.42`，`STOP $243.50`），则判定为**已脱险仓位（Financed，`open_heat_gbp = 0`，占用 `0` 个槽位）**，立即释放槽位解锁下一笔右侧开仓！未达保本止损的持仓（如 `CF`）及未成交个股 `BUY` 单各占 `1` 个槽位。
   - **单日限 1 笔 + `gate_epoch_date` 历史隔离 + 满槽 `ROTATE`**：
     - 同一交易日最多新开/新挂 `1` 笔战术个股；
     - 「近 14 天连损 2 笔 → 冷静 5 交易日 + 首单 0.5R（£42.5）」**只统计成交时间 `>= gate_epoch_date` 且建仓也在新纪元之后的 Invest 战术止损单**！历史 `-£1,335` 旧账与消化 `gate_epoch_date` 之前的老仓平仓单全部豁免，绝不误触发连损惩罚！
     - 当 Invest 槽位已满（如 `NEUTRAL` 下 `1/1`），但 A 触发器出现高分右侧突破标的时，输出 **`ROTATE`**（要求平掉现有未脱险弱仓或撤旧买单等风险置换，严禁动用闲置现金净增槽位）。
5. **去汇率噪音的防死锁 Regime + 大师四步 `DEFEND`**：
   - Regime 趋势/回撤锚定美元原币种基准 `regime_benchmark_symbol`（默认 `"SPY"`，若缺失才回退 `benchmark_symbol`），结合 `^VIX` 与 `MacroSnapshot.hy_oas_spread_pct`：
     - 进 `RISK_OFF`（1 日确认）：`SPY < MA200 * (1 - 0.03)` **或** `VIX >= vix_danger_threshold (28.0)` **或** `SPY 距52周高点回撤 >= 15.0%` **或** `HY OAS >= 5.5%`；
     - **退回 `NEUTRAL`（3 日确认，防死锁！）**：只需危机缓解——`VIX < 25.0`（或较峰值回落）且 `HY OAS < 5.5%` 且（`SPY >= MA200` **或** `SPY 20日动量 > 0`）。**严禁要求 `VIX < 18` 才能退出 `RISK_OFF`！**
     - 进 `RISK_ON`（3 日确认）：`SPY > MA200` 且 `MA50 > MA200` 且 `52周回撤 < 8.0%` 且 `VIX < 18.0` 且 `HY OAS < 4.5%`；任一破坏退回 `NEUTRAL`（`-8%~-14%` 常规回调稳居 `NEUTRAL`）。
   - **`DEFEND` 四步协议**：`RISK_OFF` 时：① 优先撤除未成交个股 `BUY` 挂单 → ② 保护已锁保本仓（如 `AMZN`）与相对抗跌且已挂硬止损的对冲仓（如 `CF`），只收紧止盈不强砍 → ③ 清退无保护浮亏弱势股 → ④ **ISA 宽基 `VWRP.L` 不被高 VIX 封锁**，按 `-10% / -15% / -20%` 回撤阶梯照常逆向分批播种。

---

## 三、涉及文件与职责矩阵

| 序号 | 文件路径 | 操作类型 | 核心职责变更 | 红线等级 |
| :---: | :--- | :---: | :--- | :---: |
| 1 | `src/config.py` | 修改 | `StrategySection` 新增门控、卫星上限、槽位与 `gate_epoch_date` 字段 | 🔴 **红线（须输出逐行 diff）** |
| 2 | `config.yaml` | 修改 | 同步 v5.4.0 默认参数（`deployment_gate_mode: "advise"` 或按阶段配置） | 🟢 绿 |
| 3 | `src/version.py`, `pyproject.toml` | 修改 | 版本号推进至 `5.4.0`（`pyproject.toml` 严禁改依赖） | 🔴 `pyproject.toml` 仅改版本号 |
| 4 | `src/engine/deployment_gate.py` | **新建** | 纯确定性门控引擎：账本、挂单三分法、Regime、Invest 槽位与以盈解锁、A/B/C/D 触发器、PM 否决钳位、110 字纯文本渲染、2 年离线回放器 | 🟢 绿 |
| 5 | `src/engine/portfolio_rules.py` | 修改 | ① 宽基/ETF 买入豁免 `_cadence_violation`；② 单股全组合 `12%` 主约束 + ISA 卫星 `30% / 8只` 新股拦截（豁免宽基）；③ ISA `<£1,200` 历史边角料顺势归核温和评分（最多浮出 2 只）；④ 导出 `VIX_DANGER_THRESHOLD = 28.0` 常量 | 🟡 黄 |
| 6 | `src/storage/db.py` | 修改 | 新增只读方法 `get_recent_context_packs(limit=5)` 或 `get_context_pack_before(...)` 供 Regime 连续确认天数衔接（走只读连接，零建表） | 🟡 黄 |
| 7 | `src/agents/schemas.py` | 修改 | 新增 `PickGateDecision`，挂载至 `MorningReportDecision.pick_gate`；`render_morning_report` 在第二段首行注入确定性门控行 | 🟢 绿 |
| 8 | `src/agents/prompts.py` | 修改 | `build_investor_profile` 同步「ISA 核心≥60%/卫星≤30% + 边角料顺势归核不强割 + Invest 在险槽位以盈解锁 + 宽基/清碎单零 Quota」口径 | 🟡 黄 |
| 9 | `src/agents/report_agent.py` | 修改 | 预拉取 `SPY` / `^VIX` 及观察池本地缓存日线；在 `apply_risk_override_guardrail` **之后**调用 `evaluate_deployment_gate`；写入 `context_pack["deployment_gate"]` 并传入 `render_morning_report` | 🟢 绿 |
| 10 | `src/agents/pick_agent.py` | 修改 | 读取 `context_pack["deployment_gate"]`：ISA 个股饱和时保留候选可见性但通过提示词 + 终审护栏拦截新开 ISA 个股；`ROTATE` 模式强制校验 `funding_from`；跨账户已持仓/已挂单同票去重 | 🟢 绿 |
| 11 | `src/agents/semantic_auditor.py` | 修改 | `build_fact_baseline` 补充 `可用占比 X% / 总现金占比 Y%`，防止 DeepSeek 语义质检将门控横幅的 `21.1%` 误报为幻觉数字 | 🟢 绿 |
| 12 | `tests/test_report_pick_gate.py` | **新建** | 45+ 离线单测（覆盖实盘快照回归、二次扣减防呆、保本解锁、边角料零 Quota、2 年合成行情无死锁回放、字数硬上限） | 🟢 绿 |
| 13 | `CHANGELOG.md` | 修改 | 顶部记录 `v5.4.0` 完整变更说明 | 🟢 绿 |

---

## 四、7 步分阶段施工细则（TDD 执行清单）

### Task 1：策略配置扩充与版本号同步（`src/config.py`, `config.yaml`, `version.py`, `pyproject.toml`）

在 `src/config.py` 的 `StrategySection` 末尾新增以下带默认值的字段（保持原有全部字段向后兼容不删除）：

```yaml
  # --- v5.4.0 组合门控、ISA 核心-卫星结构与 Invest 渐进式火力引擎 ---
  deployment_gate_mode: "advise"               # off = 完全关闭旁路 | log = 仅写日志与 context_pack | advise = 早报渲染并联动 /pick
  gate_epoch_date: "2026-11-18"                # 新纪元起算日(YYYY-MM-DD)：此日期前的历史亏损与老仓出清不计入连损冷静惩罚

  # 1) 集中度与 ISA 核心-卫星结构护栏
  max_single_name_portfolio_pct: 12.0          # 单一公司跨双账户占全组合硬上限（核心宽基 ETF 豁免）
  isa_core_etf_target_pct: 60.0                # ISA 核心宽基目标占比（%）
  isa_satellite_max_total_pct: 30.0            # ISA 个股卫星层（持仓+未成交个股BUY挂单）占 ISA 硬上限（%）
  isa_satellite_max_count: 8                   # ISA 个股卫星层最大只数（持仓+未成交新个股BUY挂单）
  isa_crumb_pruning_max_per_report: 2          # 单次早报最多温和提示顺势归核的 <min_isa_equity_order_gbp 边角料只数
  isa_crumb_near_breakeven_floor_pct: -3.0     # 边角料顺风归核浮亏下限（>=-3% 视为接近回本/微盈区）
  isa_cash_band_low_pct: 5.0                   # ISA 结构现金垫下沿（%）
  isa_cash_band_high_pct: 15.0                 # ISA 常规现金上沿（%）
  min_deploy_excess_pct: 3.0                   # 超过现金上沿多少百分点触发 ISA 宽基 dca（15+3=18%）

  # 2) Invest 在险槽位与以盈解锁（仅核算 Invest 账户自身）
  invest_slots_risk_on: 2                      # RISK_ON 最大同时未脱险个股槽位数
  invest_slots_neutral: 1                      # NEUTRAL 最大同时未脱险个股槽位数（单侦察兵）
  invest_slots_risk_off: 0                     # RISK_OFF 个股槽位数（0 = 冻结新开个股）
  max_new_equity_entries_per_day: 1            # 单日最多新开/新挂个股笔数（Best Idea Only）
  consecutive_loss_limit: 2                    # gate_epoch_date 后近 14 天连续止损笔数阈值
  consecutive_loss_lookback_days: 14           # 连损回溯窗口（天）
  consecutive_loss_cooldown_days: 5            # 连损触发后的个股冷静观察天数（交易日/自然日折算 7 天）
  recovery_pilot_risk_scale: 0.5               # 冷静期满首笔试探单风险缩放系数（0.5R = £42.5）

  # 3) 去汇率失真市场状态（Regime）与 A/C 触发器
  regime_benchmark_symbol: "SPY"               # 美元原币种全球/美股核心趋势锚（防 GBP/USD 扭曲 VWRP.L）
  regime_vix_risk_on: 18.0                     # 进入 RISK_ON 的波动率上限
  regime_vix_exit_risk_off: 25.0               # 退出 RISK_OFF 回到 NEUTRAL 的波动率缓解线（严禁用 18.0 防死锁！）
  regime_ma200_band_pct: 3.0                   # 实质跌破 MA200 的缓冲带（%）
  regime_drawdown_neutral_pct: 8.0             # 常规回调进入 NEUTRAL 阈值（%）
  regime_drawdown_risk_off_pct: 15.0           # 深度回撤进入 RISK_OFF 阈值（%）
  regime_hy_oas_risk_off_pct: 5.5              # 高收益债利差进入 RISK_OFF 阈值（%）
  regime_confirm_enter_days: 1                 # 进入 RISK_OFF 快速确认天数
  regime_confirm_exit_days: 3                  # 退出 RISK_OFF / 进入 RISK_ON 稳定确认天数
  core_etf_drawdown_steps_pct: [5.0, 10.0, 15.0, 20.0]  # ISA 宽基距52周高点分档折扣触发阶梯
  trigger_ma_tolerance_pct: 1.5                # 右侧趋势个股回踩 MA50 容差带（%）
  trigger_max_symbols: 3                       # A 触发器最多输出候选标的数
  inflow_trigger_pct: 2.0                      # C 触发器单日外部净入金占账户比例（%）
  inflow_trigger_min_gbp: 400.0                # C 触发器最小净入金绝对额（£）
```
- *注意*：`isa_crumb_Pruning_max_per_report` 请统一使用规范小写蛇形命名 `isa_crumb_pruning_max_per_report: int = Field(default=2, ge=1, le=5)`。
- 为 `deployment_gate_mode` 添加枚举校验 `Literal["off", "log", "advise"] = "advise"`（非法值回退 `"advise"`）。

---

### Task 2：新建纯确定性引擎 `src/engine/deployment_gate.py`

保持**零网络请求、零数据库直接写入、纯函数可测**，内部包含 7 个核心子模块：

1. **`classify_pending_orders(portfolio, strategy, etf_set, pos_by_sym)`**：
   - 遍历 `portfolio.all_pending_orders()`：
     - `side == "SELL"` 且 `"STOP" in order_type.upper()` → `PROTECTIVE_SELL_STOP`（按账户索引，记录经 `_rescale_price_pair` 对齐后的本币 `stop_price`）；
     - `side == "BUY"` 且标的属于 `etf_set`（`{benchmark_symbol} ∪ set(universe.etf_symbols)` 或 `bundle.quote_type == "ETF"`） → `CORE_ETF_BUY`；
     - `side == "BUY"` 且非 ETF → `NEW_EQUITY_BUY`（估算 GBP 金额，计入对应账户承诺个股仓位与槽位）。
2. **`compute_account_ledgers(portfolio, classified_orders, bundles, strategy, etf_set)`**：
   - **ISA 账本**：
     - `total_cash_pct = isa.total_cash / isa_val * 100`
     - `deployable_pct = isa.available_to_trade / isa_val * 100`（**即为挂单全成交后剩余现金占比，绝不二次扣减！**）
     - 统计 `core_etf_pct`、`committed_core_etf_pct`（含 `CORE_ETF_BUY`）、`satellite_equity_pct`、`committed_satellite_pct`（含 `NEW_EQUITY_BUY`）、`committed_satellite_count`（已持仓个股种数 + 新标的单股 `BUY` 挂单种数）、`legacy_crumb_count`（市值 `< strategy.min_isa_equity_order_gbp` 的个股数）。
     - 判定 `isa_satellite_saturated = (committed_satellite_pct > isa_satellite_max_total_pct) or (committed_satellite_count > isa_satellite_max_count)`。
   - **Invest 账本与真实在险敞口（Portfolio Heat & Financed Risk Slots）**：
     - 仅遍历 `portfolio.invest.positions`（非 ETF）：
       - 查找同标的 `PROTECTIVE_SELL_STOP` 挂单；计算往返摩擦 `friction_pct`（复用 `portfolio_rules._round_trip_friction_pct`）；
       - 若存在 `STOP` 卖单且 `aligned_stop_price >= avg_price * (1.0 + friction_pct / 100.0)`：标记 `is_financed = True`（已锁保本，如实盘 `AMZN`），`slot_used = 0`，`principal_heat_gbp = 0.0`；
       - 否则：标记 `is_financed = False`（未脱险，如实盘 `CF`），`slot_used = 1`；若有 `STOP` 单则按 `max(0, avg_price - aligned_stop_price) * qty` 折算英镑在险本金，若无 `STOP` 单则按自适应止损幅度估算在险本金。
     - 加上 `Invest` 账户内的 `NEW_EQUITY_BUY` 挂单数，得出 `invest_open_slots_used` 与 `invest_total_heat_gbp`。
   - **新纪元连损与单日限频核算（`evaluate_tactical_quota_guards`）**：
     - 解析 `gate_epoch_ms`（`strategy.gate_epoch_date` 00:00 UK 时间戳）；
     - **同日单发检查**：统计今日（UK 自然日）内非 ETF 个股的新开 `BUY` 成交 + 今日新建的 `NEW_EQUITY_BUY` 挂单，若 `>= max_new_equity_entries_per_day (1)` → `daily_cap_reached = True`；
     - **连损缩火检查（严格豁免历史旧账！）**：仅筛选 `portfolio.invest.recent_orders` 中 `side == "SELL"`、`filled_at_ms >= gate_epoch_ms`、非 ETF、且该标的开仓时间亦不早于 `gate_epoch_ms`（若可由近期买单/持仓核验；若无买单对应则视为清理 `gate_epoch_date` 前老仓直接豁免！）的平仓记录。按时间倒序若最近连续 `>= consecutive_loss_limit (2)` 笔 `realised_pnl < 0`：
       - 若距最后一笔止损 `< consecutive_loss_cooldown_days` → `in_loss_cooldown = True`；
       - 若刚过冷静期且其后尚无新的 `is_financed == True` 仓位 → `risk_budget_scale = recovery_pilot_risk_scale (0.5)`（单笔风险预算提示为 `£42.5` 试探仓），否则 `1.0`。
3. **`evaluate_market_regime(histories, macro, strategy, prev_gate_state=None)`**：
   - 优先取 `histories[strategy.regime_benchmark_symbol]`（`"SPY"`），缺失时回退 `histories[strategy.benchmark_symbol]`；若均无足够数据或 `macro.is_stale` 则返回 `NEUTRAL` + 降级备注。
   - 按第 二.5 节公式计算当日原始状态 `raw_regime`（`RISK_OFF / NEUTRAL / RISK_ON`），并结合 `^VIX` 近 3 日序列或 `prev_gate_state` 执行 `enter=1d / exit=3d` 确认迟滞，彻底杜绝单日假突破与 `VIX < 18` 死锁。
4. **`scan_triggers_and_decide(...)`**：
   - 计算 A（宽基回撤阶梯 + 右侧上升趋势个股回踩 `MA50` 且 `MA50 > MA200` 且 `MA200` 20日斜率 `> 0`，排除任一账户已持有或已挂单标的、5日内财报/除息、`RSI >= 72`）、B（仅 ISA 宽基 `dca`）、C（当日 `DEPOSIT - WITHDRAW` 净外部入金）、D（`RISK_OFF` 防守协议或 ISA `<2%` 极端低现金）。
   - 输出账户级决策（`isa_gate`, `invest_gate`）与全局归并状态（`DEFEND > RUN > ROTATE > WAIT > UNKNOWN`），当 Invest 槽位已满（如 `1/1`）但 A 触发器命中右侧个股时，Invest 状态设为 `ROTATE`（附带候选腾挪源：Invest 未脱险弱仓或未成交个股买单）。
5. **`reconcile_pm_pick_gate(gate_dict, pm_pick_gate, whitelist_symbols)`**：
   - 强制执行单向收紧权：若 Python 状态为 `WAIT / DEFEND / UNKNOWN`，强制 `allow = False`；若 `ROTATE` 无合法 `funding_from`，降级为 `WAIT`；剔除不在白名单内的 `symbols`。
6. **`format_deployment_gate_banner(gate_dict)`**：
   - 生成 `<= 110` 字、最多 2 行的紧凑纯文本（禁用 `#` 与 `**`），当缺少 `gbp_usd` 汇率时优雅附注汇率待校准提示。
7. **`replay_deployment_gate_history(bench_df, vix_df, strategy, hy_oas_series=None)`**：
   - 纯离线回放工具函数：输入 2 年日线序列，逐日推进状态机并输出统计摘要（各状态占比天数、切换次数、是否存在连续 `>60` 日死锁在 `RISK_OFF` 而指数创 60 日新高的异常），直接供单测断言！

---

### Task 3：`src/engine/portfolio_rules.py` 协同改造（零 Quota 豁免 + 组合集中度 + 边角料顺势归核）

1. **导出共享常量**：定义模块级常量 `DEFAULT_VIX_DANGER_THRESHOLD: float = 28.0`，供 `apply_risk_override_guardrail` 与 `deployment_gate.py` 唯一同源引用。
2. **宽基/ETF 与老仓清理「零 Quota 豁免」（修改 `_cadence_violation` 及 `apply_risk_override_guardrail`）**：
   - 若目标动作标的是 ETF（属于 `etf_roots` 或 `strategy.benchmark_symbol` 或 `universe.etf_symbols`），**直接返回 `None`（100% 豁免 `FREQUENCY_CAP` 与冷却检查）**！
   - 统计近 7 天 / 30 天开仓笔数用于拦截新个股时，过滤掉 ETF 买单（利用 `portfolio.isa.recent_orders` + `portfolio.invest.recent_orders` 动态统计 `equity_buys_7d / equity_buys_30d`，使买 `VWRP.L` 永远消耗 `0` 次个股配额；卖出 `SELL` 本就不计入 `buys`，确保清边角料 + 买宽基 = 0 Quota）。
3. **组合口径单股上限（`12%`）与 ISA 卫星封顶（`30% / 8只`）**：
   - **保留现有 `max_isa_single_weight_pct = 30.0` 与 `max_invest_single_weight_pct = 25.0` 默认值不变**（`compute_risk_based_size_gbp` 依赖 `25.0%` 封顶计算单仓上限，存量单测 `test_risk_based_sizing_note_with_shares` 断言 `"以损定仓上限 £500"`）。
   - 在 `apply_risk_override_guardrail`（`src/engine/portfolio_rules.py:1389` `if not is_etf:` 分支内，**必须排在 `DAILY_VETO`、`REPORT_ACTION_CONFLICT`、`CORRELATION_OVERLAP`、`CADENCE` 之后**，严禁提前顺序）：
     - 对非 ETF 个股，检查跨双账户合并权重 `issuer_w >= strat.max_single_name_portfolio_pct (12.0%)`，超限返回 `("ISSUER_OVERWEIGHT", ...)` 拦截 `ADD / BUY`；
     - 新增可选入参 `deployment_gate: dict[str, Any] | None = None`：仅当 `is_pick == True` 且 `deployment_gate` 显式传入并标记 `isa_satellite_saturated == True` 时，对 `acct_up == "ISA"` 的**新标的**非 ETF `BUY` 返回 `("ISA_SATELLITE_SATURATED", "ISA 个股卫星层已饱和，暂停新开个股，请归拢核心宽基")`（不传入 `deployment_gate` 的存量旧单测保持 100% 兼容）。老仓绝不触发强制卖出。
   - **严禁删除 `apply_risk_override_guardrail` 末尾第 1468 行的 `MACRO_VIX` 检查**（存量单测 `test_analytics_and_rules.py:1125` 断言早报 `top_actions` 中的 `ADD VWRP.L` 在 `VIX=30.5` 时降级为 `HOLD`；ISA 宽基深跌阶梯播种由 `deployment_gate.py` 在门控层独立表达）。
4. **ISA `<£1,200` 历史边角料「温和顺势归核」逻辑（`evaluate_holding` & `evaluate_portfolio`）**：
   - 在 `HoldingEvaluation`（`portfolio_rules.py:21`）新增布尔字段 `is_crumb_prune_candidate: bool = False`（**注意：`action_bias` 合法枚举值严格限定为 `"ADD_CANDIDATE" | "TRIM_ALERT" | "HOLD" | "REVIEW_STOP"`，严禁自造 `"CONSIDER_TRIM"`！**）。
   - 在 `evaluate_holding` 中识别 `is_isa_legacy_crumb = (not is_invest) and (not is_etf) and (0 < pos.current_value < strategy.min_isa_equity_order_gbp)`：
     - **仅当**处于顺风反弹窗口：`pos.return_pct >= strategy.isa_crumb_near_breakeven_floor_pct (-3.0%)` **且** `tech is not None` **且**（`(tech.rsi14 is not None and tech.rsi14 >= 60.0)` 或 `(tech.drawdown_20d_pct is not None and tech.drawdown_20d_pct >= -2.0)`），且 `not earnings_soon`，将 `trim_score` 推至触发 `"TRIM_ALERT"`，追加理由：`历史小额散仓(£{pos.current_value:.0f}<£{strategy.min_isa_equity_order_gbp:.0f})反弹接近回本/阻力窗，可顺势归拢至核心宽基(零Quota损耗)`，并设 `is_crumb_prune_candidate = True`；
     - 在 `evaluate_portfolio`（`portfolio_rules.py:918`）汇总返回前，筛选所有 `ev.is_crumb_prune_candidate and ev.action_bias == "TRIM_ALERT"` 的项，按 `(ev.return_pct, -pos_val)` 降序排列，**仅保留前 `strategy.isa_crumb_pruning_max_per_report (2)` 只维持 `"TRIM_ALERT"`**，超出部分用 `model_copy` 降回 `action_bias = "HOLD"`（理由替换为 `存量小额散仓静待反弹窗口分批归核，不低位急割`），从算法层面杜绝催促盲目清仓！

---

### Task 4：`src/storage/db.py` 只读历史门控状态衔接

- 在 `AdvisorDatabase` 中新增纯只读查询方法 `get_recent_context_packs(self, limit: int = 5) -> list[dict[str, Any]]`（读取最近 `N` 条 `morning_reports` 的日期与 `context_pack_json`，走 `self._connect(write=False)`，**不改任何表结构**），供 `deployment_gate` 读取前 1~3 个交易日的 `raw_regime` 完成 3 日确认迟滞计算。

---

### Task 5：Schema、Prompt、语义质检与 `/report` 流水线接线（`schemas.py`, `prompts.py`, `semantic_auditor.py`, `report_agent.py`）

1. **`src/agents/schemas.py`**：
   - 新增 `PickGateDecision(BaseModel)`（含 `allow: bool = True`, `mode: str = "none"`, `funding_from: str = ""`, `reason: str = ""`），并在 `MorningReportDecision` 上新增带默认值的 `pick_gate: PickGateDecision = Field(default_factory=PickGateDecision)`。
   - `MorningReportDecision.macro_and_cash` 的 `description` **必须保留 `"VWRP"` 字样且严禁出现 `"4%现金安全垫"`**（存量单测 `test_db_and_end_to_end.py:2507` 强断言）。
   - `render_morning_report(..., gate_banner: str = "")`：当 `gate_banner` 非空时，将其插在 `sec2`（`【宏观、账户与今日风控红线】`）标题下一行（首行位置，防止被 `_fit_max_chars` 尾部裁剪器截断）。
2. **`src/agents/prompts.py`**：
   - 在 `build_investor_profile` 中追加「ISA 核心宽基 ≥60% / 个股卫星 ≤30%·≤8只 + 存量散仓逢反弹顺势归拢不低位急割 + 清散仓与买宽基零 Quota + Invest 在险槽位以盈解锁」规则。
   - **存量单测字面量强保留红线（`test_db_and_end_to_end.py:2485-2517` & `:1503`）**：修改 `build_investor_profile` 时，**严禁删除或改写**以下 7 处精确字面量（连空格与引号都不可改）：
     1. `"VWRP.L"`
     2. `"£85"`
     3. `"R:R ≥ 2:1"`（注意 `≥` 两侧有空格）
     4. `"卖出三分类"`
     5. `"严禁把“守住现金吃息”当默认结论"`（保留中文弯引号）
     6. `"一律按挂单限价核算"`（受 `include_order_audit_rule` 开关控制）
     7. `"照常产生同等日息"`
     且严禁出现 `"必须直白建议撤单"`、`"明显跑赢4.0%现金利息"`。
3. **`src/agents/semantic_auditor.py`**：
   - 在 `build_fact_baseline`（第 70–74 行）输出 `ISA / INVEST` 现金时，补充打印净可用现金占比与总现金占比（如 `可用 12,121.34 (可用占比 21.1%) | 挂单占用 8,741.16 | 总现金 20,862.50 (总现金占比 36.4%)`），防止 DeepSeek 语义质检把门控横幅的 `21.1%` 与 FactSheet 的 `36.4%` 对撞误判为数字幻觉告警！
4. **`src/agents/report_agent.py`**：
   - **假桩安全探测（防 `AttributeError` 暗雷）**：存量 E2E 单测中的 `_Imp12MarketData` / `_FakeMarketData` 仅实现了 `get_bundles` 与 `get_macro_snapshot`，**未实现 `get_histories` 方法**！因此预拉取 `"SPY"` / `"^VIX"` 与观察池日线时，必须先判 `if callable(getattr(market_data, "get_histories", None)):`（参考第 470 行 `hasattr(market_data, "get_us_deep_intel_batch")` 的防御写法），若假桩无此方法则直接复用已有 `histories` 优雅降级。
   - 将预计算的客观门控摘要传入 Step 3 PM Prompt，指示 PM「仅在发现重大事件风险时用 `pick_gate.allow=false` 收紧，严禁把 `WAIT/DEFEND` 升为 `RUN`」；
   - **关键顺序（防同屏打架）**：在 Step 3 解析完成并执行 `apply_risk_override_guardrail` **之后**，调用 `evaluate_deployment_gate` + `reconcile_pm_pick_gate`（**全程 0 额外 LLM 调用**，不得改变现有单测的 `llm.calls` 计数）：
     - `deployment_gate_mode == "off"`：完全跳过，报告逐字不变；
     - `deployment_gate_mode == "log"`：写入日志与 `context_pack["deployment_gate"]`，不传 `gate_banner`；
     - `deployment_gate_mode == "advise"`：同时传入 `gate_banner` 渲染至早报第二段首行，确保总长 `<= 1800` 字。

---

### Task 6：`/pick` 选股雷达闭环接线（`src/agents/pick_agent.py`）

1. **严禁在 Researcher/Critic 构造提示词前直接从 `isa_cands` 删除候选标的（防存量单测崩溃！）**：
   - *原因*：存量单测 `test_db_and_end_to_end.py:1351, 1355, 1498` 使用 `_sample_portfolio()`（其中 `NVDA` 占 ISA `70%`），并强断言 Researcher Prompt 中包含 `AZN.L` 的历史教训（`"英股医药除息前追高易回撤"`）、除息日（`"除息日 2026-11-20"`）以及早报挂单标签（`"⚠️今日早报已裁决: 挂单MODIFY(ISA)"`）！如果在前段直接删掉 `AZN.L`，这三条存量单测会立即崩溃！
2. **正确的门控联动方式（提示词约束 + 后置确定性拦截，且按 `context_pack["deployment_gate"]` 存在性门控）**：
   - 从最新 `context_pack` 读取 `gate_ctx = context_pack.get("deployment_gate")`（注意：旧单测手工构造的 `seeded_pack` 不含 `"deployment_gate"` 键，`gate_ctx` 为 `None` 时保持原有行为 100% 不变！）；
   - 当 `isinstance(gate_ctx, dict)` 且 `cfg.strategy.deployment_gate_mode == "advise"` 时：
     - 在 Researcher 与 Critic Prompt 顶部注入 `[今日组合门控约束]` 块（写明 ISA 个股是否饱和仅限宽基、Invest 当前可用在险槽位、`ROTATE` 要求的置换源）；
     - 在 `_post_screen_validate_decision` 调用 `apply_risk_override_guardrail(..., deployment_gate=gate_ctx)`：
       - 若 `isa_satellite_saturated == True`：拦截 `ISA` 账户的非 ETF **新开个股** `BUY`（归入 `rejected_or_frozen`，保留 `VWRP.L` 等宽基 ETF 通过）；
       - 跨账户同票去重：若某账户推荐 `BUY` 另一账户已持有或已有活动 `BUY` 挂单的个股，拦截并提示归口原持仓账户管理；
       - 若 Invest 为 `ROTATE` 状态：校验 Invest `BUY` 动作的 `thesis` 或 `risk_note` 中包含合法的 `funding_from` 标的，否则拦截；
       - 若 Invest `risk_budget_scale < 1.0`（连损恢复首单 `0.5R`）：通过临时 `strategy.model_copy(update={"invest_risk_budget_gbp": strategy.invest_risk_budget_gbp * scale})` 传入算仓逻辑，使建议股数自动按 `£42.5` 半仓试探单生成！

---

### Task 7：离线单测矩阵与全量验收（`tests/test_report_pick_gate.py`）

新建 `tests/test_report_pick_gate.py`，全部离线 Mock（禁止任何真实网络或读取线上 `data/advisor.db`），至少覆盖以下 **8 组核心测试集**：

1. **账本恒等式与挂单三分法回归（2026-11-17 实盘精确复现）**：
   - 复现 ISA `£57,343.73`、总现金 `£20,862.50`、挂单 `£8,741.16`（`SHEL.L` `3480 GBp` + `RKT.L` `5250 GBp` + `VWRP.L` + `VUAG.L`）、可用 `£12,121.34` → 断言 `deployable_pct == 21.14%`（绝不等于 `5.9%`！），断言 `committed_satellite_count == 15`、`committed_satellite_pct` 约 `41.9%`、`isa_satellite_saturated == True`。
   - 复现 Invest `AMZN`（均价 `$238.42`，`STOP $243.50`）与 `CF`（均价 `$121.41`，`STOP $113.50`） → 断言 `STOP` 卖单绝不触发「未审买单阻塞」，断言 `AMZN` 判定为 `is_financed == True`（占 `0` 槽位，本金在险 `£0`），`CF` 占 `1` 槽位（本金在险约 `£82~£108`）。
2. **Invest 在险槽位与以盈解锁**：
   - `NEUTRAL` 下 `CF` 未脱险（`1/1` 满槽）且无 A 触发器 → Invest 输出 `WAIT`（哪怕现金高达 `79.7%` 也绝不输出 `RUN·dca`！）；
   - 将 `CF` 的 `STOP` 挂单调高至 `$122.50`（高于均价+摩擦） → 断言槽位变为 `0/1`，A 触发器命中时立即解锁输出 `RUN·targeted`；
   - `NEUTRAL` 满槽 `1/1` 时 A 触发器命中右侧强股 → 断言输出 `ROTATE`（附 `funding_from="CF"`），绝不允许净增第 2 个在险槽位；
   - 同日已成交 1 笔新个股 → 当日再次评估输出 `WAIT`（Best Idea Only 单日上限）。
3. **新纪元起算锚点（`gate_epoch_date`）与零 Quota 豁免**：
   - 构造 `gate_epoch_date` 之前的 3 笔历史亏损卖单（模拟历史 `-£1,335` 旧账）+ 新纪元后卖出 3 只 ISA `<£1,000` 历史边角料并连买 3 笔 `VWRP.L` → 断言：**连损冷静期未触发（`in_loss_cooldown == False`）**、**个股周/月买入 Quota 占用为 `0`**、**Invest 槽位未受任何影响**！
   - 构造 `gate_epoch_date` 之后在 Invest 新开并连续止损 2 笔个股 → 断言触发冷静期 `WAIT`，且冷静期满后首笔试探仓风险缩放为 `0.5R (£42.5)`。
4. **ISA 历史边角料「循序渐进顺势瘦身」防盲目清仓回归**：
   - 构造 8 只 `<£1,000` ISA 碎单均处于缩量下跌（浮亏 `-8%~-19%`，`RSI = 42`） → 断言 `0` 只被催促清仓（全部温和 `HOLD` 等待反弹窗）；
   - 构造其中 4 只反弹至微盈且 `RSI = 65` → 断言单次最多仅浮出 `2` 只提示顺势归核，其余保持 `HOLD`。
5. **去汇率失真 Regime、无死锁退出与 2 年合成历史回放**：
   - 构造 `SPY` 大涨但英镑暴涨导致 `VWRP.L` 虚跌的场景 → 断言 Regime 以 `SPY` 为准不被汇率误导；
   - 构造 500 个交易日合成序列（含暴跌进入 `RISK_OFF`，随后指数反弹站稳 `MA200` 但 `VIX` 长期停留在 `21.0`） → 运行 `replay_deployment_gate_history`，断言 3 日确认后顺利退回 `NEUTRAL`（绝不因 `VIX > 18` 永久死锁在 `RISK_OFF`！）。
6. **大师四步 `DEFEND` 与双账户逆向耦合**：
   - `VIX = 30` 进入 `RISK_OFF`：断言 Invest 输出 `DEFEND`（提示撤未成交个股买单、保留 `AMZN` 保本仓与 `CF` 止损保护仓、清退无保护浮亏仓），而 **ISA 宽基在回撤达 `-15%` 阶梯时依然允许宽基播种，不被高 VIX 误杀锁死**。
7. **PM 单向权限钳位与早报 1800 字硬上限**：
   - PM 试图把 `WAIT` 或 `DEFEND` 改成 `allow=True` → 被 Python 强制驳回；
   - 构造超长早报文本 + 门控横幅 → 断言最终输出严格 `<= 1800` 字、不含 `#` 与 `**`、且位于第二段首行的门控横幅完整无缺；
   - `deployment_gate_mode = "off"` → 输出与旧版逐字节一致，LLM 调用次数零增加。
8. **`/pick` 赛道过滤联动**：
   - `isa_satellite_saturated == True` 时运行 `/pick`，断言 ISA 赛道个股候选被过滤，仅保留 ETF 候选。

---

## 五、次级模型防呆速查卡（9 大存量契约暗雷与精确字段字典）

> **给执行模型的特别警告**：本仓库已有近万行存量离线单测，以下 9 处细节已在源码中逐行核实，**严禁凭直觉猜字段名或符号方向**，否则必挂存量测试！

| # | 极易踩坑点 | 源码真实契约（必须严格遵守） | 关联文件与行号 |
| :---: | :--- | :--- | :--- |
| 1 | `TechnicalSnapshot` 字段名与正负号 | • RSI 字段名为 **`tech.rsi14`**（无下划线，**不是** `rsi_14`！）<br>• ATR 为 `tech.atr14` / `tech.atr_pct`，ADX 为 `tech.adx14`<br>• **`drawdown_20d_pct` 与 `drawdown_52w_pct` 为负数**（如回撤 15% 存为 **`-15.0`**），与正数配置 `regime_drawdown_risk_off_pct=15.0` 比较时必须写 **`drawdown <= -abs(threshold)`**！ | `src/data_providers/analytics.py:28-50` |
| 2 | `HoldingEvaluation.action_bias` 枚举 | 合法值严格只有 4 个：**`"ADD_CANDIDATE" \| "TRIM_ALERT" \| "HOLD" \| "REVIEW_STOP"`**（严禁发明 `"CONSIDER_TRIM"`） | `src/engine/portfolio_rules.py:27` |
| 3 | 英股便士 `GBp` 100 倍量级缩放 | T212 挂单如 `SHEL.L` 的 `limit_price=3480.0, currency="GBp"`，与英镑或现价比较前**必须调用 `_rescale_price_pair(price, ref)`**，估算英镑市值直接复用 `_estimate_pending_order_gbp` | `src/engine/portfolio_rules.py:286, 384` |
| 4 | `_cadence_violation` 当 `portfolio=None` 时的向后兼容 | `tests/test_analytics_and_rules.py:2633-2737` 在不传 `portfolio` 的情况下直测 `trade_behavior={"buys_7d": 2, ...}` 并断言 `"近 7 天已新开仓"`、`"本周额度已用完"`、`"[⚠️纪律提醒]"`、`"advisory 模式不拦截"`。因此：① 当目标标的为 ETF 时返回 `None`；② 当 `portfolio is not None` 且 `recent_orders` 非空时重算剔除 ETF 后的个股买入数覆盖 `buys_7d/buys_30d`，**`portfolio is None` 时原样读取 `trade_behavior`**！ | `src/engine/portfolio_rules.py:1210-1251` |
| 5 | `_check_symbol_violation` 拦截优先级 | 存量测试 `:1177` 断言 `REPORT_ACTION_CONFLICT` 优先于超重拦截。新增的 `12%` 组合单股上限与 `ISA_SATELLITE_SATURATED` 必须放在 `DAILY_VETO` → `REPORT_ACTION_CONFLICT` → `CORRELATION_OVERLAP` → `CADENCE` **之后**的 `if not is_etf:` 块内 | `src/engine/portfolio_rules.py:1353-1403` |
| 6 | 单仓算仓上限 `£500` 断言保护 | `test_risk_based_sizing_note_with_shares`（`:2985`）断言 `max_invest_single_weight_pct=25.0` 算出的 `"以损定仓上限 £500"`。严禁修改 `max_invest_single_weight_pct=25.0` 与 `max_isa_single_weight_pct=30.0` 默认值 | `tests/test_analytics_and_rules.py:2962-2988` |
| 7 | E2E 假桩无 `get_histories` 方法 | `_Imp12MarketData` / `_FakeMarketData` 未定义 `get_histories`。调用前必写 `if callable(getattr(market_data, "get_histories", None)):` | `tests/test_db_and_end_to_end.py:950, 1492` |
| 8 | `/pick` 不可前置删 `AZN.L` 候选 | `test_imp13` / `test_imp12_sync` 断言 Researcher Prompt 含 `AZN.L` 的教训、除息日与挂单标签。严禁在构造 Prompt 前过滤掉 `isa_cands` 中的个股，一律靠 Prompt 头部注入约束 + `_post_screen_validate_decision` 后置硬拦截（且仅在 `context_pack` 含 `"deployment_gate"` 时生效） | `tests/test_db_and_end_to_end.py:1351-1358, 1497-1498` |
| 9 | DeepSeek 语义质检现金百分比防误报 | `FactSheet` 打印总现金占比 `36.4%`，门控打印净可用占比 `21.1%`；必须在 `semantic_auditor.build_fact_baseline` 中同时写出这两个百分比，避免触发假幻觉告警 | `src/agents/semantic_auditor.py:66-75` |

> **建议次级模型的施工节奏**：每完成一个 Task（如写完 `Task 2`、改完 `Task 3`、接完 `Task 5`），**立刻跑一次** `.\.venv\Scripts\python.exe -m tests.run_tests tests` 确认存量单测 100% 绿灯，再进入下一个 Task，切忌攒完 10 个文件一起跑！

---

## 六、交付验收命令与红线自查表（施工收尾必跑）

施工全部完成后，依次运行以下三条验收命令并确保全部零报错通过：

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.run_tests tests
.\.venv\Scripts\python.exe scripts\audit_egress.py
```

并在最终交付说明中显式附上 **`AGENTS.md` §5.3 五项自查结果** 与红线文件 diff 摘要：
1. 是否新增 `http(s)://` 出网字面量或新域名？ → **否**
2. 是否新增日志/异常打印 `api_key / token / secret / cfg.env`？ → **否**
3. 是否新增读取环境变量（尤其凭据类）？ → **否**
4. 是否改动 `base_url` / 超时 / 重试 / 灾备链？ → **否**
5. 是否改动风控护栏开关？ → **是（新增 `deployment_gate_mode`，并扩充 `StrategySection` 门控与槽位参数；已在交付说明中逐行展示 `src/config.py` 与 `pyproject.toml` 版本号 diff 供人工确认）**。
