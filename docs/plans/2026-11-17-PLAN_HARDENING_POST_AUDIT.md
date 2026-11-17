# Trading 212 × Gemini AI Advisor — 风控闭环加固与记账纠偏施工蓝图

## 0. 执行约束与安全红线（Agent 动手前必读）

1. **严守仓库 `AGENTS.md` 红线**：
   - **禁止修改任何 5.2 红线文件**：本次施工**严禁触碰** `src/config.py`、`src/agents/llm_client.py`、`src/data_providers/t212_client.py`、`src/ops_notifier.py`、`pyproject.toml`、`requirements.txt`、`Dockerfile`、`docker-compose.yml`。所有修复均在非红线业务模块内闭环完成，**零新增第三方依赖**。
   - **读取规范**：读取 >400 行文件前，必须先用 `grep` 定位类/函数行号，再按 100~200 行切片读取，禁止整篇吞读大文件；严禁读取 `.env*`、`data/`、`logs/`、`*.db*`。
   - **5.3 风控开关变更显式确认**：阶段一涉及 `config.yaml` 中 `trade_cadence_guard_mode`、`pick_rr_guard_mode`、`user_preference_memory_enabled` 三项风控开关调整，已由账户持有人明确授权，提交变更说明时须在顶部显式列出。
2. **向后兼容现有离线单测契约**：
   - 修改每阶段代码后，必须运行对应单测模块与全量离线测试套件，严禁破坏既有绿灯用例（仅在修正明确公式 Bug 如 `FEE/INTEREST` 记账时同步更新对应断言并增补新用例）。
3. **严禁实施以下两项反向负优化**：
   - **不改** `isa_deep_drawdown_review_pct: -25.0`，**不把** ISA 存量个股改为机械触发 `REVIEW_STOP`（防免税仓底部洗盘割肉）；
   - **不新增** 穿透科技暴露百分比硬拦截阈值（防 `/pick` 被迫转向高印花税低增长价值陷阱做劣质分散）。

---

## 1. 施工目标与四阶段总览

| 阶段 | 核心目标 | 涉及文件 | 预期效果 |
| :--- | :--- | :--- | :--- |
| **Phase 1** | 激活硬护栏、封死 `/pick` 四大穿仓口与伪宽基漏洞 | `config.yaml`, `src/engine/deployment_gate.py`, `src/engine/portfolio_rules.py`, `src/agents/pick_agent.py` | 终结超频冲动推荐、拦截 Invest `WAIT` 漏网买单、消除 Critic 解析失败裸推与英股 `GBp/GBP` 100 倍算仓爆炸、剥夺主题/高息 ETF 的核心宽基特权 |
| **Phase 2** | 打通危机期宽基抄底、修复 `DEFEND` 进退场死锁与自适应算仓 | `src/engine/deployment_gate.py`, `src/engine/portfolio_rules.py`, `src/agents/pick_agent.py` | `VIX>=28` 不再误杀核心宽基定投、深跌阶梯兼顾 `SPY` 原币种回撤、消除熊市死猫跳 3 日假退场与见底后旧高点死锁、实现 ISA/Invest `0.5% NAV` 免调参动态算仓 |
| **Phase 3** | 清除催买 Prompt 冲突与门控财报排雷盲区 | `src/agents/report_agent.py`, `src/agents/pick_agent.py` | 补齐门控候选股财报/除息日历过滤，删除逼迫模型对 Invest 高比例现金强凑方案的催买指令 |
| **Phase 4** | 修正 T+N 结算方向符号与 TWR 现金利息归因 | `src/bot/scheduler.py`, `src/data_providers/analytics.py`, `src/agents/report_agent.py`, `CHANGELOG.md` | 摆正 `SWAP`（多头）与 `TRIM/STOP` 基准超额 Alpha 符号、修复盘后推荐误取当日盘中低点判成交、让 4% 现金利息真实计入 TWR 净值回报 |

---

## 2. 分阶段详细施工步骤

### Phase 1：硬护栏激活、伪宽基清洗与 `/pick` 穿仓封堵（最高优先级）

#### Step 1.1：`config.yaml` 切换确定性硬护栏并关闭散户情绪偏好抽取
- **文件**：`config.yaml`
- **定位**：第 124、131、184 行附近
- **修改内容**：
  1. `trade_cadence_guard_mode: "advisory"` → 改为 `trade_cadence_guard_mode: "hard"`
  2. `pick_rr_guard_mode: "advisory"` → 改为 `pick_rr_guard_mode: "hard"`
  3. `user_preference_memory_enabled: true` → 改为 `user_preference_memory_enabled: false`

#### Step 1.2：清洗 `_build_etf_set` 核心宽基白名单 + 修复 `_native_risk_to_gbp` 便士识别 + 校验保本止损股数覆盖率
- **文件**：`src/engine/deployment_gate.py`
- **修改点 A（`GBp` 小写便士百倍热力值修复，约第 167–180 行 `_native_risk_to_gbp`）**：
  - 现状：`ccy = str(pos.inst_currency or "").upper()` 把 `"GBp"` 转成了 `"GBP"`，导致英股便士风险未除以 100。
  - 改为：先保留原始字符串 `raw_ccy = str(pos.inst_currency or "").strip()`；若 `raw_ccy in {"GBp", "GBx"} or raw_ccy.upper() == "GBX"`，直接返回 `native_amount / 100.0`；再判断 `raw_ccy.upper() == "GBP"` 返回 `native_amount`。
- **修改点 B（剥离主题/高息/债券 ETF 的核心宽基特权，约第 182–196 行 `_build_etf_set`）**：
  - 从 `src.data_providers.analytics` 导入 `infer_etf_index_family`。
  - `_build_etf_set` 始终包含 `_sym_up(strategy.benchmark_symbol)`；遍历 `universe_etf_symbols` 时，**仅保留底层指数家族属于全球宽基或标普 500 的真正核心宽基**（即 `infer_etf_index_family(sym, (bundles or {}).get(_sym_up(sym))) in {"Global All-World/Developed (全球宽基)", "S&P 500 (美股标普500)"}`）。
  - 效果：`VWRP.L / VWRL.L / VUAG.L / SWDA.L` 保留在 `etf_set`（兼容全部现有单测），而 `IUKD.L / VHYL.L / SMGB.L / INRG.L / IITU.L / IGLT.L / SGLN.L / CNX1.L` 自动退入 `other_etf_gbp`，不再虚增 `core_etf_pct` 且不再享受核心宽基通道豁免。
- **修改点 C（Invest「以盈解锁」强制核验止损挂单覆盖股数，约第 493–518 行 `compute_account_ledgers`）**：
  - 现状：仅检查 `stop_orders[0]` 的价格是否 `>= avg_price * (1 + friction)`，忽略了挂单股数。
  - 改为：汇总该持仓所有有效保护止损单股数 `covered_qty = min(float(pos.quantity), sum(float(o.quantity or 0.0) for o in stop_orders if (o.stop_price or o.limit_price)))`，并按各止损单股数计算加权对齐止损价（单张挂单时即 `aligned_stop`）：
    - 仅当 `pos.quantity > 0 and covered_qty >= float(pos.quantity) * 0.99 and pos.avg_price > 0 and aligned_stop >= pos.avg_price * (1.0 + friction_pct / 100.0)` 时，才置 `is_financed = True, heat_gbp = 0.0`；
    - 否则 `is_financed = False`：已覆盖部分 `covered_qty` 按 `max(0.0, float(pos.avg_price) - aligned_stop) * covered_qty` 计算本币在险额，未覆盖裸露部分 `uncovered_qty = max(0.0, float(pos.quantity) - covered_qty)` 按自适应止损比例 `float(pos.avg_price) * abs(float(_compute_adaptive_invest_stop_pct((tech_map or {}).get(pos.yahoo_symbol), strategy))) / 100.0 * uncovered_qty` 计算本币在险额，两者相加经 `_native_risk_to_gbp` 折算为 `heat_gbp`。

#### Step 1.3：修复英股 `entry_price` 百倍错位、双钱包真实往返换汇摩擦与买后单票超限核查
- **文件**：`src/engine/portfolio_rules.py`
- **修改点 A（Invest 未留存 USD 时的双边换汇摩擦校准，约第 142–150 行 `_round_trip_friction_pct` 与第 376–384 行 `_estimate_entry_friction_pct`）**：
  - 现状：`usd_avail > 0.01` 即按 `0%` 换汇计（$0.02 零头免除整单汇费），且 `usd_avail == 0` 时仅收单边 `fx`。
  - 改为：
    - 在 `_round_trip_friction_pct` 中，针对美股（非英/非欧）：估算持仓美元市值（或用 `usd_avail >= max(50.0, float(pos.current_value or 0.0) * 0.5)` 判定是否真正启用了 USD 留存）；若 `wallet` 中 `usd_cash_total <= 0.01` 且 ` wallet.recent_orders` 存在美股 `SELL` 挂钩 `wallet_currency == "GBP"`，按 `fx * 2.0 + spread` 计双边摩擦；若 `usd_avail` 仅覆盖部分或充足留存 USD，则只有当 `usd_avail >= 100.0` 时免去买入单边 `fx`。`_estimate_entry_friction_pct` 同理将 `usd_avail > 0.01` 提升为 `usd_avail >= float(strategy.min_invest_order_gbp)`（不足整笔最小建仓额时计入单边 `fx`）。
- **修改点 B（`/pick` 入场价向实时行情做 100 倍单位对齐，约第 424–437 行 `_resolve_pick_rr` 与第 497 行 `_shares_from_gbp`）**：
  - 在 `_resolve_pick_rr` 读取 `entry` 后，若 `bundle is not None and bundle.current_price and float(bundle.current_price) > 0`：执行 `rescaled_entry = _rescale_price_pair(entry, float(bundle.current_price))`；若 `rescaled_entry != entry`，置 `out["price_rescaled"] = True` 并更新 `entry = rescaled_entry`，随后再对 `stop` 和 `target` 执行 `_rescale_price_pair(..., entry)`。
  - 在 `_shares_from_gbp` 第 497 行：删除 `elif not raw_ccy and symbol.upper().endswith(".L"): price_gbp = price_local / 100.0`（当 `.L` 标的缺失 `currency` 时返回 `None`，严禁把 `VWRP.L` 等英镑报价 ETF 盲除 100）。
- **修改点 C（买后单票集中度越界拦截與 `ADD` 真实存量市值传入，约第 590–616 行 `_build_pick_size_note` 与第 1598–1618 行 `apply_risk_override_guardrail`）**：
  - `_build_pick_size_note` 新增可选参数 `existing_position_value_gbp: float = 0.0`，并在第 1769 行调用处传入该账户对该标的（或同 `issuer_root`）的当前持仓市值，修复 `ADD` 算仓忽略存量仓位的问题。

#### Step 1.4：封死 `/pick` 的 Critic 解析失败裸推与 Invest `WAIT(mode="none")` 通道漏拦
- **文件**：`src/agents/pick_agent.py`
- **修改点 A（Invest 非开放状态全拦截 + 同轮剩余槽位递减，约第 132–205 行 `_apply_gate_channel_constraints`）**：
  - 在循环外计算 Invest 当前可新开个股配额：`rem_invest_slots = max(0, int(inv_g.get("slots_cap") or 0) - int(inv_g.get("slots_used") or 0))`（若 `status == "ROTATE" and funding_from` 则允许 `1` 笔置换）。
  - 第 151–165 行 `if acct == "INVEST":` 分支：
    1. 若 `is_etf` 且该 ETF **不在** `etf_set`（非核心宽基）时按个股通道受控；若在应税 Invest 推荐累积型 `VWRP.L/VUAG.L`，仅放行派息型或明确合规标的，但核心是：对 `not is_etf` 个股，将原 `if status == "WAIT" and mode in {"cooldown", "daily_cap", "slots_full"}:` **收紧为**：只要 `status not in {"RUN", "ROTATE"}`（即涵盖 `DEFEND`、全部 `WAIT` 子模式含 `"none"`、以及 `UNKNOWN`），一律硬拦截并记入 `notes`！
    2. 对通过检查的 Invest 新开个股（非已持仓 `ADD`），每次保留 1 笔后扣减 `rem_invest_slots -= 1`（同时受 `max_new_equity_entries_per_day - today_new_equity_entries` 约束）；当剩余配额 `<= 0` 时，同次 `/pick` 后续多余的 Invest 个股 `BUY` 一律拦截（防止单次 `/pick` 连推 2~3 笔击穿单日上限与槽位上限）。
  - 第 121–131 行与第 182–187 行跨账户同票去重：补充 `normalize_issuer_root(sym)` 比对，防止 `GOOGL` 与 `GOOG` 跨户双持漏网。
- **修改点 B（Critic 结构化解析失败改 Fail-Closed，约第 1173–1177 行 `generate_pick_radar`）**：
  - 将原 `final_text = resolved_critic_raw.strip()[:1000]` 替换为与 `/report` 对齐的 Fail-Closed 返回：
    ```python
    logger.warning("Pick Critic structured parse failed; refusing to emit unguarded raw text.")
    return "⚠️ /pick 风控终审结构化解析失败，已拦截未经护栏核验的原始输出，请稍后重试 /pick。"
    ```

---

### Phase 2：危机期宽基抄底解锁、`DEFEND` 状态机修补与 0.5% NAV 自适应算仓

#### Step 2.1：`MACRO_VIX` 豁免核心宽基 ETF + 0.5% NAV 免调参自适应风险预算
- **文件**：`src/engine/portfolio_rules.py`、`src/agents/pick_agent.py`
- **修改点 A（`MACRO_VIX` 不再误杀核心宽基定投，约第 1698–1700 行 `apply_risk_override_guardrail`）**：
  - 将 `if macro_high_risk and macro_vix is not None:` 改为：
    ```python
    if (not is_etf) and macro_high_risk and macro_vix is not None:
        return "MACRO_VIX", f"VIX={macro_vix:.1f}(≥{vix_danger_threshold:.0f}高危阈值)，暂停个股主动买入/加仓"
    ```
    *（注：配合 Step 1.2 收紧后的 `etf_roots` / 核心宽基判定，此处 `is_etf` 仅豁免真正宽基 ETF，可在 `deployment_gate` 传入时优先以 `sym_up in core_etf_set` 判定。）*
- **修改点 B（`0.5% NAV` 终身免调参自适应算仓，约第 212–261 行 `compute_risk_based_size_gbp` 与 `pick_agent.py:672-680`）**：
  - 在 `compute_risk_based_size_gbp` 新增可选参数 `risk_budget_scale: float = 1.0`。
  - 计算 `base_budget`：
    - 若 `not is_invest`（即 `ISA`）且 `account_total_value_gbp is not None and float(account_total_value_gbp) > 0`：
      `risk_budget = max(float(strategy.invest_risk_budget_gbp), float(account_total_value_gbp) * 0.005)`（不乘 Invest 的连损 `risk_budget_scale`！）；
    - 若 `is_invest`：
      `nav_scaled = float(account_total_value_gbp) * 0.005 if (account_total_value_gbp and float(account_total_value_gbp) > 0) else 0.0`
      `risk_budget = max(float(strategy.invest_risk_budget_gbp), nav_scaled) * float(risk_budget_scale)`
      *（验证：`account_total_value_gbp=10000` 或 `14191` 时 `0.005 * NAV <= 85.0`，保持 `£85.0` 与现有单测 100% 兼容；未来外部入金使 Invest 净值升至 £30,000 时自动扩容至 `£150`。）*
  - 在 `pick_agent.py:672-680`：仅当 `item.account.upper() == "INVEST"` 时才应用 `scale`，或仅在单笔为 Invest 时缩放预算，严禁让 Invest 连损把 ISA 的风险预算砍半。

#### Step 2.2：宽基深跌阶梯兼顾美元 `SPY` 回撤 + 修复 `DEFEND` 假退场与右侧复苏死锁
- **文件**：`src/engine/deployment_gate.py`
- **修改点 A（消除熊市死猫跳 3 日假退场与站稳 MA200 后旧高点死锁，约第 673–705 行 `evaluate_market_regime`）**：
  1. 第 676–680 行 `crisis_eased`：**删去 `or (mom20 is not None and mom20 > 0)`**，要求 `ma200 is not None and price >= ma200`（若 `ma200` 因数据不足为 `None`，才退化检查 `mom20 > 0`），杜绝指数仍在 MA200 下方仅靠 3 天超跌反弹就提前解除 `DEFEND`。
  2. 第 699–700 行 `elif prev_regime == "RISK_OFF":`：当 `ease_streak >= exit_days` 且当前唯一的 `risk_off_flags` 仅剩历史 `dd52` 回撤项（即此刻 `price >= ma200`、`vix < regime_vix_exit_risk_off`、`hy_oas < regime_hy_oas_risk_off_pct` 已全部站稳确认满 `exit_days` 天）时，允许从 `"RISK_OFF"` 阶梯式恢复至 `"NEUTRAL"`，防止大盘跌 `-25%` 后见底反转站稳 MA200 却被一年前旧高点锁死在 `RISK_OFF`。
- **修改点 B（危机期宽基阶梯防英镑贬值钝化与单日脉冲踏空，约第 726–749 行 `_core_etf_drawdown_step` 及第 911–918 行）**：
  - `_core_etf_drawdown_step` 同时计算 `strategy.benchmark_symbol`（`VWRP.L`）与 `strategy.regime_benchmark_symbol`（`SPY`）的 52 周高点回撤，取更深一侧判定阶梯触发（保持 `VWRP.L` 单独回撤单测绿灯的同时，解决危机期英镑暴跌掩盖美元资产 `-15%` 真回撤的问题）；
  - 在 `scan_triggers_and_decide` 第 912–918 行 `if regime == "RISK_OFF":` 中：除 `ladder_hit >= 10.0` 首日脉冲外，若当前 `SPY` 或 `VWRP.L` 52 周回撤持续 `<= -10.0%`、且 `isa["deployable_pct"] > float(strategy.isa_cash_band_high_pct)`、且 `isa["committed_core_etf_pct"] < float(strategy.isa_core_etf_target_pct)`、且当前 `len(isa["pending_core_etf_buys"]) == 0`（尚未挂出宽基接针单），同样保持 `isa_status = "RUN", isa_mode = "ladder", isa_etf_open = True`，避免错过破位首日后整个横盘底部无法补挂 `VWRP.L` 买单。
- **修改点 C（跨月持仓止损纳入连损统计，约第 324–337 行 `evaluate_tactical_quota_guards`）**：
  - 若 `sell` 在 `recent_orders` 中找不到配对 `BUY`（因持仓超过 `order_history_days: 30` 天），额外检查 `sell.filled_at_ms - epoch_ms >= int(strategy.order_history_days) * 86_400_000`：当 `gate_epoch_date` 距今已超 30 天时，不再把缺近期买单的亏损卖单盲目豁免为历史老仓。

---

### Phase 3：门控候选财报排雷补盲与催买 Prompt 清理

#### Step 3.1：为 `/report` 门控回踩候选补拉轻量财报/除息 Bundle，并删除 Invest 现金催买冲突词
- **文件**：`src/agents/report_agent.py`、`src/agents/pick_agent.py`
- **修改点 A（补齐未持仓门控候选的财报/除息排雷数据，`src/agents/report_agent.py` 约第 1273–1284 行）**：
  - 对 `pool_syms` 先计算 `snap = compute_technical_snapshot(sym, hist, bundle=bundles.get(sym))`；挑出满足均线回踩粗筛（`snap.dist_ma50_pct is not None and abs(snap.dist_ma50_pct) <= cfg.strategy.trigger_ma_tolerance_pct and sym not in bundles`）的前 `<= 6` 只候选代码，调用一次轻量 `await market_data.get_bundles(hit_syms, include_news=False, include_earnings=True)` 合并进 `cand_bundles`，再构造 `gate_candidates`（使 `deployment_gate.py:783-795` 的 5 日财报排雷与 5 日除息排雷真正生效）。
- **修改点 B（删除强迫模型为花钱而推荐交易的 Prompt 冲突句）**：
  - `src/agents/report_agent.py` 约第 1430–1431 行：删除 `（或 Invest 现金 >{cfg.strategy.invest_cash_idle_warn_pct:.0f}%）` 催买要求，改为：`4. 资金部署：若 ISA 现金占比 >{cfg.strategy.isa_cash_deploy_trigger_pct:.0f}% 且核心宽基未达标，给出核心宽基分批归拢价位与金额；Invest 机动仓严格遵循右侧猎物门控，无 R:R≥2:1 强信号时明确维持持币待命，严禁因现金占比高而勉强凑单。`
  - `src/agents/pick_agent.py` 约第 1109–1112 行：删除 `（或 Invest 现金 >{cfg.strategy.invest_cash_idle_warn_pct:.0f}%）` 诱导在应税 Invest 定投累积型宽基的表述，明确区分：`ISA 闲钱优先归拢 VWRP.L；Invest 若门控未开或无 R:R≥2:1 右侧个股，直接令 selected_actions 为空数组 []（如需在应税 Invest 配置长期底仓仅可提示派息型 VWRL.L，严禁混买累积型 VWRP.L 造成离岸 ERI 报税摩擦）。`

---

### Phase 4：T+N 结算 Alpha 符号纠偏与 TWR 利息真实入账

#### Step 4.1：修正 `SWAP` 方向、`TRIM/STOP` 基准 Alpha 符号与盘后推荐触价起算点
- **文件**：`src/bot/scheduler.py`
- **修改点 A（`SWAP` 归入多头方向，约第 190–193 行 `_trade_direction`）**：
  - 将 `{"TRIM", "STOP", "SELL", "SWAP"}` 改为 `{"TRIM", "STOP", "SELL"}`（`/pick` 中的 `SWAP` 为换入新标的做多，方向必为 `+1`）。
- **修改点 B（盘后生成的推荐不拿当日已收盘低点判成交，约第 203–232 行 `_entry_fill_status` 及第 1368 行调用处）**：
  - `_entry_fill_status` 增加可选关键字参数 `exclude_rec_date: bool = False`（默认 `False` 以保持既有函数级单测兼容）；当 `exclude_rec_date=True` 时使用 `window = ohlc[(dates > rec_date) & (dates <= target_date)]`。
  - 在 `_run_daily_settlement_impl` 第 1368 行调用处：读取 `rec_ms = int(rec.get("created_at_ms") or 0)`；若 `rec_ms > 0` 且推荐生成时刻 `datetime.fromtimestamp(rec_ms / 1000.0, tz=timezone.utc)` 当日该市场已收盘（`has_market_closed_on_date(rec_date, market=market, now_utc=rec_dt_utc)` 为 `True`，如周五 `21:30 UK` 自动 `/pick`），传 `exclude_rec_date=True`。
- **修改点 C（修正 `direction == -1` 时的基准超额收益公式，约第 1385–1393 行与第 1458–1464 行）**：
  - 现状：`alpha_pct = raw_return_pct - bench_ret_pct`，在 `direction == -1` 时变成 `-(stock_ret + bench_ret)`。
  - 改为：
    ```python
    unadjusted_stock_ret = ((exit_price - eff_entry) / eff_entry) * 100.0
    raw_return_pct = unadjusted_stock_ret * direction
    ...
    alpha_pct = (unadjusted_stock_ret - bench_ret_pct) * direction
    ```
    *（数学验证：多头 `direction=+1` 时 `alpha = stock_ret - bench_ret` 完全不变；减仓/止损 `direction=-1` 时 `alpha = bench_ret - stock_ret`，即被砍弱股跌 `-3%` 而同期基准涨 `+5%` 时，成功避损换基准超额为 `+8.0%`；大盘跌 `-10%` 而被砍抗跌股仅跌 `-1%` 时，乱砍损失超额为 `-9.0%`。）*
- **修改点 D（日报同向动作 5 日去重防刷屏污染，约 `src/agents/report_agent.py:1681-1690`）**：
  - 在写入 `source="report"` 推荐前，若近 5 天 `past_lessons` / 未结算记录中已存在同一 `(account, symbol, action)`，跳过重复入账，避免同一持仓连写 10 条高度自相关的 T+20 记录。

#### Step 4.2：让 4.0% 现金利息真实计入 TWR 净值收益，并对齐早报基准对比窗口
- **文件**：`src/data_providers/analytics.py`、`src/agents/report_agent.py`
- **修改点 A（`src/data_providers/analytics.py` 约第 1664–1710 行 `_EXTERNAL_FLOW_SIGNS` 与 `build_nav_row`）**：
  - 从 `_EXTERNAL_FLOW_SIGNS` 中**删除** `"INTEREST_ON_FREE_CASH": 1.0`、`"LENDING_INTEREST": 1.0` 与 `"FEE": -1.0`，仅保留真正的外部出入金：
    ```python
    _EXTERNAL_FLOW_SIGNS: dict[str, float] = {
        "DEPOSIT": 1.0,
        "WITHDRAW": -1.0,
        "WITHDRAWAL": -1.0,
    }
    ```
    （现金利息与分红同属账户内部投资收益，平台扣费属内部损耗，均不得作为外部充提抵消净值涨跌；同时在 `ccy == "USD"` 且 `getattr(portfolio, "gbp_usd_rate", None)` 有效 `> 0` 时，将 USD 外部出入金除以 `gbp_usd_rate` 折算计入 `net_flow`。）
  - 同步更新 `tests/test_nav_performance.py` 第 99–116 行 `test_build_nav_row_counts_only_external_gbp_flows`：将含 `_tx(5.0, "FEE")` 的 `net_flow_gbp` 预期值由 `35.0` 更新为 `40.0`，并新增 `_tx(12.0, "INTEREST_ON_FREE_CASH")` 断言其不计入 `net_flow_gbp`。
- **修改点 B（`src/agents/report_agent.py` 约第 234–257 行 `_format_benchmark_line`）**：
  - 删除第 254–255 行拿「较上次快照（1日）账户净值变动」直接减「基准近 7 天（5根K线）涨跌幅」的错位差值 `(相对基准 ...)`，仅客观并列展示「基准近7天 / 近30天」与「本账户较上次快照净值变动」，杜绝异周期相减误导 PM。

---

## 3. 新增离线单元测试清单与全量验收标准

每个阶段完成后，补充以下针对性离线单测（零外网请求）：

1. **`tests/test_report_pick_gate.py` 新增用例**：
   - `test_vix_30_does_not_override_core_etf_buy_in_guardrail`：`VIX=30` 时 `[ISA] BUY/ADD VWRP.L` 经 `apply_risk_override_guardrail` 后保持原动作不被 `MACRO_VIX` 拦截，而 `[ISA] BUY IUKD.L`（非核心高息 ETF）与个股被正常拦截；
   - `test_financed_unlock_requires_full_share_coverage`：持仓 10 股、保本止损单仅挂 6 股时，`is_financed` 为 `False` 且占用 1 个在险槽位；
   - `test_invest_wait_mode_none_blocks_pick_equity_buy`：`isa.etf_open=True` 且 `invest.status="WAIT", mode="none"` 时，`/pick` 中的 `[Invest] BUY` 100% 被拦截；
   - `test_bear_bounce_below_ma200_stays_risk_off_and_recovery_above_ma200_exits`：MA200 下方 `mom20>0` 反弹 3 天不退 `RISK_OFF`；`-25%` 见底后重新站上 MA200 且 `VIX<25` 满 3 天顺利退回 `NEUTRAL`。
2. **`tests/test_analytics_and_rules.py` 新增用例**：
   - `test_gbp_pence_llm_pound_entry_rescaled_no_100x_shares`：`SHEL.L`（`bundle.current_price=2500, currency="GBp"`），LLM 输入 `entry_price=25.0` 时，股数按 `£25/股` 而非 `£0.25/股` 计算；
   - `test_isa_risk_budget_scales_with_nav_and_isolated_from_invest_half_r`：`ISA` 净值 £57,117 时单笔风险预算自动按 `0.5%` 取 `£285.59`，且不受 Invest `0.5R` 连损缩放影响。
3. **`tests/test_calendar_and_scheduler.py` & `tests/test_nav_performance.py` 新增用例**：
   - `test_swap_is_long_and_stop_alpha_sign_matches_direction`：验证 `SWAP` 方向为 `+1`；验证 `STOP` 后标的跌 `-3%`、基准涨 `+5%` 时 `alpha_pct == +8.0%`；验证周五美股收盘后生成的推荐不取周五当日低点判 `FILLED`；
   - `test_cash_interest_counts_as_twr_return_not_external_deposit`：验证 `INTEREST_ON_FREE_CASH` 产生正 TWR 收益。

### 交付前必跑三条验收命令（全部退出码须为 0）

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m tests.run_tests tests
.\.venv\Scripts\python.exe scripts\audit_egress.py
```

最后将本次变更摘要追加至 `CHANGELOG.md` 顶部当前版本节点下，并在交付说明中显式列出 `config.yaml` 三项风控开关变更供人工最终确认。