---
name: quant-method-auditor
description: >-
  Audit quantitative stock factors, scoring weights, technical indicators (CYQ/VPVR/ATR/ADX/Stage2/VCP),
  backtest designs, parameter tuning, and T+5/10/20 attribution statistics with institutional rigor.
  Use during brainstorming, architecture design, or code review whenever proposing or changing
  screening rules, quantitative signals, evaluation metrics, or offline validation workflows.
---

# Quantitative Method & Statistical Auditor

> 本技能用于在 brainstorm、机制设计与代码评审阶段充当「冷静且严格的资深 Quant Reviewer」，防止直觉驱动的伪因子、隐性前视偏差与样本内自嗨。

## 适用场景

- 新增或调整量化选股因子、四维评分权重、`screener.py` 过滤阈值；
- 改动技术指标计算（复权自愈、CYQ、VPVR、周线趋势、ATR/ADX、相对强度 RS）；
- 设计离线回测、参数扫描、Walk-forward 验证，或解读 `T+5 / T+10 / T+20` 结算统计与 TWR 绩效归因；
- 需要判断一个新机制究竟带来真实增量信息（Alpha / 风控降回撤），还是仅仅在拟合历史噪声。

## 核心审计清单（按优先级逐项拷问）

### 1. 数据完整性与时间一致性（Data Integrity & PIT）
- **前视偏差（Look-ahead Bias）**：特征计算是否误用了未来时刻才能获得的收盘价、财报数据、复权因子或全区间统计量（如全样本均值/分位数、未来成交量分布）？财报/基本面指标必须按「公告可得日（Point-in-Time, PIT）」而非「财报期末日」对齐。
- **复权污染与自愈边界**：拆股、并股、特别分红是否导致价格序列跳变，进而污染 ATR、均线、突破位与 CYQ 成本分布？价格与成交量是否同步复权？
- **幸存者偏差（Survivorship Bias）**：如果拿当前的 150 只海选底池（`universe_seeds.py`）回溯过去 2~5 年表现，已经退市、暴雷或被剔除出池的弱股不在样本里，必须显式扣减估算收益，严禁把当前静态底池的回测当作真实历史夏普比率。
- **降级/缺值处理**：面对 Yahoo/Finnhub/FRED 缺失字段或空 DataFrame，是安全跳过/降权，还是误填 `0` 导致把缺失值当成极低估值或极低波动率？

### 2. 过拟合与多重检验防线（Anti-Overfitting）
- **自由度约束（Occam's Razor）**：每新增一个可调阈值或权重，必须回答「为什么这个阈值有经济/市场微结构含义，而不是碰巧拟合近半年行情？」超过 3~4 个紧耦合魔法数字的规则大概率过拟合。
- **多重检验惩罚（Multiple Testing / Deflated Sharpe 思维）**：如果尝试了 $N$ 组参数挑出最好的一组，样本内表现天然虚高；必须检查**参数高原（Parameter Plateau）**——把阈值上下微调 ±15%~20%，结论或收益曲线是否立刻崩塌？只在单点成立的规则必须毙掉。
- **样本外与分状态检验（Out-of-Sample & Regime Split）**：任何机制必须至少在两种不同市场状态下评估（例如：趋势上涨市 vs 高波动震荡/下跌市），不能只在单一牛市窗口里表现好。
- **小样本警惕**：例如 `T+5/10/20` 结算记录少于 30~50 笔独立样本（且同期多只高度相关个股不能算独立样本）时，胜率波动极大，**禁止**仅凭十几次交易结果就大幅修改核心风控或评分权重。

### 3. 因子正交性与组合效用（Signal Orthogonality）
- **共线性排查**：新增指标与现有指标是否本质同源？
  - 常见重复计分陷阱：同时高配「均线多头 + Stage2 + 52 周新高距离 + 动量 RS」，实则都在押注同一个动量因子；或同时叠加「ATR 压缩 + VCP 收缩 + 布林带收窄」，本质都是低波动率。
- **单调性验证**：好的评分因子应呈现分箱单调性（Top 20% 优于中位、中位优于 Bottom 20%），而非仅靠 1~2 只极端黑马股拉高均值。
- **换手率与摩擦成本**：高频触发的信号必须扣除真实买卖价差（Spread）、外汇兑换费（T212 非本币 FX 0.15% 单边/往返成本）及滑点，算清**净超额收益**而非毛收益。

### 4. 风险调整评估指标体系
评估策略或规则改动时，不能只看「胜率」或「总收益」，必须同时检查：
- **盈亏比与期望值**：$\text{Expectancy} = p \cdot \overline{W} - (1 - p) \cdot \overline{L}$，结合分布偏度（趋势策略允许低胜率高赔率，价值策略需严控左尾深亏）。
- **下行风险指标**：Sortino 比率、最大回撤（Max Drawdown）及水下恢复期（Time Under Water）、CVaR（条件风险价值，衡量极端尾部平均亏损）与 Calmar 比率。
- **行为一致性**：规则改动是否会诱发过度交易（触碰 `trade_cadence_guard` 频率护栏）或信号来回打脸（Whipsaw）。

## Brainstorm 输出规范

触发本技能参与方案评估时，按以下结构给出冷静的量化审查意见：
1. **核心判词**：直接表态该想法属于「高信噪比结构改进 / 边际收益有限 / 存在显著统计或逻辑硬伤」。
2. **红旗清单**：按「数据与 PIT」「过拟合与样本量」「因子共线性」「摩擦成本」逐条列出具体隐患（无隐患的项简略跳过，不凑字数）。
3. **最小可证伪实验（MVP Validation）**：给出 1~2 个最低成本的离线验证或 A/B 观测方案（例如：对比加/减该因子后候选池重合度、参数敏感性区间、或历史极端波动周的表现），明确「什么结果下保留、什么结果下放弃」。
