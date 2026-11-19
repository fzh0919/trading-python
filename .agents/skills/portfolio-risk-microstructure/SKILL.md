---
name: portfolio-risk-microstructure
description: >-
  Evaluate portfolio risk budgeting (portfolio heat, CVaR, correlation stress, look-through exposure),
  execution microstructure (spread, liquidity, gap/open-auction risk, UK/EU/US session alignment,
  GBP vs USD/EUR/GBp currency isolation), and hindsight-free trade postmortems.
  Use when brainstorming or modifying portfolio_rules.py, t212_client.py, intraday sentinel,
  position sizing, stop-loss guards, or T+5/10/20 review and memory mechanisms.
---

# Portfolio Risk, Execution Microstructure & Postmortem Advisor

> 本技能聚焦「从好想法到安全落地」的最后一公里：组合级风险预算、真实交易执行微结构约束、双币种安全边界，以及无后视偏差的复盘闭环。

## 适用场景

- 修改 `portfolio_rules.py` 的仓位计算、止损规则、加减仓护栏与交易频率限制；
- 优化 `t212_client.py` 的挂单量化、盘中回执与 `intraday sentinel` 盘中哨兵机制；
- 设计穿透暴露、相关性/尾部风险（CVaR）门控、TWR 归因；
- 改进侧车记忆（Sidecar Memory）、周报汇总与 `T+5/10/20` 复盘评审逻辑。

## 一、组合级风险预算（Portfolio-Level Risk Budgeting）

单笔交易看起来很安全，组合叠加起来可能极度危险。评审任何仓位与风控机制时，强制从以下 4 个维度审视：

1. **单笔风险 vs 组合热力（Portfolio Heat）**
   - 单笔风险额：$\text{Position Risk} = \text{Position Size} \times |\text{Entry} - \text{Stop}| / \text{Entry}$（须折算至账户基准币种 `GBP`）。
   - **组合热力** = 所有未锁盈仓位的止损风险之和占账户净值比例。新增一笔建仓前，不仅要检查单标的权重上限，还要检查组合总热力是否超标（防止 8 只高相关科技股同时打掉止损导致单周剧烈回撤）。
2. **相关性压力测试（Correlation -> 1）**
   - 平静期行业/个股间相关性较低，但市场压力期风险资产相关性会迅速趋近于 1。因此不能只看静态方差，必须保留**左尾度量（CVaR / Expected Shortfall）**与**穿透暴露上限**（按底层行业、主题及货币维度合并 ETF 与个股）。
3. **动态波动率缩放（Volatility Targeting / ATR Sizing）**
   - 相同名义仓位在高波动标的（日均波幅 4%）与低波动防御股（日均波幅 1%）上的风险贡献相差 4 倍。优先以 ATR / 历史波动率反推合理仓位与止损宽度，避免止损设在正常日内噪声内部导致频繁“扫损后起飞”。
4. **资金桶与现金缓冲隔离**
   - ISA 与 Invest 的现金、净值与风险预算必须独立核算，且需考虑待成交挂单（Pending Orders）已占用的名义资金与风险敞口，防止重复授信。

## 二、执行微结构审查（Setup Quality ≠ Execution Quality）

很多纸面策略失效并非形态看错，而是败在执行摩擦。涉及入场信号、挂单量化与盘中哨兵时，必须把「形态质量」和「执行风险」拆开评估：

- **流动性与价差摩擦（Liquidity & Spread）**：检查日均成交额（ADV）与买卖价差（Bid-Ask Spread）。中小盘股或盘前/盘后时段价差急剧放大，市价单或紧贴现价的止损单极易遭受严重滑点。
- **跳空与开盘集合竞价风险（Gap & Open-Auction Risk）**：
  - 财报次日或宏观数据公布后的大幅跳空会直接击穿预设止损价，真实亏损可能远超理论 1R；风控模型必须区分「连续盘中可止损风险」与「隔夜跳空不可控风险」。
  - 开盘前 15~30 分钟波动率与价差最高，盘中哨兵（Intraday Sentinel）触发告警或动作建议时需评估是否应加入开盘静默窗或量价确认防抖。
- **跨市场时段与节假日错位**：本项目覆盖 `US / UK / XETRA / EURONEXT`，必须校验半日市、夏令时切换窗（美欧夏令时错位 1~3 周）以及欧股收盘后美股仍交易导致的净值估值时滞。
- **双币种与英股 `GBp/GBP` 铁律**：
  - 账户现金、净值、盈亏与风险预算固定以 `GBP` 计量；标的现价、均价、止损位保持本地交易币种（`USD / EUR / GBp / GBP`）。
  - 凡涉及英股价格计算、挂单金额换算、或 LLM 提示词组装，**必须在确定性 Python 代码层完成 `GBp ÷ 100 → GBP` 标准化与 FX 换算，严禁交给 LLM 口算**。

## 三、反后视偏差复盘框架（Hindsight-Free Postmortem）

设计或优化侧车记忆、周报与 `T+5/10/20` 结算复盘时，遵循「过程与结果解耦」的 2×2 矩阵，避免被单次盈亏带偏规则：

| | **结果盈利** | **结果亏损** |
| :--- | :--- | :--- |
| **遵守事前纪律（好过程）** | **可复制的正向样本**：归档其市场状态与入场结构特征 | **正常的概率成本**：接受合理止损，**严禁**因单次正常止损就放宽风控或删掉止损 |
| **违背事前纪律（坏过程）** | **最危险的运气陷阱**：必须高亮警示（例如逆势扛单后侥幸回本），绝不能强化错误习惯 | **高优先级修复项**：拆解是信号缺陷、执行滑点，还是情绪/频率违规 |

复盘归因必须优先记录结构化客观指标：**入场计划 vs 实际成交滑点、R 倍数结果（实际盈亏 / 初始风险 1R）、最大有利波动（MFE）与最大不利波动（MAE）**，用 MAE/MFE 判断是「选股方向错」还是「止损过紧/止盈过早」。

## Brainstorm 输出规范

触发本技能参与设计或评审时，必须显式覆盖：
1. **最坏情景演练（Worst-Case Stress）**：如果遭遇隔夜 -10% 跳空或全市场相关性飙升，该新机制会放大还是缓冲损失？
2. **执行摩擦清单**：列出该机制引入的价差、滑点、FX 换算、跨时区或 T212 挂单占用风险。
3. **复盘可观测性**：新规则上线后，如何通过客观字段（R 倍数、MAE/MFE、执行偏差）在 1~3 个月内客观评估其好坏。
