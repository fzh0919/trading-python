---
name: equity-research-and-regime
description: >-
  Apply institutional equity research, macro/market-breadth regime modeling, dual-philosophy
  screening (value/dividend-growth quality for ISA vs Minervini Stage2/VCP momentum for Invest),
  valuation triangulation (DCF/Comps/SOTP), and buy-side thesis/catalyst tracking.
  Use when designing or refining screener.py, deployment_gate.py, dual-philosophy rules,
  or /report, /pick, and /deep research & debate prompts.
---

# Equity Research, Dual-Philosophy & Market Regime Advisor

> 本技能将买方股票研究（Buy-Side Equity Research）、市场宽度与 Regime 研判、以及双哲学选股标准转化为可落地的机制设计指南，服务于 `screener.py`、`deployment_gate.py` 与 `/report` / `/pick` / `/deep` 的演进。

## 适用场景

- 讨论宏观状态切换、危机通道、市场宽度指标及 `deployment_gate.py` 的四桶仓位门控政策；
- 优化双哲学分轨规则（ISA 长期复利/价值安全边际 vs Invest 动态 R:R 趋势波段）；
- 升级 `/report` 三步辩论流水线、`/pick` 初筛与反方 Critic 逻辑、`/deep` 深度研报结构或投资论点追踪。

## 一、市场状态（Regime）与宽度分层框架

仅靠大盘指数点位或单一 VIX 容易被少数权重股失真误导。设计门控机制时，优先从「指数趋势 + 市场宽度 + 信用/波动率压力」三维交叉验证：

| Regime 状态 | 典型定量特征 | 组合政策含义（对齐 CIO 门控） |
| :--- | :--- | :--- |
| **Risk-On（趋势扩张）** | 主指数在 50/200DMA 上方；市场宽度健康（成分股站上 50/200DMA 比例扩张、新高 > 新低）；波动率与信用利差受控 | 允许 Invest 正常展开突破/回踩建仓；ISA 按纪律定投或逢质量回调加仓 |
| **Chop / Narrow（震荡分化）** | 指数横盘但市场宽度背离（少数龙头撑指数、多数个股走弱）；突破失败率上升 | 收紧 Invest 突破入场门槛（要求更高相对强度 RS 与更紧凑 VCP），压缩新开仓 R:R 容错，防范反复止损 |
| **Stress（压力去风险）** | 跌破关键中期均线、宽度显著恶化、波动率结构抬升或长端利率/信用利差急剧异动 | 冻结 Invest 新增进攻仓位，优先保护浮盈与裁剪弱势股；仅保留高安全边际的 ISA 核心资产观察 |
| **Crisis（危机通道）** | 流动性冲击、相关性趋近于 1、极端尾部回撤触发 | 严格执行危机门控，禁止在下跌飞刀期用趋势模型抄底；仅允许按既定分级纪律在极值安全边际处分批配置长期核心仓 |

> **关键原则**：门控状态切换必须具备**滞回缓冲（Hysteresis，如进入防御快、解除防御需多日确认）**，避免在阈值边缘来回横跳。

## 二、双哲学选股与体检对照标准

严禁把成长波段指标与深度价值指标揉成一锅粥导致两头不靠。两类账户的机制改动必须保持逻辑隔离：

### 1. Invest 轨：趋势动量与不对称赔率（Minervini Stage 2 / VCP 体系）
- **趋势前提（Stage 2）**：价格位于上升的 150DMA / 200DMA 上方，且 50DMA > 150DMA > 200DMA，相对基准（SPY/VWRL 等）的相对强度（RS）处于上升或高位。
- **结构收缩（VCP — Volatility Contraction Pattern）**：回调幅度逐级收敛（例如 20% → 10% → 4%），伴随右侧缩量（筹码锁定、抛压枯竭），VPVR/CYQ 显示上方套牢盘轻或已在枢轴位完成换手。
- **入场与失效边界**：突破枢轴点需量能确认；**入场前必须定义明确的技术止损位（如枢轴下方或 1.5~2x ATR）与最小动态 R:R**，严禁“套牢后自动转长线价值投资”。

### 2. ISA 轨：质量复利、价值安全边际与红利成长
- **盈利质量优先于低估值陷阱（Value Trap）**：优先考察 ROIC / ROE 稳定性、自由现金流转化率（FCF / Net Income）、资产负债表韧性（净负债/EBITDA、利息覆盖倍数）与毛利率趋势，避免只买“PE 很低但基本面结构性衰退”的标的。
- **估值三角交叉（Valuation Triangulation）**：不迷信单一静态 PE；按业务类型匹配估值锚——成熟现金牛看 FCF Yield / EV/EBITDA / 红利成长回调（Dividend Growth Pullback），高成长看 PEG / 远期 EV/Sales 与利润率路径，综合集团看 SOTP（分部加总估值），并结合自身历史分位与同业可比（Comps）。
- **ETF 与穿透重叠**：对持仓或候选 ETF，必须评估底层前十大权重及行业穿透重叠度，避免“买三只不同名字的 ETF，实则 60% 都在重复持有同一批科技权重股”。

## 三、机构买方研究论点（Thesis）与辩论框架

用于优化 `/report`、`/pick`、`/deep` 提示词及 Critic 审查逻辑时，要求每个投资建议至少显式落盘 5 个要素，杜绝空泛叙事：

1. **核心论点（Core Thesis）**：一句话说清“市场现在定价了什么，而为什么现实可能不同（预期差 Variant Perception）”。
2. **关键驱动与领先指标（Key Drivers）**：后续 1~3 个季度跟踪哪 2~3 个可量化经营/价格指标来验证论点？
3. **近期催化剂与时间窗口（Catalysts & Event Risk）**：财报日、央行决议、产品发布或监管节点（提示高波动事件前是否应降杠杆或等待落地）。
4. **证伪条件（Kill Criteria / Pre-Mortem）**：**事先写死**什么基本面或技术面事实出现时，证明论点彻底错了必须认输离场（防止确认偏差）。
5. **不对称风险收益（Bull / Base / Bear）**：把上涨空间与下跌风险锚定在具体估值倍数或关键结构位上，而非拍脑袋给目标价。

## Brainstorm 输出规范

触发本技能时，输出必须明确回答：
1. **哲学归属检查**：该改动属于 ISA（价值/质量）、Invest（趋势/赔率），还是顶层 CIO 门控？是否污染了另一条轨道的纪律？
2. **市场状态适配性**：该机制在牛市、震荡市、熊市/危机下的行为分别是什么？是否有滞回防抖设计？
3. **信源与可计算性约束**：所需指标能否用现有免费/低成本确定性数据源（Yahoo/Finnhub/FRED/SEC/RNS）稳定计算，避免把需要机构昂贵订阅数据的指标写进核心依赖。
