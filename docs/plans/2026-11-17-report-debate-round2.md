# `/report` Step 2 第二轮辩论 + 双中立 施工计划

**Goal:** 把 `/report` 目前「Bull/Bear 并行单轮、互不可见」升级为「首轮 Hydrogen → 第二轮 DeepSeek 对称交叉质询 → 双 DeepSeek 中立（摘要 + 证据核验）→ PM 只吃精华」的每日辩论流水线；同时把 Step 2 从单一 240s 总预算改为 **R1 360s / R2 180s / R3 120s** 的分段预算，并给 PM 单独 300s，避免 Carbon 因 120s 硬超时被过早切备援。

**Architecture:** 不改 `generate_morning_report` 的三步骨架，只在 Step 2 内部新增一个**阶段编排器**（phase orchestrator），替代现有单次 `asyncio.wait(timeout=total)`。R1 复用现有 Bull/Bear 并发；R2 双向质询并发；R3 摘要与核验并发；PM 保持单次结构化调用，prompt 只增补「精华索引」而非全文。全链路仍是 LLM 只读、Python 写库、最终 ≤1800 字。

---

## 一、已确认决策（Decision Log）

| # | 决策 | 备选 | 理由 |
| :--- | :--- | :--- | :--- |
| 1 | R1 Hydrogen ×2 → R2 DeepSeek ×2 → R3 DeepSeek ×2 | 全部 DS / 一边 H 一边 DS | 同一轮内双方同模型，模型能力是常量；跨轮换厂商，参考 `_ref/REF_TradingAgents` 的 quick/deep 分档 |
| 2 | 质询轮**对称并行**（Bull→Bear 与 Bear→Bull 同时跑） | 串行交替 | 避免「谁最后发言谁占优」的偏置；延迟只加一轮 |
| 3 | R3 跑**两个中立**：摘要 + 数字核验 | 只跑摘要 | 摘要负责信息压缩，核验负责抓 R2 的无据数字；职责不重叠 |
| 4 | 每天固定跑，不做条件触发 | 仅分歧/异动日 | 定时任务对运行时间不敏感；避免「平静日漏掉突发分歧」 |
| 5 | 预算 R1 360s / R2 180s / R3 120s；PM 300s | 单一 240s 总预算 | Carbon 单轮 120s 被切是已发生的生产问题 |
| 6 | DS 输出上限翻倍：R2 每人 ≤2400 字；R3 摘要 ≤900 字、核验 ≤400 字 | 维持 1200 字 | DS 单价低；长辩论、短投喂 |
| 7 | 质询全文不直接进 PM prompt，只进 `context_pack` | 全文注入 | 保护 PM 注意力与延迟；正文只给摘要 + 最多 2 个争议标的的原文引用 |

**非目标：** 不改确定性 `deployment_gate` / `apply_risk_override_guardrail`；不做风险辩论（激进/中性/保守）；不放开 Step 2 的 Google Search；不把 PM 换掉；不改 Telegram 1800 字上限。

---

## 二、数据流

```
R1（Hydrogen，并发，phase=360s）
  Bull R1 ─┐
  Bear R1 ─┤
           ├─→ 任一缺失 → 跳过 R2/R3，PM 按现状 2 份草稿裁决
           ↓
R2（DeepSeek，并发，phase=180s）
  Bull→Bear 质询（≤2400 字）
  Bear→Bull 质询（≤2400 字）
           ↓
R3（DeepSeek，并发，phase=120s）
  中立摘要：存活多方 / 存活空方 / 已被驳倒 / 未决冲突 + 2 个争议标的原文引用（≤900 字）
  证据核验：R2 中无法在 FactSheet 找到的价格/持仓/止损/挂单/百分比（≤400 字）
           ↓
PM（Carbon，300s）
  R1 全文 ×2 + R3 摘要 + R3 核验 + 分歧矩阵 + 门控底线
  （不注入 R2 全文；只通过摘要里的引用看关键原文）
```

`context_pack["debate"]` 落库完整 R1/R2/R3 文本与元数据（模型、耗时、降级标记），供 T+5 归因与周六审计复盘；PM prompt 不读该字段。

---

## 三、Prompt 契约

### R2 Bull 质询
- 输入：`shared_context`（FactSheet + grounding）+ Bull R1 + Bear R1。
- 角色：只做**逐条反驳**，不重复首轮已说过的内容；每条必须引用上下文中的数字或事实。
- 输出：≤2400 字；开头一行「本轮最关键的 3 个反驳点」，后续按 Bear 的论点逐条回应。
- 禁止：新增标的价格、编造新闻、调用工具（R2 不挂 toolkit）。

### R2 Bear 质询
- 同上，镜像角色；重点攻击 Bull 的「支撑位/目标价/加仓理由」。
- 输出：≤2400 字；必须给出「如果 Bull 成立，最坏情况是什么」。

### R3 中立摘要
- 输入：R1 ×2 + R2 ×2 + 分歧矩阵 + 门控底线。
- 输出四段（≤900 字）：
  1. 存活的多方论点（≤3 条）
  2. 存活的空方论点（≤3 条）
  3. 已被驳倒的论点及原因（≤3 条）
  4. 仍无证据、需 PM 亲自裁决的冲突（≤3 条）
- 另加：分歧最大的 2 个标的，各引用双方最关键的一句原文（每句 ≤200 字）。
- 硬规则：**不做投资裁决、不给买卖建议、不新增事实**；只做索引。

### R3 证据核验
- 输入：`fact_sheet`（只读基线）+ R2 ×2。
- 输出：逐条列出 `[无据] 标的 / 数字 / 出自哪一方`；≤400 字。
- 硬规则：只标记、不判违规；FactSheet 中找不到不等于错（可能是合理推导），由 PM 判断。

### PM 注入
新增一段：

```
=== 第二轮交叉质询与中立索引（DeepSeek，仅供索引，不是裁决） ===
[中立摘要]
...
[证据核验]
...
```

并在 PM system 中追加三条：
1. 中立索引只是索引，不是裁决；仍须正面裁决 Bull/Bear。
2. 索引与 R1 事实冲突时，以 R1 与本地 FactSheet 为准。
3. 被标记 `[无据]` 的数字不得作为加仓/减仓的主要依据。

---

## 四、预算与超时

| 阶段 | 相位预算 | 单次调用上限 | 说明 |
| :--- | ---: | ---: | :--- |
| R1 Bull / Bear | 360s | 沿用 120s/轮（工具 ≤2 轮） | 不抬高全局，避免拖慢 `/pick` 与单聊 |
| R2 双向质询 | 180s | 150s | DeepSeek 单轮 override；失败即跳过 |
| R3 摘要 / 核验 | 120s | 100s | 两个中立并发 |
| PM | — | 300s | 新增 `report_pm_timeout_seconds`，只对本调用生效 |
| 全链路最坏 | ~660 + 300 + 搜索 | — | 约 16–18 分钟；07:30 起跑仍早于 LSE 08:00 |

实现要点：
- `generate()` 新增 `round_timeout_seconds: float | None = None` 与 `max_output_tokens: int | None = None`；**override 只作用于候选链的第一个模型**，备援仍用客户端默认，避免一个慢模型拖垮整条链。
- `report_step2_timeout_seconds` 保留为总闸（默认 660s），新增 `report_step2_round1/2/3_seconds` 三个分段值；编排器按段取 `min(segment, remaining_total)`。
- 每段到点必须 `cancel()` + `gather(return_exceptions=True)` 回收，沿用现有纪律。

---

## 五、失败隔离

| 故障 | 行为 |
| :--- | :--- |
| R1 一侧超时/哨兵 | 跳过 R2 与 R3（对称性要求双方都有首轮）；PM 用现有中性占位继续 |
| R1 双侧超时 | 跳过 R2/R3；PM 用两份占位 + 门控底线 |
| R2 一侧失败 | 该侧质询省略；**不跑 R3**（摘要需要双方质询）；PM 只拿 R1 |
| R3 摘要失败 | 只注入核验（若有）；PM 仍拿 R1 |
| R3 核验失败 | 只注入摘要；PM 仍拿 R1 |
| R2/R3 失败 | 只记 `logger.warning` + `ops.notify(WARN, alert_key="report_debate_degraded")`；**不得**触发 `report_step2_degraded`，不得让 PM 拿到 `⚠️` 运维文案 |
| DS 未配置 | `report_debate_enabled` 自动失效，退回现状 2 调用 |

---

## 六、配置项（`ModelsSection` + `config.yaml`）

```yaml
report_debate_enabled: true
report_debate_model: "deepseek-flash"
report_debate_thinking_level: "HIGH"
report_debate_max_chars: 2400
report_neutral_model: "deepseek-flash"
report_neutral_thinking_level: "HIGH"
report_neutral_digest_max_chars: 900
report_neutral_factcheck_max_chars: 400
report_debate_max_output_tokens: 4096
report_step2_round1_seconds: 360.0
report_step2_round2_seconds: 180.0
report_step2_round3_seconds: 120.0
report_step2_timeout_seconds: 660.0
report_pm_timeout_seconds: 300.0
```

全部带 Pydantic 边界校验；`report_step2_timeout_seconds >= sum(rounds)` 时以总闸为准。

---

## 七、文件改动清单

| 文件 | 改动 |
| :--- | :--- |
| `src/agents/llm_client.py`（红线） | `generate()` 新增 `round_timeout_seconds` / `max_output_tokens`；首候选 override；DS payload `max_tokens`；Gemini `max_output_tokens` |
| `src/agents/report_agent.py` | 新增 `_run_debate_rounds()` 阶段编排器；R2/R3 prompt builders；PM prompt 注入；`context_pack["debate"]` |
| `src/config.py` / `config.yaml` / `src/main.py` | 新字段与接线 |
| `src/agents/schemas.py` | 不改 schema；仅 `render_morning_report` 无需变更 |
| `tests/test_report_debate_round2.py`（新） | 6 调用顺序、对称性、失败隔离、预算取消、PM 注入边界 |
| `tests/test_report_step2_budget.py` | 适配阶段编排器；保留原 4 条中性降级断言 |
| `tests/test_llm_pricing_and_usage.py` | per-call override 只作用于首候选；`max_tokens` 映射 |
| `tests/test_db_and_end_to_end.py` | `/report` 调用标签与模型断言；PM prompt 含摘要且不含 R2 全文 |
| `CHANGELOG.md` / `DEVELOPER_GUIDE.md` / `docs/MODULE_CONTRACTS.md` | 路由表、`generate()` 签名、Step 2 预算语义 |
| `src/version.py` / `pyproject.toml` | `5.4.2`（仅 version 行） |

---

## 八、单测矩阵（新增 ≥10 条）

1. 正常 6 调用：R1 2×Hydrogen、R2 2×deepseek、R3 2×deepseek，顺序与并发正确。
2. R2 两次 prompt 都包含对方 R1 全文，且不包含自己的 R1 全文。
3. R2 输出超 2400 字时不崩溃，PM 只收到摘要。
4. R1 单侧超时 → R2/R3 完全不调用，PM 收到中性占位。
5. R2 单侧失败 → R3 不调用，PM 只收 R1。
6. R3 摘要同侧失败 → 核验仍可注入。
7. R2 阶段超时 → 任务被 cancel 且 `gather` 回收，无 pending task 告警。
8. `report_debate_enabled=false` → 调用数回到 2，逐字保持今天的行为。
9. PM prompt 含「中立索引」段与 `[无据]` 规则，但不含 R2 全文。
10. `context_pack["debate"]` 落库 JSON 可反序列化，含 degraded 标记。
11. `round_timeout_seconds` 只作用于首候选：首候选超时后备援仍用默认 120s/60s。
12. `max_output_tokens` 同时映射到 Gemini `max_output_tokens` 与 DS `max_tokens`。

---

## 九、风险与回滚

| 风险 | 缓解 |
| :--- | :--- |
| DS 质询编造数字 | R3 核验 + PM 硬规则「无据数字不得作为主依据」+ 全文存 `context_pack` 供复盘 |
| 摘要漏掉关键反驳 | 摘要必须附 2 个争议标的的原文引用；PM 以 R1 为准 |
| DS 长输出失控 | `max_output_tokens=4096` + 150s/100s 单轮上限 |
| 运行时间变长 | 分段预算 + 总闸 660s；定时任务可接受；手动 `/report` 会变慢 |
| 红线文件改动 | `llm_client.py` 仅新增两个可选参数，不碰 egress/密钥/base_url；提交前按 §5.3 人工 review |
| 全链路失败 | `report_debate_enabled=false` 一行回滚到 2 调用 |

---

## 十、验收命令

```powershell
.\.venv\Scripts\python.exe -m compileall -q src tests
.\.venv\Scripts\python.exe -m unittest tests.test_report_debate_round2 -v
.\.venv\Scripts\python.exe -m unittest tests.test_report_step2_budget -v
.\.venv\Scripts\python.exe -m unittest tests.test_llm_pricing_and_usage -v
.\.venv\Scripts\python.exe -m tests.run_tests tests
.\.venv\Scripts\python.exe scripts\audit_egress.py
```

全绿后再 `git commit` + 根目录 `scripts\deploy-n95.ps1` 发版；重启后 `/health` 应显示 `v5.4.2`。
