# Skill 创建模板

新建技能时，在 `.agents/skills/` 下创建一个小写连字符命名的文件夹（例如 `quant-risk-check/`），将下方内容保存为该文件夹内的 `SKILL.md` 即可（本 `_TEMPLATE.md` 不会被 AGY / OpenCode 当作有效技能加载）。

```text
.agents/skills/
└── your-skill-name/
    ├── SKILL.md          # 必填：YAML 元数据 + 核心步骤
    ├── references/       # 可选：大篇幅文档、公式或接口说明（按需读取）
    └── scripts/          # 可选：可复用的辅助脚本
```

---

```markdown
---
name: your-skill-name
description: >-
  用一句话说明这个技能做什么，以及「什么时候应该触发它」（建议用英文或中英混合写清关键词，
  例如：Use when analyzing T212 portfolio risk, calculating CVaR/Sharpe, or reviewing strategy backtests.）。
---

# 技能标题

## 适用场景
- 说明在什么任务下使用本技能。

## 执行步骤
1. 第一步做什么（如需调用脚本，使用相对路径 `[script.py](./scripts/script.py)`）。
2. 第二步做什么（如需查阅长文档，链接到 `[ref.md](./references/ref.md)`）。

## 验收标准
- 说明如何验证本步骤执行成功。
```
