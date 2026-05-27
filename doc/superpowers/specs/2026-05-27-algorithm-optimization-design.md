# 算法优化设计文档

## 概述

对 LangGraph 招聘信息抽取系统的四个模块进行算法优化，提升抽取准确率、路由智能性、评估合理性和运行效率。

---

## 优化一：真路由 — 从硬编码到智能文档分析

### 问题

`analyze_node`（`src/graph/nodes.py:76-98`）存在职责矛盾：
- 调用 LLM 做了完整的 8 字段抽取（当抽取器用）
- 同时返回一个写死的 `routing_decision` 常量（当路由器用）

routing 完全与文档内容无关，LLM 调用被浪费在"生成路由 JSON 附带的副产物"上。

### 方案

将 `analyze_node` 拆分为轻量路由分析和独立的全字段 LLM 抽取：

1. **analyze_node 改为轻量路由分析**：LLM 只分析文档特征（语言混合度、句式规整度、字段是否显式），返回 routing JSON。不再返回 llm_result。

2. **路由 prompt 逻辑**：
   - 格式化字段（薪资、学历等）→ 优先 regex
   - 命名实体（公司、地点、岗位）→ 优先 ner+uie
   - 关键词字段（技能）→ 优先 dictionary
   - 语义模糊或中英混排 → 兜底 llm

3. **llm_extract_node 保留**：仅当 routing 中某字段标记了 llm 时才执行全字段抽取，避免重复调用。

### 影响

- LLM token 消耗大幅降低（从全字段抽取降级为简短分析）
- 路由决策由文档特征驱动，不再"一刀切"
- 不同文档激活不同工具组合，效率提升

### 修改文件

- `src/graph/nodes.py` — analyze_node 逻辑重写
- `src/extractors/llm_extractor.py` — 新增轻量路由 prompt

---

## 优化二：技能评估 — 从完全匹配到 Jaccard 相似度

### 问题

`src/evaluation/evaluator.py:41-46` 对技能列表使用 set 完全相等：

```python
match = set(str(x).lower() for x in pred_val) == set(str(x).lower() for x in gold_val)
```

gold = `["Python","Docker","K8s"]`, pred = `["Python","Docker"]` → FP+FN 双计，字段零分。但 2/3 的召回不应被记为完全失败。

### 方案

对列表型字段使用 Jaccard 相似度：

```
Jaccard = |pred ∩ gold| / |pred ∪ gold|
```

判定规则：
- Jaccard ≥ 0.5 → TP（partial match）
- Jaccard < 0.5 → FP + FN
- 两端皆空 → 不计（与现状一致）

非列表字段保持原有完全匹配逻辑。

额外输出 `partial_match_details`，记录每个文档 skills 的 Jaccard 值，供人工检查。

### 影响

- 技能字段评估更合理，F1 分数更真实反映实际抽取质量
- 评估结果中增加 Jaccard 明细

### 修改文件

- `src/evaluation/evaluator.py` — 列表字段匹配逻辑重写

---

## 优化三：加权投票融合策略

### 问题

三种现有融合策略都是 `_pick` 模式——按固定顺序取第一个非空值。多工具对同一字段有不同输出时，没有利用"哪个工具对该字段更可靠"的信息。

### 方案

新增第四种融合策略 `weighted_vote`，在 `config.yaml` 中通过 `confidence_weights` 配置各工具对各类字段的置信度：

| 字段类型 | Regex | Dictionary | NER | UIE | LLM |
|---------|:-----:|:----------:|:---:|:---:|:---:|
| 格式化（薪资/学历/经验/联系方式） | 0.9 | — | 0.3 | 0.5 | 0.6 |
| 实体（公司/地点/岗位） | 0.3 | — | 0.7 | 0.6 | 0.7 |
| 列表（技能） | — | 0.8 | — | 0.5 | 0.6 |

**Scalar 字段逻辑**（薪资、学历等）：
- 各工具按权重排序，取最高可信的非空值
- 输出该字段的 confidence 分数

**Skills 字段逻辑**（列表型）：
- 所有工具的技能列表做并集
- 按出现频次（跨工具一致性）标记置信度：3 源一致 > 2 源一致 > 单源

每个字段附带 `confidence` 分数，在评估报告和 Web 界面展示。

### 修改文件

- `src/fusion/merger.py` — 新增 `_weighted_vote()` 函数
- `config.yaml` — 新增 `confidence_weights` 配置块

---

## 优化四：文档级并行处理

### 问题

`src/main.py` 中 150 篇文档逐个 `graph.invoke()` 串行处理，LLM API 调用的 IO 等待时间完全叠加。

### 方案

使用 `concurrent.futures.ThreadPoolExecutor` 对文档循环做并行化，线程池大小默认 `min(8, os.cpu_count() * 2)`，可通过 `config.yaml` 的 `processing.max_workers` 覆盖。每篇文档各自走独立的 graph pipeline，LLM API 调用是 IO 密集型，线程池即可胜任。

### 修改文件

- `src/main.py` — `_run_text_pipeline` 和 `_run_image_pipeline` 改用线程池

---

## 预期效果对比

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| LLM 调用次数 / 篇 | 1-2 次（analyze + 可能的 llm_extract） | 1 次轻量路由 + 按需 llm_extract |
| 路由模式 | 硬编码，所有文档相同 | 文档级动态路由 |
| 技能评估 | 完全匹配（0 或 1） | Jaccard 相似度 |
| 融合策略 | 3 种（rule_first/llm_first/field_level） | 4 种（+weighted_vote） |
| 处理方式 | 串行 | 文档级并行 |

---

## 不做的优化

- **NER 英文适配**：当前 NER 模型仅支持中文，英文文档直接返回空。不扩展（需换模型，工作量太大且偏离核心）
- **缓存机制**：不引入跨文档缓存（150 篇规模太小，缓存命中率低，引入复杂度不值得）
