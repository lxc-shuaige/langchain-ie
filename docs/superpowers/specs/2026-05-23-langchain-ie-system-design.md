
# 基于 LangChain + LangGraph 的中英双语招聘信息抽取实验系统

## 设计规格说明

**日期**: 2026-05-23
**目标**: 将原 LangChain4j (Java) 方案转为 Python LangChain + LangGraph 实现

---

## 1. 概述

构建一个基于 LangGraph 工作流的信息抽取实验系统，使用"多工具协同 + 大模型路由"架构，从中英双语招聘文本中抽取 7 个结构化信息点。

### 1.1 模型选择

- **LLM**: DeepSeek API (`deepseek-chat`)
- **NER**: HuggingFace `bert-base-chinese-ner` (免费本地模型)
- **UIE**: PaddleNLP `uie-base` (免费本地模型，百度通用信息抽取)
- **分词**: jieba
- **正则**: Python `re`

### 1.2 运行形式

Streamlit Web 界面 + 命令行入口，直接在浏览器中浏览语料、查看抽取结果、对比不同方法效果。

### 1.3 语料

程序自动生成 120-200 篇中英双语招聘信息模拟语料。

### 1.4 范围

核心模块 + 技能词典增强 + 多工具协同（NER + UIE + Regex + Dictionary + LLM）。

---

## 2. 整体架构

系统采用 LangGraph 有向图建模抽取流程。核心思想：LLM 不仅是抽取器，还是"调度员"——在每个决策点分析当前文档特征，动态选择最适合的抽取工具。

```
输入文档
   ↓
[preprocess] → 清洗、分句、分词
   ↓
[analyze] → DeepSeek 分析文档特征，输出路由决策 JSON
   ↓
┌─ 条件路由分发 ─────────────────────────┐
│  → [regex]       薪资/学历/经验/地点    │
│  → [ner]         公司名/地点/岗位       │
│  → [dictionary]  技能词                 │
│  → [llm_extract] 复杂语义/隐式信息      │
│  → [uie]         通用实体+关系联合抽取   │
└────────────────────────────────────────┘
   ↓
[fusion] → 字段级融合（规则优先/LLM优先/冲突标记）
   ↓
→ 结果写入 JSON 文件
```

整个图在 LangGraph 的 `StateGraph` 中构建，State 携带文档内容和所有中间抽取结果。后续可方便地添加新节点、改路由逻辑、并联更多工具。

---

## 3. 数据结构

### 3.1 信息点定义 (7 个字段)

```python
FIELDS = [
    "job_title",       # 岗位名称
    "company_name",    # 公司名称
    "work_location",   # 工作地点
    "salary",          # 薪资范围
    "education",       # 学历要求
    "experience",      # 工作经验
    "skills",          # 技能要求 (list[str])
]
```

### 3.2 LangGraph State

```python
class ExtractionState(TypedDict):
    # 输入
    doc_id: str
    language: str          # "zh" | "en"
    title: str
    raw_text: str
    # 预处理后
    cleaned_text: str
    sentences: list[str]
    tokens: list[str]      # jieba 分词结果
    # LLM 分析决策
    routing_decision: dict
    # 各抽取器结果 (增量更新)
    regex_result: dict
    ner_result: dict
    dictionary_result: dict
    llm_result: dict
    uie_result: dict
    # 融合后最终结果
    final_result: dict
    # 评估
    evaluation: dict
```

LangGraph 每个节点返回 dict 增量更新 State，不同节点只写自己的字段，最终汇聚到 `final_result`。

### 3.3 文件输出格式

```json
// merged_results.json — 数组，每个元素:
{
  "doc_id": "zh_001",
  "language": "zh",
  "job_title": "Java开发工程师",
  "company_name": "某科技公司",
  "work_location": "北京",
  "salary": "20k-30k/月",
  "education": "本科及以上",
  "experience": "3年以上",
  "skills": ["Java", "Spring Boot", "MySQL"],
  "extractor_breakdown": {
    "job_title": "uie+llm",
    "company_name": "ner+llm",
    "work_location": "ner+regex",
    "salary": "regex",
    "education": "regex",
    "experience": "regex",
    "skills": "dictionary+llm"
  },
  "conflicts": []   // 有冲突的字段
}
```

---

## 4. 项目结构

```
langchain-ie-system/
├── data/
│   ├── corpus/                     # 120-200 篇招聘文本 (.txt)
│   ├── labels/
│   │   └── golden_labels.json      # 人工标注评估集 (20-30 篇)
│   └── output/
│       ├── regex_results.json
│       ├── llm_results.json
│       ├── ner_results.json
│       ├── uie_results.json
│       ├── merged_results.json
│       └── evaluation_report.json
├── src/
│   ├── __init__.py
│   ├── main.py                     # 入口：构建 graph 并执行
│   ├── state.py                    # ExtractionState 定义
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── loader.py              # 扫描目录、读取文本
│   │   └── generator.py           # 模拟语料生成
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── cleaner.py             # 清洗空格/HTML/特殊符号
│   │   └── segmenter.py           # 分句、jieba 分词
│   ├── extractors/
│   │   ├── __init__.py
│   │   ├── base.py                # BaseExtractor 抽象类
│   │   ├── regex_extractor.py     # 正则抽取 (薪资/学历/经验/地点)
│   │   ├── llm_extractor.py       # DeepSeek LLM 抽取
│   │   ├── ner_extractor.py       # HuggingFace NER 抽取
│   │   ├── uie_extractor.py       # PaddleNLP UIE 抽取
│   │   └── dict_extractor.py      # 技能词典匹配
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── builder.py             # StateGraph 构建
│   │   ├── nodes.py               # 所有节点函数
│   │   └── router.py              # 条件路由决策
│   ├── fusion/
│   │   ├── __init__.py
│   │   └── merger.py              # 字段级融合策略
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           # Precision/Recall/F1
│   └── web/
│       ├── __init__.py
│       └── app.py                 # Streamlit 界面
├── config.yaml                     # API Key、模型名、路径等
├── requirements.txt
└── README.md
```

---

## 5. LangGraph 图结构

### 5.1 节点列表 (8 个)

| 节点 | 函数 | 职责 |
|---|---|---|
| preprocess | `preprocess_node` | 清洗文本、分句、jieba 分词 |
| analyze | `analyze_node` | DeepSeek 分析文档特征，输出路由决策 |
| regex | `regex_node` | 正则抽取薪资/学历/经验/地点 |
| ner | `ner_node` | HuggingFace NER 抽取公司/地点/岗位 |
| dictionary | `dictionary_node` | 技能词典匹配 |
| llm_extract | `llm_extract_node` | DeepSeek 全字段语义抽取 |
| uie | `uie_node` | PaddleNLP UIE 通用实体关系抽取 |
| fusion | `fusion_node` | 字段级融合 + 冲突检测 |

### 5.2 图结构

```
START → preprocess → analyze → [条件路由]
                                   ↓
         ┌─────────────────────────┼─────────────────────────┐
         ↓              ↓          ↓         ↓               ↓
       regex          ner      dictionary  llm_extract      uie
         ↓              ↓          ↓         ↓               ↓
         └─────────────────────────┼─────────────────────────┘
                                   ↓
                               fusion → END
```

### 5.3 路由决策

`analyze` 节点让 DeepSeek 分析文本后输出路由 JSON:

```json
{
  "salary":        "regex",
  "education":     "regex",
  "experience":    "regex",
  "work_location": "ner+regex",
  "company_name":  "ner+llm",
  "job_title":     "uie+llm",
  "skills":        "dictionary+llm"
}
```

条件路由根据此决策将 State 推送到相应节点。被选中的节点并行执行。

---

## 6. 融合策略

`fusion` 节点实现三种融合策略，可通过配置切换:

### 6.1 规则优先 (默认)
- 正则格式字段 (薪资/学历/经验) → 直接采用 regex/ner/dictionary 结果
- 语义复杂字段 (岗位/公司/技能) → 优先采用 llm/uie 结果
- 缺失字段由 LLM 补全

### 6.2 LLM 优先
- LLM 完成初步抽取
- 规则模块对薪资/学历/经验做校验修正

### 6.3 字段级融合
- 每个字段取对应工具的结果
- 按 routing_decision 中指定的优先级合并

冲突时在结果中标记 `conflicts` 字段。

---

## 7. 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| 图框架 | `langgraph` | 构建 StateGraph 工作流 |
| LLM 调用 | `langchain-core` + `langchain-deepseek` | 封装 DeepSeek API |
| 正则 | Python `re` | 薪资/学历/经验/地点抽取 |
| NER | `transformers` + `bert-base-chinese-ner` | 公司名/地点/岗位名 |
| UIE | `paddlenlp` + `uie-base` | 通用信息抽取 |
| 分词 | `jieba` | 中文分词、技能词匹配 |
| 前端 | `streamlit` | Web 展示界面 |
| 配置 | `pyyaml` | config.yaml 管理 |

---

## 8. 评估方案

1. 从语料中随机选 25 篇，人工标注 golden_labels
2. 分别计算 regex/llm/ner/uie/fusion 的字段级 Precision、Recall、F1
3. 分中英文统计
4. 对比不同融合策略效果
5. 在 Streamlit 中可视化展示

---

## 9. 实验组设计

| 实验组 | 方案 | 
|---|---|
| 1 | 仅规则 (regex + dictionary) |
| 2 | 仅 LLM (DeepSeek 全字段) |
| 3 | NER + UIE |
| 4 | 规则优先融合 |
| 5 | LLM 优先融合 |
| 6 | 字段级融合 |

---

## 10. 实现顺序

1. 项目结构 + config.yaml + requirements.txt
2. 模拟语料生成器 → 120-200 篇 .txt
3. 预处理模块 (cleaner + segmenter)
4. 正则抽取器
5. 词典抽取器 (技能词匹配)
6. DeepSeek LLM 抽取器
7. NER 抽取器 (HuggingFace)
8. UIE 抽取器 (PaddleNLP)
9. LangGraph 图构建 + 路由
10. 结果融合模块
11. golden_labels 生成 + 评估模块
12. Streamlit Web 界面
