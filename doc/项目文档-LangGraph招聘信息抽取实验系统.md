# 基于 LangChain + LangGraph 的中英双语招聘信息抽取实验系统

## 1. 项目概述

### 1.1 项目名称

基于 LangChain + LangGraph 的中英双语招聘信息抽取实验系统（含多媒体扩展）

### 1.2 项目背景

信息抽取（Information Extraction）是自然语言处理中的核心任务，旨在从非结构化文本中提取结构化信息。传统方法依赖规则、词典和统计学习模型，具有高可解释性但泛化能力有限；大语言模型（LLM）具备强大的语义理解能力，能处理多样句式和中英混合文本，但存在幻觉和输出不稳定问题。

本系统将两者结合，使用 LangGraph 构建多工具协同工作流，以"LLM 作为调度员 + 多种抽取工具并行"的架构，实现中英双语招聘信息的结构化抽取，并进一步扩展到招聘海报图片的 OCR 识别与抽取。

### 1.3 项目目标

1. 建立不少于 100 篇的中英双语招聘语料库
2. 定义 8 个结构化信息点并完成抽取
3. 实现基于正则表达式的规则抽取
4. 接入 DeepSeek 大模型实现语义抽取
5. 集成 NER（命名实体识别）和 UIE（通用信息抽取）辅助模型
6. 实现规则优先/LLM 优先/字段级/加权投票四种融合策略
7. 扩展 OCR 模块支持招聘海报图片信息抽取
8. 构建 Streamlit Web 界面进行结果展示与对比
9. 通过 golden_labels 进行 Precision/Recall/F1 评估（含 Jaccard 相似度）
10. 实现智能路由（LLM 动态分析文档特征选择工具）、文档级并行处理等算法优化

---

## 2. 技术栈

| 层 | 技术 | 用途 |
|---|---|---|
| 图框架 | `langgraph` | 构建 StateGraph 有向图工作流 |
| LLM 调用 | `langchain-core` + `langchain-deepseek` | 封装 DeepSeek API 调用 |
| 正则 | Python `re` | 薪资/学历/经验/地点/联系方式抽取 |
| NER | `transformers` + RoBERTa | 公司名/地点/岗位名识别 |
| UIE | `paddlenlp` + `uie-base` | 通用信息抽取（百度） |
| 分词 | `jieba` | 中文分词、技能词典匹配 |
| OCR | `easyocr` / `paddleocr` | 招聘海报图片文字识别 |
| 图片生成 | `Pillow` | 生成模拟招聘海报 PNG |
| 前端 | `streamlit` | Web 展示界面 |
| 配置 | `pyyaml` | config.yaml 管理 |

**开发语言**: Python 3.10  
**大模型**: DeepSeek Chat API (`deepseek-chat`)

---

## 3. 系统架构

### 3.1 总体架构

系统采用 LangGraph 有向图建模抽取流程。LLM 不仅是抽取器，还是"调度员"——在 analyze 节点分析文档特征，动态选择最适合的抽取工具。图片输入则在预处理阶段分流到 OCR 节点。

```
                      ┌─── 文本输入 ──────────────────────┐
                      │                                   ▼
START → preprocess ───┤                              analyze (LLM 路由决策)
                      │                                   ▲
                      └─── 图片输入 → ocr (EasyOCR) ──────┘
                                                          │
              ┌───────────────────────────────────────────┼───────────────────────────┐
              ▼              ▼            ▼               ▼                           ▼
           regex           ner       dictionary       llm_extract                    uie
              │              │            │               │                           │
              └───────────────────────────────────────────┼───────────────────────────┘
                                                          ▼
                                                      fusion (字段级融合)
                                                          │
                                                         END
```

### 3.2 LangGraph 节点说明

| 节点 | 职责 | 输入 | 输出 |
|------|------|------|------|
| `preprocess` | 文本清洗、分句、jieba 分词；图片输入则跳过 | raw_text | cleaned_text, sentences, tokens |
| `ocr` | 对图片运行 EasyOCR，输出文本后再清洗/分句/分词 | image_path | ocr_text, cleaned_text, tokens |
| `analyze` | DeepSeek 轻量路由分析，仅分析文档特征输出路由决策（不抽取字段） | cleaned_text | routing_decision |
| `regex` | 正则匹配薪资/学历/经验/地点/联系方式 | cleaned_text | regex_result |
| `ner` | HuggingFace NER 识别公司名/地点/岗位 | cleaned_text | ner_result |
| `dictionary` | 技能词典匹配 | cleaned_text | dictionary_result |
| `llm_extract` | DeepSeek 全字段语义抽取（仅路由激活时执行） | cleaned_text | llm_result |
| `uie` | PaddleNLP UIE 通用实体关系抽取 | cleaned_text | uie_result |
| `fusion` | 汇总所有抽取器结果，按策略合并 + 冲突检测 | 5 个 result | final_result |

### 3.3 项目目录结构

```
langchain-ie-system/
├── config.yaml                         # API Key、模型名、路径、融合策略
├── requirements.txt                    # Python 依赖
├── data/
│   ├── corpus/                         # 150 篇中英双语招聘文本 (.txt)
│   ├── images/                         # 30 张模拟招聘海报图片 (.png)
│   ├── labels/
│   │   ├── golden_labels.json          # 文本评估集人工标注（25 篇）
│   │   └── golden_labels_image.json    # 图片评估集人工标注（10 张）
│   └── output/
│       ├── merged_results.json         # 最终融合结果
│       ├── image_results.json          # 图片抽取结果
│       └── evaluation_report.json      # 评估报告
├── src/
│   ├── main.py                         # CLI 入口（text / image / all 三种模式）
│   ├── state.py                        # FIELDS 定义 + ExtractionState TypedDict
│   ├── corpus/
│   │   ├── loader.py                   # 文本/图片语料加载 + golden_labels 加载
│   │   ├── generator.py                # 模拟招聘文本语料生成
│   │   └── image_generator.py          # Pillow 生成模拟招聘海报
│   ├── preprocess/
│   │   ├── cleaner.py                  # 文本清洗（空格/HTML/特殊符号）
│   │   └── segmenter.py               # 分句、jieba 分词
│   ├── extractors/
│   │   ├── base.py                     # BaseExtractor 抽象基类
│   │   ├── regex_extractor.py          # 正则抽取（8 字段中英模式）
│   │   ├── llm_extractor.py            # DeepSeek LLM 抽取
│   │   ├── ner_extractor.py            # HuggingFace NER 抽取
│   │   ├── uie_extractor.py            # PaddleNLP UIE 抽取
│   │   └── dict_extractor.py           # 技能词典匹配
│   ├── ocr/
│   │   └── ocr_extractor.py            # EasyOCR/PaddleOCR 双后端封装
│   ├── graph/
│   │   ├── builder.py                  # StateGraph 构建与编译
│   │   ├── nodes.py                    # 9 个节点函数
│   │   └── router.py                   # 条件路由（preprocess → ocr/analyze；analyze → 工具）
│   ├── fusion/
│   │   └── merger.py                   # 3 种融合策略（rule_first / llm_first / field_level）
│   ├── evaluation/
│   │   └── evaluator.py                # Precision / Recall / F1 计算
│   └── web/
│       └── app.py                      # Streamlit 界面（5 个 Tab）
└── doc/
    └── 项目文档-LangGraph招聘信息抽取实验系统.md
```

---

## 4. 信息点定义

### 4.1 8 个抽取字段

```python
FIELDS = [
    "job_title",       # 岗位名称，如 "Java开发工程师"
    "company_name",    # 公司名称，如 "字节跳动"
    "work_location",   # 工作地点，如 "北京"
    "salary",          # 薪资范围，如 "20k-35k/月"
    "education",       # 学历要求，如 "本科及以上"
    "experience",      # 工作经验，如 "3年以上经验"
    "skills",          # 技能要求 (list[str])，如 ["Python", "Docker", "MySQL"]
    "contact_info",    # 联系方式（电话/邮箱），如 "13912345678"
]
```

### 4.2 LangGraph State

```python
class ExtractionState(TypedDict, total=False):
    # 输入
    doc_id: str
    language: str          # "zh" | "en"
    title: str
    raw_text: str
    # 预处理后
    cleaned_text: str
    sentences: list[str]
    tokens: list[str]
    # LLM 分析决策
    routing_decision: dict
    # 各抽取器结果
    regex_result: dict
    ner_result: dict
    dictionary_result: dict
    llm_result: dict
    uie_result: dict
    # 融合后
    final_result: dict
    # 评估
    evaluation: dict
    # 多媒体
    input_type: str        # "text" | "image"
    image_path: str
    ocr_text: str
```

### 4.3 输出格式示例

```json
{
  "id": "zh_001",
  "language": "zh",
  "job_title": "Java开发工程师",
  "company_name": "某科技公司",
  "work_location": "北京",
  "salary": "20k-30k/月",
  "education": "本科及以上",
  "experience": "3年以上",
  "skills": ["Java", "Spring Boot", "MySQL"],
  "contact_info": "hr@company.com",
  "extractor_breakdown": {
    "job_title": "ner+llm",
    "company_name": "ner+llm",
    "work_location": "ner+regex",
    "salary": "regex",
    "education": "regex",
    "experience": "regex",
    "skills": "dictionary+llm",
    "contact_info": "regex"
  },
  "confidence": {
    "job_title": 0.7,
    "company_name": 0.7,
    "work_location": 0.7,
    "salary": 0.9,
    "education": 0.9,
    "experience": 0.9,
    "skills": 0.67,
    "contact_info": 0.9
  },
  "conflicts": []
}
```

---

## 5. 各模块设计

### 5.1 语料管理模块

- **文本语料**: `src/corpus/generator.py` 基于模板素材池（20 个岗位名、20 家公司、14 个城市等）随机组合生成 150 篇中英双语招聘文本
- **图片语料**: `src/corpus/image_generator.py` 使用 Pillow 在 600×850 白色画布上绘制招聘海报，包含标题栏、岗位信息、技能要求和联系方式
- **语料加载**: `src/corpus/loader.py` 提供 `load_corpus()`、`load_image_corpus()`、`load_all_golden_labels()` 统一接口

### 5.2 预处理模块

`src/preprocess/` 包含：
- `cleaner.py`: 去除多余空格、HTML 标签、特殊符号
- `segmenter.py`: 中英文分句、jieba 中文分词

### 5.3 抽取器模块

所有抽取器继承 `BaseExtractor` 抽象基类（`src/extractors/base.py`），实现 `extract(text: str, language: str) -> dict` 接口。各抽取器与 8 个字段的覆盖关系如下：

| 字段 | Regex | NER | Dictionary | LLM | UIE |
|------|:-----:|:---:|:----------:|:---:|:---:|
| job_title（岗位名称） | | ✓ | | ✓ | ✓ |
| company_name（公司名称） | | ✓ | | ✓ | ✓ |
| work_location（工作地点） | ✓ | ✓ | | ✓ | ✓ |
| salary（薪资） | ✓ | | | ✓ | ✓ |
| education（学历） | ✓ | | | ✓ | ✓ |
| experience（经验） | ✓ | | | ✓ | ✓ |
| skills（技能要求） | | | ✓ | ✓ | ✓ |
| contact_info（联系方式） | ✓ | | | ✓ | |

> **Regex** 覆盖 5 个格式化字段（中英文双模式）；**NER** 覆盖 3 个实体字段（仅中文）；**Dictionary** 仅覆盖技能字段（关键词匹配）；**LLM** 覆盖全部 8 个字段（DeepSeek 语义抽取）；**UIE** 覆盖 7 个字段（无 contact_info）。

#### 抽取器选型理由

五个抽取器构成从**确定性规则 → 模式识别 → 语义理解**的互补梯度，缺一则某类字段或某种语言场景会出现短板：

| 抽取器 | 选型理由 |
|--------|----------|
| **Regex** | 薪资、学历、经验、联系方式等字段格式规律性强（数字+k、学位名称、手机号正则），正则匹配零幻觉、零成本、可复现。不适合技能字段——技能是离散关键词集合而非模式模板，写成巨型交替组 `(Python\|Java\|Docker\|...)` 维护成本远高于词典。 |
| **Dictionary** | 技能关键词是封闭的可枚举集合（60+ 常见技术栈），`set` 子串匹配比正则更简洁、增删改零成本。不适合公司名/地点——命名实体无法枚举穷尽，必须依赖模型。 |
| **NER** | 公司名、地点、岗位名无法用规则穷举，但属于命名实体这一明确定义的类别。预训练 RoBERTa-CLUENER 模型专门为中文实体识别训练，在确定性（模型固定、同输入同输出）和灵活性之间取得平衡。仅支持中文。 |
| **UIE** | PaddleNLP 的 UIE 基于 Schema 驱动的信息抽取，比正则灵活（能理解"985 高校毕业"隐含学历要求），比 LLM 稳定（模型权重固定，不会产生 LLM 的随机幻觉），且专门为信息抽取任务训练，对结构化输出的可靠性高于通用 LLM。 |
| **LLM** | 作为语义兜底覆盖全部 8 个字段。处理中英混合文本、隐式表达（"待遇从优" = 薪资面议）、非标准句式时，只有 LLM 具备足够的语言理解能力。同时兼任路由调度员（analyze 节点），分析文档特征后决定激活哪些工具。 |

五个抽取器的组合策略：同一字段被多个抽取器覆盖（如 skills 被 Dictionary + LLM + UIE 三重覆盖），融合时取并集或投票，降低单一工具偏差，同时互补各自的遗漏。

#### 正则抽取器 (RegexExtractor)

分别维护中英文模式字典，覆盖 salary、education、experience、work_location、contact_info 五个字段：

- **薪资**: `(\d{1,2}[kK]\s*[-~]\s*\d{1,2}[kK]\s*/\s*月)`、`(面议|薪资面谈|薪资面议)` 等
- **学历**: `(大专|本科|硕士|博士)(及以上|及以上学历|优先)?` 等
- **经验**: `(\d+\s*[-~]?\s*\d*\s*年\s*(以上\s*)?(相关)?(工作)?经验)` 等
- **地点**: `(工作地点[：:]\s*[一-鿿]+)`、`([一-鿿]{2,3}[市省区县])` 等
- **联系方式**: `(1[3-9]\d{9})`、`([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})` 等

#### LLM 抽取器 (LLMExtractor)

使用 DeepSeek Chat API，通过 System Prompt 约束模型以严格 JSON 格式返回 8 个字段。支持中英文两套提示词模板。

#### NER 抽取器 (NERExtractor)

使用 HuggingFace `uer/roberta-base-finetuned-cluener2020-chinese` 模型识别公司名、地点名、岗位名等命名实体。

#### UIE 抽取器 (UIEExtractor)

使用 PaddleNLP `uie-base` 模型，基于预定义的 8 个 Schema 进行通用信息抽取。

#### 词典抽取器 (DictionaryExtractor)

基于 30+ 技能关键词的中英文技能词典，在 jieba 分词结果上进行匹配。

### 5.4 OCR 模块

`src/ocr/ocr_extractor.py` 封装 EasyOCR/PaddleOCR 双后端（可通过 config.yaml 切换）：

- **EasyOCR**: 默认引擎，模型轻量，中文识别可靠性好，API 简单
- **PaddleOCR**: 备选引擎，中文识别精度更高但依赖较重

核心方法 `extract(image_path: str) -> str` 返回拼接后的 OCR 文本。角度分类自动校正倾斜文本。延迟初始化避免模型在导入时加载。

### 5.5 路由与调度

`src/graph/router.py` 包含两个路由函数：

1. **preprocess 后路由**: `input_type == "image"` → ocr 节点；否则 → analyze 节点
2. **analyze 后路由**: 解析 routing_decision JSON，将需要激活的工具映射到对应节点

**智能路由机制**：`analyze_node` 通过 LLMExtractor 的 `route()` 方法进行轻量路由分析——仅发送前 1500 字符给 DeepSeek，LLM 分析文档的语言混合度、句式规整度和字段显式程度，动态决定每个字段最适合的抽取工具组合。相比早期版本在 analyze 节点做全字段抽取 + 硬编码路由，轻量路由的 token 消耗大幅降低，且不同文档可激活不同的工具子集（而非每次都运行全部 5 个抽取器）。

### 5.6 融合策略

`src/fusion/merger.py` 支持四种策略（通过 `config.yaml` 的 `fusion_strategy` 切换）：

1. **rule_first（规则优先）**: 格式化字段（薪资/学历/经验/地点/联系方式）优先使用 regex/ner 结果，语义字段（岗位/公司/技能）优先使用 llm/uie 结果
2. **llm_first（LLM 优先）**: LLM 完成初步抽取，规则模块对关键字段做校验修正，不一致时标记冲突
3. **field_level（字段级融合）**: 按 routing_decision 中指定的优先级逐字段选择工具
4. **weighted_vote（加权投票）**: 按字段类型（格式化/实体/列表）给各抽取器分配置信度权重（通过 `config.yaml` 的 `confidence_weights` 配置），scalar 字段选最高权重工具的输出，skills 字段做跨源并集合并。每字段输出 confidence 分数

冲突检测：比较所有抽取器对同一字段的输出，有分歧时在 `conflicts` 列表中标记。

### 5.7 评估模块

`src/evaluation/evaluator.py` 计算字段级和整体的 Precision、Recall、F1。
- 从 150 篇语料中随机抽 25 篇文本 + 10 张图片作为评估集
- golden_labels 由语料生成器在生成时自动记录（生成参数即 ground truth）
- 分字段、分语言（zh/en）统计指标
- **skills 字段使用 Jaccard 相似度**：`Jaccard = |pred ∩ gold| / |pred ∪ gold|`，阈值 ≥ 0.5 计为匹配（TP），避免部分匹配的技能列表被完全判错。评估报告额外输出 `skills_jaccard` 字段记录每篇文档的技能 Jaccard 值

### 5.8 Web 界面

`src/web/app.py` 基于 Streamlit 构建，包含 5 个 Tab：

| Tab | 功能 |
|-----|------|
| 语料浏览 | 选择文档查看原文 |
| 抽取结果 | 查看融合后字段值、技能标签、各字段使用的抽取器 |
| 规则 vs LLM 对比 | 前 20 篇结果表格对比 |
| 评估报告 | 整体/分字段/分语言 Precision/Recall/F1 |
| 图片抽取 | 上传招聘海报 → OCR 识别预览 → 抽取结果展示；批量处理已有图片 |

---

## 6. 多媒体信息抽取

### 6.1 处理流程

```
招聘海报图片 (PNG/JPG)
    ↓
EasyOCR 文字识别
    ↓
文本清洗、分句、分词
    ↓
analyze (LLM 路由决策)
    ↓
regex / ner / dictionary / llm_extract / uie
    ↓
fusion (字段级融合)
    ↓
结构化结果（含 contact_info）
```

### 6.2 模拟图片生成

使用 Pillow 生成 600×850 招聘海报 PNG，包含：
- 蓝色标题栏（"招 聘" / "RECRUITMENT"）
- 公司名称 + 岗位名称（红色大字）
- 工作地点、薪资待遇、学历要求、经验要求
- 技能要求列表
- **联系方式**（联系电话 + 电子邮箱）—— 用于验证 contact_info 字段抽取

### 6.3 OCR 验证结果

在三张测试图片上的抽取效果：

| 字段 | 命中率 | 示例 |
|------|--------|------|
| job_title | 3/3 | 市场营销经理 |
| company_name | 3/3 | 百度在线 |
| work_location | 3/3 | 合肥 |
| salary | 3/3 | 薪资面谈 |
| education | 3/3 | 本科及以上 |
| experience | 3/3 | 5年以上经验 |
| skills | 3/3 | [Hadoop, Flink, Nginx] |
| contact_info | 3/3 | 13912345678 |

---

## 7. 实验设计

### 7.1 实验组

| 实验组 | 方案 |
|--------|------|
| 1 | 仅规则 (regex + dictionary) |
| 2 | 仅 LLM (DeepSeek 全字段) |
| 3 | NER + UIE |
| 4 | 规则优先融合 |
| 5 | LLM 优先融合 |
| 6 | 字段级融合 |
| 7 | 图片 OCR + 规则优先融合 |
| 8 | 加权投票融合 |

### 7.2 评估指标

- **Precision**: TP / (TP + FP)
- **Recall**: TP / (TP + FN)
- **F1**: 2 × P × R / (P + R)
- 分字段统计、分语言（zh/en）统计

---

## 8. 使用说明

### 8.1 安装依赖

```bash
pip install -r requirements.txt
```

### 8.2 配置

编辑 `config.yaml`，填入 DeepSeek API Key：

```yaml
deepseek:
  api_key: "your-api-key-here"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"
```

### 8.3 运行

**生成语料**：

```bash
python -m src.corpus.generator          # 生成 150 篇文本语料
python -m src.corpus.image_generator    # 生成 30 张招聘海报
```

**运行抽取**：

```bash
python src/main.py          # 纯文本模式
python src/main.py image    # 图片模式
python src/main.py all      # 混合模式
```

**启动 Web 界面**：

```bash
streamlit run src/web/app.py
```

### 8.4 配置说明

| 配置项 | 说明 | 可选值 |
|--------|------|--------|
| `fusion_strategy` | 融合策略 | `rule_first` / `llm_first` / `field_level` / `weighted_vote` |
| `corpus.total_docs` | 文本语料数量 | 100-200 |
| `corpus.zh_ratio` | 中文语料比例 | 0.0-1.0 |
| `ocr.engine` | OCR 引擎 | `easyocr` / `paddleocr` |
| `ocr.use_gpu` | 是否使用 GPU | `true` / `false` |
| `processing.max_workers` | 并行处理线程数 | 1-16（默认 min(8, cpu_count×2)） |
| `confidence_weights.formatted.*` | 格式化字段各抽取器置信度权重 | 0.0-1.0 |
| `confidence_weights.entity.*` | 实体字段各抽取器置信度权重 | 0.0-1.0 |
| `confidence_weights.list_field.*` | 列表字段各抽取器置信度权重 | 0.0-1.0 |

---

## 9. 项目特色与创新点

1. **LangGraph 多工具协同**: 不是简单的"规则 + LLM"二选一，而是通过 LangGraph 构建有向图，LLM 作为路由调度员，regex/ner/uie/dictionary/llm 五种工具并行工作
2. **四种融合策略可切换**: 规则优先、LLM 优先、字段级融合、加权投票，支持配置热切换，便于对比实验。加权投票策略按字段类型分配置信度权重，每字段输出 confidence 分数
3. **智能路由**: LLM 轻量分析文档特征（语言混合度、句式规整度），动态决定每个字段的抽取工具组合。不同文档激活不同工具子集，避免不必要的 LLM 全字段抽取
4. **图文混合抽取**: 通过 OCR 前置节点，将招聘海报图片纳入同一套抽取管道，无需单独开发
5. **中英双语适配**: 从语料生成、提示词模板到正则模式全部支持中英文双语
6. **端到端实验闭环**: 语料生成 → 预处理 → 多工具抽取 → 融合 → 评估 → Web 可视化，形成完整实验系统
7. **冲突检测与溯源**: 每个字段标注使用的抽取器（extractor_breakdown），多工具输出不一致时自动标记
8. **Jaccard 相似度评估**: skills 等列表字段使用 Jaccard 相似度替代完全匹配，避免部分正确的技能列表被完全判错，评估更合理
9. **文档级并行处理**: 基于 ThreadPoolExecutor 的并行化，多篇文档同时走 graph pipeline，提升处理吞吐量

---

## 10. 总结

本系统基于 LangChain + LangGraph 构建了一个面向中英双语招聘信息的多工具协同信息抽取实验平台。系统定义了 8 个结构化信息点（岗位、公司、地点、薪资、学历、经验、技能、联系方式），集成了正则、LLM、NER、UIE、词典五种抽取工具，并通过 LangGraph 的有向图工作流实现 LLM 路由调度与多工具并行。系统扩展了 OCR 模块支持招聘海报图片的文字识别与结构化抽取，形成"图文混合招聘信息抽取"能力。

在算法优化方面，系统实现了四项改进：（1）智能路由——analyze 节点从全字段抽取改为轻量路由分析，LLM 根据文档特征动态选择工具，降低 API 消耗；（2）Jaccard 相似度评估——skills 列表字段使用 Jaccard ≥ 0.5 阈值替代完全匹配；（3）加权投票融合——新增第四种融合策略，按字段类型分配置信度权重，输出 confidence 分数；（4）文档级并行——ThreadPoolExecutor 实现多文档并行处理。

通过 golden_labels（语料生成参数即 ground truth），系统可对多种抽取方案进行字段级 Precision/Recall/F1 评估，为对比实验和算法优化提供数据支撑。
