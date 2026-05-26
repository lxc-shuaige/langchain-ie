# 招聘信息抽取实验系统 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于 LangGraph + DeepSeek 的中英双语招聘信息抽取系统，融合 Regex/NER/UIE/Dictionary 多工具协同抽取 7 个信息点。

**Architecture:** LangGraph StateGraph 驱动 8 节点工作流（preprocess → analyze → 条件路由到 5 个并行抽取器 → fusion），Streamlit 做 Web 展示，所有抽取器实现统一 BaseExtractor 接口。

**Tech Stack:** langgraph, langchain-deepseek, transformers (bert-base-chinese-ner), paddlenlp (uie-base), jieba, streamlit, pyyaml

**注意:** 本项目目录不是 git 仓库，所有 `git commit` 步骤标记为可选。如需版本管理，先在 Task 1 中执行 `git init`。

---

## 文件规划

| 文件 | 职责 | 
|---|---|
| `src/state.py` | ExtractionState TypedDict 定义 + FIELDS 常量 |
| `src/corpus/generator.py` | 生成 150 篇中英双语招聘模拟语料 + golden_labels |
| `src/corpus/loader.py` | 扫描 data/corpus/ 目录，读入 DocumentRecord |
| `src/preprocess/cleaner.py` | 去空格/HTML/特殊字符 |
| `src/preprocess/segmenter.py` | 分句 + jieba 分词 |
| `src/extractors/base.py` | BaseExtractor 抽象类 |
| `src/extractors/regex_extractor.py` | 正则抽取薪资/学历/经验/地点 |
| `src/extractors/dict_extractor.py` | 技能词典匹配 |
| `src/extractors/llm_extractor.py` | DeepSeek LLM 全字段抽取 |
| `src/extractors/ner_extractor.py` | HuggingFace NER 抽取 |
| `src/extractors/uie_extractor.py` | PaddleNLP UIE 抽取 |
| `src/graph/nodes.py` | 8 个 LangGraph 节点函数 |
| `src/graph/router.py` | 条件路由逻辑 |
| `src/graph/builder.py` | StateGraph 构建 + 编译 |
| `src/fusion/merger.py` | 三种融合策略 |
| `src/evaluation/evaluator.py` | Precision/Recall/F1 计算 |
| `src/web/app.py` | Streamlit 界面 |
| `src/main.py` | CLI 入口 |
| `config.yaml` | API Key / 模型 / 路径配置 |
| `requirements.txt` | 依赖清单 |

---

### Task 1: 项目脚手架

**Files:**
- Create: `config.yaml`
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/corpus/__init__.py`
- Create: `src/preprocess/__init__.py`
- Create: `src/extractors/__init__.py`
- Create: `src/graph/__init__.py`
- Create: `src/fusion/__init__.py`
- Create: `src/evaluation/__init__.py`
- Create: `src/web/__init__.py`
- Create: `data/corpus/.gitkeep`
- Create: `data/labels/.gitkeep`
- Create: `data/output/.gitkeep`

- [ ] **Step 1: 创建 config.yaml**

```yaml
# DeepSeek API 配置
deepseek:
  api_key: "your-api-key-here"
  model: "deepseek-chat"
  base_url: "https://api.deepseek.com"
  temperature: 0.1
  max_tokens: 2048

# 路径配置
paths:
  corpus_dir: "data/corpus"
  labels_dir: "data/labels"
  output_dir: "data/output"

# 语料生成配置
corpus:
  total_docs: 150
  zh_ratio: 0.6   # 60% 中文, 40% 英文
  eval_sample: 25  # 评估集大小

# 融合策略: "rule_first" | "llm_first" | "field_level"
fusion_strategy: "rule_first"

# NER 模型
ner:
  model: "uer/roberta-base-finetuned-cluener2020-chinese"

# UIE 模型
uie:
  model: "uie-base"
  schema:
    - "岗位名称"
    - "公司名称"
    - "工作地点"
    - "薪资"
    - "学历要求"
    - "工作经验"
    - "技能要求"
```

- [ ] **Step 2: 创建 requirements.txt**

```text
langchain>=0.3.0
langchain-deepseek>=0.1.0
langgraph>=0.2.0
transformers>=4.45.0
torch>=2.0.0
paddlenlp>=3.0.0
jieba>=0.42.1
streamlit>=1.35.0
pyyaml>=6.0
```

- [ ] **Step 3: 创建所有 __init__.py 和占位目录**

```bash
cd "e:/大三下资料/infomation_acquire/ex3"
touch src/__init__.py
touch src/corpus/__init__.py
touch src/preprocess/__init__.py
touch src/extractors/__init__.py
touch src/graph/__init__.py
touch src/fusion/__init__.py
touch src/evaluation/__init__.py
touch src/web/__init__.py
mkdir -p data/corpus data/labels data/output
touch data/corpus/.gitkeep data/labels/.gitkeep data/output/.gitkeep
```

- [ ] **Step 4: 验证目录结构**

```bash
ls -R src/ data/
```

预期: 所有 `__init__.py` 和 data 子目录存在。

---

### Task 2: State 定义

**Files:**
- Create: `src/state.py`

- [ ] **Step 1: 编写 state.py**

```python
from typing import TypedDict


FIELDS = [
    "job_title",
    "company_name",
    "work_location",
    "salary",
    "education",
    "experience",
    "skills",
]


class ExtractionState(TypedDict, total=False):
    # 输入
    doc_id: str
    language: str
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
```

- [ ] **Step 2: 验证导入**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "from src.state import ExtractionState, FIELDS; print(FIELDS)"
```

预期: `['job_title', 'company_name', 'work_location', 'salary', 'education', 'experience', 'skills']`

---

### Task 3: 模拟语料生成器

**Files:**
- Create: `src/corpus/generator.py`

- [ ] **Step 1: 编写语料生成器**

```python
import json
import random
import os


# ---- 模板素材池 ----

ZH_JOB_TITLES = [
    "Java开发工程师", "Python后端开发", "前端开发工程师", "数据分析师",
    "产品经理", "UI/UX设计师", "测试工程师", "运维工程师",
    "算法工程师", "全栈开发工程师", "Android开发工程师", "iOS开发工程师",
    "网络安全工程师", "数据库管理员", "DevOps工程师", "技术支持工程师",
    "项目经理", "运营总监", "市场营销经理", "财务分析师",
]

EN_JOB_TITLES = [
    "Senior Backend Engineer", "Frontend Developer", "Data Scientist",
    "Product Manager", "UX Designer", "QA Engineer", "DevOps Engineer",
    "Machine Learning Engineer", "Full Stack Developer", "Software Engineer",
    "Cloud Architect", "Security Engineer", "Database Administrator",
    "Tech Lead", "Engineering Manager", "Business Analyst",
    "Marketing Specialist", "Financial Analyst", "HR Manager", "Sales Representative",
]

ZH_COMPANIES = [
    "字节跳动", "阿里巴巴", "腾讯科技", "华为技术", "百度在线",
    "京东集团", "美团", "滴滴出行", "小米科技", "网易集团",
    "拼多多", "快手科技", "哔哩哔哩", "知乎", "商汤科技",
    "科大讯飞", "蔚来汽车", "理想汽车", "大疆创新", "海康威视",
]

EN_COMPANIES = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta",
    "Netflix", "Tesla", "Uber", "Airbnb", "Stripe",
    "Snowflake", "Databricks", "Palantir", "Notion", "Figma",
    "Shopify", "Atlassian", "Canva", "Zoom", "Slack",
]

ZH_LOCATIONS = [
    "北京", "上海", "深圳", "杭州", "广州", "成都", "南京",
    "武汉", "西安", "苏州", "重庆", "长沙", "天津", "合肥",
]

EN_LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
    "London, UK", "Singapore", "Tokyo, Japan", "Berlin, Germany",
    "Toronto, Canada", "Sydney, Australia",
]

ZH_SALARIES = [
    "15k-25k/月", "20k-35k/月", "25k-40k/月", "30k-50k/月",
    "10k-18k/月", "18k-28k/月", "年薪30万-50万", "年薪40万-60万",
    "年薪20万-30万", "35k-55k/月", "面议", "薪资面谈",
]

EN_SALARIES = [
    "$80k-$120k annually", "$100k-$150k annually", "$120k-$180k annually",
    "$90k-$130k annually", "$70k-$100k annually", "$150k-$200k annually",
    "$60k-$85k annually", "$130k-$170k annually", "Competitive salary",
    "$40k-$60k annually",
]

ZH_EDUCATIONS = [
    "本科及以上", "硕士及以上", "本科", "硕士", "博士",
    "大专及以上", "本科及以上学历", "硕士优先", "博士优先",
    "计算机相关专业本科", "985/211本科及以上",
]

EN_EDUCATIONS = [
    "Bachelor's degree required", "Master's degree preferred",
    "PhD in related field", "Bachelor's or above",
    "Master's degree in Computer Science", "Bachelor's degree in related field",
    "MBA preferred", "Master's or PhD", "Bachelor's minimum",
]

ZH_EXPERIENCES = [
    "3年以上经验", "1-3年经验", "5年以上经验", "2年以上经验",
    "3-5年工作经验", "1年以上经验", "应届生可投", "10年以上经验",
    "3年以上相关经验", "5-8年经验",
]

EN_EXPERIENCES = [
    "3+ years of experience", "1-3 years of experience",
    "5+ years of experience", "2+ years of experience",
    "3-5 years of experience", "Entry level welcome",
    "7+ years of experience", "4-6 years of experience",
    "10+ years of experience",
]

ZH_SKILLS_POOL = [
    "Java", "Python", "Spring Boot", "MySQL", "Redis", "Docker",
    "Kubernetes", "Linux", "Git", "MongoDB", "Nginx", "RabbitMQ",
    "Kafka", "Elasticsearch", "Hadoop", "Spark", "Flink", "TensorFlow",
    "PyTorch", "React", "Vue.js", "TypeScript", "Node.js", "Go",
    "C++", "Rust", "AWS", "Azure", "PostgreSQL", "微服务架构",
    "分布式系统", "敏捷开发", "CI/CD", "Data Warehouse",
]

EN_SKILLS_POOL = [
    "Python", "Java", "Go", "Rust", "C++", "TypeScript",
    "React", "Vue", "Angular", "Node.js", "Django", "Spring",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "Kafka", "Spark", "Flink", "TensorFlow", "PyTorch", "SQL",
    "PostgreSQL", "MongoDB", "Redis", "GraphQL", "gRPC",
    "CI/CD", "Agile", "Microservices", "System Design",
]


def _gen_zh_doc(idx: int) -> dict:
    title = random.choice(ZH_JOB_TITLES)
    company = random.choice(ZH_COMPANIES)
    location = random.choice(ZH_LOCATIONS)
    salary = random.choice(ZH_SALARIES)
    education = random.choice(ZH_EDUCATIONS)
    experience = random.choice(ZH_EXPERIENCES)
    n_skills = random.randint(3, 8)
    skills = random.sample(ZH_SKILLS_POOL, n_skills)

    text = f"""{company} - 招聘{title}

【岗位名称】{title}
【公司名称】{company}
【工作地点】{location}
【薪资范围】{salary}

【岗位职责】
1. 负责公司核心业务系统的设计、开发与维护工作；
2. 参与系统架构设计，保证系统的高可用、高性能和可扩展性；
3. 与产品经理和团队成员紧密协作，推动项目按时高质量交付；
4. 撰写技术文档，参与代码评审，提升团队整体技术水平。

【任职要求】
1. 学历要求：{education}；
2. 工作经验：{experience}；
3. 熟练掌握{'、'.join(skills[:3])}等技术；
4. 具备良好的沟通能力和团队协作精神；
5. 具有较强的学习能力和问题分析解决能力。

【福利待遇】
五险一金、带薪年假、弹性工作制、免费三餐、年度体检、股票期权等。

有意者请将简历发送至 hr@{company.lower().replace(' ', '')}.com
"""

    return {
        "id": f"zh_{idx:03d}",
        "language": "zh",
        "title": title,
        "text": text,
        "labels": {
            "job_title": title,
            "company_name": company,
            "work_location": location,
            "salary": salary,
            "education": education,
            "experience": experience,
            "skills": sorted(skills),
        },
    }


def _gen_en_doc(idx: int) -> dict:
    title = random.choice(EN_JOB_TITLES)
    company = random.choice(EN_COMPANIES)
    location = random.choice(EN_LOCATIONS)
    salary = random.choice(EN_SALARIES)
    education = random.choice(EN_EDUCATIONS)
    experience = random.choice(EN_EXPERIENCES)
    n_skills = random.randint(3, 8)
    skills = random.sample(EN_SKILLS_POOL, n_skills)

    text = f"""{company} - {title}

Position: {title}
Company: {company}
Location: {location}
Salary: {salary}

About the Role:
We are looking for a talented {title} to join our growing team at {company}. You will work on cutting-edge projects and collaborate with cross-functional teams to deliver high-quality solutions.

Key Responsibilities:
- Design, develop and maintain scalable backend services and APIs.
- Collaborate with product managers and frontend engineers to ship new features.
- Participate in code reviews and contribute to engineering best practices.
- Mentor junior engineers and help grow the team.
- Optimize application performance and reliability.

Qualifications:
- Education: {education}.
- Experience: {experience}.
- Proficiency in {', '.join(skills[:3])} and related technologies.
- Strong problem-solving skills and attention to detail.
- Excellent communication and teamwork abilities.

Benefits:
Competitive compensation package, health insurance, 401(k) matching, flexible PTO, remote-friendly culture.

To apply, please send your resume to careers@{company.lower().replace(' ', '')}.com
"""

    return {
        "id": f"en_{idx:03d}",
        "language": "en",
        "title": title,
        "text": text,
        "labels": {
            "job_title": title,
            "company_name": company,
            "work_location": location,
            "salary": salary,
            "education": education,
            "experience": experience,
            "skills": sorted(skills),
        },
    }


def generate_corpus(
    total: int = 150,
    zh_ratio: float = 0.6,
    output_dir: str = "data/corpus",
) -> list[dict]:
    """生成模拟招聘语料并写入 .txt 文件，返回所有文档的元数据列表。"""
    os.makedirs(output_dir, exist_ok=True)

    n_zh = int(total * zh_ratio)
    n_en = total - n_zh

    metadata_list = []

    for i in range(n_zh):
        doc = _gen_zh_doc(i + 1)
        filepath = os.path.join(output_dir, f"{doc['id']}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        metadata_list.append({
            "id": doc["id"],
            "language": doc["language"],
            "title": doc["title"],
            "file": filepath,
            "labels": doc["labels"],
        })

    for i in range(n_en):
        doc = _gen_en_doc(i + 1)
        filepath = os.path.join(output_dir, f"{doc['id']}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        metadata_list.append({
            "id": doc["id"],
            "language": doc["language"],
            "title": doc["title"],
            "file": filepath,
            "labels": doc["labels"],
        })

    return metadata_list


def generate_golden_labels(
    metadata_list: list[dict],
    eval_sample: int = 25,
    output_dir: str = "data/labels",
) -> list[dict]:
    """从 metadata 中随机抽样生成 golden_labels。"""
    os.makedirs(output_dir, exist_ok=True)

    sample = random.sample(metadata_list, min(eval_sample, len(metadata_list)))
    golden = [{"id": m["id"], "language": m["language"], "labels": m["labels"]} for m in sample]

    filepath = os.path.join(output_dir, "golden_labels.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    return golden


if __name__ == "__main__":
    metadata = generate_corpus(total=150, zh_ratio=0.6, output_dir="data/corpus")
    print(f"生成语料: {len(metadata)} 篇")

    golden = generate_golden_labels(metadata, eval_sample=25, output_dir="data/labels")
    print(f"生成 golden_labels: {len(golden)} 篇")
```

- [ ] **Step 2: 运行生成器**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -m src.corpus.generator
```

预期: 输出 `生成语料: 150 篇` 和 `生成 golden_labels: 25 篇`，`data/corpus/` 下有 150 个 .txt 文件。

- [ ] **Step 3: 验证生成结果**

```bash
ls "e:/大三下资料/infomation_acquire/ex3/data/corpus/" | wc -l && head -5 "e:/大三下资料/infomation_acquire/ex3/data/corpus/zh_001.txt"
```

预期: `150` 且输出中文招聘文本内容。

---

### Task 4: 语料加载器

**Files:**
- Create: `src/corpus/loader.py`

- [ ] **Step 1: 编写 loader.py**

```python
import json
import os
from dataclasses import dataclass


@dataclass
class DocumentRecord:
    id: str
    language: str
    title: str
    text: str


def load_corpus(corpus_dir: str = "data/corpus") -> list[DocumentRecord]:
    """扫描语料目录，返回所有 DocumentRecord 列表。"""
    docs = []
    if not os.path.isdir(corpus_dir):
        return docs

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        doc_id = filename.replace(".txt", "")
        language = "zh" if doc_id.startswith("zh") else "en"
        title = text.strip().split("\n")[0] if text else ""

        docs.append(DocumentRecord(
            id=doc_id,
            language=language,
            title=title,
            text=text,
        ))

    return docs


def load_golden_labels(labels_path: str = "data/labels/golden_labels.json") -> list[dict]:
    """加载人工标注评估集。"""
    if not os.path.isfile(labels_path):
        return []
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)
```

- [ ] **Step 2: 验证加载器**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus, load_golden_labels
docs = load_corpus()
print(f'Loaded {len(docs)} documents')
print(f'First doc: {docs[0].id}, lang={docs[0].language}')
golden = load_golden_labels()
print(f'Golden labels: {len(golden)} entries')
"
```

预期: `Loaded 150 documents`, `First doc: en_001, lang=en`, `Golden labels: 25 entries`

---

### Task 5: 预处理模块

**Files:**
- Create: `src/preprocess/cleaner.py`
- Create: `src/preprocess/segmenter.py`

- [ ] **Step 1: 编写 cleaner.py**

```python
import re


def clean_text(text: str) -> str:
    """清洗文本：去 HTML 标签、多余空白、特殊字符。"""
    # 去 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 将多个空白符（含 \\r\\n）替换为单个换行
    text = re.sub(r"[ \t]+", " ", text)
    # 合并多余空行为单个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉首尾空白
    text = text.strip()
    return text
```

- [ ] **Step 2: 编写 segmenter.py**

```python
import re
import jieba


def split_sentences(text: str) -> list[str]:
    """中英文分句：按句号、问号、感叹号、换行等切分。"""
    # 用正则在中英文句末标点处切分
    sentences = re.split(r"(?<=[。！？.!?\n])\s*", text)
    return [s.strip() for s in sentences if s.strip()]


def tokenize(text: str) -> list[str]:
    """使用 jieba 进行中文分词，英文保留原样。"""
    return list(jieba.cut(text))
```

- [ ] **Step 3: 测试预处理**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.preprocess.cleaner import clean_text
from src.preprocess.segmenter import split_sentences, tokenize

docs = load_corpus()
doc = [d for d in docs if d.id == 'zh_001'][0]
cleaned = clean_text(doc.text)
sents = split_sentences(cleaned)
tokens = tokenize(cleaned)
print(f'Cleaned length: {len(cleaned)} chars')
print(f'Sentences: {len(sents)}')
print(f'Tokens (first 30): {tokens[:30]}')
"
```

预期: 清洗后字符数 > 0，分句数 > 3，分词结果包含中文词。

---

### Task 6: BaseExtractor 抽象类

**Files:**
- Create: `src/extractors/base.py`

- [ ] **Step 1: 编写 base.py**

```python
from abc import ABC, abstractmethod
from src.state import FIELDS


class BaseExtractor(ABC):
    """所有抽取器的统一抽象接口。"""

    name: str = "base"

    @abstractmethod
    def extract(self, text: str, language: str) -> dict:
        """
        从文本中抽取结构化信息。

        Args:
            text: 清洗后的文本
            language: "zh" | "en"

        Returns:
            dict: 包含 FIELDS 中部分或全部字段的结果，
                  如 {"job_title": "...", "salary": "...", ...}
                  未抽取到的字段不出现或值为 None。
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.name}>"


def empty_result() -> dict:
    """返回一个所有字段为 None 的空结果。"""
    return {f: None for f in FIELDS}
```

- [ ] **Step 2: 验证抽象类**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.extractors.base import BaseExtractor, empty_result
print(empty_result())
"
```

预期: `{'job_title': None, 'company_name': None, ...}`

---

### Task 7: 正则抽取器

**Files:**
- Create: `src/extractors/regex_extractor.py`

- [ ] **Step 1: 编写 regex_extractor.py**

```python
import re
from src.extractors.base import BaseExtractor, empty_result


class RegexExtractor(BaseExtractor):
    name = "regex"

    # ---- 中文模式 ----
    ZH_PATTERNS = {
        "salary": [
            r"(\d{1,2}[kK]\s*[-~]\s*\d{1,2}[kK]\s*/\s*月)",
            r"(年薪\s*\d{1,3}\s*万\s*[-~]\s*\d{1,3}\s*万)",
            r"(年薪\s*\d{1,3}\s*万起?)",
            r"(\d{1,2}\s*[-~]\s*\d{1,2}\s*[kK]\s*/\s*月)",
            r"(月薪\s*\d{1,2}\s*[-~]\s*\d{1,2}\s*[kK])",
            r"(薪资[：:]?\s*\d{1,2}[kK]\s*[-~]\s*\d{1,2}[kK])",
        ],
        "education": [
            r"(大专|本科|硕士|博士)(及以上|及以上学历|优先)?",
            r"(博士|硕士|本科|大专)学历",
        ],
        "experience": [
            r"(\d+\s*[-~]?\s*\d*\s*年\s*(以上\s*)?(相关)?(工作)?经验)",
            r"(\d+\+?\s*年\s*以\s*上?\s*(相关)?(工作)?经验)",
            r"(应届[生生]|应届毕业生)",
        ],
        "work_location": [
            r"(工作地点[：:]\s*[一-鿿]+)",
            r"([一-鿿]{2,3}[市省区县])",
        ],
    }

    # ---- 英文模式 ----
    EN_PATTERNS = {
        "salary": [
            r"(\$\d{2,3}[kK]\s*[-~]\s*\$\d{2,3}[kK]\s*(annually|per year)?)",
            r"(\$\d{2,3}[kK]\s*[-~]\s*\$\d{2,3}[kK])",
            r"(\d{2,3}[kK]\s*[-~]\s*\d{2,3}[kK]\s*(CNY|USD|EUR)?)",
        ],
        "education": [
            r"(Bachelor'?s?\s*degree\s*(required|preferred)?)",
            r"(Master'?s?\s*degree\s*(required|preferred|in\s+\w+)?)",
            r"(PhD\s*(in\s+\w+)?\s*(required|preferred)?)",
            r"(MBA\s*(preferred)?)",
        ],
        "experience": [
            r"(\d+\+?\s*years?\s*(of\s*)?(related\s*)?experience)",
            r"(\d+\s*[-~]\s*\d+\s*years?\s*(of\s*)?(related\s*)?experience)",
            r"(Entry\s*level)",
        ],
        "work_location": [
            r"(Location[：:]\s*[\w\s,]+)",
            r"(Location[：:]\s*Remote)",
        ],
    }

    def extract(self, text: str, language: str) -> dict:
        patterns = self.ZH_PATTERNS if language == "zh" else self.EN_PATTERNS
        result = empty_result()

        for field, regex_list in patterns.items():
            for regex in regex_list:
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    result[field] = match.group(0).strip()
                    break

        return result
```

- [ ] **Step 2: 测试中文正则抽取**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.extractors.regex_extractor import RegexExtractor

docs = load_corpus()
zh_doc = [d for d in docs if d.id == 'zh_001'][0]
extractor = RegexExtractor()
result = extractor.extract(zh_doc.text, 'zh')
for k, v in result.items():
    if v:
        print(f'{k}: {v}')
"
```

预期: 至少抽取出 `salary`、`education`、`experience`、`work_location`。

- [ ] **Step 3: 测试英文正则抽取**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.extractors.regex_extractor import RegexExtractor

docs = load_corpus()
en_doc = [d for d in docs if d.id == 'en_001'][0]
extractor = RegexExtractor()
result = extractor.extract(en_doc.text, 'en')
for k, v in result.items():
    if v:
        print(f'{k}: {v}')
"
```

预期: 至少抽取出 `salary`、`education`、`experience`。

---

### Task 8: 技能词典抽取器

**Files:**
- Create: `src/extractors/dict_extractor.py`

- [ ] **Step 1: 编写 dict_extractor.py**

```python
from src.extractors.base import BaseExtractor, empty_result


# 中英文技能词库
ZH_SKILLS = {
    "Java", "Python", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
    "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "Shell",
    "Spring Boot", "Spring", "MyBatis", "Hibernate", "Django", "Flask",
    "FastAPI", "React", "Vue", "Vue.js", "Angular", "Node.js",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Oracle", "SQL Server", "SQLite", "HBase", "Cassandra",
    "Docker", "Kubernetes", "K8s", "Jenkins", "GitLab CI",
    "Nginx", "Tomcat", "Apache", "Linux", "Shell脚本",
    "Hadoop", "Spark", "Flink", "Storm", "Kafka", "RabbitMQ",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas",
    "NumPy", "OpenCV", "NLP", "机器学习", "深度学习",
    "AWS", "Azure", "阿里云", "腾讯云", "华为云", "GCP",
    "Git", "SVN", "Maven", "Gradle", "Webpack", "Vite",
    "微服务", "分布式", "高并发", "敏捷开发", "DevOps",
    "RESTful", "GraphQL", "gRPC", "WebSocket",
    "数据分析", "数据挖掘", "数据仓库", "ETL",
    "Unity", "Unreal", "Cocos",
}

EN_SKILLS = {
    "Python", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
    "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "Bash",
    "Spring Boot", "Spring", "Hibernate", "Django", "Flask",
    "FastAPI", "React", "Vue", "Angular", "Node.js", "Express",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Oracle", "SQL Server", "SQLite", "DynamoDB",
    "Docker", "Kubernetes", "K8s", "Jenkins", "GitHub Actions",
    "Nginx", "Apache", "Linux", "Unix",
    "Hadoop", "Spark", "Flink", "Kafka", "RabbitMQ", "SQS",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas",
    "NumPy", "OpenCV", "NLP", "Machine Learning", "Deep Learning",
    "AWS", "Azure", "GCP", "Terraform", "Ansible",
    "Git", "Maven", "Gradle", "Webpack", "Vite",
    "Microservices", "Distributed Systems", "Agile", "DevOps", "CI/CD",
    "REST", "RESTful", "GraphQL", "gRPC", "WebSocket",
    "Data Analysis", "Data Mining", "Data Warehouse", "ETL",
    "Unity", "Unreal Engine",
}


class DictionaryExtractor(BaseExtractor):
    name = "dictionary"

    def extract(self, text: str, language: str) -> dict:
        skills_pool = ZH_SKILLS if language == "zh" else EN_SKILLS
        found = set()

        for skill in skills_pool:
            if skill.lower() in text.lower():
                found.add(skill)

        result = empty_result()
        result["skills"] = sorted(found) if found else None
        return result
```

- [ ] **Step 2: 测试词典抽取**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.extractors.dict_extractor import DictionaryExtractor

docs = load_corpus()
zh_doc = [d for d in docs if d.id == 'zh_001'][0]
extractor = DictionaryExtractor()
result = extractor.extract(zh_doc.text, 'zh')
print(f'Skills found: {result[\"skills\"]}')
"
```

预期: 输出至少 3 个技能词。

---

### Task 9: DeepSeek LLM 抽取器

**Files:**
- Create: `src/extractors/llm_extractor.py`

- [ ] **Step 1: 编写 llm_extractor.py**

```python
import json
import yaml
from langchain_deepseek import ChatDeepSeek
from langchain_core.messages import HumanMessage, SystemMessage
from src.extractors.base import BaseExtractor, empty_result


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


SYSTEM_PROMPT_ZH = """你是一个信息抽取助手。从给定的招聘文本中抽取以下字段，返回严格的 JSON 格式。

字段说明：
- job_title: 岗位名称
- company_name: 公司名称
- work_location: 工作地点
- salary: 薪资范围
- education: 学历要求
- experience: 工作经验要求
- skills: 技能要求（返回字符串数组）

要求：
1. 若某字段在原文中不存在，将其值设为 null。
2. 保持原文表述，不要编造信息。
3. skills 必须是数组格式。
4. 只返回 JSON，不要包含其他文字。"""

SYSTEM_PROMPT_EN = """You are an information extraction assistant. Extract the following fields from the given job posting text. Return strict JSON format.

Fields:
- job_title: job position title
- company_name: company name
- work_location: work location
- salary: salary range
- education: education requirement
- experience: work experience requirement
- skills: required skills (return as array of strings)

Rules:
1. If a field is not present in the text, set its value to null.
2. Keep the original wording, do not fabricate information.
3. skills must be an array.
4. Return only JSON, no other text."""


class LLMExtractor(BaseExtractor):
    name = "llm"

    def __init__(self):
        config = _load_config()
        self.model = ChatDeepSeek(
            model=config["deepseek"]["model"],
            api_key=config["deepseek"]["api_key"],
            api_base=config["deepseek"]["base_url"],
            temperature=config["deepseek"]["temperature"],
            max_tokens=config["deepseek"]["max_tokens"],
        )

    def extract(self, text: str, language: str) -> dict:
        system_prompt = SYSTEM_PROMPT_ZH if language == "zh" else SYSTEM_PROMPT_EN

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"招聘文本：\n{text}" if language == "zh" else f"Job posting text:\n{text}"),
        ]

        response = self.model.invoke(messages)
        content = response.content.strip()

        # 清理可能的 markdown 代码块包裹
        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            if content.startswith("json"):
                content = content[4:].strip()

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return empty_result()

        result = empty_result()
        for field in result:
            if field in parsed:
                val = parsed[field]
                if field == "skills" and isinstance(val, list):
                    result[field] = val
                elif isinstance(val, str):
                    result[field] = val
        return result
```

- [ ] **Step 2: 验证 LLM 抽取器（需要配置 API Key）**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.extractors.llm_extractor import LLMExtractor

docs = load_corpus()
zh_doc = [d for d in docs if d.id == 'zh_001'][0]
extractor = LLMExtractor()
result = extractor.extract(zh_doc.text, 'zh')
for k, v in result.items():
    print(f'{k}: {v}')
"
```

预期: 7 个字段均有值，`skills` 为数组。

---

### Task 10: NER 抽取器 (HuggingFace)

**Files:**
- Create: `src/extractors/ner_extractor.py`

- [ ] **Step 1: 编写 ner_extractor.py**

```python
import yaml
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from src.extractors.base import BaseExtractor, empty_result


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NERExtractor(BaseExtractor):
    name = "ner"

    def __init__(self):
        config = _load_config()
        model_name = config["ner"]["model"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipeline = pipeline(
            "token-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            aggregation_strategy="simple",
        )

    def extract(self, text: str, language: str) -> dict:
        if language == "en":
            # HuggingFace 中文 NER 模型对英文效果差，跳过
            return empty_result()

        entities = self._pipeline(text[:2000])  # 限制长度避免 OOM

        result = empty_result()
        for ent in entities:
            label = ent["entity_group"]
            word = ent["word"]
            if label == "company" and not result["company_name"]:
                result["company_name"] = word
            elif label == "address" and not result["work_location"]:
                result["work_location"] = word
            elif label == "position" and not result["job_title"]:
                result["job_title"] = word

        return result
```

- [ ] **Step 2: 验证 NER 抽取器**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.extractors.ner_extractor import NERExtractor

docs = load_corpus()
zh_doc = [d for d in docs if d.id == 'zh_001'][0]
extractor = NERExtractor()
result = extractor.extract(zh_doc.text, 'zh')
for k, v in result.items():
    if v:
        print(f'{k}: {v}')
"
```

预期: 输出 NER 识别到的实体（首次运行需下载模型 ~400MB）。

---

### Task 11: UIE 抽取器 (PaddleNLP)

**Files:**
- Create: `src/extractors/uie_extractor.py`

- [ ] **Step 1: 编写 uie_extractor.py**

```python
import yaml
from paddlenlp import Taskflow
from src.extractors.base import BaseExtractor, empty_result


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class UIEExtractor(BaseExtractor):
    name = "uie"

    def __init__(self):
        config = _load_config()
        schema = config["uie"]["schema"]
        self._uie = Taskflow("information_extraction", schema=schema, model="uie-base")

    def extract(self, text: str, language: str) -> dict:
        if language == "en":
            # UIE 主要面向中文，尝试执行但效果有限
            pass

        try:
            outputs = self._uie(text[:2000])
        except Exception:
            return empty_result()

        result = empty_result()

        # Schema 映射到 FIELDS
        schema_map = {
            "岗位名称": "job_title",
            "公司名称": "company_name",
            "工作地点": "work_location",
            "薪资": "salary",
            "学历要求": "education",
            "工作经验": "experience",
            "技能要求": "skills",
        }

        for item in outputs:
            for key, value in item.items():
                field = schema_map.get(key)
                if field and not result[field]:
                    text_val = value[0]["text"] if isinstance(value, list) and value else str(value)
                    result[field] = text_val

        return result
```

- [ ] **Step 2: 验证 UIE 抽取器**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.extractors.uie_extractor import UIEExtractor

docs = load_corpus()
zh_doc = [d for d in docs if d.id == 'zh_001'][0]
extractor = UIEExtractor()
result = extractor.extract(zh_doc.text, 'zh')
for k, v in result.items():
    if v:
        print(f'{k}: {v}')
"
```

预期: 输出 UIE 抽取结果（首次运行需下载模型 ~500MB）。

---

### Task 12: LangGraph 图构建

**Files:**
- Create: `src/graph/nodes.py`
- Create: `src/graph/router.py`
- Create: `src/graph/builder.py`

- [ ] **Step 1: 编写 nodes.py**

```python
"""LangGraph 8 个节点函数。"""
from src.state import ExtractionState
from src.preprocess.cleaner import clean_text
from src.preprocess.segmenter import split_sentences, tokenize
from src.extractors.regex_extractor import RegexExtractor
from src.extractors.llm_extractor import LLMExtractor
from src.extractors.ner_extractor import NERExtractor
from src.extractors.uie_extractor import UIEExtractor
from src.extractors.dict_extractor import DictionaryExtractor
from src.fusion.merger import merge_results


# 延迟初始化：避免在模块加载时就下载模型
_extractors: dict[str, object] = {}


def _get_extractor(name: str):
    if name not in _extractors:
        if name == "regex":
            _extractors[name] = RegexExtractor()
        elif name == "llm":
            _extractors[name] = LLMExtractor()
        elif name == "ner":
            _extractors[name] = NERExtractor()
        elif name == "uie":
            _extractors[name] = UIEExtractor()
        elif name == "dictionary":
            _extractors[name] = DictionaryExtractor()
    return _extractors[name]


def preprocess_node(state: ExtractionState) -> dict:
    """预处理节点：清洗、分句、分词。"""
    text = state.get("raw_text", "")
    cleaned = clean_text(text)
    sentences = split_sentences(cleaned)
    tokens = tokenize(cleaned)
    return {
        "cleaned_text": cleaned,
        "sentences": sentences,
        "tokens": tokens,
    }


def analyze_node(state: ExtractionState) -> dict:
    """分析节点：让 LLM 分析文档特征，输出路由决策。"""
    llm = _get_extractor("llm")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    title = state.get("title", "")

    # 用 LLM 做一次轻量抽取 + 路由建议
    # 由于我们的 LLMExtractor.extract 已经返回全部字段，
    # 这里直接用它的结果作为路由决策的基础
    full_result = llm.extract(text, language)

    routing = {
        "salary": "regex",
        "education": "regex",
        "experience": "regex",
        "work_location": "ner+regex",
        "company_name": "ner+llm",
        "job_title": "uie+llm",
        "skills": "dictionary+llm",
    }

    return {
        "routing_decision": routing,
        "llm_result": full_result,
    }


def regex_node(state: ExtractionState) -> dict:
    regex = _get_extractor("regex")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    result = regex.extract(text, language)
    return {"regex_result": result}


def ner_node(state: ExtractionState) -> dict:
    ner = _get_extractor("ner")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    result = ner.extract(text, language)
    return {"ner_result": result}


def dictionary_node(state: ExtractionState) -> dict:
    dic = _get_extractor("dictionary")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    result = dic.extract(text, language)
    return {"dictionary_result": result}


def llm_extract_node(state: ExtractionState) -> dict:
    # analyze_node 已经调用了 LLM，直接复用结果
    existing = state.get("llm_result")
    if existing and any(v for v in existing.values()):
        return {}
    llm = _get_extractor("llm")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    result = llm.extract(text, language)
    return {"llm_result": result}


def uie_node(state: ExtractionState) -> dict:
    uie = _get_extractor("uie")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    result = uie.extract(text, language)
    return {"uie_result": result}


def fusion_node(state: ExtractionState) -> dict:
    """融合节点：汇总所有抽取器结果，按策略合并。"""
    routing = state.get("routing_decision", {})
    regex_result = state.get("regex_result", {})
    ner_result = state.get("ner_result", {})
    dictionary_result = state.get("dictionary_result", {})
    llm_result = state.get("llm_result", {})
    uie_result = state.get("uie_result", {})

    final = merge_results(
        regex=regex_result,
        ner=ner_result,
        dictionary=dictionary_result,
        llm=llm_result,
        uie=uie_result,
        routing=routing,
    )

    return {"final_result": final}
```

- [ ] **Step 2: 编写 router.py**

```python
"""条件路由逻辑：根据 routing_decision 决定激活哪些节点。"""


def route_after_analyze(state: dict) -> list[str]:
    """根据 routing_decision 返回需要执行的节点列表。"""
    routing = state.get("routing_decision", {})

    needed = set()
    for field, tools in routing.items():
        for tool in tools.split("+"):
            tool = tool.strip()
            if tool in ("regex", "ner", "llm", "uie", "dictionary"):
                needed.add(tool)

    if not needed:
        needed = {"regex", "llm", "dictionary"}

    nodes = []
    for name in ["regex", "ner", "dictionary", "llm_extract", "uie"]:
        base = {"llm_extract": "llm", "dictionary": "dictionary"}.get(name, name)
        if base in needed:
            nodes.append(name)

    return nodes
```

- [ ] **Step 3: 编写 builder.py**

```python
"""LangGraph StateGraph 构建与编译。"""
from langgraph.graph import StateGraph, END
from src.state import ExtractionState
from src.graph.nodes import (
    preprocess_node,
    analyze_node,
    regex_node,
    ner_node,
    dictionary_node,
    llm_extract_node,
    uie_node,
    fusion_node,
)
from src.graph.router import route_after_analyze


def build_graph() -> StateGraph:
    """构建并编译抽取工作流图。"""
    builder = StateGraph(ExtractionState)

    # 添加节点
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("regex", regex_node)
    builder.add_node("ner", ner_node)
    builder.add_node("dictionary", dictionary_node)
    builder.add_node("llm_extract", llm_extract_node)
    builder.add_node("uie", uie_node)
    builder.add_node("fusion", fusion_node)

    # 设置入口
    builder.set_entry_point("preprocess")

    # 边: preprocess → analyze
    builder.add_edge("preprocess", "analyze")

    # 条件路由: analyze → [regex, ner, dictionary, llm_extract, uie]
    builder.add_conditional_edges(
        "analyze",
        route_after_analyze,
        {
            "regex": "regex",
            "ner": "ner",
            "dictionary": "dictionary",
            "llm_extract": "llm_extract",
            "uie": "uie",
        },
    )

    # 所有抽取节点 → fusion
    builder.add_edge("regex", "fusion")
    builder.add_edge("ner", "fusion")
    builder.add_edge("dictionary", "fusion")
    builder.add_edge("llm_extract", "fusion")
    builder.add_edge("uie", "fusion")

    # fusion → END
    builder.add_edge("fusion", END)

    return builder.compile()
```

- [ ] **Step 4: 验证图构建**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.graph.builder import build_graph
graph = build_graph()
print('Graph compiled successfully')
print(f'Nodes: {list(graph.nodes.keys())}')
"
```

预期: `Graph compiled successfully` 和节点列表。

---

### Task 13: 结果融合模块

**Files:**
- Create: `src/fusion/merger.py`

- [ ] **Step 1: 编写 merger.py**

```python
"""结果融合策略。"""
import yaml
from src.state import FIELDS


def _load_strategy() -> str:
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config.get("fusion_strategy", "rule_first")


def merge_results(
    regex: dict,
    ner: dict,
    dictionary: dict,
    llm: dict,
    uie: dict,
    routing: dict,
) -> dict:
    """根据配置的策略融合多抽取器结果。"""
    strategy = _load_strategy()

    if strategy == "rule_first":
        return _rule_first(regex, ner, dictionary, llm, uie, routing)
    elif strategy == "llm_first":
        return _llm_first(regex, ner, dictionary, llm, uie, routing)
    else:
        return _field_level(regex, ner, dictionary, llm, uie, routing)


def _pick(sources: list[dict], field: str) -> str | list[str] | None:
    """从多个来源中取第一个非空值。"""
    for src in sources:
        val = src.get(field) if src else None
        if val:
            return val
    return None


def _rule_first(regex, ner, dictionary, llm, uie, routing) -> dict:
    """规则优先：格式化字段用规则，语义字段用 LLM/UIE。"""
    result = {}
    extractor_breakdown = {}
    conflicts = []

    rule_fields = {"salary", "education", "experience", "work_location"}
    semantic_fields = {"job_title", "company_name", "skills"}

    for f in FIELDS:
        if f in rule_fields:
            val = _pick([regex, ner, dictionary, llm], f)
            extractor_breakdown[f] = "regex" if regex.get(f) else "llm"
        elif f in semantic_fields:
            val = _pick([uie, llm, ner, dictionary], f)
            extractor_breakdown[f] = "uie+llm"
        else:
            val = _pick([llm, regex, ner, uie, dictionary], f)
            extractor_breakdown[f] = "llm"

        result[f] = val

        # 冲突检测
        vals = set()
        for src in [regex, ner, dictionary, llm, uie]:
            v = src.get(f)
            if v:
                if isinstance(v, list):
                    vals.add(tuple(sorted(v)))
                else:
                    vals.add(v)
        if len(vals) > 1:
            conflicts.append(f)

    result["extractor_breakdown"] = extractor_breakdown
    result["conflicts"] = conflicts
    return result


def _llm_first(regex, ner, dictionary, llm, uie, routing) -> dict:
    """LLM 优先：LLM 做主抽取，规则校验。"""
    result = {}
    extractor_breakdown = {}
    conflicts = []

    for f in FIELDS:
        llm_val = llm.get(f)
        rule_val = _pick([regex, ner, dictionary], f)

        if llm_val:
            result[f] = llm_val
            extractor_breakdown[f] = "llm"
            if rule_val and str(rule_val) != str(llm_val):
                conflicts.append(f)
        elif rule_val:
            result[f] = rule_val
            extractor_breakdown[f] = "rule"
        else:
            result[f] = _pick([uie], f)
            extractor_breakdown[f] = "uie"

    result["extractor_breakdown"] = extractor_breakdown
    result["conflicts"] = conflicts
    return result


def _field_level(regex, ner, dictionary, llm, uie, routing) -> dict:
    """字段级融合：按 routing 决策逐字段选工具。"""
    result = {}
    extractor_breakdown = {}
    conflicts = []

    tool_map = {
        "regex": regex,
        "ner": ner,
        "dictionary": dictionary,
        "llm": llm,
        "uie": uie,
    }

    for f in FIELDS:
        tools_str = routing.get(f, "llm")
        tool_names = [t.strip() for t in tools_str.split("+")]
        sources = [tool_map.get(t, {}) for t in tool_names if t in tool_map]
        sources.append(llm)  # fallback
        val = _pick(sources, f)
        result[f] = val
        extractor_breakdown[f] = tools_str

    result["extractor_breakdown"] = extractor_breakdown
    result["conflicts"] = conflicts
    return result
```

- [ ] **Step 2: 测试融合逻辑**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.fusion.merger import merge_results

r1 = {'salary': '20k-30k', 'education': '本科以上'}
r2 = {}
r3 = {'skills': ['Python', 'Java']}
r4 = {'job_title': 'Java工程师', 'salary': '15k-25k'}
r5 = {}
routing = {'salary': 'regex', 'education': 'regex', 'skills': 'dictionary+llm', 'job_title': 'uie+llm'}

result = merge_results(r1, r2, r3, r4, r5, routing)
for k, v in result.items():
    print(f'{k}: {v}')
"
```

预期: salary 取 regex 结果，skills 取 dictionary 结果，job_title 取 llm 结果。

---

### Task 14: 评估模块

**Files:**
- Create: `src/evaluation/evaluator.py`

- [ ] **Step 1: 编写 evaluator.py**

```python
"""评估模块：计算 Precision / Recall / F1。"""
import json
from src.state import FIELDS


def evaluate(
    predicted: list[dict],
    golden: list[dict],
) -> dict:
    """计算字段级和整体 Precision / Recall / F1。

    Args:
        predicted: [{"id": "zh_001", "job_title": "...", ...}, ...]
        golden:    [{"id": "zh_001", "labels": {...}}, ...]

    Returns:
        {"overall": {"precision": ..., "recall": ..., "f1": ...},
         "per_field": {"job_title": {...}, ...},
         "per_language": {"zh": {...}, "en": {...}}}
    """
    golden_map = {g["id"]: g["labels"] for g in golden}

    # 只评估有 golden 的文档
    matched = [p for p in predicted if p["id"] in golden_map]
    if not matched:
        return {"overall": {"precision": 0, "recall": 0, "f1": 0}}

    field_stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    lang_stats = {"zh": {"tp": 0, "fp": 0, "fn": 0}, "en": {"tp": 0, "fp": 0, "fn": 0}}

    for pred in matched:
        doc_id = pred["id"]
        labels = golden_map[doc_id]
        lang = "zh" if doc_id.startswith("zh") else "en"

        for f in FIELDS:
            pred_val = pred.get(f)
            gold_val = labels.get(f)

            pred_none = pred_val is None or pred_val == [] or pred_val == ""
            gold_none = gold_val is None or gold_val == [] or gold_val == ""

            if pred_none and gold_none:
                continue  # 都不存在，不计入
            elif pred_none and not gold_none:
                field_stats[f]["fn"] += 1
                lang_stats[lang]["fn"] += 1
            elif not pred_none and gold_none:
                field_stats[f]["fp"] += 1
                lang_stats[lang]["fp"] += 1
            else:
                # 比较值
                if isinstance(gold_val, list):
                    if isinstance(pred_val, list):
                        match = set(str(x).lower() for x in pred_val) == set(str(x).lower() for x in gold_val)
                    else:
                        match = False
                else:
                    match = str(pred_val).strip().lower() == str(gold_val).strip().lower()

                if match:
                    field_stats[f]["tp"] += 1
                    lang_stats[lang]["tp"] += 1
                else:
                    field_stats[f]["fp"] += 1
                    field_stats[f]["fn"] += 1
                    lang_stats[lang]["fp"] += 1
                    lang_stats[lang]["fn"] += 1

    def _calc(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    per_field = {}
    total_tp = total_fp = total_fn = 0
    for f in FIELDS:
        s = field_stats[f]
        per_field[f] = _calc(s["tp"], s["fp"], s["fn"])
        total_tp += s["tp"]
        total_fp += s["fp"]
        total_fn += s["fn"]

    per_lang = {}
    for lang in ["zh", "en"]:
        s = lang_stats[lang]
        per_lang[lang] = _calc(s["tp"], s["fp"], s["fn"])

    return {
        "overall": _calc(total_tp, total_fp, total_fn),
        "per_field": per_field,
        "per_language": per_lang,
    }


def save_evaluation_report(eval_result: dict, path: str = "data/output/evaluation_report.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
```

- [ ] **Step 2: 测试评估逻辑**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.evaluation.evaluator import evaluate

predicted = [
    {'id': 'zh_001', 'salary': '20k-30k/月', 'education': '本科及以上', 'skills': ['Python', 'Java']},
]
golden = [
    {'id': 'zh_001', 'labels': {'salary': '20k-30k/月', 'education': '本科及以上', 'skills': ['Java', 'Python']}},
]
result = evaluate(predicted, golden)
print(f'Overall: {result[\"overall\"]}')
print(f'Per field: {result[\"per_field\"]}')
"
```

预期: `Overall: {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}`

---

### Task 15: CLI 入口

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: 编写 main.py**

```python
"""CLI 入口：运行完整抽取流程并保存结果。"""
import json
import os
import sys
import yaml
from src.corpus.loader import load_corpus, load_golden_labels
from src.graph.builder import build_graph
from src.evaluation.evaluator import evaluate, save_evaluation_report


def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("错误: 找不到 config.yaml，请先创建配置文件。")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 加载语料
    print("加载语料...")
    docs = load_corpus(config["paths"]["corpus_dir"])
    if not docs:
        print("错误: 语料目录为空，请先运行 python -m src.corpus.generator")
        sys.exit(1)
    print(f"共 {len(docs)} 篇文档")

    # 构建图
    print("构建 LangGraph 工作流...")
    graph = build_graph()

    # 逐文档执行抽取
    print("开始抽取...")
    all_results = []
    for i, doc in enumerate(docs):
        initial_state = {
            "doc_id": doc.id,
            "language": doc.language,
            "title": doc.title,
            "raw_text": doc.text,
        }
        final_state = graph.invoke(initial_state)

        result = final_state.get("final_result", {})
        result["id"] = doc.id
        result["language"] = doc.language
        all_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(docs)} 篇")

    print("抽取完成！")

    # 保存结果
    merged_path = os.path.join(output_dir, "merged_results.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {merged_path}")

    # 评估
    golden = load_golden_labels(config["paths"].get("labels_dir", "data/labels") + "/golden_labels.json")
    if golden:
        print("运行评估...")
        eval_result = evaluate(all_results, golden)
        save_evaluation_report(eval_result, os.path.join(output_dir, "evaluation_report.json"))
        print(f"整体 F1: {eval_result['overall']['f1']:.4f}")
        for f, stats in eval_result["per_field"].items():
            print(f"  {f}: P={stats['precision']:.4f}, R={stats['recall']:.4f}, F1={stats['f1']:.4f}")
    else:
        print("未找到 golden_labels，跳过评估。")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 端到端测试（仅处理前 5 篇）**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
from src.corpus.loader import load_corpus
from src.graph.builder import build_graph

docs = load_corpus()[:5]
graph = build_graph()

for doc in docs:
    state = {'doc_id': doc.id, 'language': doc.language, 'title': doc.title, 'raw_text': doc.text}
    result = graph.invoke(state)
    final = result.get('final_result', {})
    print(f\"{doc.id}: job_title={final.get('job_title')}, salary={final.get('salary')}\")
print('E2E test passed!')
"
```

预期: 5 篇文档均输出抽取结果。

---

### Task 16: Streamlit Web 界面

**Files:**
- Create: `src/web/app.py`

- [ ] **Step 1: 编写 app.py**

```python
"""Streamlit Web 界面。"""
import json
import os
import sys

import streamlit as st
import yaml

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.corpus.loader import load_corpus, load_golden_labels
from src.corpus.generator import generate_corpus, generate_golden_labels
from src.graph.builder import build_graph
from src.evaluation.evaluator import evaluate


st.set_page_config(page_title="招聘信息抽取系统", layout="wide")
st.title("基于 LangGraph 的中英双语招聘信息抽取实验系统")


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- 侧边栏 ----
st.sidebar.header("操作面板")

if st.sidebar.button("生成模拟语料"):
    config = load_config()
    total = config["corpus"]["total_docs"]
    zh_ratio = config["corpus"]["zh_ratio"]
    with st.spinner(f"正在生成 {total} 篇语料..."):
        metadata = generate_corpus(total=total, zh_ratio=zh_ratio)
        generate_golden_labels(metadata, eval_sample=config["corpus"]["eval_sample"])
    st.sidebar.success(f"已生成 {len(metadata)} 篇语料 + golden_labels")

if st.sidebar.button("运行抽取流程"):
    config = load_config()
    docs = load_corpus()
    if not docs:
        st.sidebar.error("请先生成语料")
    else:
        graph = build_graph()
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()

        all_results = []
        for i, doc in enumerate(docs):
            state = {"doc_id": doc.id, "language": doc.language, "title": doc.title, "raw_text": doc.text}
            final_state = graph.invoke(state)
            result = final_state.get("final_result", {})
            result["id"] = doc.id
            result["language"] = doc.language
            all_results.append(result)
            progress.progress((i + 1) / len(docs))
            status.text(f"处理中: {i + 1}/{len(docs)}")

        os.makedirs(config["paths"]["output_dir"], exist_ok=True)
        with open(os.path.join(config["paths"]["output_dir"], "merged_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        st.session_state["results"] = all_results
        st.session_state["docs"] = docs
        st.sidebar.success(f"抽取完成！共 {len(all_results)} 篇")

# ---- 主区域 ----
tab1, tab2, tab3, tab4 = st.tabs(["语料浏览", "抽取结果", "规则 vs LLM 对比", "评估报告"])

config = load_config()

with tab1:
    st.header("语料浏览")
    docs = load_corpus()
    if docs:
        doc_ids = [d.id for d in docs]
        selected = st.selectbox("选择文档", doc_ids)
        doc = [d for d in docs if d.id == selected][0]
        st.subheader(f"{doc.id} — {doc.title}")
        st.text_area("原文", doc.text, height=400)
    else:
        st.info("暂无语料，请点击侧边栏「生成模拟语料」")

with tab2:
    st.header("抽取结果")

    results_path = os.path.join(config["paths"]["output_dir"], "merged_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        doc_id = st.selectbox("选择文档查看结果", [r["id"] for r in all_results])
        result = [r for r in all_results if r["id"] == doc_id][0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("抽取字段")
            for field in ["job_title", "company_name", "work_location", "salary", "education", "experience"]:
                val = result.get(field)
                st.metric(field, val if val else "未抽取到")
        with col2:
            st.subheader("技能要求")
            skills = result.get("skills", [])
            if skills:
                for s in skills:
                    st.tag(f"label: {s}", body=s)
            else:
                st.text("未抽取到技能")

        if result.get("extractor_breakdown"):
            st.subheader("各字段使用的抽取器")
            st.json(result["extractor_breakdown"])
    else:
        st.info("暂无结果，请先运行抽取流程")

with tab3:
    st.header("规则 vs LLM 对比")

    results_path = os.path.join(config["paths"]["output_dir"], "merged_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        # 简化对比表
        import pandas as pd
        rows = []
        for r in all_results[:20]:
            rows.append({
                "doc_id": r["id"],
                "language": r.get("language", ""),
                "job_title": r.get("job_title", "-"),
                "company_name": r.get("company_name", "-"),
                "salary": r.get("salary", "-"),
                "skills_count": len(r.get("skills") or []),
                "conflicts": ", ".join(r.get("conflicts", [])),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("暂无结果")

with tab4:
    st.header("评估报告")

    report_path = os.path.join(config["paths"]["output_dir"], "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        st.subheader("整体指标")
        col1, col2, col3 = st.columns(3)
        overall = report["overall"]
        col1.metric("Precision", f"{overall['precision']:.2%}")
        col2.metric("Recall", f"{overall['recall']:.2%}")
        col3.metric("F1", f"{overall['f1']:.2%}")

        st.subheader("分字段指标")
        import pandas as pd
        field_rows = []
        for f, stats in report.get("per_field", {}).items():
            field_rows.append({
                "字段": f,
                "Precision": f"{stats['precision']:.2%}",
                "Recall": f"{stats['recall']:.2%}",
                "F1": f"{stats['f1']:.2%}",
            })
        st.dataframe(pd.DataFrame(field_rows), use_container_width=True)

        st.subheader("分语言指标")
        lang_rows = []
        for lang, stats in report.get("per_language", {}).items():
            lang_rows.append({
                "语言": lang,
                "Precision": f"{stats['precision']:.2%}",
                "Recall": f"{stats['recall']:.2%}",
                "F1": f"{stats['f1']:.2%}",
            })
        st.dataframe(pd.DataFrame(lang_rows), use_container_width=True)
    else:
        st.info("暂无评估报告，请先运行抽取流程")
```

- [ ] **Step 2: 验证 Streamlit 能启动**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && streamlit run src/web/app.py --server.port 8501
```

预期: 浏览器打开 http://localhost:8501 可见界面。

---

### Task 17: 最终验证 — 全流程跑通

- [ ] **Step 1: 从头跑通**

```bash
cd "e:/大三下资料/infomation_acquire/ex3" && python -c "
# 1. 生成语料
from src.corpus.generator import generate_corpus, generate_golden_labels
metadata = generate_corpus(total=150, zh_ratio=0.6)
golden = generate_golden_labels(metadata, eval_sample=25)
print(f'[OK] 语料: {len(metadata)} 篇, golden: {len(golden)} 篇')

# 2. 加载语料
from src.corpus.loader import load_corpus, load_golden_labels
docs = load_corpus()
print(f'[OK] 加载: {len(docs)} 篇文档')

# 3. 构建图
from src.graph.builder import build_graph
graph = build_graph()
print(f'[OK] 图构建成功')

# 4. 跑前 3 篇
for doc in docs[:3]:
    state = {'doc_id': doc.id, 'language': doc.language, 'title': doc.title, 'raw_text': doc.text}
    result = graph.invoke(state)
    final = result.get('final_result', {})
    print(f\"  {doc.id}: title={final.get('job_title')}, salary={final.get('salary')}\")

print('[OK] 全流程验证通过！')
"
```

预期: 所有步骤输出 `[OK]`。
