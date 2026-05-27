# Algorithm Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement four algorithm optimizations: true intelligent routing, Jaccard-based skills evaluation, weighted voting fusion strategy, and parallel document processing.

**Architecture:** Four independent optimization modules touching extractors (routing prompt), evaluation (Jaccard), fusion (weighted voting), and main pipeline (parallelism). Each optimization is self-contained and can be implemented and tested independently.

**Tech Stack:** Python 3.10, LangGraph, DeepSeek API, concurrent.futures

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `src/extractors/llm_extractor.py` | Modify | Add `route()` method with lightweight routing prompt |
| `src/graph/nodes.py` | Modify | Rewrite `analyze_node` to use routing-only call; update `llm_extract_node` |
| `src/evaluation/evaluator.py` | Modify | Add Jaccard similarity for list-type fields |
| `src/fusion/merger.py` | Modify | Add `_weighted_vote()` strategy function |
| `config.yaml` | Modify | Add `confidence_weights` and `processing` config blocks |
| `src/main.py` | Modify | ThreadPoolExecutor for document-level parallelism |

---

### Task 1: Add routing-only method to LLMExtractor

**Files:**
- Modify: `src/extractors/llm_extractor.py`

- [ ] **Step 1: Add routing system prompts and `route()` method**

Add after the existing `SYSTEM_PROMPT_EN` (after line 47):

```python
ROUTE_PROMPT_ZH = """你是一个信息抽取路由器。分析以下招聘文本的特征，决定每个字段最适合使用哪些抽取工具。

工具说明：
- regex: 正则表达式，适合格式固定的字段（薪资数字、学历关键词、手机号/邮箱）
- ner: 命名实体识别，适合公司名、地点、岗位名等实体
- dictionary: 技能词典匹配，适合技术关键词
- llm: 大模型语义理解，兜底工具，适合语义模糊或非标准表达

字段说明：
- salary: 薪资范围
- education: 学历要求
- experience: 工作经验
- work_location: 工作地点
- contact_info: 联系方式（电话/邮箱）
- job_title: 岗位名称
- company_name: 公司名称
- skills: 技能要求

返回规则：
1. 对每个字段返回用"+"连接的优先级工具列表，如 "regex" 或 "regex+llm"
2. 格式规整的数值/关键词字段优先用 regex
3. 命名实体优先用 ner
4. 技能优先用 dictionary
5. 仅当文本语义模糊、隐式表达或中英混合时才加入 llm
6. 只返回 JSON，不要包含其他文字。
7. skills 字段始终包含 dictionary。

返回格式示例：
{"salary":"regex","education":"regex","experience":"regex","work_location":"regex+ner","contact_info":"regex","job_title":"ner+llm","company_name":"ner+llm","skills":"dictionary+llm"}"""

ROUTE_PROMPT_EN = """You are an information extraction router. Analyze the following job posting text and decide which extraction tools are best suited for each field.

Tools:
- regex: regular expressions, best for formatted fields (salary numbers, degree keywords, phone/email)
- ner: named entity recognition, best for company names, locations, job titles
- dictionary: skill keyword matching, best for technical skills
- llm: LLM semantic understanding, fallback for ambiguous or non-standard expressions

Fields: salary, education, experience, work_location, contact_info, job_title, company_name, skills

Rules:
1. Return a priority tool list joined by "+" for each field, e.g. "regex" or "regex+llm"
2. Formatted numeric/keyword fields → prefer regex
3. Named entities → prefer ner
4. Skills → always include dictionary
5. Only include llm when text is semantically ambiguous, implicit, or mixed-language
6. Return only JSON, no other text.
7. skills field always includes dictionary.

Example format:
{"salary":"regex","education":"regex","experience":"regex","work_location":"regex+ner","contact_info":"regex","job_title":"ner+llm","company_name":"ner+llm","skills":"dictionary+llm"}"""
```

Then add the `route()` method to the `LLMExtractor` class, after `__init__` (after line 61):

```python
    def route(self, text: str, language: str) -> dict:
        """轻量路由分析：只分析文档特征，返回 routing JSON，不做全字段抽取。"""
        system_prompt = ROUTE_PROMPT_ZH if language == "zh" else ROUTE_PROMPT_EN
        user_text = text[:1500]  # 前 1500 字符足够判断文档特征

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"招聘文本：\n{user_text}" if language == "zh" else f"Job posting text:\n{user_text}"),
        ]

        response = self.model.invoke(messages)
        content = response.content.strip()

        if content.startswith("```"):
            content = content.split("\n", 1)[-1]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            if content.startswith("json"):
                content = content[4:].strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {
                "salary": "regex",
                "education": "regex",
                "experience": "regex",
                "work_location": "regex+llm",
                "contact_info": "regex",
                "job_title": "llm+regex",
                "company_name": "llm+regex",
                "skills": "dictionary+llm",
            }
```

- [ ] **Step 2: Verify the file is syntactically valid**

Run: `python -c "import ast; ast.parse(open('src/extractors/llm_extractor.py').read()); print('OK')"`

---

### Task 2: Rewrite analyze_node and update llm_extract_node

**Files:**
- Modify: `src/graph/nodes.py`

- [ ] **Step 1: Rewrite `analyze_node` to use routing-only LLM call**

Replace the existing `analyze_node` function (lines 76-99):

```python
def analyze_node(state: ExtractionState) -> dict:
    """分析节点：轻量路由分析，LLM 只判断文档特征不抽取字段。"""
    llm = _get_extractor("llm")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")

    routing = llm.route(text, language)

    return {"routing_decision": routing}
```

- [ ] **Step 2: Update `llm_extract_node` to always do full extraction (remove stale result check)**

Replace the existing `llm_extract_node` function (lines 126-134):

```python
def llm_extract_node(state: ExtractionState) -> dict:
    """LLM 全字段抽取节点：仅当路由激活时才被调用。"""
    llm = _get_extractor("llm")
    language = state.get("language", "zh")
    text = state.get("cleaned_text", "")
    result = llm.extract(text, language)
    return {"llm_result": result}
```

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/graph/nodes.py').read()); print('OK')"`

---

### Task 3: Jaccard-based skills evaluation

**Files:**
- Modify: `src/evaluation/evaluator.py`

- [ ] **Step 1: Add Jaccard helper and update list-field matching logic**

Replace lines 40-56 (the `else` block in `evaluate()`) with:

```python
            else:
                if isinstance(gold_val, list):
                    if isinstance(pred_val, list):
                        pred_set = set(str(x).lower() for x in pred_val)
                        gold_set = set(str(x).lower() for x in gold_val)
                        if not gold_set:
                            match = False
                        else:
                            intersection = pred_set & gold_set
                            union = pred_set | gold_set
                            jaccard = len(intersection) / len(union)
                            match = jaccard >= 0.5
                    else:
                        match = False
                else:
                    match = str(pred_val).strip().lower() == str(gold_val).strip().lower()
```

- [ ] **Step 2: Add Jaccard details to evaluation output**

After line 76 (`per_lang` section), before the `return` statement, add a `per_doc_jaccard` section. In the main loop (after line 55 `lang_stats[lang]["fn"] += 1` for the non-match case), track per-document Jaccard for skills. Add this after the `per_lang` block (after line 76):

```python
    doc_jaccard = {}
    for pred in matched:
        doc_id = pred["id"]
        gold_labels = golden_map[doc_id]
        pred_skills = pred.get("skills")
        gold_skills = gold_labels.get("skills")
        if isinstance(pred_skills, list) and isinstance(gold_skills, list) and gold_skills:
            pred_set = set(str(x).lower() for x in pred_skills)
            gold_set = set(str(x).lower() for x in gold_skills)
            inter = pred_set & gold_set
            union = pred_set | gold_set
            doc_jaccard[doc_id] = round(len(inter) / len(union), 4) if union else 0.0

    return {
        "overall": _calc(total_tp, total_fp, total_fn),
        "per_field": per_field,
        "per_language": per_lang,
        "skills_jaccard": doc_jaccard,
    }
```

Update the existing `return` statement (lines 78-82) to match the above.

- [ ] **Step 3: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/evaluation/evaluator.py').read()); print('OK')"`

---

### Task 4: Weighted voting fusion strategy

**Files:**
- Modify: `src/fusion/merger.py`
- Modify: `config.yaml`

- [ ] **Step 1: Add config.yaml confidence_weights block**

Insert after `fusion_strategy: "rule_first"` (after line 23):

```yaml
# 加权投票置信度权重
confidence_weights:
  # 格式化字段：薪资、学历、经验、联系方式
  formatted:
    regex: 0.9
    dictionary: 0.0
    ner: 0.3
    uie: 0.5
    llm: 0.6
  # 实体字段：公司名、地点、岗位名
  entity:
    regex: 0.3
    dictionary: 0.0
    ner: 0.7
    uie: 0.6
    llm: 0.7
  # 列表字段：技能
  list_field:
    regex: 0.0
    dictionary: 0.8
    ner: 0.0
    uie: 0.5
    llm: 0.6
```

- [ ] **Step 2: Refactor config loading and add `_load_weights()` + `_weighted_vote()` to merger.py**

First, replace `_load_strategy()` (lines 6-9) to use a shared config loader:

```python
def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_strategy() -> str:
    return _load_config().get("fusion_strategy", "rule_first")


def _load_weights() -> dict:
    config = _load_config()
    return config.get("confidence_weights", {})


def _field_type(field: str) -> str:
    """返回字段的类型分类。"""
    if field in ("salary", "education", "experience", "contact_info"):
        return "formatted"
    elif field in ("job_title", "company_name", "work_location"):
        return "entity"
    elif field == "skills":
        return "list_field"
    return "formatted"


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
    elif strategy == "weighted_vote":
        return _weighted_vote(regex, ner, dictionary, llm, uie, routing)
    else:
        return _field_level(regex, ner, dictionary, llm, uie, routing)
```

- [ ] **Step 3: Add `_weighted_vote()` function to merger.py**

Add before the final line of the file:

```python
def _weighted_vote(regex, ner, dictionary, llm, uie, routing) -> dict:
    """加权投票融合：按字段类型给各抽取器分配置信度权重。"""
    weights = _load_weights()
    result = {}
    extractor_breakdown = {}
    conflicts = []
    confidence = {}

    tool_map = {
        "regex": regex,
        "ner": ner,
        "dictionary": dictionary,
        "llm": llm,
        "uie": uie,
    }

    for f in FIELDS:
        ftype = _field_type(f)
        w = weights.get(ftype, {})

        if f == "skills":
            val, label = _merge_skills(regex, ner, dictionary, llm, uie)
            result[f] = val
            extractor_breakdown[f] = label
            # skills confidence: 按贡献源数量估计
            contributor_count = len(label.split("+")) if label != "none" else 0
            confidence[f] = min(1.0, contributor_count / 3.0)
        else:
            # 按权重排各工具的输出，取最高权重的非空值
            scored = []
            for name, src in tool_map.items():
                v = src.get(f)
                if v:
                    scored.append((w.get(name, 0.1), name, v))
            scored.sort(key=lambda x: x[0], reverse=True)

            if scored:
                best_weight, best_name, best_val = scored[0]
                result[f] = best_val
                confidence[f] = best_weight
                contributors = _which_extractors(regex, ner, dictionary, llm, uie, f)
                extractor_breakdown[f] = "+".join(contributors) if contributors else best_name
            else:
                result[f] = None
                extractor_breakdown[f] = "none"
                confidence[f] = 0.0

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
    result["confidence"] = confidence
    return result
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/fusion/merger.py').read()); print('OK')"`

- [ ] **Step 5: Verify config.yaml is valid YAML**

Run: `python -c "import yaml; yaml.safe_load(open('config.yaml')); print('OK')"`

---

### Task 5: Document-level parallel processing

**Files:**
- Modify: `src/main.py`

- [ ] **Step 1: Add parallel processing to `_run_text_pipeline`**

Replace the existing `_run_text_pipeline` function (lines 22-49):

```python
def _run_text_pipeline(config: dict) -> list[dict]:
    """运行纯文本抽取流程，返回结果列表。"""
    import concurrent.futures
    import os as _os

    docs = load_corpus(config["paths"]["corpus_dir"])
    if not docs:
        print("错误: 语料目录为空，请先运行 python -m src.corpus.generator")
        return []

    print(f"共 {len(docs)} 篇文本文档")

    max_workers = config.get("processing", {}).get("max_workers", min(8, (_os.cpu_count() or 1) * 2))

    def _process_one(doc):
        graph = build_graph()
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
        return result

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, doc): i for i, doc in enumerate(docs)}
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())
            if (len(all_results)) % 10 == 0:
                print(f"  已处理文本 {len(all_results)}/{len(docs)} 篇")

    return all_results
```

- [ ] **Step 2: Add parallel processing to `_run_image_pipeline`**

Replace the existing `_run_image_pipeline` function (lines 52-82):

```python
def _run_image_pipeline(config: dict) -> list[dict]:
    """运行图片抽取流程，返回结果列表。"""
    import concurrent.futures
    import os as _os

    image_dir = config["paths"].get("image_dir", "data/images")
    docs = load_image_corpus(image_dir)
    if not docs:
        print("错误: 图片目录为空，请先运行 python -m src.corpus.image_generator")
        return []

    print(f"共 {len(docs)} 张图片")

    max_workers = config.get("processing", {}).get("max_workers", min(8, (_os.cpu_count() or 1) * 2))

    def _process_one(doc):
        graph = build_graph()
        initial_state = {
            "doc_id": doc.id,
            "language": doc.language,
            "title": doc.title,
            "raw_text": doc.text,
            "input_type": "image",
            "image_path": doc.image_path,
        }
        final_state = graph.invoke(initial_state)
        result = final_state.get("final_result", {})
        result["id"] = doc.id
        result["language"] = doc.language
        return result

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, doc): i for i, doc in enumerate(docs)}
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())
            if (len(all_results)) % 5 == 0:
                print(f"  已处理图片 {len(all_results)}/{len(docs)} 张")

    return all_results
```

- [ ] **Step 3: Add processing config to config.yaml**

Insert at end of file:

```yaml
# 处理配置
processing:
  max_workers: 4
```

- [ ] **Step 4: Verify syntax**

Run: `python -c "import ast; ast.parse(open('src/main.py').read()); print('OK')"`

---

### Task 6: Full integration test

**Files:**
- None (test only)

- [ ] **Step 1: Run text mode and verify all four optimizations work end-to-end**

Run: `python src/main.py text`

Expected:
- LLM routing calls happen (lighter than before)
- Results saved to `data/output/merged_results.json`
- Evaluation report includes `skills_jaccard` field
- Processing shows parallel progress

- [ ] **Step 2: Verify weighted_vote strategy works**

Change `config.yaml` `fusion_strategy` to `"weighted_vote"`, then:
Run: `python src/main.py text`

Expected:
- Results include `confidence` field per document
- `extractor_breakdown` reflects weighted choices

- [ ] **Step 3: Restore default config and commit**

Change `fusion_strategy` back to `"rule_first"`.

```bash
git add src/extractors/llm_extractor.py src/graph/nodes.py src/evaluation/evaluator.py src/fusion/merger.py src/main.py config.yaml
git commit -m "feat: four algorithm optimizations — true routing, Jaccard eval, weighted voting, parallel processing

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```
