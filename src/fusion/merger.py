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


def _which_extractors(regex, ner, dictionary, llm, uie, field: str) -> list[str]:
    """返回哪些抽取器对该字段有非空输出。"""
    sources = {
        "regex": regex,
        "ner": ner,
        "dictionary": dictionary,
        "llm": llm,
        "uie": uie,
    }
    return [name for name, src in sources.items() if src.get(field)]


def _merge_skills(regex, ner, dictionary, llm, uie) -> tuple[list[str] | None, str]:
    """合并所有抽取器的技能列表（去重并集），返回 (技能列表, 来源标签)。"""
    all_skills = set()
    contributors = []
    for name, src in [("dictionary", dictionary), ("llm", llm), ("uie", uie), ("ner", ner)]:
        skills = src.get("skills")
        if skills:
            all_skills.update(skills)
            contributors.append(name)
    if not all_skills:
        return None, "none"
    return sorted(all_skills), "+".join(contributors)


def _rule_first(regex, ner, dictionary, llm, uie, routing) -> dict:
    """规则优先：格式化字段用规则，语义字段用 LLM/UIE。"""
    result = {}
    extractor_breakdown = {}
    conflicts = []

    rule_fields = {"salary", "education", "experience", "work_location", "contact_info"}
    semantic_fields = {"job_title", "company_name"}

    for f in FIELDS:
        if f in rule_fields:
            val = _pick([regex, ner, dictionary, llm], f)
            contributors = _which_extractors(regex, ner, dictionary, llm, uie, f)
            extractor_breakdown[f] = "+".join(contributors) if contributors else "none"
        elif f in semantic_fields:
            val = _pick([uie, llm, ner, dictionary], f)
            contributors = _which_extractors(regex, ner, dictionary, llm, uie, f)
            extractor_breakdown[f] = "+".join(contributors) if contributors else "none"
        elif f == "skills":
            val, label = _merge_skills(regex, ner, dictionary, llm, uie)
            extractor_breakdown[f] = label
        else:
            val = _pick([llm, regex, ner, uie, dictionary], f)
            contributors = _which_extractors(regex, ner, dictionary, llm, uie, f)
            extractor_breakdown[f] = "+".join(contributors) if contributors else "none"

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

        if f == "skills":
            val, label = _merge_skills(regex, ner, dictionary, llm, uie)
            result[f] = val
            extractor_breakdown[f] = label
        elif llm_val:
            result[f] = llm_val
            extractor_breakdown[f] = "llm"
            if rule_val and str(rule_val) != str(llm_val):
                conflicts.append(f)
        elif rule_val:
            result[f] = rule_val
            contributors = _which_extractors(regex, ner, dictionary, llm, uie, f)
            extractor_breakdown[f] = "+".join(contributors)
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
        if f == "skills":
            val, label = _merge_skills(regex, ner, dictionary, llm, uie)
            result[f] = val
            extractor_breakdown[f] = label
        else:
            tools_str = routing.get(f, "llm")
            tool_names = [t.strip() for t in tools_str.split("+")]
            sources = [tool_map.get(t, {}) for t in tool_names if t in tool_map]
            sources.append(llm)  # fallback
            val = _pick(sources, f)
            result[f] = val
            contributors = _which_extractors(regex, ner, dictionary, llm, uie, f)
            extractor_breakdown[f] = "+".join(contributors) if contributors else tools_str

    result["extractor_breakdown"] = extractor_breakdown
    result["conflicts"] = conflicts
    return result
