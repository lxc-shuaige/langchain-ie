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

    full_result = llm.extract(text, language)

    routing = {
        "salary": "regex",
        "education": "regex",
        "experience": "regex",
        "work_location": "regex+llm",
        "company_name": "llm+regex",
        "job_title": "llm+regex",
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
