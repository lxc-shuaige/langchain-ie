"""LangGraph StateGraph 构建与编译。"""
from langgraph.graph import StateGraph, END
from src.state import ExtractionState
from src.graph.nodes import (
    preprocess_node,
    analyze_node,
    ocr_node,
    regex_node,
    ner_node,
    dictionary_node,
    llm_extract_node,
    uie_node,
    fusion_node,
)
from src.graph.router import route_after_analyze, route_after_preprocess


def build_graph() -> StateGraph:
    """构建并编译抽取工作流图。"""
    builder = StateGraph(ExtractionState)

    # 添加节点
    builder.add_node("preprocess", preprocess_node)
    builder.add_node("analyze", analyze_node)
    builder.add_node("ocr", ocr_node)
    builder.add_node("regex", regex_node)
    builder.add_node("ner", ner_node)
    builder.add_node("dictionary", dictionary_node)
    builder.add_node("llm_extract", llm_extract_node)
    builder.add_node("uie", uie_node)
    builder.add_node("fusion", fusion_node)

    # 设置入口
    builder.set_entry_point("preprocess")

    # 条件边: preprocess → ocr (image) 或 analyze (text)
    builder.add_conditional_edges(
        "preprocess",
        route_after_preprocess,
        {
            "ocr": "ocr",
            "analyze": "analyze",
        },
    )

    # ocr → analyze
    builder.add_edge("ocr", "analyze")

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
