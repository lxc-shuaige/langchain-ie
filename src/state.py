from typing import TypedDict


FIELDS = [
    "job_title",
    "company_name",
    "work_location",
    "salary",
    "education",
    "experience",
    "skills",
    "contact_info",
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
    # 多媒体
    input_type: str
    image_path: str
    ocr_text: str
