"""条件路由逻辑：根据 routing_decision 决定激活哪些节点。"""


def route_after_preprocess(state: dict) -> str:
    """图片输入走 OCR 节点，文本输入直通 analyze。"""
    if state.get("input_type") == "image":
        return "ocr"
    return "analyze"


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
