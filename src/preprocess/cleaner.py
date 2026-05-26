import re


def clean_text(text: str) -> str:
    """清洗文本：去 HTML 标签、多余空白、特殊字符。"""
    # 去 HTML 标签
    text = re.sub(r"<[^>]+>", "", text)
    # 将多个空白符（含 \r\n）替换为单个换行
    text = re.sub(r"[ \t]+", " ", text)
    # 合并多余空行为单个空行
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 去掉首尾空白
    text = text.strip()
    return text
