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
