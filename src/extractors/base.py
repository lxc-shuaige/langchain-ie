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
