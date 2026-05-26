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
