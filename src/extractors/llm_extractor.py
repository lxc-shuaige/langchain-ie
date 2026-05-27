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
- contact_info: 联系电话或邮箱

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
- contact_info: contact phone or email

Rules:
1. If a field is not present in the text, set its value to null.
2. Keep the original wording, do not fabricate information.
3. skills must be an array.
4. Return only JSON, no other text."""

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

    def route(self, text: str, language: str) -> dict:
        """轻量路由分析：只分析文档特征，返回 routing JSON，不做全字段抽取。"""
        system_prompt = ROUTE_PROMPT_ZH if language == "zh" else ROUTE_PROMPT_EN
        user_text = text[:1500]

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
