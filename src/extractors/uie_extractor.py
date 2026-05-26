import yaml
from paddlenlp import Taskflow
from src.extractors.base import BaseExtractor, empty_result


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class UIEExtractor(BaseExtractor):
    name = "uie"

    def __init__(self):
        config = _load_config()
        schema = config["uie"]["schema"]
        self._uie = Taskflow("information_extraction", schema=schema, model="uie-base")

    def extract(self, text: str, language: str) -> dict:
        if language == "en":
            pass

        try:
            outputs = self._uie(text[:2000])
        except Exception:
            return empty_result()

        result = empty_result()

        schema_map = {
            "岗位名称": "job_title",
            "公司名称": "company_name",
            "工作地点": "work_location",
            "薪资": "salary",
            "学历要求": "education",
            "工作经验": "experience",
            "技能要求": "skills",
        }

        for item in outputs:
            for key, value in item.items():
                field = schema_map.get(key)
                if field and not result[field]:
                    text_val = value[0]["text"] if isinstance(value, list) and value else str(value)
                    result[field] = text_val

        return result
