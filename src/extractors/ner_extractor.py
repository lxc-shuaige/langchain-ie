import yaml
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from src.extractors.base import BaseExtractor, empty_result


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class NERExtractor(BaseExtractor):
    name = "ner"

    def __init__(self):
        config = _load_config()
        model_name = config["ner"]["model"]
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForTokenClassification.from_pretrained(model_name)
        self._pipeline = pipeline(
            "token-classification",
            model=self._model,
            tokenizer=self._tokenizer,
            aggregation_strategy="simple",
        )

    def extract(self, text: str, language: str) -> dict:
        if language == "en":
            return empty_result()

        entities = self._pipeline(text[:2000])

        result = empty_result()
        for ent in entities:
            label = ent["entity_group"]
            word = ent["word"]
            if label == "company" and not result["company_name"]:
                result["company_name"] = word
            elif label == "address" and not result["work_location"]:
                result["work_location"] = word
            elif label == "position" and not result["job_title"]:
                result["job_title"] = word

        return result
