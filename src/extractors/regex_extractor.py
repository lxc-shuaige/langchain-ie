import re
from src.extractors.base import BaseExtractor, empty_result


class RegexExtractor(BaseExtractor):
    name = "regex"

    # ---- 中文模式 ----
    ZH_PATTERNS = {
        "salary": [
            r"(\d{1,2}[kK]\s*[-~]\s*\d{1,2}[kK]\s*/\s*月)",
            r"(年薪\s*\d{1,3}\s*万\s*[-~]\s*\d{1,3}\s*万)",
            r"(年薪\s*\d{1,3}\s*万起?)",
            r"(\d{1,2}\s*[-~]\s*\d{1,2}\s*[kK]\s*/\s*月)",
            r"(月薪\s*\d{1,2}\s*[-~]\s*\d{1,2}\s*[kK])",
            r"(薪资[：:]?\s*\d{1,2}[kK]\s*[-~]\s*\d{1,2}[kK])",
            r"(面议|薪资面谈|薪资面议)",
        ],
        "education": [
            r"(大专|本科|硕士|博士)(及以上|及以上学历|优先)?",
            r"(博士|硕士|本科|大专)学历",
        ],
        "experience": [
            r"(\d+\s*[-~]?\s*\d*\s*年\s*(以上\s*)?(相关)?(工作)?经验)",
            r"(\d+\+?\s*年\s*以\s*上?\s*(相关)?(工作)?经验)",
            r"(应届[生生]|应届毕业生)",
        ],
        "work_location": [
            r"(工作地点[：:]\s*[一-鿿]+)",
            r"([一-鿿]{2,3}[市省区县])",
        ],
        "contact_info": [
            r"(1[3-9]\d{9})",
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"(电话[：:]\s*1[3-9]\d{9})",
            r"(邮箱[：:]\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"(联系电话[：:]\s*\d[\d\s-]{7,15})",
        ],
    }

    # ---- 英文模式 ----
    EN_PATTERNS = {
        "salary": [
            r"(\$\d{2,3}[kK]\s*[-~]\s*\$\d{2,3}[kK]\s*(annually|per year)?)",
            r"(\$\d{2,3}[kK]\s*[-~]\s*\$\d{2,3}[kK])",
            r"(\d{2,3}[kK]\s*[-~]\s*\d{2,3}[kK]\s*(CNY|USD|EUR)?)",
        ],
        "education": [
            r"(Bachelor'?s?\s*degree\s*(required|preferred)?)",
            r"(Master'?s?\s*degree\s*(required|preferred|in\s+\w+)?)",
            r"(PhD\s*(in\s+\w+)?\s*(required|preferred)?)",
            r"(MBA\s*(preferred)?)",
        ],
        "experience": [
            r"(\d+\+?\s*years?\s*(of\s*)?(related\s*)?experience)",
            r"(\d+\s*[-~]\s*\d+\s*years?\s*(of\s*)?(related\s*)?experience)",
            r"(Entry\s*level)",
        ],
        "work_location": [
            r"(Location[：:]\s*[\w\s,]+)",
            r"(Location[：:]\s*Remote)",
        ],
        "contact_info": [
            r"([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
            r"(\+?\d[\d\s-]{7,15})",
            r"(Phone[：:]\s*\+?\d[\d\s-]{7,15})",
            r"(Email[：:]\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})",
        ],
    }

    def extract(self, text: str, language: str) -> dict:
        patterns = self.ZH_PATTERNS if language == "zh" else self.EN_PATTERNS
        result = empty_result()

        for field, regex_list in patterns.items():
            for regex in regex_list:
                match = re.search(regex, text, re.IGNORECASE)
                if match:
                    result[field] = match.group(0).strip()
                    break

        return result
