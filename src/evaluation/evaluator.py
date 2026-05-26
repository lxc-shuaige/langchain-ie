"""评估模块：计算 Precision / Recall / F1。"""
import json
from src.state import FIELDS


def evaluate(
    predicted: list[dict],
    golden: list[dict],
) -> dict:
    """计算字段级和整体 Precision / Recall / F1。"""
    golden_map = {g["id"]: g["labels"] for g in golden}

    matched = [p for p in predicted if p["id"] in golden_map]
    if not matched:
        return {"overall": {"precision": 0, "recall": 0, "f1": 0}}

    field_stats = {f: {"tp": 0, "fp": 0, "fn": 0} for f in FIELDS}
    lang_stats = {"zh": {"tp": 0, "fp": 0, "fn": 0}, "en": {"tp": 0, "fp": 0, "fn": 0}}

    for pred in matched:
        doc_id = pred["id"]
        labels = golden_map[doc_id]
        lang = "zh" if (doc_id.startswith("zh") or doc_id.startswith("img")) else "en"

        for f in FIELDS:
            pred_val = pred.get(f)
            gold_val = labels.get(f)

            pred_none = pred_val is None or pred_val == [] or pred_val == ""
            gold_none = gold_val is None or gold_val == [] or gold_val == ""

            if pred_none and gold_none:
                continue
            elif pred_none and not gold_none:
                field_stats[f]["fn"] += 1
                lang_stats[lang]["fn"] += 1
            elif not pred_none and gold_none:
                field_stats[f]["fp"] += 1
                lang_stats[lang]["fp"] += 1
            else:
                if isinstance(gold_val, list):
                    if isinstance(pred_val, list):
                        match = set(str(x).lower() for x in pred_val) == set(str(x).lower() for x in gold_val)
                    else:
                        match = False
                else:
                    match = str(pred_val).strip().lower() == str(gold_val).strip().lower()

                if match:
                    field_stats[f]["tp"] += 1
                    lang_stats[lang]["tp"] += 1
                else:
                    field_stats[f]["fp"] += 1
                    field_stats[f]["fn"] += 1
                    lang_stats[lang]["fp"] += 1
                    lang_stats[lang]["fn"] += 1

    def _calc(tp, fp, fn):
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}

    per_field = {}
    total_tp = total_fp = total_fn = 0
    for f in FIELDS:
        s = field_stats[f]
        per_field[f] = _calc(s["tp"], s["fp"], s["fn"])
        total_tp += s["tp"]
        total_fp += s["fp"]
        total_fn += s["fn"]

    per_lang = {}
    for lang in ["zh", "en"]:
        s = lang_stats[lang]
        per_lang[lang] = _calc(s["tp"], s["fp"], s["fn"])

    return {
        "overall": _calc(total_tp, total_fp, total_fn),
        "per_field": per_field,
        "per_language": per_lang,
    }


def save_evaluation_report(eval_result: dict, path: str = "data/output/evaluation_report.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=2)
