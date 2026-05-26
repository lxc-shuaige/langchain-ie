import json
import os
from dataclasses import dataclass


@dataclass
class DocumentRecord:
    id: str
    language: str
    title: str
    text: str


def load_corpus(corpus_dir: str = "data/corpus") -> list[DocumentRecord]:
    """扫描语料目录，返回所有 DocumentRecord 列表。"""
    docs = []
    if not os.path.isdir(corpus_dir):
        return docs

    for filename in sorted(os.listdir(corpus_dir)):
        if not filename.endswith(".txt"):
            continue
        filepath = os.path.join(corpus_dir, filename)
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        doc_id = filename.replace(".txt", "")
        language = "zh" if doc_id.startswith("zh") else "en"
        title = text.strip().split("\n")[0] if text else ""

        docs.append(DocumentRecord(
            id=doc_id,
            language=language,
            title=title,
            text=text,
        ))

    return docs


def load_golden_labels(labels_path: str = "data/labels/golden_labels.json") -> list[dict]:
    """加载人工标注评估集。"""
    if not os.path.isfile(labels_path):
        return []
    with open(labels_path, "r", encoding="utf-8") as f:
        return json.load(f)
