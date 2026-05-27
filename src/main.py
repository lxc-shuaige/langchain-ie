"""CLI 入口：运行完整抽取流程并保存结果。

用法:
    python src/main.py           纯文本模式（默认）
    python src/main.py image     图片模式（OCR + 抽取）
    python src/main.py all       文本 + 图片混合模式
"""
import json
import os
import sys
import yaml
from src.corpus.loader import load_corpus, load_image_corpus, load_all_golden_labels
from src.graph.builder import build_graph
from src.evaluation.evaluator import evaluate, save_evaluation_report


def _load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _run_text_pipeline(config: dict) -> list[dict]:
    """运行纯文本抽取流程，返回结果列表。"""
    import concurrent.futures
    import os as _os

    docs = load_corpus(config["paths"]["corpus_dir"])
    if not docs:
        print("错误: 语料目录为空，请先运行 python -m src.corpus.generator")
        return []

    print(f"共 {len(docs)} 篇文本文档")

    max_workers = config.get("processing", {}).get("max_workers", min(8, (_os.cpu_count() or 1) * 2))

    def _process_one(doc):
        graph = build_graph()
        initial_state = {
            "doc_id": doc.id,
            "language": doc.language,
            "title": doc.title,
            "raw_text": doc.text,
        }
        final_state = graph.invoke(initial_state)
        result = final_state.get("final_result", {})
        result["id"] = doc.id
        result["language"] = doc.language
        return result

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, doc): i for i, doc in enumerate(docs)}
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())
            if (len(all_results)) % 10 == 0:
                print(f"  已处理文本 {len(all_results)}/{len(docs)} 篇")

    return all_results


def _run_image_pipeline(config: dict) -> list[dict]:
    """运行图片抽取流程，返回结果列表。"""
    import concurrent.futures
    import os as _os

    image_dir = config["paths"].get("image_dir", "data/images")
    docs = load_image_corpus(image_dir)
    if not docs:
        print("错误: 图片目录为空，请先运行 python -m src.corpus.image_generator")
        return []

    print(f"共 {len(docs)} 张图片")

    max_workers = config.get("processing", {}).get("max_workers", min(8, (_os.cpu_count() or 1) * 2))

    def _process_one(doc):
        graph = build_graph()
        initial_state = {
            "doc_id": doc.id,
            "language": doc.language,
            "title": doc.title,
            "raw_text": doc.text,
            "input_type": "image",
            "image_path": doc.image_path,
        }
        final_state = graph.invoke(initial_state)
        result = final_state.get("final_result", {})
        result["id"] = doc.id
        result["language"] = doc.language
        return result

    all_results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, doc): i for i, doc in enumerate(docs)}
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())
            if (len(all_results)) % 5 == 0:
                print(f"  已处理图片 {len(all_results)}/{len(docs)} 张")

    return all_results


def main():
    config = _load_config()

    mode = sys.argv[1] if len(sys.argv) > 1 else "text"
    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    all_results = []

    if mode in ("text", "all"):
        print("=" * 50)
        print("文本模式抽取")
        print("=" * 50)
        text_results = _run_text_pipeline(config)
        all_results.extend(text_results)

    if mode in ("image", "all"):
        print("=" * 50)
        print("图片模式抽取（OCR + IE）")
        print("=" * 50)
        image_results = _run_image_pipeline(config)
        all_results.extend(image_results)

    if not all_results:
        return

    print(f"\n抽取完成！共 {len(all_results)} 篇")

    # 保存结果
    merged_path = os.path.join(output_dir, "merged_results.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {merged_path}")

    # 评估
    golden = load_all_golden_labels(config["paths"].get("labels_dir", "data/labels"))
    if golden:
        print("\n运行评估...")
        eval_result = evaluate(all_results, golden)
        save_evaluation_report(eval_result, os.path.join(output_dir, "evaluation_report.json"))
        print(f"整体 F1: {eval_result['overall']['f1']:.4f}")
        for f, stats in eval_result["per_field"].items():
            print(f"  {f}: P={stats['precision']:.4f}, R={stats['recall']:.4f}, F1={stats['f1']:.4f}")
    else:
        print("未找到 golden_labels，跳过评估。")


if __name__ == "__main__":
    main()
