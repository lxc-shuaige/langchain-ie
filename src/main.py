"""CLI 入口：运行完整抽取流程并保存结果。"""
import json
import os
import sys
import yaml
from src.corpus.loader import load_corpus, load_golden_labels
from src.graph.builder import build_graph
from src.evaluation.evaluator import evaluate, save_evaluation_report


def main():
    config_path = "config.yaml"
    if not os.path.exists(config_path):
        print("错误: 找不到 config.yaml，请先创建配置文件。")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = config["paths"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)

    # 加载语料
    print("加载语料...")
    docs = load_corpus(config["paths"]["corpus_dir"])
    if not docs:
        print("错误: 语料目录为空，请先运行 python -m src.corpus.generator")
        sys.exit(1)
    print(f"共 {len(docs)} 篇文档")

    # 构建图
    print("构建 LangGraph 工作流...")
    graph = build_graph()

    # 逐文档执行抽取
    print("开始抽取...")
    all_results = []
    for i, doc in enumerate(docs):
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
        all_results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(docs)} 篇")

    print("抽取完成！")

    # 保存结果
    merged_path = os.path.join(output_dir, "merged_results.json")
    with open(merged_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"结果已保存到 {merged_path}")

    # 评估
    golden = load_golden_labels(config["paths"].get("labels_dir", "data/labels") + "/golden_labels.json")
    if golden:
        print("运行评估...")
        eval_result = evaluate(all_results, golden)
        save_evaluation_report(eval_result, os.path.join(output_dir, "evaluation_report.json"))
        print(f"整体 F1: {eval_result['overall']['f1']:.4f}")
        for f, stats in eval_result["per_field"].items():
            print(f"  {f}: P={stats['precision']:.4f}, R={stats['recall']:.4f}, F1={stats['f1']:.4f}")
    else:
        print("未找到 golden_labels，跳过评估。")


if __name__ == "__main__":
    main()
