"""Streamlit Web 界面。"""
import json
import os
import sys

import streamlit as st
import yaml

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.corpus.loader import load_corpus, load_golden_labels
from src.corpus.generator import generate_corpus, generate_golden_labels
from src.graph.builder import build_graph
from src.evaluation.evaluator import evaluate


st.set_page_config(page_title="招聘信息抽取系统", layout="wide")
st.title("基于 LangGraph 的中英双语招聘信息抽取实验系统")


def load_config():
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---- 侧边栏 ----
st.sidebar.header("操作面板")

if st.sidebar.button("生成模拟语料"):
    config = load_config()
    total = config["corpus"]["total_docs"]
    zh_ratio = config["corpus"]["zh_ratio"]
    with st.spinner(f"正在生成 {total} 篇语料..."):
        metadata = generate_corpus(total=total, zh_ratio=zh_ratio)
        generate_golden_labels(metadata, eval_sample=config["corpus"]["eval_sample"])
    st.sidebar.success(f"已生成 {len(metadata)} 篇语料 + golden_labels")

if st.sidebar.button("运行抽取流程"):
    config = load_config()
    docs = load_corpus()
    if not docs:
        st.sidebar.error("请先生成语料")
    else:
        graph = build_graph()
        progress = st.sidebar.progress(0)
        status = st.sidebar.empty()

        all_results = []
        for i, doc in enumerate(docs):
            state = {"doc_id": doc.id, "language": doc.language, "title": doc.title, "raw_text": doc.text}
            final_state = graph.invoke(state)
            result = final_state.get("final_result", {})
            result["id"] = doc.id
            result["language"] = doc.language
            all_results.append(result)
            progress.progress((i + 1) / len(docs))
            status.text(f"处理中: {i + 1}/{len(docs)}")

        os.makedirs(config["paths"]["output_dir"], exist_ok=True)
        with open(os.path.join(config["paths"]["output_dir"], "merged_results.json"), "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        st.session_state["results"] = all_results
        st.session_state["docs"] = docs
        st.sidebar.success(f"抽取完成！共 {len(all_results)} 篇")

# ---- 主区域 ----
tab1, tab2, tab3, tab4 = st.tabs(["语料浏览", "抽取结果", "规则 vs LLM 对比", "评估报告"])

config = load_config()

with tab1:
    st.header("语料浏览")
    docs = load_corpus()
    if docs:
        doc_ids = [d.id for d in docs]
        selected = st.selectbox("选择文档", doc_ids)
        doc = [d for d in docs if d.id == selected][0]
        st.subheader(f"{doc.id} — {doc.title}")
        st.text_area("原文", doc.text, height=400)
    else:
        st.info("暂无语料，请点击侧边栏「生成模拟语料」")

with tab2:
    st.header("抽取结果")

    results_path = os.path.join(config["paths"]["output_dir"], "merged_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        doc_id = st.selectbox("选择文档查看结果", [r["id"] for r in all_results])
        result = [r for r in all_results if r["id"] == doc_id][0]

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("抽取字段")
            for field in ["job_title", "company_name", "work_location", "salary", "education", "experience"]:
                val = result.get(field)
                st.metric(field, val if val else "未抽取到")
        with col2:
            st.subheader("技能要求")
            skills = result.get("skills", [])
            if skills:
                badges = " ".join(
                    f'<span style="display:inline-block;background:#e8f0fe;color:#1967d2;'
                    f'padding:2px 10px;margin:2px;border-radius:12px;font-size:14px;">{s}</span>'
                    for s in skills
                )
                st.markdown(badges, unsafe_allow_html=True)
            else:
                st.text("未抽取到技能")

        if result.get("extractor_breakdown"):
            st.subheader("各字段使用的抽取器")
            st.json(result["extractor_breakdown"])
    else:
        st.info("暂无结果，请先运行抽取流程")

with tab3:
    st.header("规则 vs LLM 对比")

    results_path = os.path.join(config["paths"]["output_dir"], "merged_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r", encoding="utf-8") as f:
            all_results = json.load(f)

        import pandas as pd
        rows = []
        for r in all_results[:20]:
            rows.append({
                "doc_id": r["id"],
                "language": r.get("language", ""),
                "job_title": r.get("job_title", "-"),
                "company_name": r.get("company_name", "-"),
                "salary": r.get("salary", "-"),
                "skills_count": len(r.get("skills") or []),
                "conflicts": ", ".join(r.get("conflicts", [])),
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("暂无结果")

with tab4:
    st.header("评估报告")

    report_path = os.path.join(config["paths"]["output_dir"], "evaluation_report.json")
    if os.path.exists(report_path):
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        st.subheader("整体指标")
        col1, col2, col3 = st.columns(3)
        overall = report["overall"]
        col1.metric("Precision", f"{overall['precision']:.2%}")
        col2.metric("Recall", f"{overall['recall']:.2%}")
        col3.metric("F1", f"{overall['f1']:.2%}")

        st.subheader("分字段指标")
        import pandas as pd
        field_rows = []
        for f, stats in report.get("per_field", {}).items():
            field_rows.append({
                "字段": f,
                "Precision": f"{stats['precision']:.2%}",
                "Recall": f"{stats['recall']:.2%}",
                "F1": f"{stats['f1']:.2%}",
            })
        st.dataframe(pd.DataFrame(field_rows), use_container_width=True)

        st.subheader("分语言指标")
        lang_rows = []
        for lang, stats in report.get("per_language", {}).items():
            lang_rows.append({
                "语言": lang,
                "Precision": f"{stats['precision']:.2%}",
                "Recall": f"{stats['recall']:.2%}",
                "F1": f"{stats['f1']:.2%}",
            })
        st.dataframe(pd.DataFrame(lang_rows), use_container_width=True)
    else:
        st.info("暂无评估报告，请先运行抽取流程")
