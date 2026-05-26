"""Streamlit Web 界面。"""
import json
import os
import sys
import tempfile

import streamlit as st
import yaml
from PIL import Image

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.corpus.loader import load_corpus, load_image_corpus, load_all_golden_labels
from src.corpus.generator import generate_corpus, generate_golden_labels
from src.corpus.image_generator import generate_image_corpus, generate_image_golden_labels
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

if st.sidebar.button("生成招聘海报图片"):
    with st.spinner("正在生成 30 张招聘海报..."):
        img_meta = generate_image_corpus(total=30)
        generate_image_golden_labels(img_meta, eval_sample=10)
    st.sidebar.success(f"已生成 {len(img_meta)} 张海报 + golden_labels")

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
tab1, tab2, tab3, tab4, tab5 = st.tabs(["语料浏览", "抽取结果", "规则 vs LLM 对比", "评估报告", "图片抽取"])

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

with tab5:
    st.header("图片抽取（OCR + IE）")
    st.markdown("上传招聘海报图片，系统将通过 OCR 识别文字并自动抽取招聘信息。")

    uploaded_file = st.file_uploader("上传招聘海报", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # 保存上传文件到临时目录
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, uploaded_file.name)
        with open(tmp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 显示图片
        col_img, col_ocr = st.columns(2)
        with col_img:
            st.image(uploaded_file, caption="上传的招聘海报", use_container_width=True)

        # OCR 预览
        with col_ocr:
            with st.spinner("正在进行 OCR 识别..."):
                from src.ocr.ocr_extractor import OCRExtractor
                ocr = OCRExtractor()
                ocr_text = ocr.extract(tmp_path)
            st.subheader("OCR 识别结果")
            if ocr_text:
                st.text_area("OCR 文字", ocr_text, height=300)
            else:
                st.warning("OCR 未检测到文字，请确认图片中包含清晰的中文文本。")

        # 抽取
        if st.button("开始抽取") and ocr_text:
            with st.spinner("正在运行抽取流程..."):
                graph = build_graph()
                initial_state = {
                    "doc_id": uploaded_file.name.rsplit(".", 1)[0],
                    "language": "zh",
                    "title": uploaded_file.name,
                    "raw_text": "",
                    "input_type": "image",
                    "image_path": tmp_path,
                }
                final_state = graph.invoke(initial_state)
                result = final_state.get("final_result", {})

            st.subheader("抽取结果")
            col1, col2, col3 = st.columns(3)
            fields_display = [
                ("job_title", "岗位名称"),
                ("company_name", "公司名称"),
                ("work_location", "工作地点"),
                ("salary", "薪资待遇"),
                ("education", "学历要求"),
                ("experience", "经验要求"),
                ("contact_info", "联系方式"),
            ]
            for i, (key, label) in enumerate(fields_display):
                col = [col1, col2, col3][i % 3]
                val = result.get(key)
                col.metric(label, val if val else "未抽取到")

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
                with st.expander("各字段使用的抽取器"):
                    st.json(result["extractor_breakdown"])

            if result.get("conflicts"):
                with st.expander("冲突字段"):
                    st.warning(f"存在冲突的字段: {', '.join(result['conflicts'])}")

    # 批量处理已有图片
    st.markdown("---")
    st.subheader("批量处理已有招聘海报")

    image_dir = config["paths"].get("image_dir", "data/images")
    if os.path.isdir(image_dir):
        pngs = [f for f in os.listdir(image_dir) if f.endswith(".png")]
        if pngs:
            st.info(f"图片目录 `{image_dir}` 中共有 {len(pngs)} 张图片")
            if st.button("批量运行图片抽取"):
                img_docs = load_image_corpus(image_dir)
                graph = build_graph()
                progress = st.progress(0)
                status = st.empty()

                img_results = []
                for i, doc in enumerate(img_docs):
                    state = {
                        "doc_id": doc.id,
                        "language": doc.language,
                        "title": doc.title,
                        "raw_text": "",
                        "input_type": "image",
                        "image_path": doc.image_path,
                    }
                    final_state = graph.invoke(state)
                    r = final_state.get("final_result", {})
                    r["id"] = doc.id
                    r["language"] = doc.language
                    img_results.append(r)
                    progress.progress((i + 1) / len(img_docs))
                    status.text(f"处理中: {i + 1}/{len(img_docs)}")

                output_dir = config["paths"]["output_dir"]
                os.makedirs(output_dir, exist_ok=True)
                img_out = os.path.join(output_dir, "image_results.json")
                with open(img_out, "w", encoding="utf-8") as f:
                    json.dump(img_results, f, ensure_ascii=False, indent=2)

                st.success(f"批量抽取完成！结果保存至 {img_out}")

                golden = load_all_golden_labels(config["paths"].get("labels_dir", "data/labels"))
                if golden:
                    eval_result = evaluate(img_results, golden)
                    st.subheader("图片抽取评估")
                    col1, col2, col3 = st.columns(3)
                    overall = eval_result["overall"]
                    col1.metric("Precision", f"{overall['precision']:.2%}")
                    col2.metric("Recall", f"{overall['recall']:.2%}")
                    col3.metric("F1", f"{overall['f1']:.2%}")
        else:
            st.info(f"暂无图片，请先点击侧边栏「生成招聘海报图片」或上传图片")
    else:
        st.info("图片目录不存在，请先点击侧边栏「生成招聘海报图片」")
