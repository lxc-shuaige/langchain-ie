"""Generate synthetic recruitment poster images for OCR-based extraction testing."""
import json
import os
import random

from PIL import Image, ImageDraw, ImageFont

from src.corpus.generator import (
    ZH_JOB_TITLES, ZH_COMPANIES, ZH_LOCATIONS, ZH_SALARIES,
    ZH_EDUCATIONS, ZH_EXPERIENCES, ZH_SKILLS_POOL,
)

ZH_PHONES = [
    "13912345678", "18612345678", "13898765432", "15987654321",
    "18812349876", "13611223344", "17788990011", "15233445566",
    "13566778899", "18900112233", "15811223344", "13799887766",
]

ZH_EMAILS = [
    "hr@bytedance.com", "recruit@alibaba-inc.com", "jobs@tencent.com",
    "career@huawei.com", "hr@baidu.com", "talent@meituan.com",
    "hr@didichuxing.com", "jobs@xiaomi.com", "recruit@netease.com",
    "career@pinduoduo.com", "hr@kuaishou.com", "jobs@bilibili.com",
]


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Try to load a CJK-capable font; fall back to default."""
    font_paths = [
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simsun.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _render_poster(
    company: str,
    job: str,
    location: str,
    salary: str,
    education: str,
    experience: str,
    skills: list[str],
    phone: str,
    email: str,
) -> Image.Image:
    """Draw a recruitment poster on a 600x850 white canvas."""
    W, H = 600, 850
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    # Header banner
    draw.rectangle([(0, 0), (W, 120)], fill="#1a5276")
    font_title = _get_font(36)
    font_sub = _get_font(24)
    font_body = _get_font(22)
    font_small = _get_font(18)

    # Centered header text
    header_text = "招  聘"
    bbox = draw.textbbox((0, 0), header_text, font=font_title)
    tw = bbox[2] - bbox[0]
    draw.text(((W - tw) // 2, 20), header_text, fill="white", font=font_title)

    sub_text = "RECRUITMENT"
    bbox = draw.textbbox((0, 0), sub_text, font=font_small)
    sw = bbox[2] - bbox[0]
    draw.text(((W - sw) // 2, 72), sub_text, fill="#aed6f1", font=font_small)

    y = 140

    # Company name
    draw.text((40, y), company, fill="#1a5276", font=font_title)
    y += 55

    # Job title
    job_text = f"招聘岗位：{job}"
    draw.text((40, y), job_text, fill="#c0392b", font=font_sub)
    y += 50

    # Divider
    draw.line([(40, y), (W - 40, y)], fill="#1a5276", width=2)
    y += 25

    # Info fields
    fields = [
        ("工作地点", location),
        ("薪资待遇", salary),
        ("学历要求", education),
        ("经验要求", experience),
    ]
    for label, value in fields:
        draw.text((60, y), f"{label}：{value}", fill="#2c3e50", font=font_body)
        y += 38

    y += 5

    # Skills section
    draw.text((60, y), "技能要求：", fill="#2c3e50", font=font_body)
    y += 35

    skill_text = "、".join(skills[:6])
    # Wrap skills to fit width
    words = skill_text.split("、")
    line = ""
    x0 = 80
    for i, w in enumerate(words):
        test = f"{line}、{w}" if line else w
        bbox = draw.textbbox((0, 0), test, font=font_small)
        tw = bbox[2] - bbox[0]
        if tw > W - 100 and line:
            draw.text((x0, y), line, fill="#2c3e50", font=font_small)
            y += 28
            line = w
        else:
            line = test
    if line:
        draw.text((x0, y), line, fill="#2c3e50", font=font_small)
        y += 35

    y += 10

    # Divider
    draw.line([(40, y), (W - 40, y)], fill="#bdc3c7", width=1)
    y += 25

    # Contact info (key for contact_info extraction)
    draw.text((60, y), "联系方式", fill="#1a5276", font=font_sub)
    y += 38
    draw.text((80, y), f"联系电话：{phone}", fill="#2c3e50", font=font_body)
    y += 35
    draw.text((80, y), f"电子邮箱：{email}", fill="#2c3e50", font=font_body)
    y += 45

    # Footer
    draw.rectangle([(0, H - 60), (W, H)], fill="#1a5276")
    footer_text = f"{company} 人力资源部"
    bbox = draw.textbbox((0, 0), footer_text, font=font_small)
    fw = bbox[2] - bbox[0]
    draw.text(((W - fw) // 2, H - 45), footer_text, fill="#aed6f1", font=font_small)

    return img


def generate_image_corpus(total: int = 30, output_dir: str = "data/images") -> list[dict]:
    """Generate synthetic recruitment poster images.

    Returns metadata list suitable for golden_labels generation.
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata_list = []

    for i in range(total):
        company = random.choice(ZH_COMPANIES)
        job = random.choice(ZH_JOB_TITLES)
        location = random.choice(ZH_LOCATIONS)
        salary = random.choice(ZH_SALARIES)
        education = random.choice(ZH_EDUCATIONS)
        experience = random.choice(ZH_EXPERIENCES)
        n_skills = random.randint(3, 6)
        skills = random.sample(ZH_SKILLS_POOL, n_skills)
        phone = random.choice(ZH_PHONES)
        email = random.choice(ZH_EMAILS)

        img = _render_poster(
            company=company,
            job=job,
            location=location,
            salary=salary,
            education=education,
            experience=experience,
            skills=skills,
            phone=phone,
            email=email,
        )

        doc_id = f"img_{i + 1:03d}"
        filepath = os.path.join(output_dir, f"{doc_id}.png")
        img.save(filepath, "PNG")

        metadata_list.append({
            "id": doc_id,
            "language": "zh",
            "title": f"{company}-{job}",
            "file": filepath,
            "labels": {
                "job_title": job,
                "company_name": company,
                "work_location": location,
                "salary": salary,
                "education": education,
                "experience": experience,
                "skills": sorted(skills),
                "contact_info": phone,
            },
        })

    return metadata_list


def generate_image_golden_labels(
    metadata_list: list[dict],
    eval_sample: int = 10,
    output_dir: str = "data/labels",
) -> list[dict]:
    """Generate golden_labels for image corpus (separate file for independent eval)."""
    os.makedirs(output_dir, exist_ok=True)

    sample = random.sample(metadata_list, min(eval_sample, len(metadata_list)))
    golden = [{"id": m["id"], "language": m["language"], "labels": m["labels"]} for m in sample]

    filepath = os.path.join(output_dir, "golden_labels_image.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    return golden


if __name__ == "__main__":
    metadata = generate_image_corpus(total=30, output_dir="data/images")
    print(f"生成招聘海报图片: {len(metadata)} 张")

    golden = generate_image_golden_labels(metadata, eval_sample=10, output_dir="data/labels")
    print(f"生成图片 golden_labels: {len(golden)} 篇")
