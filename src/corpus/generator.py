import json
import random
import os


# ---- 模板素材池 ----

ZH_JOB_TITLES = [
    "Java开发工程师", "Python后端开发", "前端开发工程师", "数据分析师",
    "产品经理", "UI/UX设计师", "测试工程师", "运维工程师",
    "算法工程师", "全栈开发工程师", "Android开发工程师", "iOS开发工程师",
    "网络安全工程师", "数据库管理员", "DevOps工程师", "技术支持工程师",
    "项目经理", "运营总监", "市场营销经理", "财务分析师",
]

EN_JOB_TITLES = [
    "Senior Backend Engineer", "Frontend Developer", "Data Scientist",
    "Product Manager", "UX Designer", "QA Engineer", "DevOps Engineer",
    "Machine Learning Engineer", "Full Stack Developer", "Software Engineer",
    "Cloud Architect", "Security Engineer", "Database Administrator",
    "Tech Lead", "Engineering Manager", "Business Analyst",
    "Marketing Specialist", "Financial Analyst", "HR Manager", "Sales Representative",
]

ZH_COMPANIES = [
    "字节跳动", "阿里巴巴", "腾讯科技", "华为技术", "百度在线",
    "京东集团", "美团", "滴滴出行", "小米科技", "网易集团",
    "拼多多", "快手科技", "哔哩哔哩", "知乎", "商汤科技",
    "科大讯飞", "蔚来汽车", "理想汽车", "大疆创新", "海康威视",
]

EN_COMPANIES = [
    "Google", "Microsoft", "Amazon", "Apple", "Meta",
    "Netflix", "Tesla", "Uber", "Airbnb", "Stripe",
    "Snowflake", "Databricks", "Palantir", "Notion", "Figma",
    "Shopify", "Atlassian", "Canva", "Zoom", "Slack",
]

ZH_LOCATIONS = [
    "北京", "上海", "深圳", "杭州", "广州", "成都", "南京",
    "武汉", "西安", "苏州", "重庆", "长沙", "天津", "合肥",
]

EN_LOCATIONS = [
    "San Francisco, CA", "New York, NY", "Seattle, WA", "Austin, TX",
    "Boston, MA", "Los Angeles, CA", "Chicago, IL", "Denver, CO",
    "London, UK", "Singapore", "Tokyo, Japan", "Berlin, Germany",
    "Toronto, Canada", "Sydney, Australia",
]

ZH_SALARIES = [
    "15k-25k/月", "20k-35k/月", "25k-40k/月", "30k-50k/月",
    "10k-18k/月", "18k-28k/月", "年薪30万-50万", "年薪40万-60万",
    "年薪20万-30万", "35k-55k/月", "面议", "薪资面谈",
]

EN_SALARIES = [
    "$80k-$120k annually", "$100k-$150k annually", "$120k-$180k annually",
    "$90k-$130k annually", "$70k-$100k annually", "$150k-$200k annually",
    "$60k-$85k annually", "$130k-$170k annually", "Competitive salary",
    "$40k-$60k annually",
]

ZH_EDUCATIONS = [
    "本科及以上", "硕士及以上", "本科", "硕士", "博士",
    "大专及以上", "本科及以上学历", "硕士优先", "博士优先",
    "计算机相关专业本科", "985/211本科及以上",
]

EN_EDUCATIONS = [
    "Bachelor's degree required", "Master's degree preferred",
    "PhD in related field", "Bachelor's or above",
    "Master's degree in Computer Science", "Bachelor's degree in related field",
    "MBA preferred", "Master's or PhD", "Bachelor's minimum",
]

ZH_EXPERIENCES = [
    "3年以上经验", "1-3年经验", "5年以上经验", "2年以上经验",
    "3-5年工作经验", "1年以上经验", "应届生可投", "10年以上经验",
    "3年以上相关经验", "5-8年经验",
]

EN_EXPERIENCES = [
    "3+ years of experience", "1-3 years of experience",
    "5+ years of experience", "2+ years of experience",
    "3-5 years of experience", "Entry level welcome",
    "7+ years of experience", "4-6 years of experience",
    "10+ years of experience",
]

ZH_SKILLS_POOL = [
    "Java", "Python", "Spring Boot", "MySQL", "Redis", "Docker",
    "Kubernetes", "Linux", "Git", "MongoDB", "Nginx", "RabbitMQ",
    "Kafka", "Elasticsearch", "Hadoop", "Spark", "Flink", "TensorFlow",
    "PyTorch", "React", "Vue.js", "TypeScript", "Node.js", "Go",
    "C++", "Rust", "AWS", "Azure", "PostgreSQL", "微服务架构",
    "分布式系统", "敏捷开发", "CI/CD", "Data Warehouse",
]

EN_SKILLS_POOL = [
    "Python", "Java", "Go", "Rust", "C++", "TypeScript",
    "React", "Vue", "Angular", "Node.js", "Django", "Spring",
    "AWS", "GCP", "Azure", "Docker", "Kubernetes", "Terraform",
    "Kafka", "Spark", "Flink", "TensorFlow", "PyTorch", "SQL",
    "PostgreSQL", "MongoDB", "Redis", "GraphQL", "gRPC",
    "CI/CD", "Agile", "Microservices", "System Design",
]


def _gen_zh_doc(idx: int) -> dict:
    title = random.choice(ZH_JOB_TITLES)
    company = random.choice(ZH_COMPANIES)
    location = random.choice(ZH_LOCATIONS)
    salary = random.choice(ZH_SALARIES)
    education = random.choice(ZH_EDUCATIONS)
    experience = random.choice(ZH_EXPERIENCES)
    n_skills = random.randint(3, 8)
    skills = random.sample(ZH_SKILLS_POOL, n_skills)

    text = f"""{company} - 招聘{title}

【岗位名称】{title}
【公司名称】{company}
【工作地点】{location}
【薪资范围】{salary}

【岗位职责】
1. 负责公司核心业务系统的设计、开发与维护工作；
2. 参与系统架构设计，保证系统的高可用、高性能和可扩展性；
3. 与产品经理和团队成员紧密协作，推动项目按时高质量交付；
4. 撰写技术文档，参与代码评审，提升团队整体技术水平。

【任职要求】
1. 学历要求：{education}；
2. 工作经验：{experience}；
3. 熟练掌握{'、'.join(skills[:3])}等技术；
4. 具备良好的沟通能力和团队协作精神；
5. 具有较强的学习能力和问题分析解决能力。

【福利待遇】
五险一金、带薪年假、弹性工作制、免费三餐、年度体检、股票期权等。

有意者请将简历发送至 hr@{company.lower().replace(' ', '')}.com
"""

    return {
        "id": f"zh_{idx:03d}",
        "language": "zh",
        "title": title,
        "text": text,
        "labels": {
            "job_title": title,
            "company_name": company,
            "work_location": location,
            "salary": salary,
            "education": education,
            "experience": experience,
            "skills": sorted(skills),
        },
    }


def _gen_en_doc(idx: int) -> dict:
    title = random.choice(EN_JOB_TITLES)
    company = random.choice(EN_COMPANIES)
    location = random.choice(EN_LOCATIONS)
    salary = random.choice(EN_SALARIES)
    education = random.choice(EN_EDUCATIONS)
    experience = random.choice(EN_EXPERIENCES)
    n_skills = random.randint(3, 8)
    skills = random.sample(EN_SKILLS_POOL, n_skills)

    text = f"""{company} - {title}

Position: {title}
Company: {company}
Location: {location}
Salary: {salary}

About the Role:
We are looking for a talented {title} to join our growing team at {company}. You will work on cutting-edge projects and collaborate with cross-functional teams to deliver high-quality solutions.

Key Responsibilities:
- Design, develop and maintain scalable backend services and APIs.
- Collaborate with product managers and frontend engineers to ship new features.
- Participate in code reviews and contribute to engineering best practices.
- Mentor junior engineers and help grow the team.
- Optimize application performance and reliability.

Qualifications:
- Education: {education}.
- Experience: {experience}.
- Proficiency in {', '.join(skills[:3])} and related technologies.
- Strong problem-solving skills and attention to detail.
- Excellent communication and teamwork abilities.

Benefits:
Competitive compensation package, health insurance, 401(k) matching, flexible PTO, remote-friendly culture.

To apply, please send your resume to careers@{company.lower().replace(' ', '')}.com
"""

    return {
        "id": f"en_{idx:03d}",
        "language": "en",
        "title": title,
        "text": text,
        "labels": {
            "job_title": title,
            "company_name": company,
            "work_location": location,
            "salary": salary,
            "education": education,
            "experience": experience,
            "skills": sorted(skills),
        },
    }


def generate_corpus(
    total: int = 150,
    zh_ratio: float = 0.6,
    output_dir: str = "data/corpus",
) -> list[dict]:
    """生成模拟招聘语料并写入 .txt 文件，返回所有文档的元数据列表。"""
    os.makedirs(output_dir, exist_ok=True)

    n_zh = int(total * zh_ratio)
    n_en = total - n_zh

    metadata_list = []

    for i in range(n_zh):
        doc = _gen_zh_doc(i + 1)
        filepath = os.path.join(output_dir, f"{doc['id']}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        metadata_list.append({
            "id": doc["id"],
            "language": doc["language"],
            "title": doc["title"],
            "file": filepath,
            "labels": doc["labels"],
        })

    for i in range(n_en):
        doc = _gen_en_doc(i + 1)
        filepath = os.path.join(output_dir, f"{doc['id']}.txt")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(doc["text"])
        metadata_list.append({
            "id": doc["id"],
            "language": doc["language"],
            "title": doc["title"],
            "file": filepath,
            "labels": doc["labels"],
        })

    return metadata_list


def generate_golden_labels(
    metadata_list: list[dict],
    eval_sample: int = 25,
    output_dir: str = "data/labels",
) -> list[dict]:
    """从 metadata 中随机抽样生成 golden_labels。"""
    os.makedirs(output_dir, exist_ok=True)

    sample = random.sample(metadata_list, min(eval_sample, len(metadata_list)))
    golden = [{"id": m["id"], "language": m["language"], "labels": m["labels"]} for m in sample]

    filepath = os.path.join(output_dir, "golden_labels.json")
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)

    return golden


if __name__ == "__main__":
    metadata = generate_corpus(total=150, zh_ratio=0.6, output_dir="data/corpus")
    print(f"生成语料: {len(metadata)} 篇")

    golden = generate_golden_labels(metadata, eval_sample=25, output_dir="data/labels")
    print(f"生成 golden_labels: {len(golden)} 篇")
