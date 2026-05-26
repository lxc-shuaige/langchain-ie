from src.extractors.base import BaseExtractor, empty_result


# 中英文技能词库
ZH_SKILLS = {
    "Java", "Python", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
    "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "Shell",
    "Spring Boot", "Spring", "MyBatis", "Hibernate", "Django", "Flask",
    "FastAPI", "React", "Vue", "Vue.js", "Angular", "Node.js",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Oracle", "SQL Server", "SQLite", "HBase", "Cassandra",
    "Docker", "Kubernetes", "K8s", "Jenkins", "GitLab CI",
    "Nginx", "Tomcat", "Apache", "Linux", "Shell脚本",
    "Hadoop", "Spark", "Flink", "Storm", "Kafka", "RabbitMQ",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas",
    "NumPy", "OpenCV", "NLP", "机器学习", "深度学习",
    "AWS", "Azure", "阿里云", "腾讯云", "华为云", "GCP",
    "Git", "SVN", "Maven", "Gradle", "Webpack", "Vite",
    "微服务", "分布式", "高并发", "敏捷开发", "DevOps",
    "RESTful", "GraphQL", "gRPC", "WebSocket",
    "数据分析", "数据挖掘", "数据仓库", "ETL",
    "Unity", "Unreal", "Cocos",
}

EN_SKILLS = {
    "Python", "Java", "C++", "C#", "Go", "Rust", "Ruby", "PHP",
    "JavaScript", "TypeScript", "HTML", "CSS", "SQL", "Bash",
    "Spring Boot", "Spring", "Hibernate", "Django", "Flask",
    "FastAPI", "React", "Vue", "Angular", "Node.js", "Express",
    "MySQL", "PostgreSQL", "MongoDB", "Redis", "Elasticsearch",
    "Oracle", "SQL Server", "SQLite", "DynamoDB",
    "Docker", "Kubernetes", "K8s", "Jenkins", "GitHub Actions",
    "Nginx", "Apache", "Linux", "Unix",
    "Hadoop", "Spark", "Flink", "Kafka", "RabbitMQ", "SQS",
    "TensorFlow", "PyTorch", "Keras", "Scikit-learn", "Pandas",
    "NumPy", "OpenCV", "NLP", "Machine Learning", "Deep Learning",
    "AWS", "Azure", "GCP", "Terraform", "Ansible",
    "Git", "Maven", "Gradle", "Webpack", "Vite",
    "Microservices", "Distributed Systems", "Agile", "DevOps", "CI/CD",
    "REST", "RESTful", "GraphQL", "gRPC", "WebSocket",
    "Data Analysis", "Data Mining", "Data Warehouse", "ETL",
    "Unity", "Unreal Engine",
}


class DictionaryExtractor(BaseExtractor):
    name = "dictionary"

    def extract(self, text: str, language: str) -> dict:
        skills_pool = ZH_SKILLS if language == "zh" else EN_SKILLS
        found = set()

        for skill in skills_pool:
            if skill.lower() in text.lower():
                found.add(skill)

        result = empty_result()
        result["skills"] = sorted(found) if found else None
        return result
