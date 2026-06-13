"""
classifier.py
-------------
Classifies a resume into a career category using TF-IDF + Logistic Regression.

Why TF-IDF + Logistic Regression?
  - TF-IDF: converts resume text to a numeric vector. Terms that appear
    often in one category but rarely in others get high weight.
  - Logistic Regression: a linear model that learns decision boundaries.
    It's interpretable (you can see which words pushed toward each class),
    trains in milliseconds, and doesn't overfit on small datasets.
  
Why not deep learning?
  - We don't have thousands of labeled resumes
  - A student can understand logistic regression fully
  - Results are competitive for this text-length task
  - No GPU required

Training strategy: We use synthetic training data built from our skill
taxonomy + domain keywords. This is seeded text, not plagiarized content.
The model re-trains on every app startup (takes ~50ms) so no model files
need to be stored or versioned.
"""

import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import numpy as np


# ── Synthetic Training Data ──────────────────────────────────────────────────
# Each entry is a bag of domain keywords that represent that role.
# Multiple variants simulate resume diversity.

TRAINING_DATA = {
    "Data Science": [
        "python pandas numpy matplotlib seaborn scikit-learn statistics machine learning data analysis exploratory data analysis feature engineering regression classification clustering jupyter notebook sql data visualization",
        "data scientist pandas python statistics hypothesis testing regression model evaluation cross validation feature selection sklearn matplotlib time series forecasting eda sql reporting",
        "python numpy scipy statsmodels pandas jupyter machine learning classification regression random forest gradient boosting model deployment data pipeline",
        "data analysis visualization pandas matplotlib seaborn plotly business intelligence reporting sql python statistics probability distributions",
        "python machine learning scikit-learn cross validation feature engineering pipeline hyperparameter tuning grid search random forest xgboost lightgbm",
    ],
    "AI/ML": [
        "deep learning neural networks tensorflow pytorch keras computer vision nlp natural language processing convolutional neural network recurrent neural network lstm transformer bert",
        "machine learning deep learning pytorch tensorflow keras hugging face transformers bert gpt transfer learning model training gpu cuda training inference",
        "natural language processing nlp text classification sentiment analysis named entity recognition bert transformers attention mechanism tokenization embeddings",
        "computer vision image classification object detection yolo opencv convolutional neural network cnn image segmentation tensorflow pytorch",
        "reinforcement learning deep q network policy gradient actor critic openai gym reward optimization autonomous agents",
        "generative ai llm rag vector database langchain prompt engineering fine tuning lora quantization inference optimization",
    ],
    "Web Development": [
        "html css javascript react nodejs express mongodb restful api frontend backend web development bootstrap tailwind",
        "react javascript typescript nextjs nodejs graphql postgresql mongodb rest api web developer frontend css html responsive design",
        "vue angular javascript html css backend api django flask python web application database postgresql mysql",
        "full stack developer react nodejs express mongodb sql javascript typescript css html webpack vite deployment",
        "frontend developer html5 css3 javascript react redux state management responsive ui ux design figma",
        "backend developer nodejs express python django flask restful api postgresql mysql redis authentication authorization jwt",
    ],
    "Android Development": [
        "android kotlin java android studio jetpack compose mvvm room retrofit coroutines firebase google play material design",
        "android developer kotlin java xml layouts fragments activities intents viewmodel livedata retrofit room database",
        "mobile development android kotlin jetpack compose navigation component dagger hilt dependency injection unit testing",
        "android sdk kotlin coroutines flow rx java mvp mvvm architecture components firebase crashlytics analytics push notifications",
    ],
    "Cybersecurity": [
        "penetration testing ethical hacking network security vulnerability assessment owasp burp suite metasploit kali linux wireshark nmap",
        "cybersecurity analyst siem soc incident response digital forensics threat intelligence firewall ids ips intrusion detection",
        "security engineering application security appsec secure sdlc code review threat modeling cryptography pki ssl tls zero trust",
        "network security cisco firewall vpn ids ips packet analysis wireshark security audit compliance iso 27001 gdpr",
        "ethical hacking penetration testing web application security sql injection xss csrf privilege escalation ctf challenges",
    ],
    "DevOps": [
        "docker kubernetes jenkins github actions ci cd terraform ansible linux bash aws azure gcp monitoring prometheus grafana",
        "devops engineer docker kubernetes helm argocd gitops infrastructure as code terraform ansible ci cd pipeline jenkins gitlab",
        "site reliability engineering sre kubernetes docker prometheus grafana alerting incident management on call linux bash python",
        "cloud engineering aws ec2 s3 lambda cloudformation terraform vpc iam auto scaling load balancer rds dynamo db",
        "platform engineering kubernetes docker helm istio service mesh observability logging tracing monitoring elk stack datadog",
    ],
}

# Flatten into (text, label) pairs
CORPUS = []
LABELS = []
for label, texts in TRAINING_DATA.items():
    for text in texts:
        CORPUS.append(text)
        LABELS.append(label)


# ── Build Pipeline ───────────────────────────────────────────────────────────

def build_classifier() -> Pipeline:
    """
    Builds and trains a TF-IDF + Logistic Regression pipeline.
    Returns a fitted sklearn Pipeline.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=1.0,          # regularization strength (higher = less regularization)
            solver="lbfgs",
        )),
    ])
    pipeline.fit(CORPUS, LABELS)
    return pipeline


# Eager load at import time — takes ~50ms, negligible
_CLASSIFIER = build_classifier()


def classify_resume(text: str) -> dict:
    """
    Predicts the career category of a resume.

    Returns:
      {
        "category": "Data Science",
        "confidence": 87.3,
        "all_scores": {"Data Science": 87.3, "AI/ML": 8.1, ...}
      }
    """
    if not text.strip():
        return {"category": "Unknown", "confidence": 0, "all_scores": {}}

    clean = re.sub(r"\s+", " ", text.lower())
    proba = _CLASSIFIER.predict_proba([clean])[0]
    classes = _CLASSIFIER.classes_

    all_scores = {
        cls: round(float(p) * 100, 1)
        for cls, p in zip(classes, proba)
    }

    best_idx   = int(np.argmax(proba))
    category   = classes[best_idx]
    confidence = all_scores[category]

    return {
        "category":   category,
        "confidence": confidence,
        "all_scores": dict(sorted(all_scores.items(), key=lambda x: -x[1])),
    }
