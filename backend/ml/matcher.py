"""
matcher.py
----------
Matches a resume against a job description using TF-IDF + Cosine Similarity.

Mathematics:
  1. TF-IDF converts text to a vector where each dimension is a term.
     TF (term frequency) = count of term / total terms in doc
     IDF (inverse doc frequency) = log(N / df+1)  — rare terms get higher weight
     
  2. Cosine Similarity = dot(A, B) / (||A|| × ||B||)
     Measures the angle between two vectors, not their magnitude.
     Score of 1.0 = identical, 0.0 = nothing in common.
     
  Why not Euclidean distance?
     Euclidean distance is sensitive to document length. A long resume
     would always appear "far" from a short JD even if they share terms.
     Cosine similarity normalizes for length — perfect for this use case.

Design choice: sklearn's TfidfVectorizer handles:
  - Stop word removal ("the", "and", "is" don't matter)
  - Tokenization
  - IDF weighting
  - L2 normalization (so cosine = just dot product)
"""

import re
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SKILL_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "skills_db", "skills.json"
)
with open(_SKILL_DB_PATH) as f:
    SKILL_DB = json.load(f)

ALL_SKILLS_FLAT = []
for cat_skills in SKILL_DB.values():
    ALL_SKILLS_FLAT.extend(cat_skills)


def _preprocess(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_skill_set(text: str) -> set:
    """Returns set of skills found in text using the skill taxonomy."""
    text_lower = text.lower()
    found = set()
    for skill in ALL_SKILLS_FLAT:
        escaped = re.escape(skill)
        pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found


def compute_match(resume_text: str, jd_text: str) -> dict:
    """
    Computes similarity between resume and job description.

    Returns:
      {
        "match_pct": 73.5,
        "matching_skills": ["python", "docker", ...],
        "missing_skills":  ["kubernetes", "spark", ...],
        "jd_skills":       ["python", "docker", "kubernetes", ...],
        "resume_skills":   ["python", "docker", ...],
        "tfidf_score":     0.735,
      }
    """
    if not resume_text.strip() or not jd_text.strip():
        return {"error": "Both resume and JD must be non-empty"}

    # ── TF-IDF Cosine Similarity ────────────────────────────────────────────
    clean_resume = _preprocess(resume_text)
    clean_jd     = _preprocess(jd_text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),   # unigrams + bigrams (catches "machine learning")
        max_features=5000,
        sublinear_tf=True,    # log(1+tf) dampens very frequent terms
    )

    try:
        tfidf_matrix = vectorizer.fit_transform([clean_resume, clean_jd])
        score = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception:
        score = 0.0

    # ── Skill-Level Analysis ────────────────────────────────────────────────
    resume_skills = _extract_skill_set(resume_text)
    jd_skills     = _extract_skill_set(jd_text)

    matching = sorted(resume_skills & jd_skills)
    missing  = sorted(jd_skills - resume_skills)

    # Blend TF-IDF with skill overlap for a more intuitive final score
    if jd_skills:
        skill_overlap_pct = len(matching) / len(jd_skills)
    else:
        skill_overlap_pct = score

    # Weighted blend: 60% TF-IDF (captures phrasing), 40% skill overlap (actionable)
    blended = 0.6 * score + 0.4 * skill_overlap_pct
    match_pct = round(min(blended * 100, 100), 1)

    return {
        "match_pct":      match_pct,
        "tfidf_score":    round(score * 100, 1),
        "skill_overlap":  round(skill_overlap_pct * 100, 1),
        "matching_skills": matching,
        "missing_skills":  missing,
        "jd_skills":       sorted(jd_skills),
        "resume_skills":   sorted(resume_skills),
    }
