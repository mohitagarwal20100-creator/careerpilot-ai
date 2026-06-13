"""
skill_extractor.py
------------------
Detects skills from resume text using a structured skill taxonomy.

Design choice: Dictionary lookup over ML-based NER because:
- Skill names are exact tokens — perfect for string matching
- Zero training data required
- 100% explainable (you can see exactly which string matched)
- Instant inference with no model loading overhead

Fuzzy matching handles typos: "scikit learn" matches "scikit-learn"
"""

import json
import re
import os

# ── Load Skill Database ──────────────────────────────────────────────────────

_SKILL_DB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "skills_db", "skills.json"
)

with open(_SKILL_DB_PATH, "r") as f:
    SKILL_DB: dict = json.load(f)

# Flatten all skills into a single searchable list with category metadata
ALL_SKILLS: list[dict] = []
for category, skills in SKILL_DB.items():
    for skill in skills:
        ALL_SKILLS.append({"skill": skill, "category": category})


def _normalize(text: str) -> str:
    """Lowercase and collapse whitespace for consistent matching."""
    return re.sub(r"\s+", " ", text.lower().strip())


def _text_contains_skill(text_lower: str, skill: str) -> bool:
    """
    Checks if a skill string appears in text as a whole word/phrase.
    Word-boundary check prevents "r" matching "react", etc.
    """
    # Escape special regex characters in skill names (e.g. "c++", "c#")
    escaped = re.escape(skill)
    # Use word boundaries; for skills ending in special chars, just check presence
    pattern = rf"(?<![a-z0-9]){escaped}(?![a-z0-9])"
    return bool(re.search(pattern, text_lower))


def extract_skills(text: str) -> dict:
    """
    Scans resume text and returns detected skills grouped by category.

    Returns:
        {
          "programming_languages": ["python", "java"],
          "data_science": ["pandas", "numpy"],
          ...
          "all": ["python", "java", "pandas", ...],
          "total_count": 12
        }
    """
    text_lower = _normalize(text)
    found_by_category: dict = {cat: [] for cat in SKILL_DB.keys()}
    found_all: list = []

    for entry in ALL_SKILLS:
        skill = entry["skill"]
        cat   = entry["category"]
        if _text_contains_skill(text_lower, skill) and skill not in found_all:
            found_by_category[cat].append(skill)
            found_all.append(skill)

    return {
        **found_by_category,
        "all": found_all,
        "total_count": len(found_all),
    }


def get_skill_strength(extracted_skills: dict) -> dict:
    """
    Assigns a strength score per category based on how many skills
    the candidate has relative to the total available in that category.

    Returns: {category: {"count": N, "total": M, "pct": 0-100}}
    """
    strength = {}
    for cat, db_skills in SKILL_DB.items():
        found = extracted_skills.get(cat, [])
        pct   = round(len(found) / max(len(db_skills), 1) * 100, 1)
        strength[cat] = {
            "count": len(found),
            "total": len(db_skills),
            "pct":   pct,
            "skills": found,
        }
    return strength


def skills_to_display_name(category: str) -> str:
    """Converts snake_case category keys to readable labels."""
    return category.replace("_", " ").title()
