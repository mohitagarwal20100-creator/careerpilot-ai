"""
question_gen.py
---------------
Generates interview questions from detected skills + target role.
Uses a local JSON dataset — zero external API calls.

Design: Questions are indexed by skill and difficulty level.
The generator picks questions based on skill intersection between
the candidate's profile and the target role, prioritizing
required role skills the candidate already knows (likely to be tested).
"""

import json
import os
import random

_QDB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "questions", "interview_questions.json"
)

with open(_QDB_PATH) as f:
    QUESTION_DB: dict = json.load(f)

LEVELS = ["beginner", "intermediate", "advanced"]


def _find_matching_keys(skills: list) -> list:
    """Returns question DB keys that match any of the candidate's skills."""
    db_keys = list(QUESTION_DB.keys())
    matches = []
    for skill in skills:
        for key in db_keys:
            if key in skill.lower() or skill.lower() in key:
                if key not in matches:
                    matches.append(key)
    return matches


def generate_questions(
    candidate_skills: list,
    target_role: str = None,
    level: str = "intermediate",
    count: int = 10,
) -> dict:
    """
    Generates interview questions relevant to the candidate's skills.

    Args:
        candidate_skills: list of detected skills
        target_role: optional role to bias question selection
        level: "beginner" | "intermediate" | "advanced"
        count: how many questions to return

    Returns:
        {
          "questions": [{"skill": "python", "level": "intermediate", "question": "..."}],
          "skills_covered": ["python", "machine learning"],
          "total_found": 10,
        }
    """
    if level not in LEVELS:
        level = "intermediate"

    matching_keys = _find_matching_keys(candidate_skills)

    if not matching_keys:
        # Fallback: return generic questions from any available key
        matching_keys = list(QUESTION_DB.keys())[:3]

    questions = []
    skills_covered = []

    for key in matching_keys:
        skill_qs = QUESTION_DB.get(key, {})
        level_qs = skill_qs.get(level, [])
        # Also grab adjacent level for variety
        if level == "intermediate":
            level_qs += skill_qs.get("beginner", [])[:1]

        for q in level_qs:
            questions.append({
                "skill":    key.title(),
                "level":    level.title(),
                "question": q,
            })
        if level_qs:
            skills_covered.append(key.title())

    # Shuffle so it doesn't always show the same order
    random.shuffle(questions)

    return {
        "questions":      questions[:count],
        "skills_covered": skills_covered,
        "total_found":    len(questions),
        "level":          level.title(),
    }


def generate_quick_quiz(candidate_skills: list) -> list:
    """
    Returns a 5-question mixed-level quiz from the candidate's top skills.
    Good for a 'Quick Prep' feature.
    """
    questions = []
    keys = _find_matching_keys(candidate_skills)[:5]
    for key in keys:
        all_q = []
        for lvl in LEVELS:
            lvl_qs = QUESTION_DB.get(key, {}).get(lvl, [])
            all_q.extend([(q, lvl, key) for q in lvl_qs])
        if all_q:
            q, lvl, k = random.choice(all_q)
            questions.append({"skill": k.title(), "level": lvl.title(), "question": q})
    return questions
