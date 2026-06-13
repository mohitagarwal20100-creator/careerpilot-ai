"""
recommender.py
--------------
Recommends career paths based on extracted skills + classification result.

Strategy (no LLMs):
  1. For every role in our template database, compute a compatibility score
     using weighted skill overlap.
  2. Required skills contribute 70% of the score; preferred skills 30%.
  3. Sort roles by compatibility and return top-N.

This is content-based filtering — the same family of algorithms used
by Netflix and Spotify, applied to role-skill matching.
"""

import json
import os
from backend.nlp.gap_analyzer import ROLE_TEMPLATES, analyze_gap


def recommend_roles(candidate_skills: list, top_n: int = 5) -> list:
    """
    Scores all roles against the candidate's skill profile.

    Returns a sorted list of role recommendation dicts:
      [
        {
          "role": "Data Scientist",
          "readiness_pct": 72.0,
          "matching_required": 7,
          "total_required": 10,
          "missing_required": [...],
          "description": "...",
          "avg_salary": "...",
          "fit_label": "Strong Fit",
        },
        ...
      ]
    """
    results = []

    for role in ROLE_TEMPLATES.keys():
        gap = analyze_gap(candidate_skills, role)

        pct = gap["readiness_pct"]
        if pct >= 70:   fit_label = "🔥 Strong Fit"
        elif pct >= 45: fit_label = "✅ Good Fit"
        elif pct >= 25: fit_label = "🌱 Emerging Fit"
        else:           fit_label = "📚 Needs Work"

        results.append({
            "role":              role,
            "readiness_pct":     pct,
            "matching_required": len(gap["has_required"]),
            "total_required":    len(gap["required"]),
            "missing_required":  gap["missing_required"][:5],  # top 5 gaps
            "description":       gap["description"],
            "avg_salary":        gap["avg_salary"],
            "fit_label":         fit_label,
        })

    # Sort by readiness descending
    results.sort(key=lambda x: -x["readiness_pct"])
    return results[:top_n]


def get_career_insights(candidate_skills: list, classification: dict) -> dict:
    """
    Combines classification + recommendation for a full insight package.

    Returns:
      {
        "primary_category": "Data Science",
        "confidence": 87.3,
        "top_roles": [...],
        "skill_count": 15,
        "top_skill_areas": [...],
      }
    """
    from backend.nlp.skill_extractor import SKILL_DB, skills_to_display_name

    # Which categories does the candidate have the most skills in?
    from backend.nlp.skill_extractor import extract_skills
    skill_data = {}
    for cat, db_skills in SKILL_DB.items():
        candidate_set = {s.lower() for s in candidate_skills}
        found = [s for s in db_skills if s in candidate_set]
        if found:
            skill_data[skills_to_display_name(cat)] = len(found)

    top_areas = sorted(skill_data.items(), key=lambda x: -x[1])[:4]

    return {
        "primary_category": classification.get("category", "Unknown"),
        "confidence":       classification.get("confidence", 0),
        "top_roles":        recommend_roles(candidate_skills, top_n=5),
        "skill_count":      len(candidate_skills),
        "top_skill_areas":  top_areas,
    }
