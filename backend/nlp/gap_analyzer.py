"""
gap_analyzer.py
---------------
Compares a candidate's skills against role requirements.
Uses set operations — no ML needed for this feature.

Gap = required_skills - candidate_skills
"""

import json
import os

_ROLES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "roles", "role_templates.json"
)

with open(_ROLES_PATH) as f:
    ROLE_TEMPLATES: dict = json.load(f)


def get_available_roles() -> list:
    return list(ROLE_TEMPLATES.keys())


def analyze_gap(candidate_skills: list, target_role: str) -> dict:
    """
    Compares candidate skills against a target role's requirements.

    Args:
        candidate_skills: flat list of detected skills (lowercase)
        target_role: key from ROLE_TEMPLATES

    Returns:
        {
          "role": "Data Scientist",
          "required": [...],
          "preferred": [...],
          "has_required": [...],
          "missing_required": [...],
          "has_preferred": [...],
          "missing_preferred": [...],
          "readiness_pct": 65.0,
          "description": "...",
          "avg_salary": "...",
        }
    """
    if target_role not in ROLE_TEMPLATES:
        return {"error": f"Role '{target_role}' not found"}

    template = ROLE_TEMPLATES[target_role]
    candidate_set = {s.lower() for s in candidate_skills}

    required  = [s.lower() for s in template["required"]]
    preferred = [s.lower() for s in template["preferred"]]

    has_required     = [s for s in required  if s in candidate_set]
    missing_required = [s for s in required  if s not in candidate_set]
    has_preferred    = [s for s in preferred if s in candidate_set]
    missing_preferred= [s for s in preferred if s not in candidate_set]

    # Readiness = weighted: required skills count 70%, preferred 30%
    req_score  = len(has_required)  / max(len(required),  1)
    pref_score = len(has_preferred) / max(len(preferred), 1)
    readiness  = round((0.7 * req_score + 0.3 * pref_score) * 100, 1)

    return {
        "role":              target_role,
        "required":          required,
        "preferred":         preferred,
        "has_required":      has_required,
        "missing_required":  missing_required,
        "has_preferred":     has_preferred,
        "missing_preferred": missing_preferred,
        "readiness_pct":     readiness,
        "description":       template.get("description", ""),
        "avg_salary":        template.get("avg_salary", ""),
    }
