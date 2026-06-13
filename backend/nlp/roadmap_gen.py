"""
roadmap_gen.py
--------------
Generates a personalized learning roadmap based on current vs missing skills.

Strategy (rule-based, no LLMs):
  1. Determine candidate level from skill count + quality
  2. Select role roadmap from JSON templates
  3. Filter steps: skip skills already acquired, prioritize missing required skills
  4. Return ordered learning path with time estimates
"""

import json
import os

_ROADMAP_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "roadmaps", "roadmaps.json"
)

with open(_ROADMAP_PATH) as f:
    ROADMAPS: dict = json.load(f)


def _infer_level(candidate_skills: list, readiness_pct: float) -> str:
    """
    Infers candidate level from skill count and gap analysis readiness.
    Returns "beginner" | "intermediate" | "advanced"
    """
    count = len(candidate_skills)
    if readiness_pct >= 65 or count >= 15:
        return "advanced"
    elif readiness_pct >= 35 or count >= 7:
        return "intermediate"
    else:
        return "beginner"


def _get_fallback_roadmap(target_role: str, level: str) -> dict:
    """Returns a generic roadmap when specific role data isn't available."""
    return {
        "title": f"{level.title()} Path for {target_role}",
        "steps": [
            {"skill": "Core fundamentals for your role", "resources": ["Google", "YouTube"], "time": "4 weeks"},
            {"skill": "Build hands-on projects", "resources": ["GitHub", "Kaggle"], "time": "4 weeks"},
            {"skill": "Learn industry tools", "resources": ["Official docs", "Tutorials"], "time": "3 weeks"},
        ]
    }


def generate_roadmap(
    candidate_skills: list,
    target_role: str,
    missing_required: list,
    readiness_pct: float = 0.0,
    force_level: str = None,
) -> dict:
    """
    Generates a personalized, filtered learning roadmap.

    Args:
        candidate_skills: skills the candidate already has
        target_role: target role name
        missing_required: skills the candidate needs for the role
        readiness_pct: from gap analysis (0-100)
        force_level: override auto-detection ("beginner"|"intermediate"|"advanced")

    Returns:
        {
          "level": "intermediate",
          "role": "Data Scientist",
          "phase_title": "Core ML Phase (3-6 months)",
          "steps": [
            {
              "skill": "Scikit-learn",
              "resources": ["...", "..."],
              "time": "3 weeks",
              "already_have": False,
              "priority": "high",
            }
          ],
          "estimated_total_weeks": 14,
          "next_milestone": "...",
        }
    """
    level = force_level or _infer_level(candidate_skills, readiness_pct)

    # Normalize role name for roadmap lookup
    role_key = None
    for key in ROADMAPS.keys():
        if key.lower() in target_role.lower() or target_role.lower() in key.lower():
            role_key = key
            break

    # Fall back to closest match or generic
    if role_key is None:
        if "data" in target_role.lower():
            role_key = "Data Scientist"
        elif "web" in target_role.lower():
            role_key = "Web Developer"
        elif "devops" in target_role.lower() or "cloud" in target_role.lower():
            role_key = "DevOps Engineer"
        elif "ml" in target_role.lower() or "machine" in target_role.lower():
            role_key = "ML Engineer"

    if role_key and role_key in ROADMAPS:
        phase_data = ROADMAPS[role_key].get(level, ROADMAPS[role_key]["beginner"])
    else:
        phase_data = _get_fallback_roadmap(target_role, level)

    candidate_lower = {s.lower() for s in candidate_skills}
    missing_lower   = {s.lower() for s in missing_required}

    enriched_steps = []
    total_weeks    = 0

    for step in phase_data["steps"]:
        skill_lower   = step["skill"].lower()
        already_have  = any(s in skill_lower or skill_lower in s for s in candidate_lower)
        is_priority   = any(m in skill_lower or skill_lower in m for m in missing_lower)

        # Parse time to weeks for total estimate
        time_str = step.get("time", "2 weeks")
        try:
            weeks = int(time_str.split()[0])
        except (ValueError, IndexError):
            weeks = 2

        if not already_have:
            total_weeks += weeks

        enriched_steps.append({
            **step,
            "already_have": already_have,
            "priority":     "high" if is_priority else ("done" if already_have else "normal"),
            "weeks":        weeks,
        })

    # Sort: high priority first, done last
    priority_order = {"high": 0, "normal": 1, "done": 2}
    enriched_steps.sort(key=lambda x: priority_order.get(x["priority"], 1))

    # Next milestone = first high-priority skill
    next_milestone = next(
        (s["skill"] for s in enriched_steps if s["priority"] == "high"),
        enriched_steps[0]["skill"] if enriched_steps else "Start learning!"
    )

    return {
        "level":                  level,
        "role":                   target_role,
        "phase_title":            phase_data.get("title", f"{level.title()} Phase"),
        "steps":                  enriched_steps,
        "estimated_total_weeks":  total_weeks,
        "next_milestone":         next_milestone,
    }
