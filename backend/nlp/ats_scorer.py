"""
ats_scorer.py
-------------
Generates an ATS (Applicant Tracking System) score out of 100.

Scoring philosophy:
  Real ATS systems parse resumes and check for completeness, keywords,
  formatting signals, and section presence. We replicate this with a
  transparent weighted rubric — every point deduction is explainable.

Score Breakdown (100 points total):
  Contact Info       20 pts  (email 8, phone 6, linkedin 4, github 2)
  Education          15 pts  (degree detected)
  Skills Section     20 pts  (number of skills detected, capped)
  Projects Section   15 pts  (project section present + quality signals)
  Experience         10 pts  (experience/internship section present)
  Resume Length      10 pts  (word count in sweet spot 300-700)
  GitHub             5 pts   (already in contact, bonus for project links)
  LinkedIn           5 pts   (already in contact, bonus for profile)
"""

import re


# ── Scoring Constants ────────────────────────────────────────────────────────

MAX_SCORE = 100

WEIGHTS = {
    "email":      8,
    "phone":      6,
    "linkedin":   6,
    "github":     5,
    "education":  15,
    "skills":     20,
    "projects":   15,
    "experience": 10,
    "word_count": 10,
    "summary":    5,
}


def score_contact(parsed: dict) -> tuple[int, list]:
    """Returns (points_earned, feedback_list)"""
    pts = 0
    feedback = []

    if parsed.get("email"):
        pts += WEIGHTS["email"]
    else:
        feedback.append(("❌", "Email address missing — ATS systems require this"))

    if parsed.get("phone"):
        pts += WEIGHTS["phone"]
    else:
        feedback.append(("⚠️", "Phone number not detected"))

    if parsed.get("linkedin"):
        pts += WEIGHTS["linkedin"]
    else:
        feedback.append(("💡", "Add a LinkedIn URL to stand out"))

    if parsed.get("github"):
        pts += WEIGHTS["github"]
    else:
        feedback.append(("💡", "GitHub link boosts credibility for tech roles"))

    return pts, feedback


def score_education(parsed: dict) -> tuple[int, list]:
    edu = parsed.get("education", [])
    feedback = []
    if edu:
        pts = WEIGHTS["education"]
    else:
        pts = 0
        feedback.append(("❌", "Education section not detected — add your degree & university"))
    return pts, feedback


def score_skills(skills: dict) -> tuple[int, list]:
    """
    Skills scoring is progressive:
      0  skills → 0 pts
      5  skills → 8 pts
      10 skills → 14 pts
      15+ skills → 20 pts (full marks)
    """
    count = skills.get("total_count", 0)
    feedback = []

    if count == 0:
        pts = 0
        feedback.append(("❌", "No recognizable skills detected — add a dedicated Skills section"))
    elif count < 5:
        pts = 6
        feedback.append(("⚠️", f"Only {count} skills found — list more tools and technologies"))
    elif count < 10:
        pts = 12
        feedback.append(("⚠️", f"{count} skills detected — aim for 12+ to rank higher"))
    elif count < 15:
        pts = 16
        feedback.append(("✅", f"{count} skills found — good coverage"))
    else:
        pts = WEIGHTS["skills"]
        feedback.append(("✅", f"{count} skills detected — excellent skill coverage"))

    return pts, feedback


def score_sections(parsed: dict) -> tuple[int, list]:
    """Check for presence of projects, experience, and summary sections."""
    sections = {k.lower(): v for k, v in parsed.get("sections", {}).items()}
    raw_text = parsed.get("raw_text", "").lower()
    pts = 0
    feedback = []

    # Projects
    has_projects = (
        "projects" in sections
        or "project" in raw_text
    )
    if has_projects:
        pts += WEIGHTS["projects"]
        feedback.append(("✅", "Projects section detected"))
    else:
        feedback.append(("❌", "No Projects section found — add 2-3 projects with impact metrics"))

    # Experience
    has_exp = any(k in sections for k in ["experience", "work experience", "internship", "training"])
    if has_exp:
        pts += WEIGHTS["experience"]
        feedback.append(("✅", "Experience/Internship section detected"))
    else:
        feedback.append(("⚠️", "No Experience section — add internships or freelance work"))

    # Summary/Objective
    has_summary = any(k in sections for k in ["summary", "objective", "profile", "about"])
    if has_summary:
        pts += WEIGHTS["summary"]
        feedback.append(("✅", "Professional summary detected"))
    else:
        feedback.append(("💡", "Add a 2-line professional summary at the top"))

    return pts, feedback


def score_length(parsed: dict) -> tuple[int, list]:
    word_count = parsed.get("word_count", 0)
    feedback = []

    if 300 <= word_count <= 800:
        pts = WEIGHTS["word_count"]
        feedback.append(("✅", f"Resume length is ideal ({word_count} words)"))
    elif word_count < 150:
        pts = 2
        feedback.append(("❌", f"Resume too short ({word_count} words) — expand your content"))
    elif word_count < 300:
        pts = 6
        feedback.append(("⚠️", f"Resume a bit thin ({word_count} words) — add more detail"))
    elif word_count <= 1200:
        pts = 8
        feedback.append(("⚠️", f"Resume slightly long ({word_count} words) — consider trimming"))
    else:
        pts = 4
        feedback.append(("❌", f"Resume too long ({word_count} words) — keep it to 1 page for freshers"))

    return pts, feedback


def compute_ats_score(parsed: dict, skills: dict) -> dict:
    """
    Master scoring function. Returns full breakdown + total score.

    Returns:
      {
        "total": 72,
        "breakdown": { "contact": 20, "education": 15, ... },
        "feedback": [("✅", "msg"), ("❌", "msg"), ...],
        "grade": "B+"
      }
    """
    all_feedback = []
    breakdown = {}

    contact_pts, fb = score_contact(parsed)
    breakdown["Contact Info"]  = contact_pts
    all_feedback.extend(fb)

    edu_pts, fb = score_education(parsed)
    breakdown["Education"] = edu_pts
    all_feedback.extend(fb)

    skill_pts, fb = score_skills(skills)
    breakdown["Skills"] = skill_pts
    all_feedback.extend(fb)

    section_pts, fb = score_sections(parsed)
    breakdown["Sections"] = section_pts
    all_feedback.extend(fb)

    length_pts, fb = score_length(parsed)
    breakdown["Resume Length"] = length_pts
    all_feedback.extend(fb)

    total = min(sum(breakdown.values()), MAX_SCORE)

    # Grade
    if total >= 90:   grade = "A+"
    elif total >= 80: grade = "A"
    elif total >= 70: grade = "B+"
    elif total >= 60: grade = "B"
    elif total >= 50: grade = "C+"
    elif total >= 40: grade = "C"
    else:             grade = "D"

    return {
        "total":     total,
        "breakdown": breakdown,
        "feedback":  all_feedback,
        "grade":     grade,
        "max":       MAX_SCORE,
    }
