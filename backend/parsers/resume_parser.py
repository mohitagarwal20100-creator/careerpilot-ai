"""
resume_parser.py
----------------
Extracts structured fields from raw resume text.
Uses regex patterns — no LLMs, no external APIs.

Design choice: regex over ML for parsing because:
- Resumes follow semi-structured patterns (email, phone, URLs)
- Regex is 100% deterministic and explainable
- Zero latency, zero cost
- Easy to extend with new patterns
"""

import re
from pypdf import PdfReader
import io


# ── PDF Text Extraction ─────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_file) -> str:
    """
    Reads a PDF file object and returns all text joined across pages.
    pdf_file: a file-like object (from st.file_uploader or open())
    """
    try:
        reader = PdfReader(pdf_file)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n".join(pages).strip()
    except Exception as e:
        return f"ERROR: Could not read PDF — {e}"


# ── Field Extraction Patterns ───────────────────────────────────────────────

EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE    = re.compile(r"(?:\+?91[\s\-]?)?[6-9]\d{9}|(?:\+?1[\s\-]?)?\(?\d{3}\)?[\s\-]?\d{3}[\s\-]?\d{4}")
LINKEDIN_RE = re.compile(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[A-Za-z0-9\-_%]+")
GITHUB_RE   = re.compile(r"(?:https?://)?(?:www\.)?github\.com/[A-Za-z0-9\-_%]+")

# Education keywords to anchor section detection
EDU_KEYWORDS = [
    "b.tech", "b.e.", "b.sc", "bsc", "b.com", "bca", "bba",
    "m.tech", "m.sc", "msc", "mca", "mba", "m.e.",
    "bachelor", "master", "ph.d", "phd", "doctorate",
    "university", "college", "institute", "iit", "nit", "bits",
    "engineering", "computer science", "information technology",
]

SECTION_HEADERS = [
    "education", "experience", "skills", "projects", "certifications",
    "achievements", "publications", "summary", "objective", "profile",
    "work experience", "internship", "training", "languages", "interests",
]


def extract_email(text: str) -> str:
    match = EMAIL_RE.search(text)
    return match.group(0) if match else ""


def extract_phone(text: str) -> str:
    match = PHONE_RE.search(text)
    return match.group(0) if match else ""


def extract_linkedin(text: str) -> str:
    match = LINKEDIN_RE.search(text)
    return match.group(0) if match else ""


def extract_github(text: str) -> str:
    match = GITHUB_RE.search(text)
    return match.group(0) if match else ""


def extract_name(text: str) -> str:
    """
    Heuristic: The name is usually one of the first 1-3 non-empty lines,
    contains only alphabets and spaces, and is longer than 3 chars.
    We skip lines that look like section headers or contain special chars.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    for line in lines[:6]:
        # Skip lines that are obviously not names
        if any(c in line for c in ["@", "http", "+", "|", "/", "\\", ":", "."]):
            continue
        if len(line) < 3 or len(line) > 60:
            continue
        if re.search(r"\d", line):
            continue
        lower = line.lower()
        if any(kw in lower for kw in SECTION_HEADERS + EDU_KEYWORDS):
            continue
        # Looks like a name
        if re.match(r"^[A-Za-z\s\.\-]+$", line):
            return line.strip()
    return "Not detected"


def extract_education(text: str) -> list:
    """
    Finds lines containing education keywords and returns them as a list.
    Deduplicates and trims noise.
    """
    results = []
    lines = text.split("\n")
    for i, line in enumerate(lines):
        lower = line.lower()
        if any(kw in lower for kw in EDU_KEYWORDS):
            clean = line.strip()
            if 5 < len(clean) < 200 and clean not in results:
                results.append(clean)
    return results[:6]  # cap at 6 entries


def extract_sections(text: str) -> dict:
    """
    Splits resume text into named sections by detecting section headers.
    Returns a dict: {section_name: section_text}
    """
    sections = {}
    current = "header"
    lines = text.split("\n")
    buffer = []

    for line in lines:
        clean = line.strip()
        lower = clean.lower()
        matched_header = None
        for h in SECTION_HEADERS:
            # A line is a header if it IS the keyword (possibly with punctuation)
            if re.match(rf"^{re.escape(h)}[\s:]*$", lower):
                matched_header = h
                break
        if matched_header:
            sections[current] = "\n".join(buffer).strip()
            current = matched_header
            buffer = []
        else:
            buffer.append(clean)

    sections[current] = "\n".join(buffer).strip()
    return sections


def parse_resume(pdf_file) -> dict:
    """
    Master function: extracts all fields and returns a structured dict.
    This is the only function the rest of the app needs to call.
    """
    text = extract_text_from_pdf(pdf_file)
    if text.startswith("ERROR"):
        return {"error": text, "raw_text": ""}

    sections = extract_sections(text)

    return {
        "raw_text": text,
        "sections": sections,
        "name":     extract_name(text),
        "email":    extract_email(text),
        "phone":    extract_phone(text),
        "linkedin": extract_linkedin(text),
        "github":   extract_github(text),
        "education": extract_education(text),
        "word_count": len(text.split()),
        "char_count": len(text),
    }
