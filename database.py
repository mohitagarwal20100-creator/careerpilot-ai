"""
database.py
-----------
SQLite persistence layer for CareerPilot AI.

Stores resume analysis results so users can compare sessions
and track progress over time. No server needed — SQLite is file-based.

Design: One table per entity. Simple CRUD functions only.
The app can work fully without the DB (in-session state) but the DB
enables the version comparison and history features.
"""

import sqlite3
import json
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "careerpilot.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS resume_analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            filename    TEXT,
            name        TEXT,
            email       TEXT,
            ats_score   INTEGER,
            grade       TEXT,
            skills_json TEXT,
            category    TEXT,
            confidence  REAL,
            word_count  INTEGER
        );

        CREATE TABLE IF NOT EXISTS comparisons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at  TEXT NOT NULL,
            resume_a_id INTEGER,
            resume_b_id INTEGER,
            FOREIGN KEY(resume_a_id) REFERENCES resume_analyses(id),
            FOREIGN KEY(resume_b_id) REFERENCES resume_analyses(id)
        );
    """)
    conn.commit()
    conn.close()


def save_analysis(
    filename: str,
    parsed: dict,
    ats: dict,
    skills: dict,
    classification: dict,
) -> int:
    """Saves a resume analysis and returns the new row ID."""
    conn = get_connection()
    cur  = conn.cursor()
    cur.execute("""
        INSERT INTO resume_analyses
          (created_at, filename, name, email, ats_score, grade,
           skills_json, category, confidence, word_count)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        filename,
        parsed.get("name", ""),
        parsed.get("email", ""),
        ats.get("total", 0),
        ats.get("grade", ""),
        json.dumps(skills.get("all", [])),
        classification.get("category", ""),
        classification.get("confidence", 0.0),
        parsed.get("word_count", 0),
    ))
    conn.commit()
    row_id = cur.lastrowid
    conn.close()
    return row_id


def get_recent_analyses(limit: int = 10) -> list:
    """Returns the most recent analyses for the history panel."""
    conn = get_connection()
    rows = conn.execute("""
        SELECT id, created_at, filename, name, ats_score, grade, category
        FROM resume_analyses
        ORDER BY id DESC
        LIMIT ?
    """, (limit,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_analysis_by_id(row_id: int) -> dict:
    conn = get_connection()
    row  = conn.execute(
        "SELECT * FROM resume_analyses WHERE id = ?", (row_id,)
    ).fetchone()
    conn.close()
    if row:
        d = dict(row)
        d["skills"] = json.loads(d.get("skills_json") or "[]")
        return d
    return {}


# Initialize on import
init_db()
