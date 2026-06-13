"""
app.py — CareerPilot AI
=======================
Entry point for the Streamlit application.
Run with: streamlit run app.py

All 11 features wired together with a Pinterest/Gen-Z aesthetic.
No external LLM APIs. No LangChain. Pure Python + traditional NLP/ML.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import json
import random

# ── Page config must be FIRST ────────────────────────────────────────────────
st.set_page_config(
    page_title="CareerPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Backend imports ──────────────────────────────────────────────────────────
from backend.parsers.resume_parser  import parse_resume
from backend.nlp.skill_extractor    import extract_skills, get_skill_strength, skills_to_display_name
from backend.nlp.ats_scorer         import compute_ats_score
from backend.ml.classifier          import classify_resume
from backend.ml.matcher             import compute_match
from backend.nlp.gap_analyzer       import analyze_gap, get_available_roles
from backend.nlp.recommender        import recommend_roles, get_career_insights
from backend.nlp.question_gen       import generate_questions, generate_quick_quiz
from backend.nlp.roadmap_gen        import generate_roadmap
from database                       import save_analysis, get_recent_analyses

# ── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
  --bg:        #0a0a0f;
  --surface:   #12121a;
  --card:      #1a1a28;
  --border:    #2a2a3d;
  --accent1:   #b48bff;   /* lavender purple */
  --accent2:   #ff7eb3;   /* hot pink */
  --accent3:   #7fdbca;   /* mint */
  --accent4:   #ffd166;   /* golden yellow */
  --text:      #e8e6f0;
  --muted:     #7a788a;
  --grad1: linear-gradient(135deg, #b48bff 0%, #ff7eb3 100%);
  --grad2: linear-gradient(135deg, #7fdbca 0%, #b48bff 100%);
  --grad3: linear-gradient(135deg, #ffd166 0%, #ff7eb3 100%);
}

/* ── Base reset ── */
html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1rem 2rem 2rem 2rem !important; max-width: 100% !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: var(--surface) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label { color: var(--text) !important; }

/* ── Cards ── */
.cp-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 20px;
  padding: 1.5rem;
  margin-bottom: 1rem;
  transition: transform .2s, box-shadow .2s;
}
.cp-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 8px 32px rgba(180,139,255,0.15);
}

/* ── Hero ── */
.cp-hero {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 24px;
  padding: 2.5rem 2rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.cp-hero::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 220px; height: 220px;
  background: radial-gradient(circle, rgba(180,139,255,0.18) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.cp-hero::after {
  content: '';
  position: absolute;
  bottom: -40px; left: -40px;
  width: 160px; height: 160px;
  background: radial-gradient(circle, rgba(255,126,179,0.13) 0%, transparent 70%);
  border-radius: 50%;
  pointer-events: none;
}
.cp-hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: 2.6rem;
  font-weight: 800;
  background: var(--grad1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin: 0 0 .4rem 0;
  line-height: 1.1;
}
.cp-hero p {
  color: var(--muted);
  font-size: 1.05rem;
  margin: 0;
}

/* ── Score ring ── */
.score-ring {
  text-align: center;
  padding: 1.2rem;
}
.score-num {
  font-family: 'Syne', sans-serif;
  font-size: 3.5rem;
  font-weight: 800;
  background: var(--grad1);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  line-height: 1;
}
.score-label {
  font-size: .85rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-top: .3rem;
}
.grade-badge {
  display: inline-block;
  padding: .25rem .9rem;
  border-radius: 999px;
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: 1.1rem;
  background: var(--grad1);
  color: #fff;
  margin-top: .5rem;
}

/* ── Skill pills ── */
.skill-pill {
  display: inline-block;
  padding: .25rem .75rem;
  border-radius: 999px;
  font-size: .8rem;
  font-weight: 500;
  margin: .2rem .15rem;
  border: 1px solid;
  cursor: default;
  transition: transform .15s;
}
.skill-pill:hover { transform: scale(1.05); }
.skill-found   { background: rgba(127,219,202,.12); border-color: #7fdbca; color: #7fdbca; }
.skill-missing { background: rgba(255,126,179,.10); border-color: #ff7eb3; color: #ff7eb3; }
.skill-neutral { background: rgba(180,139,255,.10); border-color: #b48bff; color: #b48bff; }

/* ── Stat tile ── */
.stat-tile {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.2rem 1rem;
  text-align: center;
}
.stat-tile .val {
  font-family: 'Syne', sans-serif;
  font-size: 2rem;
  font-weight: 800;
  color: var(--accent1);
}
.stat-tile .lbl {
  font-size: .78rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
}

/* ── Section heading ── */
.cp-section-head {
  font-family: 'Syne', sans-serif;
  font-size: 1.3rem;
  font-weight: 700;
  color: var(--text);
  margin: 1.5rem 0 .8rem 0;
  display: flex;
  align-items: center;
  gap: .5rem;
}

/* ── Progress bar override ── */
.stProgress > div > div > div > div {
  background: var(--grad1) !important;
  border-radius: 999px !important;
}

/* ── Feedback items ── */
.fb-item {
  padding: .5rem .8rem;
  border-radius: 10px;
  margin: .3rem 0;
  font-size: .88rem;
  background: rgba(255,255,255,.03);
  border-left: 3px solid var(--border);
}

/* ── Roadmap step ── */
.roadmap-step {
  display: flex;
  align-items: flex-start;
  gap: 1rem;
  padding: .9rem 1rem;
  border-radius: 14px;
  margin: .5rem 0;
  background: var(--card);
  border: 1px solid var(--border);
  transition: all .2s;
}
.roadmap-step:hover { border-color: var(--accent1); }
.roadmap-step.done  { opacity: .5; }
.roadmap-step.high  { border-color: #ff7eb3; }
.step-dot {
  width: 34px; height: 34px;
  border-radius: 50%;
  display: flex; align-items: center; justify-content: center;
  font-size: 1rem; flex-shrink: 0;
  font-weight: 700;
}
.dot-high   { background: rgba(255,126,179,.2); color: #ff7eb3; }
.dot-normal { background: rgba(180,139,255,.2); color: #b48bff; }
.dot-done   { background: rgba(127,219,202,.2); color: #7fdbca; }
.step-info .step-title { font-weight: 600; color: var(--text); }
.step-info .step-time  { font-size: .78rem; color: var(--muted); }
.step-info .step-res   { font-size: .78rem; color: var(--accent3); margin-top: .2rem; }

/* ── Role card ── */
.role-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1rem 1.2rem;
  margin: .4rem 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  transition: all .2s;
}
.role-card:hover { border-color: var(--accent1); transform: translateX(4px); }
.role-name { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; }
.role-fit  { font-size: .85rem; }
.role-bar  { width: 100%; height: 6px; background: var(--border); border-radius: 999px; margin-top: .4rem; }
.role-bar-fill { height: 100%; border-radius: 999px; background: var(--grad1); }

/* ── Question card ── */
.q-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-left: 4px solid var(--accent1);
  border-radius: 0 14px 14px 0;
  padding: .9rem 1.1rem;
  margin: .5rem 0;
}
.q-meta { font-size: .75rem; color: var(--muted); margin-bottom: .3rem; }
.q-text { font-size: .95rem; color: var(--text); font-weight: 500; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
  background: var(--card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: 16px !important;
}
[data-testid="stFileUploader"]:hover {
  border-color: var(--accent1) !important;
}

/* ── Buttons ── */
.stButton > button {
  background: var(--grad1) !important;
  color: #fff !important;
  border: none !important;
  border-radius: 12px !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: .9rem !important;
  padding: .55rem 1.4rem !important;
  transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover {
  opacity: .88 !important;
  transform: translateY(-1px) !important;
}

/* ── Select boxes ── */
[data-testid="stSelectbox"] > div { background: var(--card) !important; border-color: var(--border) !important; border-radius: 12px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] button {
  color: var(--muted) !important;
  font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
  color: var(--accent1) !important;
  border-bottom-color: var(--accent1) !important;
}

/* ── Info / success / warning boxes ── */
.stAlert { border-radius: 14px !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; }

/* ── Comparison table ── */
.compare-row {
  display: grid;
  grid-template-columns: 1fr auto 1fr;
  gap: .5rem;
  align-items: center;
  padding: .6rem 0;
  border-bottom: 1px solid var(--border);
}
.compare-label { font-size: .8rem; color: var(--muted); text-align: center; }
.compare-val   { font-weight: 600; }
.c-win   { color: var(--accent3); }
.c-lose  { color: var(--accent2); }
.c-tie   { color: var(--accent4); }

/* ── Unique features badge ── */
.unique-badge {
  display: inline-block;
  background: var(--grad3);
  color: #1a1a28;
  font-family: 'Syne', sans-serif;
  font-weight: 700;
  font-size: .7rem;
  padding: .15rem .6rem;
  border-radius: 999px;
  text-transform: uppercase;
  letter-spacing: .06em;
  margin-left: .5rem;
  vertical-align: middle;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--surface); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent1); }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
def init_session():
    defaults = {
        "parsed":     None,
        "skills":     None,
        "ats":        None,
        "classified": None,
        "filename":   None,
        "gap":        None,
        "match":      None,
        "roadmap":    None,
        "parsed_b":   None,
        "skills_b":   None,
        "ats_b":      None,
        "filename_b": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_session()


# ── Helper: render skill pills ───────────────────────────────────────────────
def skill_pills(skills: list, cls: str = "skill-neutral") -> str:
    return "".join(f'<span class="skill-pill {cls}">{s}</span>' for s in skills)


# ── Helper: gauge chart ──────────────────────────────────────────────────────
def gauge_chart(value: float, title: str, color: str = "#b48bff") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={"x": [0,1], "y": [0,1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#7a788a",
                     "tickfont": {"color": "#7a788a", "size": 11}},
            "bar":  {"color": color, "thickness": .25},
            "bgcolor": "#1a1a28",
            "borderwidth": 0,
            "steps": [
                {"range": [0,  40], "color": "#1e1e2e"},
                {"range": [40, 70], "color": "#221e2e"},
                {"range": [70,100], "color": "#251e35"},
            ],
            "threshold": {
                "line":  {"color": "#ff7eb3", "width": 3},
                "thickness": .8,
                "value": 70,
            },
        },
        number={"suffix": "%", "font": {"color": color, "family": "Syne", "size": 36}},
        title={"text": title, "font": {"color": "#7a788a", "size": 13}},
    ))
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"t":30, "b":10, "l":10, "r":10},
        height=220,
    )
    return fig


# ── Helper: run full analysis ────────────────────────────────────────────────
def run_analysis(pdf_file, slot: str = "a"):
    with st.spinner("Parsing resume…"):
        parsed = parse_resume(pdf_file)
    with st.spinner("Extracting skills…"):
        skills = extract_skills(parsed["raw_text"])
    with st.spinner("Computing ATS score…"):
        ats    = compute_ats_score(parsed, skills)
    with st.spinner("Classifying resume…"):
        clf    = classify_resume(parsed["raw_text"])

    if slot == "a":
        st.session_state.parsed     = parsed
        st.session_state.skills     = skills
        st.session_state.ats        = ats
        st.session_state.classified = clf
        st.session_state.filename   = pdf_file.name
        try:
            save_analysis(pdf_file.name, parsed, ats, skills, clf)
        except Exception:
            pass
    else:
        st.session_state.parsed_b   = parsed
        st.session_state.skills_b   = skills
        st.session_state.ats_b      = ats
        st.session_state.filename_b = pdf_file.name


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="padding:.5rem 0 1.5rem 0">
      <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;
                  background:linear-gradient(135deg,#b48bff,#ff7eb3);
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                  background-clip:text;">🚀 CareerPilot AI</div>
      <div style="font-size:.75rem;color:#7a788a;margin-top:.2rem;">Resume Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        [
            "🏠  Dashboard",
            "📄  Resume Analyzer",
            "🎯  JD Matcher",
            "📊  Skill Gap",
            "🌟  Career Recommender",
            "💬  Interview Prep",
            "🗺️  Learning Roadmap",
            "⚖️  Resume Comparator",
            "🧬  Resume DNA",
            "💪  Strength Dashboard",
            "📈  History",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if st.session_state.parsed:
        p = st.session_state.parsed
        a = st.session_state.ats
        s = st.session_state.skills
        st.markdown(f"""
        <div style="background:#1a1a28;border:1px solid #2a2a3d;border-radius:14px;padding:1rem;">
          <div style="font-size:.75rem;color:#7a788a;text-transform:uppercase;letter-spacing:.08em;">Active Resume</div>
          <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1rem;margin:.3rem 0;">{p.get("name","Unknown")}</div>
          <div style="font-size:.8rem;color:#7a788a;">{st.session_state.filename}</div>
          <div style="margin-top:.7rem;display:flex;gap:.5rem;align-items:center;">
            <span style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                         background:linear-gradient(135deg,#b48bff,#ff7eb3);
                         -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
              {a["total"]}
            </span>
            <span style="font-size:.78rem;color:#7a788a;">ATS Score<br><strong style="color:#b48bff;">{a["grade"]}</strong></span>
          </div>
          <div style="font-size:.8rem;color:#7fdbca;margin-top:.4rem;">✦ {s["total_count"]} skills detected</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="position:absolute;bottom:1rem;left:1rem;right:1rem;font-size:.7rem;color:#3a3a50;text-align:center;">
      Built with Python · No LLMs · Open Source
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
if "Dashboard" in page:
    st.markdown("""
    <div class="cp-hero">
      <h1>CareerPilot AI 🚀</h1>
      <p>Your AI-powered resume intelligence platform. Upload a resume and get instant ATS scores,
         skill analysis, career recommendations, and a personalized roadmap — all without any LLMs.</p>
    </div>
    """, unsafe_allow_html=True)

    # Upload in hero
    col1, col2 = st.columns([2, 1])
    with col1:
        pdf = st.file_uploader("Drop your resume PDF here", type=["pdf"], key="hero_upload")
        if pdf and st.button("⚡ Analyze Now", key="hero_btn"):
            run_analysis(pdf)
            st.success("Analysis complete! Navigate the tabs to explore your results.")

    with col2:
        st.markdown("""
        <div class="cp-card" style="text-align:center;padding:1.5rem .5rem;">
          <div style="font-size:2rem">✦</div>
          <div style="font-family:'Syne',sans-serif;font-size:1rem;font-weight:700;
                      background:linear-gradient(135deg,#b48bff,#ff7eb3);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            11 AI Features
          </div>
          <div style="font-size:.78rem;color:#7a788a;margin-top:.3rem;">
            All powered by traditional ML —<br>transparent, fast, deployable
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Feature grid
    features = [
        ("📄", "Resume Parser",       "Extracts name, email, phone, education instantly"),
        ("🏅", "ATS Score",           "100-point weighted scoring with grade and feedback"),
        ("🔍", "Skill Extractor",     "200+ skills across 10 domains detected via taxonomy"),
        ("🤖", "ML Classifier",       "TF-IDF + Logistic Regression — predicts your role"),
        ("🎯", "JD Matcher",          "Cosine similarity matching with missing skill gap"),
        ("📊", "Skill Gap Analysis",  "Required vs possessed skills for target roles"),
        ("🌟", "Career Recommender",  "Content-based role scoring engine"),
        ("💬", "Interview Prep",      "Skill-specific Q&A from local dataset"),
        ("🗺️", "Learning Roadmap",   "Personalized beginner → advanced paths"),
        ("⚖️", "Resume Comparator",  "Side-by-side ATS and skill delta dashboard"),
        ("🧬", "Resume DNA",          "Unique skill fingerprint + personality radar"),
        ("📈", "Analysis History",    "SQLite-backed session history"),
    ]
    cols = st.columns(3)
    for i, (icon, title, desc) in enumerate(features):
        with cols[i % 3]:
            st.markdown(f"""
            <div class="cp-card" style="min-height:110px;">
              <div style="font-size:1.4rem;margin-bottom:.4rem;">{icon}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:.95rem;
                          color:#e8e6f0;margin-bottom:.3rem;">{title}</div>
              <div style="font-size:.78rem;color:#7a788a;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESUME ANALYZER
# ══════════════════════════════════════════════════════════════════════════════
elif "Resume Analyzer" in page:
    st.markdown('<div class="cp-section-head">📄 Resume Analyzer</div>', unsafe_allow_html=True)

    pdf = st.file_uploader("Upload resume PDF", type=["pdf"], key="analyzer_upload")
    if pdf:
        if st.button("🔍 Analyze Resume"):
            run_analysis(pdf)

    if not st.session_state.parsed:
        st.info("Upload a PDF resume above to begin analysis.")
        st.stop()

    p = st.session_state.parsed
    s = st.session_state.skills
    a = st.session_state.ats
    c = st.session_state.classified

    # ── Top stats row ──────────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">Overview</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (a["total"], "ATS Score"),
        (s["total_count"], "Skills Found"),
        (p["word_count"], "Words"),
        (len(p["education"]), "Edu. Entries"),
        (round(c["confidence"], 1), "ML Confidence %"),
    ]
    for col, (val, lbl) in zip([c1,c2,c3,c4,c5], metrics):
        col.markdown(f"""
        <div class="stat-tile">
          <div class="val">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two-column layout ──────────────────────────────────────────────────
    left, right = st.columns([1.2, 1])

    with left:
        # Contact info card
        st.markdown('<div class="cp-section-head">👤 Contact Information</div>', unsafe_allow_html=True)
        fields = [
            ("Name",     p.get("name","—"),     "🙋"),
            ("Email",    p.get("email","—"),    "📧"),
            ("Phone",    p.get("phone","—"),    "📱"),
            ("LinkedIn", p.get("linkedin","—"), "🔗"),
            ("GitHub",   p.get("github","—"),   "💻"),
        ]
        rows_html = ""
        for label, val, icon in fields:
            color = "#7fdbca" if val and val != "—" else "#ff7eb3"
            rows_html += f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:.5rem 0;border-bottom:1px solid #2a2a3d;">
              <span style="color:#7a788a;font-size:.85rem;">{icon} {label}</span>
              <span style="color:{color};font-size:.85rem;font-weight:500;
                           max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{val}</span>
            </div>"""
        st.markdown(f'<div class="cp-card">{rows_html}</div>', unsafe_allow_html=True)

        # Education
        if p["education"]:
            st.markdown('<div class="cp-section-head">🎓 Education</div>', unsafe_allow_html=True)
            edu_html = ""
            for e in p["education"][:4]:
                edu_html += f'<div style="padding:.4rem 0;border-bottom:1px solid #2a2a3d;font-size:.88rem;">{e}</div>'
            st.markdown(f'<div class="cp-card">{edu_html}</div>', unsafe_allow_html=True)

    with right:
        # ATS Score
        st.markdown('<div class="cp-section-head">🏅 ATS Score</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cp-card score-ring">
          <div class="score-num">{a["total"]}</div>
          <div class="score-label">out of 100</div>
          <div class="grade-badge">{a["grade"]}</div>
        </div>
        """, unsafe_allow_html=True)

        # ATS Breakdown
        st.markdown('<div class="cp-section-head">Score Breakdown</div>', unsafe_allow_html=True)
        for section, pts in a["breakdown"].items():
            maxes = {"Contact Info": 25, "Education": 15, "Skills": 20, "Sections": 30, "Resume Length": 10}
            mx = maxes.get(section, 20)
            pct = int(pts / mx * 100)
            st.markdown(f"""
            <div style="margin:.4rem 0;">
              <div style="display:flex;justify-content:space-between;margin-bottom:.2rem;">
                <span style="font-size:.83rem;color:#e8e6f0;">{section}</span>
                <span style="font-size:.83rem;color:#b48bff;font-weight:600;">{pts}/{mx}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(pct / 100)

        # ML Classification
        st.markdown('<div class="cp-section-head">🤖 Resume Category</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="cp-card">
          <div style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                      background:linear-gradient(135deg,#7fdbca,#b48bff);
                      -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
            {c["category"]}
          </div>
          <div style="font-size:.82rem;color:#7a788a;margin-top:.3rem;">
            Model confidence: <strong style="color:#b48bff;">{c["confidence"]}%</strong>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ATS Feedback ──────────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">📋 ATS Feedback & Recommendations</div>', unsafe_allow_html=True)
    fb_html = ""
    for icon, msg in a["feedback"]:
        color_map = {"✅": "#7fdbca", "❌": "#ff7eb3", "⚠️": "#ffd166", "💡": "#b48bff"}
        color = color_map.get(icon, "#7a788a")
        fb_html += f'<div class="fb-item" style="border-left-color:{color};">{icon} {msg}</div>'
    st.markdown(f'<div class="cp-card">{fb_html}</div>', unsafe_allow_html=True)

    # ── Skills Grid ────────────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">🔍 Detected Skills by Category</div>', unsafe_allow_html=True)
    skill_cats = {k: v for k, v in s.items() if k not in ["all", "total_count"] and v}
    if skill_cats:
        cat_keys = list(skill_cats.keys())
        col_a, col_b = st.columns(2)
        for i, cat in enumerate(cat_keys):
            container = col_a if i % 2 == 0 else col_b
            with container:
                pills = skill_pills(skill_cats[cat], "skill-neutral")
                st.markdown(f"""
                <div class="cp-card">
                  <div style="font-size:.78rem;text-transform:uppercase;letter-spacing:.08em;
                               color:#7a788a;margin-bottom:.5rem;">{skills_to_display_name(cat)}</div>
                  {pills}
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: JD MATCHER
# ══════════════════════════════════════════════════════════════════════════════
elif "JD Matcher" in page:
    st.markdown('<div class="cp-section-head">🎯 Resume ↔ Job Description Matcher</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.9rem;margin-bottom:1rem;">Powered by TF-IDF + Cosine Similarity — the same approach used in search engines.</div>', unsafe_allow_html=True)

    col_l, col_r = st.columns(2)
    with col_l:
        pdf = st.file_uploader("📄 Upload Resume PDF", type=["pdf"], key="jd_resume")
    with col_r:
        jd_text = st.text_area("📝 Paste Job Description", height=220,
                                placeholder="Paste the full job description here…")

    if pdf and jd_text.strip():
        if st.button("🎯 Match Now"):
            with st.spinner("Extracting resume text…"):
                if st.session_state.parsed and st.session_state.filename == pdf.name:
                    parsed = st.session_state.parsed
                else:
                    parsed = parse_resume(pdf)
            with st.spinner("Running cosine similarity…"):
                match = compute_match(parsed["raw_text"], jd_text)
                st.session_state.match = match

    if not st.session_state.match:
        st.info("Upload a resume and paste a job description above to compute the match.")
        st.stop()

    m = st.session_state.match

    # ── Match score ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(gauge_chart(m["match_pct"],   "Overall Match",    "#b48bff"), use_container_width=True)
    c2.plotly_chart(gauge_chart(m["tfidf_score"],  "TF-IDF Similarity","#7fdbca"), use_container_width=True)
    c3.plotly_chart(gauge_chart(m["skill_overlap"],"Skill Overlap",    "#ff7eb3"), use_container_width=True)

    # ── Skills breakdown ───────────────────────────────────────────────────
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="cp-section-head">✅ Matching Skills</div>', unsafe_allow_html=True)
        if m["matching_skills"]:
            st.markdown(f'<div class="cp-card">{skill_pills(m["matching_skills"], "skill-found")}</div>',
                        unsafe_allow_html=True)
        else:
            st.warning("No direct skill matches found.")

    with col_b:
        st.markdown('<div class="cp-section-head">❌ Missing Skills</div>', unsafe_allow_html=True)
        if m["missing_skills"]:
            st.markdown(f'<div class="cp-card">{skill_pills(m["missing_skills"], "skill-missing")}</div>',
                        unsafe_allow_html=True)
        else:
            st.success("Your resume covers all skills in the JD!")

    # ── JD skills list ──────────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">📋 All Skills Required by JD</div>', unsafe_allow_html=True)
    if m["jd_skills"]:
        st.markdown(f'<div class="cp-card">{skill_pills(m["jd_skills"], "skill-neutral")}</div>',
                    unsafe_allow_html=True)

    # ── Math explanation ────────────────────────────────────────────────────
    with st.expander("🔬 How does the matching work?"):
        st.markdown("""
        **Step 1 — TF-IDF Vectorization**
        Both resume and JD are converted to TF-IDF vectors. Terms that appear in
        one doc but rarely elsewhere get higher weight.
        
        **Step 2 — Cosine Similarity**
        `similarity = dot(resume_vec, jd_vec) / (||resume_vec|| × ||jd_vec||)`
        The angle between vectors tells us how similar the text distributions are,
        regardless of document length.
        
        **Step 3 — Skill Overlap**
        We independently count which skills from our 200+ taxonomy appear in both
        documents. This catches matches that TF-IDF might miss (e.g., exact skill names).
        
        **Final Score = 60% TF-IDF + 40% Skill Overlap**
        The blend is more robust than either metric alone.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: SKILL GAP
# ══════════════════════════════════════════════════════════════════════════════
elif "Skill Gap" in page:
    st.markdown('<div class="cp-section-head">📊 Skill Gap Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.skills:
        pdf = st.file_uploader("Upload resume PDF first", type=["pdf"], key="gap_upload")
        if pdf and st.button("Analyze"):
            run_analysis(pdf)
        st.stop()

    s = st.session_state.skills
    roles = get_available_roles()
    target = st.selectbox("🎯 Select target role", roles)

    if st.button("📊 Analyze Gap"):
        st.session_state.gap = analyze_gap(s["all"], target)

    if not st.session_state.gap:
        st.info("Select a role and click Analyze Gap.")
        st.stop()

    g = st.session_state.gap

    # ── Readiness gauge ────────────────────────────────────────────────────
    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(gauge_chart(g["readiness_pct"], "Role Readiness", "#b48bff"),
                        use_container_width=True)
        st.markdown(f"""
        <div class="cp-card" style="text-align:center;">
          <div style="font-size:.78rem;color:#7a788a;">{g["description"]}</div>
          <div style="margin-top:.5rem;font-size:.85rem;color:#ffd166;">💰 {g["avg_salary"]}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f'<div class="cp-section-head">✅ You Have ({len(g["has_required"])})</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="cp-card">{skill_pills(g["has_required"], "skill-found")}</div>',
                        unsafe_allow_html=True)
            if g["has_preferred"]:
                st.markdown(f'<div style="font-size:.78rem;color:#7a788a;margin:.3rem 0;">Preferred skills you have:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cp-card">{skill_pills(g["has_preferred"], "skill-found")}</div>',
                            unsafe_allow_html=True)

        with col_b:
            st.markdown(f'<div class="cp-section-head">❌ Missing ({len(g["missing_required"])})</div>',
                        unsafe_allow_html=True)
            st.markdown(f'<div class="cp-card">{skill_pills(g["missing_required"], "skill-missing")}</div>',
                        unsafe_allow_html=True)
            if g["missing_preferred"]:
                st.markdown(f'<div style="font-size:.78rem;color:#7a788a;margin:.3rem 0;">Preferred skills to add:</div>', unsafe_allow_html=True)
                st.markdown(f'<div class="cp-card">{skill_pills(g["missing_preferred"][:8], "skill-missing")}</div>',
                            unsafe_allow_html=True)

    # ── Radar chart ────────────────────────────────────────────────────────
    req_total   = len(g["required"])
    req_have    = len(g["has_required"])
    pref_total  = len(g["preferred"])
    pref_have   = len(g["has_preferred"])

    fig = go.Figure()
    categories = ["Required\nSkills", "Preferred\nSkills", "Overall\nReadiness",
                  "Req. Coverage", "Pref. Coverage"]
    you_vals = [
        req_have / max(req_total,1) * 100,
        pref_have / max(pref_total,1) * 100,
        g["readiness_pct"],
        req_have / max(req_total,1) * 100,
        pref_have / max(pref_total,1) * 100,
    ]
    target_vals = [100, 100, 100, 100, 100]

    fig.add_trace(go.Scatterpolar(r=target_vals, theta=categories, fill="toself",
                                   fillcolor="rgba(180,139,255,.05)",
                                   line={"color":"#2a2a3d","width":1}, name="Target"))
    fig.add_trace(go.Scatterpolar(r=you_vals, theta=categories, fill="toself",
                                   fillcolor="rgba(180,139,255,.25)",
                                   line={"color":"#b48bff","width":2}, name="You"))
    fig.update_layout(
        polar={"radialaxis":{"visible":True,"range":[0,100],
                              "gridcolor":"#2a2a3d","linecolor":"#2a2a3d",
                              "tickfont":{"color":"#7a788a"}},
               "angularaxis":{"gridcolor":"#2a2a3d","linecolor":"#2a2a3d",
                               "tickfont":{"color":"#7a788a"}},
               "bgcolor":"rgba(0,0,0,0)"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend={"font":{"color":"#e8e6f0"},"bgcolor":"rgba(0,0,0,0)"},
        margin={"t":30,"b":30,"l":30,"r":30},
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: CAREER RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════
elif "Career Recommender" in page:
    st.markdown('<div class="cp-section-head">🌟 Career Recommendation Engine</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.9rem;margin-bottom:1rem;">Content-based filtering — scored across 8 role templates with weighted skill overlap.</div>', unsafe_allow_html=True)

    if not st.session_state.skills:
        pdf = st.file_uploader("Upload resume PDF first", type=["pdf"], key="rec_upload")
        if pdf and st.button("Analyze"):
            run_analysis(pdf)
        st.stop()

    s   = st.session_state.skills
    c   = st.session_state.classified
    rec = recommend_roles(s["all"], top_n=8)

    # Top recommendation hero
    if rec:
        top = rec[0]
        st.markdown(f"""
        <div class="cp-card" style="border-color:#b48bff;padding:1.5rem 2rem;">
          <div style="font-size:.78rem;color:#b48bff;text-transform:uppercase;letter-spacing:.1em;">Best Match</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.8rem;font-weight:800;color:#e8e6f0;">{top["role"]}</div>
          <div style="color:#7a788a;font-size:.88rem;margin:.4rem 0 .8rem 0;">{top["description"]}</div>
          <div style="display:flex;gap:2rem;">
            <div><span style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#b48bff;">{top["readiness_pct"]}%</span>
              <span style="font-size:.78rem;color:#7a788a;margin-left:.3rem;">Readiness</span></div>
            <div><span style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:#ffd166;">{top["avg_salary"]}</span>
              <span style="font-size:.78rem;color:#7a788a;margin-left:.3rem;">Avg Salary</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="cp-section-head">All Role Fits</div>', unsafe_allow_html=True)

    for r in rec:
        pct = r["readiness_pct"]
        bar_color = "#7fdbca" if pct >= 65 else ("#ffd166" if pct >= 40 else "#ff7eb3")
        st.markdown(f"""
        <div class="role-card">
          <div style="flex:1">
            <div class="role-name">{r["role"]}</div>
            <div class="role-fit">{r["fit_label"]} &nbsp;·&nbsp; <span style="color:#7a788a;font-size:.8rem;">{r["avg_salary"]}</span></div>
            <div class="role-bar"><div class="role-bar-fill" style="width:{pct}%;background:{bar_color};"></div></div>
            <div style="font-size:.75rem;color:#7a788a;margin-top:.2rem;">{r["matching_required"]}/{r["total_required"]} required skills matched</div>
          </div>
          <div style="font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;color:{bar_color};margin-left:1rem;">{pct}%</div>
        </div>
        """, unsafe_allow_html=True)

    # Skill areas breakdown
    st.markdown('<div class="cp-section-head">Your Skill Distribution</div>', unsafe_allow_html=True)
    from backend.nlp.skill_extractor import SKILL_DB
    cat_counts = {}
    for cat, db_skills in SKILL_DB.items():
        cand = {x.lower() for x in s["all"]}
        found = [x for x in db_skills if x in cand]
        if found:
            cat_counts[skills_to_display_name(cat)] = len(found)

    if cat_counts:
        fig = px.bar(
            x=list(cat_counts.keys()),
            y=list(cat_counts.values()),
            color=list(cat_counts.values()),
            color_continuous_scale=["#b48bff", "#ff7eb3"],
            labels={"x": "Category", "y": "Skills"},
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color":"#e8e6f0"},
            coloraxis_showscale=False,
            margin={"t":10,"b":10},
            height=280,
            xaxis={"gridcolor":"#2a2a3d"},
            yaxis={"gridcolor":"#2a2a3d"},
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: INTERVIEW PREP
# ══════════════════════════════════════════════════════════════════════════════
elif "Interview Prep" in page:
    st.markdown('<div class="cp-section-head">💬 Interview Question Generator</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.9rem;margin-bottom:1rem;">Skill-targeted questions from a local dataset — zero AI APIs.</div>', unsafe_allow_html=True)

    if not st.session_state.skills:
        pdf = st.file_uploader("Upload resume PDF first", type=["pdf"], key="iq_upload")
        if pdf and st.button("Analyze"):
            run_analysis(pdf)
        st.stop()

    s = st.session_state.skills
    col1, col2, col3 = st.columns(3)
    with col1:
        level = st.selectbox("Difficulty", ["beginner", "intermediate", "advanced"])
    with col2:
        count = st.slider("Questions", 5, 20, 10)
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        gen = st.button("🎲 Generate Questions")

    if gen or "questions_result" not in st.session_state:
        result = generate_questions(s["all"], level=level, count=count)
        st.session_state["questions_result"] = result

    result = st.session_state.get("questions_result")
    if not result or not result["questions"]:
        st.warning("No questions found for detected skills. Make sure your resume contains recognizable skills.")
        st.stop()

    st.markdown(f"""
    <div style="display:flex;gap:1rem;margin:.5rem 0 1rem 0;flex-wrap:wrap;">
      <span class="skill-pill skill-neutral">{result["level"]} level</span>
      <span class="skill-pill skill-found">{len(result["questions"])} questions</span>
      {"".join(f'<span class="skill-pill skill-neutral">{sk}</span>' for sk in result["skills_covered"][:5])}
    </div>
    """, unsafe_allow_html=True)

    for i, q in enumerate(result["questions"], 1):
        accent = ["#b48bff", "#ff7eb3", "#7fdbca", "#ffd166"][i % 4]
        st.markdown(f"""
        <div class="q-card" style="border-left-color:{accent};">
          <div class="q-meta">Q{i} · {q["skill"]} · {q["level"]}</div>
          <div class="q-text">{q["question"]}</div>
        </div>
        """, unsafe_allow_html=True)

    # Quick quiz
    st.markdown("---")
    st.markdown("""
    <div class="cp-section-head">⚡ Quick Prep Quiz
      <span class="unique-badge">Unique</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.85rem;margin-bottom:.8rem;">5 random questions across all difficulty levels from your skills.</div>', unsafe_allow_html=True)

    if st.button("🎰 Shuffle Quiz"):
        st.session_state["quick_quiz"] = generate_quick_quiz(s["all"])

    quiz = st.session_state.get("quick_quiz") or generate_quick_quiz(s["all"])
    for i, q in enumerate(quiz, 1):
        lvl_colors = {"Beginner": "#7fdbca", "Intermediate": "#b48bff", "Advanced": "#ff7eb3"}
        color = lvl_colors.get(q["level"], "#b48bff")
        st.markdown(f"""
        <div class="q-card" style="border-left-color:{color};">
          <div class="q-meta">{q["skill"]} · <span style="color:{color};">{q["level"]}</span></div>
          <div class="q-text">{q["question"]}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: LEARNING ROADMAP
# ══════════════════════════════════════════════════════════════════════════════
elif "Learning Roadmap" in page:
    st.markdown('<div class="cp-section-head">🗺️ Personalized Learning Roadmap</div>', unsafe_allow_html=True)

    if not st.session_state.skills:
        pdf = st.file_uploader("Upload resume PDF first", type=["pdf"], key="road_upload")
        if pdf and st.button("Analyze"):
            run_analysis(pdf)
        st.stop()

    s     = st.session_state.skills
    roles = get_available_roles()

    col1, col2 = st.columns(2)
    with col1:
        target_role = st.selectbox("🎯 Target Role", roles)
    with col2:
        force_lvl = st.selectbox("📶 Level override", ["Auto-detect", "beginner", "intermediate", "advanced"])

    if st.button("🗺️ Generate My Roadmap"):
        gap = analyze_gap(s["all"], target_role)
        level_arg = None if force_lvl == "Auto-detect" else force_lvl
        roadmap = generate_roadmap(
            candidate_skills=s["all"],
            target_role=target_role,
            missing_required=gap["missing_required"],
            readiness_pct=gap["readiness_pct"],
            force_level=level_arg,
        )
        st.session_state.roadmap = roadmap
        st.session_state.gap     = gap

    if not st.session_state.roadmap:
        st.info("Select a role and generate your roadmap above.")
        st.stop()

    rm = st.session_state.roadmap
    gap = st.session_state.gap

    # Header
    st.markdown(f"""
    <div class="cp-card" style="border-color:#b48bff;">
      <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:1rem;">
        <div>
          <div style="font-size:.75rem;color:#b48bff;text-transform:uppercase;letter-spacing:.1em;">Your Path</div>
          <div style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;">{rm["phase_title"]}</div>
          <div style="color:#7a788a;font-size:.85rem;">Target: {rm["role"]} · Level: {rm["level"].title()}</div>
        </div>
        <div style="text-align:right;">
          <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;color:#ffd166;">{rm["estimated_total_weeks"]} weeks</div>
          <div style="font-size:.78rem;color:#7a788a;">estimated remaining</div>
          <div style="font-size:.82rem;color:#7fdbca;margin-top:.2rem;">Next: {rm["next_milestone"]}</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Steps
    priority_labels = {"high": "🔥 Priority", "normal": "📖 Learn", "done": "✅ Done"}
    dot_cls         = {"high": "dot-high", "normal": "dot-normal", "done": "dot-done"}
    step_cls        = {"high": "high", "normal": "", "done": "done"}

    for i, step in enumerate(rm["steps"], 1):
        p = step["priority"]
        resources_html = " · ".join(step.get("resources", []))
        st.markdown(f"""
        <div class="roadmap-step {step_cls[p]}">
          <div class="step-dot {dot_cls[p]}">{i}</div>
          <div class="step-info" style="flex:1;">
            <div class="step-title">{step["skill"]}</div>
            <div class="step-time">⏱ {step.get("time","?")} &nbsp;·&nbsp; {priority_labels[p]}</div>
            <div class="step-res">📚 {resources_html}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # Level switcher
    st.markdown("---")
    st.markdown('<div class="cp-section-head">Switch Level</div>', unsafe_allow_html=True)
    cols = st.columns(3)
    for lvl, col in zip(["beginner", "intermediate", "advanced"], cols):
        with col:
            if st.button(f"📶 {lvl.title()} Roadmap"):
                gap = analyze_gap(s["all"], rm["role"])
                roadmap = generate_roadmap(
                    s["all"], rm["role"], gap["missing_required"],
                    gap["readiness_pct"], force_level=lvl
                )
                st.session_state.roadmap = roadmap
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESUME COMPARATOR
# ══════════════════════════════════════════════════════════════════════════════
elif "Resume Comparator" in page:
    st.markdown("""
    <div class="cp-section-head">⚖️ Resume Version Comparator
      <span class="unique-badge">Unique</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.9rem;margin-bottom:1rem;">Compare two resume versions side-by-side — ATS score delta, skill diff, and category shift.</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div style="font-weight:600;color:#b48bff;margin-bottom:.4rem;">Resume A</div>', unsafe_allow_html=True)
        pdf_a = st.file_uploader("Upload Resume A", type=["pdf"], key="cmp_a")
    with col2:
        st.markdown('<div style="font-weight:600;color:#ff7eb3;margin-bottom:.4rem;">Resume B</div>', unsafe_allow_html=True)
        pdf_b = st.file_uploader("Upload Resume B", type=["pdf"], key="cmp_b")

    if pdf_a and pdf_b:
        if st.button("⚖️ Compare Resumes"):
            run_analysis(pdf_a, slot="a")
            run_analysis(pdf_b, slot="b")

    if not (st.session_state.ats and st.session_state.ats_b):
        st.info("Upload both resumes to compare.")
        st.stop()

    a_ats = st.session_state.ats
    b_ats = st.session_state.ats_b
    a_sk  = st.session_state.skills
    b_sk  = st.session_state.skills_b
    a_fn  = st.session_state.filename   or "Resume A"
    b_fn  = st.session_state.filename_b or "Resume B"

    # ── Score comparison ────────────────────────────────────────────────────
    delta = a_ats["total"] - b_ats["total"]
    delta_color = "#7fdbca" if delta > 0 else ("#ff7eb3" if delta < 0 else "#ffd166")
    delta_sym   = "▲" if delta > 0 else ("▼" if delta < 0 else "=")

    st.markdown(f"""
    <div class="cp-card" style="text-align:center;border-color:{delta_color};">
      <div style="font-family:'Syne',sans-serif;font-size:.85rem;color:#7a788a;text-transform:uppercase;">ATS Score Delta</div>
      <div style="font-family:'Syne',sans-serif;font-size:3rem;font-weight:800;color:{delta_color};">
        {delta_sym} {abs(delta)} pts
      </div>
      <div style="font-size:.85rem;color:#7a788a;">
        A: <strong style="color:#b48bff;">{a_ats["total"]} ({a_ats["grade"]})</strong>
        &nbsp;vs&nbsp;
        B: <strong style="color:#ff7eb3;">{b_ats["total"]} ({b_ats["grade"]})</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Section breakdown comparison ────────────────────────────────────────
    st.markdown('<div class="cp-section-head">Score Breakdown Comparison</div>', unsafe_allow_html=True)
    sections = list(a_ats["breakdown"].keys())
    a_vals   = [a_ats["breakdown"][s] for s in sections]
    b_vals   = [b_ats["breakdown"][s] for s in sections]

    fig = go.Figure()
    fig.add_trace(go.Bar(name=a_fn[:20], x=sections, y=a_vals,
                         marker_color="#b48bff", opacity=.85))
    fig.add_trace(go.Bar(name=b_fn[:20], x=sections, y=b_vals,
                         marker_color="#ff7eb3", opacity=.85))
    fig.update_layout(
        barmode="group",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e8e6f0"},
        legend={"bgcolor":"rgba(0,0,0,0)"},
        margin={"t":10,"b":10},
        height=300,
        xaxis={"gridcolor":"#2a2a3d"},
        yaxis={"gridcolor":"#2a2a3d"},
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Skill diff ──────────────────────────────────────────────────────────
    a_set = set(a_sk["all"])
    b_set = set(b_sk["all"])
    only_a = sorted(a_set - b_set)
    only_b = sorted(b_set - a_set)
    shared = sorted(a_set & b_set)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown(f'<div class="cp-section-head" style="color:#b48bff;">Only in A ({len(only_a)})</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cp-card">{skill_pills(only_a, "skill-neutral") or "<span style=color:#7a788a>None</span>"}</div>', unsafe_allow_html=True)
    with col_b:
        st.markdown(f'<div class="cp-section-head" style="color:#7fdbca;">Shared ({len(shared)})</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cp-card">{skill_pills(shared[:15], "skill-found") or "<span style=color:#7a788a>None</span>"}</div>', unsafe_allow_html=True)
    with col_c:
        st.markdown(f'<div class="cp-section-head" style="color:#ff7eb3;">Only in B ({len(only_b)})</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cp-card">{skill_pills(only_b, "skill-missing") or "<span style=color:#7a788a>None</span>"}</div>', unsafe_allow_html=True)

    # ── Word count & metrics ────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">Quick Stats</div>', unsafe_allow_html=True)
    rows = [
        ("Skills Count",  a_sk["total_count"], b_sk["total_count"]),
        ("Word Count",    st.session_state.parsed.get("word_count",0),
                          st.session_state.parsed_b.get("word_count",0)),
        ("ATS Score",     a_ats["total"], b_ats["total"]),
    ]
    for label, av, bv in rows:
        a_cls = "c-win" if av > bv else ("c-lose" if av < bv else "c-tie")
        b_cls = "c-win" if bv > av else ("c-lose" if bv < av else "c-tie")
        st.markdown(f"""
        <div class="compare-row">
          <div class="compare-val {a_cls}" style="text-align:right;">{av}</div>
          <div class="compare-label">{label}</div>
          <div class="compare-val {b_cls}">{bv}</div>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: RESUME DNA  (unique feature)
# ══════════════════════════════════════════════════════════════════════════════
elif "Resume DNA" in page:
    st.markdown("""
    <div class="cp-section-head">🧬 Resume DNA Fingerprint
      <span class="unique-badge">Exclusive</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.9rem;margin-bottom:1rem;">Your unique skill fingerprint — a multi-dimensional profile not seen in other resume tools.</div>', unsafe_allow_html=True)

    if not st.session_state.skills:
        pdf = st.file_uploader("Upload resume PDF first", type=["pdf"], key="dna_upload")
        if pdf and st.button("Analyze"):
            run_analysis(pdf)
        st.stop()

    s = st.session_state.skills
    a = st.session_state.ats
    c = st.session_state.classified
    p = st.session_state.parsed

    from backend.nlp.skill_extractor import SKILL_DB
    from backend.nlp.gap_analyzer    import ROLE_TEMPLATES

    # ── Radar: skill coverage across all domains ───────────────────────────
    st.markdown('<div class="cp-section-head">Domain Coverage Radar</div>', unsafe_allow_html=True)
    cats = list(SKILL_DB.keys())
    cand = {x.lower() for x in s["all"]}
    coverage = []
    for cat in cats:
        pct = len([x for x in SKILL_DB[cat] if x in cand]) / max(len(SKILL_DB[cat]),1) * 100
        coverage.append(round(pct, 1))

    labels = [skills_to_display_name(c) for c in cats]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=coverage, theta=labels, fill="toself",
        fillcolor="rgba(180,139,255,.2)",
        line={"color":"#b48bff","width":2},
        name="You",
    ))
    fig.update_layout(
        polar={
            "radialaxis":{"visible":True,"range":[0,100],"gridcolor":"#2a2a3d",
                          "tickfont":{"color":"#7a788a"},"linecolor":"#2a2a3d"},
            "angularaxis":{"gridcolor":"#2a2a3d","linecolor":"#2a2a3d",
                           "tickfont":{"color":"#7a788a","size":10}},
            "bgcolor":"rgba(0,0,0,0)",
        },
        paper_bgcolor="rgba(0,0,0,0)",
        legend={"font":{"color":"#e8e6f0"},"bgcolor":"rgba(0,0,0,0)"},
        margin={"t":20,"b":20,"l":20,"r":20},
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Role fit heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">Role Compatibility Heatmap</div>', unsafe_allow_html=True)
    from backend.nlp.gap_analyzer import analyze_gap as _ag
    roles = list(ROLE_TEMPLATES.keys())
    role_pcts = [_ag(s["all"], r)["readiness_pct"] for r in roles]

    fig2 = go.Figure(go.Bar(
        x=role_pcts, y=roles, orientation="h",
        marker={"color": role_pcts,
                "colorscale": [[0,"#ff7eb3"],[0.5,"#b48bff"],[1,"#7fdbca"]],
                "line":{"width":0}},
        text=[f"{v}%" for v in role_pcts],
        textposition="auto",
        textfont={"color":"#e8e6f0","family":"Syne","size":11},
    ))
    fig2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e8e6f0"},
        xaxis={"range":[0,105],"gridcolor":"#2a2a3d"},
        yaxis={"gridcolor":"#2a2a3d"},
        margin={"t":10,"b":10,"l":10,"r":10},
        height=300,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ── DNA card ───────────────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">Your Resume DNA</div>', unsafe_allow_html=True)
    top_domain = max(zip(cats, coverage), key=lambda x: x[1])
    weak_domain = min(zip(cats, coverage), key=lambda x: x[1])

    # Compute a "versatility" score — std dev of coverage (lower = more focused)
    import numpy as np
    versatility = round(100 - float(np.std(coverage)), 1)

    dna_cards = [
        ("🔬", "Primary Domain",    skills_to_display_name(top_domain[0]),   f"{top_domain[1]:.0f}% coverage"),
        ("⚡", "Growth Area",       skills_to_display_name(weak_domain[0]),  f"Only {weak_domain[1]:.0f}% — big opportunity"),
        ("🧩", "Versatility Score", f"{versatility}/100",                    "How broadly your skills spread"),
        ("🎯", "ML Classification", c["category"],                           f"{c['confidence']}% confident"),
        ("📊", "ATS Performance",   f"{a['total']}/100 ({a['grade']})",      "Resume completeness"),
        ("✦",  "Skill Signature",   f"{s['total_count']} unique skills",     "Total detected across all domains"),
    ]
    dna_cols = st.columns(3)
    for i, (icon, label, val, sub) in enumerate(dna_cards):
        with dna_cols[i % 3]:
            st.markdown(f"""
            <div class="cp-card">
              <div style="font-size:1.4rem;">{icon}</div>
              <div style="font-size:.72rem;text-transform:uppercase;letter-spacing:.08em;color:#7a788a;margin:.3rem 0 .1rem;">{label}</div>
              <div style="font-family:'Syne',sans-serif;font-weight:700;font-size:1.05rem;color:#e8e6f0;">{val}</div>
              <div style="font-size:.75rem;color:#7a788a;margin-top:.2rem;">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Treemap ── skill distribution
    st.markdown('<div class="cp-section-head">Skill Treemap</div>', unsafe_allow_html=True)
    tree_data = []
    for cat in cats:
        found = [x for x in SKILL_DB[cat] if x in cand]
        for sk in found:
            tree_data.append({"category": skills_to_display_name(cat), "skill": sk, "count": 1})

    if tree_data:
        df = pd.DataFrame(tree_data)
        fig3 = px.treemap(df, path=["category", "skill"], values="count",
                           color="category",
                           color_discrete_sequence=["#b48bff","#ff7eb3","#7fdbca","#ffd166",
                                                    "#6ec6ff","#ff9966","#a8ff78","#78ffd6"])
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin={"t":10,"b":10,"l":10,"r":10},
            height=350,
        )
        fig3.update_traces(textfont={"color":"#fff","size":11})
        st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: HISTORY
# ══════════════════════════════════════════════════════════════════════════════
elif "History" in page:
    st.markdown('<div class="cp-section-head">📈 Analysis History</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#7a788a;font-size:.9rem;margin-bottom:1rem;">Past analyses stored in SQLite — track your resume improvement over time.</div>', unsafe_allow_html=True)

    rows = get_recent_analyses(limit=20)
    if not rows:
        st.info("No analyses saved yet. Run the Resume Analyzer to start tracking progress.")
        st.stop()

    df = pd.DataFrame(rows)
    df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%b %d, %Y %H:%M")

    # Score trend
    if len(df) > 1:
        st.markdown('<div class="cp-section-head">ATS Score Trend</div>', unsafe_allow_html=True)
        fig = px.line(df.iloc[::-1], x="created_at", y="ats_score",
                      markers=True, line_shape="spline",
                      color_discrete_sequence=["#b48bff"])
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color":"#e8e6f0"},
            xaxis={"gridcolor":"#2a2a3d"},
            yaxis={"gridcolor":"#2a2a3d","range":[0,100]},
            margin={"t":10,"b":10},
            height=240,
        )
        fig.update_traces(line_width=2, marker_size=8, marker_color="#ff7eb3")
        st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown('<div class="cp-section-head">Recent Analyses</div>', unsafe_allow_html=True)
    for r in rows:
        grade_color = "#7fdbca" if r["ats_score"] >= 70 else ("#ffd166" if r["ats_score"] >= 50 else "#ff7eb3")
        st.markdown(f"""
        <div class="cp-card" style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem;">
          <div>
            <div style="font-family:'Syne',sans-serif;font-weight:700;">{r.get("name") or "Unknown"}</div>
            <div style="font-size:.78rem;color:#7a788a;">{r.get("filename","?")} · {r.get("created_at","")}</div>
            <div style="font-size:.8rem;color:#b48bff;margin-top:.2rem;">{r.get("category","")}</div>
          </div>
          <div style="text-align:right;">
            <span style="font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:800;color:{grade_color};">{r.get("ats_score",0)}</span>
            <span style="font-size:.8rem;color:#7a788a;margin-left:.3rem;">{r.get("grade","")}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  PAGE: STRENGTH DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
elif "Strength Dashboard" in page:
    st.markdown("""
    <div class="cp-section-head">💪 Resume Strength Dashboard
      <span class="unique-badge">All-in-One</span>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.parsed:
        pdf = st.file_uploader("Upload resume PDF", type=["pdf"], key="dash_upload")
        if pdf and st.button("⚡ Analyze"):
            run_analysis(pdf)
        if not st.session_state.parsed:
            st.info("Upload a resume to see your full strength dashboard.")
            st.stop()

    p  = st.session_state.parsed
    s  = st.session_state.skills
    a  = st.session_state.ats
    c  = st.session_state.classified

    from backend.nlp.skill_extractor import SKILL_DB
    from backend.nlp.gap_analyzer import ROLE_TEMPLATES, analyze_gap as _ag

    # ── Top KPI strip ────────────────────────────────────────────────────────
    kpis = [
        ("ATS Score",       f"{a['total']}/100",        a["total"],   "#b48bff"),
        ("Grade",           a["grade"],                  None,         "#ffd166"),
        ("Skills",          str(s["total_count"]),        None,         "#7fdbca"),
        ("Category",        c["category"],               None,         "#ff7eb3"),
        ("Word Count",      str(p["word_count"]),         None,         "#6ec6ff"),
        ("Completeness",    f"{a['total']}%",             a["total"],   "#b48bff"),
    ]
    cols = st.columns(len(kpis))
    for col, (lbl, val, bar_val, color) in zip(cols, kpis):
        col.markdown(f"""
        <div class="stat-tile">
          <div class="val" style="color:{color};font-size:1.4rem;">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Three gauge row ──────────────────────────────────────────────────────
    best_role_gap  = _ag(s["all"], "Data Scientist")
    best_role_name = "Data Scientist"
    best_pct       = best_role_gap["readiness_pct"]
    for role in list(ROLE_TEMPLATES.keys()):
        g = _ag(s["all"], role)
        if g["readiness_pct"] > best_pct:
            best_pct       = g["readiness_pct"]
            best_role_name = role
            best_role_gap  = g

    skill_coverage = round(s["total_count"] / 80 * 100, 1)  # 80 = rough expected max

    g1, g2, g3 = st.columns(3)
    g1.plotly_chart(gauge_chart(a["total"],      "ATS Score",        "#b48bff"), use_container_width=True)
    g2.plotly_chart(gauge_chart(min(skill_coverage, 100), "Skill Coverage", "#7fdbca"), use_container_width=True)
    g3.plotly_chart(gauge_chart(best_pct,        f"Best Role Fit\n({best_role_name})", "#ff7eb3"), use_container_width=True)

    # ── ATS Feedback panel ───────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">📋 Resume Health Report</div>', unsafe_allow_html=True)
    icon_map = {"✅": "#7fdbca", "❌": "#ff7eb3", "⚠️": "#ffd166", "💡": "#b48bff"}

    passes  = [(ic, msg) for ic, msg in a["feedback"] if ic == "✅"]
    fixes   = [(ic, msg) for ic, msg in a["feedback"] if ic in ("❌", "⚠️")]
    tips    = [(ic, msg) for ic, msg in a["feedback"] if ic == "💡"]

    col_pass, col_fix = st.columns(2)
    with col_pass:
        st.markdown('<div style="font-size:.85rem;color:#7fdbca;font-weight:600;margin-bottom:.4rem;">✅ What\'s Working</div>', unsafe_allow_html=True)
        if passes:
            html = "".join(f'<div class="fb-item" style="border-left-color:#7fdbca;">✅ {m}</div>' for _, m in passes)
            st.markdown(f'<div class="cp-card">{html}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="cp-card" style="color:#7a788a;font-size:.85rem;">Nothing flagged as passing yet.</div>', unsafe_allow_html=True)

    with col_fix:
        st.markdown('<div style="font-size:.85rem;color:#ff7eb3;font-weight:600;margin-bottom:.4rem;">🔧 Fix These</div>', unsafe_allow_html=True)
        if fixes:
            html = "".join(f'<div class="fb-item" style="border-left-color:{icon_map[ic]};">{ic} {m}</div>' for ic, m in fixes)
            st.markdown(f'<div class="cp-card">{html}</div>', unsafe_allow_html=True)
        else:
            st.success("No critical issues found!")

    if tips:
        st.markdown('<div style="font-size:.85rem;color:#b48bff;font-weight:600;margin:.6rem 0 .3rem;">💡 Pro Tips</div>', unsafe_allow_html=True)
        html = "".join(f'<div class="fb-item" style="border-left-color:#b48bff;">💡 {m}</div>' for _, m in tips)
        st.markdown(f'<div class="cp-card">{html}</div>', unsafe_allow_html=True)

    # ── Skills by category ───────────────────────────────────────────────────
    st.markdown('<div class="cp-section-head">🔍 Skill Coverage by Domain</div>', unsafe_allow_html=True)
    import plotly.graph_objects as go
    cat_names, cat_have, cat_total = [], [], []
    for cat, db_skills in SKILL_DB.items():
        cand_set = {x.lower() for x in s["all"]}
        found    = [x for x in db_skills if x in cand_set]
        if found or len(db_skills) > 0:
            cat_names.append(skills_to_display_name(cat))
            cat_have.append(len(found))
            cat_total.append(len(db_skills))

    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(name="You Have", x=cat_names, y=cat_have,
                         marker_color="#b48bff", opacity=.9))
    fig.add_trace(go.Bar(name="Total in DB", x=cat_names, y=cat_total,
                         marker_color="#2a2a3d", opacity=.8))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color":"#e8e6f0","size":11},
        legend={"bgcolor":"rgba(0,0,0,0)"},
        xaxis={"gridcolor":"#2a2a3d","tickangle":-30},
        yaxis={"gridcolor":"#2a2a3d"},
        margin={"t":10,"b":60,"l":10,"r":10},
        height=300,
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Missing skills for best role ─────────────────────────────────────────
    if best_role_gap["missing_required"]:
        st.markdown(f'<div class="cp-section-head">🎯 Top Skills to Add for {best_role_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="cp-card">{skill_pills(best_role_gap["missing_required"], "skill-missing")}</div>',
                    unsafe_allow_html=True)

    # ── Contact completeness checklist ───────────────────────────────────────
    st.markdown('<div class="cp-section-head">📇 Profile Completeness</div>', unsafe_allow_html=True)
    checks = [
        ("Name",     bool(p.get("name") and p["name"] != "Not detected")),
        ("Email",    bool(p.get("email"))),
        ("Phone",    bool(p.get("phone"))),
        ("LinkedIn", bool(p.get("linkedin"))),
        ("GitHub",   bool(p.get("github"))),
        ("Education",bool(p.get("education"))),
    ]
    total_checks = len(checks)
    done_checks  = sum(1 for _, v in checks if v)
    completeness_pct = round(done_checks / total_checks * 100)

    check_html = ""
    for label, done in checks:
        icon  = "✅" if done else "☐"
        color = "#7fdbca" if done else "#3a3a50"
        check_html += f'<div style="display:flex;align-items:center;gap:.6rem;padding:.35rem 0;border-bottom:1px solid #1e1e2e;"><span style="color:{color};font-size:1rem;">{icon}</span><span style="font-size:.88rem;color:{"#e8e6f0" if done else "#7a788a"};">{label}</span></div>'

    st.markdown(f"""
    <div class="cp-card">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:.6rem;">
        <span style="font-family:'Syne',sans-serif;font-weight:700;">Profile Fields</span>
        <span style="font-family:'Syne',sans-serif;font-size:1.2rem;font-weight:800;color:#b48bff;">{completeness_pct}%</span>
      </div>
      {check_html}
    </div>
    """, unsafe_allow_html=True)
