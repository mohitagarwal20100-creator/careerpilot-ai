# CareerPilot AI — Resume Intelligence Platform

## Project Overview

CareerPilot AI is a production-quality Resume Intelligence Platform built entirely with traditional NLP, machine learning, and software engineering. No external LLM APIs are used. Every algorithm is explainable, every design choice is documented.

## Features

| # | Feature | Algorithm |
|---|---------|-----------|
| 1 | Resume Parsing | Regex + pypdf |
| 2 | ATS Score | Weighted rule-based scoring |
| 3 | Skill Extraction | Dictionary lookup + fuzzy matching |
| 4 | Resume Classification | TF-IDF + Logistic Regression |
| 5 | JD Matching | TF-IDF + Cosine Similarity |
| 6 | Skill Gap Analysis | Set difference on role templates |
| 7 | Career Recommendation | Skill-weighted scoring engine |
| 8 | Interview Questions | Local JSON dataset lookup |
| 9 | Learning Roadmap | Rule-based progression engine |
| 10 | Resume Comparison | Side-by-side ATS delta analysis |
| 11 | Strength Dashboard | Aggregated metric visualization |

## Architecture

```
careerpilot/
├── app.py                    # Streamlit entry point
├── backend/
│   ├── parsers/
│   │   └── resume_parser.py  # PDF text extraction + field parsing
│   ├── ml/
│   │   ├── classifier.py     # TF-IDF + Logistic Regression
│   │   └── matcher.py        # Cosine similarity JD matching
│   └── nlp/
│       ├── ats_scorer.py     # ATS scoring engine
│       ├── skill_extractor.py# Skill detection
│       ├── gap_analyzer.py   # Skill gap analysis
│       ├── recommender.py    # Career recommendation engine
│       ├── question_gen.py   # Interview question generator
│       └── roadmap_gen.py    # Learning roadmap generator
├── data/
│   ├── skills_db/            # Skill taxonomy JSON
│   ├── questions/            # Interview questions JSON
│   ├── roadmaps/             # Role roadmaps JSON
│   └── roles/                # Role skill templates JSON
├── frontend/
│   └── components.py         # Reusable UI components
├── database.py               # SQLite persistence layer
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourusername/careerpilot-ai
cd careerpilot-ai
pip install -r requirements.txt
streamlit run app.py
```

## Deployment

Deploy free on [Streamlit Community Cloud](https://streamlit.io/cloud):
1. Push to GitHub
2. Connect repo on share.streamlit.io
3. Set main file to `app.py`

## Design Decisions

### Why TF-IDF over word embeddings?
TF-IDF is interpretable, requires no GPU, trains in seconds, and produces competitive results for short-document classification. A student can fully understand the math. Word embeddings (Word2Vec, BERT) are black boxes requiring significant compute.

### Why Logistic Regression over deep learning?
Logistic Regression trains instantly, is fully interpretable (feature weights show which terms matter), generalizes well on small datasets, and needs no GPU. Deep learning would overfit on a small resume dataset without extensive data augmentation.

### Why Cosine Similarity for JD matching?
Cosine similarity measures the angle between TF-IDF vectors — not raw term counts — making it robust to document length differences. A short JD and a long resume can still score high if they share the same vocabulary distribution.

### Why SQLite?
Zero-config, file-based, requires no server. Perfect for a portfolio project. Swap to PostgreSQL in one line when scaling.

### Why Streamlit?
Fastest path from Python functions to a deployable web UI. A CS student can read the entire frontend codebase in one sitting.
