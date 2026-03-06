---
title: XSS CTF Intelligence Platform
emoji: 🔐
colorFrom: red
colorTo: blue
sdk: gradio
sdk_version: 4.40.0
app_file: app.py
pinned: false
python_version: "3.10"
---

# 🔐 XSS CTF Intelligence Platform

[![Live Demo](https://img.shields.io/badge/Demo-HuggingFace%20Spaces-yellow)](https://huggingface.co/spaces/darkcyberwizard/xss-ctf-intelligence)
[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

Two tools in one platform for cybersecurity education research:

1. **CTF Challenge Generator** — LLM-powered generation of structured Capture The Flag challenges across vulnerability types, difficulty levels, and contexts
2. **Learning Analytics Dashboard** — Statistical analysis and visualisation of student learning patterns from pre/post test scores and game interaction data

> Built as a portfolio project connecting AI engineering with cybersecurity education research (XSS game study).

**[▶ Try the Live Demo](https://huggingface.co/spaces/darkcyberwizard/xss-ctf-intelligence)**

---

## Features

### CTF Challenge Generator
- Generates complete challenges: scenario, vulnerable code, hints, solution, remediation
- Supports XSS (Reflected/Stored/DOM), SQL Injection, CSRF, Broken Access Control, Command Injection
- 3 difficulty levels with calibrated complexity
- 7 real-world contexts (login page, comment section, admin panel etc.)
- Structured JSON output for programmatic use

### Learning Analytics
- Upload student CSV data (pre/post scores + simulator engagement)
- Computes learning gain and Hake's normalised gain (g)
- Simulator vs non-simulator comparison
- V1 vs V2 game version comparison
- Student segmentation by learning profile
- Interactive Plotly visualisations

---

## Quickstart

```bash
git clone https://github.com/darkcyberwizard/xss-ctf-intelligence
cd xss-ctf-intelligence
pip install -r requirements.txt
cp .env.example .env  # Add your HF_TOKEN
python ui/gradio_app.py
```

Open: http://localhost:7860

---

## Data Format

Upload a CSV with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `pre_score` | float | Score before the game/intervention |
| `post_score` | float | Score after the game/intervention |
| `used_simulator` | int (0/1) | Whether student used the simulator zone |
| `time_in_simulator` | float | Minutes spent in simulator (optional) |
| `version` | string | Game version e.g. V1, V2 (optional) |

A sample dataset is provided at `data/sample_data.csv`.

---

## Architecture

```
User Input
    │
    ├── Tab 1: CTF Generator
    │       │
    │       ▼
    │   LLM (Llama-3.2-3B via HF Inference Router)
    │       │
    │       ▼
    │   Structured JSON Challenge
    │
    └── Tab 2: Learning Analytics
            │
            ▼
        CSV Upload → pandas → Feature Engineering
            │
            ├── Summary Statistics (Hake's g, learning gain)
            ├── Simulator Impact Analysis
            ├── Student Segmentation
            └── Plotly Visualisations
```

---

## CV Bullets

> Built LLM-powered CTF challenge generator (Llama-3.2-3B via HF Inference API) producing structured challenges across 5 vulnerability types, 3 difficulty levels, and 7 real-world contexts

> Developed learning analytics pipeline analysing student pre/post test scores and simulator engagement data; computed Hake's normalised gain, simulator impact, and learning profile segmentation across V1/V2 game versions

---

## License

MIT
