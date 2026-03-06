"""
config.py - Central configuration for XSS CTF Intelligence Platform
"""
import os
from dotenv import load_dotenv
load_dotenv()

# --- HuggingFace ---
HF_TOKEN        = os.getenv("HF_TOKEN", "")
HF_TEXT_MODEL   = os.getenv("HF_TEXT_MODEL", "meta-llama/Llama-3.2-3B-Instruct")

# --- Generation settings ---
MAX_TOKENS      = 1024
TEMPERATURE     = 0.7

# --- Vulnerability types supported ---
VULN_TYPES = ["XSS", "SQL Injection", "CSRF", "Broken Access Control", "Command Injection"]

# --- Difficulty levels ---
DIFFICULTIES = ["Easy", "Medium", "Hard"]

# --- XSS subtypes ---
XSS_SUBTYPES = ["Reflected XSS", "Stored XSS", "DOM-based XSS"]

# --- Challenge contexts ---
CONTEXTS = [
    "Login page",
    "Comment/forum section",
    "Search bar",
    "User profile page",
    "File upload form",
    "Admin panel",
    "Contact form",
]
