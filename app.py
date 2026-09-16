"""
SentientSpend AI — v5
All 17 advanced features implemented for real (no mock/placeholder logic).

Run:
    pip install -r requirements.txt
    streamlit run app.py

Optional extras unlock extra power but the app never crashes without them:
    thefuzz + python-Levenshtein   -> better auto-tagging
    pytesseract + pillow           -> real receipt OCR
    reportlab                      -> PDF reports
    yfinance / requests            -> crypto + FX rates
    anthropic (+ ANTHROPIC_API_KEY)-> LLM-powered chat answers
"""
import streamlit as st

st.set_page_config(
    page_title="SentientSpend AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import sqlite3, hashlib, hmac, os, io, re, json, random, secrets, base64
import calendar as calmod
from datetime import datetime, timedelta, date

import streamlit.components.v1 as components

# ─────────────────────────────────────────────────────────────────────
# OPTIONAL DEPENDENCIES — every one has a working fallback
# ─────────────────────────────────────────────────────────────────────
try:
    from thefuzz import fuzz
    FUZZ_OK = True
except Exception:
    FUZZ_OK = False

try:
    import requests
    REQ_OK = True
except Exception:
    REQ_OK = False

try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle,
                                    Paragraph, Spacer, Image as RLImage)
    REPORTLAB_OK = True
except Exception:
    REPORTLAB_OK = False

try:
    import pytesseract
    from PIL import Image
    import shutil

    if not shutil.which("tesseract"):
        _candidates = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
            os.environ.get("TESSERACT_PATH", ""),
        ]
        for _path in _candidates:
            if _path and os.path.isfile(_path):
                pytesseract.pytesseract.tesseract_cmd = _path
                break

    _base_dir = os.path.dirname(os.path.abspath(__file__))
    _local_tessdata = os.path.join(_base_dir, "tessdata")
    _candidates_tessdata = [
        _local_tessdata,
        r"C:\Program Files\Tesseract-OCR\tessdata",
        r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tessdata"),
        os.environ.get("TESSDATA_PREFIX", ""),
    ]
    TESSDATA_DIR = ""
    for _td in _candidates_tessdata:
        if _td and os.path.isfile(os.path.join(_td, "eng.traineddata")):
            TESSDATA_DIR = _td
            os.environ["TESSDATA_PREFIX"] = _td
            break

    # Auto-fetch eng.traineddata if missing
    if not TESSDATA_DIR:
        try:
            import urllib.request
            os.makedirs(_local_tessdata, exist_ok=True)
            _eng_dest = os.path.join(_local_tessdata, "eng.traineddata")
            if not os.path.isfile(_eng_dest):
                urllib.request.urlretrieve(
                    "https://github.com/tesseract-ocr/tessdata_fast/raw/main/eng.traineddata",
                    _eng_dest
                )
            if os.path.isfile(_eng_dest):
                TESSDATA_DIR = _local_tessdata
                os.environ["TESSDATA_PREFIX"] = _local_tessdata
        except Exception:
            pass

    pytesseract.get_tesseract_version()
    OCR_OK = True
except Exception:
    OCR_OK = False
    TESSDATA_DIR = ""

try:
    import speech_recognition as sr
    SPEECH_OK = True
except Exception:
    SPEECH_OK = False


def transcribe_audio(audio_bytes):
    if not SPEECH_OK:
        return "SpeechRecognition package not installed."
        
    try:
        # Convert streamlit audio bytes to a recognizable format
        import io
        import speech_recognition as sr
        
        # Write bytes to a temporary wav structure in memory
        audio_file = io.BytesIO(audio_bytes)
        
        r = sr.Recognizer()
        with sr.AudioFile(audio_file) as source:
            audio_data = r.record(source)
            # recognize_google requires internet
            text = r.recognize_google(audio_data)
            return text
    except Exception as e:
        return f"Could not transcribe: {str(e)}"

def get_secret(name, default=""):
    """Key lookup order: environment variable -> .streamlit/secrets.toml -> default.

    Keys are never written into this file. Set them with, e.g.:
        export GEMINI_API_KEY="your-key"
    or put GEMINI_API_KEY = "your-key" in .streamlit/secrets.toml (git-ignore it).
    """
    val = os.getenv(name, "")
    if not val:
        try:
            val = st.secrets.get(name, "")
        except Exception:
            val = ""
    return val or default


try:
    import anthropic
    ANTHROPIC_OK = bool(get_secret("ANTHROPIC_API_KEY"))
except Exception:
    ANTHROPIC_OK = False

GEMINI_OK = bool(get_secret("GEMINI_API_KEY") or get_secret("GOOGLE_API_KEY"))
GEMINI_MODEL = get_secret("GEMINI_MODEL", "gemini-2.0-flash")

# Groq (Llama 3) LLM Backend Constants
GROQ_API_KEY = get_secret("GROQ_API_KEY")
GROQ_OK = bool(GROQ_API_KEY) and REQ_OK
GROQ_LABEL = "Groq (Llama 3)"
GROQ_MODEL = get_secret("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_BASE_URL = get_secret("GROQ_BASE_URL", "https://api.groq.com/openai/v1")


def llm_providers():
    """Which chat back-ends are usable right now."""
    out = []
    if GROQ_OK:
        out.append(GROQ_LABEL)
    if ANTHROPIC_OK:
        out.append("Claude")
    return out

try:
    import smtplib
    from email.mime.text import MIMEText
    SMTP_OK = True
except Exception:
    SMTP_OK = False

# ─────────────────────────────────────────────────────────────────────
# DATABASE  (schema + forward-compatible migrations)
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("SENTIENTSPEND_DB", os.path.join(BASE_DIR, "sentientspend.db"))


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def _ensure_column(c, table, column, decl):
    cols = [r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()]
    if column not in cols:
        c.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")


def init_db():
    with get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT UNIQUE NOT NULL,
            email         TEXT UNIQUE NOT NULL,
            pw_hash       TEXT NOT NULL,
            budget        INTEGER DEFAULT 55000,
            alert_email   TEXT DEFAULT '',
            base_currency TEXT DEFAULT 'INR',
            created       TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            date        TEXT NOT NULL,
            type        TEXT NOT NULL,
            category    TEXT NOT NULL,
            amount      REAL NOT NULL,
            description TEXT DEFAULT '',
            source      TEXT DEFAULT 'manual',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS challenges (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            name        TEXT NOT NULL,
            target_amt  REAL NOT NULL,
            start_date  TEXT NOT NULL,
            end_date    TEXT NOT NULL,
            status      TEXT DEFAULT 'active',
            kind        TEXT DEFAULT 'save',
            category    TEXT DEFAULT 'All',
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS family_accounts (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            admin_id   INTEGER NOT NULL,
            member_id  INTEGER NOT NULL,
            role       TEXT DEFAULT 'member',
            created    TEXT DEFAULT (datetime('now')),
            UNIQUE(admin_id, member_id),
            FOREIGN KEY (admin_id)  REFERENCES users(id) ON DELETE CASCADE,
            FOREIGN KEY (member_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS badges (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER NOT NULL,
            code     TEXT NOT NULL,
            earned   TEXT DEFAULT (datetime('now')),
            UNIQUE(user_id, code)
        );
        CREATE INDEX IF NOT EXISTS ix_txn_user_date ON transactions(user_id, date);
        """)
        # migrations for databases created by older versions
        _ensure_column(c, "users", "base_currency", "TEXT DEFAULT 'INR'")
        _ensure_column(c, "users", "alert_email", "TEXT DEFAULT ''")
        _ensure_column(c, "transactions", "source", "TEXT DEFAULT 'manual'")
        _ensure_column(c, "challenges", "kind", "TEXT DEFAULT 'save'")
        _ensure_column(c, "challenges", "category", "TEXT DEFAULT 'All'")


init_db()


# Streamlit width-API compatibility (use_container_width was replaced by width="stretch")
try:
    _SV = tuple(int(x) for x in st.__version__.split(".")[:2])
except Exception:
    _SV = (1, 0)
WIDE = {"width": "stretch"} if _SV >= (1, 49) else {"use_container_width": True}

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
CATEGORIES = ["Food", "Transport", "Shopping", "Bills", "Entertainment",
              "Healthcare", "Education", "Other"]
INCOME_CATEGORIES = ["Salary", "Freelance", "Investment", "Refund", "Other"]
ALL_CATEGORIES = CATEGORIES + [c for c in INCOME_CATEGORIES if c not in CATEGORIES]

CAT_COLORS = {
    "Food": "#1d4ed8", "Shopping": "#7c3aed", "Bills": "#0891b2",
    "Transport": "#d97706", "Entertainment": "#dc2626",
    "Healthcare": "#059669", "Education": "#ea580c", "Other": "#64748b",
    "Salary": "#0f766e", "Freelance": "#4338ca", "Investment": "#166534",
    "Refund": "#a16207",
}
CAT_ICONS = {"Food": "🍔", "Transport": "🚗", "Shopping": "🛍️", "Bills": "🏠",
             "Entertainment": "🎬", "Healthcare": "💊", "Education": "📚",
             "Other": "📌", "Salary": "💼", "Freelance": "🧾",
             "Investment": "📈", "Refund": "↩️"}

ESSENTIAL_CATS = {"Bills", "Healthcare", "Transport", "Education"}

CURRENCIES = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "AED": "د.إ",
              "JPY": "¥", "AUD": "A$", "CAD": "C$", "SGD": "S$"}
# used only when the live FX endpoint is unreachable (1 INR = X)
FALLBACK_RATES = {"INR": 1.0, "USD": 0.0120, "EUR": 0.0110, "GBP": 0.0094,
                  "AED": 0.0441, "JPY": 1.78, "AUD": 0.0182, "CAD": 0.0164,
                  "SGD": 0.0157}

BADGE_DEFS = {
    "first_txn":    ("🌱", "First Step", "Logged your first transaction"),
    "saver_10":     ("🐖", "Saver", "Hit a 10% savings rate"),
    "saver_30":     ("💎", "Super Saver", "Hit a 30% savings rate"),
    "challenge_1":  ("🎯", "Challenger", "Completed your first challenge"),
    "challenge_5":  ("🏆", "Champion", "Completed 5 challenges"),
    "no_spend_7":   ("🧘", "Zen Wallet", "7-day no-spend streak"),
    "under_budget": ("🛡️", "Disciplined", "3 months in a row under budget"),
    "importer":     ("🏦", "Data Wrangler", "Imported a bank statement"),
}

# ─────────────────────────────────────────────────────────────────────
# CSS — solid, high-contrast, no blur/fade anywhere
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
:root {
    --text-color: #0f172a !important;
    --background-color: #f1f5f9 !important;
    --secondary-background-color: #ffffff !important;
    --primary-color: #1d4ed8 !important;
}
*, *::before, *::after { backdrop-filter: none !important; -webkit-backdrop-filter: none !important; }
#MainMenu, footer { visibility: hidden; }

.stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"],
[data-testid="stBottom"], [data-testid="stDecoration"], [data-testid="stToolbar"] {
    background: #f1f5f9 !important; background-image: none !important; opacity: 1 !important;
}
.block-container { background: transparent !important; padding-top: 2rem !important; padding-bottom: 2rem !important; }

[data-testid="stSidebar"], [data-testid="stSidebar"] > div, [data-testid="stSidebarContent"] {
    background: #ffffff !important; border-right: 1px solid #e2e8f0 !important; opacity: 1 !important;
}
[data-testid="stSidebar"] p, [data-testid="stSidebar"] span,
[data-testid="stSidebar"] label, [data-testid="stSidebar"] div { color: #0f172a !important; opacity: 1 !important; }

.stApp p, .stApp span, .stApp div, .stApp label,
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown p, .stMarkdown span, .element-container { opacity: 1 !important; color: #0f172a !important; }
.stApp [style*="color:"] { opacity: 1 !important; }

[data-testid="stMetric"], [data-testid="metric-container"] {
    background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 12px !important;
    padding: 16px 18px !important; box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p {
    font-size: 11px !important; color: #64748b !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: .06em !important;
}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] > div {
    font-size: 24px !important; font-weight: 800 !important; color: #0f172a !important;
}
[data-testid="stMetricDelta"] { font-size: 12px !important; }

.stTabs [data-baseweb="tab-list"] { background: #e2e8f0 !important; border-radius: 10px !important; padding: 4px !important; gap: 4px !important; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { border-radius: 8px !important; font-size: 13px !important; font-weight: 600 !important;
    color: #374151 !important; background: transparent !important; padding: 8px 14px !important; }
.stTabs [data-baseweb="tab"] p, .stTabs [data-baseweb="tab"] span { color: #374151 !important; }
.stTabs [aria-selected="true"] { background: #ffffff !important; box-shadow: 0 1px 4px rgba(0,0,0,.10) !important; }
.stTabs [aria-selected="true"] p, .stTabs [aria-selected="true"] span { color: #1d4ed8 !important; }

.stButton > button { border-radius: 8px !important; font-weight: 600 !important; font-size: 13px !important;
    border: 1px solid #cbd5e1 !important; color: #0f172a !important; background: #ffffff !important; }
.stButton > button:hover { background: #f1f5f9 !important; border-color: #94a3b8 !important; }
.stButton > button[kind="primary"] { background: #1d4ed8 !important; color: #fff !important; border-color: #1d4ed8 !important; }
.stButton > button[kind="primary"]:hover { background: #1e40af !important; }
.stDownloadButton > button { background: #1d4ed8 !important; color: #fff !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; }

.stTextInput input, .stNumberInput input, .stDateInput input,
[data-baseweb="input"] input, [data-baseweb="select"] > div, .stTextArea textarea {
    background: #ffffff !important; border: 1px solid #cbd5e1 !important; border-radius: 8px !important;
    color: #0f172a !important; font-size: 14px !important; }
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] { background: #ffffff !important; }
[role="option"] { color: #0f172a !important; }

.stRadio label, .stRadio span, .stSlider label, .stSlider span, .stCheckbox label { color: #0f172a !important; }
[data-testid="stExpander"] { background: #ffffff !important; border: 1px solid #e2e8f0 !important; border-radius: 10px !important; }
[data-testid="stExpander"] summary { background: #ffffff !important; font-weight: 700 !important; color: #0f172a !important; }
[data-testid="stExpanderDetails"] { background: #f8fafc !important; border-top: 1px solid #e2e8f0 !important; }
[data-testid="stAlert"] p, [data-testid="stAlert"] div { color: #0f172a !important; }
[data-testid="stFileUploader"], [data-testid="stFileUploadDropzone"] { background: #ffffff !important; }
[data-testid="stDataFrame"], .dvn-scroller { background: #ffffff !important; }
.stCaption, [data-testid="stCaptionContainer"] p { color: #64748b !important; font-size: 12px !important; }

.ss-card { background:#fff; border-radius:14px; padding:18px 22px; border:1px solid #e2e8f0;
           box-shadow:0 1px 4px rgba(0,0,0,.06); margin-bottom:14px; }
.sh  { font-size:15px; font-weight:700; color:#0f172a; margin-bottom:2px; }
.sub { font-size:12px; color:#64748b; margin-bottom:12px; }
.badge-ok   { background:#dcfce7; color:#15803d; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-warn { background:#fef9c3; color:#a16207; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-over { background:#fee2e2; color:#b91c1c; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; }
.alert-box   { background:#fef2f2; border:1px solid #fca5a5; border-radius:10px; padding:14px 18px; margin:8px 0; font-size:13px; color:#991b1b; font-weight:600; }
.success-box { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:14px 18px; margin:8px 0; font-size:13px; color:#166534; font-weight:600; }
.info-box    { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:14px 18px; margin:8px 0; font-size:13px; color:#1e40af; font-weight:500; }
.roast-box   { background:#1f2937; border:1px solid #111827; border-radius:12px; padding:16px 20px; margin:8px 0; font-size:14px; color:#f9fafb !important; font-weight:600; line-height:1.6; }
.roast-box * { color:#f9fafb !important; }
.persona-box { background:linear-gradient(135deg,#1d4ed8,#3b82f6); border-radius:14px; padding:22px; color:#fff; margin-bottom:12px; }
.persona-box h3 { font-size:22px; font-weight:800; margin:6px 0 10px; color:#fff !important; }
.persona-box p  { font-size:13px; line-height:1.7; color:#e8efff !important; }
.persona-box div { color:#fff !important; }
.page-title { font-size:26px; font-weight:800; color:#0f172a; }
.page-sub   { font-size:13px; color:#64748b; margin-top:2px; }
.badge-chip { display:inline-block; background:#f8fafc; border:1px solid #e2e8f0; border-radius:10px;
              padding:8px 12px; margin:4px 6px 4px 0; font-size:12px; font-weight:600; color:#0f172a; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# AUTH / DB HELPERS
# ─────────────────────────────────────────────────────────────────────
def hash_pw(pw, salt=None):
    """PBKDF2-SHA256 with a per-user salt (old sha256 hashes still verify)."""
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt.encode(), 120_000)
    return f"pbkdf2${salt}${dk.hex()}"


def verify_pw(pw, stored):
    if stored and stored.startswith("pbkdf2$"):
        _, salt, _ = stored.split("$", 2)
        return hmac.compare_digest(hash_pw(pw, salt), stored)
    return hmac.compare_digest(hashlib.sha256(pw.encode()).hexdigest(), stored or "")


USER_FIELDS = "id,username,email,budget,alert_email,base_currency"


def _user_row_to_dict(row):
    return {"id": row[0], "username": row[1], "email": row[2], "budget": row[3],
            "alert_email": row[4] or "", "base_currency": row[5] or "INR"}


def create_user(username, email, pw):
    try:
        with get_conn() as c:
            c.execute("INSERT INTO users (username,email,pw_hash) VALUES (?,?,?)",
                      (username.strip(), email.strip().lower(), hash_pw(pw)))
        return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "That username or email is already registered."


def login_user(username, pw):
    with get_conn() as c:
        row = c.execute(f"SELECT {USER_FIELDS},pw_hash FROM users WHERE username=?",
                        (username.strip(),)).fetchone()
    if not row or not verify_pw(pw, row[6]):
        return False, None
    # transparently upgrade legacy sha256 hashes
    if not row[6].startswith("pbkdf2$"):
        with get_conn() as c:
            c.execute("UPDATE users SET pw_hash=? WHERE id=?", (hash_pw(pw), row[0]))
    return True, _user_row_to_dict(row)


def get_user(uid):
    with get_conn() as c:
        row = c.execute(f"SELECT {USER_FIELDS} FROM users WHERE id=?", (uid,)).fetchone()
    return _user_row_to_dict(row) if row else None


def find_user_by_username(username):
    with get_conn() as c:
        row = c.execute(f"SELECT {USER_FIELDS} FROM users WHERE username=?",
                        (username.strip(),)).fetchone()
    return _user_row_to_dict(row) if row else None


def get_transactions(user_id):
    with get_conn() as c:
        rows = c.execute(
            "SELECT id,date,type,category,amount,description,source "
            "FROM transactions WHERE user_id=? ORDER BY date DESC, id DESC",
            (user_id,)).fetchall()
    df = pd.DataFrame(rows, columns=["id", "Date", "Type", "Category",
                                     "Amount", "Description", "Source"])
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"])
        df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
        df["Description"] = df["Description"].fillna("")
    return df


def add_transaction(uid, tdate, typ, cat, amt, desc, source="manual"):
    with get_conn() as c:
        cur = c.execute(
            "INSERT INTO transactions (user_id,date,type,category,amount,description,source) "
            "VALUES (?,?,?,?,?,?,?)",
            (uid, str(tdate), typ, cat, float(amt), (desc or "").strip(), source))
        return cur.lastrowid


def transaction_exists(uid, tdate, amt, desc):
    with get_conn() as c:
        return c.execute(
            "SELECT 1 FROM transactions WHERE user_id=? AND date=? AND "
            "ABS(amount-?)<0.01 AND description=? LIMIT 1",
            (uid, str(tdate), float(amt), (desc or "").strip())).fetchone() is not None


def delete_transaction(txn_id, uid):
    with get_conn() as c:
        c.execute("DELETE FROM transactions WHERE id=? AND user_id=?", (txn_id, uid))


def update_transaction(txn_id, uid, tdate, typ, cat, amt, desc):
    with get_conn() as c:
        c.execute("UPDATE transactions SET date=?,type=?,category=?,amount=?,description=? "
                  "WHERE id=? AND user_id=?",
                  (str(tdate), typ, cat, float(amt), desc, txn_id, uid))


def update_user_field(uid, field, value):
    assert field in {"budget", "alert_email", "base_currency"}
    with get_conn() as c:
        c.execute(f"UPDATE users SET {field}=? WHERE id=?", (value, uid))


def award_badge(uid, code):
    try:
        with get_conn() as c:
            c.execute("INSERT OR IGNORE INTO badges (user_id,code) VALUES (?,?)", (uid, code))
    except Exception:
        pass


def get_badges(uid):
    with get_conn() as c:
        return [r[0] for r in c.execute(
            "SELECT code FROM badges WHERE user_id=? ORDER BY earned", (uid,)).fetchall()]


def seed_demo_data(uid):
    """12 months of realistic data: salary, rent, subscriptions, daily spends, outliers."""
    with get_conn() as c:
        if c.execute("SELECT COUNT(*) FROM transactions WHERE user_id=?", (uid,)).fetchone()[0] > 0:
            return
    rng = np.random.default_rng(42)
    rnd = random.Random(42)
    end = date.today().replace(day=1)
    months = pd.date_range(end=pd.Timestamp(end), periods=12, freq="MS")
    salary = 62000
    subs = [("Netflix subscription", 649, "Entertainment"),
            ("Spotify Premium", 119, "Entertainment"),
            ("Gold's Gym membership", 1500, "Healthcare"),
            ("Airtel Fiber broadband", 999, "Bills")]
    rows = []
    for i, m in enumerate(months):
        if i and i % 4 == 0:
            salary += 4000
        rows.append((m.date().replace(day=1), "Income", "Salary", salary, "Monthly salary credit"))
        if i % 5 == 3:
            rows.append((m.date().replace(day=12), "Income", "Freelance",
                         int(rng.integers(6000, 15000)), "Freelance project payment"))
        rows.append((m.date().replace(day=3), "Expense", "Bills", 18000, "Monthly house rent"))
        for name, amt, cat in subs:
            rows.append((m.date().replace(day=min(7 + subs.index((name, amt, cat)), 27)),
                         "Expense", cat, amt, name))
        merchants = {
            "Food": ["Swiggy order", "Zomato dinner", "Cafe Coffee Day", "Dominos Pizza", "Local grocery"],
            "Transport": ["Uber trip", "Ola cab", "Petrol HP", "Metro recharge"],
            "Shopping": ["Amazon order", "Flipkart order", "Myntra clothing", "DMart supplies"],
            "Entertainment": ["PVR movie tickets", "BookMyShow event", "Steam game"],
            "Healthcare": ["Apollo pharmacy", "Doctor consultation"],
            "Education": ["Udemy course", "Kindle book"],
            "Other": ["Gift for friend", "Misc purchase"],
        }
        for _ in range(int(rng.integers(14, 22))):
            day = int(rng.integers(1, 28))
            d = m.date().replace(day=day)
            weekend = d.weekday() >= 5
            cat = rnd.choice(["Food", "Entertainment", "Shopping"]) if weekend \
                else rnd.choice(["Food", "Transport", "Food", "Shopping", "Other", "Healthcare", "Education"])
            base = {"Food": (180, 900), "Transport": (90, 600), "Shopping": (600, 3500),
                    "Entertainment": (300, 1800), "Healthcare": (200, 1800),
                    "Education": (300, 2500), "Other": (150, 1200)}[cat]
            amt = int(rng.integers(*base) * (1.35 if weekend else 1.0))
            rows.append((d, "Expense", cat, amt, rnd.choice(merchants[cat])))
    # a few genuine outliers so anomaly detection has something to find
    rows.append((months[-3].date().replace(day=17), "Expense", "Shopping", 64000, "Laptop purchase"))
    rows.append((months[-2].date().replace(day=9), "Expense", "Healthcare", 28000, "Dental treatment"))
    with get_conn() as c:
        c.executemany(
            "INSERT INTO transactions (user_id,date,type,category,amount,description,source) "
            "VALUES (?,?,?,?,?,?, 'demo')",
            [(uid, str(d), t, cat, float(a), desc) for d, t, cat, a, desc in rows])

# ─────────────────────────────────────────────────────────────────────
# CORE ANALYTICS
# ─────────────────────────────────────────────────────────────────────
def build_summary(df):
    if df.empty:
        return pd.DataFrame(columns=["Income", "Expense", "Savings"])
    d2 = df.copy()
    d2["Month"] = d2["Date"].dt.to_period("M")
    inc = d2[d2["Type"] == "Income"].groupby("Month")["Amount"].sum()
    exp = d2[d2["Type"] == "Expense"].groupby("Month")["Amount"].sum()
    idx = sorted(set(inc.index) | set(exp.index))
    s = pd.DataFrame({"Income": inc, "Expense": exp}).reindex(idx).fillna(0)
    s["Savings"] = s["Income"] - s["Expense"]
    return s


def ml_forecast(summary):
    """Linear regression on monthly expense. Returns (prediction, slope, fitted+1 line, r2)."""
    if len(summary) < 2:
        return 0.0, 0.0, [], 0.0
    X = np.arange(len(summary)).reshape(-1, 1)
    y = summary["Expense"].values.astype(float)
    model = LinearRegression().fit(X, y)
    future = np.arange(len(summary) + 1).reshape(-1, 1)
    line = model.predict(future)
    pred = float(line[-1])
    r2 = float(model.score(X, y))
    return max(0.0, pred), float(model.coef_[0]), line.tolist(), r2


def ml_cluster(df):
    """KMeans on expense behaviour; clusters are *named from their centroids*, not hardcoded."""
    exp = df[df["Type"] == "Expense"].copy()
    if exp.empty:
        return exp, None, {}
    exp["DayOfWeek"] = exp["Date"].dt.dayofweek
    exp["IsWeekend"] = (exp["DayOfWeek"] >= 5).astype(int)
    exp["DayOfMonth"] = exp["Date"].dt.day
    exp["IsEssential"] = exp["Category"].isin(ESSENTIAL_CATS).astype(int)

    if len(exp) < 12:
        exp["Cluster"] = 0
        exp["ClusterName"] = "Balanced Spender"
        return exp, "Balanced Spender", {"Balanced Spender": "#1d4ed8"}

    feats = exp[["Amount", "IsWeekend", "IsEssential", "DayOfMonth"]].astype(float)
    k = 3 if len(exp) >= 30 else 2
    scaled = StandardScaler().fit_transform(feats)
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    exp["Cluster"] = km.fit_predict(scaled)

    stats = exp.groupby("Cluster").agg(
        avg=("Amount", "mean"), share=("Amount", "sum"),
        weekend=("IsWeekend", "mean"), essential=("IsEssential", "mean"),
        n=("Amount", "size")).sort_values("avg", ascending=False)

    names, used = {}, set()
    for cid, row in stats.iterrows():
        if row["avg"] >= stats["avg"].max() * 0.95 and "Big-Ticket Buyer" not in used:
            nm = "Big-Ticket Buyer"
        elif row["essential"] >= 0.55:
            nm = "Essential Focused"
        elif row["weekend"] >= 0.45:
            nm = "Weekend Splurger"
        else:
            nm = "Everyday Small Spender"
        while nm in used:
            nm += " II"
        used.add(nm)
        names[cid] = nm
    exp["ClusterName"] = exp["Cluster"].map(names)

    dominant = names[stats["share"].idxmax()]
    palette = ["#1d4ed8", "#7c3aed", "#059669", "#d97706"]
    colors = {names[cid]: palette[i % len(palette)] for i, cid in enumerate(sorted(names))}
    return exp, dominant, colors


PERSONA_DESC = {
    "Big-Ticket Buyer": "A handful of large purchases drive most of your outflow. Your day-to-day habits are fine — it's the occasional big hit that moves the needle. Plan large buys a month ahead and pre-fund them from a sinking fund.",
    "Weekend Splurger": "Saturdays and Sundays carry a disproportionate share of your spending — dining, entertainment and shopping. A fixed weekend allowance is the single highest-leverage fix for you.",
    "Essential Focused": "Most of your money goes to bills, transport, health and education. Discipline is strong; the remaining wins come from renegotiating fixed costs rather than cutting more.",
    "Everyday Small Spender": "Lots of small, frequent transactions that quietly add up. Death by a thousand cuts — batch your purchases and try a 24-hour rule on anything non-essential.",
    "Balanced Spender": "Spending is well distributed with low variance. Keep doing what you're doing and push the savings rate up a notch.",
}
PERSONA_ICON = {"Big-Ticket Buyer": "💥", "Weekend Splurger": "🎉",
                "Essential Focused": "🎯", "Everyday Small Spender": "🐜",
                "Balanced Spender": "⚖️"}


def detect_anomalies(df, ignore_recurring=True):
    """Isolation Forest on amount + timing + category z-score.

    Known recurring bills (rent, subscriptions) are excluded first, otherwise every
    salary-sized fixed charge gets flagged forever. Flags are then filtered for
    materiality so the list stays short and actionable.
    """
    exp = df[df["Type"] == "Expense"].copy()
    if len(exp) < 20:
        return pd.DataFrame(columns=list(exp.columns) + ["Score", "Reason"])
    if ignore_recurring:
        rec = detect_recurring(df)
        if not rec.empty:
            rec_keys = set(rec["Merchant"].map(_normalize_desc))
            exp = exp[~exp["Description"].map(_normalize_desc).isin(rec_keys)]
        if len(exp) < 20:
            return pd.DataFrame(columns=list(exp.columns) + ["Score", "Reason"])
    exp["DayOfWeek"] = exp["Date"].dt.dayofweek
    exp["MonthNum"] = exp["Date"].dt.month
    cat_mean = exp.groupby("Category")["Amount"].transform("mean")
    cat_std = exp.groupby("Category")["Amount"].transform("std").replace(0, np.nan)
    exp["CatZ"] = ((exp["Amount"] - cat_mean) / cat_std).fillna(0)

    feats = exp[["Amount", "DayOfWeek", "MonthNum", "CatZ"]].astype(float)
    model = IsolationForest(contamination="auto", random_state=42, n_estimators=200)
    exp["Flag"] = model.fit_predict(StandardScaler().fit_transform(feats))
    exp["Score"] = model.score_samples(StandardScaler().fit_transform(feats))
    out = exp[exp["Flag"] == -1].copy()
    if out.empty:
        return out
    # materiality: only surface things that are genuinely large for their category
    p85 = exp.groupby("Category")["Amount"].quantile(0.85)
    out = out[(out["CatZ"] >= 1.5) | (out["Amount"] >= out["Category"].map(p85) * 1.5)]
    if out.empty:
        return out
    out["Reason"] = np.where(
        out["CatZ"] > 2.0, "Far above your usual " + out["Category"] + " spend",
        np.where(out["Date"].dt.dayofweek >= 5, "Unusually large weekend transaction",
                 "Unusual amount/timing combination"))
    return out.sort_values("Amount", ascending=False)


def _normalize_desc(s):
    s = re.sub(r"[^a-z ]", " ", str(s).lower())
    s = re.sub(r"\b(upi|neft|imps|pos|txn|ref|no|dr|cr|payment|paid|to|from|india|pvt|ltd)\b", " ", s)
    return " ".join(s.split()[:3]).strip()


def detect_recurring(df):
    """Real cadence detection: >=3 hits, >=3 distinct months, stable amount, ~monthly gap."""
    exp = df[df["Type"] == "Expense"].copy()
    if exp.empty:
        return pd.DataFrame()
    exp["Key"] = exp["Description"].map(_normalize_desc)
    exp = exp[exp["Key"].str.len() >= 3]
    if exp.empty:
        return pd.DataFrame()
    rows = []
    for key, g in exp.groupby("Key"):
        g = g.sort_values("Date")
        if len(g) < 3 or g["Date"].dt.to_period("M").nunique() < 3:
            continue
        amt = g["Amount"]
        cv = amt.std() / amt.mean() if amt.mean() else 1
        gaps = g["Date"].diff().dt.days.dropna()
        med_gap = gaps.median() if not gaps.empty else 0
        gap_cv = (gaps.std() / med_gap) if len(gaps) > 1 and med_gap else 1
        dom_spread = g["Date"].dt.day.std() if len(g) > 1 else 99
        monthly = (20 <= med_gap <= 45) and (gap_cv <= 0.35 or dom_spread <= 4)
        weekly = (5 <= med_gap <= 9) and gap_cv <= 0.25
        if cv > 0.25 or not (monthly or weekly):
            continue
        cadence = "Monthly" if monthly else "Weekly"
        per_year = 12 if cadence == "Monthly" else 52
        rows.append({
            "Merchant": g["Description"].mode().iloc[0][:40],
            "Category": g["Category"].mode().iloc[0],
            "Cadence": cadence,
            "Occurrences": int(len(g)),
            "Avg Amount": round(float(amt.mean()), 2),
            "Last Charged": g["Date"].max().date(),
            "Next Expected": (g["Date"].max() + timedelta(days=int(med_gap))).date(),
            "Annual Cost": round(float(amt.mean()) * per_year, 2),
        })
    out = pd.DataFrame(rows)
    return out.sort_values("Annual Cost", ascending=False) if not out.empty else out


def calculate_daily_allowance(budget, df, ref=None):
    """Safe-to-spend today = (budget - this month's spend so far) / days left incl. today."""
    ref = ref or date.today()
    last_day = calmod.monthrange(ref.year, ref.month)[1]
    days_left = max(1, last_day - ref.day + 1)
    if df.empty:
        return int(budget / days_left), 0, days_left
    m = df[(df["Type"] == "Expense") &
           (df["Date"].dt.year == ref.year) & (df["Date"].dt.month == ref.month)]
    spent = float(m["Amount"].sum())
    return int(max(0, (budget - spent)) / days_left), spent, days_left


def calculate_runway(df):
    """Days of cash left = net savings / recent daily burn (last 90 days)."""
    exp = df[df["Type"] == "Expense"]
    if df.empty or exp.empty:
        return None, 0.0, 0.0
    net = float(df[df["Type"] == "Income"]["Amount"].sum() - exp["Amount"].sum())
    end = df["Date"].max()
    window = exp[exp["Date"] >= end - pd.Timedelta(days=90)]
    span = max(1, (end - max(window["Date"].min(), end - pd.Timedelta(days=90))).days + 1) \
        if not window.empty else 1
    burn = float(window["Amount"].sum()) / span if not window.empty else 0.0
    if burn <= 0:
        return None, net, 0.0
    if net <= 0:
        return 0, net, burn
    return int(net / burn), net, burn


def calculate_health_score(summary, budget, df):
    """Transparent 0-100 score with a component breakdown."""
    if summary.empty:
        return 50, {}
    inc, exp = summary["Income"].sum(), summary["Expense"].sum()
    rate = (inc - exp) / inc if inc > 0 else 0.0
    savings_pts = float(np.clip(rate / 0.30, 0, 1) * 40)

    if budget > 0:
        under = (summary["Expense"] <= budget).mean()
        overshoot = np.clip((summary["Expense"] / budget - 1).clip(lower=0).mean(), 0, 1)
        adherence_pts = float(np.clip(under * 25 - overshoot * 10, 0, 25))
    else:
        adherence_pts = 12.5

    e = summary["Expense"]
    cv = (e.std() / e.mean()) if len(e) > 1 and e.mean() else 0.5
    consistency_pts = float(np.clip(1 - cv / 0.5, 0, 1) * 15)

    days, _, _ = calculate_runway(df)
    runway_pts = float(np.clip((days or 0) / 180, 0, 1) * 10)

    an = detect_anomalies(df)
    flagged_value = float(an["Amount"].sum()) if not an.empty else 0.0
    share = flagged_value / exp if exp else 0.0
    anomaly_pts = float(np.clip(1 - share / 0.25, 0, 1) * 10)

    parts = {"Savings rate": (savings_pts, 40), "Budget adherence": (adherence_pts, 25),
             "Spending consistency": (consistency_pts, 15), "Cash runway": (runway_pts, 10),
             "Anomaly-free": (anomaly_pts, 10)}
    score = int(round(sum(v for v, _ in parts.values())))
    return int(np.clip(score, 0, 100)), parts


def score_band(score):
    if score >= 80:  return "Excellent", "#059669"
    if score >= 65:  return "Good", "#0891b2"
    if score >= 45:  return "Fair", "#d97706"
    return "Needs Work", "#dc2626"


def no_spend_streak(df, ref=None):
    """Consecutive days (ending yesterday/today) with zero expenses."""
    ref = ref or date.today()
    if df.empty:
        return 0
    spend_days = set(df[df["Type"] == "Expense"]["Date"].dt.date)
    streak, d = 0, ref
    while d not in spend_days and streak < 365:
        streak += 1
        d -= timedelta(days=1)
    return streak


def daily_spend_calendar(df, days=182):
    """Real daily totals for the calendar heatmap."""
    if df.empty:
        return pd.Series(dtype=float)
    end = max(df["Date"].max().date(), date.today())
    start = end - timedelta(days=days)
    exp = df[(df["Type"] == "Expense") & (df["Date"].dt.date >= start)]
    s = exp.groupby(exp["Date"].dt.date)["Amount"].sum()
    idx = pd.date_range(start, end, freq="D").date
    return s.reindex(idx, fill_value=0.0)

# ─────────────────────────────────────────────────────────────────────
# SMART AUTO-TAGGING  (keyword + fuzzy, works with or without thefuzz)
# ─────────────────────────────────────────────────────────────────────
KEYWORD_MAP = {
    "Food": ["swiggy", "zomato", "restaurant", "cafe", "coffee", "pizza", "dominos", "mcdonald",
             "kfc", "starbucks", "bakery", "dinner", "lunch", "breakfast", "grocery", "bigbasket",
             "blinkit", "zepto", "dmart", "hotel", "dhaba", "food", "eat", "biryani", "juice"],
    "Transport": ["uber", "ola", "rapido", "petrol", "fuel", "diesel", "metro", "irctc", "train",
                  "bus", "taxi", "cab", "toll", "fastag", "parking", "flight", "indigo", "airlines",
                  "redbus", "auto"],
    "Shopping": ["amazon", "amzn", "flipkart", "myntra", "ajio", "meesho", "nykaa", "mall", "store",
                 "clothes", "apparel", "mktp", "decathlon", "ikea", "croma", "reliance digital", "shop",
                 "lifestyle", "westside", "zara"],
    "Bills": ["electricity", "wifi", "internet", "broadband", "rent", "mobile", "recharge",
              "water", "gas", "airtel", "jio", "vodafone", "bsnl", "dth", "maintenance",
              "insurance", "premium", "emi", "loan", "tax", "bill"],
    "Entertainment": ["netflix", "spotify", "prime video", "hotstar", "jiocinema", "sony liv",
                      "bookmyshow", "pvr", "inox", "cinema", "movie", "game", "steam", "playstation",
                      "concert", "youtube premium", "subscription"],
    "Healthcare": ["pharmacy", "apollo", "medplus", "hospital", "clinic", "doctor", "medicine",
                   "dental", "lab", "diagnostic", "gym", "fitness", "cult", "health"],
    "Education": ["udemy", "coursera", "byju", "unacademy", "course", "tuition", "school",
                  "college", "exam", "book", "kindle", "stationery", "fees"],
}
INCOME_KEYWORDS = ["salary", "credited by employer", "freelance", "invoice", "refund", "cashback",
                   "dividend", "interest credit", "payout", "bonus", "stipend", "reimbursement"]


def auto_tag_category(description, amount=None, income=False):
    """Returns (category, confidence 0-100). Exact keyword hit beats fuzzy."""
    if income:
        d = str(description or "").lower()
        for cat, words in (("Salary", ["salary", "payroll", "wages", "stipend"]),
                           ("Freelance", ["freelance", "invoice", "client", "consult", "gig"]),
                           ("Investment", ["dividend", "interest", "mutual fund", "stock", "sip"]),
                           ("Refund", ["refund", "cashback", "reimburse", "return"])):
            if any(w in d for w in words):
                return cat, 95
        return "Other", 40
    desc = str(description or "").lower().strip()
    if not desc:
        return "Other", 0
    best_cat, best_score = "Other", 0
    for cat, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw), desc):
                score = 100 if len(kw) > 4 else 88
                if score > best_score:
                    best_cat, best_score = cat, score
    if best_score >= 88:
        return best_cat, best_score
    if FUZZ_OK:
        for cat, keywords in KEYWORD_MAP.items():
            for kw in keywords:
                if len(kw) < 4:
                    continue
                score = max(fuzz.partial_ratio(desc, kw), fuzz.token_set_ratio(desc, kw))
                if score > best_score:
                    best_cat, best_score = cat, score
        if best_score >= 82:
            return best_cat, int(best_score)
    if amount is not None and amount >= 8000:
        return "Bills", 40  # large untagged debits are usually fixed costs
    return "Other", int(best_score)


def guess_type_from_text(text):
    t = str(text or "").lower()
    if any(k in t for k in INCOME_KEYWORDS) or re.search(r"\b(received|got|earned|credit)\b", t):
        return "Income"
    return "Expense"


# ─────────────────────────────────────────────────────────────────────
# VOICE / NATURAL LANGUAGE TRANSACTION PARSER
# ─────────────────────────────────────────────────────────────────────
WORD_NUM = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100, "thousand": 1000,
    "lakh": 100000, "lac": 100000, "crore": 10000000
}


def parse_amount_text(text):
    t = str(text or "").lower().replace(",", "")
    m = re.search(r"(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d{1,2})?)\s*(k|thousand|lakh|lac|crore)?", t)
    if m:
        val = float(m.group(1))
        unit = m.group(2)
        if unit in ("k", "thousand"):
            val *= 1000
        elif unit in ("lakh", "lac"):
            val *= 100000
        elif unit == "crore":
            val *= 10000000
        return round(val, 2)
    tokens = [tok for tok in re.findall(r"\b[a-z]+\b", t) if tok in WORD_NUM]
    if tokens:
        total = 0
        curr = 0
        for tok in tokens:
            v = WORD_NUM[tok]
            if v in (100, 1000, 100000, 10000000):
                curr = (curr or 1) * v
                total += curr
                curr = 0
            else:
                curr += v
        total += curr
        if total > 0:
            return float(total)
    return None


def parse_relative_date(text, ref=None):
    ref = ref or date.today()
    t = str(text or "").lower()
    if "day before yesterday" in t:
        return ref - timedelta(days=2)
    if "yesterday" in t:
        return ref - timedelta(days=1)
    if "tomorrow" in t:
        return ref + timedelta(days=1)
    m = re.search(r"(\d+)\s*days?\s*ago", t)
    if m:
        return ref - timedelta(days=int(m.group(1)))
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", t)
    if m:
        try:
            d, mo = int(m.group(1)), int(m.group(2))
            y = int(m.group(3) or ref.year)
            y = y + 2000 if y < 100 else y
            return date(y, mo, d)
        except ValueError:
            pass
    return ref


def parse_voice_text(text, ref=None):
    """'spent 450 on uber yesterday' -> dict ready to save. Returns None if no amount."""
    amt = parse_amount_text(text)
    if not amt:
        return None
    typ = guess_type_from_text(text)
    cat, conf = auto_tag_category(text, amt, income=(typ == "Income"))
    tail = re.sub(r"(?i)[\d,]+(?:\.\d+)?\s*(k|thousand|lakh|lac)?", " ", str(text))
    tail = re.sub(r"(?i)\b(spent|paid|spend|for|on|rs\.?|rupees|inr|₹|received|got|earned|"
                  r"today|yesterday|day before|ago|days)\b", " ", tail)
    desc = " ".join(tail.split()).strip(" .,-") or str(text).strip()
    return {"date": parse_relative_date(text, ref), "type": typ, "category": cat,
            "amount": amt, "description": desc[:60], "confidence": conf}


# ─────────────────────────────────────────────────────────────────────
# RECEIPT OCR
# ─────────────────────────────────────────────────────────────────────
AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr)?\s*([0-9][0-9,]*\.?[0-9]{0,2})")


def ocr_receipt(file_bytes):
    """Real OCR when pytesseract is installed. Returns (data|None, raw_text, error|None)."""
    if not OCR_OK:
        return None, "", ("Tesseract OCR is not installed. Install it to enable automatic "
                          "extraction:  pip install pytesseract pillow  +  the Tesseract binary "
                          "(apt install tesseract-ocr / brew install tesseract).")
    try:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != "L":
            img = img.convert("L")
        if max(img.size) < 1200:
            ratio = 1200 / max(img.size)
            img = img.resize((int(img.width * ratio), int(img.height * ratio)))
        tess_cfg = f'--tessdata-dir {TESSDATA_DIR}' if TESSDATA_DIR else None
        text = pytesseract.image_to_string(img, config=tess_cfg)
    except Exception as e:
        return None, "", f"OCR failed: {e}"

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    total = None
    for line in reversed(lines):
        if re.search(r"(?i)\b(grand\s*total|total amount|net payable|amount payable|total)\b", line):
            cands = [float(m.replace(",", "")) for m in AMOUNT_RE.findall(line) if m.strip(".,")]
            cands = [c for c in cands if c > 0]
            if cands:
                total = max(cands)
                break
    if total is None:
        cands = []
        for line in lines:
            for m in AMOUNT_RE.findall(line):
                try:
                    v = float(m.replace(",", ""))
                    if 1 <= v <= 1_000_000:
                        cands.append(v)
                except ValueError:
                    pass
        total = max(cands) if cands else None

    merchant = next((l for l in lines[:6] if len(re.sub(r"[^A-Za-z]", "", l)) >= 4), "Receipt")
    rdate = None
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", text)
    if m:
        try:
            y = int(m.group(3)); y = y + 2000 if y < 100 else y
            rdate = date(y, int(m.group(2)), int(m.group(1)))
        except ValueError:
            rdate = None
    cat, conf = auto_tag_category(merchant + " " + text[:300], total)
    if total is None:
        return None, text, "Could not find an amount on that receipt — enter it manually below."
    return {"amount": total, "merchant": merchant[:50], "date": rdate or date.today(),
            "category": cat, "confidence": conf}, text, None


# ─────────────────────────────────────────────────────────────────────
# FX + CRYPTO
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def get_fx_rates(base="INR"):
    """1 base -> X other. Live when possible, cached 1h, static fallback."""
    if REQ_OK:
        for url in (f"https://open.er-api.com/v6/latest/{base}",
                    f"https://api.exchangerate-api.com/v4/latest/{base}"):
            try:
                r = requests.get(url, timeout=6)
                if r.ok:
                    data = r.json()
                    rates = data.get("rates") or {}
                    if rates:
                        return {k: float(v) for k, v in rates.items() if k in CURRENCIES}, "live"
            except Exception:
                continue
    return dict(FALLBACK_RATES), "offline"


@st.cache_data(ttl=600, show_spinner=False)
def get_crypto_prices():
    """BTC/ETH in USD. yfinance first, CoinGecko second, else None."""
    if YF_OK:
        try:
            out = {}
            for sym, key in (("BTC-USD", "BTC"), ("ETH-USD", "ETH")):
                h = yf.Ticker(sym).history(period="2d")["Close"]
                if len(h):
                    out[key] = float(h.iloc[-1])
                    if len(h) > 1:
                        out[key + "_chg"] = float((h.iloc[-1] / h.iloc[-2] - 1) * 100)
            if out:
                return out, "yfinance"
        except Exception:
            pass
    if REQ_OK:
        try:
            r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                             params={"ids": "bitcoin,ethereum", "vs_currencies": "usd",
                                     "include_24hr_change": "true"}, timeout=6)
            if r.ok:
                d = r.json()
                return {"BTC": d["bitcoin"]["usd"], "BTC_chg": d["bitcoin"].get("usd_24h_change", 0),
                        "ETH": d["ethereum"]["usd"], "ETH_chg": d["ethereum"].get("usd_24h_change", 0)}, "coingecko"
        except Exception:
            pass
    return None, "unavailable"


def fmt_money(amount_inr, currency="INR", rates=None):
    """Format an INR amount in the user's display currency."""
    rate = 1.0
    if currency != "INR":
        rates = rates or get_fx_rates("INR")[0]
        rate = rates.get(currency, FALLBACK_RATES.get(currency, 1.0))
    sym = CURRENCIES.get(currency, "")
    val = amount_inr * rate
    if abs(val) >= 1000 or float(val).is_integer():
        return f"{sym}{val:,.0f}"
    return f"{sym}{val:,.2f}"


# ─────────────────────────────────────────────────────────────────────
# MULTI-BANK CSV/EXCEL ETL
# ─────────────────────────────────────────────────────────────────────
COL_PATTERNS = {
    "date": [r"^txn.?date", r"^value.?date", r"^transaction.?date", r"\bdate\b", r"^dt$", r"posting date"],
    "description": [r"^description", r"narration", r"particular", r"^remarks", r"^merchant",
                    r"^payee", r"transaction remarks", r"details"],
    "debit": [r"withdrawal", r"debit", r"dr amount", r"^dr$", r"money out", r"paid out"],
    "credit": [r"deposit", r"credit", r"cr amount", r"^cr$", r"money in", r"paid in"],
    "amount": [r"^amount", r"\bamt\b", r"transaction amount", r"^value$"],
    "type": [r"^type$", r"dr\s*/\s*cr", r"credit/debit", r"txn type", r"transaction type"],
    "category": [r"^category$", r"^cat$", r"tag"],
}


def _match_column(colname):
    """Returns (field, score) for the best matching field, or (None, 0)."""
    c = str(colname).strip().lower()
    best, best_score = None, 0
    for field, pats in COL_PATTERNS.items():
        for rank, p in enumerate(pats):
            if re.search(p, c):
                score = 100 - rank * 5 + (10 if re.fullmatch(p.strip("^$"), c) else 0)
                if score > best_score:
                    best, best_score = field, score
    return best, best_score


def _resolve_columns(header):
    """Assign each field to its single best column (a column is used once)."""
    scored = []
    for i, h in enumerate(header):
        f, s = _match_column(h)
        if f:
            scored.append((s, i, f, h))
    scored.sort(reverse=True)
    col, taken = {}, set()
    for s, i, f, h in scored:
        if f in col or i in taken:
            continue
        col[f] = h
        taken.add(i)
    return col


def _repair_split_numbers(cells, target):
    """'1,240.00' split by an unquoted comma comes through as ['1','240.00'] — rejoin it."""
    out, i = [], 0
    while i < len(cells):
        cur = str(cells[i]).strip()
        while (len(cells) - i > 1 and len(out) + (len(cells) - i) > target
               and re.fullmatch(r"-?\d{1,3}", cur)
               and re.fullmatch(r"\d{3}(\.\d+)?", str(cells[i + 1]).strip())):
            cur = cur + str(cells[i + 1]).strip()
            i += 1
        out.append(cur)
        i += 1
    return out


def _read_any(file):
    """Read CSV/Excel into a header-less, ragged-safe DataFrame of strings."""
    import csv as _csv
    name = getattr(file, "name", "upload.csv").lower()
    raw_bytes = file.read() if hasattr(file, "read") else open(file, "rb").read()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw_bytes), header=None, dtype=str)
    text = None
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(enc)
            break
        except Exception:
            continue
    if text is None:
        raise ValueError("Unreadable file — export it as a plain CSV and try again.")
    sample = "\n".join(text.splitlines()[:30])
    try:
        delim = _csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        delim = max([",", ";", "\t", "|"], key=sample.count)
    rows = [r for r in _csv.reader(io.StringIO(text), delimiter=delim)]
    rows = [r for r in rows if any(str(x).strip() for x in r)]
    if not rows:
        return pd.DataFrame()
    width = max(len(r) for r in rows)          # pad ragged rows instead of dropping them
    df = pd.DataFrame([r + [""] * (width - len(r)) for r in rows], dtype=str)
    df.attrs["rows"] = rows
    return df


def _find_header_row(raw):
    """Bank statements often have 5-20 junk rows before the real header."""
    best, best_hits = 0, 0
    for i in range(min(25, len(raw))):
        cells = [str(x) for x in raw.iloc[i].tolist()]
        fields = {f for f, _ in (_match_column(x) for x in cells) if f}
        hits = len(fields)
        if "date" in fields and (fields & {"amount", "debit", "credit"}):
            hits += 2
        if hits > best_hits:
            best, best_hits = i, hits
    return best


def import_file(file, uid):
    """Detects header row, maps columns across bank formats, auto-tags, de-duplicates."""
    stats = {"imported": 0, "duplicates": 0, "skipped": 0, "errors": [], "mapping": {}}
    try:
        raw = _read_any(file)
    except Exception as e:
        stats["errors"].append(str(e))
        return stats
    if raw.empty:
        stats["errors"].append("The file is empty.")
        return stats

    hrow = _find_header_row(raw)
    header = [str(x).strip() for x in raw.iloc[hrow].tolist()]
    rows = raw.attrs.get("rows")
    if rows:
        # realign rows whose numbers were split by unquoted thousands separators
        hwidth = len(rows[hrow])
        body = [_repair_split_numbers(r, hwidth) if len(r) > hwidth else r
                for r in rows[hrow + 1:]]
        body = [r + [""] * (len(header) - len(r)) for r in body]
        data = pd.DataFrame([r[:len(header)] for r in body], columns=header, dtype=str)
    else:
        data = raw.iloc[hrow + 1:].copy()
        data.columns = header
    data = data.dropna(how="all")

    col = _resolve_columns(header)
    stats["mapping"] = dict(col)

    if "date" not in col or not (col.get("amount") or col.get("debit") or col.get("credit")):
        stats["errors"].append(
            "Could not find a date column plus an amount (or debit/credit) column. "
            f"Columns seen: {', '.join([h for h in header if h and h != 'nan'][:12])}")
        return stats

    # decide day-first vs month-first once, from the whole column
    dayfirst = True
    try:
        firsts = pd.to_numeric(data[col["date"]].astype(str)
                               .str.extract(r"^\s*(\d{1,2})[/-]")[0], errors="coerce").dropna()
        seconds = pd.to_numeric(data[col["date"]].astype(str)
                                .str.extract(r"^\s*\d{1,2}[/-](\d{1,2})")[0], errors="coerce").dropna()
        if (firsts > 12).any():
            dayfirst = True
        elif (seconds > 12).any():
            dayfirst = False
    except Exception:
        pass
    stats["date_format"] = "day-first (DD/MM)" if dayfirst else "month-first (MM/DD)"

    for i, row in data.iterrows():
        try:
            rawdate = row[col["date"]]
            if pd.isna(rawdate) or str(rawdate).strip() == "":
                stats["skipped"] += 1
                continue
            d = pd.to_datetime(str(rawdate).strip(), dayfirst=dayfirst, errors="coerce")
            if pd.isna(d):
                d = pd.to_datetime(str(rawdate).strip(), errors="coerce")
            if pd.isna(d):
                stats["skipped"] += 1
                continue
            d = d.date()

            def num(key):
                if key not in col:
                    return None
                v = str(row.get(col[key], "")).replace(",", "").replace("₹", "").strip()
                v = re.sub(r"[^\d.\-]", "", v)
                if v in ("", "-", "."):
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None

            debit, credit, amount = num("debit"), num("credit"), num("amount")
            if debit and debit > 0:
                typ, amt = "Expense", debit
            elif credit and credit > 0:
                typ, amt = "Income", credit
            elif amount is not None and amount != 0:
                amt = abs(amount)
                typ = "Expense" if amount < 0 else "Income"
                if "type" in col:
                    tokens = set(re.split(r"[^a-z]+", str(row.get(col["type"], "")).lower()))
                    if tokens & {"cr", "credit", "income", "deposit", "inflow", "received"}:
                        typ = "Income"
                    elif tokens & {"dr", "debit", "expense", "withdrawal", "withdraw", "outflow"}:
                        typ = "Expense"
                elif amount > 0:
                    typ = "Expense"     # single unsigned amount column = spending statement
            else:
                stats["skipped"] += 1
                continue

            desc = str(row.get(col.get("description", ""), "") or "").strip()
            desc = re.sub(r"\s+", " ", desc)[:120]
            if "category" in col and str(row.get(col["category"], "")).strip() in ALL_CATEGORIES:
                cat = str(row[col["category"]]).strip()
            elif typ == "Income":
                cat, _ = auto_tag_category(desc, amt, income=True)
            else:
                cat, _ = auto_tag_category(desc, amt)

            if transaction_exists(uid, d, amt, desc):
                stats["duplicates"] += 1
                continue
            add_transaction(uid, d, typ, cat, amt, desc, source="import")
            stats["imported"] += 1
        except Exception as e:
            stats["errors"].append(f"Row {i + 2}: {e}")
    if stats["imported"]:
        award_badge(uid, "importer")
    return stats

# ─────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────
def export_excel(df, summary, recurring=None, anomalies=None):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        out = df.drop(columns=["id"], errors="ignore").copy()
        if not out.empty:
            out["Date"] = out["Date"].dt.date
        out.to_excel(w, sheet_name="Transactions", index=False)
        s = summary.copy()
        s.index = s.index.astype(str)
        s.to_excel(w, sheet_name="Monthly Summary")
        if not df.empty:
            cat = (df[df["Type"] == "Expense"].groupby("Category")["Amount"]
                   .agg(["sum", "mean", "count"]).rename(
                       columns={"sum": "Total", "mean": "Average", "count": "Transactions"}))
            cat.to_excel(w, sheet_name="By Category")
        if recurring is not None and not recurring.empty:
            recurring.to_excel(w, sheet_name="Subscriptions", index=False)
        if anomalies is not None and not anomalies.empty:
            a = anomalies[["Date", "Category", "Amount", "Description", "Reason"]].copy()
            a["Date"] = a["Date"].dt.date
            a.to_excel(w, sheet_name="Anomalies", index=False)
    return buf.getvalue()


def export_pdf(summary, user, df, health, persona, recurring):
    if not REPORTLAB_OK:
        return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=30, bottomMargin=30,
                            leftMargin=30, rightMargin=30)
    sty = getSampleStyleSheet()
    h2 = ParagraphStyle("h2x", parent=sty["Heading2"], textColor=rl_colors.HexColor("#1d4ed8"),
                        spaceBefore=14, fontSize=13)
    small = ParagraphStyle("small", parent=sty["Normal"], fontSize=9,
                           textColor=rl_colors.HexColor("#475569"))
    el = [Paragraph("SentientSpend AI — Financial Report", sty["Title"]),
          Paragraph(f"{user['username']} &nbsp;|&nbsp; generated {datetime.now():%d %b %Y, %H:%M}", small),
          Spacer(1, 10)]

    inc, exp = summary["Income"].sum(), summary["Expense"].sum()
    rate = (inc - exp) / inc * 100 if inc else 0
    kpi = Table([["Total income", "Total expense", "Net savings", "Savings rate", "Health score"],
                 [f"Rs {inc:,.0f}", f"Rs {exp:,.0f}", f"Rs {inc - exp:,.0f}",
                  f"{rate:.1f}%", f"{health}/100"]], colWidths=[105] * 5)
    kpi.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9), ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7)]))
    el += [kpi, Paragraph("Monthly breakdown", h2)]

    rows = [["Month", "Income", "Expense", "Savings", "Savings %"]]
    for m, r in summary.iterrows():
        pct = (r["Savings"] / r["Income"] * 100) if r["Income"] else 0
        rows.append([str(m), f"{r['Income']:,.0f}", f"{r['Expense']:,.0f}",
                     f"{r['Savings']:,.0f}", f"{pct:.0f}%"])
    rows.append(["TOTAL", f"{inc:,.0f}", f"{exp:,.0f}", f"{inc - exp:,.0f}", f"{rate:.0f}%"])
    t = Table(rows, colWidths=[105] * 5)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [rl_colors.white, rl_colors.HexColor("#f8fafc")]),
        ("BACKGROUND", (0, -1), (-1, -1), rl_colors.HexColor("#dcfce7")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#e2e8f0")),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    el.append(t)

    if not df.empty:
        cat = df[df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values(ascending=False)
        crows = [["Category", "Total", "Share"]] + [
            [c, f"{v:,.0f}", f"{v / cat.sum() * 100:.1f}%"] for c, v in cat.items()]
        ct = Table(crows, colWidths=[175, 175, 175])
        ct.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#7c3aed")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [rl_colors.white, rl_colors.HexColor("#faf5ff")]),
            ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#e2e8f0")),
            ("ALIGN", (1, 0), (-1, -1), "RIGHT")]))
        el += [Paragraph("Spending by category", h2), ct]

    if recurring is not None and not recurring.empty:
        rrows = [["Merchant", "Cadence", "Avg", "Annual cost"]] + [
            [str(r["Merchant"])[:32], r["Cadence"], f"{r['Avg Amount']:,.0f}",
             f"{r['Annual Cost']:,.0f}"] for _, r in recurring.head(12).iterrows()]
        rt = Table(rrows, colWidths=[220, 90, 105, 110])
        rt.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0891b2")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.4, rl_colors.HexColor("#e2e8f0")),
            ("ALIGN", (2, 0), (-1, -1), "RIGHT")]))
        el += [Paragraph("Recurring subscriptions detected", h2), rt]

    el += [Paragraph("ML spending persona", h2),
           Paragraph(f"<b>{persona}</b> — {PERSONA_DESC.get(persona, '')}", sty["Normal"]),
           Spacer(1, 8),
           Paragraph("Generated by SentientSpend AI. Figures are in INR.", small)]
    doc.build(el)
    return buf.getvalue()


def send_alert_email(to, subject, body):
    if not SMTP_OK:
        return False, "smtplib unavailable in this environment."
    host, port = os.getenv("SMTP_HOST", ""), int(os.getenv("SMTP_PORT", "465"))
    user, pw = os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", "")
    if not (host and user and pw and to):
        return False, "SMTP not configured — set SMTP_HOST, SMTP_USER, SMTP_PASS (and a recipient)."
    try:
        msg = MIMEText(body, "html")
        msg["Subject"], msg["From"], msg["To"] = subject, user, to
        if port == 587:
            with smtplib.SMTP(host, port, timeout=15) as s:
                s.starttls(); s.login(user, pw); s.send_message(msg)
        else:
            with smtplib.SMTP_SSL(host, port, timeout=15) as s:
                s.login(user, pw); s.send_message(msg)
        return True, f"Alert email sent to {to}."
    except Exception as e:
        return False, f"Send failed: {e}"


# ─────────────────────────────────────────────────────────────────────
# CHAT WITH YOUR DATA  (real pandas querying; optional LLM layer)
# ─────────────────────────────────────────────────────────────────────
MONTH_NAMES = {m.lower(): i for i, m in enumerate(calmod.month_name) if m}
MONTH_ABBR = {m.lower(): i for i, m in enumerate(calmod.month_abbr) if m}


def _resolve_period(q, df):
    """Returns (filtered_df, label)."""
    today = df["Date"].max().date() if not df.empty else date.today()
    if "last month" in q:
        first_this = today.replace(day=1)
        last_prev = first_this - timedelta(days=1)
        mask = (df["Date"].dt.year == last_prev.year) & (df["Date"].dt.month == last_prev.month)
        return df[mask], f"in {last_prev:%B %Y}"
    if "this month" in q or "current month" in q:
        mask = (df["Date"].dt.year == today.year) & (df["Date"].dt.month == today.month)
        return df[mask], f"in {today:%B %Y}"
    if "this week" in q or "last 7" in q or "past week" in q:
        return df[df["Date"] >= pd.Timestamp(today) - pd.Timedelta(days=7)], "in the last 7 days"
    if "last 30" in q or "past month" in q:
        return df[df["Date"] >= pd.Timestamp(today) - pd.Timedelta(days=30)], "in the last 30 days"
    if "this year" in q:
        return df[df["Date"].dt.year == today.year], f"in {today.year}"
    for name, num in {**MONTH_NAMES, **MONTH_ABBR}.items():
        if re.search(rf"\b{name}\b", q):
            sub = df[df["Date"].dt.month == num]
            if not sub.empty:
                return sub, f"in {calmod.month_name[num]}"
    return df, "overall"


def _resolve_category(q):
    for cat in ALL_CATEGORIES:
        if re.search(rf"\b{cat.lower()}", q):
            return cat
    synonyms = {"Food": ["eat", "dining", "restaurant", "grocery", "swiggy", "zomato"],
                "Transport": ["travel", "commute", "uber", "ola", "petrol", "fuel"],
                "Shopping": ["amazon", "clothes", "shop"],
                "Bills": ["rent", "utility", "electricity", "recharge"],
                "Entertainment": ["movie", "netflix", "fun", "subscription"],
                "Healthcare": ["medical", "doctor", "gym", "pharmacy"],
                "Education": ["course", "study", "book"]}
    for cat, words in synonyms.items():
        if any(re.search(rf"\b{w}", q) for w in words):
            return cat
    return None


def roast_finances(df, summary, budget):
    inc = summary["Income"].sum()
    exp = summary["Expense"].sum()
    rate = (inc - exp) / inc * 100 if inc else 0
    lines = []
    if exp > inc:
        lines.append(f"You spent ₹{exp:,.0f} while earning ₹{inc:,.0f}. That's not a budget, that's a "
                     f"slow-motion crime scene with a ₹{exp - inc:,.0f} body count.")
    elif rate < 10:
        lines.append(f"A {rate:.1f}% savings rate. At this pace your emergency fund will be ready "
                     f"roughly around the heat death of the universe.")
    else:
        lines.append(f"Fine — {rate:.1f}% saved. Respectable. Don't let it go to your head, "
                     f"you're still one sale notification away from relapse.")
    e = df[df["Type"] == "Expense"]
    if not e.empty:
        top_cat = e.groupby("Category")["Amount"].sum().idxmax()
        top_amt = e.groupby("Category")["Amount"].sum().max()
        lines.append(f"₹{top_amt:,.0f} on {top_cat}. You didn't build a habit, you built a "
                     f"subscription to your own worst instincts.")
        big = e.loc[e["Amount"].idxmax()]
        lines.append(f"Special mention: ₹{big['Amount']:,.0f} on \"{big['Description'] or big['Category']}\" "
                     f"({big['Date']:%d %b}). Hope it was worth {int(big['Amount'] / max(1, budget) * 100)}% "
                     f"of a whole month's budget.")
        wk = e[e["Date"].dt.dayofweek >= 5]["Amount"].sum() / e["Amount"].sum() * 100
        if wk > 35:
            lines.append(f"{wk:.0f}% of your spending happens on weekends. Two days a week you behave "
                         f"like someone else is paying.")
    subs = detect_recurring(df)
    if not subs.empty:
        lines.append(f"You're quietly bleeding ₹{subs['Annual Cost'].sum():,.0f}/year on "
                     f"{len(subs)} recurring charges. Half of those apps don't even remember you.")
    lines.append("Roast over. Go cancel something.")
    return "🔥 " + " ".join(lines)


def _llm_context(df, summary, budget):
    """Aggregates only — individual transaction rows never leave the machine."""
    e = df[df["Type"] == "Expense"]
    return {
        "currency": "INR",
        "monthly_budget": budget,
        "months": {str(k): {"income": float(v["Income"]), "expense": float(v["Expense"]),
                            "savings": float(v["Savings"])} for k, v in summary.iterrows()},
        "category_totals": e.groupby("Category")["Amount"].sum().round(0).to_dict(),
        "top_merchants": e.groupby(e["Description"].str.slice(0, 30))["Amount"]
                          .sum().nlargest(10).round(0).to_dict(),
        "recurring": detect_recurring(df).to_dict("records")[:8],
    }


LLM_SYSTEM = ("You are a concise personal-finance analyst. Answer ONLY from the JSON data "
              "provided. Amounts are Indian rupees. Be specific with numbers, max 120 words, "
              "no markdown headings. If the data cannot answer it, say so plainly.")


def groq_answer(query, df, summary, budget):
    """Groq (Llama 3) via OpenAI-compatible REST API."""
    if not GROQ_OK:
        return f"({GROQ_LABEL} unavailable: no GROQ_API_KEY set)"
    if not REQ_OK:
        return f"({GROQ_LABEL} unavailable: pip install requests)"
    
    url = f"{GROQ_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": LLM_SYSTEM},
            {"role": "user", "content": f"DATA:\n{json.dumps(_llm_context(df, summary, budget), default=str)}\n\nQUESTION: {query}"}
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        if r.status_code == 401:
            return f"({GROQ_LABEL} unavailable: Invalid API key. Check your GROQ_API_KEY.)"
        if r.status_code == 429:
            return f"({GROQ_LABEL} rate limit hit — wait a moment, or use the built-in engine.)"
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices") or []
        if not choices:
            return f"({GROQ_LABEL} returned no answer.)"
        text = choices[0].get("message", {}).get("content", "").strip()
        return text or f"({GROQ_LABEL} returned an empty answer.)"
    except Exception as e:
        return f"({GROQ_LABEL} unavailable: {e})"
def claude_answer(query, df, summary, budget):
    if not ANTHROPIC_OK:
        return "(Claude unavailable: no ANTHROPIC_API_KEY set)"
    try:
        client = anthropic.Anthropic(api_key=get_secret("ANTHROPIC_API_KEY"))
        msg = client.messages.create(
            model=get_secret("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
            max_tokens=400, system=LLM_SYSTEM,
            messages=[{"role": "user", "content":
                       f"DATA:\n{json.dumps(_llm_context(df, summary, budget), default=str)}"
                       f"\n\nQUESTION: {query}"}])
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:
        return f"(Claude unavailable: {e})"


def llm_answer(query, df, summary, budget, provider=None):
    if provider == GROQ_LABEL:
        return groq_answer(query, df, summary, budget)
    elif provider == "Claude":
        return claude_answer(query, df, summary, budget)
    return groq_answer(query, df, summary, budget)


def ai_chat_with_data(query, df, summary, budget, use_llm=False, provider=None):
    """Deterministic analytics engine over the real dataframe."""
    q = str(query or "").lower().strip()
    if df.empty:
        return "There's no data yet — add a transaction or import a statement first."
    if "roast" in q:
        return roast_finances(df, summary, budget)

    if use_llm:
        out = llm_answer(query, df, summary, budget, provider or GROQ_LABEL)
        if out and not out.startswith("("):      # "(...)" means the provider failed
            return out
        if out:
            st.warning(out + " Falling back to the built-in analytics engine.")

    scope, label = _resolve_period(q, df)
    exp = scope[scope["Type"] == "Expense"]
    inc = scope[scope["Type"] == "Income"]
    total_exp, total_inc = exp["Amount"].sum(), inc["Amount"].sum()
    cat = _resolve_category(q)

    if any(k in q for k in ("health", "score", "how am i doing")):
        score, parts = calculate_health_score(summary, budget, df)
        band, _ = score_band(score)
        worst = min(parts.items(), key=lambda kv: kv[1][0] / kv[1][1]) if parts else None
        extra = f" Weakest area: {worst[0]} ({worst[1][0]:.0f}/{worst[1][1]})." if worst else ""
        return f"Your financial health score is {score}/100 — {band}.{extra}"

    if any(k in q for k in ("runway", "how long", "run out")):
        days, net, burn = calculate_runway(df)
        if days is None:
            return "Not enough expense history to compute a runway yet."
        return (f"At your recent burn rate of ₹{burn:,.0f}/day, your net savings of ₹{net:,.0f} "
                f"last about {days} days ({days / 30:.1f} months).")

    if any(k in q for k in ("today", "safe to spend", "can i spend", "allowance")):
        allow, spent, days_left = calculate_daily_allowance(budget, df)
        return (f"You've spent ₹{spent:,.0f} of your ₹{budget:,.0f} budget this month with "
                f"{days_left} days left — that's ₹{allow:,.0f} safe to spend per day.")

    if any(k in q for k in ("predict", "forecast", "next month", "will i")):
        pred, slope, _, r2 = ml_forecast(summary)
        trend = "rising" if slope > 0 else "falling"
        return (f"Linear regression on {len(summary)} months predicts ₹{pred:,.0f} of expenses next "
                f"month ({trend} ₹{abs(slope):,.0f}/month, R²={r2:.2f}). Budget is ₹{budget:,.0f}, so "
                f"you'd be {'over' if pred > budget else 'under'} by ₹{abs(pred - budget):,.0f}.")

    if any(k in q for k in ("subscription", "recurring", "netflix", "cancel")):
        subs = detect_recurring(df)
        if subs.empty:
            return "No recurring charges detected yet — I need at least 3 months of similar charges."
        top = subs.head(5)
        body = "; ".join(f"{r['Merchant']} ₹{r['Avg Amount']:,.0f}/{r['Cadence'].lower()}"
                         for _, r in top.iterrows())
        return (f"I found {len(subs)} recurring charges costing ₹{subs['Annual Cost'].sum():,.0f}/year. "
                f"Top ones: {body}.")

    if any(k in q for k in ("anomal", "fraud", "unusual", "weird", "suspicious")):
        an = detect_anomalies(df)
        if an.empty:
            return "No anomalous transactions flagged — Isolation Forest sees nothing unusual."
        r = an.iloc[0]
        return (f"{len(an)} transactions flagged as unusual. Biggest: ₹{r['Amount']:,.0f} on "
                f"{r['Date']:%d %b %Y} ({r['Category']} — {r['Description'] or 'no description'}). "
                f"Reason: {r['Reason'].lower()}.")

    if any(k in q for k in ("biggest", "largest", "most expensive", "top transaction")):
        if exp.empty:
            return f"No expenses recorded {label}."
        r = exp.loc[exp["Amount"].idxmax()]
        return (f"Your largest expense {label} was ₹{r['Amount']:,.0f} on {r['Date']:%d %b %Y} — "
                f"{r['Description'] or r['Category']} ({r['Category']}).")

    if "top" in q and any(k in q for k in ("categor", "spend", "merchant")):
        if "merchant" in q:
            top = exp.groupby(exp["Description"].str.slice(0, 30))["Amount"].sum().nlargest(5)
            return f"Top merchants {label}: " + ", ".join(
                f"{k or 'unlabelled'} ₹{v:,.0f}" for k, v in top.items()) + "."
        top = exp.groupby("Category")["Amount"].sum().nlargest(5)
        return f"Top categories {label}: " + ", ".join(
            f"{k} ₹{v:,.0f}" for k, v in top.items()) + "."

    if any(k in q for k in ("save", "saving", "saved")) and "how" in q:
        net = total_inc - total_exp
        rate = net / total_inc * 100 if total_inc else 0
        return f"You saved ₹{net:,.0f} {label} — a {rate:.1f}% savings rate on ₹{total_inc:,.0f} income."

    if cat:
        sub = exp[exp["Category"] == cat]
        if sub.empty:
            return f"No {cat} spending {label}."
        share = sub["Amount"].sum() / total_exp * 100 if total_exp else 0
        months = max(1, sub["Date"].dt.to_period("M").nunique())
        return (f"You spent ₹{sub['Amount'].sum():,.0f} on {cat} {label} across {len(sub)} "
                f"transactions — {share:.1f}% of expenses, averaging ₹{sub['Amount'].sum() / months:,.0f}/month "
                f"(₹{sub['Amount'].mean():,.0f} per transaction).")

    if any(k in q for k in ("income", "earn", "salary")):
        return f"Income {label}: ₹{total_inc:,.0f} across {len(inc)} credits."

    if any(k in q for k in ("spend", "spent", "expense", "how much", "total")):
        return (f"Total spending {label}: ₹{total_exp:,.0f} across {len(exp)} transactions "
                f"(avg ₹{exp['Amount'].mean() if len(exp) else 0:,.0f} each). Income was ₹{total_inc:,.0f}, "
                f"so net {'savings' if total_inc >= total_exp else 'shortfall'} of "
                f"₹{abs(total_inc - total_exp):,.0f}.")

    score, _ = calculate_health_score(summary, budget, df)
    return (f"Here's the picture {label}: income ₹{total_inc:,.0f}, expenses ₹{total_exp:,.0f}, "
            f"net ₹{total_inc - total_exp:,.0f}, health score {score}/100. "
            f"Try asking: “how much did I spend on food last month”, “what's my biggest expense”, "
            f"“any unusual transactions”, “predict next month”, or “roast my finances”.")


# ─────────────────────────────────────────────────────────────────────
# CHALLENGES
# ─────────────────────────────────────────────────────────────────────
def create_challenge(uid, name, kind, target, days, category="All"):
    start = date.today()
    end = start + timedelta(days=int(days))
    with get_conn() as c:
        c.execute("INSERT INTO challenges (user_id,name,target_amt,start_date,end_date,status,kind,category) "
                  "VALUES (?,?,?,?,?,'active',?,?)",
                  (uid, name, float(target), str(start), str(end), kind, category))


def get_challenges(uid, status=None):
    q = ("SELECT id,name,target_amt,start_date,end_date,status,kind,category "
         "FROM challenges WHERE user_id=?")
    args = [uid]
    if status:
        q += " AND status=?"
        args.append(status)
    with get_conn() as c:
        rows = c.execute(q + " ORDER BY id DESC", args).fetchall()
    return [dict(zip(["id", "name", "target", "start", "end", "status", "kind", "category"], r))
            for r in rows]


def challenge_progress(ch, df):
    """Real progress from transactions inside the challenge window."""
    start = pd.Timestamp(ch["start"])
    end = pd.Timestamp(ch["end"])
    today = pd.Timestamp(date.today())
    window = df[(df["Date"] >= start) & (df["Date"] <= min(end, today))] if not df.empty else df
    days_total = max(1, (end.date() - start.date()).days)
    days_done = max(0, min(days_total, (min(end, today).date() - start.date()).days))
    exp = window[window["Type"] == "Expense"] if not window.empty else window
    if ch["category"] and ch["category"] != "All" and not exp.empty:
        exp = exp[exp["Category"] == ch["category"]]
    spent = float(exp["Amount"].sum()) if not exp.empty else 0.0

    if ch["kind"] == "save":
        earned = float(window[window["Type"] == "Income"]["Amount"].sum()) if not window.empty else 0.0
        saved = earned - float(window[window["Type"] == "Expense"]["Amount"].sum() if not window.empty else 0)
        pct = np.clip(saved / ch["target"] * 100, 0, 100) if ch["target"] else 0
        detail = f"Saved ₹{saved:,.0f} of ₹{ch['target']:,.0f}"
        success = saved >= ch["target"]
    elif ch["kind"] == "no_spend":
        spend_days = set(exp["Date"].dt.date) if not exp.empty else set()
        clean = sum(1 for i in range(days_done + 1)
                    if (start.date() + timedelta(days=i)) not in spend_days)
        pct = np.clip(clean / days_total * 100, 0, 100)
        detail = f"{clean} no-spend days of {days_total}"
        success = clean >= days_total
    else:  # spend_less: survive the window under a spending cap
        success = spent <= ch["target"]
        # progress = share of the window survived while still under the cap
        pct = float(np.clip(days_done / days_total * 100, 0, 100)) if success else 0.0
        headroom = ch["target"] - spent
        detail = (f"Spent ₹{spent:,.0f} of the ₹{ch['target']:,.0f} cap "
                  f"({'₹%s left' % f'{headroom:,.0f}' if success else 'cap broken'})")
    return float(pct), detail, days_total - days_done, success


def refresh_challenge_statuses(uid, df):
    """Auto-complete or fail challenges whose window has ended; award badges."""
    completed = 0
    for ch in get_challenges(uid, "active"):
        if pd.Timestamp(ch["end"]).date() < date.today():
            _, _, _, success = challenge_progress(ch, df)
            with get_conn() as c:
                c.execute("UPDATE challenges SET status=? WHERE id=?",
                          ("completed" if success else "failed", ch["id"]))
    with get_conn() as c:
        completed = c.execute("SELECT COUNT(*) FROM challenges WHERE user_id=? AND status='completed'",
                              (uid,)).fetchone()[0]
    if completed >= 1:
        award_badge(uid, "challenge_1")
    if completed >= 5:
        award_badge(uid, "challenge_5")
    return completed


def evaluate_badges(uid, df, summary, budget):
    if not df.empty:
        award_badge(uid, "first_txn")
    if not summary.empty and summary["Income"].sum() > 0:
        rate = summary["Savings"].sum() / summary["Income"].sum()
        if rate >= 0.10:
            award_badge(uid, "saver_10")
        if rate >= 0.30:
            award_badge(uid, "saver_30")
        under = (summary["Expense"] <= budget).astype(int).tolist()
        run = best = 0
        for u in under:
            run = run + 1 if u else 0
            best = max(best, run)
        if best >= 3:
            award_badge(uid, "under_budget")
    if no_spend_streak(df) >= 7:
        award_badge(uid, "no_spend_7")


# ─────────────────────────────────────────────────────────────────────
# FAMILY / RBAC
# ─────────────────────────────────────────────────────────────────────
def add_family_member(admin_id, username, role="member"):
    target = find_user_by_username(username)
    if not target:
        return False, f"No user named '{username}'. They need to sign up first."
    if target["id"] == admin_id:
        return False, "You're already the admin of this household."
    try:
        with get_conn() as c:
            c.execute("INSERT INTO family_accounts (admin_id,member_id,role) VALUES (?,?,?)",
                      (admin_id, target["id"], role))
        return True, f"{username} added as {role}."
    except sqlite3.IntegrityError:
        return False, f"{username} is already in your household."


def remove_family_member(admin_id, member_id):
    with get_conn() as c:
        c.execute("DELETE FROM family_accounts WHERE admin_id=? AND member_id=?",
                  (admin_id, member_id))


def get_household_members(admin_id):
    with get_conn() as c:
        rows = c.execute(
            "SELECT u.id,u.username,u.email,u.budget,f.role FROM family_accounts f "
            "JOIN users u ON u.id=f.member_id WHERE f.admin_id=? ORDER BY u.username",
            (admin_id,)).fetchall()
    return [dict(zip(["id", "username", "email", "budget", "role"], r)) for r in rows]


def get_my_households(member_id):
    with get_conn() as c:
        rows = c.execute(
            "SELECT u.id,u.username,f.role FROM family_accounts f "
            "JOIN users u ON u.id=f.admin_id WHERE f.member_id=?", (member_id,)).fetchall()
    return [dict(zip(["admin_id", "admin_name", "role"], r)) for r in rows]


def can_view_member(viewer_id, target_id):
    """RBAC: admins see their members; members see only themselves."""
    if viewer_id == target_id:
        return True
    with get_conn() as c:
        return c.execute("SELECT 1 FROM family_accounts WHERE admin_id=? AND member_id=?",
                         (viewer_id, target_id)).fetchone() is not None

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
for key, default in [("user", None), ("pending_txn", None), ("chat_log", []),
                     ("ocr_result", None), ("voice_parsed", None), ("toast", None),
                     ("voice_cmd_input", ""), ("_last_voice_audio", None)]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────────────────────────────
def auth_screen():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:36px 0 24px">
            <div style="font-size:52px">💳</div>
            <div style="font-size:30px;font-weight:800;color:#0f172a;margin:10px 0 6px">SentientSpend AI</div>
            <div style="font-size:14px;color:#64748b">AI-powered personal finance dashboard</div>
        </div>""", unsafe_allow_html=True)

        tl, tr = st.tabs(["🔐  Login", "📝  Create Account"])
        with tl:
            u = st.text_input("Username", placeholder="Enter username", key="li_u")
            p = st.text_input("Password", type="password", placeholder="Enter password", key="li_p")
            if st.button("Login →", **WIDE, type="primary", key="btn_login"):
                ok, usr = login_user(u, p)
                if ok:
                    st.session_state.user = usr
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tr:
            ru = st.text_input("Username", placeholder="Min 3 characters", key="ru")
            remail = st.text_input("Email", placeholder="your@email.com", key="remail")
            rp = st.text_input("Password", type="password", placeholder="Min 6 characters", key="rp")
            rc = st.text_input("Confirm password", type="password", key="rc")
            demo = st.checkbox("Load 12 months of realistic demo data", value=True, key="rdemo")
            if st.button("Create Account →", **WIDE, type="primary", key="btn_reg"):
                if len(ru.strip()) < 3:
                    st.error("Username needs at least 3 characters.")
                elif not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", remail.strip()):
                    st.error("Enter a valid email address.")
                elif len(rp) < 6:
                    st.error("Password needs at least 6 characters.")
                elif rp != rc:
                    st.error("Passwords don't match.")
                else:
                    ok, msg = create_user(ru, remail, rp)
                    if ok:
                        _, usr = login_user(ru, rp)
                        st.session_state.user = usr
                        if demo:
                            seed_demo_data(usr["id"])
                        st.rerun()
                    else:
                        st.error(msg)


# ─────────────────────────────────────────────────────────────────────
# SMALL UI HELPERS
# ─────────────────────────────────────────────────────────────────────
def section(title, sub=""):
    st.markdown(f'<p class="sh">{title}</p>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<p class="sub">{sub}</p>', unsafe_allow_html=True)


def base_layout(fig, height=300, legend=True):
    fig.update_layout(
        height=height, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=0, r=0, t=10, b=0), font=dict(color="#0f172a"),
        showlegend=legend,
        legend=dict(orientation="h", y=1.14, font=dict(size=11, color="#0f172a")))
    return fig


def save_transaction_with_nudge(uid, tdate, ttype, tcat, tamt, tdesc, budget, df):
    """Behavioural nudge: intercept large / budget-busting expenses before committing."""
    if ttype == "Expense":
        allow, spent, days_left = calculate_daily_allowance(budget, df)
        big = tamt >= budget * 0.25
        busts = (spent + tamt) > budget
        if big or busts:
            reasons = []
            if big:
                reasons.append(f"it's {tamt / budget * 100:.0f}% of your entire monthly budget")
            if busts:
                over = spent + tamt - budget
                reasons.append(f"it pushes you ₹{over:,.0f} past this month's budget")
            st.session_state.pending_txn = {
                "uid": uid, "date": tdate, "type": ttype, "category": tcat,
                "amount": tamt, "description": tdesc,
                "reasons": reasons, "allow": allow, "days_left": days_left}
            return False
    add_transaction(uid, tdate, ttype, tcat, tamt, tdesc)
    return True

# ─────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────
def dashboard():
    user = st.session_state.user
    uid = user["id"]

    # ── SIDEBAR ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:4px 0 14px">
            <div style="font-size:20px;font-weight:800;color:#0f172a">💳 SentientSpend</div>
            <div style="font-size:12px;color:#64748b;margin-top:2px">AI Finance Dashboard</div>
        </div>
        <div style="background:#f1f5f9;border-radius:10px;padding:12px 14px;margin-bottom:14px;
                    border:1px solid #e2e8f0">
            <div style="font-size:13px;font-weight:700;color:#0f172a">👤 {user['username']}</div>
            <div style="font-size:11px;color:#64748b;margin-top:2px">{user['email']}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**💰 Monthly budget (₹)**")
        budget = st.number_input("Budget", min_value=1000, max_value=10_000_000,
                                 value=int(user["budget"]), step=1000,
                                 label_visibility="collapsed", key="budget_input")
        if budget != user["budget"]:
            update_user_field(uid, "budget", int(budget))
            st.session_state.user["budget"] = int(budget)

        st.markdown("**💱 Display currency**")
        cur_list = list(CURRENCIES)
        currency = st.selectbox("Currency", cur_list,
                                index=cur_list.index(user.get("base_currency", "INR")),
                                label_visibility="collapsed", key="cur_sel")
        if currency != user.get("base_currency"):
            update_user_field(uid, "base_currency", currency)
            st.session_state.user["base_currency"] = currency
        rates, rate_src = get_fx_rates("INR")
        if currency != "INR":
            st.caption(f"1 ₹ = {rates.get(currency, FALLBACK_RATES.get(currency,1)):.4f} {currency} "
                       f"({'live' if rate_src == 'live' else 'offline fallback'} rate)")

        st.markdown("---")
        st.markdown("**🏦 Multi-bank import (ETL)**")
        st.caption("CSV/Excel from any bank — headers and columns are detected automatically")
        uploaded = st.file_uploader("Upload", type=["csv", "xlsx", "xls"],
                                    label_visibility="collapsed", key="bank_upload")
        if uploaded is not None and st.button("⚙️ Run import", **WIDE, key="run_import"):
            with st.spinner("Parsing statement…"):
                res = import_file(uploaded, uid)
            if res["imported"]:
                st.success(f"Imported {res['imported']} transactions")
            if res["duplicates"]:
                st.info(f"{res['duplicates']} duplicates skipped")
            if res["skipped"]:
                st.caption(f"{res['skipped']} non-transaction rows ignored")
            if res["mapping"]:
                st.caption("Mapped: " + ", ".join(f"{k}→{v}" for k, v in res["mapping"].items()))
            for e in res["errors"][:3]:
                st.warning(e)
            if res["imported"]:
                st.rerun()

        st.markdown("---")
        st.markdown("**🔔 Budget alert email**")
        ae = st.text_input("Alert email", value=user.get("alert_email", ""),
                           placeholder="your@email.com", label_visibility="collapsed", key="ae")
        if st.button("💾 Save email", **WIDE, key="save_ae"):
            update_user_field(uid, "alert_email", ae.strip())
            st.session_state.user["alert_email"] = ae.strip()
            st.success("Saved")
        st.caption("Sending needs SMTP_HOST / SMTP_USER / SMTP_PASS env vars."
                   if not os.getenv("SMTP_HOST") else "SMTP configured ✅")

        st.markdown("---")
        st.markdown("**₿ Crypto tracker**")
        crypto, csrc = get_crypto_prices()
        if crypto:
            for sym, label in (("BTC", "Bitcoin"), ("ETH", "Ethereum")):
                if sym in crypto:
                    chg = crypto.get(sym + "_chg", 0)
                    col = "#059669" if chg >= 0 else "#dc2626"
                    st.markdown(
                        f"<div style='font-size:13px'>{label}: <b>${crypto[sym]:,.0f}</b> "
                        f"<span style='color:{col}'>{chg:+.2f}%</span></div>",
                        unsafe_allow_html=True)
            st.caption(f"source: {csrc}")
        else:
            st.caption("Offline — install yfinance or requests for live prices")

        st.markdown("---")
        st.markdown(f"""
        <div style="font-size:11px;font-weight:700;line-height:1.9">
            <div style="color:#059669">✅ Linear Regression (forecast)</div>
            <div style="color:#7c3aed">✅ KMeans (persona clustering)</div>
            <div style="color:#dc2626">✅ Isolation Forest (anomalies)</div>
            <div style="color:{'#059669' if FUZZ_OK else '#94a3b8'}">
                {'✅' if FUZZ_OK else '○'} Fuzzy auto-tagging (thefuzz)</div>
            <div style="color:{'#059669' if OCR_OK else '#94a3b8'}">
                {'✅' if OCR_OK else '○'} Receipt OCR (tesseract)</div>
            <div style="color:{'#059669' if GROQ_OK else '#94a3b8'}">
                {'✅' if GROQ_OK else '○'} {GROQ_LABEL} chat ({GROQ_MODEL if GROQ_OK else 'no key'})</div>
            <div style="color:{'#059669' if ANTHROPIC_OK else '#94a3b8'}">
                {'✅' if ANTHROPIC_OK else '○'} Claude chat (Anthropic key)</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Logout", **WIDE, key="logout"):
            st.session_state.user = None
            st.session_state.pending_txn = None
            st.session_state.chat_log = []
            st.rerun()

    # ── DATA ──────────────────────────────────────────────────────────
    df = get_transactions(uid)
    summary = build_summary(df)
    has_data = not df.empty and not summary.empty
    cur = st.session_state.user.get("base_currency", "INR")
    money = lambda v: fmt_money(v, cur, rates)

    st.markdown(f"""
    <div style="margin-bottom:18px">
        <div class="page-title">📊 SentientSpend AI Dashboard</div>
        <div class="page-sub">Welcome back, <strong>{user['username']}</strong>
        &nbsp;·&nbsp; Budget: ₹{int(budget):,}/month
        &nbsp;·&nbsp; {len(df)} transactions on record</div>
    </div>""", unsafe_allow_html=True)

    # ── PENDING NUDGE (behavioural finance intercept) ─────────────────
    p = st.session_state.pending_txn
    if p:
        allow = p["allow"]
        st.markdown(f"""
        <div class="alert-box" style="font-size:14px">
            🧠 <strong>Hold on — {money(p['amount'])} on {p['category']}.</strong><br>
            {'; '.join(p['reasons']).capitalize()}.<br>
            Your safe-to-spend is <strong>{money(allow)}/day</strong> for the next
            {p['days_left']} days. Sleep on it for 24 hours and see if you still want it.
        </div>""", unsafe_allow_html=True)
        n1, n2, n3 = st.columns([1.2, 1.2, 4])
        if n1.button("✅ I still want it", type="primary", key="nudge_yes"):
            add_transaction(p["uid"], p["date"], p["type"], p["category"],
                            p["amount"], p["description"])
            st.session_state.pending_txn = None
            st.rerun()
        if n2.button("🚫 Skip it — save the money", key="nudge_no"):
            st.session_state.pending_txn = None
            st.success("Nice. That's money still in your account.")
            st.rerun()

    # ── QUICK ADD ─────────────────────────────────────────────────────
    with st.expander("➕  Add a transaction", expanded=not has_data):
        cl, cr = st.columns(2)
        with cl:
            t_date = st.date_input("Date", date.today(), key="t_date")
            t_type_raw = st.radio("Money in or out?",
                                  ["➖ Expense (money I spent)", "➕ Income (money I received)"],
                                  key="t_type")
            t_type = "Income" if "Income" in t_type_raw else "Expense"
            cat_pool = CATEGORIES if t_type == "Expense" else INCOME_CATEGORIES
            t_cat_raw = st.selectbox("Category",
                                     [f"{CAT_ICONS.get(c, '📌')}  {c}" for c in cat_pool], key="t_cat")
            t_cat = t_cat_raw.split("  ", 1)[1]
        with cr:
            t_amt = st.number_input("Amount (₹)", min_value=1.0, value=500.0, step=100.0, key="t_amt")
            t_desc = st.text_input("Description (optional)",
                                   placeholder="e.g. Swiggy dinner, electricity bill", key="t_desc")
            if t_desc and t_type == "Expense":
                sug, conf = auto_tag_category(t_desc, t_amt)
                if sug != "Other" and sug != t_cat:
                    st.caption(f"🏷️ Auto-tagger suggests **{sug}** ({conf}% match) for that description.")
            sign = "+" if t_type == "Income" else "-"
            color = "#059669" if t_type == "Income" else "#dc2626"
            st.markdown(f"""
            <div style="background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;padding:14px">
                <div style="font-size:10px;color:#94a3b8;font-weight:700">PREVIEW</div>
                <div style="font-size:24px;font-weight:800;color:{color}">{sign}{money(t_amt)}</div>
                <div style="font-size:13px;color:#475569">{t_cat} · {t_type} · {t_date:%d %b %Y}</div>
                <div style="font-size:12px;color:#94a3b8">{t_desc or '(no description)'}</div>
            </div>""", unsafe_allow_html=True)

        sa, sb, _ = st.columns([1.3, 1.2, 4])
        if sa.button("💾  Save transaction", type="primary", **WIDE, key="save_txn"):
            saved = save_transaction_with_nudge(uid, t_date, t_type, t_cat, t_amt, t_desc, budget, df)
            if saved:
                st.success(f"Saved — {t_type} of {money(t_amt)} ({t_cat})")
            st.rerun()
        if sb.button("🔄  Reset form", **WIDE, key="reset_form"):
            st.rerun()

    if not has_data:
        st.markdown("""
        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;
                    padding:24px;text-align:center;margin-top:18px">
            <div style="font-size:36px">📭</div>
            <div style="font-size:17px;font-weight:700;color:#9a3412">No transactions yet</div>
            <div style="font-size:13px;color:#c2410c;margin-top:6px">
                Add one above, or import a bank statement from the sidebar.</div>
        </div>""", unsafe_allow_html=True)
        return

    # ── DERIVED METRICS ───────────────────────────────────────────────
    ti, te = float(summary["Income"].sum()), float(summary["Expense"].sum())
    ts = ti - te
    sr = ts / ti * 100 if ti else 0
    health, parts = calculate_health_score(summary, budget, df)
    allow, spent_this_month, days_left = calculate_daily_allowance(budget, df)
    run_days, net_cash, burn = calculate_runway(df)
    anomalies = detect_anomalies(df)
    recurring = detect_recurring(df)
    evaluate_badges(uid, df, summary, budget)
    refresh_challenge_statuses(uid, df)
    band, band_color = score_band(health)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("💰 Total income", money(ti))
    c2.metric("💸 Total expense", money(te))
    c3.metric("🏦 Net savings", money(ts), f"{sr:.1f}% rate")
    c4.metric("🏆 Health score", f"{health}/100", band)
    c5.metric("🛡️ Safe to spend", money(allow) + "/day",
              f"{days_left} days left", help=f"Budget minus ₹{spent_this_month:,.0f} spent this month")
    c6.metric("⏳ Runway", f"{run_days} days" if run_days is not None else "—",
              f"₹{burn:,.0f}/day burn" if burn else None,
              help="How long your net savings last at the recent daily burn rate")

    last_exp = float(summary["Expense"].iloc[-1])
    if last_exp > budget:
        st.markdown(f'<div class="alert-box">🚨 {summary.index[-1]} expenses of {money(last_exp)} '
                    f'exceeded your {money(budget)} budget by {money(last_exp - budget)}.</div>',
                    unsafe_allow_html=True)
    elif last_exp >= budget * 0.85:
        st.markdown(f'<div class="alert-box">⚠️ You used {last_exp / budget * 100:.0f}% of your '
                    f'budget in {summary.index[-1]} — close to the limit.</div>',
                    unsafe_allow_html=True)

    tabs = st.tabs(["📈  Trends", "🎯  Budget", "🔮  Simulator", "🧠  ML Persona",
                    "🚨  Anomalies", "📋  Transactions", "🤖  AI & Tools",
                    "🎮  Challenges", "👨‍👩‍👧  Family", "⬇️  Export"])
    t1, t2, t3, t4, t5, t6, t7, t8, t9, t10 = tabs

    # ══ TRENDS ════════════════════════════════════════════════════════
    with t1:
        pred, slope, tline, r2 = ml_forecast(summary)
        mlab = [str(m) for m in summary.index]
        cl1, cr1 = st.columns([2, 1])
        with cl1:
            section("Income vs expense vs savings", "Dashed line = linear-regression forecast")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mlab, y=summary["Income"], name="Income",
                                     line=dict(color="#1d4ed8", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(29,78,216,0.07)"))
            fig.add_trace(go.Scatter(x=mlab, y=summary["Expense"], name="Expense",
                                     line=dict(color="#dc2626", width=2.5),
                                     fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
            fig.add_trace(go.Scatter(x=mlab, y=summary["Savings"], name="Savings",
                                     line=dict(color="#059669", width=2)))
            if tline:
                fig.add_trace(go.Scatter(x=mlab + ["Next"], y=tline, name=f"ML trend (R²={r2:.2f})",
                                         line=dict(color="#d97706", width=1.5, dash="dash")))
            fig.add_hline(y=budget, line_dash="dot", line_color="#94a3b8",
                          annotation_text=f"Budget ₹{int(budget):,}",
                          annotation_font=dict(color="#475569", size=11))
            base_layout(fig, 300)
            fig.update_yaxes(gridcolor="#f1f5f9", tickprefix="₹", tickformat=",")
            fig.update_xaxes(showgrid=False)
            st.plotly_chart(fig, **WIDE)
        with cr1:
            section("By category", "Whole period")
            cat_sum = df[df["Type"] == "Expense"].groupby("Category")["Amount"].sum().sort_values()
            fig2 = go.Figure(go.Bar(x=cat_sum.values, y=cat_sum.index, orientation="h",
                                    marker_color=[CAT_COLORS.get(c, "#888") for c in cat_sum.index],
                                    text=[f"₹{v:,.0f}" for v in cat_sum.values],
                                    textposition="outside", textfont=dict(size=11, color="#0f172a")))
            base_layout(fig2, 300, legend=False)
            fig2.update_layout(margin=dict(l=0, r=80, t=10, b=0))
            fig2.update_xaxes(showgrid=False, showticklabels=False)
            fig2.update_yaxes(showgrid=False, tickfont=dict(size=12, color="#0f172a"))
            st.plotly_chart(fig2, **WIDE)

        cl2, cr2 = st.columns(2)
        with cl2:
            section("Monthly savings")
            fig3 = go.Figure(go.Bar(
                x=mlab, y=summary["Savings"],
                marker_color=["#059669" if v >= 0 else "#dc2626" for v in summary["Savings"]],
                text=[f"₹{v:,.0f}" for v in summary["Savings"]], textposition="outside",
                textfont=dict(size=10, color="#0f172a")))
            base_layout(fig3, 230, legend=False)
            fig3.update_yaxes(showgrid=False, showticklabels=False)
            st.plotly_chart(fig3, **WIDE)
        with cr2:
            section("AI prediction — next month")
            direction = "📈 rising" if slope > 0 else "📉 falling"
            bg = "#fef2f2" if pred > budget else "#f0fdf4"
            bc = "#fca5a5" if pred > budget else "#86efac"
            tc = "#991b1b" if pred > budget else "#166534"
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {bc};border-radius:12px;padding:20px">
                <div style="font-size:11px;color:#64748b;font-weight:700">PREDICTED EXPENSE</div>
                <div style="font-size:32px;font-weight:800;color:{tc};margin:6px 0">{money(pred)}</div>
                <div style="font-size:13px;color:#475569">
                    Trend {slope:+,.0f}/month · {direction} · model fit R²={r2:.2f}<br>
                    That is {money(abs(pred - budget))} {'over' if pred > budget else 'under'} budget.
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        section("Spending calendar heatmap", "Daily expense totals — darker means a heavier day")
        cal = daily_spend_calendar(df, 182)
        if not cal.empty:
            idx = pd.to_datetime(pd.Series(cal.index))
            weeks = idx.dt.isocalendar().week.astype(int).tolist()
            years = idx.dt.isocalendar().year.astype(int).tolist()
            wkey = [f"{y}-W{w:02d}" for y, w in zip(years, weeks)]
            order, seen = [], set()
            for k in wkey:
                if k not in seen:
                    seen.add(k); order.append(k)
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            z = np.full((7, len(order)), np.nan)
            txt = [["" for _ in order] for _ in range(7)]
            for d, val, k in zip(cal.index, cal.values, wkey):
                r, c = pd.Timestamp(d).weekday(), order.index(k)
                z[r][c] = val
                txt[r][c] = f"{d:%d %b %Y}<br>₹{val:,.0f}"
            xlabels = []
            last_month = None
            for k in order:
                dates = [d for d, kk in zip(cal.index, wkey) if kk == k]
                mo = pd.Timestamp(dates[0]).strftime("%b")
                xlabels.append(mo if mo != last_month else "")
                last_month = mo
            figc = go.Figure(go.Heatmap(z=z, x=list(range(len(order))), y=dow_names,
                                        colorscale="Blues", hoverinfo="text", text=txt,
                                        xgap=3, ygap=3, colorbar=dict(title="₹")))
            base_layout(figc, 250, legend=False)
            figc.update_xaxes(tickmode="array", tickvals=list(range(len(order))),
                              ticktext=xlabels, showgrid=False)
            figc.update_yaxes(autorange="reversed", showgrid=False)
            st.plotly_chart(figc, **WIDE)
            busiest = cal.idxmax()
            st.caption(f"Heaviest day: {busiest:%d %b %Y} at ₹{cal.max():,.0f} · "
                       f"{int((cal == 0).sum())} zero-spend days in the last {len(cal)} days · "
                       f"current no-spend streak: {no_spend_streak(df)} days")

    # ══ BUDGET ════════════════════════════════════════════════════════
    with t2:
        section("Monthly budget status", f"Limit: {money(budget)} — change it in the sidebar")
        exceeded, rows_html = [], ""
        for m in summary.index:
            e_, i_, s_ = (float(summary.loc[m, "Expense"]), float(summary.loc[m, "Income"]),
                          float(summary.loc[m, "Savings"]))
            pct = e_ / budget if budget else 0
            bar = min(int(pct * 100), 100)
            bc = "#dc2626" if pct > 1 else ("#d97706" if pct >= 0.85 else "#059669")
            badge = ('<span class="badge-over">🚨 Exceeded</span>' if pct > 1 else
                     '<span class="badge-warn">⚠️ Near limit</span>' if pct >= 0.85 else
                     '<span class="badge-ok">✅ OK</span>')
            if pct > 1:
                exceeded.append(str(m))
            rem = budget - e_
            rc = "#166534" if rem >= 0 else "#b91c1c"
            rows_html += f"""
            <tr style="border-bottom:1px solid #f1f5f9">
              <td style="padding:10px;font-weight:700;color:#0f172a">{m}</td>
              <td style="padding:10px;color:#059669;font-weight:600">{money(i_)}</td>
              <td style="padding:10px;color:#dc2626;font-weight:600">{money(e_)}</td>
              <td style="padding:10px">
                <div style="background:#f1f5f9;border-radius:999px;height:8px;width:110px">
                  <div style="background:{bc};border-radius:999px;height:8px;width:{bar}%"></div>
                </div>
                <span style="font-size:10px;color:#64748b">{pct * 100:.0f}%</span>
              </td>
              <td style="padding:10px;color:{rc};font-weight:700">{money(rem)}</td>
              <td style="padding:10px;color:#059669;font-weight:600">{money(s_)}</td>
              <td style="padding:10px">{badge}</td>
            </tr>"""
        head = "".join(f'<th style="padding:10px;text-align:left;color:#475569;font-weight:700;'
                       f'font-size:11px;text-transform:uppercase">{h}</th>'
                       for h in ["Month", "Income", "Expense", "Usage", "Remaining", "Savings", "Status"])
        st.markdown(f"""
        <div style="background:#fff;border-radius:14px;border:1px solid #e2e8f0;overflow:hidden">
          <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">{head}</tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        if exceeded:
            st.markdown(f'<div class="alert-box" style="margin-top:12px">🚨 Budget exceeded in: '
                        f'<strong>{", ".join(exceeded)}</strong></div>', unsafe_allow_html=True)
            if st.button("📧 Email me this alert", key="send_alert"):
                ok, info = send_alert_email(
                    user.get("alert_email", ""), "SentientSpend budget alert",
                    f"<h2>Budget exceeded</h2><p>Months over ₹{int(budget):,}: "
                    f"{', '.join(exceeded)}</p><p>Health score: {health}/100</p>")
                (st.success if ok else st.warning)(info)

        st.markdown("<br>", unsafe_allow_html=True)
        g1, g2 = st.columns([1, 1])
        with g1:
            avg_exp = float(summary["Expense"].mean())
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta", value=avg_exp,
                delta={"reference": budget, "valueformat": ",", "increasing": {"color": "#dc2626"},
                       "decreasing": {"color": "#059669"}},
                title={"text": "Average monthly expense vs budget",
                       "font": {"size": 13, "color": "#0f172a"}},
                number={"prefix": "₹", "valueformat": ",", "font": {"color": "#0f172a"}},
                gauge={"axis": {"range": [0, budget * 1.4], "tickformat": ","},
                       "bar": {"color": "#1d4ed8"},
                       "steps": [{"range": [0, budget * .85], "color": "#f0fdf4"},
                                 {"range": [budget * .85, budget], "color": "#fef9c3"},
                                 {"range": [budget, budget * 1.4], "color": "#fee2e2"}],
                       "threshold": {"line": {"color": "#dc2626", "width": 3}, "value": budget}}))
            fig_g.update_layout(height=280, paper_bgcolor="#fff", margin=dict(l=20, r=20, t=50, b=10),
                                font=dict(color="#0f172a"))
            st.plotly_chart(fig_g, **WIDE)
        with g2:
            section("Health score breakdown", f"{health}/100 — {band}")
            for name, (got, maxi) in parts.items():
                pctp = got / maxi * 100
                col = "#059669" if pctp >= 75 else ("#d97706" if pctp >= 45 else "#dc2626")
                st.markdown(f"""
                <div style="margin-bottom:8px">
                  <div style="display:flex;justify-content:space-between;font-size:12px;
                              font-weight:600;color:#0f172a">
                    <span>{name}</span><span>{got:.0f} / {maxi}</span></div>
                  <div style="background:#f1f5f9;border-radius:999px;height:7px">
                    <div style="background:{col};border-radius:999px;height:7px;width:{pctp:.0f}%"></div>
                  </div>
                </div>""", unsafe_allow_html=True)

    # ══ SIMULATOR ═════════════════════════════════════════════════════
    with t3:
        section("🔮 What-if simulator", "Model spending cuts, extra income and compounding returns")
        avg_exp_s = float(summary["Expense"].mean())
        avg_inc_s = float(summary["Income"].mean())
        avg_sav_s = avg_inc_s - avg_exp_s
        sl, sr_col = st.columns(2)
        with sl:
            red_pct = st.slider("Cut non-essential spending by", 0, 60, 15, format="%d%%", key="sim_cut")
            extra_inc = st.slider("Extra monthly income (₹)", 0, 100000, 0, 500, key="sim_inc")
            months_n = st.slider("Projection horizon (months)", 1, 60, 24, key="sim_months")
            inv_pct = st.slider("Invest this share of savings", 0, 100, 40, format="%d%%", key="sim_inv")
            ret_pct = st.slider("Expected annual return on investments", 0, 20, 12, format="%d%%", key="sim_ret")
        non_essential = float(df[(df["Type"] == "Expense") &
                                 (~df["Category"].isin(ESSENTIAL_CATS))]["Amount"].sum())
        share_ne = non_essential / te if te else 0
        with sr_col:
            new_exp = avg_exp_s * (1 - share_ne * red_pct / 100)
            new_inc = avg_inc_s + extra_inc
            new_sav = new_inc - new_exp
            invested = max(0.0, new_sav) * inv_pct / 100
            r1, r2c = st.columns(2)
            r1.metric("Monthly savings", money(new_sav), f"{new_sav - avg_sav_s:+,.0f} ₹")
            r2c.metric("Annual savings", money(new_sav * 12))
            r3, r4 = st.columns(2)
            r3.metric("Invested monthly", money(invested), f"{inv_pct}% of savings")
            r4.metric("New savings rate", f"{new_sav / new_inc * 100:.1f}%" if new_inc else "—")
            st.caption(f"Only non-essential categories are cut — they are {share_ne * 100:.0f}% "
                       f"of your spending, so a {red_pct}% cut removes "
                       f"₹{avg_exp_s - new_exp:,.0f}/month.")

        r_m = (1 + ret_pct / 100) ** (1 / 12) - 1
        base_cum, opt_cum, inv_cum = [], [], []
        b = o = v = 0.0
        for _ in range(months_n):
            b += avg_sav_s
            o += new_sav
            v = v * (1 + r_m) + invested
            base_cum.append(b); opt_cum.append(o); inv_cum.append(v + (o - invested * len(inv_cum)))
        fmons = [f"M{i + 1}" for i in range(months_n)]
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=fmons, y=base_cum, name="Current path",
                                   line=dict(color="#dc2626", width=2, dash="dot")))
        fig_p.add_trace(go.Scatter(x=fmons, y=opt_cum, name="Optimised (cash only)",
                                   line=dict(color="#0891b2", width=2)))
        fig_p.add_trace(go.Scatter(x=fmons, y=inv_cum, name=f"Optimised + {ret_pct}% returns",
                                   line=dict(color="#059669", width=2.5),
                                   fill="tonexty", fillcolor="rgba(5,150,105,0.07)"))
        base_layout(fig_p, 280)
        fig_p.update_yaxes(gridcolor="#f1f5f9", tickprefix="₹", tickformat=",")
        st.plotly_chart(fig_p, **WIDE)
        gain = opt_cum[-1] - base_cum[-1]
        comp = inv_cum[-1] - opt_cum[-1]
        st.markdown(f'<div class="success-box">💡 Over {months_n} months this plan puts an extra '
                    f'<strong>{money(gain)}</strong> in your pocket, and compounding at {ret_pct}% adds '
                    f'another <strong>{money(max(comp, 0))}</strong> — ending balance '
                    f'<strong>{money(inv_cum[-1])}</strong>.</div>', unsafe_allow_html=True)

    # ══ ML PERSONA ════════════════════════════════════════════════════
    with t4:
        exp_df, persona, pcolors = ml_cluster(df)
        persona = persona or "Balanced Spender"
        pl, pr = st.columns(2)
        with pl:
            share = (exp_df[exp_df["ClusterName"] == persona]["Amount"].sum() /
                     exp_df["Amount"].sum() * 100) if not exp_df.empty else 0
            st.markdown(f"""
            <div class="persona-box">
                <div style="font-size:10px;letter-spacing:.1em;font-weight:700">ML SPENDING PERSONA</div>
                <h3>{PERSONA_ICON.get(persona, '🧠')} {persona}</h3>
                <p>{PERSONA_DESC.get(persona, '')}</p>
                <div style="background:rgba(255,255,255,0.18);border-radius:10px;padding:12px 14px;
                            margin-top:10px;display:flex;justify-content:space-between">
                    <div>
                        <div style="font-size:10px;font-weight:700">SHARE OF YOUR SPENDING</div>
                        <div style="font-size:26px;font-weight:800">{share:.1f}%</div>
                    </div>
                    <div style="font-size:11px;text-align:right">KMeans<br>
                        {exp_df['Cluster'].nunique()} clusters<br>
                        {len(exp_df)} expenses analysed</div>
                </div>
            </div>""", unsafe_allow_html=True)
            cc = exp_df.groupby("ClusterName")["Amount"].sum()
            fig_d = go.Figure(go.Pie(labels=cc.index.tolist(), values=cc.values, hole=0.55,
                                     marker_colors=[pcolors.get(n, "#64748b") for n in cc.index],
                                     textfont=dict(size=12, color="#0f172a")))
            base_layout(fig_d, 240)
            st.plotly_chart(fig_d, **WIDE)
        with pr:
            section("Cluster scatter — amount vs day of week")
            fig_sc = px.scatter(exp_df, x="DayOfWeek", y="Amount", color="ClusterName",
                                color_discrete_map=pcolors, opacity=0.75,
                                hover_data=["Category", "Description"], height=240)
            fig_sc.update_traces(marker=dict(size=7))
            base_layout(fig_sc, 240)
            fig_sc.update_xaxes(tickvals=list(range(7)),
                                ticktext=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
                                showgrid=False, title=None)
            fig_sc.update_yaxes(gridcolor="#f1f5f9", tickprefix="₹", tickformat=",", title=None)
            st.plotly_chart(fig_sc, **WIDE)

            section("Category × month heatmap")
            piv = df[df["Type"] == "Expense"].copy()
            piv["MonthP"] = piv["Date"].dt.to_period("M").astype(str)
            piv2 = piv.pivot_table(index="Category", columns="MonthP",
                                   values="Amount", aggfunc="sum").fillna(0)
            piv2 = piv2.reindex(sorted(piv2.columns), axis=1)
            fig_h = go.Figure(go.Heatmap(
                z=piv2.values, x=piv2.columns.tolist(), y=piv2.index.tolist(), colorscale="Blues",
                text=[[f"₹{v:,.0f}" for v in row] for row in piv2.values],
                texttemplate="%{text}", textfont=dict(size=9, color="#0f172a")))
            base_layout(fig_h, 240, legend=False)
            st.plotly_chart(fig_h, **WIDE)

        st.markdown("---")
        wk = exp_df[exp_df["DayOfWeek"] >= 5]["Amount"].sum()
        wd = exp_df[exp_df["DayOfWeek"] < 5]["Amount"].sum()
        ess = exp_df[exp_df["IsEssential"] == 1]["Amount"].sum()
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Weekend share", f"{wk / (wk + wd) * 100:.0f}%" if wk + wd else "—")
        k2.metric("Essentials share", f"{ess / exp_df['Amount'].sum() * 100:.0f}%" if len(exp_df) else "—")
        k3.metric("Median transaction", money(exp_df["Amount"].median() if len(exp_df) else 0))
        k4.metric("Busiest day", ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][
            int(exp_df.groupby("DayOfWeek")["Amount"].sum().idxmax())] if len(exp_df) else "—")

    # ══ ANOMALIES ═════════════════════════════════════════════════════
    with t5:
        section("🚨 Anomaly & fraud detection",
                "Isolation Forest over amount, timing and category-relative z-score")
        if anomalies.empty:
            st.markdown('<div class="success-box">✅ Nothing unusual found. Every transaction sits '
                        'inside your normal spending pattern.</div>', unsafe_allow_html=True)
        else:
            a1, a2, a3 = st.columns(3)
            a1.metric("Flagged", len(anomalies))
            a2.metric("Value flagged", money(anomalies["Amount"].sum()))
            a3.metric("Share of spend", f"{anomalies['Amount'].sum() / te * 100:.1f}%")
            st.markdown("<br>", unsafe_allow_html=True)
            for _, r in anomalies.head(15).iterrows():
                st.markdown(f"""
                <div class="ss-card" style="border-left:4px solid #dc2626">
                  <div style="display:flex;justify-content:space-between">
                    <div>
                      <div style="font-size:15px;font-weight:700;color:#0f172a">
                        {CAT_ICONS.get(r['Category'],'📌')} {money(r['Amount'])} · {r['Category']}</div>
                      <div style="font-size:12px;color:#64748b">
                        {r['Date']:%d %b %Y} · {r['Description'] or 'no description'}</div>
                    </div>
                    <div style="font-size:11px;color:#b91c1c;font-weight:700;text-align:right;
                                max-width:220px">{r['Reason']}</div>
                  </div>
                </div>""", unsafe_allow_html=True)
            chart = anomalies.head(20)
            fig_a = go.Figure()
            normal = df[(df["Type"] == "Expense") & (~df["id"].isin(anomalies["id"]))]
            fig_a.add_trace(go.Scatter(x=normal["Date"], y=normal["Amount"], mode="markers",
                                       name="Normal", marker=dict(color="#cbd5e1", size=6)))
            fig_a.add_trace(go.Scatter(x=chart["Date"], y=chart["Amount"], mode="markers",
                                       name="Flagged", marker=dict(color="#dc2626", size=11,
                                                                   symbol="x")))
            base_layout(fig_a, 280)
            fig_a.update_yaxes(gridcolor="#f1f5f9", tickprefix="₹", tickformat=",")
            st.plotly_chart(fig_a, **WIDE)

    # ══ TRANSACTIONS ══════════════════════════════════════════════════
    with t6:
        section("📋 All transactions", "Filter, search, edit or delete")
        f1, f2, f3, f4 = st.columns([2.5, 1, 1, 1.5])
        search = f1.text_input("Search", placeholder="Search description or category…",
                               label_visibility="collapsed", key="txn_search")
        f_type = f2.selectbox("Type", ["All", "Income", "Expense"],
                              label_visibility="collapsed", key="txn_type")
        f_cat = f3.selectbox("Category", ["All"] + ALL_CATEGORIES,
                             label_visibility="collapsed", key="txn_cat")
        f_sort = f4.selectbox("Sort", ["Newest first", "Oldest first", "Highest amount", "Lowest amount"],
                              label_visibility="collapsed", key="txn_sort")
        view = df.copy()
        if search:
            view = view[view["Description"].str.contains(search, case=False, na=False) |
                        view["Category"].str.contains(search, case=False, na=False)]
        if f_type != "All":
            view = view[view["Type"] == f_type]
        if f_cat != "All":
            view = view[view["Category"] == f_cat]
        sc, asc = {"Newest first": ("Date", False), "Oldest first": ("Date", True),
                   "Highest amount": ("Amount", False), "Lowest amount": ("Amount", True)}[f_sort]
        view = view.sort_values(sc, ascending=asc)

        m1, m2, m3, m4 = st.columns([1, 1, 1, 3])
        m1.metric("Shown", len(view))
        m2.metric("Income shown", money(view[view["Type"] == "Income"]["Amount"].sum()))
        m3.metric("Expense shown", money(view[view["Type"] == "Expense"]["Amount"].sum()))

        if view.empty:
            st.info("No transactions match those filters.")
        else:
            page_size = 25
            pages = max(1, int(np.ceil(len(view) / page_size)))
            page = st.number_input("Page", 1, pages, 1, key="txn_page") if pages > 1 else 1
            chunk = view.iloc[(page - 1) * page_size: page * page_size]
            hcols = st.columns([1.1, 0.8, 1.1, 1.1, 2.2, 0.5, 0.5])
            for h, colh in zip(["Date", "Type", "Category", "Amount", "Description", "", ""], hcols):
                colh.markdown(f"<div style='font-size:11px;font-weight:700;color:#64748b;"
                              f"text-transform:uppercase'>{h}</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:4px 0;border-color:#e2e8f0'>", unsafe_allow_html=True)
            for _, row in chunk.iterrows():
                ac = "#059669" if row["Type"] == "Income" else "#dc2626"
                sign = "+" if row["Type"] == "Income" else "-"
                q1, q2, q3, q4, q5, q6, q7 = st.columns([1.1, 0.8, 1.1, 1.1, 2.2, 0.5, 0.5])
                q1.markdown(f"<span style='font-size:13px;color:#475569'>{row['Date']:%d %b %Y}</span>",
                            unsafe_allow_html=True)
                q2.markdown(f"<span style='font-size:12px;color:#94a3b8'>{row['Type']}</span>",
                            unsafe_allow_html=True)
                q3.markdown(f"<span style='font-size:13px'>{CAT_ICONS.get(row['Category'],'📌')} "
                            f"{row['Category']}</span>", unsafe_allow_html=True)
                q4.markdown(f"<span style='font-size:13px;font-weight:700;color:{ac}'>"
                            f"{sign}{money(row['Amount'])}</span>", unsafe_allow_html=True)
                q5.markdown(f"<span style='font-size:12px;color:#64748b'>"
                            f"{str(row['Description'])[:46] or '—'}</span>", unsafe_allow_html=True)
                if q6.button("✏️", key=f"e_{row['id']}", help="Edit"):
                    st.session_state[f"edit_{row['id']}"] = not st.session_state.get(f"edit_{row['id']}", False)
                if q7.button("🗑️", key=f"d_{row['id']}", help="Delete"):
                    delete_transaction(int(row["id"]), uid)
                    st.rerun()
                if st.session_state.get(f"edit_{row['id']}", False):
                    with st.container():
                        w1, w2, w3, w4, w5 = st.columns([1.2, 1, 1.2, 2, 1])
                        nd = w1.date_input("Date", row["Date"].date(), key=f"ed_{row['id']}",
                                           label_visibility="collapsed")
                        nt = w2.selectbox("Type", ["Expense", "Income"],
                                          index=0 if row["Type"] == "Expense" else 1,
                                          key=f"et_{row['id']}", label_visibility="collapsed")
                        pool = CATEGORIES if nt == "Expense" else INCOME_CATEGORIES
                        nc = w3.selectbox("Category", pool,
                                          index=pool.index(row["Category"]) if row["Category"] in pool else 0,
                                          key=f"ec_{row['id']}", label_visibility="collapsed")
                        nde = w4.text_input("Description", value=str(row["Description"]),
                                            key=f"ex_{row['id']}", label_visibility="collapsed")
                        na = w5.number_input("Amount", min_value=0.0, value=float(row["Amount"]),
                                             key=f"ea_{row['id']}", label_visibility="collapsed")
                        if st.button("Save changes", key=f"es_{row['id']}", type="primary"):
                            update_transaction(int(row["id"]), uid, nd, nt, nc, na, nde)
                            st.session_state[f"edit_{row['id']}"] = False
                            st.rerun()
                st.markdown("<hr style='margin:2px 0;border-color:#f8fafc'>", unsafe_allow_html=True)
            if pages > 1:
                st.caption(f"Page {page} of {pages} · {len(view)} matching transactions")

    # ══ AI & TOOLS ════════════════════════════════════════════════════
    with t7:
        section("🤖 AI assistant & smart tools")
        ai_l, ai_r = st.columns(2)

        with ai_l:
            st.markdown("#### 💬 Chat with your data")
            providers = llm_providers()
            use_llm, provider = False, GROQ_LABEL
            if providers:
                pc1, pc2 = st.columns([1.3, 1])
                use_llm = pc1.toggle("Use an AI model for free-form answers",
                                     value=False, key="use_llm")
                provider = pc2.selectbox("Model", providers, label_visibility="collapsed",
                                         key="llm_provider")
                if use_llm:
                    st.caption(f"{provider} sees only aggregates — monthly totals, category sums "
                               f"and top merchant names. Individual transactions never leave your machine. "
                               f"If it fails, the built-in engine answers instead.")
            else:
                st.caption("Set GROQ_API_KEY (or GEMINI_API_KEY) to add free-form AI answers. "
                           "The built-in analytics engine works without any key.")
            quick = st.columns(3)
            preset = None
            if quick[0].button("Top categories", key="q1"):
                preset = "top categories this year"
            if quick[1].button("Anything unusual?", key="q2"):
                preset = "any unusual transactions"
            if quick[2].button("🔥 Roast me", key="q3"):
                preset = "roast my finances"
            chat_q = st.text_input("Ask anything about your money",
                                   placeholder="e.g. how much did I spend on food last month?",
                                   key="chat_q")
            question = preset or (chat_q if st.session_state.get("chat_q") else None)
            if question:
                with st.spinner("Crunching your numbers…"):
                    answer = ai_chat_with_data(question, df, summary, budget,
                                               use_llm=use_llm, provider=provider)
                    st.session_state.chat_log.insert(0, (question, answer))
                    st.session_state.chat_log = st.session_state.chat_log[:6]
                for q_, a_ in st.session_state.chat_log:
                    box = "roast-box" if a_.startswith("🔥") else "info-box"
                    st.markdown(f'<div class="{box}"><strong>You:</strong> {q_}<br><br>{a_}</div>',
                                unsafe_allow_html=True)
            st.markdown("---")
            st.markdown("#### 📸 Receipt scanner (OCR)")
            if not OCR_OK:
                st.caption("Install pytesseract + the Tesseract binary for automatic extraction.")
            receipt = st.file_uploader("Receipt image", type=["png", "jpg", "jpeg", "webp"],
                                       key="receipt_up", label_visibility="collapsed")
            if receipt is not None:
                img_bytes = receipt.getvalue()
                st.image(img_bytes, width=220)
                if st.button("🔍 Scan receipt", key="scan_receipt"):
                    data, rawtext, err = ocr_receipt(img_bytes)
                    st.session_state.ocr_result = {"data": data, "text": rawtext, "err": err}
                res = st.session_state.ocr_result
                if res:
                    if res["err"]:
                        st.warning(res["err"])
                    d = res["data"] or {}
                    o1, o2 = st.columns(2)
                    o_amt = o1.number_input("Amount (₹)", min_value=0.0,
                                            value=float(d.get("amount", 0.0)), key="ocr_amt")
                    o_date = o2.date_input("Date", d.get("date", date.today()), key="ocr_date")
                    pool_idx = CATEGORIES.index(d["category"]) if d.get("category") in CATEGORIES else 7
                    o_cat = o1.selectbox("Category", CATEGORIES, index=pool_idx, key="ocr_cat")
                    o_desc = o2.text_input("Merchant / note", value=d.get("merchant", ""), key="ocr_desc")
                    if d.get("confidence"):
                        st.caption(f"Auto-tagged as {d['category']} ({d['confidence']}% match)")
                    if res["text"]:
                        with st.expander("Raw OCR text"):
                            st.text(res["text"][:2000])
                    if st.button("➕ Add this transaction", type="primary", key="ocr_add"):
                        if o_amt <= 0:
                            st.error("Enter an amount greater than zero.")
                        else:
                            add_transaction(uid, o_date, "Expense", o_cat, o_amt,
                                            o_desc or "Receipt", source="ocr")
                            st.session_state.ocr_result = None
                            st.success("Added from receipt")
                            st.rerun()

        with ai_r:
            st.markdown("#### 🎤 Voice transaction entry")
            st.caption("Record your voice naturally (e.g. *“spent 450 on uber yesterday”*) or type below:")

            audio_value = st.audio_input("🎙️ Record voice expense", key="voice_audio_record")

            if "transcribed_text" not in st.session_state:
                st.session_state["transcribed_text"] = ""

            if audio_value:
                curr_bytes = audio_value.read()
                if st.session_state.get("_last_voice_audio") != curr_bytes:
                    st.session_state["_last_voice_audio"] = curr_bytes
                    with st.spinner("🎧 Transcribing your voice..."):
                        transcript = transcribe_audio(curr_bytes)
                        if transcript and not transcript.startswith("SpeechRecognition") and not transcript.startswith("Could not"):
                            st.session_state["transcribed_text"] = transcript
                            st.toast(f"🎙️ Heard: {transcript}")
                        else:
                            st.warning(transcript or "Could not detect clear speech. Please try again.")

            final_text = st.text_input("Review / Edit Transcript",
                                       value=st.session_state["transcribed_text"],
                                       placeholder="spent 450 on uber yesterday",
                                       key="voice_cmd_text")

            # Quick suggestion buttons
            col_p1, col_p2, col_p3 = st.columns(3)
            if col_p1.button("🚕 450 Uber", use_container_width=True, key="quick_v1"):
                st.session_state["transcribed_text"] = "spent 450 on uber yesterday"
                st.rerun()
            if col_p2.button("🛒 1200 Blinkit", use_container_width=True, key="quick_v2"):
                st.session_state["transcribed_text"] = "bought 1200 groceries at blinkit"
                st.rerun()
            if col_p3.button("💰 50000 Salary", use_container_width=True, key="quick_v3"):
                st.session_state["transcribed_text"] = "received 50000 salary"
                st.rerun()

            btn_col1, btn_col2 = st.columns([3, 1])
            if btn_col1.button("🎙️ Process Voice Expense", key="voice_parse", type="primary"):
                if final_text:
                    with st.spinner("Parsing voice input..."):
                        st.session_state.voice_parsed = parse_voice_text(final_text)
                        if st.session_state.voice_parsed is None:
                            st.error("I couldn't find an amount in that. Try “spent 450 on uber yesterday”.")
                else:
                    st.info("Record audio above or type a command to parse.")
            if btn_col2.button("🗑️ Clear", key="voice_clear"):
                st.session_state["transcribed_text"] = ""
                st.session_state.voice_parsed = None
                st.rerun()

            # Optional fallback Web Speech in-browser widget
            with st.expander("🌐 Web Speech API (Live Chrome/Edge Speech Recognition)"):
                components.html("""
<div style="font-family:'Source Sans Pro',sans-serif;padding:6px 0">
<button id="mic" style="background:#1d4ed8;color:#fff;border:none;border-radius:8px;
padding:8px 14px;font-size:13px;font-weight:600;cursor:pointer">🎙️ Start live listening</button>
<span id="status" style="font-size:12px;color:#64748b;margin-left:10px"></span>
<div id="out" style="margin-top:8px;padding:8px;border:1px solid #cbd5e1;
border-radius:6px;background:#fff;color:#0f172a;font-size:13px;min-height:36px"></div>
<button id="copy" style="margin-top:6px;background:#fff;border:1px solid #cbd5e1;
border-radius:6px;padding:5px 10px;font-size:12px;cursor:pointer">📋 Copy text</button>
</div>
<script>
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
const out = document.getElementById('out'), status = document.getElementById('status');
if (!SR) { status.textContent = 'Web Speech not supported in this browser — use the recorder above.'; }
document.getElementById('mic').onclick = () => {
    if (!SR) return;
    try {
        const r = new SR(); r.lang = 'en-IN'; r.interimResults = true; r.continuous = false;
        status.textContent = 'listening…';
        r.onresult = e => {
            const transcript = Array.from(e.results).map(x => x[0].transcript).join(' ');
            out.textContent = transcript;
            try {
                const parentInput = window.parent.document.querySelector('input[aria-label="Transcript / Spoken command"], input[placeholder*="uber"]');
                if (parentInput) {
                    parentInput.value = transcript;
                    parentInput.dispatchEvent(new Event('input', { bubbles: true }));
                    parentInput.dispatchEvent(new Event('change', { bubbles: true }));
                }
            } catch(err) {}
        };
        r.onend = () => { status.textContent = 'done — copied to input above'; };
        r.onerror = e => {
            status.textContent = 'error: ' + e.error + ' (Use the native recorder above if microphone is blocked in iframes)';
        };
        r.start();
    } catch(err) {
        status.textContent = 'error: ' + err.message;
    }
};
document.getElementById('copy').onclick = () => {
    const text = out.textContent;
    if (!text) return;
    if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(() => {
            status.textContent = 'copied!';
        }).catch(() => {
            fallbackCopy(text);
        });
    } else {
        fallbackCopy(text);
    }
    function fallbackCopy(val) {
        const ta = document.createElement('textarea');
        ta.value = val;
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        status.textContent = 'copied!';
    }
};
</script>
""", height=150)

            vp = st.session_state.voice_parsed
            if vp:
                st.markdown(f"""
<div class="info-box">
Parsed → <strong>{vp['type']} of ₹{vp['amount']:,.0f}</strong> ·
{vp['category']} ({vp['confidence']}% tag confidence) ·
{vp['date']:%d %b %Y}<br><span style="color:#475569">“{vp['description']}”</span>
</div>""", unsafe_allow_html=True)
                v1, v2 = st.columns([1, 1])
                if v1.button("✅ Save it", type="primary", key="voice_save"):
                    add_transaction(uid, vp["date"], vp["type"], vp["category"], vp["amount"],
                                vp["description"], source="voice")
                    st.session_state.voice_parsed = None
                    st.session_state["transcribed_text"] = ""
                    st.success("Saved from voice")
                    st.rerun()
                if v2.button("✖️ Discard", key="voice_discard"):
                    st.session_state.voice_parsed = None
                    st.rerun()
            st.markdown("---")
            st.markdown("#### 🔄 Recurring subscriptions")
            if recurring.empty:
                st.info("No recurring charges detected yet — I need at least 3 similar charges "
                        "across 3 different months.")
            else:
                st.metric("Annual cost of subscriptions", money(recurring["Annual Cost"].sum()),
                      f"{len(recurring)} charges")
                st.dataframe(recurring[["Merchant", "Category", "Cadence", "Avg Amount",
                                    "Next Expected", "Annual Cost"]],
                         **WIDE, hide_index=True)
                worst = recurring.iloc[0]
                st.caption(f"Cancelling “{worst['Merchant']}” alone would save "
                       f"₹{worst['Annual Cost']:,.0f} a year.")
    # ══ CHALLENGES ════════════════════════════════════════════════════
    with t8:
        section("🎮 Savings challenges", "Progress is computed from your real transactions")
        streak = no_spend_streak(df)
        earned = get_badges(uid)
        s1, s2, s3 = st.columns(3)
        s1.metric("🔥 No-spend streak", f"{streak} days")
        s2.metric("🎯 Active challenges", len(get_challenges(uid, "active")))
        s3.metric("🏅 Badges earned", f"{len(earned)}/{len(BADGE_DEFS)}")

        with st.expander("➕ Start a new challenge"):
            ch_kind = st.selectbox("Challenge type", [
                "save — put aside a target amount",
                "spend_less — stay under a spending cap",
                "no_spend — zero-spend days"], key="ch_kind")
            kind = ch_kind.split(" — ")[0]
            cc1, cc2, cc3 = st.columns(3)
            ch_name = cc1.text_input("Name", placeholder="No-spend weekend", key="ch_name")
            ch_days = cc2.slider("Duration (days)", 2, 90, 7, key="ch_days")
            ch_cat = cc3.selectbox("Category (for spend caps)", ["All"] + CATEGORIES, key="ch_cat")
            default_target = 1000.0 if kind != "no_spend" else float(ch_days)
            ch_amt = st.number_input("Target amount (₹) — ignored for no-spend challenges",
                                     min_value=0.0, value=default_target, step=500.0, key="ch_amt")
            if st.button("Start challenge", type="primary", key="ch_start"):
                if not ch_name.strip():
                    st.error("Give your challenge a name.")
                else:
                    create_challenge(uid, ch_name.strip(), kind, ch_amt or ch_days, ch_days, ch_cat)
                    st.success("Challenge started — good luck!")
                    st.rerun()

        active = get_challenges(uid, "active")
        if not active:
            st.info("No active challenges yet. Start one above.")
        for ch in active:
            pct, detail, days_left_ch, success = challenge_progress(ch, df)
            col = "#059669" if pct >= 70 else ("#d97706" if pct >= 35 else "#dc2626")
            st.markdown(f"""
            <div class="ss-card">
              <div style="display:flex;justify-content:space-between;align-items:center">
                <div>
                  <div style="font-size:16px;font-weight:700;color:#0f172a">🏆 {ch['name']}</div>
                  <div style="font-size:12px;color:#64748b">{detail} · {ch['kind'].replace('_',' ')}
                    {'· ' + ch['category'] if ch['category'] != 'All' else ''} ·
                    ends {ch['end']} ({max(0, days_left_ch)} days left)</div>
                </div>
                <div style="font-size:18px;font-weight:800;color:{col}">{pct:.0f}%</div>
              </div>
              <div style="background:#e2e8f0;border-radius:999px;height:9px;margin-top:10px">
                <div style="background:{col};border-radius:999px;height:9px;width:{pct:.0f}%"></div>
              </div>
            </div>""", unsafe_allow_html=True)
            b1, b2, _ = st.columns([1, 1, 4])
            if b1.button("✅ Finish now", key=f"chf_{ch['id']}"):
                with get_conn() as c:
                    c.execute("UPDATE challenges SET status=? WHERE id=?",
                              ("completed" if success else "failed", ch["id"]))
                st.rerun()
            if b2.button("🗑️ Abandon", key=f"chd_{ch['id']}"):
                with get_conn() as c:
                    c.execute("DELETE FROM challenges WHERE id=? AND user_id=?", (ch["id"], uid))
                st.rerun()

        done = [c_ for c_ in get_challenges(uid) if c_["status"] != "active"]
        if done:
            with st.expander(f"📜 History ({len(done)})"):
                st.dataframe(pd.DataFrame(done)[["name", "kind", "target", "start", "end", "status"]],
                             **WIDE, hide_index=True)

        st.markdown("---")
        section("🏅 Badges")
        chips = ""
        for code, (icon, title, desc) in BADGE_DEFS.items():
            got = code in earned
            style = ("background:#f0fdf4;border-color:#86efac" if got
                     else "background:#f8fafc;opacity:.55")
            chips += (f'<span class="badge-chip" style="{style}">{icon} <strong>{title}</strong>'
                      f'<br><span style="font-size:11px;color:#64748b">{desc}</span></span>')
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)

    # ══ FAMILY / RBAC ═════════════════════════════════════════════════
    with t9:
        section("👨‍👩‍👧 Household accounts (RBAC)",
                "Admins see everyone's spending; members only see their own")
        members = get_household_members(uid)
        my_households = get_my_households(uid)
        role_label = "Admin" if members else ("Member" if my_households else "Solo")
        st.markdown(f'<div class="info-box">Your role: <strong>{role_label}</strong> · '
                    f'{len(members)} member(s) in your household' +
                    (f" · you also belong to {', '.join(h['admin_name'] for h in my_households)}'s household"
                     if my_households else "") + '</div>', unsafe_allow_html=True)

        ac1, ac2 = st.columns([2, 1])
        new_member = ac1.text_input("Add an existing user by username", key="fam_user",
                                    placeholder="their SentientSpend username")
        new_role = ac2.selectbox("Role", ["member", "viewer"], key="fam_role")
        if st.button("➕ Add to household", key="fam_add"):
            ok, msg = add_family_member(uid, new_member, new_role)
            (st.success if ok else st.error)(msg)
            if ok:
                st.rerun()

        if members:
            st.markdown("<br>", unsafe_allow_html=True)
            rows = []
            for m in members:
                if not can_view_member(uid, m["id"]):
                    continue
                mdf = get_transactions(m["id"])
                msum = build_summary(mdf)
                mexp = float(msum["Expense"].sum()) if not msum.empty else 0.0
                minc = float(msum["Income"].sum()) if not msum.empty else 0.0
                this_month = 0.0
                if not mdf.empty:
                    today = date.today()
                    this_month = float(mdf[(mdf["Type"] == "Expense") &
                                           (mdf["Date"].dt.year == today.year) &
                                           (mdf["Date"].dt.month == today.month)]["Amount"].sum())
                rows.append({"Member": m["username"], "Role": m["role"], "Budget": m["budget"],
                             "Spent this month": this_month, "Total income": minc,
                             "Total expense": mexp,
                             "Status": "Over budget" if this_month > m["budget"] else "On track"})
            house = pd.DataFrame(rows)
            if not house.empty:
                h1, h2, h3 = st.columns(3)
                h1.metric("Household spend this month",
                          money(house["Spent this month"].sum() + spent_this_month))
                h2.metric("Combined budgets", money(house["Budget"].sum() + budget))
                h3.metric("Members over budget", int((house["Status"] == "Over budget").sum()))
                st.dataframe(house, **WIDE, hide_index=True)
                fig_f = go.Figure(go.Bar(
                    x=[user["username"]] + house["Member"].tolist(),
                    y=[spent_this_month] + house["Spent this month"].tolist(),
                    marker_color="#1d4ed8",
                    text=[f"₹{v:,.0f}" for v in [spent_this_month] + house["Spent this month"].tolist()],
                    textposition="outside"))
                base_layout(fig_f, 260, legend=False)
                fig_f.update_yaxes(gridcolor="#f1f5f9", tickprefix="₹", tickformat=",")
                st.plotly_chart(fig_f, **WIDE)
                rm = st.selectbox("Remove a member", ["—"] + house["Member"].tolist(), key="fam_rm")
                if rm != "—" and st.button("Remove", key="fam_rm_btn"):
                    target = find_user_by_username(rm)
                    if target:
                        remove_family_member(uid, target["id"])
                        st.rerun()
        else:
            st.caption("No members yet. Anyone you add must already have a SentientSpend account.")

    # ══ EXPORT ════════════════════════════════════════════════════════
    with t10:
        section("⬇️ Export your data")
        e1, e2, e3 = st.columns(3)
        with e1:
            st.markdown("""<div style="background:#eff6ff;border-radius:12px;padding:18px;
                border:1px solid #bfdbfe;margin-bottom:10px">
                <div style="font-size:26px">📊</div>
                <div style="font-size:15px;font-weight:700;color:#1e40af">Excel workbook</div>
                <div style="font-size:12px;color:#3b82f6">Transactions, monthly summary, categories,
                subscriptions and anomalies</div></div>""", unsafe_allow_html=True)
            st.download_button("⬇️ Download .xlsx",
                               data=export_excel(df, summary, recurring, anomalies),
                               file_name=f"sentientspend_{user['username']}_{datetime.now():%Y%m%d}.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                               **WIDE, type="primary", key="dl_xlsx")
        with e2:
            st.markdown("""<div style="background:#fdf4ff;border-radius:12px;padding:18px;
                border:1px solid #e9d5ff;margin-bottom:10px">
                <div style="font-size:26px">📄</div>
                <div style="font-size:15px;font-weight:700;color:#7e22ce">PDF report</div>
                <div style="font-size:12px;color:#9333ea">Branded summary with KPIs, categories,
                subscriptions and your ML persona</div></div>""", unsafe_allow_html=True)
            if REPORTLAB_OK:
                persona_name = ml_cluster(df)[1] or "Balanced Spender"
                pdf_bytes = export_pdf(summary, user, df, health, persona_name, recurring)
                st.download_button("⬇️ Download .pdf", data=pdf_bytes,
                                   file_name=f"sentientspend_report_{datetime.now():%Y%m%d}.pdf",
                                   mime="application/pdf", **WIDE,
                                   type="primary", key="dl_pdf")
            else:
                st.warning("Run `pip install reportlab` to enable PDF export.")
        with e3:
            st.markdown("""<div style="background:#f0fdf4;border-radius:12px;padding:18px;
                border:1px solid #bbf7d0;margin-bottom:10px">
                <div style="font-size:26px">📋</div>
                <div style="font-size:15px;font-weight:700;color:#166534">CSV</div>
                <div style="font-size:12px;color:#16a34a">Raw transactions for Excel, Sheets
                or another tool</div></div>""", unsafe_allow_html=True)
            csv_df = df.drop(columns=["id"], errors="ignore").copy()
            csv_df["Date"] = csv_df["Date"].dt.date
            st.download_button("⬇️ Download .csv", data=csv_df.to_csv(index=False).encode(),
                               file_name=f"transactions_{datetime.now():%Y%m%d}.csv",
                               mime="text/csv", **WIDE, type="primary", key="dl_csv")

        st.markdown("---")
        section("Monthly summary preview")
        prev = summary.copy()
        prev.index = prev.index.astype(str)
        st.dataframe(prev.style.format("₹{:,.0f}")
                     .background_gradient(subset=["Savings"], cmap="RdYlGn"),
                     **WIDE)


# ─────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────
if st.session_state.user is None:
    auth_screen()
else:
    dashboard()