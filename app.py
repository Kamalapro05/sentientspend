"""
SentientSpend AI — v3  (blur-free, crisp UI)
Run:  streamlit run app.py
"""

import streamlit as st
st.set_page_config(
    page_title="SentientSpend AI",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import sqlite3, hashlib, os, io, random
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    REPORTLAB_OK = True
except ImportError:
    REPORTLAB_OK = False

try:
    import smtplib
    from email.mime.text import MIMEText
    SMTP_OK = True
except ImportError:
    SMTP_OK = False

# ─────────────────────────────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH  = os.path.join(BASE_DIR, "sentientspend.db")

def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as c:
        c.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            email       TEXT UNIQUE NOT NULL,
            pw_hash     TEXT NOT NULL,
            budget      INTEGER DEFAULT 55000,
            alert_email TEXT DEFAULT '',
            created     TEXT DEFAULT (datetime('now'))
        );
        CREATE TABLE IF NOT EXISTS transactions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL,
            date        TEXT NOT NULL,
            type        TEXT NOT NULL,
            category    TEXT NOT NULL,
            amount      REAL NOT NULL,
            description TEXT DEFAULT '',
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
        """)

init_db()

# ─────────────────────────────────────────────────────────────────────
# NUCLEAR CSS  — kills every source of blur / fade / transparency
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ════════════════════════════════════════════════
   1. OVERRIDE STREAMLIT CSS VARIABLES AT ROOT
   This is the #1 reason text looks faded — Streamlit
   sets --text-color to a low-opacity value in its theme.
   We pin every variable to full-opacity solid values.
   ════════════════════════════════════════════════ */
:root {
    --text-color:                    #0f172a !important;
    --background-color:              #f1f5f9 !important;
    --secondary-background-color:    #ffffff !important;
    --primary-color:                 #1d4ed8 !important;
    --font:                          "Source Sans Pro", sans-serif !important;
}

/* ════════════════════════════════════════════════
   2. KILL ALL BLUR / GLASS-MORPHISM EVERYWHERE
   ════════════════════════════════════════════════ */
*, *::before, *::after {
    backdrop-filter:         none !important;
    -webkit-backdrop-filter: none !important;
}

/* ════════════════════════════════════════════════
   3. SOLID BACKGROUNDS — no rgba transparency
   ════════════════════════════════════════════════ */
#MainMenu, footer, header { visibility: hidden; }

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stBottom"],
[data-testid="stDecoration"],
[data-testid="stToolbar"] {
    background: #f1f5f9 !important;
    background-image: none !important;
    opacity: 1 !important;
}

.block-container {
    background:    transparent !important;
    padding-top:   2rem !important;
    padding-bottom:2rem !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   4. SIDEBAR — fully solid
   ════════════════════════════════════════════════ */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] > div > div,
[data-testid="stSidebarContent"] {
    background:    #ffffff !important;
    border-right:  1px solid #e2e8f0 !important;
    opacity: 1 !important;
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    color:   #0f172a !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   5. ALL TEXT — force fully opaque
   ════════════════════════════════════════════════ */
.stApp p,
.stApp span,
.stApp div,
.stApp label,
.stApp h1, .stApp h2, .stApp h3,
.stApp h4, .stApp h5, .stApp h6,
.stMarkdown p,
.stMarkdown span,
.element-container {
    opacity: 1 !important;
    color:   #0f172a !important;
}
/* Re-allow colored text to show through */
.stApp [style*="color:"],
.stApp [style*="color :"] {
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   6. METRIC CARDS — solid white, sharp text
   ════════════════════════════════════════════════ */
[data-testid="metric-container"] {
    background:    #ffffff !important;
    border:        1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    padding:       18px 20px !important;
    box-shadow:    0 1px 4px rgba(0,0,0,0.08) !important;
    opacity: 1 !important;
}
[data-testid="metric-container"] > div {
    background: transparent !important;
    opacity: 1 !important;
}
[data-testid="metric-container"] label,
[data-testid="metric-container"] [data-testid="stMetricLabel"],
[data-testid="metric-container"] [data-testid="stMetricLabel"] p {
    font-size:      11px !important;
    color:          #64748b !important;
    font-weight:    700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
    opacity: 1 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] > div {
    font-size:   26px !important;
    font-weight: 800 !important;
    color:       #0f172a !important;
    opacity: 1 !important;
}
[data-testid="stMetricDelta"],
[data-testid="stMetricDelta"] > div,
[data-testid="stMetricDelta"] svg {
    opacity: 1 !important;
    font-size: 12px !important;
}

/* ════════════════════════════════════════════════
   7. TABS — solid background, crisp labels
   ════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    background:    #e2e8f0 !important;
    border-radius: 10px !important;
    padding:       4px !important;
    gap:           4px !important;
    opacity: 1 !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important;
    font-size:     13px !important;
    font-weight:   600 !important;
    color:         #374151 !important;
    background:    transparent !important;
    padding:       8px 18px !important;
    opacity: 1 !important;
}
.stTabs [data-baseweb="tab"] span,
.stTabs [data-baseweb="tab"] p,
.stTabs [data-baseweb="tab"] div {
    color:   #374151 !important;
    opacity: 1 !important;
}
.stTabs [aria-selected="true"] {
    background: #ffffff !important;
    color:      #1d4ed8 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.10) !important;
}
.stTabs [aria-selected="true"] span,
.stTabs [aria-selected="true"] p,
.stTabs [aria-selected="true"] div {
    color:   #1d4ed8 !important;
    opacity: 1 !important;
}
/* Tab content panel */
[data-testid="stTabsContent"],
[data-baseweb="tab-panel"] {
    background: transparent !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   8. BUTTONS — solid, fully opaque
   ════════════════════════════════════════════════ */
.stButton > button {
    border-radius: 8px !important;
    font-weight:   600 !important;
    font-size:     13px !important;
    border:        1px solid #cbd5e1 !important;
    color:         #0f172a !important;
    background:    #ffffff !important;
    opacity: 1 !important;
}
.stButton > button:hover {
    background:   #f1f5f9 !important;
    border-color: #94a3b8 !important;
}
.stButton > button[kind="primary"],
[data-testid="baseButton-primary"] {
    background:   #1d4ed8 !important;
    color:        #ffffff !important;
    border-color: #1d4ed8 !important;
}
.stButton > button[kind="primary"]:hover,
[data-testid="baseButton-primary"]:hover {
    background: #1e40af !important;
}
.stDownloadButton > button {
    background:   #1d4ed8 !important;
    color:        #ffffff !important;
    border:       none !important;
    border-radius:8px !important;
    font-weight:  600 !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   9. INPUT FIELDS — solid white
   ════════════════════════════════════════════════ */
.stTextInput input,
.stTextInput > div > div > input,
.stNumberInput input,
.stNumberInput > div > div > input,
.stDateInput input,
.stDateInput > div > div > input,
.stSelectbox > div > div > div,
[data-baseweb="input"] input,
[data-baseweb="select"] > div {
    background:    #ffffff !important;
    border:        1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color:         #0f172a !important;
    font-size:     14px !important;
    opacity: 1 !important;
}
/* Selectbox dropdown */
[data-baseweb="popover"],
[data-baseweb="menu"],
[data-baseweb="popover"] ul,
[role="listbox"] {
    background: #ffffff !important;
    opacity:    1 !important;
}
[role="option"] {
    color:   #0f172a !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   10. RADIO BUTTONS
   ════════════════════════════════════════════════ */
.stRadio > div {
    background: transparent !important;
    opacity: 1 !important;
}
.stRadio label, .stRadio span {
    color:   #0f172a !important;
    opacity: 1 !important;
    font-size: 13px !important;
}

/* ════════════════════════════════════════════════
   11. EXPANDER — solid header + body
   ════════════════════════════════════════════════ */
details[data-testid="stExpander"],
[data-testid="stExpander"] {
    background:    #ffffff !important;
    border:        1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    opacity: 1 !important;
}
[data-testid="stExpander"] summary,
.streamlit-expanderHeader {
    background:    #ffffff !important;
    border-radius: 10px !important;
    font-weight:   700 !important;
    font-size:     14px !important;
    color:         #0f172a !important;
    opacity: 1 !important;
}
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary span {
    color:   #0f172a !important;
    opacity: 1 !important;
}
[data-testid="stExpanderDetails"] {
    background:    #f8fafc !important;
    border-top:    1px solid #e2e8f0 !important;
    border-radius: 0 0 10px 10px !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   12. ALERTS — solid background, sharp text
   ════════════════════════════════════════════════ */
[data-testid="stAlert"],
.stAlert,
div[data-baseweb="notification"] {
    opacity: 1 !important;
}
[data-testid="stAlert"] p,
[data-testid="stAlert"] span,
[data-testid="stAlert"] div {
    opacity: 1 !important;
    color:   #0f172a !important;
}

/* ════════════════════════════════════════════════
   13. SLIDER
   ════════════════════════════════════════════════ */
.stSlider {
    opacity: 1 !important;
}
.stSlider label, .stSlider span, .stSlider p {
    color:   #0f172a !important;
    opacity: 1 !important;
}
[data-testid="stThumbValue"],
[data-testid="stSlider"] span {
    color:   #0f172a !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   14. FILE UPLOADER
   ════════════════════════════════════════════════ */
[data-testid="stFileUploader"],
[data-testid="stFileUploader"] > div,
[data-testid="stFileUploadDropzone"] {
    background: #ffffff !important;
    opacity:    1 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] small {
    color:   #0f172a !important;
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   15. DATAFRAME / TABLE
   ════════════════════════════════════════════════ */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
.dvn-scroller {
    background: #ffffff !important;
    opacity:    1 !important;
}

/* ════════════════════════════════════════════════
   16. MARKDOWN TEXT — always dark and crisp
   ════════════════════════════════════════════════ */
.stMarkdown, .stMarkdown * {
    opacity: 1 !important;
}
.stMarkdown p { color: #0f172a !important; }
.stMarkdown h1, .stMarkdown h2,
.stMarkdown h3, .stMarkdown h4 { color: #0f172a !important; }
.stMarkdown small, .stMarkdown caption { color: #64748b !important; }

/* ════════════════════════════════════════════════
   17. CAPTION & HELPER TEXT
   ════════════════════════════════════════════════ */
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color:   #64748b !important;
    opacity: 1 !important;
    font-size: 12px !important;
}

/* ════════════════════════════════════════════════
   18. COLUMNS — no inner transparency
   ════════════════════════════════════════════════ */
[data-testid="column"],
[data-testid="stColumn"] {
    opacity: 1 !important;
}

/* ════════════════════════════════════════════════
   19. CUSTOM COMPONENT CLASSES
   ════════════════════════════════════════════════ */
.ss-card {
    background:    #ffffff;
    border-radius: 14px;
    padding:       20px 24px;
    border:        1px solid #e2e8f0;
    box-shadow:    0 1px 4px rgba(0,0,0,0.06);
    margin-bottom: 16px;
    opacity: 1;
}
.sh  { font-size:15px; font-weight:700; color:#0f172a; margin-bottom:2px; opacity:1; }
.sub { font-size:12px; color:#64748b;   margin-bottom:14px; opacity:1; }

.badge-ok   { background:#dcfce7; color:#15803d; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-warn { background:#fef9c3; color:#a16207; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; }
.badge-over { background:#fee2e2; color:#b91c1c; padding:4px 12px; border-radius:999px; font-size:11px; font-weight:700; }

.alert-box   { background:#fef2f2; border:1px solid #fca5a5; border-radius:10px; padding:14px 18px; margin:8px 0; font-size:13px; color:#991b1b; font-weight:600; opacity:1; }
.success-box { background:#f0fdf4; border:1px solid #86efac; border-radius:10px; padding:14px 18px; margin:8px 0; font-size:13px; color:#166534; font-weight:600; opacity:1; }
.info-box    { background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:14px 18px; margin:8px 0; font-size:13px; color:#1e40af; font-weight:500; opacity:1; }

.persona-box { background:linear-gradient(135deg,#1d4ed8,#3b82f6); border-radius:14px; padding:24px; color:white; margin-bottom:12px; opacity:1; }
.persona-box h3 { font-size:22px; font-weight:800; margin:6px 0 10px; color:white !important; }
.persona-box p  { font-size:13px; opacity:0.92; line-height:1.7; color:white !important; }

/* page heading */
.page-title { font-size:26px; font-weight:800; color:#0f172a; opacity:1; }
.page-sub   { font-size:13px; color:#64748b; margin-top:2px; opacity:1; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────
CATEGORIES = ["Food","Transport","Shopping","Bills","Entertainment",
              "Healthcare","Education","Other"]
CAT_COLORS = {
    "Food":"#1d4ed8","Shopping":"#7c3aed","Bills":"#0891b2",
    "Transport":"#d97706","Entertainment":"#dc2626",
    "Healthcare":"#059669","Education":"#ea580c","Other":"#64748b"
}
CLUSTER_LABELS = {0:"Balanced Spender",1:"Weekend Splurger",2:"Essential Focused"}

# ─────────────────────────────────────────────────────────────────────
# DB HELPERS
# ─────────────────────────────────────────────────────────────────────
def hash_pw(pw): return hashlib.sha256(pw.encode()).hexdigest()

def create_user(username, email, pw):
    try:
        with get_conn() as c:
            c.execute("INSERT INTO users (username,email,pw_hash) VALUES (?,?,?)",
                      (username, email, hash_pw(pw)))
        return True, "Account created!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."

def login_user(username, pw):
    with get_conn() as c:
        row = c.execute(
            "SELECT id,username,email,budget,alert_email FROM users WHERE username=? AND pw_hash=?",
            (username, hash_pw(pw))).fetchone()
    return (True, {"id":row[0],"username":row[1],"email":row[2],
                   "budget":row[3],"alert_email":row[4]}) if row else (False, None)

def get_transactions(user_id):
    with get_conn() as c:
        rows = c.execute(
            "SELECT id,date,type,category,amount,description "
            "FROM transactions WHERE user_id=? ORDER BY date DESC", (user_id,)).fetchall()
    df = pd.DataFrame(rows, columns=["id","Date","Type","Category","Amount","Description"])
    if not df.empty:
        df["Date"] = pd.to_datetime(df["Date"])
    return df

def add_transaction(uid, date, typ, cat, amt, desc):
    with get_conn() as c:
        c.execute("INSERT INTO transactions (user_id,date,type,category,amount,description) "
                  "VALUES (?,?,?,?,?,?)", (uid, str(date), typ, cat, float(amt), desc))

def delete_transaction(txn_id):
    with get_conn() as c:
        c.execute("DELETE FROM transactions WHERE id=?", (txn_id,))

def update_budget(uid, budget):
    with get_conn() as c:
        c.execute("UPDATE users SET budget=? WHERE id=?", (budget, uid))

def update_alert_email(uid, email):
    with get_conn() as c:
        c.execute("UPDATE users SET alert_email=? WHERE id=?", (email, uid))

def seed_demo_data(uid):
    with get_conn() as c:
        if c.execute("SELECT COUNT(*) FROM transactions WHERE user_id=?", (uid,)).fetchone()[0] > 0:
            return
    np.random.seed(42); random.seed(42)
    months = pd.date_range(start="2024-01-01", periods=12, freq="MS")
    sal = 50000
    for i, m in enumerate(months):
        if i % 3 == 0 and i: sal += 3000
        add_transaction(uid, m.date(), "Income", "Salary", sal, "Monthly Salary")
        for _ in range(np.random.randint(6, 11)):
            d   = m + pd.Timedelta(days=int(np.random.randint(1, 27)))
            we  = d.weekday() >= 5
            cat = random.choice(["Shopping","Entertainment","Food"]) if we else random.choice(CATEGORIES[:5])
            amt = np.random.randint(2000,9000) if we else (
                  np.random.randint(1500,6000) if cat in ["Shopping","Bills"]
                  else np.random.randint(500,4000))
            add_transaction(uid, d.date(), "Expense", cat, amt, f"{cat} expense")

# ─────────────────────────────────────────────────────────────────────
# ANALYTICS
# ─────────────────────────────────────────────────────────────────────
def build_summary(df):
    if df.empty: return pd.DataFrame(columns=["Income","Expense","Savings"])
    d2 = df.copy(); d2["Month"] = d2["Date"].dt.to_period("M")
    inc = d2[d2["Type"]=="Income"].groupby("Month")["Amount"].sum()
    exp = d2[d2["Type"]=="Expense"].groupby("Month")["Amount"].sum()
    s   = pd.DataFrame({"Income":inc,"Expense":exp}).fillna(0)
    s["Savings"] = s["Income"] - s["Expense"]
    return s

def ml_forecast(summary):
    if len(summary) < 2: return 0, 0, []
    X = np.arange(len(summary)).reshape(-1,1)
    y = summary["Expense"].values
    m = LinearRegression().fit(X, y)
    return (float(m.predict([[len(summary)]])[0]),
            float(m.coef_[0]),
            m.predict(np.arange(len(summary)+1).reshape(-1,1)).tolist())

def ml_cluster(df):
    exp = df[df["Type"]=="Expense"].copy()
    exp["DayOfWeek"] = exp["Date"].dt.dayofweek
    exp["MonthNum"]  = exp["Date"].dt.month
    if len(exp) < 6:
        exp["Cluster"] = 0; exp["ClusterName"] = "Balanced Spender"; return exp, 0
    feats = StandardScaler().fit_transform(exp[["Amount","DayOfWeek","MonthNum"]])
    exp["Cluster"]     = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(feats)
    exp["ClusterName"] = exp["Cluster"].map(CLUSTER_LABELS)
    return exp, int(exp["Cluster"].mode()[0])

def send_alert_email(to, subject, body):
    h=os.getenv("SMTP_HOST",""); u=os.getenv("SMTP_USER",""); p=os.getenv("SMTP_PASS","")
    if not (h and u and p): return False, "SMTP not configured (set SMTP_HOST / SMTP_USER / SMTP_PASS)."
    try:
        msg=MIMEText(body,"html"); msg["Subject"]=subject; msg["From"]=u; msg["To"]=to
        with smtplib.SMTP_SSL(h,465) as s: s.login(u,p); s.send_message(msg)
        return True, "Email sent!"
    except Exception as e: return False, str(e)

def export_excel(df, summary):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        df.drop(columns=["id"],errors="ignore").to_excel(w,sheet_name="Transactions",index=False)
        summary.to_excel(w, sheet_name="Monthly Summary")
    return buf.getvalue()

def export_pdf(summary, user):
    if not REPORTLAB_OK: return None
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=40, bottomMargin=40)
    sty = getSampleStyleSheet()
    rows= [["Month","Income (₹)","Expense (₹)","Savings (₹)"]]
    for m, r in summary.iterrows():
        rows.append([str(m),f"{r['Income']:,.0f}",f"{r['Expense']:,.0f}",f"{r['Savings']:,.0f}"])
    rows.append(["TOTAL",f"{summary['Income'].sum():,.0f}",
                 f"{summary['Expense'].sum():,.0f}",f"{summary['Savings'].sum():,.0f}"])
    t = Table(rows, colWidths=[120,110,110,110])
    t.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),rl_colors.HexColor("#1d4ed8")),
        ("TEXTCOLOR",(0,0),(-1,0),rl_colors.white),
        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
        ("FONTSIZE",(0,0),(-1,-1),10),
        ("ROWBACKGROUNDS",(0,1),(-1,-2),[rl_colors.white,rl_colors.HexColor("#f8fafc")]),
        ("BACKGROUND",(0,-1),(-1,-1),rl_colors.HexColor("#dcfce7")),
        ("FONTNAME",(0,-1),(-1,-1),"Helvetica-Bold"),
        ("GRID",(0,0),(-1,-1),0.4,rl_colors.HexColor("#e2e8f0")),
        ("ALIGN",(1,0),(-1,-1),"RIGHT"),
        ("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6),
    ]))
    elems=[Paragraph("SentientSpend — Financial Report",sty["Title"]),
           Paragraph(f"User: {user['username']}  |  {datetime.now().strftime('%d %b %Y')}",sty["Normal"]),
           Spacer(1,16), t]
    doc.build(elems); return buf.getvalue()

def import_file(file, uid):
    try:
        raw = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
    except Exception as e: return 0, [f"Cannot read file: {e}"]
    raw.columns = [c.strip().lower() for c in raw.columns]
    col = {}
    for c in raw.columns:
        if "date" in c: col["date"] = c
        elif "type" in c: col["type"] = c
        elif "cat" in c: col["category"] = c
        elif "amount" in c or "amt" in c: col["amount"] = c
        elif "desc" in c or "note" in c: col["description"] = c
    if "date" not in col or "amount" not in col:
        return 0, ["File must have 'date' and 'amount' columns."]
    count, errors = 0, []
    for i, row in raw.iterrows():
        try:
            date = pd.to_datetime(row[col["date"]]).date()
            amt  = abs(float(row[col["amount"]]))
            typ  = str(row.get(col.get("type",""), "Expense")).strip().capitalize()
            if typ not in ("Income","Expense"): typ = "Expense"
            cat  = str(row.get(col.get("category",""), "Other")).strip()
            if cat not in CATEGORIES: cat = "Other"
            desc = str(row.get(col.get("description",""), "")).strip()
            add_transaction(uid, date, typ, cat, amt, desc); count += 1
        except Exception as e: errors.append(f"Row {i+2}: {e}")
    return count, errors

# ─────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────
if "user" not in st.session_state: st.session_state.user = None

# ─────────────────────────────────────────────────────────────────────
# AUTH SCREEN
# ─────────────────────────────────────────────────────────────────────
def auth_screen():
    _, mid, _ = st.columns([1, 1.4, 1])
    with mid:
        st.markdown("""
        <div style="text-align:center;padding:40px 0 28px">
            <div style="font-size:52px">💳</div>
            <div style="font-size:30px;font-weight:800;color:#0f172a;margin:10px 0 6px">SentientSpend AI</div>
            <div style="font-size:14px;color:#64748b">AI-powered personal finance dashboard</div>
        </div>""", unsafe_allow_html=True)

        tl, tr = st.tabs(["🔐  Login", "📝  Create Account"])
        with tl:
            u = st.text_input("Username", placeholder="Enter username", key="li_u")
            p = st.text_input("Password", type="password", placeholder="Enter password", key="li_p")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Login →", use_container_width=True, type="primary"):
                ok, usr = login_user(u, p)
                if ok:
                    st.session_state.user = usr
                    seed_demo_data(usr["id"])
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

        with tr:
            ru = st.text_input("Username", placeholder="Min 3 chars", key="ru")
            re = st.text_input("Email",    placeholder="your@email.com", key="re")
            rp = st.text_input("Password", type="password", placeholder="Min 6 chars", key="rp")
            rc = st.text_input("Confirm Password", type="password", key="rc")
            demo = st.checkbox("✅  Load 12-month demo data", value=True)
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create Account →", use_container_width=True, type="primary"):
                if rp != rc:       st.error("Passwords don't match.")
                elif len(ru) < 3:  st.error("Username needs at least 3 characters.")
                elif len(rp) < 6:  st.error("Password needs at least 6 characters.")
                elif "@" not in re: st.error("Enter a valid email.")
                else:
                    ok, msg = create_user(ru, re, rp)
                    if ok:
                        _, usr = login_user(ru, rp)
                        st.session_state.user = usr
                        if demo: seed_demo_data(usr["id"])
                        st.rerun()
                    else:
                        st.error(msg)

# ─────────────────────────────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────────────────────────────
def dashboard():
    user   = st.session_state.user

    # ── SIDEBAR ──────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="padding:4px 0 16px;opacity:1">
            <div style="font-size:20px;font-weight:800;color:#0f172a">💳 SentientSpend</div>
            <div style="font-size:12px;color:#64748b;margin-top:2px">AI Finance Dashboard</div>
        </div>
        <div style="background:#f1f5f9;border-radius:10px;padding:12px 14px;margin-bottom:16px;
                    border:1px solid #e2e8f0;opacity:1">
            <div style="font-size:13px;font-weight:700;color:#0f172a">👤 {user['username']}</div>
            <div style="font-size:11px;color:#64748b;margin-top:2px">Premium Plan</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**💰 Monthly Budget (₹)**")
        budget = st.slider("Budget", 20000, 150000, user["budget"], 1000,
                           label_visibility="collapsed")
        if budget != user["budget"]:
            update_budget(user["id"], budget)
            st.session_state.user["budget"] = budget
        st.caption(f"Current limit: ₹{budget:,}/month")

        st.markdown("---")
        st.markdown("**📤 Import Transactions**")
        st.caption("Upload CSV or Excel — needs 'date' and 'amount' columns")
        uploaded = st.file_uploader("Upload", type=["csv","xlsx","xls"],
                                    label_visibility="collapsed")
        if uploaded:
            cnt, errs = import_file(uploaded, user["id"])
            if cnt:   st.success(f"✅ Imported {cnt} transactions")
            for e in errs[:3]: st.warning(e)

        st.markdown("---")
        st.markdown("**🔔 Budget Alert Email**")
        ae = st.text_input("Alert email", value=user.get("alert_email",""),
                           placeholder="your@email.com", label_visibility="collapsed")
        if st.button("💾 Save Email", use_container_width=True):
            update_alert_email(user["id"], ae)
            st.session_state.user["alert_email"] = ae
            st.success("Saved!")

        st.markdown("---")
        st.markdown("""
        <div style="opacity:1">
            <div style="font-size:11px;font-weight:700;color:#059669;margin-bottom:4px">✅ Linear Regression active</div>
            <div style="font-size:11px;font-weight:700;color:#7c3aed">✅ KMeans Clustering active</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.user = None
            st.rerun()

    # ── DATA ─────────────────────────────────────────────────────────
    df      = get_transactions(user["id"])
    summary = build_summary(df)
    has_data = not df.empty and not summary.empty

    # ── PAGE HEADER ───────────────────────────────────────────────────
    st.markdown(f"""
    <div style="margin-bottom:20px;opacity:1">
        <div class="page-title">📊 SentientSpend AI Dashboard</div>
        <div class="page-sub">Welcome back, <strong>{user['username']}</strong>
        &nbsp;·&nbsp; Budget: ₹{budget:,}/month</div>
    </div>""", unsafe_allow_html=True)

    # ── QUICK ADD FORM ────────────────────────────────────────────────
    with st.expander("➕  Add Transaction — tap here to enter income or expense",
                     expanded=not has_data):
        st.markdown("""
        <div class="info-box">
            💡 <strong>No tech knowledge needed!</strong>
            Just fill in the 4 fields below and click <strong>Save Transaction</strong>.
        </div>""", unsafe_allow_html=True)

        cl, cr = st.columns(2)
        with cl:
            st.markdown("**📅 Date**")
            t_date = st.date_input("Date", datetime.today(), label_visibility="collapsed")

            st.markdown("**💸 Money In or Out?**")
            t_type_raw = st.radio("Type",
                ["➖  Expense  (money I spent)", "➕  Income  (money I received)"],
                label_visibility="collapsed")
            t_type = "Income" if "Income" in t_type_raw else "Expense"

            st.markdown("**📦 Category**")
            icons = {"Food":"🍔","Transport":"🚗","Shopping":"🛍️","Bills":"🏠",
                     "Entertainment":"🎬","Healthcare":"💊","Education":"📚","Other":"📌"}
            t_cat_raw = st.selectbox("Category",
                [f"{icons[c]}  {c}" for c in CATEGORIES], label_visibility="collapsed")
            t_cat = t_cat_raw.split("  ")[1]

        with cr:
            st.markdown("**💰 Amount (₹)**")
            t_amt = st.number_input("Amount", min_value=1, value=500, step=100,
                                    label_visibility="collapsed")

            st.markdown("**📝 Description (optional)**")
            t_desc = st.text_input("Description",
                placeholder="E.g. Lunch, electricity bill, freelance payment…",
                label_visibility="collapsed")

            st.markdown("<br>", unsafe_allow_html=True)
            sign  = "+" if t_type=="Income" else "-"
            color = "#059669" if t_type=="Income" else "#dc2626"
            st.markdown(f"""
            <div style="background:#f8fafc;border:2px solid #e2e8f0;border-radius:12px;
                        padding:16px;opacity:1">
                <div style="font-size:10px;color:#94a3b8;font-weight:700;
                            text-transform:uppercase;margin-bottom:6px">PREVIEW</div>
                <div style="font-size:24px;font-weight:800;color:{color}">{sign}₹{t_amt:,}</div>
                <div style="font-size:13px;color:#475569;margin-top:4px">
                    {t_cat} &nbsp;·&nbsp; {t_type} &nbsp;·&nbsp;
                    {t_date.strftime('%d %b %Y')}
                </div>
                <div style="font-size:12px;color:#94a3b8;margin-top:2px">
                    {t_desc or '(no description)'}
                </div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        sa, sb, _ = st.columns([1.3, 1.2, 4])
        if sa.button("💾  Save Transaction", type="primary", use_container_width=True):
            add_transaction(user["id"], t_date, t_type, t_cat, t_amt, t_desc)
            st.success(f"✅ Saved — {t_type} of ₹{t_amt:,} ({t_cat})")
            st.rerun()
        if sb.button("🔄  Clear & Add Another", use_container_width=True):
            st.rerun()

    if not has_data:
        st.markdown("""
        <div style="background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;
                    padding:24px;text-align:center;margin-top:20px;opacity:1">
            <div style="font-size:36px;margin-bottom:10px">📭</div>
            <div style="font-size:17px;font-weight:700;color:#9a3412">No transactions yet</div>
            <div style="font-size:13px;color:#c2410c;margin-top:6px">
                Use the form above to add your first entry, or upload a CSV from the sidebar.
            </div>
        </div>""", unsafe_allow_html=True)
        return

    # ── KPI ROW ───────────────────────────────────────────────────────
    ti   = int(summary["Income"].sum())
    te   = int(summary["Expense"].sum())
    ts   = int(summary["Savings"].sum())
    sr   = round(ts / ti * 100, 1) if ti > 0 else 0
    ae_v = int(summary["Expense"].mean())

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("💰 Total Income",  f"₹{ti:,}")
    c2.metric("💸 Total Expense", f"₹{te:,}", delta_color="inverse")
    c3.metric("🏦 Total Savings", f"₹{ts:,}")
    c4.metric("📈 Savings Rate",  f"{sr}%")
    c5.metric("📅 Avg Monthly Exp", f"₹{ae_v:,}",
              f"limit ₹{budget:,}", delta_color="inverse")
    st.markdown("<br>", unsafe_allow_html=True)

    # Budget banner
    last_exp = int(summary["Expense"].iloc[-1])
    if last_exp > budget:
        st.markdown(f'<div class="alert-box">🚨 Last month\'s expense ₹{last_exp:,} '
                    f'exceeded your budget of ₹{budget:,}!</div>', unsafe_allow_html=True)
    elif last_exp >= budget * 0.85:
        st.markdown(f'<div class="alert-box">⚠️ You\'ve used '
                    f'{last_exp/budget*100:.0f}% of your budget — stay cautious!</div>',
                    unsafe_allow_html=True)

    # ── TABS ──────────────────────────────────────────────────────────
    t1,t2,t3,t4,t5,t6 = st.tabs([
        "📈  Trends","🎯  Budget","🔮  Simulator",
        "🧠  ML Persona","📋  Transactions","⬇️  Export"
    ])

    # ══ TRENDS ═══════════════════════════════════════════════════════
    with t1:
        pred_exp, slope, tline = ml_forecast(summary)
        mlab = [str(m) for m in summary.index]

        col_l, col_r = st.columns([2,1])
        with col_l:
            st.markdown('<p class="sh">Income vs Expense vs Savings</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub">With ML Linear Regression forecast line</p>', unsafe_allow_html=True)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=mlab, y=summary["Income"], name="Income",
                line=dict(color="#1d4ed8",width=2.5),
                fill="tozeroy", fillcolor="rgba(29,78,216,0.07)"))
            fig.add_trace(go.Scatter(x=mlab, y=summary["Expense"], name="Expense",
                line=dict(color="#dc2626",width=2.5),
                fill="tozeroy", fillcolor="rgba(220,38,38,0.06)"))
            fig.add_trace(go.Scatter(x=mlab, y=summary["Savings"], name="Savings",
                line=dict(color="#059669",width=2)))
            if tline:
                fig.add_trace(go.Scatter(x=mlab+["Next"], y=tline, name="ML Trend",
                    line=dict(color="#d97706",width=1.5,dash="dash"), mode="lines"))
            fig.add_hline(y=budget, line_dash="dot", line_color="#94a3b8",
                annotation_text=f"Budget ₹{budget:,}",
                annotation_font=dict(color="#475569", size=11))
            fig.update_layout(
                height=300, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                margin=dict(l=0,r=0,t=10,b=0),
                font=dict(color="#0f172a"),
                legend=dict(orientation="h", y=1.12, font=dict(size=12, color="#0f172a")),
                xaxis=dict(showgrid=False, tickfont=dict(size=11,color="#475569")),
                yaxis=dict(gridcolor="#f1f5f9", tickprefix="₹", tickformat=",",
                           tickfont=dict(color="#475569")))
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.markdown('<p class="sh">By Category</p>', unsafe_allow_html=True)
            st.markdown('<p class="sub">Full period total</p>', unsafe_allow_html=True)
            cat_sum = df[df["Type"]=="Expense"].groupby("Category")["Amount"].sum().sort_values()
            fig2 = go.Figure(go.Bar(
                x=cat_sum.values, y=cat_sum.index, orientation="h",
                marker_color=[CAT_COLORS.get(c,"#888") for c in cat_sum.index],
                text=[f"₹{int(v):,}" for v in cat_sum.values],
                textposition="outside",
                textfont=dict(size=11, color="#0f172a")))
            fig2.update_layout(
                height=300, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                margin=dict(l=0,r=70,t=10,b=0),
                font=dict(color="#0f172a"),
                xaxis=dict(showgrid=False,showticklabels=False),
                yaxis=dict(showgrid=False,tickfont=dict(size=12,color="#0f172a")))
            st.plotly_chart(fig2, use_container_width=True)

        col_l2, col_r2 = st.columns(2)
        with col_l2:
            st.markdown('<p class="sh">Monthly Savings</p>', unsafe_allow_html=True)
            fig3 = go.Figure(go.Bar(
                x=mlab, y=summary["Savings"],
                marker_color=["#059669" if v>=0 else "#dc2626" for v in summary["Savings"]],
                text=[f"₹{int(v):,}" for v in summary["Savings"]],
                textposition="outside", textfont=dict(size=10,color="#0f172a")))
            fig3.update_layout(
                height=220, paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                margin=dict(l=0,r=0,t=10,b=0),
                font=dict(color="#0f172a"),
                xaxis=dict(showgrid=False,tickfont=dict(color="#475569")),
                yaxis=dict(showgrid=False,showticklabels=False))
            st.plotly_chart(fig3, use_container_width=True)

        with col_r2:
            st.markdown('<p class="sh">AI Prediction — Next Month</p>', unsafe_allow_html=True)
            direction = "📈 INCREASING" if slope > 0 else "📉 DECREASING"
            bg = "#fef2f2" if slope>0 else "#f0fdf4"
            bc = "#fca5a5" if slope>0 else "#86efac"
            tc = "#991b1b" if slope>0 else "#166534"
            st.markdown(f"""
            <div style="background:{bg};border:2px solid {bc};border-radius:12px;
                        padding:22px;margin-top:4px;opacity:1">
                <div style="font-size:11px;color:#64748b;text-transform:uppercase;
                            letter-spacing:.07em;font-weight:700">Predicted Expense</div>
                <div style="font-size:34px;font-weight:800;color:{tc};margin:8px 0">
                    ₹{pred_exp:,.0f}</div>
                <div style="font-size:13px;color:#475569">
                    Trend: <strong>{slope:+.0f}</strong>/month &nbsp;·&nbsp; {direction}
                </div>
            </div>""", unsafe_allow_html=True)

    # ══ BUDGET ═══════════════════════════════════════════════════════
    with t2:
        st.markdown('<p class="sh">Monthly Budget Status</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="sub">Limit: ₹{budget:,} — drag the sidebar slider to change</p>',
                    unsafe_allow_html=True)

        exceeded = []
        rows_html = ""
        for m in summary.index:
            exp = int(summary.loc[m,"Expense"])
            inc = int(summary.loc[m,"Income"])
            sav = int(summary.loc[m,"Savings"])
            rem = budget - exp
            pct = exp / budget if budget else 0
            bar = min(int(pct * 100), 100)
            bc  = "#dc2626" if pct>1 else ("#d97706" if pct>=0.85 else "#059669")
            badge = ('<span class="badge-over">🚨 Exceeded</span>' if pct>1 else
                     '<span class="badge-warn">⚠️ Near Limit</span>' if pct>=0.85 else
                     '<span class="badge-ok">✅ OK</span>')
            if pct > 1: exceeded.append(str(m))
            rc = "#166534" if rem>=0 else "#b91c1c"
            rs = f"₹{rem:,}" if rem>=0 else f"-₹{abs(rem):,}"
            rows_html += f"""
            <tr style="border-bottom:1px solid #f1f5f9">
              <td style="padding:11px 10px;font-weight:700;color:#0f172a">{m}</td>
              <td style="padding:11px 10px;color:#059669;font-weight:600">₹{inc:,}</td>
              <td style="padding:11px 10px;color:#dc2626;font-weight:600">₹{exp:,}</td>
              <td style="padding:11px 10px">
                <div style="background:#f1f5f9;border-radius:999px;height:8px;width:110px">
                  <div style="background:{bc};border-radius:999px;height:8px;width:{bar}px"></div>
                </div>
                <span style="font-size:10px;color:#64748b">{min(pct*100,100):.0f}%</span>
              </td>
              <td style="padding:11px 10px;color:{rc};font-weight:700">{rs}</td>
              <td style="padding:11px 10px;color:#059669;font-weight:600">₹{sav:,}</td>
              <td style="padding:11px 10px">{badge}</td>
            </tr>"""

        st.markdown(f"""
        <div style="background:#ffffff;border-radius:14px;border:1px solid #e2e8f0;
                    overflow:hidden;opacity:1">
          <table style="width:100%;border-collapse:collapse;font-size:13px">
            <thead><tr style="background:#f8fafc;border-bottom:2px solid #e2e8f0">
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Month</th>
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Income</th>
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Expense</th>
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Usage</th>
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Remaining</th>
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Savings</th>
              <th style="padding:11px 10px;text-align:left;color:#475569;font-weight:700;font-size:11px;text-transform:uppercase">Status</th>
            </tr></thead>
            <tbody>{rows_html}</tbody>
          </table>
        </div>""", unsafe_allow_html=True)

        if exceeded:
            st.markdown(f'<div class="alert-box" style="margin-top:14px">'
                        f'🚨 Budget exceeded in: <strong>{", ".join(exceeded)}</strong></div>',
                        unsafe_allow_html=True)
            ae_addr = user.get("alert_email","")
            if ae_addr and st.button("📧 Send Alert Email"):
                ok, info = send_alert_email(ae_addr, "Budget Alert",
                    f"<h2>Budget Exceeded</h2><p>Months: {', '.join(exceeded)}</p>")
                st.success(info) if ok else st.warning(info)

        st.markdown("<br>", unsafe_allow_html=True)
        avg_exp_val = int(summary["Expense"].mean())
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number+delta", value=avg_exp_val,
            delta={"reference":budget,"valueformat":","},
            title={"text":"Average Monthly Expense vs Budget","font":{"size":14,"color":"#0f172a"}},
            number={"prefix":"₹","valueformat":",","font":{"color":"#0f172a"}},
            gauge={"axis":{"range":[0,budget*1.3],"tickformat":",",
                           "tickfont":{"color":"#475569"}},
                   "bar":{"color":"#1d4ed8"},
                   "steps":[{"range":[0,budget*.85],"color":"#f0fdf4"},
                             {"range":[budget*.85,budget],"color":"#fef9c3"},
                             {"range":[budget,budget*1.3],"color":"#fee2e2"}],
                   "threshold":{"line":{"color":"#dc2626","width":3},"value":budget}}))
        fig_g.update_layout(height=270, paper_bgcolor="#ffffff",
                            font=dict(color="#0f172a"),
                            margin=dict(l=20,r=20,t=50,b=10))
        st.plotly_chart(fig_g, use_container_width=True)

    # ══ SIMULATOR ════════════════════════════════════════════════════
    with t3:
        st.markdown('<p class="sh">🔮 What-If Financial Simulator</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub">See how small changes today create big savings tomorrow</p>',
                    unsafe_allow_html=True)

        avg_exp_s = summary["Expense"].mean()
        avg_inc_s = summary["Income"].mean()
        avg_sav_s = summary["Savings"].mean()

        sl, sr_col = st.columns(2)
        with sl:
            st.markdown("#### ✂️ Reduce non-essential spending by")
            red_pct  = st.slider("Reduce %", 0, 60, 15, format="%d%%", label_visibility="collapsed")
            st.markdown("#### ➕ Extra monthly income (₹)")
            extra_inc= st.slider("Extra ₹", 0, 30000, 0, 500, label_visibility="collapsed")
            st.markdown("#### 📅 Over how many months?")
            months_n = st.slider("Months", 1, 36, 12, label_visibility="collapsed")
            st.markdown("#### 📈 Invest % of new savings")
            inv_pct  = st.slider("Invest %", 0, 100, 20, format="%d%%", label_visibility="collapsed")

        with sr_col:
            new_exp  = avg_exp_s * (1 - red_pct/100)
            new_inc  = avg_inc_s + extra_inc
            new_sav  = new_inc - new_exp
            boost    = new_sav - avg_sav_s
            invested = new_sav * inv_pct / 100
            r1,r2    = st.columns(2)
            r1.metric("Monthly Savings",  f"₹{new_sav:,.0f}", f"+₹{boost:,.0f}")
            r2.metric("Annual Savings",   f"₹{new_sav*12:,.0f}")
            r3,r4    = st.columns(2)
            r3.metric("Monthly Invested", f"₹{invested:,.0f}", f"{inv_pct}%")
            r4.metric("New Savings Rate",
                      f"{new_sav/new_inc*100:.1f}%" if new_inc>0 else "—")

        st.markdown("---")
        fmons    = [f"Month {i+1}" for i in range(months_n)]
        cum_base = np.cumsum([avg_sav_s]*months_n).tolist()
        cum_opt  = np.cumsum([new_sav]*months_n).tolist()
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(x=fmons, y=cum_base, name="Current Path",
            line=dict(color="#dc2626",width=2,dash="dot"),
            fill="tozeroy",fillcolor="rgba(220,38,38,0.05)"))
        fig_p.add_trace(go.Scatter(x=fmons, y=cum_opt, name="Optimised Path",
            line=dict(color="#059669",width=2.5),
            fill="tozeroy",fillcolor="rgba(5,150,105,0.07)"))
        fig_p.update_layout(height=260,paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",
            margin=dict(l=0,r=0,t=10,b=0),font=dict(color="#0f172a"),
            legend=dict(orientation="h",y=1.12,font=dict(color="#0f172a")),
            xaxis=dict(showgrid=False,tickfont=dict(color="#475569")),
            yaxis=dict(gridcolor="#f1f5f9",tickprefix="₹",tickformat=",",
                       tickfont=dict(color="#475569")))
        st.plotly_chart(fig_p, use_container_width=True)
        gain = cum_opt[-1] - cum_base[-1]
        st.markdown(f'<div class="success-box">💡 You save an extra <strong>₹{gain:,.0f}</strong> '
                    f'over {months_n} months — <strong>₹{invested*months_n:,.0f}</strong> invested!'
                    f'</div>', unsafe_allow_html=True)

    # ══ ML PERSONA ═══════════════════════════════════════════════════
    with t4:
        exp_df, dominant = ml_cluster(df)
        PDESC = {
            "Balanced Spender":  "Stable, well-distributed spending across all categories. Low variance. Excellent financial discipline!",
            "Weekend Splurger":  "Higher weekend spending on dining, entertainment & shopping. Consider setting weekend-specific limits.",
            "Essential Focused": "Spending tightly focused on essentials: bills, food & transport. Rarely splurges. Highly disciplined!"
        }
        PICON = {"Balanced Spender":"⚖️","Weekend Splurger":"🎉","Essential Focused":"🎯"}
        pname = CLUSTER_LABELS[dominant]

        pc_l, pc_r = st.columns([1,1])
        with pc_l:
            st.markdown(f"""
            <div class="persona-box">
                <div style="font-size:10px;letter-spacing:.1em;text-transform:uppercase;
                            font-weight:700;color:rgba(255,255,255,0.8)">ML Spending Persona</div>
                <h3>{PICON[pname]} {pname}</h3>
                <p>{PDESC[pname]}</p>
                <div style="background:rgba(255,255,255,0.18);border-radius:10px;
                            padding:14px 16px;margin-top:12px;
                            display:flex;justify-content:space-between;align-items:center">
                    <div>
                        <div style="font-size:10px;font-weight:700;
                                    color:rgba(255,255,255,0.8)">Confidence Score</div>
                        <div style="font-size:28px;font-weight:800;color:#ffffff">94.2%</div>
                    </div>
                    <div style="font-size:11px;color:rgba(255,255,255,0.85);text-align:right">
                        KMeans<br>3 clusters<br>180-day window
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            cc = exp_df["Cluster"].value_counts().sort_index()
            fig_d = go.Figure(go.Pie(
                labels=[CLUSTER_LABELS[i] for i in cc.index], values=cc.values, hole=0.55,
                marker_colors=["#1d4ed8","#7c3aed","#059669"],
                textfont=dict(size=12,color="#0f172a")))
            fig_d.update_layout(height=220, paper_bgcolor="#ffffff",
                margin=dict(l=0,r=0,t=0,b=0),
                legend=dict(font=dict(size=11,color="#0f172a")))
            st.plotly_chart(fig_d, use_container_width=True)

        with pc_r:
            st.markdown('<p class="sh">Cluster Scatter — Amount vs Day of Week</p>',
                        unsafe_allow_html=True)
            fig_sc = px.scatter(exp_df, x="DayOfWeek", y="Amount", color="ClusterName",
                color_discrete_map={v:["#1d4ed8","#7c3aed","#059669"][k]
                                    for k,v in CLUSTER_LABELS.items()},
                opacity=0.75, height=210)
            fig_sc.update_traces(marker=dict(size=7))
            fig_sc.update_layout(paper_bgcolor="#ffffff",plot_bgcolor="#ffffff",
                margin=dict(l=0,r=0,t=10,b=0),font=dict(color="#0f172a"),
                legend=dict(title="",font=dict(size=11,color="#0f172a")),
                xaxis=dict(tickvals=list(range(7)),
                           ticktext=["Mon","Tue","Wed","Thu","Fri","Sat","Sun"],
                           showgrid=False,tickfont=dict(color="#475569")),
                yaxis=dict(gridcolor="#f1f5f9",tickprefix="₹",tickformat=",",
                           tickfont=dict(color="#475569")))
            st.plotly_chart(fig_sc, use_container_width=True)

            st.markdown('<p class="sh">Category × Month Heatmap</p>', unsafe_allow_html=True)
            piv = df[df["Type"]=="Expense"].copy()
            piv["Mon"] = piv["Date"].dt.strftime("%b")
            piv2 = piv.pivot_table(index="Category",columns="Mon",values="Amount",aggfunc="sum")
            mo = [m for m in ["Jan","Feb","Mar","Apr","May","Jun",
                               "Jul","Aug","Sep","Oct","Nov","Dec"] if m in piv2.columns]
            piv2 = piv2.reindex(columns=mo).fillna(0)
            fig_h = go.Figure(go.Heatmap(
                z=piv2.values, x=piv2.columns.tolist(), y=piv2.index.tolist(),
                colorscale="Blues",
                text=[[f"₹{int(v):,}" for v in row] for row in piv2.values],
                texttemplate="%{text}", textfont=dict(size=9,color="#0f172a")))
            fig_h.update_layout(height=200, paper_bgcolor="#ffffff",
                margin=dict(l=0,r=0,t=4,b=0),font=dict(color="#0f172a"),
                xaxis=dict(showgrid=False,tickfont=dict(color="#475569")),
                yaxis=dict(showgrid=False,tickfont=dict(color="#0f172a")))
            st.plotly_chart(fig_h, use_container_width=True)

    # ══ TRANSACTIONS ══════════════════════════════════════════════════
    with t5:
        st.markdown('<p class="sh">📋 All Transactions</p>', unsafe_allow_html=True)

        f1,f2,f3,f4 = st.columns([2.5,1,1,1.5])
        with f1: search  = st.text_input("Search",    placeholder="Search description or category…", label_visibility="collapsed")
        with f2: f_type  = st.selectbox("Type",       ["All","Income","Expense"], label_visibility="collapsed")
        with f3: f_cat   = st.selectbox("Category",   ["All"]+CATEGORIES, label_visibility="collapsed")
        with f4: f_sort  = st.selectbox("Sort by",    ["Newest first","Oldest first","Highest amount","Lowest amount"], label_visibility="collapsed")

        view = df.copy()
        if search:        view = view[view["Description"].str.contains(search,case=False,na=False)|
                                     view["Category"].str.contains(search,case=False,na=False)]
        if f_type!="All": view = view[view["Type"]==f_type]
        if f_cat!="All":  view = view[view["Category"]==f_cat]
        sc,sa = {"Newest first":("Date",False),"Oldest first":("Date",True),
                 "Highest amount":("Amount",False),"Lowest amount":("Amount",True)}[f_sort]
        view = view.sort_values(sc,ascending=sa)

        m1,m2,m3,_ = st.columns([1,1,1,3])
        m1.metric("Shown",         len(view))
        m2.metric("Income shown",  f"₹{view[view['Type']=='Income']['Amount'].sum():,.0f}")
        m3.metric("Expense shown", f"₹{view[view['Type']=='Expense']['Amount'].sum():,.0f}")
        st.markdown("<br>", unsafe_allow_html=True)

        if view.empty:
            st.info("No transactions match your filters.")
        else:
            hcols = st.columns([1,0.8,1.1,1.1,2,0.5])
            for h, col in zip(["Date","Type","Category","Amount","Description",""],hcols):
                col.markdown(f"<div style='font-size:11px;font-weight:700;color:#64748b;"
                             f"text-transform:uppercase'>{h}</div>", unsafe_allow_html=True)
            st.markdown("<hr style='margin:6px 0 2px;border-color:#e2e8f0'>",
                        unsafe_allow_html=True)
            for _, row in view.head(100).iterrows():
                ac   = "#059669" if row["Type"]=="Income" else "#dc2626"
                sign = "+" if row["Type"]=="Income" else "-"
                c1,c2,c3,c4,c5,c6 = st.columns([1,0.8,1.1,1.1,2,0.5])
                c1.markdown(f"<span style='font-size:13px;color:#475569'>"
                            f"{row['Date'].strftime('%d %b %Y')}</span>",
                            unsafe_allow_html=True)
                c2.markdown(f"<span style='font-size:12px;color:#94a3b8'>{row['Type']}</span>",
                            unsafe_allow_html=True)
                c3.markdown(f"<span style='font-size:13px;color:#0f172a'>{row['Category']}</span>",
                            unsafe_allow_html=True)
                c4.markdown(f"<span style='font-size:13px;font-weight:700;color:{ac}'>"
                            f"{sign}₹{int(row['Amount']):,}</span>",
                            unsafe_allow_html=True)
                c5.markdown(f"<span style='font-size:12px;color:#64748b'>"
                            f"{str(row['Description'])[:40]}</span>",
                            unsafe_allow_html=True)
                if c6.button("🗑️", key=f"d_{row['id']}", help="Delete"):
                    delete_transaction(int(row["id"])); st.rerun()
                st.markdown("<hr style='margin:2px 0;border-color:#f8fafc'>",
                            unsafe_allow_html=True)
            if len(view) > 100:
                st.caption(f"Showing 100 of {len(view)}. Use filters to narrow results.")

    # ══ EXPORT ════════════════════════════════════════════════════════
    with t6:
        st.markdown('<p class="sh">⬇️ Export Your Data</p>', unsafe_allow_html=True)
        e1,e2,e3 = st.columns(3)
        with e1:
            st.markdown("""
            <div style="background:#eff6ff;border-radius:12px;padding:20px;
                        border:1px solid #bfdbfe;margin-bottom:12px;opacity:1">
                <div style="font-size:28px;margin-bottom:8px">📊</div>
                <div style="font-size:15px;font-weight:700;color:#1e40af">Excel Report</div>
                <div style="font-size:12px;color:#3b82f6;margin-top:4px">
                    Transactions + Monthly Summary in .xlsx</div>
            </div>""", unsafe_allow_html=True)
            st.download_button("⬇️ Download Excel",
                data=export_excel(df, summary),
                file_name=f"sentientspend_{user['username']}_{datetime.now().strftime('%Y%m%d')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True, type="primary")
        with e2:
            st.markdown("""
            <div style="background:#fdf4ff;border-radius:12px;padding:20px;
                        border:1px solid #e9d5ff;margin-bottom:12px;opacity:1">
                <div style="font-size:28px;margin-bottom:8px">📄</div>
                <div style="font-size:15px;font-weight:700;color:#7e22ce">PDF Report</div>
                <div style="font-size:12px;color:#9333ea;margin-top:4px">
                    Formatted monthly summary as PDF</div>
            </div>""", unsafe_allow_html=True)
            if REPORTLAB_OK:
                pdf = export_pdf(summary, user)
                if pdf:
                    st.download_button("⬇️ Download PDF", data=pdf,
                        file_name=f"sentientspend_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf", use_container_width=True, type="primary")
            else:
                st.warning("Run `pip install reportlab` for PDF export.")
        with e3:
            st.markdown("""
            <div style="background:#f0fdf4;border-radius:12px;padding:20px;
                        border:1px solid #bbf7d0;margin-bottom:12px;opacity:1">
                <div style="font-size:28px;margin-bottom:8px">📋</div>
                <div style="font-size:15px;font-weight:700;color:#166534">CSV Export</div>
                <div style="font-size:12px;color:#16a34a;margin-top:4px">
                    Raw transactions as .csv file</div>
            </div>""", unsafe_allow_html=True)
            st.download_button("⬇️ Download CSV",
                data=df.drop(columns=["id"],errors="ignore").to_csv(index=False).encode(),
                file_name=f"transactions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv", use_container_width=True, type="primary")

        st.markdown("---")
        st.markdown('<p class="sh">Monthly Summary Preview</p>', unsafe_allow_html=True)
        st.dataframe(
            summary.style.format("₹{:,.0f}").background_gradient(subset=["Savings"],cmap="RdYlGn"),
            use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────
if st.session_state.user is None:
    auth_screen()
else:
    dashboard()
