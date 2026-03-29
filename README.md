# 💳 SentientSpend AI — Personal Finance Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-brightgreen?style=flat-square)

> An AI-powered personal finance dashboard built with Python and Streamlit.  
> Track income, expenses, and savings — with ML predictions and spending persona analysis.

---

## 🌐 Live Demo

👉 **[sentientspend.streamlit.app](https://kamalapro05-sentientspend.streamlit.app)**

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Dashboard** | KPI cards for income, expense, savings, and savings rate |
| 📈 **Trend Charts** | Monthly income vs expense with ML Linear Regression forecast |
| 🎯 **Budget Monitor** | Live budget alerts — Under / Near Limit / Exceeded |
| 🔮 **What-If Simulator** | Adjust spending sliders and see future savings projections |
| 🧠 **ML Persona** | KMeans clustering identifies your spending personality |
| 📋 **Transactions** | Add, search, filter, and delete transactions manually |
| 📤 **CSV / Excel Import** | Upload your bank statement directly |
| ⬇️ **Export** | Download your data as Excel, PDF, or CSV |
| 🔐 **Login System** | Secure user accounts with SQLite database |
| 🔔 **Email Alerts** | Get notified when budget is exceeded |

---

## 🚀 Installation — Run Locally on Windows

### Step 1 — Make sure Python is installed

Download Python from **[python.org](https://python.org)**

> ⚠️ During install — tick **"Add Python to PATH"** before clicking Install

Check it worked — open Command Prompt and type:
```
python --version
```
You should see something like `Python 3.12.0`

---

### Step 2 — Get the project files

You have **two options** — pick whichever is easier:

---

**Option A — Download ZIP (easiest, no Git needed)**

Click the green **"Code"** button on this page → **"Download ZIP"**

Extract the ZIP to a folder — for example:
```
C:\Users\YourName\Desktop\sentientspend\
```

---

**Option B — Clone with Git (recommended)**

First make sure Git is installed — download from **[git-scm.com](https://git-scm.com)** if you don't have it.

Open Command Prompt anywhere and run:
```
git clone https://github.com/Kamalapro05/sentientspend.git
```

Then move into the folder:
```
cd sentientspend
```

> 💡 To get future updates later, just run `git pull` inside the folder.

---

### Step 3 — Open Command Prompt in the folder

**If you used Option A (ZIP):**
Open the extracted `sentientspend` folder → click the **address bar** at the top → type `cmd` → press **Enter**.

**If you used Option B (Git clone):**
You are already inside the folder from the previous step. ✅

---

### Step 4 — Install required libraries

Paste this command and press **Enter**. Wait 2–3 minutes:

```
pip install streamlit pandas numpy plotly scikit-learn openpyxl reportlab
```

---

### Step 5 — Run the app

```
streamlit run app.py
```

Your browser opens automatically at **http://localhost:8501** 🎉

---

### Step 6 — Create your account

- Click **"Create Account"** tab
- Enter a username, email, and password
- Tick **"Load 12-month demo data"** to explore with sample data
- Click **Create Account**

---

## ☁️ Deploy to Streamlit Cloud (Free Hosting)

### Step 1 — Push to GitHub

1. Create a free account at **[github.com](https://github.com)**
2. Create a new **public** repository named `sentientspend`
3. Upload all project files:
   - `app.py`
   - `requirements.txt`
   - `packages.txt` ← must be completely empty
   - `sample_data.csv`
   - `.streamlit/config.toml`

### Step 2 — Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository:** `Kamalapro05/sentientspend`
   - **Branch:** `main`
   - **Main file:** `app.py`
5. Click **"Deploy!"** — wait 2–3 minutes

Your app is live at `https://kamalapro05-sentientspend.streamlit.app` ✅

---

## 📁 Project Structure

```
sentientspend/
├── app.py                  ← Main application
├── requirements.txt        ← Python libraries
├── packages.txt            ← System packages (keep empty)
├── sample_data.csv         ← Demo data for testing
├── sentientspend.db        ← SQLite database (auto-created, never upload)
└── .streamlit/
    └── config.toml         ← Theme config (fixes blur issue)
```

---

## 📋 CSV Import Format

You can import your own bank data. The CSV needs at minimum a `date` and `amount` column:

```csv
date,type,category,amount,description
2024-01-01,Income,Salary,50000,Monthly Salary
2024-01-05,Expense,Food,1200,Groceries
2024-01-08,Expense,Transport,800,Fuel
```

**Supported categories:** Food, Transport, Shopping, Bills, Entertainment, Healthcare, Education, Other

---

## 🔔 Email Alerts Setup (Optional)

To receive budget alert emails, set these environment variables.

**Locally** — create a `.env` file in the project folder:
```
SMTP_HOST=smtp.gmail.com
SMTP_USER=your@gmail.com
SMTP_PASS=your_app_password
```

**On Streamlit Cloud** — go to your app → Settings → Secrets → add the same values.

> For Gmail, use an **App Password** — Google Account → Security → 2-Step Verification → App Passwords

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit, Plotly, Custom CSS |
| Backend | Python 3.12 |
| Database | SQLite (local) |
| Machine Learning | Scikit-learn (Linear Regression + KMeans) |
| Data Processing | Pandas, NumPy |
| Export | OpenPyXL (Excel), ReportLab (PDF) |

---

## ❓ Troubleshooting

| Problem | Fix |
|---|---|
| `streamlit: command not found` | Try `python -m streamlit run app.py` |
| `ModuleNotFoundError` | Run the pip install command again (Step 4) |
| App looks blurry / faded | Make sure `.streamlit/config.toml` exists in the folder |
| `background_gradient requires matplotlib` | You have the old app.py — download the latest version |
| `Error installing requirements` on cloud | Make sure `packages.txt` is completely empty |
| Database resets on cloud | Expected — Streamlit free tier resets on inactivity |
| Browser doesn't open | Go to `http://localhost:8501` manually |

---

## 🗺️ Roadmap

- [ ] Supabase integration for permanent cloud database
- [ ] Dark mode toggle
- [ ] Mobile-friendly layout
- [ ] Recurring transaction support
- [ ] Multi-currency support

---

## 📄 License

MIT License — free to use, modify, and share.

---

## 🙏 Built With

- [Streamlit](https://streamlit.io) — Python web app framework
- [Plotly](https://plotly.com) — Interactive charts
- [Scikit-learn](https://scikit-learn.org) — Machine learning
- [Pandas](https://pandas.pydata.org) — Data processing

---

<div align="center">
Made with ❤️ using Python & Streamlit
</div>
