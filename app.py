import warnings
warnings.filterwarnings("ignore")

import io
import os
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import requests
import json

from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    VotingRegressor, IsolationForest
)
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, mean_absolute_percentage_error
from sklearn.inspection import permutation_importance

# Optional libraries
try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ========================== CONFIGURATION ==========================
MAX_FILE_SIZE_MB = 1000
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
GUEST_MAX_ROWS = 1000  # Guest users have the same limit as Free plan

CONFIG_DIR = Path(".streamlit")
CONFIG_FILE = CONFIG_DIR / "config.toml"
if not CONFIG_FILE.exists():
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        f.write(f"""
[server]
maxUploadSize = {MAX_FILE_SIZE_MB}
""")

# ========================== ENCODING HELPER ==========================
def read_csv_with_encoding(uploaded_file):
    encodings = ['utf-8', 'windows-1256', 'iso-8859-1', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            return df, enc
        except (UnicodeDecodeError, Exception):
            continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, "Unable to decode file with common encodings.")

# ========================== DATABASE ==========================
DB_PATH = "nexus_users.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS login_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT,
            success INTEGER,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS system_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_email TEXT,
            action TEXT,
            details TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS subscription_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL,
            price_monthly REAL,
            price_yearly REAL,
            max_rows INTEGER,
            features TEXT,
            is_active INTEGER DEFAULT 1
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            plan_id INTEGER NOT NULL,
            start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            end_date TIMESTAMP,
            is_active INTEGER DEFAULT 1,
            payment_method TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id),
            FOREIGN KEY (plan_id) REFERENCES subscription_plans(id)
        )
    ''')
    
    c.execute('''
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    c.execute("SELECT COUNT(*) FROM subscription_plans")
    if c.fetchone()[0] == 0:
        plans = [
            ("Free", 0, 0, 1000, "Basic analytics, limited rows, no export, no forecasting, no market basket, no clustering", 1),
            ("Pro", 19.99, 199.99, 50000, "Full analytics, forecasting, market basket, clustering, export, priority support", 1),
            ("Enterprise", 49.99, 499.99, 999999999, "Unlimited rows, all Pro features + custom models, dedicated support, API access", 1)
        ]
        c.executemany("INSERT INTO subscription_plans (name, price_monthly, price_yearly, max_rows, features, is_active) VALUES (?,?,?,?,?,?)", plans)
    
    # AI settings defaults
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('deepseek_api_key', '')")
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('groq_api_key', '')")
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('ai_provider', 'deepseek')")
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('custom_ai_url', '')")
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('custom_ai_api_key', '')")
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('custom_ai_model', 'gpt-3.5-turbo')")
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('custom_ai_enabled', '0')")
    
    conn.commit()
    
    admin_email = "kareemeltemsah7@gmail.com"
    admin_pass = "temsah1!"
    hashed = hashlib.sha256(admin_pass.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE email = ?", (admin_email,))
    existing = c.fetchone()
    if existing:
        c.execute("UPDATE users SET password_hash = ?, is_admin = 1 WHERE email = ?", (hashed, admin_email))
    else:
        c.execute("INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, 1)",
                  (admin_email, hashed))
    
    conn.commit()
    conn.close()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def hash_password(pwd):
    return hashlib.sha256(pwd.encode()).hexdigest()

def verify_password(email, pwd):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT password_hash, is_admin FROM users WHERE email = ?", (email,))
        row = c.fetchone()
        if row and row["password_hash"] == hash_password(pwd):
            c.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE email = ?", (email,))
            conn.commit()
            return True, bool(row["is_admin"])
    return False, False

def register_user(email, pwd, is_admin=False):
    with get_db() as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
                      (email, hash_password(pwd), 1 if is_admin else 0))
            conn.commit()
            c.execute("SELECT id FROM subscription_plans WHERE name = 'Free'")
            free_plan = c.fetchone()
            if free_plan:
                c.execute("INSERT INTO user_subscriptions (user_id, plan_id, start_date, end_date, is_active) VALUES (?, ?, ?, ?, 1)",
                          (c.lastrowid, free_plan["id"], datetime.now(), datetime.now() + timedelta(days=365*100)))
            return True
        except sqlite3.IntegrityError:
            return False

def log_login_attempt(email, success):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO login_logs (email, success) VALUES (?, ?)", (email, 1 if success else 0))
        conn.commit()

def log_system_action(user_email, action, details=""):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT INTO system_logs (user_email, action, details) VALUES (?, ?, ?)",
                  (user_email, action, details[:500]))
        conn.commit()

def get_all_users():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, email, is_admin, created_at, last_login FROM users ORDER BY id")
        return c.fetchall()

def delete_user(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT email FROM users WHERE id = ?", (user_id,))
        row = c.fetchone()
        if row:
            log_system_action("system", "delete_user", f"Deleted user {row['email']} (ID: {user_id})")
        c.execute("DELETE FROM user_subscriptions WHERE user_id = ?", (user_id,))
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

def toggle_admin(user_id, make_admin):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if make_admin else 0, user_id))
        conn.commit()

def get_user_by_id(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, email, is_admin FROM users WHERE id = ?", (user_id,))
        return c.fetchone()

def get_user_by_email(email):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, email, is_admin FROM users WHERE email = ?", (email,))
        return c.fetchone()

def add_admin_by_email(email):
    with get_db() as conn:
        c = conn.cursor()
        user = get_user_by_email(email)
        if user:
            c.execute("UPDATE users SET is_admin = 1 WHERE email = ?", (email,))
            conn.commit()
            log_system_action("system", "promote_to_admin", f"Promoted user {email} to admin")
            return True, f"User {email} is now admin."
        else:
            return False, f"User {email} not found. Please ask them to register first."

def reset_user_password(user_id, new_password):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new_password), user_id))
        conn.commit()
        log_system_action("system", "reset_password", f"Reset password for user ID {user_id}")

def get_login_logs(limit=200):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT email, success, timestamp FROM login_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return c.fetchall()

def get_system_logs(limit=500):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT user_email, action, details, timestamp FROM system_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return c.fetchall()

def get_stats():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT COUNT(*) FROM users")
        total_users = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM users WHERE is_admin = 1")
        total_admins = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM login_logs WHERE success = 1")
        total_success_logins = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM login_logs WHERE success = 0")
        total_failed_logins = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM system_logs")
        total_actions = c.fetchone()[0]
        today = pd.Timestamp.now().date()
        c.execute("SELECT COUNT(*) FROM login_logs WHERE success = 1 AND date(timestamp) = ?", (str(today),))
        today_logins = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM login_logs WHERE date(timestamp) = ?", (str(today),))
        today_attempts = c.fetchone()[0]
        return {
            "total_users": total_users,
            "total_admins": total_admins,
            "total_success_logins": total_success_logins,
            "total_failed_logins": total_failed_logins,
            "total_actions": total_actions,
            "today_logins": today_logins,
            "today_attempts": today_attempts,
        }

# ======================== SUBSCRIPTION HELPERS ========================
def get_available_plans():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name, price_monthly, price_yearly, max_rows, features FROM subscription_plans WHERE is_active = 1 ORDER BY price_monthly")
        return c.fetchall()

def get_all_plans():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id, name, price_monthly, price_yearly, max_rows, features FROM subscription_plans ORDER BY id")
        return c.fetchall()

def update_plan(plan_id, price_monthly, price_yearly, max_rows, features):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE subscription_plans SET price_monthly = ?, price_yearly = ?, max_rows = ?, features = ? WHERE id = ?",
                  (price_monthly, price_yearly, max_rows, features, plan_id))
        conn.commit()
        log_system_action("system", "update_plan", f"Updated plan ID {plan_id}")

def get_user_subscription(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT sp.name, sp.max_rows, sp.features, us.start_date, us.end_date, us.is_active, sp.id as plan_id
            FROM user_subscriptions us
            JOIN subscription_plans sp ON us.plan_id = sp.id
            WHERE us.user_id = ? AND us.is_active = 1
            ORDER BY us.start_date DESC LIMIT 1
        ''', (user_id,))
        row = c.fetchone()
        if row:
            return dict(row)
        else:
            c.execute("SELECT id, name, max_rows, features FROM subscription_plans WHERE name = 'Free'")
            free = c.fetchone()
            if free:
                c.execute("INSERT INTO user_subscriptions (user_id, plan_id, start_date, end_date, is_active) VALUES (?, ?, ?, ?, 1)",
                          (user_id, free["id"], datetime.now(), datetime.now() + timedelta(days=365*100)))
                return {"name": free["name"], "max_rows": free["max_rows"], "features": free["features"], "plan_id": free["id"]}
            return None

def upgrade_subscription(user_id, plan_id, duration_months=1, payment_method="manual"):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE user_subscriptions SET is_active = 0 WHERE user_id = ?", (user_id,))
        start = datetime.now()
        end = start + timedelta(days=30*duration_months)
        c.execute("INSERT INTO user_subscriptions (user_id, plan_id, start_date, end_date, is_active, payment_method) VALUES (?, ?, ?, ?, 1, ?)",
                  (user_id, plan_id, start, end, payment_method))
        conn.commit()
        log_system_action("system", "subscription_upgrade", f"User {user_id} upgraded to plan {plan_id} for {duration_months} months (method: {payment_method})")
        return True

def get_all_subscriptions():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT u.id as user_id, u.email, sp.name as plan_name, us.start_date, us.end_date, us.is_active
            FROM user_subscriptions us
            JOIN users u ON us.user_id = u.id
            JOIN subscription_plans sp ON us.plan_id = sp.id
            ORDER BY us.start_date DESC
        ''')
        rows = c.fetchall()
        result = []
        for row in rows:
            result.append({
                "user_id": row["user_id"],
                "email": row["email"],
                "plan_name": row["plan_name"],
                "start_date": row["start_date"],
                "end_date": row["end_date"],
                "is_active": row["is_active"]
            })
        return result

def get_subscription_stats():
    with get_db() as conn:
        c = conn.cursor()
        c.execute('''
            SELECT sp.name, COUNT(us.id) as user_count
            FROM user_subscriptions us
            JOIN subscription_plans sp ON us.plan_id = sp.id
            WHERE us.is_active = 1
            GROUP BY sp.name
        ''')
        plan_counts = {row["name"]: row["user_count"] for row in c.fetchall()}
        c.execute("SELECT SUM(sp.price_monthly) FROM user_subscriptions us JOIN subscription_plans sp ON us.plan_id = sp.id WHERE us.is_active = 1")
        monthly_revenue = c.fetchone()[0] or 0
        return plan_counts, monthly_revenue

def cancel_subscription(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE user_subscriptions SET is_active = 0 WHERE user_id = ? AND is_active = 1", (user_id,))
        c.execute("SELECT id FROM subscription_plans WHERE name = 'Free'")
        free_plan = c.fetchone()
        if free_plan:
            c.execute("INSERT INTO user_subscriptions (user_id, plan_id, start_date, end_date, is_active) VALUES (?, ?, ?, ?, 1)",
                      (user_id, free_plan["id"], datetime.now(), datetime.now() + timedelta(days=365*100)))
        conn.commit()

def extend_subscription(user_id, extra_months=1):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT end_date FROM user_subscriptions WHERE user_id = ? AND is_active = 1", (user_id,))
        row = c.fetchone()
        if row:
            new_end = datetime.strptime(row["end_date"], "%Y-%m-%d %H:%M:%S") + timedelta(days=30*extra_months)
            c.execute("UPDATE user_subscriptions SET end_date = ? WHERE user_id = ? AND is_active = 1", (new_end, user_id))
            conn.commit()
            return True
        return False

# ======================== GLOBAL SETTINGS ========================
def get_setting(key, default=""):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
        row = c.fetchone()
        return row["value"] if row else default

def set_setting(key, value):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                  (key, value))
        conn.commit()
        log_system_action("system", "update_setting", f"Updated {key}")

# ========================== PAGE CONFIG WITH MOBILE OPTIMIZATION ==========================
st.set_page_config(
    page_title="NEXUS Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ================== DARK THEME CSS ==================
st.markdown("""
<style>
/* Global dark background */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #0f172a !important;
    color: #f1f5f9 !important;
}

[data-testid="stSidebar"] {
    background-color: #0f172a !important;
    border-right: 1px solid #334155 !important;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

/* Cards, expanders, metrics */
.nx-header, .element-container, [data-testid="stExpander"], [data-testid="stMetric"] {
    background-color: #1e293b !important;
    border-radius: 20px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3) !important;
    color: #f1f5f9 !important;
}

/* Chat bubbles */
.chat-message.user .chat-bubble {
    background-color: #334155 !important;
    color: white !important;
}
.chat-message.assistant .chat-bubble {
    background-color: #1e293b !important;
    border: 1px solid #475569 !important;
    color: #f1f5f9 !important;
}

/* Input fields */
.stTextInput > div > div > input,
.stSelectbox > div > div,
.stTextArea textarea,
.stNumberInput input {
    background-color: #334155 !important;
    color: white !important;
    border-color: #475569 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.5rem 1rem !important;
    font-weight: 600 !important;
}
.stButton button:hover {
    opacity: 0.9 !important;
    transform: scale(1.02) !important;
}

/* Dataframes */
.dataframe, .stDataFrame {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
}
.dataframe th {
    background-color: #334155 !important;
    color: white !important;
}
.dataframe td {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 1rem;
    background-color: #1e293b;
    border-radius: 20px;
    padding: 0.5rem;
}
.stTabs [data-baseweb="tab"] {
    background-color: #334155;
    color: white;
    border-radius: 12px;
    padding: 0.5rem 1rem;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4);
    color: white;
}

/* Plotly charts - background override */
.js-plotly-plot .plotly .main-svg {
    background-color: #1e293b !important;
}
.js-plotly-plot .plotly .bg {
    fill: #1e293b !important;
}

/* Metrics and other text */
[data-testid="stMetricLabel"], [data-testid="stMetricValue"] {
    color: #f1f5f9 !important;
}
</style>
""", unsafe_allow_html=True)

# Set default Plotly template to dark
pio.templates.default = "plotly_dark"

# ========================== HELPER FUNCTIONS ==========================
def sec_header(tag, title, sub=""):
    st.markdown(f"""
    <div class="nx-header">
        <span class="nx-tag">{tag}</span>
        <span class="nx-title">{title}</span>
        <span style="margin-left:auto; font-size:0.7rem; color:#94a3b8;">{sub}</span>
    </div>""", unsafe_allow_html=True)

def fmt_num(n, prefix="", suffix="", decimals=1):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)):
            return "N/A"
        n = float(n)
        if abs(n) >= 1e9:
            return f"{prefix}{n/1e9:.{decimals}f}B{suffix}"
        if abs(n) >= 1e6:
            return f"{prefix}{n/1e6:.{decimals}f}M{suffix}"
        if abs(n) >= 1e3:
            return f"{prefix}{n/1e3:.{decimals}f}K{suffix}"
        return f"{prefix}{n:.{decimals}f}{suffix}"
    except Exception:
        return "N/A"

# ========================== AI API FUNCTIONS (DEEPSEEK + GROQ + CUSTOM) ==========================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_deepseek_api(messages, api_key, max_tokens=2000, temperature=0.7):
    if not api_key:
        return None, "DeepSeek API key not configured"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        else:
            return None, f"DeepSeek Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"DeepSeek Exception: {str(e)}"

def call_groq_api(messages, api_key, max_tokens=2000, temperature=0.7):
    if not api_key:
        return None, "Groq API key not configured"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        else:
            return None, f"Groq Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"Groq Exception: {str(e)}"

def call_custom_ai_api(messages, api_url, api_key, model, max_tokens=2000, temperature=0.7):
    if not api_url or not api_key:
        return None, "Custom AI endpoint or API key not configured"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        else:
            return None, f"Custom AI Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"Custom AI Exception: {str(e)}"

def get_ai_response(messages, provider, deepseek_key, groq_key, custom_url, custom_key, custom_model, max_tokens=2000, temperature=0.7):
    """Try provider first, fallback to others if available."""
    if provider == "deepseek":
        response, error = call_deepseek_api(messages, deepseek_key, max_tokens, temperature)
        if response:
            return response, None
        # Fallback to groq if available
        if groq_key:
            response2, error2 = call_groq_api(messages, groq_key, max_tokens, temperature)
            if response2:
                return response2, None
            return None, f"DeepSeek failed: {error}. Groq failed: {error2}"
        # Fallback to custom if enabled
        if custom_url and custom_key:
            response3, error3 = call_custom_ai_api(messages, custom_url, custom_key, custom_model, max_tokens, temperature)
            if response3:
                return response3, None
            return None, f"DeepSeek failed: {error}. Custom AI failed: {error3}"
        return None, error
    elif provider == "groq":
        response, error = call_groq_api(messages, groq_key, max_tokens, temperature)
        if response:
            return response, None
        if deepseek_key:
            response2, error2 = call_deepseek_api(messages, deepseek_key, max_tokens, temperature)
            if response2:
                return response2, None
            return None, f"Groq failed: {error}. DeepSeek failed: {error2}"
        if custom_url and custom_key:
            response3, error3 = call_custom_ai_api(messages, custom_url, custom_key, custom_model, max_tokens, temperature)
            if response3:
                return response3, None
            return None, f"Groq failed: {error}. Custom AI failed: {error3}"
        return None, error
    elif provider == "custom":
        response, error = call_custom_ai_api(messages, custom_url, custom_key, custom_model, max_tokens, temperature)
        if response:
            return response, None
        if deepseek_key:
            response2, error2 = call_deepseek_api(messages, deepseek_key, max_tokens, temperature)
            if response2:
                return response2, None
            return None, f"Custom AI failed: {error}. DeepSeek failed: {error2}"
        if groq_key:
            response3, error3 = call_groq_api(messages, groq_key, max_tokens, temperature)
            if response3:
                return response3, None
            return None, f"Custom AI failed: {error}. Groq failed: {error3}"
        return None, error
    else:
        return None, f"Unknown provider: {provider}"

def get_data_context(df):
    if df is None or len(df) == 0:
        return "No data currently loaded."
    num_rows = len(df)
    num_cols = len(df.columns)
    columns_list = list(df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_stats = {}
    for col in numeric_cols[:10]:
        try:
            numeric_stats[col] = {
                "min": float(df[col].min()) if not pd.isna(df[col].min()) else None,
                "max": float(df[col].max()) if not pd.isna(df[col].max()) else None,
                "mean": float(df[col].mean()) if not pd.isna(df[col].mean()) else None,
                "sum": float(df[col].sum()) if not pd.isna(df[col].sum()) else None
            }
        except:
            pass
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_samples = {}
    for col in categorical_cols[:5]:
        try:
            unique_vals = df[col].dropna().unique()[:5].tolist()
            categorical_samples[col] = unique_vals
        except:
            pass
    context = f"""
Dataset Overview:
- Total rows: {num_rows:,}
- Total columns: {num_cols}
- Column names: {columns_list}

Numeric Columns Summary (first {len(numeric_stats)}):
{json.dumps(numeric_stats, indent=2, default=str)}

Categorical Columns Samples:
{json.dumps(categorical_samples, indent=2, default=str)}

First 5 rows of data:
{df.head(5).to_string()}
"""
    return context

def get_chatbot_response(user_message, provider, deepseek_key, groq_key, custom_url, custom_key, custom_model, chat_history, df=None):
    system_prompt = """You are NEXUS AI, an intelligent assistant integrated into the NEXUS Analytics Pro platform. ..."""  # (same as original)
    
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    
    if df is not None and len(df) > 0:
        data_context = get_data_context(df)
        user_content = f"""
Current Dataset Context:
{data_context}

User Question: {user_message}

Please answer based on the dataset context provided. If the question is not related to the data, ignore the data context.
"""
    else:
        user_content = user_message
    
    messages.append({"role": "user", "content": user_content})
    
    response, error = get_ai_response(messages, provider, deepseek_key, groq_key, custom_url, custom_key, custom_model)
    return response, error

# ========================== DATA FUNCTIONS ==========================
@st.cache_data(show_spinner=False)
def load_builtin_dataset():
    np.random.seed(99)
    n = 5000
    start = pd.Timestamp("2022-01-01")
    dates = [start + pd.Timedelta(days=int(x)) for x in np.sort(np.random.randint(0, 1095, n))]
    categories = np.random.choice(
        ["Electronics", "Fashion", "Home & Kitchen", "Beauty", "Sports", "Books", "Toys", "Groceries"],
        n, p=[0.22, 0.18, 0.17, 0.12, 0.11, 0.08, 0.07, 0.05]
    )
    regions = np.random.choice(["Riyadh", "Dubai", "Cairo", "Jeddah", "Kuwait City", "Doha", "Amman", "Manama"], n)
    segments = np.random.choice(["Premium", "Standard", "Economy"], n, p=[0.25, 0.5, 0.25])
    base_prices = {
        "Electronics": 1200, "Fashion": 180, "Home & Kitchen": 250,
        "Beauty": 120, "Sports": 300, "Books": 60, "Toys": 90, "Groceries": 45
    }
    sales = np.array([base_prices[c] * np.random.uniform(0.7, 2.5) for c in categories])
    discount = np.random.choice([0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3], n, p=[0.3, 0.15, 0.2, 0.15, 0.1, 0.06, 0.04])
    sales_final = sales * (1 - discount)
    profit_margin = np.where(
        categories == "Electronics", 0.12,
        np.where(categories == "Fashion", 0.35,
                 np.where(categories == "Books", 0.4, 0.22))
    )
    profit = sales_final * (profit_margin + np.random.normal(0, 0.03, n))
    qty = np.random.randint(1, 8, n)
    returns = np.random.choice([0, 1], n, p=[0.88, 0.12])
    rating = np.round(np.random.normal(4.1, 0.5, n).clip(1, 5), 1)
    shipping_days = np.random.randint(1, 7, n)
    df = pd.DataFrame({
        "Order Date": dates, "Category": categories, "Sub-Region": regions,
        "Customer Segment": segments, "Sales": np.round(sales_final, 2),
        "Profit": np.round(profit, 2), "Discount": discount,
        "Quantity": qty, "Returns": returns, "Rating": rating,
        "Shipping Days": shipping_days
    })
    return df

def detect_column_types(df):
    roles = {"date": [], "numeric": [], "categorical": [], "id": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            roles["date"].append(col)
        elif s.dtype == object:
            sample = s.dropna().head(100)
            if len(sample) > 0:
                is_date = False
                for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d', '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d'):
                    try:
                        pd.to_datetime(sample, format=fmt, errors='raise')
                        is_date = True
                        break
                    except:
                        continue
                if not is_date:
                    try:
                        if pd.to_datetime(sample, errors='coerce').notna().mean() > 0.6:
                            is_date = True
                    except:
                        pass
                if is_date:
                    roles["date"].append(col)
                elif s.nunique() < 50:
                    roles["categorical"].append(col)
                else:
                    roles["id"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            lc = col.lower()
            if lc in ["id", "row id", "index", "customer id", "order id"] or col.endswith("_id"):
                roles["id"].append(col)
            else:
                roles["numeric"].append(col)
        else:
            roles["categorical"].append(col)
    return roles

def smart_clean(df, roles, manual_date_format=None):
    df = df.copy()
    for col in roles["date"]:
        if manual_date_format:
            try:
                df[col] = pd.to_datetime(df[col], format=manual_date_format, errors='coerce')
            except:
                df[col] = pd.to_datetime(df[col], errors='coerce')
        else:
            df[col] = pd.to_datetime(df[col], errors='coerce')
        if df[col].isnull().all():
            df[col] = df[col].astype(str)
    for col in roles["numeric"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isnull().all():
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    for col in roles["categorical"]:
        mode_val = df[col].mode()
        df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown", inplace=True)
    return df

# ========================== ML FUNCTIONS ==========================
@st.cache_resource(show_spinner=False)
def train_ml_ensemble(df_json, target_col, feature_cols):
    df = pd.read_json(io.StringIO(df_json))
    X = df[feature_cols].copy()
    y = df[target_col].fillna(df[target_col].median())
    le_map = {}
    for col in X.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le
    X = X.fillna(X.median(numeric_only=True))
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)
    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=150, max_depth=5, random_state=42)
    rg = Ridge(alpha=1.0)
    en = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000)
    estimators = [("rf", rf), ("gb", gb), ("ridge", rg), ("en", en)]
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(Xs, y)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_r2 = None
    cv_mape = None
    try:
        cv_r2 = float(cross_val_score(ensemble, Xs, y, cv=tscv, scoring="r2").mean())
        mape_total = 0
        count = 0
        for tr, te in tscv.split(Xs):
            ensemble.fit(Xs[tr], y.iloc[tr])
            pred = ensemble.predict(Xs[te])
            mape_total += mean_absolute_percentage_error(y.iloc[te], np.maximum(pred, 1e-6))
            count += 1
        cv_mape = (mape_total / count) * 100
        ensemble.fit(Xs, y)
    except Exception:
        pass
    rf.fit(Xs, y)
    importances = dict(zip(X.columns, rf.feature_importances_))
    perm_imp_result = permutation_importance(rf, Xs, y, n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = dict(zip(X.columns, perm_imp_result.importances_mean))
    return ensemble, le_map, scaler, cv_r2, cv_mape, importances, perm_imp, list(X.columns)

@st.cache_data(show_spinner=False)
def build_forecast(date_series_json, value_series_json, horizon=90, freq='ME'):
    date_series = pd.Series(pd.read_json(io.StringIO(date_series_json), typ='series'))
    value_series = pd.Series(pd.read_json(io.StringIO(value_series_json), typ='series'))

    df = pd.DataFrame({"ds": pd.to_datetime(date_series), "y": value_series.values})
    # Convert freq: 'ME' -> 'M' for Prophet, else 'W'
    period_key = 'M' if freq == 'ME' else 'W'
    # Group by period and convert to timestamp
    df['period'] = df['ds'].dt.to_period(period_key)
    ts = df.groupby('period')['y'].sum().reset_index()
    ts['ds'] = ts['period'].dt.to_timestamp()
    ts = ts[['ds', 'y']].sort_values('ds').reset_index(drop=True)

    if PROPHET_AVAILABLE and len(ts) > 10:
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=(freq == 'W'), daily_seasonality=False)
            model.fit(ts)
            # Prophet uses 'M' for monthly
            prophet_freq = 'M' if freq == 'ME' else 'W'
            future = model.make_future_dataframe(periods=horizon, freq=prophet_freq)
            forecast = model.predict(future)
            forecast = forecast.tail(horizon)
            hist = ts.rename(columns={"ds": "Date", "y": "Value"})
            fcast = pd.DataFrame({
                "Date": forecast["ds"],
                "Value": forecast["yhat"],
                "Lower": forecast["yhat_lower"],
                "Upper": forecast["yhat_upper"]
            })
            return hist, fcast, "Prophet"
        except Exception:
            pass

    if STATSMODELS_AVAILABLE:
        try:
            model = ExponentialSmoothing(ts["y"], trend='add', seasonal=None, initialization_method='estimated')
            fit = model.fit()
            forecast_vals = fit.forecast(horizon)
            last_date = ts["ds"].iloc[-1]
            if freq == 'ME':
                forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)]
            else:
                forecast_dates = [last_date + pd.Timedelta(days=7*(i+1)) for i in range(horizon)]
            hist = ts.rename(columns={"ds": "Date", "y": "Value"})
            fcast = pd.DataFrame({
                "Date": forecast_dates,
                "Value": forecast_vals.values,
                "Lower": forecast_vals.values * 0.85,
                "Upper": forecast_vals.values * 1.15
            })
            return hist, fcast, "Exponential Smoothing"
        except Exception:
            pass

    return None, None, "Error"

@st.cache_data(show_spinner=False)
def run_advanced_clustering(df_json, feature_cols, method='kmeans', n_clusters=3, eps=0.5):
    df = pd.read_json(io.StringIO(df_json))
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        labels = model.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else None
        inertias = {}
        for ki in range(2, min(8, len(df))):
            km_i = KMeans(n_clusters=ki, random_state=42, n_init=10)
            km_i.fit(Xs)
            inertias[ki] = km_i.inertia_
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)
        return labels, sil, coords, inertias, pca.explained_variance_ratio_, model

    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(Xs)
        unique_labels = set(labels)
        n_clusters_found = len(unique_labels) - (1 if -1 in unique_labels else 0)
        sil = silhouette_score(Xs, labels) if n_clusters_found > 1 else None
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)
        return labels, sil, coords, None, pca.explained_variance_ratio_, model

    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else None
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)
        return labels, sil, coords, None, pca.explained_variance_ratio_, model

    return None, None, None, None, None, None

@st.cache_data(show_spinner=False)
def detect_anomalies_iforest(df_json, feature_cols, contamination=0.05):
    df = pd.read_json(io.StringIO(df_json))
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(Xs)
    anomalies = preds == -1
    return anomalies, iso_forest

def compute_rfm(df, date_col, sales_col, id_col):
    if id_col == "—" or id_col not in df.columns:
        return None
    ref = df[date_col].max()
    rfm = df.groupby(id_col).agg(
        Recency=(date_col, lambda x: (ref - x.max()).days),
        Frequency=(date_col, "count"),
        Monetary=(sales_col, "sum"),
    ).reset_index()
    try:
        rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1], duplicates='drop').astype(float)
        rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float)
        rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates='drop').astype(float)
    except Exception:
        return None
    rfm["RFM_Score"] = rfm["Recency_Score"] * 100 + rfm["Frequency_Score"] * 10 + rfm["Monetary_Score"]

    def seg(row):
        r, f, m = row["Recency_Score"], row["Frequency_Score"], row["Monetary_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 3 and f >= 3:
            return "Loyal"
        if r >= 4:
            return "Recent"
        if f >= 3:
            return "Potential"
        if r <= 2 and f <= 2:
            return "At Risk"
        return "Others"

    rfm["Segment"] = rfm.apply(seg, axis=1)
    return rfm

def market_basket_analysis(df, customer_col, product_col, min_support=0.01):
    if not MLXTEND_AVAILABLE:
        return None, None, "mlxtend not installed. Run: pip install mlxtend"
    basket = (
        df.groupby([customer_col, product_col])
        .size()
        .unstack()
        .fillna(0)
        .map(lambda x: 1 if x > 0 else 0)
    )
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if len(frequent) == 0:
        return None, None, "No frequent itemsets found. Try lowering min_support."
    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    return frequent, rules, "Success"

# ========================== LOGIN SECTION ==========================
def login_section():
    st.sidebar.markdown("### 🔐 Account")
    if st.session_state.get("logged_in", False):
        role_label = "👑 Admin" if st.session_state.get("is_admin") else "👤 User"
        st.sidebar.success(f"{role_label}: {st.session_state['user_email']}")
        if st.sidebar.button("Logout", key="logout_btn"):
            st.session_state["logged_in"] = False
            st.session_state["user_email"] = None
            st.session_state["is_admin"] = False
            st.rerun()
        return True
    else:
        with st.sidebar.expander("🔑 Login", expanded=False):
            email = st.text_input("Email", key="login_email")
            pwd = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", key="login_submit"):
                success, is_admin = verify_password(email, pwd)
                log_login_attempt(email, success)
                if success:
                    st.session_state["logged_in"] = True
                    st.session_state["user_email"] = email
                    st.session_state["is_admin"] = is_admin
                    log_system_action(email, "login", "User logged in")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        with st.sidebar.expander("📝 Register", expanded=False):
            new_email = st.text_input("Email", key="reg_email")
            new_pwd = st.text_input("Password", type="password", key="reg_password")
            if st.button("Register", key="reg_submit"):
                if not new_email or not new_pwd:
                    st.error("Please fill all fields.")
                elif register_user(new_email, new_pwd, is_admin=False):
                    st.success("✅ Account created! Please login.")
                    log_system_action(new_email, "register", "New user registered")
                else:
                    st.error("⚠️ Email already exists.")
        st.sidebar.info("💡 Guest mode: limited features (max 1000 rows).")
        return False

# ========================== CHATBOT TAB ==========================
def chatbot_tab():
    deepseek_key = get_setting("deepseek_api_key")
    groq_key = get_setting("groq_api_key")
    provider = get_setting("ai_provider", "deepseek")
    custom_url = get_setting("custom_ai_url")
    custom_key = get_setting("custom_ai_api_key")
    custom_model = get_setting("custom_ai_model")
    custom_enabled = get_setting("custom_ai_enabled") == "1"

    if custom_enabled and provider == "custom":
        # Override provider to custom if enabled
        pass

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #8b5cf6, #06b6d4); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 28px;">🚀</div>
        <h1 style="background: linear-gradient(135deg, #8b5cf6, #06b6d4); -webkit-background-clip: text; background-clip: text; color: transparent; font-size: 1.8rem;">NEXUS AI Assistant</h1>
        <p style="color: #94a3b8;">Powered by DeepSeek, Groq, or your custom AI — Ask me anything</p>
    </div>
    """, unsafe_allow_html=True)
    
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "Hello! I'm NEXUS AI. How can I help you today? You can ask about your data, platform features, or troubleshooting."
        })
    
    col1, col2 = st.columns([6, 1])
    with col2:
        if st.button("🗑️ Clear", key="clear_chat"):
            st.session_state.chat_messages = []
            st.session_state.chat_messages.append({
                "role": "assistant",
                "content": "Chat cleared! How can I help you today?"
            })
            st.rerun()
    
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <div class="avatar">👤</div>
                <div class="chat-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <div class="avatar">🤖</div>
                <div class="chat-bubble">{msg['content']}</div>
            </div>
            """, unsafe_allow_html=True)
    
    prompt = st.chat_input("Ask me about your data or the platform...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        current_df = st.session_state.get("df", None)
        
        # Check if any API key is configured
        has_key = (provider == "deepseek" and deepseek_key) or \
                  (provider == "groq" and groq_key) or \
                  (provider == "custom" and custom_enabled and custom_key)
        if not has_key:
            error_msg = "⚠️ No AI API keys configured. Please ask the administrator to set up DeepSeek, Groq, or a custom AI in the Admin Settings."
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
        else:
            with st.spinner("Thinking..."):
                response, error = get_chatbot_response(
                    prompt, provider, deepseek_key, groq_key,
                    custom_url if custom_enabled else "",
                    custom_key if custom_enabled else "",
                    custom_model if custom_enabled else "",
                    st.session_state.chat_messages[:-1], current_df
                )
                if error:
                    response_text = f"Sorry, an error occurred: {error}"
                else:
                    response_text = response
                st.session_state.chat_messages.append({"role": "assistant", "content": response_text})
                with st.chat_message("assistant"):
                    st.markdown(response_text)
        st.rerun()

# ========================== MEGA ADMIN DASHBOARD ==========================
def mega_admin_dashboard():
    # Ensure only admin can see this (already checked in main, but double-check)
    if not st.session_state.get("is_admin", False):
        st.error("Access denied. Admins only.")
        return

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #4f46e5 50%, #06b6d4 100%);
         border-radius: 24px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;">
        <h1 style="color: white; font-size: 1.6rem; font-weight: 800; background: none; margin-bottom: 0.5rem;">
            🛡️ NEXUS Admin Control Center
        </h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem;">Full system visibility · User management · Activity monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_stats()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("👥 Users", stats["total_users"])
    with c2:
        st.metric("👑 Admins", stats["total_admins"])
    with c3:
        st.metric("✅ Today", stats["today_logins"])
    with c4:
        st.metric("🔢 All-Time", stats["total_success_logins"])
    with c5:
        st.metric("❌ Failed", stats["total_failed_logins"])
    with c6:
        st.metric("📋 Actions", stats["total_actions"])

    st.markdown("---")

    admin_tabs = st.tabs([
        "📊 Overview", "👤 Users", "📋 Logs", "📋 Subs", "💰 Plans", "⚙️ Settings", "📈 Analytics"
    ])

    # ---------- Overview (unchanged) ----------
    with admin_tabs[0]:
        sec_header("OVERVIEW", "Platform Health", "Real-time insights")
        logs = get_login_logs(limit=500)
        if logs:
            df_logs = pd.DataFrame(logs, columns=["email", "success", "timestamp"])
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
            df_logs["date"] = df_logs["timestamp"].dt.date
            col1, col2 = st.columns(2)
            with col1:
                login_counts = df_logs.groupby(["date", "success"]).size().reset_index(name="count")
                login_counts["status"] = login_counts["success"].map({1: "✅ Success", 0: "❌ Failed"})
                fig = px.area(login_counts, x="date", y="count", color="status", title="Login Activity",
                              color_discrete_map={"✅ Success": "#10b981", "❌ Failed": "#ef4444"})
                fig.update_layout(height=350, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                success_rate = df_logs["success"].sum() / len(df_logs) * 100 if len(df_logs) > 0 else 0
                fig3 = go.Figure(go.Indicator(mode="gauge+number", value=round(success_rate, 1),
                                              title={"text": "Login Success Rate (%)"},
                                              gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#8b5cf6"}}))
                fig3.update_layout(height=300, template="plotly_dark")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No login data yet.")

    # ---------- User Management (unchanged) ----------
    with admin_tabs[1]:
        sec_header("USERS", "User Management", "Create · Edit · Delete · Promote")
        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users, columns=["ID", "Email", "Is Admin", "Created At", "Last Login"])
            df_users["Role"] = df_users["Is Admin"].map({1: "👑 Admin", 0: "👤 User"})
            df_users["Created At"] = pd.to_datetime(df_users["Created At"]).dt.strftime("%Y-%m-%d")
            df_users["Last Login"] = pd.to_datetime(df_users["Last Login"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("Never")
            st.dataframe(df_users[["ID", "Email", "Role", "Created At", "Last Login"]], use_container_width=True)
        
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            uid_del = st.number_input("User ID to Delete", min_value=1, step=1)
            if st.button("Delete User"):
                delete_user(uid_del)
                st.success(f"User {uid_del} deleted.")
                st.rerun()
        with col2:
            uid_toggle = st.number_input("User ID to Toggle Admin", min_value=1, step=1)
            make_ad = st.checkbox("Grant Admin?")
            if st.button("Toggle Admin"):
                toggle_admin(uid_toggle, make_ad)
                st.success(f"Admin {'granted' if make_ad else 'revoked'} for user {uid_toggle}.")
                st.rerun()
        with col3:
            uid_reset = st.number_input("User ID to Reset Password", min_value=1, step=1)
            new_pass = st.text_input("New Password", type="password")
            if st.button("Reset Password"):
                if new_pass:
                    reset_user_password(uid_reset, new_pass)
                    st.success(f"Password reset for user {uid_reset}.")
        
        st.markdown("---")
        st.markdown("#### Promote User to Admin")
        new_admin_email = st.text_input("Email to promote")
        if st.button("Promote to Admin"):
            if new_admin_email:
                success, msg = add_admin_by_email(new_admin_email)
                st.success(msg) if success else st.error(msg)
                st.rerun()
        
        st.markdown("#### Register New User")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_email = st.text_input("Email")
        with col2:
            new_passw = st.text_input("Password", type="password")
        with col3:
            is_ad = st.checkbox("Make Admin")
            if st.button("Create User"):
                if new_email and new_passw:
                    if register_user(new_email, new_passw, is_admin=is_ad):
                        st.success(f"User {new_email} created.")
                        st.rerun()
                    else:
                        st.error("Email already exists.")
                else:
                    st.error("Fill all fields.")

    # ---------- Activity Logs (unchanged) ----------
    with admin_tabs[2]:
        sec_header("LOGS", "Activity Logs", "Audit trail")
        col1, col2 = st.columns(2)
        with col1:
            logs = get_login_logs(limit=300)
            if logs:
                df_login = pd.DataFrame(logs, columns=["Email", "Success", "Timestamp"])
                df_login["Status"] = df_login["Success"].map({1: "✅ Success", 0: "❌ Failed"})
                st.dataframe(df_login[["Email", "Status", "Timestamp"]], use_container_width=True, height=400)
        with col2:
            sys_logs = get_system_logs(limit=300)
            if sys_logs:
                df_sys = pd.DataFrame(sys_logs, columns=["User", "Action", "Details", "Timestamp"])
                st.dataframe(df_sys, use_container_width=True, height=400)

    # ---------- Subscription Management (unchanged) ----------
    with admin_tabs[3]:
        sec_header("SUBSCRIPTIONS", "Manage User Plans", "Upgrade, downgrade, extend")
        subs = get_all_subscriptions()
        if subs:
            df_subs = pd.DataFrame(subs)
            if "start_date" in df_subs.columns:
                df_subs["start_date"] = pd.to_datetime(df_subs["start_date"]).dt.strftime("%Y-%m-%d")
            if "end_date" in df_subs.columns:
                df_subs["end_date"] = pd.to_datetime(df_subs["end_date"]).dt.strftime("%Y-%m-%d")
            st.dataframe(df_subs, use_container_width=True)
        
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            user_id_up = st.number_input("User ID", min_value=1, step=1, key="sub_user")
            plans = get_available_plans()
            plan_opts = {p["name"]: p["id"] for p in plans}
            plan_name = st.selectbox("Plan", list(plan_opts.keys()))
            duration = st.selectbox("Duration (months)", [1,3,6,12])
            if st.button("Upgrade User"):
                if upgrade_subscription(user_id_up, plan_opts[plan_name], duration):
                    st.success(f"User {user_id_up} upgraded to {plan_name}.")
                    st.rerun()
        with col2:
            user_id_ext = st.number_input("User ID to Extend", min_value=1, step=1, key="ext_user")
            extra = st.number_input("Extra months", min_value=1, max_value=12, value=1)
            if st.button("Extend Subscription"):
                if extend_subscription(user_id_ext, extra):
                    st.success(f"Extended by {extra} months.")
                    st.rerun()
            if st.button("Cancel Subscription (Revert to Free)"):
                cancel_subscription(user_id_ext)
                st.success(f"Subscription cancelled for user {user_id_ext}.")
                st.rerun()
        
        plan_counts, monthly_revenue = get_subscription_stats()
        st.metric("💰 Monthly Recurring Revenue (MRR)", f"${monthly_revenue:,.2f}")
        for plan, cnt in plan_counts.items():
            st.write(f"- {plan}: {cnt} users")

    # ---------- Plan Management (unchanged) ----------
    with admin_tabs[4]:
        sec_header("PLANS", "Edit Subscription Plans", "Prices, limits, features")
        all_plans = get_all_plans()
        for plan in all_plans:
            with st.expander(f"✏️ Edit {plan['name']}"):
                new_price_m = st.number_input(f"Monthly Price (${plan['name']})", value=float(plan['price_monthly']), step=1.0, key=f"pm_{plan['id']}")
                new_price_y = st.number_input(f"Yearly Price (${plan['name']})", value=float(plan['price_yearly']), step=10.0, key=f"py_{plan['id']}")
                new_rows = st.number_input(f"Max Rows", value=int(plan['max_rows']), step=1000, key=f"rows_{plan['id']}")
                new_feat = st.text_area(f"Features", value=plan['features'], height=100, key=f"feat_{plan['id']}")
                if st.button(f"Update {plan['name']}", key=f"upd_{plan['id']}"):
                    update_plan(plan['id'], new_price_m, new_price_y, new_rows, new_feat)
                    st.success(f"{plan['name']} updated.")
                    st.rerun()

    # ---------- System Settings (AI Keys + Custom AI) ----------
    with admin_tabs[5]:
        sec_header("SETTINGS", "System Configuration", "API Keys & Limits (Admin only)")
        
        st.markdown("#### 🤖 AI Provider Configuration")
        current_provider = get_setting("ai_provider", "deepseek")
        custom_enabled = get_setting("custom_ai_enabled") == "1"
        
        # Show provider options
        provider_options = ["deepseek", "groq"]
        if custom_enabled:
            provider_options.append("custom")
        provider_choice = st.selectbox("Primary AI Provider", provider_options, index=provider_options.index(current_provider) if current_provider in provider_options else 0)
        
        # DeepSeek settings
        current_deepseek = get_setting("deepseek_api_key")
        new_deepseek = st.text_input("DeepSeek API Key", type="password", value=current_deepseek, key="ds_key")
        
        # Groq settings
        current_groq = get_setting("groq_api_key")
        new_groq = st.text_input("Groq API Key (free tier)", type="password", value=current_groq, key="groq_key")
        
        # Custom AI (OpenAI-compatible) settings
        st.markdown("---")
        st.markdown("#### 🔌 Custom AI (OpenAI-compatible)")
        enable_custom = st.checkbox("Enable Custom AI", value=custom_enabled)
        custom_url = st.text_input("Custom API URL", value=get_setting("custom_ai_url"), placeholder="https://api.openai.com/v1/chat/completions")
        custom_key = st.text_input("Custom API Key", type="password", value=get_setting("custom_ai_api_key"), placeholder="sk-...")
        custom_model = st.text_input("Model Name", value=get_setting("custom_ai_model"), placeholder="gpt-3.5-turbo")
        
        if st.button("💾 Save AI Settings"):
            set_setting("deepseek_api_key", new_deepseek)
            set_setting("groq_api_key", new_groq)
            set_setting("ai_provider", provider_choice)
            set_setting("custom_ai_enabled", "1" if enable_custom else "0")
            if enable_custom:
                set_setting("custom_ai_url", custom_url)
                set_setting("custom_ai_api_key", custom_key)
                set_setting("custom_ai_model", custom_model)
            st.success("AI settings saved. Chatbot will use these keys.")
            st.rerun()
        st.caption("DeepSeek and Groq are built-in. Custom AI allows you to connect to any OpenAI-compatible endpoint (e.g., OpenAI, Azure, local LLM). The system will fallback to other providers if the primary fails.")
        
        st.markdown("---")
        st.markdown("#### 📁 Upload Limit")
        new_limit = st.number_input("Max Upload Size (MB)", min_value=100, max_value=5000, value=MAX_FILE_SIZE_MB, step=50)
        if st.button("Apply New Limit"):
            try:
                CONFIG_DIR.mkdir(exist_ok=True)
                with open(CONFIG_FILE, "w") as f:
                    f.write(f"[server]\nmaxUploadSize = {new_limit}\n")
                st.success(f"Upload limit set to {new_limit} MB. Restart app to apply.")
            except Exception as e:
                st.error(f"Error: {e}")
        
        st.markdown("---")
        if st.button("🧹 Clear All Cache"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.success("Cache cleared. Refresh page.")
        
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
        st.metric("Database Size", f"{db_size:.3f} MB")

    # ---------- Platform Analytics (unchanged) ----------
    with admin_tabs[6]:
        sec_header("ANALYTICS", "Platform Usage", "User behavior insights")
        sys_logs = get_system_logs(limit=500)
        if sys_logs:
            df_sys = pd.DataFrame(sys_logs, columns=["User", "Action", "Details", "Timestamp"])
            df_sys["Timestamp"] = pd.to_datetime(df_sys["Timestamp"])
            df_sys["date"] = df_sys["Timestamp"].dt.date
            col1, col2 = st.columns(2)
            with col1:
                action_counts = df_sys["Action"].value_counts().reset_index()
                action_counts.columns = ["Action", "Count"]
                fig = px.pie(action_counts, names="Action", values="Count", title="Action Distribution")
                fig.update_layout(template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                daily = df_sys.groupby("date").size().reset_index(name="actions")
                fig2 = px.bar(daily, x="date", y="actions", title="Daily Activity")
                fig2.update_layout(template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
            top_users = df_sys.groupby("User").size().reset_index(name="actions").sort_values("actions", ascending=False).head(10)
            fig3 = px.bar(top_users, x="User", y="actions", title="Most Active Users")
            fig3.update_layout(template="plotly_dark")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No analytics data yet.")

# ========================== SUBSCRIPTION PLANS TAB ==========================
def subscription_plans_tab():
    sec_header("PLANS", "Choose Your Plan", "Upgrade for full features")
    if not st.session_state.get("logged_in", False):
        st.info("Please login to view and subscribe to plans.")
        return
    user_email = st.session_state["user_email"]
    user = get_user_by_email(user_email)
    if not user:
        st.error("User not found.")
        return
    current_sub = get_user_subscription(user["id"])
    if current_sub:
        st.info(f"**Current Plan:** {current_sub['name']} | Max rows: {current_sub['max_rows']:,} | Features: {current_sub['features']}")
    
    plans = get_available_plans()
    cols = st.columns(len(plans))
    for idx, plan in enumerate(plans):
        with cols[idx]:
            st.markdown(f"""
            <div style="background: #1e293b; border-radius: 20px; padding: 1rem; box-shadow: 0 4px 12px rgba(0,0,0,0.3); text-align: center; margin-bottom: 1rem;">
                <h3 style="margin-bottom: 0.5rem; color:#f1f5f9;">{plan['name']}</h3>
                <p style="font-size: 1.5rem; font-weight: 800; color: #8b5cf6;">${plan['price_monthly']:.2f}<span style="font-size: 0.9rem;">/month</span></p>
                <p style="font-size: 0.8rem; color:#94a3b8;">or ${plan['price_yearly']:.2f}/year</p>
                <hr style="border-color:#334155;">
                <p>📊 Max rows: {plan['max_rows']:,}</p>
                <p>✨ {plan['features'][:80]}...</p>
            </div>
            """, unsafe_allow_html=True)
            if plan['name'] != current_sub['name']:
                if st.session_state.get("is_admin", False):
                    if st.button(f"Assign {plan['name']}", key=f"admin_assign_{plan['id']}"):
                        if upgrade_subscription(user["id"], plan["id"], duration_months=1, payment_method="admin_manual"):
                            st.success(f"Assigned {plan['name']} to {user_email}.")
                            st.rerun()
                else:
                    with st.form(key=f"payment_form_{plan['id']}"):
                        st.caption("💳 Payment Simulation")
                        card = st.text_input("Card Number", placeholder="4242 4242 4242 4242", key=f"card_{plan['id']}")
                        exp = st.text_input("Expiry (MM/YY)", placeholder="12/25", key=f"exp_{plan['id']}")
                        cvv = st.text_input("CVV", type="password", placeholder="123", key=f"cvv_{plan['id']}")
                        name = st.text_input("Cardholder Name", key=f"name_{plan['id']}")
                        if st.form_submit_button(f"Subscribe - ${plan['price_monthly']}/month"):
                            if card and exp and cvv and name:
                                if upgrade_subscription(user["id"], plan["id"], duration_months=1, payment_method="simulated_card"):
                                    st.success(f"Subscribed to {plan['name']}!")
                                    st.rerun()
                                else:
                                    st.error("Upgrade failed.")
                            else:
                                st.error("Fill all fields.")
            else:
                st.button("Current Plan", disabled=True)

# ========================== ANALYTICS APP ==========================
def render_analytics_app():
    if "df" not in st.session_state:
        st.session_state["df"] = None
        st.session_state["roles"] = None
        st.session_state["source"] = None
        st.session_state["col_map"] = {}

    user_plan = None
    if st.session_state.get("logged_in", False):
        user = get_user_by_email(st.session_state["user_email"])
        if user:
            user_plan = get_user_subscription(user["id"])
    
    with st.sidebar:
        st.markdown("## 🚀 NEXUS Analytics")
        if st.session_state.get("logged_in", False):
            st.markdown(f"👤 **{st.session_state['user_email']}**")
            if user_plan:
                st.caption(f"📋 Plan: {user_plan['name']} (Max rows: {user_plan['max_rows']:,})")
            else:
                st.caption("📋 Plan: Free (Max rows: 1000)")
        else:
            st.markdown("👤 **Guest Mode** (Limited to 1000 rows)")
        st.markdown("---")
        st.markdown("### 📂 Data Source")
        source = st.radio("", ["📦 Built-in Dataset", "📂 Upload File"], label_visibility="collapsed", key="data_source")

        if source == "📂 Upload File":
            uploaded = st.file_uploader(f"CSV / Excel / JSON (Max {MAX_FILE_SIZE_MB} MB)", type=["csv", "xlsx", "xls", "json"], key="file_upload")
            if uploaded:
                if uploaded.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"File exceeds {MAX_FILE_SIZE_MB} MB.")
                else:
                    progress_bar = st.progress(0)
                    try:
                        if uploaded.name.endswith('.csv'):
                            try:
                                df_new, used_enc = read_csv_with_encoding(uploaded)
                                st.success(f"Loaded ({used_enc})")
                            except:
                                manual_enc = st.selectbox("Encoding", ['utf-8', 'windows-1256', 'iso-8859-1', 'cp1252'])
                                uploaded.seek(0)
                                df_new = pd.read_csv(uploaded, encoding=manual_enc)
                                st.success(f"Loaded ({manual_enc})")
                        elif uploaded.name.endswith('.json'):
                            df_new = pd.read_json(uploaded)
                        else:
                            df_new = pd.read_excel(uploaded)
                        progress_bar.progress(100)
                        # Enforce row limit based on plan or guest
                        max_allowed = user_plan['max_rows'] if user_plan else GUEST_MAX_ROWS
                        if len(df_new) > max_allowed:
                            st.error(f"Dataset has {len(df_new):,} rows, but your plan allows only {max_allowed:,}. Upgrade or use smaller file.")
                        else:
                            if st.session_state["source"] != uploaded.name:
                                st.session_state["df_raw"] = df_new
                                st.session_state["roles"] = detect_column_types(df_new)
                                st.session_state["df"] = smart_clean(df_new, st.session_state["roles"])
                                st.session_state["source"] = uploaded.name
                                st.session_state["col_map"] = {}
                                log_system_action(st.session_state.get("user_email", "guest"), "upload_file", f"Uploaded {uploaded.name}")
                            st.success(f"Loaded {uploaded.name}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        progress_bar.empty()
        else:
            if st.session_state["source"] != "builtin":
                df_bi = load_builtin_dataset()
                max_allowed = user_plan['max_rows'] if user_plan else GUEST_MAX_ROWS
                if len(df_bi) > max_allowed:
                    st.error(f"Built-in dataset has {len(df_bi):,} rows, but your plan allows only {max_allowed:,}. Upgrade or use a smaller file.")
                else:
                    st.session_state["df_raw"] = df_bi
                    st.session_state["roles"] = detect_column_types(df_bi)
                    st.session_state["df"] = smart_clean(df_bi, st.session_state["roles"])
                    st.session_state["source"] = "builtin"
                    st.session_state["col_map"] = {}
                    log_system_action(st.session_state.get("user_email", "guest"), "load_builtin", "Loaded built-in dataset")
                st.success("Built-in dataset ready")

        df = st.session_state.get("df")
        if df is not None:
            st.markdown("---")
            st.markdown("### 🗂️ Column Mapping")
            roles = st.session_state["roles"]
            num_c = ["—"] + roles["numeric"]
            date_c = ["—"] + roles["date"]
            cat_c = ["—"] + roles["categorical"]
            id_c = ["—"] + roles["id"] + roles["categorical"]
            cm = st.session_state["col_map"]

            def safe_index(lst, val):
                return lst.index(val) if val in lst else 0

            cm["sales"] = st.selectbox("💰 Sales", num_c, index=safe_index(num_c, cm.get("sales", "—")), key="map_sales")
            cm["profit"] = st.selectbox("📈 Profit", num_c, index=safe_index(num_c, cm.get("profit", "—")), key="map_profit")
            cm["date"] = st.selectbox("📅 Date", date_c, index=safe_index(date_c, cm.get("date", "—")), key="map_date")
            cm["category"] = st.selectbox("🏷️ Category", cat_c, index=safe_index(cat_c, cm.get("category", "—")), key="map_cat")
            cm["customer"] = st.selectbox("👤 Customer ID", id_c, index=safe_index(id_c, cm.get("customer", "—")), key="map_cust")
            cm["product"] = st.selectbox("📦 Product (Basket)", cat_c, index=safe_index(cat_c, cm.get("product", "—")), key="map_prod")
            st.session_state["col_map"] = cm

    df = st.session_state.get("df")
    if df is None:
        st.info("👈 Please load data from the sidebar to get started.")
        return

    is_pro_or_enterprise = False
    if user_plan and user_plan['name'] in ['Pro', 'Enterprise']:
        is_pro_or_enterprise = True
    elif not st.session_state.get("logged_in", False):
        is_pro_or_enterprise = False

    tabs = st.tabs([
        "📊 Data", "💰 KPIs", "🔮 Forecast", "🤖 Optimizer",
        "👥 Segments", "🛒 Basket", "📈 Advanced", "📄 Report", "💎 Plans", "💬 AI"
    ])

    # ---------- Data Hub ----------
    with tabs[0]:
        sec_header("00", "Data Hub", "Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{df.shape[1]}")
        col3.metric("Missing", f"{df.isnull().sum().sum():,}")
        col4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        st.dataframe(df.head(100), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.json({k: len(v) for k, v in st.session_state["roles"].items()})
        with col2:
            with st.expander("Stats"):
                st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ---------- KPIs ----------
    with tabs[1]:
        sec_header("01", "Key Performance Indicators", "Revenue & Profit")
        sales_col = st.session_state["col_map"].get("sales", "—")
        profit_col = st.session_state["col_map"].get("profit", "—")
        date_col = st.session_state["col_map"].get("date", "—")
        cat_col = st.session_state["col_map"].get("category", "—")
        if sales_col != "—" and sales_col in df.columns:
            total_rev = df[sales_col].sum()
            total_profit = df[profit_col].sum() if profit_col != "—" and profit_col in df.columns else None
            margin = (total_profit / total_rev * 100) if total_profit and total_rev else None
            avg_order = df[sales_col].mean()
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("💰 Total Revenue", fmt_num(total_rev, prefix="$"))
            if total_profit:
                c2.metric("📈 Total Profit", fmt_num(total_profit, prefix="$"))
            if margin:
                c3.metric("📊 Profit Margin", f"{margin:.1f}%")
            c4.metric("🛒 Avg Order", fmt_num(avg_order, prefix="$"))
            if date_col != "—" and date_col in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    df_ts = df.set_index(date_col).resample('ME')[sales_col].sum().reset_index()
                    fig = px.line(df_ts, x=date_col, y=sales_col, title="Monthly Sales", markers=True)
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if cat_col != "—" and cat_col in df.columns:
                        cat_sales = df.groupby(cat_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
                        fig2 = px.bar(cat_sales, x=cat_col, y=sales_col, title="Sales by Category")
                        fig2.update_layout(template="plotly_dark")
                        st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Map a Sales column in sidebar.")

    # ---------- Forecasting ----------
    with tabs[2]:
        sec_header("02", "Demand Forecasting", "AI-powered prediction")
        if not is_pro_or_enterprise:
            st.warning("🔒 Forecasting is available in Pro and Enterprise plans only.")
        else:
            sales_col = st.session_state["col_map"].get("sales", "—")
            date_col = st.session_state["col_map"].get("date", "—")
            if sales_col != "—" and date_col != "—" and sales_col in df.columns and date_col in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    horizon = st.slider("Horizon (periods)", 3, 36, 12)
                with col2:
                    freq_label = st.selectbox("Aggregation", ["Monthly", "Weekly"])
                freq_key = 'ME' if freq_label == "Monthly" else 'W'
                if st.button("Run Forecast"):
                    with st.spinner("Building forecast..."):
                        try:
                            date_json = df[date_col].astype(str).to_json()
                            val_json = df[sales_col].to_json()
                            hist, fcast, model_name = build_forecast(date_json, val_json, horizon, freq_key)
                            if hist is None:
                                st.error("Failed to build forecast. Install statsmodels or prophet.")
                            else:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Value"], name="Historical", line=dict(color="#06b6d4")))
                                fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Upper"], fill=None, line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                                fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Lower"], fill="tonexty", name="Confidence", fillcolor="rgba(139,92,246,0.15)", line=dict(color="rgba(0,0,0,0)")))
                                fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Value"], name=f"Forecast ({model_name})", line=dict(color="#f59e0b", dash="dash")))
                                fig.update_layout(height=500, template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(fcast.round(2), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("Map Sales and Date columns.")

    # ---------- Profit Optimizer ----------
    with tabs[3]:
        sec_header("03", "AI Profit Optimizer", "Voting Ensemble")
        profit_col = st.session_state["col_map"].get("profit", "—")
        if profit_col != "—" and profit_col in df.columns:
            available_features = [c for c in df.columns if c != profit_col]
            features = st.multiselect("Select features", available_features, default=[])
            if features and st.button("Train Model"):
                with st.spinner("Training ensemble..."):
                    try:
                        train_df = df[features + [profit_col]].dropna()
                        if len(train_df) < 20:
                            st.error("Need at least 20 rows.")
                        else:
                            result = train_ml_ensemble(train_df.to_json(), profit_col, features)
                            _, _, _, r2, mape, imp, perm_imp, _ = result
                            c1, c2, c3 = st.columns(3)
                            c1.metric("R² Score", f"{r2:.3f}" if r2 else "N/A")
                            c2.metric("MAPE", f"{mape:.1f}%" if mape else "N/A")
                            c3.metric("Rows", f"{len(train_df):,}")
                            col1, col2 = st.columns(2)
                            with col1:
                                imp_df = pd.Series(imp).sort_values(ascending=True)
                                fig = px.bar(imp_df, orientation="h", title="Feature Importance")
                                fig.update_layout(template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                perm_df = pd.Series(perm_imp).sort_values(ascending=True)
                                fig2 = px.bar(perm_df, orientation="h", title="Permutation Importance")
                                fig2.update_layout(template="plotly_dark")
                                st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("Map a Profit column.")

    # ---------- Segmentation ----------
    with tabs[4]:
        sec_header("04", "Customer Intelligence", "RFM & Clustering")
        cust_col = st.session_state["col_map"].get("customer", "—")
        sales_col = st.session_state["col_map"].get("sales", "—")
        date_col = st.session_state["col_map"].get("date", "—")
        seg_tab1, seg_tab2 = st.tabs(["RFM Analysis", "Clustering"])
        with seg_tab1:
            if cust_col != "—" and sales_col != "—" and date_col != "—" and all(c in df.columns for c in [cust_col, sales_col, date_col]):
                if st.button("Run RFM"):
                    rfm = compute_rfm(df, date_col, sales_col, cust_col)
                    if rfm is not None:
                        st.dataframe(rfm.head(50), use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            seg_counts = rfm["Segment"].value_counts()
                            fig = px.pie(seg_counts, names=seg_counts.index, values=seg_counts.values, title="Segments")
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig2 = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment", size="RFM_Score", title="RFM Scatter")
                            fig2.update_layout(template="plotly_dark")
                            st.plotly_chart(fig2, use_container_width=True)
                    else:
                        st.warning("RFM failed.")
            else:
                st.info("Map Customer ID, Sales, and Date.")
        with seg_tab2:
            if not is_pro_or_enterprise:
                st.info("Clustering is Pro/Enterprise feature.")
            else:
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    method = st.selectbox("Method", ["kmeans", "dbscan", "hierarchical"])
                    col1, col2 = st.columns(2)
                    with col1:
                        if method in ['kmeans', 'hierarchical']:
                            k = st.slider("Clusters", 2, 8, 3)
                        else:
                            eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.05)
                    with col2:
                        feat_clust = st.multiselect("Features", num_cols, default=num_cols[:min(3, len(num_cols))])
                    if feat_clust and st.button("Run Clustering"):
                        try:
                            eps_val = eps if method == 'dbscan' else 0.5
                            labels, sil, coords, inertias, var, _ = run_advanced_clustering(
                                df[feat_clust].dropna().to_json(), feat_clust, method=method,
                                n_clusters=k, eps=eps_val
                            )
                            if labels is not None:
                                if sil:
                                    st.metric("Silhouette Score", f"{sil:.3f}")
                                fig = px.scatter(x=coords[:, 0], y=coords[:, 1], color=labels.astype(str),
                                                 title=f"PCA Projection - {method.upper()}",
                                                 labels={"x": f"PC1 ({var[0]*100:.1f}%)", "y": f"PC2 ({var[1]*100:.1f}%)"})
                                fig.update_layout(template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True)
                                if inertias:
                                    fig2 = px.line(x=list(inertias.keys()), y=list(inertias.values()), markers=True, title="Elbow Method")
                                    fig2.update_layout(template="plotly_dark")
                                    st.plotly_chart(fig2, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
                else:
                    st.info("Need at least 2 numeric columns.")

    # ---------- Market Basket ----------
    with tabs[5]:
        sec_header("05", "Market Basket", "Apriori rules")
        if not is_pro_or_enterprise:
            st.warning("🔒 Market Basket is Pro/Enterprise only.")
        elif not MLXTEND_AVAILABLE:
            st.warning("Install mlxtend: pip install mlxtend")
        else:
            cust_col = st.session_state["col_map"].get("customer", "—")
            prod_col = st.session_state["col_map"].get("product", "—")
            if cust_col != "—" and prod_col != "—" and cust_col in df.columns and prod_col in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    min_sup = st.slider("Min Support", 0.005, 0.1, 0.01, 0.005)
                with col2:
                    min_lift = st.slider("Min Lift", 1.0, 5.0, 1.0, 0.1)
                if st.button("Run Apriori"):
                    with st.spinner("Running..."):
                        try:
                            freq, rules, msg = market_basket_analysis(df, cust_col, prod_col, min_sup)
                            if rules is not None and len(rules) > 0:
                                rules_filtered = rules[rules["lift"] >= min_lift]
                                st.success(f"Found {len(rules_filtered)} rules.")
                                st.dataframe(rules_filtered[["antecedents", "consequents", "support", "confidence", "lift"]].sort_values("lift", ascending=False), use_container_width=True)
                                fig = px.scatter(rules_filtered, x="support", y="confidence", color="lift", size="lift", title="Support vs Confidence")
                                fig.update_layout(template="plotly_dark")
                                st.plotly_chart(fig, use_container_width=True)
                            else:
                                st.info(msg)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("Map Customer ID and Product columns.")

    # ---------- Advanced Analytics ----------
    with tabs[6]:
        sec_header("06", "Advanced Analytics", "Correlations, Anomalies")
        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["Correlations", "Anomaly Detection", "Data Explorer"])
        with adv_tab1:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix", aspect="auto")
                fig.update_layout(height=500, template="plotly_dark")
                st.plotly_chart(fig, use_container_width=True)
                top4 = num_cols[:min(4, len(num_cols))]
                fig2 = px.scatter_matrix(df[top4].dropna().sample(min(500, len(df))), title="Scatter Matrix")
                fig2.update_layout(height=600, template="plotly_dark")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Need 2+ numeric columns.")
        with adv_tab2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            feat_anom = st.multiselect("Features for anomaly", num_cols, default=num_cols[:min(3, len(num_cols))])
            contamination = st.slider("Contamination %", 0.01, 0.2, 0.05, 0.01)
            if feat_anom and st.button("Detect Anomalies"):
                with st.spinner("Training Isolation Forest..."):
                    try:
                        clean_df = df[feat_anom].dropna()
                        anomalies, _ = detect_anomalies_iforest(clean_df.to_json(), feat_anom, contamination)
                        n_anom = int(anomalies.sum())
                        st.metric("Anomalies Detected", n_anom)
                        st.metric("Anomaly Rate", f"{n_anom/len(anomalies)*100:.1f}%")
                        if n_anom > 0 and len(feat_anom) >= 2:
                            fig = px.scatter(x=clean_df.iloc[:,0], y=clean_df.iloc[:,1], color=np.where(anomalies, "Anomaly", "Normal"),
                                             title="Anomaly Visualization", color_discrete_map={"Anomaly":"#ef4444","Normal":"#10b981"})
                            fig.update_layout(template="plotly_dark")
                            st.plotly_chart(fig, use_container_width=True)
                        if n_anom > 0:
                            clean_idx = clean_df.index
                            anom_idx = clean_idx[anomalies]
                            st.dataframe(df.loc[anom_idx].head(50), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        with adv_tab3:
            date_col = st.session_state["col_map"].get("date", "—")
            if date_col != "—" and date_col in df.columns:
                min_d = df[date_col].min().date()
                max_d = df[date_col].max().date()
                date_range = st.date_input("Date Range", [min_d, max_d])
                if len(date_range) == 2:
                    mask = (df[date_col] >= pd.to_datetime(date_range[0])) & (df[date_col] <= pd.to_datetime(date_range[1]))
                    filtered_df = df[mask]
                    st.write(f"Showing {len(filtered_df):,} of {len(df):,} rows")
                    st.dataframe(filtered_df, use_container_width=True)
                    csv = filtered_df.to_csv(index=False)
                    st.download_button("Export CSV", csv, "filtered.csv", "text/csv")
            else:
                st.info("Map a Date column for filtering.")
                st.dataframe(df.sample(min(200, len(df))), use_container_width=True)

    # ---------- Executive Report ----------
    with tabs[7]:
        sec_header("07", "Executive Report", "AI-generated summary")
        if st.button("Generate Report"):
            sales_col = st.session_state["col_map"].get("sales", "—")
            profit_col = st.session_state["col_map"].get("profit", "—")
            cat_col = st.session_state["col_map"].get("category", "—")
            date_col = st.session_state["col_map"].get("date", "—")
            total_rev = df[sales_col].sum() if sales_col != "—" and sales_col in df.columns else 0
            total_profit = df[profit_col].sum() if profit_col != "—" and profit_col in df.columns else 0
            margin = (total_profit / total_rev * 100) if total_rev and total_rev > 0 else 0
            top_category = "N/A"
            if cat_col != "—" and cat_col in df.columns and sales_col != "—" and sales_col in df.columns:
                try:
                    top_category = df.groupby(cat_col)[sales_col].sum().idxmax()
                except:
                    pass
            date_range_str = "N/A"
            if date_col != "—" and date_col in df.columns:
                try:
                    date_range_str = f"{df[date_col].min().date()} → {df[date_col].max().date()}"
                except:
                    pass
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100
            report = f"""# NEXUS Analytics Pro Executive Report

## Dataset Overview
- Records: {len(df):,}
- Columns: {df.shape[1]}
- Date Range: {date_range_str}
- Missing Data: {missing_pct:.2f}%

## Financial Performance
- Total Revenue: {fmt_num(total_rev, prefix='$')}
- Total Profit: {fmt_num(total_profit, prefix='$')}
- Profit Margin: {margin:.1f}%
- Top Category: {top_category}

## Recommendations
1. Focus on {top_category} category.
2. Improve margin by reviewing discounts.
3. Use RFM to retain loyal customers.
4. Deploy forecasting for inventory.
5. Run anomaly detection weekly.

---
Report generated by NEXUS Analytics Pro
"""
            st.markdown(report)
            st.download_button("Download Report", report, "nexus_report.md", "text/markdown")

    # ---------- Subscription Plans ----------
    with tabs[8]:
        subscription_plans_tab()

    # ---------- AI Assistant ----------
    with tabs[9]:
        chatbot_tab()

# ========================== MAIN ==========================
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = None
        st.session_state["is_admin"] = False

    login_section()

    if st.session_state.get("is_admin", False):
        admin_mode = st.sidebar.checkbox("🔧 Admin Panel", key="admin_mode_switch")
        if admin_mode:
            mega_admin_dashboard()
        else:
            render_analytics_app()
    else:
        render_analytics_app()

if __name__ == "__main__":
    main()
