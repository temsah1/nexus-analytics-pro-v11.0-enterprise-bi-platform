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
GUEST_MAX_ROWS = 1000

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

# ========================== PAGE CONFIG WITH DARK MOBILE-FRIENDLY CSS ==========================
st.set_page_config(
    page_title="NEXUS Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# DARK THEME HIGH CONTRAST CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

html, body, .stApp {
    background: #0f172a !important;
    font-family: 'Inter', sans-serif;
    color: #f1f5f9 !important;
}

/* Sidebar dark */
[data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
}

[data-testid="stSidebar"] * {
    color: #f1f5f9 !important;
}

/* Main content background */
.main > div {
    background: #0f172a;
}

/* Headers */
h1, h2, h3, h4, h5, h6 {
    color: #38bdf8 !important;
    font-weight: 700;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #1e293b !important;
    border-radius: 20px;
    padding: 1rem;
    border: 1px solid #334155;
    box-shadow: 0 4px 12px rgba(0,0,0,0.3);
}

[data-testid="stMetric"] * {
    color: #f1f5f9 !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    color: #38bdf8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(95deg, #8b5cf6, #06b6d4) !important;
    border: none !important;
    border-radius: 60px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(6,182,212,0.3);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    transform: scale(1.02);
    background: linear-gradient(95deg, #7c3aed, #0891b2) !important;
    color: white !important;
}

/* Text inputs, textareas, selects */
input, textarea, select, [data-baseweb="input"], [data-baseweb="textarea"], [data-baseweb="select"] {
    background-color: #1e293b !important;
    color: #f1f5f9 !important;
    border: 1px solid #475569 !important;
    border-radius: 12px !important;
    padding: 0.6rem !important;
}

input::placeholder, textarea::placeholder {
    color: #94a3b8 !important;
}

/* Dataframe tables */
.dataframe, .stDataFrame, [data-testid="stDataFrame"] {
    background: #1e293b !important;
    color: #f1f5f9 !important;
    border-radius: 12px;
    overflow-x: auto;
}

.dataframe th, .stDataFrame th {
    background: #334155 !important;
    color: #f1f5f9 !important;
    font-weight: 600;
}

.dataframe td, .stDataFrame td {
    color: #e2e8f0 !important;
    border-bottom: 1px solid #334155;
}

/* Expander */
.streamlit-expanderHeader {
    background: #1e293b !important;
    color: #38bdf8 !important;
    border-radius: 12px;
}

.streamlit-expanderContent {
    background: #0f172a !important;
    color: #f1f5f9 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.5rem;
    background: #1e293b;
    border-radius: 12px;
    padding: 0.5rem;
}

.stTabs [data-baseweb="tab"] {
    background: #334155 !important;
    border-radius: 30px !important;
    padding: 0.5rem 1rem !important;
    color: #f1f5f9 !important;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4) !important;
    color: white !important;
}

/* Chat messages */
.chat-message {
    display: flex;
    gap: 12px;
    margin: 16px 0;
}

.chat-message.user {
    justify-content: flex-end;
}

.chat-message.assistant {
    justify-content: flex-start;
}

.chat-bubble {
    max-width: 75%;
    padding: 12px 16px;
    border-radius: 24px;
    font-size: 0.95rem;
    line-height: 1.5;
    word-wrap: break-word;
}

.user .chat-bubble {
    background: linear-gradient(135deg, #8b5cf6, #06b6d4);
    color: white;
    border-bottom-right-radius: 4px;
}

.assistant .chat-bubble {
    background: #1e293b;
    border: 1px solid #475569;
    color: #f1f5f9;
    border-bottom-left-radius: 4px;
}

.avatar {
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    flex-shrink: 0;
}

.user .avatar {
    background: #8b5cf6;
    color: white;
    order: 1;
}

.assistant .avatar {
    background: #06b6d4;
    color: white;
}

/* Info, warning, error boxes */
.stAlert {
    background: #1e293b !important;
    border-left: 4px solid #38bdf8 !important;
    color: #f1f5f9 !important;
}

.stAlert p {
    color: #f1f5f9 !important;
}

/* Selectbox dropdown menu */
[data-baseweb="popover"] {
    background: #1e293b !important;
    border: 1px solid #475569;
}

[data-baseweb="menu"] div {
    color: #f1f5f9 !important;
    background: #1e293b !important;
}

[data-baseweb="menu"] div:hover {
    background: #334155 !important;
}

/* Mobile responsiveness */
@media (max-width: 768px) {
    .stColumns {
        flex-direction: column !important;
    }
    .stColumns > div {
        width: 100% !important;
        margin-bottom: 1rem;
    }
    [data-testid="stMetric"] {
        padding: 0.8rem !important;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
    }
    h1, h2, h3 {
        font-size: 1.2rem !important;
    }
    .nx-title {
        font-size: 1.1rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        flex-wrap: wrap !important;
        gap: 0.25rem;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 0.3rem 0.6rem !important;
        font-size: 0.7rem !important;
    }
    .stDataFrame {
        overflow-x: auto !important;
    }
    .stButton > button {
        width: 100%;
        padding: 0.5rem;
    }
    .chat-bubble {
        max-width: 90% !important;
        font-size: 0.85rem !important;
    }
    .avatar {
        width: 28px !important;
        height: 28px !important;
        font-size: 0.8rem !important;
    }
}

/* Custom header styling */
.nx-header {
    display: flex;
    align-items: baseline;
    gap: 12px;
    margin: 1rem 0 1.2rem;
    flex-wrap: wrap;
}

.nx-tag {
    background: linear-gradient(95deg, #8b5cf6, #06b6d4);
    color: white;
    border-radius: 60px;
    padding: 0.2rem 0.8rem;
    font-size: 0.7rem;
    font-weight: 600;
}

.nx-title {
    font-size: 1.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #38bdf8, #8b5cf6);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

/* Insight boxes */
.insight-box {
    background: #1e293b;
    border-left: 5px solid #8b5cf6;
    border-radius: 20px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    color: #f1f5f9;
}

/* Admin panel cards */
.admin-kpi-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}

/* Plotly charts background fix */
.js-plotly-plot, .plotly, .plotly .main-svg {
    background: #1e293b !important;
}

/* Code blocks */
code {
    background: #334155 !important;
    color: #facc15 !important;
    padding: 0.2rem 0.4rem;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

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

# ========================== AI API FUNCTIONS (same as before) ==========================
# ... (keep all AI functions identical to your original, they remain unchanged)
# For brevity, I'm not repeating them here, but in your final code you keep them exactly as in your original.

# ========================== DATA FUNCTIONS (unchanged) ==========================
# ... (keep all data, ML, clustering functions unchanged)

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

# ========================== CHATBOT TAB (unchanged logic, only CSS affects look) ==========================
def chatbot_tab():
    deepseek_key = get_setting("deepseek_api_key")
    groq_key = get_setting("groq_api_key")
    provider = get_setting("ai_provider", "deepseek")
    custom_url = get_setting("custom_ai_url")
    custom_key = get_setting("custom_ai_api_key")
    custom_model = get_setting("custom_ai_model")
    custom_enabled = get_setting("custom_ai_enabled") == "1"

    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <div style="width: 60px; height: 60px; background: linear-gradient(135deg, #8b5cf6, #06b6d4); border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto; font-size: 28px;">🚀</div>
        <h1 style="color: #38bdf8; font-size: 1.8rem;">NEXUS AI Assistant</h1>
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

# ========================== MEGA ADMIN DASHBOARD (unchanged logic) ==========================
def mega_admin_dashboard():
    if not st.session_state.get("is_admin", False):
        st.error("Access denied. Admins only.")
        return

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #4f46e5 50%, #06b6d4 100%);
         border-radius: 24px; padding: 1.5rem; margin-bottom: 1.5rem; text-align: center;">
        <h1 style="color: white; font-size: 1.6rem; font-weight: 800; margin-bottom: 0.5rem;">
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

    # ... (rest of admin tabs remain exactly as in your original, no changes needed)

# ========================== SUBSCRIPTION PLANS TAB (unchanged) ==========================
def subscription_plans_tab():
    # ... (same as original)

# ========================== ANALYTICS APP (unchanged) ==========================
def render_analytics_app():
    # ... (same as original)

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
