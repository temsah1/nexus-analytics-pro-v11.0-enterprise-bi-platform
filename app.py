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
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024 * 5
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

# ========================== FIX #1: ENHANCED ENCODING & DATE DETECTION ==========================
def read_csv_with_encoding(uploaded_file):
    """
    قراءة ملفات CSV مع دعم كامل للترميزات المختلفة واكتشاف التواريخ تلقائياً
    """
    encodings = ['utf-8', 'windows-1256', 'iso-8859-6', 'iso-8859-1', 'cp1252', 'latin1', 'utf-8-sig']
    
    # أنماط التواريخ الشائعة للكشف
    DATE_FORMATS = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d',
        '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y',
        '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ',
        '%d-%b-%Y', '%d-%B-%Y',
    ]
    
    def try_parse_date_column(series):
        """محاولة تحويل عمود إلى تاريخ بأمان"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return series

        if pd.api.types.is_numeric_dtype(series):
            if series.dropna().mean() < 100000:  
                return None

        sample = series.dropna().astype(str).head(50)
        if len(sample) == 0:
            return None
        
        for fmt in DATE_FORMATS:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='raise')
                full_parsed = pd.to_datetime(series.astype(str), format=fmt, errors='coerce')
                success_rate = full_parsed.notna().mean()
                if success_rate > 0.7:
                    return full_parsed
            except (ValueError, TypeError):
                continue
        
        try:
            full_parsed = pd.to_datetime(series.astype(str), format='mixed', errors='coerce')
            success_rate = full_parsed.notna().mean()
            if success_rate > 0.7:
                return full_parsed
        except:
            pass
        
        return None

    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            
            for col in df.columns:
                parsed = try_parse_date_column(df[col])
                if parsed is not None:
                    df[col] = parsed
            
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
            ("Free", 0, 0, 5000, "Basic analytics, limited rows, no export, no forecasting, no market basket, no clustering", 1),
            ("Pro", 19.99, 199.99, 50000, "Full analytics, forecasting, market basket, clustering, export, priority support", 1),
            ("Enterprise", 49.99, 499.99, 999999999, "Unlimited rows, all Pro features + custom models, dedicated support, API access", 1)
        ]
        c.executemany("INSERT INTO subscription_plans (name, price_monthly, price_yearly, max_rows, features, is_active) VALUES (?,?,?,?,?,?)", plans)
    
    for key in ['deepseek_api_key', 'groq_api_key', 'custom_ai_url', 'custom_ai_api_key', 'custom_ai_model']:
        c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES (?, '')", (key,))
    c.execute("INSERT OR IGNORE INTO app_settings (key, value) VALUES ('ai_provider', 'deepseek')")
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
        c.execute("INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, 1)", (admin_email, hashed))
    
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
            last_id = c.lastrowid
            c.execute("SELECT id FROM subscription_plans WHERE name = 'Free'")
            free_plan = c.fetchone()
            if free_plan:
                c.execute("INSERT INTO user_subscriptions (user_id, plan_id, start_date, end_date, is_active) VALUES (?, ?, ?, ?, 1)",
                          (last_id, free_plan["id"], datetime.now(), datetime.now() + timedelta(days=365*100)))
            conn.commit()
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
            return False, f"User {email} not found."

def reset_user_password(user_id, new_password):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET password_hash = ? WHERE id = ?", (hash_password(new_password), user_id))
        conn.commit()

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
                conn.commit()
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
        return [dict(row) for row in rows]

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
            try:
                new_end = datetime.strptime(str(row["end_date"]), "%Y-%m-%d %H:%M:%S") + timedelta(days=30*extra_months)
            except:
                new_end = datetime.now() + timedelta(days=30*extra_months)
            c.execute("UPDATE user_subscriptions SET end_date = ? WHERE user_id = ? AND is_active = 1", (new_end, user_id))
            conn.commit()
            return True
        return False

def get_setting(key, default=""):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT value FROM app_settings WHERE key = ?", (key,))
        row = c.fetchone()
        return row["value"] if row else default

def set_setting(key, value):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("INSERT OR REPLACE INTO app_settings (key, value, updated_at) VALUES (?, ?, CURRENT_TIMESTAMP)", (key, value))
        conn.commit()

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="NEXUS Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ========================== FIX #3: ENHANCED UI CSS ==========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=Orbitron:wght@400;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ===== ROOT VARIABLES ===== */
:root {
    --bg-primary: #060b18;
    --bg-secondary: #0d1626;
    --bg-card: #111827;
    --bg-card-hover: #1a2438;
    --accent-purple: #7c3aed;
    --accent-cyan: #06b6d4;
    --accent-orange: #f97316;
    --accent-green: #10b981;
    --accent-pink: #ec4899;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --border: #1e293b;
    --border-bright: #334155;
    --glow-purple: rgba(124, 58, 237, 0.25);
    --glow-cyan: rgba(6, 182, 212, 0.25);
    --radius-sm: 10px;
    --radius-md: 16px;
    --radius-lg: 24px;
    --radius-xl: 32px;
}

/* ===== GLOBAL RESET ===== */
html, body, [data-testid="stAppViewContainer"], .main {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ===== ANIMATED BACKGROUND ===== */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: 
        radial-gradient(ellipse at 20% 20%, rgba(124, 58, 237, 0.08) 0%, transparent 50%),
        radial-gradient(ellipse at 80% 80%, rgba(6, 182, 212, 0.06) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 50%, rgba(249, 115, 22, 0.04) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #060b18 0%, #0d1626 100%) !important;
    border-right: 1px solid rgba(124, 58, 237, 0.3) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.5) !important;
}
[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Orbitron', sans-serif !important;
    font-size: 0.9rem !important;
    color: var(--accent-cyan) !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
}

/* ===== CARDS & CONTAINERS ===== */
.nx-header {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.15), rgba(6, 182, 212, 0.1)) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1.2rem 1.5rem !important;
    margin-bottom: 1.5rem !important;
    backdrop-filter: blur(10px) !important;
    display: flex;
    align-items: center;
    gap: 1rem;
}
.nx-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-cyan);
    background: rgba(6, 182, 212, 0.1);
    border: 1px solid rgba(6, 182, 212, 0.3);
    padding: 4px 10px;
    border-radius: 6px;
    letter-spacing: 1px;
}
.nx-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    background: linear-gradient(135deg, #f1f5f9, var(--accent-cyan));
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}

/* ===== METRICS ===== */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-card-hover)) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    padding: 1.5rem !important;
    transition: all 0.3s ease !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stMetric"]::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan));
}
[data-testid="stMetric"]:hover {
    border-color: rgba(124, 58, 237, 0.5) !important;
    box-shadow: 0 0 20px var(--glow-purple) !important;
    transform: translateY(-2px) !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-family: 'Orbitron', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-lg) !important;
    padding: 6px !important;
    flex-wrap: wrap !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text-secondary) !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(124, 58, 237, 0.1) !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--accent-purple), var(--accent-cyan)) !important;
    color: white !important;
    box-shadow: 0 4px 15px var(--glow-purple) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-purple), #4f46e5) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.65rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.3) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 8px !important;
}

/* ===== INPUT FIELDS ===== */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox > div > div {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1rem !important;
    transition: border-color 0.2s ease !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
    border-color: var(--accent-purple) !important;
    box-shadow: 0 0 0 2px var(--glow-purple) !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploadDropzone"] {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.05), rgba(6, 182, 212, 0.05)) !important;
    border: 2px dashed rgba(124, 58, 237, 0.4) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.3s ease !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(6, 182, 212, 0.08) !important;
}
[data-testid="stFileUploadDropzone"] * {
    color: var(--text-primary) !important;
}

/* ===== DATAFRAME ===== */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}
.dataframe {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.dataframe th {
    background: linear-gradient(135deg, var(--accent-purple), #4f46e5) !important;
    color: white !important;
    font-family: 'Space Grotesk', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-size: 0.75rem !important;
}
.dataframe td {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-bottom: 1px solid var(--border) !important;
}
.dataframe tr:hover td {
    background: var(--bg-card-hover) !important;
}

/* ===== EXPANDER ===== */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-bright) !important;
    border-radius: var(--radius-md) !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] summary {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
    padding: 1rem 1.5rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ===== SUCCESS / ERROR / WARNING / INFO ===== */
[data-testid="stAlert"] {
    border-radius: var(--radius-md) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ===== CHAT - FIX #2: PROPER MESSAGE CONTAINERS ===== */
[data-testid="stChatInput"] {
    background: var(--bg-secondary) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: var(--radius-lg) !important;
    padding: 4px !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
    border-radius: var(--radius-md) !important;
    resize: none !important;
    min-height: 52px !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin-bottom: 1rem !important;
}

[data-testid="stChatMessage"][data-testid*="user"] .stMarkdown,
.stChatMessage[aria-label*="user"] .stMarkdown {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.2), rgba(79, 70, 229, 0.15)) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md) !important;
    padding: 1rem 1.25rem !important;
    width: fit-content !important;
    max-width: 80% !important;
    margin-left: auto !important;
    word-wrap: break-word !important;
    white-space: pre-wrap !important;
    height: auto !important;
    min-height: unset !important;
}

[data-testid="stChatMessage"][data-testid*="assistant"] .stMarkdown,
.stChatMessage[aria-label*="assistant"] .stMarkdown {
    background: linear-gradient(135deg, var(--bg-card), rgba(6, 182, 212, 0.05)) !important;
    border: 1px solid rgba(6, 182, 212, 0.2) !important;
    border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px !important;
    padding: 1rem 1.25rem !important;
    width: fit-content !important;
    max-width: 85% !important;
    margin-right: auto !important;
    word-wrap: break-word !important;
    white-space: pre-wrap !important;
    height: auto !important;
    min-height: unset !important;
}

[data-testid="stChatMessage"] .stMarkdown p,
[data-testid="stChatMessage"] .stMarkdown {
    height: auto !important;
    overflow: visible !important;
    white-space: pre-wrap !important;
    word-break: break-word !important;
}

.chat-message {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 1.5rem;
    animation: fadeIn 0.3s ease;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.chat-message .avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
}
.chat-message.user {
    flex-direction: row-reverse;
}
.chat-message.user .avatar {
    background: linear-gradient(135deg, var(--accent-purple), #4f46e5);
}
.chat-message.assistant .avatar {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    border: 1px solid rgba(6, 182, 212, 0.4);
}

.chat-bubble {
    padding: 1rem 1.25rem !important;
    border-radius: var(--radius-md) !important;
    line-height: 1.7 !important;
    font-size: 0.95rem !important;
    font-family: 'Space Grotesk', sans-serif !important;
    word-wrap: break-word !important;
    word-break: break-word !important;
    white-space: pre-wrap !important;
    height: auto !important;
    min-height: unset !important;
    max-height: none !important;
    overflow: visible !important;
    max-width: 75vw;
}
.chat-message.user .chat-bubble {
    background: linear-gradient(135deg, rgba(124, 58, 237, 0.25), rgba(79, 70, 229, 0.2)) !important;
    border: 1px solid rgba(124, 58, 237, 0.4) !important;
    border-radius: var(--radius-md) var(--radius-md) 4px var(--radius-md) !important;
    color: var(--text-primary) !important;
    margin-left: auto;
}
.chat-message.assistant .chat-bubble {
    background: linear-gradient(135deg, var(--bg-card), rgba(6, 182, 212, 0.08)) !important;
    border: 1px solid rgba(6, 182, 212, 0.25) !important;
    border-radius: var(--radius-md) var(--radius-md) var(--radius-md) 4px !important;
    color: var(--text-primary) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}

/* ===== PROGRESS BAR ===== */
.stProgress > div > div {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan)) !important;
    border-radius: 999px !important;
}
.stProgress > div {
    background: var(--border) !important;
    border-radius: 999px !important;
}

/* ===== SLIDER ===== */
[data-testid="stSlider"] > div > div > div {
    background: linear-gradient(90deg, var(--accent-purple), var(--accent-cyan)) !important;
}

/* ===== CHECKBOX ===== */
[data-testid="stCheckbox"] > label > div {
    border-color: var(--accent-purple) !important;
}
[data-testid="stCheckbox"] > label > div[aria-checked="true"] {
    background: var(--accent-purple) !important;
}

/* ===== RADIO ===== */
[data-testid="stRadio"] label {
    color: var(--text-primary) !important;
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ===== SCROLLBAR ===== */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { 
    background: linear-gradient(180deg, var(--accent-purple), var(--accent-cyan));
    border-radius: 999px;
}

/* ===== DIVIDER ===== */
hr { border-color: var(--border-bright) !important; opacity: 0.4 !important; }

/* ===== CAPTION ===== */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* ===== MULTISELECT ===== */
[data-testid="stMultiSelect"] > div > div {
    background: var(--bg-card) !important;
    border-color: var(--border-bright) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="tag"] {
    background: linear-gradient(135deg, var(--accent-purple), #4f46e5) !important;
    border: none !important;
    border-radius: 6px !important;
}

/* ===== DOWNLOAD BUTTON ===== */
[data-testid="stDownloadButton"] > button {
    background: linear-gradient(135deg, var(--accent-green), #059669) !important;
    box-shadow: 0 4px 15px rgba(16, 185, 129, 0.3) !important;
}

/* ===== ADMIN PANEL SPECIFIC ===== */
.admin-hero {
    background: linear-gradient(135deg, #0d1626 0%, rgba(124, 58, 237, 0.15) 50%, rgba(6, 182, 212, 0.1) 100%);
    border: 1px solid rgba(124, 58, 237, 0.3);
    border-radius: var(--radius-xl);
    padding: 2.5rem;
    margin-bottom: 2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.admin-hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: conic-gradient(from 0deg, transparent 0deg, rgba(124, 58, 237, 0.05) 60deg, transparent 120deg);
    animation: spin 20s linear infinite;
    pointer-events: none;
}
@keyframes spin { to { transform: rotate(360deg); } }

/* ===== PLAN CARDS ===== */
.plan-card {
    background: linear-gradient(145deg, var(--bg-card), var(--bg-card-hover));
    border: 1px solid var(--border-bright);
    border-radius: var(--radius-xl);
    padding: 2rem;
    text-align: center;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.plan-card:hover {
    border-color: rgba(124, 58, 237, 0.5);
    box-shadow: 0 0 40px var(--glow-purple);
    transform: translateY(-4px);
}
.plan-card.featured {
    border-color: rgba(124, 58, 237, 0.6);
    background: linear-gradient(145deg, rgba(124, 58, 237, 0.1), rgba(6, 182, 212, 0.05));
}
.plan-card.featured::before {
    content: '⭐ POPULAR';
    position: absolute;
    top: 12px; right: 12px;
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 1px;
    color: var(--accent-orange);
    background: rgba(249, 115, 22, 0.15);
    border: 1px solid rgba(249, 115, 22, 0.3);
    padding: 3px 8px;
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)

# ========== PLOT TEMPLATE ==========
custom_dark_template = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        font=dict(color='#f1f5f9', size=12, family='Space Grotesk'),
        title_font=dict(size=16, color='#f1f5f9', family='Orbitron'),
        xaxis=dict(
            title_font=dict(size=13, color='#94a3b8'),
            tickfont=dict(size=11, color='#94a3b8'),
            gridcolor='#1e293b',
            linecolor='#334155',
            zerolinecolor='#334155'
        ),
        yaxis=dict(
            title_font=dict(size=13, color='#94a3b8'),
            tickfont=dict(size=11, color='#94a3b8'),
            gridcolor='#1e293b',
            linecolor='#334155',
            zerolinecolor='#334155'
        ),
        legend=dict(
            font=dict(size=12, color='#f1f5f9'),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='#334155'
        ),
        hoverlabel=dict(bgcolor='#0f172a', font_size=13, font_color='#f1f5f9', font_family='Space Grotesk'),
        colorway=['#f97316', '#10b981', '#3b82f6', '#f43f5e', '#8b5cf6', '#06b6d4', '#eab308', '#ec4899']
    )
)

def apply_plot_style(fig):
    fig.update_layout(template=custom_dark_template)
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='#1e293b')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='#1e293b')
    return fig

pio.templates["custom_dark"] = custom_dark_template
pio.templates.default = "custom_dark"

# ========================== HELPERS ==========================
def sec_header(tag, title, sub=""):
    st.markdown(f"""
    <div class="nx-header">
        <span class="nx-tag">{tag}</span>
        <span class="nx-title">{title}</span>
        <span style="margin-left:auto; font-size:0.75rem; color:#64748b; font-family:'JetBrains Mono',monospace;">{sub}</span>
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

# ========================== AI API ==========================
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"

def call_deepseek_api(messages, api_key, max_tokens=2000, temperature=0.7):
    if not api_key:
        return None, "DeepSeek API key not configured"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "deepseek-chat", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        return None, f"DeepSeek Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"DeepSeek Exception: {str(e)}"

def call_groq_api(messages, api_key, max_tokens=2000, temperature=0.7):
    if not api_key:
        return None, "Groq API key not configured"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": "llama-3.3-70b-versatile", "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        return None, f"Groq Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"Groq Exception: {str(e)}"

def call_custom_ai_api(messages, api_url, api_key, model, max_tokens=2000, temperature=0.7):
    if not api_url or not api_key:
        return None, "Custom AI endpoint or API key not configured"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": False}
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"], None
        return None, f"Custom AI Error {response.status_code}: {response.text[:200]}"
    except Exception as e:
        return None, f"Custom AI Exception: {str(e)}"

def get_ai_response(messages, provider, deepseek_key, groq_key, custom_url, custom_key, custom_model, max_tokens=2000, temperature=0.7):
    if provider == "deepseek":
        r, e = call_deepseek_api(messages, deepseek_key, max_tokens, temperature)
        if r: return r, None
        if groq_key:
            r2, e2 = call_groq_api(messages, groq_key, max_tokens, temperature)
            if r2: return r2, None
        return None, e
    elif provider == "groq":
        r, e = call_groq_api(messages, groq_key, max_tokens, temperature)
        if r: return r, None
        if deepseek_key:
            r2, e2 = call_deepseek_api(messages, deepseek_key, max_tokens, temperature)
            if r2: return r2, None
        return None, e
    elif provider == "custom":
        r, e = call_custom_ai_api(messages, custom_url, custom_key, custom_model, max_tokens, temperature)
        if r: return r, None
        if deepseek_key:
            r2, e2 = call_deepseek_api(messages, deepseek_key, max_tokens, temperature)
            if r2: return r2, None
        return None, e
    return None, f"Unknown provider: {provider}"

def get_data_context(df):
    if df is None or len(df) == 0:
        return "No data currently loaded."
    num_rows, num_cols = len(df), len(df.columns)
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    numeric_stats = {}
    for col in numeric_cols[:10]:
        try:
            numeric_stats[col] = {"min": float(df[col].min()), "max": float(df[col].max()), "mean": float(df[col].mean()), "sum": float(df[col].sum())}
        except: pass
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    categorical_samples = {}
    for col in categorical_cols[:5]:
        try:
            categorical_samples[col] = df[col].dropna().unique()[:5].tolist()
        except: pass
    return f"""Dataset: {num_rows:,} rows, {num_cols} columns\nColumns: {list(df.columns)}\nNumeric Stats: {json.dumps(numeric_stats, indent=2, default=str)}\nCategorical Samples: {json.dumps(categorical_samples, indent=2, default=str)}\nFirst 5 rows:\n{df.head(5).to_string()}"""

def get_chatbot_response(user_message, provider, deepseek_key, groq_key, custom_url, custom_key, custom_model, chat_history, df=None):
    system_prompt = "You are NEXUS AI, an intelligent analytics assistant. Be helpful, concise, and data-driven."
    messages = [{"role": "system", "content": system_prompt}]
    for msg in chat_history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    if df is not None and len(df) > 0:
        data_context = get_data_context(df)
        user_content = f"Dataset Context:\n{data_context}\n\nUser Question: {user_message}"
    else:
        user_content = user_message
    messages.append({"role": "user", "content": user_content})
    return get_ai_response(messages, provider, deepseek_key, groq_key, custom_url, custom_key, custom_model)

# ========================== DATA FUNCTIONS ==========================
@st.cache_data(show_spinner=False)
def load_builtin_dataset():
    np.random.seed(99)
    n = 5000
    start = pd.Timestamp("2022-01-01")
    dates = [start + pd.Timedelta(days=int(x)) for x in np.sort(np.random.randint(0, 1095, n))]
    categories = np.random.choice(["Electronics","Fashion","Home & Kitchen","Beauty","Sports","Books","Toys","Groceries"], n, p=[0.22,0.18,0.17,0.12,0.11,0.08,0.07,0.05])
    regions = np.random.choice(["Riyadh","Dubai","Cairo","Jeddah","Kuwait City","Doha","Amman","Manama"], n)
    segments = np.random.choice(["Premium","Standard","Economy"], n, p=[0.25,0.5,0.25])
    base_prices = {"Electronics":1200,"Fashion":180,"Home & Kitchen":250,"Beauty":120,"Sports":300,"Books":60,"Toys":90,"Groceries":45}
    sales = np.array([base_prices[c]*np.random.uniform(0.7,2.5) for c in categories])
    discount = np.random.choice([0,0.05,0.1,0.15,0.2,0.25,0.3], n, p=[0.3,0.15,0.2,0.15,0.1,0.06,0.04])
    sales_final = sales*(1-discount)
    profit_margin = np.where(categories=="Electronics",0.12,np.where(categories=="Fashion",0.35,np.where(categories=="Books",0.4,0.22)))
    profit = sales_final*(profit_margin+np.random.normal(0,0.03,n))
    qty = np.random.randint(1,8,n)
    returns = np.random.choice([0,1],n,p=[0.88,0.12])
    rating = np.round(np.random.normal(4.1,0.5,n).clip(1,5),1)
    shipping_days = np.random.randint(1,7,n)
    return pd.DataFrame({"Order Date":dates,"Category":categories,"Sub-Region":regions,"Customer Segment":segments,"Sales":np.round(sales_final,2),"Profit":np.round(profit,2),"Discount":discount,"Quantity":qty,"Returns":returns,"Rating":rating,"Shipping Days":shipping_days})

def detect_column_types(df):
    """FIX #1: Enhanced date detection for uploaded files"""
    roles = {"date": [], "numeric": [], "categorical": [], "id": []}
    
    DATE_FORMATS = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d',
        '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y',
        '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S', '%m/%d/%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d-%b-%Y',
    ]
    
    def is_date_column(series):
        """فحص ما إذا كان العمود يحتوي على بيانات تاريخ"""
        if pd.api.types.is_datetime64_any_dtype(series):
            return True
        if series.dtype != 'object':
            return False
        
        sample = series.dropna().head(50)
        if len(sample) == 0:
            return False
        
        for fmt in DATE_FORMATS:
            try:
                parsed = pd.to_datetime(sample, format=fmt, errors='raise')
                if len(parsed) > 0:
                    full_parsed = pd.to_datetime(series, format=fmt, errors='coerce')
                    if full_parsed.notna().mean() > 0.7:
                        return True
            except:
                continue
        
        try:
            parsed = pd.to_datetime(sample, infer_datetime_format=True, errors='coerce')
            if parsed.notna().mean() > 0.8:
                full_parsed = pd.to_datetime(series, infer_datetime_format=True, errors='coerce')
                if full_parsed.notna().mean() > 0.7:
                    return True
        except:
            pass
        
        return False
    
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            roles["date"].append(col)
        elif is_date_column(s):
            roles["date"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            lc = col.lower()
            if lc in ["id", "row id", "index", "customer id", "order id"] or col.endswith("_id"):
                roles["id"].append(col)
            else:
                roles["numeric"].append(col)
        elif s.dtype == object:
            if s.nunique() < 50:
                roles["categorical"].append(col)
            else:
                roles["id"].append(col)
        else:
            roles["categorical"].append(col)
    
    return roles

def smart_clean(df, roles, manual_date_format=None):
    """FIX #1: Enhanced cleaning with proper date conversion"""
    df = df.copy()
    
    DATE_FORMATS = [
        '%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%Y/%m/%d',
        '%d-%m-%Y', '%m-%d-%Y', '%Y%m%d',
        '%d %b %Y', '%d %B %Y', '%b %d, %Y', '%B %d, %Y',
        '%Y-%m-%d %H:%M:%S', '%d/%m/%Y %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S', '%Y-%m-%dT%H:%M:%SZ', '%d-%b-%Y',
    ]
    
    for col in roles["date"]:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        
        if manual_date_format:
            try:
                df[col] = pd.to_datetime(df[col], format=manual_date_format, errors='coerce')
                if df[col].isna().all():
                    raise ValueError("All NaT after manual format")
                continue
            except:
                pass
        
        success = False
        for fmt in DATE_FORMATS:
            try:
                parsed = pd.to_datetime(df[col], format=fmt, errors='coerce')
                if parsed.notna().mean() > 0.7:
                    df[col] = parsed
                    success = True
                    break
            except:
                continue
        
        if not success:
            try:
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            except:
                df[col] = pd.to_datetime(df[col], errors='coerce')
    
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
    ensemble = VotingRegressor(estimators=[("rf",rf),("gb",gb),("ridge",rg),("en",en)])
    ensemble.fit(Xs, y)
    tscv = TimeSeriesSplit(n_splits=3)
    cv_r2 = cv_mape = None
    try:
        cv_r2 = float(cross_val_score(ensemble, Xs, y, cv=tscv, scoring="r2").mean())
        mape_total = 0; count = 0
        for tr, te in tscv.split(Xs):
            ensemble.fit(Xs[tr], y.iloc[tr])
            pred = ensemble.predict(Xs[te])
            mape_total += mean_absolute_percentage_error(y.iloc[te], np.maximum(pred, 1e-6))
            count += 1
        cv_mape = (mape_total/count)*100
        ensemble.fit(Xs, y)
    except: pass
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
    period_key = 'M' if freq == 'ME' else 'W'
    df['period'] = df['ds'].dt.to_period(period_key)
    ts = df.groupby('period')['y'].sum().reset_index()
    ts['ds'] = ts['period'].dt.to_timestamp()
    ts = ts[['ds','y']].sort_values('ds').reset_index(drop=True)
    if PROPHET_AVAILABLE and len(ts) > 10:
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=(freq=='W'), daily_seasonality=False)
            model.fit(ts)
            prophet_freq = 'M' if freq == 'ME' else 'W'
            future = model.make_future_dataframe(periods=horizon, freq=prophet_freq)
            forecast = model.predict(future).tail(horizon)
            hist = ts.rename(columns={"ds":"Date","y":"Value"})
            fcast = pd.DataFrame({"Date":forecast["ds"],"Value":forecast["yhat"],"Lower":forecast["yhat_lower"],"Upper":forecast["yhat_upper"]})
            return hist, fcast, "Prophet"
        except: pass
    if STATSMODELS_AVAILABLE:
        try:
            model = ExponentialSmoothing(ts["y"], trend='add', seasonal=None, initialization_method='estimated')
            fit = model.fit()
            forecast_vals = fit.forecast(horizon)
            last_date = ts["ds"].iloc[-1]
            forecast_dates = [last_date + pd.DateOffset(months=i+1) for i in range(horizon)] if freq == 'ME' else [last_date + pd.Timedelta(days=7*(i+1)) for i in range(horizon)]
            hist = ts.rename(columns={"ds":"Date","y":"Value"})
            fcast = pd.DataFrame({"Date":forecast_dates,"Value":forecast_vals.values,"Lower":forecast_vals.values*0.85,"Upper":forecast_vals.values*1.15})
            return hist, fcast, "Exponential Smoothing"
        except: pass
    return None, None, "Error"

@st.cache_data(show_spinner=False)
def run_advanced_clustering(df_json, feature_cols, method='kmeans', n_clusters=3, eps=0.5):
    df = pd.read_json(io.StringIO(df_json))
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    Xs = StandardScaler().fit_transform(X)
    if method == 'kmeans':
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=15)
        labels = model.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else None
        inertias = {ki: KMeans(n_clusters=ki, random_state=42, n_init=10).fit(Xs).inertia_ for ki in range(2, min(8, len(df)))}
        coords = PCA(n_components=2).fit_transform(Xs)
        return labels, sil, coords, inertias, PCA(n_components=2).fit(Xs).explained_variance_ratio_, model
    elif method == 'dbscan':
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(Xs)
        n_cl = len(set(labels)) - (1 if -1 in labels else 0)
        sil = silhouette_score(Xs, labels) if n_cl > 1 else None
        coords = PCA(n_components=2).fit_transform(Xs)
        return labels, sil, coords, None, PCA(n_components=2).fit(Xs).explained_variance_ratio_, model
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(np.unique(labels)) > 1 else None
        coords = PCA(n_components=2).fit_transform(Xs)
        return labels, sil, coords, None, PCA(n_components=2).fit(Xs).explained_variance_ratio_, model
    return None, None, None, None, None, None

@st.cache_data(show_spinner=False)
def detect_anomalies_iforest(df_json, feature_cols, contamination=0.05):
    df = pd.read_json(io.StringIO(df_json))
    X = df[feature_cols].copy()
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    Xs = StandardScaler().fit_transform(X)
    iso_forest = IsolationForest(contamination=contamination, random_state=42)
    preds = iso_forest.fit_predict(Xs)
    return preds == -1, iso_forest

def compute_rfm(df, date_col, sales_col, id_col):
    if id_col == "—" or id_col not in df.columns:
        return None
    ref = df[date_col].max()
    rfm = df.groupby(id_col).agg(Recency=(date_col, lambda x: (ref-x.max()).days), Frequency=(date_col, "count"), Monetary=(sales_col, "sum")).reset_index()
    try:
        rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1], duplicates='drop').astype(float)
        rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
        rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
    except: return None
    rfm["RFM_Score"] = rfm["Recency_Score"]*100 + rfm["Frequency_Score"]*10 + rfm["Monetary_Score"]
    def seg(row):
        r,f,m = row["Recency_Score"], row["Frequency_Score"], row["Monetary_Score"]
        if r>=4 and f>=4 and m>=4: return "Champions"
        if r>=3 and f>=3: return "Loyal"
        if r>=4: return "Recent"
        if f>=3: return "Potential"
        if r<=2 and f<=2: return "At Risk"
        return "Others"
    rfm["Segment"] = rfm.apply(seg, axis=1)
    return rfm

def market_basket_analysis(df, customer_col, product_col, min_support=0.01):
    if not MLXTEND_AVAILABLE:
        return None, None, "mlxtend not installed."
    basket = df.groupby([customer_col, product_col]).size().unstack().fillna(0).map(lambda x: 1 if x > 0 else 0)
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if len(frequent) == 0:
        return None, None, "No frequent itemsets found."
    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    return frequent, rules, "Success"

# ========================== LOGIN ==========================
def login_section():
    st.sidebar.markdown("### 🔐 Account")
    if st.session_state.get("logged_in", False):
        role_label = "👑 Admin" if st.session_state.get("is_admin") else "👤 User"
        st.sidebar.success(f"{role_label}: {st.session_state['user_email']}")
        if st.sidebar.button("Logout", key="logout_btn"):
            for k in ["logged_in", "user_email", "is_admin"]:
                st.session_state[k] = False if k != "user_email" else None
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
                    st.session_state.update({"logged_in": True, "user_email": email, "is_admin": is_admin})
                    log_system_action(email, "login")
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        with st.sidebar.expander("📝 Register", expanded=False):
            new_email = st.text_input("Email", key="reg_email")
            new_pwd = st.text_input("Password", type="password", key="reg_password")
            if st.button("Register", key="reg_submit"):
                if not new_email or not new_pwd:
                    st.error("Please fill all fields.")
                elif register_user(new_email, new_pwd):
                    st.success("✅ Account created! Please login.")
                    log_system_action(new_email, "register")
                else:
                    st.error("⚠️ Email already exists.")
        st.sidebar.info("💡 Guest mode: limited to 1000 rows.")
        return False

# ========================== FIX #2 + #3: ENHANCED CHATBOT ==========================
def chatbot_tab():
    deepseek_key = get_setting("deepseek_api_key")
    groq_key = get_setting("groq_api_key")
    provider = get_setting("ai_provider", "deepseek")
    custom_url = get_setting("custom_ai_url")
    custom_key = get_setting("custom_ai_api_key")
    custom_model = get_setting("custom_ai_model")
    custom_enabled = get_setting("custom_ai_enabled") == "1"

    st.markdown("""
    <div style="text-align:center; margin-bottom:2rem;">
        <div style="width:80px;height:80px;background:linear-gradient(135deg,#7c3aed,#06b6d4);
                    border-radius:50%;display:flex;align-items:center;justify-content:center;
                    margin:0 auto 1rem;font-size:36px;box-shadow:0 0 40px rgba(124,58,237,0.4);">🚀</div>
        <h1 style="font-family:'Orbitron',sans-serif;font-size:2rem;font-weight:800;
                   background:linear-gradient(135deg,#7c3aed,#06b6d4);
                   -webkit-background-clip:text;background-clip:text;color:transparent;margin-bottom:0.5rem;">
            NEXUS AI Assistant
        </h1>
        <p style="color:#64748b;font-size:0.95rem;">Powered by DeepSeek · Groq · Custom AI</p>
    </div>
    """, unsafe_allow_html=True)

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role": "assistant", "content": "مرحباً! أنا NEXUS AI 🚀\n\nكيف يمكنني مساعدتك اليوم؟ يمكنك سؤالي عن بياناتك، أو ميزات المنصة، أو أي استفسار تحليلي."}]

    col1, col2 = st.columns([8, 1])
    with col2:
        if st.button("🗑️ مسح", key="clear_chat"):
            st.session_state.chat_messages = [{"role": "assistant", "content": "تم مسح المحادثة! كيف يمكنني مساعدتك؟"}]
            st.rerun()

    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt = st.chat_input("اسألني عن بياناتك أو المنصة...")
    if prompt:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        has_key = (provider == "deepseek" and deepseek_key) or \
                  (provider == "groq" and groq_key) or \
                  (provider == "custom" and custom_enabled and custom_key)

        if not has_key:
            error_msg = "⚠️ لم يتم تكوين مفاتيح AI API. يرجى التواصل مع المسؤول لإعداد DeepSeek أو Groq أو Custom AI في لوحة الإدارة."
            st.session_state.chat_messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.markdown(error_msg)
        else:
            with st.chat_message("assistant"):
                with st.spinner("جاري التفكير..."):
                    current_df = st.session_state.get("df")
                    response, error = get_chatbot_response(
                        prompt, provider, deepseek_key, groq_key,
                        custom_url if custom_enabled else "",
                        custom_key if custom_enabled else "",
                        custom_model if custom_enabled else "",
                        st.session_state.chat_messages[:-1], current_df
                    )
                    response_text = f"عذراً، حدث خطأ: {error}" if error else response
                    st.markdown(response_text)
                    st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

# ========================== ADMIN DASHBOARD ==========================
def mega_admin_dashboard():
    if not st.session_state.get("is_admin", False):
        st.error("Access denied. Admins only.")
        return

    st.markdown("""
    <div class="admin-hero">
        <h1 style="font-family:'Orbitron',sans-serif;color:white;font-size:2rem;font-weight:800;margin-bottom:0.5rem;">
            🛡️ NEXUS Control Center
        </h1>
        <p style="color:rgba(255,255,255,0.7);font-size:1rem;">Full system visibility · User management · Activity monitoring</p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_stats()
    cols = st.columns(6)
    metrics = [("👥 Users", stats["total_users"]), ("👑 Admins", stats["total_admins"]),
               ("✅ Today", stats["today_logins"]), ("🔢 All-Time", stats["total_success_logins"]),
               ("❌ Failed", stats["total_failed_logins"]), ("📋 Actions", stats["total_actions"])]
    for col, (label, val) in zip(cols, metrics):
        col.metric(label, val)

    st.markdown("---")
    admin_tabs = st.tabs(["📊 Overview","👤 Users","📋 Logs","📋 Subs","💰 Plans","⚙️ Settings","📈 Analytics"])

    with admin_tabs[0]:
        sec_header("OVERVIEW", "Platform Health", "Real-time insights")
        logs = get_login_logs(limit=500)
        if logs:
            df_logs = pd.DataFrame(logs, columns=["email","success","timestamp"])
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
            df_logs["date"] = df_logs["timestamp"].dt.date
            col1, col2 = st.columns(2)
            with col1:
                lc = df_logs.groupby(["date","success"]).size().reset_index(name="count")
                lc["status"] = lc["success"].map({1:"✅ Success",0:"❌ Failed"})
                fig = px.area(lc, x="date", y="count", color="status", title="Login Activity", color_discrete_map={"✅ Success":"#10b981","❌ Failed":"#ef4444"})
                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
            with col2:
                success_rate = df_logs["success"].sum()/len(df_logs)*100 if len(df_logs) > 0 else 0
                fig3 = go.Figure(go.Indicator(mode="gauge+number", value=round(success_rate,1), title={"text":"Login Success Rate (%)"},
                                              gauge={"axis":{"range":[0,100]},"bar":{"color":"#7c3aed"},"steps":[{"range":[0,50],"color":"#1e293b"},{"range":[50,100],"color":"#111827"}]}))
                st.plotly_chart(apply_plot_style(fig3), use_container_width=True)

    with admin_tabs[1]:
        sec_header("USERS", "User Management", "Create · Edit · Delete · Promote")
        users = get_all_users()
        if users:
            df_u = pd.DataFrame(users, columns=["ID","Email","Is Admin","Created At","Last Login"])
            df_u["Role"] = df_u["Is Admin"].map({1:"👑 Admin",0:"👤 User"})
            df_u["Created At"] = pd.to_datetime(df_u["Created At"]).dt.strftime("%Y-%m-%d")
            df_u["Last Login"] = pd.to_datetime(df_u["Last Login"], errors="coerce").dt.strftime("%Y-%m-%d").fillna("Never")
            st.dataframe(df_u[["ID","Email","Role","Created At","Last Login"]], use_container_width=True)
        st.markdown("---")
        col1,col2,col3 = st.columns(3)
        with col1:
            uid_del = st.number_input("User ID to Delete", min_value=1, step=1)
            if st.button("🗑️ Delete User"):
                delete_user(uid_del); st.success(f"User {uid_del} deleted."); st.rerun()
        with col2:
            uid_toggle = st.number_input("User ID for Admin Toggle", min_value=1, step=1)
            make_ad = st.checkbox("Grant Admin?")
            if st.button("🔄 Toggle Admin"):
                toggle_admin(uid_toggle, make_ad); st.success(f"Admin {'granted' if make_ad else 'revoked'}."); st.rerun()
        with col3:
            uid_reset = st.number_input("User ID to Reset Pwd", min_value=1, step=1)
            new_pass = st.text_input("New Password", type="password")
            if st.button("🔑 Reset Password"):
                if new_pass: reset_user_password(uid_reset, new_pass); st.success("Password reset!")

    with admin_tabs[2]:
        sec_header("LOGS", "Activity Logs", "Audit trail")
        col1,col2 = st.columns(2)
        with col1:
            logs = get_login_logs(limit=300)
            if logs:
                df_login = pd.DataFrame(logs, columns=["Email","Success","Timestamp"])
                df_login["Status"] = df_login["Success"].map({1:"✅ Success",0:"❌ Failed"})
                st.dataframe(df_login[["Email","Status","Timestamp"]], use_container_width=True, height=400)
        with col2:
            sys_logs = get_system_logs(limit=300)
            if sys_logs:
                df_sys = pd.DataFrame(sys_logs, columns=["User","Action","Details","Timestamp"])
                st.dataframe(df_sys, use_container_width=True, height=400)

    with admin_tabs[3]:
        sec_header("SUBSCRIPTIONS", "Manage User Plans")
        subs = get_all_subscriptions()
        if subs:
            df_subs = pd.DataFrame(subs)
            for dc in ["start_date","end_date"]:
                if dc in df_subs.columns:
                    df_subs[dc] = pd.to_datetime(df_subs[dc]).dt.strftime("%Y-%m-%d")
            st.dataframe(df_subs, use_container_width=True)
        st.markdown("---")
        col1,col2 = st.columns(2)
        with col1:
            uid_up = st.number_input("User ID", min_value=1, step=1, key="sub_user")
            plans = get_available_plans()
            plan_opts = {p["name"]: p["id"] for p in plans}
            plan_name = st.selectbox("Plan", list(plan_opts.keys()))
            duration = st.selectbox("Duration (months)", [1,3,6,12])
            if st.button("⬆️ Upgrade User"):
                if upgrade_subscription(uid_up, plan_opts[plan_name], duration):
                    st.success(f"User {uid_up} → {plan_name}"); st.rerun()
        with col2:
            uid_ext = st.number_input("User ID to Extend", min_value=1, step=1, key="ext_user")
            extra = st.number_input("Extra months", min_value=1, max_value=12, value=1)
            if st.button("📅 Extend"):
                if extend_subscription(uid_ext, extra): st.success(f"Extended {extra} months."); st.rerun()
            if st.button("❌ Cancel → Free"):
                cancel_subscription(uid_ext); st.success("Cancelled."); st.rerun()
        plan_counts, monthly_revenue = get_subscription_stats()
        st.metric("💰 Monthly Revenue (MRR)", f"${monthly_revenue:,.2f}")

    with admin_tabs[4]:
        sec_header("PLANS", "Edit Subscription Plans")
        for plan in get_all_plans():
            with st.expander(f"✏️ {plan['name']}"):
                c1,c2,c3 = st.columns(3)
                with c1: pm = st.number_input("Monthly $", value=float(plan['price_monthly']), key=f"pm_{plan['id']}")
                with c2: py = st.number_input("Yearly $", value=float(plan['price_yearly']), key=f"py_{plan['id']}")
                with c3: rows = st.number_input("Max Rows", value=int(plan['max_rows']), step=1000, key=f"rows_{plan['id']}")
                feat = st.text_area("Features", value=plan['features'], key=f"feat_{plan['id']}")
                if st.button(f"💾 Update {plan['name']}", key=f"upd_{plan['id']}"):
                    update_plan(plan['id'], pm, py, rows, feat); st.success("Updated!"); st.rerun()

    with admin_tabs[5]:
        sec_header("SETTINGS", "System Configuration")
        current_provider = get_setting("ai_provider", "deepseek")
        custom_enabled_val = get_setting("custom_ai_enabled") == "1"
        provider_opts = ["deepseek","groq"] + (["custom"] if custom_enabled_val else [])
        provider_choice = st.selectbox("Primary AI Provider", provider_opts, index=provider_opts.index(current_provider) if current_provider in provider_opts else 0)
        new_ds = st.text_input("DeepSeek API Key", type="password", value=get_setting("deepseek_api_key"))
        new_groq = st.text_input("Groq API Key", type="password", value=get_setting("groq_api_key"))
        st.markdown("---")
        st.markdown("#### 🔌 Custom AI (OpenAI-compatible)")
        enable_custom = st.checkbox("Enable Custom AI", value=custom_enabled_val)
        custom_url = st.text_input("Custom API URL", value=get_setting("custom_ai_url"))
        custom_key = st.text_input("Custom API Key", type="password", value=get_setting("custom_ai_api_key"))
        custom_model_val = st.text_input("Model Name", value=get_setting("custom_ai_model"))
        if st.button("💾 Save AI Settings"):
            set_setting("deepseek_api_key", new_ds)
            set_setting("groq_api_key", new_groq)
            set_setting("ai_provider", provider_choice)
            set_setting("custom_ai_enabled", "1" if enable_custom else "0")
            if enable_custom:
                set_setting("custom_ai_url", custom_url)
                set_setting("custom_ai_api_key", custom_key)
                set_setting("custom_ai_model", custom_model_val)
            st.success("AI settings saved!"); st.rerun()
        st.markdown("---")
        if st.button("🧹 Clear All Cache"):
            st.cache_data.clear(); st.cache_resource.clear(); st.success("Cache cleared.")
        db_size = os.path.getsize(DB_PATH)/(1024*1024) if os.path.exists(DB_PATH) else 0
        st.metric("Database Size", f"{db_size:.3f} MB")

    with admin_tabs[6]:
        sec_header("ANALYTICS", "Platform Usage")
        sys_logs = get_system_logs(limit=500)
        if sys_logs:
            df_sys = pd.DataFrame(sys_logs, columns=["User","Action","Details","Timestamp"])
            df_sys["Timestamp"] = pd.to_datetime(df_sys["Timestamp"])
            df_sys["date"] = df_sys["Timestamp"].dt.date
            col1,col2 = st.columns(2)
            with col1:
                ac = df_sys["Action"].value_counts().reset_index(); ac.columns = ["Action","Count"]
                fig = px.pie(ac, names="Action", values="Count", title="Action Distribution")
                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
            with col2:
                daily = df_sys.groupby("date").size().reset_index(name="actions")
                fig2 = px.bar(daily, x="date", y="actions", title="Daily Activity")
                st.plotly_chart(apply_plot_style(fig2), use_container_width=True)

# ========================== SUBSCRIPTION TAB ==========================
def subscription_plans_tab():
    sec_header("PLANS", "Choose Your Plan", "Upgrade for full features")
    if not st.session_state.get("logged_in", False):
        st.info("Please login to view and subscribe to plans.")
        return
    user = get_user_by_email(st.session_state["user_email"])
    if not user: st.error("User not found."); return
    current_sub = get_user_subscription(user["id"])
    if current_sub:
        st.info(f"**Current Plan:** {current_sub['name']} | Max rows: {current_sub['max_rows']:,}")
    plans = get_available_plans()
    cols = st.columns(len(plans))
    for idx, plan in enumerate(plans):
        with cols[idx]:
            is_featured = plan['name'] == 'Pro'
            card_class = "plan-card featured" if is_featured else "plan-card"
            st.markdown(f"""
            <div class="{card_class}">
                <h3 style="font-family:'Orbitron',sans-serif;color:#f1f5f9;font-size:1.2rem;">{plan['name']}</h3>
                <p style="font-size:2.5rem;font-weight:800;background:linear-gradient(135deg,#7c3aed,#06b6d4);-webkit-background-clip:text;background-clip:text;color:transparent;">
                    ${plan['price_monthly']:.2f}<span style="font-size:1rem;color:#94a3b8;">/mo</span>
                </p>
                <p style="color:#64748b;font-size:0.85rem;">${plan['price_yearly']:.2f}/year</p>
                <hr style="border-color:#1e293b;margin:1rem 0;">
                <p style="color:#94a3b8;font-size:0.9rem;">📊 Max: {plan['max_rows']:,} rows</p>
                <p style="color:#94a3b8;font-size:0.8rem;">{plan['features'][:80]}...</p>
            </div>
            """, unsafe_allow_html=True)
            if plan['name'] != (current_sub['name'] if current_sub else ''):
                if st.button(f"Subscribe to {plan['name']}", key=f"sub_{plan['id']}"):
                    if upgrade_subscription(user["id"], plan["id"], duration_months=1, payment_method="simulated"):
                        st.success(f"Subscribed to {plan['name']}!"); st.rerun()
            else:
                st.button("✅ Current Plan", disabled=True, key=f"current_{plan['id']}")

# ========================== ANALYTICS APP ==========================
def render_analytics_app():
    # تهيئة session_state بشكل آمن
    defaults = {
        "df": None,
        "roles": {},
        "source": None,
        "col_map": {
            "sales": "—",
            "profit": "—",
            "date": "—",
            "category": "—",
            "customer": "—",
            "product": "—"
        }
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val
    # تأكد من وجود جميع مفاتيح col_map
    for k in ["sales", "profit", "date", "category", "customer", "product"]:
        if k not in st.session_state["col_map"]:
            st.session_state["col_map"][k] = "—"

    user_plan = None
    if st.session_state.get("logged_in", False):
        user = get_user_by_email(st.session_state["user_email"])
        if user:
            user_plan = get_user_subscription(user["id"])

    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;border-bottom:1px solid #1e293b;margin-bottom:1rem;">
            <div style="font-family:'Orbitron',sans-serif;font-size:1.1rem;font-weight:800;
                        background:linear-gradient(135deg,#7c3aed,#06b6d4);
                        -webkit-background-clip:text;background-clip:text;color:transparent;">
                🚀 NEXUS Analytics
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get("logged_in"):
            plan_name = user_plan['name'] if user_plan else 'Free'
            max_rows = user_plan['max_rows'] if user_plan else 1000
            st.markdown(f"""
            <div style="background:rgba(124,58,237,0.1);border:1px solid rgba(124,58,237,0.3);
                        border-radius:12px;padding:0.75rem;margin-bottom:1rem;text-align:center;">
                <div style="font-size:0.8rem;color:#94a3b8;">👤 {st.session_state['user_email']}</div>
                <div style="font-size:0.75rem;color:#7c3aed;font-weight:600;margin-top:4px;">
                    📋 {plan_name} · {max_rows:,} rows
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:rgba(6,182,212,0.1);border:1px solid rgba(6,182,212,0.3);
                        border-radius:12px;padding:0.75rem;margin-bottom:1rem;text-align:center;">
                <div style="font-size:0.85rem;color:#06b6d4;">👤 Guest Mode</div>
                <div style="font-size:0.75rem;color:#64748b;">Max 1,000 rows</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### 📂 Data Source")
        source = st.radio("", ["📦 Built-in Dataset","📂 Upload File"], label_visibility="collapsed", key="data_source")

        if source == "📂 Upload File":
            uploaded = st.file_uploader(f"CSV / Excel / JSON", type=["csv","xlsx","xls","json"], key="file_upload")
            if uploaded:
                if uploaded.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"File exceeds {MAX_FILE_SIZE_MB} MB.")
                else:
                    try:
                        if uploaded.name.endswith('.csv'):
                            try:
                                df_new, used_enc = read_csv_with_encoding(uploaded)
                                st.success(f"✅ Loaded ({used_enc})")
                            except:
                                manual_enc = st.selectbox("Encoding", ['utf-8','windows-1256','iso-8859-1','cp1252'])
                                uploaded.seek(0)
                                df_new = pd.read_csv(uploaded, encoding=manual_enc)
                                st.success(f"✅ Loaded ({manual_enc})")
                        elif uploaded.name.endswith('.json'):
                            df_new = pd.read_json(uploaded)
                        else:
                            df_new = pd.read_excel(uploaded)

                        max_allowed = user_plan['max_rows'] if user_plan else GUEST_MAX_ROWS
                        if len(df_new) > max_allowed:
                            st.error(f"⚠️ {len(df_new):,} rows > {max_allowed:,} limit. Upgrade plan.")
                        else:
                            if st.session_state["source"] != uploaded.name:
                                roles = detect_column_types(df_new)
                                df_clean = smart_clean(df_new, roles)
                                st.session_state.update({"df_raw": df_new, "roles": roles, "df": df_clean, "source": uploaded.name})
                                # لا نعيد تعيين col_map بالكامل، فقط نحاول التخمين التلقائي
                                cm = st.session_state["col_map"].copy()
                                for col in df_new.columns:
                                    col_lower = col.lower()
                                    if cm["sales"] == "—" and col in roles["numeric"]:
                                        if any(kw in col_lower for kw in ['sales','revenue','amount','price','total']):
                                            cm["sales"] = col
                                    if cm["profit"] == "—" and col in roles["numeric"]:
                                        if any(kw in col_lower for kw in ['profit','income','earning']):
                                            cm["profit"] = col
                                    if cm["date"] == "—" and col in roles["date"]:
                                        cm["date"] = col
                                    if cm["category"] == "—" and col in roles["categorical"]:
                                        if any(kw in col_lower for kw in ['category','type','class','segment','product']):
                                            cm["category"] = col
                                    if cm["customer"] == "—" and (col in roles["id"] or col in roles["categorical"]):
                                        if any(kw in col_lower for kw in ['customer','client','user','id']):
                                            cm["customer"] = col
                                    if cm["product"] == "—" and col in roles["categorical"]:
                                        if 'product' in col_lower:
                                            cm["product"] = col
                                st.session_state["col_map"] = cm
                                log_system_action(st.session_state.get("user_email","guest"), "upload_file", uploaded.name)
                            
                            date_cols = st.session_state["roles"].get("date", [])
                            if date_cols:
                                st.success(f"📅 Date columns detected: {', '.join(date_cols)}")
                            else:
                                st.warning("⚠️ No date columns detected. Check column mapping below.")
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            if st.session_state["source"] != "builtin":
                df_bi = load_builtin_dataset()
                max_allowed = user_plan['max_rows'] if user_plan else GUEST_MAX_ROWS
                if len(df_bi) > max_allowed:
                    st.error(f"Dataset {len(df_bi):,} rows > {max_allowed:,} limit.")
                else:
                    roles = detect_column_types(df_bi)
                    df_clean = smart_clean(df_bi, roles)
                    st.session_state.update({"df_raw": df_bi, "roles": roles, "df": df_clean, "source": "builtin"})
                    # تخمين تلقائي للأعمدة للبيانات المدمجة
                    cm = st.session_state["col_map"].copy()
                    cm["sales"] = "Sales"
                    cm["profit"] = "Profit"
                    cm["date"] = "Order Date"
                    cm["category"] = "Category"
                    cm["customer"] = "Sub-Region"  # لا يوجد معرف عميل، نستخدم المنطقة كمثال
                    cm["product"] = "Category"  # نستخدم الفئة كمنتج
                    st.session_state["col_map"] = cm
                    log_system_action(st.session_state.get("user_email","guest"), "load_builtin")
                st.success("✅ Built-in dataset ready")

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
                try:
                    return lst.index(val)
                except ValueError:
                    return 0

            cm["sales"] = st.selectbox("💰 Sales", num_c, index=safe_index(num_c, cm.get("sales","—")), key="map_sales")
            cm["profit"] = st.selectbox("📈 Profit", num_c, index=safe_index(num_c, cm.get("profit","—")), key="map_profit")
            cm["date"] = st.selectbox("📅 Date", date_c, index=safe_index(date_c, cm.get("date","—")), key="map_date")
            cm["category"] = st.selectbox("🏷️ Category", cat_c, index=safe_index(cat_c, cm.get("category","—")), key="map_cat")
            cm["customer"] = st.selectbox("👤 Customer ID", id_c, index=safe_index(id_c, cm.get("customer","—")), key="map_cust")
            cm["product"] = st.selectbox("📦 Product", cat_c, index=safe_index(cat_c, cm.get("product","—")), key="map_prod")
            st.session_state["col_map"] = cm

    df = st.session_state.get("df")
    if df is None:
        st.markdown("""
        <div style="text-align:center;padding:5rem 2rem;">
            <div style="font-size:5rem;margin-bottom:1.5rem;">📊</div>
            <h2 style="font-family:'Orbitron',sans-serif;color:#7c3aed;margin-bottom:1rem;">Welcome to NEXUS Analytics Pro</h2>
            <p style="color:#64748b;font-size:1.1rem;">Load data from the sidebar to get started</p>
        </div>
        """, unsafe_allow_html=True)
        return

    is_pro = user_plan and user_plan['name'] in ['Pro', 'Enterprise']

    tabs = st.tabs(["📊 Data","💰 KPIs","🔮 Forecast","🤖 Optimizer","👥 Segments","🛒 Basket","📈 Advanced","📄 Report","💎 Plans","💬 AI"])

    # ---------- DATA HUB ----------
    with tabs[0]:
        sec_header("00", "Data Hub", "Overview & Exploration")
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("📋 Rows", f"{len(df):,}")
        c2.metric("📐 Columns", f"{df.shape[1]}")
        c3.metric("⚠️ Missing", f"{df.isnull().sum().sum():,}")
        c4.metric("💾 Memory", f"{df.memory_usage(deep=True).sum()/1024/1024:.2f} MB")
        
        roles = st.session_state["roles"]
        col1, col2 = st.columns([3,1])
        with col2:
            st.markdown("**Column Types Detected:**")
            for role, cols in roles.items():
                if cols:
                    icon = {"date":"📅","numeric":"🔢","categorical":"🏷️","id":"🔑"}.get(role,"")
                    st.markdown(f"{icon} **{role.title()}**: {', '.join(cols[:3])}{'...' if len(cols)>3 else ''}")
        with col1:
            st.dataframe(df.head(100), use_container_width=True)
        
        with st.expander("📊 Statistical Summary"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ---------- KPIs ----------
    with tabs[1]:
        sec_header("01", "Key Performance Indicators", "Revenue & Profit Analysis")
        cm = st.session_state.get("col_map", {})
        sales_col = cm.get("sales", "—")
        profit_col = cm.get("profit", "—")
        date_col = cm.get("date", "—")
        cat_col = cm.get("category", "—")
        
        if sales_col != "—" and sales_col in df.columns:
            total_rev = df[sales_col].sum()
            total_profit = df[profit_col].sum() if profit_col != "—" and profit_col in df.columns else None
            margin = (total_profit/total_rev*100) if total_profit and total_rev else None
            avg_order = df[sales_col].mean()
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("💰 Total Revenue", fmt_num(total_rev, prefix="$"))
            if total_profit: c2.metric("📈 Total Profit", fmt_num(total_profit, prefix="$"))
            if margin: c3.metric("📊 Profit Margin", f"{margin:.1f}%")
            c4.metric("🛒 Avg Order", fmt_num(avg_order, prefix="$"))
            
            if date_col != "—" and date_col in df.columns:
                col1,col2 = st.columns(2)
                with col1:
                    df_ts = df.set_index(date_col).resample('ME')[sales_col].sum().reset_index()
                    fig = px.line(df_ts, x=date_col, y=sales_col, title="Monthly Sales Trend", markers=True)
                    st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                with col2:
                    if cat_col != "—" and cat_col in df.columns:
                        cat_sales = df.groupby(cat_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
                        fig2 = px.bar(cat_sales, x=cat_col, y=sales_col, title="Sales by Category", color=sales_col, color_continuous_scale="Viridis")
                        st.plotly_chart(apply_plot_style(fig2), use_container_width=True)
        else:
            st.info("👈 قم بتعيين عمود المبيعات (Sales) من الشريط الجانبي")

    # ---------- FORECAST ----------
    with tabs[2]:
        sec_header("02", "Demand Forecasting", "AI-powered prediction")
        if not is_pro:
            st.warning("🔒 Forecasting is available in Pro and Enterprise plans only.")
        else:
            cm = st.session_state.get("col_map", {})
            sales_col = cm.get("sales", "—")
            date_col = cm.get("date", "—")
            if sales_col != "—" and date_col != "—" and all(c in df.columns for c in [sales_col, date_col]):
                c1,c2 = st.columns(2)
                with c1: horizon = st.slider("Horizon (periods)", 3, 36, 12)
                with c2: freq_label = st.selectbox("Aggregation", ["Monthly","Weekly"])
                freq_key = 'ME' if freq_label == "Monthly" else 'W'
                if st.button("🔮 Run Forecast"):
                    with st.spinner("Building AI forecast model..."):
                        try:
                            hist, fcast, model_name = build_forecast(df[date_col].astype(str).to_json(), df[sales_col].to_json(), horizon, freq_key)
                            if hist is None:
                                st.error("Failed. Install statsmodels or prophet.")
                            else:
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Value"], name="Historical", line=dict(color="#06b6d4", width=2)))
                                fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Upper"], fill=None, line=dict(color="rgba(0,0,0,0)"), showlegend=False))
                                fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Lower"], fill="tonexty", name="Confidence Band", fillcolor="rgba(124,58,237,0.2)", line=dict(color="rgba(0,0,0,0)")))
                                fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Value"], name=f"Forecast ({model_name})", line=dict(color="#f97316", width=2, dash="dash")))
                                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                                st.dataframe(fcast.round(2), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("قم بتعيين عمودي التاريخ والمبيعات من الشريط الجانبي")

    # ---------- OPTIMIZER ----------
    with tabs[3]:
        sec_header("03", "AI Profit Optimizer", "Ensemble ML Model")
        cm = st.session_state.get("col_map", {})
        profit_col = cm.get("profit", "—")
        if profit_col != "—" and profit_col in df.columns:
            available = [c for c in df.columns if c != profit_col]
            features = st.multiselect("Select Features", available, default=[])
            if features and st.button("🤖 Train Model"):
                with st.spinner("Training ensemble..."):
                    try:
                        train_df = df[features+[profit_col]].dropna()
                        if len(train_df) < 20:
                            st.error("Need at least 20 rows.")
                        else:
                            result = train_ml_ensemble(train_df.to_json(), profit_col, features)
                            _, _, _, r2, mape, imp, perm_imp, _ = result
                            c1,c2,c3 = st.columns(3)
                            c1.metric("R² Score", f"{r2:.3f}" if r2 else "N/A")
                            c2.metric("MAPE", f"{mape:.1f}%" if mape else "N/A")
                            c3.metric("Training Rows", f"{len(train_df):,}")
                            col1,col2 = st.columns(2)
                            with col1:
                                imp_df = pd.Series(imp).sort_values(ascending=True)
                                fig = px.bar(imp_df, orientation="h", title="Feature Importance (RF)")
                                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                            with col2:
                                perm_df = pd.Series(perm_imp).sort_values(ascending=True)
                                fig2 = px.bar(perm_df, orientation="h", title="Permutation Importance")
                                st.plotly_chart(apply_plot_style(fig2), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.info("قم بتعيين عمود الربح (Profit) من الشريط الجانبي")

    # ---------- SEGMENTS ----------
    with tabs[4]:
        sec_header("04", "Customer Intelligence", "RFM & Clustering")
        seg_tab1, seg_tab2 = st.tabs(["RFM Analysis","Advanced Clustering"])
        cm = st.session_state.get("col_map", {})
        with seg_tab1:
            cust_col = cm.get("customer", "—")
            sales_col = cm.get("sales", "—")
            date_col = cm.get("date", "—")
            if all(c != "—" and c in df.columns for c in [cust_col, sales_col, date_col]):
                if st.button("🔍 Run RFM Analysis"):
                    rfm = compute_rfm(df, date_col, sales_col, cust_col)
                    if rfm is not None:
                        st.dataframe(rfm.head(50), use_container_width=True)
                        col1,col2 = st.columns(2)
                        with col1:
                            seg_counts = rfm["Segment"].value_counts()
                            fig = px.pie(seg_counts, names=seg_counts.index, values=seg_counts.values, title="Customer Segments")
                            st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                        with col2:
                            fig2 = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment", size="RFM_Score", title="RFM Scatter")
                            st.plotly_chart(apply_plot_style(fig2), use_container_width=True)
            else:
                st.info("قم بتعيين معرف العميل والتاريخ والمبيعات من الشريط الجانبي")
        with seg_tab2:
            if not is_pro:
                st.info("🔒 Clustering is a Pro/Enterprise feature.")
            else:
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if len(num_cols) >= 2:
                    method = st.selectbox("Clustering Method", ["kmeans","dbscan","hierarchical"])
                    c1,c2 = st.columns(2)
                    with c1:
                        if method in ['kmeans','hierarchical']: k = st.slider("Clusters", 2, 8, 3)
                        else: eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.05)
                    with c2:
                        feat_clust = st.multiselect("Features", num_cols, default=num_cols[:min(3,len(num_cols))])
                    if feat_clust and st.button("🎯 Run Clustering"):
                        try:
                            eps_val = eps if method == 'dbscan' else 0.5
                            labels, sil, coords, inertias, var, _ = run_advanced_clustering(df[feat_clust].dropna().to_json(), feat_clust, method=method, n_clusters=k if method != 'dbscan' else 3, eps=eps_val)
                            if labels is not None:
                                if sil: st.metric("Silhouette Score", f"{sil:.3f}")
                                fig = px.scatter(x=coords[:,0], y=coords[:,1], color=labels.astype(str),
                                                 title=f"PCA Projection - {method.upper()}",
                                                 labels={"x":f"PC1 ({var[0]*100:.1f}%)","y":f"PC2 ({var[1]*100:.1f}%)"})
                                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                                if inertias:
                                    fig2 = px.line(x=list(inertias.keys()), y=list(inertias.values()), markers=True, title="Elbow Method")
                                    st.plotly_chart(apply_plot_style(fig2), use_container_width=True)
                        except Exception as e:
                            st.error(f"Error: {e}")

    # ---------- BASKET ----------
    with tabs[5]:
        sec_header("05", "Market Basket Analysis", "Apriori Association Rules")
        if not is_pro:
            st.warning("🔒 Market Basket is a Pro/Enterprise feature.")
        elif not MLXTEND_AVAILABLE:
            st.warning("Install mlxtend: pip install mlxtend")
        else:
            cm = st.session_state.get("col_map", {})
            cust_col = cm.get("customer", "—")
            prod_col = cm.get("product", "—")
            if cust_col != "—" and prod_col != "—" and all(c in df.columns for c in [cust_col, prod_col]):
                c1,c2 = st.columns(2)
                with c1: min_sup = st.slider("Min Support", 0.005, 0.1, 0.01, 0.005)
                with c2: min_lift = st.slider("Min Lift", 1.0, 5.0, 1.0, 0.1)
                if st.button("🛒 Run Apriori"):
                    with st.spinner("Mining association rules..."):
                        try:
                            freq, rules, msg = market_basket_analysis(df, cust_col, prod_col, min_sup)
                            if rules is not None and len(rules) > 0:
                                rules_f = rules[rules["lift"] >= min_lift]
                                st.success(f"Found {len(rules_f)} rules.")
                                st.dataframe(rules_f[["antecedents","consequents","support","confidence","lift"]].sort_values("lift", ascending=False), use_container_width=True)
                                fig = px.scatter(rules_f, x="support", y="confidence", color="lift", size="lift", title="Support vs Confidence")
                                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                            else:
                                st.info(msg)
                        except Exception as e:
                            st.error(f"Error: {e}")
            else:
                st.info("قم بتعيين معرف العميل والمنتج من الشريط الجانبي")

    # ---------- ADVANCED ----------
    with tabs[6]:
        sec_header("06", "Advanced Analytics", "Correlations, Anomalies & Exploration")
        adv1, adv2, adv3 = st.tabs(["Correlations","Anomaly Detection","Data Explorer"])
        with adv1:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix", aspect="auto")
                st.plotly_chart(apply_plot_style(fig), use_container_width=True)
        with adv2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            feat_anom = st.multiselect("Features for Anomaly Detection", num_cols, default=num_cols[:min(3,len(num_cols))])
            contamination = st.slider("Contamination %", 0.01, 0.2, 0.05, 0.01)
            if feat_anom and st.button("🔍 Detect Anomalies"):
                with st.spinner("Training Isolation Forest..."):
                    try:
                        clean_df = df[feat_anom].dropna()
                        anomalies, _ = detect_anomalies_iforest(clean_df.to_json(), feat_anom, contamination)
                        n_anom = int(anomalies.sum())
                        c1,c2 = st.columns(2)
                        c1.metric("Anomalies Detected", n_anom)
                        c2.metric("Anomaly Rate", f"{n_anom/len(anomalies)*100:.1f}%")
                        if n_anom > 0 and len(feat_anom) >= 2:
                            fig = px.scatter(x=clean_df.iloc[:,0], y=clean_df.iloc[:,1], color=np.where(anomalies,"Anomaly","Normal"),
                                             title="Anomaly Visualization", color_discrete_map={"Anomaly":"#ef4444","Normal":"#10b981"})
                            st.plotly_chart(apply_plot_style(fig), use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        with adv3:
            cm = st.session_state.get("col_map", {})
            date_col = cm.get("date", "—")
            if date_col != "—" and date_col in df.columns:
                min_d, max_d = df[date_col].min().date(), df[date_col].max().date()
                date_range = st.date_input("Date Range", [min_d, max_d])
                if len(date_range) == 2:
                    mask = (df[date_col] >= pd.to_datetime(date_range[0])) & (df[date_col] <= pd.to_datetime(date_range[1]))
                    filtered = df[mask]
                    st.write(f"Showing **{len(filtered):,}** of {len(df):,} rows")
                    st.dataframe(filtered, use_container_width=True)
                    csv = filtered.to_csv(index=False)
                    st.download_button("📥 Export CSV", csv, "filtered.csv", "text/csv")
            else:
                st.dataframe(df.sample(min(200,len(df))), use_container_width=True)

    # ---------- REPORT ----------
    with tabs[7]:
        sec_header("07", "Executive Report", "AI-generated summary")
        if st.button("📄 Generate Report"):
            cm = st.session_state.get("col_map", {})
            sales_col = cm.get("sales", "—")
            profit_col = cm.get("profit", "—")
            cat_col = cm.get("category", "—")
            date_col = cm.get("date", "—")
            total_rev = df[sales_col].sum() if sales_col != "—" and sales_col in df.columns else 0
            total_profit = df[profit_col].sum() if profit_col != "—" and profit_col in df.columns else 0
            margin = (total_profit/total_rev*100) if total_rev else 0
            top_cat = df.groupby(cat_col)[sales_col].sum().idxmax() if cat_col != "—" and cat_col in df.columns and sales_col != "—" and sales_col in df.columns else "N/A"
            date_range_str = f"{df[date_col].min().date()} → {df[date_col].max().date()}" if date_col != "—" and date_col in df.columns else "N/A"
            missing_pct = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100
            report = f"""# NEXUS Analytics Pro — Executive Report

## 📊 Dataset Overview
- **Records:** {len(df):,}
- **Columns:** {df.shape[1]}
- **Date Range:** {date_range_str}
- **Missing Data:** {missing_pct:.2f}%

## 💰 Financial Performance
- **Total Revenue:** {fmt_num(total_rev, prefix='$')}
- **Total Profit:** {fmt_num(total_profit, prefix='$')}
- **Profit Margin:** {margin:.1f}%
- **Top Category:** {top_cat}

## 🎯 Strategic Recommendations
1. **Double down on {top_cat}** — highest revenue generator
2. **Review discount strategy** — reduce margin erosion
3. **Activate RFM segmentation** — target Champions & Loyal customers
4. **Deploy demand forecasting** — optimize inventory levels
5. **Run weekly anomaly detection** — catch data quality issues early

---
*Generated by NEXUS Analytics Pro · {datetime.now().strftime('%Y-%m-%d %H:%M')}*
"""
            st.markdown(report)
            st.download_button("📥 Download Report", report, "nexus_report.md", "text/markdown")

    # ---------- PLANS ----------
    with tabs[8]:
        subscription_plans_tab()

    # ---------- AI CHAT ----------
    with tabs[9]:
        chatbot_tab()

# ========================== MAIN ==========================
def main():
    if "logged_in" not in st.session_state:
        st.session_state.update({"logged_in": False, "user_email": None, "is_admin": False})
    login_section()
    if st.session_state.get("is_admin", False):
        if st.sidebar.checkbox("🔧 Admin Panel", key="admin_mode_switch"):
            mega_admin_dashboard()
        else:
            render_analytics_app()
    else:
        render_analytics_app()

if __name__ == "__main__":
    main()
