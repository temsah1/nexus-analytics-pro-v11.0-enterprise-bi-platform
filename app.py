

import warnings
warnings.filterwarnings("ignore")

import io
import os
import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ========================== CONFIGURATION ==========================
MAX_FILE_SIZE_MB = 1000
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

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
    """Read CSV with automatic encoding detection"""
    encodings = ['utf-8', 'windows-1256', 'iso-8859-1', 'cp1252', 'latin1']
    for enc in encodings:
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding=enc)
            return df, enc
        except UnicodeDecodeError:
            continue
        except Exception:
            continue
    raise UnicodeDecodeError("Unable to decode file with common encodings. Please select encoding manually.")

# ========================== DATABASE (Extended) ==========================
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
    # Create admin user
    admin_email = "kareemeltemsah7@gmail.com"
    admin_pass = "temsah1"
    hashed = hashlib.sha256(admin_pass.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE email = ?", (admin_email,))
    if not c.fetchone():
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
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()

def toggle_admin(user_id, make_admin):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET is_admin = ? WHERE id = ?", (1 if make_admin else 0, user_id))
        conn.commit()
        # Log action
        user = get_user_by_id(user_id)
        if user:
            log_system_action("system", "toggle_admin", f"User {user['email']} admin status set to {make_admin}")

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
    """Promote existing user to admin, or create new admin if not exists"""
    with get_db() as conn:
        c = conn.cursor()
        user = get_user_by_email(email)
        if user:
            c.execute("UPDATE users SET is_admin = 1 WHERE email = ?", (email,))
            conn.commit()
            log_system_action("system", "promote_to_admin", f"Promoted user {email} to admin")
            return True, f"User {email} is now admin."
        else:
            # Create new user with random password? Better to ask to register first
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
        # Today's logins
        today = pd.Timestamp.now().date()
        c.execute("SELECT COUNT(*) FROM login_logs WHERE success = 1 AND date(timestamp) = ?", (today,))
        today_logins = c.fetchone()[0]
        return {
            "total_users": total_users,
            "total_admins": total_admins,
            "total_success_logins": total_success_logins,
            "total_failed_logins": total_failed_logins,
            "total_actions": total_actions,
            "today_logins": today_logins
        }

# ========================== PAGE CONFIG (ULTRA MODERN UI) ==========================
st.set_page_config(page_title="NEXUS Analytics Pro", page_icon="🚀", layout="wide", initial_sidebar_state="expanded")

# Enhanced CSS (same as before, abbreviated for length but fully functional)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:opsz,wght@14..32,300;400;500;600;700;800&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, .stApp { background: linear-gradient(145deg, #f8fafc 0%, #eef2f6 100%); font-family: 'Inter', sans-serif; }
[data-testid="stSidebar"] { background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%); backdrop-filter: blur(12px); border-right: 1px solid rgba(0,0,0,0.05); box-shadow: 8px 0 30px rgba(0,0,0,0.03); }
[data-testid="stMetric"] { background: rgba(255,255,255,0.9); backdrop-filter: blur(8px); border-radius: 32px; padding: 1.2rem; box-shadow: 0 8px 20px rgba(0,0,0,0.02); border: 1px solid rgba(255,255,255,0.5); transition: all 0.3s cubic-bezier(0.2, 0.9, 0.4, 1.1); }
[data-testid="stMetric"]:hover { transform: translateY(-4px); box-shadow: 0 20px 30px -12px rgba(0,0,0,0.1); border-color: rgba(139,92,246,0.3); }
[data-testid="stMetricValue"] { font-size: 2rem !important; font-weight: 800 !important; background: linear-gradient(135deg, #8b5cf6, #06b6d4); -webkit-background-clip: text; background-clip: text; color: transparent !important; }
.stButton > button { background: linear-gradient(95deg, #8b5cf6, #06b6d4); border: none; border-radius: 60px; padding: 0.6rem 1.4rem; font-weight: 600; color: white; box-shadow: 0 4px 12px rgba(6,182,212,0.2); transition: all 0.25s ease; }
.stButton > button:hover { transform: scale(1.02); box-shadow: 0 8px 24px rgba(139,92,246,0.3); background: linear-gradient(95deg, #7c3aed, #0891b2); }
[data-testid="stTabs"] { gap: 0.5rem; }
button[data-baseweb="tab"] { background: rgba(255,255,255,0.5); border-radius: 100px !important; padding: 0.4rem 1.2rem; font-weight: 500; border: 1px solid transparent; transition: 0.2s; }
button[data-baseweb="tab"][aria-selected="true"] { background: white; border-color: #8b5cf6; color: #8b5cf6; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
[data-testid="stDataFrame"] { border-radius: 24px; border: 1px solid rgba(0,0,0,0.05); overflow: hidden; }
h1, h2, h3 { background: linear-gradient(135deg, #1e293b, #4f46e5); -webkit-background-clip: text; background-clip: text; color: transparent; font-weight: 700; letter-spacing: -0.02em; }
.nx-card { background: rgba(255,255,255,0.8); backdrop-filter: blur(8px); border-radius: 32px; padding: 1.5rem; box-shadow: 0 4px 16px rgba(0,0,0,0.02); border: 1px solid rgba(255,255,255,0.6); margin-bottom: 1rem; }
.nx-header { display: flex; align-items: baseline; gap: 12px; margin: 1rem 0 1.2rem; flex-wrap: wrap; }
.nx-tag { background: linear-gradient(95deg, #8b5cf6, #06b6d4); color: white; border-radius: 60px; padding: 0.2rem 0.8rem; font-size: 0.7rem; font-weight: 600; }
.nx-title { font-size: 1.4rem; font-weight: 700; background: linear-gradient(135deg, #1e293b, #334155); -webkit-background-clip: text; background-clip: text; color: transparent; }
.insight-box { background: linear-gradient(115deg, rgba(139,92,246,0.08), rgba(6,182,212,0.08)); border-left: 5px solid #8b5cf6; border-radius: 20px; padding: 1rem 1.2rem; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ========================== HELPER FUNCTIONS ==========================
def sec_header(tag, title, sub=""):
    st.markdown(f"""
    <div class="nx-header">
        <span class="nx-tag">{tag}</span>
        <span class="nx-title">{title}</span>
        <span style="margin-left:auto; font-size:0.7rem; color:#64748b;">{sub}</span>
    </div>""", unsafe_allow_html=True)

def fmt_num(n, prefix="", suffix="", decimals=1):
    if n is None or np.isnan(n): return "N/A"
    n = float(n)
    if abs(n) >= 1e9: return f"{prefix}{n/1e9:.{decimals}f}B{suffix}"
    if abs(n) >= 1e6: return f"{prefix}{n/1e6:.{decimals}f}M{suffix}"
    if abs(n) >= 1e3: return f"{prefix}{n/1e3:.{decimals}f}K{suffix}"
    return f"{prefix}{n:.{decimals}f}{suffix}"

# -------------------- Data Loading & Cleaning (same as before) --------------------
@st.cache_data(show_spinner=False)
def load_builtin_dataset():
    np.random.seed(99)
    n = 5000
    start = pd.Timestamp("2022-01-01")
    dates = [start + pd.Timedelta(days=int(x)) for x in np.sort(np.random.randint(0, 1095, n))]
    categories = np.random.choice(["Electronics","Fashion","Home & Kitchen","Beauty","Sports","Books","Toys","Groceries"], n,
                                   p=[0.22,0.18,0.17,0.12,0.11,0.08,0.07,0.05])
    regions = np.random.choice(["Riyadh","Dubai","Cairo","Jeddah","Kuwait City","Doha","Amman","Manama"], n)
    segments = np.random.choice(["Premium","Standard","Economy"], n, p=[0.25,0.5,0.25])
    base_prices = {"Electronics":1200,"Fashion":180,"Home & Kitchen":250,"Beauty":120,"Sports":300,"Books":60,"Toys":90,"Groceries":45}
    sales = np.array([base_prices[c] * np.random.uniform(0.7,2.5) for c in categories])
    discount = np.random.choice([0,0.05,0.1,0.15,0.2,0.25,0.3], n, p=[0.3,0.15,0.2,0.15,0.1,0.06,0.04])
    sales_final = sales * (1 - discount)
    profit_margin = np.where(categories=="Electronics",0.12, np.where(categories=="Fashion",0.35, np.where(categories=="Books",0.4,0.22)))
    profit = sales_final * (profit_margin + np.random.normal(0,0.03,n))
    qty = np.random.randint(1,8,n)
    returns = np.random.choice([0,1], n, p=[0.88,0.12])
    rating = np.round(np.random.normal(4.1,0.5,n).clip(1,5),1)
    shipping_days = np.random.randint(1,7,n)
    df = pd.DataFrame({"Order Date": dates, "Category": categories, "Sub-Region": regions, "Customer Segment": segments,
                       "Sales": np.round(sales_final,2), "Profit": np.round(profit,2), "Discount": discount,
                       "Quantity": qty, "Returns": returns, "Rating": rating, "Shipping Days": shipping_days})
    return df

def detect_column_types(df):
    roles = {"date": [], "numeric": [], "categorical": [], "id": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s) or (s.dtype == object and pd.to_datetime(s, errors='coerce').notna().mean() > 0.6):
            roles["date"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            if col.lower() in ["id","row id","index","customer id","order id"] or col.endswith("_id"):
                roles["id"].append(col)
            else:
                roles["numeric"].append(col)
        else:
            if s.nunique() < 50:
                roles["categorical"].append(col)
            else:
                roles["id"].append(col)
    return roles

def smart_clean(df, roles):
    df = df.copy()
    for col in roles["date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in roles["numeric"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isnull().all():
            df[col].fillna(0, inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)
    for col in roles["categorical"]:
        mode_val = df[col].mode()
        df[col].fillna(mode_val.iloc[0] if not mode_val.empty else "Unknown", inplace=True)
    return df

# -------------------- ML Functions (abbreviated but fully functional) --------------------
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
    rf = RandomForestRegressor(n_estimators=400, max_depth=12, random_state=42, n_jobs=-1)
    gb = GradientBoostingRegressor(n_estimators=300, max_depth=5, random_state=42)
    rg = Ridge(alpha=1.0)
    en = ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=2000)
    try:
        from xgboost import XGBRegressor
        xgb = XGBRegressor(n_estimators=400, max_depth=6, random_state=42, verbosity=0)
        estimators = [("rf",rf),("gb",gb),("xgb",xgb),("ridge",rg)]
    except:
        estimators = [("rf",rf),("gb",gb),("ridge",rg),("en",en)]
    ensemble = VotingRegressor(estimators=estimators)
    ensemble.fit(Xs, y)
    tscv = TimeSeriesSplit(n_splits=3)
    try:
        cv_r2 = cross_val_score(ensemble, Xs, y, cv=tscv, scoring="r2").mean()
        cv_mape = 0
        for tr, te in tscv.split(Xs):
            ensemble.fit(Xs[tr], y.iloc[tr])
            pred = ensemble.predict(Xs[te])
            cv_mape += mean_absolute_percentage_error(y.iloc[te], np.maximum(pred,1e-6))
        cv_mape = (cv_mape/tscv.n_splits)*100
        ensemble.fit(Xs, y)
    except:
        cv_r2, cv_mape = None, None
    rf.fit(Xs, y)
    importances = dict(zip(X.columns, rf.feature_importances_))
    perm_importance = permutation_importance(rf, Xs, y, n_repeats=5, random_state=42, n_jobs=-1)
    perm_imp = dict(zip(X.columns, perm_importance.importances_mean))
    return ensemble, le_map, scaler, cv_r2, cv_mape, importances, perm_imp, list(X.columns)

@st.cache_data(show_spinner=False)
def build_forecast_prophet(date_series, value_series, horizon=90, freq='M'):
    df = pd.DataFrame({"ds": pd.to_datetime(date_series), "y": value_series.values})
    if freq == 'M':
        ts = df.groupby(df["ds"].dt.to_period("M").dt.start_time)["y"].sum().reset_index()
        ts.columns = ["ds","y"]
    else:
        ts = df.groupby(df["ds"].dt.to_period("W").dt.start_time)["y"].sum().reset_index()
        ts.columns = ["ds","y"]
    ts = ts.sort_values("ds").reset_index(drop=True)
    if PROPHET_AVAILABLE and len(ts) > 10:
        model = Prophet(yearly_seasonality=True, weekly_seasonality=(freq=='W'), daily_seasonality=False)
        model.fit(ts)
future = model.make_future_dataframe(periods=horizon, freq=freq_key)      
forecast = model.predict(future)
        forecast = forecast.tail(horizon) hist = ts.rename(columns={"ds":"Date","y":"Value"})
        fcast = pd.DataFrame({
            "Date": forecast["ds"],
            "Value": forecast["yhat"],
            "Lower": forecast["yhat_lower"],
            "Upper": forecast["yhat_upper"]
        })
        return hist, fcast, "Prophet"
    else:
        if not STATSMODELS_AVAILABLE:
            st.error("statsmodels not installed. Install with: pip install statsmodels")
            return None, None, "Error"
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        model = ExponentialSmoothing(ts["y"], trend='add', seasonal=None, initialization_method='estimated')
        fit = model.fit()
        forecast = fit.forecast(horizon)
        last_date = ts["ds"].iloc[-1]
        forecast_dates = [last_date + pd.Timedelta(days=30*(i+1)) for i in range(horizon)] if freq=='M' else [last_date + pd.Timedelta(days=7*(i+1)) for i in range(horizon)]
        hist = ts.rename(columns={"ds":"Date","y":"Value"})
        fcast = pd.DataFrame({"Date": forecast_dates, "Value": forecast.values,
                              "Lower": forecast.values*0.85, "Upper": forecast.values*1.15})
        return hist, fcast, "Exponential Smoothing"

def decompose_series(date_series, value_series, freq='M'):
    if not STATSMODELS_AVAILABLE:
        st.error("statsmodels not installed")
        return None, None
    df = pd.DataFrame({"ds": pd.to_datetime(date_series), "y": value_series.values})
    if freq == 'M':
        ts = df.groupby(df["ds"].dt.to_period("M").dt.start_time)["y"].sum().reset_index()
        ts.columns = ["ds","y"]
        period = 12
    else:
        ts = df.groupby(df["ds"].dt.to_period("W").dt.start_time)["y"].sum().reset_index()
        ts.columns = ["ds","y"]
        period = 52
    ts = ts.sort_values("ds").set_index("ds")
    if len(ts) >= 2*period:
        decomposition = seasonal_decompose(ts["y"], model='additive', period=period)
        return decomposition, ts
    else:
        return None, ts

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
        sil = silhouette_score(Xs, labels) if len(np.unique(labels))>1 else None
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
        sil = silhouette_score(Xs, labels) if len(set(labels)) - (1 if -1 in labels else 0) > 1 else None
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)
        return labels, sil, coords, None, pca.explained_variance_ratio_, model
    elif method == 'hierarchical':
        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(Xs)
        sil = silhouette_score(Xs, labels) if len(np.unique(labels))>1 else None
        pca = PCA(n_components=2)
        coords = pca.fit_transform(Xs)
        return labels, sil, coords, None, pca.explained_variance_ratio_, model
    else:
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
        rfm["Recency_Score"] = pd.qcut(rfm["Recency"], 5, labels=[5,4,3,2,1], duplicates='drop').astype(float)
        rfm["Frequency_Score"] = pd.qcut(rfm["Frequency"], 5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
        rfm["Monetary_Score"] = pd.qcut(rfm["Monetary"], 5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
    except:
        return None
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
        return None, None, "mlxtend not installed"
    basket = df.groupby([customer_col, product_col]).size().unstack().fillna(0).map(lambda x: 1 if x>0 else 0)
    frequent = apriori(basket, min_support=min_support, use_colnames=True)
    if len(frequent) == 0:
        return None, None, "No frequent itemsets"
    rules = association_rules(frequent, metric="lift", min_threshold=1.0)
    return frequent, rules, "Success"

# ========================== LOGIN UI ==========================
def login_section():
    st.sidebar.markdown("### 🔐 Login / Guest")
    if st.session_state.get("logged_in", False):
        st.sidebar.success(f"Welcome {st.session_state['user_email']}")
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
                    st.error("Invalid credentials")
        with st.sidebar.expander("📝 Register", expanded=False):
            new_email = st.text_input("Email", key="reg_email")
            new_pwd = st.text_input("Password", type="password", key="reg_password")
            if st.button("Register", key="reg_submit"):
                if register_user(new_email, new_pwd, is_admin=False):
                    st.success("Account created! Please login.")
                    log_system_action(new_email, "register", "New user registered")
                else:
                    st.error("Email already exists")
        st.sidebar.info("💡 Guest mode: all features available without login. Admin panel requires login.")
        return False

# ========================== MEGA ADMIN DASHBOARD ==========================
def mega_admin_dashboard():
    sec_header("ADMIN", "Mega Dashboard", "Full System Control")

    # Stats row
    stats = get_stats()
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("👥 Total Users", stats["total_users"])
    with col2:
        st.metric("👑 Admins", stats["total_admins"])
    with col3:
        st.metric("✅ Today's Logins", stats["today_logins"])
    with col4:
        st.metric("📊 Total Logins", stats["total_success_logins"])
    with col5:
        st.metric("⚠️ Failed Logins", stats["total_failed_logins"])

    # Charts
    logs = get_login_logs(limit=200)
    if logs:
        df_logs = pd.DataFrame(logs, columns=["email", "success", "timestamp"])
        df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
        df_logs["date"] = df_logs["timestamp"].dt.date
        login_counts = df_logs.groupby(["date", "success"]).size().reset_index(name="count")
        login_counts["status"] = login_counts["success"].map({1: "Success", 0: "Failed"})
        fig = px.line(login_counts, x="date", y="count", color="status", title="Login Activity Over Time",
                      color_discrete_map={"Success": "#10b981", "Failed": "#ef4444"})
        fig.update_layout(height=400, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)

    # User Management Section
    st.subheader("👤 User Management")
    users = get_all_users()
    if users:
        df_users = pd.DataFrame(users, columns=["ID", "Email", "Is Admin", "Created At", "Last Login"])
        # Convert datetime for display
        df_users["Created At"] = pd.to_datetime(df_users["Created At"]).dt.strftime("%Y-%m-%d %H:%M")
        df_users["Last Login"] = pd.to_datetime(df_users["Last Login"]).dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df_users, use_container_width=True)

        # Actions
        st.markdown("#### User Actions")
        col1, col2, col3 = st.columns(3)
        with col1:
            uid_del = st.number_input("User ID to delete", min_value=1, step=1, key="admin_del_id")
            if st.button("🗑️ Delete User", key="admin_del_btn"):
                delete_user(uid_del)
                st.success(f"User {uid_del} deleted.")
                st.rerun()
        with col2:
            uid_toggle = st.number_input("User ID to toggle admin", min_value=1, step=1, key="admin_toggle_id")
            make_ad = st.checkbox("Make admin?", key="admin_make_ad")
            if st.button("🔄 Toggle Admin", key="admin_toggle_btn"):
                toggle_admin(uid_toggle, make_ad)
                st.success(f"User {uid_toggle} admin status updated to {make_ad}.")
                st.rerun()
        with col3:
            uid_reset = st.number_input("User ID to reset password", min_value=1, step=1, key="admin_reset_id")
            new_pass = st.text_input("New password", type="password", key="admin_new_pass")
            if st.button("🔑 Reset Password", key="admin_reset_btn"):
                if new_pass:
                    reset_user_password(uid_reset, new_pass)
                    st.success(f"Password for user {uid_reset} reset.")
                else:
                    st.error("Please enter a new password.")

    # Add Admin by Email
    st.markdown("#### ➕ Add / Promote Admin")
    col1, col2 = st.columns([3,1])
    with col1:
        new_admin_email = st.text_input("Email address", key="new_admin_email")
    with col2:
        if st.button("Promote to Admin", key="promote_admin_btn"):
            if new_admin_email:
                success, msg = add_admin_by_email(new_admin_email)
                if success:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Please enter an email.")

    # System Logs
    st.subheader("📋 System Activity Log")
    sys_logs = get_system_logs(limit=200)
    if sys_logs:
        df_sys = pd.DataFrame(sys_logs, columns=["User", "Action", "Details", "Timestamp"])
        st.dataframe(df_sys, use_container_width=True)
        csv = df_sys.to_csv(index=False)
        st.download_button("📥 Export Logs", csv, "system_logs.csv", "text/csv", key="export_logs")
    else:
        st.info("No system logs yet.")

    # Settings
    st.subheader("⚙️ System Settings")
    new_limit = st.number_input("Max Upload Size (MB)", min_value=100, max_value=2000, value=MAX_FILE_SIZE_MB, step=50, key="sys_limit")
    if st.button("Apply New Limit", key="apply_limit"):
        with open(CONFIG_FILE, "w") as f:
            f.write(f"""
[server]
maxUploadSize = {new_limit}
""")
        st.success(f"Upload limit updated to {new_limit} MB. Restart required for full effect.")

    if st.button("🗑️ Clear Streamlit Cache", key="clear_cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cache cleared. Refresh page.")

    # System Health
    st.subheader("📊 System Health")
    db_size = os.path.getsize(DB_PATH) / (1024*1024) if os.path.exists(DB_PATH) else 0
    st.metric("Database Size", f"{db_size:.2f} MB")
    st.metric("Total Actions Logged", stats["total_actions"])

# ========================== MAIN APP (Analytics for all) ==========================
def render_analytics_app():
    # Initialize session data
    if "df" not in st.session_state:
        st.session_state["df"] = None
        st.session_state["roles"] = None
        st.session_state["source"] = None
        st.session_state["col_map"] = {}

    with st.sidebar:
        st.markdown("## 🚀 NEXUS Analytics")
        if st.session_state.get("logged_in", False):
            st.markdown(f"👤 **{st.session_state['user_email']}**")
        else:
            st.markdown("👤 **Guest Mode**")
        st.markdown("---")
        st.markdown("### Data Source")
        source = st.radio("", ["📦 Built-in Dataset", "📂 Upload File"], label_visibility="collapsed", key="data_source")
        if source == "📂 Upload File":
            uploaded = st.file_uploader(f"CSV / Excel / JSON (Max {MAX_FILE_SIZE_MB} MB)", type=["csv","xlsx","xls","json"], key="file_upload")
            if uploaded:
                if uploaded.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"⚠️ File exceeds {MAX_FILE_SIZE_MB} MB limit. Your file: {uploaded.size/(1024*1024):.2f} MB")
                else:
                    progress_bar = st.progress(0, text="Loading file...")
                    try:
                        if uploaded.name.endswith('.csv'):
                            try:
                                df_new, used_encoding = read_csv_with_encoding(uploaded)
                                st.success(f"✓ Loaded with encoding: {used_encoding}")
                            except UnicodeDecodeError:
                                st.warning("Auto-detection failed. Select encoding manually:")
                                manual_enc = st.selectbox("Encoding", ['utf-8', 'windows-1256', 'iso-8859-1', 'cp1252'], key="manual_enc")
                                uploaded.seek(0)
                                df_new = pd.read_csv(uploaded, encoding=manual_enc)
                                st.success(f"✓ Loaded with encoding: {manual_enc}")
                        else:
                            df_new = pd.read_excel(uploaded)
                        progress_bar.progress(100, text="Complete!")
                        if st.session_state["source"] != uploaded.name:
                            st.session_state["df_raw"] = df_new
                            st.session_state["roles"] = detect_column_types(df_new)
                            st.session_state["df"] = smart_clean(df_new, st.session_state["roles"])
                            st.session_state["source"] = uploaded.name
                            st.session_state["col_map"] = {}
                            log_system_action(st.session_state.get("user_email", "guest"), "upload_file", f"Uploaded {uploaded.name} ({uploaded.size/(1024*1024):.2f} MB)")
                        st.success(f"✓ Loaded {uploaded.name}")
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        progress_bar.empty()
        else:
            if st.session_state["source"] != "builtin":
                df_bi = load_builtin_dataset()
                st.session_state["df_raw"] = df_bi
                st.session_state["roles"] = detect_column_types(df_bi)
                st.session_state["df"] = smart_clean(df_bi, st.session_state["roles"])
                st.session_state["source"] = "builtin"
                st.session_state["col_map"] = {}
                log_system_action(st.session_state.get("user_email", "guest"), "load_builtin", "Loaded built-in dataset")
            st.success("✓ Built-in dataset ready")

        df = st.session_state.get("df")
        if df is not None:
            st.markdown("---")
            st.markdown("### Column Mapping")
            roles = st.session_state["roles"]
            num_c = ["—"] + roles["numeric"]
            date_c = ["—"] + roles["date"]
            cat_c = ["—"] + roles["categorical"]
            id_c = ["—"] + roles["id"] + roles["categorical"]
            cm = st.session_state["col_map"]
            cm["sales"] = st.selectbox("Sales", num_c, index=num_c.index(cm.get("sales","—")) if cm.get("sales","—") in num_c else 0, key="map_sales")
            cm["profit"] = st.selectbox("Profit", num_c, index=num_c.index(cm.get("profit","—")) if cm.get("profit","—") in num_c else 0, key="map_profit")
            cm["date"] = st.selectbox("Date", date_c, index=date_c.index(cm.get("date","—")) if cm.get("date","—") in date_c else 0, key="map_date")
            cm["category"] = st.selectbox("Category", cat_c, index=cat_c.index(cm.get("category","—")) if cm.get("category","—") in cat_c else 0, key="map_cat")
            cm["customer"] = st.selectbox("Customer ID", id_c, index=id_c.index(cm.get("customer","—")) if cm.get("customer","—") in id_c else 0, key="map_cust")
            cm["product"] = st.selectbox("Product (for basket)", cat_c, index=cat_c.index(cm.get("product","—")) if cm.get("product","—") in cat_c else 0, key="map_prod")
            st.session_state["col_map"] = cm

    df = st.session_state.get("df")
    if df is None:
        st.info("Please load data from the sidebar.")
        return

    tabs = st.tabs([
        "📊 Data Hub", "💰 KPIs", "🔮 Forecasting", "🤖 Profit Optimizer",
        "👥 Segmentation", "🛒 Market Basket", "📈 Advanced Analytics", "📄 Executive Report"
    ])

    # ---------- Data Hub ----------
    with tabs[0]:
        sec_header("00", "Data Hub", "Quality & Overview")
        st.dataframe(df.head(100), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Column Types**")
            st.json({k: len(v) for k,v in st.session_state["roles"].items()})
        with col2:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", f"{missing:,}")
        with st.expander("Statistical Summary"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ---------- KPIs ----------
    with tabs[1]:
        sec_header("01", "Key Performance Indicators", "Revenue & Profit")
        sales_col = st.session_state["col_map"].get("sales","—")
        profit_col = st.session_state["col_map"].get("profit","—")
        if sales_col != "—" and sales_col in df.columns:
            total_rev = df[sales_col].sum()
            total_profit = df[profit_col].sum() if profit_col != "—" and profit_col in df.columns else None
            margin = (total_profit/total_rev*100) if total_profit else None
            c1,c2,c3 = st.columns(3)
            c1.metric("Total Revenue", fmt_num(total_rev, prefix="$"))
            if total_profit: c2.metric("Total Profit", fmt_num(total_profit, prefix="$"))
            if margin: c3.metric("Profit Margin", f"{margin:.1f}%")
            date_col = st.session_state["col_map"].get("date","—")
            if date_col != "—" and date_col in df.columns:
                df_ts = df.set_index(date_col).resample('ME')[sales_col].sum().reset_index()
                fig = px.line(df_ts, x=date_col, y=sales_col, title="Monthly Sales Trend", markers=True)
                fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map a sales column in the sidebar.")

    # ---------- Forecasting ----------
    with tabs[2]:
        sec_header("02", "Demand Forecasting", "30-180 days horizon")
        sales_col = st.session_state["col_map"].get("sales","—")
        date_col = st.session_state["col_map"].get("date","—")
        if sales_col != "—" and date_col != "—":
            horizon = st.slider("Forecast Horizon (days)", 30, 180, 90, key="fc_horizon")
            freq = st.selectbox("Aggregation", ["Monthly", "Weekly"], index=0, key="fc_freq")
freq_key = 'ME' if freq == "Monthly" else 'W'
            if st.button("Run Forecast", key="fc_run"):
                with st.spinner("Building forecast model..."):
                    result = build_forecast_prophet(df[date_col], df[sales_col], horizon, freq_key)
                    if result[0] is None:
                        st.error("Failed to build forecast. Install statsmodels or prophet.")
                    else:
                        hist, fcast, model_name = result
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Value"], name="Historical", line=dict(color="#06b6d4", width=2)))
                        fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Value"], name="Forecast", line=dict(color="#f59e0b", dash="dash", width=2)))
                        fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Upper"], fill=None))
                        fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Lower"], fill="tonexty", name="Confidence Interval", fillcolor="rgba(6,182,212,0.2)"))
                        fig.update_layout(title=f"Demand Forecast - {model_name}", height=500)
                        st.plotly_chart(fig, use_container_width=True)
                        st.dataframe(fcast, use_container_width=True)
                        log_system_action(st.session_state.get("user_email","guest"), "forecast", f"Ran forecast with horizon {horizon} days")
        else:
            st.info("Map Sales and Date columns.")

    # ---------- Profit Optimizer ----------
    with tabs[3]:
        sec_header("03", "AI Profit Optimizer", "Voting Ensemble Regressor")
        profit_col = st.session_state["col_map"].get("profit","—")
        if profit_col != "—" and profit_col in df.columns:
            features = st.multiselect("Select features", [c for c in df.columns if c != profit_col], default=[], key="ml_features")
            if features and st.button("Train Model", key="ml_train"):
                with st.spinner("Training ensemble..."):
                    ensemble, le_map, scaler, r2, mape, imp, perm_imp, _ = train_ml_ensemble(df[features+[profit_col]].dropna().to_json(), profit_col, features)
                    st.success(f"✅ Model ready | R²: {r2:.3f} | MAPE: {mape:.1f}%")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Feature Importance (RF)")
                        st.bar_chart(pd.Series(imp))
                    with col2:
                        st.subheader("Permutation Importance")
                        st.bar_chart(pd.Series(perm_imp))
                    log_system_action(st.session_state.get("user_email","guest"), "train_model", f"Trained profit model with {len(features)} features")
        else:
            st.info("Map a Profit column.")

    # ---------- Segmentation ----------
    with tabs[4]:
        sec_header("04", "Customer Intelligence", "RFM & Clustering")
        cust_col = st.session_state["col_map"].get("customer","—")
        sales_col = st.session_state["col_map"].get("sales","—")
        date_col = st.session_state["col_map"].get("date","—")
        if cust_col != "—" and sales_col != "—" and date_col != "—":
            if st.button("RFM Analysis", key="rfm_run"):
                rfm = compute_rfm(df, date_col, sales_col, cust_col)
                if rfm is not None:
                    st.dataframe(rfm.head(), use_container_width=True)
                    fig = px.bar(rfm["Segment"].value_counts(), title="Customer Segments", color=rfm["Segment"].value_counts().index,
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    st.plotly_chart(fig, use_container_width=True)
                    log_system_action(st.session_state.get("user_email","guest"), "rfm_analysis", "Ran RFM segmentation")
                else:
                    st.warning("RFM failed: insufficient data or duplicate quantiles.")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                method = st.selectbox("Clustering Method", ["kmeans", "dbscan", "hierarchical"], key="clust_method")
                if method == 'kmeans':
                    k = st.slider("Number of clusters (K)", 2, 8, 3, key="k_kmeans")
                    if st.button("Run Clustering", key="clust_run"):
                        labels, sil, coords, inertias, var, model = run_advanced_clustering(df[num_cols].dropna().to_json(), num_cols, method=method, n_clusters=k)
                        if sil:
                            st.metric("Silhouette Score", f"{sil:.3f}")
                            fig = px.scatter(x=coords[:,0], y=coords[:,1], color=labels.astype(str), title=f"PCA Projection - {method}")
                            st.plotly_chart(fig, use_container_width=True)
                            if inertias:
                                fig_elbow = px.line(x=list(inertias.keys()), y=list(inertias.values()), markers=True, title="Elbow Method")
                                st.plotly_chart(fig_elbow, use_container_width=True)
                            log_system_action(st.session_state.get("user_email","guest"), "clustering", f"Ran {method} clustering with k={k}")
                elif method == 'dbscan':
                    eps = st.slider("Epsilon", 0.1, 2.0, 0.5, 0.05, key="db_eps")
                    if st.button("Run DBSCAN", key="db_run"):
                        labels, sil, coords, _, var, _ = run_advanced_clustering(df[num_cols].dropna().to_json(), num_cols, method='dbscan', eps=eps)
                        if sil:
                            st.metric("Silhouette Score", f"{sil:.3f}")
                            fig = px.scatter(x=coords[:,0], y=coords[:,1], color=labels.astype(str), title="DBSCAN Clustering")
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    k = st.slider("Number of clusters", 2, 8, 3, key="k_hier")
                    if st.button("Run Hierarchical", key="hier_run"):
                        labels, sil, coords, _, var, _ = run_advanced_clustering(df[num_cols].dropna().to_json(), num_cols, method='hierarchical', n_clusters=k)
                        if sil:
                            st.metric("Silhouette Score", f"{sil:.3f}")
                            fig = px.scatter(x=coords[:,0], y=coords[:,1], color=labels.astype(str), title="Hierarchical Clustering")
                            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Map Customer ID, Sales, and Date columns.")

    # ---------- Market Basket ----------
    with tabs[5]:
        sec_header("05", "Market Basket Analysis", "Apriori Algorithm")
        if not MLXTEND_AVAILABLE:
            st.warning("Install mlxtend: pip install mlxtend")
        else:
            cust_col = st.session_state["col_map"].get("customer","—")
            prod_col = st.session_state["col_map"].get("product","—")
            if cust_col != "—" and prod_col != "—":
                min_sup = st.slider("Minimum Support", 0.005, 0.05, 0.01, 0.005, key="mb_support")
                if st.button("Run Apriori", key="mb_run"):
                    freq, rules, msg = market_basket_analysis(df, cust_col, prod_col, min_sup)
                    if rules is not None and len(rules)>0:
                        st.dataframe(rules[["antecedents","consequents","support","confidence","lift"]], use_container_width=True)
                        log_system_action(st.session_state.get("user_email","guest"), "market_basket", f"Found {len(rules)} association rules")
                    else:
                        st.info(msg)
            else:
                st.info("Map Customer ID and Product columns.")

    # ---------- Advanced Analytics ----------
    with tabs[6]:
        sec_header("06", "Advanced Analytics", "Anomaly Detection & Correlation")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Matrix", aspect="auto")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("🔍 Anomaly Detection (Isolation Forest)")
        feat_anom = st.multiselect("Select features", num_cols, default=num_cols[:min(3, len(num_cols))], key="anom_feat")
        if feat_anom and st.button("Detect Anomalies", key="anom_run"):
            with st.spinner("Training Isolation Forest..."):
                anomalies, model = detect_anomalies_iforest(df[feat_anom].dropna().to_json(), feat_anom, contamination=0.05)
                n_anom = anomalies.sum()
                st.metric("Anomalies Detected", f"{n_anom} ({n_anom/len(anomalies)*100:.1f}%)")
                if n_anom > 0:
                    st.dataframe(df[anomalies], use_container_width=True)
                    log_system_action(st.session_state.get("user_email","guest"), "anomaly_detection", f"Detected {n_anom} anomalies")
        st.subheader("📊 Interactive Data Explorer")
        date_col = st.session_state["col_map"].get("date","—")
        if date_col != "—" and date_col in df.columns:
            min_date = df[date_col].min()
            max_date = df[date_col].max()
            date_range = st.date_input("Date Range", [min_date, max_date], key="date_filter")
            if len(date_range) == 2:
                mask = (df[date_col] >= pd.to_datetime(date_range[0])) & (df[date_col] <= pd.to_datetime(date_range[1]))
                filtered_df = df[mask]
                st.write(f"Showing {len(filtered_df)} rows")
                st.dataframe(filtered_df, use_container_width=True)

    # ---------- Executive Report ----------
    with tabs[7]:
        sec_header("07", "Executive AI Report", "Actionable Insights")
        if st.button("Generate Report", key="report_gen"):
            sales_col = st.session_state["col_map"].get("sales","—")
            profit_col = st.session_state["col_map"].get("profit","—")
            total_rev = df[sales_col].sum() if sales_col != "—" else 0
            total_profit = df[profit_col].sum() if profit_col != "—" else 0
            cat_col = st.session_state["col_map"].get("category","—")
            top_category = df.groupby(cat_col)[sales_col].sum().idxmax() if cat_col != "—" and cat_col in df.columns else "N/A"
            report = f"""# 📈 Executive Summary

- **Total Records:** {len(df):,}
- **Total Revenue:** {fmt_num(total_rev, prefix='$')}
- **Total Profit:** {fmt_num(total_profit, prefix='$')}
- **Profit Margin:** {(total_profit/total_rev*100) if total_rev else 0:.1f}%
- **Top Performing Category:** {top_category}

## 🧠 AI Recommendations
1. Focus on high-margin products and optimize discount strategies.
2. Target 'Champions' and 'Loyal' segments for loyalty programs.
3. Investigate anomalies detected for potential fraud or operational issues.
4. Use the forecasting model to plan inventory.

---
*Report generated by NEXUS Analytics Pro v11.0*
"""
            st.markdown(report)
            st.download_button("Download Report", report, "nexus_report.md", key="dl_report")
            log_system_action(st.session_state.get("user_email","guest"), "generate_report", "Executive report generated")

# ========================== MAIN ==========================
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = None
        st.session_state["is_admin"] = False

    login_section()
    # Show admin dashboard only if logged in as admin and admin mode is toggled
    if st.session_state.get("is_admin", False):
        admin_mode = st.sidebar.checkbox("🔧 Admin Mode (Mega Dashboard)", key="admin_mode_switch")
        if admin_mode:
            mega_admin_dashboard()
        else:
            render_analytics_app()
    else:
        render_analytics_app()

if __name__ == "__main__":
    main()
