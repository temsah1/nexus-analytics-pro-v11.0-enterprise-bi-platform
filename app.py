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
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
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

# ========================== PAGE CONFIG ==========================
st.set_page_config(
    page_title="NEXUS Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
* { margin: 0; padding: 0; box-sizing: border-box; }
html, body, .stApp {
    background: linear-gradient(145deg, #f8fafc 0%, #eef2f6 100%);
    font-family: 'Inter', sans-serif;
}
[data-testid="stSidebar"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.98) 0%, rgba(248,250,252,0.98) 100%);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(0,0,0,0.05);
    box-shadow: 8px 0 30px rgba(0,0,0,0.03);
}
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.9);
    backdrop-filter: blur(8px);
    border-radius: 20px;
    padding: 1.2rem;
    box-shadow: 0 8px 20px rgba(0,0,0,0.04);
    border: 1px solid rgba(255,255,255,0.6);
    transition: all 0.3s ease;
}
[data-testid="stMetric"]:hover {
    transform: translateY(-4px);
    box-shadow: 0 20px 30px -12px rgba(0,0,0,0.12);
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    background: linear-gradient(135deg, #8b5cf6, #06b6d4);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent !important;
}
.stButton > button {
    background: linear-gradient(95deg, #8b5cf6, #06b6d4);
    border: none;
    border-radius: 60px;
    padding: 0.6rem 1.4rem;
    font-weight: 600;
    color: white;
    box-shadow: 0 4px 12px rgba(6,182,212,0.2);
    transition: all 0.25s ease;
}
.stButton > button:hover {
    transform: scale(1.02);
    box-shadow: 0 8px 24px rgba(139,92,246,0.3);
}
h1, h2, h3 {
    background: linear-gradient(135deg, #1e293b, #4f46e5);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    font-weight: 700;
}
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
    background: linear-gradient(135deg, #1e293b, #334155);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
}
.admin-kpi-card {
    background: linear-gradient(135deg, rgba(139,92,246,0.08), rgba(6,182,212,0.08));
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 20px;
    padding: 1.5rem;
    text-align: center;
    margin-bottom: 1rem;
}
.insight-box {
    background: linear-gradient(115deg, rgba(139,92,246,0.08), rgba(6,182,212,0.08));
    border-left: 5px solid #8b5cf6;
    border-radius: 20px;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
}
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
        if pd.api.types.is_datetime64_any_dtype(s) or (
            s.dtype == object and pd.to_datetime(s, errors='coerce').notna().mean() > 0.6
        ):
            roles["date"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            lc = col.lower()
            if lc in ["id", "row id", "index", "customer id", "order id"] or col.endswith("_id"):
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
    period_key = "M" if freq == "ME" else "W"
    ts = df.groupby(df["ds"].dt.to_period(period_key).dt.start_time)["y"].sum().reset_index()
    ts.columns = ["ds", "y"]
    ts = ts.sort_values("ds").reset_index(drop=True)

    if PROPHET_AVAILABLE and len(ts) > 10:
        try:
            model = Prophet(yearly_seasonality=True, weekly_seasonality=(freq == 'W'), daily_seasonality=False)
            model.fit(ts)
            future = model.make_future_dataframe(periods=horizon, freq=freq)
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
                forecast_dates = [last_date + pd.DateOffset(months=i + 1) for i in range(horizon)]
            else:
                forecast_dates = [last_date + pd.Timedelta(days=7 * (i + 1)) for i in range(horizon)]
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
        st.sidebar.info("💡 Guest mode: all analytics features available.")
        return False

# ========================== MEGA ADMIN DASHBOARD ==========================
def mega_admin_dashboard():
    st.markdown("""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #4f46e5 50%, #06b6d4 100%);
         border-radius: 24px; padding: 2.5rem; margin-bottom: 2rem; text-align: center;">
        <h1 style="color: white; font-size: 2.2rem; font-weight: 800; background: none;
            -webkit-background-clip: unset; background-clip: unset; margin-bottom: 0.5rem;">
            🛡️ NEXUS Admin Control Center
        </h1>
        <p style="color: rgba(255,255,255,0.8); font-size: 1rem; margin: 0;">
            Full system visibility · User management · Activity monitoring · Platform health
        </p>
    </div>
    """, unsafe_allow_html=True)

    stats = get_stats()

    # ---- TOP KPI ROW ----
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        st.metric("👥 Total Users", stats["total_users"])
    with c2:
        st.metric("👑 Admins", stats["total_admins"])
    with c3:
        st.metric("✅ Today's Logins", stats["today_logins"])
    with c4:
        st.metric("🔢 All-Time Logins", stats["total_success_logins"])
    with c5:
        st.metric("❌ Failed Attempts", stats["total_failed_logins"])
    with c6:
        st.metric("📋 System Actions", stats["total_actions"])

    st.markdown("---")

    # ---- ADMIN TABS ----
    admin_tabs = st.tabs([
        "📊 Dashboard Overview",
        "👤 User Management",
        "📋 Activity Logs",
        "⚙️ System Settings",
        "📈 Platform Analytics"
    ])

    # ---- TAB 0: DASHBOARD OVERVIEW ----
    with admin_tabs[0]:
        sec_header("OVERVIEW", "Platform Health Dashboard", "Real-time insights")

        logs = get_login_logs(limit=500)
        if logs:
            df_logs = pd.DataFrame(logs, columns=["email", "success", "timestamp"])
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])
            df_logs["date"] = df_logs["timestamp"].dt.date
            df_logs["hour"] = df_logs["timestamp"].dt.hour

            col1, col2 = st.columns(2)

            with col1:
                login_counts = df_logs.groupby(["date", "success"]).size().reset_index(name="count")
                login_counts["status"] = login_counts["success"].map({1: "✅ Success", 0: "❌ Failed"})
                fig = px.area(
                    login_counts, x="date", y="count", color="status",
                    title="📅 Login Activity Over Time",
                    color_discrete_map={"✅ Success": "#10b981", "❌ Failed": "#ef4444"}
                )
                fig.update_layout(height=350, template="plotly_white", plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                hourly = df_logs.groupby("hour").size().reset_index(name="logins")
                fig2 = px.bar(
                    hourly, x="hour", y="logins",
                    title="🕐 Login Distribution by Hour",
                    color="logins", color_continuous_scale="Viridis"
                )
                fig2.update_layout(height=350, template="plotly_white", plot_bgcolor='rgba(0,0,0,0)')
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                success_rate = (
                    df_logs["success"].sum() / len(df_logs) * 100
                    if len(df_logs) > 0 else 0
                )
                fig3 = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=round(success_rate, 1),
                    title={"text": "Login Success Rate (%)"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "#8b5cf6"},
                        "steps": [
                            {"range": [0, 50], "color": "#fee2e2"},
                            {"range": [50, 75], "color": "#fef3c7"},
                            {"range": [75, 100], "color": "#d1fae5"},
                        ]
                    }
                ))
                fig3.update_layout(height=300)
                st.plotly_chart(fig3, use_container_width=True)

            with col4:
                # Top users by login count
                top_users = (
                    df_logs[df_logs["success"] == 1]
                    .groupby("email")
                    .size()
                    .reset_index(name="logins")
                    .sort_values("logins", ascending=False)
                    .head(8)
                )
                fig4 = px.bar(
                    top_users, x="logins", y="email", orientation="h",
                    title="🏆 Most Active Users",
                    color="logins", color_continuous_scale="Blues"
                )
                fig4.update_layout(height=300, template="plotly_white")
                st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("No login data yet.")

        # System summary boxes
        st.markdown("### 🖥️ Platform Summary")
        col1, col2, col3 = st.columns(3)
        with col1:
            db_size = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
            st.markdown(f"""
            <div class="insight-box">
                <strong>💾 Database Size</strong><br>
                <span style="font-size:1.5rem; font-weight:800; color:#8b5cf6;">{db_size:.3f} MB</span><br>
                <small>SQLite · nexus_users.db</small>
            </div>""", unsafe_allow_html=True)
        with col2:
            fail_rate = (
                stats["total_failed_logins"] /
                max(stats["total_success_logins"] + stats["total_failed_logins"], 1) * 100
            )
            st.markdown(f"""
            <div class="insight-box">
                <strong>⚠️ Failure Rate</strong><br>
                <span style="font-size:1.5rem; font-weight:800; color:#f59e0b;">{fail_rate:.1f}%</span><br>
                <small>of all login attempts</small>
            </div>""", unsafe_allow_html=True)
        with col3:
            regular_users = stats["total_users"] - stats["total_admins"]
            st.markdown(f"""
            <div class="insight-box">
                <strong>👤 Regular Users</strong><br>
                <span style="font-size:1.5rem; font-weight:800; color:#06b6d4;">{regular_users}</span><br>
                <small>Non-admin accounts</small>
            </div>""", unsafe_allow_html=True)

    # ---- TAB 1: USER MANAGEMENT ----
    with admin_tabs[1]:
        sec_header("USERS", "User Management Center", "Create · Edit · Delete · Promote")

        users = get_all_users()
        if users:
            df_users = pd.DataFrame(users, columns=["ID", "Email", "Is Admin", "Created At", "Last Login"])
            df_users["Role"] = df_users["Is Admin"].map({1: "👑 Admin", 0: "👤 User"})
            df_users["Created At"] = pd.to_datetime(df_users["Created At"]).dt.strftime("%Y-%m-%d %H:%M")
            df_users["Last Login"] = pd.to_datetime(df_users["Last Login"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M").fillna("Never")
            st.dataframe(
                df_users[["ID", "Email", "Role", "Created At", "Last Login"]],
                use_container_width=True
            )

        st.markdown("---")
        st.markdown("#### 🔧 User Actions")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**🗑️ Delete User**")
            uid_del = st.number_input("User ID", min_value=1, step=1, key="admin_del_id")
            if st.button("Delete User", key="admin_del_btn"):
                delete_user(uid_del)
                st.success(f"✅ User {uid_del} deleted.")
                st.rerun()

        with col2:
            st.markdown("**🔄 Toggle Admin Role**")
            uid_toggle = st.number_input("User ID", min_value=1, step=1, key="admin_toggle_id")
            make_ad = st.checkbox("Grant Admin?", key="admin_make_ad")
            if st.button("Toggle Admin", key="admin_toggle_btn"):
                toggle_admin(uid_toggle, make_ad)
                action_text = "granted" if make_ad else "revoked"
                st.success(f"✅ Admin {action_text} for user {uid_toggle}.")
                st.rerun()

        with col3:
            st.markdown("**🔑 Reset Password**")
            uid_reset = st.number_input("User ID", min_value=1, step=1, key="admin_reset_id")
            new_pass = st.text_input("New Password", type="password", key="admin_new_pass")
            if st.button("Reset Password", key="admin_reset_btn"):
                if new_pass:
                    reset_user_password(uid_reset, new_pass)
                    st.success(f"✅ Password for user {uid_reset} has been reset.")
                else:
                    st.error("Please enter a new password.")

        st.markdown("---")
        st.markdown("#### ➕ Promote User to Admin")
        col1, col2 = st.columns([3, 1])
        with col1:
            new_admin_email = st.text_input("Email address to promote", key="new_admin_email")
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("⬆️ Promote to Admin", key="promote_admin_btn"):
                if new_admin_email:
                    success_flag, msg = add_admin_by_email(new_admin_email)
                    if success_flag:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)
                else:
                    st.error("Please enter an email.")

        st.markdown("---")
        st.markdown("#### 📝 Register New User (as Admin)")
        col1, col2, col3 = st.columns(3)
        with col1:
            new_u_email = st.text_input("Email", key="admin_new_user_email")
        with col2:
            new_u_pass = st.text_input("Password", type="password", key="admin_new_user_pass")
        with col3:
            new_u_admin = st.checkbox("Make Admin?", key="admin_new_user_is_admin")
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Create User", key="admin_create_user_btn"):
                if new_u_email and new_u_pass:
                    if register_user(new_u_email, new_u_pass, is_admin=new_u_admin):
                        st.success(f"✅ User {new_u_email} created.")
                        log_system_action(
                            st.session_state.get("user_email", "admin"),
                            "create_user",
                            f"Admin created user {new_u_email}"
                        )
                        st.rerun()
                    else:
                        st.error("⚠️ Email already exists.")
                else:
                    st.error("Please fill email and password.")

    # ---- TAB 2: ACTIVITY LOGS ----
    with admin_tabs[2]:
        sec_header("LOGS", "Activity & Audit Logs", "Full audit trail")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### 🔐 Login Logs")
            logs = get_login_logs(limit=300)
            if logs:
                df_login_logs = pd.DataFrame(logs, columns=["Email", "Success", "Timestamp"])
                df_login_logs["Status"] = df_login_logs["Success"].map({1: "✅ Success", 0: "❌ Failed"})
                st.dataframe(
                    df_login_logs[["Email", "Status", "Timestamp"]],
                    use_container_width=True, height=400
                )
                csv_login = df_login_logs.to_csv(index=False)
                st.download_button(
                    "📥 Export Login Logs", csv_login,
                    "login_logs.csv", "text/csv", key="export_login_logs"
                )
            else:
                st.info("No login logs yet.")

        with col2:
            st.markdown("##### 📋 System Action Logs")
            sys_logs = get_system_logs(limit=300)
            if sys_logs:
                df_sys = pd.DataFrame(sys_logs, columns=["User", "Action", "Details", "Timestamp"])
                st.dataframe(df_sys, use_container_width=True, height=400)
                csv_sys = df_sys.to_csv(index=False)
                st.download_button(
                    "📥 Export System Logs", csv_sys,
                    "system_logs.csv", "text/csv", key="export_sys_logs"
                )
            else:
                st.info("No system logs yet.")

    # ---- TAB 3: SYSTEM SETTINGS ----
    with admin_tabs[3]:
        sec_header("SETTINGS", "System Configuration", "Platform controls")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### 📁 Upload Limit")
            new_limit = st.number_input(
                "Max Upload Size (MB)", min_value=100, max_value=5000,
                value=MAX_FILE_SIZE_MB, step=50, key="sys_limit"
            )
            if st.button("💾 Apply New Limit", key="apply_limit"):
                try:
                    CONFIG_DIR.mkdir(exist_ok=True)
                    with open(CONFIG_FILE, "w") as f:
                        f.write(f"[server]\nmaxUploadSize = {new_limit}\n")
                    st.success(f"✅ Upload limit set to {new_limit} MB. Restart app to apply.")
                    log_system_action(
                        st.session_state.get("user_email", "admin"),
                        "change_upload_limit",
                        f"Changed to {new_limit} MB"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

        with col2:
            st.markdown("#### 🗑️ Cache Management")
            st.info("Clear Streamlit cache to reload models and data.")
            if st.button("🧹 Clear All Cache", key="clear_cache"):
                st.cache_data.clear()
                st.cache_resource.clear()
                log_system_action(
                    st.session_state.get("user_email", "admin"),
                    "clear_cache", "Cleared all Streamlit cache"
                )
                st.success("✅ Cache cleared. Please refresh the page.")

        st.markdown("---")
        st.markdown("#### 💾 Database Info")
        db_size = os.path.getsize(DB_PATH) / (1024 * 1024) if os.path.exists(DB_PATH) else 0
        col1, col2, col3 = st.columns(3)
        col1.metric("Database Size", f"{db_size:.3f} MB")
        col2.metric("Total Users", stats["total_users"])
        col3.metric("Total Log Entries", stats["total_actions"])

    # ---- TAB 4: PLATFORM ANALYTICS ----
    with admin_tabs[4]:
        sec_header("ANALYTICS", "Platform Usage Analytics", "Behavior & engagement insights")

        sys_logs = get_system_logs(limit=500)
        if sys_logs:
            df_sys = pd.DataFrame(sys_logs, columns=["User", "Action", "Details", "Timestamp"])
            df_sys["Timestamp"] = pd.to_datetime(df_sys["Timestamp"])
            df_sys["date"] = df_sys["Timestamp"].dt.date

            col1, col2 = st.columns(2)
            with col1:
                action_counts = df_sys["Action"].value_counts().reset_index()
                action_counts.columns = ["Action", "Count"]
                fig = px.pie(
                    action_counts, names="Action", values="Count",
                    title="🎯 Action Distribution",
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig.update_layout(height=380)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                daily_actions = df_sys.groupby("date").size().reset_index(name="actions")
                fig2 = px.bar(
                    daily_actions, x="date", y="actions",
                    title="📅 Daily Platform Activity",
                    color="actions", color_continuous_scale="Purples"
                )
                fig2.update_layout(height=380, template="plotly_white")
                st.plotly_chart(fig2, use_container_width=True)

            # Most active users in the platform
            top_active = df_sys.groupby("User").size().reset_index(name="actions").sort_values("actions", ascending=False).head(10)
            fig3 = px.bar(
                top_active, x="User", y="actions",
                title="🏅 Most Active Users (by Platform Actions)",
                color="actions", color_continuous_scale="Teal"
            )
            fig3.update_layout(height=350, template="plotly_white")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("No platform analytics data yet.")

# ========================== ANALYTICS APP ==========================
def render_analytics_app():
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
        st.markdown("### 📂 Data Source")
        source = st.radio(
            "", ["📦 Built-in Dataset", "📂 Upload File"],
            label_visibility="collapsed", key="data_source"
        )

        if source == "📂 Upload File":
            uploaded = st.file_uploader(
                f"CSV / Excel / JSON (Max {MAX_FILE_SIZE_MB} MB)",
                type=["csv", "xlsx", "xls", "json"],
                key="file_upload"
            )
            if uploaded:
                if uploaded.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"⚠️ File exceeds {MAX_FILE_SIZE_MB} MB. Your file: {uploaded.size / (1024 * 1024):.2f} MB")
                else:
                    progress_bar = st.progress(0, text="Loading file...")
                    try:
                        if uploaded.name.endswith('.csv'):
                            try:
                                df_new, used_enc = read_csv_with_encoding(uploaded)
                                st.success(f"✓ Loaded ({used_enc})")
                            except Exception:
                                manual_enc = st.selectbox("Encoding", ['utf-8', 'windows-1256', 'iso-8859-1', 'cp1252'], key="manual_enc")
                                uploaded.seek(0)
                                df_new = pd.read_csv(uploaded, encoding=manual_enc)
                                st.success(f"✓ Loaded ({manual_enc})")
                        elif uploaded.name.endswith('.json'):
                            df_new = pd.read_json(uploaded)
                        else:
                            df_new = pd.read_excel(uploaded)
                        progress_bar.progress(100, text="Complete!")
                        if st.session_state["source"] != uploaded.name:
                            st.session_state["df_raw"] = df_new
                            st.session_state["roles"] = detect_column_types(df_new)
                            st.session_state["df"] = smart_clean(df_new, st.session_state["roles"])
                            st.session_state["source"] = uploaded.name
                            st.session_state["col_map"] = {}
                            log_system_action(
                                st.session_state.get("user_email", "guest"),
                                "upload_file",
                                f"Uploaded {uploaded.name} ({uploaded.size / (1024 * 1024):.2f} MB)"
                            )
                        st.success(f"✓ {uploaded.name}")
                    except Exception as e:
                        st.error(f"Error loading file: {e}")
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

    tabs = st.tabs([
        "📊 Data Hub", "💰 KPIs", "🔮 Forecasting", "🤖 Profit Optimizer",
        "👥 Segmentation", "🛒 Market Basket", "📈 Advanced Analytics", "📄 Executive Report"
    ])

    # ---------- TAB 0: DATA HUB ----------
    with tabs[0]:
        sec_header("00", "Data Hub", "Quality & Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Rows", f"{len(df):,}")
        col2.metric("Columns", f"{df.shape[1]}")
        col3.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        col4.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

        st.dataframe(df.head(100), use_container_width=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**📂 Column Types**")
            st.json({k: len(v) for k, v in st.session_state["roles"].items()})
        with col2:
            with st.expander("📊 Statistical Summary"):
                st.dataframe(df.describe(include="all").T, use_container_width=True)

    # ---------- TAB 1: KPIs ----------
    with tabs[1]:
        sec_header("01", "Key Performance Indicators", "Revenue & Profit Analysis")
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
            c4.metric("🛒 Avg Order Value", fmt_num(avg_order, prefix="$"))

            if date_col != "—" and date_col in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    df_ts = df.set_index(date_col).resample('ME')[sales_col].sum().reset_index()
                    fig = px.line(df_ts, x=date_col, y=sales_col, title="📅 Monthly Sales Trend", markers=True,
                                  color_discrete_sequence=["#8b5cf6"])
                    fig.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    if cat_col != "—" and cat_col in df.columns:
                        cat_sales = df.groupby(cat_col)[sales_col].sum().reset_index().sort_values(sales_col, ascending=False)
                        fig2 = px.bar(cat_sales, x=cat_col, y=sales_col, title="🏷️ Sales by Category",
                                      color=sales_col, color_continuous_scale="Purples")
                        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig2, use_container_width=True)

            if profit_col != "—" and profit_col in df.columns and cat_col != "—" and cat_col in df.columns:
                profit_cat = df.groupby(cat_col)[[sales_col, profit_col]].sum().reset_index()
                fig3 = px.scatter(profit_cat, x=sales_col, y=profit_col, text=cat_col,
                                  title="💡 Sales vs Profit by Category",
                                  color=profit_col, color_continuous_scale="RdYlGn")
                fig3.update_traces(textposition="top center")
                st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("👈 Map a Sales column in the sidebar to see KPIs.")

    # ---------- TAB 2: FORECASTING ----------
    with tabs[2]:
        sec_header("02", "Demand Forecasting", "AI-powered time series prediction")
        sales_col = st.session_state["col_map"].get("sales", "—")
        date_col = st.session_state["col_map"].get("date", "—")

        if sales_col != "—" and date_col != "—" and sales_col in df.columns and date_col in df.columns:
            col1, col2 = st.columns(2)
            with col1:
                horizon = st.slider("Forecast Horizon (periods)", 3, 36, 12, key="fc_horizon")
            with col2:
                freq_label = st.selectbox("Aggregation", ["Monthly", "Weekly"], index=0, key="fc_freq")
            freq_key = 'ME' if freq_label == "Monthly" else 'W'

            if st.button("🔮 Run Forecast", key="fc_run"):
                with st.spinner("Building forecast model..."):
                    try:
                        date_json = df[date_col].astype(str).to_json()
                        val_json = df[sales_col].to_json()
                        hist, fcast, model_name = build_forecast(date_json, val_json, horizon, freq_key)
                        if hist is None:
                            st.error("❌ Failed to build forecast. Install statsmodels or prophet.")
                        else:
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=hist["Date"], y=hist["Value"], name="Historical",
                                line=dict(color="#06b6d4", width=2.5)
                            ))
                            fig.add_trace(go.Scatter(
                                x=fcast["Date"], y=fcast["Upper"],
                                fill=None, line=dict(color="rgba(139,92,246,0)"), showlegend=False
                            ))
                            fig.add_trace(go.Scatter(
                                x=fcast["Date"], y=fcast["Lower"],
                                fill="tonexty", name="Confidence Band",
                                fillcolor="rgba(139,92,246,0.15)",
                                line=dict(color="rgba(139,92,246,0)")
                            ))
                            fig.add_trace(go.Scatter(
                                x=fcast["Date"], y=fcast["Value"], name=f"Forecast ({model_name})",
                                line=dict(color="#f59e0b", dash="dash", width=2.5)
                            ))
                            fig.update_layout(
                                title=f"📈 Demand Forecast — {model_name}",
                                height=500, template="plotly_white",
                                plot_bgcolor='rgba(0,0,0,0)'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(fcast.round(2), use_container_width=True)
                            log_system_action(
                                st.session_state.get("user_email", "guest"),
                                "forecast", f"Ran {freq_label} forecast, horizon={horizon}"
                            )
                    except Exception as e:
                        st.error(f"Forecast error: {e}")
        else:
            st.info("👈 Map both Sales and Date columns in the sidebar.")

    # ---------- TAB 3: PROFIT OPTIMIZER ----------
    with tabs[3]:
        sec_header("03", "AI Profit Optimizer", "Voting Ensemble ML Model")
        profit_col = st.session_state["col_map"].get("profit", "—")

        if profit_col != "—" and profit_col in df.columns:
            available_features = [c for c in df.columns if c != profit_col]
            features = st.multiselect(
                "Select features for training",
                available_features,
                default=[],
                key="ml_features"
            )
            if features:
                if st.button("🤖 Train Ensemble Model", key="ml_train"):
                    with st.spinner("Training ensemble model (RF + GB + Ridge + ElasticNet)..."):
                        try:
                            train_df = df[features + [profit_col]].dropna()
                            if len(train_df) < 20:
                                st.error("Not enough data (need at least 20 rows after dropping nulls).")
                            else:
                                result = train_ml_ensemble(train_df.to_json(), profit_col, features)
                                ensemble, le_map, scaler, r2, mape, imp, perm_imp, _ = result
                                col1, col2, col3 = st.columns(3)
                                col1.metric("R² Score", f"{r2:.3f}" if r2 is not None else "N/A")
                                col2.metric("MAPE", f"{mape:.1f}%" if mape is not None else "N/A")
                                col3.metric("Training Rows", f"{len(train_df):,}")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.subheader("🌲 Feature Importance (RF)")
                                    imp_df = pd.Series(imp).sort_values(ascending=True)
                                    fig = px.bar(imp_df, orientation="h", title="Feature Importance")
                                    st.plotly_chart(fig, use_container_width=True)
                                with col2:
                                    st.subheader("🔀 Permutation Importance")
                                    perm_df = pd.Series(perm_imp).sort_values(ascending=True)
                                    fig2 = px.bar(perm_df, orientation="h", title="Permutation Importance",
                                                  color_discrete_sequence=["#06b6d4"])
                                    st.plotly_chart(fig2, use_container_width=True)

                                log_system_action(
                                    st.session_state.get("user_email", "guest"),
                                    "train_model",
                                    f"Trained profit model with features: {features}"
                                )
                        except Exception as e:
                            st.error(f"Training error: {e}")
        else:
            st.info("👈 Map a Profit column in the sidebar.")

    # ---------- TAB 4: SEGMENTATION ----------
    with tabs[4]:
        sec_header("04", "Customer Intelligence", "RFM Analysis & Clustering")
        cust_col = st.session_state["col_map"].get("customer", "—")
        sales_col = st.session_state["col_map"].get("sales", "—")
        date_col = st.session_state["col_map"].get("date", "—")

        seg_tab1, seg_tab2 = st.tabs(["👤 RFM Analysis", "🔵 Clustering"])

        with seg_tab1:
            if cust_col != "—" and sales_col != "—" and date_col != "—" and \
               all(c in df.columns for c in [cust_col, sales_col, date_col]):
                if st.button("🔍 Run RFM Analysis", key="rfm_run"):
                    rfm = compute_rfm(df, date_col, sales_col, cust_col)
                    if rfm is not None:
                        st.dataframe(rfm.head(50), use_container_width=True)
                        col1, col2 = st.columns(2)
                        with col1:
                            seg_counts = rfm["Segment"].value_counts()
                            fig = px.pie(seg_counts, names=seg_counts.index, values=seg_counts.values,
                                         title="Customer Segments", color_discrete_sequence=px.colors.qualitative.Pastel)
                            st.plotly_chart(fig, use_container_width=True)
                        with col2:
                            fig2 = px.scatter(
                                rfm, x="Frequency", y="Monetary", color="Segment",
                                size="RFM_Score", title="RFM Scatter",
                                color_discrete_sequence=px.colors.qualitative.Set2
                            )
                            st.plotly_chart(fig2, use_container_width=True)
                        log_system_action(
                            st.session_state.get("user_email", "guest"), "rfm_analysis", "Ran RFM segmentation"
                        )
                    else:
                        st.warning("⚠️ RFM failed: insufficient data or duplicate quantiles.")
            else:
                st.info("👈 Map Customer ID, Sales, and Date columns.")

        with seg_tab2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                method = st.selectbox("Clustering Method", ["kmeans", "dbscan", "hierarchical"], key="clust_method")
                col1, col2 = st.columns(2)
                with col1:
                    if method in ['kmeans', 'hierarchical']:
                        k = st.slider("Number of clusters", 2, 8, 3, key="k_clust")
                    else:
                        eps = st.slider("Epsilon (DBSCAN)", 0.1, 2.0, 0.5, 0.05, key="db_eps")
                        k = 3
                with col2:
                    feat_clust = st.multiselect("Features", num_cols, default=num_cols[:min(3, len(num_cols))], key="clust_feat")

                if feat_clust and st.button("🔵 Run Clustering", key="clust_run"):
                    try:
                        eps_val = eps if method == 'dbscan' else 0.5
                        labels, sil, coords, inertias, var, model = run_advanced_clustering(
                            df[feat_clust].dropna().to_json(), feat_clust, method=method,
                            n_clusters=k, eps=eps_val
                        )
                        if labels is not None:
                            if sil:
                                st.metric("Silhouette Score", f"{sil:.3f}")
                            col1, col2 = st.columns(2)
                            with col1:
                                fig = px.scatter(
                                    x=coords[:, 0], y=coords[:, 1], color=labels.astype(str),
                                    title=f"PCA Projection — {method.upper()}",
                                    labels={"x": f"PC1 ({var[0]*100:.1f}%)", "y": f"PC2 ({var[1]*100:.1f}%)"}
                                )
                                st.plotly_chart(fig, use_container_width=True)
                            with col2:
                                if inertias:
                                    fig2 = px.line(
                                        x=list(inertias.keys()), y=list(inertias.values()),
                                        markers=True, title="📐 Elbow Method",
                                        labels={"x": "K", "y": "Inertia"}
                                    )
                                    st.plotly_chart(fig2, use_container_width=True)
                            log_system_action(
                                st.session_state.get("user_email", "guest"),
                                "clustering", f"Ran {method} clustering"
                            )
                    except Exception as e:
                        st.error(f"Clustering error: {e}")
            else:
                st.info("Need at least 2 numeric columns for clustering.")

    # ---------- TAB 5: MARKET BASKET ----------
    with tabs[5]:
        sec_header("05", "Market Basket Analysis", "Apriori — Association Rules")
        if not MLXTEND_AVAILABLE:
            st.warning("⚠️ Install mlxtend: `pip install mlxtend`")
        else:
            cust_col = st.session_state["col_map"].get("customer", "—")
            prod_col = st.session_state["col_map"].get("product", "—")
            if cust_col != "—" and prod_col != "—" and cust_col in df.columns and prod_col in df.columns:
                col1, col2 = st.columns(2)
                with col1:
                    min_sup = st.slider("Minimum Support", 0.005, 0.1, 0.01, 0.005, key="mb_support")
                with col2:
                    min_lift = st.slider("Minimum Lift", 1.0, 5.0, 1.0, 0.1, key="mb_lift")

                if st.button("🛒 Run Apriori", key="mb_run"):
                    with st.spinner("Running Apriori algorithm..."):
                        try:
                            freq, rules, msg = market_basket_analysis(df, cust_col, prod_col, min_sup)
                            if rules is not None and len(rules) > 0:
                                rules_filtered = rules[rules["lift"] >= min_lift]
                                st.success(f"✅ Found {len(rules_filtered)} association rules.")
                                st.dataframe(
                                    rules_filtered[["antecedents", "consequents", "support", "confidence", "lift"]]
                                    .sort_values("lift", ascending=False),
                                    use_container_width=True
                                )
                                fig = px.scatter(
                                    rules_filtered, x="support", y="confidence", color="lift",
                                    size="lift", title="Support vs Confidence (color=Lift)",
                                    color_continuous_scale="Viridis"
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                log_system_action(
                                    st.session_state.get("user_email", "guest"),
                                    "market_basket", f"Found {len(rules_filtered)} rules"
                                )
                            else:
                                st.info(f"ℹ️ {msg}")
                        except Exception as e:
                            st.error(f"Market basket error: {e}")
            else:
                st.info("👈 Map both Customer ID and Product columns.")

    # ---------- TAB 6: ADVANCED ANALYTICS ----------
    with tabs[6]:
        sec_header("06", "Advanced Analytics", "Anomaly Detection & Deep Insights")

        adv_tab1, adv_tab2, adv_tab3 = st.tabs(["🔗 Correlations", "🚨 Anomaly Detection", "🔍 Data Explorer"])

        with adv_tab1:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_cols) >= 2:
                corr = df[num_cols].corr()
                fig = px.imshow(
                    corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    title="🔗 Correlation Matrix", aspect="auto"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)

                # Pair plot for top 4 numeric
                top4 = num_cols[:min(4, len(num_cols))]
                fig2 = px.scatter_matrix(df[top4].dropna().sample(min(500, len(df))),
                                         title="📊 Scatter Matrix (top 4 numeric)")
                fig2.update_layout(height=600)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns.")

        with adv_tab2:
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            feat_anom = st.multiselect(
                "Select features for anomaly detection",
                num_cols, default=num_cols[:min(3, len(num_cols))], key="anom_feat"
            )
            contamination = st.slider("Contamination (expected anomaly %)", 0.01, 0.2, 0.05, 0.01, key="anom_contam")
            if feat_anom and st.button("🔎 Detect Anomalies", key="anom_run"):
                with st.spinner("Training Isolation Forest..."):
                    try:
                        clean_df = df[feat_anom].dropna()
                        anomalies, model = detect_anomalies_iforest(clean_df.to_json(), feat_anom, contamination)
                        n_anom = int(anomalies.sum())
                        col1, col2 = st.columns(2)
                        col1.metric("🚨 Anomalies Detected", n_anom)
                        col2.metric("📊 Anomaly Rate", f"{n_anom / len(anomalies) * 100:.1f}%")

                        if n_anom > 0 and len(feat_anom) >= 2:
                            fig = px.scatter(
                                x=clean_df.iloc[:, 0],
                                y=clean_df.iloc[:, 1],
                                color=np.where(anomalies, "🔴 Anomaly", "🟢 Normal"),
                                title="Anomaly Visualization",
                                labels={"x": feat_anom[0], "y": feat_anom[1]},
                                color_discrete_map={"🔴 Anomaly": "#ef4444", "🟢 Normal": "#10b981"}
                            )
                            st.plotly_chart(fig, use_container_width=True)

                        if n_anom > 0:
                            st.markdown("**Anomalous Records:**")
                            # Align index properly
                            clean_idx = clean_df.index
                            anom_idx = clean_idx[anomalies]
                            st.dataframe(df.loc[anom_idx].head(50), use_container_width=True)

                        log_system_action(
                            st.session_state.get("user_email", "guest"),
                            "anomaly_detection", f"Detected {n_anom} anomalies"
                        )
                    except Exception as e:
                        st.error(f"Anomaly detection error: {e}")

        with adv_tab3:
            date_col = st.session_state["col_map"].get("date", "—")
            if date_col != "—" and date_col in df.columns:
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                date_range = st.date_input("📅 Date Range", [min_date, max_date], key="date_filter")
                if len(date_range) == 2:
                    mask = (
                        (df[date_col] >= pd.to_datetime(date_range[0])) &
                        (df[date_col] <= pd.to_datetime(date_range[1]))
                    )
                    filtered_df = df[mask]
                    st.write(f"Showing {len(filtered_df):,} of {len(df):,} rows")
                    st.dataframe(filtered_df, use_container_width=True)
                    csv = filtered_df.to_csv(index=False)
                    st.download_button("📥 Export Filtered Data", csv, "filtered_data.csv", "text/csv", key="dl_filtered")
            else:
                st.info("Map a Date column to enable date filtering.")
                st.dataframe(df.sample(min(200, len(df))), use_container_width=True)

    # ---------- TAB 7: EXECUTIVE REPORT ----------
    with tabs[7]:
        sec_header("07", "Executive AI Report", "Comprehensive Performance Summary")

        if st.button("📄 Generate Executive Report", key="report_gen"):
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
                except Exception:
                    pass

            date_range_str = "N/A"
            if date_col != "—" and date_col in df.columns:
                try:
                    date_range_str = f"{df[date_col].min().date()} → {df[date_col].max().date()}"
                except Exception:
                    pass

            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100

            report = f"""# 📈 NEXUS Analytics Pro — Executive Report

## 📋 Dataset Overview
| Metric | Value |
|--------|-------|
| Total Records | {len(df):,} |
| Total Columns | {df.shape[1]} |
| Date Range | {date_range_str} |
| Missing Data | {missing_pct:.2f}% |
| Numeric Features | {len(num_cols)} |

## 💰 Financial Performance
| Metric | Value |
|--------|-------|
| Total Revenue | {fmt_num(total_rev, prefix='$')} |
| Total Profit | {fmt_num(total_profit, prefix='$')} |
| Profit Margin | {margin:.1f}% |
| Avg Order Value | {fmt_num(df[sales_col].mean() if sales_col != '—' and sales_col in df.columns else 0, prefix='$')} |
| Top Category | {top_category} |

## 🧠 AI-Powered Recommendations

1. **Revenue Optimization**: Focus on the `{top_category}` category which drives the highest revenue. Expand product lines and promotional spend here.

2. **Margin Enhancement**: Current profit margin is {margin:.1f}%. Review discount strategies — high discounts in low-margin categories compress profits significantly.

3. **Customer Strategy**: Use RFM segmentation to identify Champions and Loyal customers. Implement tiered loyalty programs to retain high-value segments.

4. **Demand Planning**: Deploy the forecasting model output for inventory optimization. Reduce stockouts in top-performing categories by 15–20%.

5. **Anomaly Monitoring**: Run the Isolation Forest anomaly detector weekly to flag unusual transactions, potential fraud, or data quality issues.

6. **Operational Efficiency**: Analyze shipping days distribution — reducing average shipping time can improve customer ratings and repeat purchase rates.

## ⚠️ Risk Flags
- Monitor failed login attempts (visible in Admin panel) for security posture.
- Watch for categories with declining sales trends over consecutive months.
- High return rates in any category signal quality or listing accuracy issues.

---
*🚀 Report generated by NEXUS Analytics Pro · Powered by AI & ML*
"""
            st.markdown(report)
            st.download_button(
                "📥 Download Report (Markdown)", report,
                "nexus_executive_report.md", "text/markdown", key="dl_report"
            )
            log_system_action(
                st.session_state.get("user_email", "guest"),
                "generate_report", "Executive report generated"
            )

# ========================== MAIN ==========================
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
        st.session_state["user_email"] = None
        st.session_state["is_admin"] = False

    login_section()

    if st.session_state.get("is_admin", False):
        admin_mode = st.sidebar.checkbox("🔧 Admin Control Center", key="admin_mode_switch")
        if admin_mode:
            mega_admin_dashboard()
        else:
            render_analytics_app()
    else:
        render_analytics_app()

if __name__ == "__main__":
    main()
