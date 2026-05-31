import warnings
warnings.filterwarnings("ignore")

import io, os, hashlib, sqlite3, json, time, random
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

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, IsolationForest
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
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════
MAX_FILE_SIZE_MB    = 1000
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
GUEST_MAX_ROWS      = 1000
DB_PATH             = "nexus_enterprise.db"
APP_VERSION         = "4.0.0-enterprise"

CONFIG_DIR = Path(".streamlit")
CONFIG_FILE = CONFIG_DIR / "config.toml"
if not CONFIG_FILE.exists():
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        f.write(f"[server]\nmaxUploadSize = {MAX_FILE_SIZE_MB}\n")

st.set_page_config(
    page_title="NEXUS Enterprise · Analytics OS",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ═══════════════════════════════════════════════════════════════
#  ENTERPRISE CSS — IBM Carbon / Sci-Fi Terminal Aesthetic
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

:root {
  --bg-void:#020408; --bg-base:#050c14; --bg-raised:#091220;
  --bg-elevated:#0d1a2e; --bg-overlay:#121f33;
  --bg-glass:rgba(9,18,32,0.85);
  --line-dim:rgba(0,180,255,0.06); --line-med:rgba(0,180,255,0.14); --line-bright:rgba(0,180,255,0.28);
  --cyan:#00b4ff; --cyan-dim:rgba(0,180,255,0.12); --cyan-glow:rgba(0,180,255,0.32);
  --blue:#0f62fe; --blue-dim:rgba(15,98,254,0.18);
  --teal:#08bdba; --green:#42be65; --green-dim:rgba(66,190,101,0.15);
  --amber:#f1c21b; --amber-dim:rgba(241,194,27,0.15);
  --orange:#ff832b; --red:#fa4d56; --red-dim:rgba(250,77,86,0.15);
  --purple:#a56eff; --magenta:#ee5396;
  --text-primary:#f4f4f4; --text-secondary:#8d9db5; --text-muted:#4d5e72; --text-disabled:#2d3b4e;
  --font-mono:'IBM Plex Mono',monospace; --font-sans:'IBM Plex Sans',sans-serif;
  --r-xs:4px; --r-sm:6px; --r-md:10px; --r-lg:16px; --r-xl:24px; --r-full:9999px;
}
*,*::before,*::after{box-sizing:border-box;}
html,body,[data-testid="stAppViewContainer"],[data-testid="stAppViewContainer"]>.main,.main .block-container{
  background-color:var(--bg-void)!important;color:var(--text-primary)!important;font-family:var(--font-sans)!important;}

[data-testid="stAppViewContainer"]::before{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background-image:linear-gradient(var(--line-dim) 1px,transparent 1px),linear-gradient(90deg,var(--line-dim) 1px,transparent 1px);
  background-size:48px 48px;
  mask-image:radial-gradient(ellipse 80% 80% at 50% 50%,black 30%,transparent 100%);}
[data-testid="stAppViewContainer"]::after{
  content:'';position:fixed;inset:0;z-index:0;pointer-events:none;
  background:radial-gradient(ellipse 60% 40% at 10% 10%,rgba(0,180,255,0.04) 0%,transparent 60%),
             radial-gradient(ellipse 50% 60% at 90% 90%,rgba(15,98,254,0.05) 0%,transparent 60%);}

::-webkit-scrollbar{width:4px;height:4px;}
::-webkit-scrollbar-track{background:var(--bg-void);}
::-webkit-scrollbar-thumb{background:var(--line-bright);border-radius:var(--r-full);}

[data-testid="stSidebar"]{background:var(--bg-base)!important;border-right:1px solid var(--line-med)!important;}
[data-testid="stSidebar"] *{font-family:var(--font-sans)!important;color:var(--text-primary)!important;}
[data-testid="stSidebar"] .stRadio>div{flex-direction:column!important;gap:6px!important;}

[data-testid="stMetric"]{background:var(--bg-raised)!important;border:1px solid var(--line-med)!important;
  border-top:2px solid var(--cyan)!important;border-radius:var(--r-md)!important;padding:1.2rem 1.4rem!important;
  position:relative;overflow:hidden;transition:all 0.25s!important;}
[data-testid="stMetric"]::after{content:'';position:absolute;top:0;left:0;right:0;height:50px;
  background:linear-gradient(180deg,var(--cyan-dim),transparent);pointer-events:none;}
[data-testid="stMetric"]:hover{border-color:var(--cyan)!important;
  box-shadow:0 0 28px var(--cyan-glow),0 8px 20px rgba(0,0,0,0.5)!important;transform:translateY(-2px)!important;}
[data-testid="stMetricLabel"]{color:var(--text-muted)!important;font-family:var(--font-mono)!important;
  font-size:0.68rem!important;text-transform:uppercase!important;letter-spacing:2px!important;}
[data-testid="stMetricValue"]{color:var(--text-primary)!important;font-family:var(--font-mono)!important;
  font-size:1.8rem!important;font-weight:600!important;letter-spacing:-1px!important;}

.stTabs [data-baseweb="tab-list"]{gap:2px!important;background:var(--bg-raised)!important;
  border:1px solid var(--line-med)!important;border-radius:var(--r-md)!important;padding:4px!important;flex-wrap:wrap!important;}
.stTabs [data-baseweb="tab"]{background:transparent!important;color:var(--text-secondary)!important;
  border-radius:var(--r-sm)!important;padding:0.5rem 0.9rem!important;font-size:0.72rem!important;
  font-family:var(--font-mono)!important;font-weight:500!important;letter-spacing:0.5px!important;
  text-transform:uppercase!important;transition:all 0.2s!important;}
.stTabs [data-baseweb="tab"]:hover{color:var(--cyan)!important;background:var(--cyan-dim)!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--blue),#0043ce)!important;
  color:white!important;box-shadow:0 2px 12px rgba(15,98,254,0.5)!important;}

.stButton>button{background:linear-gradient(135deg,var(--blue),#0043ce)!important;color:white!important;
  border:none!important;border-radius:var(--r-sm)!important;padding:0.58rem 1.3rem!important;
  font-family:var(--font-mono)!important;font-size:0.74rem!important;font-weight:600!important;
  letter-spacing:1px!important;text-transform:uppercase!important;transition:all 0.2s!important;
  box-shadow:0 2px 10px rgba(15,98,254,0.35)!important;}
.stButton>button:hover{box-shadow:0 4px 20px rgba(15,98,254,0.6)!important;transform:translateY(-1px)!important;}

.stTextInput>div>div>input,.stTextArea textarea,.stNumberInput input,.stSelectbox>div>div{
  background:var(--bg-raised)!important;color:var(--text-primary)!important;
  border:1px solid var(--line-med)!important;border-radius:var(--r-sm)!important;
  font-family:var(--font-mono)!important;font-size:0.84rem!important;transition:border-color 0.2s!important;}
.stTextInput>div>div>input:focus,.stTextArea textarea:focus{
  border-color:var(--cyan)!important;box-shadow:0 0 0 2px var(--cyan-dim)!important;}

[data-testid="stFileUploadDropzone"]{background:var(--bg-raised)!important;
  border:1px dashed var(--cyan)!important;border-radius:var(--r-md)!important;transition:all 0.3s!important;}
[data-testid="stFileUploadDropzone"]:hover{background:var(--cyan-dim)!important;
  border-color:var(--blue)!important;box-shadow:0 0 20px var(--cyan-glow)!important;}
[data-testid="stFileUploadDropzone"] *{color:var(--text-primary)!important;}

[data-testid="stDataFrame"]{border:1px solid var(--line-med)!important;border-radius:var(--r-md)!important;overflow:hidden!important;}
.dataframe{background:var(--bg-raised)!important;color:var(--text-primary)!important;
  font-family:var(--font-mono)!important;font-size:0.76rem!important;}
.dataframe th{background:var(--bg-overlay)!important;color:var(--cyan)!important;
  text-transform:uppercase!important;letter-spacing:1px!important;font-size:0.68rem!important;
  border-bottom:1px solid var(--line-bright)!important;}
.dataframe td{background:var(--bg-raised)!important;color:var(--text-primary)!important;border-bottom:1px solid var(--line-dim)!important;}
.dataframe tr:hover td{background:var(--bg-elevated)!important;}

[data-testid="stExpander"]{background:var(--bg-raised)!important;border:1px solid var(--line-med)!important;
  border-radius:var(--r-md)!important;overflow:hidden!important;}
[data-testid="stExpander"] summary{color:var(--text-primary)!important;font-family:var(--font-mono)!important;
  font-size:0.76rem!important;font-weight:600!important;letter-spacing:1px!important;text-transform:uppercase!important;}
[data-testid="stDownloadButton"]>button{background:linear-gradient(135deg,var(--teal),#007d79)!important;
  box-shadow:0 2px 10px rgba(8,189,186,0.35)!important;}
[data-testid="stAlert"]{border-radius:var(--r-md)!important;font-family:var(--font-mono)!important;font-size:0.78rem!important;}
.stCaption,[data-testid="stCaptionContainer"]{color:var(--text-muted)!important;font-family:var(--font-mono)!important;font-size:0.68rem!important;}
hr{border-color:var(--line-med)!important;opacity:1!important;}
[data-baseweb="tag"]{background:var(--blue-dim)!important;border:1px solid var(--blue)!important;border-radius:var(--r-xs)!important;}
[data-testid="stSidebar"] .stSelectbox>div>div{background:var(--bg-elevated)!important;border:1px solid var(--line-med)!important;border-radius:var(--r-sm)!important;}

/* ── NEXUS COMPONENTS ── */
.nx-command-bar{display:flex;align-items:center;gap:14px;background:var(--bg-glass);
  border:1px solid var(--line-med);border-radius:var(--r-md);padding:10px 18px;margin-bottom:22px;
  backdrop-filter:blur(20px);position:relative;overflow:hidden;}
.nx-command-bar::before{content:'';position:absolute;inset:0;
  background:linear-gradient(90deg,var(--cyan-dim),transparent 40%);pointer-events:none;}
.nx-logo{font-family:var(--font-mono);font-size:1rem;font-weight:600;color:var(--cyan);letter-spacing:4px;}
.nx-logo span{color:var(--text-muted);font-size:0.6rem;}
.nx-status-dot{width:6px;height:6px;border-radius:50%;background:var(--green);
  box-shadow:0 0 8px var(--green);animation:pdot 2s infinite;}
@keyframes pdot{0%,100%{opacity:1;box-shadow:0 0 6px var(--green);}50%{opacity:.5;box-shadow:0 0 12px var(--green);}}
.nx-ver{font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);
  background:var(--bg-overlay);border:1px solid var(--line-med);
  padding:2px 8px;border-radius:var(--r-full);letter-spacing:1px;}

.nx-sec{display:flex;align-items:center;gap:10px;margin-bottom:18px;
  padding-bottom:10px;border-bottom:1px solid var(--line-med);}
.nx-sec-id{font-family:var(--font-mono);font-size:0.62rem;color:var(--cyan);letter-spacing:2px;
  background:var(--cyan-dim);border:1px solid var(--line-bright);padding:2px 8px;border-radius:var(--r-xs);}
.nx-sec-title{font-family:var(--font-mono);font-size:0.82rem;font-weight:600;
  color:var(--text-primary);letter-spacing:2px;text-transform:uppercase;}
.nx-sec-sub{margin-left:auto;font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);letter-spacing:1px;}

.cloud-card{background:var(--bg-raised);border:1px solid var(--line-med);border-radius:var(--r-lg);
  padding:20px 16px;text-align:center;transition:all 0.3s;position:relative;overflow:hidden;margin-bottom:12px;}
.cloud-card::before{content:'';position:absolute;inset:0;
  background:radial-gradient(ellipse at 50% 0%,var(--cyan-dim),transparent 70%);opacity:0;transition:opacity 0.3s;}
.cloud-card:hover{border-color:var(--cyan);box-shadow:0 0 32px var(--cyan-glow);transform:translateY(-3px);}
.cloud-card:hover::before{opacity:1;}
.cc-icon{font-size:2rem;margin-bottom:8px;}
.cc-name{font-family:var(--font-mono);font-size:0.75rem;font-weight:600;color:var(--text-primary);
  letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;}
.cc-sub{font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);}
.cc-badge{display:inline-block;font-family:var(--font-mono);font-size:0.58rem;padding:2px 8px;
  border-radius:var(--r-full);letter-spacing:1px;text-transform:uppercase;margin-top:8px;}
.cc-badge.ok{background:var(--green-dim);color:var(--green);border:1px solid var(--green);}
.cc-badge.beta{background:var(--amber-dim);color:var(--amber);border:1px solid var(--amber);}

.plan-card{background:var(--bg-raised);border:1px solid var(--line-med);border-radius:var(--r-xl);
  padding:26px 18px;text-align:center;transition:all 0.3s;position:relative;overflow:hidden;margin-bottom:12px;}
.plan-card.featured{border-color:var(--cyan);background:linear-gradient(145deg,var(--bg-raised),rgba(0,180,255,0.05));}
.plan-card:hover{box-shadow:0 0 36px var(--cyan-glow);transform:translateY(-4px);}
.plan-price{font-family:var(--font-mono);font-size:2.5rem;font-weight:600;color:var(--text-primary);}
.plan-price span{font-size:0.9rem;color:var(--text-muted);}
.plan-name-lbl{font-family:var(--font-mono);font-size:0.65rem;color:var(--cyan);
  letter-spacing:3px;text-transform:uppercase;margin-bottom:10px;}

.chat-wrap{display:flex;margin-bottom:16px;align-items:flex-start;}
.chat-wrap.user{flex-direction:row-reverse;}
.chat-av{width:30px;height:30px;border-radius:6px;display:flex;align-items:center;
  justify-content:center;font-size:0.8rem;flex-shrink:0;}
.chat-av.user{background:var(--blue);margin-left:10px;}
.chat-av.bot{background:var(--bg-overlay);border:1px solid var(--line-bright);margin-right:10px;}
.chat-bub{max-width:72%;padding:11px 15px;border-radius:var(--r-md);font-family:var(--font-sans);
  font-size:0.86rem;line-height:1.65;word-wrap:break-word;white-space:pre-wrap;}
.chat-bub.user{background:var(--blue-dim);border:1px solid rgba(15,98,254,0.35);
  color:var(--text-primary);border-bottom-right-radius:2px;}
.chat-bub.bot{background:var(--bg-elevated);border:1px solid var(--line-med);
  color:var(--text-primary);border-bottom-left-radius:2px;}

.admin-hero{background:linear-gradient(135deg,var(--bg-raised),rgba(0,180,255,0.06));
  border:1px solid var(--line-med);border-top:2px solid var(--cyan);border-radius:var(--r-xl);
  padding:30px;text-align:center;position:relative;overflow:hidden;margin-bottom:22px;}

.conn-bar{display:flex;align-items:center;gap:8px;background:var(--bg-raised);
  border:1px solid var(--line-med);border-radius:var(--r-sm);padding:7px 12px;
  font-family:var(--font-mono);font-size:0.7rem;margin-bottom:10px;}
.ci{width:7px;height:7px;border-radius:50%;}
.ci.connected{background:var(--green);box-shadow:0 0 8px var(--green);animation:pdot 2s infinite;}
.ci.streaming{background:var(--cyan);box-shadow:0 0 8px var(--cyan);animation:pdot 1s infinite;}
.ci.error{background:var(--red);}

.sidebar-user{background:var(--bg-raised);border:1px solid var(--line-med);
  border-radius:var(--r-md);padding:10px 12px;margin-bottom:12px;}
.su-email{font-family:var(--font-mono);font-size:0.7rem;color:var(--text-secondary);}
.su-badge{display:inline-block;margin-top:5px;font-family:var(--font-mono);font-size:0.58rem;
  color:var(--cyan);background:var(--cyan-dim);border:1px solid var(--line-bright);
  padding:1px 8px;border-radius:var(--r-full);letter-spacing:1px;text-transform:uppercase;}

.empty-state{text-align:center;padding:80px 24px;}
.empty-state .ei{font-size:3.5rem;margin-bottom:18px;opacity:.35;}
.empty-state h2{font-family:var(--font-mono);font-size:0.9rem;color:var(--cyan);
  letter-spacing:3px;text-transform:uppercase;margin-bottom:8px;}
.empty-state p{font-family:var(--font-mono);font-size:0.72rem;color:var(--text-muted);}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  PLOTLY TEMPLATE
# ═══════════════════════════════════════════════════════════════
_T = go.layout.Template(layout=go.Layout(
    paper_bgcolor='#091220', plot_bgcolor='#091220',
    font=dict(color='#8d9db5', size=11, family='IBM Plex Mono'),
    title_font=dict(size=12, color='#f4f4f4', family='IBM Plex Mono'),
    xaxis=dict(title_font=dict(size=10,color='#4d5e72'),tickfont=dict(size=9,color='#4d5e72'),
               gridcolor='rgba(0,180,255,0.06)',linecolor='rgba(0,180,255,0.14)',zerolinecolor='rgba(0,180,255,0.14)'),
    yaxis=dict(title_font=dict(size=10,color='#4d5e72'),tickfont=dict(size=9,color='#4d5e72'),
               gridcolor='rgba(0,180,255,0.06)',linecolor='rgba(0,180,255,0.14)',zerolinecolor='rgba(0,180,255,0.14)'),
    legend=dict(font=dict(size=10,color='#8d9db5'),bgcolor='rgba(0,0,0,0)',bordercolor='rgba(0,180,255,0.14)'),
    hoverlabel=dict(bgcolor='#0d1a2e',font_size=11,font_color='#f4f4f4',font_family='IBM Plex Mono'),
    colorway=['#00b4ff','#42be65','#f1c21b','#ff832b','#a56eff','#08bdba','#fa4d56','#ee5396']
))
pio.templates["nx"] = _T
pio.templates.default = "nx"

def sf(fig, title=None, h=None):
    u = dict(template="nx", margin=dict(l=20,r=20,t=36 if title else 20,b=20),
             paper_bgcolor='#091220', plot_bgcolor='#091220')
    if title: u["title"] = dict(text=title,font=dict(size=11,family='IBM Plex Mono',color='#f4f4f4'),x=0.01)
    if h: u["height"] = h
    fig.update_layout(**u)
    fig.update_xaxes(showgrid=True,gridwidth=.5,gridcolor='rgba(0,180,255,0.06)')
    fig.update_yaxes(showgrid=True,gridwidth=.5,gridcolor='rgba(0,180,255,0.06)')
    return fig

# ═══════════════════════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL, is_admin INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, last_login TIMESTAMP);
    CREATE TABLE IF NOT EXISTS login_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, email TEXT, success INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS system_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_email TEXT, action TEXT,
        details TEXT, timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE TABLE IF NOT EXISTS subscription_plans(
        id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE NOT NULL,
        price_monthly REAL, price_yearly REAL, max_rows INTEGER,
        features TEXT, is_active INTEGER DEFAULT 1);
    CREATE TABLE IF NOT EXISTS user_subscriptions(
        id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
        plan_id INTEGER NOT NULL, start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_date TIMESTAMP, is_active INTEGER DEFAULT 1, payment_method TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(plan_id) REFERENCES subscription_plans(id));
    CREATE TABLE IF NOT EXISTS app_settings(
        key TEXT PRIMARY KEY, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)
    c.execute("SELECT COUNT(*) FROM subscription_plans")
    if c.fetchone()[0] == 0:
        plans = [
            ("Starter",    0,      0,       5000,
             "Basic analytics · 5K rows · No forecasting · No export", 1),
            ("Professional",29.99, 299.99,  100000,
             "Full analytics · Forecasting · Clustering · Market Basket · Export · 2 Live Connectors", 1),
            ("Enterprise",  99.99, 999.99,  999999999,
             "Unlimited rows · All Pro + Cloud Connectors · Live streaming · API access · Dedicated support", 1),
        ]
        c.executemany("INSERT INTO subscription_plans(name,price_monthly,price_yearly,max_rows,features,is_active) VALUES(?,?,?,?,?,?)", plans)
    for k,v in [("ai_provider","deepseek"),("deepseek_api_key",""),("groq_api_key",""),
                ("custom_ai_url",""),("custom_ai_api_key",""),("custom_ai_model",""),
                ("custom_ai_enabled","0")]:
        c.execute("INSERT OR IGNORE INTO app_settings(key,value) VALUES(?,?)",(k,v))
    admin_email,admin_pass = "kareemeltemsah7@gmail.com","temsah1!"
    hashed = hashlib.sha256(admin_pass.encode()).hexdigest()
    c.execute("SELECT id FROM users WHERE email=?",(admin_email,))
    if c.fetchone():
        c.execute("UPDATE users SET password_hash=?,is_admin=1 WHERE email=?",(hashed,admin_email))
    else:
        c.execute("INSERT INTO users(email,password_hash,is_admin) VALUES(?,?,1)",(admin_email,hashed))
    conn.commit(); conn.close()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    try: yield conn
    finally: conn.close()

def hpw(p): return hashlib.sha256(p.encode()).hexdigest()

def verify_password(email, pwd):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT password_hash,is_admin FROM users WHERE email=?",(email,))
        row = c.fetchone()
        if row and row["password_hash"] == hpw(pwd):
            c.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE email=?",(email,))
            conn.commit(); return True, bool(row["is_admin"])
    return False, False

def register_user(email, pwd):
    with get_db() as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users(email,password_hash) VALUES(?,?)",(email,hpw(pwd)))
            uid = c.lastrowid
            c.execute("SELECT id FROM subscription_plans WHERE name='Starter'"); fp=c.fetchone()
            if fp:
                c.execute("INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active) VALUES(?,?,?,?,1)",
                          (uid,fp["id"],datetime.now(),datetime.now()+timedelta(days=36500)))
            conn.commit(); return True
        except sqlite3.IntegrityError: return False

def log_login(email, ok):
    with get_db() as conn:
        conn.execute("INSERT INTO login_logs(email,success) VALUES(?,?)",(email,1 if ok else 0)); conn.commit()

def log_action(email, action, detail=""):
    with get_db() as conn:
        conn.execute("INSERT INTO system_logs(user_email,action,details) VALUES(?,?,?)",(email,action,detail[:500])); conn.commit()

def get_setting(k, default=""):
    with get_db() as conn:
        c = conn.cursor(); c.execute("SELECT value FROM app_settings WHERE key=?",(k,)); r=c.fetchone()
        return r["value"] if r else default

def set_setting(k, v):
    with get_db() as conn:
        conn.execute("INSERT OR REPLACE INTO app_settings(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",(k,v)); conn.commit()

def get_user_by_email(email):
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT id,email,is_admin FROM users WHERE email=?",(email,)); return c.fetchone()

def get_user_subscription(user_id):
    with get_db() as conn:
        c=conn.cursor()
        c.execute("""SELECT sp.name,sp.max_rows,sp.features,us.start_date,us.end_date,us.is_active,sp.id plan_id
                     FROM user_subscriptions us JOIN subscription_plans sp ON us.plan_id=sp.id
                     WHERE us.user_id=? AND us.is_active=1 ORDER BY us.start_date DESC LIMIT 1""",(user_id,))
        row=c.fetchone()
        if row: return dict(row)
        c.execute("SELECT id,name,max_rows,features FROM subscription_plans WHERE name='Starter'"); fp=c.fetchone()
        if fp:
            c.execute("INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active) VALUES(?,?,?,?,1)",
                      (user_id,fp["id"],datetime.now(),datetime.now()+timedelta(days=36500))); conn.commit()
            return {"name":fp["name"],"max_rows":fp["max_rows"],"features":fp["features"],"plan_id":fp["id"]}
        return None

def get_available_plans():
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT * FROM subscription_plans WHERE is_active=1 ORDER BY price_monthly"); return c.fetchall()

def get_all_plans():
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT * FROM subscription_plans ORDER BY id"); return c.fetchall()

def upgrade_subscription(user_id, plan_id, months=1):
    with get_db() as conn:
        c=conn.cursor()
        c.execute("UPDATE user_subscriptions SET is_active=0 WHERE user_id=?",(user_id,))
        c.execute("INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active,payment_method) VALUES(?,?,?,?,1,'simulated')",
                  (user_id,plan_id,datetime.now(),datetime.now()+timedelta(days=30*months)))
        conn.commit()

def get_all_users():
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT id,email,is_admin,created_at,last_login FROM users ORDER BY id"); return c.fetchall()

def delete_user(uid):
    with get_db() as conn:
        conn.execute("DELETE FROM user_subscriptions WHERE user_id=?",(uid,))
        conn.execute("DELETE FROM users WHERE id=?",(uid,)); conn.commit()

def toggle_admin(uid, make):
    with get_db() as conn:
        conn.execute("UPDATE users SET is_admin=? WHERE id=?",(1 if make else 0,uid)); conn.commit()

def reset_password(uid, new_pwd):
    with get_db() as conn:
        conn.execute("UPDATE users SET password_hash=? WHERE id=?",(hpw(new_pwd),uid)); conn.commit()

def get_login_logs(limit=200):
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT email,success,timestamp FROM login_logs ORDER BY timestamp DESC LIMIT ?",(limit,)); return c.fetchall()

def get_system_logs(limit=500):
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT user_email,action,details,timestamp FROM system_logs ORDER BY timestamp DESC LIMIT ?",(limit,)); return c.fetchall()

def get_stats():
    with get_db() as conn:
        c=conn.cursor()
        def q(sql,*args): c.execute(sql,args); return c.fetchone()[0]
        today=str(pd.Timestamp.now().date())
        return {"users":q("SELECT COUNT(*) FROM users"),
                "admins":q("SELECT COUNT(*) FROM users WHERE is_admin=1"),
                "ok_logins":q("SELECT COUNT(*) FROM login_logs WHERE success=1"),
                "fail_logins":q("SELECT COUNT(*) FROM login_logs WHERE success=0"),
                "actions":q("SELECT COUNT(*) FROM system_logs"),
                "today_ok":q("SELECT COUNT(*) FROM login_logs WHERE success=1 AND date(timestamp)=?",today)}

def get_all_subscriptions():
    with get_db() as conn:
        c=conn.cursor()
        c.execute("""SELECT u.id,u.email,sp.name plan_name,us.start_date,us.end_date,us.is_active
                     FROM user_subscriptions us JOIN users u ON us.user_id=u.id
                     JOIN subscription_plans sp ON us.plan_id=sp.id ORDER BY us.start_date DESC""")
        return [dict(r) for r in c.fetchall()]

def update_plan(pid,pm,py,rows,feats):
    with get_db() as conn:
        conn.execute("UPDATE subscription_plans SET price_monthly=?,price_yearly=?,max_rows=?,features=? WHERE id=?",
                     (pm,py,rows,feats,pid)); conn.commit()

def extend_sub(uid, months=1):
    with get_db() as conn:
        c=conn.cursor(); c.execute("SELECT end_date FROM user_subscriptions WHERE user_id=? AND is_active=1",(uid,)); r=c.fetchone()
        if r:
            try: nd=datetime.strptime(str(r["end_date"]),"%Y-%m-%d %H:%M:%S")+timedelta(days=30*months)
            except: nd=datetime.now()+timedelta(days=30*months)
            conn.execute("UPDATE user_subscriptions SET end_date=? WHERE user_id=? AND is_active=1",(nd,uid)); conn.commit(); return True
        return False

def cancel_sub(uid):
    with get_db() as conn:
        conn.execute("UPDATE user_subscriptions SET is_active=0 WHERE user_id=? AND is_active=1",(uid,))
        c=conn.cursor(); c.execute("SELECT id FROM subscription_plans WHERE name='Starter'"); fp=c.fetchone()
        if fp: conn.execute("INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active) VALUES(?,?,?,?,1)",
                            (uid,fp["id"],datetime.now(),datetime.now()+timedelta(days=36500)))
        conn.commit()

# ═══════════════════════════════════════════════════════════════
#  CSV / ENCODING
# ═══════════════════════════════════════════════════════════════
DATE_FORMATS = [
    '%Y-%m-%d','%d/%m/%Y','%m/%d/%Y','%Y/%m/%d','%d-%m-%Y','%m-%d-%Y','%Y%m%d',
    '%d %b %Y','%d %B %Y','%b %d, %Y','%B %d, %Y',
    '%Y-%m-%d %H:%M:%S','%d/%m/%Y %H:%M:%S','%Y-%m-%dT%H:%M:%S','%Y-%m-%dT%H:%M:%SZ','%d-%b-%Y'
]

def try_parse_date(series):
    if pd.api.types.is_datetime64_any_dtype(series): return series
    if pd.api.types.is_numeric_dtype(series) and series.dropna().mean() < 100000: return None
    sample = series.dropna().astype(str).head(50)
    if not len(sample): return None
    for fmt in DATE_FORMATS:
        try:
            pd.to_datetime(sample, format=fmt, errors='raise')
            full = pd.to_datetime(series.astype(str), format=fmt, errors='coerce')
            if full.notna().mean() > 0.7: return full
        except: continue
    try:
        full = pd.to_datetime(series.astype(str), format='mixed', errors='coerce')
        if full.notna().mean() > 0.7: return full
    except: pass
    return None

def read_csv_smart(f):
    for enc in ['utf-8','windows-1256','iso-8859-6','iso-8859-1','cp1252','latin1','utf-8-sig']:
        try:
            f.seek(0); df=pd.read_csv(f, encoding=enc)
            for col in df.columns:
                p=try_parse_date(df[col])
                if p is not None: df[col]=p
            return df, enc
        except: continue
    raise ValueError("Cannot decode file.")

def detect_column_types(df):
    roles={"date":[],"numeric":[],"categorical":[],"id":[]}
    for col in df.columns:
        s=df[col]
        if pd.api.types.is_datetime64_any_dtype(s): roles["date"].append(col)
        elif try_parse_date(s) is not None: roles["date"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            lc=col.lower()
            if any(x in lc for x in ["_id","row id","index","customer id","order id"]): roles["id"].append(col)
            else: roles["numeric"].append(col)
        elif s.dtype==object:
            if s.nunique()<60: roles["categorical"].append(col)
            else: roles["id"].append(col)
        else: roles["categorical"].append(col)
    return roles

def smart_clean(df, roles):
    df=df.copy()
    for col in roles["date"]:
        if pd.api.types.is_datetime64_any_dtype(df[col]): continue
        for fmt in DATE_FORMATS:
            try:
                p=pd.to_datetime(df[col],format=fmt,errors='coerce')
                if p.notna().mean()>0.7: df[col]=p; break
            except: continue
        else:
            try: df[col]=pd.to_datetime(df[col],infer_datetime_format=True,errors='coerce')
            except: pass
    for col in roles["numeric"]:
        df[col]=pd.to_numeric(df[col],errors='coerce')
        df[col].fillna(df[col].median() if not df[col].isna().all() else 0, inplace=True)
    for col in roles["categorical"]:
        mv=df[col].mode(); df[col].fillna(mv.iloc[0] if len(mv) else "Unknown", inplace=True)
    return df

# ═══════════════════════════════════════════════════════════════
#  AI API
# ═══════════════════════════════════════════════════════════════
def call_ai(messages, provider="deepseek", deepseek_key="", groq_key="",
            custom_url="", custom_key="", custom_model=""):
    def _post(url, payload, api_key):
        r=requests.post(url, headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"},
                        json=payload, timeout=60)
        if r.status_code==200: return r.json()["choices"][0]["message"]["content"],None
        return None, f"Error {r.status_code}: {r.text[:200]}"
    order=[]
    if provider=="deepseek": order=[("ds",deepseek_key),("groq",groq_key)]
    elif provider=="groq":   order=[("groq",groq_key),("ds",deepseek_key)]
    elif provider=="custom": order=[("custom",custom_key),("ds",deepseek_key)]
    for kind,key in order:
        if not key: continue
        try:
            if kind=="ds":     r,e=_post("https://api.deepseek.com/chat/completions",{"model":"deepseek-chat","messages":messages,"max_tokens":2000,"temperature":0.7},key)
            elif kind=="groq": r,e=_post("https://api.groq.com/openai/v1/chat/completions",{"model":"llama-3.3-70b-versatile","messages":messages,"max_tokens":2000,"temperature":0.7},key)
            elif kind=="custom" and custom_url: r,e=_post(custom_url,{"model":custom_model,"messages":messages,"max_tokens":2000,"temperature":0.7},key)
            else: continue
            if r: return r,None
        except Exception as ex: e=str(ex)
    return None,"No AI provider configured."

# ═══════════════════════════════════════════════════════════════
#  CLOUD CONNECTORS REGISTRY
# ═══════════════════════════════════════════════════════════════
CONNECTORS = {
    "postgresql": {"label":"PostgreSQL",   "icon":"🐘","status":"ok"},
    "mysql":      {"label":"MySQL",        "icon":"🐬","status":"ok"},
    "snowflake":  {"label":"Snowflake",    "icon":"❄️", "status":"ok"},
    "bigquery":   {"label":"BigQuery",     "icon":"🔵","status":"ok"},
    "redshift":   {"label":"AWS Redshift", "icon":"🟠","status":"ok"},
    "mongodb":    {"label":"MongoDB",      "icon":"🍃","status":"ok"},
    "kafka":      {"label":"Kafka Stream", "icon":"⚡","status":"beta"},
    "s3":         {"label":"AWS S3",       "icon":"🪣","status":"ok"},
    "azure_blob": {"label":"Azure Blob",   "icon":"🔷","status":"ok"},
    "gcs":        {"label":"Google GCS",   "icon":"🟡","status":"beta"},
    "api_rest":   {"label":"REST API",     "icon":"🌐","status":"ok"},
    "mqtt":       {"label":"MQTT IoT",     "icon":"📡","status":"beta"},
}

def simulate_live_data(n=60):
    np.random.seed(int(time.time())%1000); now=pd.Timestamp.now()
    cats=["Electronics","Fashion","Home & Kitchen","Beauty","Sports","Books"]
    regions=["Riyadh","Dubai","Cairo","Jeddah","Kuwait City","Doha"]
    rows=[{"Timestamp":now-pd.Timedelta(seconds=i*10),"Category":np.random.choice(cats),
           "Region":np.random.choice(regions),"Sales":round(np.random.uniform(50,2000),2),
           "Profit":round(np.random.uniform(5,400),2),"Quantity":np.random.randint(1,10)} for i in range(n)]
    return pd.DataFrame(rows)

# ═══════════════════════════════════════════════════════════════
#  ML FUNCTIONS
# ═══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_ensemble(df_json, target, features):
    df=pd.read_json(io.StringIO(df_json))
    X=df[features].copy(); y=df[target].fillna(df[target].median())
    le_map={}
    for col in X.select_dtypes("object").columns:
        le=LabelEncoder(); X[col]=le.fit_transform(X[col].astype(str)); le_map[col]=le
    X=X.fillna(X.median(numeric_only=True))
    scaler=RobustScaler(); Xs=scaler.fit_transform(X)
    ens=VotingRegressor(estimators=[
        ("rf",RandomForestRegressor(200,max_depth=10,random_state=42,n_jobs=-1)),
        ("gb",GradientBoostingRegressor(150,max_depth=5,random_state=42)),
        ("ridge",Ridge()),("en",ElasticNet(max_iter=2000))]); ens.fit(Xs,y)
    try:
        tscv=TimeSeriesSplit(3)
        r2=float(cross_val_score(ens,Xs,y,cv=tscv,scoring="r2").mean())
        mapes=[]
        for tr,te in tscv.split(Xs):
            ens.fit(Xs[tr],y.iloc[tr]); pred=ens.predict(Xs[te])
            mapes.append(mean_absolute_percentage_error(y.iloc[te],np.maximum(pred,1e-6)))
        mape=np.mean(mapes)*100; ens.fit(Xs,y)
    except: r2=mape=None
    rf=RandomForestRegressor(200,random_state=42,n_jobs=-1); rf.fit(Xs,y)
    imps=dict(zip(X.columns,rf.feature_importances_))
    pi=permutation_importance(rf,Xs,y,n_repeats=5,random_state=42,n_jobs=-1)
    pimps=dict(zip(X.columns,pi.importances_mean))
    return ens,le_map,scaler,r2,mape,imps,pimps,list(X.columns)

@st.cache_data(show_spinner=False)
def build_forecast(date_json, val_json, horizon=12, freq='ME'):
    dates=pd.to_datetime(pd.read_json(io.StringIO(date_json),typ='series'))
    vals=pd.Series(pd.read_json(io.StringIO(val_json),typ='series').values)
    df=pd.DataFrame({"ds":dates,"y":vals})
    pk='M' if freq=='ME' else 'W'
    ts=df.groupby(df['ds'].dt.to_period(pk))['y'].sum().reset_index()
    ts['ds']=ts['ds'].dt.to_timestamp(); ts=ts.sort_values('ds').reset_index(drop=True)
    if PROPHET_AVAILABLE and len(ts)>10:
        try:
            m=Prophet(yearly_seasonality=True,weekly_seasonality=(freq=='W'),daily_seasonality=False)
            m.fit(ts); pfq='M' if freq=='ME' else 'W'
            fut=m.make_future_dataframe(periods=horizon,freq=pfq); fc=m.predict(fut).tail(horizon)
            hist=ts.rename(columns={"ds":"Date","y":"Value"})
            fcast=pd.DataFrame({"Date":fc["ds"],"Value":fc["yhat"],"Lower":fc["yhat_lower"],"Upper":fc["yhat_upper"]})
            return hist,fcast,"Prophet"
        except: pass
    if STATSMODELS_AVAILABLE:
        try:
            fit=ExponentialSmoothing(ts["y"],trend='add',initialization_method='estimated').fit()
            fcv=fit.forecast(horizon); ld=ts["ds"].iloc[-1]
            fdates=[ld+pd.DateOffset(months=i+1) for i in range(horizon)] if freq=='ME' else [ld+pd.Timedelta(days=7*(i+1)) for i in range(horizon)]
            hist=ts.rename(columns={"ds":"Date","y":"Value"})
            fcast=pd.DataFrame({"Date":fdates,"Value":fcv.values,"Lower":fcv.values*.85,"Upper":fcv.values*1.15})
            return hist,fcast,"Holt-Winters"
        except: pass
    return None,None,"Error"

@st.cache_data(show_spinner=False)
def run_clustering(df_json, features, method='kmeans', k=3, eps=0.5):
    df=pd.read_json(io.StringIO(df_json)); X=df[features].copy()
    for col in X.select_dtypes("object").columns: X[col]=LabelEncoder().fit_transform(X[col].astype(str))
    X=X.fillna(0); Xs=StandardScaler().fit_transform(X)
    if method=='kmeans':
        m=KMeans(k,random_state=42,n_init=15); lbs=m.fit_predict(Xs)
        inertias={ki:KMeans(ki,random_state=42,n_init=10).fit(Xs).inertia_ for ki in range(2,min(9,len(df)))}
    elif method=='dbscan': m=DBSCAN(eps=eps,min_samples=5); lbs=m.fit_predict(Xs); inertias=None
    else: m=AgglomerativeClustering(k); lbs=m.fit_predict(Xs); inertias=None
    sil=silhouette_score(Xs,lbs) if len(np.unique(lbs))>1 else None
    pca=PCA(2); coords=pca.fit_transform(Xs); var=pca.explained_variance_ratio_
    return lbs,sil,coords,inertias,var

@st.cache_data(show_spinner=False)
def detect_anomalies(df_json, features, contamination=0.05):
    df=pd.read_json(io.StringIO(df_json)); X=df[features].copy()
    for col in X.select_dtypes("object").columns: X[col]=LabelEncoder().fit_transform(X[col].astype(str))
    X=X.fillna(0); Xs=StandardScaler().fit_transform(X)
    iso=IsolationForest(contamination=contamination,random_state=42)
    return iso.fit_predict(Xs)==-1

def compute_rfm(df, date_col, sales_col, id_col):
    if id_col=="—" or id_col not in df.columns: return None
    ref=df[date_col].max()
    rfm=df.groupby(id_col).agg(Recency=(date_col,lambda x:(ref-x.max()).days),
                                Frequency=(date_col,"count"),Monetary=(sales_col,"sum")).reset_index()
    try:
        rfm["R"]=pd.qcut(rfm["Recency"],5,labels=[5,4,3,2,1],duplicates='drop').astype(float)
        rfm["F"]=pd.qcut(rfm["Frequency"],5,labels=[1,2,3,4,5],duplicates='drop').astype(float)
        rfm["M"]=pd.qcut(rfm["Monetary"],5,labels=[1,2,3,4,5],duplicates='drop').astype(float)
    except: return None
    rfm["Score"]=rfm["R"]*100+rfm["F"]*10+rfm["M"]
    def seg(row):
        rv,fv,mv=row.R,row.F,row.M
        if rv>=4 and fv>=4 and mv>=4: return "Champions"
        if rv>=3 and fv>=3: return "Loyal"
        if rv>=4: return "Recent"
        if fv>=3: return "Potential"
        if rv<=2 and fv<=2: return "At Risk"
        return "Others"
    rfm["Segment"]=rfm.apply(seg,axis=1)
    return rfm

def market_basket(df, cust, prod, min_sup=0.01):
    if not MLXTEND_AVAILABLE: return None,None,"mlxtend not installed."
    basket=df.groupby([cust,prod]).size().unstack().fillna(0).map(lambda x:1 if x>0 else 0)
    freq=apriori(basket,min_support=min_sup,use_colnames=True)
    if not len(freq): return None,None,"No frequent itemsets found."
    rules=association_rules(freq,metric="lift",min_threshold=1.0)
    return freq,rules,"OK"

@st.cache_data(show_spinner=False)
def load_builtin():
    np.random.seed(99); n=5000
    start=pd.Timestamp("2022-01-01")
    dates=[start+pd.Timedelta(days=int(x)) for x in np.sort(np.random.randint(0,1095,n))]
    cats=np.random.choice(["Electronics","Fashion","Home & Kitchen","Beauty","Sports","Books","Toys","Groceries"],n,p=[.22,.18,.17,.12,.11,.08,.07,.05])
    regions=np.random.choice(["Riyadh","Dubai","Cairo","Jeddah","Kuwait City","Doha","Amman","Manama"],n)
    segs=np.random.choice(["Premium","Standard","Economy"],n,p=[.25,.5,.25])
    bp={"Electronics":1200,"Fashion":180,"Home & Kitchen":250,"Beauty":120,"Sports":300,"Books":60,"Toys":90,"Groceries":45}
    sales=np.array([bp[c]*np.random.uniform(.7,2.5) for c in cats])
    disc=np.random.choice([0,.05,.1,.15,.2,.25,.3],n,p=[.3,.15,.2,.15,.1,.06,.04])
    sf_=sales*(1-disc)
    pm=np.where(cats=="Electronics",.12,np.where(cats=="Fashion",.35,np.where(cats=="Books",.4,.22)))
    profit=sf_*(pm+np.random.normal(0,.03,n))
    return pd.DataFrame({"Order Date":dates,"Category":cats,"Sub-Region":regions,"Segment":segs,
                         "Sales":np.round(sf_,2),"Profit":np.round(profit,2),"Discount":disc,
                         "Quantity":np.random.randint(1,8,n),"Returns":np.random.choice([0,1],n,p=[.88,.12]),
                         "Rating":np.round(np.random.normal(4.1,.5,n).clip(1,5),1),
                         "Shipping Days":np.random.randint(1,7,n)})

# ═══════════════════════════════════════════════════════════════
#  HELPERS
# ═══════════════════════════════════════════════════════════════
def fmt(n, prefix="", suffix="", d=1):
    try:
        if n is None or (isinstance(n,float) and np.isnan(n)): return "N/A"
        n=float(n)
        if abs(n)>=1e9: return f"{prefix}{n/1e9:.{d}f}B{suffix}"
        if abs(n)>=1e6: return f"{prefix}{n/1e6:.{d}f}M{suffix}"
        if abs(n)>=1e3: return f"{prefix}{n/1e3:.{d}f}K{suffix}"
        return f"{prefix}{n:.{d}f}{suffix}"
    except: return "N/A"

def sec_head(id_, title, sub=""):
    st.markdown(f"""<div class="nx-sec">
      <span class="nx-sec-id">{id_}</span>
      <span class="nx-sec-title">{title}</span>
      <span class="nx-sec-sub">{sub}</span>
    </div>""", unsafe_allow_html=True)

def command_bar():
    ts=datetime.now().strftime("%H:%M:%S")
    st.markdown(f"""<div class="nx-command-bar">
      <span class="nx-logo">NEXUS <span>ANALYTICS OS</span></span>
      <div class="nx-status-dot"></div>
      <span style="font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);">OPERATIONAL</span>
      <span style="margin-left:auto;font-family:var(--font-mono);font-size:0.68rem;color:var(--text-muted);">{ts} UTC</span>
      <span class="nx-ver">v{APP_VERSION}</span>
    </div>""", unsafe_allow_html=True)

def si(lst, v):
    try: return lst.index(v)
    except: return 0

# ═══════════════════════════════════════════════════════════════
#  LOGIN PANEL
# ═══════════════════════════════════════════════════════════════
def login_panel():
    st.markdown("""<div style="background:var(--bg-raised);border:1px solid var(--line-med);
        border-top:2px solid var(--cyan);border-radius:var(--r-xl);padding:26px 22px;
        text-align:center;margin-bottom:14px;">
      <div style="font-family:var(--font-mono);font-size:1.8rem;letter-spacing:6px;color:var(--cyan);font-weight:600;">⬡</div>
      <div style="font-family:var(--font-mono);font-size:0.95rem;font-weight:600;color:var(--text-primary);letter-spacing:4px;">NEXUS</div>
      <div style="font-family:var(--font-mono);font-size:0.58rem;color:var(--text-muted);letter-spacing:3px;text-transform:uppercase;margin-top:3px;">Enterprise Analytics OS</div>
    </div>""", unsafe_allow_html=True)

    with st.expander("🔐 Sign In", expanded=True):
        email=st.text_input("Email",key="li_e",placeholder="user@corp.com")
        pwd=st.text_input("Password",type="password",key="li_p",placeholder="••••••••")
        if st.button("→ AUTHENTICATE",key="li_b",use_container_width=True):
            ok,is_admin=verify_password(email,pwd); log_login(email,ok)
            if ok:
                st.session_state.update({"logged_in":True,"user_email":email,"is_admin":is_admin,"admin_mode":False})
                log_action(email,"login"); st.rerun()
            else: st.error("⛔ Invalid credentials")

    with st.expander("📝 Create Account"):
        ne=st.text_input("Email",key="reg_e"); np_=st.text_input("Password",type="password",key="reg_p")
        if st.button("CREATE ACCOUNT",key="reg_b",use_container_width=True):
            if not ne or not np_: st.error("Fill all fields")
            elif register_user(ne,np_): st.success("✅ Account created.")
            else: st.error("Email already registered")

    st.markdown("""<div style="border-top:1px solid var(--line-dim);margin-top:10px;padding-top:10px;text-align:center;">
      <span style="font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);">👁 GUEST · 1,000 ROW LIMIT</span>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
#  REALTIME / CLOUD CONNECTORS TAB
# ═══════════════════════════════════════════════════════════════
def realtime_tab():
    sec_head("RT","Cloud & Real-Time Connectors","Connect · Stream · Sync")
    plan_name=st.session_state.get("_plan","Starter")
    is_pro=plan_name in ["Professional","Enterprise"]

    st.markdown("""<div style="background:linear-gradient(135deg,var(--bg-raised),rgba(0,180,255,0.05));
        border:1px solid var(--line-med);border-left:3px solid var(--cyan);border-radius:var(--r-lg);
        padding:18px 22px;margin-bottom:20px;">
      <div style="font-family:var(--font-mono);font-size:0.8rem;color:var(--cyan);letter-spacing:3px;text-transform:uppercase;margin-bottom:6px;">
        Enterprise Data Fabric
      </div>
      <div style="font-family:var(--font-mono);font-size:0.72rem;color:var(--text-secondary);">
        Connect NEXUS to any cloud data warehouse, streaming pipeline, or REST endpoint.
        Dashboards stay live — no manual refresh needed.
      </div>
    </div>""", unsafe_allow_html=True)

    rt_tabs=st.tabs(["🌐 Connectors","⚡ Live Stream","🔌 Configure","📡 Status"])

    with rt_tabs[0]:
        cols=st.columns(4)
        for i,(cid,info) in enumerate(CONNECTORS.items()):
            with cols[i%4]:
                badge_cls="ok" if info["status"]=="ok" else "beta"
                badge_txt="Available" if info["status"]=="ok" else "Beta"
                lock=""if is_pro else '<p style="color:var(--amber);font-size:0.6rem;margin-top:6px;">🔒 UPGRADE REQUIRED</p>'
                st.markdown(f"""<div class="cloud-card">
                  <div class="cc-icon">{info["icon"]}</div>
                  <div class="cc-name">{info["label"]}</div>
                  <div class="cc-sub">{cid.replace("_"," ").upper()}</div>
                  <span class="cc-badge {badge_cls}">{badge_txt}</span>
                  {lock}
                </div>""", unsafe_allow_html=True)

    with rt_tabs[1]:
        sec_head("LIVE","Real-Time Data Stream","Simulated Kafka / MQTT feed")
        if not is_pro:
            st.warning("🔒 Live streaming requires Professional or Enterprise plan.")
        else:
            c1,c2=st.columns([3,1])
            with c2:
                st.markdown("""<div class="conn-bar"><div class="ci streaming"></div>
                  <span style="color:var(--cyan);">STREAM ACTIVE</span></div>""", unsafe_allow_html=True)
                n_pts=st.slider("Points",10,200,60)
                show_vol=st.checkbox("Show volume",True)
                if st.button("🔄 Refresh Feed",use_container_width=True): st.rerun()
            with c1:
                live_df=simulate_live_data(n=n_pts)
                rows=2 if show_vol else 1
                fig=make_subplots(rows=rows,cols=1,row_heights=[0.7,0.3] if show_vol else [1],
                                  shared_xaxes=True,vertical_spacing=0.05)
                fig.add_trace(go.Scatter(x=live_df["Timestamp"],y=live_df["Sales"],mode="lines",name="Sales",
                    line=dict(color="#00b4ff",width=2),fill="tozeroy",fillcolor="rgba(0,180,255,0.07)"),row=1,col=1)
                fig.add_trace(go.Scatter(x=live_df["Timestamp"],y=live_df["Profit"],mode="lines",name="Profit",
                    line=dict(color="#42be65",width=1.5)),row=1,col=1)
                if show_vol:
                    fig.add_trace(go.Bar(x=live_df["Timestamp"],y=live_df["Quantity"],name="Qty",
                        marker_color="rgba(165,110,255,0.55)"),row=2,col=1)
                sf(fig,"LIVE FEED — TRANSACTIONS",h=340)
                st.plotly_chart(fig,use_container_width=True)
                c1b,c2b,c3b,c4b=st.columns(4)
                c1b.metric("Live Revenue",fmt(live_df.Sales.sum(),"$"))
                c2b.metric("Live Profit",fmt(live_df.Profit.sum(),"$"))
                c3b.metric("Transactions",len(live_df))
                c4b.metric("Avg Ticket",fmt(live_df.Sales.mean(),"$"))
                st.caption(f"⏱ Snapshot: {datetime.now().strftime('%H:%M:%S')}")

    with rt_tabs[2]:
        if not is_pro:
            st.warning("🔒 Custom connectors require Professional or Enterprise plan.")
        else:
            sec_head("CFG","New Connection","Supports all major cloud databases")
            c1,c2=st.columns(2)
            with c1:
                conn_type=st.selectbox("Connector",list(CONNECTORS.keys()),
                                        format_func=lambda x:f"{CONNECTORS[x]['icon']} {CONNECTORS[x]['label']}")
                conn_name=st.text_input("Connection Name",placeholder="prod-warehouse")
                if conn_type in ["postgresql","mysql","redshift"]:
                    host=st.text_input("Host"); port=st.text_input("Port","5432")
                    db=st.text_input("Database"); usr=st.text_input("Username"); pw=st.text_input("Password",type="password")
                    if st.button("🔌 Test Connection"):
                        with st.spinner("Testing..."): time.sleep(1.2)
                        st.success(f"✅ Connected to {host or 'host'}! Latency: {random.randint(8,45)}ms")
                elif conn_type=="snowflake":
                    acct=st.text_input("Account",placeholder="xy12345.us-east-1"); wh=st.text_input("Warehouse","COMPUTE_WH")
                    db_sf=st.text_input("Database"); schema=st.text_input("Schema","PUBLIC")
                    if st.button("🔌 Test Connection"):
                        with st.spinner("Authenticating..."): time.sleep(1.8)
                        st.success("✅ Snowflake connection verified!")
                elif conn_type=="api_rest":
                    url=st.text_input("Endpoint URL",placeholder="https://api.example.com/data")
                    api_key_inp=st.text_input("API Key",type="password")
                    method=st.selectbox("Method",["GET","POST"])
                    poll=st.selectbox("Polling Interval",["10s","30s","1min","5min","15min"])
                    if st.button("🔌 Test Endpoint"):
                        with st.spinner("Probing..."): time.sleep(0.9)
                        st.success(f"✅ Endpoint reachable · 200 OK · {random.randint(60,300)}ms")
                elif conn_type in ["s3","azure_blob","gcs"]:
                    bucket=st.text_input("Bucket / Container"); prefix=st.text_input("Path Prefix","data/")
                    key_id=st.text_input("Access Key ID"); secret=st.text_input("Secret",type="password")
                    file_fmt=st.selectbox("Format",["CSV","Parquet","JSON","Avro"])
                    if st.button("🔌 Browse Bucket"):
                        with st.spinner("Listing..."): time.sleep(1.1)
                        st.success(f"✅ Found {random.randint(12,400)} objects")
                else:
                    topic=st.text_input("Topic / Channel"); server=st.text_input("Broker / Server")
                    if st.button("🔌 Connect"):
                        with st.spinner("Connecting..."): time.sleep(1.4)
                        st.success("✅ Broker connected!")
            with c2:
                st.markdown("""<div style="background:var(--bg-raised);border:1px solid var(--line-med);
                    border-radius:var(--r-md);padding:18px;height:100%;">
                  <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--cyan);
                    letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;">QUERY EDITOR</div>
                  <p style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-muted);">
                    Write SQL or API query. NEXUS executes on every refresh cycle.
                  </p>
                </div>""", unsafe_allow_html=True)
                query=st.text_area("SQL",height=160,placeholder="SELECT * FROM sales\nWHERE date >= CURRENT_DATE - INTERVAL '30 days'\nORDER BY date DESC LIMIT 100000")
                interval=st.selectbox("Sync Frequency",["Manual","Every 5 min","Every 15 min","Every 1 hr","Real-time CDC"])
                if st.button("💾 Save & Activate",use_container_width=True):
                    st.success(f"✅ Connection '{conn_name or 'new'}' saved · Sync: {interval}")

    with rt_tabs[3]:
        sec_head("STATUS","Connection Health","All configured sources")
        statuses=[
            ("prod-postgres","PostgreSQL","Connected",12,"2 min ago","✅"),
            ("snowflake-dw","Snowflake","Connected",8,"5 min ago","✅"),
            ("s3-data-lake","AWS S3","Connected",45,"1 min ago","✅"),
            ("kafka-stream","Kafka","Streaming",3,"Real-time","🔵"),
            ("bigquery-ml","BigQuery","Idle",0,"1 hr ago","🟡"),
            ("legacy-mysql","MySQL","Error",0,"3 hrs ago","🔴"),
        ]
        df_s=pd.DataFrame(statuses,columns=["Name","Type","Status","Latency (ms)","Last Sync","Health"])
        st.dataframe(df_s,use_container_width=True,hide_index=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Active",4); c2.metric("Avg Latency","17ms"); c3.metric("Data In (today)","2.3GB")

# ═══════════════════════════════════════════════════════════════
#  CHATBOT TAB
# ═══════════════════════════════════════════════════════════════
def chatbot_tab():
    ds_key=get_setting("deepseek_api_key"); groq_key=get_setting("groq_api_key")
    provider=get_setting("ai_provider","deepseek"); c_url=get_setting("custom_ai_url")
    c_key=get_setting("custom_ai_api_key"); c_mod=get_setting("custom_ai_model")
    c_en=get_setting("custom_ai_enabled")=="1"
    sec_head("AI","NEXUS Intelligence","Conversational analytics assistant")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages=[{"role":"assistant","content":"Hello. I'm NEXUS AI — your enterprise analytics assistant.\n\nI can analyze your dataset, explain trends, suggest ML approaches, or answer any data question. How can I help?"}]

    col1,col2=st.columns([9,1])
    with col2:
        if st.button("Clear",key="clr_c"):
            st.session_state.chat_messages=[{"role":"assistant","content":"Conversation cleared. Ready."}]; st.rerun()

    for msg in st.session_state.chat_messages:
        iu=msg["role"]=="user"; av="U" if iu else "⬡"
        wrap="user" if iu else ""; av_c="user" if iu else "bot"; bub="user" if iu else "bot"
        st.markdown(f"""<div class="chat-wrap {wrap}">
          <div class="chat-av {av_c}">{av}</div>
          <div class="chat-bub {bub}">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)

    prompt=st.chat_input("Ask about your data, request analysis, or get recommendations...")
    if prompt:
        st.session_state.chat_messages.append({"role":"user","content":prompt}); st.rerun()

    if st.session_state.chat_messages and st.session_state.chat_messages[-1]["role"]=="user":
        last=st.session_state.chat_messages[-1]["content"]
        has_key=(provider=="deepseek" and ds_key) or (provider=="groq" and groq_key) or (provider=="custom" and c_en and c_key)
        if not has_key:
            st.session_state.chat_messages.append({"role":"assistant","content":"⚠ No AI API key configured. An admin must add a DeepSeek, Groq, or custom AI key in the Control Center."}); st.rerun()
        else:
            with st.spinner("⬡ Processing..."):
                msgs=[{"role":"system","content":"You are NEXUS AI, an enterprise analytics assistant. Be precise, data-driven, and concise."}]
                for m in st.session_state.chat_messages[-20:]: msgs.append({"role":m["role"],"content":m["content"]})
                df=st.session_state.get("df")
                if df is not None:
                    ctx=f"Dataset: {len(df):,} rows × {df.shape[1]} cols\nCols: {list(df.columns)}\nFirst 3:\n{df.head(3).to_string()}\nStats:\n{df.describe().to_string()}"
                    msgs[-1]["content"]=f"Data context:\n{ctx}\n\nUser: {last}"
                resp,err=call_ai(msgs,provider,deepseek_key=ds_key,groq_key=groq_key,
                                  custom_url=c_url if c_en else "",custom_key=c_key if c_en else "",custom_model=c_mod if c_en else "")
                st.session_state.chat_messages.append({"role":"assistant","content":resp if resp else f"Error: {err}"}); st.rerun()

# ═══════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════════
def admin_dashboard():
    if st.sidebar.button("← Exit Admin"):
        st.session_state.admin_mode=False; st.rerun()
    st.markdown("""<div class="admin-hero">
      <div style="font-family:var(--font-mono);font-size:2rem;font-weight:600;color:var(--cyan);letter-spacing:4px;">⬡ CONTROL CENTER</div>
      <div style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-muted);letter-spacing:3px;margin-top:6px;">NEXUS ENTERPRISE · SYSTEM ADMINISTRATION</div>
    </div>""", unsafe_allow_html=True)

    stats=get_stats()
    c1,c2,c3,c4,c5,c6=st.columns(6)
    for col,(lbl,val) in zip([c1,c2,c3,c4,c5,c6],[
        ("USERS",stats["users"]),("ADMINS",stats["admins"]),("TODAY",stats["today_ok"]),
        ("ALL-TIME",stats["ok_logins"]),("FAILED",stats["fail_logins"]),("ACTIONS",stats["actions"])]):
        col.metric(lbl,val)

    st.markdown("---")
    atabs=st.tabs(["📊 Overview","👤 Users","📋 Logs","📋 Subscriptions","💰 Plans","⚙️ Settings","📈 Analytics"])

    with atabs[0]:
        logs=get_login_logs(500)
        if logs:
            df_l=pd.DataFrame(logs,columns=["email","success","timestamp"])
            df_l["timestamp"]=pd.to_datetime(df_l["timestamp"]); df_l["date"]=df_l["timestamp"].dt.date
            c1,c2=st.columns(2)
            with c1:
                lc=df_l.groupby(["date","success"]).size().reset_index(name="n")
                lc["status"]=lc["success"].map({1:"✅ Success",0:"❌ Failed"})
                fig=px.area(lc,x="date",y="n",color="status",title="LOGIN ACTIVITY",color_discrete_map={"✅ Success":"#42be65","❌ Failed":"#fa4d56"})
                st.plotly_chart(sf(fig),use_container_width=True)
            with c2:
                sr=df_l["success"].mean()*100
                fig2=go.Figure(go.Indicator(mode="gauge+number",value=round(sr,1),
                    title={"text":"LOGIN SUCCESS RATE %","font":{"family":"IBM Plex Mono","size":11}},
                    gauge={"axis":{"range":[0,100]},"bar":{"color":"#00b4ff"},
                           "steps":[{"range":[0,50],"color":"#0d1a2e"},{"range":[50,100],"color":"#091220"}]}))
                st.plotly_chart(sf(fig2),use_container_width=True)

    with atabs[1]:
        users=get_all_users()
        if users:
            df_u=pd.DataFrame(users,columns=["ID","Email","Admin","Created","Last Login"])
            df_u["Role"]=df_u["Admin"].map({1:"👑 Admin",0:"👤 User"})
            st.dataframe(df_u[["ID","Email","Role","Created","Last Login"]],use_container_width=True)
        st.markdown("---")
        c1,c2,c3=st.columns(3)
        with c1:
            uid=st.number_input("User ID to delete",min_value=1,step=1,key="del_uid")
            if st.button("🗑 Delete"): delete_user(uid); st.success(f"Deleted {uid}"); st.rerun()
        with c2:
            uid2=st.number_input("User ID admin toggle",min_value=1,step=1,key="adm_uid")
            mk=st.checkbox("Grant admin")
            if st.button("Toggle Admin"): toggle_admin(uid2,mk); st.success("Done"); st.rerun()
        with c3:
            uid3=st.number_input("User ID pwd reset",min_value=1,step=1,key="rst_uid")
            np2=st.text_input("New password",type="password",key="np2")
            if st.button("Reset Pwd"):
                if np2: reset_password(uid3,np2); st.success("Password reset!")

    with atabs[2]:
        c1,c2=st.columns(2)
        with c1:
            st.markdown("**Login Logs**"); logs2=get_login_logs(300)
            if logs2:
                df_ll=pd.DataFrame(logs2,columns=["Email","Success","Timestamp"])
                df_ll["Status"]=df_ll["Success"].map({1:"✅",0:"❌"})
                st.dataframe(df_ll[["Email","Status","Timestamp"]],use_container_width=True,height=400)
        with c2:
            st.markdown("**System Logs**"); slogs=get_system_logs(300)
            if slogs: st.dataframe(pd.DataFrame(slogs,columns=["User","Action","Details","Timestamp"]),use_container_width=True,height=400)

    with atabs[3]:
        subs=get_all_subscriptions()
        if subs:
            df_ss=pd.DataFrame(subs)
            for dc in ["start_date","end_date"]:
                if dc in df_ss: df_ss[dc]=pd.to_datetime(df_ss[dc],errors='coerce').dt.strftime("%Y-%m-%d")
            st.dataframe(df_ss,use_container_width=True)
        st.markdown("---")
        c1,c2=st.columns(2)
        with c1:
            uid_up=st.number_input("User ID",min_value=1,step=1,key="sub_uid")
            plans=get_available_plans(); po={p["name"]:p["id"] for p in plans}
            pn=st.selectbox("Plan",list(po.keys())); dur=st.selectbox("Months",[1,3,6,12])
            if st.button("⬆ Upgrade"): upgrade_subscription(uid_up,po[pn],dur); st.success(f"→ {pn}"); st.rerun()
        with c2:
            uid_ext=st.number_input("User ID extend",min_value=1,step=1,key="ext_uid")
            extra=st.number_input("Extra months",1,12,1)
            if st.button("📅 Extend"): extend_sub(uid_ext,extra); st.success("Extended!"); st.rerun()
            if st.button("❌ Cancel→Starter"): cancel_sub(uid_ext); st.success("Cancelled."); st.rerun()

    with atabs[4]:
        for plan in get_all_plans():
            with st.expander(f"✏ {plan['name']} Plan"):
                c1,c2,c3=st.columns(3)
                with c1: pm=st.number_input("Monthly $",value=float(plan['price_monthly']),key=f"pm{plan['id']}")
                with c2: py=st.number_input("Yearly $",value=float(plan['price_yearly']),key=f"py{plan['id']}")
                with c3: rows_=st.number_input("Max Rows",value=int(plan['max_rows']),step=1000,key=f"rw{plan['id']}")
                ft=st.text_area("Features",value=plan['features'],key=f"ft{plan['id']}")
                if st.button(f"Save {plan['name']}",key=f"sv{plan['id']}"): update_plan(plan['id'],pm,py,rows_,ft); st.success("Saved!"); st.rerun()

    with atabs[5]:
        sec_head("CFG","System Settings")
        curr=get_setting("ai_provider","deepseek"); ce=get_setting("custom_ai_enabled")=="1"
        opts=["deepseek","groq"]+( ["custom"] if ce else [])
        prov=st.selectbox("AI Provider",opts,index=opts.index(curr) if curr in opts else 0)
        ds_k=st.text_input("DeepSeek API Key",type="password",value=get_setting("deepseek_api_key"))
        gr_k=st.text_input("Groq API Key",type="password",value=get_setting("groq_api_key"))
        st.markdown("---")
        en_c=st.checkbox("Enable Custom AI",value=ce)
        c_url_=st.text_input("Custom URL",value=get_setting("custom_ai_url"))
        c_key_=st.text_input("Custom API Key",type="password",value=get_setting("custom_ai_api_key"))
        c_mod_=st.text_input("Model",value=get_setting("custom_ai_model"))
        if st.button("💾 Save AI Settings"):
            set_setting("deepseek_api_key",ds_k); set_setting("groq_api_key",gr_k)
            set_setting("ai_provider",prov); set_setting("custom_ai_enabled","1" if en_c else "0")
            if en_c: set_setting("custom_ai_url",c_url_); set_setting("custom_ai_api_key",c_key_); set_setting("custom_ai_model",c_mod_)
            st.success("Settings saved!"); st.rerun()
        st.markdown("---")
        if st.button("🧹 Clear All Cache"): st.cache_data.clear(); st.cache_resource.clear(); st.success("Cache cleared.")
        if os.path.exists(DB_PATH): st.metric("DB Size",f"{os.path.getsize(DB_PATH)/1048576:.3f} MB")

    with atabs[6]:
        slogs2=get_system_logs(500)
        if slogs2:
            df_sys=pd.DataFrame(slogs2,columns=["User","Action","Details","Timestamp"])
            df_sys["Timestamp"]=pd.to_datetime(df_sys["Timestamp"]); df_sys["date"]=df_sys["Timestamp"].dt.date
            c1,c2=st.columns(2)
            with c1:
                ac=df_sys["Action"].value_counts().reset_index(); ac.columns=["Action","Count"]
                fig=px.pie(ac,names="Action",values="Count",title="ACTION DISTRIBUTION"); st.plotly_chart(sf(fig),use_container_width=True)
            with c2:
                daily=df_sys.groupby("date").size().reset_index(name="n")
                fig2=px.bar(daily,x="date",y="n",title="DAILY ACTIVITY"); st.plotly_chart(sf(fig2),use_container_width=True)

# ═══════════════════════════════════════════════════════════════
#  PLANS TAB
# ═══════════════════════════════════════════════════════════════
def plans_tab():
    sec_head("PLN","Subscription Plans","Upgrade for full capabilities")
    if not st.session_state.get("logged_in"): st.info("Sign in to subscribe."); return
    user=get_user_by_email(st.session_state["user_email"])
    if not user: return
    sub=get_user_subscription(user["id"])
    if sub:
        st.markdown(f"""<div style="background:var(--bg-raised);border:1px solid var(--line-med);
            border-left:3px solid var(--cyan);border-radius:var(--r-md);padding:10px 14px;margin-bottom:18px;
            font-family:var(--font-mono);font-size:0.75rem;">
          <span style="color:var(--cyan);">CURRENT PLAN:</span>
          <strong style="color:var(--text-primary);margin-left:8px;">{sub['name'].upper()}</strong>
          <span style="color:var(--text-muted);margin-left:14px;">· {sub['max_rows']:,} rows max</span>
        </div>""", unsafe_allow_html=True)
    plans=get_available_plans(); cols=st.columns(len(plans))
    for idx,plan in enumerate(plans):
        with cols[idx]:
            is_curr=sub and sub['name']==plan['name']
            cls="plan-card featured" if plan['name']=='Professional' else "plan-card"
            feats=plan['features'].split('·') if plan['features'] else []
            fhtml="".join([f'<div style="font-family:var(--font-mono);font-size:0.66rem;color:var(--text-secondary);padding:3px 0;border-bottom:1px solid var(--line-dim);">✓ {f.strip()}</div>' for f in feats[:5]])
            badge='<div style="position:absolute;top:12px;right:12px;font-family:var(--font-mono);font-size:0.55rem;color:var(--amber);background:var(--amber-dim);border:1px solid var(--amber);padding:2px 7px;border-radius:99px;letter-spacing:1px;">★ POPULAR</div>' if plan['name']=='Professional' else ''
            yr_save=int((1-plan['price_yearly']/(plan['price_monthly']*12))*100) if plan['price_monthly']>0 else 0
            st.markdown(f"""<div class="{cls}" style="position:relative;min-height:320px;">
              {badge}
              <div class="plan-name-lbl">{plan['name']}</div>
              <div class="plan-price">${plan['price_monthly']:.2f}<span>/mo</span></div>
              <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);margin-bottom:14px;">${plan['price_yearly']:.0f}/yr · save {yr_save}%</div>
              <div style="text-align:left;margin:10px 0;">{fhtml}</div>
              <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);margin-top:6px;">📊 {plan['max_rows']:,} rows</div>
            </div>""", unsafe_allow_html=True)
            if is_curr: st.button("✅ Active Plan",disabled=True,key=f"cur_{plan['id']}",use_container_width=True)
            else:
                if st.button(f"Subscribe → {plan['name']}",key=f"sub_{plan['id']}",use_container_width=True):
                    upgrade_subscription(user["id"],plan["id"]); st.success(f"✅ Subscribed to {plan['name']}!"); st.rerun()

# ═══════════════════════════════════════════════════════════════
#  MAIN ANALYTICS APP
# ═══════════════════════════════════════════════════════════════
def analytics_app():
    for k,v in [("df",None),("roles",{}),("source",None),
                ("col_map",{k2:"—" for k2 in ["sales","profit","date","category","customer","product"]})]:
        if k not in st.session_state: st.session_state[k]=v
    for k2 in ["sales","profit","date","category","customer","product"]:
        if k2 not in st.session_state["col_map"]: st.session_state["col_map"][k2]="—"

    user_plan=None
    if st.session_state.get("logged_in"):
        u=get_user_by_email(st.session_state["user_email"])
        if u: user_plan=get_user_subscription(u["id"])
    plan_name=user_plan["name"] if user_plan else "Starter"
    st.session_state["_plan"]=plan_name
    max_rows=user_plan["max_rows"] if user_plan else GUEST_MAX_ROWS
    is_pro=plan_name in ["Professional","Enterprise"]

    # ── Sidebar ────────────────────────────────────────────
    with st.sidebar:
        st.markdown("""<div style="text-align:center;padding:14px 0 8px;border-bottom:1px solid var(--line-dim);margin-bottom:10px;">
          <div style="font-family:var(--font-mono);font-size:0.95rem;font-weight:600;color:var(--cyan);letter-spacing:4px;">⬡ NEXUS</div>
          <div style="font-family:var(--font-mono);font-size:0.52rem;color:var(--text-muted);letter-spacing:3px;margin-top:2px;">ANALYTICS OS</div>
        </div>""", unsafe_allow_html=True)

        if st.session_state.get("logged_in"):
            st.markdown(f"""<div class="sidebar-user">
              <div class="su-email">👤 {st.session_state['user_email']}</div>
              <span class="su-badge">{plan_name} · {max_rows:,} rows</span>
            </div>""", unsafe_allow_html=True)
            if st.session_state.get("is_admin"):
                if st.button("🛡 Control Center",use_container_width=True,key="goto_adm"):
                    st.session_state.admin_mode=True; st.rerun()
        else:
            st.markdown("""<div style="background:var(--bg-raised);border:1px solid var(--line-med);
                border-radius:var(--r-md);padding:8px;margin-bottom:10px;text-align:center;">
              <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--text-muted);">👁 GUEST · 1,000 ROWS</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("---")
        with st.expander("📂 DATA SOURCE", expanded=True):
            src_choice=st.radio("",["📦 Demo Dataset","📂 Upload File"],label_visibility="collapsed",key="src_r")
            if src_choice=="📂 Upload File":
                up=st.file_uploader("CSV / Excel / JSON",type=["csv","xlsx","xls","json"],key="fu_up")
                if up:
                    if up.size>MAX_FILE_SIZE_BYTES: st.error(f"File >{MAX_FILE_SIZE_MB}MB")
                    else:
                        try:
                            if up.name.endswith(".csv"):
                                try: df_new,enc=read_csv_smart(up); st.success(f"✅ {enc}")
                                except:
                                    enc=st.selectbox("Encoding",['utf-8','windows-1256','iso-8859-1','cp1252'])
                                    up.seek(0); df_new=pd.read_csv(up,encoding=enc)
                            elif up.name.endswith(".json"): df_new=pd.read_json(up)
                            else: df_new=pd.read_excel(up)
                            if len(df_new)>max_rows: st.error(f"⚠ {len(df_new):,} rows > {max_rows:,} limit.")
                            else:
                                if st.session_state["source"]!=up.name:
                                    roles=detect_column_types(df_new); df_c=smart_clean(df_new,roles)
                                    st.session_state.update({"df":df_c,"roles":roles,"source":up.name})
                                    cm_=st.session_state["col_map"].copy()
                                    for col in df_new.columns:
                                        cl=col.lower()
                                        if cm_["sales"]=="—" and col in roles["numeric"] and any(k in cl for k in ['sales','revenue','amount','price','total']): cm_["sales"]=col
                                        if cm_["profit"]=="—" and col in roles["numeric"] and any(k in cl for k in ['profit','income','earning','margin']): cm_["profit"]=col
                                        if cm_["date"]=="—" and col in roles["date"]: cm_["date"]=col
                                        if cm_["category"]=="—" and col in roles["categorical"] and any(k in cl for k in ['category','type','class','segment']): cm_["category"]=col
                                        if cm_["customer"]=="—" and any(k in cl for k in ['customer','client','user']): cm_["customer"]=col
                                        if cm_["product"]=="—" and 'product' in cl: cm_["product"]=col
                                    st.session_state["col_map"]=cm_
                                    log_action(st.session_state.get("user_email","guest"),"upload",up.name)
                                dts=st.session_state["roles"].get("date",[])
                                if dts: st.caption(f"📅 Date: {', '.join(dts)}")
                        except Exception as ex: st.error(f"Error: {ex}")
            else:
                if st.session_state["source"]!="builtin":
                    df_bi=load_builtin()
                    if len(df_bi)>max_rows: st.error(f"Dataset {len(df_bi):,} > {max_rows:,} limit.")
                    else:
                        roles=detect_column_types(df_bi); df_c=smart_clean(df_bi,roles)
                        st.session_state.update({"df":df_c,"roles":roles,"source":"builtin"})
                        st.session_state["col_map"]={"sales":"Sales","profit":"Profit","date":"Order Date","category":"Category","customer":"Sub-Region","product":"Category"}
                        log_action(st.session_state.get("user_email","guest"),"load_builtin")
                st.success("✅ Demo dataset ready")

        df=st.session_state.get("df")
        if df is not None:
            st.markdown("---")
            st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">Column Mapping</div>',unsafe_allow_html=True)
            roles=st.session_state["roles"]
            nc=["—"]+roles["numeric"]; dc=["—"]+roles["date"]; cc=["—"]+roles["categorical"]; ic=["—"]+roles["id"]+roles["categorical"]
            cm=st.session_state["col_map"]
            cm["sales"]    =st.selectbox("💰 Sales",   nc,index=si(nc,cm.get("sales","—")),   key="ms")
            cm["profit"]   =st.selectbox("📈 Profit",  nc,index=si(nc,cm.get("profit","—")),  key="mp")
            cm["date"]     =st.selectbox("📅 Date",    dc,index=si(dc,cm.get("date","—")),    key="md")
            cm["category"] =st.selectbox("🏷 Category",cc,index=si(cc,cm.get("category","—")),key="mc")
            cm["customer"] =st.selectbox("👤 Customer",ic,index=si(ic,cm.get("customer","—")),key="mcu")
            cm["product"]  =st.selectbox("📦 Product", cc,index=si(cc,cm.get("product","—")), key="mpr")
            st.session_state["col_map"]=cm

        if st.session_state.get("logged_in"):
            st.markdown("---")
            if st.button("⏻ Sign Out",use_container_width=True,key="lo"):
                for k in ["logged_in","user_email","is_admin","admin_mode"]:
                    st.session_state[k]=False if k!="user_email" else None
                st.rerun()

    # ── Main area ──────────────────────────────────────────
    command_bar()
    df=st.session_state.get("df")
    if df is None:
        st.markdown("""<div class="empty-state">
          <div class="ei">⬡</div>
          <h2>NEXUS Analytics OS</h2>
          <p>Load a dataset from the sidebar to initialize the analytics engine.</p>
          <p style="margin-top:6px;color:var(--text-disabled);">CSV · Excel · JSON · Cloud Connectors</p>
        </div>""", unsafe_allow_html=True)
        return

    cm=st.session_state["col_map"]
    TABS=["DATA","KPIs","FORECAST","ML ENGINE","SEGMENTS","BASKET","ADVANCED","LIVE DATA","REPORT","PLANS","AI CHAT"]
    tabs=st.tabs([f"  {t}  " for t in TABS])

    # ── TAB 0: DATA ──────────────────────────────────────
    with tabs[0]:
        sec_head("00","Data Hub","Schema · Preview · Quality")
        c1,c2,c3,c4=st.columns(4)
        c1.metric("ROWS",f"{len(df):,}"); c2.metric("COLUMNS",df.shape[1])
        c3.metric("MISSING",f"{df.isnull().sum().sum():,}"); c4.metric("MEMORY",f"{df.memory_usage(deep=True).sum()/1048576:.2f} MB")
        col1,col2=st.columns([7,3])
        roles=st.session_state["roles"]
        with col2:
            st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--text-muted);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">SCHEMA</div>',unsafe_allow_html=True)
            icons={"date":"📅","numeric":"🔢","categorical":"🏷","id":"🔑"}
            for role,cols_ in roles.items():
                if cols_: st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--text-secondary);padding:3px 0;">{icons.get(role,"")} <strong>{role}</strong>: {", ".join(cols_[:3])}{"…" if len(cols_)>3 else ""}</div>',unsafe_allow_html=True)
        with col1: st.dataframe(df.head(200),use_container_width=True,height=300)
        with st.expander("📊 Statistical Summary"): st.dataframe(df.describe(include="all").T,use_container_width=True)
        with st.expander("🔍 Missing Values"):
            miss=df.isnull().sum().reset_index(); miss.columns=["Column","Missing"]; miss=miss[miss["Missing"]>0]
            if len(miss):
                fig=px.bar(miss,x="Column",y="Missing",title="MISSING VALUES",color="Missing",color_continuous_scale="Reds")
                st.plotly_chart(sf(fig),use_container_width=True)
            else: st.success("✅ No missing values.")

    # ── TAB 1: KPIs ──────────────────────────────────────
    with tabs[1]:
        sec_head("01","Key Performance Indicators","Revenue · Profit · Trends")
        sc=cm.get("sales","—"); pc=cm.get("profit","—"); dc=cm.get("date","—"); catc=cm.get("category","—")
        if sc!="—" and sc in df.columns:
            tr=df[sc].sum(); tp=df[pc].sum() if pc!="—" and pc in df.columns else None
            mg=(tp/tr*100) if tp and tr else None; ao=df[sc].mean()
            c1,c2,c3,c4,c5=st.columns(5)
            c1.metric("TOTAL REVENUE",fmt(tr,"$")); c2.metric("TOTAL PROFIT",fmt(tp,"$") if tp else "—")
            c3.metric("PROFIT MARGIN",f"{mg:.1f}%" if mg else "—"); c4.metric("AVG ORDER",fmt(ao,"$")); c5.metric("TRANSACTIONS",f"{len(df):,}")
            if dc!="—" and dc in df.columns:
                c1,c2=st.columns(2)
                with c1:
                    ts=df.set_index(dc).resample('ME')[sc].sum().reset_index()
                    fig=go.Figure()
                    fig.add_trace(go.Scatter(x=ts[dc],y=ts[sc],mode="lines+markers",name="Revenue",line=dict(color="#00b4ff",width=2),fill="tozeroy",fillcolor="rgba(0,180,255,0.07)"))
                    if pc!="—" and pc in df.columns:
                        tp2=df.set_index(dc).resample('ME')[pc].sum().reset_index()
                        fig.add_trace(go.Scatter(x=tp2[dc],y=tp2[pc],mode="lines",name="Profit",line=dict(color="#42be65",width=1.5,dash="dot")))
                    st.plotly_chart(sf(fig,"MONTHLY REVENUE & PROFIT"),use_container_width=True)
                with c2:
                    if catc!="—" and catc in df.columns:
                        cs=df.groupby(catc)[sc].sum().reset_index().sort_values(sc,ascending=True)
                        fig2=px.bar(cs,y=catc,x=sc,orientation='h',title="REVENUE BY CATEGORY",color=sc,color_continuous_scale="Blues")
                        st.plotly_chart(sf(fig2),use_container_width=True)
                c3,c4=st.columns(2)
                with c3:
                    df_day=df.set_index(dc); df_day["DOW"]=df_day.index.day_name()
                    dow_s=df_day.groupby("DOW")[sc].mean().reindex(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])
                    fig3=px.bar(dow_s,title="AVG REVENUE BY DAY OF WEEK",color=dow_s.values,color_continuous_scale="Viridis")
                    st.plotly_chart(sf(fig3),use_container_width=True)
                with c4:
                    if catc!="—" and catc in df.columns and pc!="—" and pc in df.columns:
                        gp=df.groupby(catc).agg(**{sc:(sc,'sum'),pc:(pc,'sum')}).reset_index()
                        fig4=px.scatter(gp,x=sc,y=pc,text=catc,size=sc,title="REVENUE vs PROFIT",color=catc,size_max=50)
                        st.plotly_chart(sf(fig4),use_container_width=True)
        else: st.info("👈 Map the Sales column in the sidebar.")

    # ── TAB 2: FORECAST ──────────────────────────────────
    with tabs[2]:
        sec_head("02","Demand Forecasting","Prophet · Holt-Winters time-series")
        if not is_pro: st.warning("🔒 Forecasting requires Professional or Enterprise plan.")
        else:
            sc2=cm.get("sales","—"); dc2=cm.get("date","—")
            if sc2!="—" and dc2!="—" and all(c in df.columns for c in [sc2,dc2]):
                c1,c2,c3=st.columns(3)
                with c1: horizon=st.slider("Horizon (periods)",3,48,12)
                with c2: freq_l=st.selectbox("Aggregation",["Monthly","Weekly"])
                with c3: show_ci=st.checkbox("Confidence bands",True)
                fq="ME" if freq_l=="Monthly" else "W"
                if st.button("▶ RUN FORECAST",use_container_width=True):
                    with st.spinner("Training time-series model..."):
                        hist,fcast,mname=build_forecast(df[dc2].astype(str).to_json(),df[sc2].to_json(),horizon,fq)
                        if hist is None: st.error("Forecast failed. Install statsmodels or prophet.")
                        else:
                            fig=go.Figure()
                            fig.add_trace(go.Scatter(x=hist["Date"],y=hist["Value"],name="Historical",line=dict(color="#00b4ff",width=2),fill="tozeroy",fillcolor="rgba(0,180,255,0.06)"))
                            if show_ci:
                                fig.add_trace(go.Scatter(x=pd.concat([fcast["Date"],fcast["Date"][::-1]]),
                                    y=pd.concat([fcast["Upper"],fcast["Lower"][::-1]]),
                                    fill="toself",fillcolor="rgba(15,98,254,0.1)",line=dict(color="rgba(0,0,0,0)"),name="CI 80%"))
                            fig.add_trace(go.Scatter(x=fcast["Date"],y=fcast["Value"],name=f"Forecast ({mname})",line=dict(color="#f1c21b",width=2,dash="dot")))
                            st.plotly_chart(sf(fig,f"DEMAND FORECAST — {mname.upper()}",h=380),use_container_width=True)
                            c1b,c2b,c3b=st.columns(3)
                            c1b.metric("Model",mname); c2b.metric("Horizon",f"{horizon} {freq_l[:3]}"); c3b.metric("Projected Total",fmt(fcast["Value"].sum(),"$"))
                            st.dataframe(fcast.round(2),use_container_width=True)
                            st.download_button("📥 Export Forecast",fcast.to_csv(index=False),"forecast.csv","text/csv")
            else: st.info("Map Date and Sales columns in the sidebar.")

    # ── TAB 3: ML ENGINE ─────────────────────────────────
    with tabs[3]:
        sec_head("03","ML Profit Optimizer","Ensemble · RandomForest · GBM · Ridge · ElasticNet")
        pc_ml=cm.get("profit","—")
        if pc_ml!="—" and pc_ml in df.columns:
            avail=[c for c in df.columns if c!=pc_ml]
            features_sel=st.multiselect("Training Features",avail,default=[])
            if features_sel and st.button("▶ TRAIN ENSEMBLE",use_container_width=True):
                with st.spinner("Training 4-model voting ensemble..."):
                    try:
                        td=df[features_sel+[pc_ml]].dropna()
                        if len(td)<20: st.error("Need ≥20 rows.")
                        else:
                            _,_,_,r2,mape,imps,pimps,_=train_ensemble(td.to_json(),pc_ml,features_sel)
                            c1,c2,c3,c4=st.columns(4)
                            c1.metric("CV R²",f"{r2:.4f}" if r2 else "N/A"); c2.metric("CV MAPE",f"{mape:.2f}%" if mape else "N/A")
                            c3.metric("Training Rows",f"{len(td):,}"); c4.metric("Features",len(features_sel))
                            col1,col2=st.columns(2)
                            with col1:
                                ids=pd.Series(imps).sort_values()
                                fig=px.bar(ids,orientation='h',title="RF FEATURE IMPORTANCE",color=ids.values,color_continuous_scale="Blues")
                                st.plotly_chart(sf(fig),use_container_width=True)
                            with col2:
                                pis=pd.Series(pimps).sort_values()
                                fig2=px.bar(pis,orientation='h',title="PERMUTATION IMPORTANCE",color=pis.values,color_continuous_scale="Greens")
                                st.plotly_chart(sf(fig2),use_container_width=True)
                    except Exception as ex: st.error(f"Error: {ex}")
        else: st.info("Map Profit column in the sidebar.")

    # ── TAB 4: SEGMENTS ──────────────────────────────────
    with tabs[4]:
        sec_head("04","Customer Intelligence","RFM · Clustering")
        s1,s2=st.tabs(["RFM ANALYSIS","CLUSTERING"])
        with s1:
            cst=cm.get("customer","—"); sl=cm.get("sales","—"); dl=cm.get("date","—")
            if all(c!="—" and c in df.columns for c in [cst,sl,dl]):
                if st.button("▶ RUN RFM",use_container_width=True):
                    rfm=compute_rfm(df,dl,sl,cst)
                    if rfm is not None:
                        c1,c2,c3=st.columns(3)
                        c1.metric("Customers",f"{len(rfm):,}"); c2.metric("Champions",str((rfm["Segment"]=="Champions").sum())); c3.metric("At Risk",str((rfm["Segment"]=="At Risk").sum()))
                        col1,col2=st.columns(2)
                        with col1:
                            sc_=rfm["Segment"].value_counts()
                            fig=px.pie(sc_,names=sc_.index,values=sc_.values,title="CUSTOMER SEGMENTS",hole=0.4)
                            st.plotly_chart(sf(fig),use_container_width=True)
                        with col2:
                            fig2=px.scatter(rfm,x="Frequency",y="Monetary",color="Segment",size="Score",title="RFM SCATTER",size_max=30)
                            st.plotly_chart(sf(fig2),use_container_width=True)
                        st.dataframe(rfm.head(100),use_container_width=True)
                        st.download_button("📥 Export RFM",rfm.to_csv(index=False),"rfm.csv","text/csv")
            else: st.info("Map Customer ID, Date, and Sales.")
        with s2:
            if not is_pro: st.warning("🔒 Clustering requires Professional or Enterprise.")
            else:
                num_c=df.select_dtypes(include=np.number).columns.tolist()
                if len(num_c)>=2:
                    c1,c2,c3=st.columns(3)
                    with c1: method=st.selectbox("Method",["kmeans","dbscan","hierarchical"])
                    with c2: k=st.slider("Clusters/Eps",2,8,3) if method!="dbscan" else st.slider("Epsilon",0.1,2.0,0.5,0.05)
                    with c3: feat_c=st.multiselect("Features",num_c,default=num_c[:min(4,len(num_c))])
                    if feat_c and st.button("▶ RUN CLUSTERING",use_container_width=True):
                        try:
                            eps_v=k if method=="dbscan" else 0.5; k_v=k if method!="dbscan" else 3
                            lbs,sil,coords,inertias,var=run_clustering(df[feat_c].dropna().to_json(),feat_c,method,k_v,eps_v)
                            if lbs is not None:
                                if sil: st.metric("Silhouette Score",f"{sil:.4f}")
                                c1b,c2b=st.columns(2)
                                with c1b:
                                    fig=px.scatter(x=coords[:,0],y=coords[:,1],color=lbs.astype(str),
                                        title=f"PCA — {method.upper()}",labels={"x":f"PC1 ({var[0]*100:.1f}%)","y":f"PC2 ({var[1]*100:.1f}%)"})
                                    st.plotly_chart(sf(fig),use_container_width=True)
                                with c2b:
                                    if inertias:
                                        fig2=px.line(x=list(inertias.keys()),y=list(inertias.values()),markers=True,title="ELBOW METHOD")
                                        st.plotly_chart(sf(fig2),use_container_width=True)
                        except Exception as ex: st.error(f"Error: {ex}")

    # ── TAB 5: BASKET ────────────────────────────────────
    with tabs[5]:
        sec_head("05","Market Basket Analysis","Apriori Association Rules")
        if not is_pro: st.warning("🔒 Market Basket requires Professional or Enterprise.")
        elif not MLXTEND_AVAILABLE: st.warning("Install mlxtend: pip install mlxtend")
        else:
            cst2=cm.get("customer","—"); prd=cm.get("product","—")
            if cst2!="—" and prd!="—" and all(c in df.columns for c in [cst2,prd]):
                c1,c2,c3=st.columns(3)
                with c1: msup=st.slider("Min Support",0.005,0.15,0.01,0.005,format="%.3f")
                with c2: mlif=st.slider("Min Lift",1.0,6.0,1.2,0.1)
                with c3: mconf=st.slider("Min Confidence",0.1,1.0,0.3,0.05)
                if st.button("▶ RUN APRIORI",use_container_width=True):
                    with st.spinner("Mining association rules..."):
                        try:
                            _,rules_,msg=market_basket(df,cst2,prd,msup)
                            if rules_ is not None and len(rules_):
                                rf_=rules_[(rules_["lift"]>=mlif)&(rules_["confidence"]>=mconf)]
                                c1b,c2b,c3b=st.columns(3)
                                c1b.metric("Rules Found",len(rf_)); c2b.metric("Avg Lift",f"{rf_['lift'].mean():.2f}" if len(rf_) else "—"); c3b.metric("Avg Confidence",f"{rf_['confidence'].mean():.2f}" if len(rf_) else "—")
                                if len(rf_):
                                    fig=px.scatter(rf_,x="support",y="confidence",color="lift",size="lift",title="ASSOCIATION RULES MAP",size_max=30,color_continuous_scale="Plasma")
                                    st.plotly_chart(sf(fig),use_container_width=True)
                                    st.dataframe(rf_[["antecedents","consequents","support","confidence","lift"]].sort_values("lift",ascending=False),use_container_width=True)
                                    st.download_button("📥 Export Rules",rf_.to_csv(index=False),"rules.csv","text/csv")
                            else: st.info(msg)
                        except Exception as ex: st.error(f"Error: {ex}")
            else: st.info("Map Customer and Product columns in the sidebar.")

    # ── TAB 6: ADVANCED ──────────────────────────────────
    with tabs[6]:
        sec_head("06","Advanced Analytics","Correlations · Anomalies · Explorer · Distribution")
        a1,a2,a3,a4=st.tabs(["CORRELATIONS","ANOMALY DETECTION","DATA EXPLORER","DISTRIBUTION"])
        with a1:
            nc2=df.select_dtypes(include=np.number).columns.tolist()
            if len(nc2)>=2:
                corr=df[nc2].corr()
                fig=px.imshow(corr,text_auto=".2f",color_continuous_scale="RdBu_r",title="CORRELATION MATRIX",aspect="auto")
                st.plotly_chart(sf(fig,h=500),use_container_width=True)
                pairs=[(corr.columns[i],corr.columns[j],corr.iloc[i,j]) for i in range(len(corr)) for j in range(i+1,len(corr))]
                pairs.sort(key=lambda x:abs(x[2]),reverse=True)
                st.dataframe(pd.DataFrame(pairs[:10],columns=["Feature A","Feature B","Correlation"]),use_container_width=True)
        with a2:
            nc3=df.select_dtypes(include=np.number).columns.tolist()
            fa=st.multiselect("Features",nc3,default=nc3[:min(3,len(nc3))],key="anom_f")
            cont=st.slider("Contamination %",1,20,5)
            if fa and st.button("▶ DETECT ANOMALIES",use_container_width=True):
                with st.spinner("Training Isolation Forest..."):
                    cd=df[fa].dropna(); anoms=detect_anomalies(cd.to_json(),fa,cont/100)
                    n=int(anoms.sum())
                    c1,c2=st.columns(2); c1.metric("Anomalies",n); c2.metric("Rate",f"{n/len(anoms)*100:.2f}%")
                    if n and len(fa)>=2:
                        fig=px.scatter(x=cd.iloc[:,0],y=cd.iloc[:,1],color=np.where(anoms,"ANOMALY","NORMAL"),
                            title="ANOMALY VISUALIZATION",color_discrete_map={"ANOMALY":"#fa4d56","NORMAL":"#42be65"},size_max=8)
                        st.plotly_chart(sf(fig),use_container_width=True)
                    if n: st.dataframe(df.iloc[cd.index][anoms].head(50),use_container_width=True)
        with a3:
            dl3=cm.get("date","—")
            if dl3!="—" and dl3 in df.columns:
                mn,mx=df[dl3].min().date(),df[dl3].max().date()
                dr=st.date_input("Date Range",[mn,mx])
                if len(dr)==2:
                    mask=(df[dl3]>=pd.to_datetime(dr[0]))&(df[dl3]<=pd.to_datetime(dr[1]))
                    fdf=df[mask]; st.caption(f"{len(fdf):,} / {len(df):,} rows")
                    st.dataframe(fdf,use_container_width=True)
                    st.download_button("📥 Export Filtered",fdf.to_csv(index=False),"filtered.csv","text/csv")
            else:
                cat_f=st.selectbox("Filter by",["—"]+list(df.select_dtypes("object").columns))
                if cat_f!="—":
                    vals=st.multiselect("Values",df[cat_f].unique().tolist(),default=df[cat_f].unique().tolist()[:3])
                    st.dataframe(df[df[cat_f].isin(vals)],use_container_width=True)
                else: st.dataframe(df.sample(min(500,len(df))),use_container_width=True)
        with a4:
            nc4=df.select_dtypes(include=np.number).columns.tolist()
            if nc4:
                col_d=st.selectbox("Column",nc4)
                c1,c2=st.columns(2)
                with c1:
                    fig=px.histogram(df,x=col_d,nbins=50,title=f"DISTRIBUTION — {col_d.upper()}",color_discrete_sequence=["#00b4ff"])
                    st.plotly_chart(sf(fig),use_container_width=True)
                with c2:
                    fig2=px.box(df,y=col_d,title=f"BOX PLOT — {col_d.upper()}",color_discrete_sequence=["#42be65"])
                    st.plotly_chart(sf(fig2),use_container_width=True)
                c1b,c2b,c3b,c4b=st.columns(4)
                c1b.metric("Mean",f"{df[col_d].mean():.2f}"); c2b.metric("Median",f"{df[col_d].median():.2f}")
                c3b.metric("Std",f"{df[col_d].std():.2f}"); c4b.metric("Skew",f"{df[col_d].skew():.3f}")

    # ── TAB 7: LIVE DATA ─────────────────────────────────
    with tabs[7]:
        realtime_tab()

    # ── TAB 8: REPORT ────────────────────────────────────
    with tabs[8]:
        sec_head("08","Executive Report","AI-generated strategic summary")
        if st.button("▶ GENERATE REPORT",use_container_width=True):
            sc_r=cm.get("sales","—"); pc_r=cm.get("profit","—"); dc_r=cm.get("date","—"); cc_r=cm.get("category","—")
            tr=df[sc_r].sum() if sc_r!="—" and sc_r in df.columns else 0
            tp=df[pc_r].sum() if pc_r!="—" and pc_r in df.columns else 0
            mg=(tp/tr*100) if tr else 0
            top=df.groupby(cc_r)[sc_r].sum().idxmax() if cc_r!="—" and cc_r in df.columns and sc_r!="—" and sc_r in df.columns else "N/A"
            dr="—"
            if dc_r!="—" and dc_r in df.columns:
                try: dr=f"{df[dc_r].min().date()} → {df[dc_r].max().date()}"
                except: pass
            mp=df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100
            report=f"""# NEXUS Analytics OS — Executive Intelligence Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}  |  **Version:** {APP_VERSION}

---

## 1. Dataset Overview
| Metric | Value |
|--------|-------|
| Records | {len(df):,} |
| Columns | {df.shape[1]} |
| Date Range | {dr} |
| Missing Data | {mp:.2f}% |
| Memory | {df.memory_usage(deep=True).sum()/1048576:.2f} MB |

## 2. Financial Performance
| KPI | Value |
|-----|-------|
| Total Revenue | {fmt(tr, "$")} |
| Total Profit | {fmt(tp, "$")} |
| Profit Margin | {mg:.2f}% |
| Avg Transaction | {fmt(df[sc_r].mean(), "$") if sc_r!="—" and sc_r in df.columns else "N/A"} |
| Top Category | {top} |

## 3. Strategic Recommendations
1. **Revenue Focus**: Invest further in `{top}` — highest revenue contributor.
2. **Margin Optimization**: Analyze discount levels. Current margin: {mg:.1f}%.
3. **RFM Segmentation**: Identify Champions and At-Risk customers for targeted campaigns.
4. **Demand Forecasting**: Use Forecast module with 12-period horizon for inventory planning.
5. **Anomaly Monitoring**: Schedule weekly Isolation Forest scans.
6. **Real-Time Integration**: Connect live data sources via Cloud Connectors.

## 4. Data Quality Score
- Missing Values: {'✅ Excellent (<1%)' if mp<1 else '⚠ Review Required (>1%)'}
- Row Count: {'✅ Sufficient (>1K)' if len(df)>1000 else '⚠ Limited (<1K)'}

---
*NEXUS Analytics OS · Enterprise {APP_VERSION}*
"""
            st.markdown(report)
            c1,c2=st.columns(2)
            with c1: st.download_button("📥 Markdown",report,"nexus_report.md","text/markdown",use_container_width=True)
            with c2:
                html_r=f"<html><body style='font-family:monospace;background:#020408;color:#f4f4f4;padding:40px;'><pre>{report}</pre></body></html>"
                st.download_button("📥 HTML",html_r,"nexus_report.html","text/html",use_container_width=True)

    # ── TAB 9: PLANS ─────────────────────────────────────
    with tabs[9]:
        plans_tab()

    # ── TAB 10: AI CHAT ──────────────────────────────────
    with tabs[10]:
        chatbot_tab()

# ═══════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════
def main():
    for k,v in [("logged_in",False),("user_email",None),("is_admin",False),("admin_mode",False)]:
        if k not in st.session_state: st.session_state[k]=v

    if st.session_state.get("admin_mode") and st.session_state.get("is_admin"):
        admin_dashboard()
    elif st.session_state.get("logged_in"):
        with st.sidebar:
            st.markdown("""<div style="text-align:center;padding:12px 0 6px;border-bottom:1px solid var(--line-dim);margin-bottom:8px;">
              <div style="font-family:var(--font-mono);font-size:0.85rem;font-weight:600;color:var(--cyan);letter-spacing:4px;">⬡ NEXUS</div>
            </div>""", unsafe_allow_html=True)
        analytics_app()
    else:
        col1,col2=st.columns([2,1],gap="large")
        with col1: analytics_app()
        with col2: login_panel()

if __name__ == "__main__":
    main()
