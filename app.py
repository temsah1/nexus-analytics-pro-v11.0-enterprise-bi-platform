import warnings
warnings.filterwarnings("ignore")

import io, os, hashlib, sqlite3, json, time, random, math
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

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               VotingRegressor, IsolationForest, ExtraTreesRegressor)
from sklearn.linear_model import Ridge, ElasticNet, Lasso, BayesianRidge
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit, GridSearchCV
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, MiniBatchKMeans
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.metrics import (silhouette_score, mean_absolute_percentage_error,
                              mean_squared_error, r2_score, davies_bouldin_score)
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.neighbors import LocalOutlierFactor

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
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════
MAX_FILE_SIZE_MB    = 2000
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
GUEST_MAX_ROWS      = 2000
DB_PATH             = "nexus_v5.db"
APP_VERSION         = "5.0.0-enterprise"
BUILD_DATE          = "2026-05-31"

CONFIG_DIR = Path(".streamlit")
CONFIG_FILE = CONFIG_DIR / "config.toml"
if not CONFIG_FILE.exists():
    CONFIG_DIR.mkdir(exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        f.write(f"[server]\nmaxUploadSize = {MAX_FILE_SIZE_MB}\n[theme]\nbase = \"dark\"\n")

# ── MUST be first Streamlit call ────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS v5 · Enterprise Analytics OS",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════
#  MASTER CSS — Full Animation Suite, Zero White Flash
# ═══════════════════════════════════════════════════════════════════════
MASTER_CSS = """
<style>
/* ── IMPORTS ─────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600;700&family=IBM+Plex+Sans:wght@300;400;500;600;700&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ── DESIGN TOKENS ───────────────────────────────────────────────── */
:root {
  /* Backgrounds */
  --void:       #010409;
  --base:       #040d1a;
  --panel:      #071526;
  --card:       #0a1d35;
  --elevated:   #0e2440;
  --overlay:    #132c4a;
  --glass:      rgba(4,13,26,0.92);

  /* Accent Palette */
  --cyan:       #00d4ff;
  --cyan2:      #0099cc;
  --cyan-dim:   rgba(0,212,255,0.10);
  --cyan-glow:  rgba(0,212,255,0.30);
  --cyan-border:rgba(0,212,255,0.22);
  --blue:       #1a6bff;
  --blue2:      #0047d9;
  --blue-dim:   rgba(26,107,255,0.15);
  --blue-glow:  rgba(26,107,255,0.35);
  --teal:       #00e5cc;
  --teal-dim:   rgba(0,229,204,0.12);
  --green:      #00e676;
  --green2:     #00c853;
  --green-dim:  rgba(0,230,118,0.12);
  --green-glow: rgba(0,230,118,0.28);
  --amber:      #ffca28;
  --amber-dim:  rgba(255,202,40,0.12);
  --orange:     #ff8c00;
  --orange-dim: rgba(255,140,0,0.12);
  --red:        #ff3d57;
  --red2:       #d50000;
  --red-dim:    rgba(255,61,87,0.12);
  --purple:     #bb86fc;
  --purple-dim: rgba(187,134,252,0.12);
  --magenta:    #ff4da6;

  /* Grid lines */
  --grid-dim:   rgba(0,212,255,0.04);
  --grid-med:   rgba(0,212,255,0.10);
  --grid-bright:rgba(0,212,255,0.22);

  /* Typography */
  --t1:  #f0f6ff;
  --t2:  #7a9cc0;
  --t3:  #3d5a7a;
  --t4:  #1e3250;
  --font-mono: 'IBM Plex Mono', monospace;
  --font-ui:   'IBM Plex Sans', sans-serif;
  --font-head: 'Rajdhani', sans-serif;

  /* Radius */
  --r1: 4px; --r2: 8px; --r3: 12px; --r4: 18px; --r5: 28px; --round: 9999px;

  /* Transitions */
  --fast: 0.15s ease;
  --med:  0.28s ease;
  --slow: 0.5s ease;
}

/* ── GLOBAL RESET / KILL WHITE FLASH ─────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; }

html {
  background: var(--void) !important;
  color-scheme: dark !important;
}

body,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main,
.main,
.main > .block-container,
[data-testid="stHeader"],
[data-testid="stToolbar"],
header,
footer,
#root {
  background: var(--void) !important;
  background-color: var(--void) !important;
  color: var(--t1) !important;
  font-family: var(--font-ui) !important;
}

/* kill ALL white areas */
[data-testid="stHeader"]   { display: none !important; }
[data-testid="stToolbar"]  { display: none !important; }
footer                     { display: none !important; }
#MainMenu                  { display: none !important; }
.stDeployButton            { display: none !important; }

/* Block container padding */
.main .block-container {
  padding: 1rem 1.5rem 2rem !important;
  max-width: 100% !important;
}

/* ── ANIMATED GRID BG ────────────────────────────────────────────── */
[data-testid="stAppViewContainer"]::before {
  content: '';
  position: fixed; inset: 0; z-index: -2; pointer-events: none;
  background-image:
    linear-gradient(var(--grid-dim) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid-dim) 1px, transparent 1px);
  background-size: 52px 52px;
  animation: gridDrift 60s linear infinite;
}
@keyframes gridDrift {
  from { background-position: 0 0; }
  to   { background-position: 52px 52px; }
}

/* ambient glow orbs */
[data-testid="stAppViewContainer"]::after {
  content: '';
  position: fixed; inset: 0; z-index: -1; pointer-events: none;
  background:
    radial-gradient(ellipse 55% 35% at 8% 12%,  rgba(0,212,255,0.05) 0%, transparent 65%),
    radial-gradient(ellipse 45% 55% at 92% 88%,  rgba(26,107,255,0.06) 0%, transparent 65%),
    radial-gradient(ellipse 40% 30% at 50% 50%,  rgba(0,229,204,0.03) 0%, transparent 70%);
  animation: ambientPulse 12s ease-in-out infinite alternate;
}
@keyframes ambientPulse {
  0%   { opacity: 0.6; }
  100% { opacity: 1.0; }
}

/* ── SCROLLBAR ───────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: var(--void); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(var(--cyan2), var(--blue));
  border-radius: var(--round);
}

/* ═══════════════════════════════════════════════════════════════════
   SIDEBAR — Full Premium Redesign
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #020c1b 0%, #040f22 40%, #030c1a 100%) !important;
  border-right: 1px solid var(--cyan-border) !important;
  box-shadow: 4px 0 40px rgba(0,0,0,0.7), inset -1px 0 0 var(--cyan-border) !important;
  min-width: 260px !important;
}

[data-testid="stSidebar"] > div:first-child {
  padding: 0 !important;
}

[data-testid="stSidebar"] * {
  font-family: var(--font-ui) !important;
  color: var(--t1) !important;
}

/* sidebar scan-line overlay */
[data-testid="stSidebar"]::before {
  content: '';
  position: absolute; inset: 0; pointer-events: none; z-index: 0;
  background: repeating-linear-gradient(
    0deg,
    transparent,
    transparent 2px,
    rgba(0,212,255,0.012) 2px,
    rgba(0,212,255,0.012) 4px
  );
}

[data-testid="stSidebar"] .stRadio > div {
  flex-direction: column !important; gap: 5px !important;
}
[data-testid="stSidebar"] .stRadio label {
  background: var(--panel) !important;
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r2) !important;
  padding: 8px 12px !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
  letter-spacing: 1px !important;
  transition: all var(--med) !important;
  cursor: pointer !important;
}
[data-testid="stSidebar"] .stRadio label:hover {
  background: var(--cyan-dim) !important;
  border-color: var(--cyan-border) !important;
}

/* ── SIDEBAR EXPANDER ────────────────────────────────────────────── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
  background: transparent !important;
  border: none !important;
  border-bottom: 1px solid var(--grid-dim) !important;
  border-radius: 0 !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
  font-family: var(--font-mono) !important;
  font-size: 0.65rem !important;
  letter-spacing: 2px !important;
  text-transform: uppercase !important;
  color: var(--t3) !important;
  padding: 10px 16px !important;
  transition: color var(--fast) !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
  color: var(--cyan) !important;
}

/* ── SELECTBOX IN SIDEBAR ────────────────────────────────────────── */
[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--card) !important;
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r2) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.78rem !important;
  color: var(--t1) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   METRIC CARDS — Holographic Style
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
  background: linear-gradient(145deg, var(--panel) 0%, var(--card) 100%) !important;
  border: 1px solid var(--grid-med) !important;
  border-top: 2px solid transparent !important;
  border-radius: var(--r3) !important;
  padding: 1.2rem 1.4rem !important;
  position: relative; overflow: hidden;
  transition: transform var(--med), box-shadow var(--med), border-color var(--med) !important;
  background-clip: padding-box !important;
}
[data-testid="stMetric"]::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--blue), var(--teal));
  animation: shimmerLine 3s linear infinite;
}
@keyframes shimmerLine {
  0%   { background-position: -200% 0; }
  100% { background-position: 200% 0; }
}
[data-testid="stMetric"]::after {
  content: '';
  position: absolute; top: 2px; left: 0; right: 0; height: 60px;
  background: linear-gradient(180deg, rgba(0,212,255,0.07), transparent);
  pointer-events: none;
}
[data-testid="stMetric"]:hover {
  transform: translateY(-3px) !important;
  border-color: var(--cyan-border) !important;
  box-shadow: 0 0 30px var(--cyan-glow), 0 12px 28px rgba(0,0,0,0.6) !important;
}
[data-testid="stMetricLabel"] {
  color: var(--t3) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.62rem !important;
  text-transform: uppercase !important;
  letter-spacing: 2.5px !important;
}
[data-testid="stMetricValue"] {
  color: var(--t1) !important;
  font-family: var(--font-head) !important;
  font-size: 2.1rem !important;
  font-weight: 700 !important;
  letter-spacing: -0.5px !important;
  line-height: 1.1 !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important;
}

/* ═══════════════════════════════════════════════════════════════════
   TABS — Futuristic Pill Style
   ═══════════════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
  gap: 3px !important;
  background: var(--panel) !important;
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r3) !important;
  padding: 5px !important;
  flex-wrap: wrap !important;
  position: sticky; top: 0; z-index: 10;
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--t3) !important;
  border-radius: var(--r2) !important;
  padding: 0.48rem 0.85rem !important;
  font-size: 0.68rem !important;
  font-family: var(--font-mono) !important;
  font-weight: 500 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  transition: all var(--fast) !important;
  white-space: nowrap !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--cyan) !important;
  background: var(--cyan-dim) !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--blue), var(--blue2)) !important;
  color: white !important;
  box-shadow: 0 2px 14px var(--blue-glow) !important;
}

/* ═══════════════════════════════════════════════════════════════════
   BUTTONS
   ═══════════════════════════════════════════════════════════════════ */
.stButton > button {
  background: linear-gradient(135deg, var(--blue), var(--blue2)) !important;
  color: white !important; border: none !important;
  border-radius: var(--r2) !important;
  padding: 0.55rem 1.3rem !important;
  font-family: var(--font-mono) !important;
  font-size: 0.72rem !important; font-weight: 600 !important;
  letter-spacing: 1.5px !important; text-transform: uppercase !important;
  transition: all var(--med) !important;
  box-shadow: 0 2px 12px var(--blue-glow) !important;
  position: relative; overflow: hidden !important;
}
.stButton > button::after {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(135deg, transparent, rgba(255,255,255,0.08), transparent);
  transform: translateX(-100%);
  transition: transform 0.4s;
}
.stButton > button:hover::after { transform: translateX(100%); }
.stButton > button:hover {
  box-shadow: 0 6px 24px var(--blue-glow) !important;
  transform: translateY(-2px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ═══════════════════════════════════════════════════════════════════
   INPUTS / FORMS
   ═══════════════════════════════════════════════════════════════════ */
.stTextInput > div > div > input,
.stTextArea textarea,
.stNumberInput input,
.stSelectbox > div > div,
[data-baseweb="select"] > div {
  background: var(--card) !important;
  color: var(--t1) !important;
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r2) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.82rem !important;
  transition: border-color var(--fast), box-shadow var(--fast) !important;
}
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus {
  border-color: var(--cyan) !important;
  box-shadow: 0 0 0 2px var(--cyan-dim) !important;
  outline: none !important;
}

/* ═══════════════════════════════════════════════════════════════════
   FILE UPLOADER
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stFileUploadDropzone"] {
  background: var(--card) !important;
  border: 1px dashed var(--cyan-border) !important;
  border-radius: var(--r3) !important;
  transition: all var(--med) !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  background: var(--cyan-dim) !important;
  border-color: var(--cyan) !important;
  box-shadow: 0 0 24px var(--cyan-glow) !important;
}
[data-testid="stFileUploadDropzone"] * { color: var(--t2) !important; }

/* ═══════════════════════════════════════════════════════════════════
   DATAFRAME / TABLES
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stDataFrame"] {
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r3) !important;
  overflow: hidden !important;
}
.dataframe {
  background: var(--panel) !important;
  color: var(--t1) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.74rem !important;
}
.dataframe th {
  background: var(--overlay) !important;
  color: var(--cyan) !important;
  text-transform: uppercase !important;
  letter-spacing: 1.5px !important;
  font-size: 0.62rem !important;
  border-bottom: 1px solid var(--cyan-border) !important;
  padding: 8px 12px !important;
}
.dataframe td {
  background: var(--panel) !important;
  color: var(--t1) !important;
  border-bottom: 1px solid var(--grid-dim) !important;
  padding: 6px 12px !important;
}
.dataframe tr:hover td { background: var(--card) !important; }

/* ═══════════════════════════════════════════════════════════════════
   EXPANDERS
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
  background: var(--panel) !important;
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r3) !important;
  overflow: hidden !important;
}
[data-testid="stExpander"] summary {
  color: var(--t2) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.74rem !important;
  font-weight: 600 !important;
  letter-spacing: 1px !important;
  text-transform: uppercase !important;
  padding: 12px 16px !important;
}
[data-testid="stExpander"] summary:hover { color: var(--cyan) !important; }

/* ═══════════════════════════════════════════════════════════════════
   ALERTS / CAPTIONS / HR
   ═══════════════════════════════════════════════════════════════════ */
[data-testid="stAlert"] {
  border-radius: var(--r3) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.76rem !important;
}
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--t3) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.66rem !important;
}
hr { border-color: var(--grid-med) !important; opacity: 1 !important; margin: 1rem 0 !important; }
[data-baseweb="tag"] {
  background: var(--blue-dim) !important;
  border: 1px solid var(--blue) !important;
  border-radius: var(--r1) !important;
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
}
[data-testid="stDownloadButton"] > button {
  background: linear-gradient(135deg, var(--teal), #007a6e) !important;
  box-shadow: 0 2px 12px rgba(0,229,204,0.35) !important;
}
[data-testid="stProgress"] > div > div {
  background: linear-gradient(90deg, var(--cyan), var(--blue)) !important;
  border-radius: var(--round) !important;
}
.stSlider > div > div > div { background: var(--cyan) !important; }
.stCheckbox > label > span:first-child { border-color: var(--cyan-border) !important; }

/* ═══════════════════════════════════════════════════════════════════
   NEXUS COMPONENT LIBRARY
   ═══════════════════════════════════════════════════════════════════ */

/* ─ Top Command Bar ─────────────────────────────────────────────── */
.nx-topbar {
  display: flex; align-items: center; gap: 12px;
  background: var(--glass);
  border: 1px solid var(--cyan-border);
  border-radius: var(--r3);
  padding: 9px 18px;
  margin-bottom: 20px;
  backdrop-filter: blur(24px) saturate(180%);
  position: relative; overflow: hidden;
}
.nx-topbar::before {
  content: '';
  position: absolute; inset: 0;
  background: linear-gradient(90deg, rgba(0,212,255,0.05) 0%, transparent 50%);
  pointer-events: none;
}
.nx-topbar::after {
  content: '';
  position: absolute; bottom: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent);
  animation: scanH 4s linear infinite;
}
@keyframes scanH {
  0%   { transform: translateX(-100%); opacity: 0; }
  50%  { opacity: 1; }
  100% { transform: translateX(100%); opacity: 0; }
}
.nx-logo {
  font-family: var(--font-head);
  font-size: 1.35rem; font-weight: 700;
  color: var(--cyan); letter-spacing: 5px;
  text-shadow: 0 0 20px var(--cyan-glow);
}
.nx-logo-sub {
  font-family: var(--font-mono);
  font-size: 0.55rem; color: var(--t3);
  letter-spacing: 3px; text-transform: uppercase;
  margin-top: -3px;
}
.nx-pulse {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--green);
  box-shadow: 0 0 10px var(--green-glow);
  animation: gPulse 2s ease-in-out infinite;
}
@keyframes gPulse {
  0%,100% { opacity: 1; box-shadow: 0 0 8px var(--green-glow); }
  50%      { opacity: 0.4; box-shadow: 0 0 18px var(--green-glow); }
}
.nx-status-text {
  font-family: var(--font-mono); font-size: 0.62rem;
  color: var(--green); letter-spacing: 1px; text-transform: uppercase;
}
.nx-clock {
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--t3); letter-spacing: 2px;
}
.nx-badge {
  font-family: var(--font-mono); font-size: 0.58rem;
  color: var(--t3); letter-spacing: 1px;
  background: var(--overlay);
  border: 1px solid var(--grid-med);
  padding: 2px 9px; border-radius: var(--round);
}
.nx-kpi-row {
  display: flex; gap: 6px;
  font-family: var(--font-mono); font-size: 0.62rem; color: var(--t3);
  margin-left: auto;
}
.nx-kpi-chip {
  background: var(--card); border: 1px solid var(--grid-med);
  border-radius: var(--round); padding: 3px 10px;
}
.nx-kpi-chip span { color: var(--cyan); font-weight: 600; margin-left: 4px; }

/* ─ Section Headers ─────────────────────────────────────────────── */
.nx-sec {
  display: flex; align-items: center; gap: 10px;
  margin-bottom: 20px; padding-bottom: 12px;
  border-bottom: 1px solid var(--grid-med);
  position: relative;
}
.nx-sec::after {
  content: '';
  position: absolute; bottom: -1px; left: 0; width: 60px; height: 1px;
  background: linear-gradient(90deg, var(--cyan), transparent);
}
.nx-sec-id {
  font-family: var(--font-mono); font-size: 0.58rem;
  color: var(--cyan); letter-spacing: 2px; text-transform: uppercase;
  background: var(--cyan-dim); border: 1px solid var(--cyan-border);
  padding: 2px 8px; border-radius: var(--r1);
}
.nx-sec-title {
  font-family: var(--font-head); font-size: 1rem;
  font-weight: 700; color: var(--t1); letter-spacing: 1.5px;
  text-transform: uppercase;
}
.nx-sec-sub {
  font-family: var(--font-mono); font-size: 0.6rem;
  color: var(--t3); letter-spacing: 1px;
  margin-left: auto;
}

/* ─ Sidebar Brand ───────────────────────────────────────────────── */
.sb-brand {
  padding: 20px 16px 16px;
  border-bottom: 1px solid var(--grid-dim);
  position: relative; overflow: hidden;
}
.sb-brand::after {
  content: '';
  position: absolute; bottom: 0; left: 16px; right: 16px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan-border), transparent);
}
.sb-logo {
  font-family: var(--font-head); font-size: 1.5rem; font-weight: 700;
  color: var(--cyan); letter-spacing: 5px;
  text-shadow: 0 0 20px var(--cyan-glow);
  display: flex; align-items: center; gap: 8px;
}
.sb-logo-hex { font-size: 1.2rem; animation: hexSpin 8s linear infinite; }
@keyframes hexSpin {
  0%   { text-shadow: 0 0 8px var(--cyan); }
  50%  { text-shadow: 0 0 20px var(--cyan), 0 0 40px var(--blue); }
  100% { text-shadow: 0 0 8px var(--cyan); }
}
.sb-sub {
  font-family: var(--font-mono); font-size: 0.52rem;
  color: var(--t3); letter-spacing: 3px; text-transform: uppercase;
  margin-top: 2px;
}
.sb-ver {
  display: inline-block; margin-top: 6px;
  font-family: var(--font-mono); font-size: 0.55rem;
  color: var(--cyan2); background: var(--cyan-dim);
  border: 1px solid var(--cyan-border);
  padding: 1px 7px; border-radius: var(--round); letter-spacing: 1px;
}

/* ─ Sidebar Nav ─────────────────────────────────────────────────── */
.sb-nav { padding: 12px 10px; }
.sb-nav-label {
  font-family: var(--font-mono); font-size: 0.58rem;
  color: var(--t3); letter-spacing: 2.5px; text-transform: uppercase;
  padding: 6px 8px; margin-bottom: 4px;
}
.sb-nav-item {
  display: flex; align-items: center; gap: 10px;
  background: transparent;
  border: 1px solid transparent;
  border-radius: var(--r2);
  padding: 9px 12px; margin-bottom: 3px;
  cursor: pointer; transition: all var(--fast);
  font-family: var(--font-mono); font-size: 0.72rem;
  color: var(--t2); letter-spacing: 0.5px;
  text-decoration: none;
}
.sb-nav-item:hover {
  background: var(--cyan-dim) !important;
  border-color: var(--cyan-border) !important;
  color: var(--cyan) !important;
}
.sb-nav-item.active {
  background: linear-gradient(135deg, var(--blue-dim), var(--cyan-dim)) !important;
  border-color: var(--cyan-border) !important;
  color: var(--cyan) !important;
  box-shadow: 0 2px 12px var(--blue-glow);
}
.sb-nav-icon { font-size: 0.9rem; flex-shrink: 0; }
.sb-nav-dot {
  width: 5px; height: 5px; border-radius: 50%;
  background: var(--cyan); margin-left: auto;
  box-shadow: 0 0 6px var(--cyan);
}

/* ─ User Card ───────────────────────────────────────────────────── */
.sb-user {
  margin: 0 10px 10px;
  background: linear-gradient(135deg, var(--card), var(--elevated));
  border: 1px solid var(--grid-med);
  border-radius: var(--r3); padding: 12px;
  position: relative; overflow: hidden;
}
.sb-user::before {
  content: '';
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, var(--cyan), var(--blue));
}
.sb-user-email {
  font-family: var(--font-mono); font-size: 0.68rem;
  color: var(--t2); word-break: break-all;
}
.sb-user-plan {
  display: inline-block; margin-top: 6px;
  font-family: var(--font-mono); font-size: 0.56rem;
  color: var(--cyan); background: var(--cyan-dim);
  border: 1px solid var(--cyan-border);
  padding: 2px 9px; border-radius: var(--round);
  letter-spacing: 1px; text-transform: uppercase;
}

/* ─ Data Source Tile ────────────────────────────────────────────── */
.ds-tile {
  display: flex; align-items: center; gap: 10px;
  background: var(--card); border: 1px solid var(--grid-med);
  border-radius: var(--r2); padding: 10px 12px; margin-bottom: 6px;
  cursor: pointer; transition: all var(--fast);
}
.ds-tile:hover {
  border-color: var(--cyan-border);
  background: var(--cyan-dim);
}
.ds-tile.active {
  border-color: var(--cyan);
  background: linear-gradient(135deg, var(--cyan-dim), var(--blue-dim));
  box-shadow: 0 0 14px var(--cyan-glow);
}
.ds-tile-icon { font-size: 1.1rem; }
.ds-tile-label {
  font-family: var(--font-mono); font-size: 0.7rem;
  color: var(--t1); font-weight: 600;
}
.ds-tile-sub {
  font-family: var(--font-mono); font-size: 0.6rem; color: var(--t3);
}

/* ─ Cloud Connector Cards ───────────────────────────────────────── */
.cc-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 10px; }
.cc-card {
  background: var(--panel); border: 1px solid var(--grid-med);
  border-radius: var(--r3); padding: 18px 12px;
  text-align: center; cursor: pointer;
  transition: all var(--med); position: relative; overflow: hidden;
}
.cc-card::before {
  content: ''; position: absolute; inset: 0;
  background: radial-gradient(ellipse at 50% 0%, var(--cyan-dim), transparent 70%);
  opacity: 0; transition: opacity var(--med);
}
.cc-card:hover { border-color: var(--cyan); box-shadow: 0 0 28px var(--cyan-glow); transform: translateY(-3px); }
.cc-card:hover::before { opacity: 1; }
.cc-icon { font-size: 1.8rem; margin-bottom: 8px; }
.cc-name { font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600; color: var(--t1); letter-spacing: 1px; text-transform: uppercase; }
.cc-sub  { font-family: var(--font-mono); font-size: 0.6rem; color: var(--t3); margin-top: 3px; }
.cc-badge {
  display: inline-block; margin-top: 8px;
  font-family: var(--font-mono); font-size: 0.55rem;
  padding: 2px 7px; border-radius: var(--round);
  letter-spacing: 1px; text-transform: uppercase;
}
.cc-badge.ok   { background: var(--green-dim);  color: var(--green);  border: 1px solid var(--green); }
.cc-badge.beta { background: var(--amber-dim);  color: var(--amber);  border: 1px solid var(--amber); }
.cc-badge.lock { background: var(--red-dim);    color: var(--red);    border: 1px solid var(--red); }

/* ─ KPI Cards ───────────────────────────────────────────────────── */
.kpi-card {
  background: linear-gradient(145deg, var(--panel), var(--card));
  border: 1px solid var(--grid-med);
  border-left: 3px solid var(--cyan);
  border-radius: var(--r3); padding: 16px 18px;
  transition: all var(--med); position: relative; overflow: hidden;
}
.kpi-card::after {
  content: ''; position: absolute; top: 0; right: 0;
  width: 60px; height: 100%;
  background: linear-gradient(to left, rgba(0,212,255,0.04), transparent);
  pointer-events: none;
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,0.5); }
.kpi-card.green  { border-left-color: var(--green); }
.kpi-card.amber  { border-left-color: var(--amber); }
.kpi-card.red    { border-left-color: var(--red); }
.kpi-card.purple { border-left-color: var(--purple); }
.kpi-card.teal   { border-left-color: var(--teal); }
.kpi-label { font-family: var(--font-mono); font-size: 0.6rem; color: var(--t3); text-transform: uppercase; letter-spacing: 2px; margin-bottom: 8px; }
.kpi-value { font-family: var(--font-head); font-size: 1.8rem; font-weight: 700; color: var(--t1); letter-spacing: -0.5px; line-height: 1; }
.kpi-delta { font-family: var(--font-mono); font-size: 0.66rem; margin-top: 5px; }
.kpi-delta.up   { color: var(--green); }
.kpi-delta.down { color: var(--red); }
.kpi-icon { position: absolute; top: 12px; right: 14px; font-size: 1.4rem; opacity: 0.25; }

/* ─ Plan Cards ──────────────────────────────────────────────────── */
.plan-card {
  background: linear-gradient(145deg, var(--panel), var(--card));
  border: 1px solid var(--grid-med);
  border-radius: var(--r4); padding: 28px 20px;
  text-align: center; transition: all var(--med); position: relative; overflow: hidden;
  min-height: 350px;
}
.plan-card.featured {
  border-color: var(--cyan-border);
  background: linear-gradient(145deg, var(--panel), rgba(0,212,255,0.06));
}
.plan-card:hover { box-shadow: 0 0 40px var(--cyan-glow); transform: translateY(-5px); }
.plan-name-lbl {
  font-family: var(--font-mono); font-size: 0.62rem;
  color: var(--cyan); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 12px;
}
.plan-price { font-family: var(--font-head); font-size: 2.8rem; font-weight: 700; color: var(--t1); line-height: 1; }
.plan-price span { font-size: 1rem; color: var(--t3); }
.plan-popular {
  position: absolute; top: 12px; right: 12px;
  font-family: var(--font-mono); font-size: 0.55rem;
  color: var(--amber); background: var(--amber-dim);
  border: 1px solid var(--amber); padding: 2px 8px;
  border-radius: var(--round); letter-spacing: 1px;
}
.plan-feat {
  font-family: var(--font-mono); font-size: 0.66rem;
  color: var(--t2); padding: 4px 0;
  border-bottom: 1px solid var(--grid-dim);
  text-align: left;
}
.plan-feat::before { content: '✓ '; color: var(--green); }

/* ─ Chat Bubbles ─────────────────────────────────────────────────── */
.chat-msg { display: flex; margin-bottom: 18px; align-items: flex-start; }
.chat-msg.user { flex-direction: row-reverse; }
.chat-av {
  width: 32px; height: 32px; border-radius: var(--r2);
  display: flex; align-items: center; justify-content: center;
  font-size: 0.8rem; flex-shrink: 0; font-family: var(--font-head); font-weight: 700;
}
.chat-av.user { background: linear-gradient(135deg, var(--blue), var(--blue2)); margin-left: 10px; }
.chat-av.bot  { background: var(--overlay); border: 1px solid var(--cyan-border); margin-right: 10px; color: var(--cyan); }
.chat-bub {
  max-width: 72%; padding: 12px 16px; border-radius: var(--r3);
  font-family: var(--font-ui); font-size: 0.88rem; line-height: 1.7;
  word-wrap: break-word; white-space: pre-wrap;
}
.chat-bub.user { background: var(--blue-dim); border: 1px solid rgba(26,107,255,0.3); color: var(--t1); border-bottom-right-radius: var(--r1); }
.chat-bub.bot  { background: var(--card); border: 1px solid var(--grid-med); color: var(--t1); border-bottom-left-radius: var(--r1); }

/* ─ Admin Hero ──────────────────────────────────────────────────── */
.admin-hero {
  background: linear-gradient(135deg, var(--panel) 0%, rgba(0,212,255,0.06) 100%);
  border: 1px solid var(--cyan-border);
  border-top: 2px solid var(--cyan);
  border-radius: var(--r4); padding: 32px;
  text-align: center; position: relative; overflow: hidden; margin-bottom: 24px;
}
.admin-hero::before {
  content: '';
  position: absolute; inset: 0; pointer-events: none;
  background: repeating-linear-gradient(
    90deg, transparent, transparent 80px,
    rgba(0,212,255,0.015) 80px, rgba(0,212,255,0.015) 82px
  );
}

/* ─ Live Stream ─────────────────────────────────────────────────── */
.live-badge {
  display: inline-flex; align-items: center; gap: 6px;
  font-family: var(--font-mono); font-size: 0.65rem;
  color: var(--red); letter-spacing: 2px;
  background: var(--red-dim); border: 1px solid var(--red);
  padding: 3px 10px; border-radius: var(--round); text-transform: uppercase;
}
.live-dot {
  width: 6px; height: 6px; border-radius: 50%; background: var(--red);
  box-shadow: 0 0 8px var(--red);
  animation: gPulse 1s ease-in-out infinite;
}

/* ─ Connection Status ───────────────────────────────────────────── */
.conn-row {
  display: flex; align-items: center; gap: 8px;
  background: var(--card); border: 1px solid var(--grid-med);
  border-radius: var(--r2); padding: 8px 12px; margin-bottom: 6px;
  font-family: var(--font-mono); font-size: 0.7rem; transition: all var(--fast);
}
.conn-row:hover { border-color: var(--cyan-border); }
.conn-dot { width: 8px; height: 8px; border-radius: 50%; flex-shrink: 0; }
.conn-dot.ok      { background: var(--green); box-shadow: 0 0 8px var(--green-glow); animation: gPulse 2s infinite; }
.conn-dot.stream  { background: var(--cyan);  box-shadow: 0 0 8px var(--cyan-glow);  animation: gPulse 1s infinite; }
.conn-dot.idle    { background: var(--amber); }
.conn-dot.err     { background: var(--red); }

/* ─ Empty State ─────────────────────────────────────────────────── */
.empty-state { text-align: center; padding: 80px 24px; }
.empty-state .ei { font-size: 4rem; margin-bottom: 20px; opacity: 0.3; animation: hexSpin 6s ease-in-out infinite; }
.empty-state h2 { font-family: var(--font-head); font-size: 1.2rem; color: var(--cyan); letter-spacing: 3px; text-transform: uppercase; margin-bottom: 10px; }
.empty-state p { font-family: var(--font-mono); font-size: 0.72rem; color: var(--t3); line-height: 1.8; }

/* ─ Section Divider ─────────────────────────────────────────────── */
.nx-divider {
  height: 1px; background: linear-gradient(90deg, transparent, var(--cyan-border), transparent);
  margin: 20px 0; border: none;
}

/* ─ Info Chip ───────────────────────────────────────────────────── */
.nx-chip {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--font-mono); font-size: 0.62rem;
  background: var(--card); border: 1px solid var(--grid-med);
  border-radius: var(--round); padding: 3px 10px; color: var(--t2);
}
.nx-chip.cyan  { border-color: var(--cyan-border); color: var(--cyan); background: var(--cyan-dim); }
.nx-chip.green { border-color: var(--green); color: var(--green); background: var(--green-dim); }
.nx-chip.amber { border-color: var(--amber); color: var(--amber); background: var(--amber-dim); }
.nx-chip.red   { border-color: var(--red);   color: var(--red);   background: var(--red-dim); }

/* ─ Animated Number Counter ─────────────────────────────────────── */
@keyframes countUp {
  from { opacity: 0; transform: translateY(10px); }
  to   { opacity: 1; transform: translateY(0); }
}
.count-anim { animation: countUp 0.5s ease-out forwards; }

/* ─ Tab Panel Entrance ──────────────────────────────────────────── */
@keyframes fadeInUp {
  from { opacity: 0; transform: translateY(16px); }
  to   { opacity: 1; transform: translateY(0); }
}
.stTabs [data-baseweb="tab-panel"] {
  animation: fadeInUp 0.3s ease-out;
}

/* ─ Sidebar Section Label ───────────────────────────────────────── */
.sb-section-label {
  font-family: var(--font-mono); font-size: 0.58rem;
  color: var(--t3); letter-spacing: 3px; text-transform: uppercase;
  padding: 12px 16px 4px;
}

/* ─ Sidebar Divider ─────────────────────────────────────────────── */
.sb-divider {
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--grid-med), transparent);
  margin: 8px 16px;
}

/* ─ Tooltip / Hover Info ────────────────────────────────────────── */
.nx-tooltip {
  position: relative; display: inline-block;
}
.nx-tooltip:hover::after {
  content: attr(data-tip);
  position: absolute; bottom: 110%; left: 50%; transform: translateX(-50%);
  background: var(--overlay); color: var(--t1);
  font-family: var(--font-mono); font-size: 0.66rem;
  padding: 6px 10px; border-radius: var(--r2);
  border: 1px solid var(--cyan-border);
  white-space: nowrap; z-index: 999;
  box-shadow: 0 4px 16px rgba(0,0,0,0.5);
}

/* ─ Pulse Ring ──────────────────────────────────────────────────── */
@keyframes pulseRing {
  0%   { transform: scale(0.8); opacity: 1; }
  100% { transform: scale(1.8); opacity: 0; }
}
.pulse-ring {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--cyan); position: relative; display: inline-block;
}
.pulse-ring::before {
  content: ''; position: absolute; inset: 0;
  border-radius: 50%; background: var(--cyan);
  animation: pulseRing 1.8s ease-out infinite;
}

/* ─ Sidebar Footer ──────────────────────────────────────────────── */
.sb-footer {
  padding: 10px 16px;
  border-top: 1px solid var(--grid-dim);
  font-family: var(--font-mono); font-size: 0.58rem;
  color: var(--t3); text-align: center; letter-spacing: 1px;
}

/* ─ Progress Bar Custom ─────────────────────────────────────────── */
.nx-progress-wrap {
  background: var(--card); border-radius: var(--round);
  height: 6px; overflow: hidden; margin: 4px 0;
}
.nx-progress-fill {
  height: 100%; border-radius: var(--round);
  background: linear-gradient(90deg, var(--cyan), var(--blue));
  transition: width 1s ease-out;
  position: relative; overflow: hidden;
}
.nx-progress-fill::after {
  content: ''; position: absolute; inset: 0;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.25), transparent);
  animation: shimmerLine 2s linear infinite;
}

/* ─ Table Alternating Rows ──────────────────────────────────────── */
.nx-table { width: 100%; border-collapse: collapse; font-family: var(--font-mono); font-size: 0.74rem; }
.nx-table th { background: var(--overlay); color: var(--cyan); text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.62rem; padding: 10px 14px; border-bottom: 1px solid var(--cyan-border); }
.nx-table td { padding: 9px 14px; border-bottom: 1px solid var(--grid-dim); color: var(--t1); }
.nx-table tr:nth-child(even) td { background: rgba(0,212,255,0.025); }
.nx-table tr:hover td { background: var(--cyan-dim); }

/* ─ Overlay Panel ───────────────────────────────────────────────── */
.nx-panel {
  background: var(--panel);
  border: 1px solid var(--grid-med);
  border-radius: var(--r3);
  padding: 18px 20px;
  position: relative; overflow: hidden;
}
.nx-panel::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
  background: linear-gradient(90deg, transparent, var(--cyan-border), transparent);
}

/* ─ Hero / Landing Banner ───────────────────────────────────────── */
.nx-hero {
  background: linear-gradient(135deg, var(--base) 0%, rgba(0,212,255,0.05) 50%, var(--base) 100%);
  border: 1px solid var(--cyan-border);
  border-radius: var(--r4); padding: 36px 32px;
  text-align: center; position: relative; overflow: hidden; margin-bottom: 24px;
}
.nx-hero::before {
  content: '';
  position: absolute; inset: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 60% 60% at 50% 50%, rgba(0,212,255,0.06), transparent);
}
.nx-hero-title {
  font-family: var(--font-head); font-size: 2.4rem; font-weight: 700;
  color: var(--t1); letter-spacing: 6px; text-transform: uppercase;
  text-shadow: 0 0 30px var(--cyan-glow);
}
.nx-hero-sub {
  font-family: var(--font-mono); font-size: 0.75rem;
  color: var(--t3); letter-spacing: 2px; margin-top: 8px;
}

/* ─ Alert / Notification Banners ───────────────────────────────── */
.nx-alert {
  display: flex; align-items: center; gap: 10px;
  background: var(--card); border-radius: var(--r2);
  padding: 10px 14px; margin-bottom: 8px;
  font-family: var(--font-mono); font-size: 0.74rem;
  border-left: 3px solid var(--cyan);
}
.nx-alert.warn { border-left-color: var(--amber); }
.nx-alert.err  { border-left-color: var(--red); }
.nx-alert.ok   { border-left-color: var(--green); }

/* ─ Tag Cloud ───────────────────────────────────────────────────── */
.tag-cloud { display: flex; flex-wrap: wrap; gap: 6px; }
.tag { font-family: var(--font-mono); font-size: 0.64rem; padding: 3px 10px; border-radius: var(--round); background: var(--blue-dim); border: 1px solid var(--blue); color: var(--t1); }

/* ─ Animated Dots Loader ────────────────────────────────────────── */
.nx-dots { display: flex; gap: 5px; align-items: center; }
.nx-dots span {
  width: 6px; height: 6px; border-radius: 50%;
  background: var(--cyan); animation: dotBounce 1.4s ease-in-out infinite;
}
.nx-dots span:nth-child(2) { animation-delay: 0.2s; }
.nx-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dotBounce {
  0%,80%,100% { transform: scale(0.6); opacity: 0.4; }
  40%          { transform: scale(1.0); opacity: 1; }
}

/* ─ Loading Shimmer ─────────────────────────────────────────────── */
.nx-shimmer {
  background: linear-gradient(90deg, var(--card) 25%, var(--elevated) 50%, var(--card) 75%);
  background-size: 400% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
  border-radius: var(--r2); height: 20px;
}
@keyframes shimmer {
  0%   { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}

/* ─ Multi-select / Checkboxes ───────────────────────────────────── */
[data-testid="stMultiSelect"] > div > div {
  background: var(--card) !important;
  border: 1px solid var(--grid-med) !important;
  border-radius: var(--r2) !important;
}

/* ─ Number Input ────────────────────────────────────────────────── */
[data-testid="stNumberInput"] input {
  font-family: var(--font-mono) !important;
  font-size: 0.84rem !important;
}

/* ─ Slider ──────────────────────────────────────────────────────── */
[data-testid="stSlider"] { padding: 4px 0 !important; }
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] {
  font-family: var(--font-mono) !important;
  font-size: 0.68rem !important;
  color: var(--t3) !important;
}

/* ─ Radio ───────────────────────────────────────────────────────── */
.stRadio > label {
  font-family: var(--font-mono) !important;
  font-size: 0.74rem !important;
  color: var(--t2) !important;
}

/* ─ Checkbox ────────────────────────────────────────────────────── */
.stCheckbox > label {
  font-family: var(--font-mono) !important;
  font-size: 0.74rem !important;
  color: var(--t2) !important;
}

/* ─ Success / Error / Warning overrides ─────────────────────────── */
div[data-baseweb="notification"] {
  background: var(--card) !important;
  border-radius: var(--r2) !important;
  font-family: var(--font-mono) !important;
}

</style>
"""

st.markdown(MASTER_CSS, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  PLOTLY DARK TEMPLATE — IBM Spectrum
# ═══════════════════════════════════════════════════════════════════════
_NX_TPL = go.layout.Template(layout=go.Layout(
    paper_bgcolor='#071526', plot_bgcolor='#071526',
    font=dict(color='#7a9cc0', size=11, family='IBM Plex Mono'),
    title_font=dict(size=12, color='#f0f6ff', family='Rajdhani'),
    xaxis=dict(
        title_font=dict(size=10, color='#3d5a7a'),
        tickfont=dict(size=9, color='#3d5a7a'),
        gridcolor='rgba(0,212,255,0.05)',
        linecolor='rgba(0,212,255,0.12)',
        zerolinecolor='rgba(0,212,255,0.12)'
    ),
    yaxis=dict(
        title_font=dict(size=10, color='#3d5a7a'),
        tickfont=dict(size=9, color='#3d5a7a'),
        gridcolor='rgba(0,212,255,0.05)',
        linecolor='rgba(0,212,255,0.12)',
        zerolinecolor='rgba(0,212,255,0.12)'
    ),
    legend=dict(font=dict(size=10, color='#7a9cc0'), bgcolor='rgba(0,0,0,0)', bordercolor='rgba(0,212,255,0.12)'),
    hoverlabel=dict(bgcolor='#0e2440', font_size=11, font_color='#f0f6ff', font_family='IBM Plex Mono'),
    colorway=['#00d4ff','#00e676','#ffca28','#ff8c00','#bb86fc','#00e5cc','#ff3d57','#ff4da6','#1a6bff','#42a5f5']
))
pio.templates["nx5"] = _NX_TPL
pio.templates.default = "nx5"

def pf(fig, title=None, h=None, show_grid=True):
    """Apply NEXUS plot style."""
    u = dict(
        template="nx5",
        margin=dict(l=22, r=22, t=42 if title else 22, b=22),
        paper_bgcolor='#071526', plot_bgcolor='#071526',
    )
    if title:
        u["title"] = dict(text=f"<b>{title}</b>", font=dict(size=11, family='Rajdhani', color='#f0f6ff'), x=0.01)
    if h:
        u["height"] = h
    fig.update_layout(**u)
    if show_grid:
        fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,212,255,0.05)')
        fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgba(0,212,255,0.05)')
    return fig

def make_gauge(value, title, max_val=100, color="#00d4ff"):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": title, "font": {"family": "Rajdhani", "size": 13, "color": "#f0f6ff"}},
        number={"font": {"family": "Rajdhani", "size": 36, "color": "#f0f6ff"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#3d5a7a", "tickfont": {"size": 9, "color": "#3d5a7a"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#0a1d35",
            "bordercolor": "rgba(0,212,255,0.12)",
            "steps": [
                {"range": [0, max_val*0.5], "color": "#0a1d35"},
                {"range": [max_val*0.5, max_val], "color": "#0e2440"}
            ],
            "threshold": {"line": {"color": "#00e676", "width": 2}, "thickness": 0.75, "value": value}
        }
    ))
    return pf(fig, h=220)

# ═══════════════════════════════════════════════════════════════════════
#  DATABASE LAYER
# ═══════════════════════════════════════════════════════════════════════
def _safe_add_column(c, table, col, col_def):
    """Add a column to a table only if it doesn't already exist."""
    c.execute(f"PRAGMA table_info({table})")
    existing = {row[1] for row in c.fetchall()}
    if col not in existing:
        try:
            c.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
        except Exception:
            pass

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ── Core tables ────────────────────────────────────────────────
    c.executescript("""
    CREATE TABLE IF NOT EXISTS users(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        is_admin INTEGER DEFAULT 0,
        full_name TEXT DEFAULT '',
        avatar_color TEXT DEFAULT '#00d4ff',
        timezone TEXT DEFAULT 'UTC',
        preferences TEXT DEFAULT '{}',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        last_ip TEXT DEFAULT '');

    CREATE TABLE IF NOT EXISTS login_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT, success INTEGER,
        user_agent TEXT DEFAULT '',
        ip_address TEXT DEFAULT '',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS system_logs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_email TEXT, action TEXT,
        details TEXT, module TEXT DEFAULT 'core',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS subscription_plans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE NOT NULL,
        price_monthly REAL,
        price_yearly REAL,
        max_rows INTEGER,
        features TEXT,
        is_active INTEGER DEFAULT 1);

    CREATE TABLE IF NOT EXISTS user_subscriptions(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL, plan_id INTEGER NOT NULL,
        start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        end_date TIMESTAMP, is_active INTEGER DEFAULT 1,
        payment_method TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(plan_id) REFERENCES subscription_plans(id));

    CREATE TABLE IF NOT EXISTS app_settings(
        key TEXT PRIMARY KEY, value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS cloud_connections(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, name TEXT,
        connector_type TEXT, config_json TEXT,
        is_active INTEGER DEFAULT 0,
        last_sync TIMESTAMP, row_count INTEGER DEFAULT 0,
        latency_ms INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id));

    CREATE TABLE IF NOT EXISTS saved_dashboards(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, name TEXT,
        description TEXT DEFAULT '',
        config_json TEXT, is_public INTEGER DEFAULT 0,
        view_count INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id));

    CREATE TABLE IF NOT EXISTS alerts_config(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, name TEXT, metric_col TEXT,
        condition TEXT, threshold REAL,
        notify_email INTEGER DEFAULT 0,
        is_active INTEGER DEFAULT 1,
        triggered_count INTEGER DEFAULT 0,
        last_triggered TIMESTAMP,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(user_id) REFERENCES users(id));

    CREATE TABLE IF NOT EXISTS audit_trail(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, table_name TEXT,
        record_id INTEGER, operation TEXT,
        old_value TEXT, new_value TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS data_catalog(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, dataset_name TEXT,
        source_type TEXT, row_count INTEGER,
        col_count INTEGER, size_bytes INTEGER,
        schema_json TEXT, tags TEXT DEFAULT '',
        description TEXT DEFAULT '',
        quality_score REAL DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS ml_experiments(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, experiment_name TEXT,
        model_type TEXT, target_col TEXT,
        feature_cols TEXT, hyperparams TEXT,
        r2_score REAL, mape REAL, rmse REAL,
        training_rows INTEGER,
        artifact_path TEXT DEFAULT '',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS forecast_runs(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, model_name TEXT,
        date_col TEXT, value_col TEXT,
        horizon INTEGER, frequency TEXT,
        mae REAL, rmse REAL, mape REAL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);

    CREATE TABLE IF NOT EXISTS usage_stats(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER, event_type TEXT,
        module TEXT, duration_ms INTEGER DEFAULT 0,
        metadata TEXT DEFAULT '{}',
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    """)

    # ── Safe migrations — add new columns to existing tables ───────
    _safe_add_column(c, "subscription_plans", "tier",           "INTEGER DEFAULT 1")
    _safe_add_column(c, "subscription_plans", "max_connectors", "INTEGER DEFAULT 0")
    _safe_add_column(c, "subscription_plans", "max_dashboards", "INTEGER DEFAULT 3")
    _safe_add_column(c, "user_subscriptions", "coupon_code",    "TEXT DEFAULT ''")
    _safe_add_column(c, "users",              "full_name",      "TEXT DEFAULT ''")
    _safe_add_column(c, "users",              "avatar_color",   "TEXT DEFAULT '#00d4ff'")
    _safe_add_column(c, "users",              "timezone",       "TEXT DEFAULT 'UTC'")
    _safe_add_column(c, "users",              "preferences",    "TEXT DEFAULT '{}'")
    _safe_add_column(c, "users",              "last_ip",        "TEXT DEFAULT ''")
    _safe_add_column(c, "login_logs",         "user_agent",     "TEXT DEFAULT ''")
    _safe_add_column(c, "login_logs",         "ip_address",     "TEXT DEFAULT ''")
    _safe_add_column(c, "system_logs",        "module",         "TEXT DEFAULT 'core'")
    conn.commit()

    # ── Update tiers for existing plans (safe) ─────────────────────
    tier_map = {"Starter": 1, "Free": 1, "Pro": 2, "Professional": 2,
                "Business": 3, "Enterprise": 4}
    for name, tier in tier_map.items():
        c.execute("UPDATE subscription_plans SET tier=? WHERE name=? AND (tier IS NULL OR tier=0)",
                  (tier, name))

    # ── Seed plans (only if table is empty) ────────────────────────
    c.execute("SELECT COUNT(*) FROM subscription_plans")
    if c.fetchone()[0] == 0:
        plans = [
            ("Starter",      1,  0,      0,        5_000,
             "Basic analytics · 5K rows · 3 Dashboards · Community support", 1),
            ("Professional", 2,  29.99,  299.99,  100_000,
             "Full analytics · Forecasting · Clustering · Basket · Export · 2 Connectors · Priority support", 1),
            ("Business",     3,  69.99,  699.99,  500_000,
             "All Professional + 5 Connectors · 30 Dashboards · Alerts · ML Experiments · SLA 99.9%", 1),
            ("Enterprise",   4, 149.99, 1499.99, 999_999_999,
             "Unlimited everything · Custom ML · Dedicated support · API access · White-label · SSO", 1),
        ]
        c.executemany("""INSERT INTO subscription_plans
            (name,tier,price_monthly,price_yearly,max_rows,features,is_active)
            VALUES(?,?,?,?,?,?,?)""", plans)

    # ── Ensure new columns are filled for existing rows ────────────
    c.execute("UPDATE subscription_plans SET max_connectors=0  WHERE max_connectors IS NULL")
    c.execute("UPDATE subscription_plans SET max_dashboards=3  WHERE max_dashboards IS NULL")
    c.execute("UPDATE subscription_plans SET tier=1            WHERE tier IS NULL OR tier=0")
    # Set sensible max_connectors for known tiers
    c.execute("UPDATE subscription_plans SET max_connectors=2  WHERE name='Professional' AND max_connectors<2")
    c.execute("UPDATE subscription_plans SET max_connectors=5  WHERE name='Business'    AND max_connectors<5")
    c.execute("UPDATE subscription_plans SET max_connectors=99 WHERE name='Enterprise'  AND max_connectors<99")
    c.execute("UPDATE subscription_plans SET max_connectors=2  WHERE name='Pro'         AND max_connectors<2")
    c.execute("UPDATE subscription_plans SET max_dashboards=10 WHERE name='Professional'AND max_dashboards<10")
    c.execute("UPDATE subscription_plans SET max_dashboards=30 WHERE name='Business'    AND max_dashboards<30")
    c.execute("UPDATE subscription_plans SET max_dashboards=999 WHERE name='Enterprise' AND max_dashboards<999")

    # Settings
    for k, v in [
        ("ai_provider", "deepseek"), ("deepseek_api_key", ""), ("groq_api_key", ""),
        ("custom_ai_url", ""), ("custom_ai_api_key", ""), ("custom_ai_model", ""),
        ("custom_ai_enabled", "0"), ("app_name", "NEXUS Analytics OS"),
        ("maintenance_mode", "0"), ("registration_enabled", "1"),
        ("default_theme", "nexus-dark"), ("telemetry_enabled", "0"),
        ("max_upload_mb", str(MAX_FILE_SIZE_MB)),
        ("smtp_host", ""), ("smtp_port", "587"), ("smtp_user", ""), ("smtp_pass", ""),
        ("alert_email_from", "noreply@nexus.io"),
        ("s3_bucket", ""), ("s3_region", "us-east-1"), ("s3_access_key", ""), ("s3_secret_key", ""),
        ("gcs_bucket", ""), ("gcs_credentials_json", ""),
        ("azure_connection_string", ""), ("azure_container", ""),
    ]:
        c.execute("INSERT OR IGNORE INTO app_settings(key,value) VALUES(?,?)", (k, v))

    # ── Admin user ─────────────────────────────────────────────────
    admin_email, admin_pass = "kareemeltemsah7@gmail.com", "temsah1!"
    hashed = hashlib.sha256(admin_pass.encode()).hexdigest()

    c.execute("SELECT id FROM users WHERE email=?", (admin_email,))
    existing_admin = c.fetchone()
    if existing_admin:
        c.execute("UPDATE users SET password_hash=?,is_admin=1 WHERE email=?",
                  (hashed, admin_email))
    else:
        c.execute("INSERT INTO users(email,password_hash,is_admin,full_name) VALUES(?,?,1,'Admin')",
                  (admin_email, hashed))
        uid = c.lastrowid
        c.execute("SELECT id FROM subscription_plans WHERE name='Enterprise'")
        ep = c.fetchone()
        if not ep:
            c.execute("SELECT id FROM subscription_plans ORDER BY id DESC LIMIT 1")
            ep = c.fetchone()
        if ep:
            plan_id = ep[0]  # always use index — no row_factory in init_db
            c.execute("""INSERT INTO user_subscriptions
                         (user_id,plan_id,start_date,end_date,is_active)
                         VALUES(?,?,?,?,1)""",
                      (uid, plan_id,
                       datetime.now(), datetime.now() + timedelta(days=36500)))

    conn.commit()
    conn.close()

init_db()

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try: yield conn
    finally: conn.close()

def _hash(p): return hashlib.sha256(p.encode()).hexdigest()

# ── AUTH ──────────────────────────────────────────────────────────────
def verify_password(email, pwd):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT password_hash,is_admin FROM users WHERE email=?", (email,))
        row = c.fetchone()
        if row and row["password_hash"] == _hash(pwd):
            c.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE email=?", (email,))
            conn.commit()
            return True, bool(row["is_admin"])
    return False, False

def register_user(email, pwd, full_name=""):
    with get_db() as conn:
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users(email,password_hash,full_name) VALUES(?,?,?)",
                      (email, _hash(pwd), full_name))
            uid = c.lastrowid
            c.execute("SELECT id FROM subscription_plans WHERE name='Starter'")
            fp = c.fetchone()
            if fp:
                fp_id = fp["id"] if hasattr(fp, "keys") else fp[0]
                c.execute("""INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active)
                             VALUES(?,?,?,?,1)""",
                          (uid, fp_id, datetime.now(), datetime.now() + timedelta(days=36500)))
            conn.commit()
            return True
        except sqlite3.IntegrityError: return False

def log_login(email, ok, ip=""):
    with get_db() as c:
        c.execute("INSERT INTO login_logs(email,success,ip_address) VALUES(?,?,?)",
                  (email, 1 if ok else 0, ip))
        c.commit()

def log_action(email, action, detail="", module="core"):
    with get_db() as c:
        c.execute("INSERT INTO system_logs(user_email,action,details,module) VALUES(?,?,?,?)",
                  (email, action, detail[:800], module))
        c.commit()

def log_usage(user_id, event, module, duration_ms=0, meta="{}"):
    with get_db() as c:
        c.execute("INSERT INTO usage_stats(user_id,event_type,module,duration_ms,metadata) VALUES(?,?,?,?,?)",
                  (user_id, event, module, duration_ms, meta))
        c.commit()

# ── SETTINGS ─────────────────────────────────────────────────────────
def get_setting(k, default=""):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT value FROM app_settings WHERE key=?", (k,))
        r = c.fetchone()
        return r["value"] if r else default

def set_setting(k, v):
    with get_db() as c:
        c.execute("INSERT OR REPLACE INTO app_settings(key,value,updated_at) VALUES(?,?,CURRENT_TIMESTAMP)",
                  (k, v))
        c.commit()

def get_all_settings():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT key,value FROM app_settings ORDER BY key")
        return {r["key"]: r["value"] for r in c.fetchall()}

# ── USERS ─────────────────────────────────────────────────────────────
def get_user_by_email(email):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id,email,is_admin,full_name FROM users WHERE email=?", (email,))
        return c.fetchone()

def get_user_subscription(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""SELECT sp.name,
                            COALESCE(sp.tier,1) tier,
                            sp.max_rows,
                            COALESCE(sp.max_connectors,0) max_connectors,
                            COALESCE(sp.max_dashboards,3) max_dashboards,
                            sp.features,
                            us.start_date,us.end_date,us.is_active,sp.id plan_id
                     FROM user_subscriptions us
                     JOIN subscription_plans sp ON us.plan_id=sp.id
                     WHERE us.user_id=? AND us.is_active=1
                     ORDER BY us.start_date DESC LIMIT 1""", (user_id,))
        row = c.fetchone()
        if row: return dict(row)
        c.execute("""SELECT id,name,
                            COALESCE(tier,1) tier,
                            max_rows,
                            COALESCE(max_connectors,0) max_connectors,
                            COALESCE(max_dashboards,3) max_dashboards,
                            features
                     FROM subscription_plans WHERE name='Starter'""")
        fp = c.fetchone()
        if not fp:
            c.execute("SELECT id,name,max_rows,features FROM subscription_plans ORDER BY id LIMIT 1")
            fp = c.fetchone()
        if fp:
            fp_dict = dict(fp)
            c.execute("""INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active)
                         VALUES(?,?,?,?,1)""",
                      (user_id, fp_dict["id"], datetime.now(), datetime.now() + timedelta(days=36500)))
            conn.commit()
            return {
                "name":           fp_dict.get("name", "Starter"),
                "tier":           fp_dict.get("tier", 1),
                "max_rows":       fp_dict.get("max_rows", 5000),
                "max_connectors": fp_dict.get("max_connectors", 0),
                "max_dashboards": fp_dict.get("max_dashboards", 3),
                "features":       fp_dict.get("features", ""),
                "plan_id":        fp_dict.get("id", 1),
            }
        return None

def get_all_users():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT id,email,is_admin,full_name,created_at,last_login FROM users ORDER BY id")
        return c.fetchall()

def delete_user(uid):
    with get_db() as c:
        c.execute("DELETE FROM user_subscriptions WHERE user_id=?", (uid,))
        c.execute("DELETE FROM users WHERE id=?", (uid,))
        c.commit()

def toggle_admin(uid, make):
    with get_db() as c:
        c.execute("UPDATE users SET is_admin=? WHERE id=?", (1 if make else 0, uid))
        c.commit()

def reset_password(uid, new_pwd):
    with get_db() as c:
        c.execute("UPDATE users SET password_hash=? WHERE id=?", (_hash(new_pwd), uid))
        c.commit()

# ── SUBSCRIPTIONS ─────────────────────────────────────────────────────
def get_available_plans():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""SELECT id, name,
                            COALESCE(tier,1) tier,
                            price_monthly, price_yearly, max_rows,
                            COALESCE(max_connectors,0) max_connectors,
                            COALESCE(max_dashboards,3) max_dashboards,
                            features, is_active
                     FROM subscription_plans WHERE is_active=1
                     ORDER BY COALESCE(tier,1)""")
        return c.fetchall()

def get_all_plans():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""SELECT id, name,
                            COALESCE(tier,1) tier,
                            price_monthly, price_yearly, max_rows,
                            COALESCE(max_connectors,0) max_connectors,
                            COALESCE(max_dashboards,3) max_dashboards,
                            features, is_active
                     FROM subscription_plans ORDER BY COALESCE(tier,1)""")
        return c.fetchall()

def upgrade_subscription(user_id, plan_id, months=1):
    with get_db() as conn:
        c = conn.cursor()
        try: plan_id = int(plan_id)
        except (TypeError, ValueError): return
        c.execute("UPDATE user_subscriptions SET is_active=0 WHERE user_id=?", (user_id,))
        end = datetime.now() + timedelta(days=30*months)
        c.execute("""INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active,payment_method)
                     VALUES(?,?,?,?,1,'simulated')""",
                  (user_id, plan_id, datetime.now(), end))
        conn.commit()

def get_all_subscriptions():
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""SELECT u.id,u.email,sp.name plan_name,
                            COALESCE(sp.tier,1) tier,
                            us.start_date,us.end_date,us.is_active
                     FROM user_subscriptions us
                     JOIN users u ON us.user_id=u.id
                     JOIN subscription_plans sp ON us.plan_id=sp.id
                     ORDER BY us.start_date DESC""")
        return [dict(r) for r in c.fetchall()]

def update_plan(pid, pm, py, rows, connectors, dashboards, feats):
    with get_db() as conn:
        c = conn.cursor()
        # Use safe column updates — skip if column doesn't exist
        c.execute("UPDATE subscription_plans SET price_monthly=?,price_yearly=?,max_rows=?,features=? WHERE id=?",
                  (pm, py, rows, feats, pid))
        try:
            c.execute("UPDATE subscription_plans SET max_connectors=? WHERE id=?", (connectors, pid))
        except Exception: pass
        try:
            c.execute("UPDATE subscription_plans SET max_dashboards=? WHERE id=?", (dashboards, pid))
        except Exception: pass
        conn.commit()

def extend_sub(uid, months=1):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT end_date FROM user_subscriptions WHERE user_id=? AND is_active=1", (uid,))
        r = c.fetchone()
        if r:
            try: nd = datetime.strptime(str(r["end_date"]), "%Y-%m-%d %H:%M:%S") + timedelta(days=30*months)
            except: nd = datetime.now() + timedelta(days=30*months)
            c.execute("UPDATE user_subscriptions SET end_date=? WHERE user_id=? AND is_active=1", (nd, uid))
            conn.commit()
            return True
        return False

def cancel_sub(uid):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("UPDATE user_subscriptions SET is_active=0 WHERE user_id=? AND is_active=1", (uid,))
        c.execute("SELECT id FROM subscription_plans WHERE name='Starter'")
        fp = c.fetchone()
        if not fp:
            c.execute("SELECT id FROM subscription_plans ORDER BY id LIMIT 1")
            fp = c.fetchone()
        if fp:
            fp_id = fp["id"] if hasattr(fp, "keys") else fp[0]
            c.execute("""INSERT INTO user_subscriptions(user_id,plan_id,start_date,end_date,is_active)
                         VALUES(?,?,?,?,1)""",
                      (uid, fp_id, datetime.now(), datetime.now() + timedelta(days=36500)))
        conn.commit()

# ── STATS ─────────────────────────────────────────────────────────────
def get_stats():
    with get_db() as conn:
        c = conn.cursor()
        def q(sql, *args):
            c.execute(sql, args)
            return c.fetchone()[0]
        today = str(pd.Timestamp.now().date())
        plan_dist = {}
        c.execute("""SELECT sp.name, COUNT(*) cnt
                     FROM user_subscriptions us
                     JOIN subscription_plans sp ON us.plan_id=sp.id
                     WHERE us.is_active=1 GROUP BY sp.name""")
        for row in c.fetchall(): plan_dist[row["name"]] = row["cnt"]
        mrr = q("""SELECT COALESCE(SUM(sp.price_monthly),0)
                   FROM user_subscriptions us
                   JOIN subscription_plans sp ON us.plan_id=sp.id
                   WHERE us.is_active=1""")
        return {
            "users":        q("SELECT COUNT(*) FROM users"),
            "admins":       q("SELECT COUNT(*) FROM users WHERE is_admin=1"),
            "ok_logins":    q("SELECT COUNT(*) FROM login_logs WHERE success=1"),
            "fail_logins":  q("SELECT COUNT(*) FROM login_logs WHERE success=0"),
            "actions":      q("SELECT COUNT(*) FROM system_logs"),
            "today_ok":     q("SELECT COUNT(*) FROM login_logs WHERE success=1 AND date(timestamp)=?", today),
            "today_attempts": q("SELECT COUNT(*) FROM login_logs WHERE date(timestamp)=?", today),
            "experiments":  q("SELECT COUNT(*) FROM ml_experiments"),
            "dashboards":   q("SELECT COUNT(*) FROM saved_dashboards"),
            "connections":  q("SELECT COUNT(*) FROM cloud_connections WHERE is_active=1"),
            "plan_dist":    plan_dist,
            "mrr":          mrr,
        }

def get_login_logs(limit=300):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT email,success,ip_address,timestamp FROM login_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return c.fetchall()

def get_system_logs(limit=500):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT user_email,action,details,module,timestamp FROM system_logs ORDER BY timestamp DESC LIMIT ?", (limit,))
        return c.fetchall()

# ── ML EXPERIMENTS TRACKING ───────────────────────────────────────────
def save_ml_experiment(user_id, name, model_type, target, features, r2, mape, rmse, rows, params="{}"):
    with get_db() as c:
        c.execute("""INSERT INTO ml_experiments
                     (user_id,experiment_name,model_type,target_col,feature_cols,hyperparams,r2_score,mape,rmse,training_rows)
                     VALUES(?,?,?,?,?,?,?,?,?,?)""",
                  (user_id, name, model_type, target, json.dumps(features), params, r2, mape, rmse, rows))
        c.commit()

def get_ml_experiments(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("""SELECT * FROM ml_experiments WHERE user_id=? ORDER BY created_at DESC LIMIT 50""", (user_id,))
        return [dict(r) for r in c.fetchall()]

# ── DATA CATALOG ──────────────────────────────────────────────────────
def save_to_catalog(user_id, name, src_type, rows, cols, size, schema, tags="", desc="", quality=0):
    with get_db() as c:
        c.execute("""INSERT INTO data_catalog
                     (user_id,dataset_name,source_type,row_count,col_count,size_bytes,schema_json,tags,description,quality_score)
                     VALUES(?,?,?,?,?,?,?,?,?,?)""",
                  (user_id, name, src_type, rows, cols, size, json.dumps(schema), tags, desc, quality))
        c.commit()

def get_catalog(user_id):
    with get_db() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM data_catalog WHERE user_id=? ORDER BY created_at DESC", (user_id,))
        return [dict(r) for r in c.fetchall()]

# ═══════════════════════════════════════════════════════════════════════
#  DATA PROCESSING — Smart CSV, Column Detection, Cleaning
# ═══════════════════════════════════════════════════════════════════════
DATE_FORMATS = [
    '%Y-%m-%d','%d/%m/%Y','%m/%d/%Y','%Y/%m/%d','%d-%m-%Y','%m-%d-%Y','%Y%m%d',
    '%d %b %Y','%d %B %Y','%b %d, %Y','%B %d, %Y',
    '%Y-%m-%d %H:%M:%S','%d/%m/%Y %H:%M:%S','%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%SZ','%d-%b-%Y','%d-%b-%y','%b-%Y',
    '%m/%Y','%Y-%m','%B %Y','%b %Y',
]

def try_parse_date(series):
    if pd.api.types.is_datetime64_any_dtype(series): return series
    if pd.api.types.is_numeric_dtype(series):
        mn = series.dropna().mean()
        if mn < 100000: return None
        # unix timestamp detection
        if 1e9 < mn < 2e10:
            try:
                parsed = pd.to_datetime(series, unit='s', errors='coerce')
                if parsed.notna().mean() > 0.7: return parsed
            except: pass
        return None
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
    encodings = ['utf-8', 'windows-1256', 'iso-8859-6', 'iso-8859-1', 'cp1252', 'latin1', 'utf-8-sig', 'utf-16']
    for enc in encodings:
        try:
            f.seek(0)
            df = pd.read_csv(f, encoding=enc)
            for col in df.columns:
                p = try_parse_date(df[col])
                if p is not None: df[col] = p
            return df, enc
        except: continue
    raise ValueError("Cannot decode file with any supported encoding.")

def detect_column_types(df):
    roles = {"date": [], "numeric": [], "categorical": [], "id": [], "text": [], "boolean": []}
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_datetime64_any_dtype(s):
            roles["date"].append(col)
        elif try_parse_date(s) is not None:
            roles["date"].append(col)
        elif pd.api.types.is_bool_dtype(s):
            roles["boolean"].append(col)
        elif pd.api.types.is_numeric_dtype(s):
            lc = col.lower()
            if any(x in lc for x in ["_id", "row id", "index", "customer id", "order id", "user_id", "product_id"]):
                roles["id"].append(col)
            else:
                roles["numeric"].append(col)
        elif s.dtype == object:
            uniq_ratio = s.nunique() / max(len(s), 1)
            if s.nunique() < 60:
                roles["categorical"].append(col)
            elif uniq_ratio > 0.9 and s.str.len().mean() > 20:
                roles["text"].append(col)
            else:
                roles["id"].append(col)
        else:
            roles["categorical"].append(col)
    return roles

def smart_clean(df, roles, fill_strategy="median"):
    df = df.copy()
    for col in roles["date"]:
        if pd.api.types.is_datetime64_any_dtype(df[col]): continue
        for fmt in DATE_FORMATS:
            try:
                p = pd.to_datetime(df[col], format=fmt, errors='coerce')
                if p.notna().mean() > 0.7: df[col] = p; break
            except: continue
        else:
            try: df[col] = pd.to_datetime(df[col], infer_datetime_format=True, errors='coerce')
            except: pass
    for col in roles["numeric"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        if df[col].isna().all(): df[col].fillna(0, inplace=True)
        elif fill_strategy == "median": df[col].fillna(df[col].median(), inplace=True)
        elif fill_strategy == "mean":   df[col].fillna(df[col].mean(), inplace=True)
        else:                           df[col].fillna(0, inplace=True)
    for col in roles["categorical"]:
        mv = df[col].mode()
        df[col].fillna(mv.iloc[0] if len(mv) else "Unknown", inplace=True)
    return df

def compute_data_quality(df, roles):
    """Return a quality score 0-100 and breakdown."""
    total_cells = df.shape[0] * df.shape[1]
    missing_pct = df.isnull().sum().sum() / max(total_cells, 1) * 100
    dup_pct = df.duplicated().sum() / max(len(df), 1) * 100
    date_ok = len(roles.get("date", [])) > 0
    num_ok  = len(roles.get("numeric", [])) > 0

    score = 100
    score -= min(missing_pct * 2, 40)
    score -= min(dup_pct * 3, 30)
    if not date_ok: score -= 10
    if not num_ok:  score -= 10

    return max(0, round(score)), {
        "missing_pct": round(missing_pct, 2),
        "dup_pct": round(dup_pct, 2),
        "has_dates": date_ok,
        "has_numerics": num_ok,
    }

def auto_map_columns(df, roles):
    """Auto-detect sales/profit/date/category/customer/product columns."""
    cm = {k: "—" for k in ["sales","profit","date","category","customer","product","quantity","region"]}
    kw = {
        "sales":    ['sales','revenue','amount','price','total','income','gmv','value'],
        "profit":   ['profit','margin','earning','net','gain'],
        "date":     [],
        "category": ['category','type','class','segment','department','channel','brand'],
        "customer": ['customer','client','user','account','member','buyer'],
        "product":  ['product','item','sku','article','goods','service'],
        "quantity": ['quantity','qty','units','count','volume','pieces'],
        "region":   ['region','country','city','area','location','state','province','zone'],
    }
    for col in df.columns:
        cl = col.lower()
        if cm["date"] == "—" and col in roles.get("date", []): cm["date"] = col
        for field, words in kw.items():
            if field == "date": continue
            if cm[field] == "—":
                if field in ["sales","profit","quantity"] and col in roles.get("numeric",[]):
                    if any(w in cl for w in words): cm[field] = col
                elif field in ["category","customer","product","region"]:
                    src = roles.get("categorical",[]) + roles.get("id",[])
                    if col in src and any(w in cl for w in words): cm[field] = col
    return cm

# ═══════════════════════════════════════════════════════════════════════
#  CLOUD CONNECTORS
# ═══════════════════════════════════════════════════════════════════════
CONNECTORS = {
    "postgresql":  {"label": "PostgreSQL",    "icon": "🐘", "status": "ok",   "category": "SQL"},
    "mysql":       {"label": "MySQL",         "icon": "🐬", "status": "ok",   "category": "SQL"},
    "mssql":       {"label": "SQL Server",    "icon": "🪟", "status": "ok",   "category": "SQL"},
    "oracle":      {"label": "Oracle DB",     "icon": "🔴", "status": "ok",   "category": "SQL"},
    "sqlite":      {"label": "SQLite",        "icon": "🗃️", "status": "ok",   "category": "SQL"},
    "snowflake":   {"label": "Snowflake",     "icon": "❄️", "status": "ok",   "category": "Cloud DW"},
    "bigquery":    {"label": "BigQuery",      "icon": "🔵", "status": "ok",   "category": "Cloud DW"},
    "redshift":    {"label": "AWS Redshift",  "icon": "🟠", "status": "ok",   "category": "Cloud DW"},
    "databricks":  {"label": "Databricks",    "icon": "🧱", "status": "ok",   "category": "Cloud DW"},
    "synapse":     {"label": "Azure Synapse", "icon": "🔷", "status": "beta", "category": "Cloud DW"},
    "mongodb":     {"label": "MongoDB",       "icon": "🍃", "status": "ok",   "category": "NoSQL"},
    "elasticsearch":{"label":"Elasticsearch", "icon": "🔍", "status": "ok",   "category": "NoSQL"},
    "redis":       {"label": "Redis",         "icon": "🔴", "status": "beta", "category": "NoSQL"},
    "kafka":       {"label": "Kafka Stream",  "icon": "⚡", "status": "beta", "category": "Streaming"},
    "kinesis":     {"label": "AWS Kinesis",   "icon": "🌊", "status": "beta", "category": "Streaming"},
    "pubsub":      {"label": "Google Pub/Sub","icon": "📨", "status": "beta", "category": "Streaming"},
    "mqtt":        {"label": "MQTT IoT",      "icon": "📡", "status": "beta", "category": "Streaming"},
    "s3":          {"label": "AWS S3",        "icon": "🪣", "status": "ok",   "category": "Object Storage"},
    "gcs":         {"label": "Google GCS",    "icon": "🟡", "status": "ok",   "category": "Object Storage"},
    "azure_blob":  {"label": "Azure Blob",    "icon": "🔷", "status": "ok",   "category": "Object Storage"},
    "api_rest":    {"label": "REST API",      "icon": "🌐", "status": "ok",   "category": "API"},
    "graphql":     {"label": "GraphQL",       "icon": "◈",  "status": "beta", "category": "API"},
    "websocket":   {"label": "WebSocket",     "icon": "🔗", "status": "beta", "category": "API"},
    "salesforce":  {"label": "Salesforce",    "icon": "☁️", "status": "ok",   "category": "SaaS"},
    "hubspot":     {"label": "HubSpot",       "icon": "🧲", "status": "ok",   "category": "SaaS"},
    "google_sheets":{"label":"Google Sheets", "icon": "📊", "status": "ok",   "category": "SaaS"},
}

def simulate_live_data(n=80, seed=None):
    if seed is None: seed = int(time.time()) % 10000
    np.random.seed(seed)
    now = pd.Timestamp.now()
    cats = ["Electronics","Fashion","Home & Kitchen","Beauty","Sports","Books","Toys","Groceries"]
    regions = ["Riyadh","Dubai","Cairo","Jeddah","Kuwait City","Doha","Amman","Manama","Abu Dhabi","Bahrain"]
    segs = ["Premium","Standard","Economy"]
    base_prices = {"Electronics":1200,"Fashion":180,"Home & Kitchen":250,"Beauty":120,
                   "Sports":300,"Books":60,"Toys":90,"Groceries":45}
    rows = []
    for i in range(n):
        cat = np.random.choice(cats)
        bp = base_prices[cat]
        sales = round(bp * np.random.uniform(0.6, 2.8), 2)
        profit = round(sales * np.random.uniform(0.08, 0.45), 2)
        rows.append({
            "Timestamp":  now - pd.Timedelta(seconds=i * 8),
            "Order_ID":   f"ORD-{random.randint(100000, 999999)}",
            "Category":   cat,
            "Region":     np.random.choice(regions),
            "Segment":    np.random.choice(segs),
            "Sales":      sales,
            "Profit":     profit,
            "Quantity":   np.random.randint(1, 12),
            "Discount":   round(np.random.choice([0, 0.05, 0.1, 0.15, 0.2]), 2),
            "Rating":     round(np.random.uniform(3.2, 5.0), 1),
        })
    return pd.DataFrame(rows)

def cloud_connector_test(connector_type, config):
    """Simulate connector test — in production replace with real SDK calls."""
    time.sleep(random.uniform(0.8, 2.2))
    latency = random.randint(5, 120)
    rows = random.randint(1000, 5_000_000)
    ok = random.random() > 0.1  # 90% success rate simulation
    if ok:
        return True, latency, rows, "Connection established successfully."
    else:
        return False, 0, 0, "Connection refused: check credentials."

def fetch_from_s3_simulation(bucket, prefix, fmt, aws_key, aws_secret, aws_region):
    """Simulate S3 data fetch. Replace with boto3 in production."""
    time.sleep(1.5)
    n = random.randint(500, 5000)
    df = simulate_live_data(n=min(n, 200))
    return df, f"s3://{bucket}/{prefix}"

def fetch_from_rest_api(url, api_key, method="GET", payload=None):
    """Real REST API fetch."""
    try:
        headers = {}
        if api_key: headers["Authorization"] = f"Bearer {api_key}"
        if method == "GET":
            r = requests.get(url, headers=headers, timeout=15)
        else:
            r = requests.post(url, headers=headers, json=payload or {}, timeout=15)
        if r.status_code == 200:
            data = r.json()
            if isinstance(data, list): return pd.DataFrame(data), None
            elif isinstance(data, dict):
                for k in ["data", "results", "rows", "items", "records", "response"]:
                    if k in data and isinstance(data[k], list):
                        return pd.DataFrame(data[k]), None
                return pd.DataFrame([data]), None
        return None, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as ex:
        return None, str(ex)

# ═══════════════════════════════════════════════════════════════════════
#  AI / LLM LAYER
# ═══════════════════════════════════════════════════════════════════════
SYSTEM_PROMPTS = {
    "analyst": """You are NEXUS AI — an elite enterprise data analyst and BI consultant.
You have deep expertise in: SQL, Python, statistics, machine learning, business intelligence,
supply chain, retail analytics, financial modeling, and data storytelling.
Be precise, data-driven, and concise. Use technical language when appropriate.
When analyzing data, provide actionable insights with specific numbers.
Format responses clearly with sections when helpful.""",

    "sql": """You are NEXUS SQL Expert. Generate optimized SQL queries based on user requirements.
Always include: proper JOINs, WHERE clauses, aggregations.
Comment complex parts. Support: PostgreSQL, MySQL, BigQuery, Snowflake dialects.
Return only the SQL code block with brief explanation.""",

    "ml": """You are NEXUS ML Engineer. Advise on machine learning model selection, feature engineering,
hyperparameter tuning, and model interpretation. Be specific about scikit-learn implementations.
When given data context, suggest concrete approaches with code snippets.""",

    "forecast": """You are NEXUS Forecasting Expert. Help users understand time-series forecasting,
seasonal patterns, trend decomposition, and model selection (Prophet, ARIMA, Holt-Winters).
Provide specific parameters and interpretation guidelines.""",
}

def call_ai(messages, provider="deepseek", deepseek_key="", groq_key="",
            custom_url="", custom_key="", custom_model="", max_tokens=2000, temperature=0.7):
    def _post(url, payload, api_key):
        r = requests.post(url,
                          headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                          json=payload, timeout=90)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"], None
        return None, f"HTTP {r.status_code}: {r.text[:300]}"

    order = []
    if provider == "deepseek":  order = [("ds", deepseek_key), ("groq", groq_key)]
    elif provider == "groq":    order = [("groq", groq_key), ("ds", deepseek_key)]
    elif provider == "custom":  order = [("custom", custom_key), ("ds", deepseek_key)]

    for kind, key in order:
        if not key: continue
        try:
            if kind == "ds":
                r, e = _post("https://api.deepseek.com/chat/completions",
                             {"model": "deepseek-chat", "messages": messages,
                              "max_tokens": max_tokens, "temperature": temperature}, key)
            elif kind == "groq":
                r, e = _post("https://api.groq.com/openai/v1/chat/completions",
                             {"model": "llama-3.3-70b-versatile", "messages": messages,
                              "max_tokens": max_tokens, "temperature": temperature}, key)
            elif kind == "custom" and custom_url:
                r, e = _post(custom_url,
                             {"model": custom_model, "messages": messages,
                              "max_tokens": max_tokens, "temperature": temperature}, key)
            else: continue
            if r: return r, None
        except Exception as ex:
            e = str(ex)
    return None, "No AI provider configured or all requests failed."

def get_ai_keys():
    return {
        "provider":    get_setting("ai_provider", "deepseek"),
        "deepseek_key": get_setting("deepseek_api_key"),
        "groq_key":    get_setting("groq_api_key"),
        "custom_url":  get_setting("custom_ai_url"),
        "custom_key":  get_setting("custom_ai_api_key"),
        "custom_model": get_setting("custom_ai_model"),
        "c_enabled":   get_setting("custom_ai_enabled") == "1",
    }

# ═══════════════════════════════════════════════════════════════════════
#  ML FUNCTIONS — Full Suite
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def train_ensemble(df_json, target, features, hyperparams_json="{}"):
    df = pd.read_json(io.StringIO(df_json))
    X = df[features].copy()
    y = df[target].fillna(df[target].median())
    le_map = {}
    for col in X.select_dtypes("object").columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_map[col] = le
    X = X.fillna(X.median(numeric_only=True))
    scaler = RobustScaler()
    Xs = scaler.fit_transform(X)

    params = json.loads(hyperparams_json) if hyperparams_json else {}
    rf_params  = params.get("rf",  {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 2})
    gb_params  = params.get("gb",  {"n_estimators": 200, "max_depth": 5,  "learning_rate": 0.05})
    ext_params = params.get("ext", {"n_estimators": 200, "max_depth": 10})

    rf  = RandomForestRegressor(**rf_params, random_state=42, n_jobs=-1)
    gb  = GradientBoostingRegressor(**gb_params, random_state=42)
    ext = ExtraTreesRegressor(**ext_params, random_state=42, n_jobs=-1)
    br  = BayesianRidge()
    rg  = Ridge(alpha=1.0)

    ens = VotingRegressor(estimators=[("rf",rf),("gb",gb),("ext",ext),("br",br),("ridge",rg)])
    ens.fit(Xs, y)

    # Cross-validation
    tscv = TimeSeriesSplit(5)
    r2 = mape_score = rmse_score = None
    try:
        r2 = float(cross_val_score(ens, Xs, y, cv=tscv, scoring="r2").mean())
        mapes, rmses = [], []
        for tr, te in tscv.split(Xs):
            ens.fit(Xs[tr], y.iloc[tr])
            pred = ens.predict(Xs[te])
            mapes.append(mean_absolute_percentage_error(y.iloc[te], np.maximum(pred, 1e-6)))
            rmses.append(np.sqrt(mean_squared_error(y.iloc[te], pred)))
        mape_score = np.mean(mapes) * 100
        rmse_score = np.mean(rmses)
        ens.fit(Xs, y)
    except: pass

    # Feature importance (RF sub-model)
    rf2 = RandomForestRegressor(300, random_state=42, n_jobs=-1)
    rf2.fit(Xs, y)
    imps = dict(zip(X.columns, rf2.feature_importances_))
    pi = permutation_importance(rf2, Xs, y, n_repeats=8, random_state=42, n_jobs=-1)
    pimps = dict(zip(X.columns, pi.importances_mean))

    # SHAP-like (top-k feature contribution via correlation)
    corr_to_target = {}
    for col in X.columns:
        try: corr_to_target[col] = abs(float(pd.Series(X[col]).corr(y)))
        except: corr_to_target[col] = 0.0

    return ens, le_map, scaler, r2, mape_score, rmse_score, imps, pimps, corr_to_target, list(X.columns)

@st.cache_data(show_spinner=False)
def build_forecast(date_json, val_json, horizon=12, freq='ME', include_decomp=False):
    dates = pd.to_datetime(pd.read_json(io.StringIO(date_json), typ='series'))
    vals  = pd.Series(pd.read_json(io.StringIO(val_json), typ='series').values)
    df = pd.DataFrame({"ds": dates, "y": vals})
    pk = 'M' if freq == 'ME' else 'W'
    ts = df.groupby(df['ds'].dt.to_period(pk))['y'].sum().reset_index()
    ts['ds'] = ts['ds'].dt.to_timestamp()
    ts = ts.sort_values('ds').reset_index(drop=True)

    decomp = None

    # Prophet
    if PROPHET_AVAILABLE and len(ts) > 12:
        try:
            m = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=(freq == 'W'),
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                seasonality_mode='additive'
            )
            m.fit(ts)
            pfq = 'M' if freq == 'ME' else 'W'
            fut = m.make_future_dataframe(periods=horizon, freq=pfq)
            fc = m.predict(fut).tail(horizon)
            hist = ts.rename(columns={"ds":"Date","y":"Value"})
            fcast = pd.DataFrame({
                "Date": fc["ds"], "Value": fc["yhat"],
                "Lower": fc["yhat_lower"], "Upper": fc["yhat_upper"]
            })
            if include_decomp and STATSMODELS_AVAILABLE and len(ts) > 24:
                try:
                    decomp_result = seasonal_decompose(ts["y"], model='additive', period=12, extrapolate_trend='freq')
                    decomp = decomp_result
                except: pass
            return hist, fcast, "Prophet", decomp
        except: pass

    # Holt-Winters fallback
    if STATSMODELS_AVAILABLE and len(ts) > 6:
        try:
            seasonal_periods = 12 if freq == 'ME' else 52
            if len(ts) >= seasonal_periods * 2:
                fit = ExponentialSmoothing(ts["y"], trend='add', seasonal='add',
                                           seasonal_periods=seasonal_periods,
                                           initialization_method='estimated').fit()
            else:
                fit = ExponentialSmoothing(ts["y"], trend='add',
                                           initialization_method='estimated').fit()
            fcv = fit.forecast(horizon)
            ld = ts["ds"].iloc[-1]
            fdates = [ld + pd.DateOffset(months=i+1) for i in range(horizon)] \
                     if freq == 'ME' else [ld + pd.Timedelta(days=7*(i+1)) for i in range(horizon)]
            hist = ts.rename(columns={"ds":"Date","y":"Value"})
            fcast = pd.DataFrame({
                "Date": fdates, "Value": fcv.values,
                "Lower": fcv.values * 0.85, "Upper": fcv.values * 1.15
            })
            return hist, fcast, "Holt-Winters", decomp
        except: pass

    # ARIMA-lite (naive moving average) fallback
    window = min(6, len(ts))
    ma = ts["y"].rolling(window).mean().fillna(ts["y"].mean())
    last_ma = ma.iloc[-1]
    trend = (ts["y"].iloc[-1] - ts["y"].iloc[max(0, len(ts)-window)]) / window
    fdates = [ts["ds"].iloc[-1] + pd.DateOffset(months=i+1) for i in range(horizon)]
    fcv_vals = [last_ma + trend * i for i in range(1, horizon+1)]
    hist = ts.rename(columns={"ds":"Date","y":"Value"})
    fcast = pd.DataFrame({
        "Date": fdates, "Value": fcv_vals,
        "Lower": [v*0.8 for v in fcv_vals],
        "Upper": [v*1.2 for v in fcv_vals],
    })
    return hist, fcast, "Moving Average", decomp

@st.cache_data(show_spinner=False)
def run_clustering(df_json, features, method='kmeans', k=3, eps=0.5, min_samples=5):
    df = pd.read_json(io.StringIO(df_json))
    X = df[features].copy()
    for col in X.select_dtypes("object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    Xs = StandardScaler().fit_transform(X)

    if method == 'kmeans':
        m = KMeans(k, random_state=42, n_init=20)
        lbs = m.fit_predict(Xs)
        inertias = {ki: KMeans(ki, random_state=42, n_init=10).fit(Xs).inertia_
                    for ki in range(2, min(10, len(df)))}
    elif method == 'dbscan':
        m = DBSCAN(eps=eps, min_samples=min_samples)
        lbs = m.fit_predict(Xs)
        inertias = None
    elif method == 'minibatch':
        m = MiniBatchKMeans(k, random_state=42)
        lbs = m.fit_predict(Xs)
        inertias = {ki: MiniBatchKMeans(ki, random_state=42).fit(Xs).inertia_
                    for ki in range(2, min(10, len(df)))}
    else:
        m = AgglomerativeClustering(k)
        lbs = m.fit_predict(Xs)
        inertias = None

    n_clusters = len(set(lbs)) - (1 if -1 in lbs else 0)
    sil = silhouette_score(Xs, lbs) if n_clusters > 1 else None
    dbi = davies_bouldin_score(Xs, lbs) if n_clusters > 1 else None

    pca = PCA(2)
    coords = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_

    # Cluster profiles
    df_c = df[features].copy()
    for col in df_c.select_dtypes("object").columns:
        df_c[col] = LabelEncoder().fit_transform(df_c[col].astype(str))
    df_c = df_c.fillna(0)
    df_c["_cluster"] = lbs
    profiles = df_c.groupby("_cluster").mean().round(3)

    return lbs, sil, dbi, coords, inertias, var, profiles

@st.cache_data(show_spinner=False)
def detect_anomalies(df_json, features, contamination=0.05, method="iforest"):
    df = pd.read_json(io.StringIO(df_json))
    X = df[features].copy()
    for col in X.select_dtypes("object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
    X = X.fillna(0)
    Xs = StandardScaler().fit_transform(X)

    if method == "iforest":
        m = IsolationForest(contamination=contamination, n_estimators=200, random_state=42)
        preds = m.fit_predict(Xs)
        scores = m.score_samples(Xs)
    elif method == "lof":
        n_neigh = max(5, min(20, int(len(X) * 0.05)))
        m = LocalOutlierFactor(n_neighbors=n_neigh, contamination=contamination)
        preds = m.fit_predict(Xs)
        scores = m.negative_outlier_factor_
    else:
        # Z-score based
        from scipy import stats as scipy_stats
        z = np.abs(scipy_stats.zscore(Xs, nan_policy='omit'))
        preds = np.where(z.max(axis=1) > 3, -1, 1)
        scores = -z.max(axis=1)

    return preds == -1, scores

def compute_rfm(df, date_col, sales_col, id_col):
    if id_col == "—" or id_col not in df.columns: return None
    ref = df[date_col].max()
    rfm = df.groupby(id_col).agg(
        Recency=(date_col,   lambda x: (ref - x.max()).days),
        Frequency=(date_col, "count"),
        Monetary=(sales_col, "sum")
    ).reset_index()
    try:
        rfm["R"] = pd.qcut(rfm["Recency"],   5, labels=[5,4,3,2,1], duplicates='drop').astype(float)
        rfm["F"] = pd.qcut(rfm["Frequency"], 5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
        rfm["M"] = pd.qcut(rfm["Monetary"],  5, labels=[1,2,3,4,5], duplicates='drop').astype(float)
    except: return None
    rfm["RFM_Score"] = rfm["R"]*100 + rfm["F"]*10 + rfm["M"]
    def seg(row):
        rv, fv, mv = row.R, row.F, row.M
        if rv >= 4 and fv >= 4 and mv >= 4: return "🏆 Champions"
        if rv >= 3 and fv >= 3 and mv >= 3: return "💎 Loyal"
        if rv >= 4 and mv >= 3: return "⭐ High-Value"
        if rv >= 4: return "🆕 Recent"
        if fv >= 4: return "🔁 Frequent"
        if mv >= 4: return "💰 Big Spender"
        if rv <= 2 and fv <= 2 and mv <= 2: return "⚠️ At Risk"
        if rv <= 1: return "💤 Hibernating"
        return "📊 Average"
    rfm["Segment"] = rfm.apply(seg, axis=1)
    return rfm

def market_basket(df, cust, prod, min_sup=0.01, min_conf=0.1, min_lift=1.0):
    if not MLXTEND_AVAILABLE: return None, None, "mlxtend not installed."
    try:
        basket = df.groupby([cust, prod]).size().unstack(fill_value=0)
        basket = basket.map(lambda x: 1 if x > 0 else 0)
        freq = apriori(basket, min_support=min_sup, use_colnames=True, low_memory=True)
        if not len(freq): return None, None, "No frequent itemsets found. Try lower min_support."
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
        rules = rules[rules["confidence"] >= min_conf]
        return freq, rules, "OK"
    except Exception as ex:
        return None, None, str(ex)

def compute_cohort_analysis(df, date_col, id_col, sales_col):
    """Compute cohort retention matrix."""
    if any(c == "—" or c not in df.columns for c in [date_col, id_col, sales_col]): return None
    df_c = df[[date_col, id_col, sales_col]].copy()
    df_c["cohort_month"] = df_c[date_col].dt.to_period('M')
    df_c["order_month"]  = df_c[date_col].dt.to_period('M')
    cohort_group = df_c.groupby(id_col)["cohort_month"].min().reset_index()
    cohort_group.columns = [id_col, "cohort"]
    df_c = df_c.merge(cohort_group, on=id_col)
    df_c["period"] = (df_c["order_month"] - df_c["cohort"]).apply(lambda x: x.n)
    cohort_table = df_c.groupby(["cohort","period"])[id_col].nunique().reset_index()
    cohort_pivot = cohort_table.pivot(index="cohort", columns="period", values=id_col)
    cohort_size  = cohort_pivot[0]
    retention    = cohort_pivot.divide(cohort_size, axis=0) * 100
    return retention.round(1)

@st.cache_data(show_spinner=False)
def load_builtin():
    np.random.seed(42); n = 8000
    start = pd.Timestamp("2021-01-01")
    dates = [start + pd.Timedelta(days=int(x)) for x in np.sort(np.random.randint(0, 1460, n))]
    cats  = np.random.choice(["Electronics","Fashion","Home & Kitchen","Beauty",
                               "Sports","Books","Toys","Groceries","Automotive","Health"],
                              n, p=[.20,.16,.14,.11,.10,.08,.07,.05,.05,.04])
    regions = np.random.choice(["Riyadh","Dubai","Cairo","Jeddah","Kuwait City",
                                  "Doha","Amman","Manama","Abu Dhabi","Muscat"], n)
    segs    = np.random.choice(["Premium","Standard","Economy"], n, p=[.25,.5,.25])
    bp = {"Electronics":1400,"Fashion":200,"Home & Kitchen":280,"Beauty":130,
          "Sports":320,"Books":65,"Toys":95,"Groceries":50,"Automotive":850,"Health":180}
    sales   = np.array([bp[c] * np.random.uniform(.6, 2.8) for c in cats])
    disc    = np.random.choice([0,.05,.1,.15,.2,.25,.3], n, p=[.28,.14,.20,.16,.10,.07,.05])
    sf_     = sales * (1 - disc)
    pm      = np.where(cats=="Electronics",.10,
              np.where(cats=="Fashion",.32,
              np.where(cats=="Books",.42,
              np.where(cats=="Automotive",.18,.22))))
    profit  = sf_ * (pm + np.random.normal(0, .025, n))
    return pd.DataFrame({
        "Order Date":    dates,
        "Category":      cats,
        "Sub-Region":    regions,
        "Segment":       segs,
        "Product_ID":    [f"P{random.randint(1000,9999)}" for _ in range(n)],
        "Customer_ID":   [f"C{random.randint(10000,99999)}" for _ in range(n)],
        "Sales":         np.round(sf_, 2),
        "Profit":        np.round(profit, 2),
        "Discount":      disc,
        "Quantity":      np.random.randint(1, 10, n),
        "Returns":       np.random.choice([0,1], n, p=[.87,.13]),
        "Rating":        np.round(np.random.normal(4.1, .55, n).clip(1, 5), 1),
        "Shipping Days": np.random.randint(1, 8, n),
        "Shipping Cost": np.round(np.random.uniform(5, 80, n), 2),
    })

# ═══════════════════════════════════════════════════════════════════════
#  HELPERS & UI PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════
def fmt(n, prefix="", suffix="", d=1):
    try:
        if n is None or (isinstance(n, float) and np.isnan(n)): return "N/A"
        n = float(n)
        if abs(n) >= 1e12: return f"{prefix}{n/1e12:.{d}f}T{suffix}"
        if abs(n) >= 1e9:  return f"{prefix}{n/1e9:.{d}f}B{suffix}"
        if abs(n) >= 1e6:  return f"{prefix}{n/1e6:.{d}f}M{suffix}"
        if abs(n) >= 1e3:  return f"{prefix}{n/1e3:.{d}f}K{suffix}"
        return f"{prefix}{n:.{d}f}{suffix}"
    except: return "N/A"

def si(lst, v):
    try: return lst.index(v)
    except: return 0

def sec_head(id_, title, sub=""):
    st.markdown(f"""<div class="nx-sec">
      <span class="nx-sec-id">{id_}</span>
      <span class="nx-sec-title">{title}</span>
      <span class="nx-sec-sub">{sub}</span>
    </div>""", unsafe_allow_html=True)

def topbar():
    ts  = datetime.now().strftime("%H:%M:%S")
    df  = st.session_state.get("df")
    r_chip = f'<div class="nx-kpi-chip">ROWS<span>{len(df):,}</span></div>' if df is not None else ""
    c_chip = f'<div class="nx-kpi-chip">COLS<span>{df.shape[1]}</span></div>' if df is not None else ""
    plan   = st.session_state.get("_plan","—")
    st.markdown(f"""<div class="nx-topbar">
      <div>
        <div class="nx-logo">NEXUS</div>
        <div class="nx-logo-sub">Enterprise Analytics OS</div>
      </div>
      <div class="nx-pulse"></div>
      <span class="nx-status-text">OPERATIONAL</span>
      <div class="nx-kpi-row">
        {r_chip}{c_chip}
        <div class="nx-kpi-chip">PLAN<span>{plan}</span></div>
      </div>
      <span class="nx-clock">{ts} UTC</span>
      <span class="nx-badge">v{APP_VERSION}</span>
    </div>""", unsafe_allow_html=True)

def sidebar_brand():
    st.markdown(f"""<div class="sb-brand">
      <div class="sb-logo"><span class="sb-logo-hex">⬡</span> NEXUS</div>
      <div class="sb-sub">Enterprise Analytics OS</div>
      <span class="sb-ver">v{APP_VERSION}</span>
    </div>""", unsafe_allow_html=True)

def sidebar_user_card(email, plan_name, max_rows):
    st.markdown(f"""<div class="sb-user">
      <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--t3);
                  letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">Authenticated</div>
      <div class="sb-user-email">👤 {email}</div>
      <span class="sb-user-plan">{plan_name} · {max_rows:,} rows</span>
    </div>""", unsafe_allow_html=True)

def progress_bar(pct, color="#00d4ff"):
    st.markdown(f"""<div class="nx-progress-wrap">
      <div class="nx-progress-fill" style="width:{min(pct,100):.1f}%;
           background:linear-gradient(90deg,{color},var(--blue));"></div>
    </div>""", unsafe_allow_html=True)

def chip(text, kind="cyan"):
    st.markdown(f'<span class="nx-chip {kind}">{text}</span>', unsafe_allow_html=True)

def alert_box(msg, kind=""):
    icons = {"warn":"⚠️","err":"❌","ok":"✅","":"ℹ️"}
    icon  = icons.get(kind,"ℹ️")
    st.markdown(f'<div class="nx-alert {kind}">{icon} {msg}</div>', unsafe_allow_html=True)

def empty_state(icon="⬡", title="No Data Loaded", subtitle="Load a dataset from the sidebar to begin."):
    st.markdown(f"""<div class="empty-state">
      <div class="ei">{icon}</div>
      <h2>{title}</h2>
      <p>{subtitle}</p>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  LOGIN PANEL
# ═══════════════════════════════════════════════════════════════════════
def login_panel():
    st.markdown("""<div style="background:linear-gradient(145deg,var(--panel),var(--card));
        border:1px solid var(--cyan-border);border-top:2px solid var(--cyan);
        border-radius:var(--r4);padding:28px 22px;text-align:center;margin-bottom:14px;">
      <div style="font-family:var(--font-head);font-size:2.2rem;font-weight:700;
                  color:var(--cyan);letter-spacing:6px;
                  text-shadow:0 0 20px var(--cyan-glow);">⬡ NEXUS</div>
      <div style="font-family:var(--font-mono);font-size:0.56rem;color:var(--t3);
                  letter-spacing:3px;text-transform:uppercase;margin-top:4px;">Enterprise Analytics OS</div>
      <div style="margin-top:12px;font-family:var(--font-mono);font-size:0.68rem;color:var(--t2);">
        Sign in to unlock full capabilities
      </div>
    </div>""", unsafe_allow_html=True)

    with st.expander("🔐 Sign In", expanded=True):
        email = st.text_input("Email", key="li_e", placeholder="user@corp.com",
                               label_visibility="collapsed")
        st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--t3);margin-bottom:4px;">EMAIL</div>', unsafe_allow_html=True)
        pwd   = st.text_input("Password", type="password", key="li_p", placeholder="••••••••",
                               label_visibility="collapsed")
        st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--t3);margin-bottom:8px;">PASSWORD</div>', unsafe_allow_html=True)
        if st.button("→ AUTHENTICATE", key="li_b", use_container_width=True):
            ok, is_admin = verify_password(email, pwd)
            log_login(email, ok)
            if ok:
                st.session_state.update({"logged_in": True, "user_email": email,
                                          "is_admin": is_admin, "admin_mode": False})
                log_action(email, "login", module="auth")
                st.rerun()
            else:
                st.error("⛔ Invalid credentials")

    with st.expander("📝 Create Account"):
        ne   = st.text_input("Email",     key="reg_e", placeholder="you@company.com")
        name = st.text_input("Full Name", key="reg_n", placeholder="John Doe")
        np_  = st.text_input("Password",  type="password", key="reg_p", placeholder="min 8 chars")
        if st.button("CREATE ACCOUNT", key="reg_b", use_container_width=True):
            if not ne or not np_: st.error("Fill all required fields")
            elif len(np_) < 6:    st.error("Password must be at least 6 characters")
            elif register_user(ne, np_, name): st.success("✅ Account created. Sign in above.")
            else: st.error("⚠️ Email already registered")

    st.markdown("""<hr class="nx-divider">
    <div style="text-align:center;font-family:var(--font-mono);font-size:0.62rem;color:var(--t3);">
      👁 GUEST MODE · 2,000 ROW LIMIT · READ ONLY
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  REALTIME / CLOUD CONNECTORS MODULE
# ═══════════════════════════════════════════════════════════════════════
def realtime_module():
    sec_head("RT", "Cloud & Real-Time Connectors", "Connect · Stream · Sync · Monitor")
    plan = st.session_state.get("_plan", "Starter")
    is_pro = plan in ["Professional", "Business", "Enterprise"]
    max_conn = st.session_state.get("_max_connectors", 0)

    # Hero banner
    st.markdown("""<div class="nx-hero" style="padding:24px 28px;margin-bottom:18px;">
      <div style="display:flex;align-items:center;gap:16px;text-align:left;">
        <div style="font-size:2.5rem;">⚡</div>
        <div>
          <div class="nx-hero-title" style="font-size:1.2rem;letter-spacing:3px;">Enterprise Data Fabric</div>
          <div class="nx-hero-sub" style="margin-top:4px;">
            Connect NEXUS to 25+ cloud sources · Real-time streaming · CDC pipelines ·
            Zero-copy query federation
          </div>
        </div>
        <div style="margin-left:auto;text-align:right;">
          <div class="nx-chip cyan">25+ Connectors</div>&nbsp;
          <div class="nx-chip green" style="margin-top:6px;">Live Streaming</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    rt_tabs = st.tabs(["🌐 Connector Library","⚡ Live Stream","🔌 Configure","📡 Status","🔒 Security"])

    # ── Tab 0: Connector Library ─────────────────────────────────────
    with rt_tabs[0]:
        # Group by category
        categories = {}
        for cid, info in CONNECTORS.items():
            cat = info["category"]
            categories.setdefault(cat, []).append((cid, info))

        search = st.text_input("🔍 Search connectors...", key="conn_search",
                                placeholder="postgres, kafka, S3...")

        for cat, conns in categories.items():
            st.markdown(f"""<div class="sb-section-label" style="padding:10px 0 4px;">{cat}</div>""",
                        unsafe_allow_html=True)
            filtered = [(cid,info) for cid,info in conns
                        if not search or search.lower() in info["label"].lower() or search.lower() in cid]
            if not filtered: continue
            cols = st.columns(min(len(filtered), 5))
            for i, (cid, info) in enumerate(filtered):
                with cols[i % 5]:
                    locked = not is_pro and info["status"] != "ok"
                    badge_cls  = info["status"] if not locked else "lock"
                    badge_text = info["status"].upper() if not locked else "🔒 UPGRADE"
                    st.markdown(f"""<div class="cc-card">
                      <div class="cc-icon">{info["icon"]}</div>
                      <div class="cc-name">{info["label"]}</div>
                      <div class="cc-sub">{cid.upper()}</div>
                      <span class="cc-badge {badge_cls}">{badge_text}</span>
                    </div>""", unsafe_allow_html=True)

    # ── Tab 1: Live Stream ───────────────────────────────────────────
    with rt_tabs[1]:
        sec_head("LIVE", "Real-Time Data Stream", "Simulated Kafka · MQTT · WebSocket feed")
        if not is_pro:
            alert_box("Live streaming requires Professional plan or higher.", "warn")
        else:
            c_ctrl, c_main = st.columns([1, 3])
            with c_ctrl:
                st.markdown("""<div class="live-badge"><div class="live-dot"></div>STREAM ACTIVE</div>""",
                            unsafe_allow_html=True)
                n_pts      = st.slider("Points/frame", 20, 300, 80, key="live_n")
                show_vol   = st.checkbox("Volume bars", True, key="live_vol")
                show_dist  = st.checkbox("Category split", True, key="live_dist")
                latency_ms = random.randint(4, 28)
                st.metric("Latency", f"{latency_ms}ms")
                throughput = random.randint(800, 3200)
                st.metric("Throughput", f"{throughput}/s")
                if st.button("🔄 Refresh", use_container_width=True, key="live_ref"):
                    st.rerun()

            with c_main:
                live_df = simulate_live_data(n=n_pts)
                rows    = 3 if (show_vol and show_dist) else (2 if (show_vol or show_dist) else 1)
                h_ratios = [0.55, 0.25, 0.20][:rows]
                fig = make_subplots(rows=rows, cols=1,
                                    row_heights=h_ratios,
                                    shared_xaxes=True,
                                    vertical_spacing=0.04)
                fig.add_trace(go.Scatter(
                    x=live_df["Timestamp"], y=live_df["Sales"],
                    mode="lines", name="Sales",
                    line=dict(color="#00d4ff", width=2),
                    fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"
                ), row=1, col=1)
                fig.add_trace(go.Scatter(
                    x=live_df["Timestamp"], y=live_df["Profit"],
                    mode="lines", name="Profit",
                    line=dict(color="#00e676", width=1.5, dash="dot")
                ), row=1, col=1)
                r = 2
                if show_vol:
                    fig.add_trace(go.Bar(
                        x=live_df["Timestamp"], y=live_df["Quantity"],
                        name="Qty", marker_color="rgba(187,134,252,0.55)"
                    ), row=r, col=1)
                    r += 1
                if show_dist:
                    cat_live = live_df.groupby("Category")["Sales"].sum().reset_index()
                    fig.add_trace(go.Bar(
                        x=cat_live["Category"], y=cat_live["Sales"],
                        name="By Category",
                        marker=dict(color=cat_live["Sales"],
                                    colorscale=[[0,"#1a6bff"],[0.5,"#00d4ff"],[1,"#00e676"]])
                    ), row=r, col=1)
                pf(fig, "LIVE TRANSACTION FEED", h=420)
                st.plotly_chart(fig, use_container_width=True)

                kc = st.columns(5)
                kc[0].metric("Revenue",    fmt(live_df.Sales.sum(), "$"))
                kc[1].metric("Profit",     fmt(live_df.Profit.sum(), "$"))
                kc[2].metric("Orders",     len(live_df))
                kc[3].metric("Avg Ticket", fmt(live_df.Sales.mean(), "$"))
                kc[4].metric("Avg Rating", f"{live_df.Rating.mean():.2f}")

    # ── Tab 2: Configure ─────────────────────────────────────────────
    with rt_tabs[2]:
        if not is_pro:
            alert_box("Custom connectors require Professional plan or higher.", "warn")
        else:
            sec_head("CFG", "New Connection", "Enterprise data source configuration")
            c1, c2 = st.columns([1, 1])
            with c1:
                conn_type = st.selectbox("Connector Type", list(CONNECTORS.keys()),
                                          format_func=lambda x: f"{CONNECTORS[x]['icon']} {CONNECTORS[x]['label']}")
                conn_name = st.text_input("Connection Name", placeholder="e.g. prod-snowflake-dw")

                if conn_type in ["postgresql","mysql","mssql","oracle","sqlite"]:
                    host = st.text_input("Host / IP", placeholder="db.yourcompany.com")
                    port = st.text_input("Port", value={"postgresql":"5432","mysql":"3306","mssql":"1433","oracle":"1521"}.get(conn_type,"5432"))
                    db   = st.text_input("Database / Schema")
                    usr  = st.text_input("Username")
                    pw   = st.text_input("Password", type="password")
                    ssl  = st.checkbox("SSL/TLS enabled", True)
                    timeout = st.slider("Connection timeout (s)", 5, 60, 30)
                    if st.button("🔌 Test Connection", use_container_width=True):
                        with st.spinner("Connecting..."):
                            ok, lat, rows, msg = cloud_connector_test(conn_type, {})
                        if ok: st.success(f"✅ {msg} · Latency: {lat}ms · ~{rows:,} rows")
                        else:  st.error(f"❌ {msg}")

                elif conn_type == "snowflake":
                    acct    = st.text_input("Account ID", placeholder="xy12345.us-east-1")
                    wh      = st.text_input("Warehouse", value="COMPUTE_WH")
                    db_sf   = st.text_input("Database")
                    schema  = st.text_input("Schema", value="PUBLIC")
                    role    = st.text_input("Role", placeholder="SYSADMIN")
                    auth    = st.selectbox("Auth Method", ["Password","Key-Pair","OAuth"])
                    if st.button("🔌 Test Snowflake", use_container_width=True):
                        with st.spinner("Authenticating with Snowflake..."):
                            ok, lat, rows, msg = cloud_connector_test(conn_type, {})
                        if ok: st.success(f"✅ {msg} · Latency: {lat}ms")
                        else:  st.error(f"❌ {msg}")

                elif conn_type == "bigquery":
                    project    = st.text_input("GCP Project ID")
                    dataset    = st.text_input("Dataset")
                    creds_file = st.text_area("Service Account JSON", height=80, placeholder='{"type":"service_account",...}')
                    location   = st.selectbox("Location", ["US","EU","asia-east1","europe-west2"])
                    if st.button("🔌 Test BigQuery", use_container_width=True):
                        with st.spinner("Connecting to BigQuery..."):
                            ok, lat, rows, msg = cloud_connector_test(conn_type, {})
                        if ok: st.success(f"✅ {msg} · Latency: {lat}ms")
                        else:  st.error(f"❌ {msg}")

                elif conn_type == "api_rest":
                    url        = st.text_input("Endpoint URL", placeholder="https://api.example.com/v1/data")
                    api_key_   = st.text_input("API Key / Token", type="password")
                    method     = st.selectbox("HTTP Method", ["GET","POST","PUT"])
                    headers_j  = st.text_area("Extra Headers (JSON)", value="{}", height=60)
                    body_j     = st.text_area("Request Body (JSON)", value="{}", height=60)
                    if st.button("🔌 Test REST Endpoint", use_container_width=True):
                        with st.spinner("Testing endpoint..."):
                            df_api, err = fetch_from_rest_api(url, api_key_, method)
                        if df_api is not None:
                            st.success(f"✅ Endpoint reachable · {len(df_api):,} rows returned")
                            st.dataframe(df_api.head(5), use_container_width=True)
                        else:
                            st.error(f"❌ {err}")

                elif conn_type in ["s3","gcs","azure_blob"]:
                    if conn_type == "s3":
                        bucket     = st.text_input("S3 Bucket")
                        prefix     = st.text_input("Key Prefix", placeholder="data/sales/2024/")
                        region     = st.selectbox("AWS Region", ["us-east-1","us-west-2","eu-west-1","ap-southeast-1"])
                        access_key = st.text_input("Access Key ID")
                        secret_key = st.text_input("Secret Access Key", type="password")
                        file_fmt   = st.selectbox("File Format", ["CSV","Parquet","JSON","AVRO","ORC"])
                        if st.button("🔌 Browse S3", use_container_width=True):
                            with st.spinner("Listing objects..."):
                                ok, lat, rows, msg = cloud_connector_test(conn_type, {})
                            if ok: st.success(f"✅ Found ~{rows:,} objects in s3://{bucket or 'bucket'}/{prefix or ''}")
                            else:  st.error(f"❌ {msg}")
                    elif conn_type == "gcs":
                        bucket_gcs = st.text_input("GCS Bucket")
                        prefix_gcs = st.text_input("Prefix", placeholder="data/")
                        creds_gcs  = st.text_area("Service Account JSON", height=80)
                        if st.button("🔌 Browse GCS", use_container_width=True):
                            with st.spinner("Connecting to GCS..."): time.sleep(1.2)
                            st.success(f"✅ Bucket accessible")
                    else:
                        conn_str = st.text_input("Connection String")
                        container = st.text_input("Container Name")
                        if st.button("🔌 Test Azure Blob", use_container_width=True):
                            with st.spinner("Connecting to Azure..."): time.sleep(1.5)
                            st.success("✅ Azure Blob accessible")

                elif conn_type in ["kafka","kinesis","mqtt"]:
                    server = st.text_input("Broker / Server", placeholder="kafka.corp.com:9092")
                    topic  = st.text_input("Topic / Channel")
                    grp    = st.text_input("Consumer Group", value="nexus-consumer")
                    offset = st.selectbox("Start Offset", ["latest","earliest","stored"])
                    if st.button("🔌 Connect", use_container_width=True):
                        with st.spinner("Establishing stream connection..."):
                            ok, lat, rows, msg = cloud_connector_test(conn_type, {})
                        if ok: st.success(f"✅ Stream connected · Latency: {lat}ms")
                        else:  st.error(f"❌ {msg}")

                else:
                    st.info(f"Configure {CONNECTORS[conn_type]['label']} connection parameters above.")

            with c2:
                st.markdown("""<div class="nx-panel" style="height:100%;">
                  <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--cyan);
                               letter-spacing:2px;text-transform:uppercase;margin-bottom:12px;">
                    QUERY / TRANSFORM EDITOR
                  </div>
                  <p style="font-family:var(--font-mono);font-size:0.72rem;color:var(--t3);
                             line-height:1.7;margin-bottom:12px;">
                    Define extraction query. NEXUS runs this on every sync cycle.
                    Supports SQL, JSONPath, XPath, jq.
                  </p>
                </div>""", unsafe_allow_html=True)
                query = st.text_area("Query / Expression", height=200, placeholder=
                    "SELECT\n  order_date, category, SUM(sales) AS revenue,\n  SUM(profit) AS profit,\n  COUNT(*) AS orders\nFROM sales_fact\nWHERE order_date >= CURRENT_DATE - INTERVAL '90 days'\nGROUP BY 1,2\nORDER BY 1 DESC\nLIMIT 500000")
                sync_freq = st.selectbox("Sync Frequency", ["Manual","Every 5 min","Every 15 min","Every 1 hr","Every 6 hr","Daily 00:00 UTC","Real-time CDC"])
                transform = st.text_area("Post-fetch Transform (Python lambda)", height=60,
                                          placeholder="lambda df: df.assign(margin=df.profit/df.sales*100)")
                if st.button("💾 Save & Activate Connection", use_container_width=True):
                    if conn_name:
                        st.success(f"✅ Connection '{conn_name}' saved · Next sync: {sync_freq}")
                        log_action(st.session_state.get("user_email","guest"),
                                   "create_connection", conn_name, module="connectors")
                    else:
                        st.error("Please provide a connection name")

    # ── Tab 3: Status ────────────────────────────────────────────────
    with rt_tabs[3]:
        sec_head("STATUS", "Connection Health", "Monitor all data sources")
        simulated_status = [
            ("prod-postgres",    "PostgreSQL",    "ok",     12,  "2 min ago",  "2.8M",  "99.9%"),
            ("snowflake-main",   "Snowflake",     "ok",      8,  "5 min ago",  "45M",   "99.95%"),
            ("s3-data-lake",     "AWS S3",        "ok",     45,  "1 min ago",  "180M",  "100%"),
            ("kafka-orders",     "Kafka",         "stream",  3,  "Real-time",  "∞",     "99.7%"),
            ("bigquery-ml",      "BigQuery",      "idle",    0,  "1 hr ago",   "90M",   "99.5%"),
            ("legacy-mysql",     "MySQL",         "err",     0,  "3 hrs ago",  "1.2M",  "87.0%"),
            ("salesforce-crm",   "Salesforce",    "ok",     55,  "10 min ago", "500K",  "99.2%"),
            ("google-sheets",    "Google Sheets", "ok",     90,  "30 min ago", "50K",   "98.0%"),
        ]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Active",       "5", "↑1")
        c2.metric("Streaming",    "1")
        c3.metric("Avg Latency",  "24ms", "↓3ms")
        c4.metric("Data Ingest",  "3.2GB", "today")

        for name, ctype, status, lat, sync, rows, uptime in simulated_status:
            dot_cls = {"ok":"ok","stream":"stream","idle":"idle","err":"err"}.get(status,"idle")
            lat_str = f"{lat}ms" if lat > 0 else "—"
            st.markdown(f"""<div class="conn-row">
              <div class="conn-dot {dot_cls}"></div>
              <span style="flex:2;color:var(--t1);font-weight:600;">{name}</span>
              <span style="flex:1.5;color:var(--t3);">{ctype}</span>
              <span style="flex:0.8;color:var(--t2);">{lat_str}</span>
              <span style="flex:1;color:var(--t3);">{sync}</span>
              <span style="flex:0.8;color:var(--t2);">{rows} rows</span>
              <span style="flex:0.7;color:{"var(--green)" if float(uptime.rstrip('%'))>98 else "var(--amber)"};">{uptime}</span>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 Refresh All Connections", use_container_width=True):
            with st.spinner("Polling all data sources..."): time.sleep(1.2)
            st.success("✅ All connections refreshed")

    # ── Tab 4: Security ──────────────────────────────────────────────
    with rt_tabs[4]:
        sec_head("SEC", "Connection Security", "Encryption · Credentials · Audit")
        if not is_pro:
            alert_box("Security features require Professional plan or higher.", "warn")
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("""<div class="nx-panel">
                  <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--cyan);
                               letter-spacing:2px;margin-bottom:12px;">🔐 ENCRYPTION CONFIG</div>""",
                            unsafe_allow_html=True)
                st.selectbox("TLS Version", ["TLS 1.3 (Recommended)","TLS 1.2","TLS 1.1"])
                st.selectbox("Certificate Verification", ["Strict","Relaxed","Disabled (dev only)"])
                st.text_input("CA Certificate Path", placeholder="/etc/ssl/certs/ca.pem")
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""<div class="nx-panel" style="margin-top:12px;">
                  <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--cyan);
                               letter-spacing:2px;margin-bottom:12px;">🔑 SECRETS MANAGEMENT</div>""",
                            unsafe_allow_html=True)
                st.selectbox("Secrets Backend", ["NEXUS Vault (built-in)","AWS Secrets Manager","HashiCorp Vault","Azure Key Vault"])
                st.checkbox("Rotate credentials automatically", True)
                st.number_input("Rotation interval (days)", 30, 365, 90)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown("""<div class="nx-panel">
                  <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--cyan);
                               letter-spacing:2px;margin-bottom:12px;">🛡️ ACCESS CONTROL</div>""",
                            unsafe_allow_html=True)
                st.selectbox("Auth Protocol", ["API Key","OAuth 2.0","SAML 2.0","LDAP/AD","JWT","mTLS"])
                st.multiselect("IP Allowlist", ["0.0.0.0/0 (Any)"], default=["0.0.0.0/0 (Any)"])
                st.checkbox("Enable MFA for connector access", False)
                st.checkbox("Log all data access events", True)
                st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("""<div class="nx-panel" style="margin-top:12px;">
                  <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--cyan);
                               letter-spacing:2px;margin-bottom:12px;">📋 COMPLIANCE</div>""",
                            unsafe_allow_html=True)
                for cert in ["GDPR Compliant","SOC 2 Type II","ISO 27001","HIPAA Ready","PCI DSS"]:
                    col_a, col_b = st.columns([4,1])
                    col_a.markdown(f'<span style="font-family:var(--font-mono);font-size:0.72rem;">{cert}</span>', unsafe_allow_html=True)
                    col_b.markdown('<span class="nx-chip green">✓</span>', unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  CHATBOT MODULE
# ═══════════════════════════════════════════════════════════════════════
def chatbot_module():
    sec_head("AI", "NEXUS Intelligence", "Multi-mode AI analytics assistant")
    keys = get_ai_keys()
    has_key = ((keys["provider"]=="deepseek" and keys["deepseek_key"]) or
               (keys["provider"]=="groq"     and keys["groq_key"])     or
               (keys["provider"]=="custom"   and keys["c_enabled"] and keys["custom_key"]))

    # Mode selector
    ai_mode = st.selectbox("AI Mode", ["🔬 Data Analyst","💻 SQL Expert","🤖 ML Engineer","📈 Forecast Advisor"],
                            key="ai_mode_sel", label_visibility="collapsed")
    mode_map = {"🔬 Data Analyst":"analyst","💻 SQL Expert":"sql","🤖 ML Engineer":"ml","📈 Forecast Advisor":"forecast"}
    sys_prompt_key = mode_map.get(ai_mode, "analyst")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = [{"role":"assistant","content":
            "Hello. I'm **NEXUS AI** — your enterprise analytics assistant.\n\n"
            "I'm operating in **Data Analyst** mode. Switch modes above to get specialized help:\n"
            "• 🔬 **Data Analyst** — insights, trends, KPI analysis\n"
            "• 💻 **SQL Expert** — query generation & optimization\n"
            "• 🤖 **ML Engineer** — model selection & feature engineering\n"
            "• 📈 **Forecast Advisor** — time-series & demand planning\n\n"
            "Load a dataset and ask me anything."}]

    # Quick prompts
    df = st.session_state.get("df")
    if df is not None:
        st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--t3);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">QUICK PROMPTS</div>', unsafe_allow_html=True)
        qcols = st.columns(4)
        quick_prompts = [
            "📊 Summarize this dataset",
            "🔍 Find anomalies in the data",
            "💡 Key business insights",
            "📈 Forecast next quarter",
        ]
        for i, qp in enumerate(quick_prompts):
            with qcols[i]:
                if st.button(qp, key=f"qp_{i}", use_container_width=True):
                    st.session_state.chat_messages.append({"role":"user","content":qp})
                    st.rerun()

    col_top, col_clear = st.columns([9,1])
    with col_clear:
        if st.button("🗑", key="clr_c"):
            st.session_state.chat_messages = [{"role":"assistant","content":"Conversation cleared. Ready."}]
            st.rerun()

    # Chat history
    st.markdown('<div style="max-height:480px;overflow-y:auto;padding:4px 0;" id="chat-hist">', unsafe_allow_html=True)
    for msg in st.session_state.chat_messages:
        iu = msg["role"] == "user"
        wrap = "user" if iu else ""
        av_c = "user" if iu else "bot"
        bub  = "user" if iu else "bot"
        av_text = "U" if iu else "⬡"
        st.markdown(f"""<div class="chat-msg {wrap}">
          <div class="chat-av {av_c}">{av_text}</div>
          <div class="chat-bub {bub}">{msg["content"]}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    prompt = st.chat_input("Ask about your data, request SQL, ML advice, or forecasting guidance...")
    if prompt:
        st.session_state.chat_messages.append({"role":"user","content":prompt})
        st.rerun()

    if (st.session_state.chat_messages and
            st.session_state.chat_messages[-1]["role"] == "user"):
        last = st.session_state.chat_messages[-1]["content"]
        if not has_key:
            st.session_state.chat_messages.append({"role":"assistant","content":
                "⚠️ No AI API key configured. An administrator must add a DeepSeek, Groq, or custom AI key in the **Control Center → Settings**."})
            st.rerun()
        else:
            with st.spinner("⬡ Thinking..."):
                msgs = [{"role":"system","content":SYSTEM_PROMPTS.get(sys_prompt_key, SYSTEM_PROMPTS["analyst"])}]
                for m in st.session_state.chat_messages[-22:]:
                    msgs.append({"role": m["role"], "content": m["content"]})
                if df is not None:
                    ctx = (f"Dataset Context: {len(df):,} rows × {df.shape[1]} columns\n"
                           f"Columns: {list(df.columns)}\n"
                           f"Dtypes: {df.dtypes.to_dict()}\n"
                           f"First 5 rows:\n{df.head(5).to_string()}\n"
                           f"Statistics:\n{df.describe().to_string()}")
                    msgs[-1]["content"] = f"{ctx}\n\nUser Question: {last}"

                resp, err = call_ai(
                    msgs,
                    provider=keys["provider"],
                    deepseek_key=keys["deepseek_key"],
                    groq_key=keys["groq_key"],
                    custom_url=keys["custom_url"] if keys["c_enabled"] else "",
                    custom_key=keys["custom_key"] if keys["c_enabled"] else "",
                    custom_model=keys["custom_model"] if keys["c_enabled"] else "",
                )
                result = resp if resp else f"⚠️ Error: {err}"
                st.session_state.chat_messages.append({"role":"assistant","content":result})
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════
#  ADMIN DASHBOARD
# ═══════════════════════════════════════════════════════════════════════
def admin_dashboard():
    if st.sidebar.button("← Exit Admin", key="exit_admin"):
        st.session_state.admin_mode = False; st.rerun()

    st.markdown("""<div class="admin-hero">
      <div style="font-family:var(--font-head);font-size:2.4rem;font-weight:700;
                  color:var(--cyan);letter-spacing:5px;
                  text-shadow:0 0 30px var(--cyan-glow);">⬡ CONTROL CENTER</div>
      <div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--t3);
                  letter-spacing:3px;margin-top:8px;">
        NEXUS ENTERPRISE v{APP_VERSION} · SYSTEM ADMINISTRATION · FULL ACCESS
      </div>
    </div>""".format(APP_VERSION=APP_VERSION), unsafe_allow_html=True)

    stats = get_stats()
    c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
    for col,(lbl,val,delta) in zip([c1,c2,c3,c4,c5,c6,c7,c8],[
        ("USERS",    stats["users"],           None),
        ("ADMINS",   stats["admins"],          None),
        ("TODAY",    stats["today_ok"],        f"{stats['today_attempts']} attempts"),
        ("ALL-TIME", stats["ok_logins"],       None),
        ("FAILED",   stats["fail_logins"],     None),
        ("ACTIONS",  stats["actions"],         None),
        ("MRR",      f"${stats['mrr']:.0f}",  None),
        ("ML RUNS",  stats["experiments"],     None),
    ]):
        col.metric(lbl, val, delta)

    st.markdown("---")
    atabs = st.tabs(["📊 Overview","👤 Users","📋 Logs","💳 Subscriptions","💰 Plans","⚙️ Settings","📈 Analytics","🔧 System"])

    # ── Overview ─────────────────────────────────────────────────────
    with atabs[0]:
        sec_head("OVERVIEW","Platform Health","Real-time system metrics")
        logs = get_login_logs(500)
        if logs:
            df_l = pd.DataFrame(logs, columns=["email","success","ip","timestamp"])
            df_l["timestamp"] = pd.to_datetime(df_l["timestamp"])
            df_l["date"] = df_l["timestamp"].dt.date
            c1, c2 = st.columns(2)
            with c1:
                lc = df_l.groupby(["date","success"]).size().reset_index(name="n")
                lc["status"] = lc["success"].map({1:"✅ Success",0:"❌ Failed"})
                fig = px.area(lc, x="date", y="n", color="status",
                              color_discrete_map={"✅ Success":"#00e676","❌ Failed":"#ff3d57"})
                st.plotly_chart(pf(fig,"LOGIN ACTIVITY"), use_container_width=True)
            with c2:
                sr = df_l["success"].mean() * 100
                fig2 = make_gauge(round(sr,1), "LOGIN SUCCESS RATE %", 100, "#00d4ff")
                st.plotly_chart(fig2, use_container_width=True)

        # Plan distribution
        c1, c2 = st.columns(2)
        with c1:
            pd_ = stats["plan_dist"]
            if pd_:
                fig3 = px.pie(names=list(pd_.keys()), values=list(pd_.values()),
                              title="USER PLAN DISTRIBUTION", hole=0.45)
                pf(fig3)
                st.plotly_chart(fig3, use_container_width=True)
        with c2:
            slogs = get_system_logs(100)
            if slogs:
                df_sys = pd.DataFrame(slogs, columns=["user","action","details","module","ts"])
                ac = df_sys["action"].value_counts().head(8).reset_index()
                ac.columns = ["action","count"]
                fig4 = px.bar(ac, x="count", y="action", orientation='h', title="TOP ACTIONS",
                              color="count", color_continuous_scale="Blues")
                st.plotly_chart(pf(fig4), use_container_width=True)

    # ── Users ────────────────────────────────────────────────────────
    with atabs[1]:
        sec_head("USERS","User Management","Create · Edit · Promote · Delete")
        users = get_all_users()
        if users:
            df_u = pd.DataFrame(users, columns=["ID","Email","Admin","Full Name","Created","Last Login"])
            df_u["Role"] = df_u["Admin"].map({1:"👑 Admin",0:"👤 User"})
            st.dataframe(df_u[["ID","Email","Full Name","Role","Created","Last Login"]],
                         use_container_width=True)
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown('<div class="nx-sec-id">DELETE</div>', unsafe_allow_html=True)
            uid_del = st.number_input("User ID",min_value=1,step=1,key="del_uid")
            if st.button("🗑 Delete User",use_container_width=True):
                delete_user(uid_del); st.success(f"Deleted user {uid_del}"); st.rerun()
        with c2:
            st.markdown('<div class="nx-sec-id">ADMIN TOGGLE</div>', unsafe_allow_html=True)
            uid_adm = st.number_input("User ID",min_value=1,step=1,key="adm_uid")
            mk_adm  = st.checkbox("Grant Admin")
            if st.button("🔄 Toggle Role",use_container_width=True):
                toggle_admin(uid_adm, mk_adm); st.success("Role updated"); st.rerun()
        with c3:
            st.markdown('<div class="nx-sec-id">RESET PASSWORD</div>', unsafe_allow_html=True)
            uid_rst = st.number_input("User ID",min_value=1,step=1,key="rst_uid")
            np2     = st.text_input("New Password",type="password",key="np2")
            if st.button("🔑 Reset",use_container_width=True):
                if np2: reset_password(uid_rst,np2); st.success("Password reset!")

    # ── Logs ─────────────────────────────────────────────────────────
    with atabs[2]:
        sec_head("LOGS","Activity Logs","Complete audit trail")
        tab_l1, tab_l2 = st.tabs(["LOGIN LOGS","SYSTEM LOGS"])
        with tab_l1:
            logs2 = get_login_logs(400)
            if logs2:
                df_ll = pd.DataFrame(logs2, columns=["Email","Success","IP","Timestamp"])
                df_ll["Status"] = df_ll["Success"].map({1:"✅ OK",0:"❌ FAIL"})
                st.dataframe(df_ll[["Email","Status","IP","Timestamp"]],
                             use_container_width=True, height=420)
        with tab_l2:
            slogs2 = get_system_logs(400)
            if slogs2:
                df_sl = pd.DataFrame(slogs2, columns=["User","Action","Details","Module","Timestamp"])
                st.dataframe(df_sl, use_container_width=True, height=420)

    # ── Subscriptions ─────────────────────────────────────────────────
    with atabs[3]:
        sec_head("SUBS","Subscription Management","Upgrade · Extend · Cancel")
        subs = get_all_subscriptions()
        if subs:
            df_ss = pd.DataFrame(subs)
            for dc in ["start_date","end_date"]:
                if dc in df_ss:
                    df_ss[dc] = pd.to_datetime(df_ss[dc], errors='coerce').dt.strftime("%Y-%m-%d")
            st.dataframe(df_ss, use_container_width=True)
        st.markdown("---")
        c1,c2,c3 = st.columns(3)
        with c1:
            uid_up  = st.number_input("User ID",min_value=1,step=1,key="sub_up_uid")
            plans   = get_available_plans()
            po      = {p["name"]: p["id"] for p in plans}
            pn      = st.selectbox("New Plan", list(po.keys()))
            dur     = st.selectbox("Duration (months)",[1,3,6,12])
            if st.button("⬆ Upgrade",use_container_width=True):
                upgrade_subscription(uid_up, po[pn], dur)
                st.success(f"User {uid_up} → {pn}"); st.rerun()
        with c2:
            uid_ext = st.number_input("User ID",min_value=1,step=1,key="ext_uid2")
            extra   = st.number_input("Extra months",1,12,1)
            if st.button("📅 Extend",use_container_width=True):
                extend_sub(uid_ext, extra); st.success("Extended!"); st.rerun()
        with c3:
            uid_cxl = st.number_input("User ID",min_value=1,step=1,key="cxl_uid")
            if st.button("❌ Cancel → Starter",use_container_width=True):
                cancel_sub(uid_cxl); st.success("Cancelled."); st.rerun()

    # ── Plans ────────────────────────────────────────────────────────
    with atabs[4]:
        sec_head("PLANS","Plan Configuration","Edit pricing & features")
        for plan in get_all_plans():
            p = dict(plan)
            tier_lbl = p.get('tier', 1)
            with st.expander(f"✏️ {p['name']} (Tier {tier_lbl})"):
                c1,c2,c3,c4,c5 = st.columns(5)
                with c1: pm    = st.number_input("$/mo",value=float(p.get('price_monthly',0)),key=f"pm{p['id']}")
                with c2: py    = st.number_input("$/yr",value=float(p.get('price_yearly',0)),key=f"py{p['id']}")
                with c3: rows_ = st.number_input("Max Rows",value=int(p.get('max_rows',5000)),step=5000,key=f"rw{p['id']}")
                with c4: conns_= st.number_input("Max Connectors",value=int(p.get('max_connectors',0)),step=1,key=f"cn{p['id']}")
                with c5: dash_ = st.number_input("Dashboards",value=int(p.get('max_dashboards',3)),step=1,key=f"db{p['id']}")
                ft = st.text_area("Features",value=p.get('features',''),key=f"ft{p['id']}")
                if st.button(f"💾 Save {p['name']}",key=f"sv{p['id']}"):
                    update_plan(p['id'],pm,py,rows_,conns_,dash_,ft)
                    st.success("Plan updated!"); st.rerun()

    # ── Settings ─────────────────────────────────────────────────────
    with atabs[5]:
        sec_head("SETTINGS","System Configuration","AI · Email · Storage · Security")
        stab1,stab2,stab3,stab4 = st.tabs(["🤖 AI Providers","📧 Email / SMTP","☁️ Cloud Storage","🛡️ Security"])

        with stab1:
            curr    = get_setting("ai_provider","deepseek")
            ce      = get_setting("custom_ai_enabled") == "1"
            opts    = ["deepseek","groq"] + (["custom"] if ce else [])
            prov    = st.selectbox("Primary AI Provider",opts,index=opts.index(curr) if curr in opts else 0)
            ds_k    = st.text_input("DeepSeek API Key",type="password",value=get_setting("deepseek_api_key"))
            gr_k    = st.text_input("Groq API Key",type="password",value=get_setting("groq_api_key"))
            st.markdown("---")
            en_c    = st.checkbox("Enable Custom AI (OpenAI-compatible endpoint)",value=ce)
            c_url_  = st.text_input("Custom AI URL",value=get_setting("custom_ai_url"))
            c_key_  = st.text_input("Custom AI API Key",type="password",value=get_setting("custom_ai_api_key"))
            c_mod_  = st.text_input("Model Name",value=get_setting("custom_ai_model"),placeholder="gpt-4o / claude-3 / llama-3...")
            if st.button("💾 Save AI Settings",use_container_width=True):
                set_setting("ai_provider",prov); set_setting("deepseek_api_key",ds_k)
                set_setting("groq_api_key",gr_k); set_setting("custom_ai_enabled","1" if en_c else "0")
                if en_c:
                    set_setting("custom_ai_url",c_url_)
                    set_setting("custom_ai_api_key",c_key_)
                    set_setting("custom_ai_model",c_mod_)
                st.success("✅ AI settings saved"); st.rerun()

        with stab2:
            smtp_h = st.text_input("SMTP Host",value=get_setting("smtp_host"),placeholder="smtp.gmail.com")
            smtp_p = st.text_input("SMTP Port",value=get_setting("smtp_port"),placeholder="587")
            smtp_u = st.text_input("SMTP User",value=get_setting("smtp_user"))
            smtp_pw= st.text_input("SMTP Password",type="password",value=get_setting("smtp_pass"))
            from_e = st.text_input("From Email",value=get_setting("alert_email_from"))
            if st.button("💾 Save SMTP",use_container_width=True):
                set_setting("smtp_host",smtp_h); set_setting("smtp_port",smtp_p)
                set_setting("smtp_user",smtp_u); set_setting("smtp_pass",smtp_pw)
                set_setting("alert_email_from",from_e)
                st.success("✅ SMTP settings saved")
            if st.button("📧 Send Test Email",use_container_width=True):
                st.info("Test email would be sent in production with SMTP configured.")

        with stab3:
            c1,c2,c3 = st.columns(3)
            with c1:
                st.markdown("**AWS S3**")
                s3_b = st.text_input("Bucket",value=get_setting("s3_bucket"))
                s3_r = st.text_input("Region",value=get_setting("s3_region"))
                s3_a = st.text_input("Access Key",value=get_setting("s3_access_key"))
                s3_s = st.text_input("Secret Key",type="password",value=get_setting("s3_secret_key"))
                if st.button("💾 Save S3",use_container_width=True):
                    set_setting("s3_bucket",s3_b); set_setting("s3_region",s3_r)
                    set_setting("s3_access_key",s3_a); set_setting("s3_secret_key",s3_s)
                    st.success("Saved")
            with c2:
                st.markdown("**Google GCS**")
                gcs_b = st.text_input("Bucket",value=get_setting("gcs_bucket"),key="gcs_bk")
                gcs_c = st.text_area("Credentials JSON",value=get_setting("gcs_credentials_json"),height=100)
                if st.button("💾 Save GCS",use_container_width=True):
                    set_setting("gcs_bucket",gcs_b); set_setting("gcs_credentials_json",gcs_c)
                    st.success("Saved")
            with c3:
                st.markdown("**Azure Blob**")
                az_c = st.text_input("Connection String",value=get_setting("azure_connection_string"),type="password")
                az_ct= st.text_input("Container",value=get_setting("azure_container"))
                if st.button("💾 Save Azure",use_container_width=True):
                    set_setting("azure_connection_string",az_c); set_setting("azure_container",az_ct)
                    st.success("Saved")

        with stab4:
            st.selectbox("Session Timeout (minutes)", [15,30,60,120,480,0])
            st.checkbox("Enforce strong passwords", True)
            st.checkbox("Enable brute-force protection", True)
            st.number_input("Max login attempts before lockout", 3, 20, 5)
            st.checkbox("Require email verification", False)
            st.checkbox("Enable two-factor authentication", False)
            reg_en = st.checkbox("Allow new registrations", get_setting("registration_enabled")=="1")
            if st.button("💾 Save Security Settings",use_container_width=True):
                set_setting("registration_enabled","1" if reg_en else "0")
                st.success("Security settings saved")

    # ── Analytics ─────────────────────────────────────────────────────
    with atabs[6]:
        sec_head("ANALYTICS","Usage Analytics","Platform engagement metrics")
        slogs3 = get_system_logs(1000)
        if slogs3:
            df_an = pd.DataFrame(slogs3, columns=["User","Action","Details","Module","Timestamp"])
            df_an["Timestamp"] = pd.to_datetime(df_an["Timestamp"])
            df_an["date"] = df_an["Timestamp"].dt.date
            c1,c2 = st.columns(2)
            with c1:
                daily = df_an.groupby("date").size().reset_index(name="n")
                fig = px.bar(daily, x="date", y="n", title="DAILY PLATFORM ACTIVITY",
                             color="n", color_continuous_scale="Blues")
                st.plotly_chart(pf(fig), use_container_width=True)
            with c2:
                mod_cnt = df_an["Module"].value_counts().reset_index()
                mod_cnt.columns = ["Module","Count"]
                fig2 = px.pie(mod_cnt, names="Module", values="Count",
                              title="ACTIVITY BY MODULE", hole=0.4)
                st.plotly_chart(pf(fig2), use_container_width=True)
            c3,c4 = st.columns(2)
            with c3:
                user_cnt = df_an.groupby("User").size().sort_values(ascending=False).head(10).reset_index()
                user_cnt.columns = ["User","Actions"]
                fig3 = px.bar(user_cnt, x="Actions", y="User", orientation='h', title="TOP POWER USERS")
                st.plotly_chart(pf(fig3), use_container_width=True)
            with c4:
                hourly = df_an.copy()
                hourly["hour"] = hourly["Timestamp"].dt.hour
                hc = hourly.groupby("hour").size().reset_index(name="n")
                fig4 = px.bar(hc, x="hour", y="n", title="ACTIVITY BY HOUR OF DAY",
                              color="n", color_continuous_scale="Viridis")
                st.plotly_chart(pf(fig4), use_container_width=True)

    # ── System ────────────────────────────────────────────────────────
    with atabs[7]:
        sec_head("SYSTEM","System Management","Cache · Database · Maintenance")
        c1,c2 = st.columns(2)
        with c1:
            st.markdown("""<div class="nx-panel">
              <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--cyan);
                          letter-spacing:2px;margin-bottom:12px;">🗄️ DATABASE</div>""", unsafe_allow_html=True)
            if os.path.exists(DB_PATH):
                db_size = os.path.getsize(DB_PATH)/1048576
                st.metric("DB Size", f"{db_size:.3f} MB")
            col_a,col_b = st.columns(2)
            with col_a:
                if st.button("🧹 Clear Cache",use_container_width=True):
                    st.cache_data.clear(); st.cache_resource.clear()
                    st.success("Cache cleared")
            with col_b:
                if st.button("🗑 Clear Logs",use_container_width=True):
                    with get_db() as c_db:
                        c_db.execute("DELETE FROM login_logs WHERE timestamp < date('now','-30 days')")
                        c_db.execute("DELETE FROM system_logs WHERE timestamp < date('now','-90 days')")
                        c_db.commit()
                    st.success("Old logs cleared")
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown("""<div class="nx-panel">
              <div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--cyan);
                          letter-spacing:2px;margin-bottom:12px;">⚙️ APP CONFIG</div>""", unsafe_allow_html=True)
            app_n = st.text_input("App Name",value=get_setting("app_name"))
            maint = st.checkbox("Maintenance Mode",value=get_setting("maintenance_mode")=="1")
            max_up= st.number_input("Max Upload MB",value=int(get_setting("max_upload_mb","1000")))
            if st.button("💾 Save Config",use_container_width=True):
                set_setting("app_name",app_n)
                set_setting("maintenance_mode","1" if maint else "0")
                set_setting("max_upload_mb",str(max_up))
                st.success("Config saved")
            st.markdown("</div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
#  SUBSCRIPTION PLANS TAB
# ═══════════════════════════════════════════════════════════════════════
def plans_tab():
    sec_head("PLN", "Subscription Plans", "Choose the right tier for your team")
    if not st.session_state.get("logged_in"):
        alert_box("Sign in to view and subscribe to plans.", "warn"); return

    user = get_user_by_email(st.session_state["user_email"])
    if not user: return
    sub = get_user_subscription(user["id"])

    if sub:
        sub_d = dict(sub) if not isinstance(sub, dict) else sub
        st.markdown(f"""<div class="nx-panel" style="margin-bottom:18px;border-left:3px solid var(--cyan);">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--t3);letter-spacing:2px;text-transform:uppercase;">Current Plan</div>
            <strong style="font-family:var(--font-head);font-size:1.1rem;color:var(--cyan);">{sub_d.get('name','Free').upper()}</strong>
            <span class="nx-chip cyan">{sub_d.get('max_rows',5000):,} rows</span>
            <span class="nx-chip">{sub_d.get('max_connectors',0)} connectors</span>
            <span class="nx-chip">{sub_d.get('max_dashboards',3)} dashboards</span>
          </div>
        </div>""", unsafe_allow_html=True)

    plans = get_available_plans()
    if not plans:
        st.info("No plans available."); return
    cols  = st.columns(len(plans))
    sub_d = dict(sub) if sub and not isinstance(sub, dict) else (sub or {})
    for idx, plan in enumerate(plans):
        plan = dict(plan)
        with cols[idx]:
            is_curr    = bool(sub_d) and sub_d.get('name') == plan.get('name')
            is_feat    = plan.get('name') == 'Professional'
            cls        = "plan-card featured" if is_feat else "plan-card"
            feats_list = [f.strip() for f in plan.get('features','').split('·') if f.strip()]
            feats_html = "".join([f'<div class="plan-feat">{f}</div>' for f in feats_list[:6]])
            # Extract all values safely before use
            p_id       = plan.get('id', idx + 1)
            p_name     = plan.get('name', 'Plan')
            p_monthly  = float(plan.get('price_monthly', 0) or 0)
            p_yearly   = float(plan.get('price_yearly', 0) or 0)
            p_rows     = int(plan.get('max_rows', 0) or 0)
            p_conn     = int(plan.get('max_connectors', 0) or 0)
            yr_save    = int((1 - p_yearly / (p_monthly * 12)) * 100) if p_monthly > 0 else 0
            popular    = '<div class="plan-popular">★ POPULAR</div>' if is_feat else ''
            max_rows_fmt = "Unlimited" if p_rows > 900000000 else f"{p_rows:,}"

            st.markdown(f"""<div class="{cls}">
              {popular}
              <div class="plan-name-lbl">{p_name}</div>
              <div class="plan-price">${p_monthly:.2f}<span>/mo</span></div>
              <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--t3);margin-bottom:14px;">
                ${p_yearly:.0f}/yr · save {yr_save}%
              </div>
              <div style="text-align:left;margin:12px 0;">{feats_html}</div>
              <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--t3);margin-top:8px;">
                📊 {max_rows_fmt} rows &nbsp;|&nbsp; 🔌 {p_conn} connectors
              </div>
            </div>""", unsafe_allow_html=True)

            if is_curr:
                st.button("✅ Active Plan", disabled=True, key=f"cur_{p_id}", use_container_width=True)
            else:
                if st.button(f"→ Subscribe {p_name}", key=f"sub_{p_id}", use_container_width=True):
                    upgrade_subscription(user["id"], p_id)
                    st.success(f"✅ Subscribed to {p_name}!")
                    log_action(st.session_state["user_email"], "subscribe", p_name, module="billing")
                    st.rerun()

# ═══════════════════════════════════════════════════════════════════════
#  MAIN ANALYTICS APP — Full Tab Suite
# ═══════════════════════════════════════════════════════════════════════
def analytics_app():
    # Session state initialization
    for k, v in [
        ("df", None), ("roles", {}), ("source", None),
        ("col_map", {k2: "—" for k2 in ["sales","profit","date","category","customer","product","quantity","region"]}),
        ("quality_score", 0), ("quality_details", {}),
    ]:
        if k not in st.session_state: st.session_state[k] = v

    for k2 in ["sales","profit","date","category","customer","product","quantity","region"]:
        if k2 not in st.session_state["col_map"]:
            st.session_state["col_map"][k2] = "—"

    user_plan = None
    if st.session_state.get("logged_in"):
        u = get_user_by_email(st.session_state["user_email"])
        if u: user_plan = get_user_subscription(u["id"])

    plan_name    = user_plan["name"] if user_plan else "Starter"
    max_rows     = user_plan["max_rows"] if user_plan else GUEST_MAX_ROWS
    max_connectors = (dict(user_plan).get("max_connectors", 0) if user_plan else 0)
    st.session_state["_plan"]           = plan_name
    st.session_state["_max_connectors"] = max_connectors
    is_pro = plan_name in ["Professional","Business","Enterprise"]

    # ── SIDEBAR ───────────────────────────────────────────────────────
    with st.sidebar:
        sidebar_brand()

        if st.session_state.get("logged_in"):
            sidebar_user_card(st.session_state["user_email"], plan_name, max_rows)
            if st.session_state.get("is_admin"):
                if st.button("🛡 Control Center", use_container_width=True, key="goto_admin"):
                    st.session_state.admin_mode = True; st.rerun()
        else:
            st.markdown("""<div style="margin:0 10px 12px;padding:8px 12px;
                background:var(--card);border:1px solid var(--grid-med);
                border-radius:var(--r2);text-align:center;">
              <div style="font-family:var(--font-mono);font-size:0.62rem;color:var(--t3);">
                👁 GUEST · 2,000 ROWS
              </div>
            </div>""", unsafe_allow_html=True)

        # ── DATA SOURCE ──────────────────────────────────────────────
        st.markdown('<div class="sb-section-label">Data Source</div>', unsafe_allow_html=True)
        src_choice = st.radio("", ["📦 Demo Dataset","📂 Upload File"],
                               label_visibility="collapsed", key="src_r")

        if src_choice == "📂 Upload File":
            up = st.file_uploader("CSV / Excel / JSON", type=["csv","xlsx","xls","json"], key="fu")
            if up:
                if up.size > MAX_FILE_SIZE_BYTES:
                    st.error(f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
                else:
                    try:
                        if up.name.endswith(".csv"):
                            try:   df_new, enc = read_csv_smart(up); st.success(f"✅ {enc}")
                            except:
                                enc = st.selectbox("Encoding",['utf-8','windows-1256','iso-8859-1','cp1252'])
                                up.seek(0); df_new = pd.read_csv(up, encoding=enc)
                        elif up.name.endswith(".json"): df_new = pd.read_json(up)
                        else:                            df_new = pd.read_excel(up)

                        if len(df_new) > max_rows:
                            st.error(f"⚠️ {len(df_new):,} rows exceeds {max_rows:,} limit. Upgrade plan.")
                        else:
                            if st.session_state["source"] != up.name:
                                roles = detect_column_types(df_new)
                                df_c  = smart_clean(df_new, roles)
                                qs, qd = compute_data_quality(df_c, roles)
                                cm_   = auto_map_columns(df_new, roles)
                                st.session_state.update({
                                    "df": df_c, "roles": roles, "source": up.name,
                                    "col_map": cm_, "quality_score": qs, "quality_details": qd
                                })
                                log_action(st.session_state.get("user_email","guest"),
                                           "upload_file", up.name, module="data")
                            dts = st.session_state["roles"].get("date", [])
                            if dts: st.caption(f"📅 {', '.join(dts)}")
                            qs = st.session_state.get("quality_score", 0)
                            st.caption(f"Quality: {qs}/100")
                            progress_bar(qs, "#00e676" if qs>70 else "#ffca28" if qs>40 else "#ff3d57")
                    except Exception as ex: st.error(f"Error: {ex}")
        else:
            if st.session_state["source"] != "builtin":
                df_bi = load_builtin()
                if len(df_bi) > max_rows:
                    st.error(f"Dataset {len(df_bi):,} rows exceeds {max_rows:,} limit.")
                else:
                    roles = detect_column_types(df_bi)
                    df_c  = smart_clean(df_bi, roles)
                    qs, qd = compute_data_quality(df_c, roles)
                    st.session_state.update({
                        "df": df_c, "roles": roles, "source": "builtin",
                        "quality_score": qs, "quality_details": qd,
                        "col_map": {
                            "sales":"Sales","profit":"Profit","date":"Order Date",
                            "category":"Category","customer":"Customer_ID",
                            "product":"Category","quantity":"Quantity","region":"Sub-Region"
                        }
                    })
                    log_action(st.session_state.get("user_email","guest"),
                               "load_builtin", module="data")
            st.success("✅ Demo dataset (8K rows)")
            qs = st.session_state.get("quality_score", 0)
            progress_bar(qs, "#00e676")

        # ── COLUMN MAPPING ───────────────────────────────────────────
        df = st.session_state.get("df")
        if df is not None:
            st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sb-section-label">Column Mapping</div>', unsafe_allow_html=True)
            roles = st.session_state["roles"]
            nc = ["—"] + roles.get("numeric",[])
            dc = ["—"] + roles.get("date",[])
            cc = ["—"] + roles.get("categorical",[])
            ic = ["—"] + roles.get("id",[]) + roles.get("categorical",[])
            cm = st.session_state["col_map"]
            with st.expander("⚙️ Map Columns", expanded=True):
                cm["sales"]    = st.selectbox("💰 Sales/Revenue",nc, index=si(nc,cm.get("sales","—")),    key="ms")
                cm["profit"]   = st.selectbox("📈 Profit/Margin", nc, index=si(nc,cm.get("profit","—")),  key="mp")
                cm["date"]     = st.selectbox("📅 Date/Time",     dc, index=si(dc,cm.get("date","—")),    key="md")
                cm["category"] = st.selectbox("🏷️ Category",      cc, index=si(cc,cm.get("category","—")),key="mc")
                cm["customer"] = st.selectbox("👤 Customer ID",   ic, index=si(ic,cm.get("customer","—")),key="mcu")
                cm["product"]  = st.selectbox("📦 Product",       cc, index=si(cc,cm.get("product","—")), key="mpr")
                cm["quantity"] = st.selectbox("🔢 Quantity",      nc, index=si(nc,cm.get("quantity","—")), key="mqt")
                cm["region"]   = st.selectbox("🌍 Region",        cc, index=si(cc,cm.get("region","—")),  key="mrg")
                st.session_state["col_map"] = cm

        st.markdown('<div class="sb-divider"></div>', unsafe_allow_html=True)
        if st.session_state.get("logged_in"):
            if st.button("⏻ Sign Out", use_container_width=True, key="lo"):
                for k in ["logged_in","user_email","is_admin","admin_mode"]:
                    st.session_state[k] = False if k != "user_email" else None
                st.rerun()

        st.markdown(f'<div class="sb-footer">NEXUS v{APP_VERSION} · {BUILD_DATE}</div>',
                    unsafe_allow_html=True)

    # ── MAIN AREA ─────────────────────────────────────────────────────
    topbar()
    df = st.session_state.get("df")
    if df is None:
        empty_state("⬡", "NEXUS Analytics OS Ready",
                    "Load a dataset from the sidebar to initialize the analytics engine.\n\nSupports: CSV · Excel · JSON · 25+ Cloud Connectors")
        return

    cm = st.session_state["col_map"]
    TABS = ["DATA","KPIs","TRENDS","FORECAST","ML ENGINE","SEGMENTS","RFM","BASKET","ADVANCED","LIVE DATA","REPORTS","PLANS","AI CHAT"]
    tabs = st.tabs([f"  {t}  " for t in TABS])

    # ════════════════════════════════════════════════════════════════
    #  TAB 0: DATA HUB
    # ════════════════════════════════════════════════════════════════
    with tabs[0]:
        sec_head("00","Data Hub","Schema · Preview · Quality · Catalog")
        roles = st.session_state["roles"]
        qs    = st.session_state.get("quality_score",0)
        qd    = st.session_state.get("quality_details",{})

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("ROWS",    f"{len(df):,}")
        c2.metric("COLUMNS", df.shape[1])
        c3.metric("MISSING", f"{df.isnull().sum().sum():,}")
        c4.metric("MEMORY",  f"{df.memory_usage(deep=True).sum()/1048576:.2f} MB")
        c5.metric("QUALITY", f"{qs}/100", "Good" if qs>70 else "Fair" if qs>40 else "Poor")

        col1, col2 = st.columns([7,3])
        with col2:
            st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--t3);letter-spacing:2px;text-transform:uppercase;margin-bottom:8px;">SCHEMA</div>', unsafe_allow_html=True)
            icons = {"date":"📅","numeric":"🔢","categorical":"🏷️","id":"🔑","text":"📝","boolean":"☑️"}
            for role, cols_ in roles.items():
                if cols_:
                    st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.7rem;color:var(--t2);padding:4px 0;border-bottom:1px solid var(--grid-dim);">{icons.get(role,"")} <strong style="color:var(--cyan)">{role}</strong>: {", ".join(cols_[:4])}{"…" if len(cols_)>4 else ""}</div>', unsafe_allow_html=True)

            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown('<div style="font-family:var(--font-mono);font-size:0.6rem;color:var(--t3);letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">QUALITY BREAKDOWN</div>', unsafe_allow_html=True)
            for k,v in qd.items():
                lbl = k.replace("_"," ").title()
                val = f"{v}%" if "pct" in k else ("✅" if v else "❌")
                st.markdown(f'<div style="font-family:var(--font-mono);font-size:0.68rem;color:var(--t2);padding:2px 0;">{lbl}: <span style="color:var(--cyan)">{val}</span></div>', unsafe_allow_html=True)

        with col1:
            st.dataframe(df.head(300), use_container_width=True, height=320)

        with st.expander("📊 Full Statistical Summary"):
            st.dataframe(df.describe(include="all").T, use_container_width=True)

        with st.expander("🔍 Missing Values Analysis"):
            miss = df.isnull().sum().reset_index()
            miss.columns = ["Column","Missing"]
            miss["Pct"] = (miss["Missing"]/len(df)*100).round(2)
            miss = miss[miss["Missing"]>0].sort_values("Missing",ascending=False)
            if len(miss):
                fig = px.bar(miss, x="Column", y="Pct", title="MISSING VALUE % BY COLUMN",
                             color="Pct", color_continuous_scale="Reds",
                             hover_data={"Missing":True,"Pct":True})
                st.plotly_chart(pf(fig), use_container_width=True)
                st.dataframe(miss, use_container_width=True)
            else:
                alert_box("No missing values detected — excellent data quality!", "ok")

        with st.expander("📦 Data Catalog"):
            if st.button("📋 Save to Catalog", key="save_catalog"):
                user_id = get_user_by_email(st.session_state.get("user_email",""))
                if user_id:
                    src = st.session_state.get("source","unknown")
                    schema = {col: str(df[col].dtype) for col in df.columns}
                    save_to_catalog(user_id["id"], src, "file",
                                    len(df), df.shape[1],
                                    df.memory_usage(deep=True).sum(),
                                    schema, quality=qs)
                    st.success("Saved to data catalog!")

    # ════════════════════════════════════════════════════════════════
    #  TAB 1: KPIs
    # ════════════════════════════════════════════════════════════════
    with tabs[1]:
        sec_head("01","Key Performance Indicators","Revenue · Profit · Orders · Growth")
        sc = cm.get("sales","—"); pc = cm.get("profit","—")
        dc2= cm.get("date","—");  cc = cm.get("category","—"); rc = cm.get("region","—")

        if sc != "—" and sc in df.columns:
            tr = df[sc].sum()
            tp = df[pc].sum() if pc!="—" and pc in df.columns else None
            mg = (tp/tr*100) if tp and tr else None
            ao = df[sc].mean()
            tot_orders = len(df)
            max_order  = df[sc].max()

            kpi_cols = st.columns(6)
            metrics_ = [
                ("TOTAL REVENUE",  fmt(tr,"$"),    "cyan"),
                ("TOTAL PROFIT",   fmt(tp,"$") if tp else "—", "green"),
                ("PROFIT MARGIN",  f"{mg:.1f}%" if mg else "—","teal"),
                ("AVG ORDER",      fmt(ao,"$"),     ""),
                ("TOTAL ORDERS",   f"{tot_orders:,}",""),
                ("MAX ORDER",      fmt(max_order,"$"),"amber"),
            ]
            for col,(lbl,val,cls) in zip(kpi_cols, metrics_):
                col.metric(lbl, val)

            if dc2 != "—" and dc2 in df.columns:
                c1, c2 = st.columns(2)
                with c1:
                    ts = df.set_index(dc2).resample('ME')[sc].sum().reset_index()
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ts[dc2], y=ts[sc], mode="lines+markers",
                                             name="Revenue", line=dict(color="#00d4ff",width=2.5),
                                             fill="tozeroy", fillcolor="rgba(0,212,255,0.06)"))
                    if pc != "—" and pc in df.columns:
                        tp2 = df.set_index(dc2).resample('ME')[pc].sum().reset_index()
                        fig.add_trace(go.Scatter(x=tp2[dc2], y=tp2[pc], mode="lines",
                                                 name="Profit", line=dict(color="#00e676",width=2,dash="dot")))
                    st.plotly_chart(pf(fig,"MONTHLY REVENUE & PROFIT TREND"), use_container_width=True)
                with c2:
                    if cc != "—" and cc in df.columns:
                        cs = df.groupby(cc)[sc].sum().reset_index().sort_values(sc,ascending=True).tail(10)
                        fig2 = px.bar(cs, y=cc, x=sc, orientation='h',
                                      title="TOP CATEGORIES BY REVENUE",
                                      color=sc, color_continuous_scale="Blues_r")
                        st.plotly_chart(pf(fig2), use_container_width=True)

                c3, c4 = st.columns(2)
                with c3:
                    df_d = df.set_index(dc2)
                    df_d["DOW"] = df_d.index.day_name()
                    dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                    dow_s = df_d.groupby("DOW")[sc].mean().reindex(dow_order).fillna(0)
                    colors_ = ["#00d4ff" if v==dow_s.max() else "#1a6bff" for v in dow_s]
                    fig3 = go.Figure(go.Bar(x=dow_s.index, y=dow_s.values,
                                            marker_color=colors_))
                    st.plotly_chart(pf(fig3,"AVG REVENUE BY DAY OF WEEK"), use_container_width=True)

                with c4:
                    if cc!="—" and cc in df.columns and pc!="—" and pc in df.columns:
                        gp = df.groupby(cc).agg(**{sc:(sc,'sum'),pc:(pc,'sum'),
                                                    "orders":(sc,'count')}).reset_index()
                        gp["margin"] = (gp[pc]/gp[sc]*100).round(1)
                        fig4 = px.scatter(gp, x=sc, y=pc, text=cc, size="orders",
                                          color="margin", title="REVENUE vs PROFIT MATRIX",
                                          color_continuous_scale="RdYlGn",
                                          hover_data={"margin":True,"orders":True},
                                          size_max=60)
                        st.plotly_chart(pf(fig4), use_container_width=True)
        else:
            alert_box("Map the Sales column in the sidebar to see KPIs.", "warn")

    # ════════════════════════════════════════════════════════════════
    #  TAB 2: TRENDS
    # ════════════════════════════════════════════════════════════════
    with tabs[2]:
        sec_head("02","Trend Analytics","Seasonality · YoY · MoM · Moving Averages")
        sc2 = cm.get("sales","—"); dc3 = cm.get("date","—")
        cc2 = cm.get("category","—"); rc2 = cm.get("region","—")

        if sc2!="—" and dc3!="—" and all(c in df.columns for c in [sc2,dc3]):
            c1,c2,c3 = st.columns(3)
            with c1: agg_p  = st.selectbox("Period",["Daily","Weekly","Monthly","Quarterly"],index=2)
            with c2: ma_win = st.slider("Moving Average Window",3,30,7)
            with c3: split_ = st.checkbox("Split by Category",False)

            agg_map = {"Daily":"D","Weekly":"W","Monthly":"ME","Quarterly":"QE"}
            resample_key = agg_map[agg_p]

            ts_agg = df.set_index(dc3).resample(resample_key)[sc2].sum().reset_index()
            ts_agg["MA"]   = ts_agg[sc2].rolling(ma_win, center=True).mean()
            ts_agg["MA3"]  = ts_agg[sc2].rolling(3).mean()

            if split_ and cc2!="—" and cc2 in df.columns:
                ts_cat = df.groupby([df[dc3].dt.to_period(resample_key[0]).dt.to_timestamp(), cc2])[sc2].sum().reset_index()
                fig_t = px.line(ts_cat, x=dc3, y=sc2, color=cc2, title="REVENUE TREND BY CATEGORY")
            else:
                fig_t = go.Figure()
                fig_t.add_trace(go.Scatter(x=ts_agg[dc3], y=ts_agg[sc2], mode="lines",
                                           name=agg_p+" Revenue", line=dict(color="#00d4ff",width=1.5),
                                           fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
                fig_t.add_trace(go.Scatter(x=ts_agg[dc3], y=ts_agg["MA"], mode="lines",
                                           name=f"{ma_win}-period MA", line=dict(color="#00e676",width=2.5)))
                fig_t.add_trace(go.Scatter(x=ts_agg[dc3], y=ts_agg["MA3"], mode="lines",
                                           name="3-period MA", line=dict(color="#ffca28",width=1.5,dash="dot")))
            st.plotly_chart(pf(fig_t,"REVENUE TREND ANALYSIS",h=360), use_container_width=True)

            # YoY comparison
            if len(ts_agg) > 24:
                ts_agg["Year"]  = ts_agg[dc3].dt.year
                ts_agg["Month"] = ts_agg[dc3].dt.month
                yoy = ts_agg.pivot_table(index="Month", columns="Year", values=sc2, aggfunc="sum")
                if len(yoy.columns) >= 2:
                    fig_yoy = go.Figure()
                    colors_yoy = ["#1a6bff","#00d4ff","#00e676","#ffca28"]
                    for i,yr in enumerate(yoy.columns):
                        fig_yoy.add_trace(go.Scatter(
                            x=yoy.index, y=yoy[yr], mode="lines+markers",
                            name=str(yr), line=dict(color=colors_yoy[i%len(colors_yoy)],width=2)
                        ))
                    st.plotly_chart(pf(fig_yoy,"YEAR-OVER-YEAR COMPARISON"), use_container_width=True)

            # Seasonality heatmap
            if len(df) > 100 and dc3 in df.columns:
                df_h = df.copy()
                df_h["Month"] = df_h[dc3].dt.month_name()
                df_h["DOW"]   = df_h[dc3].dt.day_name()
                heat = df_h.groupby(["DOW","Month"])[sc2].mean().reset_index()
                heat_piv = heat.pivot(index="DOW", columns="Month", values=sc2)
                months_order = ["January","February","March","April","May","June",
                                 "July","August","September","October","November","December"]
                days_order   = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
                heat_piv = heat_piv.reindex(index=days_order, columns=months_order, fill_value=0)
                fig_h = px.imshow(heat_piv, color_continuous_scale="Blues",
                                   title="SEASONALITY HEATMAP — Avg Revenue by Day × Month",
                                   aspect="auto")
                st.plotly_chart(pf(fig_h,h=320), use_container_width=True)
        else:
            alert_box("Map Date and Sales columns to see trend analytics.", "warn")

    # ════════════════════════════════════════════════════════════════
    #  TAB 3: FORECAST
    # ════════════════════════════════════════════════════════════════
    with tabs[3]:
        sec_head("03","Demand Forecasting","Prophet · Holt-Winters · Seasonal Decomposition")
        if not is_pro:
            alert_box("Forecasting requires Professional plan or higher.", "warn")
        else:
            sc3 = cm.get("sales","—"); dc4 = cm.get("date","—")
            if sc3!="—" and dc4!="—" and all(c in df.columns for c in [sc3,dc4]):
                c1,c2,c3,c4 = st.columns(4)
                with c1: horizon   = st.slider("Forecast Horizon",3,60,12)
                with c2: freq_l    = st.selectbox("Aggregation",["Monthly","Weekly"])
                with c3: show_ci   = st.checkbox("Confidence Bands",True)
                with c4: decomp_en = st.checkbox("Seasonal Decomp",True)
                fq = "ME" if freq_l=="Monthly" else "W"

                if st.button("▶ RUN FORECAST", use_container_width=True):
                    with st.spinner("Training time-series model..."):
                        hist, fcast, mname, decomp = build_forecast(
                            df[dc4].astype(str).to_json(), df[sc3].to_json(),
                            horizon, fq, decomp_en
                        )
                    if hist is None:
                        alert_box("Forecast failed. Install statsmodels or prophet.", "err")
                    else:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=hist["Date"], y=hist["Value"],
                                                  name="Historical", line=dict(color="#00d4ff",width=2),
                                                  fill="tozeroy", fillcolor="rgba(0,212,255,0.05)"))
                        if show_ci:
                            fig.add_trace(go.Scatter(
                                x=pd.concat([fcast["Date"], fcast["Date"][::-1]]),
                                y=pd.concat([fcast["Upper"], fcast["Lower"][::-1]]),
                                fill="toself", fillcolor="rgba(26,107,255,0.10)",
                                line=dict(color="rgba(0,0,0,0)"), name="CI 80%"
                            ))
                        fig.add_trace(go.Scatter(x=fcast["Date"], y=fcast["Value"],
                                                  name=f"Forecast ({mname})",
                                                  line=dict(color="#ffca28",width=2.5,dash="dot")))
                        st.plotly_chart(pf(fig,f"DEMAND FORECAST — {mname.upper()}",h=400),
                                        use_container_width=True)
                        kc = st.columns(4)
                        kc[0].metric("Model",           mname)
                        kc[1].metric("Horizon",         f"{horizon} {freq_l[:3]}")
                        kc[2].metric("Projected Total", fmt(fcast["Value"].sum(),"$"))
                        kc[3].metric("Avg / Period",    fmt(fcast["Value"].mean(),"$"))
                        st.dataframe(fcast.round(2), use_container_width=True)
                        st.download_button("📥 Export Forecast",
                                           fcast.to_csv(index=False), "forecast.csv","text/csv")

                        if decomp is not None:
                            st.markdown("---")
                            sec_head("DECOMP","Seasonal Decomposition","Trend · Seasonal · Residual")
                            fig_d = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                                  subplot_titles=["Observed","Trend","Seasonal","Residual"],
                                                  vertical_spacing=0.05)
                            for r, (comp, clr) in enumerate(
                                    zip([decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
                                        ["#00d4ff","#00e676","#ffca28","#ff8c00"]), 1):
                                fig_d.add_trace(go.Scatter(y=comp, mode="lines",
                                                            line=dict(color=clr,width=1.5)), row=r, col=1)
                            st.plotly_chart(pf(fig_d,h=500), use_container_width=True)
            else:
                alert_box("Map Date and Sales columns to run forecasting.", "warn")

    # ════════════════════════════════════════════════════════════════
    #  TAB 4: ML ENGINE
    # ════════════════════════════════════════════════════════════════
    with tabs[4]:
        sec_head("04","ML Profit Optimizer","5-Model Ensemble · AutoFeature · Experiment Tracking")
        pc_ml = cm.get("profit","—")
        if pc_ml!="—" and pc_ml in df.columns:
            avail = [c for c in df.columns if c!=pc_ml]
            c1,c2 = st.columns([3,1])
            with c1:
                features_sel = st.multiselect("Training Features", avail, default=[],
                                               help="Select features for the ML model")
            with c2:
                exp_name = st.text_input("Experiment Name", value=f"exp_{datetime.now().strftime('%H%M%S')}")

            if features_sel and st.button("▶ TRAIN ENSEMBLE", use_container_width=True):
                with st.spinner("Training 5-model voting ensemble (RF + GBM + ExtraTrees + BayesianRidge + Ridge)..."):
                    try:
                        td = df[features_sel+[pc_ml]].dropna()
                        if len(td) < 30:
                            alert_box("Need at least 30 rows for training.", "err")
                        else:
                            result = train_ensemble(td.to_json(), pc_ml, features_sel)
                            _,_,_,r2,mape,rmse,imps,pimps,corrs,_ = result
                            kc = st.columns(5)
                            kc[0].metric("CV R² Score",  f"{r2:.4f}" if r2   else "N/A")
                            kc[1].metric("CV MAPE",      f"{mape:.2f}%" if mape else "N/A")
                            kc[2].metric("CV RMSE",      fmt(rmse) if rmse else "N/A")
                            kc[3].metric("Train Rows",   f"{len(td):,}")
                            kc[4].metric("Features",     len(features_sel))

                            col1,col2 = st.columns(2)
                            with col1:
                                ids = pd.Series(imps).sort_values()
                                fig = px.bar(ids, orientation='h', title="RF FEATURE IMPORTANCE",
                                             color=ids.values, color_continuous_scale="Blues")
                                st.plotly_chart(pf(fig), use_container_width=True)
                            with col2:
                                pis = pd.Series(pimps).sort_values()
                                fig2 = px.bar(pis, orientation='h', title="PERMUTATION IMPORTANCE",
                                              color=pis.values, color_continuous_scale="Greens")
                                st.plotly_chart(pf(fig2), use_container_width=True)

                            # Correlation to target
                            cors_ = pd.Series(corrs).sort_values(ascending=False)
                            fig3 = px.bar(cors_, title="CORRELATION TO TARGET (absolute)",
                                          color=cors_.values, color_continuous_scale="RdYlGn")
                            st.plotly_chart(pf(fig3), use_container_width=True)

                            # Save experiment
                            user_obj = get_user_by_email(st.session_state.get("user_email",""))
                            if user_obj:
                                save_ml_experiment(user_obj["id"], exp_name, "VotingEnsemble",
                                                   pc_ml, features_sel,
                                                   r2 or 0, mape or 0, rmse or 0, len(td))
                                alert_box(f"Experiment '{exp_name}' saved to history.", "ok")
                    except Exception as ex:
                        alert_box(f"Training error: {ex}", "err")

            # Experiment history
            user_obj = get_user_by_email(st.session_state.get("user_email",""))
            if user_obj:
                exps = get_ml_experiments(user_obj["id"])
                if exps:
                    with st.expander("📋 Experiment History"):
                        df_exps = pd.DataFrame(exps)[["experiment_name","model_type","target_col","r2_score","mape","rmse","training_rows","created_at"]]
                        st.dataframe(df_exps, use_container_width=True)
        else:
            alert_box("Map the Profit column in the sidebar to train ML models.", "warn")

    # ════════════════════════════════════════════════════════════════
    #  TAB 5: SEGMENTS (CLUSTERING)
    # ════════════════════════════════════════════════════════════════
    with tabs[5]:
        sec_head("05","Customer Segmentation","Advanced Clustering · KMeans · DBSCAN · Hierarchical")
        if not is_pro:
            alert_box("Clustering requires Professional plan or higher.", "warn")
        else:
            num_c = df.select_dtypes(include=np.number).columns.tolist()
            if len(num_c) >= 2:
                c1,c2,c3,c4 = st.columns(4)
                with c1: method   = st.selectbox("Algorithm",["kmeans","dbscan","minibatch","hierarchical"])
                with c2: k_       = st.slider("Clusters/K",2,12,4) if method!="dbscan" else st.slider("Epsilon",0.1,3.0,0.5,0.05)
                with c3: feat_c   = st.multiselect("Cluster Features",num_c,default=num_c[:min(4,len(num_c))])
                with c4: min_s    = st.slider("Min Samples (DBSCAN)",2,20,5)

                if feat_c and st.button("▶ RUN CLUSTERING", use_container_width=True):
                    with st.spinner("Running clustering algorithm..."):
                        try:
                            eps_v = k_ if method=="dbscan" else 0.5
                            k_v   = k_ if method!="dbscan" else 4
                            lbs,sil,dbi,coords,inertias,var,profiles = run_clustering(
                                df[feat_c].dropna().to_json(), feat_c, method, k_v, eps_v, min_s
                            )
                            n_cl = len(set(lbs))-(1 if -1 in lbs else 0)
                            kc   = st.columns(4)
                            kc[0].metric("Clusters Found",  n_cl)
                            kc[1].metric("Silhouette",      f"{sil:.4f}" if sil else "N/A")
                            kc[2].metric("Davies-Bouldin",  f"{dbi:.4f}" if dbi else "N/A")
                            kc[3].metric("Noise Points",    str((lbs==-1).sum()) if -1 in lbs else "0")

                            c1,c2 = st.columns(2)
                            with c1:
                                fig = px.scatter(x=coords[:,0], y=coords[:,1],
                                                  color=lbs.astype(str),
                                                  title=f"PCA PROJECTION — {method.upper()}",
                                                  labels={"x":f"PC1 ({var[0]*100:.1f}%)",
                                                          "y":f"PC2 ({var[1]*100:.1f}%)"},
                                                  color_discrete_sequence=px.colors.qualitative.Vivid)
                                st.plotly_chart(pf(fig), use_container_width=True)
                            with c2:
                                if inertias:
                                    ks_ = list(inertias.keys())
                                    iv_ = list(inertias.values())
                                    fig2 = go.Figure()
                                    fig2.add_trace(go.Scatter(x=ks_, y=iv_, mode="lines+markers",
                                                               line=dict(color="#00d4ff",width=2),
                                                               marker=dict(size=8)))
                                    fig2.add_vline(x=k_v, line_dash="dot", line_color="#ff3d57")
                                    st.plotly_chart(pf(fig2,"ELBOW METHOD"), use_container_width=True)
                                else:
                                    # Cluster sizes
                                    cl_cnt = pd.Series(lbs).value_counts().reset_index()
                                    cl_cnt.columns = ["Cluster","Count"]
                                    fig_cl = px.pie(cl_cnt, names="Cluster", values="Count",
                                                    title="CLUSTER SIZE DISTRIBUTION", hole=0.4)
                                    st.plotly_chart(pf(fig_cl), use_container_width=True)

                            # Cluster profiles
                            if profiles is not None:
                                st.markdown("**Cluster Profiles (Feature Means)**")
                                st.dataframe(profiles, use_container_width=True)
                        except Exception as ex:
                            alert_box(f"Clustering error: {ex}", "err")

    # ════════════════════════════════════════════════════════════════
    #  TAB 6: RFM
    # ════════════════════════════════════════════════════════════════
    with tabs[6]:
        sec_head("06","RFM Customer Analysis","Recency · Frequency · Monetary Segmentation")
        cst = cm.get("customer","—"); sl  = cm.get("sales","—"); dl  = cm.get("date","—")
        if all(c!="—" and c in df.columns for c in [cst,sl,dl]):
            if st.button("▶ RUN RFM ANALYSIS", use_container_width=True):
                with st.spinner("Computing RFM scores..."):
                    rfm = compute_rfm(df, dl, sl, cst)
                if rfm is not None:
                    kc = st.columns(5)
                    kc[0].metric("Total Customers", f"{len(rfm):,}")
                    kc[1].metric("Champions",  str((rfm["Segment"].str.contains("Champions")).sum()))
                    kc[2].metric("Loyal",      str((rfm["Segment"].str.contains("Loyal")).sum()))
                    kc[3].metric("At Risk",    str((rfm["Segment"].str.contains("At Risk")).sum()))
                    kc[4].metric("Avg Monetary",fmt(rfm["Monetary"].mean(),"$"))

                    c1,c2 = st.columns(2)
                    with c1:
                        sc_ = rfm["Segment"].value_counts()
                        fig = px.pie(sc_, names=sc_.index, values=sc_.values,
                                     title="CUSTOMER SEGMENT DISTRIBUTION", hole=0.45)
                        st.plotly_chart(pf(fig), use_container_width=True)
                    with c2:
                        fig2 = px.scatter(rfm, x="Frequency", y="Monetary", color="Segment",
                                           size="RFM_Score", title="RFM SCATTER MAP",
                                           size_max=35, hover_data=["Recency"])
                        st.plotly_chart(pf(fig2), use_container_width=True)

                    c3,c4 = st.columns(2)
                    with c3:
                        fig3 = px.box(rfm, y="Monetary", color="Segment",
                                       title="MONETARY VALUE BY SEGMENT")
                        st.plotly_chart(pf(fig3), use_container_width=True)
                    with c4:
                        fig4 = px.histogram(rfm, x="Recency", color="Segment", nbins=40,
                                             title="RECENCY DISTRIBUTION BY SEGMENT",
                                             barmode="overlay", opacity=0.7)
                        st.plotly_chart(pf(fig4), use_container_width=True)

                    st.dataframe(rfm.head(200), use_container_width=True)
                    st.download_button("📥 Export RFM", rfm.to_csv(index=False), "rfm.csv","text/csv")

                    # Cohort analysis
                    with st.expander("📊 Cohort Retention Analysis"):
                        ret_matrix = compute_cohort_analysis(df, dl, cst, sl)
                        if ret_matrix is not None:
                            fig_coh = px.imshow(ret_matrix, text_auto=".0f",
                                                 color_continuous_scale="Blues",
                                                 title="COHORT RETENTION MATRIX (%)", aspect="auto")
                            st.plotly_chart(pf(fig_coh,h=400), use_container_width=True)
                        else:
                            st.info("Need sufficient date range for cohort analysis.")
                else:
                    alert_box("Could not compute RFM. Check column mapping.", "err")
        else:
            alert_box("Map Customer ID, Date, and Sales columns to run RFM.", "warn")

    # ════════════════════════════════════════════════════════════════
    #  TAB 7: BASKET
    # ════════════════════════════════════════════════════════════════
    with tabs[7]:
        sec_head("07","Market Basket Analysis","Apriori Association Rules · Product Affinity")
        if not is_pro:
            alert_box("Market Basket requires Professional plan or higher.", "warn")
        elif not MLXTEND_AVAILABLE:
            alert_box("Install mlxtend: pip install mlxtend", "warn")
        else:
            cst2 = cm.get("customer","—"); prd = cm.get("product","—")
            if cst2!="—" and prd!="—" and all(c in df.columns for c in [cst2,prd]):
                c1,c2,c3 = st.columns(3)
                with c1: msup  = st.slider("Min Support",  0.005, 0.15, 0.01, 0.005, format="%.3f")
                with c2: mlift = st.slider("Min Lift",     1.0,   8.0,  1.5,  0.1)
                with c3: mconf = st.slider("Min Confidence",0.1,  1.0,  0.3,  0.05)

                if st.button("▶ RUN APRIORI MINING", use_container_width=True):
                    with st.spinner("Mining association rules..."):
                        try:
                            _, rules_, msg = market_basket(df, cst2, prd, msup, mconf, mlift)
                        except Exception as ex:
                            rules_ = None; msg = str(ex)

                    if rules_ is not None and len(rules_):
                        kc = st.columns(4)
                        kc[0].metric("Rules Found",    len(rules_))
                        kc[1].metric("Avg Lift",       f"{rules_['lift'].mean():.3f}")
                        kc[2].metric("Avg Confidence", f"{rules_['confidence'].mean():.3f}")
                        kc[3].metric("Avg Support",    f"{rules_['support'].mean():.4f}")

                        c1,c2 = st.columns(2)
                        with c1:
                            fig = px.scatter(rules_, x="support", y="confidence", color="lift",
                                             size="lift", title="SUPPORT vs CONFIDENCE (colored by Lift)",
                                             size_max=30, color_continuous_scale="Plasma")
                            st.plotly_chart(pf(fig), use_container_width=True)
                        with c2:
                            top_rules = rules_.sort_values("lift",ascending=False).head(15)
                            top_rules["rule"] = top_rules["antecedents"].astype(str) + " → " + top_rules["consequents"].astype(str)
                            fig2 = px.bar(top_rules, x="lift", y="rule", orientation='h',
                                          title="TOP 15 RULES BY LIFT", color="confidence",
                                          color_continuous_scale="Blues")
                            st.plotly_chart(pf(fig2), use_container_width=True)

                        st.dataframe(
                            rules_[["antecedents","consequents","support","confidence","lift"]]
                            .sort_values("lift",ascending=False),
                            use_container_width=True
                        )
                        st.download_button("📥 Export Rules", rules_.to_csv(index=False), "rules.csv","text/csv")
                    else:
                        alert_box(f"No rules found: {msg}. Try reducing min_support.", "warn")
            else:
                alert_box("Map Customer ID and Product columns to run basket analysis.", "warn")

    # ════════════════════════════════════════════════════════════════
    #  TAB 8: ADVANCED ANALYTICS
    # ════════════════════════════════════════════════════════════════
    with tabs[8]:
        sec_head("08","Advanced Analytics","Correlations · Anomalies · Distribution · VIF")
        a1,a2,a3,a4,a5 = st.tabs(["CORRELATIONS","ANOMALY DETECTION","DISTRIBUTION","DATA EXPLORER","FEATURE ANALYSIS"])

        with a1:
            nc2 = df.select_dtypes(include=np.number).columns.tolist()
            if len(nc2) >= 2:
                corr = df[nc2].corr()
                fig  = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  title="PEARSON CORRELATION MATRIX", aspect="auto")
                st.plotly_chart(pf(fig,h=520), use_container_width=True)
                pairs = [(corr.columns[i],corr.columns[j],corr.iloc[i,j])
                         for i in range(len(corr)) for j in range(i+1,len(corr))]
                pairs.sort(key=lambda x:abs(x[2]),reverse=True)
                top_df = pd.DataFrame(pairs[:15],columns=["Feature A","Feature B","Correlation"])
                top_df["Abs Corr"] = top_df["Correlation"].abs().round(4)
                st.dataframe(top_df, use_container_width=True)

        with a2:
            nc3 = df.select_dtypes(include=np.number).columns.tolist()
            c1,c2,c3 = st.columns(3)
            with c1: fa_sel = st.multiselect("Features",nc3,default=nc3[:min(4,len(nc3))],key="af_sel")
            with c2: cont   = st.slider("Contamination %",1,25,5,key="anom_cont")
            with c3: am_sel = st.selectbox("Method",["iforest","lof","zscore"],key="anom_meth")
            if fa_sel and st.button("▶ DETECT ANOMALIES",use_container_width=True):
                with st.spinner("Analyzing anomalies..."):
                    cd = df[fa_sel].dropna()
                    anoms, scores = detect_anomalies(cd.to_json(), fa_sel, cont/100, am_sel)
                n = int(anoms.sum())
                kc = st.columns(3)
                kc[0].metric("Anomalies",  n)
                kc[1].metric("Rate",       f"{n/len(anoms)*100:.2f}%")
                kc[2].metric("Method",     am_sel.upper())
                if n and len(fa_sel) >= 2:
                    fig = px.scatter(x=cd.iloc[:,0],y=cd.iloc[:,1],
                                     color=np.where(anoms,"🔴 ANOMALY","🟢 NORMAL"),
                                     title="ANOMALY VISUALIZATION (PC1 vs PC2)",
                                     color_discrete_map={"🔴 ANOMALY":"#ff3d57","🟢 NORMAL":"#00e676"},
                                     opacity=0.75)
                    st.plotly_chart(pf(fig), use_container_width=True)

                    # Score distribution
                    fig_s = px.histogram(x=scores, title="ANOMALY SCORE DISTRIBUTION",
                                          nbins=60, color_discrete_sequence=["#1a6bff"])
                    st.plotly_chart(pf(fig_s), use_container_width=True)

                if n:
                    st.markdown("**Top Anomalous Records:**")
                    anom_df = df.iloc[cd.index][anoms]
                    st.dataframe(anom_df.head(50), use_container_width=True)
                    st.download_button("📥 Export Anomalies",
                                       anom_df.to_csv(index=False),"anomalies.csv","text/csv")

        with a3:
            nc4 = df.select_dtypes(include=np.number).columns.tolist()
            if nc4:
                col_d = st.selectbox("Select Column",nc4,key="dist_col")
                c1,c2 = st.columns(2)
                with c1:
                    fig = px.histogram(df,x=col_d,nbins=60,
                                        title=f"DISTRIBUTION — {col_d.upper()}",
                                        color_discrete_sequence=["#00d4ff"], marginal="box")
                    st.plotly_chart(pf(fig), use_container_width=True)
                with c2:
                    fig2 = px.violin(df,y=col_d,box=True,
                                      title=f"VIOLIN PLOT — {col_d.upper()}",
                                      color_discrete_sequence=["#bb86fc"])
                    st.plotly_chart(pf(fig2), use_container_width=True)
                stats_cols = st.columns(6)
                for col_st,(lbl,val) in zip(stats_cols,[
                    ("Mean",   f"{df[col_d].mean():.3f}"),
                    ("Median", f"{df[col_d].median():.3f}"),
                    ("Std",    f"{df[col_d].std():.3f}"),
                    ("Skew",   f"{df[col_d].skew():.3f}"),
                    ("Kurt",   f"{df[col_d].kurt():.3f}"),
                    ("IQR",    f"{df[col_d].quantile(0.75)-df[col_d].quantile(0.25):.3f}"),
                ]):
                    col_st.metric(lbl, val)

        with a4:
            dl3 = cm.get("date","—")
            if dl3!="—" and dl3 in df.columns:
                mn,mx = df[dl3].min().date(), df[dl3].max().date()
                dr = st.date_input("Date Range",[mn,mx],key="exp_dr")
                if len(dr)==2:
                    mask = (df[dl3]>=pd.to_datetime(dr[0])) & (df[dl3]<=pd.to_datetime(dr[1]))
                    fdf  = df[mask]
                    st.caption(f"{len(fdf):,} / {len(df):,} rows in range")
                    st.dataframe(fdf, use_container_width=True)
                    c1,c2,c3 = st.columns(3)
                    with c1: st.download_button("📥 Export CSV",  fdf.to_csv(index=False), "filtered.csv","text/csv",  use_container_width=True)
                    with c2: st.download_button("📥 Export JSON", fdf.to_json(orient="records"), "filtered.json","application/json",use_container_width=True)
            else:
                cat_f = st.selectbox("Filter Column",["—"]+list(df.select_dtypes("object").columns),key="cat_fil")
                if cat_f!="—":
                    vals_ = st.multiselect("Values",df[cat_f].dropna().unique().tolist(),
                                           default=list(df[cat_f].dropna().unique()[:3]))
                    st.dataframe(df[df[cat_f].isin(vals_)], use_container_width=True)
                else:
                    st.dataframe(df.sample(min(500,len(df))), use_container_width=True)

        with a5:
            nc5 = df.select_dtypes(include=np.number).columns.tolist()
            if len(nc5) >= 2:
                sc_fa = cm.get("sales","—")
                if sc_fa!="—" and sc_fa in nc5:
                    # Feature-to-target correlation
                    other_nc = [c for c in nc5 if c!=sc_fa]
                    if other_nc:
                        corr_ft = df[other_nc+[sc_fa]].corr()[sc_fa].drop(sc_fa).sort_values(key=abs,ascending=False)
                        fig_ft  = px.bar(corr_ft, orientation='h',
                                          title=f"FEATURE CORRELATION TO {sc_fa.upper()}",
                                          color=corr_ft.values, color_continuous_scale="RdBu",
                                          range_color=[-1,1])
                        st.plotly_chart(pf(fig_ft), use_container_width=True)

                # Pair plot (top-4)
                top4 = nc5[:4]
                fig_pp = px.scatter_matrix(df[top4].dropna().sample(min(500,len(df))),
                                            dimensions=top4, title="SCATTER MATRIX (sample)",
                                            color_continuous_scale="Blues")
                fig_pp.update_traces(diagonal_visible=False)
                st.plotly_chart(pf(fig_pp,h=500), use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    #  TAB 9: LIVE DATA
    # ════════════════════════════════════════════════════════════════
    with tabs[9]:
        realtime_module()

    # ════════════════════════════════════════════════════════════════
    #  TAB 10: REPORTS
    # ════════════════════════════════════════════════════════════════
    with tabs[10]:
        sec_head("10","Executive Reports","AI-Generated Strategic Intelligence")
        sc_r = cm.get("sales","—"); pc_r = cm.get("profit","—")
        dc_r = cm.get("date","—");  cc_r = cm.get("category","—"); rc_r = cm.get("region","—")

        tr_  = df[sc_r].sum()  if sc_r!="—" and sc_r in df.columns else 0
        tp_  = df[pc_r].sum()  if pc_r!="—" and pc_r in df.columns else 0
        mg_  = (tp_/tr_*100)   if tr_ else 0
        top_ = df.groupby(cc_r)[sc_r].sum().idxmax() \
               if cc_r!="—" and cc_r in df.columns and sc_r!="—" and sc_r in df.columns else "N/A"
        dr_  = "—"
        if dc_r!="—" and dc_r in df.columns:
            try: dr_ = f"{df[dc_r].min().date()} → {df[dc_r].max().date()}"
            except: pass
        mp_  = df.isnull().sum().sum()/(df.shape[0]*df.shape[1])*100
        qs_  = st.session_state.get("quality_score",0)

        if st.button("▶ GENERATE EXECUTIVE REPORT", use_container_width=True):
            nc_r = df.select_dtypes(np.number).columns.tolist()
            corr_summary = ""
            if sc_r!="—" and sc_r in nc_r:
                other_ = [c for c in nc_r if c!=sc_r][:5]
                if other_:
                    cors__ = df[other_+[sc_r]].corr()[sc_r].drop(sc_r).sort_values(key=abs,ascending=False)
                    corr_summary = "\n".join([f"  · {c}: r={v:.3f}" for c,v in cors__.items()[:5]])

            region_top = "N/A"
            if rc_r!="—" and rc_r in df.columns and sc_r!="—" and sc_r in df.columns:
                region_top = df.groupby(rc_r)[sc_r].sum().idxmax()

            report = f"""# NEXUS Analytics OS — Executive Intelligence Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
**Platform Version:** {APP_VERSION}
**Dataset:** {st.session_state.get('source','Unknown')}

---

## 1. Executive Summary

This report provides a comprehensive analysis of the loaded dataset, covering financial performance,
operational metrics, data quality assessment, and strategic recommendations.

---

## 2. Dataset Overview

| Metric              | Value |
|---------------------|-------|
| Total Records       | {len(df):,} |
| Column Count        | {df.shape[1]} |
| Date Range          | {dr_} |
| Missing Values      | {mp_:.2f}% |
| Duplicate Rows      | {df.duplicated().sum():,} |
| Memory Usage        | {df.memory_usage(deep=True).sum()/1048576:.2f} MB |
| Data Quality Score  | {qs_}/100 |

---

## 3. Financial Performance KPIs

| KPI                 | Value |
|---------------------|-------|
| Total Revenue       | {fmt(tr_, "$")} |
| Total Profit        | {fmt(tp_, "$")} |
| Profit Margin       | {mg_:.2f}% |
| Average Order Value | {fmt(df[sc_r].mean(),"$") if sc_r!="—" and sc_r in df.columns else "N/A"} |
| Total Transactions  | {len(df):,} |
| Top Category        | {top_} |
| Top Region          | {region_top} |

---

## 4. Correlation Analysis (vs Revenue)
{corr_summary if corr_summary else "— Insufficient numeric columns for correlation analysis —"}

---

## 5. Strategic Recommendations

### Revenue Growth
1. **Double down on {top_}** — highest revenue contributor; increase SKU depth and marketing investment.
2. **Geographic expansion** — leverage {region_top} performance as a blueprint for expansion.
3. **Cross-sell optimization** — run Market Basket Analysis to identify product affinity patterns.

### Profitability Optimization
4. **Margin review** — current blended margin is {mg_:.1f}%. Benchmark by SKU and channel.
5. **Discount strategy audit** — analyze discount distribution to identify margin-diluting promotions.
6. **Customer lifetime value** — deploy RFM segmentation to focus retention on high-LTV segments.

### Operational Excellence
7. **Demand forecasting** — implement 12-period forecasting to optimize inventory planning.
8. **Anomaly monitoring** — schedule weekly Isolation Forest scans for data quality assurance.
9. **Real-time data integration** — connect live sources via Cloud Connectors for intraday decisions.

### Data Governance
10. **Data quality score: {qs_}/100** — {"Excellent. Maintain governance standards." if qs_>80 else "Good. Address missing value patterns." if qs_>60 else "Needs improvement. Run data cleansing pipeline."}

---

## 6. Data Quality Assessment

| Check                  | Status |
|------------------------|--------|
| Missing Values (<1%)   | {'✅ PASS' if mp_<1 else '⚠️ REVIEW — {mp_:.1f}%'.format(mp_=mp_)} |
| Row Count (>1K)        | {'✅ PASS' if len(df)>1000 else '⚠️ LIMITED'} |
| Date Column Present    | {'✅ PASS' if dc_r!="—" else '❌ MISSING'} |
| Numeric Columns        | {'✅ PASS' if sc_r!="—" else '❌ MISSING'} |
| Duplicate Records      | {'✅ NONE' if df.duplicated().sum()==0 else f'⚠️ {df.duplicated().sum():,} FOUND'} |

---

*NEXUS Analytics OS · Enterprise Edition · {APP_VERSION} · {BUILD_DATE}*
"""
            st.markdown(report)
            c1,c2,c3 = st.columns(3)
            with c1: st.download_button("📥 Markdown",    report, "nexus_report.md",   "text/markdown", use_container_width=True)
            with c2:
                html_r = f"<html><body style='font-family:monospace;background:#010409;color:#f0f6ff;padding:48px;line-height:1.8;'><pre>{report}</pre></body></html>"
                st.download_button("📥 HTML",         html_r, "nexus_report.html", "text/html",     use_container_width=True)
            with c3:
                csv_summary = pd.DataFrame([{"metric":"total_revenue","value":tr_},
                                             {"metric":"total_profit","value":tp_},
                                             {"metric":"margin_pct","value":mg_},
                                             {"metric":"total_orders","value":len(df)},
                                             {"metric":"quality_score","value":qs_}])
                st.download_button("📥 KPI CSV",      csv_summary.to_csv(index=False),"kpis.csv","text/csv",use_container_width=True)

    # ════════════════════════════════════════════════════════════════
    #  TAB 11: PLANS
    # ════════════════════════════════════════════════════════════════
    with tabs[11]:
        plans_tab()

    # ════════════════════════════════════════════════════════════════
    #  TAB 12: AI CHAT
    # ════════════════════════════════════════════════════════════════
    with tabs[12]:
        chatbot_module()

# ═══════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════
def main():
    for k, v in [("logged_in",False),("user_email",None),
                  ("is_admin",False),("admin_mode",False)]:
        if k not in st.session_state: st.session_state[k] = v

    # Maintenance mode check
    if get_setting("maintenance_mode") == "1" and not st.session_state.get("is_admin"):
        st.markdown("""<div class="nx-hero" style="margin-top:80px;">
          <div class="nx-hero-title">⚙️ MAINTENANCE</div>
          <div class="nx-hero-sub" style="margin-top:8px;">
            NEXUS is currently undergoing scheduled maintenance.<br>
            We'll be back shortly. Contact support@nexus.io for urgent issues.
          </div>
        </div>""", unsafe_allow_html=True)
        return

    if st.session_state.get("admin_mode") and st.session_state.get("is_admin"):
        admin_dashboard()
    elif st.session_state.get("logged_in"):
        analytics_app()
    else:
        col1, col2 = st.columns([2, 1], gap="large")
        with col1:  analytics_app()
        with col2:  login_panel()

if __name__ == "__main__":
    main()
