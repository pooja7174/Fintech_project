"""
LendSmart AI - Financial EDA & Credit Scoring Engine
Day 1: Know Your Customer Before You Lend
Team Project - Fintech AI Platform
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime
from PIL import Image

# ---- Module Imports ----
from modules.model_engine import load_model_and_data
from modules.kyc_engine import run_kyc_verification
from modules.credit_profiler import run_credit_profiling
from modules.batch_scorer import run_batch_scoring
from modules.analytics import run_analytics
from modules.report_generator import generate_pdf_report
from modules.risk_compliance import run_risk_dashboard
from modules.fraud_detection import run_fraud_detection
from modules.portfolio import run_portfolio

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="FinGuard Analytics & Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= CUSTOM CSS =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=JetBrains+Mono:wght@400;700&display=swap');

* { font-family: 'Outfit', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 40%, #0f2040 100%);
    color: #e0e8f0;
}

h1, h2, h3, h4, h5, h6 { color: #e0e8f0 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050c17 0%, #0a1628 100%);
    border-right: 1px solid rgba(0, 212, 255, 0.15);
}

[data-testid="stSidebar"] .stRadio label {
    color: #94a3b8 !important;
    font-size: 14px;
    padding: 4px 0;
}

.stButton > button {
    background: linear-gradient(135deg, #00d4ff, #0077ff);
    color: #000;
    border-radius: 8px;
    border: none;
    font-weight: 700;
    font-size: 14px;
    letter-spacing: 0.5px;
    padding: 10px 24px;
    transition: all 0.3s ease;
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.4);
}

.stDownloadButton > button {
    background: linear-gradient(135deg, #ff4757, #c0392b);
    color: white;
    border-radius: 8px;
    border: none;
    font-weight: 700;
}

[data-testid="metric-container"] {
    background: rgba(0, 212, 255, 0.06);
    border: 1px solid rgba(0, 212, 255, 0.2);
    border-radius: 12px;
    padding: 16px;
}

[data-testid="metric-container"] label {
    color: #64748b !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #00d4ff !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

.stDataFrame {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 10px;
}

.stAlert {
    border-radius: 10px;
}

div[data-testid="stForm"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 14px;
    padding: 20px;
}

.stSelectbox > div > div,
.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.06) !important;
    color: #e0e8f0 !important;
    border: 1px solid rgba(0,212,255,0.2) !important;
    border-radius: 8px !important;
}

.stTabs [data-baseweb="tab"] {
    color: #64748b;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
}

/* Header banner */
.main-header {
    background: linear-gradient(135deg, rgba(0,212,255,0.1), rgba(0,119,255,0.1));
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(0,212,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}

.badge {
    display: inline-block;
    background: rgba(0,212,255,0.15);
    border: 1px solid rgba(0,212,255,0.3);
    color: #00d4ff;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 12px;
}

.role-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(0,212,255,0.15);
    border-radius: 12px;
    padding: 16px;
    margin: 8px 0;
}

.team-tag {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 3px 10px;
    border-radius: 12px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

# ================= LOAD MODEL =================
model, encoder, accuracy, dataset = load_model_and_data()

# ================= HEADER =================
st.markdown("""
<div class="main-header">
    <div class="badge"></div>
    <h1 style='font-size:42px; font-weight:900; margin:0; 
               background: linear-gradient(135deg, #00d4ff, #0077ff);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        💳 FinGuard Analytics & Intelligence
    </h1>
    <p style='color:#94a3b8; font-size:16px; margin:8px 0 0 0; font-weight:300;'>
        AI-Powered Credit Risk & KYC Platform &nbsp;·&nbsp; Know Your Customer Before You Lend
    </p>
</div>
""", unsafe_allow_html=True)

# ================= SIDEBAR =================
st.sidebar.markdown("""
<div style='text-align:center; padding: 16px 0 24px 0;'>
    <div style='font-size:28px;'>💳</div>
    <div style='font-size:13px; font-weight:700; color:#00d4ff; letter-spacing:2px;'>FinGuard Analytics & Intelligence</div>
    <div style='font-size:10px; color:#475569; margin-top:4px;'>Credit Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("**📌 Navigation**")
page = st.sidebar.radio(
    "",
    [
        "🏠 Dashboard",
        "📊 Financial Analytics",
        "🛡️ Risk & Compliance",
        "🔍 Fraud Detection",
        "🪪 KYC Verification",
        "🎯 Credit Profiling",
        "📈 Batch Scoring",
        "📋 Portfolio Reports",
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='background:rgba(0,212,255,0.08); border:1px solid rgba(0,212,255,0.2);
            border-radius:10px; padding:12px; font-size:12px;'>
    <div style='color:#00d4ff; font-weight:700; margin-bottom:8px;'>🤖 Model Status</div>
    <div style='color:#94a3b8;'>Algorithm: Random Forest</div>
    <div style='color:#94a3b8;'>Accuracy: <span style='color:#4ade80; font-weight:700;'>{accuracy*100:.1f}%</span></div>
    <div style='color:#94a3b8;'>Dataset: {len(dataset):,} records</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='font-size:11px; color:#334155; text-align:center;'>
    <div style='color:#475569; font-weight:600; margin-bottom:6px;'>TEAM ROLES</div>
    <div>🏗️ ML Architect — Module A</div>
    <div>⚖️ Risk Analyst — Module D</div>
    <div>🔬 Data Scientist — Module B+E</div>
    <div>📊 Business Analyst — Module C</div>
</div>
""", unsafe_allow_html=True)

# ================= PAGE ROUTING =================

if page == "🏠 Dashboard":
    applications = len(dataset)
    approval_rate = round((dataset['default'].value_counts(normalize=True).get(0, 0)) * 100, 2)
    default_risk = round((dataset['default'].value_counts(normalize=True).get(1, 0)) * 100, 2)
    avg_score = int(850 - (default_risk * 5))

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📁 Total Applications", f"{applications:,}")
    col2.metric("✅ Approval Rate", f"{approval_rate}%")
    col3.metric("⚠️ Default Risk", f"{default_risk}%")
    col4.metric("💯 Avg Credit Score", avg_score)

    st.markdown("---")

    chart1, chart2 = st.columns(2)
    with chart1:
        fig = px.pie(
            names=["Approved", "Defaulted"],
            values=[approval_rate, default_risk],
            hole=0.55,
            title="📊 Loan Decision Distribution",
            color_discrete_sequence=["#00d4ff", "#ff4757"]
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8'
        )
          
        st.plotly_chart(fig, use_container_width=True)

    with chart2:
        fig2 = px.histogram(
            dataset, x="income", nbins=40,
            title="💰 Income Distribution of Applicants",
            color_discrete_sequence=["#0077ff"]
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Team module overview
   # st.markdown("### 🧩 Module Overview — Day 1 Deliverables")
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        st.markdown("""<div class='role-card'>
            <div style='color:#00d4ff; font-weight:700; font-size:13px;'>🏗️ ML Architect</div>
            <div style='color:#64748b; font-size:11px; margin-top:4px;'>Module A + Integration</div>
            <div style='color:#94a3b8; font-size:12px; margin-top:8px;'>Pipeline design, GitHub repo, end-to-end integration</div>
        </div>""", unsafe_allow_html=True)
    with tc2:
        st.markdown("""<div class='role-card'>
            <div style='color:#f59e0b; font-weight:700; font-size:13px;'>⚖️ Risk Analyst</div>
            <div style='color:#64748b; font-size:11px; margin-top:4px;'>Module D + Ethics</div>
            <div style='color:#94a3b8; font-size:12px; margin-top:8px;'>VaR dashboard, AML engine, compliance audit</div>
        </div>""", unsafe_allow_html=True)
    with tc3:
        st.markdown("""<div class='role-card'>
            <div style='color:#4ade80; font-weight:700; font-size:13px;'>🔬 Data Scientist</div>
            <div style='color:#64748b; font-size:11px; margin-top:4px;'>Module B + E</div>
            <div style='color:#94a3b8; font-size:12px; margin-top:8px;'>Fraud detection, segmentation, LSTM</div>
        </div>""", unsafe_allow_html=True)
    with tc4:
        st.markdown("""<div class='role-card'>
            <div style='color:#f472b6; font-weight:700; font-size:13px;'>📊 Biz Analyst</div>
            <div style='color:#64748b; font-size:11px; margin-top:4px;'>Module C + Pitch</div>
            <div style='color:#94a3b8; font-size:12px; margin-top:8px;'>Loan default pipeline, EL, pitch deck</div>
        </div>""", unsafe_allow_html=True)

elif page == "📊 Financial Analytics":
    run_analytics(dataset)

elif page == "🛡️ Risk & Compliance":
    run_risk_dashboard(dataset)

elif page == "🔍 Fraud Detection":
    run_fraud_detection(dataset)

elif page == "🪪 KYC Verification":
    run_kyc_verification()

elif page == "🎯 Credit Profiling":
    run_credit_profiling(model, encoder, accuracy)

elif page == "📈 Batch Scoring":
    run_batch_scoring(model, encoder)

elif page == "📋 Portfolio Reports":
    run_portfolio(dataset, accuracy)

# ================= FOOTER =================
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#334155; font-size:12px; padding: 12px;'>
    <span style='color:#475569;'>💳 FinGuard Analytics & Intelligence/span> &nbsp;·&nbsp; 
     &nbsp;·&nbsp; 
    <span style='color:#00d4ff;'>Financial EDA, Credit Scoring Engine and Fraud Detection</span>
</div>
""", unsafe_allow_html=True)
