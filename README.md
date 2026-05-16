## FinGuard Analytics & Intelligence
💳 AI-Powered Credit Risk, KYC & Fraud Detection Platform

FinGuard Analytics & Intelligence is an advanced FinTech AI platform built using Python, Streamlit, Machine Learning, and Financial Analytics techniques.
The platform helps financial institutions analyze customer risk, detect fraud, perform KYC verification, and automate credit scoring processes.

This project was developed as part of a FinTech & Data Science team project focused on:

Credit Risk Analysis
Financial EDA
Fraud Detection
AI-Based Credit Scoring
Risk & Compliance Monitoring
Portfolio Analytics

## 🚀 Features

📊 Financial Analytics Dashboard
Loan approval insights
Income distribution analysis
Credit risk visualization
Interactive charts using Plotly

🎯 AI Credit Scoring Engine
Random Forest Machine Learning model
Predicts customer default risk
Automated credit profiling

🛡️ Risk & Compliance Dashboard
Financial risk monitoring
Compliance analytics
Risk distribution analysis

🔍 Fraud Detection System
Detect suspicious financial patterns
Analyze fraudulent transactions
Real-time fraud monitoring

🪪 KYC Verification
Customer verification workflow
Identity validation system
Customer onboarding checks

📈 Batch Scoring
Bulk customer credit scoring
CSV upload support
Automated prediction generation

📋 Portfolio Reports
Portfolio performance analytics
Loan distribution reporting
Risk segmentation

🧠 Machine Learning Model

The platform uses:

Random Forest Classifier
Scikit-learn
Financial Risk Prediction Techniques
Model Objectives
Predict loan defaults
Analyze borrower behavior
Improve lending decisions

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd lendsmart_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app1.py
```

---

## 👥 Team Roles & Module Ownership

| Role | Modules | Key Deliverables |
|------|---------|-----------------|
| 🏗️ **Team Lead / ML Architect** | Module A + Integration | Pipeline design, model engine, GitHub repo |
| ⚖️ **Risk & Compliance Analyst** | Module D + Ethics | VaR dashboard, AML engine, compliance checklist |
| 🔬 **Data Scientist** | Module B + E | Fraud detection, segmentation, anomaly scoring |
| 📊 **Business Analyst** | Module C + Pitch | Credit profiling, EL computation, batch scoring, PDF report |

---

## 📦 Project Structure

```
lendsmart_ai/
├── app.py                        ← Main Streamlit entry point
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
├── loan_dataset.csv              ← (Optional) Real dataset; synthetic generated if missing
└── modules/
    ├── __init__.py
    ├── model_engine.py           ← Module A: ML model training & loading
    ├── analytics.py              ← Module A: Financial EDA visualizations
    ├── risk_compliance.py        ← Module D: VaR, AML, compliance checklist
    ├── fraud_detection.py        ← Module B: Isolation Forest + K-Means segmentation
    ├── kyc_engine.py             ← KYC: YOLO face detection (+ fallback simulation)
    ├── credit_profiler.py        ← Module C: Individual credit scoring UI
    ├── batch_scorer.py           ← Module C: Bulk CSV scoring pipeline
    ├── portfolio.py              ← Module C/D: Portfolio dashboard + PDF export
    └── report_generator.py       ← Utility: PDF report wrapper
```

---

## 🧠 ML Stack

| Component | Technology |
|-----------|-----------|
| Credit Scoring | `RandomForestClassifier` (200 trees) |
| Fraud Detection | `IsolationForest` (anomaly scoring) |
| Segmentation | `KMeans` clustering |
| KYC Vision | `YOLOv8n` (face detection) |
| Data | Synthetic (5,000 records) or `loan_dataset.csv` |

---

## 📊 Key Features

- **Dashboard** — Portfolio KPIs, approval rates, income distribution
- **Financial Analytics** — EDA: scatter, histogram, correlation heatmap, box plots
- **Risk & Compliance** — VaR at configurable confidence, AML risk scoring, regulatory checklist
- **Fraud Detection** — Isolation Forest anomaly map, suspicious records table, K-Means segmentation
- **KYC Verification** — Upload ID + selfie → YOLO face detection → pass/fail decision
- **Credit Profiling** — Individual applicant form → credit score gauge + EL breakdown
- **Batch Scoring** — Upload CSV → score hundreds of applicants → download results
- **Portfolio Reports** — Stress testing, grade-wise analysis, PDF export

---

## 🔧 Using a Real Dataset

Place `loan_dataset.csv` in the project root. Required columns:

```
age, income, loan_amount, loan_grade, employment_years, default
```

Optional columns (automatically used if present):
```
credit_history_years, num_open_accounts, delinquency_count
```

---

## 🔐 FinTech Use Cases
Digital Lending Platforms
NBFC Risk Analysis
Banking Credit Assessment
Loan Approval Automation
AML & Fraud Monitoring
Customer Risk Profiling

## 📈 Future Enhancements
Deep Learning Models
Real-time Transaction Monitoring
API Integration
Cloud Deployment
Blockchain-based KYC
Advanced Fraud Detection AI
LSTM Risk Forecasting
