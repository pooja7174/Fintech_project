## FinGuard Analytics & Intelligence
💳 AI-Powered Credit Risk, KYC & Fraud Detection Platform

FinGuard Analytics & Intelligence is an advanced FinTech AI platform built using Python, Streamlit, Machine Learning, and Financial Analytics techniques.
The platform helps financial institutions analyze customer risk, detect fraud, perform KYC verification, and automate credit scoring processes.

## 🏠 Main Dashboard
<img width="1907" height="902" alt="Screenshot 2026-05-14 152556" src="https://github.com/user-attachments/assets/b9277b60-3f4e-405d-8083-7e22f6503ea5" />
<img width="1918" height="817" alt="Screenshot 2026-05-14 152613" src="https://github.com/user-attachments/assets/3f991027-2e6c-4cbd-ad5f-bd8c972dd6fa" />
<img width="1918" height="898" alt="Screenshot 2026-05-14 152633" src="https://github.com/user-attachments/assets/3e90da21-79f9-4bd9-a474-ed8ecc432278" />
<img width="1918" height="883" alt="Screenshot 2026-05-14 152653" src="https://github.com/user-attachments/assets/f653ddfe-4e28-4c77-ab82-6f7588cf9a25" />
<img width="1912" height="895" alt="Screenshot 2026-05-14 152728" src="https://github.com/user-attachments/assets/2ca50336-3141-4bbc-86a2-9f8fcc123ccb" />
<img width="1918" height="906" alt="Screenshot 2026-05-14 152747" src="https://github.com/user-attachments/assets/9a363aa7-91e9-46fc-aab4-32b47cda27eb" />
<img width="1918" height="897" alt="Screenshot 2026-05-14 152803" src="https://github.com/user-attachments/assets/66c66778-b595-4937-b750-871f081d442e" />
<img width="1908" height="898" alt="Screenshot 2026-05-14 152818" src="https://github.com/user-attachments/assets/aab29b32-1c9a-4da1-8590-8a78830f0487" />
<img width="1918" height="797" alt="Screenshot 2026-05-14 152908" src="https://github.com/user-attachments/assets/41957ecf-def4-4e41-a916-f15802f7f626" />
<img width="1901" height="900" alt="Screenshot 2026-05-14 152937" src="https://github.com/user-attachments/assets/2359b5dd-b943-4acf-989a-b5c487aac076" />
<img width="1915" height="911" alt="Screenshot 2026-05-14 152953" src="https://github.com/user-attachments/assets/6e6a14b9-e317-4dc8-9050-3e335d77b4c0" />

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
