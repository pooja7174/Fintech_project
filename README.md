# 💳 LendSmart AI — Fintech Day 1 Project

> **Financial EDA & Credit Scoring Engine | Know Your Customer Before You Lend**

---

## 🚀 Quick Start

```bash
# 1. Clone / download the project
cd lendsmart_ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
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

## ⚡ Day 1 Deliverables Checklist

- [x] Streamlit app running with modular structure
- [x] Random Forest credit scoring model (>80% accuracy)
- [x] EDA dashboard with 6+ interactive charts
- [x] VaR & Expected Loss computation
- [x] AML risk scoring engine
- [x] Isolation Forest fraud detection
- [x] K-Means customer segmentation
- [x] KYC verification flow (YOLO + fallback)
- [x] Individual credit profiling with gauge chart
- [x] Batch CSV scoring pipeline
- [x] PDF portfolio report generator
- [x] Regulatory compliance checklist

---

*LendSmart AI | Fintech Day 1 | Team Project*
