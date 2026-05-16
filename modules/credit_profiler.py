"""
Module C — Loan Default Pipeline & Credit Profiling
Owner: Business Analyst / Presenter
Responsibilities: Individual credit assessment, EL computation, loan decision
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go


GRADE_MAP = {
    'A — Very Low Risk': 'A',
    'B — Low Risk': 'B',
    'C — Medium Risk': 'C',
    'D — High Risk': 'D',
    'E — Very High Risk': 'E'
}


def run_credit_profiling(model, encoder, accuracy: float):
    st.subheader("🎯 AI Credit Profiling Engine")
    st.markdown("*Module C — Individual Customer Risk Assessment & Loan Decision*")

    with st.form("credit_assessment_form"):
        st.markdown("#### 👤 Customer Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Customer Name", placeholder="John Doe")
            age = st.number_input("Age", 18, 80, 32)
            employment_years = st.number_input("Employment Years", 0, 40, 5)

        with col2:
            income = st.number_input("Annual Income (₹)", 100000, 5000000, 700000, step=50000)
            loan_amount = st.number_input("Loan Amount Requested (₹)", 10000, 5000000, 250000, step=10000)
            grade_label = st.selectbox("Loan Grade / Risk Category", list(GRADE_MAP.keys()))

        with col3:
            loan_purpose = st.selectbox("Loan Purpose", [
                "Home Purchase", "Education", "Business", "Medical", "Vehicle", "Personal"
            ])
            collateral = st.selectbox("Collateral Type", [
                "None", "Property", "Vehicle", "Fixed Deposit", "Gold"
            ])
            num_dependents = st.number_input("Number of Dependents", 0, 10, 1)

        submit = st.form_submit_button("🤖 Run AI Credit Assessment", use_container_width=True)

    if submit:
        grade = GRADE_MAP[grade_label]

        # Encode grade
        try:
            grade_encoded = encoder["loan_grade"].transform([grade])[0]
        except Exception:
            grade_map_manual = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
            grade_encoded = grade_map_manual.get(grade, 2)

        input_data = pd.DataFrame({
            'age': [age],
            'income': [income],
            'loan_amount': [loan_amount],
            'loan_grade': [grade_encoded],
            'employment_years': [employment_years]
        })

        # Add extra features if model expects them
        for col in ['credit_history_years', 'num_open_accounts', 'delinquency_count']:
            if col not in input_data.columns:
                input_data[col] = [0]

        try:
            prediction = model.predict(input_data[model.feature_names_in_])[0]
            probability = model.predict_proba(input_data[model.feature_names_in_])[0][1]
        except Exception:
            prediction = model.predict(input_data[['age', 'income', 'loan_amount', 'loan_grade', 'employment_years']])[0]
            probability = model.predict_proba(input_data[['age', 'income', 'loan_amount', 'loan_grade', 'employment_years']])[0][1]

        credit_score = int(850 - probability * 550)
        dti_ratio = (loan_amount / income) * 100

        # Decision logic
        if prediction == 0 and credit_score >= 650:
            decision = "✅ APPROVED"
            decision_color = "#4ade80"
            decision_bg = "rgba(74,222,128,0.1)"
        elif prediction == 0 and credit_score >= 580:
            decision = "⚠️ CONDITIONAL APPROVAL"
            decision_color = "#f59e0b"
            decision_bg = "rgba(245,158,11,0.1)"
        else:
            decision = "❌ DECLINED"
            decision_color = "#ff4757"
            decision_bg = "rgba(255,71,87,0.1)"

        # ---- Metrics ----
        st.markdown("---")
        st.markdown(f"""
        <div style='background:{decision_bg}; border:2px solid {decision_color}; 
                    border-radius:14px; padding:20px; text-align:center; margin-bottom:20px;'>
            <div style='font-size:12px; color:#94a3b8; letter-spacing:2px; text-transform:uppercase;'>
                Loan Decision for {name or "Applicant"}
            </div>
            <div style='font-size:32px; font-weight:900; color:{decision_color}; margin:8px 0;'>
                {decision}
            </div>
            <div style='font-size:13px; color:#94a3b8;'>
                Loan Amount: ₹{loan_amount:,} &nbsp;·&nbsp; Purpose: {loan_purpose}
            </div>
        </div>
        """, unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💯 Credit Score", credit_score)
        c2.metric("⚠️ Default Probability", f"{probability*100:.1f}%")
        c3.metric("📊 DTI Ratio", f"{dti_ratio:.1f}%")
        c4.metric("🤖 Model Accuracy", f"{accuracy*100:.1f}%")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=credit_score,
                title={'text': "Credit Score", 'font': {'color': '#94a3b8', 'size': 14}},
                number={'font': {'color': '#00d4ff', 'size': 36}},
                gauge={
                    'axis': {'range': [300, 850], 'tickcolor': '#94a3b8'},
                    'bar': {'color': '#00d4ff', 'thickness': 0.25},
                    'steps': [
                        {'range': [300, 580], 'color': 'rgba(255,71,87,0.2)'},
                        {'range': [580, 670], 'color': 'rgba(245,158,11,0.2)'},
                        {'range': [670, 850], 'color': 'rgba(74,222,128,0.2)'}
                    ],
                    'threshold': {
                        'line': {'color': '#00d4ff', 'width': 3},
                        'thickness': 0.75,
                        'value': credit_score
                    }
                }
            ))
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # EL Computation
            pd_val = probability
            lgd = 0.45 if collateral == "None" else 0.25
            ead = loan_amount
            el = pd_val * lgd * ead

            st.markdown("#### 📐 Expected Loss Breakdown")
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid rgba(0,212,255,0.15);
                        border-radius:12px; padding:20px; font-size:14px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                    <span style='color:#64748b;'>Probability of Default (PD)</span>
                    <span style='color:#00d4ff; font-weight:700;'>{pd_val*100:.2f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                    <span style='color:#64748b;'>Loss Given Default (LGD)</span>
                    <span style='color:#f59e0b; font-weight:700;'>{lgd*100:.0f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                    <span style='color:#64748b;'>Exposure at Default (EAD)</span>
                    <span style='color:#94a3b8; font-weight:700;'>₹{ead:,.0f}</span>
                </div>
                <div style='border-top:1px solid rgba(0,212,255,0.2); padding-top:10px; display:flex; justify-content:space-between;'>
                    <span style='color:#e0e8f0; font-weight:700;'>Expected Loss (EL = PD × LGD × EAD)</span>
                    <span style='color:#ff4757; font-weight:900;'>₹{el:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Risk factors
            st.markdown("#### ⚡ Key Risk Factors")
            factors = {
                "Income-to-Loan Ratio": f"{(income/loan_amount):.1f}x {'✅' if income/loan_amount > 3 else '⚠️'}",
                "Employment Stability": f"{employment_years} yrs {'✅' if employment_years > 2 else '⚠️'}",
                "DTI Ratio": f"{dti_ratio:.1f}% {'✅' if dti_ratio < 40 else '❌'}",
                "Grade Category": f"{grade_label.split('—')[0].strip()} {'✅' if grade in ['A','B'] else '⚠️' if grade == 'C' else '❌'}",
                "Collateral": f"{collateral} {'✅' if collateral != 'None' else '⚠️'}",
            }
            for k, v in factors.items():
                st.markdown(f"**{k}:** {v}")
