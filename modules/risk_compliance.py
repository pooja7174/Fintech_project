"""
Module D — Risk & Compliance Dashboard
Owner: Risk & Compliance Analyst
Responsibilities: VaR dashboard, AML engine, compliance checklist, regulatory mapping
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def run_risk_dashboard(dataset: pd.DataFrame):
    st.subheader("🛡️ Risk & Compliance Dashboard")
    st.markdown("*Module D — VaR Analysis, AML Engine & Regulatory Compliance*")

    tab1, tab2, tab3 = st.tabs(["📉 VaR Dashboard", "🚨 AML Engine", "✅ Compliance Checklist"])

    # ---- VaR DASHBOARD ----
    with tab1:
        st.markdown("#### Value at Risk (VaR) Analysis")

        confidence = st.slider("Confidence Level (%)", 90, 99, 95)
        portfolio_size = st.number_input("Portfolio Size ($)", value=10_000_000, step=500_000)

        returns = np.random.normal(0.001, 0.02, 1000)
        var_pct = np.percentile(returns, 100 - confidence)
        var_dollar = abs(var_pct) * portfolio_size
        cvar = abs(np.mean(returns[returns <= var_pct])) * portfolio_size

        c1, c2, c3 = st.columns(3)
        c1.metric("VaR (1-day)", f"${var_dollar:,.0f}", f"At {confidence}% confidence")
        c2.metric("CVaR (Expected Shortfall)", f"${cvar:,.0f}")
        c3.metric("Portfolio Size", f"${portfolio_size:,.0f}")

        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=returns * portfolio_size,
            nbinsx=60,
            name="P&L Distribution",
            marker_color='#0077ff',
            opacity=0.7
        ))
        fig.add_vline(
            x=-var_dollar, line_dash="dash", line_color="#ff4757",
            annotation_text=f"VaR: ${var_dollar:,.0f}",
            annotation_font_color="#ff4757"
        )
        fig.update_layout(
            title=f"Portfolio P&L Distribution with {confidence}% VaR",
            xaxis_title="P&L ($)",
            yaxis_title="Frequency",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='#94a3b8',
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)

        # Expected Loss Computation
        st.markdown("#### 📐 Expected Loss (EL) Computation")
        default_rate = dataset['default'].mean()
        avg_loan = dataset['loan_amount'].mean()
        lgd = 0.45  # Loss Given Default (industry standard)
        ead = avg_loan
        el = default_rate * lgd * ead

        ec1, ec2, ec3, ec4 = st.columns(4)
        ec1.metric("Probability of Default (PD)", f"{default_rate*100:.2f}%")
        ec2.metric("Loss Given Default (LGD)", f"{lgd*100:.0f}%")
        ec3.metric("Exposure at Default (EAD)", f"${ead:,.0f}")
        ec4.metric("Expected Loss (EL)", f"${el:,.0f}")

    # ---- AML ENGINE ----
    with tab2:
        st.markdown("#### 🚨 Anti-Money Laundering (AML) Screening")

        col1, col2 = st.columns(2)
        with col1:
            cust_name = st.text_input("Customer Name", placeholder="e.g. John Doe")
            cust_income = st.number_input("Declared Income ($)", value=50000)
            cust_loan = st.number_input("Loan Amount Requested ($)", value=15000)
            transaction_freq = st.selectbox("Transaction Frequency", ["Normal", "High", "Very High"])

        with col2:
            country = st.selectbox("Country of Origin", [
                "India", "USA", "UAE", "China", "Nigeria", "Pakistan", "Russia", "UK", "Germany", "Brazil"
            ])
            source_of_funds = st.selectbox("Source of Funds", [
                "Salary", "Business", "Inheritance", "Investment", "Unknown", "Mixed"
            ])
            politically_exposed = st.checkbox("Politically Exposed Person (PEP)")

        if st.button("🔍 Run AML Check"):
            risk_score = 0

            # Income-to-loan ratio check
            ratio = cust_loan / max(cust_income, 1)
            if ratio > 0.8:
                risk_score += 25

            # High-risk countries
            high_risk_countries = ["Nigeria", "Pakistan", "Russia"]
            if country in high_risk_countries:
                risk_score += 30

            # Source of funds
            if source_of_funds in ["Unknown", "Mixed"]:
                risk_score += 25

            # PEP flag
            if politically_exposed:
                risk_score += 20

            # Transaction frequency
            if transaction_freq == "Very High":
                risk_score += 15
            elif transaction_freq == "High":
                risk_score += 8

            risk_score = min(risk_score, 100)

            if risk_score < 30:
                st.success(f"✅ AML Score: {risk_score}/100 — LOW RISK — Customer cleared for processing")
            elif risk_score < 60:
                st.warning(f"⚠️ AML Score: {risk_score}/100 — MEDIUM RISK — Enhanced Due Diligence required")
            else:
                st.error(f"🚨 AML Score: {risk_score}/100 — HIGH RISK — Escalate to compliance officer")

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_score,
                title={'text': "AML Risk Score", 'font': {'color': '#94a3b8'}},
                gauge={
                    'axis': {'range': [0, 100], 'tickcolor': '#94a3b8'},
                    'bar': {'color': "#ff4757" if risk_score >= 60 else "#f59e0b" if risk_score >= 30 else "#4ade80"},
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(74,222,128,0.15)'},
                        {'range': [30, 60], 'color': 'rgba(245,158,11,0.15)'},
                        {'range': [60, 100], 'color': 'rgba(255,71,87,0.15)'}
                    ]
                }
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#94a3b8', height=300)
            st.plotly_chart(fig, use_container_width=True)

    # ---- COMPLIANCE CHECKLIST ----
    with tab3:
        st.markdown("#### ✅ Regulatory Compliance Checklist")

        compliance_items = {
            "KYC/AML": [
                ("Customer identity verified (PAN/Aadhaar)", True),
                ("Source of funds documented", True),
                ("AML screening completed", True),
                ("PEP check performed", True),
                ("Beneficial ownership identified", False),
            ],
            "RBI Guidelines": [
                ("Fair Practice Code followed", True),
                ("Interest rate disclosed upfront", True),
                ("Cooling-off period offered", True),
                ("Loan agreement signed digitally", True),
                ("Repayment schedule provided", True),
            ],
            "Data Privacy": [
                ("DPDP Act compliance", True),
                ("Customer data encrypted", True),
                ("Consent recorded", True),
                ("Data retention policy enforced", False),
                ("Third-party data sharing restricted", True),
            ],
            "Credit Bureau": [
                ("CIBIL score fetched", True),
                ("Experian report checked", False),
                ("Bureau consent obtained", True),
                ("Score disclosed to applicant", True),
            ]
        }

        for category, items in compliance_items.items():
            st.markdown(f"**📌 {category}**")
            passed = sum(1 for _, v in items if v)
            total = len(items)
            st.progress(passed / total, text=f"{passed}/{total} checks passed")
            for label, status in items:
                icon = "✅" if status else "❌"
                color = "#4ade80" if status else "#ff4757"
                st.markdown(
                    f"<span style='color:{color};'>{icon}</span> &nbsp; {label}",
                    unsafe_allow_html=True
                )
            st.markdown("")
