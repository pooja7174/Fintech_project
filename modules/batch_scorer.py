"""
Module C — Batch Credit Scoring
Owner: Business Analyst
Responsibilities: Bulk loan assessment pipeline
"""

import streamlit as st
import pandas as pd
import numpy as np


def run_batch_scoring(model, encoder):
    st.subheader("📈 Batch Credit Scoring Pipeline")
    st.markdown("*Upload a CSV file to score multiple applicants at once*")

    st.markdown("""
    <div style='background:rgba(0,212,255,0.06); border:1px solid rgba(0,212,255,0.2);
                border-radius:10px; padding:16px; margin-bottom:16px; font-size:13px;'>
        <div style='color:#00d4ff; font-weight:700; margin-bottom:8px;'>📋 Required CSV Columns</div>
        <div style='color:#94a3b8;'>
            <code style='color:#f59e0b;'>age</code> &nbsp;·&nbsp; 
            <code style='color:#f59e0b;'>income</code> &nbsp;·&nbsp; 
            <code style='color:#f59e0b;'>loan_amount</code> &nbsp;·&nbsp; 
            <code style='color:#f59e0b;'>loan_grade</code> (A/B/C/D/E) &nbsp;·&nbsp; 
            <code style='color:#f59e0b;'>employment_years</code>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample CSV download
    sample_df = pd.DataFrame({
        'age': [28, 45, 33, 52, 25],
        'income': [60000, 120000, 80000, 150000, 35000],
        'loan_amount': [15000, 40000, 20000, 55000, 8000],
        'loan_grade': ['B', 'A', 'C', 'A', 'D'],
        'employment_years': [3, 15, 6, 22, 1]
    })
    st.download_button(
        "⬇️ Download Sample CSV Template",
        sample_df.to_csv(index=False),
        file_name="sample_loan_applications.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("📤 Upload Applications CSV", type=["csv"])

    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.markdown(f"**Preview — {len(batch_df)} applications loaded**")
        st.dataframe(batch_df.head(), use_container_width=True)

        required_cols = ['age', 'income', 'loan_amount', 'loan_grade', 'employment_years']
        missing = [c for c in required_cols if c not in batch_df.columns]

        if missing:
            st.error(f"❌ Missing columns: {missing}")
            return

        if st.button("🚀 Run Batch AI Scoring", use_container_width=True):
            with st.spinner("Processing all applications..."):
                processed = batch_df.copy()

                # Encode loan grade
                try:
                    if processed['loan_grade'].dtype == object:
                        processed['loan_grade'] = encoder.transform(processed['loan_grade'])
                except Exception:
                    grade_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
                    processed['loan_grade'] = processed['loan_grade'].map(grade_map).fillna(2).astype(int)

                features = ['age', 'income', 'loan_amount', 'loan_grade', 'employment_years']

                # Add missing features if model needs more
                for col in ['credit_history_years', 'num_open_accounts', 'delinquency_count']:
                    processed[col] = 0

                try:
                    feat_cols = model.feature_names_in_
                    preds = model.predict(processed[feat_cols])
                    probs = model.predict_proba(processed[feat_cols])[:, 1]
                except Exception:
                    preds = model.predict(processed[features])
                    probs = model.predict_proba(processed[features])[:, 1]

                result_df = batch_df.copy()
                result_df['default_probability_%'] = (probs * 100).round(2)
                result_df['credit_score'] = (850 - probs * 550).astype(int)
                result_df['decision'] = np.where(preds == 0, '✅ APPROVED', '❌ DECLINED')
                result_df['risk_tier'] = pd.cut(
                    probs,
                    bins=[0, 0.1, 0.25, 0.5, 1.0],
                    labels=['🟢 Low', '🟡 Medium', '🟠 High', '🔴 Very High']
                )

            # Summary metrics
            approved = (preds == 0).sum()
            declined = (preds == 1).sum()

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Applications", len(result_df))
            c2.metric("✅ Approved", approved)
            c3.metric("❌ Declined", declined)
            c4.metric("Avg Credit Score", int(result_df['credit_score'].mean()))

            st.markdown("#### 📊 Scored Results")
            st.dataframe(result_df, use_container_width=True)

            # Risk tier breakdown
            risk_counts = result_df['risk_tier'].value_counts()
            import plotly.express as px
            fig = px.bar(
                x=risk_counts.index.astype(str),
                y=risk_counts.values,
                title="Risk Tier Distribution",
                color=risk_counts.index.astype(str),
                color_discrete_sequence=["#4ade80", "#f59e0b", "#ff8c00", "#ff4757"]
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#94a3b8',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

            csv_out = result_df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Scored Results CSV",
                csv_out,
                file_name="batch_scoring_results.csv",
                mime="text/csv"
            )
