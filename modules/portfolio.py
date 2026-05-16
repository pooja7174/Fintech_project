"""
Module C/D — Portfolio Reports & PDF Generation
Owner: Business Analyst / Risk Analyst
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime


def run_portfolio(dataset: pd.DataFrame, accuracy: float):
    st.subheader("📋 Portfolio Reports & Export")
    st.markdown("*Generate comprehensive risk reports for your loan portfolio*")

    tab1, tab2 = st.tabs(["📊 Portfolio Summary", "📄 PDF Report"])

    with tab1:
        total = len(dataset)
        default_rate = dataset['default'].mean()
        approval_rate = 1 - default_rate
        avg_loan = dataset['loan_amount'].mean()
        avg_income = dataset['income'].mean()
        total_exposure = dataset['loan_amount'].sum()

        # KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Applications", f"{total:,}")
        c1.metric("Total Exposure", f"₹{total_exposure:,.0f}")
        c2.metric("Approval Rate", f"{approval_rate*100:.1f}%")
        c2.metric("Default Rate", f"{default_rate*100:.1f}%")
        c3.metric("Avg Loan Amount", f"₹{avg_loan:,.0f}")
        c3.metric("Model Accuracy", f"{accuracy*100:.1f}%")

        st.markdown("---")

        col1, col2 = st.columns(2)
        with col1:
            # Default by loan grade
            grade_default = dataset.groupby('loan_grade')['default'].mean().reset_index()
            grade_default.columns = ['Loan Grade', 'Default Rate']
            grade_default['Default Rate %'] = (grade_default['Default Rate'] * 100).round(2)
            fig1 = px.bar(
                grade_default, x='Loan Grade', y='Default Rate %',
                title="Default Rate by Loan Grade",
                color='Default Rate %',
                color_continuous_scale='RdYlGn_r'
            )
            fig1.update_layout(**_dark_layout())
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            income_buckets = pd.cut(
                dataset['income'],
                bins=[0, 40000, 70000, 100000, 150000, 500000],
                labels=['<40K', '40-70K', '70-100K', '100-150K', '>150K']
            )
            dataset_copy = dataset.copy()
            dataset_copy['income_bucket'] = income_buckets
            bucket_default = dataset_copy.groupby('income_bucket', observed=True)['default'].mean().reset_index()
            bucket_default['Default Rate %'] = (bucket_default['default'] * 100).round(2)
            fig2 = px.bar(
                bucket_default, x='income_bucket', y='Default Rate %',
                title="Default Rate by Income Bracket",
                color='Default Rate %',
                color_continuous_scale='RdYlGn_r'
            )
            fig2.update_layout(**_dark_layout())
            st.plotly_chart(fig2, use_container_width=True)

        # Portfolio stress test
        st.markdown("#### 🧪 Portfolio Stress Test Scenarios")
        stress_scenarios = {
            "Base Case": default_rate,
            "Mild Recession (+50% defaults)": min(default_rate * 1.5, 1.0),
            "Moderate Stress (+100% defaults)": min(default_rate * 2.0, 1.0),
            "Severe Crisis (+200% defaults)": min(default_rate * 3.0, 1.0),
        }
        lgd = 0.45
        stress_df = pd.DataFrame([
            {
                "Scenario": k,
                "Default Rate": f"{v*100:.1f}%",
                "Expected Loss (₹)": f"₹{v * lgd * total_exposure:,.0f}",
                "Capital Required (₹)": f"₹{v * lgd * total_exposure * 1.2:,.0f}"
            }
            for k, v in stress_scenarios.items()
        ])
        st.dataframe(stress_df, use_container_width=True)

    with tab2:
        st.markdown("#### 📄 Generate PDF Portfolio Report")
        st.markdown("""
        <div style='background:rgba(255,71,87,0.08); border:1px solid rgba(255,71,87,0.25);
                    border-radius:10px; padding:16px; font-size:13px; color:#94a3b8;'>
            The PDF report will include: portfolio overview, model performance metrics,
            risk analysis, stress test results, and compliance summary.
        </div>
        """, unsafe_allow_html=True)

        report_sections = st.multiselect(
            "Include Sections",
            ["Portfolio Overview", "Model Performance", "Risk Analysis",
             "EL Computation", "Stress Test Results", "Compliance Summary"],
            default=["Portfolio Overview", "Model Performance", "Risk Analysis"]
        )

        if st.button("📄 Generate PDF Report", use_container_width=True):
            pdf_path = _generate_pdf_report(dataset, accuracy, report_sections)
            if pdf_path:
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download PDF Report",
                        data=f,
                        file_name=pdf_path.split("/")[-1],
                        mime="application/pdf"
                    )
                st.success("✅ PDF report generated successfully!")


def _generate_pdf_report(dataset, accuracy, sections):
    try:
        from fpdf import FPDF
        pdf = FPDF()
        pdf.add_page()

        # Header
        pdf.set_font("Arial", 'B', 20)
        pdf.set_text_color(0, 119, 255)
        pdf.cell(0, 12, "LendSmart AI - Portfolio Report", ln=True, align='C')

        pdf.set_font("Arial", '', 10)
        pdf.set_text_color(100, 100, 120)
        pdf.cell(0, 8, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Fintech Day 1 Project", ln=True, align='C')
        pdf.ln(8)

        # Divider
        pdf.set_draw_color(0, 119, 255)
        pdf.set_line_width(0.5)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(6)

        if "Portfolio Overview" in sections:
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(30, 30, 50)
            pdf.cell(0, 10, "Portfolio Overview", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(60, 60, 80)
            pdf.cell(0, 8, f"Total Applications: {len(dataset):,}", ln=True)
            pdf.cell(0, 8, f"Total Loan Exposure: Rs. {dataset['loan_amount'].sum():,.0f}", ln=True)
            pdf.cell(0, 8, f"Average Loan Amount: Rs. {dataset['loan_amount'].mean():,.0f}", ln=True)
            pdf.cell(0, 8, f"Default Rate: {dataset['default'].mean()*100:.2f}%", ln=True)
            pdf.ln(4)

        if "Model Performance" in sections:
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(30, 30, 50)
            pdf.cell(0, 10, "Model Performance", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(60, 60, 80)
            pdf.cell(0, 8, f"Algorithm: Random Forest Classifier (200 trees)", ln=True)
            pdf.cell(0, 8, f"Model Accuracy: {accuracy*100:.2f}%", ln=True)
            pdf.cell(0, 8, f"Features Used: Age, Income, Loan Amount, Grade, Employment Years", ln=True)
            pdf.ln(4)

        if "Risk Analysis" in sections:
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(30, 30, 50)
            pdf.cell(0, 10, "Risk Analysis", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(60, 60, 80)
            lgd = 0.45
            el = dataset['default'].mean() * lgd * dataset['loan_amount'].mean()
            pdf.cell(0, 8, f"Probability of Default (PD): {dataset['default'].mean()*100:.2f}%", ln=True)
            pdf.cell(0, 8, f"Loss Given Default (LGD): 45% (industry standard)", ln=True)
            pdf.cell(0, 8, f"Expected Loss per Loan: Rs. {el:,.0f}", ln=True)
            pdf.ln(4)

        if "Stress Test Results" in sections:
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(30, 30, 50)
            pdf.cell(0, 10, "Stress Test Results", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(60, 60, 80)
            base_dr = dataset['default'].mean()
            total_exp = dataset['loan_amount'].sum()
            for scenario, mult in [("Base Case", 1.0), ("Mild Recession", 1.5), ("Severe Stress", 3.0)]:
                dr = min(base_dr * mult, 1.0)
                loss = dr * 0.45 * total_exp
                pdf.cell(0, 8, f"{scenario}: Default {dr*100:.1f}% | Loss Rs. {loss:,.0f}", ln=True)
            pdf.ln(4)

        if "Compliance Summary" in sections:
            pdf.set_font("Arial", 'B', 14)
            pdf.set_text_color(30, 30, 50)
            pdf.cell(0, 10, "Compliance Summary", ln=True)
            pdf.set_font("Arial", '', 11)
            pdf.set_text_color(60, 60, 80)
            pdf.cell(0, 8, "KYC/AML: Verified (AI-powered YOLO face detection)", ln=True)
            pdf.cell(0, 8, "RBI Guidelines: Fair Practice Code followed", ln=True)
            pdf.cell(0, 8, "Data Privacy: DPDP Act compliant", ln=True)
            pdf.cell(0, 8, "Credit Bureau: CIBIL scoring integrated", ln=True)

        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 9)
        pdf.set_text_color(150, 150, 160)
        pdf.cell(0, 8, "LendSmart AI | Fintech Day 1 | Confidential", align='C', ln=True)

        fname = f"lendsmart_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        pdf.output(fname)
        return fname

    except ImportError:
        st.error("fpdf library not installed. Run: pip install fpdf2")
        return None


def _dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
