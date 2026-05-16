"""
Module A — Financial EDA & Analytics Dashboard
Owner: Team Lead / ML Architect
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def run_analytics(dataset: pd.DataFrame):
    st.subheader("📊 Financial EDA Dashboard")
    st.markdown("*Exploratory Data Analysis — Know Your Portfolio*")

    tab1, tab2, tab3 = st.tabs(["📉 Risk Analysis", "👥 Demographics", "🔗 Correlations"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.scatter(
                dataset, x="income", y="loan_amount",
                color="default",
                color_discrete_map={0: "#00d4ff", 1: "#ff4757"},
                title="Income vs Loan Amount by Default Status",
                labels={"default": "Default", "income": "Income ($)", "loan_amount": "Loan Amount ($)"},
                opacity=0.6
            )
            fig.update_layout(**_dark_layout())
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            fig2 = px.histogram(
                dataset, x="age", color="default",
                color_discrete_map={0: "#00d4ff", 1: "#ff4757"},
                barmode="overlay",
                title="Age Distribution by Default Status",
                labels={"default": "Default", "age": "Age"},
                opacity=0.7
            )
            fig2.update_layout(**_dark_layout())
            st.plotly_chart(fig2, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            if 'employment_years' in dataset.columns:
                fig3 = px.box(
                    dataset, x="default", y="employment_years",
                    color="default",
                    color_discrete_map={0: "#00d4ff", 1: "#ff4757"},
                    title="Employment Years vs Default Risk",
                    labels={"default": "Default (0=No, 1=Yes)", "employment_years": "Years Employed"}
                )
                fig3.update_layout(**_dark_layout())
                st.plotly_chart(fig3, use_container_width=True)

        with c4:
            grade_counts = dataset.groupby(['loan_grade', 'default']).size().reset_index(name='count')
            fig4 = px.bar(
                grade_counts, x="loan_grade", y="count", color="default",
                color_discrete_map={0: "#00d4ff", 1: "#ff4757"},
                title="Loan Grade vs Default Count",
                barmode="group",
                labels={"loan_grade": "Loan Grade (Encoded)", "count": "Count"}
            )
            fig4.update_layout(**_dark_layout())
            st.plotly_chart(fig4, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig = px.histogram(
                dataset, x="income", nbins=50,
                title="Income Distribution",
                color_discrete_sequence=["#0077ff"]
            )
            fig.update_layout(**_dark_layout())
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            loan_bins = pd.cut(dataset['loan_amount'], bins=5).value_counts().reset_index()
            loan_bins.columns = ['Range', 'Count']
            loan_bins['Range'] = loan_bins['Range'].astype(str)
            fig2 = px.bar(
                loan_bins, x='Range', y='Count',
                title="Loan Amount Distribution",
                color_discrete_sequence=["#00d4ff"]
            )
            fig2.update_layout(**_dark_layout())
            st.plotly_chart(fig2, use_container_width=True)

        # Summary stats table
        st.markdown("#### 📋 Dataset Summary Statistics")
        st.dataframe(
            dataset.describe().round(2).style.background_gradient(cmap='Blues'),
            use_container_width=True
        )

    with tab3:
        numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
        corr = dataset[numeric_cols].corr()
        fig = px.imshow(
            corr,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Heatmap",
            zmin=-1, zmax=1
        )
        fig.update_layout(**_dark_layout())
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 🔑 Key Insight")
        default_corr = corr['default'].drop('default').abs().sort_values(ascending=False)
        st.markdown(f"**Strongest predictor of default:** `{default_corr.index[0]}` "
                    f"(correlation: {corr['default'][default_corr.index[0]]:.3f})")


def _dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
