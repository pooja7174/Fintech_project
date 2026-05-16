"""
Module B — Fraud Detection Engine
Owner: Data Scientist
Responsibilities: Fraud detection, anomaly scoring, customer segmentation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def run_fraud_detection(dataset: pd.DataFrame):
    st.subheader("🔍 Fraud Detection & Anomaly Engine")
    st.markdown("*Module B — Isolation Forest Anomaly Detection + Customer Segmentation*")

    tab1, tab2 = st.tabs(["🚨 Anomaly Detection", "👥 Customer Segmentation"])

    with tab1:
        st.markdown("#### Isolation Forest — Fraud Anomaly Scoring")
        contamination = st.slider(
            "Expected Fraud Rate (%)", 1, 20, 5, 1,
            help="Set the expected proportion of fraud cases in your portfolio"
        ) / 100

        feature_cols = [c for c in ['income', 'loan_amount', 'age', 'employment_years']
                        if c in dataset.columns]

        X = dataset[feature_cols].copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        preds = iso.fit_predict(X_scaled)
        scores = iso.score_samples(X_scaled)

        fraud_df = dataset.copy()
        fraud_df['fraud_flag'] = np.where(preds == -1, 'Suspicious 🚨', 'Normal ✅')
        fraud_df['anomaly_score'] = -scores  # higher = more anomalous

        suspicious = (preds == -1).sum()
        normal = (preds == 1).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("🚨 Suspicious Records", f"{suspicious:,}")
        c2.metric("✅ Normal Records", f"{normal:,}")
        c3.metric("Fraud Rate Detected", f"{suspicious/len(dataset)*100:.2f}%")

        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(
                fraud_df, x="income", y="loan_amount",
                color="fraud_flag",
                color_discrete_map={
                    'Suspicious 🚨': "#ff4757",
                    'Normal ✅': "#00d4ff"
                },
                title="Income vs Loan: Fraud Anomaly Map",
                opacity=0.6
            )
            fig.update_layout(**_dark_layout())
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(
                fraud_df, x="anomaly_score",
                color="fraud_flag",
                color_discrete_map={
                    'Suspicious 🚨': "#ff4757",
                    'Normal ✅': "#00d4ff"
                },
                title="Anomaly Score Distribution",
                nbins=50, opacity=0.7
            )
            fig2.update_layout(**_dark_layout())
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### 🚨 Top Suspicious Records")
        top_fraud = fraud_df[fraud_df['fraud_flag'] == 'Suspicious 🚨'].nlargest(10, 'anomaly_score')
        st.dataframe(top_fraud[feature_cols + ['anomaly_score', 'fraud_flag']].reset_index(drop=True),
                     use_container_width=True)

    with tab2:
        st.markdown("#### 👥 K-Means Customer Segmentation")
        n_clusters = st.slider("Number of Customer Segments", 2, 6, 3)

        seg_features = [c for c in ['income', 'loan_amount', 'age'] if c in dataset.columns]
        X_seg = dataset[seg_features].copy()
        X_scaled_seg = StandardScaler().fit_transform(X_seg)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        dataset_seg = dataset.copy()
        dataset_seg['segment'] = kmeans.fit_predict(X_scaled_seg).astype(str)

        segment_labels = {str(i): f"Segment {i+1}" for i in range(n_clusters)}
        dataset_seg['segment_label'] = dataset_seg['segment'].map(segment_labels)

        fig = px.scatter(
            dataset_seg, x="income", y="loan_amount",
            color="segment_label",
            title=f"Customer Segments (K={n_clusters})",
            color_discrete_sequence=px.colors.qualitative.Set2,
            opacity=0.6
        )
        fig.update_layout(**_dark_layout())
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### 📊 Segment Profiles")
        seg_summary = dataset_seg.groupby('segment_label')[seg_features + ['default']].mean().round(2)
        seg_summary['default_rate_%'] = (seg_summary['default'] * 100).round(1)
        st.dataframe(seg_summary.drop('default', axis=1), use_container_width=True)


def _dark_layout():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='#94a3b8',
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
    )
