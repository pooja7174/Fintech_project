import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

import streamlit as st


@st.cache_resource
def load_model_and_data():

    # ================= LOAD DATA =================
    df = pd.read_csv(r"C:\Users\DELL\OneDrive\Pictures\Desktop\mba\fintech\MODULE4\LendSmartAI\lendsmart_ai\credit_data.csv")

    # ================= HANDLE CATEGORICAL DATA =================

    encoder = {}

    categorical_cols = [
        "loan_grade"
    ]

    for col in categorical_cols:

        le = LabelEncoder()

        df[col] = le.fit_transform(df[col])

        encoder[col] = le

    # ================= FEATURES & TARGET =================

    X = df.drop("default", axis=1)

    y = df["default"]

    # ================= TRAIN TEST SPLIT =================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # ================= MODEL =================

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    # ================= ACCURACY =================

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)

    return model, encoder, accuracy, df