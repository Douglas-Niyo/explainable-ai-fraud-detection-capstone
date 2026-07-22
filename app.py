from __future__ import annotations

import pathlib

import pandas as pd
import streamlit as st

from src.modeling import FEATURES, explain_transaction, train_fraud_model, validate_transaction_dataframe
from src.sample_data import build_sample_transactions

st.set_page_config(page_title="Explainable Fraud Detection Prototype", page_icon="🔐", layout="wide")

DATA_PATH = pathlib.Path("data/sample_transactions.csv")

st.title("🔐 Explainable AI Fraud Detection Prototype")
st.caption("MSIT 5910 initial system demo — financial security, explainability, and human review")

with st.sidebar:
    st.header("Demo Controls")
    uploaded_file = st.file_uploader("Optional: upload a CSV", type=["csv"])
    model_choice = st.selectbox("Model", ["Random Forest", "Logistic Regression"])
    st.markdown("**Core modules shown:**")
    st.markdown("1. Data intake and validation")
    st.markdown("2. Model scoring and explanations")
    st.markdown("3. Dashboard reporting")

@st.cache_data
def load_data(uploaded):
    if uploaded is not None:
        return pd.read_csv(uploaded)
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH)
    return build_sample_transactions()


df = load_data(uploaded_file)

try:
    validate_transaction_dataframe(df)
except ValueError as error:
    st.error(str(error))
    st.stop()

st.subheader("1. Data Intake and Quality Check")
col1, col2, col3 = st.columns(3)
col1.metric("Transactions", f"{len(df):,}")
col2.metric("Fraud cases", f"{int(df['is_fraud'].sum()):,}")
fraud_rate = df["is_fraud"].mean() * 100
col3.metric("Fraud rate", f"{fraud_rate:.2f}%")

st.write("Sample of incoming transaction data:")
st.dataframe(df.head(10), use_container_width=True)

st.subheader("2. Model Training, Scoring, and Evaluation")
result = train_fraud_model(df, model_choice)
left, right = st.columns(2)
with left:
    st.write("Model performance by class:")
    st.dataframe(result.metrics, use_container_width=True)
with right:
    st.write("Confusion matrix:")
    st.dataframe(result.confusion, use_container_width=True)

st.subheader("3. Risk Queue and Plain-Language Explanations")
st.write("Top transactions sorted by model risk score:")
st.dataframe(result.scored_test_data.head(15), use_container_width=True)

selected_index = st.selectbox(
    "Pick one transaction from the risk queue to explain",
    result.scored_test_data.head(15).index.astype(str).tolist(),
)
selected_row = result.scored_test_data.loc[int(selected_index)]

score_col, flag_col = st.columns(2)
score_col.metric("Selected transaction risk score", f"{selected_row['risk_score']:.3f}")
flag_col.metric("Model flag", "Review" if selected_row["predicted_flag"] == 1 else "Monitor")

st.write("Why the system flagged or monitored this transaction:")
st.dataframe(explain_transaction(selected_row), use_container_width=True)

st.info(
    "This prototype is a decision-support tool. It does not block transactions or accuse customers of fraud. "
    "A human analyst should review the alert and supporting context before action is taken."
)
