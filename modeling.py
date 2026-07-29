"""Modeling helpers for the Streamlit fraud-detection prototype."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "amount",
    "hour",
    "merchant_risk",
    "distance_from_home",
    "previous_txn_count_24h",
    "foreign_transaction",
    "card_present",
]
TARGET = "is_fraud"


@dataclass
class ModelResult:
    model_name: str
    model: object
    metrics: pd.DataFrame
    confusion: pd.DataFrame
    scored_test_data: pd.DataFrame


def validate_transaction_dataframe(df: pd.DataFrame) -> None:
    """Validate that incoming transaction data can be scored by the model."""
    required_columns = set(FEATURES + [TARGET])
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    if df.empty:
        raise ValueError("Transaction dataset cannot be empty.")
    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("is_fraud must contain only 0 or 1 labels.")


def train_fraud_model(df: pd.DataFrame, model_name: str = "Random Forest") -> ModelResult:
    """Train a fraud classifier and return metrics plus a ranked risk queue."""
    validate_transaction_dataframe(df)
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=7, stratify=y
    )

    if model_name == "Logistic Regression":
        model = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=1000, class_weight="balanced", random_state=7),
                ),
            ]
        )
    else:
        model = RandomForestClassifier(
            n_estimators=120,
            max_depth=7,
            class_weight="balanced_subsample",
            random_state=7,
        )
        model_name = "Random Forest"

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    metrics = pd.DataFrame(
        {
            "class": ["Legitimate", "Suspicious/Fraud"],
            "precision": precision.round(3),
            "recall": recall.round(3),
            "f1_score": f1.round(3),
            "support": support,
        }
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    confusion = pd.DataFrame(
        cm,
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"],
    )
    scored = X_test.copy()
    scored["actual_is_fraud"] = y_test.values
    scored["risk_score"] = y_prob.round(3)
    scored["predicted_flag"] = y_pred
    scored = scored.sort_values("risk_score", ascending=False)

    return ModelResult(
        model_name=model_name,
        model=model,
        metrics=metrics,
        confusion=confusion,
        scored_test_data=scored,
    )


def explain_transaction(row: pd.Series) -> pd.DataFrame:
    """Return plain-language reason codes for the selected transaction."""
    reasons = []
    if row["amount"] >= 140:
        reasons.append(("High amount", "Transaction amount is unusually high for this sample."))
    if row["merchant_risk"] >= 0.45:
        reasons.append(("Higher merchant risk", "Merchant risk score is above the normal range."))
    if row["distance_from_home"] >= 40:
        reasons.append(("Unusual distance", "Transaction is far from the typical home area."))
    if row["previous_txn_count_24h"] >= 5:
        reasons.append(("Rapid transaction activity", "Several transactions occurred within 24 hours."))
    if row["foreign_transaction"] == 1:
        reasons.append(("Foreign transaction", "Transaction occurred outside the normal domestic context."))
    if row["card_present"] == 0:
        reasons.append(("Card not present", "Card-not-present activity can carry higher fraud risk."))
    if row["hour"] <= 5 or row["hour"] >= 23:
        reasons.append(("Unusual hour", "Transaction occurred late night or early morning."))
    if not reasons:
        reasons.append(("No major risk driver", "This transaction has no single obvious high-risk pattern."))
    return pd.DataFrame(reasons, columns=["Reason", "Plain-language explanation"])
