"""Unit tests for the fraud-detection prototype."""
from __future__ import annotations

import pandas as pd
import pytest

from src.modeling import FEATURES, explain_transaction, train_fraud_model, validate_transaction_dataframe
from src.sample_data import build_sample_transactions


def test_sample_data_contains_required_columns_and_binary_target():
    """Black-box test: generated data should match the expected input contract."""
    df = build_sample_transactions(n=300, random_state=11)
    for column in FEATURES + ["is_fraud"]:
        assert column in df.columns
    assert set(df["is_fraud"].unique()).issubset({0, 1})
    assert len(df) == 300


def test_train_fraud_model_returns_metrics_and_risk_scores():
    """White-box test: model output should include metrics, confusion matrix, and ranked scores."""
    df = build_sample_transactions(n=600, random_state=42)
    result = train_fraud_model(df, model_name="Random Forest")
    assert result.model_name == "Random Forest"
    assert {"precision", "recall", "f1_score"}.issubset(result.metrics.columns)
    assert result.confusion.shape == (2, 2)
    assert result.scored_test_data["risk_score"].between(0, 1).all()
    assert result.scored_test_data["risk_score"].is_monotonic_decreasing


def test_explain_transaction_returns_expected_reason_codes():
    """Black-box test: a visibly risky transaction should produce useful reason codes."""
    row = pd.Series(
        {
            "amount": 220,
            "hour": 2,
            "merchant_risk": 0.80,
            "distance_from_home": 75,
            "previous_txn_count_24h": 7,
            "foreign_transaction": 1,
            "card_present": 0,
        }
    )
    explanation = explain_transaction(row)
    reasons = set(explanation["Reason"].tolist())
    assert "High amount" in reasons
    assert "Higher merchant risk" in reasons
    assert "Foreign transaction" in reasons
    assert len(explanation) >= 4


def test_validation_rejects_missing_required_columns():
    """Negative unit test: bad input should fail before training begins."""
    bad_df = pd.DataFrame({"amount": [10, 20], "is_fraud": [0, 1]})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_transaction_dataframe(bad_df)
