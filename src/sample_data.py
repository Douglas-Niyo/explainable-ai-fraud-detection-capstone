"""Create a small synthetic credit-card transaction dataset for the Unit 4 demo.
This file avoids real customer data and keeps the prototype safe to share.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_sample_transactions(n: int = 1200, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    amount = rng.gamma(shape=2.0, scale=35.0, size=n).round(2)
    hour = rng.integers(0, 24, size=n)
    merchant_risk = rng.beta(2, 7, size=n)
    distance_from_home = rng.exponential(scale=18, size=n).round(1)
    previous_txn_count_24h = rng.poisson(lam=2.0, size=n)
    foreign_transaction = rng.binomial(1, 0.08, size=n)
    card_present = rng.binomial(1, 0.72, size=n)

    # Fraud probability: intentionally simple, explainable, and safe for demo use.
    logit = (
        -4.2
        + 0.018 * amount
        + 1.8 * merchant_risk
        + 0.018 * distance_from_home
        + 0.35 * previous_txn_count_24h
        + 1.2 * foreign_transaction
        - 0.7 * card_present
        + 0.4 * ((hour <= 5) | (hour >= 23))
    )
    probability = 1 / (1 + np.exp(-logit))
    fraud = rng.binomial(1, probability)

    df = pd.DataFrame(
        {
            "transaction_id": [f"TXN-{i:05d}" for i in range(1, n + 1)],
            "amount": amount,
            "hour": hour,
            "merchant_risk": merchant_risk.round(3),
            "distance_from_home": distance_from_home,
            "previous_txn_count_24h": previous_txn_count_24h,
            "foreign_transaction": foreign_transaction,
            "card_present": card_present,
            "is_fraud": fraud,
        }
    )
    return df


if __name__ == "__main__":
    df = build_sample_transactions()
    df.to_csv("data/sample_transactions.csv", index=False)
    print("Saved data/sample_transactions.csv", df.shape)
