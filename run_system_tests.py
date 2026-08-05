from __future__ import annotations

import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sample_data import build_sample_transactions
from src.modeling import explain_transaction, train_fraud_model, validate_transaction_dataframe


def check(name: str, condition: bool, detail: str) -> tuple[str, str, str]:
    return (name, "PASS" if condition else "FAIL", detail)


def main() -> int:
    results = []
    df = build_sample_transactions(n=1200, random_state=42)

    # Integration: data generator -> validation -> model -> explanation
    try:
        validate_transaction_dataframe(df)
        model_result = train_fraud_model(df, "Random Forest")
        explanation = explain_transaction(model_result.scored_test_data.iloc[0])
        results.append(check("INT-01 end-to-end module flow", True, "Data, model, risk queue, and explanation connected."))
    except Exception as exc:
        results.append(check("INT-01 end-to-end module flow", False, f"Exception: {exc}"))
        model_result = None
        explanation = None

    if model_result is not None:
        results.append(check(
            "SYS-01 required outputs",
            not model_result.metrics.empty and model_result.confusion.shape == (2, 2) and len(model_result.scored_test_data) == 300,
            "Metrics, 2x2 confusion matrix, and 300 scored test records returned.",
        ))
        results.append(check(
            "REG-01 risk queue ordering",
            model_result.scored_test_data["risk_score"].is_monotonic_decreasing,
            "Risk queue remains sorted from highest to lowest score.",
        ))
        results.append(check(
            "ACC-01 analyst explanation",
            explanation is not None and len(explanation) >= 1,
            "Selected transaction returns at least one plain-language reason code.",
        ))

    start = time.perf_counter()
    perf_result = train_fraud_model(df, "Random Forest")
    elapsed = time.perf_counter() - start
    throughput = len(df) / elapsed
    results.append(check(
        "PERF-01 training and scoring latency",
        elapsed < 3.0,
        f"Completed in {elapsed:.3f} seconds; throughput {throughput:.1f} transactions/sec.",
    ))

    print("Unit 7 Integrated System Testing Log")
    print("Project: Explainable AI Fraud Detection Prototype")
    print("Test data: 1,200 synthetic transactions; 25% test split")
    print("-" * 90)
    print(f"{'Test Case':40} {'Status':8} Details")
    print("-" * 90)
    for name, status, detail in results:
        print(f"{name:40} {status:8} {detail}")
    print("-" * 90)
    total = len(results)
    passed = sum(1 for _, status, _ in results if status == "PASS")
    print(f"Summary: {passed}/{total} tests passed.")
    print("Limitation noted: fraud recall remains moderate and needs future tuning.")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
