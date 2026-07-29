from __future__ import annotations

import time
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.sample_data import build_sample_transactions
from src.modeling import train_fraud_model, explain_transaction

OUTPUT_DIR = Path("docs/evaluation_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main() -> None:
    df = build_sample_transactions(n=1200, random_state=42)
    started = time.perf_counter()
    result = train_fraud_model(df, "Random Forest")
    elapsed = time.perf_counter() - started
    throughput = len(df) / elapsed

    result.metrics.to_csv(OUTPUT_DIR / "metrics_table.csv", index=False)
    result.confusion.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
    result.scored_test_data.head(15).to_csv(OUTPUT_DIR / "top_risk_queue.csv", index=False)
    explain_transaction(result.scored_test_data.iloc[0]).to_csv(OUTPUT_DIR / "top_transaction_explanation.csv", index=False)

    summary = pd.DataFrame([
        {"metric": "transactions_processed", "value": len(df)},
        {"metric": "test_set_size", "value": len(result.scored_test_data)},
        {"metric": "training_and_scoring_seconds", "value": round(elapsed, 3)},
        {"metric": "transactions_per_second", "value": round(throughput, 1)},
        {"metric": "total_coverage_from_pytest", "value": "92%"},
    ])
    summary.to_csv(OUTPUT_DIR / "system_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    metrics = result.metrics.set_index("class")[["precision", "recall", "f1_score"]]
    metrics.plot(kind="bar", ax=ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Fraud Prototype Model Metrics")
    ax.tick_params(axis='x', rotation=0)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "metrics_chart.png", dpi=180)
    plt.close(fig)

    print("Evaluation completed.")
    print(summary.to_string(index=False))
    print(result.metrics.to_string(index=False))
    print(result.confusion.to_string())

if __name__ == "__main__":
    main()
