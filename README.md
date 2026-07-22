# Explainable AI Fraud Detection Prototype

This is the initial working prototype for the MSIT 5910 capstone project: **Development of an Explainable AI Prototype for Detecting Suspicious Credit Card Transactions**.

## What the prototype demonstrates

1. Data intake and validation
2. Model training and fraud-risk scoring
3. Plain-language explanations for selected suspicious transactions
4. Dashboard-style reporting for human review

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # Mac/Linux
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/sample_data.py
streamlit run app.py
```

## Folder structure

- `app.py` - Streamlit dashboard
- `src/` - data generation and modeling code
- `data/` - sample synthetic transaction file
- `docs/` - project documentation
- `design/` - architecture/design materials
- `tests/` - future test files

## Safety note

This demo uses synthetic data generated for a classroom prototype. It does not use live cardholder data or personally identifiable information.
