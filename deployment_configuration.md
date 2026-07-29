# Deployment Configuration Summary

Recommended approach: Docker container hosted locally for the capstone demonstration or on a small cloud instance for controlled review.

Prerequisites:
- macOS, Windows, or Linux
- Python 3.11+
- Git
- Docker Desktop if using container deployment

Local setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Docker setup:
```bash
docker build -t fraud-demo-prototype .
docker run -p 8501:8501 fraud-demo-prototype
```

Environment settings:
- No production secrets are required for the classroom prototype.
- Dataset path: data/sample_transactions.csv
- Streamlit port: 8501
- The .venv folder, raw private datasets, and credentials should never be committed.
