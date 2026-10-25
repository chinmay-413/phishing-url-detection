# AI-Based Phishing URL Detection System

Classifies a URL as legitimate or phishing, with a risk score and an explanation.

## Features
- Ensemble of three models: gradient boosting on URL features, character n-gram model, and a model pooled from two datasets
- Tested on websites never seen in training (domain-separated split)
- Streamlit app: single scan, batch scan, model insights

## Datasets (download separately from Kaggle)
- malicious_phish.csv
- phishing_url_dataset.csv

## Run
    pip install -r requirements.txt
    python train.py
    streamlit run app.py

## Limitations
URL-only analysis: no page content, domain age or live blocklists.