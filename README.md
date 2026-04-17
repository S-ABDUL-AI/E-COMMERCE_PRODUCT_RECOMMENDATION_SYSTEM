$ErrorActionPreference = "Stop"

$repoUrl  = "https://github.com/S-ABDUL-AI/E-COMMERCE_PRODUCT_RECOMMENDATION_SYSTEM.git"
$repoPath = "C:/Users/sherr/OneDrive/Desktop/PERSONAL WEBSITE/app_repos/E-COMMERCE_PRODUCT_RECOMMENDATION_SYSTEM"

if (Test-Path "$repoPath/.git") {
  Write-Host "Using existing repo: $repoPath"
} else {
  New-Item -ItemType Directory -Path $repoPath -Force | Out-Null
  git clone $repoUrl "$repoPath"
}

Set-Location "$repoPath"

$readme = @'
# Strategic E-Commerce Personalization & Recommendation Engine

## Strategic Overview
This project is a scalable personalization engine that combines behavioral intelligence and product understanding to deliver high-relevance recommendations that improve commercial performance.

## Business Problem
- Information Overload: Large catalogs create friction and choice paralysis.
- Low Conversion Rates: Generic displays miss user intent.
- Cold-Start Challenges: New users/products have sparse interaction signals.

## Solution Objectives
- Improve recommendation relevance across user journeys.
- Lift CTR and conversion through contextual ranking.
- Increase AOV and long-term LTV.
- Reduce cold-start impact using hybrid logic.
- Maintain explainable, production-ready recommendation workflows.

## Recommendation Methodology
- Collaborative Filtering for user-item affinity.
- Content-Based Retrieval using TF-IDF/embedding-driven product similarity.
- Hybrid ranking that blends CF + content scores with practical business rules.

## Architecture
1. Data ingestion (events, transactions, product metadata)
2. Feature engineering (interaction matrix, vectors, embeddings)
3. Model training/tuning
4. Inference delivery via app/UI for low-latency top-N recommendations

## Business Impact
- Higher AOV via better cross-sell/upsell recommendations.
- Stronger LTV through sustained relevance.
- Higher CTR from improved recommendation quality.
- Conversion lift via lower discovery friction.

## Tech Stack
- Python, pandas, NumPy, scikit-learn
- TF-IDF / embeddings for content understanding
- Streamlit for interactive delivery
- GitHub for versioned deployment workflows

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
