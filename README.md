# Strategic E-Commerce Personalization & Recommendation Engine

This Streamlit dashboard delivers explainable product recommendations in an executive-friendly format so teams can improve conversion, average order value, and campaign targeting.

## Strategic Overview

The project combines behavioral intelligence (collaborative signals) and product understanding (content similarity) to generate high-relevance "next best product" recommendations with clear business actions.

## Business Problem

- Information overload: large catalogs create friction and choice paralysis.
- Low conversion rates: generic merchandising misses user intent.
- Cold-start challenges: new users and items have sparse interaction signals.

## Solution Objectives

- Improve recommendation relevance across customer journeys.
- Increase CTR and conversion through better ranking.
- Lift AOV and long-term LTV with stronger cross-sell opportunities.
- Reduce empty results via robust fallback logic.
- Keep outputs explainable and deployment-ready.

## What the App Includes

- Executive overview with KPI cards
  - Customers, products, Precision@5, catalog coverage
- Single customer recommendation brief
  - Recommended products, confidence score, and plain-language reasons
- Similar product explorer
  - Content-based related items using TF-IDF text similarity
- Batch recommendations
  - Upload `user_id` CSV, generate recommendations, and download output
- Data quality and trust checks
  - Validation stats, interaction density, and evaluation snapshot

## Recommendation Methodology

The app uses a hybrid strategy:

1. Collaborative filtering (customer-to-customer affinity)
2. Content-based similarity (product name + category TF-IDF)
3. Popularity fallback for sparse or dense interaction edge cases

## Business Impact

- Higher AOV via better cross-sell and upsell suggestions.
- Better conversion with more relevant product exposure.
- Improved merchandising decisions through explainable recommendation reasons.

## Dataset Format

The app expects `product_recs.csv` with:

- `user_id`
- `product_id`
- `rating`
- `product_name`
- `category`

## Data Quality Rules

- Required columns are validated at load time.
- Ratings are coerced to numeric and filtered to `0..5`.
- Duplicate `(user_id, product_id)` rows are deduplicated.
- Positive interactions are defined as `rating >= 4`.

## Tech Stack

- Python, pandas, NumPy, scikit-learn
- TF-IDF and cosine similarity for retrieval/scoring
- Streamlit for interactive delivery

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Author

Sherriff Abdul-Hamid  
[GitHub](https://github.com/S-ABDUL-AI)  
[LinkedIn](https://www.linkedin.com/in/abdul-hamid-sherriff-08583354/)
