# CSAO Recommendation System

## Overview
Hybrid cart-based retrieval and user repeat-memory recommendation system designed for real-time Cart Super Add-On (CSAO) optimization.

The system combines:
- Item-to-item hybrid retrieval (conditional probability + lift)
- User repeat-memory modeling
- Popularity fallback

---

## Key Results

- Precision@10: ~0.24 (±0.01 variation due to random masking)
- 100% recommendation coverage
- Order-level temporal evaluation

---

## Dataset

This project uses the **Instacart Market Basket Analysis** dataset from Kaggle:

https://www.kaggle.com/competitions/instacart-market-basket-analysis

⚠ The dataset (~1GB) is NOT included in this repository.

After downloading, place the following files inside the `data/` directory:

- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

---

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy lightgbm scikit-learn joblib

## Run Pipeline

```bash
python run_pipeline.py
