# CSAO Recommendation System

## Overview
Hybrid cart-based retrieval and user repeat-memory recommendation system designed for real-time Cart Super Add-On (CSAO) optimization.

The system combines:
- Item-to-item hybrid retrieval (conditional probability + lift)
- User repeat-memory modeling
- Popularity fallback

---

## Final Results

- Precision@10: ~0.45 (±0.01 variation due to random masking)
- 100% recommendation coverage
- Order-level temporal evaluation
- Strict order-level temporal evaluation (80/20 per user split)

---

## Dataset

This project uses the **Instacart Market Basket Analysis** dataset from Kaggle:

https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis

⚠ The dataset (~1GB) is NOT included in this repository.

After downloading, place the following files inside the `data/` directory:

- `orders.csv`
- `order_products__prior.csv`
- `products.csv`
- `aisles.csv`
- `departments.csv`

---

## Environment Setup
 
### Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy lightgbm scikit-learn joblib
```
### Windows
```bash
python -m venv .venv
.venv\Scripts\activate
pip install pandas numpy lightgbm scikit-learn joblib
```


## Run Pipeline

```bash
python run_pipeline.py
```
## 🏗️ Architecture

### 1. Hybrid Item-to-Item Retrieval

Combines:
- **Conditional Probability**
- **Lift Score**

> Co-occurrence relationships are pruned to the **top-100 related items** per product to maintain memory efficiency.

---

### 2. User Repeat Memory

Models repeat purchase behavior:
```python
repeat_score = user_product_orders / user_total_orders
```

This acts as the **primary ranking signal**.

---

### 3. Popularity Fallback

Used when no repeat or cart-based signals exist, ensuring **full coverage**.

---

## 🧪 Experimental / Baseline ML Track

The repository also includes:

| File | Purpose |
|------|---------|
| `create_training_data.py` | Training data generation |
| `feature_engineering.py` | Feature construction |
| `train_model.py` | Model training |

> These were used for **LightGBM experimentation** and are not part of the final hybrid CSAO pipeline. They are retained for completeness and comparison.

---

## 📁 Project Structure
```
prepare_data.py          → Dataset preprocessing
split_data.py            → Temporal train/test split
build_*.py               → Offline signal construction
recommend.py             → Final hybrid ranking logic
evaluate_recommender.py  → Evaluation pipeline
run_pipeline.py          → One-command execution
train_model.py           → Baseline ML experimentation
```

---

## 📝 Notes

- 📦 Large `.pkl` artifacts are **excluded from GitHub**
- ♻️ All artifacts are **reproducible** using `run_pipeline.py`
- 💾 Designed for execution on **~8GB RAM** systems
- 🔒 **Deterministic evaluation** supported via fixed random seeds
