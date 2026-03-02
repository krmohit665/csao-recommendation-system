import pandas as pd
import pickle
import random

from recommend import recommend  # ✅ USE ACTUAL RECOMMENDER

print("Loading data...")

data = pd.read_csv("data/test_cart_data.csv")

with open("data/hybrid_lookup.pkl", "rb") as f:
    hybrid_lookup = pickle.load(f)

with open("data/user_preferences.pkl", "rb") as f:
    user_preferences = pickle.load(f)

with open("data/user_repeat.pkl", "rb") as f:
    user_repeat = pickle.load(f)

with open("data/global_popularity.pkl", "rb") as f:
    global_popularity = pickle.load(f)

# Only food-related departments
food_departments = [
    "dairy eggs",
    "bakery",
    "produce",
    "frozen",
    "snacks",
    "beverages"
]

data = data[data["department"].isin(food_departments)]

# Keep only orders where all items belong to same department
dept_counts = data.groupby("order_id")["department"].nunique()
single_dept_orders = dept_counts[dept_counts == 1].index
data = data[data["order_id"].isin(single_dept_orders)]

# Keep basket size 2–5
basket_sizes = data.groupby("order_id")["product_id"].count()
valid_orders = basket_sizes[(basket_sizes >= 2) & (basket_sizes <= 5)].index
data = data[data["order_id"].isin(valid_orders)]

# Sample for speed
data = data.sample(n=min(20000, len(data)), random_state=42)

print("Evaluating...")

hits = 0
total = 0

for order_id, group in data.groupby("order_id"):
    products = group["product_id"].tolist()
    user_id = group["user_id"].iloc[0]

    if len(products) < 2:
        continue

    hidden_item = random.choice(products)
    observed_cart = [p for p in products if p != hidden_item]

    recs = recommend(user_id, observed_cart, top_k=10)

    if hidden_item in recs:
        hits += 1

    total += 1

precision_at_10 = hits / total

print("Precision@10:", precision_at_10)