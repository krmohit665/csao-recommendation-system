import pandas as pd
import pickle
import random

random.seed(42)

from recommend import recommend  

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

# SELECT MODE HERE
mode = "hybrid"   # change to: "cart", "repeat", "hybrid"

for order_id, group in data.groupby("order_id"):
    products = group["product_id"].tolist()
    user_id = group["user_id"].iloc[0]

    if len(products) < 2:
        continue

    hidden_item = random.choice(products)
    observed_cart = [p for p in products if p != hidden_item]

    # ---- MODE SWITCH ----
    if mode == "cart":
        recs = recommend(user_id, observed_cart, top_k=10,
                         w_cart=1.0, w_repeat=0.0)

    elif mode == "repeat":
        recs = recommend(user_id, observed_cart, top_k=10,
                         w_cart=0.0, w_repeat=1.0)

    else:  # hybrid
        recs = recommend(user_id, observed_cart, top_k=10,
                         w_cart=0.05, w_repeat=2.0)

    if hidden_item in recs:
        hits += 1

    total += 1

if total == 0:
    print("No valid evaluation cases found.")
    precision_at_10 = 0
else:
    precision_at_10 = hits / total

print("Total evaluated baskets:", total)
print("Hits:", hits)
print("Precision@10:", precision_at_10)