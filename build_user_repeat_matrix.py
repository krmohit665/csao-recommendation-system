import pandas as pd
import pickle
from collections import defaultdict

print("Loading train data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Computing user total orders...")
user_total_orders = data.groupby("user_id")["order_id"].nunique().to_dict()

print("Computing user-item purchase counts...")
user_item_counts = data.groupby(["user_id", "product_id"])["order_id"].count()

print("Building user repeat scores...")

user_repeat = defaultdict(dict)

for (user, product), count in user_item_counts.items():
    total_orders = user_total_orders.get(user, 1)
    repeat_score = count / total_orders
    user_repeat[user][product] = repeat_score

print("Saving user repeat matrix...")

with open("data/user_repeat.pkl", "wb") as f:
    pickle.dump(dict(user_repeat), f)

print("Done. Total users:", len(user_repeat))