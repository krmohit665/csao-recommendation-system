import pandas as pd
from itertools import combinations
from collections import defaultdict
import pickle

print("Loading train data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Reducing size for development...")
data = data.sample(n=300000, random_state=42)

print("Computing item counts...")
item_counts = data["product_id"].value_counts().to_dict()

print("Counting co-occurrences...")
co_counts = defaultdict(int)

grouped = data.groupby("order_id")

for order_id, group in grouped:
    products = group["product_id"].unique()
    for a, b in combinations(products, 2):
        co_counts[(a, b)] += 1
        co_counts[(b, a)] += 1

N = len(data["order_id"].unique())

print("Computing lift scores...")

lift_lookup = defaultdict(list)

for (a, b), count_ab in co_counts.items():
    count_a = item_counts.get(a, 1)
    count_b = item_counts.get(b, 1)
    
    lift = (count_ab * N) / (count_a * count_b)
    
    lift_lookup[a].append((b, lift))

print("Saving lift lookup...")

with open("data/lift_lookup.pkl", "wb") as f:
    pickle.dump(dict(lift_lookup), f)

print("Done. Base items:", len(lift_lookup))