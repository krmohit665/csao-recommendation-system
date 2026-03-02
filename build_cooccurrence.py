import pandas as pd
from itertools import combinations
from collections import defaultdict
import pickle

print("Loading cart data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Reducing size for development...")
data = data.sample(n=300000, random_state=42)

print("Building co-occurrence counts...")

co_matrix = defaultdict(int)

grouped = data.groupby("order_id")

for order_id, group in grouped:
    products = group["product_id"].unique()
    
    for item1, item2 in combinations(products, 2):
        co_matrix[(item1, item2)] += 1
        co_matrix[(item2, item1)] += 1

print("Saving co-occurrence matrix...")

with open("data/cooccurrence.pkl", "wb") as f:
    pickle.dump(dict(co_matrix), f)

print("Done. Total pairs:", len(co_matrix))