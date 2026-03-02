import pandas as pd
from itertools import combinations
from collections import defaultdict
import pickle

print("Loading cart data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Building co-occurrence counts with pruning...")

co_matrix = defaultdict(lambda: defaultdict(int))

grouped = data.groupby("order_id")

for _, group in grouped:
    products = group["product_id"].unique()

    for item1, item2 in combinations(products, 2):
        co_matrix[item1][item2] += 1
        co_matrix[item2][item1] += 1

print("Pruning to top 100 related items per product...")

pruned_matrix = {}

for item, related_dict in co_matrix.items():
    # Sort by co-occurrence count
    top_related = sorted(
        related_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )[:100]

    pruned_matrix[item] = top_related

print("Saving pruned co-occurrence matrix...")

with open("data/cooccurrence.pkl", "wb") as f:
    pickle.dump(pruned_matrix, f)

print("Done. Base items:", len(pruned_matrix))