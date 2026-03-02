import pandas as pd
import pickle

print("Loading train data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Computing item counts...")
item_counts = data["product_id"].value_counts().to_dict()

print("Loading pruned co-occurrence matrix...")
with open("data/cooccurrence.pkl", "rb") as f:
    co_matrix = pickle.load(f)

print("Computing lift scores...")

N = data["order_id"].nunique()

lift_lookup = {}

for item1, related_list in co_matrix.items():
    lift_list = []

    count_a = item_counts.get(item1, 1)

    for item2, co_count in related_list:
        count_b = item_counts.get(item2, 1)

        lift = (co_count * N) / (count_a * count_b)
        lift_list.append((item2, lift))

    lift_lookup[item1] = lift_list

print("Saving lift lookup...")

with open("data/lift_lookup.pkl", "wb") as f:
    pickle.dump(lift_lookup, f)

print("Done. Base items:", len(lift_lookup))