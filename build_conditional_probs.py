import pandas as pd
import pickle

print("Loading train data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Computing item frequencies...")
item_counts = data["product_id"].value_counts().to_dict()

print("Loading co-occurrence matrix...")
with open("data/cooccurrence.pkl", "rb") as f:
    co_matrix = pickle.load(f)

print("Computing conditional probabilities...")

conditional_probs = {}

for item1, related_list in co_matrix.items():
    count_item1 = item_counts.get(item1, 0)

    if count_item1 == 0:
        continue

    for item2, co_count in related_list:
        conditional_probs[(item1, item2)] = co_count / count_item1

print("Saving conditional probability matrix...")

with open("data/conditional_probs.pkl", "wb") as f:
    pickle.dump(conditional_probs, f)

print("Done. Total conditional pairs:", len(conditional_probs))