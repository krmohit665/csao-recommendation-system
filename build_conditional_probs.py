import pandas as pd
import pickle
from collections import defaultdict

print("Loading data...")
data = pd.read_csv("data/train_cart_data.csv")
data = data.sample(n=300000, random_state=42)

print("Computing item frequencies...")
item_counts = data["product_id"].value_counts().to_dict()

print("Loading co-occurrence matrix...")
with open("data/cooccurrence.pkl", "rb") as f:
    co_matrix = pickle.load(f)

print("Computing conditional probabilities...")

conditional_probs = {}

for (item1, item2), count in co_matrix.items():
    if item_counts.get(item1, 0) > 0:
        conditional_probs[(item1, item2)] = count / item_counts[item1]

print("Saving conditional probability matrix...")

with open("data/conditional_probs.pkl", "wb") as f:
    pickle.dump(conditional_probs, f)

print("Done. Total conditional pairs:", len(conditional_probs))