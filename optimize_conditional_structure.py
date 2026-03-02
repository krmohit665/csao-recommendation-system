import pickle
from collections import defaultdict

print("Loading conditional probabilities...")

with open("data/conditional_probs.pkl", "rb") as f:
    conditional_probs = pickle.load(f)

print("Restructuring for fast lookup...")

lookup = defaultdict(list)

for (item1, item2), prob in conditional_probs.items():
    lookup[item1].append((item2, prob))

print("Saving optimized lookup...")

with open("data/conditional_lookup.pkl", "wb") as f:
    pickle.dump(dict(lookup), f)

print("Done. Total base items:", len(lookup))