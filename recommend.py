import pickle
import pandas as pd
from collections import defaultdict

print("Loading data...")

with open("data/hybrid_lookup.pkl", "rb") as f:
    hybrid_lookup = pickle.load(f)

with open("data/user_preferences.pkl", "rb") as f:
    user_preferences = pickle.load(f)

with open("data/user_repeat.pkl", "rb") as f:
    user_repeat = pickle.load(f)

with open("data/global_popularity.pkl", "rb") as f:
    global_popularity = pickle.load(f)

# Load product → department mapping
product_data = pd.read_csv("data/final_cart_data.csv")
product_dept = product_data[["product_id", "department"]].drop_duplicates()
product_dept = dict(zip(product_dept.product_id, product_dept.department))


def recommend(user_id, cart_items, top_k=10,
              w_cart=0.05, w_repeat=2.0):

    scores = defaultdict(float)

    # ----- USER REPEAT (PRIMARY SIGNAL) -----
    if user_id in user_repeat:
        for item, repeat_score in user_repeat[user_id].items():
            if item not in cart_items:
                scores[item] += w_repeat * repeat_score

    # ----- CART SIGNAL -----
    for item in cart_items:
        if item in hybrid_lookup:
            for related_item, hybrid_score in hybrid_lookup[item]:
                if related_item not in cart_items:
                    scores[related_item] += w_cart * hybrid_score

    # ----- FALLBACK: GLOBAL POPULARITY -----
    if not scores:
        for item, pop_score in global_popularity.items():
            if item not in cart_items:
                scores[item] += pop_score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [item for item, _ in ranked[:top_k]]


# Example usage
if __name__ == "__main__":
    example_user = list(user_preferences.keys())[0]
    example_cart = [24852, 13176]

    recs = recommend(example_user, example_cart)

    print("User:", example_user)
    print("Cart:", example_cart)
    print("Top Recommendations:")
    for item in recs:
        print(item)