import pandas as pd
import pickle

print("Loading train data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Computing product popularity...")
product_popularity = data.groupby("product_id")["order_id"].count()

# normalize
product_popularity = product_popularity / product_popularity.sum()

product_popularity = product_popularity.to_dict()

print("Saving popularity table...")
with open("data/global_popularity.pkl", "wb") as f:
    pickle.dump(product_popularity, f)

print("Done. Total products:", len(product_popularity))