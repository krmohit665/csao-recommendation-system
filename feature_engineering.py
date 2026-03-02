import pandas as pd

print("Loading training data...")
data = pd.read_csv("data/training_data.csv")

print("Creating user_total_orders feature...")
user_orders = data.groupby("user_id")["order_id"].nunique().reset_index()
user_orders.columns = ["user_id", "user_total_orders"]

data = data.merge(user_orders, on="user_id", how="left")

print("Creating product_popularity feature...")
product_popularity = data.groupby("product_id")["order_id"].count().reset_index()
product_popularity.columns = ["product_id", "product_popularity"]

data = data.merge(product_popularity, on="product_id", how="left")

print("Creating product_reorder_rate feature...")
product_reorder = data.groupby("product_id")["label"].mean().reset_index()
product_reorder.columns = ["product_id", "product_reorder_rate"]

data = data.merge(product_reorder, on="product_id", how="left")



data = data.fillna(0)

data.to_csv("data/training_data_with_features.csv", index=False)

print("Feature engineering complete.")
print(data.head())