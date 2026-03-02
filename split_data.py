import pandas as pd

print("Loading full data...")
data = pd.read_csv("data/final_cart_data.csv")

print("Sorting by user and order_number...")
data = data.sort_values(["user_id", "order_number"])

print("Building user → ordered list of orders mapping...")

# Only keep minimal columns for splitting
user_orders = (
    data[["user_id", "order_id", "order_number"]]
    .drop_duplicates()
    .sort_values(["user_id", "order_number"])
)

train_orders = set()
test_orders = set()

print("Performing temporal split (order-level)...")

for user_id, group in user_orders.groupby("user_id"):
    orders = group["order_id"].tolist()
    split_index = int(len(orders) * 0.8)

    train_orders.update(orders[:split_index])
    test_orders.update(orders[split_index:])

print("Creating train/test datasets...")

train_data = data[data["order_id"].isin(train_orders)]
test_data = data[data["order_id"].isin(test_orders)]

train_data.to_csv("data/train_cart_data.csv", index=False)
test_data.to_csv("data/test_cart_data.csv", index=False)

print("Train shape:", train_data.shape)
print("Test shape:", test_data.shape)