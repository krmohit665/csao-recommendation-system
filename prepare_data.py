import pandas as pd

# Load data
orders = pd.read_csv("data/orders.csv")
order_products = pd.read_csv("data/order_products__prior.csv")
products = pd.read_csv("data/products.csv")
aisles = pd.read_csv("data/aisles.csv")
departments = pd.read_csv("data/departments.csv")

# Merge order_products with products
data = order_products.merge(products, on="product_id", how="left")

# Merge with aisles
data = data.merge(aisles, on="aisle_id", how="left")

# Merge with departments
data = data.merge(departments, on="department_id", how="left")

# Merge with orders
data = data.merge(orders, on="order_id", how="left")

# Keep only important columns
data = data[[
    "user_id",
    "order_id",
    "product_id",
    "product_name",
    "aisle",
    "department",
    "order_number",
    "order_hour_of_day",
    "days_since_prior_order"
]]

# Save processed data
data.to_csv("data/final_cart_data.csv", index=False)

print("Final dataset shape:", data.shape)
print(data.head())