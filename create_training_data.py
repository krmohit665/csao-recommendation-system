import pandas as pd
import numpy as np

print("Loading processed data...")
data = pd.read_csv("data/final_cart_data.csv")

# Reduce dataset size for faster development (VERY IMPORTANT)
print("Reducing dataset size...")
data = data.sample(n=500000, random_state=42)

# Create positive samples
positive = data.copy()
positive["label"] = 1

# Create negative samples
print("Creating negative samples...")

# Get all unique products
all_products = data["product_id"].unique()

negative_rows = []

# Group by user and order
grouped = data.groupby(["user_id", "order_id"])

for (user, order), group in grouped:

    cart_products = group["product_id"].values
    
    # sample products not in cart
    negative_products = np.random.choice(
        all_products,
        size=min(len(cart_products), 5),
        replace=False
    )
    
    for product in negative_products:
        if product not in cart_products:
            
            row = group.iloc[0].copy()
            row["product_id"] = product
            row["label"] = 0
            
            negative_rows.append(row)

negative = pd.DataFrame(negative_rows)

# Combine positive and negative
training_data = pd.concat([positive, negative])

# Shuffle
training_data = training_data.sample(frac=1, random_state=42)

# Save
training_data.to_csv("data/training_data.csv", index=False)

print("Training data shape:", training_data.shape)
print(training_data["label"].value_counts())
print(training_data.head())