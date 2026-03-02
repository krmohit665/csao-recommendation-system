import pandas as pd
import pickle

print("Loading cart data...")
data = pd.read_csv("data/train_cart_data.csv")

print("Reducing size for development...")
data = data.sample(n=500000, random_state=42)

print("Computing user total orders...")
user_orders = data.groupby("user_id")["order_id"].nunique()

print("Computing user department counts...")
user_dept_counts = data.groupby(["user_id", "department"])["order_id"].count()

print("Building preference scores...")

user_preferences = {}

for (user, dept), count in user_dept_counts.items():
    total = user_orders.get(user, 1)
    score = count / total
    
    if user not in user_preferences:
        user_preferences[user] = {}
    
    user_preferences[user][dept] = score

print("Saving user preferences...")

with open("data/user_preferences.pkl", "wb") as f:
    pickle.dump(user_preferences, f)

print("Done. Total users:", len(user_preferences))