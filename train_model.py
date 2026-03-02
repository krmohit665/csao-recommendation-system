import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("Loading training data...")
data = pd.read_csv("data/training_data_with_features.csv")

# Select features (initial simple features)
features = [
    "order_hour_of_day",
    "days_since_prior_order",
    "order_number",
    "user_total_orders",
    "product_popularity",
    "product_reorder_rate"
]

# Remove rows with missing values
data = data.dropna(subset=features)

X = data[features]
y = data["label"]

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training LightGBM model...")

model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

print("Making predictions...")
y_pred = model.predict_proba(X_test)[:, 1]

auc = roc_auc_score(y_test, y_pred)

print("AUC Score:", auc)

# Save model
import joblib
joblib.dump(model, "model.pkl")

print("Model saved as model.pkl")