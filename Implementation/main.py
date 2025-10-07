import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------
# Reproducibility
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------
# Step 1: Load Data
# ----------------------------
def load_data():
    orders = pd.read_csv("orders.csv")
    order_products_train = pd.read_csv("order_products__train.csv")
    order_products_prior = pd.read_csv("order_products__prior.csv")
    products = pd.read_csv("products.csv")
    aisles = pd.read_csv("aisles.csv")
    departments = pd.read_csv("departments.csv")

    products = products.merge(aisles, on="aisle_id", how="left").merge(
        departments, on="department_id", how="left"
    )
    return orders, order_products_train, order_products_prior, products

# ----------------------------
# Step 2: Feature Engineering (CRITICAL FIX)
# ----------------------------
def create_features(orders, order_products_prior, products):
    """
    Create features ONLY from prior orders to avoid data leakage
    The training target comes from train orders, but features come from prior
    """
    prior_orders = order_products_prior.merge(orders, on="order_id", how="left")
    prior_orders = prior_orders.merge(products, on="product_id", how="left")

    # User features from PRIOR orders only
    user_features = prior_orders.groupby("user_id").agg(
        total_orders=("order_number", "max"),
        avg_days_between_orders=("days_since_prior_order", "mean"),
        user_total_products=("product_id", "count"),
    ).reset_index()

    # Product features from PRIOR orders only
    product_features = prior_orders.groupby("product_id").agg(
        reorder_ratio=("reordered", "mean"),
        times_ordered=("order_id", "count"),
        product_avg_cart_position=("add_to_cart_order", "mean"),
    ).reset_index()

    # User-product features from PRIOR orders only
    user_product_features = prior_orders.groupby(["user_id", "product_id"]).agg(
        user_product_orders=("order_id", "count"),
        user_product_reorders=("reordered", "sum"),
        user_product_avg_order_dow=("order_dow", "mean"),
        user_product_avg_order_hod=("order_hour_of_day", "mean"),
    ).reset_index()
    user_product_features["user_product_reorder_rate"] = (
        user_product_features["user_product_reorders"] / 
        user_product_features["user_product_orders"]
    )
    user_product_features["user_product_reorder_rate"].fillna(0, inplace=True)

    return user_features, product_features, user_product_features

def prepare_training_data(orders, order_products_train, user_features, product_features, user_product_features):
    """
    Prepare training data by joining features with target from train orders
    """
    train_orders = order_products_train.merge(orders, on="order_id", how="left")
    
    # Merge features with training target
    train_data = train_orders.merge(user_features, on="user_id", how="left")
    train_data = train_data.merge(product_features, on="product_id", how="left")
    train_data = train_data.merge(
        user_product_features, on=["user_id", "product_id"], how="left"
    )
    
    return train_data

# ----------------------------
# Step 3: Preprocessing
# ----------------------------
def preprocess_data(train_data):
    train_data.fillna(0, inplace=True)

    # Only encode categorical features that are safe
    le_aisle = LabelEncoder()
    train_data["aisle"] = le_aisle.fit_transform(train_data["aisle"])
    le_dept = LabelEncoder()
    train_data["department"] = le_dept.fit_transform(train_data["department"])

    # SAFE features - only use historical information
    features = [
        "total_orders",
        "avg_days_between_orders",
        "user_total_products",
        "reorder_ratio",
        "times_ordered",
        "product_avg_cart_position",
        "user_product_orders",
        "user_product_reorders",
        "user_product_reorder_rate",
        "user_product_avg_order_dow",
        "user_product_avg_order_hod",
        "aisle",
        "department",
        # REMOVED: order_dow, order_hour_of_day, days_since_prior_order
        # These could leak information about the current order
    ]
    
    X = train_data[features]
    y = train_data["reordered"]

    # Identify all numerical columns for scaling
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    scaler = StandardScaler()
    X[num_cols] = scaler.fit_transform(X[num_cols])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Apply SMOTE to training set only
    print(f"Class distribution before SMOTE: {np.bincount(y_train)}")
    sm = SMOTE(random_state=SEED)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")

    return X_train_res, X_val, y_train_res, y_val, scaler, features

# ----------------------------
# Step 4: Build Model
# ----------------------------
def build_model(input_dim):
    model = Sequential()
    model.add(Dense(128, input_dim=input_dim, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.4))

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.3))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', 
                 metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# ----------------------------
# Step 5: Train Model
# ----------------------------
def train_model(model, X_train, y_train, X_val, y_val):
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.00001, verbose=1)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=2048,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    return model, history

# ----------------------------
# Step 6: Evaluate Model
# ----------------------------
def evaluate_model(model, X_val, y_val):
    y_probs = model.predict(X_val, verbose=0).flatten()

    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    
    for t in thresholds:
        y_pred_t = (y_probs > t).astype(int)
        f1_scores.append(f1_score(y_val, y_pred_t, zero_division=0))

    best_threshold = thresholds[np.argmax(f1_scores)]
    y_pred = (y_probs > best_threshold).astype(int)

    print("\n" + "="*50)
    print("MODEL PERFORMANCE")
    print("="*50)
    print(f"Best Threshold: {best_threshold:.3f}")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Precision:", precision_score(y_val, y_pred))
    print("Recall:", recall_score(y_val, y_pred))
    print("F1 Score:", f1_score(y_val, y_pred))
    print("ROC-AUC:", roc_auc_score(y_val, y_pred))

    cm = confusion_matrix(y_val, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Not Reordered', 'Reordered'],
                yticklabels=['Not Reordered', 'Reordered'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix (Threshold={best_threshold:.2f})')
    plt.tight_layout()
    plt.show()

    return best_threshold

# ----------------------------
# Main Pipeline
# ----------------------------
def main():
    print("Loading data...")
    orders, order_products_train, order_products_prior, products = load_data()

    print("Creating features from PRIOR orders only...")
    user_features, product_features, user_product_features = create_features(
        orders, order_products_prior, products
    )

    print("Preparing training data...")
    train_data = prepare_training_data(
        orders, order_products_train, user_features, product_features, user_product_features
    )

    print("Preprocessing data...")
    X_train, X_val, y_train, y_val, scaler, feature_names = preprocess_data(train_data)
    
    print(f"Using {len(feature_names)} features: {feature_names}")

    print("Building model...")
    model = build_model(X_train.shape[1])

    print("Training model...")
    model, history = train_model(model, X_train, y_train, X_val, y_val)

    print("Evaluating model...")
    best_threshold = evaluate_model(model, X_val, y_val)

    print("\nIf you still get perfect scores, check for:")
    print("1. Data leakage in your source files")
    print("2. Target variable accidentally included in features")
    print("3. Validation set contamination")

if __name__ == "__main__":
    main()