import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler


# =========================================================
# PATHS
# =========================================================

GOLD_PATH = r"F:\Git-Hub\Trading model\data\gold112_dataset.csv"
US30_PATH = r"F:\Git-Hub\Trading model\data\US30Cash_balanced.csv"

GOLD_SCALER_PATH = r"F:\Git-Hub\Trading model\data\gold112_robust_scaler.pkl"
US30_SCALER_PATH = r"F:\Git-Hub\Trading model\data\US30_robust_scaler.pkl"


# =========================================================
# CREATE SCALER
# =========================================================

def create_scaler(csv_path, scaler_path):

    print(f"\nLoading: {csv_path}")

    df = pd.read_csv(csv_path)

    # Numeric columns
    numeric_df = df.select_dtypes(include=[np.number])

    # Columns that are NOT model features
    exclude = {
        "signal",
        "target",
        "y"
    }

    feature_cols = [
        col for col in numeric_df.columns
        if col.lower() not in exclude
    ]

    X = numeric_df[feature_cols].astype(np.float64)

    print("Shape:", X.shape)
    print("Features:", len(feature_cols))

    # Fit RobustScaler
    scaler = RobustScaler()
    scaler.fit(X)

    # Save scaler
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    print("Scaler saved:")
    print(scaler_path)

    return scaler


# =========================================================
# GOLD
# =========================================================

gold_scaler = create_scaler(
    GOLD_PATH,
    GOLD_SCALER_PATH
)


# =========================================================
# US30
# =========================================================

us30_scaler = create_scaler(
    US30_PATH,
    US30_SCALER_PATH
)


print("\n========================================")
print("BOTH SCALERS CREATED SUCCESSFULLY")
print("========================================")