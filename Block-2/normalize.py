from sklearn.preprocessing import RobustScaler
import pandas as pd

path = r"F:\Git-Hub\Trading model\Block-2\goldthresold0.5.csv"

df = pd.read_csv(path)

# Features
X = df.iloc[:, :-1]

# Target
y = df.iloc[:, -1]

# Robust Scaling
scaler = RobustScaler()

X_scaled = scaler.fit_transform(X)

# DataFrame
X_scaled = pd.DataFrame(
    X_scaled,
    columns=X.columns
)

# Signal wapas add
X_scaled["signal"] = y.values

# Save with _norm
output_path = path.replace(
    ".csv",
    "_norm.csv"
)

X_scaled.to_csv(
    output_path,
    index=False
)

print("Saved:", output_path)