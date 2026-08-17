import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset

torch.manual_seed(42)

# ---------------------------------------------------------
# 1. Data
# ---------------------------------------------------------
df = pd.read_csv("robust_normalized_data.csv")

# Supports:
# HOLD=0, BUY=1, SELL=2
# Also supports old numeric labels: SELL=-1, HOLD=0, BUY=1
signal_map = {
    "HOLD": 0,
    "BUY": 1,
    "SELL": 2,
    -1: 2,
    0: 0,
    1: 1,
    2: 2,
}

df["signal"] = df["signal"].map(signal_map)

if df["signal"].isna().any():
    raise ValueError("signal column mein unknown/missing labels hain.")

df["signal"] = df["signal"].astype(int)

X = df.drop(columns=["signal"]).values
y = df["signal"].values

print("Labels:", np.unique(y, return_counts=True))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Financial time-series: chronological split; shuffle=False
n = len(X)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train, y_train = X[:train_end], y[:train_end]
X_val, y_val = X[train_end:val_end], y[train_end:val_end]
X_test, y_test = X[val_end:], y[val_end:]

# ---------------------------------------------------------
# 2. Dataset / DataLoaders
# ---------------------------------------------------------
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

train_loader = DataLoader(
    CustomDataset(X_train, y_train),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    CustomDataset(X_val, y_val),
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    CustomDataset(X_test, y_test),
    batch_size=32,
    shuffle=False
)

# ---------------------------------------------------------
# 3. Class weights — only training labels
# ---------------------------------------------------------
classes = np.array([0, 1, 2])

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

class_weights = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)

print("Class weights:", class_weights)

# ---------------------------------------------------------
# 4. Model
# ---------------------------------------------------------
class MyNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
 
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 3)   # HOLD=0, BUY=1, SELL=2
        )

    def forward(self, x):
        return self.model(x)

model = MyNN(num_features=X_train.shape[1]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# SGD lr=0.1 unstable ho sakta hai; yeh safer option hai
optimizer = optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)

# ---------------------------------------------------------
# 5. Training + validation + early stopping
# ---------------------------------------------------------
epochs = 100
patience = 15

best_f1 = -1.0
best_epoch = 0
best_model_state = None
wait = 0

for epoch in range(epochs):
    # Training
    model.train()
    total_train_loss = 0.0

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(batch_features)
        loss = criterion(outputs, batch_labels)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    # Validation — yahan val_loader use hoga, train_loader nahi
    model.eval()
    total_val_loss = 0.0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            total_val_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)

            val_true.extend(batch_labels.cpu().numpy())
            val_pred.extend(predictions.cpu().numpy())

    val_loss = total_val_loss / len(val_loader)
    val_accuracy = accuracy_score(val_true, val_pred)
    val_macro_f1 = f1_score(val_true, val_pred, average="macro")

    # Best model checkpoint
    if val_macro_f1 > best_f1:
        best_f1 = val_macro_f1
        best_epoch = epoch + 1
        best_model_state = copy.deepcopy(model.state_dict())
        wait = 0
    else:
        wait += 1

    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy * 100:.2f}% | "
        f"Val Macro F1: {val_macro_f1:.4f}"
    )

    if wait >= patience:
        print(f"Early stopping at epoch {epoch + 1}")
        break

# ---------------------------------------------------------
# 6. Final test evaluation — only once
# ---------------------------------------------------------
model.load_state_dict(best_model_state)
model.eval()

test_true = []
test_pred = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)

        outputs = model(batch_features)
        predictions = torch.argmax(outputs, dim=1)

        test_true.extend(batch_labels.numpy())
        test_pred.extend(predictions.cpu().numpy())

test_accuracy = accuracy_score(test_true, test_pred)
test_macro_f1 = f1_score(test_true, test_pred, average="macro")

print("\n========================================")
print(f"BEST EPOCH: {best_epoch}")
print(f"BEST VALIDATION MACRO F1: {best_f1:.4f}")
print(f"TEST ACCURACY: {test_accuracy * 100:.2f}%")
print(f"TEST MACRO F1: {test_macro_f1:.4f}")

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(classification_report(
    test_true,
    test_pred,
    labels=[0, 1, 2],
    target_names=["HOLD", "BUY", "SELL"],
    zero_division=0
))

print("\n========================================")
print("CONFUSION MATRIX")
print("Rows = Actual, Columns = Predicted")
print("Order: HOLD, BUY, SELL")
print("========================================")

model.load_state_dict(best_model_state)

torch.save({
    "model_state_dict": model.state_dict(),
    "input_size": X_train.shape[1],
    "feature_names": df.drop(columns=["signal"]).columns.tolist(),
    "class_mapping": {0: "HOLD", 1: "BUY", 2: "SELL"},
    "confidence_threshold": 0.60
}, "best_gold_model.pt")
print(confusion_matrix(test_true, test_pred, labels=[0, 1, 2]))