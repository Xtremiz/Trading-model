import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset


# =========================================================
# 0. SETTINGS
# =========================================================

torch.manual_seed(42)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================================================
# 1. DATA
# =========================================================

df = pd.read_csv(
    r"F:\Git-Hub\Trading model\block-2, new idea\gold_target_40_30_30.csv"
)

df.rename(
    columns={"target": "signal"},
    inplace=True
)

df.dropna(inplace=True)


# ---------------------------------------------------------
# Label mapping
# HOLD = 0
# BUY  = 1
# SELL = 2
#
# Also supports old:
# SELL = -1
# HOLD = 0
# BUY  = 1
# ---------------------------------------------------------

signal_map = {
    0: 0,
    1: 1,
    -1: 2
}

df["signal"] = df["signal"].map(signal_map)

if df["signal"].isna().any():
    raise ValueError(
        "signal column mein unknown/missing labels hain."
    )

df["signal"] = df["signal"].astype(int)


# ---------------------------------------------------------
# Features / labels
# ---------------------------------------------------------

feature_names = df.drop(
    columns=["signal"]
).columns.tolist()

X = df[feature_names].values
y = df["signal"].values

print("\nLabels:")
print(
    np.unique(
        y,
        return_counts=True
    )
)

print("Number of features:", X.shape[1])
print("Total samples:", len(X))


# =========================================================
# 2. CHRONOLOGICAL SPLIT
# =========================================================

n = len(X)

train_end = int(n * 0.70)
val_end = int(n * 0.85)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]


print("\nDataset split:")
print("Train:", len(X_train))
print("Validation:", len(X_val))
print("Test:", len(X_test))


# =========================================================
# 3. DATASET
# =========================================================

class CustomDataset(Dataset):

    def __init__(self, features, labels):

        self.features = torch.tensor(
            features,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        return (
            self.features[index],
            self.labels[index]
        )


# =========================================================
# 4. DATALOADERS
# =========================================================

train_loader = DataLoader(
    CustomDataset(
        X_train,
        y_train
    ),
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    CustomDataset(
        X_val,
        y_val
    ),
    batch_size=32,
    shuffle=False
)

test_loader = DataLoader(
    CustomDataset(
        X_test,
        y_test
    ),
    batch_size=32,
    shuffle=False
)


# =========================================================
# 5. CLASS WEIGHTS
# =========================================================

classes = np.array([0, 1, 2])
hold_weight = 1.2
buy_weight = 1.7
sell_weight = 1.5


class_weights = torch.tensor(
    [hold_weight, buy_weight, sell_weight],
    dtype=torch.float32,
    device=device
)

print("\nClass weights:")
print(class_weights)


# =========================================================
# 6. ANN MODEL
# =========================================================

class MyNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(
                num_features,
                256
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(
                            256,
                            128
                        ),
            
            nn.BatchNorm1d(128),
            
            nn.ReLU(),
            
            nn.Dropout(0.3),

            nn.Linear(
                128,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                64,
                32
            ),

            nn.ReLU(),


            nn.Linear(
                32,
                3
            )

        )

    def forward(self, x):

        return self.model(x)


# =========================================================
# 7. MODEL / LOSS / OPTIMIZER
# =========================================================

model = MyNN(
    num_features=X_train.shape[1]
).to(device)


criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


optimizer = optim.AdamW(
    model.parameters(),
    lr=0.006856,
    weight_decay=1e-4
)


# =========================================================
# 8. TRAINING SETTINGS
# =========================================================

epochs = 100
patience = 50

best_f1 = -1.0
best_epoch = 0

best_model_state = None

wait = 0


# =========================================================
# 9. TRAINING LOOP
# =========================================================

for epoch in range(epochs):


    # -----------------------------------------------------
    # TRAIN
    # -----------------------------------------------------

    model.train()

    total_train_loss = 0.0


    for batch_features, batch_labels in train_loader:

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            batch_features
        )


        loss = criterion(
            outputs,
            batch_labels
        )


        loss.backward()

        optimizer.step()


        total_train_loss += loss.item()


    train_loss = (
        total_train_loss
        /
        len(train_loader)
    )


    # -----------------------------------------------------
    # VALIDATION
    # -----------------------------------------------------

    model.eval()

    total_val_loss = 0.0

    val_true = []
    val_pred = []


    with torch.no_grad():

        for batch_features, batch_labels in val_loader:

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)


            outputs = model(
                batch_features
            )


            loss = criterion(
                outputs,
                batch_labels
            )


            total_val_loss += loss.item()


            predictions = torch.argmax(
                outputs,
                dim=1
            )


            val_true.extend(
                batch_labels.cpu().numpy()
            )

            val_pred.extend(
                predictions.cpu().numpy()
            )


    val_loss = (
        total_val_loss
        /
        len(val_loader)
    )


    val_accuracy = accuracy_score(
        val_true,
        val_pred
    )


    val_balanced_accuracy = balanced_accuracy_score(
        val_true,
        val_pred
    )


    val_macro_f1 = f1_score(
        val_true,
        val_pred,
        average="macro"
    )


    # -----------------------------------------------------
    # BEST MODEL
    # -----------------------------------------------------

    if val_macro_f1 > best_f1:

        best_f1 = val_macro_f1

        best_epoch = epoch + 1

        best_model_state = copy.deepcopy(
            model.state_dict()
        )

        wait = 0

    else:

        wait += 1


    print(
        f"Epoch [{epoch + 1}/{epochs}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy * 100:.2f}% | "
        f"Val Bal Acc: {val_balanced_accuracy * 100:.2f}% | "
        f"Val Macro F1: {val_macro_f1:.4f}"
    )


    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    if wait >= patience:

        print(
            f"\nEarly stopping at epoch {epoch + 1}"
        )

        break


# =========================================================
# 10. LOAD BEST MODEL
# =========================================================

model.load_state_dict(
    best_model_state
)

model.eval()


# =========================================================
# 11. FINAL TEST
# =========================================================

test_true = []
test_pred = []


with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        batch_features = batch_features.to(device)


        outputs = model(
            batch_features
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        test_true.extend(
            batch_labels.numpy()
        )

        test_pred.extend(
            predictions.cpu().numpy()
        )


# =========================================================
# 12. METRICS
# =========================================================

test_accuracy = accuracy_score(
    test_true,
    test_pred
)


test_balanced_accuracy = balanced_accuracy_score(
    test_true,
    test_pred
)


test_macro_f1 = f1_score(
    test_true,
    test_pred,
    average="macro"
)


test_weighted_f1 = f1_score(
    test_true,
    test_pred,
    average="weighted"
)


# ---------------------------------------------------------
# Per-class F1
# ---------------------------------------------------------

class_f1 = f1_score(
    test_true,
    test_pred,
    labels=[0, 1, 2],
    average=None
)


hold_f1 = class_f1[0]
buy_f1 = class_f1[1]
sell_f1 = class_f1[2]


# =========================================================
# 13. PRINT MAIN RESULTS
# =========================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Macro F1: "
    f"{best_f1:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Balanced Accuracy: "
    f"{test_balanced_accuracy * 100:.2f}%"
)

print(
    f"Test Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Test Weighted F1: "
    f"{test_weighted_f1:.4f}"
)

print(
    f"HOLD F1: "
    f"{hold_f1:.4f}"
)

print(
    f"BUY F1: "
    f"{buy_f1:.4f}"
)

print(
    f"SELL F1: "
    f"{sell_f1:.4f}"
)


# =========================================================
# 14. CLASSIFICATION REPORT
# =========================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")


print(
    classification_report(
        test_true,
        test_pred,
        labels=[0, 1, 2],
        target_names=[
            "HOLD",
            "BUY",
            "SELL"
        ],
        zero_division=0
    )
)


# =========================================================
# 15. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(
    test_true,
    test_pred,
    labels=[0, 1, 2]
)


print("\n========================================")
print("CONFUSION MATRIX")
print("Rows = Actual")
print("Columns = Predicted")
print("Order: HOLD, BUY, SELL")
print("========================================")


print(cm)


print("\n              Predicted")
print("              HOLD   BUY   SELL")


print(
    f"Actual HOLD   "
    f"{cm[0][0]:5d} "
    f"{cm[0][1]:5d} "
    f"{cm[0][2]:5d}"
)


print(
    f"Actual BUY    "
    f"{cm[1][0]:5d} "
    f"{cm[1][1]:5d} "
    f"{cm[1][2]:5d}"
)


print(
    f"Actual SELL   "
    f"{cm[2][0]:5d} "
    f"{cm[2][1]:5d} "
    f"{cm[2][2]:5d}"
)


# =========================================================
# 16. SAVE BEST MODEL
# =========================================================

model.load_state_dict(
    best_model_state
)


torch.save(

    {
        "model_state_dict":
            model.state_dict(),

        "input_size":
            X_train.shape[1],

        "feature_names":
            feature_names,

        "class_mapping":
            {
                0: "HOLD",
                1: "BUY",
                2: "SELL"
            },

        "confidence_threshold":
            0.60,

        "best_epoch":
            best_epoch,

        "validation_macro_f1":
            best_f1,

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_accuracy,

        "test_macro_f1":
            test_macro_f1,

        "hold_f1":
            hold_f1,

        "buy_f1":
            buy_f1,

        "sell_f1":
            sell_f1
    },

    "best_ann_model.pt"
)


print(
    "\nBest model saved as: "
    "best_ann_model.pt"
)

