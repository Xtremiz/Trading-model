import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. DEVICE
# ============================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print(f"Using device: {device}")
print("=" * 60)


# ============================================================
# 2. SETTINGS
# ============================================================

DATA_PATH = r"F:\Git-Hub\Trading model\block-2, new idea\goldbuydata.csv"

MODEL_PATH = r"F:\Git-Hub\Trading model\block-2, new idea\best_gold_buy_model.pt"

SEQUENCE_LENGTH = 50

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.001

PATIENCE = 12

# ------------------------------------------------------------
# BUY CLASS WEIGHT
#
# Class mapping:
# 0 = HOLD
# 1 = BUY
#
# BUY ko priority di ja rahi hai.
# Pehle 4.0 try karna reasonable hai.
# Agar BUY recall kam ho -> 5.0 try karo.
# Agar false BUY bohot zyada hon -> 3.0 try karo.
# ------------------------------------------------------------

HOLD_WEIGHT = 0.5
BUY_WEIGHT = 3.4

CLASS_WEIGHTS = torch.tensor(
    [HOLD_WEIGHT, BUY_WEIGHT],
    dtype=torch.float32
).to(device)


# ============================================================
# 3. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

df = df.dropna().reset_index(drop=True)

print(f"Dataset shape: {df.shape}")


# ============================================================
# 4. SIGNAL MAPPING
# ============================================================

signal_mapping = {
    "hold": 0,
    "buy": 1
}

df["signal"] = df["signal"].map(signal_mapping)

if df["signal"].isna().any():
    print("\nERROR: Unknown signal values found:")
    print(df.loc[df["signal"].isna(), "signal"])
    raise ValueError(
        "Signal column contains values other than 'hold' and 'buy'."
    )

df["signal"] = df["signal"].astype(int)

print("\nOriginal class distribution:")
print(df["signal"].value_counts().sort_index())

print("\nClass distribution:")
print(
    df["signal"]
    .value_counts()
    .rename(index={0: "HOLD", 1: "BUY"})
)


# ============================================================
# 5. FEATURES
# ============================================================

feature_names = [col for col in df.columns if col != "signal"]

X = df[feature_names].values.astype(np.float32)
y = df["signal"].values.astype(np.int64)

print("\nNumber of features:", len(feature_names))


# ============================================================
# 6. CREATE SEQUENCES
# ============================================================

print("\nCreating sequences...")

X_sequences = []
y_sequences = []

for i in range(SEQUENCE_LENGTH, len(X)):
    X_sequences.append(X[i - SEQUENCE_LENGTH:i])
    y_sequences.append(y[i])

X_sequences = np.array(X_sequences, dtype=np.float32)
y_sequences = np.array(y_sequences, dtype=np.int64)

print("X shape:", X_sequences.shape)
print("y shape:", y_sequences.shape)


# ============================================================
# 7. CHRONOLOGICAL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

total_samples = len(X_sequences)

train_end = int(total_samples * 0.70)
validation_end = int(total_samples * 0.85)

X_train = X_sequences[:train_end]
y_train = y_sequences[:train_end]

X_val = X_sequences[train_end:validation_end]
y_val = y_sequences[train_end:validation_end]

X_test = X_sequences[validation_end:]
y_test = y_sequences[validation_end:]

print("\nSplit:")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)


# ============================================================
# 8. DATASET
# ============================================================

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])


# ============================================================
# 9. DATALOADERS
# ============================================================

train_data = CustomDataset(X_train, y_train)
validation_data = CustomDataset(X_val, y_val)
test_data = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
validation_loader = DataLoader(validation_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# 10. CNN MODEL
# ============================================================

class MyCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.20)
        )

        # Adaptive pooling - sequence length change hone par bhi model chalega
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.30),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, 2)
        )

    def forward(self, x):
        # Input: [batch, sequence, features]
        x = x.permute(0, 2, 1)
        # [batch, features, sequence]

        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)

        return x


# ============================================================
# 11. CREATE MODEL
# ============================================================

num_features = X_train.shape[2]

model = MyCNN(num_features=num_features).to(device)

print("\nModel:")
print(model)


# ============================================================
# 12. LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)


# ============================================================
# 13. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)


# ============================================================
# 14. LR SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=4
)


# ============================================================
# 15. VALIDATION FUNCTION
# ============================================================

def evaluate_model(model, loader, buy_threshold=0.50):
    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)

            probabilities = torch.softmax(outputs, dim=1)

            # BUY probability
            buy_probability = probabilities[:, 1]

            # Agar BUY probability threshold se zyada hai to BUY predict hoga
            predictions = (buy_probability >= buy_threshold).long()

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(buy_probability.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)

    accuracy = accuracy_score(all_labels, all_predictions)

    balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)

    buy_f1 = f1_score(
        all_labels, all_predictions, pos_label=1, zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["HOLD", "BUY"],
        output_dict=True,
        zero_division=0
    )

    buy_precision = report["BUY"]["precision"]
    buy_recall = report["BUY"]["recall"]

    return (
        accuracy,
        balanced_accuracy,
        buy_f1,
        buy_precision,
        buy_recall,
        all_labels,
        all_predictions,
        all_probabilities
    )


# ============================================================
# 16. TRAINING
# ============================================================

print("\n")
print("=" * 60)
print("STARTING TRAINING")
print("=" * 60)

best_buy_f1 = -1.0
best_buy_recall = -1.0
best_model_weights = None
best_epoch = 0
epochs_without_improvement = 0


for epoch in range(EPOCHS):

    # ========================================================
    # TRAIN
    # ========================================================

    model.train()
    total_loss = 0.0

    for batch_features, batch_labels in train_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(batch_features)

        loss = criterion(outputs, batch_labels)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_accuracy,
        val_balanced_accuracy,
        val_buy_f1,
        val_buy_precision,
        val_buy_recall,
        _,
        _,
        _
    ) = evaluate_model(model, validation_loader, buy_threshold=0.50)

    # ========================================================
    # LR SCHEDULER
    # ========================================================

    scheduler.step(val_buy_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    # ========================================================
    # PRINT
    # ========================================================

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
    print(f"Loss: {average_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")
    print(f"Validation Balanced Accuracy: {val_balanced_accuracy * 100:.2f}%")
    print(f"BUY Precision: {val_buy_precision * 100:.2f}%")
    print(f"BUY Recall: {val_buy_recall * 100:.2f}%")
    print(f"BUY F1: {val_buy_f1:.4f}")
    print(f"Learning Rate: {current_lr:.8f}")

    # ========================================================
    # BEST MODEL
    #
    # Primary objective = BUY F1
    # Tie-breaker = BUY Recall
    # ========================================================

    is_better = False

    if val_buy_f1 > best_buy_f1:
        is_better = True
    elif val_buy_f1 == best_buy_f1 and val_buy_recall > best_buy_recall:
        is_better = True

    if is_better:
        best_buy_f1 = val_buy_f1
        best_buy_recall = val_buy_recall
        best_model_weights = copy.deepcopy(model.state_dict())
        best_epoch = epoch + 1
        epochs_without_improvement = 0

        print("\n>>> NEW BEST MODEL <<<")
        print(f"BUY F1: {best_buy_f1:.4f}")
        print(f"BUY Recall: {best_buy_recall * 100:.2f}%")
    else:
        epochs_without_improvement += 1

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# 17. LOAD BEST MODEL
# ============================================================

if best_model_weights is None:
    raise RuntimeError("No best model was saved.")

model.load_state_dict(best_model_weights)


# ============================================================
# 18. FINAL TEST
# ============================================================

print("\n")
print("=" * 60)
print("FINAL TEST RESULTS")
print("=" * 60)

(
    test_accuracy,
    test_balanced_accuracy,
    test_buy_f1,
    test_buy_precision,
    test_buy_recall,
    test_labels,
    test_predictions,
    test_probabilities
) = evaluate_model(model, test_loader, buy_threshold=0.50)

print(f"\nBest Epoch: {best_epoch}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Balanced Accuracy: {test_balanced_accuracy * 100:.2f}%")
print(f"BUY Precision: {test_buy_precision * 100:.2f}%")
print(f"BUY Recall: {test_buy_recall * 100:.2f}%")
print(f"BUY F1: {test_buy_f1:.4f}")


# ============================================================
# 19. CLASSIFICATION REPORT
# ============================================================

print("\n")
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["HOLD", "BUY"],
        digits=4,
        zero_division=0
    )
)


# ============================================================
# 20. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(test_labels, test_predictions)

print("\n")
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print("\n              Predicted")
print("              HOLD   BUY")
print(f"Actual HOLD   {cm[0][0]:5d} {cm[0][1]:5d}")
print(f"Actual BUY    {cm[1][0]:5d} {cm[1][1]:5d}")


# ============================================================
# 21. SAVE COMPLETE CHECKPOINT
# ============================================================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "input_size": num_features,
    "feature_names": feature_names,
    "sequence_length": SEQUENCE_LENGTH,
    "class_mapping": {0: "HOLD", 1: "BUY"},
    "class_weights": CLASS_WEIGHTS.cpu(),
    "hold_weight": HOLD_WEIGHT,
    "buy_weight": BUY_WEIGHT,
    "best_epoch": best_epoch,
    "best_buy_f1": best_buy_f1,
    "best_buy_recall": best_buy_recall,
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "batch_size": BATCH_SIZE,
    "model_type": "CNN",
    "test_accuracy": test_accuracy,
    "test_balanced_accuracy": test_balanced_accuracy,
    "test_buy_precision": test_buy_precision,
    "test_buy_recall": test_buy_recall,
    "test_buy_f1": test_buy_f1
}

torch.save(checkpoint, MODEL_PATH)

print("\n")
print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(f"\nPath:\n{MODEL_PATH}")
print(f"\nBest BUY F1: {best_buy_f1:.4f}")
print(f"Best BUY Recall: {best_buy_recall * 100:.2f}%")