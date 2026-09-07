import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. HYPERPARAMETERS
# =========================================================

LEARNING_RATE = 0.00684998
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 50   # early stopping patience


sequence_length = 100


# =========================================================
# DEVICE
# =========================================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(
    r"F:\Git-Hub\Trading model\block-2, new idea\gold_target_40_30_30.csv"
)

df = df.dropna()


# =========================================================
# 3. SIGNAL MAPPING
# =========================================================

df.rename(columns={"target": "signal"}, inplace=True)
signal_mapping = {
    0: 0,
    1: 1,
    -1: 2
}

df["signal"] = df["signal"].map(signal_mapping)

# Remove any rows where signal mapping failed
df = df.dropna(subset=["signal"])

df["signal"] = df["signal"].astype(int)


# =========================================================
# 4. FEATURES AND LABELS
# =========================================================

X = df.drop(columns=["signal"]).values
y = df["signal"].values


# =========================================================
# 5. CREATE SEQUENCES
# Previous 50 candles -> predict current signal
# =========================================================

X_sequences = []
y_sequences = []

for i in range(sequence_length, len(X)):

    X_sequences.append(
        X[i - sequence_length:i]
    )

    y_sequences.append(
        y[i]
    )


X_sequences = np.array(X_sequences)
y_sequences = np.array(y_sequences)


print("X Shape:", X_sequences.shape)
print("y Shape:", y_sequences.shape)


# =========================================================
# 6. TRAIN / VAL / TEST SPLIT
# First split off test, then split remainder into train/val
# =========================================================

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.1,   # 10% of remaining data used for validation
    random_state=42,
    shuffle=True
)


print("\nTrain Shape:", X_train.shape)
print("Val Shape  :", X_val.shape)
print("Test Shape :", X_test.shape)


# =========================================================
# 7. CLASS WEIGHTS
# =========================================================

hold_weight = 1.75
buy_weight = 2.28
sell_weight = 2.17

CLASS_WEIGHTS = torch.tensor(
    [
        hold_weight,
        buy_weight,
        sell_weight
    ],
    dtype=torch.float32
).to(device)


# =========================================================
# 8. DATASET
# =========================================================

class CustomDataset(Dataset):

    def __init__(self, features, labels):

        self.data = torch.tensor(
            features,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        return (
            self.data[idx],
            self.labels[idx]
        )


# =========================================================
# 9. DATASETS
# =========================================================

train_data = CustomDataset(X_train, y_train)
val_data = CustomDataset(X_val, y_val)
test_data = CustomDataset(X_test, y_test)


# =========================================================
# 10. DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# =========================================================
# 11. CNN MODEL
# =========================================================

class MyCNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        # -------------------------------------------------
        # CNN FEATURE EXTRACTOR
        # -------------------------------------------------

        self.features = nn.Sequential(

            # Input:
            # (batch, num_features, 50)

            nn.Conv1d(
                in_channels=num_features,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.MaxPool1d(kernel_size=2, stride=2),

            # Sequence: 50 -> 25

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.MaxPool1d(kernel_size=2, stride=2)

            # Sequence: 25 -> 12
        )

        # -------------------------------------------------
        # LSTM
        # -------------------------------------------------

        # Takes the CNN's downsampled sequence
        # (batch, reduced_seq_len, 128)
        # and learns temporal dependencies across it.
        # in_channels (input_size) = 128  -> matches conv output channels
        # out_channels (hidden_size) = 128, bidirectional -> 256 out

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # -------------------------------------------------
        # ADAPTIVE POOLING
        # -------------------------------------------------

        # Collapses the (now variable-length) LSTM output
        # sequence down to a single vector per sample, so
        # we no longer depend on a fixed sequence_length
        # (previously hardcoded as 12/25 after the conv
        # stack — that's what adaptive pooling removes).

        self.pool = nn.AdaptiveAvgPool1d(1)

        # -------------------------------------------------
        # CLASSIFIER
        # -------------------------------------------------

        self.classifier = nn.Sequential(

            # in = 256 (lstm hidden_size 128 * 2 directions), out = 64
            nn.Linear(256, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            # in = 64, out = 128
            nn.Linear(64, 128),

            nn.ReLU(),

            nn.Dropout(0.2),

            # Output:
            # HOLD = 0
            # BUY  = 1
            # SELL = 2

            nn.Linear(128, 3)
        )

    def forward(self, x):

        # Original: (batch, 50, num_features)
        x = x.permute(0, 2, 1)
        # After: (batch, num_features, 50)

        x = self.features(x)
        # After CNN: (batch, conv2_channels, reduced_seq_len)

        x = x.permute(0, 2, 1)
        # For LSTM: (batch, reduced_seq_len, conv2_channels)

        x, (h_n, c_n) = self.lstm(x)
        # LSTM out: (batch, reduced_seq_len, lstm_hidden * directions)

        x = x.permute(0, 2, 1)
        # For pooling: (batch, lstm_hidden * directions, reduced_seq_len)

        x = self.pool(x)
        # After pooling: (batch, lstm_hidden * directions, 1)

        x = x.squeeze(-1)
        # Flattened: (batch, lstm_hidden * directions)

        x = self.classifier(x)

        return x


# =========================================================
# 12. CREATE MODEL
# =========================================================

num_features = X_train.shape[2]

model = MyCNN(num_features=num_features).to(device)

print("\nModel:")
print(model)


# =========================================================
# 13. LOSS, OPTIMIZER, SCHEDULER
# =========================================================

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=5
)


# =========================================================
# 14. EVALUATION FUNCTION
# =========================================================

def evaluate_model(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_features, batch_labels in loader:

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)

            predictions = torch.argmax(outputs, dim=1)

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)

    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)

    macro_f1 = f1_score(
        all_labels, all_predictions, average="macro", zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels, all_predictions, average="weighted", zero_division=0
    )

    report = classification_report(
        all_labels,
        all_predictions,
        target_names=["HOLD", "BUY", "SELL"],
        output_dict=True,
        zero_division=0
    )

    hold_f1 = report["HOLD"]["f1-score"]
    buy_f1 = report["BUY"]["f1-score"]
    sell_f1 = report["SELL"]["f1-score"]

    return (
        accuracy,
        balanced_acc,
        macro_f1,
        weighted_f1,
        hold_f1,
        buy_f1,
        sell_f1,
        all_labels,
        all_predictions
    )


# =========================================================
# 15. TRAINING LOOP (with validation + early stopping)
# =========================================================

print("\nStarting Training...\n")

best_macro_f1 = -1.0
best_balanced_acc = -1.0

best_model_weights = None
best_epoch = 0

epochs_without_improvement = 0


for epoch in range(EPOCHS):

    model.train()

    total_loss = 0

    for batch_features, batch_labels in train_loader:

        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        output = model(batch_features)

        loss = criterion(output, batch_labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)

    (
        val_accuracy,
        val_balanced_acc,
        val_macro_f1,
        val_weighted_f1,
        val_hold_f1,
        val_buy_f1,
        val_sell_f1,
        _,
        _
    ) = evaluate_model(model, val_loader)

    scheduler.step(val_macro_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} | "
        f"Macro F1: {val_macro_f1:.4f} | "
        f"Balanced Acc: {val_balanced_acc * 100:.2f}% | "
        f"HOLD F1: {val_hold_f1:.4f} | "
        f"BUY F1: {val_buy_f1:.4f} | "
        f"SELL F1: {val_sell_f1:.4f} | "
        f"LR: {current_lr:.8f}"
    )

    is_better = False

    if val_macro_f1 > best_macro_f1:
        is_better = True
    elif (
        val_macro_f1 == best_macro_f1
        and val_balanced_acc > best_balanced_acc
    ):
        is_better = True

    if is_better:

        best_macro_f1 = val_macro_f1
        best_balanced_acc = val_balanced_acc

        best_model_weights = copy.deepcopy(model.state_dict())

        best_epoch = epoch + 1

        epochs_without_improvement = 0

        print(f"  --> New best model! Macro F1: {best_macro_f1:.4f}")

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")

        break


# =========================================================
# 16. SAVE + LOAD BEST MODEL
# =========================================================

if best_model_weights is not None:

    torch.save(best_model_weights, "best_gold_model_hybrid2.pt")

    print(
        f"\nBest model saved (Epoch {best_epoch}) "
        f"with Macro F1: {best_macro_f1:.4f}"
    )

    model.load_state_dict(best_model_weights)

else:

    print("\nWarning: No best model found.")


# =========================================================
# 17. FINAL TEST EVALUATION
# =========================================================

(
    test_accuracy,
    test_balanced_acc,
    test_macro_f1,
    test_weighted_f1,
    test_hold_f1,
    test_buy_f1,
    test_sell_f1,
    test_labels,
    test_predictions
) = evaluate_model(model, test_loader)


print("\n" + "=" * 50)
print("FINAL TEST RESULTS (Best Model)")
print("=" * 50)

print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"Test Balanced Accuracy: {test_balanced_acc * 100:.2f}%")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"Test Weighted F1: {test_weighted_f1:.4f}")
print(f"HOLD F1: {test_hold_f1:.4f}")
print(f"BUY F1: {test_buy_f1:.4f}")
print(f"SELL F1: {test_sell_f1:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        test_labels,
        test_predictions,
        target_names=["HOLD", "BUY", "SELL"],
        digits=4,
        zero_division=0
    )
)

cm = confusion_matrix(test_labels, test_predictions)

print("\nConfusion Matrix:")
print("              Predicted")
print("              HOLD   BUY   SELL")

print(
    f"Actual HOLD   {cm[0][0]:5d} {cm[0][1]:5d} {cm[0][2]:5d}"
)
print(
    f"Actual BUY    {cm[1][0]:5d} {cm[1][1]:5d} {cm[1][2]:5d}"
)
print(
    f"Actual SELL   {cm[2][0]:5d} {cm[2][1]:5d} {cm[2][2]:5d}"
)