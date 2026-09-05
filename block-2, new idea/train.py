import copy
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
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ============================================================
# SETTINGS
# ============================================================

BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001
PATIENCE = 15          # early stopping patience
SEQUENCE_LENGTH = 50

HOLD_WEIGHT = 0.7
BUY_WEIGHT = 2

CLASS_WEIGHTS = torch.tensor(
    [HOLD_WEIGHT, BUY_WEIGHT], dtype=torch.float32
).to(device)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(r"F:\Git-Hub\Trading model\block-2, new idea\goldbuydata.csv")
df = df.dropna().reset_index(drop=True)

signal_mapping = {"hold": 0, "buy": 1}
df["signal"] = df["signal"].map(signal_mapping)

feature_names = df.drop(columns=["signal"]).columns.tolist()

X = df[feature_names].values.astype(np.float32)
y = df["signal"].values.astype(np.int64)



# ============================================================
# CREATE SEQUENCES
# ============================================================

X_sequence = []
y_sequence = []
for i in range(SEQUENCE_LENGTH, len(X)):
    X_sequence.append(X[i - SEQUENCE_LENGTH:i])
    y_sequence.append(y[i])

X_sequence = np.array(X_sequence, dtype=np.float32)
y_sequence = np.array(y_sequence, dtype=np.int64)


# ============================================================
# CHRONOLOGICAL SPLIT (TRAIN / VAL / TEST)
# TECHNIQUE 1: Random shuffle split hataya, kyunki trading
# data mein future information leak ho sakti hai agar random
# shuffle kiya. Ab time order maintain hota hai.
# ============================================================

total = len(X_sequence)
train_end = int(total * 0.70)
val_end = int(total * 0.85)

X_train, y_train = X_sequence[:train_end], y_sequence[:train_end]
X_val, y_val = X_sequence[train_end:val_end], y_sequence[train_end:val_end]
X_test, y_test = X_sequence[val_end:], y_sequence[val_end:]

print("\nSplit sizes:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


# ============================================================
# DATASET
# ============================================================

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])


train_data = CustomDataset(X_train, y_train)
val_data = CustomDataset(X_val, y_val)
test_data = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_data, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, BATCH_SIZE, shuffle=False)


# ============================================================
# MODEL
# TECHNIQUE 2: AdaptiveAvgPool1d use kiya, hardcoded 128*12
# ki jagah - agar sequence length change ho to bhi chalega.
# TECHNIQUE 3: Ek teesri Conv layer (256 channels) add ki
# taaki model deeper patterns seekh sake.
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
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x


num_features = X_train.shape[2]
model = MyCNN(num_features=num_features).to(device)

criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)

# TECHNIQUE 4: AdamW (weight_decay wala Adam) - overfitting
# kam karne mein help karta hai.
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4
)

# TECHNIQUE 5: LR Scheduler - agar BUY F1 improve hona band
# ho jaye to learning rate automatically kam ho jayegi, jisse
# model fine-tune ho sake bina overshoot kiye.
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=5
)


# ============================================================
# EVALUATION FUNCTION
# TECHNIQUE 6: buy_threshold parameter - 0.5 se neeche/upar
# karke BUY predictions ko tune kar sakte ho (precision vs recall).
# ============================================================

def evaluate_model(model, loader, buy_threshold=0.50):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)
            probabilities = torch.softmax(outputs, dim=1)
            buy_probability = probabilities[:, 1]

            predictions = (buy_probability >= buy_threshold).long()

            all_labels.extend(batch_labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    buy_f1 = f1_score(all_labels, all_predictions, pos_label=1, zero_division=0)

    report = classification_report(
        all_labels, all_predictions,
        target_names=["HOLD", "BUY"],
        output_dict=True, zero_division=0
    )
    buy_precision = report["BUY"]["precision"]
    buy_recall = report["BUY"]["recall"]

    return accuracy, balanced_acc, buy_f1, buy_precision, buy_recall, all_labels, all_predictions


# ============================================================
# TRAINING
# TECHNIQUE 7: Best model ab BUY F1 pe select hota hai,
# balanced accuracy pe nahi.
# TECHNIQUE 8: Gradient clipping - exploding gradients rokta hai.
# TECHNIQUE 9: Early stopping - agar PATIENCE epochs tak
# improvement na ho to training rok do (overfitting/time waste avoid).
# ============================================================

best_buy_f1 = -1.0
best_buy_recall = -1.0
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

    (val_accuracy, val_balanced_acc, val_buy_f1,
     val_buy_precision, val_buy_recall, _, _) = evaluate_model(model, val_loader)

    scheduler.step(val_buy_f1)
    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} | "
        f"BUY F1: {val_buy_f1:.4f} | "
        f"BUY Precision: {val_buy_precision*100:.2f}% | "
        f"BUY Recall: {val_buy_recall*100:.2f}% | "
        f"Balanced Acc: {val_balanced_acc*100:.2f}% | "
        f"LR: {current_lr:.8f}"
    )

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
        print(f"  --> New best model! BUY F1: {best_buy_f1:.4f}")
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping triggered.")
        break


# ============================================================
# LOAD BEST MODEL
# ============================================================

if best_model_weights is not None:
    torch.save(best_model_weights, "best_gold_model.pt")
    print(f"\nBest model saved (Epoch {best_epoch}) with BUY F1: {best_buy_f1:.4f}")
    model.load_state_dict(best_model_weights)
else:
    print("\nWarning: No best model found.")


# ============================================================
# FINAL TEST (best model pe)
# ============================================================

(test_accuracy, test_balanced_acc, test_buy_f1,
 test_buy_precision, test_buy_recall,
 test_labels, test_predictions) = evaluate_model(model, test_loader)

print("\n" + "=" * 50)
print("FINAL TEST RESULTS (Best Model)")
print("=" * 50)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
print(f"Test Balanced Accuracy: {test_balanced_acc*100:.2f}%")
print(f"BUY Precision: {test_buy_precision*100:.2f}%")
print(f"BUY Recall: {test_buy_recall*100:.2f}%")
print(f"BUY F1: {test_buy_f1:.4f}")

print("\nClassification Report:\n")
print(
    classification_report(
        test_labels, test_predictions,
        target_names=["HOLD", "BUY"],
        digits=4, zero_division=0
    )
)

cm = confusion_matrix(test_labels, test_predictions)
print("\nConfusion Matrix:")
print("              Predicted")
print("              HOLD   BUY")
print(f"Actual HOLD   {cm[0][0]:5d} {cm[0][1]:5d}")
print(f"Actual BUY    {cm[1][0]:5d} {cm[1][1]:5d}")