import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)


# =========================================================
# 1. HYPERPARAMETERS
# =========================================================

learning_rate = 0.00684998
batch_size = 64
epochs = 100

hold_weight = 0.8
buy_weight = 2.0
sell_weight = 2.9

conv1_channels = 64
conv2_channels = 128
kernel_size = 3

dropout = 0.2

linear1 = 64
linear2 = 128

sequence_length = 50


# =========================================================
# 2. LOAD DATA
# =========================================================

df = pd.read_csv(
    r"F:\Git-Hub\Trading model\Block-2\goldthresold0.4_norm.csv"
)

df = df.dropna()


# =========================================================
# 3. SIGNAL MAPPING
# =========================================================

signal_mapping = {
    "hold": 0,
    "buy": 1,
    "sell": 2
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
# 6. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


print("\nTrain Shape:", X_train.shape)
print("Test Shape :", X_test.shape)


# =========================================================
# 7. CLASS WEIGHTS
# =========================================================

# Class mapping:
# 0 = HOLD
# 1 = BUY
# 2 = SELL

class_weights = {
    0: hold_weight,
    1: buy_weight,
    2: sell_weight
}

print("\nClass Weights:")
print(class_weights)


# PyTorch Tensor
class_weights_tensor = torch.tensor(
    [
        hold_weight,
        buy_weight,
        sell_weight
    ],
    dtype=torch.float32
)


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

train_data = CustomDataset(
    X_train,
    y_train
)

test_data = CustomDataset(
    X_test,
    y_test
)


# =========================================================
# 10. DATALOADERS
# =========================================================

train_loader = DataLoader(
    train_data,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=batch_size,
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
                out_channels=conv1_channels,
                kernel_size=kernel_size,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                conv1_channels
            ),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),

            # Sequence:
            # 50 -> 25


            nn.Conv1d(
                in_channels=conv1_channels,
                out_channels=conv2_channels,
                kernel_size=kernel_size,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                conv2_channels
            ),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            )

            # Sequence:
            # 25 -> 12
        )


        # -------------------------------------------------
        # CLASSIFIER
        # -------------------------------------------------

        # Final shape:
        #
        # batch
        # 128 channels
        # 12 sequence length
        #
        # Flatten = 128 * 12 = 1536

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                conv2_channels * 12,
                linear1
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),


            nn.Linear(
                linear1,
                linear2
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),


            # Output:
            # HOLD = 0
            # BUY  = 1
            # SELL = 2

            nn.Linear(
                linear2,
                3
            )
        )


    def forward(self, x):

        # Original:
        # (batch, 50, num_features)

        x = x.permute(
            0,
            2,
            1
        )

        # After:
        # (batch, num_features, 50)

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================================================
# 12. CREATE MODEL
# =========================================================

num_features = X_train.shape[2]

model = MyCNN(
    num_features=num_features
)


print("\nModel:")
print(model)


# =========================================================
# 13. LOSS FUNCTION
# =========================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)


# =========================================================
# 14. OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# =========================================================
# 15. TRAIN MODEL
# =========================================================

print("\nStarting Training...\n")


for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch_features, batch_labels in train_loader:

        # ---------------------------------------------
        # Forward Pass
        # ---------------------------------------------

        outputs = model(
            batch_features
        )


        # ---------------------------------------------
        # Weighted Loss
        # ---------------------------------------------

        loss = criterion(
            outputs,
            batch_labels
        )


        # ---------------------------------------------
        # Reset Gradients
        # ---------------------------------------------

        optimizer.zero_grad()


        # ---------------------------------------------
        # Backpropagation
        # ---------------------------------------------

        loss.backward()


        # ---------------------------------------------
        # Update Weights
        # ---------------------------------------------

        optimizer.step()


        total_loss += loss.item()


    average_loss = (
        total_loss /
        len(train_loader)
    )


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {average_loss:.4f}"
    )


# =========================================================
# 16. TESTING
# =========================================================

model.eval()

total = 0
correct = 0


with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        outputs = model(
            batch_features
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += batch_labels.size(0)

        correct += (
            predicted == batch_labels
        ).sum().item()


accuracy = (
    correct /
    total
) * 100


print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)


# =========================================================
# 17. COLLECT PREDICTIONS
# =========================================================

model.eval()

all_labels = []
all_predictions = []


with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        outputs = model(
            batch_features
        )

        _, predicted = torch.max(
            outputs,
            1
        )


        all_labels.extend(
            batch_labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# =========================================================
# 18. RAW ACCURACY
# =========================================================

raw_accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print(
    f"Raw Accuracy: "
    f"{raw_accuracy * 100:.2f}%"
)


# =========================================================
# 19. BALANCED ACCURACY
# =========================================================

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_predictions
)


print(
    f"Balanced Accuracy: "
    f"{balanced_accuracy * 100:.2f}%"
)


# =========================================================
# 20. CLASSIFICATION REPORT
# =========================================================

print(
    "\nClassification Report:\n"
)


print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=[
            "HOLD",
            "BUY",
            "SELL"
        ],
        digits=4
    )
)


# =========================================================
# 21. SAVE MODEL
# =========================================================

torch.save(
    model.state_dict(),
    "best_gold_model.pt"
)

print(
    "\nModel saved as: best_gold_model.pt"
)