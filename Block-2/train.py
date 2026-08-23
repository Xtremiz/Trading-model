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


df = pd.read_csv(r"C:\Users\fozan\OneDrive\Desktop\Code\Data\Block=2 prep\goldthresold0.5.csv")
df = df.dropna()

signal_mapping = {
    "hold": 0,
    "buy": 1,
    "sell": 2
}

df["signal"] = df["signal"].map(signal_mapping)


# ==========================================
# FEATURES AND LABELS
# ==========================================

X = df.drop(columns=["signal"]).values
y = df["signal"].values


# ==========================================
# CREATE SEQUENCES
# Example: previous 50 candles -> predict signal
# ==========================================

sequence_length = 50

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


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    random_state=42,
    shuffle=True
)


# ==========================================
# DATASET
# ==========================================

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

        return self.data[idx], self.labels[idx]


# ==========================================
# DATASETS
# ==========================================

train_data = CustomDataset(
    X_train,
    y_train
)

test_data = CustomDataset(
    X_test,
    y_test
)


# ==========================================
# DATALOADERS
# ==========================================

train_loader = DataLoader(
    train_data,
    batch_size=32,
    shuffle=True
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False
)


# ==========================================
# CNN MODEL
# ==========================================

class MyCNN(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.features = nn.Sequential(

            # Input:
            # (batch, num_features, 50)

            nn.Conv1d(
                in_channels=num_features,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(32),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),

            # (batch, 32, 25)

            nn.Conv1d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            )

            # Final:
            # (batch, 64, 12)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                64 * 12,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.3),


            nn.Linear(
                128,
                64
            ),

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

        # Before:
        # (batch, 50, 13)

        x = x.permute(0, 2, 1)

        # After:
        # (batch, 13, 50)

        x = self.features(x)

        x = self.classifier(x)

        return x


# ==========================================
# CREATE MODEL
# ==========================================

num_features = X_train.shape[2]

model = MyCNN(
    num_features=num_features
)


# ==========================================
# TRAINING SETTINGS
# ==========================================

learning_rate = 0.001
epochs = 50

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)


# ==========================================
# TRAIN MODEL
# ==========================================

for epoch in range(epochs):

    model.train()

    total_loss = 0

    for batch_features, batch_labels in train_loader:

        # Forward pass
        outputs = model(batch_features)

        # Loss
        loss = criterion(
            outputs,
            batch_labels
        )

        # Reset gradients
        optimizer.zero_grad()

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item()


    print(
        f"Epoch [{epoch + 1}/{epochs}] "
        f"Loss: {total_loss / len(train_loader):.4f}"
    )


# ==========================================
# TESTING
# ==========================================

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


accuracy = correct / total * 100


print(
    f"\nTest Accuracy: {accuracy:.2f}%"
)


model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for batch_features, batch_labels in test_loader:

        outputs = model(batch_features)

        _, predicted = torch.max(outputs, 1)

        all_labels.extend(
            batch_labels.cpu().numpy()
        )

        all_predictions.extend(
            predicted.cpu().numpy()
        )


# ==========================================
# RAW ACCURACY
# ==========================================

raw_accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print(
    f"Raw Accuracy: {raw_accuracy * 100:.2f}%"
)


# ==========================================
# BALANCED / ACTUAL ACCURACY
# ==========================================

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_predictions
)

print(
    f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%"
)


# ==========================================
# CLASSIFICATION REPORT
# ==========================================

print("\nClassification Report:\n")

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