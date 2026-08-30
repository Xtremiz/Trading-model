import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch = 32

df = pd.read_csv(r"F:\Git-Hub\Trading model\block-2, new idea\goldbuydata.csv")
df = df.dropna()

signal_mapping = {"hold": 0, "buy": 1}
df["signal"] = df["signal"].map(signal_mapping)

classes = np.array([0,1])
weights = [1,]
class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

feature_names = df.drop(columns=["signal"]).columns.tolist()

X = df[feature_names].values
y = df["signal"].values

sequence = 50
X_sequence = []
y_sequence = []
for i in range(sequence, len(X)):
    X_sequence.append(X[i-sequence:i])
    y_sequence.append(y[i])

X_sequence = np.array(X_sequence)
y_sequence = np.array(y_sequence)

# FIX: train_test_split ka return order galat tha
X_train, X_test, y_train, y_test = train_test_split(
    X_sequence, y_sequence, test_size=0.2, random_state=42, shuffle=True
)


class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.data = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])


train_data = CustomDataset(X_train, y_train)
test_data = CustomDataset(X_test, y_test)

train_loader = DataLoader(train_data, batch, shuffle=True)
test_loader = DataLoader(test_data, batch, shuffle=False)


class MyCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(in_channels=num_features, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 12, 64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)   # tumhare paas sirf 2 classes hain: hold, buy
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)   # FIX: "premute" ko "permute" kiya
        x = self.features(x)
        x = self.classifier(x)
        return x


num_features = X_train.shape[2]

# CUDA FIX: model ko device pe bhejo
model = MyCNN(num_features=num_features).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for i in range(100):
    model.train()   # FIX: () lagana zaroori hai warna training mode set nahi hoga
    total_loss = 0
    for batch_features, batch_labels in train_loader:
        # CUDA FIX: har batch ko bhi device pe bhejo
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        output = model(batch_features)
        loss = criterion(output, batch_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    average_loss = total_loss / len(train_loader)
    print(f"Epoch [{i + 1}/100] Loss: {average_loss:.4f}")


model.eval()
total = 0
correct = 0
with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        output = model(batch_features)
        _, predicted = torch.max(output, 1)
        total += batch_labels.size(0)
        correct += (predicted == batch_labels).sum().item()

accuracy = (correct / total) * 100
print(f"\nTest Accuracy: {accuracy:.2f}%")


all_labels = []
all_predictions = []

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        batch_features = batch_features.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)

        all_labels.extend(batch_labels.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())


raw_accuracy = accuracy_score(all_labels, all_predictions)
print(f"Raw Accuracy: {raw_accuracy * 100:.2f}%")

balanced_accuracy = balanced_accuracy_score(all_labels, all_predictions)
print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")

print("\nClassification Report:\n")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["HOLD", "BUY"],   
        digits=4
    )
)


torch.save(model.state_dict(), "best_gold_model.pt")
print("\nModel saved as: best_gold_model.pt")