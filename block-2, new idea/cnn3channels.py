import copy
import channells
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


BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.0001
PATIENCE = 40
SEQUENCE_LENGTH = 100

HOLD_WEIGHT = 1
BUY_WEIGHT = 1.7
SELL_WEIGHT = 1.2

CLASS_WEIGHTS = torch.tensor(
    [HOLD_WEIGHT, BUY_WEIGHT, SELL_WEIGHT],
    dtype=torch.float32
).to(device)


df = pd.read_csv(
    r"F:\Git-Hub\Trading model\block-2, new idea\gold_target_40_30_30.csv"
)

df = df.dropna().reset_index(drop=True)

signal_mapping = {
    0: 0,
    1: 1,
    -1: 2
}

df["target"] = df["target"].map(signal_mapping)
channel_1_features = channells.channel_1
channel_2_features = channells.channel_2
channel_3_features = channells.channel_3


feature_names = df.drop(columns=["target"]).columns.tolist()

X = df[feature_names].values.astype(np.float32)
y = df["target"].values.astype(np.int64)


feature_index = {
    feature: i
    for i, feature in enumerate(feature_names)
}


X_channel_1 = np.zeros_like(X, dtype=np.float32)
X_channel_2 = np.zeros_like(X, dtype=np.float32)
X_channel_3 = np.zeros_like(X, dtype=np.float32)


for feature in channel_1_features:
    idx = feature_index[feature]
    X_channel_1[:, idx] = X[:, idx]


for feature in channel_2_features:
    idx = feature_index[feature]
    X_channel_2[:, idx] = X[:, idx]


for feature in channel_3_features:
    idx = feature_index[feature]
    X_channel_3[:, idx] = X[:, idx]


X_sequence = []
y_sequence = []

for i in range(SEQUENCE_LENGTH, len(X)):

    seq_1 = X_channel_1[i - SEQUENCE_LENGTH:i]
    seq_2 = X_channel_2[i - SEQUENCE_LENGTH:i]
    seq_3 = X_channel_3[i - SEQUENCE_LENGTH:i]

    sequence = np.stack(
        [seq_1, seq_2, seq_3],
        axis=0
    )

    X_sequence.append(sequence)
    y_sequence.append(y[i])


X_sequence = np.array(
    X_sequence,
    dtype=np.float32
)

y_sequence = np.array(
    y_sequence,
    dtype=np.int64
)


print("\nFull sequence shape:")
print(X_sequence.shape)

print("\nClass distribution:")
print("HOLD:", np.sum(y_sequence == 0))
print("BUY :", np.sum(y_sequence == 1))
print("SELL:", np.sum(y_sequence == 2))


total = len(X_sequence)

train_end = int(total * 0.70)
val_end = int(total * 0.85)

X_train = X_sequence[:train_end]
y_train = y_sequence[:train_end]

X_val = X_sequence[train_end:val_end]
y_val = y_sequence[train_end:val_end]

X_test = X_sequence[val_end:]
y_test = y_sequence[val_end:]


print("\nSplit sizes:")
print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)


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

    def __getitem__(self, index):

        return (
            self.data[index],
            self.labels[index]
        )


train_data = CustomDataset(
    X_train,
    y_train
)

val_data = CustomDataset(
    X_val,
    y_val
)

test_data = CustomDataset(
    X_test,
    y_test
)


train_loader = DataLoader(
    train_data,
    BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_data,
    BATCH_SIZE,
    shuffle=False
)


class MyCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv2d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Dropout2d(0.20)
        )


        self.pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )


        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                128,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                3
            )
        )


    def forward(self, x):

        x = self.features(x)

        x = self.pool(x)

        x = self.classifier(x)

        return x


model = MyCNN().to(device)


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


def evaluate_model(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_features, batch_labels in loader:

            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            outputs = model(batch_features)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                batch_labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )


    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)


    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    balanced_acc = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0
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


        output = model(
            batch_features
        )


        loss = criterion(
            output,
            batch_labels
        )


        loss.backward()


        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )


        optimizer.step()


        total_loss += loss.item()


    average_loss = (
        total_loss /
        len(train_loader)
    )


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
    ) = evaluate_model(
        model,
        val_loader
    )


    scheduler.step(
        val_macro_f1
    )


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

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        best_epoch = epoch + 1

        epochs_without_improvement = 0

        print(
            f"  --> New best model! "
            f"Macro F1: {best_macro_f1:.4f}"
        )

    else:

        epochs_without_improvement += 1


    if epochs_without_improvement >= PATIENCE:

        print(
            "\nEarly stopping triggered."
        )

        break


if best_model_weights is not None:

    torch.save(
        best_model_weights,
        "best_gold_model.pt"
    )

    print(
        f"\nBest model saved "
        f"(Epoch {best_epoch}) "
        f"with Macro F1: {best_macro_f1:.4f}"
    )

    model.load_state_dict(
        best_model_weights
    )

else:

    print(
        "\nWarning: No best model found."
    )


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
) = evaluate_model(
    model,
    test_loader
)


print("\n" + "=" * 50)
print("FINAL TEST RESULTS (Best Model)")
print("=" * 50)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Test Balanced Accuracy: "
    f"{test_balanced_acc * 100:.2f}%"
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
    f"{test_hold_f1:.4f}"
)

print(
    f"BUY F1: "
    f"{test_buy_f1:.4f}"
)

print(
    f"SELL F1: "
    f"{test_sell_f1:.4f}"
)


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


cm = confusion_matrix(
    test_labels,
    test_predictions
)


print("\nConfusion Matrix:")

print("              Predicted")
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

