import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset,WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score
)
import pandas as pd
import numpy as np
import optuna


# ==========================================
# DEVICE
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)

if torch.cuda.is_available():
    print(
        "GPU:",
        torch.cuda.get_device_name(0)
    )


# ==========================================
# LOAD DATA
# ==========================================

df = pd.read_csv(
    r"F:\Git-Hub\Trading model\Block-2\goldthresold0.4_norm.csv"
)

df = df.dropna()


signal_mapping = {
    "hold": 0,
    "buy": 1,
    "sell": 2
}

df["signal"] = df["signal"].map(
    signal_mapping
)


# ==========================================
# FEATURES / LABELS
# ==========================================

feature_names = df.drop(
    columns=["signal"]
).columns.tolist()

X = df[feature_names].values

y = df["signal"].values


# ==========================================
# SEQUENCES
# ==========================================

sequence_length = 50

X_sequences = []
y_sequences = []

for i in range(
    sequence_length,
    len(X)
):

    X_sequences.append(X[i - sequence_length:i])
    y_sequences.append(y[i])


X_sequences = np.array(
    X_sequences
)

y_sequences = np.array(
    y_sequences
)


print(
    "X Shape:",
    X_sequences.shape
)

print(
    "y Shape:",
    y_sequences.shape
)


# ==========================================
# TRAIN / TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_sequences,
    y_sequences,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

class_counts = np.bincount(y_train)

class_weights = 1.0 / class_counts
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(
    weights=torch.DoubleTensor(sample_weights),
    num_samples=len(y_train),
    replacement=True
)

# ==========================================
# CUSTOM DATASET
# ==========================================

class CustomDataset(Dataset):

    def __init__(
        self,
        features,
        labels
    ):

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


    def __getitem__(
        self,
        idx
    ):

        return (
            self.data[idx],
            self.labels[idx]
        )


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
# CLASS WEIGHTS
# ==========================================

classes = np.array([
    0,
    1,
    2
])

weights = [
    0.7,
    2.1,
    2.5
]

class_weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)




# ==========================================
# OPTUNA CNN
# ==========================================

# ==========================================
# OPTUNA OBJECTIVE
# ==========================================

def objective(trial):

    # ======================================
    # HYPERPARAMETERS
    # ======================================

    learning_rate = trial.suggest_float(
        "learning_rate",
        2.5e-3,
        9e-3,
        log=True
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [32, 64]
    )

    epochs = trial.suggest_categorical(
        "epochs",
        [50]
    )

    # ======================================
    # CLASS WEIGHTS
    # ======================================

    hold_weight = trial.suggest_float(
        "hold_weight",
        0.6,
        0.8,
        step=0.05
    )

    buy_weight = trial.suggest_float(
        "buy_weight",
        1.9,
        2.2,
        step=0.05
    )

    sell_weight = trial.suggest_float(
        "sell_weight",
        2.7,
        3.2,
        step=0.05
    )

    class_weights = torch.tensor(
        [
            hold_weight,
            buy_weight,
            sell_weight
        ],
        dtype=torch.float32,
        device=device
    )

    # ======================================
    # CNN HYPERPARAMETERS
    # ======================================

    conv1_channels = trial.suggest_categorical(
        "conv1_channels",
        [ 32, 64]
    )

    conv2_channels = trial.suggest_categorical(
        "conv2_channels",
        [64, 128]
    )

    kernel_size = trial.suggest_categorical(
        "kernel_size",
        [3, 5]
    )

    dropout = trial.suggest_categorical(
        "dropout",
       [0.1, 0.2, 0.3]
    )

    linear1 = trial.suggest_categorical(
        "linear1",
        [ 128, 256]
    )

    linear2 = trial.suggest_categorical(
        "linear2",
        [64, 128]
    )

    # ======================================
    # DATA LOADERS
    # ======================================

    train_data = CustomDataset(
        X_train,
        y_train
    )

    test_data = CustomDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )

    # ======================================
    # CNN MODEL
    # ======================================

    class OptunaCNN(nn.Module):

        def __init__(self, num_features):

            super().__init__()

            self.features = nn.Sequential(

                nn.Conv1d(
                    in_channels=num_features,
                    out_channels=conv1_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),

                nn.ReLU(),

                nn.BatchNorm1d(
                    conv1_channels
                ),

                nn.MaxPool1d(
                    kernel_size=2,
                    stride=2
                ),

                nn.Conv1d(
                    in_channels=conv1_channels,
                    out_channels=conv2_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                ),

                nn.ReLU(),

                nn.BatchNorm1d(
                    conv2_channels
                ),

                nn.MaxPool1d(
                    kernel_size=2,
                    stride=2
                )
            )

            # 50 -> 25 -> 12
            flattened_size = (
                conv2_channels * 12
            )

            self.classifier = nn.Sequential(

                nn.Flatten(),

                nn.Linear(
                    flattened_size,
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

                nn.Linear(
                    linear2,
                    3
                )
            )

        def forward(self, x):

            # (batch, 50, features)

            x = x.permute(
                0,
                2,
                1
            )

            # (batch, features, 50)

            x = self.features(x)

            x = self.classifier(x)

            return x

    # ======================================
    # CREATE MODEL
    # ======================================

    num_features = X_train.shape[2]

    model = OptunaCNN(
        num_features
    ).to(device)

    # ======================================
    # LOSS
    # ======================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ======================================
    # OPTIMIZER
    # ======================================

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        [
            "AdamW",
            "RMSprop"
        ]
    )

   

    if optimizer_name == "AdamW":

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )

    else:

        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate
        )

    # ======================================
    # TRAINING
    # ======================================

    for epoch in range(epochs):

        model.train()

        total_loss = 0

        for batch_features, batch_labels in train_loader:

            batch_features = batch_features.to(
                device,
                non_blocking=True
            )

            batch_labels = batch_labels.to(
                device,
                non_blocking=True
            )

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

            total_loss += loss.item()

    # ======================================
    # VALIDATION / TEST
    # ======================================

    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for batch_features, batch_labels in test_loader:

            batch_features = batch_features.to(
                device,
                non_blocking=True
            )

            outputs = model(
                batch_features
            )

            _, predicted = torch.max(
                outputs,
                1
            )

            all_labels.extend(
                batch_labels.numpy()
            )

            all_predictions.extend(
                predicted.cpu().numpy()
            )

    # ======================================
    # BALANCED ACCURACY
    # ======================================

    balanced_accuracy = balanced_accuracy_score(
        all_labels,
        all_predictions
    )

    raw_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    print(
        f"\nTrial {trial.number}"
    )

    print(
        f"Raw Accuracy: "
        f"{raw_accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{balanced_accuracy * 100:.2f}%"
    )

    print(
        f"Weights: "
        f"HOLD={hold_weight:.1f}, "
        f"BUY={buy_weight:.1f}, "
        f"SELL={sell_weight:.1f}"
    )

    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if balanced_accuracy > objective.best_score:

        objective.best_score = balanced_accuracy

        torch.save(
    {
        # ==========================================
        # MODEL
        # ==========================================

        "model_state_dict":
            model.state_dict(),

        # ==========================================
        # INPUT INFORMATION
        # ==========================================

        "input_size":
            num_features,

        "feature_names":
            feature_names,

        "sequence_length":
            sequence_length,

        # ==========================================
        # CLASS INFORMATION
        # ==========================================

        "class_mapping":
            signal_mapping,

        "class_weights":
            [
                hold_weight,
                buy_weight,
                sell_weight
            ],

        # ==========================================
        # CNN ARCHITECTURE
        # ==========================================

        "conv1_channels":
            conv1_channels,

        "conv2_channels":
            conv2_channels,

        "kernel_size":
            kernel_size,

        "dropout":
            dropout,

        "linear1":
            linear1,

        "linear2":
            linear2,

        # ==========================================
        # TRAINING PARAMETERS
        # ==========================================

        "learning_rate":
            learning_rate,

        "batch_size":
            batch_size,

        "epochs":
            epochs,

        "optimizer":
            optimizer_name,

        # ==========================================
        # OPTUNA PARAMETERS
        # ==========================================

        "best_hyperparameters":
            trial.params,

        # ==========================================
        # PERFORMANCE
        # ==========================================

        "validation_balanced_accuracy":
            balanced_accuracy,

        "validation_accuracy":
            raw_accuracy,

        "best_epoch":
            epoch + 1

    },

    "best_cnn_model.pt")

        print(
            "\n🔥 NEW BEST MODEL SAVED!"
        )

        print(
            f"Balanced Accuracy: "
            f"{balanced_accuracy * 100:.2f}%"
        )

    return balanced_accuracy


# ==========================================
# INITIAL BEST SCORE
# ==========================================

objective.best_score = -float("inf")


# ==========================================
# OPTUNA STUDY
# ==========================================

study = optuna.create_study(
    direction="maximize"
)


study.optimize(
    objective,
    n_trials=60
)


# ==========================================
# FINAL RESULTS
# ==========================================

print(
    "\n================================"
)

print(
    "OPTUNA FINISHED"
)

print(
    "================================"
)

print(
    f"Best Balanced Accuracy: "
    f"{study.best_value * 100:.2f}%"
)


print(
    "\nBEST PARAMETERS:"
)


for key, value in study.best_params.items():

    print(
        f"{key}: {value}"
    )


print(
    "\nBest model saved as:"
)

print(
    "best_cnn_model.pt"
)