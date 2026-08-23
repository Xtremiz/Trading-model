import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

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

X = df.drop(
    columns=["signal"]
).values

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

    X_sequences.append(
        X[i - sequence_length:i]
    )

    y_sequences.append(
        y[i]
    )


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


print("\nClass Weights:")

for cls, weight in zip(
    classes,
    class_weights
):

    print(
        f"{cls}: {weight.item():.4f}"
    )


# ==========================================
# OPTUNA CNN
# ==========================================

class OptunaCNN(nn.Module):

    def __init__(
        self,
        num_features,
        trial
    ):

        super().__init__()


        # ==================================
        # CONVOLUTION SETTINGS
        # ==================================

        n_conv_layers = trial.suggest_int(
            "n_conv_layers",
            1,
            4
        )


        conv_layers = []

        in_channels = num_features


        for i in range(
            n_conv_layers
        ):

            out_channels = trial.suggest_categorical(
                f"conv_{i}_channels",
                [
                    16,
                    32,
                    64,
                    128,
                    256
                ]
            )


            kernel_size = trial.suggest_categorical(
                f"conv_{i}_kernel",
                [
                    3,
                    5,
                    7
                ]
            )


            use_batchnorm = trial.suggest_categorical(
                f"conv_{i}_batchnorm",
                [
                    True,
                    False
                ]
            )


            conv_layers.append(

                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2
                )

            )


            if use_batchnorm:

                conv_layers.append(
                    nn.BatchNorm1d(
                        out_channels
                    )
                )


            conv_layers.append(
                nn.ReLU()
            )


            # ==================================
            # POOLING
            # ==================================

            use_pool = trial.suggest_categorical(
                f"conv_{i}_pool",
                [
                    True,
                    False
                ]
            )


            if use_pool:

                pool_size = trial.suggest_categorical(
                    f"conv_{i}_pool_size",
                    [
                        2,
                        3
                    ]
                )

                conv_layers.append(
                    nn.MaxPool1d(
                        kernel_size=pool_size,
                        stride=pool_size
                    )
                )


            in_channels = out_channels


        self.features = nn.Sequential(
            *conv_layers
        )


        # ==================================
        # GLOBAL POOLING
        # ==================================

        self.global_pool = nn.AdaptiveAvgPool1d(
            1
        )


        # ==================================
        # LINEAR HIDDEN LAYERS
        # ==================================

        n_linear_layers = trial.suggest_int(
            "n_linear_layers",
            1,
            4
        )


        linear_layers = []

        linear_input = in_channels


        for i in range(
            n_linear_layers
        ):

            hidden_size = trial.suggest_categorical(
                f"linear_{i}_size",
                [
                    32,
                    64,
                    128,
                    256,
                    512
                ]
            )


            linear_layers.append(
                nn.Linear(
                    linear_input,
                    hidden_size
                )
            )


            activation = trial.suggest_categorical(
                f"linear_{i}_activation",
                [
                    "relu",
                    "gelu",
                    "silu"
                ]
            )


            if activation == "relu":

                linear_layers.append(
                    nn.ReLU()
                )

            elif activation == "gelu":

                linear_layers.append(
                    nn.GELU()
                )

            else:

                linear_layers.append(
                    nn.SiLU()
                )


            use_dropout = trial.suggest_categorical(
                f"linear_{i}_dropout",
                [
                    True,
                    False
                ]
            )


            if use_dropout:

                dropout = trial.suggest_float(
                    f"linear_{i}_dropout_rate",
                    0.1,
                    0.5
                )

                linear_layers.append(
                    nn.Dropout(
                        dropout
                    )
                )


            linear_input = hidden_size


        # ==================================
        # OUTPUT
        # ==================================

        linear_layers.append(
            nn.Linear(
                linear_input,
                3
            )
        )


        self.classifier = nn.Sequential(
            *linear_layers
        )


    def forward(
        self,
        x
    ):

        # (batch, 50, features)

        x = x.permute(
            0,
            2,
            1
        )

        # (batch, features, 50)

        x = self.features(x)

        # (batch, channels, sequence)

        x = self.global_pool(x)

        # (batch, channels, 1)

        x = x.squeeze(-1)

        # (batch, channels)

        x = self.classifier(x)

        return x


# ==========================================
# OPTUNA OBJECTIVE
# ==========================================

def objective(trial):


    # ==================================
    # BATCH SIZE
    # ==================================

    batch_size = trial.suggest_categorical(
        "batch_size",
        [
            16,
            32,
            64,
            128
        ]
    )


    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True
    )


    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True
    )


    # ==================================
    # MODEL
    # ==================================

    model = OptunaCNN(
        num_features=X_train.shape[2],
        trial=trial
    ).to(device)


    # ==================================
    # LEARNING RATE
    # ==================================

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        3e-3,
        log=True
    )


    # ==================================
    # WEIGHT DECAY
    # ==================================

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-7,
        1e-3,
        log=True
    )
    epochs = trial.suggest_int(
        "epochs",20,50,step=10
    )

    # ==================================
    # OPTIMIZER
    # ==================================

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        [
            "Adam",
            "AdamW",
            "RMSprop"
        ]
    )


    if optimizer_name == "Adam":

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    elif optimizer_name == "AdamW":

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    else:

        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


    # ==================================
    # LOSS
    # ==================================

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # ==================================
    # EPOCHS
    # ==================================

    


    # ==================================
    # TRAINING
    # ==================================

    for epoch in range(
        epochs
    ):

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


        # ==================================
        # VALIDATION
        # ==================================

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


                predictions = torch.argmax(
                    outputs,
                    dim=1
                )


                all_labels.extend(
                    batch_labels.numpy()
                )

                all_predictions.extend(
                    predictions.cpu().numpy()
                )


        balanced_accuracy = balanced_accuracy_score(
            all_labels,
            all_predictions
        )


        # ==================================
        # OPTUNA PRUNING
        # ==================================

        trial.report(
            balanced_accuracy,
            epoch
        )


        if trial.should_prune():

            raise optuna.TrialPruned()


    return balanced_accuracy


# ==========================================
# CREATE STUDY
# ==========================================

study = optuna.create_study(
    direction="maximize",
    study_name="Gold_CNN_Optimization"
)


# ==========================================
# START OPTIMIZATION
# ==========================================

study.optimize(
    objective,
    n_trials=50
)


# ==========================================
# BEST RESULT
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
    "\nBest Balanced Accuracy:",
    study.best_value
)


print(
    "\nBest Hyperparameters:"
)


for key, value in study.best_params.items():

    print(
        f"{key}: {value}"
    )