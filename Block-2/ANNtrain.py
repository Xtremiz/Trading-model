import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

import optuna

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# 1. SETTINGS
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("DEVICE")
print("========================================")
print("Using device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))


# ============================================================
# 2. LOAD DATA
# ============================================================

DATA_PATH = r"F:\Git-Hub\Trading model\Block-2\goldthresold0.4_norm.csv"

df = pd.read_csv(DATA_PATH)

print("\nOriginal shape:", df.shape)

# Remove NaN / inf
df = df.replace(
    [np.inf, -np.inf],
    np.nan
)

df = df.dropna().reset_index(drop=True)

print("After dropna:", df.shape)


# ============================================================
# 3. SIGNAL MAPPING
# ============================================================

# Supports:
# HOLD = 0
# BUY  = 1
# SELL = 2
#
# Also supports:
# -1 = SELL

signal_mapping = {
    "hold": 0,
    "buy": 1,
    "sell": 2,

    "HOLD": 0,
    "BUY": 1,
    "SELL": 2,

    -1: 2,
    0: 0,
    1: 1,
    2: 2
}

df["signal"] = df["signal"].map(signal_mapping)

if df["signal"].isna().any():

    print("\nUnique problematic signals:")

    print(
        df.loc[
            df["signal"].isna(),
            "signal"
        ].unique()
    )

    raise ValueError(
        "signal column mein unknown labels hain."
    )

df["signal"] = df["signal"].astype(int)


# ============================================================
# 4. FEATURES / LABELS
# ============================================================

feature_names = [
    col
    for col in df.columns
    if col != "signal"
]

X = df[feature_names].values.astype(
    np.float32
)

y = df["signal"].values.astype(
    np.int64
)

print("\n========================================")
print("DATA")
print("========================================")

print("Features:", X.shape)
print("Labels:", y.shape)

unique_labels, label_counts = np.unique(
    y,
    return_counts=True
)

print("\nFull dataset distribution:")

for label, count in zip(
    unique_labels,
    label_counts
):

    name = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }[label]

    print(
        f"{name}: {count}"
    )


# ============================================================
# 5. CHRONOLOGICAL SPLIT
# ============================================================

# Trading data hai, isliye random shuffle nahi karna.

n = len(X)

train_end = int(
    n * 0.70
)

val_end = int(
    n * 0.85
)

X_train = X[:train_end]
y_train = y[:train_end]

X_val = X[train_end:val_end]
y_val = y[train_end:val_end]

X_test = X[val_end:]
y_test = y[val_end:]


print("\n========================================")
print("SPLIT")
print("========================================")

print(
    "Train:",
    X_train.shape,
    y_train.shape
)

print(
    "Validation:",
    X_val.shape,
    y_val.shape
)

print(
    "Test:",
    X_test.shape,
    y_test.shape
)


# ============================================================
# 6. DATASET
# ============================================================

class CustomDataset(Dataset):

    def __init__(
        self,
        features,
        labels
    ):

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


train_dataset = CustomDataset(
    X_train,
    y_train
)

val_dataset = CustomDataset(
    X_val,
    y_val
)

test_dataset = CustomDataset(
    X_test,
    y_test
)


# ============================================================
# 7. CLASS WEIGHTS
# ============================================================

# Base balanced weights are only printed for reference.
# Optuna will tune the FINAL weights:
# HOLD -> 0.30 to 0.70
# BUY  -> 2.20 to 4.00
# SELL -> 2.60 to 4.00
#
# These are deliberately kept as explicit ranges because you
# requested HOLD maximum 0.7, BUY minimum 2.2 and SELL minimum 2.6.

classes = np.array([0, 1, 2])

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=y_train
)

print("\n========================================")
print("BASE BALANCED CLASS WEIGHTS")
print("========================================")
print(f"HOLD: {balanced_weights[0]:.4f}")
print(f"BUY : {balanced_weights[1]:.4f}")
print(f"SELL: {balanced_weights[2]:.4f}")


def make_class_weights(params):
    """Create the Optuna-selected class-weight tensor."""
    weights = np.array([
        params["weight_hold"],
        params["weight_buy"],
        params["weight_sell"]
    ], dtype=np.float32)

    return torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )


# ============================================================
# 8. ANN MODEL
# ============================================================

class MyNN(nn.Module):

    def __init__(
        self,
        num_features,
        hidden_layers,
        dropout
    ):

        super().__init__()

        layers = []

        input_size = num_features

        # ----------------------------------------
        # Dynamic hidden layers
        # ----------------------------------------

        for hidden_size in hidden_layers:

            layers.append(
                nn.Linear(
                    input_size,
                    hidden_size
                )
            )

            layers.append(
                nn.BatchNorm1d(
                    hidden_size
                )
            )

            layers.append(
                nn.ReLU()
            )

            layers.append(
                nn.Dropout(
                    dropout
                )
            )

            input_size = hidden_size

        # ----------------------------------------
        # Final classification layer
        # ----------------------------------------

        layers.append(
            nn.Linear(
                input_size,
                3
            )
        )

        self.model = nn.Sequential(
            *layers
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# 9. CREATE MODEL FROM PARAMS
# ============================================================

def create_model(
    num_features,
    hidden_layers,
    dropout
):

    model = MyNN(
        num_features=num_features,
        hidden_layers=hidden_layers,
        dropout=dropout
    )

    return model.to(device)


# ============================================================
# 10. TRAIN ONE MODEL
# ============================================================

def train_model(
    params,
    epochs,
    patience,
    trial=None,
    verbose=False
):

    # ----------------------------------------
    # DataLoaders
    # ----------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        shuffle=True,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        shuffle=False,
        pin_memory=torch.cuda.is_available()
    )

    # ----------------------------------------
    # Hidden layer list
    # ----------------------------------------

    hidden_layers = []

    for i in range(
        params["n_layers"]
    ):

        hidden_layers.append(
            params[f"hidden_{i}"]
        )

    # ----------------------------------------
    # Model
    # ----------------------------------------

    model = create_model(
        num_features=X_train.shape[1],
        hidden_layers=hidden_layers,
        dropout=params["dropout"]
    )

    # ----------------------------------------
    # Loss
    # ----------------------------------------

    # ----------------------------------------
    # Optuna-selected class weights
    # ----------------------------------------

    class_weights = make_class_weights(params)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    # ----------------------------------------
    # Optuna-selected optimizer
    # ----------------------------------------

    optimizer_name = params["optimizer"]

    if optimizer_name == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"]
        )

    elif optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"]
        )

    elif optimizer_name == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=params["lr"],
            weight_decay=params["weight_decay"]
        )

    else:
        raise ValueError(
            f"Unknown optimizer: {optimizer_name}"
        )

    # ----------------------------------------
    # Best values
    # ----------------------------------------

    best_balanced_acc = -float("inf")

    best_accuracy = 0.0

    best_f1 = 0.0

    best_epoch = 0

    best_model_state = None

    wait = 0

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(epochs):

        model.train()

        total_train_loss = 0.0

        for (
            batch_features,
            batch_labels
        ) in train_loader:

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

            # Safety check
            if not torch.isfinite(loss):

                print(
                    "\nWARNING: NaN/Inf loss detected."
                )

                return (
                    None,
                    0.0,
                    0.0,
                    0.0,
                    0
                )

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0
            )

            optimizer.step()

            total_train_loss += loss.item()

        train_loss = (
            total_train_loss /
            max(len(train_loader), 1)
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        total_val_loss = 0.0

        val_true = []

        val_pred = []

        with torch.no_grad():

            for (
                batch_features,
                batch_labels
            ) in val_loader:

                batch_features = batch_features.to(
                    device,
                    non_blocking=True
                )

                batch_labels = batch_labels.to(
                    device,
                    non_blocking=True
                )

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
            total_val_loss /
            max(len(val_loader), 1)
        )

        val_accuracy = accuracy_score(
            val_true,
            val_pred
        )

        val_balanced_acc = balanced_accuracy_score(
            val_true,
            val_pred
        )

        val_macro_f1 = f1_score(
            val_true,
            val_pred,
            average="macro",
            zero_division=0
        )

        # ====================================================
        # OPTUNA PRUNING
        # ====================================================

        if trial is not None:

            trial.report(
                val_balanced_acc,
                epoch
            )

            if trial.should_prune():

                raise optuna.TrialPruned()

        # ====================================================
        # BEST MODEL
        # ====================================================

        if (
            val_balanced_acc >
            best_balanced_acc
        ):

            best_balanced_acc = (
                val_balanced_acc
            )

            best_accuracy = (
                val_accuracy
            )

            best_f1 = (
                val_macro_f1
            )

            best_epoch = (
                epoch + 1
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )

            wait = 0

        else:

            wait += 1

        # ====================================================
        # PRINT
        # ====================================================

        if verbose:

            print(
                f"Epoch "
                f"[{epoch + 1}/{epochs}] | "
                f"Train Loss: "
                f"{train_loss:.4f} | "
                f"Val Loss: "
                f"{val_loss:.4f} | "
                f"Val Acc: "
                f"{val_accuracy * 100:.2f}% | "
                f"Val Balanced: "
                f"{val_balanced_acc * 100:.2f}% | "
                f"Macro F1: "
                f"{val_macro_f1:.4f}"
            )

        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if wait >= patience:

            if verbose:

                print(
                    f"\nEarly stopping at "
                    f"epoch {epoch + 1}"
                )

            break

    return (
        best_model_state,
        best_balanced_acc,
        best_accuracy,
        best_f1,
        best_epoch
    )


# ============================================================
# 11. OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # ----------------------------------------
    # Basic hyperparameters
    # ----------------------------------------

    params = {

        "lr": trial.suggest_float(
            "lr",
            1e-4,
            1e-2,
            log=True
        ),

        "weight_decay": trial.suggest_float(
            "weight_decay",
            1e-6,
            1e-2,
            log=True
        ),

        # Optimizer is also an Optuna hyperparameter.
        "optimizer": trial.suggest_categorical(
            "optimizer",
            [
                "Adam",
                "AdamW",
                "RMSprop"
            ]
        ),

        # Class weights are also optimized.
        # HOLD maximum = 0.70
        # BUY minimum  = 2.20
        # SELL minimum = 2.60
        "weight_hold": trial.suggest_float(
            "weight_hold",
            0.30,
            0.70,
            step=0.05
        ),

        "weight_buy": trial.suggest_float(
            "weight_buy",
            2.20,
            4.00,
            step=0.10
        ),

        "weight_sell": trial.suggest_float(
            "weight_sell",
            2.60,
            4.00,
            step=0.10
        ),

        "dropout": trial.suggest_float(
            "dropout",
            0.1,
            0.5
        ),

        "batch_size": trial.suggest_categorical(
            "batch_size",
            [32, 64, 128]
        ),

        # Number of hidden layers
        "n_layers": trial.suggest_int(
            "n_layers",
            2,
            5
        )
    }

    # ----------------------------------------
    # Dynamic hidden layer sizes
    # ----------------------------------------

    for i in range(
        params["n_layers"]
    ):

        params[f"hidden_{i}"] = (
            trial.suggest_categorical(
                f"hidden_{i}",
                [
                    32,
                    64,
                    128,
                    256,
                    512
                ]
            )
        )

    # ----------------------------------------
    # Train
    # ----------------------------------------

    (
        _,
        best_balanced_acc,
        best_accuracy,
        best_f1,
        best_epoch
    ) = train_model(
        params=params,
        epochs=50,
        patience=8,
        trial=trial,
        verbose=False
    )

    # ----------------------------------------
    # Print trial result
    # ----------------------------------------

    print(
        f"\nTrial {trial.number}"
    )

    print(
        f"Best Epoch: {best_epoch}"
    )

    print(
        f"Accuracy: "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        f"Balanced Accuracy: "
        f"{best_balanced_acc * 100:.2f}%"
    )

    print(
        f"Macro F1: "
        f"{best_f1:.4f}"
    )

    print(
        "Hidden Layers:",
        [
            params[f"hidden_{i}"]
            for i in range(
                params["n_layers"]
            )
        ]
    )

    print(
        f"Optimizer: {params['optimizer']}"
    )

    print(
        f"Weights -> HOLD: {params['weight_hold']:.2f}, "
        f"BUY: {params['weight_buy']:.2f}, "
        f"SELL: {params['weight_sell']:.2f}"
    )

    # Optuna objective
    return best_balanced_acc


# ============================================================
# 12. OPTUNA STUDY
# ============================================================

print("\n========================================")
print("STARTING OPTUNA")
print("========================================")

N_TRIALS = 50

study = optuna.create_study(
    direction="maximize",
    study_name="gold_ann"
)

study.optimize(
    objective,
    n_trials=N_TRIALS
)


# ============================================================
# 13. BEST PARAMETERS
# ============================================================

print("\n========================================")
print("OPTUNA COMPLETE")
print("========================================")

print(
    f"Best Balanced Accuracy: "
    f"{study.best_value * 100:.2f}%"
)

print("\nBest Parameters:")

for key, value in (
    study.best_params.items()
):

    print(
        f"{key}: {value}"
    )


# ============================================================
# 14. FINAL TRAINING
# ============================================================

best_params = dict(
    study.best_params
)

print("\n========================================")
print("FINAL TRAINING")
print("========================================")

(
    best_model_state,
    best_balanced_acc,
    best_accuracy,
    best_f1,
    best_epoch
) = train_model(
    params=best_params,
    epochs=100,
    patience=15,
    verbose=True
)


# ============================================================
# 15. CREATE FINAL MODEL
# ============================================================

final_hidden_layers = []

for i in range(
    best_params["n_layers"]
):

    final_hidden_layers.append(
        best_params[f"hidden_{i}"]
    )

print(
    "\nFinal Hidden Layers:",
    final_hidden_layers
)

model = create_model(
    num_features=X_train.shape[1],
    hidden_layers=final_hidden_layers,
    dropout=best_params["dropout"]
)

if best_model_state is None:

    raise RuntimeError(
        "Best model state is None."
    )

model.load_state_dict(
    best_model_state
)

model.eval()


# ============================================================
# 16. TEST DATA
# ============================================================

test_loader = DataLoader(
    test_dataset,
    batch_size=best_params["batch_size"],
    shuffle=False,
    pin_memory=torch.cuda.is_available()
)

test_true = []

test_pred = []

test_probabilities = []


with torch.no_grad():

    for (
        batch_features,
        batch_labels
    ) in test_loader:

        batch_features = batch_features.to(
            device,
            non_blocking=True
        )

        outputs = model(
            batch_features
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
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

        test_probabilities.extend(
            probabilities.cpu().numpy()
        )


# ============================================================
# 17. TEST METRICS
# ============================================================

test_accuracy = accuracy_score(
    test_true,
    test_pred
)

test_balanced_acc = balanced_accuracy_score(
    test_true,
    test_pred
)

test_macro_f1 = f1_score(
    test_true,
    test_pred,
    average="macro",
    zero_division=0
)


print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(
    f"Best Epoch: "
    f"{best_epoch}"
)

print(
    f"Validation Accuracy: "
    f"{best_accuracy * 100:.2f}%"
)

print(
    f"Validation Balanced Accuracy: "
    f"{best_balanced_acc * 100:.2f}%"
)

print(
    f"Validation Macro F1: "
    f"{best_f1:.4f}"
)

print(
    f"TEST Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"TEST Balanced Accuracy: "
    f"{test_balanced_acc * 100:.2f}%"
)

print(
    f"TEST Macro F1: "
    f"{test_macro_f1:.4f}"
)

print(
    f"Best Optimizer: {best_params['optimizer']}"
)

print(
    f"Best Weights -> "
    f"HOLD: {best_params['weight_hold']:.2f}, "
    f"BUY: {best_params['weight_buy']:.2f}, "
    f"SELL: {best_params['weight_sell']:.2f}"
)


# ============================================================
# 18. CLASSIFICATION REPORT
# ============================================================

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


# ============================================================
# 19. CONFUSION MATRIX
# ============================================================

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(
    "Rows    = Actual"
)

print(
    "Columns = Predicted"
)

print(
    "Order   = HOLD, BUY, SELL"
)

print()

print(
    confusion_matrix(
        test_true,
        test_pred,
        labels=[0, 1, 2]
    )
)


# ============================================================
# 20. SAVE BEST MODEL
# ============================================================

MODEL_PATH = "best_gold_ann_model.pt"

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

        "class_weights":
            np.array([
                best_params["weight_hold"],
                best_params["weight_buy"],
                best_params["weight_sell"]
            ], dtype=np.float32),

        "hidden_layers":
            final_hidden_layers,

        "best_hyperparameters":
            best_params,

        "optimizer":
            best_params["optimizer"],

        "validation_balanced_accuracy":
            best_balanced_acc,

        "validation_accuracy":
            best_accuracy,

        "validation_macro_f1":
            best_f1,

        "test_accuracy":
            test_accuracy,

        "test_balanced_accuracy":
            test_balanced_acc,

        "test_macro_f1":
            test_macro_f1,

        "best_epoch":
            best_epoch
    },

    MODEL_PATH
)


print("\n========================================")
print("MODEL SAVED")
print("========================================")

print(
    f"Saved as: {MODEL_PATH}"
)