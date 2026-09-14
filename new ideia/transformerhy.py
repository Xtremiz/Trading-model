import copy
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. SETTINGS
# =========================================================

SEED = 42

LEARNING_RATE = 0.0003
BATCH_SIZE = 64

EPOCHS = 100
PATIENCE = 50

SEQUENCE_LENGTH = 100

# -------------------------
# LSTM
# -------------------------

LSTM_HIDDEN = 128
LSTM_LAYERS = 1
BIDIRECTIONAL = True

# -------------------------
# Transformer
# -------------------------

D_MODEL = 128
N_HEADS = 8
NUM_LAYERS = 2
DIM_FEEDFORWARD = 256

DROPOUT = 0.2

# -------------------------
# Class weights
# -------------------------

HOLD_WEIGHT = 1
BUY_WEIGHT = 1.7
SELL_WEIGHT = 2


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# =========================================================
# 3. DEVICE
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print(f"Device: {device}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    print(
        f"GPU Memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print("=" * 60)


# =========================================================
# 4. LOAD DATA
# =========================================================

CSV_PATH = (
    r"F:\Git-Hub\Trading model\data\gold112_Robust-Scaled.csv"
)

print("\nLoading data...")

df = pd.read_csv(CSV_PATH)

print(
    "Original shape:",
    df.shape
)


# =========================================================
# 5. CLEAN DATA
# =========================================================

df = (
    df
    .dropna()
    .reset_index(drop=True)
)


# =========================================================
# 6. SIGNAL MAPPING
# =========================================================

signal_mapping = {
    0: 0,      # HOLD
    1: 1,      # BUY
    -1: 2      # SELL
}

df["signal"] = (
    df["signal"]
    .map(signal_mapping)
)

df = (
    df
    .dropna(subset=["signal"])
    .reset_index(drop=True)
)

df["signal"] = (
    df["signal"]
    .astype(np.int64)
)


# =========================================================
# 7. FEATURES / LABELS
# =========================================================

feature_columns = [
    col
    for col in df.columns
    if col != "signal"
]

X = (
    df[feature_columns]
    .values
    .astype(np.float32)
)

y = (
    df["signal"]
    .values
    .astype(np.int64)
)


print("\nData:")
print(
    "Rows:",
    len(X)
)

print(
    "Features:",
    X.shape[1]
)


# =========================================================
# 8. CLASS DISTRIBUTION
# =========================================================

print("\nClass distribution:")

unique, counts = np.unique(
    y,
    return_counts=True
)

for label, count in zip(
    unique,
    counts
):

    name = {
        0: "HOLD",
        1: "BUY",
        2: "SELL"
    }[label]

    print(
        f"{name:5s}: "
        f"{count:6d} "
        f"({count / len(y) * 100:.2f}%)"
    )


# =========================================================
# 9. CHECK DATA
# =========================================================

if len(X) <= SEQUENCE_LENGTH:

    raise ValueError(
        f"Not enough data. "
        f"Rows={len(X)}, "
        f"Sequence={SEQUENCE_LENGTH}"
    )


# =========================================================
# 10. CHRONOLOGICAL SPLIT
# =========================================================
#
# OLD
#  |
#  |------ TRAIN ------|-- VAL --|------ TEST ------|
#  |
# NEW
#
# =========================================================

total_samples = (
    len(X) -
    SEQUENCE_LENGTH
)

train_end = int(
    total_samples * 0.70
)

val_end = int(
    total_samples * 0.80
)


print("\nSequence samples:")
print(
    total_samples
)

print(
    f"Train samples: "
    f"{train_end}"
)

print(
    f"Val samples:   "
    f"{val_end - train_end}"
)

print(
    f"Test samples:  "
    f"{total_samples - val_end}"
)


# =========================================================
# 11. MEMORY-EFFICIENT DATASET
# =========================================================
#
# IMPORTANT:
#
# We DO NOT create:
#
# X_sequences = []
#
# Every sequence is generated when needed.
#
# =========================================================

class TradingDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        start_index,
        end_index,
        sequence_length
    ):

        self.X = X
        self.y = y

        self.start_index = (
            start_index
        )

        self.end_index = (
            end_index
        )

        self.sequence_length = (
            sequence_length
        )


    def __len__(self):

        return (
            self.end_index -
            self.start_index
        )


    def __getitem__(self, idx):

        target_index = (
            self.start_index +
            idx
        )

        start = (
            target_index -
            self.sequence_length
        )

        end = target_index

        sequence = self.X[
            start:end
        ]

        label = self.y[
            target_index
        ]

        return (
            torch.from_numpy(sequence),
            torch.tensor(
                label,
                dtype=torch.long
            )
        )


# =========================================================
# 12. DATASETS
# =========================================================

train_dataset = TradingDataset(
    X,
    y,
    start_index=SEQUENCE_LENGTH,
    end_index=(
        SEQUENCE_LENGTH +
        train_end
    ),
    sequence_length=SEQUENCE_LENGTH
)


val_dataset = TradingDataset(
    X,
    y,
    start_index=(
        SEQUENCE_LENGTH +
        train_end
    ),
    end_index=(
        SEQUENCE_LENGTH +
        val_end
    ),
    sequence_length=SEQUENCE_LENGTH
)


test_dataset = TradingDataset(
    X,
    y,
    start_index=(
        SEQUENCE_LENGTH +
        val_end
    ),
    end_index=len(X),
    sequence_length=SEQUENCE_LENGTH
)


# =========================================================
# 13. DATALOADERS
# =========================================================

pin_memory = (
    torch.cuda.is_available()
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    pin_memory=pin_memory
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    pin_memory=pin_memory
)


print("\nDataLoaders ready.")


# =========================================================
# 14. LSTM + TRANSFORMER MODEL
# =========================================================

class LSTMTransformer(nn.Module):

    def __init__(
        self,
        num_features,
        sequence_length=100,
        lstm_hidden=128,
        lstm_layers=1,
        bidirectional=True,
        d_model=128,
        nhead=8,
        num_layers=2,
        dim_feedforward=256,
        dropout=0.2
    ):

        super().__init__()


        # =================================================
        # LSTM
        # =================================================

        self.lstm = nn.LSTM(

            input_size=num_features,

            hidden_size=lstm_hidden,

            num_layers=lstm_layers,

            batch_first=True,

            bidirectional=bidirectional,

            dropout=(
                dropout
                if lstm_layers > 1
                else 0
            )
        )


        if bidirectional:

            lstm_output_size = (
                lstm_hidden * 2
            )

        else:

            lstm_output_size = (
                lstm_hidden
            )


        # =================================================
        # LSTM OUTPUT → TRANSFORMER DIMENSION
        # =================================================

        self.projection = nn.Linear(

            lstm_output_size,

            d_model
        )


        # =================================================
        # POSITIONAL ENCODING
        # =================================================

        self.positional_encoding = nn.Parameter(

            torch.zeros(
                1,
                sequence_length,
                d_model
            )
        )


        nn.init.trunc_normal_(

            self.positional_encoding,

            std=0.02
        )


        # =================================================
        # DROPOUT
        # =================================================

        self.dropout = nn.Dropout(
            dropout
        )


        # =================================================
        # TRANSFORMER
        # =================================================

        encoder_layer = (
            nn.TransformerEncoderLayer(

                d_model=d_model,

                nhead=nhead,

                dim_feedforward=(
                    dim_feedforward
                ),

                dropout=dropout,

                activation="gelu",

                batch_first=True,

                norm_first=False
            )
        )


        self.transformer = (
            nn.TransformerEncoder(

                encoder_layer,

                num_layers=num_layers
            )
        )


        # =================================================
        # CLASSIFIER
        # =================================================

        self.classifier = nn.Sequential(

            nn.Linear(
                d_model,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                64,
                3
            )
        )


    def forward(self, x):

        # =================================================
        # INPUT
        #
        # (batch, 100, 112)
        # =================================================

        x, _ = self.lstm(x)


        # =================================================
        # LSTM OUTPUT
        #
        # BiLSTM:
        #
        # (batch, 100, 256)
        # =================================================


        # =================================================
        # PROJECT TO TRANSFORMER
        #
        # (batch, 100, 128)
        # =================================================

        x = self.projection(x)


        # =================================================
        # POSITION
        # =================================================

        x = (
            x +
            self.positional_encoding
        )

        x = self.dropout(x)


        # =================================================
        # TRANSFORMER
        #
        # (batch, 100, 128)
        # =================================================

        x = self.transformer(x)


        # =================================================
        # GLOBAL AVERAGE POOLING
        #
        # (batch, 128)
        # =================================================

        x = x.mean(
            dim=1
        )


        # =================================================
        # CLASSIFIER
        # =================================================

        x = self.classifier(x)


        return x


# =========================================================
# 15. CREATE MODEL
# =========================================================

num_features = X.shape[1]


model = LSTMTransformer(

    num_features=num_features,

    sequence_length=SEQUENCE_LENGTH,

    lstm_hidden=LSTM_HIDDEN,

    lstm_layers=LSTM_LAYERS,

    bidirectional=BIDIRECTIONAL,

    d_model=D_MODEL,

    nhead=N_HEADS,

    num_layers=NUM_LAYERS,

    dim_feedforward=DIM_FEEDFORWARD,

    dropout=DROPOUT

).to(device)


print("\nModel:")
print(model)


# =========================================================
# 16. CLASS WEIGHTS
# =========================================================

class_weights = torch.tensor(

    [
        HOLD_WEIGHT,
        BUY_WEIGHT,
        SELL_WEIGHT
    ],

    dtype=torch.float32,

    device=device
)


criterion = nn.CrossEntropyLoss(

    weight=class_weights
)


# =========================================================
# 17. OPTIMIZER
# =========================================================

optimizer = torch.optim.AdamW(

    model.parameters(),

    lr=LEARNING_RATE,

    weight_decay=1e-4
)


# =========================================================
# 18. SCHEDULER
# =========================================================

scheduler = (
    torch.optim.lr_scheduler.ReduceLROnPlateau(

        optimizer,

        mode="max",

        factor=0.5,

        patience=3
    )
)


# =========================================================
# 19. MIXED PRECISION
# =========================================================

use_amp = (
    torch.cuda.is_available()
)


if use_amp:

    scaler = torch.amp.GradScaler(
        "cuda"
    )

else:

    scaler = None


print(
    f"\nMixed Precision: "
    f"{use_amp}"
)


# =========================================================
# 20. EVALUATION
# =========================================================

def evaluate_model(
    model,
    loader
):

    model.eval()


    all_labels = []

    all_predictions = []


    total_loss = 0.0

    total_samples = 0


    with torch.no_grad():

        for (
            batch_features,
            batch_labels
        ) in loader:


            batch_features = (
                batch_features.to(

                    device,

                    non_blocking=True
                )
            )


            batch_labels = (
                batch_labels.to(

                    device,

                    non_blocking=True
                )
            )


            if use_amp:

                with torch.amp.autocast(
                    device_type="cuda"
                ):

                    outputs = model(
                        batch_features
                    )

                    loss = criterion(
                        outputs,
                        batch_labels
                    )

            else:

                outputs = model(
                    batch_features
                )

                loss = criterion(
                    outputs,
                    batch_labels
                )


            predictions = torch.argmax(

                outputs,

                dim=1
            )


            batch_size = (
                batch_labels.size(0)
            )


            total_loss += (
                loss.item() *
                batch_size
            )


            total_samples += (
                batch_size
            )


            all_labels.extend(

                batch_labels
                .cpu()
                .numpy()
            )


            all_predictions.extend(

                predictions
                .cpu()
                .numpy()
            )


    all_labels = np.array(
        all_labels
    )

    all_predictions = np.array(
        all_predictions
    )


    average_loss = (
        total_loss /
        total_samples
    )


    accuracy = accuracy_score(

        all_labels,

        all_predictions
    )


    balanced_acc = (
        balanced_accuracy_score(

            all_labels,

            all_predictions
        )
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

        target_names=[

            "HOLD",

            "BUY",

            "SELL"
        ],

        output_dict=True,

        zero_division=0
    )


    hold_f1 = (
        report["HOLD"]["f1-score"]
    )

    buy_f1 = (
        report["BUY"]["f1-score"]
    )

    sell_f1 = (
        report["SELL"]["f1-score"]
    )


    return (

        average_loss,

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
# 21. TRAINING
# =========================================================

print("\n")

print("=" * 60)

print(
    "STARTING LSTM + TRANSFORMER TRAINING"
)

print("=" * 60)


best_macro_f1 = -1.0

best_balanced_acc = -1.0

best_model_weights = None

best_epoch = 0

epochs_without_improvement = 0


for epoch in range(EPOCHS):


    # =====================================================
    # TRAIN MODE
    # =====================================================

    model.train()


    total_train_loss = 0.0

    total_train_samples = 0


    # =====================================================
    # TRAIN BATCHES
    # =====================================================

    for (
        batch_features,
        batch_labels
    ) in train_loader:


        batch_features = (
            batch_features.to(

                device,

                non_blocking=True
            )
        )


        batch_labels = (
            batch_labels.to(

                device,

                non_blocking=True
            )
        )


        optimizer.zero_grad(
            set_to_none=True
        )


        # =================================================
        # FORWARD
        # =================================================

        if use_amp:

            with torch.amp.autocast(
                device_type="cuda"
            ):

                outputs = model(
                    batch_features
                )

                loss = criterion(
                    outputs,
                    batch_labels
                )


            # =============================================
            # BACKWARD
            # =============================================

            scaler.scale(
                loss
            ).backward()


            scaler.unscale_(
                optimizer
            )


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0
            )


            scaler.step(
                optimizer
            )


            scaler.update()


        else:

            outputs = model(
                batch_features
            )


            loss = criterion(

                outputs,

                batch_labels
            )


            loss.backward()


            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0
            )


            optimizer.step()


        batch_size = (
            batch_labels.size(0)
        )


        total_train_loss += (

            loss.item() *
            batch_size
        )


        total_train_samples += (
            batch_size
        )


    average_train_loss = (

        total_train_loss /
        total_train_samples
    )


    # =====================================================
    # VALIDATION
    # =====================================================

    (
        val_loss,
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


    # =====================================================
    # SCHEDULER
    # =====================================================

    scheduler.step(
        val_macro_f1
    )


    current_lr = (
        optimizer
        .param_groups[0]["lr"]
    )


    # =====================================================
    # PRINT RESULTS
    # =====================================================

    print(

        f"Epoch "
        f"[{epoch + 1:02d}/{EPOCHS}] "

        f"Train Loss: "
        f"{average_train_loss:.4f} | "

        f"Val Loss: "
        f"{val_loss:.4f} | "

        f"Macro F1: "
        f"{val_macro_f1:.4f} | "

        f"Balanced: "
        f"{val_balanced_acc * 100:.2f}% | "

        f"HOLD: "
        f"{val_hold_f1:.4f} | "

        f"BUY: "
        f"{val_buy_f1:.4f} | "

        f"SELL: "
        f"{val_sell_f1:.4f} | "

        f"LR: "
        f"{current_lr:.7f}"
    )


    # =====================================================
    # BEST MODEL CHECK
    # =====================================================

    is_better = False


    if (
        val_macro_f1 >
        best_macro_f1
    ):

        is_better = True


    elif (

        val_macro_f1 ==
        best_macro_f1

        and

        val_balanced_acc >
        best_balanced_acc

    ):

        is_better = True


    if is_better:


        best_macro_f1 = (
            val_macro_f1
        )


        best_balanced_acc = (
            val_balanced_acc
        )


        best_model_weights = (
            copy.deepcopy(

                model.state_dict()
            )
        )


        best_epoch = (
            epoch + 1
        )


        epochs_without_improvement = 0


        print(

            f"  --> NEW BEST | "

            f"Macro F1: "
            f"{best_macro_f1:.4f} | "

            f"Balanced: "
            f"{best_balanced_acc * 100:.2f}%"
        )


    else:

        epochs_without_improvement += 1


    # =====================================================
    # EARLY STOPPING
    # =====================================================

    if (
        epochs_without_improvement >=
        PATIENCE
    ):

        print(
            "\nEarly stopping triggered."
        )

        break


# =========================================================
# 22. LOAD BEST MODEL
# =========================================================

if best_model_weights is None:

    raise RuntimeError(
        "No best model was saved."
    )


model.load_state_dict(
    best_model_weights
)


# =========================================================
# 23. SAVE MODEL
# =========================================================

MODEL_PATH = (
    "gold112_lstm_transformer_best.pt"
)


torch.save(

    {
        "model_state_dict":
            best_model_weights,

        "num_features":
            num_features,

        "sequence_length":
            SEQUENCE_LENGTH,

        "lstm_hidden":
            LSTM_HIDDEN,

        "lstm_layers":
            LSTM_LAYERS,

        "bidirectional":
            BIDIRECTIONAL,

        "d_model":
            D_MODEL,

        "nhead":
            N_HEADS,

        "num_layers":
            NUM_LAYERS,

        "dim_feedforward":
            DIM_FEEDFORWARD,

        "dropout":
            DROPOUT,

        "best_epoch":
            best_epoch,

        "best_macro_f1":
            best_macro_f1,

        "best_balanced_acc":
            best_balanced_acc,

        "feature_columns":
            feature_columns
    },

    MODEL_PATH
)


print("\n")

print("=" * 60)

print(
    "BEST MODEL SAVED"
)

print(
    f"Epoch: {best_epoch}"
)

print(
    f"Macro F1: "
    f"{best_macro_f1:.4f}"
)

print(
    f"Balanced Accuracy: "
    f"{best_balanced_acc * 100:.2f}%"
)

print(
    f"File: {MODEL_PATH}"
)

print("=" * 60)


# =========================================================
# 24. FINAL TEST
# =========================================================

(
    test_loss,
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


# =========================================================
# 25. FINAL RESULTS
# =========================================================

print("\n")

print("=" * 60)

print(
    "FINAL TEST RESULTS"
)

print("=" * 60)


print(
    f"Test Loss: "
    f"{test_loss:.4f}"
)


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


# =========================================================
# 26. CLASSIFICATION REPORT
# =========================================================

print(
    "\nClassification Report:\n"
)


print(

    classification_report(

        test_labels,

        test_predictions,

        target_names=[

            "HOLD",

            "BUY",

            "SELL"
        ],

        digits=4,

        zero_division=0
    )
)


# =========================================================
# 27. CONFUSION MATRIX
# =========================================================

cm = confusion_matrix(

    test_labels,

    test_predictions
)


print(
    "\nConfusion Matrix:"
)


print(
    "              Predicted"
)


print(
    "              HOLD   BUY   SELL"
)


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


# =========================================================
# 28. GPU CLEANUP
# =========================================================

if torch.cuda.is_available():

    torch.cuda.empty_cache()

    print(
        "\nGPU cache cleared."
    )


print(
    "\nTraining complete."
)