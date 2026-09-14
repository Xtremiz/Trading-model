
import time
import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from probablity_func import predict_signal, get_signal
import torch
import torch.nn as nn

from pipeline import pipeline

c1d = []
h1 = []
h2 = []

# =========================================================
# SETTINGS
# =========================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SYMBOL1 = "GOLD"
TIMEFRAME = mt5.TIMEFRAME_M15

SEQUENCE_LENGTH = 100
NUM_FEATURES = 87

CONV1D_PATH = r"F:\Git-Hub\Trading model\models\best_gold_model_conv1d.pt"
HYBRID1_PATH = r"F:\Git-Hub\Trading model\models\best_gold_model_hybrid.pt"
HYBRID2_PATH = r"F:\Git-Hub\Trading model\models\best_gold_model_hybrid2.pt"


# =========================================================
# MT5 INITIALIZATION
# =========================================================

if not mt5.initialize():

    print("MT5 initialize failed!")
    print("Error:", mt5.last_error())

    raise SystemExit


print("MT5 connected successfully!")


# =========================================================
# ACCOUNT INFORMATION
# =========================================================

account_info = mt5.account_info()

if account_info is not None:

    print("Account:", account_info.login)
    print("Server :", account_info.server)


# =========================================================
# SYMBOL CHECK
# =========================================================

if not mt5.symbol_select(SYMBOL1, True):

    print(f"Symbol select failed: {SYMBOL1}")
    print("MT5 Error:", mt5.last_error())

    mt5.shutdown()

    raise SystemExit


print(f"Symbol selected: {SYMBOL1}")


# =========================================================
# LABELS
# =========================================================

LABELS = {
    0: "HOLD",
    1: "BUY",
    2: "SELL"
}


# =========================================================
# CONV1D MODEL
# =========================================================

class ConvNet(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv1d(
                num_features,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(64),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            ),

            nn.Conv1d(
                64,
                128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.BatchNorm1d(128),

            nn.MaxPool1d(
                kernel_size=2,
                stride=2
            )
        )

        # Sequence 100
        # 100 -> 50 -> 25

        flattened_size = 128 * 25

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                flattened_size,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                64,
                128
            ),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(
                128,
                3
            )
        )


    def forward(self, x):

        # (batch, sequence, features)

        x = x.permute(
            0,
            2,
            1
        )

        x = self.features(x)

        x = self.classifier(x)

        return x


# =========================================================
# HYBRID MODEL
# =========================================================

class HybridNet(nn.Module):

    def __init__(self, num_features):

        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(
                in_channels=num_features,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(
                kernel_size=2,
                stride=2)
        )


        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )


        self.pool = nn.AdaptiveAvgPool1d(1)


        self.classifier = nn.Sequential(

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                64,
                128
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                128,
                3
            )
        )


    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.features(x)
        x = x.permute(0,2,1)
        x, _ = self.lstm(x)
        x = x.permute(0,2,1)
        x = self.pool(x)
        x = x.squeeze(-1)
        x = self.classifier(x)

        return x


# =========================================================
# LOAD CONV1D
# =========================================================

print("\nLoading Conv1D model...")

conv1d_state = torch.load(
    CONV1D_PATH,
    map_location=DEVICE
)


conv1d_model = ConvNet(
    num_features=NUM_FEATURES
).to(DEVICE)


conv1d_model.load_state_dict(
    conv1d_state
)

conv1d_model.eval()


print("Conv1D loaded successfully.")


# =========================================================
# LOAD HYBRID 1
# =========================================================

print("Loading Hybrid1 model...")

hybrid1_state = torch.load(
    HYBRID1_PATH,
    map_location=DEVICE
)


hybrid1_model = HybridNet(
    num_features=NUM_FEATURES
).to(DEVICE)


hybrid1_model.load_state_dict(
    hybrid1_state
)

hybrid1_model.eval()


print("Hybrid1 loaded successfully.")


# =========================================================
# LOAD HYBRID 2
# =========================================================

print("Loading Hybrid2 model...")

hybrid2_state = torch.load(
    HYBRID2_PATH,
    map_location=DEVICE
)


hybrid2_model = HybridNet(
    num_features=NUM_FEATURES
).to(DEVICE)


hybrid2_model.load_state_dict(
    hybrid2_state
)

hybrid2_model.eval()


print("Hybrid2 loaded successfully.")


print("\nAll models loaded successfully.")

print("Device:", DEVICE)


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_model(model, data):

    # -----------------------------------------------------
    # Last 100 candles
    # -----------------------------------------------------

    data = data[-SEQUENCE_LENGTH:]


    # -----------------------------------------------------
    # Convert to tensor
    # -----------------------------------------------------

    x = torch.tensor(
        data,
        dtype=torch.float32
    )


    # -----------------------------------------------------
    # Add batch dimension
    # -----------------------------------------------------

    x = x.unsqueeze(0)


    # -----------------------------------------------------
    # Move to device
    # -----------------------------------------------------

    x = x.to(DEVICE)


    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    with torch.no_grad():

        output = model(x)

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]


    # -----------------------------------------------------
    # Prediction class
    # -----------------------------------------------------

    prediction = torch.argmax(
        probabilities
    ).item()


    # -----------------------------------------------------
    # Individual probabilities
    # -----------------------------------------------------

    hold_probability = probabilities[0].item()

    buy_probability = probabilities[1].item()

    sell_probability = probabilities[2].item()


    return (
        LABELS[prediction],
        hold_probability,
        buy_probability,
        sell_probability
    )


# =========================================================
# MAJORITY VOTING
# =========================================================


def get_candle_timer():

    """

    Returns:

        current candle time
        remaining MM:SS

    """

    # Current UTC time

    now = pd.Timestamp.now(
        tz="UTC"
    )


    # Current 15-minute candle

    current_candle = now.floor(
        "15min"
    )


    # Next candle

    next_candle = (
        current_candle
        + pd.Timedelta(minutes=15)
    )


    # Remaining time

    remaining = (
        next_candle - now
    )


    total_seconds = max(
        0,
        int(
            remaining.total_seconds()
        )
    )


    minutes = total_seconds // 60

    seconds = total_seconds % 60


    return (
        current_candle,
        f"{minutes:02d}:{seconds:02d}"
    )


# =========================================================
# MAIN LOOP
# =========================================================

last_candle_time = None


try:

    while True:

        rates = mt5.copy_rates_from_pos(SYMBOL1,TIMEFRAME,1,150)


        if rates is None:

            print(
                "MT5 data nahi mili:",
                mt5.last_error()
            )

            time.sleep(1)

            continue


        # =================================================
        # DATAFRAME
        # =================================================

        df = pd.DataFrame(
            rates
        )


        if df.empty:

            print(
                "MT5 returned empty dataframe."
            )

            time.sleep(1)

            continue


        # =================================================
        # TIME CONVERSION
        # =================================================

        df["time"] = pd.to_datetime(

            df["time"],

            unit="s",

            utc=True
        )


        # =================================================
        # LAST COMPLETED CANDLE
        # =================================================

        candle_time = df["time"].iloc[-1]


        # =================================================
        # ONLY PREDICT ON NEW CANDLE
        # =================================================

        if candle_time == last_candle_time:

            # Still show countdown

            current_candle, timer = (
                get_candle_timer()
            )

            print(
                f"\rCurrent Candle: "
                f"{current_candle.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                f"Time Remaining: {timer}",
                end=""
            )

            time.sleep(1)

            continue


        last_candle_time = candle_time


        # =================================================
        # PIPELINE
        # =================================================

        features = pipeline(
            df
        )


        features = np.asarray(

            features,

            dtype=np.float32
        )


        # =================================================
        # CHECK SEQUENCE
        # =================================================

        if len(features) < SEQUENCE_LENGTH:

            print(
                f"\nNot enough features: "
                f"{len(features)}/{SEQUENCE_LENGTH}"
            )

            time.sleep(1)

            continue


        # =================================================
        # LAST 100 SEQUENCES
        # =================================================

        features = features[
            -SEQUENCE_LENGTH:
        ]


        # =================================================
        # CHECK FEATURE COUNT
        # =================================================

        if features.ndim != 2:

            print(
                "\nFeature array shape invalid:",
                features.shape
            )

            time.sleep(1)

            continue


        if features.shape[1] != NUM_FEATURES:

            print(

                f"\nFeature mismatch! "

                f"Expected: {NUM_FEATURES}, "

                f"Got: {features.shape[1]}"

            )

            time.sleep(1)

            continue


        # =================================================
        # PREDICTIONS
        # =================================================

        (
            conv_prediction,
            conv_hold,
            conv_buy,
            conv_sell
        ) = predict_model(

            conv1d_model,

            features
        )

        c1d = [conv_hold, conv_buy, conv_sell]
        (
            hybrid1_prediction,
            hybrid1_hold,
            hybrid1_buy,
            hybrid1_sell
        ) = predict_model(

            hybrid1_model,

            features
        )
        h1 = [hybrid1_hold, hybrid1_buy, hybrid1_sell]

        (
            hybrid2_prediction,
            hybrid2_hold,
            hybrid2_buy,
            hybrid2_sell
        ) = predict_model(

            hybrid2_model,

            features
        )
        h2 = [hybrid2_hold, hybrid2_buy, hybrid2_sell]

        # =================================================
        # MAJORITY VOTE
        # =================================================

        predictions = [

            conv_prediction,

            hybrid1_prediction,

            hybrid2_prediction
        ]

        # =================================================
        # TIMER
        # =================================================

        current_candle, timer = (
            get_candle_timer()
        )


        # =================================================
        # PRINT RESULT
        # =================================================

        print(
            "\n\n" + "=" * 75
        )


        print(
            f"Candle : "
            f"{candle_time.strftime('%Y-%m-%d %H:%M:%S')} UTC"
        )


        print(
            f"Current Candle: "
            f"{current_candle.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
            f"Time Remaining: {timer}"
        )


        print("-" * 75)


        # -------------------------------------------------
        # Conv1D
        # -------------------------------------------------

        print(

            f"Conv1D : {conv_prediction:<5} | "

            f"HOLD {conv_hold * 100:6.2f}% | "

            f"BUY {conv_buy * 100:6.2f}% | "

            f"SELL {conv_sell * 100:6.2f}%"

        )


        # -------------------------------------------------
        # Hybrid1
        # -------------------------------------------------

        print(

            f"Hybrid1: {hybrid1_prediction:<5} | "

            f"HOLD {hybrid1_hold * 100:6.2f}% | "

            f"BUY {hybrid1_buy * 100:6.2f}% | "

            f"SELL {hybrid1_sell * 100:6.2f}%"

        )


        # -------------------------------------------------
        # Hybrid2
        # -------------------------------------------------

        print(

            f"Hybrid2: {hybrid2_prediction:<5} | "

            f"HOLD {hybrid2_hold * 100:6.2f}% | "

            f"BUY {hybrid2_buy * 100:6.2f}% | "

            f"SELL {hybrid2_sell * 100:6.2f}%"

        )
        

        print(
            "=" * 75
        )
        p1 =predict_signal(c1d, h1, h2)
        p2 =get_signal(c1d, h1, h2)
        print(
            f"Majority Vote: {p1} | Ensemble Signal: {p2}"
        )

         # -------------------------------------------------
        # Record BUY/SELL signals to record.txt
        # -------------------------------------------------
        if p1 in ("BUY", "SELL") or p2 in ("BUY", "SELL"):
            with open("record.txt", "a") as f:
                f.write(
                    f"{current_candle.strftime('%Y-%m-%d %H:%M:%S')} UTC | "
                    f"Majority Vote: {p1} | Ensemble Signal: {p2}\n"
                )
            print(
                f"Signal recorded to record.txt"
            )
        time.sleep(1)



except KeyboardInterrupt:

    print(
        "\n\nStopping program..."
    )


finally:

    mt5.shutdown()

    print(
        "MT5 disconnected."
    )

