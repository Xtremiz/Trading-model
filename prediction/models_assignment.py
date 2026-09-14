import torch
import torch.nn as nn
NUM_FEATURES = 112
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
gold_CONV1D_PATH = r"F:\Git-Hub\Trading model\Gold models\gold112_scaled_model_conv1d.pt"
gold_LSTM_PATH = r"F:\Git-Hub\Trading model\Gold models\gold112_scaled_model_lstm.pt"
gold_HYBRID1_PATH = r"F:\Git-Hub\Trading model\Gold models\gold112_scaled_model_lstmhy.pt"
gold_HYBRID2_PATH = r"F:\Git-Hub\Trading model\Gold models\gold112_scaled_model_lstmhy2.pt"

US30_CONV1D_PATH = r"F:\Git-Hub\Trading model\US30 model\US30Cash_balanced_Scaled_model_conv1d.pt"
US30_LSTM_PATH = r"F:\Git-Hub\Trading model\US30 model\US30Cash_balanced_Scaled_model_lstm.pt"
US30_HYBRID1_PATH = r"F:\Git-Hub\Trading model\US30 model\US30Cash_balanced_Scaled_model_lstmhy.pt"
US30_HYBRID2_PATH = r"F:\Git-Hub\Trading model\US30 model\US30Cash_balanced_Scaled_model_lstmhy2.pt"
class MyConv1d(nn.Module):

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
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2)

        )
        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(128 * 25, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 3)
        )

    def forward(self, x):

        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = self.classifier(x)

        return x


class MyLSTM(nn.Module):

    def __init__(self, num_features):

        super().__init__()

      

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        lstm_out_size = 128 * 2
    
        self.classifier = nn.Sequential(

            nn.Linear(lstm_out_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 3)
        )

    def forward(self, x):

        x, (h_n, c_n) = self.lstm(x)
    
        x =x[:, -1, :]
    
        x = self.classifier(x)

        return x

class MyHybrid(nn.Module):

    def __init__(self, num_features):

        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        lstm_out_size = 128 * 2

        self.features = nn.Sequential(

            nn.Conv1d(
                in_channels=lstm_out_size,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(kernel_size=2, stride=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.MaxPool1d(kernel_size=2, stride=2)

        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(

            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 3)
        )

    def forward(self, x):

        x, (h_n, c_n) = self.lstm(x)
        x = x.permute(0, 2, 1)
        

        x = self.features(x)

        x = self.pool(x)
  
        x = x.squeeze(-1)

        x = self.classifier(x)

        return x

gold_conv1d_state = torch.load(gold_CONV1D_PATH, map_location=DEVICE)
gold_conv1d_model = MyConv1d(num_features=NUM_FEATURES).to(DEVICE)
gold_conv1d_model.load_state_dict(gold_conv1d_state)

gold_lstm_state = torch.load(gold_LSTM_PATH, map_location=DEVICE)
gold_lstm_model = MyLSTM(num_features=NUM_FEATURES).to(DEVICE)
gold_lstm_model.load_state_dict(gold_lstm_state)

gold_hybrid1_state = torch.load(gold_HYBRID1_PATH, map_location=DEVICE)
gold_hybrid1_model = MyHybrid(num_features=NUM_FEATURES).to(DEVICE)
gold_hybrid1_model.load_state_dict(gold_hybrid1_state)

gold_hybrid2_state = torch.load(gold_HYBRID2_PATH, map_location=DEVICE)
gold_hybrid2_model = MyHybrid(num_features=NUM_FEATURES).to(DEVICE)
gold_hybrid2_model.load_state_dict(gold_hybrid2_state)

US30_conv1d_state = torch.load(US30_CONV1D_PATH, map_location=DEVICE)
US30_conv1d_model = MyConv1d(num_features=NUM_FEATURES).to(DEVICE)
US30_conv1d_model.load_state_dict(US30_conv1d_state)

US30_lstm_state = torch.load(US30_LSTM_PATH, map_location=DEVICE)
US30_lstm_model = MyLSTM(num_features=NUM_FEATURES).to(DEVICE)
US30_lstm_model.load_state_dict(US30_lstm_state)

US30_hybrid1_state = torch.load(US30_HYBRID1_PATH, map_location=DEVICE)
US30_hybrid1_model = MyHybrid(num_features=NUM_FEATURES).to(DEVICE)
US30_hybrid1_model.load_state_dict(US30_hybrid1_state)

US30_hybrid2_state = torch.load(US30_HYBRID2_PATH, map_location=DEVICE)
US30_hybrid2_model = MyHybrid(num_features=NUM_FEATURES).to(DEVICE)
US30_hybrid2_model.load_state_dict(US30_hybrid2_state)
