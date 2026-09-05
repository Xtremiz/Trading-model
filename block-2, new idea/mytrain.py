import channells
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
df= pd.read_csv(r"")
ch1 = channells.channel_1
ch2 = channells.channel_2
ch3 = channells.channel_3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
df.dropna(inplace=True)
signal_mapping = {
    0: 0,
    1: 1,
    -1: 2
}
df['target'] = df['target'].map(signal_mapping)
feature_names = df.drop(columns=['target']).columns.tolist()
X = df[feature_names].values.astype(np.float32)
y = df['target'].values.astype(np.int64)

feature_index = {
    feature: i
    for i, feature in enumerate(feature_names)
}
