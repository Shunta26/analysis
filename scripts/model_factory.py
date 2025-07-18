#モデル定義
# scripts/model_factory.py

import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2, regression=False):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.activation = nn.Identity() if regression else nn.Sigmoid()
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return self.activation(out)

class GRUModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))

class SimpleRNNModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.rnn(x)
        out = out[:, -1, :]
        return torch.sigmoid(self.fc(out))

def get_model(model_name: str, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, use_dropout: bool = True, dropout_rate: float = 0.2, regression: bool = False):
    model_name = model_name.lower()
    dropout = dropout_rate if use_dropout else 0.0
    if model_name == "lstm":
        return LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, regression=regression)
    elif model_name == "gru":
        return GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_name == "rnn":
        return SimpleRNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    else:
        raise ValueError(f"未対応のモデル名です: {model_name}")
