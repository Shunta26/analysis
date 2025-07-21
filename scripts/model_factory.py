#モデル定義
# scripts/model_factory.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, nhead=4, dropout=0.2, regression=False):
        super(TransformerModel, self).__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(hidden_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=hidden_size*4, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, 1)
        self.activation = nn.Identity() if regression else nn.Sigmoid()

    def forward(self, src):
        src = self.encoder(src) * math.sqrt(self.encoder.out_features)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output[:, -1, :])
        return self.activation(output)

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

def get_model(model_name: str, input_size: int = 1, hidden_size: int = 64, num_layers: int = 2, nhead: int = 4, use_dropout: bool = True, dropout_rate: float = 0.2, regression: bool = False):
    model_name = model_name.lower()
    dropout = dropout_rate if use_dropout else 0.0
    if model_name == "lstm":
        return LSTMModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout, regression=regression)
    elif model_name == "gru":
        return GRUModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_name == "rnn":
        return SimpleRNNModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout)
    elif model_name == "transformer":
        return TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, nhead=nhead, dropout=dropout, regression=regression)
    else:
        raise ValueError(f"未対応のモデル名です: {model_name}")
