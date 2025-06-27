#推論ロジック（モデル読み込み、予測）

#models/フォルダ内に保存されたPyTorchモデルを読み込む
#GUIから渡されたデータを前処理しモデルに入力
#推定された覚醒度スコアを返す

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# ========================
# LSTMモデルの構造（保存と一致させる）
# ========================
class LSTMModel(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_size]
        out, _ = self.lstm(x)               # out shape: [batch, seq, hidden]
        return self.fc(out[:, -1, :])       # 最後の時刻の出力だけを使う → [batch, output_size]

# ========================
# モデル読み込み関数
# ========================
def load_model(model_path: str, input_size=4):
    model = LSTMModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

# ========================
# 前処理（標準化）
# ========================
def normalize(x: np.ndarray):
    return (x - np.mean(x, axis=0)) / (np.std(x, axis=0) + 1e-8)

# ========================
# 推論関数（複数行に対応）
# ========================
def predict_awakenness(model, df: pd.DataFrame):
    # 必要な列を取り出し（順番に注意）
    try:
        features = df[["pupil", "eda", "eeg", "hr"]].values
    except KeyError as e:
        raise ValueError(f"必要な列がCSVにありません: {e}")

    # 正規化
    features = normalize(features)

    # 1行ずつ [1, 1, 4] 形式で推論
    X = torch.tensor(features, dtype=torch.float32)
    results = []

    with torch.no_grad():
        for row in X:
            input_tensor = row.unsqueeze(0).unsqueeze(0)  # shape: [1, 1, 4]
            y_pred = model(input_tensor)
            results.append(y_pred.item())

    return results  # リストで返す
