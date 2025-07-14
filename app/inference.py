#推論ロジック（モデル読み込み、予測）

import torch
import pandas as pd
import numpy as np
import os
import sys
import json

# model_factory を scripts/ からインポートできるようにパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from scripts.model_factory import get_model

def load_model(model_path):
    config_path = model_path.replace(".pt", ".json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    model_type = config["model_type"].lower()
    input_size = len(config["selected_features"])
    hidden_size = config.get("hidden_size", 64)  # デフォルト値64
    num_layers = config.get("num_layers", 2)    # デフォルト値2

    model = get_model(model_type, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, regression=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model, config["selected_features"]

def predict_awakenness(model, df, selected_features, sequence_length=10):
    """
    DataFrameから覚醒度を予測します。
    入力は 'pupil', 'eda', 'eeg', 'hr' カラムを持つ前提です。
    """
    for col in selected_features:
        if col not in df.columns:
            raise ValueError(f"必要なカラムが不足しています: {col}")

    # データ抽出・正規化（0-1スケーリング）
    data = df[selected_features].values.astype(np.float32)
    data_min = np.min(data, axis=0)
    data_max = np.max(data, axis=0)
    data = (data - data_min) / (data_max - data_min + 1e-8)

    # 時系列に変換（sequenceごとにスライス）
    sequences = []
    for i in range(len(data) - sequence_length + 1):
        sequences.append(data[i:i + sequence_length])
    sequences = np.stack(sequences)

    inputs = torch.tensor(sequences)
    with torch.no_grad():
        outputs = model(inputs).squeeze().numpy()

    # sequence_length未満の先頭には NaN を補完しておく（時系列揃え）
    return np.concatenate([np.full(sequence_length - 1, np.nan), outputs])
