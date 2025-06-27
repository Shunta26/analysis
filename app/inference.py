#推論ロジック（モデル読み込み、予測）

import torch
import pandas as pd
import numpy as np
import os
import sys

# model_factory を scripts/ からインポートできるようにパスを追加
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
from model_factory import get_model

def load_model(model_path, input_size=4):
    """
    保存された PyTorch モデルを読み込みます。
    モデル構造は model_factory.get_model() を使用して再構築されます。
    """
    model = get_model("lstm", input_size=input_size, regression=True)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model

def predict_awakenness(model, df, sequence_length=10):
    """
    DataFrameから覚醒度を予測します。
    入力は 'pupil', 'eda', 'eeg', 'hr' カラムを持つ前提です。
    """
    required_columns = ['pupil', 'eda', 'eeg', 'hr']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"必要なカラムが不足しています: {col}")

    # データ抽出・正規化（0-1スケーリング）
    data = df[required_columns].values.astype(np.float32)
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
