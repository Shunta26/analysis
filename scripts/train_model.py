#学習処理（データ読み込み～学習・評価・返却）
# scripts/train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from model_factory import get_model

# シーケンスを生成する関数
def create_sequences(features, labels, seq_len):
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i:i+seq_len])
        y.append(labels[i+seq_len])
    return np.array(X), np.array(y)

# モデル学習関数
def train_model(data_path, model_type="LSTM", loss_type="MSELoss", optimizer_type="Adam",
                seq_len=10, selected_features=None):

    df = pd.read_csv(data_path)

    if selected_features is None or not selected_features:
        raise ValueError("selected_features を1つ以上選択してください。")

    # 選択された生理データの列を抽出
    features = df[selected_features].astype(float).values
    labels = df["kss"].astype(float).values

    # 正規化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    features = scaler_X.fit_transform(features)
    labels = scaler_y.fit_transform(labels.reshape(-1, 1)).flatten()

    # シーケンス生成
    X, y = create_sequences(features, labels, seq_len)

    # データ分割
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tensor変換
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1).to(device)

    # モデル構築
    model = get_model(model_type, input_size=len(selected_features), regression=True).to(device)

    # 損失関数
    if loss_type == "MSELoss":
        criterion = nn.MSELoss()
    elif loss_type == "BCELoss":
        criterion = nn.BCELoss()
    else:
        raise ValueError("指定された損失関数が無効です。")

    # 最適化手法
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=0.001)
    else:
        raise ValueError("指定された最適化手法が無効です。")

    # 学習ループ
    model.train()
    for epoch in range(30):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                print(f"Epoch [{epoch+1}/30], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")
            model.train()

    return model, scaler_X, scaler_y
