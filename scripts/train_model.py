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
from scripts.model_factory import get_model


# モデル学習関数
def train_model(data_path, model_type="LSTM", loss_type="MSELoss", optimizer_type="Adam",
                selected_features=None, time_window_seconds=60):
    df = pd.read_csv(data_path, parse_dates=['timestamp'])
    df.set_index('timestamp', inplace=True)

    if selected_features is None or not selected_features:
        raise ValueError("selected_features を1つ以上選択してください。")

    # 指定された秒数でリサンプリング
    resampled_list = []
    expected_rows = time_window_seconds
    for group_name, group_df in df.resample(f'{time_window_seconds}S'):
        if len(group_df) == expected_rows:
            features = group_df[selected_features].values
            kss_mean = group_df['kss'].mean()
            resampled_list.append((features, kss_mean))

    # 指定された秒数ちょうどのデータがないものは除外されている
    if not resampled_list:
        raise ValueError("指定された秒数ちょうどのデータがありません。データを確認してください。")
        
    X_list, y_list = zip(*resampled_list)
    
    X = np.array(X_list)
    y = np.array(y_list)


    # 正規化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    # Xの形状は (サンプル数, seq_len, 特徴量数) なので、2Dに変換して正規化
    n_samples, seq_len, n_features = X.shape
    X = scaler_X.fit_transform(X.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
    y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

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


import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="モデルの学習を行います。")
    parser.add_argument("--model_type", type=str, default="LSTM", choices=["LSTM", "GRU", "RNN"],
                        help="学習するモデルの種類を選択します (LSTM, GRU, RNN)")
    args = parser.parse_args()

    data_path = "../data/train/train_bio_driver_data.csv"
    selected_features = ["pupil", "eda", "eeg", "hr"]
    
    print(f"\n----- {args.model_type} モデルの学習を開始します -----")
    
    model, scaler_X, scaler_y = train_model(data_path, 
                                              model_type=args.model_type, 
                                              selected_features=selected_features)

    # モデル保存
    now = datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.model_type.upper()}_{now}.pt")
    torch.save(model.state_dict(), model_path)

    # スケーラーの保存（任意）
    # import joblib
    # joblib.dump(scaler_X, f"scaler_X_{now}.pkl")
    # joblib.dump(scaler_y, f"scaler_y_{now}.pkl")

    print(f"モデルが {model_path} に保存されました。")
