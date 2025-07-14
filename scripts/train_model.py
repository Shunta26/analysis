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
                selected_features=None, window_size=60,
                lr=0.001, epochs=30, num_layers=2, hidden_size=64): # 引数追加
    df = pd.read_csv(data_path)

    if selected_features is None or not selected_features:
        raise ValueError("selected_features を1つ以上選択してください。")

    # 指定された個数でデータを分割
    windowed_list = []
    # スライド幅を1としてウィンドウを作成
    for i in range(len(df) - window_size + 1):
        window_df = df.iloc[i:i + window_size]
        
        features = window_df[selected_features].values
        kss_mean = window_df['kss'].mean()
        windowed_list.append((features, kss_mean))

    # ウィンドウが作成できなかった場合のエラー
    if not windowed_list:
        raise ValueError("データが少なすぎるため、ウィンドウを作成できませんでした。")
        
    X_list, y_list = zip(*windowed_list)
    
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

    # データセットとデータローダーの作成
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # モデル構築
    model = get_model(model_type, input_size=len(selected_features), 
                      hidden_size=hidden_size, num_layers=num_layers, regression=True).to(device) # 引数追加

    # 損失関数
    if loss_type == "MSELoss":
        criterion = nn.MSELoss()
    elif loss_type == "BCELoss":
        criterion = nn.BCELoss()
    else:
        raise ValueError("指定された損失関数が無効です。")

    # 最適化手法
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr) # 引数追加
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr) # 引数追加
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr) # 引数追加
    else:
        raise ValueError("指定された最適化手法が無効です。")

    # 学習ループ
    for epoch in range(epochs): # 引数追加
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # 5エポックごとに検証
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
                    val_running_loss += val_loss.item()
            
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {val_running_loss / len(val_loader):.4f}")

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
                                              selected_features=selected_features,
                                              window_size=60)  # ここを修正

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
