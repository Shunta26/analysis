#学習処理（データ読み込み～学習・評価・返却）
# scripts/train_model.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import os
from scripts.model_factory import get_model
import random

class BioSignalWindowDataset(Dataset):
    """
    生体信号データのためのカスタムDatasetクラス。
    事前に生成されたウィンドウのリストとラベル付け方法を受け取る。
    """
    def __init__(self, windows, scaler_X, scaler_y, labeling_method="last"):
        self.windows = windows
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        self.labeling_method = labeling_method

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx]
        
        features = window[:, :-1]
        kss_values = window[:, -1]

        if self.labeling_method == "average":
            label = np.mean(kss_values)
        elif self.labeling_method == "first":
            label = kss_values[0]
        elif self.labeling_method == "last":
            label = kss_values[-1]
        else:
            raise ValueError(f"無効なラベリング方法です: {self.labeling_method}")

        features_scaled = self.scaler_X.transform(features)
        label_scaled = self.scaler_y.transform(np.array([[label]]))

        return torch.tensor(features_scaled, dtype=torch.float32), torch.tensor(label_scaled, dtype=torch.float32).flatten()

# モデル学習関数
def train_model(data_paths, model_type="LSTM", loss_type="MSELoss", optimizer_type="Adam",
                selected_features=None, window_size=60, labeling_method="last",
                lr=0.001, epochs=30, num_layers=2, hidden_size=64, use_dropout=True, dropout_rate=0.2, validation_interval=5, hyper_params_modes=None,
                use_early_stopping=False, patience=10):

    if isinstance(data_paths, str):
        data_paths = [data_paths]
    
    if selected_features is None or not selected_features:
        raise ValueError("selected_features を1つ以上選択してください。")

    all_windows = []
    for path in data_paths:
        df = pd.read_csv(path)
        if 'kss' not in df.columns:
            print(f"警告: {path} に 'kss' カラムが見つかりません。スキップします。")
            continue
        
        cols_to_use = selected_features + ['kss']
        df_filtered = df[cols_to_use]
        data_np = df_filtered.to_numpy()
        
        num_windows = len(data_np) - window_size + 1
        for i in range(num_windows):
            all_windows.append(data_np[i : i + window_size])

    if not all_windows:
        raise ValueError("有効な学習データウィンドウが1つも生成されませんでした。")

    total_data_points = sum(len(w) for w in all_windows)

    if hyper_params_modes:
        if hyper_params_modes["lr_mode"] == "自動調整":
            lr = max(0.0001, 0.001 * (1 - (total_data_points / 1000000) * 0.5))
            print(f"自動調整: 学習率 = {lr:.6f}")
        if hyper_params_modes["epochs_mode"] == "自動調整":
            epochs = int(30 + (total_data_points / 10000) * 2)
            epochs = min(epochs, 100)
            print(f"自動調整: エポック数 = {epochs}")
        if hyper_params_modes["num_layers_mode"] == "自動調整":
            num_layers = int(2 + (total_data_points / 50000))
            num_layers = min(num_layers, 5)
            print(f"自動調整: モデル層 = {num_layers}")
        if hyper_params_modes["hidden_size_mode"] == "自動調整":
            hidden_size = int(64 + (total_data_points / 10000) * 10)
            hidden_size = min(hidden_size, 256)
            print(f"自動調整: 隠れ層 = {hidden_size}")
        if hyper_params_modes["dropout_mode"] == "自動調整":
            dropout_rate = max(0.1, 0.5 * (1 - (total_data_points / 1000000) * 0.8))
            print(f"自動調整: ドロップアウト率 = {dropout_rate:.2f}")
        if hyper_params_modes["validation_interval_mode"] == "自動調整":
            validation_interval = max(1, int(epochs / 10)) # エポック数の10分の1を推奨
            print(f"自動調整: 検証間隔 = {validation_interval} エポック")
        if use_early_stopping:
            print(f"早期終了: 有効 (Patience = {patience})")
        else:
            print("早期終了: 無効")

    print("\n----- 最終学習設定 -----")
    print(f"モデルタイプ: {model_type}")
    print(f"最適化関数: {optimizer_type}")
    print(f"損失関数: {loss_type}")
    print(f"使用生理指標: {', '.join(selected_features)}")
    print(f"ラベリング方法: {labeling_method}")
    print(f"ウィンドウサイズ: {window_size}")
    print(f"学習率: {lr:.6f}")
    print(f"エポック数: {epochs}")
    print(f"モデル層: {num_layers}")
    print(f"隠れ層: {hidden_size}")
    if use_dropout:
        print(f"ドロップアウト率: {dropout_rate:.2f}")
    else:
        print("ドロップアウト: 無効")
    print(f"検証間隔: {validation_interval} エポック")
    print(f"早期終了: {'有効' if use_early_stopping else '無効'}")
    if use_early_stopping:
        print(f"Patience: {patience}")
    print("----------------------")

    random.shuffle(all_windows)
    train_size = int(len(all_windows) * 0.8)
    train_windows = all_windows[:train_size]
    val_windows = all_windows[train_size:]

    def get_labels(windows, method):
        if method == "average":
            return np.array([np.mean(w[:, -1]) for w in windows]).reshape(-1, 1)
        elif method == "first":
            return np.array([w[0, -1] for w in windows]).reshape(-1, 1)
        elif method == "last":
            return np.array([w[-1, -1] for w in windows]).reshape(-1, 1)
        else:
            raise ValueError(f"無効なラベリング方法です: {method}")

    all_train_features = np.vstack([w[:, :-1] for w in train_windows])
    all_train_labels = get_labels(train_windows, labeling_method)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    scaler_X.fit(all_train_features)
    scaler_y.fit(all_train_labels)

    if np.any(scaler_X.scale_ == 0):
        print("警告: 特徴量の標準偏差が0です。スケーリングを調整します。")
        scaler_X.scale_[scaler_X.scale_ == 0] = 1.0
    if scaler_y.scale_[0] == 0:
        print("警告: ラベル（kss）の標準偏差が0です。スケーリングを調整します。")
        scaler_y.scale_[0] = 1.0

    train_dataset = BioSignalWindowDataset(train_windows, scaler_X, scaler_y, labeling_method)
    val_dataset = BioSignalWindowDataset(val_windows, scaler_X, scaler_y, labeling_method)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory_flag = True if device.type == 'cuda' else False
    batch_size = 64
    num_workers = 4
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag)

    model = get_model(model_type, input_size=len(selected_features), 
                      hidden_size=hidden_size, num_layers=num_layers, use_dropout=use_dropout, dropout_rate=dropout_rate, regression=True).to(device)

    criterion = nn.MSELoss() if loss_type == "MSELoss" else nn.BCELoss()
    optimizer_class = {"Adam": optim.Adam, "SGD": optim.SGD, "RMSprop": optim.RMSprop}.get(optimizer_type)
    if optimizer_class is None:
        raise ValueError(f"指定された最適化手法が無効です: {optimizer_type}")
    optimizer = optimizer_class(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        if (epoch + 1) % validation_interval == 0:
            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for val_inputs, val_labels in val_loader:
                    val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)
                    val_outputs = model(val_inputs)
                    val_loss = criterion(val_outputs, val_labels)
                    val_running_loss += val_loss.item()
            
            current_val_loss = val_running_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Val Loss: {current_val_loss:.4f}")

            if use_early_stopping:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict()
                    best_epoch = epoch + 1
                else:
                    patience_counter += 1
                    print(f"  [Early Stopping] Patience: {patience_counter}/{patience}")
                    if patience_counter >= patience:
                        print(f"早期終了: 検証ロスが {patience} エポック改善しなかったため学習を停止します。")
                        if best_model_state:
                           print(f"  最終モデルはエポック {best_epoch} の状態をロードしました。")
                        break
    
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, scaler_X, scaler_y, hidden_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="モデルの学習を行います。")
    parser.add_argument("--model_type", type=str, default="LSTM", choices=["LSTM", "GRU", "RNN", "Transformer"],
                        help="学習するモデルの種類を選択します (LSTM, GRU, RNN, Transformer)")
    parser.add_argument("--labeling", type=str, default="last", choices=["average", "first", "last"], help="ラベル付け方法を選択します。")
    parser.add_argument("--dropout", type=float, default=0.2, help="ドロップアウト率を指定します。")
    parser.add_argument("--no_dropout", action="store_true", help="ドロップアウトを使用しません。")
    parser.add_argument("--validation_interval", type=int, default=5, help="検証を実行するエポック間隔を指定します。")
    args = parser.parse_args()

    data_paths = [
        "../data/train/train_bio_driver_data.csv",
        "../data/train/testdata2.csv"
    ]
    selected_features = ["pupil", "eda", "eeg", "hr"]
    
    print(f"\n----- {args.model_type} モデルの学習を開始します (ラベリング: {args.labeling}) -----")
    
    model, scaler_X, scaler_y, _ = train_model(
        data_paths, 
        model_type=args.model_type, 
        selected_features=selected_features,
        window_size=60,
        labeling_method=args.labeling,
        use_dropout=not args.no_dropout,
        dropout_rate=args.dropout,
        validation_interval=args.validation_interval
    )

    now = datetime.now().strftime("%Y%m%d_%H%M")
    model_dir = "../models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{args.model_type.upper()}_{now}.pt")
    torch.save(model.state_dict(), model_path)

    print(f"モデルが {model_path} に保存されました。")