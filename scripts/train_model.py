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

class BioSignalWindowDataset(Dataset):
    """
    生体信号データのためのカスタムDatasetクラス。
    ウィンドウ単位でデータを動的に生成し、メモリ効率を向上させる。
    """
    def __init__(self, df, selected_features, window_size, scaler_X, scaler_y):
        self.df = df
        self.selected_features = selected_features
        self.window_size = window_size
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y
        
        # 'kss' カラムが存在するか確認
        if 'kss' not in self.df.columns:
            raise ValueError("DataFrameに 'kss' カラムが見つかりません。")

    def __len__(self):
        # 生成可能なウィンドウの総数を返す
        return len(self.df) - self.window_size + 1

    def __getitem__(self, idx):
        # 指定されたインデックスのデータウィンドウを取得
        window_df = self.df.iloc[idx : idx + self.window_size]
        
        # 特徴量とラベルを抽出
        features = window_df[self.selected_features].values
        label = window_df['kss'].mean() # ウィンドウ内のkssの平均値をラベルとする

        # データを正規化
        features_scaled = self.scaler_X.transform(features)
        label_scaled = self.scaler_y.transform(np.array([[label]]))

        return torch.tensor(features_scaled, dtype=torch.float32), torch.tensor(label_scaled, dtype=torch.float32).flatten()

# モデル学習関数
def train_model(data_path, model_type="LSTM", loss_type="MSELoss", optimizer_type="Adam",
                selected_features=None, window_size=60,
                lr=0.001, epochs=30, num_layers=2, hidden_size=64, hyper_params_modes=None,
                use_early_stopping=False, patience=10):
    
    df = pd.read_csv(data_path)
    total_data_points = len(df) # データ総数を取得

    # ハイパーパラメータの自動調整ロジック
    if hyper_params_modes:
        # 学習率の調整
        if hyper_params_modes["lr_mode"] == "自動調整":
            # データ量が多いほど学習率を小さくする（例: 10000データで0.001、100000データで0.0005）
            lr = max(0.0001, 0.001 * (1 - (total_data_points / 1000000) * 0.5)) # 最小値を設定
            print(f"自動調整: 学習率 = {lr:.6f}")

        # エポック数の調整
        if hyper_params_modes["epochs_mode"] == "自動調整":
            # データ量が多いほどエポック数を増やす（例: 10000データで30、100000データで50）
            epochs = int(30 + (total_data_points / 10000) * 2) # 1万データごとに2エポック追加
            epochs = min(epochs, 100) # 最大エポック数を設定
            print(f"自動調整: エポック数 = {epochs}")

        # モデル層の調整
        if hyper_params_modes["num_layers_mode"] == "自動調整":
            # データ量が多いほど層を増やす（例: 10000データで2、50000データで3）
            num_layers = int(2 + (total_data_points / 50000))
            num_layers = min(num_layers, 5) # 最大層数を設定
            print(f"自動調整: モデル層 = {num_layers}")

        # 隠れ層の調整
        if hyper_params_modes["hidden_size_mode"] == "自動調整":
            # データ量が多いほど隠れ層のサイズを増やす（例: 10000データで64、50000データで128）
            hidden_size = int(64 + (total_data_points / 10000) * 10)
            hidden_size = min(hidden_size, 256) # 最大隠れ層サイズを設定
            print(f"自動調整: 隠れ層 = {hidden_size}")

        # 早期終了設定のログ
        if use_early_stopping:
            print(f"早期終了: 有効 (Patience = {patience})")
        else:
            print("早期終了: 無効")

    if selected_features is None or not selected_features:
        raise ValueError("selected_features を1つ以上選択してください。")

    # 1. データ分割 (時間的順序を維持するため、シャッフルしない)
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # 2. スケーラーの準備と学習 (訓練データのみを使用)
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # 訓練データの特徴量とラベルを使ってスケーラーを学習させる
    # ウィンドウ化する前の全訓練データでスケーラーをfitする
    scaler_X.fit(train_df[selected_features].values)
    scaler_y.fit(train_df[['kss']].values)

    # ゼロ除算を回避するためのチェックと対策
    # 特徴量のスケーラー
    zero_std_features_indices = np.where(scaler_X.scale_ == 0)[0]
    if len(zero_std_features_indices) > 0:
        zero_std_features = [selected_features[i] for i in zero_std_features_indices]
        print(f"警告: 以下の特徴量の標準偏差が0です: {zero_std_features}")
        print("ゼロ除算を避けるため、これらの特徴量のスケーリングを調整します。")
        scaler_X.scale_[zero_std_features_indices] = 1.0

    # ラベルのスケーラー
    if scaler_y.scale_[0] == 0:
        print("警告: ラベル（kss）の標準偏差が0です。")
        print("ゼロ除算を避けるため、ラベルのスケーリングを調整します。")
        scaler_y.scale_[0] = 1.0

    # 3. カスタムDatasetのインスタンス化
    train_dataset = BioSignalWindowDataset(train_df, selected_features, window_size, scaler_X, scaler_y)
    val_dataset = BioSignalWindowDataset(val_df, selected_features, window_size, scaler_X, scaler_y)

    # 4. デバイスの決定とDataLoaderの作成
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory_flag = True if device.type == 'cuda' else False

    # num_workers > 0 を設定すると、データ読み込みが並列化され、特にGPU使用時に学習が高速化します。
    # pin_memory=True は、データをGPUに転送する際の速度を向上させます。(GPU利用時のみ有効)
    batch_size = 64
    num_workers = 4 # CPUコア数に応じて調整
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory_flag)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory_flag)

    # 5. モデルの構築
    model = get_model(model_type, input_size=len(selected_features), 
                      hidden_size=hidden_size, num_layers=num_layers, regression=True).to(device)

    # 6. 損失関数と最適化手法
    criterion = nn.MSELoss() if loss_type == "MSELoss" else nn.BCELoss()
    
    if optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
    elif optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise ValueError(f"指定された最適化手法が無効です: {optimizer_type}")

    # 7. 学習ループ
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

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

        # 5エポックごとに検証
        if (epoch + 1) % 5 == 0:
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

            # 早期終了のロジック
            if use_early_stopping:
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict() # 最も良いモデルの状態を保存
                    best_epoch = epoch + 1 # ベストエポックを記録
                else:
                    patience_counter += 1
                    print(f"  [Early Stopping] Patience: {patience_counter}/{patience}") # カウンターのログ
                    if patience_counter >= patience:
                        print(f"早期終了: 検証ロスが {patience} エポック改善しなかったため学習を停止します。")
                        print(f"  最終モデルはエポック {best_epoch} の状態をロードしました。") # ベストエポックのログ
                        break
    
    # 早期終了が有効で、かつベストモデルが保存されている場合、その状態をロード
    if use_early_stopping and best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, scaler_X, scaler_y, hidden_size


if __name__ == "__main__":
    import argparse
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
                                              window_size=60)

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