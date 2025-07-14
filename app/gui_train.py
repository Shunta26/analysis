#学習用GUI
# app/gui_train.py
# gui_train.py 全体（生理データ選択・モデル等設定付き学習GUI）

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import os
import threading
import torch
import json
from datetime import datetime
from scripts.train_model import train_model

class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ドライバー覚醒度推定アプリ（学習モード）")
        self.root.geometry("600x600")

        # 各設定用変数
        self.selected_features = {
            'pupil': tk.BooleanVar(value=True),
            'eda': tk.BooleanVar(value=True),
            'eeg': tk.BooleanVar(value=True),
            'hr': tk.BooleanVar(value=True)
        }

        self.model_var = tk.StringVar()
        self.optimizer_var = tk.StringVar()
        self.loss_var = tk.StringVar()
        self.save_model_var = tk.BooleanVar(value=True)
        self.csv_path = None
        self.window_size = tk.IntVar(value=60)

        # 詳細設定のデフォルト値
        self.lr = tk.DoubleVar(value=0.001)
        self.epochs = tk.IntVar(value=30)
        self.num_layers = tk.IntVar(value=2)
        self.hidden_size = tk.IntVar(value=64)

        self.setup_widgets()

    def setup_widgets(self):
        # モデル選択
        ttk.Label(self.root, text="モデル選択:").pack(pady=5)
        self.model_combo = ttk.Combobox(self.root, textvariable=self.model_var, state="readonly",
                                        values=["LSTM", "GRU", "RNN"])
        self.model_combo.current(0)
        self.model_combo.pack()

        # 最適化関数選択
        ttk.Label(self.root, text="最適化関数選択:").pack(pady=5)
        self.optimizer_combo = ttk.Combobox(self.root, textvariable=self.optimizer_var, state="readonly",
                                            values=["Adam", "SGD", "RMSprop"])
        self.optimizer_combo.current(0)
        self.optimizer_combo.pack()

        # 損失関数選択
        ttk.Label(self.root, text="損失関数選択:").pack(pady=5)
        self.loss_combo = ttk.Combobox(self.root, textvariable=self.loss_var, state="readonly",
                                       values=["MSELoss", "BCELoss"])
        self.loss_combo.current(0)
        self.loss_combo.pack()

        # CSVファイル読み込み
        self.file_label = ttk.Label(self.root, text="CSVファイルが選択されていません")
        self.file_label.pack(pady=10)
        ttk.Button(self.root, text="CSVファイルを選択", command=self.select_file).pack()

        # 詳細設定ボタン
        ttk.Button(self.root, text="詳細設定", command=self.open_settings_window).pack(pady=10)

        # モデル保存チェック
        ttk.Checkbutton(self.root, text="学習後にモデルを保存する", variable=self.save_model_var).pack(pady=15)

        # 学習実行ボタン
        ttk.Button(self.root, text="学習開始", command=self.start_training).pack(pady=15)

        # 戻るボタン
        ttk.Button(self.root, text="← モード選択に戻る", command=self.back_to_menu).pack(pady=15)

    def open_settings_window(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("詳細設定") # タイトルも変更
        settings_win.geometry("300x300") # ジオメトリを少し大きくする

        # 生理データ選択
        ttk.Label(settings_win, text="使用する生理データの選択:").pack(pady=5)
        temp_vars = {key: tk.BooleanVar(value=var.get()) for key, var in self.selected_features.items()}

        for key in temp_vars:
            tk.Checkbutton(settings_win, text=key.upper(), variable=temp_vars[key]).pack(anchor='w')

        # 時間窓（秒数）設定
        ttk.Label(settings_win, text="データ個数: ").pack(pady=5)
        temp_window_size = tk.IntVar(value=self.window_size.get())
        ttk.Entry(settings_win, textvariable=temp_window_size).pack()

        # 学習率
        ttk.Label(settings_win, text="学習率:").pack(pady=5)
        temp_lr = tk.DoubleVar(value=self.lr.get())
        ttk.Entry(settings_win, textvariable=temp_lr).pack()

        # エポック数
        ttk.Label(settings_win, text="エポック数:").pack(pady=5)
        temp_epochs = tk.IntVar(value=self.epochs.get())
        ttk.Entry(settings_win, textvariable=temp_epochs).pack()

        # モデル層
        ttk.Label(settings_win, text="モデル層:").pack(pady=5)
        temp_num_layers = tk.IntVar(value=self.num_layers.get())
        ttk.Entry(settings_win, textvariable=temp_num_layers).pack()

        # 隠れ層
        ttk.Label(settings_win, text="隠れ層:").pack(pady=5)
        temp_hidden_size = tk.IntVar(value=self.hidden_size.get())
        ttk.Entry(settings_win, textvariable=temp_hidden_size).pack()

        def save_and_close():
            # 生理データ選択の保存
            if not any(var.get() for var in temp_vars.values()):
                result = messagebox.askyesno("警告", "1つも選択されていません。変更を保存しますか？")
                if not result:
                    return
            for key in temp_vars:
                self.selected_features[key].set(temp_vars[key].get())
            
            # 時間窓（秒数）の保存とバリデーション
            try:
                self.window_size.set(temp_window_size.get())
                self.lr.set(temp_lr.get())
                self.epochs.set(temp_epochs.get())
                self.num_layers.set(temp_num_layers.get())
                self.hidden_size.set(temp_hidden_size.get())
            except ValueError as e:
                messagebox.showerror("入力エラー", str(e))
                return
            except Exception:
                messagebox.showerror("入力エラー", "有効な数値を入力してください。")
                return

            settings_win.destroy()

        def cancel_and_close():
            settings_win.destroy()

        btn_frame = ttk.Frame(settings_win)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="保存して戻る", command=save_and_close).pack(side='right', padx=10)
        ttk.Button(btn_frame, text="変更を削除して戻る", command=cancel_and_close, style="Danger.TButton").pack(side='left', padx=10)

    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSVファイル", "*.csv")])
        if filepath:
            self.csv_path = filepath
            self.file_label.config(text=f"選択ファイル: {filepath}")

    def start_training(self):
        if not self.csv_path:
            messagebox.showwarning("警告", "CSVファイルを選択してください。")
            return

        features = [key for key, var in self.selected_features.items() if var.get()]
        if not features:
            messagebox.showwarning("警告", "最低1つの生理データを選択してください。")
            return

        model = self.model_var.get()
        optimizer = self.optimizer_var.get()
        loss_func = self.loss_var.get()
        window_size = self.window_size.get()
        lr = self.lr.get()
        epochs = self.epochs.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()

        threading.Thread(target=self.run_training, args=(model, optimizer, loss_func, self.csv_path, features, window_size, lr, epochs, num_layers, hidden_size), daemon=True).start()

    def run_training(self, model, optimizer, loss_func, csv_path, selected_features, window_size, lr, epochs, num_layers, hidden_size):
        try:
            trained_model, _, _ = train_model(
                csv_path,
                model_type=model.lower(),
                optimizer_type=optimizer,
                loss_type=loss_func,
                selected_features=selected_features,
                window_size=window_size,
                lr=lr,
                epochs=epochs,
                num_layers=num_layers,
                hidden_size=hidden_size
            )

            if self.save_model_var.get():
                os.makedirs("models", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                base_filename = f"{model}_{timestamp}"

                # モデル保存
                model_path = os.path.join("models", f"{base_filename}.pt")
                torch.save(trained_model.state_dict(), model_path)

                # JSONファイル保存
                config = {
                    "model_type": model,
                    "optimizer": optimizer,
                    "loss_function": loss_func,
                    "selected_features": selected_features,
                    "window_size": window_size,
                    "lr": lr,
                    "epochs": epochs,
                    "num_layers": num_layers,
                    "hidden_size": hidden_size,
                    "timestamp": timestamp
                }
                json_path = os.path.join("models", f"{base_filename}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)

            messagebox.showinfo("完了", "学習が完了しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"学習中にエラーが発生しました:\n{e}")

    def back_to_menu(self):
        self.root.destroy()
        from main import main_menu
        main_menu()

# 起動関数

def launch_train_gui():
    root = tk.Tk()
    app = TrainApp(root)
    root.mainloop()
    