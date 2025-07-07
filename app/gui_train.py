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
        self.time_window_seconds = tk.IntVar(value=60) # 新しい変数

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
        ttk.Label(settings_win, text="時間窓 (秒): ").pack(pady=5)
        temp_time_window_seconds = tk.IntVar(value=self.time_window_seconds.get())
        ttk.Entry(settings_win, textvariable=temp_time_window_seconds).pack()

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
                new_time_window = temp_time_window_seconds.get()
                if not isinstance(new_time_window, int) or new_time_window <= 0:
                    raise ValueError("時間窓は正の整数で入力してください。")
                self.time_window_seconds.set(new_time_window)
            except ValueError as e:
                messagebox.showerror("入力エラー", str(e))
                return
            except Exception:
                messagebox.showerror("入力エラー", "時間窓は有効な数値を入力してください。")
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
        time_window = self.time_window_seconds.get() # 時間窓を取得

        threading.Thread(target=self.run_training, args=(model, optimizer, loss_func, self.csv_path, features, time_window), daemon=True).start()

    def run_training(self, model, optimizer, loss_func, csv_path, selected_features, time_window_seconds):
        try:
            trained_model, _, _ = train_model(
                csv_path,
                model_type=model.lower(),
                optimizer_type=optimizer,
                loss_type=loss_func,
                selected_features=selected_features,
                time_window_seconds=time_window_seconds # 時間窓を渡す
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
                    "time_window_seconds": time_window_seconds, # JSONにも追加
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
    