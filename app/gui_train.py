#学習用GUI
# app/gui_train.py

import os
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import pandas as pd
import torch
from scripts.train_model import train_model  # train_model関数をscripts/train_model.pyに分離して管理
from datetime import datetime

class TrainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("覚醒度推定モデル 学習GUI")
        self.root.geometry("500x480")

        # モデル選択
        ttk.Label(root, text="モデル選択:").pack(pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(root, textvariable=self.model_var, state="readonly",
                                        values=["LSTM", "GRU", "RNN"])
        self.model_combo.current(0)
        self.model_combo.pack()

        # 最適化関数選択
        ttk.Label(root, text="最適化関数選択:").pack(pady=5)
        self.optimizer_var = tk.StringVar()
        self.optimizer_combo = ttk.Combobox(root, textvariable=self.optimizer_var, state="readonly",
                                            values=["Adam", "SGD", "RMSprop"])
        self.optimizer_combo.current(0)
        self.optimizer_combo.pack()

        # 損失関数選択
        ttk.Label(root, text="損失関数選択:").pack(pady=5)
        self.loss_var = tk.StringVar()
        self.loss_combo = ttk.Combobox(root, textvariable=self.loss_var, state="readonly",
                                       values=["MSELoss", "BCELoss"])
        self.loss_combo.current(0)
        self.loss_combo.pack()

        # ファイル選択
        self.csv_path = None
        self.file_label = ttk.Label(root, text="CSVファイルが選択されていません")
        self.file_label.pack(pady=10)
        self.file_button = ttk.Button(root, text="CSVファイルを選択", command=self.select_file)
        self.file_button.pack()

        # モデル保存チェック
        self.save_model_var = tk.BooleanVar(value=True)
        self.save_check = ttk.Checkbutton(root, text="学習後にモデルを保存する", variable=self.save_model_var)
        self.save_check.pack(pady=10)

        # 学習ボタン
        self.train_button = ttk.Button(root, text="学習開始", command=self.start_training)
        self.train_button.pack(pady=20)

        # 戻るボタン
        ttk.Button(root, text="← モード選択に戻る", command=self.back_to_menu).pack(pady=10)

    def back_to_menu(self):
        self.root.destroy()
        from main import main_menu
        main_menu()

    def select_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSVファイル", "*.csv")])
        if filepath:
            self.csv_path = filepath
            self.file_label.config(text=f"選択ファイル: {filepath}")

    def start_training(self):
        if not self.csv_path:
            messagebox.showwarning("警告", "CSVファイルを選択してください。")
            return
        model = self.model_var.get()
        optimizer = self.optimizer_var.get()
        loss_func = self.loss_var.get()
        threading.Thread(target=self.run_training, args=(model, optimizer, loss_func, self.csv_path), daemon=True).start()

    def run_training(self, model, optimizer, loss_func, csv_path):
        try:
            trained_model, _, _ = train_model(csv_path, model_type=model.lower(), loss_type=loss_func, optimizer_type=optimizer)

            if self.save_model_var.get():
                # モデルの自動保存処理
                if self.save_model_var.get():
                    os.makedirs("models", exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                    filename = f"{model}_{timestamp}.pt"
                    save_path = os.path.join("models", filename)
                    torch.save(trained_model.state_dict(), save_path)
                    messagebox.showinfo("保存完了", f"モデルを保存しました:\n{save_path}")

            messagebox.showinfo("完了", f"学習が完了しました.")
        except Exception as e:
            messagebox.showerror("エラー", f"学習中にエラーが発生しました:\n{e}")

# 起動関数
def launch_train_gui():
    root = tk.Tk()
    app = TrainApp(root)
    root.mainloop()

