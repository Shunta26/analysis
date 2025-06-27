#推論用GUI

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from app.inference import load_model, predict_awakenness

class AwakeApp:
    def __init__(self, root):
        # ウィンドウの基本設定
        self.root = root
        self.root.title("ドライバー覚醒度推定アプリ（推論モード）")
        self.root.geometry("850x650")

        # 各種データ保持用変数
        self.model = None
        self.data = None
        self.result = None

        # GUIウィジェットとグラフの設定
        self.setup_widgets()
        self.setup_plot()

    def setup_widgets(self):
        # モデル選択セクション（プルダウンと更新ボタン）
        tk.Label(self.root, text="使用モデルファイル（.pt）").pack()

        model_frame = tk.Frame(self.root)
        model_frame.pack(pady=5)

        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state="readonly", width=40)
        self.model_combo.pack(side=tk.LEFT)

        refresh_button = tk.Button(model_frame, text="更新", command=self.refresh_model_list)
        refresh_button.pack(side=tk.LEFT, padx=5)

        # モデルファイルの初期読み込み
        self.refresh_model_list()

        # CSV読み込みボタン
        tk.Button(self.root, text="CSVデータを選択", command=self.load_csv).pack(pady=5)

        # 推定実行ボタン
        tk.Button(self.root, text="覚醒度を推定", command=self.run_inference).pack(pady=5)

        # 結果保存ボタン
        tk.Button(self.root, text="結果をCSVとして保存", command=self.save_csv).pack(pady=5)

        # モード選択画面に戻るボタン
        tk.Button(self.root, text="← モード選択に戻る", command=self.back_to_menu).pack(pady=10)

        # 選択されたCSVファイル名の表示欄
        self.file_label = tk.Label(self.root, text="CSVファイル未選択", fg="blue")
        self.file_label.pack(pady=5)

    def refresh_model_list(self):
        # models/ フォルダ内の .pt ファイルをリスト化して更新
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
        self.model_combo["values"] = model_files
        if model_files:
            self.model_combo.current(0)

    def setup_plot(self):
        # 覚醒度スコア推移グラフのキャンバス設定
        self.fig, self.ax = plt.subplots(figsize=(8, 3))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=10)

    def load_csv(self):
        # ユーザーがCSVファイルを選択し、読み込む処理
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.data = pd.read_csv(file_path)
            self.file_label.config(text=f"読み込みファイル: {os.path.basename(file_path)}")
            print("データ読み込み成功:", self.data.shape)

    def run_inference(self):
        # 推定処理本体
        if self.data is None:
            messagebox.showwarning("警告", "CSVファイルを読み込んでください。")
            return

        model_file = self.model_combo.get()
        if not model_file:
            messagebox.showwarning("警告", "モデルファイルを選択してください。")
            return

        model_path = os.path.join("models", model_file)
        self.model = load_model(model_path, input_size=4)

        scores = predict_awakenness(self.model, self.data)
        self.result = pd.DataFrame({
            "timestamp": range(len(scores)),
            "awakenness_score": scores
        })
        
        
        # ✅ NaNを除外してから描画
        valid = self.result.dropna()

        # グラフ描画
        self.ax.clear()
        self.ax.plot(valid["timestamp"], valid["awakenness_score"])
        self.ax.set_title("覚醒度推移グラフ")
        self.ax.set_xlabel("時刻 (index)")
        self.ax.set_ylabel("覚醒度スコア (0: 眠気強, 1: 覚醒状態)")
        self.canvas.draw()

        # 推定結果のグラフ描画
        self.ax.clear()
        self.ax.plot(self.result["timestamp"], self.result["awakenness_score"])
        self.ax.set_title("覚醒度推移グラフ")
        self.ax.set_xlabel("時刻 (index)")
        self.ax.set_ylabel("覚醒度スコア (0: 眠気強, 1: 覚醒状態)")
        self.canvas.draw()

    def save_csv(self):
        # 推定結果をCSVファイルとして保存
        if self.result is not None:
            file_path = filedialog.asksaveasfilename(defaultextension=".csv")
            if file_path:
                self.result.to_csv(file_path, index=False)
                messagebox.showinfo("保存完了", f"結果を保存しました:\n{file_path}")
        else:
            messagebox.showwarning("警告", "保存する結果がありません。")

    def back_to_menu(self):
        # モード選択画面に戻る
        self.root.destroy()
        from main import main_menu
        main_menu()

# 外部呼び出し用関数（main.py から呼ばれる）
def launch_predict_gui():
    root = tk.Tk()
    app = AwakeApp(root)
    root.mainloop()