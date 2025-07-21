#推論用GUI

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import os
from app.inference import load_model, predict_awakenness

class AwakeApp:
    def __init__(self, root):
        # ウィンドウの基本設定
        self.root = root
        self.root.title("ドライバー覚醒度推定アプリ（推論モード）")
        self.root.geometry("1080x1080")

        # 各種データ保持用変数
        self.model = None
        self.data = None
        self.result = None
        self.selected_features = None

        # GUIウィジェットとグラフの設定
        self.setup_widgets()
        self.setup_plot()
        
        self.csv_path = None
        
        # ズーム・パン機能用変数
        self.press = None       #マウスプレスイベントを保持
        self.cur_xlim = None    #現在のｘ軸の表示範囲を保持
        self.cur_ylim = None    #現在のｙ軸の表示範囲を保持

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
        
        # リセットボタンを追加
        tk.Button(self.root, text="表示をリセット", command=self.reset_view).pack(pady=5)

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
        font_path = None
        for prop in fm.findSystemFonts(fontpaths=None, fontext='ttf'):
            if 'Meiryo' in fm.FontProperties(fname=prop).get_name():
                font_path = prop
                break
        
        if font_path:
            plt.rcParams['font.family'] = fm.FontProperties(fname=font_path).get_name()
        else:
            # フォントが見つからない場合のフォールバック（警告を表示するなど）
            print("Warning: Meiryo font not found. Japanese characters might not display correctly.")
            # MacOSの場合は 'Hiragino Sans GB' など
            if os.name == 'posix': # macOS/Linux
                plt.rcParams['font.family'] = 'Hiragino Sans GB' # macOS向け
            else: # その他のOS
                plt.rcParams['font.family'] = 'sans-serif' # デフォルトのサンセリフ

        plt.rcParams['axes.unicode_minus'] = False # 負の符号の文字化けを防ぐ

        # 覚醒度スコア推移グラフのキャンバス設定
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(pady=10)
        
        # マウスイベントのバインド
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)        
        
    def on_press(self, event):
        # マウスボタンが押されたときの処理（パン開始）
        if event.inaxes == self.ax:
            self.press = event.xdata, event.ydata
            self.cur_xlim = self.ax.get_xlim()
            self.cur_ylim = self.ax.get_ylim()

    def on_release(self, event):
        # マウスボタンが離されたときの処理（パン終了）
        self.press = None
        self.ax.figure.canvas.draw()

    def on_motion(self, event):
        # マウスが移動したときの処理（パン中）
        if self.press is None or event.inaxes != self.ax:
            return
        
        # マウスの現在位置と初期位置のデータ座標の差を計算
        xpress, ypress = self.press
        xnew, ynew = event.xdata, event.ydata

        # 軸の移動量を計算
        dx = xnew - xpress
        dy = ynew - ypress

        # 新しい軸の範囲を設定
        self.ax.set_xlim(self.cur_xlim[0] - dx, self.cur_xlim[1] - dx)
        self.ax.set_ylim(self.cur_ylim[0] - dy, self.cur_ylim[1] - dy)
        self.ax.figure.canvas.draw_idle() # アイドル時に再描画

    def on_scroll(self, event):
        # マウススクロールによるズーム処理
        if event.inaxes != self.ax:
            return

        xdata = event.xdata # マウスカーソルのXデータ座標
        ydata = event.ydata # マウスカーソルのYデータ座標

        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()

        x_left = cur_xlim[0]
        x_right = cur_xlim[1]
        y_bottom = cur_ylim[0]
        y_top = cur_ylim[1]

        # ズームファクター
        zoom_factor = 1.1 if event.step > 0 else 1/1.1 # スクロール方向で拡大・縮小を決定

        # 新しいX軸の範囲を計算
        new_x_left = xdata - (xdata - x_left) / zoom_factor
        new_x_right = xdata + (x_right - xdata) / zoom_factor

        # 新しいY軸の範囲を計算
        new_y_bottom = ydata - (ydata - y_bottom) / zoom_factor
        new_y_top = ydata + (y_top - ydata) / zoom_factor

        self.ax.set_xlim(new_x_left, new_x_right)
        self.ax.set_ylim(new_y_bottom, new_y_top)
        self.ax.figure.canvas.draw_idle() # アイドル時に再描画
        
    def reset_view(self):
        # グラフの表示をリセットする
        if self.result is not None and not self.result.empty:
            # データの最小値と最大値に基づいて軸の範囲を設定
            self.ax.set_xlim(self.result["timestamp"].min(), self.result["timestamp"].max())
            self.ax.set_ylim(self.result["awakenness_score"].min() - 0.1, self.result["awakenness_score"].max() + 0.1) # 少し余白を持たせる
            self.canvas.draw()
        else:
            messagebox.showwarning("警告", "表示をリセットするデータがありません。")

    def load_csv(self):
        # ユーザーがCSVファイルを選択し、読み込む処理
        file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if file_path:
            self.csv_path = file_path
            self.file_label.config(text=f"読み込みファイル: {os.path.basename(file_path)}")
            

    def run_inference(self):
        # 推定処理本体
        if not self.csv_path:
            messagebox.showwarning("警告", "CSVファイルを読み込んでください。")
            return
        
        model_file = self.model_combo.get()
        if not model_file:
            messagebox.showwarning("警告", "モデルファイルを選択してください。")
            return

        model_path = os.path.join("models", model_file)
        
        try:
            # モデル読み込みと、モデルが学習に使用した特徴量を取得
            self.model, self.selected_features, self.scaler_X_mean, self.scaler_X_scale, self.scaler_y_mean, self.scaler_y_scale = load_model(model_path)
        except Exception as e:
            messagebox.showerror("エラー", f"モデルの読み込みに失敗しました:\n{e}")
            return

        try:
            # CSVデータ読み込み
            self.data = pd.read_csv(self.csv_path)
            print("データ読み込み成功:", self.data.shape)

            # 読み込んだモデルのselected_featuresを使ってデータをフィルタリング
            # 必要なカラムがデータに存在するかチェック
            missing_cols = [col for col in self.selected_features if col not in self.data.columns]
            if missing_cols:
                messagebox.showerror("エラー", f"選択されたモデルに必要な生理データカラムが不足しています: {', '.join(missing_cols)}")
                return
            
            # predict_awakenness 関数に selected_features を渡す
            scores = predict_awakenness(self.model, self.data, self.selected_features, self.scaler_X_mean, self.scaler_X_scale, self.scaler_y_mean, self.scaler_y_scale)
            self.result = pd.DataFrame({
                "timestamp": range(len(scores)),
                "awakenness_score": scores
            })
            
            # NaNを除外してから描画
            valid = self.result.dropna()

            # グラフ描画
            self.ax.clear()
            self.ax.plot(valid["timestamp"], valid["awakenness_score"])
            self.ax.set_title("覚醒度推移グラフ")
            self.ax.set_xlabel("時刻 (index)")
            self.ax.set_ylabel("覚醒度スコア (0: 眠気強, 1: 覚醒状態)")
            self.canvas.draw()
            
            #グラフが描画されたら、現在の軸の範囲を保存
            self.cur_xlim = self.ax.get_xlim()
            self.cur_ylim = self.ax.get_ylim()

        except Exception as e:
            messagebox.showerror("エラー", f"推定中にエラーが発生しました:\n{e}")

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