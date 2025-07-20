

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
        self.selected_features = {}

        self.model_var = tk.StringVar()
        self.optimizer_var = tk.StringVar()
        self.loss_var = tk.StringVar()
        self.save_model_var = tk.BooleanVar(value=True)
        self.csv_paths = None
        self.window_size = tk.IntVar(value=60)
        self.labeling_method_var = tk.StringVar(value="last") # 追加

        # 詳細設定のデフォルト値
        self.lr = tk.DoubleVar(value=0.001)
        self.epochs = tk.IntVar(value=30)
        self.num_layers = tk.IntVar(value=2)
        self.hidden_size = tk.IntVar(value=64)

        # 早期終了設定
        self.early_stopping_var = tk.BooleanVar(value=False)
        self.patience_var = tk.IntVar(value=10)

        # 自動/手動調整モード用の変数
        self.lr_mode = tk.StringVar(value="自動調整")
        self.epochs_mode = tk.StringVar(value="自動調整")
        self.num_layers_mode = tk.StringVar(value="自動調整")
        self.hidden_size_mode = tk.StringVar(value="自動調整")
        self.dropout_var = tk.BooleanVar(value=True)
        self.dropout_rate_var = tk.DoubleVar(value=0.2)
        self.dropout_mode = tk.StringVar(value="自動調整")

        self.validation_interval_var = tk.IntVar(value=5)
        self.validation_interval_mode = tk.StringVar(value="自動調整")

        self.load_features_from_config()
        self.setup_widgets()

    def load_features_from_config(self):
        try:
            # config.jsonはプロジェクトのルートにあると想定
            with open("config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
                features = config.get("features", [])
                for feature in features:
                    self.selected_features[feature] = tk.BooleanVar(value=True)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            messagebox.showwarning("設定ファイルエラー", f"config.jsonの読み込みに失敗しました。\n{e}\nデフォルトの指標を使います。")
            # フォールバック
            default_features = ['pupil', 'eda', 'eeg', 'hr']
            for feature in default_features:
                self.selected_features[feature] = tk.BooleanVar(value=True)

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
        self.file_label = ttk.Label(self.root, text="CSVファイルが選択されていません", wraplength=580, justify=tk.LEFT)
        self.file_label.pack(pady=10)
        ttk.Button(self.root, text="CSVファイルを選択", command=self.select_files).pack()

        # 詳細設定ボタン
        ttk.Button(self.root, text="詳細設定", command=self.open_settings_window).pack(pady=10)

        # モデル保存チェック
        ttk.Checkbutton(self.root, text="学習後にモデルを保存する", variable=self.save_model_var).pack(pady=15)

        # 学習実行ボタン
        self.train_button = ttk.Button(self.root, text="学習開始", command=self.start_training)
        self.train_button.pack(pady=15)

        # ステータスラベル
        self.status_label = ttk.Label(self.root, text="")
        self.status_label.pack(pady=5)

        # 戻るボタン
        ttk.Button(self.root, text="← モード選択に戻る", command=self.back_to_menu).pack(pady=15)

    def open_settings_window(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("詳細設定")
        settings_win.geometry("420x700") # 幅を少し広げてスクロールバーに対応

        # --- スクロール可能なフレームを作成 ---
        main_frame = ttk.Frame(settings_win)
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(main_frame)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(
                scrollregion=canvas.bbox("all")
            )
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # --- ウィジェットの状態を切り替える関数 ---
        def toggle_entry(entry_widget, mode_var):
            if mode_var.get() == "自動調整":
                entry_widget.config(state="disabled")
            else:
                entry_widget.config(state="normal")

        # --- 設定項目を作成するヘルパー関数 ---
        def create_setting_row(parent, label_text, mode_var, value_var):
            frame = ttk.Frame(parent)
            frame.pack(fill='x', padx=10, pady=5)

            ttk.Label(frame, text=label_text, width=12).pack(side='left')
            
            combo = ttk.Combobox(frame, textvariable=mode_var, values=["手動調整", "自動調整"], width=10, state="readonly")
            combo.pack(side='left', padx=5)
            
            entry = ttk.Entry(frame, textvariable=value_var, width=15)
            entry.pack(side='left')

            combo.bind("<<ComboboxSelected>>", lambda event: toggle_entry(entry, mode_var))
            toggle_entry(entry, mode_var)
            return entry

        # --- 各種設定（scrollable_frameに追加） ---
        feature_frame = ttk.LabelFrame(scrollable_frame, text="使用する生理データ")
        feature_frame.pack(fill='x', padx=10, pady=10)
        temp_vars = {key: tk.BooleanVar(value=var.get()) for key, var in self.selected_features.items()}
        for key in temp_vars:
            tk.Checkbutton(feature_frame, text=key.upper(), variable=temp_vars[key]).pack(anchor='w', padx=10)

        # ラベリング方法設定
        labeling_frame = ttk.LabelFrame(scrollable_frame, text="ラベル付け方法")
        labeling_frame.pack(fill='x', padx=10, pady=10)
        temp_labeling_method = tk.StringVar(value=self.labeling_method_var.get())
        labeling_combo = ttk.Combobox(labeling_frame, textvariable=temp_labeling_method, 
                                        values=["average", "first", "last"], state="readonly")
        labeling_combo.pack(padx=10, pady=5)

        param_frame = ttk.LabelFrame(scrollable_frame, text="ハイパーパラメータ設定")
        param_frame.pack(fill='x', padx=10, pady=10)

        temp_lr_mode = tk.StringVar(value=self.lr_mode.get())
        temp_epochs_mode = tk.StringVar(value=self.epochs_mode.get())
        temp_num_layers_mode = tk.StringVar(value=self.num_layers_mode.get())
        temp_hidden_size_mode = tk.StringVar(value=self.hidden_size_mode.get())
        
        temp_lr = tk.DoubleVar(value=self.lr.get())
        temp_epochs = tk.IntVar(value=self.epochs.get())
        temp_num_layers = tk.IntVar(value=self.num_layers.get())
        temp_hidden_size = tk.IntVar(value=self.hidden_size.get())
        temp_window_size = tk.IntVar(value=self.window_size.get())

        temp_early_stopping_var = tk.BooleanVar(value=self.early_stopping_var.get())
        temp_patience_var = tk.IntVar(value=self.patience_var.get())

        data_frame = ttk.Frame(param_frame)
        data_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(data_frame, text="データ個数:", width=12).pack(side='left')
        ttk.Entry(data_frame, textvariable=temp_window_size, width=28).pack(side='left', padx=5)

        create_setting_row(param_frame, "学習率:", temp_lr_mode, temp_lr)
        create_setting_row(param_frame, "エポック数:", temp_epochs_mode, temp_epochs)
        create_setting_row(param_frame, "モデル層:", temp_num_layers_mode, temp_num_layers)
        create_setting_row(param_frame, "隠れ層:", temp_hidden_size_mode, temp_hidden_size)

        # ドロップアウト設定
        dropout_frame = ttk.LabelFrame(scrollable_frame, text="ドロップアウト設定")
        dropout_frame.pack(fill='x', padx=10, pady=10)

        temp_dropout_var = tk.BooleanVar(value=self.dropout_var.get())
        tk.Checkbutton(dropout_frame, text="ドロップアウトを有効にする", variable=temp_dropout_var).pack(anchor='w', padx=10)

        temp_dropout_rate_var = tk.DoubleVar(value=self.dropout_rate_var.get())
        temp_dropout_mode = tk.StringVar(value=self.dropout_mode.get())
        create_setting_row(dropout_frame, "ドロップアウト率:", temp_dropout_mode, temp_dropout_rate_var)

        # 検証間隔設定
        validation_frame = ttk.LabelFrame(scrollable_frame, text="検証間隔設定")
        validation_frame.pack(fill='x', padx=10, pady=10)

        temp_validation_interval_var = tk.IntVar(value=self.validation_interval_var.get())
        temp_validation_interval_mode = tk.StringVar(value=self.validation_interval_mode.get())
        create_setting_row(validation_frame, "検証間隔(Epoch):", temp_validation_interval_mode, temp_validation_interval_var)

        es_frame = ttk.LabelFrame(scrollable_frame, text="早期終了設定")
        es_frame.pack(fill='x', padx=10, pady=10)
        tk.Checkbutton(es_frame, text="早期終了を有効にする", variable=temp_early_stopping_var).pack(anchor='w', padx=10)
        
        patience_frame = ttk.Frame(es_frame)
        patience_frame.pack(fill='x', padx=10, pady=5)
        ttk.Label(patience_frame, text="Patience (エポック数):").pack(side='left')
        ttk.Entry(patience_frame, textvariable=temp_patience_var, width=10).pack(side='left')

        # 保存・キャンセルボタンはウィンドウ下部に固定
        btn_frame = ttk.Frame(settings_win)
        btn_frame.pack(fill='x', pady=10, side='bottom')

        def unbind_mousewheel():
            canvas.unbind_all("<MouseWheel>")

        def save_and_close():
            try:
                for key in temp_vars:
                    self.selected_features[key].set(temp_vars[key].get())
                
                self.labeling_method_var.set(temp_labeling_method.get())
                self.window_size.set(temp_window_size.get())
                self.lr_mode.set(temp_lr_mode.get())
                self.epochs_mode.set(temp_epochs_mode.get())
                self.num_layers_mode.set(temp_num_layers_mode.get())
                self.hidden_size_mode.set(temp_hidden_size_mode.get())

                if self.lr_mode.get() == "手動調整": self.lr.set(temp_lr.get())
                if self.epochs_mode.get() == "手動調整": self.epochs.set(temp_epochs.get())
                if self.num_layers_mode.get() == "手動調整": self.num_layers.set(temp_num_layers.get())
                if self.hidden_size_mode.get() == "手動調整": self.hidden_size.set(temp_hidden_size.get())

                self.dropout_var.set(temp_dropout_var.get())
                self.dropout_mode.set(temp_dropout_mode.get())
                if self.dropout_mode.get() == "手動調整": self.dropout_rate_var.set(temp_dropout_rate_var.get())

                self.validation_interval_mode.set(temp_validation_interval_mode.get())
                if self.validation_interval_mode.get() == "手動調整": self.validation_interval_var.set(temp_validation_interval_var.get())

                self.early_stopping_var.set(temp_early_stopping_var.get())
                self.patience_var.set(temp_patience_var.get())

                unbind_mousewheel()
                settings_win.destroy()
            except tk.TclError as e:
                messagebox.showerror("入力エラー", f"無効な数値が入力されました。\n{e}")
            except Exception as e:
                messagebox.showerror("エラー", f"設定の保存中にエラーが発生しました。\n{e}")

        def cancel_and_close():
            unbind_mousewheel()
            settings_win.destroy()

        ttk.Button(btn_frame, text="保存して戻る", command=save_and_close).pack(side='left', padx=10, expand=True)
        ttk.Button(btn_frame, text="キャンセル", command=cancel_and_close).pack(side='right', padx=10, expand=True)

    def select_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("CSVファイル", "*.csv")])
        if filepaths:
            self.csv_paths = filepaths
            filenames = [os.path.basename(p) for p in filepaths]
            self.file_label.config(text="選択中のファイル:\n" + "\n".join(filenames))

    def start_training(self):
        if not self.csv_paths:
            messagebox.showwarning("警告", "CSVファイルを選択してください。")
            return

        features = [key for key, var in self.selected_features.items() if var.get()]
        if not features:
            messagebox.showwarning("警告", "最低1つの生理データを選択してください。")
            return

        self.train_button.config(state="disabled")
        self.status_label.config(text="学習中... コンソールの進捗をご確認ください。")

        model = self.model_var.get()
        optimizer = self.optimizer_var.get()
        loss_func = self.loss_var.get()
        window_size = self.window_size.get()
        labeling_method = self.labeling_method_var.get()
        lr = self.lr.get()
        epochs = self.epochs.get()
        num_layers = self.num_layers.get()
        hidden_size = self.hidden_size.get()
        use_dropout = self.dropout_var.get()
        dropout_rate = self.dropout_rate_var.get()
        validation_interval = self.validation_interval_var.get()

        hyper_params_modes = {
            "lr_mode": self.lr_mode.get(),
            "epochs_mode": self.epochs_mode.get(),
            "num_layers_mode": self.num_layers_mode.get(),
            "hidden_size_mode": self.hidden_size_mode.get(),
            "dropout_mode": self.dropout_mode.get(),
            "validation_interval_mode": self.validation_interval_mode.get()
        }

        threading.Thread(target=self.run_training, args=(model, optimizer, loss_func, self.csv_paths, features, window_size, labeling_method, lr, epochs, num_layers, hidden_size, use_dropout, dropout_rate, validation_interval, hyper_params_modes, self.early_stopping_var.get(), self.patience_var.get()), daemon=True).start()

    def training_finished(self):
        self.train_button.config(state="normal")
        self.status_label.config(text="")

    def run_training(self, model, optimizer, loss_func, csv_paths, selected_features, window_size, labeling_method, lr, epochs, num_layers, hidden_size, use_dropout, dropout_rate, validation_interval, hyper_params_modes, use_early_stopping, patience):
        try:
            trained_model, scaler_X, scaler_y, actual_hidden_size = train_model(
                csv_paths,
                model_type=model.lower(),
                optimizer_type=optimizer,
                loss_type=loss_func,
                selected_features=selected_features,
                window_size=window_size,
                labeling_method=labeling_method,
                lr=lr,
                epochs=epochs,
                num_layers=num_layers,
                hidden_size=hidden_size,
                use_dropout=use_dropout,
                dropout_rate=dropout_rate,
                hyper_params_modes=hyper_params_modes,
                use_early_stopping=use_early_stopping,
                patience=patience
            )

            if self.save_model_var.get():
                os.makedirs("models", exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M")
                base_filename = f"{model}_{timestamp}"

                model_path = os.path.join("models", f"{base_filename}.pt")
                torch.save(trained_model.state_dict(), model_path)

                config = {
                    "model_type": model,
                    "optimizer": optimizer,
                    "loss_function": loss_func,
                    "selected_features": selected_features,
                    "window_size": window_size,
                    "labeling_method": labeling_method,
                    "lr": lr,
                    "epochs": epochs,
                    "num_layers": num_layers,
                    "hidden_size": actual_hidden_size,
                    "timestamp": timestamp,
                    "scaler_X_mean": scaler_X.mean_.tolist(),
                    "scaler_X_scale": scaler_X.scale_.tolist(),
                    "scaler_y_mean": scaler_y.mean_.tolist(),
                    "scaler_y_scale": scaler_y.scale_.tolist()
                }
                json_path = os.path.join("models", f"{base_filename}.json")
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(config, f, ensure_ascii=False, indent=4)

            messagebox.showinfo("完了", "学習が完了しました。")
        except Exception as e:
            messagebox.showerror("エラー", f"学習中にエラーが発生しました:\n{e}")
        finally:
            self.root.after(0, self.training_finished)

    def back_to_menu(self):
        self.root.destroy()
        from main import main_menu
        main_menu()

def launch_train_gui():
    root = tk.Tk()
    app = TrainApp(root)
    root.mainloop()
    