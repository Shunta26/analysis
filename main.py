#main.py (簡易GUI起動メニュー)

import tkinter as tk
from app.gui_predict import launch_predict_gui
from app.gui_train import launch_train_gui

def main_menu():
    root = tk.Tk()
    root.title("ドライバー覚醒度アプリ - 起動選択")
    root.geometry("300x150")

    tk.Label(root, text="モードを選択してください").pack(pady=10)

    tk.Button(root, text="推論モード", command=lambda: [root.destroy(), launch_predict_gui()]).pack(pady=10)
    tk.Button(root, text="学習モード", command=lambda: [root.destroy(), launch_train_gui()]).pack(pady=10)
    root.mainloop()

if __name__ == "__main__":
    main_menu()
