# -*- coding: utf-8 -*-
"""
eval_module_controller_predict_only.py
- FF_train.py で学習した ModuleControllerNet を読み込み
- エクスプローラで選んだ任意の画像1枚を入力
- モデルが推定した左右電圧（Pred）のみターミナルに出力
"""

import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

# === 学習スクリプトからクラスとパスをインポート ==========================
from FF_train import (
    ModuleControllerNet,
    MODEL_SAVE_PATH,
)

def predict_single_image():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- 画像選択 ----
    root = tk.Tk()
    root.withdraw()
    img_path_str = filedialog.askopenfilename(
        title="予測したい画像を選択してください",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg")]
    )
    root.destroy()

    if not img_path_str:
        print("画像が選択されませんでした。終了します。")
        return

    img_path = Path(img_path_str)
    print(f"\n選択された画像: {img_path}")

    # ---- モデル読み込み ----
    model = ModuleControllerNet(feat_dim=128).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- 入力変換（学習時と同じサイズ） ----
    transform = T.Compose([
        T.Resize((62, 50)),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ---- 画像読み込み＆推論 ----
    img = Image.open(img_path).convert("RGB")
    inp = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(inp).cpu().numpy()[0]

    # ---- 結果出力 ----
    print(f"\n--- 予測結果 ---")
    print(f"左電圧 (Pred): {pred[0]:.4f} V")
    print(f"右電圧 (Pred): {pred[1]:.4f} V")

if __name__ == "__main__":
    predict_single_image()
