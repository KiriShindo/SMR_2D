# -*- coding: utf-8 -*-
"""
batch_roi_crop.py

- 事前に保存してある roi_config.json (x, y, w, h) を読み込み
- 指定フォルダ内の全画像を、そのROIで切り取り
- 元画像はそのまま、ROI画像を別フォルダに保存
"""

import os
import json
import cv2

# ===== パス設定 =====
ROI_JSON_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\roi_config_full.json"

IN_DIR  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data_new\2module_dataset_max_DAC"
OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data_new\2module_dataset_max_DAC\roi"

# 対象とする拡張子
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# ===== ROIユーティリティ =====
def load_roi_xywh(json_path: str):
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"ROI JSON not found: {json_path}")
    with open(json_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    x = int(d.get("x", 0))
    y = int(d.get("y", 0))
    w = int(d.get("w", 0))
    h = int(d.get("h", 0))
    if w <= 0 or h <= 0:
        raise ValueError(f"Invalid ROI size in JSON: w={w}, h={h}")
    return x, y, w, h

def safe_crop_xywh(img, x, y, w, h):
    H, W = img.shape[:2]
    xx = max(0, min(x, W))
    yy = max(0, min(y, H))
    ww = max(0, min(w, W - xx))
    hh = max(0, min(h, H - yy))
    if ww <= 0 or hh <= 0:
        # ROIが完全に範囲外なら元画像を返す（保険）
        return img
    return img[yy:yy+hh, xx:xx+ww]

def main():
    # ROI読み込み
    try:
        x, y, w, h = load_roi_xywh(ROI_JSON_PATH)
        print(f"[ROI] x={x}, y={y}, w={w}, h={h}")
    except Exception as e:
        print(f"[ERROR] ROI読み込みに失敗しました: {e}")
        return

    if not os.path.isdir(IN_DIR):
        print(f"[ERROR] 入力フォルダが存在しません: {IN_DIR}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] 出力フォルダ: {OUT_DIR}")

    files = sorted(os.listdir(IN_DIR))
    total = 0
    saved = 0

    for name in files:
        in_path = os.path.join(IN_DIR, name)
        if not os.path.isfile(in_path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in VALID_EXT:
            continue

        total += 1

        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] 読み込み失敗: {in_path}")
            continue

        # ROIでトリミング（解像度はそのまま）
        roi_img = safe_crop_xywh(img, x, y, w, h)

        out_path = os.path.join(OUT_DIR, name)
        ok = cv2.imwrite(out_path, roi_img)
        if not ok:
            print(f"[WARN] 書き込み失敗: {out_path}")
            continue

        saved += 1
        print(f"[OK] {name} -> {out_path}")

    print(f"\n[DONE] 対象画像数: {total}, 保存: {saved}")

if __name__ == "__main__":
    main()
