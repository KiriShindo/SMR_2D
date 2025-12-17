# -*- coding: utf-8 -*-
"""
roi_image_preview.py
- JSONのROI(x,y,w,h)で静止画像を切り取り表示
- 位置・大きさをトラックバーで調整可能
- キー:
    q / Esc : 終了
    r       : ROI再読込（JSONから）
    s       : ROI保存（JSONへ）
- 画像はエクスプローラから選択
"""

import os
import json
import cv2
import time
import numpy as np
from datetime import datetime

# 画像選択ダイアログ用
import tkinter as tk
from tkinter import filedialog

# ===== ユーザ設定 =====
ROI_JSON_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\roi_config.json"
RESIZE_W, RESIZE_H = 1920, 1080

# ===== ROIユーティリティ =====
def load_roi_xywh(json_path: str, img_w: int, img_h: int):
    """
    JSONからROIを読み込む。
    ROIが存在しない/異常値の場合は、画像全体をROIとして返す。
    """
    if not os.path.isfile(json_path):
        print(f"[WARN] ROI JSON not found: {json_path}, use full image.")
        return 0, 0, img_w, img_h

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            d = json.load(f)
        x = int(d.get("x", 0))
        y = int(d.get("y", 0))
        w = int(d.get("w", img_w))
        h = int(d.get("h", img_h))
    except Exception as e:
        print(f"[WARN] ROI JSON load error: {e}, use full image.")
        return 0, 0, img_w, img_h

    # 一応クリップ
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h

def save_roi_xywh(json_path: str, x: int, y: int, w: int, h: int):
    """
    現在のROIをJSONに保存
    """
    d = {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] ROI saved to: {json_path}  (x={x}, y={y}, w={w}, h={h})")

def safe_crop_xywh(img, x, y, w, h):
    H, W = img.shape[:2]
    xx, yy = max(0, min(x, W)), max(0, min(y, H))
    ww, hh = max(0, min(w, W - xx)), max(0, min(h, H - yy))
    if ww <= 0 or hh <= 0:
        # ROIが不正なら全体返す（見失わないための保険）
        return img
    return img[yy:yy+hh, xx:xx+ww]

# 画面にパラメータを出す
def draw_info(img, text, org=(10, 22)):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)

def select_image_via_dialog():
    """
    エクスプローラから画像ファイルを選択
    """
    root = tk.Tk()
    root.withdraw()
    root.update()
    filetypes = [
        ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
        ("All files", "*.*")
    ]
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=filetypes
    )
    root.destroy()
    return path

def main():
    # 1. 画像ファイル選択
    img_path = select_image_via_dialog()
    if not img_path:
        print("[INFO] No image selected. Exit.")
        return

    img = cv2.imread(img_path)
    if img is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        return

    H, W = img.shape[:2]
    print(f"[INFO] Loaded image: {img_path}  (W={W}, H={H})")

    # 2. ROI読み込み（なければ全体）
    x, y, w, h = load_roi_xywh(ROI_JSON_PATH, W, H)
    print(f"[ROI] x={x}, y={y}, w={w}, h={h}")

    cv2.namedWindow("Full frame (ROI box)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("ROI view", cv2.WINDOW_NORMAL)

    # 3. トラックバーを作成（位置＆サイズ調整）
    # 注意: createTrackbar の max 値は inclusive なので W-1, H-1 にする
    def _dummy(v):
        pass

    cv2.createTrackbar("x", "Full frame (ROI box)", x, max(0, W - 1), _dummy)
    cv2.createTrackbar("y", "Full frame (ROI box)", y, max(0, H - 1), _dummy)
    cv2.createTrackbar("w", "Full frame (ROI box)", w, W, _dummy)   # 幅
    cv2.createTrackbar("h", "Full frame (ROI box)", h, H, _dummy)   # 高さ

    print("操作: q/Esc=終了, r=ROI再読込(JSON), s=ROI保存(JSON)")

    while True:
        # トラックバーから最新値を取得
        x = cv2.getTrackbarPos("x", "Full frame (ROI box)")
        y = cv2.getTrackbarPos("y", "Full frame (ROI box)")
        w = cv2.getTrackbarPos("w", "Full frame (ROI box)")
        h = cv2.getTrackbarPos("h", "Full frame (ROI box)")

        # 値を少し安全側にクリップ
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        # w,h は少なくとも1ピクセル、かつ画像範囲内
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        # フルフレームにROI枠を重ねる
        full_view = img.copy()
        x1 = x
        y1 = y
        x2 = x + w
        y2 = y + h
        cv2.rectangle(full_view, (x1, y1), (x2, y2), (0, 0, 255), 2)

        draw_info(full_view, f"ROI: x={x} y={y} w={w} h={h}")
        cv2.imshow("Full frame (ROI box)", full_view)

        # ROIを切り取って表示
        roi = safe_crop_xywh(img, x, y, w, h)
        roi_view = cv2.resize(roi, (RESIZE_W, RESIZE_H), interpolation=cv2.INTER_LINEAR)
        draw_info(roi_view, f"ROI view  {RESIZE_W}x{RESIZE_H}")
        cv2.imshow("ROI view", roi_view)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('r'):
            # ROI再読込（JSONが編集された場合など）
            x2, y2, w2, h2 = load_roi_xywh(ROI_JSON_PATH, W, H)
            print(f"[ROI] reloaded: x={x2}, y={y2}, w={w2}, h={h2}")
            # トラックバーに反映
            cv2.setTrackbarPos("x", "Full frame (ROI box)", x2)
            cv2.setTrackbarPos("y", "Full frame (ROI box)", y2)
            cv2.setTrackbarPos("w", "Full frame (ROI box)", w2)
            cv2.setTrackbarPos("h", "Full frame (ROI box)", h2)
        elif key == ord('s'):
            # 現在のROIをJSONへ保存
            save_roi_xywh(ROI_JSON_PATH, x, y, w, h)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
