# -*- coding: utf-8 -*-
"""
cleanse_beam2ch_masks.py

- detectron2_rect2mask_to_2ch.py の出力に対してクレンジングを行う
- mask_relation.csv の distance / angle_diff_deg に基づき外れ値除去
- 有効サンプルだけ *_beam2ch.npy / *_beammask.png をコピー
"""

import os
import shutil
import numpy as np
import pandas as pd

# ===== 元ディレクトリ =====
SRC_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi_aug_rot_shift_beammask_2ch"
SRC_CSV = os.path.join(SRC_DIR, "mask_relation.csv")

# ===== 出力ディレクトリ =====
DST_DIR = SRC_DIR + "_filtered"
DST_CSV = os.path.join(DST_DIR, "mask_relation_filtered.csv")

os.makedirs(DST_DIR, exist_ok=True)

# ===== 外れ値判定パラメータ =====
Z_THRESH = 1.5  # 必要に応じて 2.0〜3.0 くらいまで上げてもよい


def main():
    if not os.path.exists(SRC_CSV):
        print(f"[ERROR] CSV not found: {SRC_CSV}")
        return

    df = pd.read_csv(SRC_CSV)

    required_cols = ["filename", "distance", "angle_diff_deg"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[ERROR] column '{c}' not found in CSV.")
            return

    # 統計量
    dist_mean = df["distance"].mean()
    dist_std  = df["distance"].std(ddof=0)
    ang_mean  = df["angle_diff_deg"].mean()
    ang_std   = df["angle_diff_deg"].std(ddof=0)

    print("=== 統計量 ===")
    print(f"distance     : mean={dist_mean:.3f}, std={dist_std:.3f}")
    print(f"angle_diff   : mean={ang_mean:.3f}, std={ang_std:.3f}")

    dist_std_safe = dist_std if dist_std > 1e-6 else 1e-6
    ang_std_safe  = ang_std  if ang_std  > 1e-6 else 1e-6

    # Zスコア
    df["z_dist"] = (df["distance"]      - dist_mean) / dist_std_safe
    df["z_ang"]  = (df["angle_diff_deg"] - ang_mean) / ang_std_safe

    # 有効 / 外れ値
    mask_valid = (df["z_dist"].abs() < Z_THRESH) & (df["z_ang"].abs() < Z_THRESH)
    df_valid   = df[mask_valid].copy()
    df_outlier = df[~mask_valid].copy()

    print(f"\nTotal samples : {len(df)}")
    print(f"Valid samples : {len(df_valid)}")
    print(f"Outliers      : {len(df_outlier)}")

    copied_npy = 0
    copied_png = 0

    for fname in df_valid["filename"]:
        stem, ext = os.path.splitext(fname)   # "57.png" -> "57"

        src_npy = os.path.join(SRC_DIR, f"{stem}_beam2ch.npy")
        dst_npy = os.path.join(DST_DIR, f"{stem}_beam2ch.npy")

        src_png = os.path.join(SRC_DIR, f"{stem}_beammask.png")
        dst_png = os.path.join(DST_DIR, f"{stem}_beammask.png")

        if os.path.exists(src_npy):
            shutil.copy2(src_npy, dst_npy)
            copied_npy += 1
        else:
            print(f"[WARN] NPY not found for {fname}: {src_npy}")

        if os.path.exists(src_png):
            shutil.copy2(src_png, dst_png)
            copied_png += 1
        else:
            print(f"[WARN] PNG not found for {fname}: {src_png}")

    # フィルタ済みCSVを書き出し
    df_valid = df_valid.drop(columns=["z_dist", "z_ang"])
    df_valid.to_csv(DST_CSV, index=False)

    print("\n===== Summary =====")
    print(f"  Copied NPY  : {copied_npy}")
    print(f"  Copied PNG  : {copied_png}")
    print(f"  Filtered CSV: {DST_CSV}")
    print(f"  Output dir  : {DST_DIR}")


if __name__ == "__main__":
    main()
