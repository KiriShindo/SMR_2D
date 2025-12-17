# ### 梁マスク
# import os
# import csv
# import shutil
# import numpy as np
# import pandas as pd

# # ===== 元のフォルダ =====
# SRC_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi_aug_rot_shift_scale_rect2mask"
# SRC_CSV = os.path.join(SRC_DIR, "mask_relation.csv")

# # ===== 出力フォルダ =====
# DST_DIR = SRC_DIR + "_filtered"
# DST_CSV = os.path.join(DST_DIR, "mask_relation_filtered.csv")
# os.makedirs(DST_DIR, exist_ok=True)

# # ===== 外れ値除去パラメータ =====
# Z_THRESH = 1.5  # Zスコアがこの値を超えたら外れ値扱い

# def main():
#     if not os.path.exists(SRC_CSV):
#         print(f"[ERROR] CSV not found: {SRC_CSV}")
#         return

#     # ===== CSV読み込み =====
#     df = pd.read_csv(SRC_CSV)
#     if "distance" not in df.columns or "angle_diff_deg" not in df.columns:
#         print("[ERROR] 必要な列 (distance, angle_diff_deg) がありません。")
#         return

#     # ===== 統計量計算 =====
#     dist_mean, dist_std = df["distance"].mean(), df["distance"].std(ddof=0)
#     ang_mean, ang_std = df["angle_diff_deg"].mean(), df["angle_diff_deg"].std(ddof=0)

#     print("=== 統計量 ===")
#     print(f"distance: mean={dist_mean:.3f}, std={dist_std:.3f}")
#     print(f"angle_diff_deg: mean={ang_mean:.3f}, std={ang_std:.3f}")

#     # ===== Zスコア計算 =====
#     df["z_dist"] = (df["distance"] - dist_mean) / (dist_std + 1e-6)
#     df["z_ang"]  = (df["angle_diff_deg"] - ang_mean) / (ang_std + 1e-6)

#     # ===== 外れ値除外 =====
#     mask_valid = (df["z_dist"].abs() < Z_THRESH) & (df["z_ang"].abs() < Z_THRESH)
#     df_valid = df[mask_valid].copy()
#     df_outlier = df[~mask_valid].copy()

#     print(f"\nValid samples: {len(df_valid)} / {len(df)} ({len(df_outlier)} removed)")

#     # ===== 対応するマスク画像をコピー =====
#     copied_count = 0
#     for fname in df_valid["filename"]:
#         stem, _ = os.path.splitext(fname)
#         src_img = os.path.join(SRC_DIR, f"{stem}_rect2mask.png")
#         dst_img = os.path.join(DST_DIR, f"{stem}_rect2mask.png")
#         if os.path.exists(src_img):
#             shutil.copy2(src_img, dst_img)
#             copied_count += 1

#     # ===== 新しいCSVを保存 =====
#     df_valid.drop(columns=["z_dist", "z_ang"], inplace=True)
#     df_valid.to_csv(DST_CSV, index=False)

#     print(f"\n✅ コピー完了: {copied_count} images")
#     print(f"出力フォルダ: {DST_DIR}")
#     print(f"新しいCSV: {DST_CSV}")

# if __name__ == "__main__":
#     main()



### 梁を画像の端まで伸ばした時の4点の交点の座標を保存
import os
import csv
import shutil
import numpy as np
import pandas as pd

# ===== 元のフォルダ（この中だけ触る） =====
SRC_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi_aug_rot_shift_scale_rect2mask_4point"
SRC_CSV = os.path.join(SRC_DIR, "mask_relation.csv")

# ===== 出力フォルダ（クレンジング後データ） =====
DST_DIR = SRC_DIR + "_filtered"
DST_CSV = os.path.join(DST_DIR, "mask_relation_filtered.csv")
os.makedirs(DST_DIR, exist_ok=True)

# ===== 外れ値除去パラメータ =====
# Zスコアがこの値を超えたら外れ値とみなす
Z_THRESH = 1.5  # 2.0〜3.0あたりで調整して

def main():
    if not os.path.exists(SRC_CSV):
        print(f"[ERROR] CSV not found: {SRC_CSV}")
        return

    # ===== CSV読み込み =====
    df = pd.read_csv(SRC_CSV)

    # 必須列チェック
    required_cols = ["filename", "distance", "angle_diff_deg"]
    for c in required_cols:
        if c not in df.columns:
            print(f"[ERROR] column '{c}' not found in CSV.")
            return

    # ===== 統計量計算 =====
    dist_mean = df["distance"].mean()
    dist_std  = df["distance"].std(ddof=0)
    ang_mean  = df["angle_diff_deg"].mean()
    ang_std   = df["angle_diff_deg"].std(ddof=0)

    print("=== 統計量 ===")
    print(f"distance     : mean={dist_mean:.3f}, std={dist_std:.3f}")
    print(f"angle_diff   : mean={ang_mean:.3f}, std={ang_std:.3f}")

    # ゼロ除算対策
    dist_std_safe = dist_std if dist_std > 1e-6 else 1e-6
    ang_std_safe  = ang_std  if ang_std  > 1e-6 else 1e-6

    # ===== Zスコア計算 =====
    df["z_dist"] = (df["distance"] - dist_mean) / dist_std_safe
    df["z_ang"]  = (df["angle_diff_deg"] - ang_mean) / ang_std_safe

    # ===== 外れ値判定 =====
    mask_valid = (df["z_dist"].abs() < Z_THRESH) & (df["z_ang"].abs() < Z_THRESH)
    df_valid   = df[mask_valid].copy()
    df_outlier = df[~mask_valid].copy()

    print(f"\nTotal samples : {len(df)}")
    print(f"Valid samples : {len(df_valid)}")
    print(f"Outliers      : {len(df_outlier)} (removed from filtered set)")

    # ===== 対応する画像をコピー =====
    copied_count = 0
    for fname in df_valid["filename"]:
        stem, _ = os.path.splitext(fname)
        src_img = os.path.join(SRC_DIR, f"{stem}_rect2mask.png")
        dst_img = os.path.join(DST_DIR, f"{stem}_rect2mask.png")
        if os.path.exists(src_img):
            shutil.copy2(src_img, dst_img)
            copied_count += 1
        else:
            # 画像が存在しない場合のログだけ出す
            print(f"[WARN] image not found for {fname}: {src_img}")

    # ===== フィルタ後CSVを書き出し =====
    df_valid = df_valid.drop(columns=["z_dist", "z_ang"])
    df_valid.to_csv(DST_CSV, index=False)

    print("\n===== Summary =====")
    print(f"  Copied images : {copied_count}")
    print(f"  Filtered CSV  : {DST_CSV}")
    print(f"  Output folder : {DST_DIR}")

if __name__ == "__main__":
    main()
