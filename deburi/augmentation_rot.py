# # -*- coding: utf-8 -*-
# """
# augment_rotate_roi.py

# - ROI後の画像を入力として、回転オーグメンテーションを行う
# - 回転後の解像度は元画像と同じ (W, H)
# - 回転角度範囲 [-90°, 0°, 90°] を STEP 度刻みに従って自動生成
# - 0° も含む（元ROI画像と同じ内容のデータも作る）
# - 元画像の外側に相当する領域は、クリックで計測した
#   背景色の「平均・標準偏差」に基づいてランダムサンプリングした色で補完する
# - 出力ファイル名は元画像名 + "_rot_±xxx"
# """

# import os
# import cv2
# import numpy as np

# # ===== パス設定 =====
# IN_DIR  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\no_silicon\roi"
# OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\no_silicon\roi_rot"

# VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# # ===== 背景色の統計（BGR） =====
# # click_rgb_picker_with_stats.py で表示された値をここに貼る
# # 例：
# #   mean BGR = (133.12, 136.45, 136.02)
# #   std  BGR = (  4.20,   5.10,   4.80)
# # なら：
# #   MEAN_BGR = np.array([133.12, 136.45, 136.02], dtype=np.float32)
# #   STD_BGR  = np.array([  4.20,   5.10,   4.80], dtype=np.float32)


# # ===== 統計 (クリックした画素の BGR) =====
# # mean BGR = (128.04, 131.38, 132.08)
# # std  BGR = (6.56, 6.47, 5.51)
# MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)   # ← 実測値で上書きして
# STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)   # ← 実測 std で上書きして


# def sample_border_color():
#     """
#     背景色の平均・標準偏差から BGR を1色サンプリングする。
#     ガウス分布 N(mean, std^2) からサンプリングし、[0,255] にクリップして uint8 にする。
#     """
#     c = np.random.normal(MEAN_BGR, STD_BGR)
#     c = np.clip(c, 0, 255).astype(np.uint8)
#     return (int(c[0]), int(c[1]), int(c[2]))   # OpenCVはBGR順


# # ===== 角度リスト生成（0°含む） =====
# def generate_angle_list(step_deg, max_deg=90):
#     """
#     例:
#         step_deg = 15 → [-90, -75, ..., -15, 0, 15, ..., 75, 90]
#         step_deg = 30 → [-90, -60, -30, 0, 30, 60, 90]
#     """
#     if step_deg <= 0:
#         raise ValueError("STEP must be positive.")

#     neg = list(range(-max_deg, 0, step_deg))
#     pos = list(range(0, max_deg + 1, step_deg))  # 0, step, 2*step, ...
#     return neg + pos


# # ===== 回転処理（サイズ固定・ランダム色埋め） =====
# def rotate_keep_size(img, angle_deg):
#     """
#     画像を中心で回転させ、
#     元の画像サイズ (W, H) のまま返す。
#     元画像に存在しない領域は sample_border_color() でサンプリングした色で塗りつぶす。
#     """
#     H, W = img.shape[:2]
#     center = (W / 2.0, H / 2.0)

#     M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

#     border_color = sample_border_color()

#     rotated = cv2.warpAffine(
#         img, M, (W, H),
#         flags=cv2.INTER_LINEAR,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=border_color
#     )
#     return rotated


# # ===== ファイル名用 suffix =====
# def angle_to_suffix(angle_deg):
#     """
#     角度からファイル名用サフィックスを作る。
#     例:
#         -90 → "m090"
#          15 → "p015"
#           0 → "p000"
#     """
#     sign = "p" if angle_deg >= 0 else "m"
#     val  = abs(int(round(angle_deg)))
#     return f"{sign}{val:03d}"


# # ===== メイン処理 =====
# def main():

#     # ---- ここだけ書き換えれば角度刻みを変更できる ----
#     STEP = 5   # 5, 10, 15, 30 など好きな刻みに変更可
#     ANGLES = generate_angle_list(STEP)
#     # -----------------------------------------------------

#     print(f"[INFO] 角度リスト: {ANGLES}")
#     print(f"[INFO] MEAN_BGR: {MEAN_BGR}, STD_BGR: {STD_BGR}")

#     if not os.path.isdir(IN_DIR):
#         print(f"[ERROR] 入力フォルダが存在しません: {IN_DIR}")
#         return

#     os.makedirs(OUT_DIR, exist_ok=True)
#     print(f"[INFO] 入力フォルダ: {IN_DIR}")
#     print(f"[INFO] 出力フォルダ: {OUT_DIR}")

#     files = sorted(os.listdir(IN_DIR))
#     total_in = 0
#     total_out = 0

#     for name in files:
#         in_path = os.path.join(IN_DIR, name)
#         if not os.path.isfile(in_path):
#             continue

#         ext = os.path.splitext(name)[1].lower()
#         if ext not in VALID_EXT:
#             continue

#         total_in += 1

#         img = cv2.imread(in_path)
#         if img is None:
#             print(f"[WARN] 読み込み失敗: {in_path}")
#             continue

#         base, ext = os.path.splitext(name)

#         for angle in ANGLES:
#             rotated = rotate_keep_size(img, angle)

#             suf = angle_to_suffix(angle)
#             out_name = f"{base}_rot_{suf}{ext}"
#             out_path = os.path.join(OUT_DIR, out_name)

#             ok = cv2.imwrite(out_path, rotated)
#             if not ok:
#                 print(f"[WARN] 書き込み失敗: {out_path}")
#                 continue

#             total_out += 1
#             print(f"[OK] {name} angle={angle:>4}° -> {out_name}")

#     print(f"\n[DONE] 入力画像数: {total_in}, 生成枚数: {total_out}")


# if __name__ == "__main__":
#     main()









# -*- coding: utf-8 -*-
"""
augment_rotate_roi.py

- ROI後の画像を入力として、回転オーグメンテーションを行う
- 回転後の解像度は元画像と同じ (W, H)
- 回転角度範囲 [-90°, 0°, 90°] を STEP 度刻みに従って自動生成
- 0° も含む（元ROI画像と同じ内容のデータも作る）
- 元画像の外側に相当する領域は、背景の BGR 平均・標準偏差に従って
  ピクセルごとにランダムサンプリングした色で補完しつつ、
  元画像との境界は distance transform を用いてなめらかにブレンドする
- 出力ファイル名は元画像名 + "_rot_±xxx"
"""

import os
import cv2
import numpy as np

# ===== パス設定 =====
IN_DIR  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\normal\roi"
OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\normal\roi_rot_random"

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


# ===== 背景色の統計（BGR） =====
# click_rgb_picker_with_stats.py で計測した値をここに貼り付ける
# ===== 統計 (クリックした画素の BGR) =====
# mean BGR = (128.04, 131.38, 132.08)
# std  BGR = (6.56, 6.47, 5.51)
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)   # ★ここを実測値で更新
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)   # ★ここも実測 std で更新

# 境界をどれくらいの幅でブレンドするか（ピクセル）
BLEND_WIDTH = 5.0  # 3〜10くらいで調整推奨


def sample_border_noise(height: int, width: int) -> np.ndarray:
    """
    (H, W, 3) 形状のランダムBGR画像を生成する。
    各ピクセルは N(MEAN_BGR, STD_BGR^2) に従う。
    """
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(height, width, 3))
    noise = np.clip(noise, 0, 255).astype(np.uint8)
    return noise


# ===== 角度リスト生成（0°含む） =====
def generate_angle_list(step_deg, max_deg=90):
    """
    例:
        step_deg = 15 → [-90, -75, ..., -15, 0, 15, ..., 75, 90]
        step_deg = 30 → [-90, -60, -30, 0, 30, 60, 90]
    """
    if step_deg <= 0:
        raise ValueError("STEP must be positive.")

    neg = list(range(-max_deg, 0, step_deg))
    pos = list(range(0, max_deg + 1, step_deg))  # 0, step, 2*step, ...
    return neg + pos


# ===== 回転処理（サイズ固定・境界ブレンド付きランダム補完） =====
def rotate_keep_size_with_random_border_and_blend(img, angle_deg):
    """
    画像を中心で回転させ、
    元の画像サイズ (W, H) のまま返す。

    - 元画像の有効領域は warpAffine した結果 (rot_orig)
    - 画像外には背景ノイズ (noise) を配置
    - 有効領域との境界は distance transform によって
      BLEND_WIDTH ピクセル程度の幅でなめらかにブレンドする。
    """
    H, W = img.shape[:2]
    center = (W / 2.0, H / 2.0)

    # 回転行列
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    # 1) 元画像をそのまま回転（境界色は仮値なので何でもOK）
    rot_orig = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 2) 「元画像の有効領域マスク」を作る
    #    元画像を全255の1ch画像とみなし、それを同じ変換でwarpする
    src_mask = np.ones((H, W), dtype=np.uint8) * 255  # 全255 (前景)
    warped_mask = cv2.warpAffine(
        src_mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0   # 外側 = 0
    )
    # warped_mask > 0 → 元画像由来の領域
    fg_mask = (warped_mask > 0)  # bool (H, W)

    # 3) 背景ノイズ画像
    noise = sample_border_noise(H, W)   # (H, W, 3)

    # 4) distance transform で「元画像領域からの距離」を求める
    #    warped_mask は 0 or 255 なので、そのまま距離変換に使える
    dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)  # float32

    # 5) 距離に応じたブレンド係数 alpha を計算
    #    - dist >= BLEND_WIDTH → alpha = 1.0（完全に元画像）
    #    - dist = 0 → alpha = 0.0（完全にノイズ）
    #    - 0 < dist < BLEND_WIDTH → 線形に 0〜1
    if BLEND_WIDTH <= 0:
        alpha = (dist > 0).astype(np.float32)  # 内側=1, 外側=0 (急峻)
    else:
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)

    # チャンネル方向に拡張
    alpha_3 = alpha[..., None]  # (H, W, 1)

    # 6) ブレンド： alpha * rot_orig + (1 - alpha) * noise
    rot_f = rot_orig.astype(np.float32)
    noise_f = noise.astype(np.float32)

    blended = rot_f * alpha_3 + noise_f * (1.0 - alpha_3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# ===== ファイル名用 suffix =====
def angle_to_suffix(angle_deg):
    """
    角度からファイル名用サフィックスを作る。
    例:
        -90 → "m090"
         15 → "p015"
          0 → "p000"
    """
    sign = "p" if angle_deg >= 0 else "m"
    val  = abs(int(round(angle_deg)))
    return f"{sign}{val:03d}"


# ===== メイン処理 =====
def main():

    # ---- ここだけ書き換えれば角度刻みを変更できる ----
    STEP = 5   # 5, 10, 15, 30 など好きな刻みに変更可
    ANGLES = generate_angle_list(STEP)
    # -----------------------------------------------------

    print(f"[INFO] 角度リスト: {ANGLES}")
    print(f"[INFO] MEAN_BGR: {MEAN_BGR}, STD_BGR: {STD_BGR}, BLEND_WIDTH={BLEND_WIDTH}")

    if not os.path.isdir(IN_DIR):
        print(f"[ERROR] 入力フォルダが存在しません: {IN_DIR}")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"[INFO] 入力フォルダ: {IN_DIR}")
    print(f"[INFO] 出力フォルダ: {OUT_DIR}")

    files = sorted(os.listdir(IN_DIR))
    total_in = 0
    total_out = 0

    for name in files:
        in_path = os.path.join(IN_DIR, name)
        if not os.path.isfile(in_path):
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in VALID_EXT:
            continue

        total_in += 1

        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] 読み込み失敗: {in_path}")
            continue

        base, ext = os.path.splitext(name)

        for angle in ANGLES:
            rotated = rotate_keep_size_with_random_border_and_blend(img, angle)

            suf = angle_to_suffix(angle)
            out_name = f"{base}_rot_{suf}{ext}"
            out_path = os.path.join(OUT_DIR, out_name)

            ok = cv2.imwrite(out_path, rotated)
            if not ok:
                print(f"[WARN] 書き込み失敗: {out_path}")
                continue

            total_out += 1
            print(f"[OK] {name} angle={angle:>4}° -> {out_name}")

    print(f"\n[DONE] 入力画像数: {total_in}, 生成枚数: {total_out}")


if __name__ == "__main__":
    main()
