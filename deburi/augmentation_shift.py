# -*- coding: utf-8 -*-
"""
augment_translate_xy_roi.py

- ROI後の画像を入力として、平行移動オーグメンテーションを行う
- 出力画像の解像度は元画像と同じ (W, H)
- シフト量は X, Y それぞれ [-MAX_SHIFT, ..., 0, ..., +MAX_SHIFT] ピクセル
- (0, 0) も含む（元ROI画像と同じ内容のデータも作る）
- 画像外からはみ出した領域は、背景の BGR 平均・標準偏差に従って
  ピクセルごとにランダムサンプリングした色で補完しつつ、
  元画像との境界は distance transform によってなめらかにブレンドする
- 出力ファイル名は元画像名 + "_shift_xy_..." にする
"""

import os
import cv2
import numpy as np

# ===== パス設定 =====
IN_DIR  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\normal\roi"
OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\normal\roi_shiftxy"

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


# ===== シフト量リスト生成（X,Y両方, 0含む） =====
def generate_shift_pairs(step_px: int, max_px: int):
    """
    例:
        step_px = 5, max_px = 10 のとき
        vals = [-10, -5, 0, 5, 10]
        → (dx, dy) の全組み合わせ [-10, -5, 0, 5, 10]^2

        (= 水平・垂直・斜めすべて)
    """
    if step_px <= 0:
        raise ValueError("step_px must be positive.")

    vals = list(range(-max_px, max_px + 1, step_px))
    pairs = []
    for dx in vals:
        for dy in vals:
            pairs.append((dx, dy))
    return pairs


# ===== 平行移動 + ランダム補完 + 境界ブレンド =====
def translate_xy_with_random_border_and_blend(img, shift_x_px: int, shift_y_px: int):
    """
    画像を (shift_x_px, shift_y_px) だけ平行移動させる。
    shift_x_px > 0 → 右
    shift_x_px < 0 → 左
    shift_y_px > 0 → 下
    shift_y_px < 0 → 上

    - 出力サイズは元と同じ (W, H)
    - 画像外からはみ出した領域は背景ノイズで埋める
    - 元画像との境界は distance transform によって BLEND_WIDTH ピクセル程度でブレンド
    """
    H, W = img.shape[:2]

    # 平行移動行列
    # [1, 0, tx]
    # [0, 1, ty]
    M = np.float32([[1, 0, shift_x_px],
                    [0, 1, shift_y_px]])

    # 1) 元画像をそのまま平行移動（境界色は仮値なので何でもOK）
    trans_orig = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # 2) 「元画像の有効領域マスク」を作る
    src_mask = np.ones((H, W), dtype=np.uint8) * 255  # 全255 (前景)
    warped_mask = cv2.warpAffine(
        src_mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0   # 外側 = 0
    )
    # warped_mask > 0 → 元画像由来の領域
    # ここでは fg_mask は使わず、距離だけを見る
    # fg_mask = (warped_mask > 0)

    # 3) 背景ノイズ画像
    noise = sample_border_noise(H, W)   # (H, W, 3)

    # 4) distance transform で「元画像領域からの距離」を求める
    dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)  # float32

    # 5) 距離に応じたブレンド係数 alpha を計算
    if BLEND_WIDTH <= 0:
        alpha = (dist > 0).astype(np.float32)  # 内側=1, 外側=0 (急峻)
    else:
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)

    alpha_3 = alpha[..., None]  # (H, W, 1)

    # 6) ブレンド： alpha * trans_orig + (1 - alpha) * noise
    trans_f = trans_orig.astype(np.float32)
    noise_f = noise.astype(np.float32)

    blended = trans_f * alpha_3 + noise_f * (1.0 - alpha_3)
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    return blended


# ===== ファイル名用 suffix =====
def shift_to_suffix(dx: int, dy: int) -> str:
    """
    シフト量（ピクセル）からファイル名用サフィックスを作る。

    例:
        dx = -20, dy =  15 → "mx020_py015"
        dx =   0, dy =   0 → "px000_py000"
    """
    sign_x = "p" if dx >= 0 else "m"
    sign_y = "p" if dy >= 0 else "m"
    val_x  = abs(int(round(dx)))
    val_y  = abs(int(round(dy)))
    return f"{sign_x}x{val_x:03d}_{sign_y}y{val_y:03d}"


# ===== メイン処理 =====
def main():

    # ---- ここだけ書き換えればシフト量を変更できる ----
    STEP_PX = 5    # 何ピクセル刻みでシフトするか
    MAX_PX  = 20   # X, Yともに最大シフト量（上下左右）
    SHIFT_PAIRS = generate_shift_pairs(STEP_PX, MAX_PX)
    # -----------------------------------------------------

    print(f"[INFO] シフトペア (dx, dy): {SHIFT_PAIRS}")
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

        for dx, dy in SHIFT_PAIRS:
            translated = translate_xy_with_random_border_and_blend(img, dx, dy)

            suf = shift_to_suffix(dx, dy)
            out_name = f"{base}_shift_xy_{suf}{ext}"
            out_path = os.path.join(OUT_DIR, out_name)

            ok = cv2.imwrite(out_path, translated)
            if not ok:
                print(f"[WARN] 書き込み失敗: {out_path}")
                continue

            total_out += 1
            print(f"[OK] {name} shift=(dx={dx:>3}, dy={dy:>3}) -> {out_name}")

    print(f"\n[DONE] 入力画像数: {total_in}, 生成枚数: {total_out}")


if __name__ == "__main__":
    main()
