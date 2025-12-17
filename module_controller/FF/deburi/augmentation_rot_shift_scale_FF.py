# -*- coding: utf-8 -*-
"""
augment_random_affine_roi.py
- ROI後の画像を入力として、ランダムな「回転＋平行移動＋拡大縮小」を50枚生成する
- 出力画像の解像度は元画像と同じ (W, H)
- 元画像のファイル名 (例: 57.png) を保持したまま、
  57_aug_0000.png〜57_aug_0049.png の形で保存する
- 境界はノイズ + distance transform ブレンドで自然に補間
"""
import os
import cv2
import numpy as np

# ===== パス設定 =====
IN_DIR  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\no_silicon\roi"
OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\no_silicon\roi_aug_rot_shift_scale"
VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 1枚あたり生成枚数
NUM_SAMPLES = 75

# ===== 背景色の統計（BGR） =====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)   # 必ず更新
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)   # 必ず更新
BLEND_WIDTH = 5.0  # 3〜10で調整可

def sample_border_noise(H, W):
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)

def warp_with_random_border_and_blend(img, M):
    H, W = img.shape[:2]

    # 変換後画像（外側は 0 埋め）
    warped = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,0,0)
    )

    # マスクも同じ変換
    src_mask = np.ones((H, W), dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(
        src_mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 外側をノイズで補完
    noise = sample_border_noise(H, W)

    # 距離変換で境界をなめらかにブレンド
    dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
    alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
    alpha_3 = alpha[..., None]

    blended = warped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
    return np.clip(blended, 0, 255).astype(np.uint8)

def main():
    MAX_ROT   = 90   # 回転角度ランダム範囲
    MAX_SHIFT = 5    # 平行移動の最大 px 範囲（上下左右）

    # ★ 追加: 拡大・縮小の範囲（例: 0.8〜1.2倍）
    MIN_SCALE = 0.8
    MAX_SCALE = 1.2

    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(os.listdir(IN_DIR))

    total_in = 0
    total_out = 0

    for name in files:
        in_path = os.path.join(IN_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in VALID_EXT:
            continue

        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] 読み込み失敗: {in_path}")
            continue

        total_in += 1
        base = os.path.splitext(name)[0]  # 例: "57"

        H, W = img.shape[:2]
        center = (W / 2.0, H / 2.0)

        print(f"[INFO] {name}: ランダム生成 {NUM_SAMPLES}枚")

        for i in range(NUM_SAMPLES):
            # ---- ランダム変換 ----
            angle = np.random.uniform(-MAX_ROT, MAX_ROT)
            dx    = np.random.randint(-MAX_SHIFT, MAX_SHIFT+1)
            dy    = np.random.randint(-MAX_SHIFT, MAX_SHIFT+1)

            # ★ ランダムスケールをサンプリング
            scale = np.random.uniform(MIN_SCALE, MAX_SCALE)

            # 回転＋スケーリング行列
            M_rot = cv2.getRotationMatrix2D(center, angle, scale)
            M_rot = np.vstack([M_rot, [0,0,1]]).astype(np.float32)

            # 平行移動行列
            M_shift = np.float32([[1, 0, dx],
                                  [0, 1, dy]])
            M_shift = np.vstack([M_shift, [0,0,1]]).astype(np.float32)

            # 合成アフィン（平行移動 → 回転・拡大）
            M = (M_shift @ M_rot)[0:2, :]

            aug = warp_with_random_border_and_blend(img, M)

            # ---- ファイル名 ----
            out_name = f"{base}_aug_{i:04d}.png"
            out_path = os.path.join(OUT_DIR, out_name)
            cv2.imwrite(out_path, aug)
            total_out += 1

        print(f"[OK] {name} -> {NUM_SAMPLES}枚生成")

    print(f"\n[DONE] 入力: {total_in}枚, 生成: {total_out}枚")

if __name__ == "__main__":
    main()
