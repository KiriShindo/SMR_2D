# -*- coding: utf-8 -*-
"""
augment_affine_pairs_121x121.py

- 事前に作成した signals_pairs_121x121.csv を読み込み、
  各 (target, current) ペアに対して「同じアフィン変換」をかけた
  画像ペアを生成する。
- 出力は:
    画像: OUT_DIR 配下に PNG で保存
    CSV : signals_pairs_121x121_aug.csv を OUT_DIR に保存

前提:
- ROI_DIR に 1.png ~ 121.png がある
- ROI_DIR に signals_pairs_121x121.csv がある
- signals_pairs_121x121.csv のカラム:
    target_idx, current_idx,
    target_img, current_img,
    target_left, target_right,
    current_left, current_right
"""

import os
import csv
import cv2
import numpy as np

# ===== パス設定 =====
ROI_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi"
PAIRS_CSV = os.path.join(ROI_DIR, "signals_pairs_121x121.csv")

OUT_DIR = os.path.join(ROI_DIR, "affine_pairs_121x121")
OUT_CSV = os.path.join(OUT_DIR, "signals_pairs_121x121_aug.csv")

# 1ペアあたり何セットオーグメンテーションするか
NUM_SAMPLES_PER_PAIR = 20  # 好きに増減させてOK

# ===== 背景色の統計（BGR） =====
# ★ 必要に応じて、FF用ROI画像から算出し直してもOK
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0  # 3〜10で調整可


def sample_border_noise(H, W):
    """背景を埋めるためのランダムノイズ画像を生成 (BGR 正規分布)"""
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


def warp_with_random_border_and_blend(img, M):
    """
    アフィン変換 + 外側のノイズ補完 + distance transform による境界ブレンド
    img: 入力画像 (H, W, 3)
    M:   2x3 アフィン変換行列
    """
    H, W = img.shape[:2]

    # 変換後画像（外側は 0 埋め）
    warped = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0)
    )

    # マスクも同じ変換
    src_mask = np.ones((H, W), dtype=np.uint8) * 255
    warped_mask = cv2.warpAffine(
        src_mask, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 外側はノイズで補完
    noise = sample_border_noise(H, W)

    # 距離変換で境界だけなめらかにブレンド
    dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
    alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
    alpha_3 = alpha[..., None]

    blended = warped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1.0 - alpha_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def main():
    # アフィン変換パラメータ
    MAX_ROT   = 60   # 回転角度ランダム範囲 [-MAX_ROT, +MAX_ROT]
    MAX_SHIFT = 5    # 平行移動の最大 px 範囲（上下左右）

    os.makedirs(OUT_DIR, exist_ok=True)

    # 入力ペアCSVを読む
    with open(PAIRS_CSV, "r", newline="") as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError(f"{PAIRS_CSV} にペアがありません。")

    # 出力CSVを初期化
    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "pair_id", "aug_id",
            "target_idx", "current_idx",
            "target_img_aug", "current_img_aug",
            "target_left", "target_right",
            "current_left", "current_right",
            "orig_target_img", "orig_current_img",
        ])

    total_pairs = len(rows)
    total_out_images = 0

    for pair_id, row in enumerate(rows):
        # 行から情報を取り出す
        try:
            t_idx = int(row["target_idx"])
            c_idx = int(row["current_idx"])
        except Exception:
            raise RuntimeError(f"CSVの target_idx/current_idx がパースできません: {row}")

        t_name = row["target_img"]
        c_name = row["current_img"]

        t_left  = float(row["target_left"])
        t_right = float(row["target_right"])
        c_left  = float(row["current_left"])
        c_right = float(row["current_right"])

        # 元画像を読み込み
        t_path = os.path.join(ROI_DIR, t_name)
        c_path = os.path.join(ROI_DIR, c_name)

        t_img = cv2.imread(t_path)
        c_img = cv2.imread(c_path)

        if t_img is None:
            print(f"[WARN] target画像読み込み失敗: {t_path}")
            continue
        if c_img is None:
            print(f"[WARN] current画像読み込み失敗: {c_path}")
            continue

        if t_img.shape != c_img.shape:
            print(f"[WARN] 画像サイズが異なるためスキップ: {t_path}, {c_path}")
            continue

        H, W = t_img.shape[:2]
        center = (W / 2.0, H / 2.0)

        # ここから NUM_SAMPLES_PER_PAIR セット生成
        with open(OUT_CSV, "a", newline="") as f_out:
            writer = csv.writer(f_out)

            for aug_id in range(NUM_SAMPLES_PER_PAIR):
                # ---- ランダム変換パラメータ（ペア共通）----
                angle = np.random.uniform(-MAX_ROT, MAX_ROT)
                dx    = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1)
                dy    = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1)

                # 回転行列（スケールは常に 1.0）
                M_rot = cv2.getRotationMatrix2D(center, angle, 1.0)
                M_rot = np.vstack([M_rot, [0, 0, 1]]).astype(np.float32)

                # 平行移動行列
                M_shift = np.float32([[1, 0, dx],
                                      [0, 1, dy],
                                      [0, 0, 1]])

                # 合成アフィン（平行移動 → 回転）
                M = (M_shift @ M_rot)[0:2, :]

                # 同じ変換を target / current 両方に適用
                t_aug = warp_with_random_border_and_blend(t_img, M)
                c_aug = warp_with_random_border_and_blend(c_img, M)

                # ---- ファイル名 ----
                # pair_id と aug_id を含めて一意に
                t_out_name = f"p{pair_id:05d}_t_{aug_id:03d}.png"
                c_out_name = f"p{pair_id:05d}_c_{aug_id:03d}.png"

                cv2.imwrite(os.path.join(OUT_DIR, t_out_name), t_aug)
                cv2.imwrite(os.path.join(OUT_DIR, c_out_name), c_aug)
                total_out_images += 2

                # 対応するラベルを書き出し
                writer.writerow([
                    pair_id, aug_id,
                    t_idx, c_idx,
                    t_out_name, c_out_name,
                    f"{t_left:.6f}", f"{t_right:.6f}",
                    f"{c_left:.6f}", f"{c_right:.6f}",
                    t_name, c_name,
                ])

        if (pair_id + 1) % 500 == 0:
            print(f"[INFO] {pair_id + 1}/{total_pairs} ペア処理完了")

    print(f"\n[DONE] ペア数: {total_pairs}, 生成画像枚数: {total_out_images}")
    print(f"[INFO] 出力DIR:  {OUT_DIR}")
    print(f"[INFO] 出力CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
