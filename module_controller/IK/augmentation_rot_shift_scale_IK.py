# -*- coding: utf-8 -*-
"""
augment_random_affine_babbling.py

- motor babbling で取得した画像と signals.csv を入力として、
  ランダムな「回転＋平行移動＋拡大縮小」オーグメンテーションを行う。
- 出力画像の解像度は元画像と同じ (W, H)。
- 元画像の番号 (例: 57.png) を保持したまま、
  57_aug_0000.png〜57_aug_0049.png のように保存する。
- 対応する電圧ラベル (Left_V, Right_V) は、元画像の行に紐づけたまま
  signals_aug.csv として出力する。

※ 前提:
- IN_DIR 直下に 1.png, 2.png, ... がある
- IN_CSV は元の motor babbling の signals.csv
  先頭行がヘッダ "Left_V,Right_V"、2行目以降がデータ
"""

import os
import csv
import cv2
import numpy as np

# ===== パス設定 =====
IN_DIR   = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon\roi"
IN_CSV   = os.path.join(IN_DIR, "signals.csv")

OUT_DIR  = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon\aug_rot_shift_scale"
OUT_CSV  = os.path.join(OUT_DIR, "signals_aug.csv")

VALID_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 1枚あたり生成枚数
NUM_SAMPLES = 20

# ===== 背景色の統計（BGR） =====
# ★ あなたの babbling 画像で算出し直してから更新してください
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)   # 必ず更新
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)   # 必ず更新
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

    # 距離変換で境界だけなめらかにブレンド (dist=0付近だけノイズ側に寄せる)
    dist = cv2.distanceTransform(warped_mask, cv2.DIST_L2, 3)
    alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
    alpha_3 = alpha[..., None]

    blended = warped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1.0 - alpha_3)
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_signals(in_csv):
    """
    signals.csv を読み込み、インデックス → (Left_V, Right_V) の辞書を作る。

    前提:
    - 1行目: ヘッダ
    - 2行目以降: Left_V, Right_V
    - i行目 (i>=2) が i-1.png に対応するとみなす
    """
    mapping = {}  # index(int) -> (left, right)

    with open(in_csv, 'r', newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) <= 1:
        raise RuntimeError("signals.csv にデータ行がありません。")

    # 2行目以降を読み込む
    for line_idx, row in enumerate(rows[1:], start=2):
        if len(row) < 2:
            continue
        try:
            left = float(row[0])
            right = float(row[1])
        except ValueError:
            continue
        # 行番号 line_idx に対して画像インデックスは (line_idx - 1) とする
        img_index = line_idx - 1
        mapping[img_index] = (left, right)

    return mapping


def main():
    # アフィン変換パラメータ
    MAX_ROT   = 90   # 回転角度ランダム範囲 [-MAX_ROT, +MAX_ROT]
    MAX_SHIFT = 5    # 平行移動の最大 px 範囲（上下左右）

    # 拡大・縮小の範囲（例: 0.8〜1.2倍）
    MIN_SCALE = 0.8
    MAX_SCALE = 1.2

    os.makedirs(OUT_DIR, exist_ok=True)

    # 元 signals.csv を読み込み： index -> (Left_V, Right_V)
    index_to_signal = load_signals(IN_CSV)

    # 出力CSV初期化
    with open(OUT_CSV, 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        # filename: 生成画像名, orig_index: 元画像インデックス, Left_V, Right_V
        writer.writerow(["filename", "orig_index", "Left_V", "Right_V"])

    files = sorted(os.listdir(IN_DIR))

    total_in = 0
    total_out = 0

    for name in files:
        in_path = os.path.join(IN_DIR, name)
        ext = os.path.splitext(name)[1].lower()
        if ext not in VALID_EXT:
            continue

        base = os.path.splitext(name)[0]  # "57" など
        # 元画像のインデックスを整数として取得できるかチェック
        try:
            idx = int(base)
        except ValueError:
            # 名前が数値でない場合は飛ばす
            print(f"[WARN] 数値ベース名ではないためスキップ: {name}")
            continue

        if idx not in index_to_signal:
            print(f"[WARN] signals.csv に {idx} 行目のデータが見つからないためスキップ: {name}")
            continue

        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] 読み込み失敗: {in_path}")
            continue

        total_in += 1
        left_v, right_v = index_to_signal[idx]

        H, W = img.shape[:2]
        center = (W / 2.0, H / 2.0)

        print(f"[INFO] {name}: ランダム生成 {NUM_SAMPLES} 枚 (Left={left_v:.1f}, Right={right_v:.1f})")

        # 出力CSVを追記モードで開く
        with open(OUT_CSV, 'a', newline='') as f_out:
            writer = csv.writer(f_out)

            for i in range(NUM_SAMPLES):
                # ---- ランダム変換パラメータ ----
                angle = np.random.uniform(-MAX_ROT, MAX_ROT)
                dx    = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1)
                dy    = np.random.randint(-MAX_SHIFT, MAX_SHIFT + 1)
                scale = np.random.uniform(MIN_SCALE, MAX_SCALE)

                # 回転＋スケーリング行列
                M_rot = cv2.getRotationMatrix2D(center, angle, scale)
                M_rot = np.vstack([M_rot, [0, 0, 1]]).astype(np.float32)

                # 平行移動行列
                M_shift = np.float32([[1, 0, dx],
                                      [0, 1, dy]])
                M_shift = np.vstack([M_shift, [0, 0, 1]]).astype(np.float32)

                # 合成アフィン（平行移動 → 回転・拡大）
                M = (M_shift @ M_rot)[0:2, :]

                # 画像変換
                aug = warp_with_random_border_and_blend(img, M)

                # ---- ファイル名 ----
                out_name = f"{base}_aug_{i:04d}.png"
                out_path = os.path.join(OUT_DIR, out_name)
                cv2.imwrite(out_path, aug)
                total_out += 1

                # 対応するラベルも出力
                writer.writerow([out_name, idx, left_v, right_v])

        print(f"[OK] {name} -> {NUM_SAMPLES} 枚生成")

    print(f"\n[DONE] 入力画像: {total_in} 枚, 生成画像: {total_out} 枚")
    print(f"[INFO] 出力CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
