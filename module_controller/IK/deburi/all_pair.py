# -*- coding: utf-8 -*-
"""
expand_121x121_pairs.py

1モジュール静的データ(121通り)から、
「目標状態 × 現在状態」の 121x121 通りのペアを列挙した CSV を作る。

前提:
- IN_DIR に 1.png ~ 121.png がある
- IN_DIR に signal.csv または signals.csv があり、
  ・ヘッダなし
  ・行番号 n (1始まり) が n.png に対応
  ・左電圧: 列A (index 0)
  ・右電圧: 列G (index 6)

出力:
- IN_DIR/signals_pairs_121x121.csv
  カラム:
    target_idx, current_idx,
    target_img, current_img,
    target_left, target_right,
    current_left, current_right
"""

import os
import csv

# ===== パス設定 =====
IN_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi"

# signal.csv / signals.csv どちらでも動くようにする
CANDIDATE_CSV_NAMES = ["signal.csv", "signals.csv"]

IN_CSV = None
for name in CANDIDATE_CSV_NAMES:
    path = os.path.join(IN_DIR, name)
    if os.path.exists(path):
        IN_CSV = path
        break

if IN_CSV is None:
    raise FileNotFoundError(
        f"signal.csv / signals.csv が見つかりませんでした in {IN_DIR}"
    )

OUT_CSV = os.path.join(IN_DIR, "signals_pairs_121x121.csv")


def load_signals_no_header(in_csv):
    """
    ヘッダなし signals.csv を読み込み、
    行番号(1始まり) -> (left, right) の辞書を作る。

    - 行番号 n が n.png に対応
    - 左電圧: 列A (index 0)
    - 右電圧: 列G (index 6)
    """
    mapping = {}  # idx(int) -> (left, right)

    with open(in_csv, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError("signals.csv にデータ行がありません。")

    for line_idx, row in enumerate(rows, start=1):  # 1行目 = 1.png
        if len(row) <= 6:
            raise RuntimeError(
                f"{in_csv} の {line_idx} 行目に 7 列以上のデータがありません。"
            )
        try:
            left = float(row[0])  # A列
            right = float(row[6])  # G列
        except ValueError:
            raise RuntimeError(f"{in_csv} の {line_idx} 行目が数値でパースできません: {row}")
        mapping[line_idx] = (left, right)

    return mapping


def main():
    # 1〜121 の電圧を読み込み
    idx_to_signal = load_signals_no_header(IN_CSV)

    # 一応 1〜121 が揃っているかチェック
    missing = [i for i in range(1, 122) if i not in idx_to_signal]
    if missing:
        raise RuntimeError(f"signals に存在しないインデックスがあります: {missing}")

    # 画像の存在チェック（軽く）
    for i in range(1, 122):
        img_path = os.path.join(IN_DIR, f"{i}.png")
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"画像がありません: {img_path}")

    # 出力CSV
    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.writer(f_out)
        writer.writerow([
            "target_idx", "current_idx",
            "target_img", "current_img",
            "target_left", "target_right",
            "current_left", "current_right",
        ])

        count = 0
        for target_idx in range(1, 122):   # 1..121
            t_left, t_right = idx_to_signal[target_idx]
            t_name = f"{target_idx}.png"

            for current_idx in range(1, 122):
                c_left, c_right = idx_to_signal[current_idx]
                c_name = f"{current_idx}.png"

                writer.writerow([
                    target_idx,
                    current_idx,
                    t_name,
                    c_name,
                    f"{t_left:.6f}",
                    f"{t_right:.6f}",
                    f"{c_left:.6f}",
                    f"{c_right:.6f}",
                ])
                count += 1

    print(f"[DONE] 121x121 = {121*121} ペアを書き出しました。")
    print(f"[INFO] 出力CSV: {OUT_CSV}")


if __name__ == "__main__":
    main()
