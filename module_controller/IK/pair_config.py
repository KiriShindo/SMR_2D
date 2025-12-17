# -*- coding: utf-8 -*-
"""
list_adjacent_pairs.py

augment_random_affine_babbling.py で生成した
signals_aug.csv を読み込み、
orig_index ごとにグルーピングしたうえで、

  - k と k+1 の両方にデータがある orig_index の組だけを
    「隣接ペア」として検出し、
  - それぞれについて
      * group_k のサンプル数
      * group_{k+1} のサンプル数
      * 全組み合わせペア数 = len(group_k) * len(group_{k+1})
    を標準出力に表示する。

これで「隣接ペアとして使える (k, k+1) がどれだけあるか」を確認できる。
"""

import os
import csv
from collections import defaultdict

# ===== ここをあなたの拡張フォルダに合わせて変更 =====
AUG_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon\aug_shift_500_10_test"
CSV_AUG = os.path.join(AUG_DIR, "signals_aug.csv")
# ===============================================


def load_groups(csv_path):
    """
    signals_aug.csv を読み込み、
    orig_index ごとに行をグルーピングして返す。

    戻り値:
        groups: dict[int, list[dict]]
                key = orig_index
                value = その orig_index を持つ行 (DictReader の row) のリスト
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError("signals_aug.csv にデータ行がありません。")

    # 必須カラムの確認
    required_cols = {"filename", "orig_index", "Left_V", "Right_V"}
    missing = required_cols - set(rows[0].keys())
    if missing:
        raise RuntimeError(f"signals_aug.csv に必要な列が足りません: {missing}")

    groups = defaultdict(list)

    for row in rows:
        try:
            orig = int(row["orig_index"])
        except Exception:
            # 変な値が入っていたらスキップ
            continue
        groups[orig].append(row)

    return groups


def main():
    print(f"[INFO] Loading: {CSV_AUG}")
    groups = load_groups(CSV_AUG)

    if not groups:
        print("[WARN] 有効な orig_index グループが 0 件でした。")
        return

    sorted_indices = sorted(groups.keys())

    print(f"[INFO] 検出された orig_index の数: {len(sorted_indices)}")
    print(f"[INFO] 最小 orig_index: {sorted_indices[0]}, 最大 orig_index: {sorted_indices[-1]}\n")

    total_neighbor_pairs = 0
    total_sample_pairs = 0

    for k in sorted_indices:
        k_next = k + 1
        if k_next not in groups:
            continue

        group_k = groups[k]
        group_knext = groups[k_next]

        n_k = len(group_k)
        n_k1 = len(group_knext)
        n_pairs = n_k * n_k1

        total_neighbor_pairs += 1
        total_sample_pairs += n_pairs

        print(
            f"  Neighbor orig_index pair: {k} - {k_next} "
            f"(samples: {n_k} x {n_k1} = {n_pairs})"
        )

    print("\n===== Summary =====")
    print(f"隣接 orig_index ペア数: {total_neighbor_pairs}")
    print(f"全組み合わせサンプル数の合計: {total_sample_pairs}")


if __name__ == "__main__":
    main()
