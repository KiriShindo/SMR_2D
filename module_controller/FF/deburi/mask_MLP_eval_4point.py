# -*- coding: utf-8 -*-
"""
学習済み voltage_regressor_best.pth を使って評価を行うスクリプト

- クレンジング後の mask_relation_filtered.csv を読み込む
- signals.csv から [V_left, V_right] を取得
- 学習時と同じ正規化 (x_mean, x_std) を使って入力を標準化
- train / val / 全体 について MSE, MAE を計算
- val セットの:
    - 散布図 (真値 vs 予測, y=x 線付き)
    - 誤差ヒストグラム (pred - true)
  をPNGで保存
"""

import os
import re
import numpy as np
import pandas as pd
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

import matplotlib.pyplot as plt


# ===== パス設定 =====
BASE_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon"

FILTERED_DIR = os.path.join(BASE_DIR, "roi_aug_rot_shift_scale_rect2mask_4point_filtered")
CSV_FILTERED = os.path.join(FILTERED_DIR, "mask_relation_filtered.csv")

SIGNALS_CSV = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\raw\signals.csv"

MODEL_SAVE_DIR = os.path.join(FILTERED_DIR, "training_logs")
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "voltage_regressor_best.pth")

# 評価結果保存先
EVAL_SCATTER_LEFT  = os.path.join(MODEL_SAVE_DIR, "eval_scatter_left.png")
EVAL_SCATTER_RIGHT = os.path.join(MODEL_SAVE_DIR, "eval_scatter_right.png")
EVAL_HIST_LEFT     = os.path.join(MODEL_SAVE_DIR, "error_hist_left.png")
EVAL_HIST_RIGHT    = os.path.join(MODEL_SAVE_DIR, "error_hist_right.png")

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

# ===== 評価設定 =====
VAL_RATIO = 0.2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 共通ユーティリティ
# ============================================================================

def extract_base_index(filename: str) -> int:
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot extract leading integer from filename: {filename}")
    return int(m.group(1))


def load_filtered_and_signals(filtered_csv: str, signals_csv: str):
    df = pd.read_csv(filtered_csv)
    sig = pd.read_csv(signals_csv, header=None)

    v_left_all  = sig.iloc[:, 0].values
    v_right_all = sig.iloc[:, 6].values

    X_list, y_list, base_ids = [], [], []
    for _, row in df.iterrows():
        fname = row["filename"]
        i = extract_base_index(fname)
        idx = i - 1
        if idx < 0 or idx >= len(sig):
            continue

        feat = np.array([
            row["mask1_p1_x"], row["mask1_p1_y"],
            row["mask1_p2_x"], row["mask1_p2_y"],
            row["mask2_p1_x"], row["mask2_p1_y"],
            row["mask2_p2_x"], row["mask2_p2_y"],
        ], dtype=np.float32)
        X_list.append(feat)
        y_list.append(np.array([v_left_all[idx], v_right_all[idx]], dtype=np.float32))
        base_ids.append(i)

    X = np.stack(X_list)
    y = np.stack(y_list)
    base_ids = np.array(base_ids, dtype=np.int32)
    return X, y, base_ids


class VoltageDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


class MLPRegressor(nn.Module):
    def __init__(self, in_dim=8, hidden_dim=64, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x): return self.net(x)


def create_train_val_samplers(base_ids, val_ratio=0.2, seed=42):
    rng = np.random.RandomState(seed)
    uids = np.unique(base_ids)
    rng.shuffle(uids)
    n_val = max(1, int(len(uids) * val_ratio))
    val_ids = set(uids[:n_val])
    train_ids = set(uids[n_val:])
    train_idx = [i for i, bid in enumerate(base_ids) if bid in train_ids]
    val_idx   = [i for i, bid in enumerate(base_ids) if bid in val_ids]
    return SubsetRandomSampler(train_idx), SubsetRandomSampler(val_idx)


def eval_model(model, loader):
    model.eval()
    all_t, all_p = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            all_t.append(yb.cpu().numpy())
            all_p.append(model(xb).cpu().numpy())
    if not all_t: return None, None, {}
    y_true, y_pred = np.concatenate(all_t), np.concatenate(all_p)
    diff = y_pred - y_true
    mse, mae = np.mean(diff**2), np.mean(np.abs(diff))
    metrics = dict(
        mse=mse, mae=mae,
        mse_left=np.mean(diff[:,0]**2),
        mse_right=np.mean(diff[:,1]**2),
        mae_left=np.mean(np.abs(diff[:,0])),
        mae_right=np.mean(np.abs(diff[:,1])),
        n=len(y_true)
    )
    return y_true, y_pred, metrics


# ============================================================================
# 可視化
# ============================================================================

def plot_scatter_and_error(y_true, y_pred, save_scatter, save_hist, label):
    """散布図 (y=x付き) ＆ 誤差ヒストグラム"""
    # 散布図
    plt.figure(figsize=(5,5))
    plt.scatter(y_true, y_pred, alpha=0.6)
    lims = [
        min(y_true.min(), y_pred.min()),
        max(y_true.max(), y_pred.max())
    ]
    plt.plot(lims, lims, "r--", label="y = x")
    plt.xlabel("True Voltage")
    plt.ylabel("Predicted Voltage")
    plt.title(f"True vs Pred ({label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_scatter)
    plt.close()

    # 誤差ヒストグラム
    err = y_pred - y_true
    plt.figure()
    plt.hist(err, bins=40, color="gray", alpha=0.7, edgecolor="black")
    plt.xlabel("Prediction Error (Pred - True)")
    plt.ylabel("Count")
    plt.title(f"Error Distribution ({label})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_hist)
    plt.close()


# ============================================================================
# メイン
# ============================================================================

def main():
    if not os.path.exists(BEST_MODEL_PATH):
        print(f"[ERROR] Model not found: {BEST_MODEL_PATH}")
        return

    X, y, base_ids = load_filtered_and_signals(CSV_FILTERED, SIGNALS_CSV)
    print(f"Samples: {len(X)}")

    state = torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=False)
    model = MLPRegressor().to(DEVICE)
    model.load_state_dict(state["model_state"])
    x_mean, x_std = state["x_mean"], state["x_std"]

    X_norm = (X - x_mean) / (x_std + 1e-6)
    dataset = VoltageDataset(X_norm, y)
    train_smp, val_smp = create_train_val_samplers(base_ids, VAL_RATIO, SEED)
    train_loader = DataLoader(dataset, 64, sampler=train_smp)
    val_loader   = DataLoader(dataset, 64, sampler=val_smp)

    # 評価
    print("\nEvaluating...")
    y_val, y_pred, m = eval_model(model, val_loader)
    print(f"Val metrics: {m}")

    # 可視化
    if y_val is not None:
        plot_scatter_and_error(
            y_val[:,0], y_pred[:,0],
            EVAL_SCATTER_LEFT, EVAL_HIST_LEFT, "Left"
        )
        plot_scatter_and_error(
            y_val[:,1], y_pred[:,1],
            EVAL_SCATTER_RIGHT, EVAL_HIST_RIGHT, "Right"
        )
        print(f"\nSaved plots to:\n  {EVAL_SCATTER_LEFT}\n  {EVAL_HIST_LEFT}\n  {EVAL_SCATTER_RIGHT}\n  {EVAL_HIST_RIGHT}")

    print("\n✅ Evaluation finished.")


if __name__ == "__main__":
    main()
