# -*- coding: utf-8 -*-
"""
eval_module_controller_beam2ch.py

- train_module_controller_beam2ch.py で学習した ModuleControllerNet を評価する
- 入力: 2chビームマスク (ch0=上梁, ch1=下梁) の .npy
- 評価: MSE / MAE / R^2（全体 + 左右別）
- 予測 vs 正解 の散布図、誤差ヒストグラムを PNG 出力
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

# ===== 学習スクリプトからインポート =====
# ファイル名が違うならここを修正（例: from FF_2ch_train import ...）
from FF_2ch_train import (
    BeamMaskDataset,
    ModuleControllerNet,
    DATA_DIR,
    CSV_PATH,
    MODEL_SAVE_PATH,
    LOG_DIR,
)

# 評価結果の保存先
EVAL_DIR = Path(LOG_DIR) / "eval_beam2ch_nocleansing"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== データセット / ローダ =====
    dataset = BeamMaskDataset(DATA_DIR, CSV_PATH)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # ===== モデル読み込み =====
    model = ModuleControllerNet(feat_dim=128).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 正規化に使った統計量（checkpoint から取得）
    v_left_mean  = float(ckpt.get("v_left_mean",  dataset.v_left_mean))
    v_left_std   = float(ckpt.get("v_left_std",   dataset.v_left_std))
    v_right_mean = float(ckpt.get("v_right_mean", dataset.v_right_mean))
    v_right_std  = float(ckpt.get("v_right_std",  dataset.v_right_std))

    print("Normalization stats (from checkpoint):")
    print(f"  left mean={v_left_mean:.4f}, std={v_left_std:.4f}")
    print(f"  right mean={v_right_mean:.4f}, std={v_right_std:.4f}")

    all_targets_norm = []
    all_preds_norm   = []

    # ===== 推論 =====
    with torch.no_grad():
        for x, y in loader:  # y は正規化済み
            x = x.to(device)
            y = y.to(device)

            pred = model(x)  # 正規化スケール

            all_targets_norm.append(y.cpu().numpy())
            all_preds_norm.append(pred.cpu().numpy())

    all_targets_norm = np.concatenate(all_targets_norm, axis=0)  # (N, 2)
    all_preds_norm   = np.concatenate(all_preds_norm,   axis=0)  # (N, 2)

    # ===== 正規化を元の電圧スケールに戻す =====
    # idx 0: V_left, idx 1: V_right
    all_targets = np.empty_like(all_targets_norm)
    all_preds   = np.empty_like(all_preds_norm)

    all_targets[:, 0] = all_targets_norm[:, 0] * v_left_std  + v_left_mean
    all_targets[:, 1] = all_targets_norm[:, 1] * v_right_std + v_right_mean

    all_preds[:, 0]   = all_preds_norm[:, 0]   * v_left_std  + v_left_mean
    all_preds[:, 1]   = all_preds_norm[:, 1]   * v_right_std + v_right_mean

    # ===== 指標計算（電圧スケールで） =====
    mse_total = np.mean((all_preds - all_targets) ** 2)
    mae_total = np.mean(np.abs(all_preds - all_targets))

    mse_left = np.mean((all_preds[:, 0] - all_targets[:, 0]) ** 2)
    mae_left = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))

    mse_right = np.mean((all_preds[:, 1] - all_targets[:, 1]) ** 2)
    mae_right = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))

    r2_left  = r2_score(all_targets[:, 0], all_preds[:, 0])
    r2_right = r2_score(all_targets[:, 1], all_preds[:, 1])

    print("===== Evaluation Result (in voltage scale) =====")
    print(f"MSE  (total):  {mse_total:.6f}")
    print(f"MAE  (total):  {mae_total:.6f}")
    print(f"MSE  (left):   {mse_left:.6f}")
    print(f"MAE  (left):   {mae_left:.6f}")
    print(f"MSE  (right):  {mse_right:.6f}")
    print(f"MAE  (right):  {mae_right:.6f}")
    print(f"R^2  (left):   {r2_left:.4f}")
    print(f"R^2  (right):  {r2_right:.4f}")

    # ===== CSV に保存 =====
    metrics = {
        "mse_total": mse_total,
        "mae_total": mae_total,
        "mse_left": mse_left,
        "mae_left": mae_left,
        "mse_right": mse_right,
        "mae_right": mae_right,
        "r2_left": r2_left,
        "r2_right": r2_right,
    }
    pd.DataFrame([metrics]).to_csv(EVAL_DIR / "metrics_beam2ch.csv", index=False)

    # ===== 予測 vs 正解 散布図 =====
    # Left
    plt.figure()
    plt.scatter(all_targets[:, 0], all_preds[:, 0], s=5, alpha=0.5)
    min_v = min(all_targets[:, 0].min(), all_preds[:, 0].min())
    max_v = max(all_targets[:, 0].max(), all_preds[:, 0].max())
    plt.plot([min_v, max_v], [min_v, max_v], 'k--')
    plt.xlabel("True V_left [V]")
    plt.ylabel("Pred V_left [V]")
    plt.title("Pred vs True (Left Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "pred_vs_true_left_beam2ch.png")
    plt.close()

    # Right
    plt.figure()
    plt.scatter(all_targets[:, 1], all_preds[:, 1], s=5, alpha=0.5)
    min_v = min(all_targets[:, 1].min(), all_preds[:, 1].min())
    max_v = max(all_targets[:, 1].max(), all_preds[:, 1].max())
    plt.plot([min_v, max_v], [min_v, max_v], 'k--')
    plt.xlabel("True V_right [V]")
    plt.ylabel("Pred V_right [V]")
    plt.title("Pred vs True (Right Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "pred_vs_true_right_beam2ch.png")
    plt.close()

    # ===== 誤差ヒストグラム =====
    err_left  = all_preds[:, 0] - all_targets[:, 0]
    err_right = all_preds[:, 1] - all_targets[:, 1]

    plt.figure()
    plt.hist(err_left, bins=50, alpha=0.7)
    plt.xlabel("Error (Pred - True) [V]  (Left)")
    plt.ylabel("Count")
    plt.title("Error Histogram (Left Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_hist_left_beam2ch.png")
    plt.close()

    plt.figure()
    plt.hist(err_right, bins=50, alpha=0.7)
    plt.xlabel("Error (Pred - True) [V]  (Right)")
    plt.ylabel("Count")
    plt.title("Error Histogram (Right Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_hist_right_beam2ch.png")
    plt.close()

    print(f"\nEvaluation results saved in: {EVAL_DIR}")


if __name__ == "__main__":
    evaluate()
