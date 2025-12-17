# -*- coding: utf-8 -*-
"""
eval_module_controller.py
- train_module_controller.py で学習した ModuleControllerNet を読み込んで評価する
- MSE / MAE (全体 + 左右別) を計算
- 予測 vs 正解 の散布図を PNG 出力
- 誤差ヒストグラムも出力
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T

# === 学習スクリプトからクラスとパスをインポート ==========================
from FF_train import (
    ModuleImageVoltageDataset,
    ModuleControllerNet,
    IMG_DIR,
    CSV_PATH,
    MODEL_SAVE_PATH,
)

# 評価結果の保存先（ログディレクトリを流用してもOK）
EVAL_DIR = Path(os.path.dirname(MODEL_SAVE_PATH)) / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- データセット・ローダ --------------------------
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ModuleImageVoltageDataset(IMG_DIR, CSV_PATH, transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # ---- モデル読み込み -------------------------------
    model = ModuleControllerNet(feat_dim=128).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)

    # 学習スクリプトでは {"model_state_dict": ..., "best_val_loss": ...}
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- 推論して全サンプルの予測/正解をためる ---------
    all_targets = []
    all_preds = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            outputs = model(imgs)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)  # shape (N, 2)
    all_preds = np.concatenate(all_preds, axis=0)      # shape (N, 2)

    # ---- 指標計算 --------------------------------------
    # 全体
    mse_total = np.mean((all_preds - all_targets) ** 2)
    mae_total = np.mean(np.abs(all_preds - all_targets))

    # 左右それぞれ
    mse_left = np.mean((all_preds[:, 0] - all_targets[:, 0]) ** 2)
    mae_left = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))

    mse_right = np.mean((all_preds[:, 1] - all_targets[:, 1]) ** 2)
    mae_right = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))

    # 簡易的な R^2 も出しておく
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot

    r2_left = r2_score(all_targets[:, 0], all_preds[:, 0])
    r2_right = r2_score(all_targets[:, 1], all_preds[:, 1])

    print("===== Evaluation Result =====")
    print(f"MSE  (total):  {mse_total:.6f}")
    print(f"MAE  (total):  {mae_total:.6f}")
    print(f"MSE  (left):   {mse_left:.6f}")
    print(f"MAE  (left):   {mae_left:.6f}")
    print(f"MSE  (right):  {mse_right:.6f}")
    print(f"MAE  (right):  {mae_right:.6f}")
    print(f"R^2  (left):   {r2_left:.4f}")
    print(f"R^2  (right):  {r2_right:.4f}")

    # ---- 結果を CSV に保存 -----------------------------
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
    pd.DataFrame([metrics]).to_csv(EVAL_DIR / "metrics.csv", index=False)

    # ---- 予測 vs 正解 散布図 ---------------------------
    # 左
    plt.figure()
    plt.scatter(all_targets[:, 0], all_preds[:, 0], s=5, alpha=0.5)
    min_v = min(all_targets[:, 0].min(), all_preds[:, 0].min())
    max_v = max(all_targets[:, 0].max(), all_preds[:, 0].max())
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("True V_left")
    plt.ylabel("Pred V_left")
    plt.title("Pred vs True (Left Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "pred_vs_true_left.png")
    plt.close()

    # 右
    plt.figure()
    plt.scatter(all_targets[:, 1], all_preds[:, 1], s=5, alpha=0.5)
    min_v = min(all_targets[:, 1].min(), all_preds[:, 1].min())
    max_v = max(all_targets[:, 1].max(), all_preds[:, 1].max())
    plt.plot([min_v, max_v], [min_v, max_v])
    plt.xlabel("True V_right")
    plt.ylabel("Pred V_right")
    plt.title("Pred vs True (Right Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "pred_vs_true_right.png")
    plt.close()

    # ---- 誤差ヒストグラム -----------------------------
    err_left = all_preds[:, 0] - all_targets[:, 0]
    err_right = all_preds[:, 1] - all_targets[:, 1]

    plt.figure()
    plt.hist(err_left, bins=50, alpha=0.7)
    plt.xlabel("Error (Pred - True) V_left")
    plt.ylabel("Count")
    plt.title("Error Histogram (Left Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_hist_left.png")
    plt.close()

    plt.figure()
    plt.hist(err_right, bins=50, alpha=0.7)
    plt.xlabel("Error (Pred - True) V_right")
    plt.ylabel("Count")
    plt.title("Error Histogram (Right Voltage)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "error_hist_right.png")
    plt.close()

    print(f"Evaluation results saved in: {EVAL_DIR}")


if __name__ == "__main__":
    evaluate()
