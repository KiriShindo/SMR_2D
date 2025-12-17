# -*- coding: utf-8 -*-
"""
4点の交点座標（8次元）から [V_left, V_right] を推定する回帰モデル（PyTorch）

- クレンジング後の mask_relation_filtered.csv を読み込む
- filename の "i_aug_j.png" から先頭の数字 i を取り出す
- signals.csv の i 行目（1始まり想定 → 実装では i-1）から
    - A列: V_left
    - G列: V_right
  を教師データとして取得
- 8次元入力 → 2次元出力の MLP 回帰モデルを学習する
- tqdm でプログレスバー表示
- 一定エポックごとに:
    - モデル checkpoint 保存
    - loss ログ CSV 出力
    - 学習曲線 PNG 出力
"""

import os
import re
import csv
import numpy as np
import pandas as pd
from typing import Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from tqdm import tqdm
import matplotlib.pyplot as plt

# ===== パス設定 =====
BASE_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon"

FILTERED_DIR = os.path.join(
    BASE_DIR,
    "roi_aug_rot_shift_scale_rect2mask_4point_filtered"
)
CSV_FILTERED = os.path.join(FILTERED_DIR, "mask_relation_filtered.csv")

# 教師データ（生信号）
SIGNALS_CSV = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\raw\signals.csv"

MODEL_SAVE_DIR = os.path.join(FILTERED_DIR, "training_logs")
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "voltage_regressor_best.pth")
LOSS_LOG_CSV   = os.path.join(MODEL_SAVE_DIR, "loss_log.csv")
LOSS_PLOT_PNG  = os.path.join(MODEL_SAVE_DIR, "loss_curve.png")

# ===== ハイパーパラメータ =====
BATCH_SIZE = 64
EPOCHS = 3000
LR = 1e-3
VAL_RATIO = 0.2  # base index ごとの train/val 分割比
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 何エポックごとに checkpoint & 学習曲線を保存するか
SAVE_INTERVAL = 100

# ============================================================================
# ユーティリティ
# ============================================================================

def extract_base_index(filename: str) -> int:
    """
    filename から先頭の数字 i を取り出す。
    例:
        "3_aug_0001.png" -> 3
        "12.png"         -> 12
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    m = re.match(r"(\d+)", stem)
    if not m:
        raise ValueError(f"Cannot extract leading integer from filename: {filename}")
    return int(m.group(1))


def load_filtered_and_signals(
    filtered_csv: str,
    signals_csv: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    クレンジング後のCSVと signals.csv から
    - X: (N, 8) 特徴量（4点の座標）
    - y: (N, 2) ラベル [V_left, V_right]
    - base_ids: (N,) 各サンプルに対応する i（元画像インデックス）
    を返す。
    """
    # クレンジング後 CSV 読み込み
    df = pd.read_csv(filtered_csv)

    required_cols = [
        "filename",
        "mask1_p1_x", "mask1_p1_y", "mask1_p2_x", "mask1_p2_y",
        "mask2_p1_x", "mask2_p1_y", "mask2_p2_x", "mask2_p2_y",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise RuntimeError(f"Column '{c}' is missing in filtered CSV.")

    # signals.csv 読み込み（ヘッダなし, A〜L）
    sig = pd.read_csv(signals_csv, header=None)
    # A列(0), G列(6) を取り出す
    v_left_all  = sig.iloc[:, 0].values  # (num_rows,)
    v_right_all = sig.iloc[:, 6].values

    X_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    base_ids: List[int] = []

    for _, row in df.iterrows():
        fname = row["filename"]
        i = extract_base_index(fname)  # 1,2,3,...

        # 行番号との対応: i 行目(1始まり) → pandas行 index = i-1 と仮定
        idx = i - 1
        if idx < 0 or idx >= len(sig):
            # signals.csv 側に存在しない index はスキップ
            continue

        v_left = float(v_left_all[idx])
        v_right = float(v_right_all[idx])

        # 4点の座標 → 8次元特徴量
        feat = np.array([
            row["mask1_p1_x"], row["mask1_p1_y"],
            row["mask1_p2_x"], row["mask1_p2_y"],
            row["mask2_p1_x"], row["mask2_p1_y"],
            row["mask2_p2_x"], row["mask2_p2_y"],
        ], dtype=np.float32)

        X_list.append(feat)
        y_list.append(np.array([v_left, v_right], dtype=np.float32))
        base_ids.append(i)

    X = np.stack(X_list, axis=0)  # (N,8)
    y = np.stack(y_list, axis=0)  # (N,2)
    base_ids = np.array(base_ids, dtype=np.int32)

    return X, y, base_ids


class VoltageDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int = 8, hidden_dim: int = 64, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def create_train_val_samplers(
    base_ids: np.ndarray,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[SubsetRandomSampler, SubsetRandomSampler]:
    """
    base_ids（元の i）ごとにグルーピングして train/val を分ける。
    同じ i の augment 群が train と val にまたがらないようにする。
    """
    rng = np.random.RandomState(seed)
    unique_ids = np.unique(base_ids)
    rng.shuffle(unique_ids)

    num_val_ids = max(1, int(len(unique_ids) * val_ratio))
    val_ids = set(unique_ids[:num_val_ids])
    train_ids = set(unique_ids[num_val_ids:])

    train_indices = [idx for idx, bid in enumerate(base_ids) if bid in train_ids]
    val_indices = [idx for idx, bid in enumerate(base_ids) if bid in val_ids]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    return train_sampler, val_sampler


def save_loss_log_and_plot(epochs: List[int],
                           train_losses: List[float],
                           val_losses: List[float],
                           csv_path: str,
                           fig_path: str):
    """lossのCSVと学習曲線PNGを保存"""
    # CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, vl in zip(epochs, train_losses, val_losses):
            writer.writerow([e, tr, vl])

    # PNG（学習曲線）
    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


# ============================================================================
# メイン処理
# ============================================================================

def main():
    print("Loading data...")
    X, y, base_ids = load_filtered_and_signals(CSV_FILTERED, SIGNALS_CSV)
    print(f"Total samples after matching with signals.csv: {X.shape[0]}")

    # 入力の標準化（全データで）
    x_mean = X.mean(axis=0, keepdims=True)
    x_std = X.std(axis=0, keepdims=True) + 1e-6
    X_norm = (X - x_mean) / x_std

    # PyTorch Dataset
    dataset = VoltageDataset(X_norm, y)

    # train/val split（base_ids単位）
    train_sampler, val_sampler = create_train_val_samplers(
        base_ids, val_ratio=VAL_RATIO, seed=SEED
    )

    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

    model = MLPRegressor(in_dim=8, hidden_dim=64, out_dim=2).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"Device: {DEVICE}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    best_val_loss = float("inf")
    best_state = None

    epochs_list = []
    train_losses = []
    val_losses = []

    for epoch in range(1, EPOCHS + 1):
        # ===== train =====
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [train]", leave=False)
        for xb, yb in pbar:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * xb.size(0)
            train_count += xb.size(0)
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        train_loss = train_loss_sum / max(train_count, 1)

        # ===== validation =====
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [val]", leave=False)
            for xb, yb in pbar_val:
                xb = xb.to(DEVICE)
                yb = yb.to(DEVICE)
                pred = model(xb)
                loss = criterion(pred, yb)
                val_loss_sum += loss.item() * xb.size(0)
                val_count += xb.size(0)
                pbar_val.set_postfix({"loss": f"{loss.item():.4f}"})
        val_loss = val_loss_sum / max(val_count, 1)

        epochs_list.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # ベストモデル更新
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "model_state": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
            }

        print(f"[Epoch {epoch:03d}] train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

        # 一定エポックごとに checkpoint ＆ ログ出力
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            ckpt_path = os.path.join(MODEL_SAVE_DIR, f"voltage_regressor_epoch_{epoch:04d}.pth")
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "x_mean": x_mean,
                    "x_std": x_std,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            print(f"  -> checkpoint saved: {ckpt_path}")

            # loss ログ & 学習曲線
            save_loss_log_and_plot(epochs_list, train_losses, val_losses,
                                   LOSS_LOG_CSV, LOSS_PLOT_PNG)
            print(f"  -> loss log & curve saved.")

    # ===== ベストモデル保存 =====
    if best_state is not None:
        torch.save(best_state, BEST_MODEL_PATH)
        print(f"\nSaved best model to: {BEST_MODEL_PATH}")
        print(f"Best val_loss: {best_val_loss:.6f}")
    else:
        print("No best_state saved (something went wrong).")


if __name__ == "__main__":
    main()
