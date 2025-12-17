# -*- coding: utf-8 -*-
"""
train_module_controller_beam2ch.py

- 入力: 2chビームマスク (ch0=上梁, ch1=下梁) の .npy
- 出力: [v_left, v_right] の回帰
"""

import os
from pathlib import Path
import random

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ===== パス設定 =====
DATA_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\FF\1module_dataset_max_DAC\silicon\roi_aug_rot_shift_beammask_2ch"
)
CSV_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\FF\1module_dataset_max_DAC\silicon\raw\signals.csv"
)

MODEL_SAVE_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\FF\module_controller_beam2ch_nocleansing_best.pth"
)
LOG_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\FF\logs_beam2ch_nocleansing"
)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ===== ハイパーパラメータ =====
BATCH_SIZE = 64
NUM_EPOCHS = 1000
LEARNING_RATE = 1e-3
VAL_RATIO = 0.1
WEIGHT_DECAY = 1e-4
SAVE_INTERVAL = 100

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ===== Dataset 定義 =====
class BeamMaskDataset(Dataset):
    """
    *_beam2ch.npy (2, H, W) を読み込んで電圧と紐付ける
    ファイル名: "57_beam2ch.npy" -> signals.csv の 57 行目と対応
    """
    def __init__(self, npy_dir: Path, csv_path: Path):
        self.npy_dir = Path(npy_dir)
        self.volt_df = pd.read_csv(csv_path, header=None)

        self.paths = sorted(self.npy_dir.glob("*_beam2ch.npy"))
        if len(self.paths) == 0:
            raise RuntimeError(f"No *_beam2ch.npy found in {self.npy_dir}")

        self.v_left_mean  = self.volt_df.iloc[:, 0].mean()
        self.v_left_std   = self.volt_df.iloc[:, 0].std(ddof=0) + 1e-8
        self.v_right_mean = self.volt_df.iloc[:, 6].mean()
        self.v_right_std  = self.volt_df.iloc[:, 6].std(ddof=0) + 1e-8

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        npy_path = self.paths[idx]
        mask_2ch = np.load(str(npy_path)).astype(np.float32)  # (2, H, W)

        x = torch.from_numpy(mask_2ch)  # (2, H, W)

        stem = npy_path.stem  # "57_beam2ch"
        base_idx_str = stem.split('_')[0]
        i = int(base_idx_str)

        row = self.volt_df.iloc[i - 1]
        v_left, v_right = float(row[0]), float(row[6])

        v_left_norm  = (v_left  - self.v_left_mean)  / self.v_left_std
        v_right_norm = (v_right - self.v_right_mean) / self.v_right_std
        y = torch.tensor([v_left_norm, v_right_norm], dtype=torch.float32)

        return x, y


# ===== モデル定義 (2ch入力CNN + GAP) =====
class BeamEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc  = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)           # (B, 128, 1, 1)
        x = x.view(x.size(0), -1) # (B, 128)
        x = self.fc(x)            # (B, out_dim)
        return x


class VoltageMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.net(x)


class ModuleControllerNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.encoder = BeamEncoder(feat_dim)
        self.head    = VoltageMLP(feat_dim)

    def forward(self, x):
        feat = self.encoder(x)
        out  = self.head(feat)
        return out


# ===== 学習ルーチン =====
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0.0
    with tqdm(loader, desc=f"[Epoch {epoch}/{num_epochs}] Train", leave=False) as pbar:
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device, epoch, num_epochs):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with tqdm(loader, desc=f"[Epoch {epoch}/{num_epochs}] Val", leave=False) as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                pred = model(x)
                loss = criterion(pred, y)
                total_loss += loss.item() * x.size(0)
                pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    full_dataset = BeamMaskDataset(DATA_DIR, CSV_PATH)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader   = DataLoader(
        val_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    model = ModuleControllerNet(feat_dim=128).to(device)
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Overall Progress"):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS
        )
        val_loss = eval_one_epoch(
            model, val_loader, criterion, device, epoch, NUM_EPOCHS
        )

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"[Epoch {epoch:04d}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # 定期保存
        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            ckpt_path = LOG_DIR / f"checkpoint_epoch{epoch:04d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
                "v_left_mean":  full_dataset.v_left_mean,
                "v_left_std":   full_dataset.v_left_std,
                "v_right_mean": full_dataset.v_right_mean,
                "v_right_std":  full_dataset.v_right_std,
            }, ckpt_path)
            tqdm.write(f"  -> Saved checkpoint: {ckpt_path}")

            plt.figure()
            plt.plot(train_losses, label="Train")
            plt.plot(val_losses,   label="Val")
            plt.xlabel("Epoch")
            plt.ylabel("SmoothL1 Loss")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.savefig(LOG_DIR / f"loss_curve_epoch{epoch:04d}.png")
            plt.close()

            pd.DataFrame({
                "epoch": list(range(1, epoch + 1)),
                "train_loss": train_losses,
                "val_loss": val_losses,
            }).to_csv(LOG_DIR / "loss_history.csv", index=False)

        # ベストモデルの更新（early stop はしない）
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val_loss,
                "v_left_mean":  full_dataset.v_left_mean,
                "v_left_std":   full_dataset.v_left_std,
                "v_right_mean": full_dataset.v_right_mean,
                "v_right_std":  full_dataset.v_right_std,
            }, MODEL_SAVE_PATH)
            tqdm.write(f"  -> Best model updated (val={val_loss:.6f}).")

    print("Training finished.")
    print(f"Best val loss: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
