# -*- coding: utf-8 -*-
"""
train_voltage_to_mask.py

電圧 [Left_V, Right_V] -> 2chマスク (2, H, W) を生成する NN の訓練スクリプト。

・機能
  - 電圧を平均0・分散1で正規化して入力
  - 損失は BCE + Dice Loss（細いビーム領域を重視）
  - tqdm プログレスバー
  - 各エポックの train/val loss を CSV に記録
  - save_interval ごとに:
      - モデルパラメータ保存
      - 学習曲線 (loss) の PNG 出力
      - 固定サンプルに対する GT / 生成マスク画像を保存

前提:
  - BEAMMASK_DIR 配下に {stem}_beam2ch.npy がある
  - CSV_PATH に signals_aug.csv があり、
      A列: filename
      C列: Left_V
      D列: Right_V
"""

import os
import csv
from pathlib import Path

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm


# =======================================
# Dataset: 電圧 -> 2chマスク（電圧正規化込み）
# =======================================

class VoltageToBeamMaskDataset(Dataset):
    """
    電圧 -> 2chマスク学習用 Dataset

    - beammask_dir:
        {stem}_beam2ch.npy が格納されているフォルダ
    - csv_path:
        signals_aug.csv 的な電圧情報が入った CSV
    - v_left_col, v_right_col:
        CSV 内の列名に合わせて指定
    """

    def __init__(self, beammask_dir, csv_path,
                 v_left_col="Left_V", v_right_col="Right_V"):
        self.beammask_dir = Path(beammask_dir)
        self.csv_path = Path(csv_path)
        self.v_left_col = v_left_col
        self.v_right_col = v_right_col

        self.samples = []
        volts_list = []  # 電圧の統計を取るために全件溜める

        with open(self.csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # CSV 内の画像ファイル名（A列: filename）
                fname = row["filename"]
                stem = Path(fname).stem

                npy_path = self.beammask_dir / f"{stem}_beam2ch.npy"
                if not npy_path.exists():
                    # 対応する npy が無ければスキップ
                    continue

                v_left = float(row[self.v_left_col])
                v_right = float(row[self.v_right_col])

                self.samples.append({
                    "npy_path": npy_path,
                    "v_left": v_left,
                    "v_right": v_right,
                    "stem": stem,
                })
                volts_list.append([v_left, v_right])

        if len(self.samples) == 0:
            raise RuntimeError(
                "有効なサンプルが 0 件です。"
                "beammask_dir / csv_path / 列名 (filename, Left_V, Right_V) を確認してください。"
            )

        # 1つ読んで形状チェック (全サンプル同じ前提)
        tmp = np.load(self.samples[0]["npy_path"])
        if tmp.ndim != 3 or tmp.shape[0] != 2:
            raise RuntimeError(f"想定外のマスク形状: {tmp.shape} (期待: (2, H, W))")
        self.out_ch, self.H, self.W = tmp.shape

        # 電圧の平均・標準偏差を計算しておく
        volts_arr = np.array(volts_list, dtype=np.float32)  # (N,2)
        self.v_mean = volts_arr.mean(axis=0)                 # (2,)
        self.v_std = volts_arr.std(axis=0) + 1e-6            # (2,) ゼロ割防止

        print(f"[INFO] voltage mean = {self.v_mean}, std = {self.v_std}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        # 2chマスク読み込み (2, H, W)
        mask = np.load(s["npy_path"]).astype(np.float32)
        mask = np.clip(mask, 0.0, 1.0)

        # 生電圧 (2,)
        v = np.array([s["v_left"], s["v_right"]], dtype=np.float32)
        # 平均0・分散1で正規化
        v = (v - self.v_mean) / self.v_std

        mask_t = torch.from_numpy(mask)  # (2, H, W)
        volt_t = torch.from_numpy(v)     # (2,)

        return volt_t, mask_t, s["stem"]


# =======================================
# Model: Voltage -> Mask
# =======================================

class VoltageEncoder(nn.Module):
    """
    2次元電圧 [Left_V, Right_V] -> 潜在ベクトル
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, v):
        # v: (B, 2)
        return self.net(v)  # (B, latent_dim)


class MaskDecoder(nn.Module):
    """
    潜在ベクトル -> 2chマスク (2, H, W)
    ConvTranspose2d で段階的にアップサンプル
    """
    def __init__(self, latent_dim, out_h, out_w):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w

        # 4段階アップサンプルする前提で、最初の空間サイズを決める
        self.init_h = out_h // 16
        self.init_w = out_w // 16
        if self.init_h < 1 or self.init_w < 1:
            raise ValueError(f"出力サイズが小さすぎるかも: H={out_h}, W={out_w}")

        self.fc = nn.Linear(latent_dim, 256 * self.init_h * self.init_w)

        self.deconv = nn.Sequential(
            # (B, 256, H/16, W/16) -> (B, 128, H/8, W/8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # (B, 128, H/8, W/8) -> (B, 64, H/4, W/4)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # (B, 64, H/4, W/4) -> (B, 32, H/2, W/2)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # (B, 32, H/2, W/2) -> (B, 2, H, W)
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
            # 出力はロジット (sigmoid 前)
        )

    def forward(self, z):
        # z: (B, latent_dim)
        x = self.fc(z)  # (B, 256 * H0 * W0)
        x = x.view(-1, 256, self.init_h, self.init_w)
        x = self.deconv(x)  # (B, 2, H', W')

        # H',W' がズレたときは補正
        if x.shape[2] != self.out_h or x.shape[3] != self.out_w:
            x = F.interpolate(x, size=(self.out_h, self.out_w),
                              mode="bilinear", align_corners=False)
        return x


class VoltageToMaskNet(nn.Module):
    """
    全体: 電圧(2,) -> 2chマスク(2, H, W)
    """
    def __init__(self, out_h, out_w, latent_dim=256):
        super().__init__()
        self.encoder = VoltageEncoder(latent_dim=latent_dim)
        self.decoder = MaskDecoder(latent_dim=latent_dim,
                                   out_h=out_h, out_w=out_w)

    def forward(self, v):
        # v: (B, 2)
        z = self.encoder(v)
        mask_logits = self.decoder(z)  # (B, 2, H, W)
        return mask_logits


# =======================================
# 損失: BCE + Dice
# =======================================

class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits: (B,2,H,W)
        targets: (B,2,H,W) in {0,1}
        """
        # BCE 部分
        bce_loss = self.bce(logits, targets)

        # Dice 部分
        probs = torch.sigmoid(logits)  # (B,2,H,W)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        union = probs_flat.sum(dim=1) + targets_flat.sum(dim=1) + 1e-6

        dice_loss = 1.0 - 2.0 * intersection / union
        dice_loss = dice_loss.mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# =======================================
# 可視化・ログユーティリティ
# =======================================

def save_loss_curve(log_dir, epochs, train_losses, val_losses):
    """
    学習曲線 (train / val loss) を PNG で保存
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    out_path = log_dir / "loss_curve.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[LOG] loss curve saved to: {out_path}")


def save_loss_csv(log_dir, epochs, train_losses, val_losses):
    """
    各エポックの loss を CSV に保存
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "loss_log.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss"])
        for e, tr, vl in zip(epochs, train_losses, val_losses):
            writer.writerow([e, tr, vl])

    print(f"[LOG] loss csv saved to: {csv_path}")


def save_generated_masks(
    model,
    dataset,
    indices,
    out_dir,
    device="cuda",
    epoch=None
):
    """
    固定サンプル (indices) に対して GT / 生成マスクを可視化して保存

    出力画像:
        [ GT(赤青) | Pred(赤青) ] を横並びで1枚にした PNG
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(indices):
            volt, mask_gt, stem = dataset[idx]

            volt_b = volt.unsqueeze(0).to(device)   # (1,2)
            logits = model(volt_b)                  # (1,2,H,W)
            prob = torch.sigmoid(logits)[0].cpu().numpy()  # (2,H,W)

            # 0.5 閾値でバイナリ化
            pred_mask = (prob > 0.5).astype(np.uint8)

            _, H, W = mask_gt.shape
            gt_vis = np.zeros((H, W, 3), dtype=np.uint8)
            pred_vis = np.zeros_like(gt_vis)

            mask_gt_np = mask_gt.numpy()

            # upper ビーム: 赤, lower ビーム: 青 （BGR）
            gt_vis[mask_gt_np[0].astype(bool)] = (0, 0, 255)
            gt_vis[mask_gt_np[1].astype(bool)] = (255, 0, 0)

            pred_vis[pred_mask[0].astype(bool)] = (0, 0, 255)
            pred_vis[pred_mask[1].astype(bool)] = (255, 0, 0)

            concat = np.concatenate([gt_vis, pred_vis], axis=1)

            if epoch is None:
                out_name = f"gen_{i:02d}_{stem}.png"
            else:
                out_name = f"epoch{epoch:03d}_gen_{i:02d}_{stem}.png"

            out_path = out_dir / out_name
            cv2.imwrite(str(out_path), concat)
            print(f"[LOG] generated mask saved: {out_path}")


# =======================================
# Train loop
# =======================================

def train_voltage_to_mask(
    beammask_dir,
    csv_path,
    v_left_col="Left_V",
    v_right_col="Right_V",
    batch_size=32,
    num_epochs=200,
    lr=1e-3,
    val_ratio=0.1,
    device="cuda",
    save_interval=50,
    log_subdir="training_logs_voltage_to_mask"
):
    """
    電圧 -> 2chマスク NN の学習ループ

    save_interval エポックごとに:
      - モデルパラメータ保存
      - loss 曲線保存
      - サンプル生成結果を保存
    """
    # Dataset 準備
    full_ds = VoltageToBeamMaskDataset(
        beammask_dir=beammask_dir,
        csv_path=csv_path,
        v_left_col=v_left_col,
        v_right_col=v_right_col
    )
    out_ch, H, W = full_ds.out_ch, full_ds.H, full_ds.W
    print(f"[INFO] mask shape: (C={out_ch}, H={H}, W={W})")
    print(f"[INFO] total samples: {len(full_ds)}")

    # train / val に分割
    n_total = len(full_ds)
    n_val = max(1, int(n_total * val_ratio))
    n_train = n_total - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])
    print(f"[INFO] n_train={n_train}, n_val={n_val}")

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, num_workers=0)

    # モデル・オプティマイザ・損失
    model = VoltageToMaskNet(out_h=H, out_w=W, latent_dim=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = BCEDiceLoss(bce_weight=1.0, dice_weight=1.0)

    # ログ保存用ディレクトリ
    beammask_dir = Path(beammask_dir)
    log_dir = beammask_dir / log_subdir
    log_dir.mkdir(parents=True, exist_ok=True)

    # 損失ログ
    epochs_log = []
    train_losses = []
    val_losses = []

    # 固定サンプル (フルデータセットから数件ピック)
    np.random.seed(0)
    num_gen_samples = min(5, len(full_ds))
    fixed_indices = np.random.choice(len(full_ds), size=num_gen_samples, replace=False)
    print(f"[INFO] fixed sample indices for generation: {fixed_indices}")

    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        # -------- train --------
        model.train()
        train_loss_sum = 0.0

        for volt, mask_gt, _ in tqdm(train_loader, desc=f"Epoch {epoch} [train]"):
            volt = volt.to(device)          # (B, 2)
            mask_gt = mask_gt.to(device)    # (B, 2, H, W)

            optimizer.zero_grad()
            mask_logits = model(volt)       # (B, 2, H, W)
            loss = criterion(mask_logits, mask_gt)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * volt.size(0)

        train_loss = train_loss_sum / n_train

        # -------- val --------
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for volt, mask_gt, _ in tqdm(val_loader, desc=f"Epoch {epoch} [val]"):
                volt = volt.to(device)
                mask_gt = mask_gt.to(device)
                mask_logits = model(volt)
                loss = criterion(mask_logits, mask_gt)
                val_loss_sum += loss.item() * volt.size(0)

        val_loss = val_loss_sum / n_val

        epochs_log.append(epoch)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        # ベストモデル更新
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = log_dir / "voltage_to_mask_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"  -> best model updated, saved to: {best_path}")

        # ログ CSV は毎エポック更新
        save_loss_csv(log_dir, epochs_log, train_losses, val_losses)

        # save_interval ごとにモデルと可視化を保存
        if (epoch % save_interval == 0) or (epoch == num_epochs):
            # モデルパラメータ
            ckpt_path = log_dir / f"voltage_to_mask_epoch{epoch:03d}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"[LOG] checkpoint saved to: {ckpt_path}")

            # loss 曲線
            save_loss_curve(log_dir, epochs_log, train_losses, val_losses)

            # 生成結果
            gen_out_dir = log_dir / f"gen_epoch{epoch:03d}"
            save_generated_masks(
                model=model,
                dataset=full_ds,
                indices=fixed_indices,
                out_dir=gen_out_dir,
                device=device,
                epoch=epoch
            )

    return model, log_dir


# =======================================
# メイン
# =======================================

if __name__ == "__main__":
    # ★ユーザ指定パス★
    BEAMMASK_DIR = r"C:\Users\akami\Documents\master_thesis_2025\SMR_controller\roi_aug_shift_1000_20_beammask_2ch_upper0\roi_aug_shift_1000_20_beammask_2ch_upper0"
    CSV_PATH     = r"C:\Users\akami\Documents\master_thesis_2025\SMR_controller\aug_shift_1000_20\aug_shift_1000_20\signals_aug.csv"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model, log_dir = train_voltage_to_mask(
        beammask_dir=BEAMMASK_DIR,
        csv_path=CSV_PATH,
        v_left_col="Left_V",
        v_right_col="Right_V",
        batch_size=64,
        num_epochs=5000,
        lr=1e-3,
        val_ratio=0.1,
        device=device,
        save_interval=100,
        log_subdir="training_logs_voltage_to_mask"
    )

    print(f"[INFO] training finished. logs saved in: {log_dir}")
