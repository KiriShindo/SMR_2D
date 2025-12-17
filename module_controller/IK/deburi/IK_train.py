# -*- coding: utf-8 -*-
"""
train_ik_beam2ch.py
---------------------------------------------------------
2chビームマスクを使った「画像版IKネットワーク」の学習スクリプト

入力 x:
  - 時刻 i の 2chビームマスク: mask_i (2, H, W)
  - 時刻 i+1 の 2chビームマスク: mask_ip1 (2, H, W)
  - 時刻 i の電圧 q_i = (V_L^i, V_R^i) を 2ch一様マップにしたもの: q_map (2, H, W)
  → x = concat([mask_i, mask_ip1, q_map], dim=0) なので (6, H, W)

出力 y:
  - 時刻 i+1 の電圧 (V_L^{i+1}, V_R^{i+1}) [0〜5V]

ペア構成:
  - signals_aug.csv を orig_index ごとにグルーピング
  - 隣り合う orig_index (k, k+1) のそれぞれの行群から、
    すべての組み合わせ (row ∈ group_k, row' ∈ group_{k+1}) をサンプルにする
  - その両方に対応する *_beam2ch.npy が存在する場合のみサンプルとして採用
---------------------------------------------------------
"""

import os
import csv
from pathlib import Path
from collections import defaultdict  # ★追加

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ==========================================================
# パス設定
# ==========================================================

# 2chビームマスク (*.npy) が入っているディレクトリ
BEAMMASK_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\roi_aug_shift_500_10_beammask_2ch_upper0"
)

# バブリング拡張時に出力した signals_aug.csv
CSV_AUG_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\aug_shift_500_10\signals_aug.csv"
)

# モデル保存先・ログ保存先
MODEL_SAVE_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\ik_beam2ch_babbling_500_10_shift_upper0_model.pth"
)
LOG_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\training_logs_beam2ch_babbling_500_10_shift_upper0"
)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================================
# ハイパーパラメータ
# ==========================================================
BATCH_SIZE   = 64
EPOCHS       = 5000
LR           = 1e-3
VAL_RATIO    = 0.1
WEIGHT_DECAY = 1e-4
SAVE_INTERVAL = 100

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)


# ==========================================================
# Dataset 定義
# ==========================================================

class IKBeamDataset(Dataset):
    """
    - signals_aug.csv を読み込み、
      orig_index ごとに行をグルーピング。
    - 隣り合う orig_index (k, k+1) について、
      group_k の各行と group_{k+1} の各行の
      全組み合わせをサンプルとして採用。
    - 各行は filename, orig_index, Left_V, Right_V を持っている想定。
    - filename の stem から *_beam2ch.npy を探して 2chマスクをロードする。
    - 入力 x: concat([mask_i (2ch), mask_ip1 (2ch), q_map_i (2ch)]) → (6, H, W)
    - 出力 y: q_ip1 = (Left_V^{i+1}, Right_V^{i+1}) [生の 0〜5V]
    """
    def __init__(self, beammask_dir: Path, csv_aug_path: Path):
        super().__init__()
        self.beammask_dir = Path(beammask_dir)

        # CSV 読み込み
        with open(csv_aug_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if len(rows) < 2:
            raise RuntimeError("signals_aug.csv の行数が足りません。")

        # --- orig_index ごとにグルーピング ---
        groups = defaultdict(list)  # orig_index -> [row, row, ...]
        for row in rows:
            try:
                orig = int(row["orig_index"])
            except Exception:
                continue
            groups[orig].append(row)

        samples = []

        # 隣り合う orig_index (k, k+1) のペアごとに全組み合わせを作る
        sorted_indices = sorted(groups.keys())
        for k in sorted_indices:
            k_next = k + 1
            if k_next not in groups:
                continue

            rows_k     = groups[k]
            rows_knext = groups[k_next]

            # 全組み合わせ（必要ならここにランダムサンプリングを挟める）
            for row_i in rows_k:
                for row_ip1 in rows_knext:
                    fname_i   = row_i["filename"]   # 例: "57_aug_0000.png"
                    fname_ip1 = row_ip1["filename"] # 例: "58_aug_0012.png"

                    stem_i   = os.path.splitext(fname_i)[0]
                    stem_ip1 = os.path.splitext(fname_ip1)[0]

                    npy_i   = self.beammask_dir / f"{stem_i}_beam2ch.npy"
                    npy_ip1 = self.beammask_dir / f"{stem_ip1}_beam2ch.npy"

                    # マスクが両方とも存在するペアだけ残す
                    if not npy_i.exists() or not npy_ip1.exists():
                        continue

                    try:
                        q_i    = (float(row_i["Left_V"]),    float(row_i["Right_V"]))
                        q_ip1  = (float(row_ip1["Left_V"]),  float(row_ip1["Right_V"]))
                    except Exception:
                        continue

                    samples.append({
                        "npy_i":   npy_i,
                        "npy_ip1": npy_ip1,
                        "q_i":     q_i,
                        "q_ip1":   q_ip1,
                    })

        if len(samples) == 0:
            raise RuntimeError(
                "有効な IK ペアが見つかりませんでした。"
                "orig_index ごとのグループや *_beam2ch.npy のパスを確認してください。"
            )

        self.samples = samples
        print(f"[INFO] 有効な IK ペア数（全組み合わせ）: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # 2chマスクをロード (2, H, W)
        mask_i   = np.load(str(item["npy_i"])).astype(np.float32)
        mask_ip1 = np.load(str(item["npy_ip1"])).astype(np.float32)

        # torch tensor に変換
        mask_i   = torch.from_numpy(mask_i)    # (2, H, W)
        mask_ip1 = torch.from_numpy(mask_ip1)  # (2, H, W)

        # q_i を 2ch の一様マップに展開 (V/5.0 に正規化)
        vL_i, vR_i = item["q_i"]
        _, H, W = mask_i.shape
        q_map = torch.zeros((2, H, W), dtype=torch.float32)
        q_map[0, :, :] = vL_i / 5.0
        q_map[1, :, :] = vR_i / 5.0

        # 入力 x: (6, H, W)
        x = torch.cat([mask_i, mask_ip1, q_map], dim=0)

        # 出力 y: 時刻 i+1 の電圧 (生の 0〜5V)
        vL_ip1, vR_ip1 = item["q_ip1"]
        y = torch.tensor([vL_ip1, vR_ip1], dtype=torch.float32)

        return x, y


# ==========================================================
# モデル定義
# ==========================================================

class IKBeamEncoder(nn.Module):
    """
    6ch入力 (mask_i 2ch + mask_ip1 2ch + q_map 2ch) を受け取る CNN encoder
    """
    def __init__(self, in_ch: int = 6, feat_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
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
        self.fc  = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 128, H', W')
        x = self.gap(x)               # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)     # (B, 128)
        x = self.fc(x)                # (B, feat_dim)
        return x


class VoltageHead(nn.Module):
    def __init__(self, in_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),

            nn.Linear(64, 2),  # 出力: [V_L^{i+1}, V_R^{i+1}]
        )

    def forward(self, x):
        return self.net(x)


class IKBeamNet(nn.Module):
    """
    全体モデル: Encoder(6chマスク+q_map) → MLP → 2次元電圧
    """
    def __init__(self, in_ch: int = 6, feat_dim: int = 128):
        super().__init__()
        self.encoder = IKBeamEncoder(in_ch=in_ch, feat_dim=feat_dim)
        self.head    = VoltageHead(in_dim=feat_dim)

    def forward(self, x):
        feat = self.encoder(x)
        out  = self.head(feat)
        return out


# ==========================================================
# 学習・評価ループ
# ==========================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch, max_epoch):
    model.train()
    total_loss = 0.0
    with tqdm(loader, desc=f"[Epoch {epoch}/{max_epoch}] Train", leave=False) as pbar:
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device, epoch, max_epoch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        with tqdm(loader, desc=f"[Epoch {epoch}/{max_epoch}] Val", leave=False) as pbar:
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)
                y_hat = model(x)
                loss = criterion(y_hat, y)
                total_loss += loss.item() * x.size(0)
                pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def save_curve(train_losses, val_losses, epoch):
    plt.figure()
    plt.plot(train_losses, label="Train")
    plt.plot(val_losses,   label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("SmoothL1 Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    out_path = LOG_DIR / f"loss_curve_epoch{epoch:04d}.png"
    plt.savefig(out_path)
    plt.close()


# ==========================================================
# メイン
# ==========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Dataset 構築
    dataset = IKBeamDataset(BEAMMASK_DIR, CSV_AUG_PATH)
    val_size = int(len(dataset) * VAL_RATIO)
    train_size = len(dataset) - val_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0, pin_memory=True
    )

    # モデル・オプティマイザなど
    model = IKBeamNet(in_ch=6, feat_dim=128).to(device)
    criterion = nn.SmoothL1Loss(beta=0.1)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=20
    )

    best_val = float("inf")
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, epoch, EPOCHS)
        val_loss   = eval_epoch(model, val_loader,   criterion, device, epoch, EPOCHS)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"  Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        scheduler.step(val_loss)

        # ベストモデル更新
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_loss": best_val,
            }, MODEL_SAVE_PATH)
            print(f"  → Best model updated: {best_val:.6f}")

        # スナップショット & 学習曲線保存
        if epoch % SAVE_INTERVAL == 0 or epoch == EPOCHS:
            ckpt_path = LOG_DIR / f"ik_beam2ch_epoch{epoch:04d}.pth"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, ckpt_path)
            print(f"  → Saved checkpoint: {ckpt_path}")
            save_curve(train_losses, val_losses, epoch)

    print("\nTraining finished.")
    print(f"Best Val Loss = {best_val:.6f}")
    print(f"Final best model saved to: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
