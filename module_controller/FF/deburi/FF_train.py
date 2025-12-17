# -*- coding: utf-8 -*-
"""
train_module_controller.py (with tqdm progress bar + checkpoint logging)
"""

import os
from pathlib import Path
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import matplotlib.pyplot as plt
from tqdm import tqdm  # ← 追加

# ===== パス設定 ======================================
IMG_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi_aug_rot_shift_scale"
CSV_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\raw\signals.csv"
MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller.pth"
LOG_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\logs"

# ===== ハイパーパラメータ ============================
BATCH_SIZE = 64
NUM_EPOCHS = 3000
LEARNING_RATE = 1e-3
VAL_RATIO = 0.1
SAVE_INTERVAL = 100


# ===== Dataset 定義 ==================================
class ModuleImageVoltageDataset(Dataset):
    def __init__(self, img_dir: str, csv_path: str, transform=None):
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.volt_df = pd.read_csv(csv_path, header=None)
        self.image_paths = sorted(self.img_dir.glob("*.png"))
        if len(self.image_paths) == 0:
            raise RuntimeError(f"No .png files found in {self.img_dir}")

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def _parse_index_from_filename(path: Path) -> int:
        return int(path.stem.split('_')[0])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB").resize((50, 62))
        if self.transform:
            img = self.transform(img)
        i = self._parse_index_from_filename(img_path)
        row = self.volt_df.iloc[i - 1]
        v_left, v_right = float(row[0]), float(row[6])
        return img, torch.tensor([v_left, v_right], dtype=torch.float32)


# ===== モデル定義 ====================================
class ModuleImageEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Linear(128 * 7 * 6, out_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class VoltageMLP(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x): return self.net(x)


class ModuleControllerNet(nn.Module):
    def __init__(self, feat_dim=128):
        super().__init__()
        self.encoder = ModuleImageEncoder(feat_dim)
        self.mlp = VoltageMLP(feat_dim)

    def forward(self, img):
        feat = self.encoder(img)
        return self.mlp(feat)


# ===== 学習関数 =====================================
def train_one_epoch(model, loader, criterion, optimizer, device, epoch, num_epochs):
    model.train()
    total_loss = 0
    with tqdm(loader, desc=f"[Epoch {epoch}/{num_epochs}] Training", leave=False) as pbar:
        for imgs, targets in pbar:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


def eval_one_epoch(model, loader, criterion, device, epoch, num_epochs):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        with tqdm(loader, desc=f"[Epoch {epoch}/{num_epochs}] Validation", leave=False) as pbar:
            for imgs, targets in pbar:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                total_loss += loss.item() * imgs.size(0)
                pbar.set_postfix(loss=f"{loss.item():.6f}")
    return total_loss / len(loader.dataset)


# ===== メイン =======================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(LOG_DIR, exist_ok=True)

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    dataset = ModuleImageVoltageDataset(IMG_DIR, CSV_PATH, transform)
    val_size = int(len(dataset) * VAL_RATIO)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    model = ModuleControllerNet(feat_dim=128).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_loss = float("inf")
    train_losses, val_losses = [], []

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Overall Progress"):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, NUM_EPOCHS)
        val_loss = eval_one_epoch(model, val_loader, criterion, device, epoch, NUM_EPOCHS)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        tqdm.write(f"[Epoch {epoch:04d}] Train: {train_loss:.6f} | Val: {val_loss:.6f}")

        # --- 定期保存 ---
        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            ckpt_path = os.path.join(LOG_DIR, f"checkpoint_epoch{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            }, ckpt_path)
            tqdm.write(f"  -> Saved checkpoint: {ckpt_path}")

            # 学習曲線保存
            plt.figure()
            plt.plot(train_losses, label="Train Loss")
            plt.plot(val_losses, label="Val Loss")
            plt.xlabel("Epoch")
            plt.ylabel("MSE Loss")
            plt.legend()
            plt.title(f"Loss Curve up to Epoch {epoch}")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, f"loss_curve_epoch{epoch:04d}.png"))
            plt.close()

            pd.DataFrame({"epoch": list(range(1, epoch + 1)),
                          "train_loss": train_losses,
                          "val_loss": val_losses}).to_csv(
                os.path.join(LOG_DIR, "loss_history.csv"), index=False
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict(),
                        "best_val_loss": best_val_loss}, MODEL_SAVE_PATH)
            tqdm.write(f"  -> Best model updated (val={val_loss:.6f}).")

    print("Training finished.")


if __name__ == "__main__":
    main()
