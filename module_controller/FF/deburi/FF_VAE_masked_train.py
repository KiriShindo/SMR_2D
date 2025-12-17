import os
from pathlib import Path

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
from torchvision.utils import make_grid
from tqdm import tqdm

# ===== パス設定 ======================================
IMG_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\roi_aug_rot_shift_scale_rect2mask_filtered"
CSV_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\1module_dataset_max_DAC\silicon\raw\signals.csv"
MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_vae_masked.pth"
LOG_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\logs_vae_masked"

# ===== ハイパーパラメータ ============================
BATCH_SIZE = 64
NUM_EPOCHS = 5000
LEARNING_RATE = 1e-3
VAL_RATIO = 0.1
SAVE_INTERVAL = 500

LATENT_DIM = 32          # 潜在次元
BETA_KL = 1e-3           # KL 正則化係数
LAMBDA_REG = 1.0         # 電圧回帰 MSE の重み


# ===== Dataset 定義 ==================================
class ModuleImageVoltageDataset(Dataset):
    """
    画像と 2ch 電圧 [v_left, v_right] を返す Dataset
    """
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
        # ファイル名 123_aug_0001.png → 123 を取り出す想定
        return int(path.stem.split('_')[0])

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert("RGB").resize((50, 62))  # (W=50, H=62)
        if self.transform:
            img_tensor = self.transform(img)    # (3, 62, 50)
        else:
            img_tensor = T.ToTensor()(img)      # fallback

        i = self._parse_index_from_filename(img_path)
        row = self.volt_df.iloc[i - 1]
        v_left, v_right = float(row[0]), float(row[6])

        target = torch.tensor([v_left, v_right], dtype=torch.float32)
        return img_tensor, target


# ===== モデル定義 (VAE + 回帰) =======================

class ConvEncoderVAE(nn.Module):
    """
    画像 → 潜在ベクトル z (mu, logvar)
    Global Average Pooling を用いて flatten ではなくチャネル統合。
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # (3,62,50) -> (32,62,50)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> (32,31,25)

            nn.Conv2d(32, 64, 3, padding=1),  # -> (64,31,25)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> (64,15,12)

            nn.Conv2d(64, 128, 3, padding=1), # -> (128,15,12)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),                  # -> (128,7,6)
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # -> (128,1,1)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, x):
        h = self.features(x)               # (B,128,H',W')
        h = self.gap(h)                    # (B,128,1,1)
        h = h.view(h.size(0), -1)          # (B,128)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class ConvDecoderVAE(nn.Module):
    """
    潜在ベクトル z → 画像再構成
    入力サイズ (3, 62, 50) を再現するように設計。
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 128 * 7 * 6)

        self.deconv = nn.Sequential(
            # (128,7,6)
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # -> (128,14,12)
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),   # -> (64,28,24)
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # 最終的に (3,62,50) に合わせてアップサンプル
            nn.Upsample(size=(62, 50), mode="bilinear", align_corners=False),    # -> (32,62,50)
            nn.Conv2d(32, 3, 3, padding=1),
            nn.Sigmoid(),  # 出力を [0,1] に
        )

    def forward(self, z):
        h = self.fc(z)                     # (B,128*7*6)
        h = h.view(-1, 128, 7, 6)          # (B,128,7,6)
        x_recon = self.deconv(h)           # (B,3,62,50)
        return x_recon


class VoltageMLP(nn.Module):
    """
    潜在ベクトル z → 電圧 [V_left, V_right]
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, z):
        return self.net(z)


class VAEControllerNet(nn.Module):
    """
    VAE Encoder + Decoder + 回帰 MLP をまとめたモデル
    """
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = ConvEncoderVAE(latent_dim)
        self.decoder = ConvDecoderVAE(latent_dim)
        self.reg_head = VoltageMLP(latent_dim)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        x: 入力画像 (B,3,62,50) [0,1]
        戻り値:
            x_recon: 再構成画像
            mu, logvar: VAE の潜在分布パラメータ
            pred: 電圧予測 [B,2]
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        pred = self.reg_head(z)
        return x_recon, mu, logvar, pred


# ===== 損失関数 (VAE + 回帰) =========================
def vae_regression_loss(x, x_recon, mu, logvar, pred, target,
                        beta_kl=BETA_KL, lambda_reg=LAMBDA_REG):
    """
    x        : 入力画像 (B,3,62,50)
    x_recon  : 再構成画像
    mu,logvar: VAE latent
    pred     : 予測電圧 (B,2)
    target   : 真値電圧 (B,2)
    """
    # 再構成損失 (画像空間でのMSE)
    rec_loss = F.mse_loss(x_recon, x, reduction='mean')

    # KL損失
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # 電圧回帰 MSE
    reg_loss = F.mse_loss(pred, target, reduction='mean')

    total_loss = rec_loss + beta_kl * kl_loss + lambda_reg * reg_loss
    loss_dict = {
        "total": total_loss.item(),
        "rec": rec_loss.item(),
        "kl": kl_loss.item(),
        "reg": reg_loss.item(),
    }
    return total_loss, loss_dict


# ===== 再構成画像のグリッド保存 ======================
def save_reconstruction_grid(model, val_loader, device, epoch, log_dir, num_samples=8):
    model.eval()
    with torch.no_grad():
        # val_loader から最初のバッチを一つ取る
        try:
            imgs, _ = next(iter(val_loader))
        except StopIteration:
            return
        imgs = imgs.to(device)
        x_recon, _, _, _ = model(imgs)

    # 先頭 num_samples 枚
    n = min(num_samples, imgs.size(0))
    orig = imgs[:n]       # (n,3,62,50)
    recon = x_recon[:n]   # (n,3,62,50)

    # オリジナルと再構成を縦に並べる (2行)
    # [orig0..origN-1, recon0..reconN-1] を1枚のグリッドに
    grid = make_grid(
        torch.cat([orig, recon], dim=0),
        nrow=n,  # 1行あたり n 枚 → 上段:元, 下段:再構成
        padding=2
    )

    # [C,H,W] → [H,W,C] に変換して保存
    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(n * 2, 4))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title(f"Reconstruction (top: original, bottom: recon) @ epoch {epoch}")
    plt.tight_layout()
    out_path = os.path.join(log_dir, f"reconstruction_grid_epoch{epoch:04d}.png")
    plt.savefig(out_path)
    plt.close()


# ===== メイン学習ループ ==============================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    os.makedirs(LOG_DIR, exist_ok=True)

    # VAE なので、ここではあえて Normalize は入れず、[0,1] のままにしている。
    transform = T.Compose([
        T.ToTensor(),  # 0〜1
    ])

    dataset = ModuleImageVoltageDataset(IMG_DIR, CSV_PATH, transform)
    val_size = int(len(dataset) * VAL_RATIO)
    train_dataset, val_dataset = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = VAEControllerNet(latent_dim=LATENT_DIM).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)

    best_val_reg = float("inf")  # 電圧回帰の val MSE を指標にする
    history = {
        "epoch": [],
        "train_total": [],
        "train_rec": [],
        "train_kl": [],
        "train_reg": [],
        "val_total": [],
        "val_rec": [],
        "val_kl": [],
        "val_reg": [],
    }

    for epoch in tqdm(range(1, NUM_EPOCHS + 1), desc="Overall Progress"):
        # ---- train ----
        model.train()
        train_loss_sum = {"total": 0.0, "rec": 0.0, "kl": 0.0, "reg": 0.0}
        n_train = 0

        with tqdm(train_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}] Train", leave=False) as pbar:
            for imgs, targets in pbar:
                imgs = imgs.to(device)
                targets = targets.to(device)

                optimizer.zero_grad()
                x_recon, mu, logvar, pred = model(imgs)
                loss, loss_dict = vae_regression_loss(imgs, x_recon, mu, logvar, pred, targets)
                loss.backward()
                optimizer.step()

                bs = imgs.size(0)
                n_train += bs
                for k in train_loss_sum.keys():
                    train_loss_sum[k] += loss_dict[k] * bs

                pbar.set_postfix(
                    total=f"{loss_dict['total']:.4f}",
                    rec=f"{loss_dict['rec']:.4f}",
                    kl=f"{loss_dict['kl']:.4f}",
                    reg=f"{loss_dict['reg']:.4f}",
                )

        for k in train_loss_sum.keys():
            train_loss_sum[k] /= n_train

        # ---- validation ----
        model.eval()
        val_loss_sum = {"total": 0.0, "rec": 0.0, "kl": 0.0, "reg": 0.0}
        n_val = 0

        with torch.no_grad():
            with tqdm(val_loader, desc=f"[Epoch {epoch}/{NUM_EPOCHS}] Val", leave=False) as pbar:
                for imgs, targets in pbar:
                    imgs = imgs.to(device)
                    targets = targets.to(device)

                    x_recon, mu, logvar, pred = model(imgs)
                    loss, loss_dict = vae_regression_loss(imgs, x_recon, mu, logvar, pred, targets)

                    bs = imgs.size(0)
                    n_val += bs
                    for k in val_loss_sum.keys():
                        val_loss_sum[k] += loss_dict[k] * bs

                    pbar.set_postfix(
                        total=f"{loss_dict['total']:.4f}",
                        rec=f"{loss_dict['rec']:.4f}",
                        kl=f"{loss_dict['kl']:.4f}",
                        reg=f"{loss_dict['reg']:.4f}",
                    )

        for k in val_loss_sum.keys():
            val_loss_sum[k] /= n_val

        tqdm.write(
            f"[Epoch {epoch:04d}] "
            f"Train(total={train_loss_sum['total']:.4f}, rec={train_loss_sum['rec']:.4f}, "
            f"kl={train_loss_sum['kl']:.4f}, reg={train_loss_sum['reg']:.4f}) | "
            f"Val(total={val_loss_sum['total']:.4f}, rec={val_loss_sum['rec']:.4f}, "
            f"kl={val_loss_sum['kl']:.4f}, reg={val_loss_sum['reg']:.4f})"
        )

        # 履歴に保存
        history["epoch"].append(epoch)
        history["train_total"].append(train_loss_sum["total"])
        history["train_rec"].append(train_loss_sum["rec"])
        history["train_kl"].append(train_loss_sum["kl"])
        history["train_reg"].append(train_loss_sum["reg"])
        history["val_total"].append(val_loss_sum["total"])
        history["val_rec"].append(val_loss_sum["rec"])
        history["val_kl"].append(val_loss_sum["kl"])
        history["val_reg"].append(val_loss_sum["reg"])

        # ---- checkpoint 保存 & ログ可視化 ----
        if epoch % SAVE_INTERVAL == 0 or epoch == NUM_EPOCHS:
            # checkpoint
            ckpt_path = os.path.join(LOG_DIR, f"vae_checkpoint_epoch{epoch:04d}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            }, ckpt_path)
            tqdm.write(f"  -> Saved checkpoint: {ckpt_path}")

            # CSV で全履歴を保存
            df_hist = pd.DataFrame(history)
            df_hist.to_csv(os.path.join(LOG_DIR, "loss_history.csv"), index=False)

            # 回帰損失のみのカーブ
            plt.figure()
            plt.plot(history["epoch"], history["train_reg"], label="Train Reg MSE")
            plt.plot(history["epoch"], history["val_reg"], label="Val Reg MSE")
            plt.xlabel("Epoch")
            plt.ylabel("Regression MSE")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, f"reg_loss_curve_epoch{epoch:04d}.png"))
            plt.close()

            # 全損失項目の学習曲線
            plt.figure(figsize=(10, 6))
            # total
            plt.plot(history["epoch"], history["train_total"], label="Train Total")
            plt.plot(history["epoch"], history["val_total"], label="Val Total", linestyle="--")
            # rec
            plt.plot(history["epoch"], history["train_rec"], label="Train Recon")
            plt.plot(history["epoch"], history["val_rec"], label="Val Recon", linestyle="--")
            # kl
            plt.plot(history["epoch"], history["train_kl"], label="Train KL")
            plt.plot(history["epoch"], history["val_kl"], label="Val KL", linestyle="--")
            # reg
            plt.plot(history["epoch"], history["train_reg"], label="Train Reg")
            plt.plot(history["epoch"], history["val_reg"], label="Val Reg", linestyle="--")

            plt.yscale("log")  # 桁が違うので log にしておく
            plt.xlabel("Epoch")
            plt.ylabel("Loss (log scale)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(LOG_DIR, f"all_loss_curve_epoch{epoch:04d}.png"))
            plt.close()

            # 再構成画像のグリッド保存
            save_reconstruction_grid(model, val_loader, device, epoch, LOG_DIR, num_samples=8)

        # ベストモデル更新（Val の回帰 MSE ベース）
        if val_loss_sum["reg"] < best_val_reg:
            best_val_reg = val_loss_sum["reg"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "best_val_reg": best_val_reg,
                "history": history,
            }, MODEL_SAVE_PATH)
            tqdm.write(f"  -> Best VAEController model updated (val_reg={best_val_reg:.6f}).")

    # 最終的な履歴も CSV に保存（保険）
    df_hist = pd.DataFrame(history)
    df_hist.to_csv(os.path.join(LOG_DIR, "loss_history_final.csv"), index=False)

    print("Training finished.")


if __name__ == "__main__":
    main()
