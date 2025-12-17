import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.utils import make_grid
from sklearn.decomposition import PCA

# === 学習スクリプトからクラスとパスをインポート ==========================
from FF_VAE_masked_train import (
    ModuleImageVoltageDataset,
    VAEControllerNet,
    IMG_DIR,
    CSV_PATH,
    MODEL_SAVE_PATH,
    LATENT_DIM,  # 潜在次元
)

# 評価結果の保存先
EVAL_DIR = Path(os.path.dirname(MODEL_SAVE_PATH)) / "eval_vae_masked"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ---- データセット・ローダ --------------------------
    transform = T.Compose([
        T.ToTensor(),  # VAE学習時と同じく [0,1]
    ])

    dataset = ModuleImageVoltageDataset(IMG_DIR, CSV_PATH, transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # ---- モデル読み込み -------------------------------
    model = VAEControllerNet(latent_dim=LATENT_DIM).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ---- 推論して全サンプルの予測/正解 & 潜在 mu, z をためる ----------
    all_targets = []
    all_preds = []
    all_mu = []   # 潜在平均ベクトル μ
    all_z = []    # 実際にサンプリングされた潜在表現 z

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            targets = targets.to(device)

            # VAEControllerNet: x_recon, mu, logvar, pred
            x_recon, mu, logvar, outputs = model(imgs)

            # mu, logvar から実際の潜在サンプル z を生成
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # shape: (B, LATENT_DIM)

            all_targets.append(targets.cpu().numpy())
            all_preds.append(outputs.cpu().numpy())
            all_mu.append(mu.cpu().numpy())
            all_z.append(z.cpu().numpy())

    all_targets = np.concatenate(all_targets, axis=0)  # (N, 2)
    all_preds = np.concatenate(all_preds, axis=0)      # (N, 2)
    all_mu = np.concatenate(all_mu, axis=0)            # (N, LATENT_DIM)
    all_z = np.concatenate(all_z, axis=0)              # (N, LATENT_DIM)

    # ---- 指標計算 --------------------------------------
    mse_total = np.mean((all_preds - all_targets) ** 2)
    mae_total = np.mean(np.abs(all_preds - all_targets))

    mse_left = np.mean((all_preds[:, 0] - all_targets[:, 0]) ** 2)
    mae_left = np.mean(np.abs(all_preds[:, 0] - all_targets[:, 0]))

    mse_right = np.mean((all_preds[:, 1] - all_targets[:, 1]) ** 2)
    mae_right = np.mean(np.abs(all_preds[:, 1] - all_targets[:, 1]))

    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1.0 - ss_res / ss_tot

    r2_left = r2_score(all_targets[:, 0], all_preds[:, 0])
    r2_right = r2_score(all_targets[:, 1], all_preds[:, 1])

    print("===== Evaluation Result (VAEControllerNet) =====")
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

    # ---- 再構成画像のグリッドを保存 -------------------
    model.eval()
    with torch.no_grad():
        imgs_vis, _ = next(iter(loader))
        imgs_vis = imgs_vis.to(device)
        x_recon, _, _, _ = model(imgs_vis)

    num_samples = 8
    n = min(num_samples, imgs_vis.size(0))
    orig = imgs_vis[:n]       # (n,3,H,W)
    recon = x_recon[:n]       # (n,3,H,W)

    grid = make_grid(
        torch.cat([orig, recon], dim=0),
        nrow=n,  # 上段: orig, 下段: recon
        padding=2
    )

    grid_np = grid.cpu().permute(1, 2, 0).numpy()
    plt.figure(figsize=(n * 2, 4))
    plt.imshow(grid_np)
    plt.axis("off")
    plt.title("Reconstruction (top: original, bottom: recon)")
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "reconstruction_grid.png")
    plt.close()

    # ---- 潜在空間 (mu) の分布を可視化 -----------------
    D = all_mu.shape[1]

    # 1) 各次元のヒストグラム
    n_cols = 4
    n_rows = int(np.ceil(D / n_cols))

    plt.figure(figsize=(3 * n_cols, 2.5 * n_rows))
    for d in range(D):
        ax = plt.subplot(n_rows, n_cols, d + 1)
        ax.hist(all_mu[:, d], bins=40, alpha=0.8)
        ax.set_title(f"mu[{d}]")
        ax.grid(True)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / "latent_mu_hist_all_dims.png")
    plt.close()

    # 2) 最初の2次元だけの散布図 (mu[0], mu[1])
    if D >= 2:
        plt.figure()
        plt.scatter(all_mu[:, 0], all_mu[:, 1], s=5, alpha=0.5)
        plt.xlabel("mu[0]")
        plt.ylabel("mu[1]")
        plt.title("Latent Space (mu[0] vs mu[1])")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / "latent_mu_scatter_dim01.png")
        plt.close()

    # 3) PCA による 2 次元射影 (mu)
    try:
        pca_mu = PCA(n_components=2)
        mu_2d = pca_mu.fit_transform(all_mu)  # (N, 2)

        plt.figure(figsize=(6, 5))
        plt.scatter(mu_2d[:, 0], mu_2d[:, 1], s=5, alpha=0.6)
        plt.title("Latent Space (mu, PCA 2D projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / "latent_mu_pca2d.png")
        plt.close()

        explained_mu = pca_mu.explained_variance_ratio_
        pd.DataFrame({
            "PC": [1, 2],
            "explained_variance_ratio": explained_mu,
        }).to_csv(EVAL_DIR / "latent_mu_pca_explained_variance.csv", index=False)
    except Exception as e:
        print(f"[WARN] PCA(mu) failed: {e}")

    # ---- 潜在サンプル z の PCA 2次元射影 ----------------
    try:
        pca_z = PCA(n_components=2)
        z_2d = pca_z.fit_transform(all_z)  # (N, 2)

        plt.figure(figsize=(6, 5))
        plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6, color="tab:blue")
        plt.title("Latent Space (sampled z, PCA 2D projection)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(EVAL_DIR / "latent_z_pca2d.png")
        plt.close()

        explained_z = pca_z.explained_variance_ratio_
        pd.DataFrame({
            "PC": [1, 2],
            "explained_variance_ratio": explained_z,
        }).to_csv(EVAL_DIR / "latent_z_pca_explained_variance.csv", index=False)
    except Exception as e:
        print(f"[WARN] PCA(z) failed: {e}")

    # ---- 生の潜在ベクトルも保存しておく -----------------
    np.save(EVAL_DIR / "latent_mu.npy", all_mu)
    np.save(EVAL_DIR / "latent_z.npy", all_z)

    print(f"Evaluation results (including latent plots & PCA) saved in: {EVAL_DIR}")


if __name__ == "__main__":
    evaluate()
