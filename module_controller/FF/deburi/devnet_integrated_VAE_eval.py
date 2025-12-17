# -*- coding: utf-8 -*-
"""
eval_vae_with_devnet_integrated.py

- 分割ネット (detectron2) で出力した center_crop 画像を
  VAEControllerNet に入れて電圧を予測
- 各元画像ごとに:
    - 元のroi画像
    - 分割可視化画像
    - center_crop画像
    - 再構成結果 (center_crop の recon グリッド)
    - モジュールごとの電圧誤差バー
- 全 center_crop の潜在 z を集めて、
  FF_only の z（latent_z.npy）から fit した PCA の上に
  強制3枚分の z（各画像でモジュール平均）を重ねてプロット
"""

import os
import random
import colorsys

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T
from torchvision.utils import make_grid
from sklearn.decomposition import PCA

from pathlib import Path

# detectron2 関係
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ==== VAE 学習コードから import（ファイル名は合わせて）====
from FF_VAE_train import (
    VAEControllerNet,
    LATENT_DIM,
)

# ===== パス設定 ======================================
BASE_DEVNET_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data"

# 1〜6 module 用 signals.csv & roi は
#   BASE_DEVNET_DIR / f"{i}module_dataset_max_DAC" / "roi"
#   BASE_DEVNET_DIR / f"{i}module_dataset_max_DAC" / "signals.csv"
# にある前提

# FF_only 側で保存した VAE 潜在 (★ z を読むように変更)
FF_ONLY_LATENT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_vae\FF_only"
FF_ONLY_LATENT_PATH = os.path.join(FF_ONLY_LATENT_DIR, "latent_z.npy")  # <- ここを z に

# VAE モデルの重み
MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_vae.pth"

# 今回の devnet 統合 VAE 評価結果
RESULT_ROOT = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_vae\devnet_integrated"
os.makedirs(RESULT_ROOT, exist_ok=True)

# detectron2 の学習済みモデル設定
TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
DEVNET_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01/model_final.pth"

# ==== 背景ノイズ用統計（必要なら更新）====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0  # distanceTransform のスケール


# ===== ノイズ生成関数 =====
def sample_border_noise(H, W):
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


# ===== カラー生成 =====
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append((r, g, b))
    return colors


# ===== ランダム評価画像抽出 =====
def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
    """
    1〜n_folders の各 module_dataset からランダムに roi 画像を拾う。
    戻り値: [(roi_img_path, module_idx), ...] を n_samples 件
    """
    random_entries = []
    for _ in range(n_samples):
        i = random.randint(1, n_folders)
        roi_dir = os.path.join(base_dir, f"{i}module_dataset_max_DAC", "roi")
        if not os.path.exists(roi_dir):
            continue
        files = [f for f in os.listdir(roi_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if not files:
            continue
        chosen = random.choice(files)
        img_path = os.path.join(roi_dir, chosen)
        random_entries.append((img_path, i))
    return random_entries


# ===== center_crop + ノイズ補完 & VAE推論 =====
def crop_and_predict_with_vae(
    img_bgr,
    boxes,
    module_idx,
    row_idx,
    signals_df,
    vae_model,
    device,
    transform,
    save_dir,
    crop_h=62,
    crop_w=50,
):
    """
    - detectron2 の bbox をもとに center_crop + ノイズ補完
    - 各モジュールについて:
        - center_crop画像を保存
        - VAEで再構成 & 電圧予測
        - 真値電圧を signals_df から取得
    - 再構成グリッド & 誤差バーを保存
    戻り値:
        all_mu: (num_modules, LATENT_DIM) の numpy
        all_z : (num_modules, LATENT_DIM) の numpy（実サンプル）
    """
    os.makedirs(save_dir, exist_ok=True)
    H, W, _ = img_bgr.shape

    # y1 が小さい順にソート（上から順にモジュールとみなす）
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    half_h = crop_h // 2
    half_w = crop_w // 2

    num_modules = len(boxes_sorted)

    # 真値取得用：row_idx は roi画像の j に対応
    row = signals_df.iloc[row_idx - 1]  # 0-index に直す

    # ログ用
    abs_errors_left = []
    abs_errors_right = []

    # VAE の再構成ログ用
    orig_tensors = []
    recon_tensors = []
    mus_list = []
    zs_list = []   # z をためる

    # center_crop画像を順に処理
    for n, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        x1c, x2c = cx - half_w, cx + half_w
        y1c, y2c = cy - half_h, cy + half_h

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

        x1_src, y1_src = max(0, x1c), max(0, y1c)
        x2_src, y2_src = min(W, x2c), min(H, y2c)

        x1_dst = x1_src - x1c
        y1_dst = y1_src - y1c
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)

        if x2_src > x1_src and y2_src > y1_src:
            cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img_bgr[y1_src:y2_src, x1_src:x2_src]
            mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

        # ノイズ + distance transform で補完
        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]

        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 保存 (BGRのまま)
        center_crop_path = os.path.join(save_dir, f"center_crop_{n:02d}.png")
        cv2.imwrite(center_crop_path, blended)

        # ---- VAE へ入力（RGBにして Tensor 化） ----
        img_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        pil_img = T.functional.to_pil_image(img_rgb)
        pil_img = pil_img.resize((50, 62))  # (W=50, H=62) に揃える
        tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,62,50)

        with torch.no_grad():
            x_recon, mu, logvar, pred = vae_model(tensor)

            # mu, logvar から潜在サンプル z を生成
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # (1, LATENT_DIM)

        pred = pred.cpu().numpy()[0]       # [V_left, V_right]
        mu_np = mu.cpu().numpy()[0]        # (LATENT_DIM,)
        z_np  = z.cpu().numpy()[0]         # (LATENT_DIM,)
        mus_list.append(mu_np)
        zs_list.append(z_np)

        # 再構成画像ログ用
        orig_tensors.append(tensor.cpu()[0])
        recon_tensors.append(x_recon.cpu()[0])

        # ---- 真値電圧（signals.csv） ----
        # A-L を 0-11 に対応させる。n番目モジュール:
        # 左: n列目, 右: n+6列目
        v_left_true = float(row[n])
        v_right_true = float(row[n + 6])

        v_left_pred, v_right_pred = float(pred[0]), float(pred[1])

        abs_errors_left.append(abs(v_left_pred - v_left_true))
        abs_errors_right.append(abs(v_right_pred - v_right_true))

    # ---- 再構成グリッド保存 ----
    if len(orig_tensors) > 0:
        orig_batch = torch.stack(orig_tensors, dim=0)   # (M,3,H,W)
        recon_batch = torch.stack(recon_tensors, dim=0) # (M,3,H,W)

        # 上段: orig, 下段: recon
        grid = make_grid(
            torch.cat([orig_batch, recon_batch], dim=0),
            nrow=len(orig_tensors),
            padding=2
        )
        grid_np = grid.permute(1, 2, 0).numpy()
        plt.figure(figsize=(len(orig_tensors) * 2, 4))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title("Reconstruction (top: original, bottom: recon)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "reconstruction_grid.png"))
        plt.close()

    # ---- 電圧誤差のバーグラフ ----
    # モジュールごとに左・右の誤差を並べて表示
    modules = np.arange(num_modules)

    width = 0.35
    plt.figure(figsize=(6, 4))
    plt.bar(modules - width/2, abs_errors_left, width=width, label="Left |Pred-True|")
    plt.bar(modules + width/2, abs_errors_right, width=width, label="Right |Pred-True|")
    plt.xlabel("Module index (sorted by y)")
    plt.ylabel("Absolute Voltage Error")
    plt.title("Voltage Error per Module")
    plt.xticks(modules, [str(i) for i in modules])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "voltage_error_bar.png"))
    plt.close()

    return (
        np.array(mus_list, dtype=np.float32),
        np.array(zs_list, dtype=np.float32),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== VAE モデル読み込み =====
    vae_model = VAEControllerNet(latent_dim=LATENT_DIM).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
    vae_model.load_state_dict(ckpt["model_state_dict"])
    vae_model.eval()

    # VAE用 transform（学習時と合わせる：ToTensorのみで0〜1）
    transform = T.Compose([
        T.ToTensor(),
    ])

    # ===== detectron2 設定 =====
    register_coco_instances("Train", {}, TRAIN_JSON, TRAIN_IMAGES)
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = DEVNET_WEIGHT
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)

    metadata = MetadataCatalog.get("Train")

    # ===== ランダムに評価画像を選択 =====
    eval_entries = get_random_eval_images(BASE_DEVNET_DIR, n_folders=6, n_samples=10)

    # --- 強制選択の3枚を追加 (module1_3, module4_8, module6_6) ---
    forced_entries = [
        (os.path.join(BASE_DEVNET_DIR, "1module_dataset_max_DAC", "roi", "3.png"), 1),
        (os.path.join(BASE_DEVNET_DIR, "4module_dataset_max_DAC", "roi", "8.png"), 4),
        (os.path.join(BASE_DEVNET_DIR, "6module_dataset_max_DAC", "roi", "6.png"), 6),
    ]
    eval_entries.extend(forced_entries)
    # (path, module_idx) のタプルの重複を削除
    eval_entries = list(dict.fromkeys(eval_entries))

    print("Selected evaluation images (random + forced):")
    for p, i in eval_entries:
        print(f" - {p} (module {i})")

    # μ / z ベクトル保存用
    all_mu_devnet = []        # 全評価画像の μ（必要なら解析用）
    all_z_devnet = []         # 全評価画像の z（実サンプル）
    selected_z_devnet = []    # 強制3枚用（各画像で modules×LATENT_DIM）

    # ===== 各評価画像ごとの処理 =====
    for idx, (img_path, module_idx) in enumerate(eval_entries):
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            print(f"[WARN] Image not found: {img_path}")
            continue

        img_name = os.path.splitext(os.path.basename(img_path))[0]
        try:
            row_idx = int(img_name)
        except ValueError:
            print(f"[WARN] Cannot parse index from filename as int: {img_name}")
            continue

        # signals.csv 読み込み
        signals_path = os.path.join(
            BASE_DEVNET_DIR,
            f"{module_idx}module_dataset_max_DAC",
            "signals.csv"
        )
        if not os.path.exists(signals_path):
            print(f"[WARN] signals.csv not found: {signals_path}")
            continue
        signals_df = pd.read_csv(signals_path, header=None)

        # detectron2 推論
        outputs = predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            print(f"[INFO] No detection in {img_path}")
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        num_masks = masks.shape[0]
        assigned_colors = generate_distinct_colors(num_masks)

        vis = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, scale=1.0)
        out_vis = vis.overlay_instances(
            masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
        )
        result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")

        # 保存ディレクトリ（例: module3_22）
        save_dir = os.path.join(RESULT_ROOT, f"module{module_idx}_{img_name}")
        os.makedirs(save_dir, exist_ok=True)

        # 元ROI画像 & 分割可視化画像を保存
        roi_copy_path = os.path.join(save_dir, f"{img_name}_roi.png")
        vis_path = os.path.join(save_dir, f"{img_name}_vis.png")
        cv2.imwrite(roi_copy_path, img_bgr)
        cv2.imwrite(vis_path, result_img)

        print(f"Saved ROI + visualization → {save_dir}")

        # center_crop + VAE + 電圧誤差評価
        mu_dev, z_dev = crop_and_predict_with_vae(
            img_bgr=img_bgr,
            boxes=boxes,
            module_idx=module_idx,
            row_idx=row_idx,
            signals_df=signals_df,
            vae_model=vae_model,
            device=device,
            transform=transform,
            save_dir=save_dir,
            crop_h=62,
            crop_w=50,
        )

        if mu_dev.size > 0:
            all_mu_devnet.append(mu_dev)
        if z_dev.size > 0:
            all_z_devnet.append(z_dev)

            # === z の PCA 空間に乗せる対象（module1_3, 4_8, 6_6）だけ追加 ===
            key = (module_idx, row_idx)
            if key in [(1, 3), (4, 8), (6, 6)]:
                selected_z_devnet.append(z_dev)

    # ===== FF_only(z) の PCA 空間上に、強制3枚の z を重ねる =====
    if len(selected_z_devnet) == 0:
        print("[INFO] No selected latent z found for PCA overlay (module1_3, 4_8, 6_6).")
    else:
        # 各画像の z を 1 点に代表化（モジュール方向に平均）
        selected_z_means = [z.mean(axis=0) for z in selected_z_devnet]
        selected_z_means = np.stack(selected_z_means, axis=0)  # shape (3, D)
        print(f"Selected latent z means shape: {selected_z_means.shape}")

        # FF_only の latent_z.npy を読む
        if not os.path.exists(FF_ONLY_LATENT_PATH):
            print(f"[WARN] FF_only latent z file not found: {FF_ONLY_LATENT_PATH}")
        else:
            ff_z = np.load(FF_ONLY_LATENT_PATH)  # (N_ff, D)

            # PCA は FF_only(z) に対してのみ fit（基準空間）
            pca = PCA(n_components=2)
            pca.fit(ff_z)

            ff_z2d = pca.transform(ff_z)                    # (N_ff, 2)
            selected_z2d = pca.transform(selected_z_means)  # (3, 2)

            # --- オーバーレイ散布図（FF_only z: グレー, devnet 3点 z: 赤＋ラベル） ---
            plt.figure(figsize=(6, 5))
            plt.scatter(
                ff_z2d[:, 0], ff_z2d[:, 1],
                s=5, alpha=0.2, color="gray", label="FF_only (sampled z)"
            )
            plt.scatter(
                selected_z2d[:, 0], selected_z2d[:, 1],
                s=120, alpha=0.95, color="red", edgecolor="black",
                label="Devnet (3 forced samples, mean z)"
            )

            # 各点にキャプションを付与
            labels = ["1", "2", "3"]
            for (x, y, label) in zip(selected_z2d[:, 0], selected_z2d[:, 1], labels):
                plt.text(x + 0.02, y + 0.02, label, fontsize=10, fontweight="bold", color="red")

            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Latent PCA (sampled z: FF_only vs 3 Devnet samples)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            overlay_path = os.path.join(RESULT_ROOT, "latent_z_pca_selected_overlay.png")
            plt.savefig(overlay_path)
            plt.close()
            print(f"Saved PCA overlay for selected devnet z → {overlay_path}")

            # 2D座標も保存（解析用）
            np.save(os.path.join(RESULT_ROOT, "latent_z_ff_only_pca2d.npy"), ff_z2d)
            np.save(os.path.join(RESULT_ROOT, "latent_z_selected3_devnet_pca2d.npy"), selected_z2d)

            # 寄与率ログ（任意）
            explained = pca.explained_variance_ratio_
            pd.DataFrame({
                "PC": [1, 2],
                "explained_variance_ratio": explained,
            }).to_csv(
                os.path.join(RESULT_ROOT, "latent_z_pca_explained_variance_selected.csv"),
                index=False
            )

    # ===== Devnet 側の z 全体だけで PCA 2次元射影（おまけ） =====
    if len(all_z_devnet) > 0:
        all_z = np.concatenate(all_z_devnet, axis=0)  # shape (N_all_center_crops, LATENT_DIM)
        try:
            pca_z = PCA(n_components=2)
            z_2d = pca_z.fit_transform(all_z)

            plt.figure(figsize=(6, 5))
            plt.scatter(z_2d[:, 0], z_2d[:, 1], s=5, alpha=0.6, color="tab:blue")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Latent Space (sampled z, Devnet-integrated, PCA 2D)")
            plt.grid(True)
            plt.tight_layout()
            out_zpca_path = os.path.join(RESULT_ROOT, "latent_z_pca2d_devnet.png")
            plt.savefig(out_zpca_path)
            plt.close()
            print(f"Saved PCA of sampled z (Devnet-integrated) → {out_zpca_path}")

            # 2D座標 & 寄与率も保存
            np.save(os.path.join(RESULT_ROOT, "latent_z_devnet_pca2d.npy"), z_2d)
            explained_z = pca_z.explained_variance_ratio_
            pd.DataFrame({
                "PC": [1, 2],
                "explained_variance_ratio": explained_z,
            }).to_csv(
                os.path.join(RESULT_ROOT, "latent_z_pca_explained_variance_devnet.csv"),
                index=False
            )

            # 生の z も保存
            np.save(os.path.join(RESULT_ROOT, "latent_z_devnet.npy"), all_z)

        except Exception as e:
            print(f"[WARN] PCA(z) failed: {e}")
    else:
        print("[INFO] No z collected for PCA (Devnet side).")

    print("Done.")


if __name__ == "__main__":
    main()
