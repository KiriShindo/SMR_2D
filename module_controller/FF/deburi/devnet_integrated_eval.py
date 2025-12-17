# -*- coding: utf-8 -*-
"""
eval_devnet_with_module_controller_per_image_full.py

- detectron2 でモジュールごとに分割し、62x50 の center_crop を生成
- 各 center_crop を ModuleControllerNet に入力して電圧を予測
- signals.csv から真値を取得
- 各画像フォルダに：
    - 元画像
    - セグメント可視化
    - crops/
    - 真値 vs 予測値の棒グラフ
    - 誤差棒グラフ
  を保存する
"""

import os
import cv2
import colorsys
import random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as T

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# === 学習コードからモデル定義と重みパスを import ===
from FF_train import ModuleControllerNet, MODEL_SAVE_PATH


# ===== 背景色統計 =====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0


# ===== ノイズ生成関数 =====
def sample_border_noise(H, W):
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


# ===== カラー生成関数 =====
def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / n
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append((r, g, b))
    return colors


# ===== ランダム画像選択 =====
def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
    entries = []
    for _ in range(n_samples):
        i = random.randint(1, n_folders)
        roi_dir = os.path.join(base_dir, f"{i}module_dataset_max_DAC", "roi")
        if not os.path.exists(roi_dir):
            continue
        files = [f for f in os.listdir(roi_dir) if f.lower().endswith(".png")]
        if not files:
            continue
        chosen = random.choice(files)
        entries.append((os.path.join(roi_dir, chosen), i))
    return entries


# ===== bbox切り出し＋ブレンド補完 =====
def crop_and_save_and_collect(img, boxes, crops_dir, crop_h=62, crop_w=50):
    os.makedirs(crops_dir, exist_ok=True)
    H, W, _ = img.shape
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    half_h, half_w = crop_h // 2, crop_w // 2
    center_crops = []

    for idx, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

        x1c, x2c = cx - half_w, cx + half_w
        y1c, y2c = cy - half_h, cy + half_h

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

        x1_src, y1_src = max(0, x1c), max(0, y1c)
        x2_src, y2_src = min(W, x2c), min(H, y2c)
        x1_dst, y1_dst = x1_src - x1c, y1_src - y1c
        x2_dst, y2_dst = x1_dst + (x2_src - x1_src), y1_dst + (y2_src - y1_src)

        if x2_src > x1_src and y2_src > y1_src:
            cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img[y1_src:y2_src, x1_src:x2_src]
            mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

        # --- 境界補完 ---
        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        blended = cropped * alpha[..., None] + noise * (1 - alpha[..., None])
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # 保存
        cv2.imwrite(os.path.join(crops_dir, f"crop_{idx:02d}.png"), cropped)
        cv2.imwrite(os.path.join(crops_dir, f"center_crop_{idx:02d}.png"), blended)
        center_crops.append((blended, idx))

    print(f"Saved {len(center_crops)} cropped + blended crops → {crops_dir}")
    return center_crops


def main():
    # ===== Detectron2設定 =====
    train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
    train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
    register_coco_instances("Train", {}, train_json, train_images)

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)

    # ===== ModuleControllerNet 読み込み =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for ModuleControllerNet: {device}")
    model = ModuleControllerNet(feat_dim=128).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # ===== 入出力設定 =====
    base_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data"
    result_root = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/result/devnet_integrated"
    os.makedirs(result_root, exist_ok=True)

    eval_entries = get_random_eval_images(base_dir, n_folders=6, n_samples=10)
    metadata = MetadataCatalog.get("Train")

    # ===== signals.csv 読み込み =====
    volt_dfs = {}
    for i in range(1, 7):
        csv_path = os.path.join(base_dir, f"{i}module_dataset_max_DAC", "signals.csv")
        if os.path.exists(csv_path):
            volt_dfs[i] = pd.read_csv(csv_path, header=None)

    # ===== 評価ループ =====
    for img_path, module_idx in eval_entries:
        img = cv2.imread(img_path)
        if img is None or module_idx not in volt_dfs:
            continue

        frame_j = int(os.path.splitext(os.path.basename(img_path))[0])
        row = volt_dfs[module_idx].iloc[frame_j - 1]

        save_dir = os.path.join(result_root, f"module{module_idx}_{frame_j}")
        os.makedirs(save_dir, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir, f"{frame_j}_orig.png"), img)

        # detectron2 推論
        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        colors = generate_distinct_colors(len(masks))
        vis = Visualizer(img[:, :, ::-1], metadata=metadata)
        out = vis.overlay_instances(masks=masks, assigned_colors=colors)
        vis_img = out.get_image()[:, :, ::-1]
        cv2.imwrite(os.path.join(save_dir, f"{frame_j}_vis.png"), vis_img)

        crops_dir = os.path.join(save_dir, "crops")
        center_crops = crop_and_save_and_collect(img, boxes, crops_dir)

        errs_left, errs_right = [], []
        true_left_list, true_right_list = [], []
        pred_left_list, pred_right_list = [], []
        used_idx = []

        for crop_img, local_idx in center_crops:
            if local_idx >= module_idx:
                continue
            v_left_true  = float(row[local_idx])
            v_right_true = float(row[local_idx + 6])

            img_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
            inp = transform(Image.fromarray(img_rgb)).unsqueeze(0).to(device)
            with torch.no_grad():
                pred = model(inp).cpu().numpy()[0]
            pred = np.clip(pred, 0.0, 5.0)  # optional clip

            errs_left.append(pred[0] - v_left_true)
            errs_right.append(pred[1] - v_right_true)
            true_left_list.append(v_left_true)
            true_right_list.append(v_right_true)
            pred_left_list.append(pred[0])
            pred_right_list.append(pred[1])
            used_idx.append(local_idx)

        if len(used_idx) == 0:
            continue

        # === 誤差棒グラフ ===
        x = np.arange(len(used_idx))
        width = 0.35
        plt.figure(figsize=(8, 4))
        plt.bar(x - width/2, errs_left,  width=width, label="Left error")
        plt.bar(x + width/2, errs_right, width=width, label="Right error")
        plt.axhline(0, color="black")
        plt.xticks(x, [str(i) for i in used_idx])
        plt.xlabel("Module index")
        plt.ylabel("Error (pred - true) [V]")
        plt.title(f"Voltage error per module (module{module_idx}, frame {frame_j})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{frame_j}_error_bar.png"))
        plt.close()

        # === 真値 vs 予測棒グラフ ===
        plt.figure(figsize=(8, 4))
        plt.bar(x - width/2, true_left_list,  width=width, label="True Left", color="steelblue", alpha=0.7)
        plt.bar(x + width/2, pred_left_list,  width=width, label="Pred Left", color="orange", alpha=0.7)
        plt.bar(x - width/2, true_right_list, width=width, label="True Right", color="skyblue", alpha=0.5)
        plt.bar(x + width/2, pred_right_list, width=width, label="Pred Right", color="coral", alpha=0.5)
        plt.xticks(x, [str(i) for i in used_idx])
        plt.xlabel("Module index")
        plt.ylabel("Voltage [V]")
        plt.title(f"True vs Pred voltage (module{module_idx}, frame {frame_j})")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{frame_j}_voltage_true_vs_pred.png"))
        plt.close()

        print(f"✅ Saved evaluation plots → {save_dir}")

    print("🎯 Finished integrated evaluation for 10 random images.")


if __name__ == "__main__":
    main()
