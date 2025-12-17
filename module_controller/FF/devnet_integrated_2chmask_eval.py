# -*- coding: utf-8 -*-
"""
eval_beam2ch_with_devnet_integrated.py

全体ROI画像 → devnet で各モジュールを検出・切り出し →
seg-net + rect2mask で上下2本の梁を固定サイズの細長い矩形マスクに →
上梁 / 下梁を 2ch マスク (2, H, W) に変換 →
学習済み 2ch CNN(ModuleControllerNet) で [V_left, V_right] を推定 →
signals.csv の真値と比較して誤差を可視化・評価するスクリプト
"""

import os
import random
import colorsys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as T

# detectron2 関係
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog


# =========================================================
#  パス設定（元コードからほぼそのまま）
# =========================================================

BASE_DEVNET_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data"

# devnet (1〜6 module) 用 signals.csv & roi は:
# BASE_DEVNET_DIR / f"{i}module_dataset_max_DAC" / "roi"
# BASE_DEVNET_DIR / f"{i}module_dataset_max_DAC" / "signals.csv"

# 2ch CNN の学習済み重み
MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_beam2ch_nocleansing_best.pth"

# 結果出力ルート
RESULT_ROOT = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_beam2ch_devnet_1module_nocleansing\devnet_integrated"
os.makedirs(RESULT_ROOT, exist_ok=True)

# ---- devnet (分割ネット) 用 detectron2 設定 ----
DEVNET_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
DEVNET_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
DEVNET_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01/model_final.pth"

# ---- seg-net (上下梁用) detectron2 設定 ----
SEG_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
SEG_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"
SEG_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01/model_final.pth"

# ==== center crop の周辺ノイズ補完用統計（元コード踏襲）====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0  # distanceTransform のスケール

# ===== rect2mask 用パラメータ =====
RECT_W = 40
RECT_H = 4
ANGLE_SEARCH_RANGE_DEG = 60.0
ANGLE_SEARCH_STEP_DEG = 1.0

# ===== center_crop サイズ（学習時と合わせる）=====
CROP_H = 62
CROP_W = 50


# =========================================================
# 2ch CNN モデル定義（学習スクリプトと同じ構造）
# =========================================================

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

            nn.Linear(64, 2),  # [v_left_norm, v_right_norm]
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


# =========================================================
#  共通ユーティリティ
# =========================================================

def sample_border_noise(H, W):
    noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
    return np.clip(noise, 0, 255).astype(np.uint8)


def generate_distinct_colors(n):
    colors = []
    for i in range(n):
        hue = i / max(1, n)
        r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
        colors.append((r, g, b))
    return colors


def estimate_angle_deg(mask_bool):
    m8 = (mask_bool.astype(np.uint8)) * 255
    cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0
    cnt = max(cnts, key=cv2.contourArea)
    (_, _), (w, h), ang = cv2.minAreaRect(cnt)
    if w < h:
        ang += 90.0
    if ang >= 90:
        ang -= 180
    if ang < -90:
        ang += 180
    return float(ang)


def make_rotated_rect_kernel(rect_w, rect_h, angle_deg):
    diag = int(np.ceil(np.sqrt(rect_w**2 + rect_h**2))) + 4
    kh = diag | 1
    kw = diag | 1
    kern = np.zeros((kh, kw), np.uint8)
    cx, cy = kw / 2.0, kh / 2.0
    box = cv2.boxPoints(((cx, cy), (float(rect_w), float(rect_h)), float(angle_deg)))
    box = np.int32(np.round(box))
    cv2.fillPoly(kern, [box], 1)
    return kern.astype(np.float32), (kw // 2, kh // 2)


def fit_fixed_rect_to_mask(mask_bool, rect_w, rect_h, base_angle_deg,
                           angle_search_range=0.0, angle_step=1.0):
    H, W = mask_bool.shape
    mask_f = mask_bool.astype(np.float32)
    best = dict(score=-1, angle=base_angle_deg, center=(W // 2, H // 2), rect_pts=None)
    angles = [base_angle_deg]
    if angle_search_range > 1e-6 and angle_step > 0:
        offs = np.arange(-angle_search_range, angle_search_range + 1e-9, angle_step)
        angles = [base_angle_deg + a for a in offs]
    for ang in angles:
        kern, _ = make_rotated_rect_kernel(rect_w, rect_h, ang)
        score_map = cv2.filter2D(mask_f, ddepth=-1, kernel=kern, borderType=cv2.BORDER_CONSTANT)
        idx = np.unravel_index(np.argmax(score_map), score_map.shape)
        y, x = int(idx[0]), int(idx[1])
        score = float(score_map[y, x])
        if score > best["score"]:
            cx, cy = float(x), float(y)
            box = cv2.boxPoints(((cx, cy), (float(rect_w), float(rect_h)), float(ang)))
            box = np.int32(np.round(box))
            best.update(dict(score=score, angle=ang, center=(x, y), rect_pts=box))
    rect_mask = np.zeros((H, W), np.uint8)
    if best["rect_pts"] is not None:
        cv2.fillPoly(rect_mask, [best["rect_pts"]], 1)
    rect_mask = rect_mask.astype(bool)
    rect_area = rect_mask.sum() + 1e-6
    overlap = float((rect_mask & mask_bool).sum())
    overlap_ratio = overlap / rect_area
    return rect_mask, best["angle"], best["center"], overlap_ratio


def rectify_all_to_fixed_rects(masks_bool, rect_w, rect_h,
                               angle_search_range=0.0, angle_step=1.0):
    rect_masks, angles, centers, overlaps = [], [], [], []
    for m in masks_bool:
        base_ang = estimate_angle_deg(m)
        rmask, ang, ctr, ov = fit_fixed_rect_to_mask(
            m, rect_w, rect_h, base_ang,
            angle_search_range=angle_search_range,
            angle_step=angle_step
        )
        rect_masks.append(rmask)
        angles.append(ang)
        centers.append(ctr)
        overlaps.append(ov)
    if len(rect_masks) == 0:
        return masks_bool, [], [], []
    return np.stack(rect_masks, axis=0), angles, centers, overlaps


def mask_y_top(m):
    ys = np.where(m)[0]
    return int(ys.min()) if ys.size else 10**9


def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
    """
    1〜n_folders の各 module_dataset からランダムに roi 画像を拾う。
    戻り値: [(roi_img_path, module_idx), ...]
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


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# =========================================================
#   center_crop + seg-net → 2chマスク → CNN 推論
# =========================================================

def crop_and_predict_with_cnn(
    img_bgr,
    boxes,
    module_idx,
    row_idx,
    signals_df,
    model,
    device,
    v_stats,
    seg_predictor,
    seg_metadata,
    save_dir,
    crop_h=CROP_H,
    crop_w=CROP_W,
):
    """
    - devnet の bbox をもとに center_crop (+ノイズ補完) を作る
    - seg-net で上下梁をセグメントし，rect2mask で固定長方形化
    - 上/下を 2ch ビームマスク (2, H, W) に変換
    - 2ch CNN(ModuleControllerNet) で正規化スケールの電圧を予測
    - signals.csv の真値と比較し，誤差バーや真値/予測値の棒グラフを保存

    戻り値:
        true_volts: shape (num_modules, 2) [V_left, V_right]
        pred_volts: shape (num_modules, 2) [V_left, V_right]
    """
    os.makedirs(save_dir, exist_ok=True)
    H, W, _ = img_bgr.shape

    # 上から順にモジュールを並べる
    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    half_h = crop_h // 2
    half_w = crop_w // 2
    num_modules = len(boxes_sorted)

    row = signals_df.iloc[row_idx - 1]  # 0-index に直す

    v_left_mean, v_left_std, v_right_mean, v_right_std = v_stats

    abs_errors_left = []
    abs_errors_right = []
    true_list = []
    pred_list = []

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

        # ノイズ + distance transform で境界補完
        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]
        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        center_crop_path = os.path.join(save_dir, f"center_crop_{n:02d}.png")
        cv2.imwrite(center_crop_path, blended)

        # ==== seg-net で上下梁マスク取得 → rect2mask で細長い矩形化 ====
        seg_img = blended.copy()
        outputs_seg = seg_predictor(seg_img)
        instances_seg = outputs_seg["instances"].to("cpu")

        if len(instances_seg) < 2:
            print(f"  [WARN] seg-net: detected {len(instances_seg)} (<2) for module {n}, skip this module.")
            continue

        scores = instances_seg.scores.numpy()
        masks_all = instances_seg.pred_masks.numpy().astype(bool)
        top2_idx = np.argsort(-scores)[:2]
        masks_two = masks_all[top2_idx]

        rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
            masks_two,
            RECT_W, RECT_H,
            angle_search_range=ANGLE_SEARCH_RANGE_DEG,
            angle_step=ANGLE_SEARCH_STEP_DEG
        )

        # rect_masks: (2, H, W) だと思ってる
        if rect_masks.shape[0] != 2:
            print(f"  [WARN] rect_masks not 2ch (got {rect_masks.shape[0]}), skip module {n}.")
            continue

        # 上下を y座標で決定
        tops = [mask_y_top(rect_masks[i]) for i in range(2)]
        idx_upper = int(np.argmin(tops))
        idx_lower = 1 - idx_upper

        upper = rect_masks[idx_upper].astype(np.float32)
        lower = rect_masks[idx_lower].astype(np.float32)

        # 2ch マスク (2, H, W)
        mask_2ch = np.stack([upper, lower], axis=0)

        # 可視化 BGR: 上=赤, 下=青
        vis = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        vis[upper > 0.5] = (0, 0, 255)
        vis[lower > 0.5] = (255, 0, 0)
        vis_path = os.path.join(save_dir, f"beam_mask_vis_{n:02d}.png")
        cv2.imwrite(vis_path, vis)

        # ==== CNN で電圧推定（正規化スケール → 実電圧）====
        x = torch.from_numpy(mask_2ch).unsqueeze(0).to(device)  # (1,2,H,W)
        with torch.no_grad():
            pred_norm = model(x)  # (1,2)
        pred_norm = pred_norm.cpu().numpy()[0]

        v_left_pred  = float(pred_norm[0] * v_left_std  + v_left_mean)
        v_right_pred = float(pred_norm[1] * v_right_std + v_right_mean)

        # 真値電圧: n番目モジュール → 左: n列目, 右: n+6列目
        v_left_true  = float(row[n])
        v_right_true = float(row[n + 6])

        true_list.append([v_left_true, v_right_true])
        pred_list.append([v_left_pred, v_right_pred])

        abs_errors_left.append(abs(v_left_pred - v_left_true))
        abs_errors_right.append(abs(v_right_pred - v_right_true))

    # ==== ここから可視化まわり ====    

    # 真値／予測値が1つもない場合
    if len(true_list) == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32)

    true_arr = np.array(true_list, dtype=np.float32)  # (M,2)
    pred_arr = np.array(pred_list, dtype=np.float32)  # (M,2)
    modules = np.arange(true_arr.shape[0])

    # ---- モジュールごとの絶対誤差バー（既存処理）----
    if len(abs_errors_left) > 0:
        width = 0.35
        plt.figure(figsize=(6, 4))
        plt.bar(modules - width/2, abs_errors_left, width=width, label="Left |Pred-True|")
        plt.bar(modules + width/2, abs_errors_right, width=width, label="Right |Pred-True|")
        plt.xlabel("Module index (sorted by y)")
        plt.ylabel("Absolute Voltage Error [V]")
        plt.title("Voltage Error per Module")
        plt.xticks(modules, [str(i) for i in modules])
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "voltage_error_bar.png"))
        plt.close()

    # ---- (1) 左右それぞれ：真値 vs 予測値 棒グラフ ----
    width = 0.35

    # 左
    plt.figure(figsize=(6, 4))
    plt.bar(modules - width/2, true_arr[:, 0], width=width, label="Left True")
    plt.bar(modules + width/2, pred_arr[:, 0], width=width, label="Left Pred")
    plt.xlabel("Module index (sorted by y)")
    plt.ylabel("Voltage [V]")
    plt.title("True vs Pred (Left Voltage per Module)")
    plt.xticks(modules, [str(i) for i in modules])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "voltage_bar_left_true_pred.png"))
    plt.close()

    # 右
    plt.figure(figsize=(6, 4))
    plt.bar(modules - width/2, true_arr[:, 1], width=width, label="Right True")
    plt.bar(modules + width/2, pred_arr[:, 1], width=width, label="Right Pred")
    plt.xlabel("Module index (sorted by y)")
    plt.ylabel("Voltage [V]")
    plt.title("True vs Pred (Right Voltage per Module)")
    plt.xticks(modules, [str(i) for i in modules])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "voltage_bar_right_true_pred.png"))
    plt.close()

    # ---- (2) 予測値を 0.0〜5.0 にクリッピングした版 ----
    pred_clip = np.clip(pred_arr, 0.0, 5.0)

    # 左（クリップ）
    plt.figure(figsize=(6, 4))
    plt.bar(modules - width/2, true_arr[:, 0], width=width, label="Left True")
    plt.bar(modules + width/2, pred_clip[:, 0], width=width, label="Left Pred (clipped 0–5V)")
    plt.xlabel("Module index (sorted by y)")
    plt.ylabel("Voltage [V]")
    plt.title("True vs Pred (Left, clipped 0–5V)")
    plt.xticks(modules, [str(i) for i in modules])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0.0, 5.0)  # ★ 縦軸を 0–5V に固定
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "voltage_bar_left_true_pred_clipped.png"))
    plt.close()

    # 右（クリップ）
    plt.figure(figsize=(6, 4))
    plt.bar(modules - width/2, true_arr[:, 1], width=width, label="Right True")
    plt.bar(modules + width/2, pred_clip[:, 1], width=width, label="Right Pred (clipped 0–5V)")
    plt.xlabel("Module index (sorted by y)")
    plt.ylabel("Voltage [V]")
    plt.title("True vs Pred (Right, clipped 0–5V)")
    plt.xticks(modules, [str(i) for i in modules])
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    plt.ylim(0.0, 5.0)  # ★ 縦軸を 0–5V に固定
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "voltage_bar_right_true_pred_clipped.png"))
    plt.close()

    return true_arr, pred_arr



# =========================================================
#   main: devnet + seg-net + 2ch CNN で一気通貫評価
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== 2ch CNN モデル読み込み =====
    model = ModuleControllerNet(feat_dim=128).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 正規化統計量（学習時に保存してある前提）
    v_left_mean  = float(ckpt["v_left_mean"])
    v_left_std   = float(ckpt["v_left_std"])
    v_right_mean = float(ckpt["v_right_mean"])
    v_right_std  = float(ckpt["v_right_std"])
    v_stats = (v_left_mean, v_left_std, v_right_mean, v_right_std)

    print("Voltage normalization stats:")
    print(f"  left  mean={v_left_mean:.4f}, std={v_left_std:.4f}")
    print(f"  right mean={v_right_mean:.4f}, std={v_right_std:.4f}")

    # ===== devnet 設定 =====
    try:
        register_coco_instances("DEV_TRAIN", {}, DEVNET_TRAIN_JSON, DEVNET_TRAIN_IMAGES)
    except Exception:
        pass
    cfg_dev = get_cfg()
    cfg_dev.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg_dev.MODEL.WEIGHTS = DEVNET_WEIGHT
    cfg_dev.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg_dev.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_dev.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg_dev.DATASETS.TEST = ()
    dev_predictor = DefaultPredictor(cfg_dev)
    dev_metadata = MetadataCatalog.get("DEV_TRAIN")

    # ===== seg-net 設定 =====
    try:
        register_coco_instances("SEG_TRAIN", {}, SEG_TRAIN_JSON, SEG_TRAIN_IMAGES)
    except Exception:
        pass
    cfg_seg = get_cfg()
    cfg_seg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg_seg.MODEL.WEIGHTS = SEG_WEIGHT
    cfg_seg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg_seg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg_seg.DATASETS.TEST = ()
    seg_predictor = DefaultPredictor(cfg_seg)
    seg_metadata = MetadataCatalog.get("SEG_TRAIN")

    # ===== 評価画像の選択 =====
    eval_entries = get_random_eval_images(BASE_DEVNET_DIR, n_folders=6, n_samples=10)

    # VAE版と同じく強制3枚を追加するならここで（不要ならコメントアウト）
    forced_entries = [
        (os.path.join(BASE_DEVNET_DIR, "1module_dataset_max_DAC", "roi", "3.png"), 1),
        (os.path.join(BASE_DEVNET_DIR, "4module_dataset_max_DAC", "roi", "8.png"), 4),
        (os.path.join(BASE_DEVNET_DIR, "6module_dataset_max_DAC", "roi", "6.png"), 6),
    ]
    eval_entries.extend(forced_entries)
    eval_entries = list(dict.fromkeys(eval_entries))  # 重複削除

    print("Selected evaluation images (random + forced):")
    for p, i in eval_entries:
        print(f" - {p} (module {i})")

    all_true = []
    all_pred = []

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

        signals_path = os.path.join(
            BASE_DEVNET_DIR,
            f"{module_idx}module_dataset_max_DAC",
            "signals.csv"
        )
        if not os.path.exists(signals_path):
            print(f"[WARN] signals.csv not found: {signals_path}")
            continue
        signals_df = pd.read_csv(signals_path, header=None)

        # devnet でモジュール検出
        outputs = dev_predictor(img_bgr)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            print(f"[INFO] No detection in {img_path}")
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        num_masks = masks.shape[0]
        assigned_colors = generate_distinct_colors(num_masks)

        vis = Visualizer(img_bgr[:, :, ::-1], metadata=dev_metadata, scale=1.0)
        out_vis = vis.overlay_instances(
            masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
        )
        result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")

        save_dir = os.path.join(RESULT_ROOT, f"module{module_idx}_{img_name}")
        os.makedirs(save_dir, exist_ok=True)

        roi_copy_path = os.path.join(save_dir, f"{img_name}_roi.png")
        vis_path = os.path.join(save_dir, f"{img_name}_devnet_vis.png")
        cv2.imwrite(roi_copy_path, img_bgr)
        cv2.imwrite(vis_path, result_img)

        print(f"\n[{idx+1}/{len(eval_entries)}] {img_path}  →  save_dir = {save_dir}")

        true_v, pred_v = crop_and_predict_with_cnn(
            img_bgr=img_bgr,
            boxes=boxes,
            module_idx=module_idx,
            row_idx=row_idx,
            signals_df=signals_df,
            model=model,
            device=device,
            v_stats=v_stats,
            seg_predictor=seg_predictor,
            seg_metadata=seg_metadata,
            save_dir=save_dir,
            crop_h=CROP_H,
            crop_w=CROP_W,
        )

        if true_v.shape[0] > 0:
            all_true.append(true_v)
            all_pred.append(pred_v)

    if len(all_true) == 0:
        print("\n[INFO] No valid module predictions collected. Done.")
        return

    all_true = np.concatenate(all_true, axis=0)  # (N,2)
    all_pred = np.concatenate(all_pred, axis=0)  # (N,2)

    # ===== 全体指標 =====
    mse_total = np.mean((all_pred - all_true) ** 2)
    mae_total = np.mean(np.abs(all_pred - all_true))

    mse_left = np.mean((all_pred[:, 0] - all_true[:, 0]) ** 2)
    mae_left = np.mean(np.abs(all_pred[:, 0] - all_true[:, 0]))

    mse_right = np.mean((all_pred[:, 1] - all_true[:, 1]) ** 2)
    mae_right = np.mean(np.abs(all_pred[:, 1] - all_true[:, 1]))

    r2_left  = r2_score(all_true[:, 0], all_pred[:, 0])
    r2_right = r2_score(all_true[:, 1], all_pred[:, 1])

    print("\n===== Overall Evaluation (Devnet + 2ch CNN) =====")
    print(f"MSE  (total):  {mse_total:.6f}")
    print(f"MAE  (total):  {mae_total:.6f}")
    print(f"MSE  (left):   {mse_left:.6f}")
    print(f"MAE  (left):   {mae_left:.6f}")
    print(f"MSE  (right):  {mse_right:.6f}")
    print(f"MAE  (right):  {mae_right:.6f}")
    print(f"R^2  (left):   {r2_left:.4f}")
    print(f"R^2  (right):  {r2_right:.4f}")

    # 保存
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
    pd.DataFrame([metrics]).to_csv(
        os.path.join(RESULT_ROOT, "metrics_beam2ch_devnet.csv"),
        index=False
    )

    # 散布図
    plt.figure()
    plt.scatter(all_true[:, 0], all_pred[:, 0], s=5, alpha=0.5)
    min_v = min(all_true[:, 0].min(), all_pred[:, 0].min())
    max_v = max(all_true[:, 0].max(), all_pred[:, 0].max())
    plt.plot([min_v, max_v], [min_v, max_v], 'k--')
    plt.xlabel("True V_left [V]")
    plt.ylabel("Pred V_left [V]")
    plt.title("Pred vs True (Left Voltage, Devnet+2ch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_ROOT, "pred_vs_true_left_beam2ch_devnet.png"))
    plt.close()

    plt.figure()
    plt.scatter(all_true[:, 1], all_pred[:, 1], s=5, alpha=0.5)
    min_v = min(all_true[:, 1].min(), all_pred[:, 1].min())
    max_v = max(all_true[:, 1].max(), all_pred[:, 1].max())
    plt.plot([min_v, max_v], [min_v, max_v], 'k--')
    plt.xlabel("True V_right [V]")
    plt.ylabel("Pred V_right [V]")
    plt.title("Pred vs True (Right Voltage, Devnet+2ch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_ROOT, "pred_vs_true_right_beam2ch_devnet.png"))
    plt.close()

    # 誤差ヒスト
    err_left  = all_pred[:, 0] - all_true[:, 0]
    err_right = all_pred[:, 1] - all_true[:, 1]

    plt.figure()
    plt.hist(err_left, bins=50, alpha=0.7)
    plt.xlabel("Error (Pred - True) [V]  (Left)")
    plt.ylabel("Count")
    plt.title("Error Histogram (Left Voltage, Devnet+2ch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_ROOT, "error_hist_left_beam2ch_devnet.png"))
    plt.close()

    plt.figure()
    plt.hist(err_right, bins=50, alpha=0.7)
    plt.xlabel("Error (Pred - True) [V]  (Right)")
    plt.ylabel("Count")
    plt.title("Error Histogram (Right Voltage, Devnet+2ch)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_ROOT, "error_hist_right_beam2ch_devnet.png"))
    plt.close()

    print(f"\nEvaluation results saved in: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
