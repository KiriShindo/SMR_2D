import os
import random
import colorsys
import json
import time
from pathlib import Path
from tkinter import Tk, filedialog
from PIL import Image

import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# 全グラフのデフォルトフォントを Times New Roman にする
matplotlib.rcParams["font.family"] = "Times New Roman"

import torch
import torch.nn as nn
import torch.nn.functional as F  # VoltageToMask のデコーダで使用
import imageio.v2 as imageio

# detectron2 関係
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

from IK_upper0_afterFF_eval import *  # IKBeamNet, make_beammask_upper0_2ch, beammask_to_color など

# Arduino シリアル
from serial import Serial


# =========================================================
#  パス設定
# =========================================================

BASE_DEVNET_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data_new"

MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_beam2ch_nocleansing_best.pth"

IK_MODEL_PATH = (
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\ik_beam2ch_babbling_1000_20_shift_upper0_randompair_model.pth"
)

# ★ Voltage→Mask 用 ckpt（300epoch のファイルに変更して）
V2M_CKPT_PATH = (
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025"
    r"\SMR_control\module_controller\IK\voltage_to_mask_epoch300.pth"
)

# devnet / seg-net
DEVNET_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
DEVNET_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
DEVNET_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01/model_final.pth"

SEG_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
SEG_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"
SEG_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01/model_final.pth"

# USBカメラ側の ROI 設定
ROI_CONFIG_FULL = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\roi_config_full.json"

# center crop / rect2mask / seg-net 用
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0

RECT_W = 40
RECT_H = 4
ANGLE_SEARCH_RANGE_DEG = 60.0
ANGLE_SEARCH_STEP_DEG = 1.0

CROP_H = 62
CROP_W = 50

# Arduino / カメラ
SERIAL_PORT = 'COM4'
BAUDRATE = 9600
CAMERA_INDEX = 0  # 必要なら変更

### フィードバックゲイン
K_GAIN = 1.5
### フィードバック回数
NUM_LOOP = 10

### フィードバック電圧を印加する前に0.0Vにするかどうか
RESET_EACH_LOOP = True
### 0.0V印加してから次の電圧を印加するまでの時間
WAIT_TIME = 3.0

### 電圧を印加してからカメラ撮影までの時間
LOOP_WAIT = 5.0

### モジュール数
NUM_MODULE = 5

# ★ Voltage→Mask で使う電圧正規化（訓練時の mean/std に合わせて書き換え推奨）
#   0〜5V 一様なら mean≈2.5, std≈1.44 くらいだが、訓練ログに出ている値をコピペするのがベスト。
# V2M_VMEAN = np.array([2.5, 2.5], dtype=np.float32)
# V2M_VSTD  = np.array([1.5, 1.5], dtype=np.float32)
V2M_VMEAN = np.array([2.5855, 2.3915], dtype=np.float32)
V2M_VSTD  = np.array([1.4434049, 1.4229937], dtype=np.float32)


# ===== ベースディレクトリ =====
BASE_ROOT = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\IK_only_nocamera_result"

# ===== フォルダ名自動生成 =====
folder_name = f"Module{NUM_MODULE}_K={K_GAIN:.1f}_Loop{NUM_LOOP}_Reset{RESET_EACH_LOOP}_Wait{WAIT_TIME:.1f}_Freq{LOOP_WAIT:.1f}"
RESULT_ROOT = os.path.join(BASE_ROOT, folder_name)
os.makedirs(RESULT_ROOT, exist_ok=True)


# =========================================================
# 2ch CNN モデル定義（FF用：現状未使用ならそのまま）
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
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
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


# =========================================================
# Voltage → Mask モデル定義（推論専用）
# =========================================================

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

        self.init_h = out_h // 16
        self.init_w = out_w // 16
        if self.init_h < 1 or self.init_w < 1:
            raise ValueError(f"出力サイズが小さすぎるかも: H={out_h}, W={out_w}")

        self.fc = nn.Linear(latent_dim, 256 * self.init_h * self.init_w)

        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.fc(z)  # (B, 256 * H0 * W0)
        x = x.view(-1, 256, self.init_h, self.init_w)
        x = self.deconv(x)  # (B, 2, H', W')

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
        z = self.encoder(v)               # (B, latent_dim)
        mask_logits = self.decoder(z)     # (B, 2, H, W)
        return mask_logits


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


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot


# =========================================================
# Voltage→Mask 出力を RECT_W×RECT_H の長方形マスクに整形
# =========================================================

def voltage_to_rect_beammask(
    v_left, v_right,
    v2m_model,
    v2m_mean,
    v2m_std,
    device
):
    """
    電圧 (v_left, v_right) から VoltageToMaskNet で 2ch マスクを生成し、
    RECT_W×RECT_H の細長い矩形にフィットさせて返す。
    戻り値: mask_2ch (2, H, W) float32, {0,1}
    """
    v_np = np.array([v_left, v_right], dtype=np.float32)  # (2,)
    v_norm = (v_np - v2m_mean) / v2m_std
    v_t = torch.from_numpy(v_norm).unsqueeze(0).to(device)  # (1,2)

    with torch.no_grad():
        logits = v2m_model(v_t)  # (1,2,H,W)
        prob = torch.sigmoid(logits)[0].cpu().numpy()  # (2,H,W)

    masks_bool = prob > 0.5  # (2,H,W) bool

    rect_masks, angles, centers, overlaps = rectify_all_to_fixed_rects(
        masks_bool,
        RECT_W,
        RECT_H,
        angle_search_range=ANGLE_SEARCH_RANGE_DEG,
        angle_step=ANGLE_SEARCH_STEP_DEG
    )

    if rect_masks.shape[0] != 2:
        print(f"[V2M] Warning: rect_masks shape {rect_masks.shape}, expected (2,H,W)")
        # とりあえずそのまま返す
        if masks_bool.shape[0] == 2:
            return masks_bool.astype(np.float32)
        else:
            return np.zeros_like(prob, dtype=np.float32)

    # y座標最小が upper
    tops = [mask_y_top(rect_masks[i]) for i in range(2)]
    idx_upper = int(np.argmin(tops))
    idx_lower = 1 - idx_upper

    upper = rect_masks[idx_upper].astype(np.float32)
    lower = rect_masks[idx_lower].astype(np.float32)

    mask_2ch = np.stack([upper, lower], axis=0)  # (2,H,W)
    return mask_2ch


# =========================================================
#   seg-net → 2chマスク → CNN 推論（既存、今回は主にターゲット側）
# =========================================================

def crop_and_predict_with_cnn(
    img_bgr,
    boxes,
    model,
    device,
    v_stats,
    seg_predictor,
    seg_metadata,
    save_dir,
    crop_h=CROP_H,
    crop_w=CROP_W,
    signals_df=None,
    row_idx=None,
):
    """
    これは主に FF 評価用の関数（ロジックは元コードほぼそのまま）
    """
    targ_crop_dir = os.path.join(save_dir, "targ_center_crops")
    os.makedirs(targ_crop_dir, exist_ok=True)

    os.makedirs(save_dir, exist_ok=True)
    H, W, _ = img_bgr.shape

    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    half_h = crop_h // 2
    half_w = crop_w // 2

    have_gt = (signals_df is not None) and (row_idx is not None)
    if have_gt:
        try:
            row = signals_df.iloc[row_idx - 1]
        except Exception:
            have_gt = False
            row = None
    else:
        row = None

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

        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]
        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        center_crop_path = os.path.join(targ_crop_dir, f"center_crop_{n:02d}.png")
        cv2.imwrite(center_crop_path, blended)

        # seg-net
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

        if rect_masks.shape[0] != 2:
            print(f"  [WARN] rect_masks not 2ch (got {rect_masks.shape[0]}), skip module {n}.")
            continue

        tops = [mask_y_top(rect_masks[i]) for i in range(2)]
        idx_upper = int(np.argmin(tops))
        idx_lower = 1 - idx_upper

        upper = rect_masks[idx_upper].astype(np.float32)
        lower = rect_masks[idx_lower].astype(np.float32)

        mask_2ch = np.stack([upper, lower], axis=0)

        vis = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        vis[upper > 0.5] = (0, 0, 255)
        vis[lower > 0.5] = (255, 0, 0)
        vis_path = os.path.join(targ_crop_dir, f"beam_mask_vis_{n:02d}.png")
        cv2.imwrite(vis_path, vis)

        x = torch.from_numpy(mask_2ch).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_norm = model(x)
        pred_norm = pred_norm.cpu().numpy()[0]

        v_left_pred  = float(pred_norm[0] * v_left_std  + v_left_mean)
        v_right_pred = float(pred_norm[1] * v_right_std + v_right_mean)

        pred_list.append([v_left_pred, v_right_pred])

        if have_gt:
            v_left_true  = float(row[n])
            v_right_true = float(row[n + 6])

            true_list.append([v_left_true, v_right_true])
            abs_errors_left.append(abs(v_left_pred - v_left_true))
            abs_errors_right.append(abs(v_right_pred - v_right_true))

    if len(pred_list) == 0:
        return None, None

    pred_arr = np.array(pred_list, dtype=np.float32)
    true_arr = np.array(true_list, dtype=np.float32) if len(true_list) > 0 else None

    # 以下の棒グラフ部分は省略（元コード通りなら残してOK）

    return true_arr, pred_arr


# =========================================================
#  Arduino / カメラ / ROI 合成ユーティリティ
# =========================================================

def select_image_via_dialog():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select ROI image (1 module)",
        filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    root.destroy()
    if not file_path:
        print("[INFO] No image selected.")
        return None
    return file_path


def init_serial(port: str, baudrate: int):
    print(f"[Serial] Opening {port}@{baudrate} ...")
    ser = Serial(port, baudrate, timeout=1)
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[Serial] <- {line}")
        if line == 'READY':
            break
    print("[Serial] Arduino READY")
    return ser


def send_voltages_to_arduino(ser, module_pred_volts):
    if module_pred_volts is None or module_pred_volts.size == 0:
        print("[Serial] No module voltages to send.")
        return

    M = module_pred_volts.shape[0]
    L = [0.0] * 6
    R = [0.0] * 6
    for i in range(min(6, M)):
        L[i] = float(module_pred_volts[i, 0])
        R[i] = float(module_pred_volts[i, 1])

    volts = L + R
    volts = [max(0.0, min(5.0, v)) for v in volts]

    cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in volts) + '\n'
    print(f"[Serial] -> {cmd.strip()}")
    ser.write(cmd.encode())

    while True:
        resp = ser.readline().decode(errors='ignore').strip()
        if resp:
            print(f"[Serial] <- {resp}")
        if resp == 'APPLIED':
            break
    print("[Serial] Voltages applied.")


def reset_voltages_to_zero(ser, n_channels=12):
    zeros = [0.0] * n_channels
    cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in zeros) + '\n'
    print(f"[Serial] -> {cmd.strip()}  (reset)")
    ser.write(cmd.encode())
    while True:
        resp = ser.readline().decode(errors='ignore').strip()
        if resp:
            print(f"[Serial] <- {resp}")
        if resp == 'APPLIED':
            break
    print("[Serial] All channels reset to 0.0V.")


def open_camera(index=0, width=1920, height=1080):
    print(f"[Camera] Opening camera index {index} ...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("[Camera] Failed to open camera.")
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def capture_frame(cap, grab_n=5):
    if cap is None or not cap.isOpened():
        print("[Camera] capture_frame called but cap is not opened.")
        return None

    for _ in range(grab_n):
        cap.grab()
    ret, frame = cap.read()
    if not ret:
        print("[Camera] Failed to grab frame.")
        return None

    print(f"[Camera] captured frame shape: {frame.shape}")
    return frame


def load_roi_config(json_path: str):
    with open(json_path, 'r') as f:
        roi = json.load(f)
    return roi["x"], roi["y"], roi["w"], roi["h"]


def crop_with_roi(img, x, y, w, h):
    H, W = img.shape[:2]
    if x >= W or y >= H:
        return np.empty((0, 0, 3), dtype=img.dtype)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)

    return img[y1:y2, x1:x2].copy()


def make_red_tint(img_bgr):
    red = img_bgr.copy()
    red[:, :, 0] = 0   # B
    red[:, :, 1] = 0   # G
    return red


def make_blue_tint(img_bgr):
    blue = img_bgr.copy()
    blue[:, :, 1] = 0  # G
    blue[:, :, 2] = 0  # R
    return blue


def compute_mse(img1_bgr, img2_bgr):
    a = img1_bgr.astype(np.float32)
    b = img2_bgr.astype(np.float32)
    diff = a - b
    mse = np.mean(diff ** 2)
    return float(mse)


def compute_mse_and_overlay(img_target_roi, cam_full_frame, step_idx, base_dir):
    x, y, w, h = load_roi_config(ROI_CONFIG_FULL)
    roi_cam = crop_with_roi(cam_full_frame, x, y, w, h)

    if roi_cam.size == 0:
        print(f"[WARN] step {step_idx}: roi_cam is empty, MSEをNaNにします")
        return float('nan'), None

    H_t, W_t = img_target_roi.shape[:2]
    H_c, W_c = roi_cam.shape[:2]
    if (H_t, W_t) != (H_c, W_c):
        roi_cam_resized = cv2.resize(roi_cam, (W_t, H_t))
    else:
        roi_cam_resized = roi_cam

    mse = compute_mse(img_target_roi, roi_cam_resized)

    red  = make_red_tint(roi_cam_resized)
    blue = make_blue_tint(img_target_roi)
    overlay = cv2.addWeighted(red, 0.5, blue, 0.5, 0)

    timeline_dir = os.path.join(base_dir, "timeline_overlays")
    os.makedirs(timeline_dir, exist_ok=True)

    overlay_path = os.path.join(timeline_dir, f"overlay_step_{step_idx:03d}.png")
    cv2.imwrite(overlay_path, overlay)

    print(f"[Timeline] step {step_idx}: MSE={mse:.3f}, overlay saved -> {overlay_path}")
    return mse, overlay_path


def create_mse_plot_and_gif(
    mse_list,
    overlay_paths,
    save_dir,
    gif_duration=5.0,
):
    steps = list(range(len(mse_list)))        # 0,1,2,...
    steps_shifted = [s - 1 for s in steps]    # -1,0,1,2,...

    plt.figure(figsize=(6, 4))
    plt.plot(steps_shifted, mse_list, marker='o')
    plt.xlabel("Time step")
    plt.ylabel("MSE (target ROI vs camera ROI)")
    plt.title("MSE over time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, NUM_LOOP)
    plt.tight_layout()
    mse_plot_path = os.path.join(save_dir, "mse_time_series.png")
    plt.savefig(mse_plot_path)
    plt.close()
    print(f"[Timeline] MSE plot saved -> {mse_plot_path}")

    dt = WAIT_TIME + LOOP_WAIT
    time_axis = [s * dt for s in steps_shifted]

    plt.figure(figsize=(6, 4))
    plt.plot(time_axis, mse_list, marker='o')
    plt.xlabel("Time [s]")
    plt.ylabel("MSE (target ROI vs camera ROI)")
    plt.title("MSE over time (sec)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlim(0, NUM_LOOP * dt)
    plt.tight_layout()
    mse_time_plot_path = os.path.join(save_dir, "mse_time_series_time.png")
    plt.savefig(mse_time_plot_path)
    plt.close()
    print(f"[Timeline] MSE(time) plot saved -> {mse_time_plot_path}")

    frames = []
    for p in overlay_paths:
        if p is None:
            continue
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))

    if not frames:
        print("[Timeline] No frames for GIF, skip.")
        return

    gif_path = os.path.join(save_dir, "overlay_time_series.gif")

    if isinstance(gif_duration, (list, tuple)):
        durations_ms = [int(d * 1000) for d in gif_duration]
        if len(durations_ms) < len(frames):
            durations_ms += [durations_ms[-1]] * (len(frames) - len(durations_ms))
    else:
        durations_ms = [int(gif_duration * 1000)] * len(frames)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0
    )
    print(f"[Timeline] GIF saved -> {gif_path}")


def make_center_crops_from_boxes(img_bgr, boxes, save_dir, crop_h=CROP_H, crop_w=CROP_W):
    os.makedirs(save_dir, exist_ok=True)
    H, W, _ = img_bgr.shape

    boxes_sorted = sorted(boxes, key=lambda b: b[1])
    half_h = crop_h // 2
    half_w = crop_w // 2

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

        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]
        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        out_path = os.path.join(save_dir, f"center_crop_cam_{n:02d}.png")
        cv2.imwrite(out_path, blended)


def save_roi_overlay(roi_image, captured_full, dev_predictor, save_dir):
    if not os.path.exists(ROI_CONFIG_FULL):
        print(f"[ROI] roi_config_full.json not found: {ROI_CONFIG_FULL}. Skip overlay.")
        return

    x, y, w, h = load_roi_config(ROI_CONFIG_FULL)
    roi_cap = crop_with_roi(captured_full, x, y, w, h)

    if roi_cap.size == 0:
        print("[ROI] roi_cap is empty. Check ROI config or camera resolution. Skip overlay.")
        return

    os.makedirs(save_dir, exist_ok=True)

    raw_path = os.path.join(save_dir, "roi_captured_raw.png")
    cv2.imwrite(raw_path, roi_cap)
    print(f"[ROI] Captured ROI raw image saved: {raw_path}")

    outputs_cam = dev_predictor(roi_cap)
    instances_cam = outputs_cam["instances"].to("cpu")
    if len(instances_cam) == 0:
        print("[ROI] No detections on captured ROI image. Skip center_crop_cam.")
    else:
        boxes_cam = instances_cam.pred_boxes.tensor.numpy()
        center_dir = os.path.join(save_dir, "cam_center_crops")
        make_center_crops_from_boxes(roi_cap, boxes_cam, center_dir, crop_h=CROP_H, crop_w=CROP_W)
        print(f"[ROI] Camera center crops saved in: {center_dir}")

    h_o, w_o = roi_image.shape[:2]
    roi_cap_resized = cv2.resize(roi_cap, (w_o, h_o))

    red  = make_red_tint(roi_image)
    blue = make_blue_tint(roi_cap_resized)

    overlay = cv2.addWeighted(red, 0.5, blue, 0.5, 0)

    cv2.imwrite(os.path.join(save_dir, "roi_original_red.png"), red)
    cv2.imwrite(os.path.join(save_dir, "roi_captured_blue.png"), blue)
    cv2.imwrite(os.path.join(save_dir, "roi_overlay_red_orig_blue_cap.png"), overlay)
    print(f"[ROI] Overlay image saved in: {save_dir}")

    input_center_dir = os.path.join(save_dir)
    cam_center_dir   = os.path.join(save_dir, "cam_center_crops")
    overlay_dir      = os.path.join(save_dir, "center_crop_overlays")
    os.makedirs(overlay_dir, exist_ok=True)

    input_crops = sorted([f for f in os.listdir(input_center_dir) if f.startswith("center_crop_") and f.endswith(".png")])
    cam_crops   = sorted([f for f in os.listdir(cam_center_dir) if f.startswith("center_crop_cam_") and f.endswith(".png")])

    for in_name, cam_name in zip(input_crops, cam_crops):
        in_path  = os.path.join(input_center_dir, in_name)
        cam_path = os.path.join(cam_center_dir, cam_name)
        out_path = os.path.join(overlay_dir, f"overlay_{in_name.replace('center_crop_', 'center_crop_module')}")

        img_red = make_red_tint(cv2.imread(in_path))
        img_blue = make_blue_tint(cv2.imread(cam_path))

        if img_red.shape[:2] != img_blue.shape[:2]:
            img_blue = cv2.resize(img_blue, (img_red.shape[1], img_red.shape[0]))

        overlay = cv2.addWeighted(img_red, 0.5, img_blue, 0.5, 0)
        cv2.imwrite(out_path, overlay)

    print(f"[ROI] Center crop overlay images saved in: {overlay_dir}")


def make_center_crops_in_memory(img_bgr, boxes, crop_h=CROP_H, crop_w=CROP_W):
    H, W, _ = img_bgr.shape
    boxes_sorted = sorted(boxes, key=lambda b: b[1])

    half_h = crop_h // 2
    half_w = crop_w // 2

    crops = []

    for box in boxes_sorted:
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

        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]

        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        crops.append(blended)

    return crops


def plot_voltage_bars_for_step(target_volt, pred_volt, out_path, step_idx=None):
    num_mod = target_volt.shape[0]
    x = np.arange(1, num_mod + 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bar_width = 0.35

    axes[0].bar(x - bar_width/2, target_volt[:, 0], width=bar_width, label="Target", color="lightgray", edgecolor="black")
    axes[0].bar(x + bar_width/2, pred_volt[:, 0], width=bar_width, label="Pred", color="royalblue", edgecolor="black")
    axes[0].set_ylim(0, 5)
    axes[0].set_xlabel("Module index", fontsize=12)
    axes[0].set_ylabel("Voltage [V]", fontsize=12)
    axes[0].set_title("Left side", fontsize=14)
    axes[0].set_xticks(x)
    axes[0].legend()

    axes[1].bar(x - bar_width/2, target_volt[:, 1], width=bar_width, label="Target", color="lightgray", edgecolor="black")
    axes[1].bar(x + bar_width/2, pred_volt[:, 1], width=bar_width, label="Pred", color="crimson", edgecolor="black")
    axes[1].set_ylim(0, 5)
    axes[1].set_xlabel("Module index", fontsize=12)
    axes[1].set_ylabel("Voltage [V]", fontsize=12)
    axes[1].set_title("Right side", fontsize=14)
    axes[1].set_xticks(x)
    axes[1].legend()

    if step_idx is not None:
        fig.text(0.05, 0.93, f"N = {step_idx}", fontsize=16, fontweight="bold", color="black")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150)
    plt.close()


def create_voltage_bar_gif(bar_image_paths, save_dir,
                           gif_name="voltage_bars_time_series.gif",
                           duration=1.0):
    frames = []
    for p in bar_image_paths:
        img_bgr = cv2.imread(p)
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(img_rgb))

    if not frames:
        print("[VoltageGIF] No bar images, skip GIF.")
        return

    gif_path = os.path.join(save_dir, gif_name)

    if isinstance(duration, (list, tuple)):
        durations_ms = [int(d * 1000) for d in duration]
        if len(durations_ms) < len(frames):
            durations_ms += [durations_ms[-1]] * (len(frames) - len(durations_ms))
    else:
        durations_ms = [int(duration * 1000)] * len(frames)

    frames[0].save(
        gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0
    )

    print(f"[VoltageGIF] Saved -> {gif_path}")


# =========================================================
#   フィードバックループ本体（Voltage→Mask 版）
# =========================================================

def run_feedback_loop(
    img_target_roi,
    boxes_target,
    initial_voltages,
    first_cam_frame,
    dev_predictor,
    seg_predictor,
    ik_model,
    device,
    v_stats,
    ser,
    base_save_dir,
    num_loops=30,
    wait_sec=5.0,
    start_step_idx=2,
    use_reset_between=RESET_EACH_LOOP,
    K_gain=K_GAIN,
    reset_wait_sec=WAIT_TIME,
    cap=None,
    target_voltages=None,
    v2m_model=None,
    v2m_mean=None,
    v2m_std=None,
):
    """
    Voltage→Mask モデルを使って、各ステップの「現在ビームマスク」を生成するフィードバックループ。
    カメラ画像は MSE とログ用のみで、IK入力には使わない。
    """

    if v2m_model is None or v2m_mean is None or v2m_std is None:
        print("[FB-IK] VoltageToMaskNet が渡されていません。Abort.")
        return [], [], []

    print("[FB-IK] Preparing target center crops & beam masks ...")
    target_crops = make_center_crops_in_memory(img_target_roi, boxes_target, crop_h=CROP_H, crop_w=CROP_W)
    num_modules = len(target_crops)
    if num_modules == 0:
        print("[FB-IK] No target modules found. Abort feedback.")
        return [], [], []

    target_masks_2ch = []
    for m_idx, targ_crop in enumerate(target_crops):
        ok_targ, mask_targ_2ch, img_targ_rot = make_beammask_upper0_2ch(targ_crop, seg_predictor)
        if not ok_targ:
            print(f"[FB-IK] Warning: failed to make target beammask for module {m_idx}, stop.")
            return [], [], []
        target_masks_2ch.append(mask_targ_2ch)

    targ_dbg_dir = os.path.join(base_save_dir, "fb_target_debug")
    os.makedirs(targ_dbg_dir, exist_ok=True)
    for m_idx, (targ_crop, mask2ch) in enumerate(zip(target_crops, target_masks_2ch)):
        cv2.imwrite(os.path.join(targ_dbg_dir, f"mod{m_idx}_targ_crop.png"), targ_crop)
        cv2.imwrite(
            os.path.join(targ_dbg_dir, f"mod{m_idx}_targ_beammask.png"),
            beammask_to_color(mask2ch)
        )

    mse_list_fb = []
    overlay_paths_fb = []
    bar_paths_fb = []

    cam_frame = first_cam_frame.copy()

    x_roi, y_roi, w_roi, h_roi = load_roi_config(ROI_CONFIG_FULL)

    current_voltages = np.array(initial_voltages, dtype=np.float32).copy()
    if current_voltages.shape[0] < num_modules:
        pad = np.zeros((num_modules - current_voltages.shape[0], 2), dtype=np.float32)
        current_voltages = np.vstack([current_voltages, pad])
    elif current_voltages.shape[0] > num_modules:
        current_voltages = current_voltages[:num_modules]

    current_voltages = np.clip(current_voltages, 0.0, 5.0)

    current_step_idx = start_step_idx

    for loop in range(num_loops):
        print(f"\n[FB-IK] ===== Feedback loop {loop+1}/{num_loops} =====")

        fb_dir = os.path.join(base_save_dir, f"fb_step_{loop+1:02d}")
        os.makedirs(fb_dir, exist_ok=True)

        cv2.imwrite(os.path.join(fb_dir, "captured_full.png"), cam_frame)

        if use_reset_between:
            print("[FB-IK] Resetting all channels to 0.0V before applying next voltages ...")
            reset_voltages_to_zero(ser, n_channels=12)
            print(f"[FB-IK] Waiting {reset_wait_sec} seconds after reset ...")
            time.sleep(reset_wait_sec)

        roi_cam = crop_with_roi(cam_frame, x_roi, y_roi, w_roi, h_roi)
        if roi_cam.size == 0:
            print("[FB-IK] ROI from camera is empty, break.")
            break

        cv2.imwrite(os.path.join(fb_dir, "roi_captured_raw.png"), roi_cam)

        H_t, W_t = img_target_roi.shape[:2]
        roi_cam_resized = cv2.resize(roi_cam, (W_t, H_t))
        red  = make_red_tint(roi_cam_resized)
        blue = make_blue_tint(img_target_roi)
        overlay_local = cv2.addWeighted(red, 0.5, blue, 0.5, 0)
        cv2.imwrite(os.path.join(fb_dir, "roi_overlay_red_orig_blue_cap.png"), overlay_local)

        outputs_cam = dev_predictor(roi_cam)
        instances_cam = outputs_cam["instances"].to("cpu")
        if len(instances_cam) == 0:
            print("[FB-IK] No detections in camera ROI, break.")
            break

        boxes_cam = instances_cam.pred_boxes.tensor.numpy()
        masks_cam = instances_cam.pred_masks.numpy()
        num_cam_masks = masks_cam.shape[0]

        assigned_colors = generate_distinct_colors(num_cam_masks)
        vis_cam = Visualizer(roi_cam[:, :, ::-1], metadata=None, scale=1.0)
        out_vis_cam = vis_cam.overlay_instances(
            masks=masks_cam, boxes=None, labels=None, assigned_colors=assigned_colors
        )
        result_img_cam = out_vis_cam.get_image()[:, :, ::-1].astype("uint8")
        cv2.imwrite(os.path.join(fb_dir, "cam_devnet_vis.png"), result_img_cam)

        cam_crops = make_center_crops_in_memory(roi_cam, boxes_cam, crop_h=CROP_H, crop_w=CROP_W)
        if len(cam_crops) < num_modules:
            print(f"[FB-IK] Warning: detected {len(cam_crops)} modules, but target has {num_modules}. Using min().")
        effective_modules = min(len(cam_crops), num_modules)

        V_next_raw     = np.zeros_like(current_voltages, dtype=np.float32)
        V_next_clipped = np.zeros_like(current_voltages, dtype=np.float32)

        for m_idx in range(effective_modules):
            img_cam_crop   = cam_crops[m_idx]
            mask_targ_2ch  = target_masks_2ch[m_idx]

            # 現在の電圧（既に clip 済み）
            vL_i = float(current_voltages[m_idx, 0])
            vR_i = float(current_voltages[m_idx, 1])

            # ★ Voltage→Mask で現在モジュールの beammask を生成し、
            #   RECT_W×RECT_H でフィットさせた 2ch マスクを得る
            mask_cam_2ch = voltage_to_rect_beammask(
                vL_i,
                vR_i,
                v2m_model=v2m_model,
                v2m_mean=v2m_mean,
                v2m_std=v2m_std,
                device=device,
            )

            # デバッグ保存
            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_cam_crop.png"), img_cam_crop)
            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_cam_beammask.png"), beammask_to_color(mask_cam_2ch))
            cv2.imwrite(os.path.join(fb_dir, f"mod{m_idx}_targ_beammask.png"), beammask_to_color(mask_targ_2ch))

            mask_cam_tensor  = torch.from_numpy(mask_cam_2ch).float()
            mask_targ_tensor = torch.from_numpy(mask_targ_2ch).float()

            _, H, W = mask_cam_tensor.shape
            q_map = torch.zeros((2, H, W), dtype=torch.float32)
            q_map[0, :, :] = vL_i / 5.0
            q_map[1, :, :] = vR_i / 5.0

            x = torch.cat([mask_cam_tensor, mask_targ_tensor, q_map], dim=0)
            x = x.unsqueeze(0).to(device)

            with torch.no_grad():
                y_hat = ik_model(x)
            vL_ik = float(y_hat[0, 0].cpu().item())
            vR_ik = float(y_hat[0, 1].cpu().item())

            dVL = vL_ik - vL_i
            dVR = vR_ik - vR_i

            vL_next_raw = vL_i + K_gain * dVL
            vR_next_raw = vR_i + K_gain * dVR

            vL_next_clip = max(0.0, min(5.0, vL_next_raw))
            vR_next_clip = max(0.0, min(5.0, vR_next_raw))

            V_next_raw[m_idx, 0]     = vL_next_raw
            V_next_raw[m_idx, 1]     = vR_next_raw
            V_next_clipped[m_idx, 0] = vL_next_clip
            V_next_clipped[m_idx, 1] = vR_next_clip

            print(
                f"[FB-IK] Module {m_idx}: "
                f"V_i_clip=({vL_i:.3f}, {vR_i:.3f}) , "
                f"V_next_clip=({vL_next_clip:.3f}, {vR_next_clip:.3f}) , "
                f"V_ik_raw=({vL_ik:.3f}, {vR_ik:.3f})"
            )

        np.savetxt(os.path.join(fb_dir, "ik_pred_voltages_raw.txt"),     V_next_raw,     fmt="%.4f")
        np.savetxt(os.path.join(fb_dir, "ik_pred_voltages_clipped.txt"), V_next_clipped, fmt="%.4f")

        if target_voltages is not None:
            tv = np.array(target_voltages, dtype=np.float32).copy()
            n_plot = min(tv.shape[0], V_next_clipped.shape[0])
            tv = tv[:n_plot, :]
            pv = V_next_clipped[:n_plot, :]

            valid_mask = np.any(tv > 0.0, axis=1)
            tv_plot = tv[valid_mask]
            pv_plot = pv[valid_mask]

            if tv_plot.shape[0] > 0:
                bar_path = os.path.join(fb_dir, f"voltage_bars_step_{loop+1:02d}.png")
                plot_voltage_bars_for_step(tv_plot, pv_plot, bar_path, step_idx=loop+1)
                bar_paths_fb.append(bar_path)

        send_voltages_to_arduino(ser, V_next_clipped)

        current_voltages = V_next_clipped.copy()

        print(f"[FB-IK] Waiting {wait_sec} seconds ...")
        time.sleep(wait_sec)

        cam_next = capture_frame(cap)
        if cam_next is None:
            print("[FB-IK] Camera capture failed in feedback, break.")
            break

        mse_t, overlay_t = compute_mse_and_overlay(
            img_target_roi, cam_next, current_step_idx, base_save_dir
        )
        mse_list_fb.append(mse_t)
        overlay_paths_fb.append(overlay_t)
        current_step_idx += 1

        mse_log_path = os.path.join(base_save_dir, "mse_each_step.txt")
        with open(mse_log_path, "a") as f:
            f.write(f"{loop+1},{mse_t:.6f}\n")

        cam_frame = cam_next.copy()

    return mse_list_fb, overlay_paths_fb, bar_paths_fb


# =========================================================
#   main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cap = open_camera(CAMERA_INDEX, width=1920, height=1080)
    if cap is None:
        print("[ERROR] Could not open camera. Abort.")
        return

    print("[INFO] Capturing initial camera frame (t=0) ...")
    initial_cam_frame = capture_frame(cap)
    if initial_cam_frame is None:
        print("[WARN] Failed to capture initial camera frame. Timeline may start from later steps.")

    # v_stats は現状 run_feedback_loop では未使用なのでダミーでOK(FFを使用しないので)
    v_stats = (0.0, 1.0, 0.0, 1.0)

    # ===== IK モデル =====
    ik_model = IKBeamNet(in_ch=6, feat_dim=128).to(device)
    ik_ckpt = torch.load(IK_MODEL_PATH, map_location=device)
    ik_state_dict = ik_ckpt.get("model_state_dict", ik_ckpt)
    ik_model.load_state_dict(ik_state_dict)
    ik_model.eval()
    print(f"[INFO] Loaded IK model from: {IK_MODEL_PATH}")

    # ===== Voltage→Mask モデル =====
    v2m_model = VoltageToMaskNet(out_h=CROP_H, out_w=CROP_W, latent_dim=256).to(device)
    v2m_state_dict = torch.load(V2M_CKPT_PATH, map_location=device)
    # 学習時に state_dict をそのまま保存している前提
    v2m_model.load_state_dict(v2m_state_dict)
    v2m_model.eval()
    print(f"[INFO] Loaded VoltageToMaskNet from: {V2M_CKPT_PATH}")
    print(f"[INFO] VoltageToMask normalization mean={V2M_VMEAN}, std={V2M_VSTD}")

    # ===== devnet =====
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

    # ===== seg-net =====
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
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg_seg.DATASETS.TEST = ()
    seg_predictor = DefaultPredictor(cfg_seg)
    seg_metadata = MetadataCatalog.get("SEG_TRAIN")

    # ===== ROI画像（ターゲット） =====
    img_path = select_image_via_dialog()
    if img_path is None:
        if cap is not None:
            cap.release()
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        if cap is not None:
            cap.release()
        return

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(RESULT_ROOT, f"selected_{img_name}")
    os.makedirs(save_dir, exist_ok=True)

    cv2.imwrite(os.path.join(save_dir, f"{img_name}_roi_input.png"), img_bgr)

    timeline_mse = []
    timeline_overlays = []

    if initial_cam_frame is not None:
        mse0, ov0 = compute_mse_and_overlay(
            img_target_roi=img_bgr,
            cam_full_frame=initial_cam_frame,
            step_idx=0,
            base_dir=save_dir,
        )
        timeline_mse.append(mse0)
        timeline_overlays.append(ov0)
    else:
        print("[Timeline] initial_cam_frame is None, t=0 をスキップします。")

    outputs = dev_predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        print(f"[INFO] No detection in {img_path}")
        if cap is not None:
            cap.release()
        return

    boxes = instances.pred_boxes.tensor.numpy()
    masks = instances.pred_masks.numpy()
    num_masks = masks.shape[0]
    assigned_colors = generate_distinct_colors(num_masks)

    vis = Visualizer(img_bgr[:, :, ::-1], metadata=dev_metadata, scale=1.0)
    out_vis = vis.overlay_instances(
        masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
    )
    result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_devnet_vis.png"), result_img)

    # ===== signals.csv → target_voltages =====
    target_voltages = None
    try:
        img_p = Path(img_path)
        module_dir = img_p.parent.parent
        signals_path = module_dir / "signals.csv"

        if signals_path.exists():
            print(f"[INFO] Loading target voltages from: {signals_path}")
            signals_df = pd.read_csv(signals_path, header=None)

            try:
                row_idx = int(img_name) - 1
            except ValueError:
                print(f"[WARN] img_name is not integer-like: {img_name}. Skip target_voltages.")
                row_idx = None

            if row_idx is not None and 0 <= row_idx < len(signals_df):
                row = signals_df.iloc[row_idx].values.astype(np.float32)

                if row.shape[0] >= 12:
                    left = row[0:6]
                    right = row[6:12]
                    lr = np.stack([left, right], axis=1)

                    num_modules_detected = boxes.shape[0]
                    lr = lr[:num_modules_detected, :]

                    target_voltages = lr
                    print(f"[INFO] target_voltages shape: {target_voltages.shape}")
                else:
                    print(f"[WARN] Row in signals.csv has <12 columns (got {row.shape[0]}). Skip target_voltages.")
            else:
                print(f"[WARN] Row index {row_idx} is out of range for signals.csv.")
        else:
            print(f"[INFO] signals.csv not found at: {signals_path}. target_voltages は使いません。")

    except Exception as e:
        print(f"[WARN] Failed to prepare target_voltages from signals.csv: {e}")
        target_voltages = None

    num_modules_detected = boxes.shape[0]
    initial_voltages = np.zeros((num_modules_detected, 2), dtype=np.float32)

    ser = None
    bar_paths_fb = []
    try:
        ser = init_serial(SERIAL_PORT, BAUDRATE)

        reset_voltages_to_zero(ser, n_channels=12)
        print(f"[INFO] Waiting {WAIT_TIME} seconds after initial reset ...")
        time.sleep(WAIT_TIME)

        cam_frame_init = capture_frame(cap)
        if cam_frame_init is None:
            print("[WARN] Camera capture failed after initial reset, skip feedback & ROI overlay.")
        else:
            cv2.imwrite(os.path.join(save_dir, "captured_full_init.png"), cam_frame_init)

            save_roi_overlay(img_bgr, cam_frame_init, dev_predictor, save_dir)

            mse1, ov1 = compute_mse_and_overlay(
                img_target_roi=img_bgr,
                cam_full_frame=cam_frame_init,
                step_idx=1,
                base_dir=save_dir,
            )
            timeline_mse.append(mse1)
            timeline_overlays.append(ov1)

            mse_fb, overlays_fb, bar_paths_fb = run_feedback_loop(
                img_target_roi=img_bgr,
                boxes_target=boxes,
                initial_voltages=initial_voltages,
                first_cam_frame=cam_frame_init,
                dev_predictor=dev_predictor,
                seg_predictor=seg_predictor,
                ik_model=ik_model,
                device=device,
                v_stats=v_stats,
                ser=ser,
                base_save_dir=save_dir,
                num_loops=NUM_LOOP,
                wait_sec=LOOP_WAIT,
                start_step_idx=2,
                K_gain=K_GAIN,
                use_reset_between=RESET_EACH_LOOP,
                reset_wait_sec=WAIT_TIME,
                cap=cap,
                target_voltages=target_voltages,
                v2m_model=v2m_model,
                v2m_mean=V2M_VMEAN,
                v2m_std=V2M_VSTD,
            )
            timeline_mse.extend(mse_fb)
            timeline_overlays.extend(overlays_fb)

        reset_voltages_to_zero(ser, n_channels=12)

    finally:
        if ser is not None:
            try:
                ser.close()
            except:
                pass

        if cap is not None:
            try:
                cap.release()
            except:
                pass

    if len(timeline_mse) > 0:
        create_mse_plot_and_gif(
            mse_list=timeline_mse,
            overlay_paths=timeline_overlays,
            save_dir=save_dir,
            gif_duration=1.0,
        )

    if len(bar_paths_fb) > 0:
        create_voltage_bar_gif(
            bar_image_paths=bar_paths_fb,
            save_dir=save_dir,
            gif_name="voltage_bars_time_series.gif",
            duration=1.0,
        )

    print(f"\n[INFO] All results saved in: {save_dir}")


if __name__ == "__main__":
    main()
