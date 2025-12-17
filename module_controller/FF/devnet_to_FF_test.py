# """
# devnet_to_FF_arduino_roi_overlay.py

# - 起動時にエクスプローラで「1モジュール ROI 画像」を1枚選択（※すでにROI済みの画像）
# - 選択画像に対して:
#     - devnet でモジュール検出
#     - seg-net + rect2mask で2chマスクを生成
#     - 2ch CNN(ModuleControllerNet) で [V_left, V_right] を予測
# - 予測電圧を [0.0, 5.0] にクリップして Arduino に送信 (VOLT ... プロトコル)
# - 電圧印加後 5 秒待ち、USBカメラで画像撮影
# - roi_config_full.json の ROI で **USBカメラ画像だけ** を切り取り:
#     - ROIで切り取った「生画像」を保存 (roi_captured_raw.png)
#     - そのROI画像に対して devnet でモジュール検出し、
#       center_crop_cam_00.png, center_crop_cam_01.png, ... を保存
#     - さらに、ユーザ選択画像（そのまま）を赤、
#       カメラ ROI 画像を青にして 50%/50% で合成して保存
# - 終了前に必ず全12chを 0.0V にリセットしてからシリアルを閉じる
# """

# import os
# import random
# import colorsys
# import json
# import time
# from pathlib import Path
# from tkinter import Tk, filedialog

# import cv2
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# import torch
# import torch.nn as nn

# # detectron2 関係
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import MetadataCatalog

# # Arduino シリアル
# from serial import Serial


# # =========================================================
# #  パス設定
# # =========================================================

# BASE_DEVNET_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data"

# MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_beam2ch_nocleansing_best.pth"

# RESULT_ROOT = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_beam2ch_devnet_arduino2"
# os.makedirs(RESULT_ROOT, exist_ok=True)

# DEVNET_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
# DEVNET_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
# DEVNET_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01/model_final.pth"

# SEG_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
# SEG_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"
# SEG_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01/model_final.pth"

# # USBカメラ側の ROI 設定
# ROI_CONFIG_FULL = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\roi_config_full.json"

# # center crop / rect2mask / seg-net 用
# MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
# STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
# BLEND_WIDTH = 5.0

# RECT_W = 40
# RECT_H = 4
# ANGLE_SEARCH_RANGE_DEG = 60.0
# ANGLE_SEARCH_STEP_DEG = 1.0

# CROP_H = 62
# CROP_W = 50

# # Arduino / カメラ
# SERIAL_PORT = 'COM4'
# BAUDRATE = 9600
# CAMERA_INDEX = 0  # 必要なら変更


# # =========================================================
# # 2ch CNN モデル定義
# # =========================================================

# class BeamEncoder(nn.Module):
#     def __init__(self, out_dim: int = 128):
#         super().__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(2, 32, 3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(32, 64, 3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),

#             nn.Conv2d(64, 128, 3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(2),
#         )
#         self.gap = nn.AdaptiveAvgPool2d(1)
#         self.fc  = nn.Linear(128, out_dim)

#     def forward(self, x):
#         x = self.features(x)
#         x = self.gap(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x


# class VoltageMLP(nn.Module):
#     def __init__(self, in_dim: int):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),

#             nn.Linear(128, 64),
#             nn.ReLU(inplace=True),
#             nn.Dropout(p=0.3),

#             nn.Linear(64, 2),
#         )

#     def forward(self, x):
#         return self.net(x)


# class ModuleControllerNet(nn.Module):
#     def __init__(self, feat_dim=128):
#         super().__init__()
#         self.encoder = BeamEncoder(feat_dim)
#         self.head    = VoltageMLP(feat_dim)

#     def forward(self, x):
#         feat = self.encoder(x)
#         out  = self.head(feat)
#         return out


# # =========================================================
# #  共通ユーティリティ
# # =========================================================

# def sample_border_noise(H, W):
#     noise = np.random.normal(MEAN_BGR, STD_BGR, size=(H, W, 3))
#     return np.clip(noise, 0, 255).astype(np.uint8)


# def generate_distinct_colors(n):
#     colors = []
#     for i in range(n):
#         hue = i / max(1, n)
#         r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
#         colors.append((r, g, b))
#     return colors


# def estimate_angle_deg(mask_bool):
#     m8 = (mask_bool.astype(np.uint8)) * 255
#     cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return 0.0
#     cnt = max(cnts, key=cv2.contourArea)
#     (_, _), (w, h), ang = cv2.minAreaRect(cnt)
#     if w < h:
#         ang += 90.0
#     if ang >= 90:
#         ang -= 180
#     if ang < -90:
#         ang += 180
#     return float(ang)


# def make_rotated_rect_kernel(rect_w, rect_h, angle_deg):
#     diag = int(np.ceil(np.sqrt(rect_w**2 + rect_h**2))) + 4
#     kh = diag | 1
#     kw = diag | 1
#     kern = np.zeros((kh, kw), np.uint8)
#     cx, cy = kw / 2.0, kh / 2.0
#     box = cv2.boxPoints(((cx, cy), (float(rect_w), float(rect_h)), float(angle_deg)))
#     box = np.int32(np.round(box))
#     cv2.fillPoly(kern, [box], 1)
#     return kern.astype(np.float32), (kw // 2, kh // 2)


# def fit_fixed_rect_to_mask(mask_bool, rect_w, rect_h, base_angle_deg,
#                            angle_search_range=0.0, angle_step=1.0):
#     H, W = mask_bool.shape
#     mask_f = mask_bool.astype(np.float32)
#     best = dict(score=-1, angle=base_angle_deg, center=(W // 2, H // 2), rect_pts=None)
#     angles = [base_angle_deg]
#     if angle_search_range > 1e-6 and angle_step > 0:
#         offs = np.arange(-angle_search_range, angle_search_range + 1e-9, angle_step)
#         angles = [base_angle_deg + a for a in offs]
#     for ang in angles:
#         kern, _ = make_rotated_rect_kernel(rect_w, rect_h, ang)
#         score_map = cv2.filter2D(mask_f, ddepth=-1, kernel=kern, borderType=cv2.BORDER_CONSTANT)
#         idx = np.unravel_index(np.argmax(score_map), score_map.shape)
#         y, x = int(idx[0]), int(idx[1])
#         score = float(score_map[y, x])
#         if score > best["score"]:
#             cx, cy = float(x), float(y)
#             box = cv2.boxPoints(((cx, cy), (float(rect_w), float(rect_h)), float(ang)))
#             box = np.int32(np.round(box))
#             best.update(dict(score=score, angle=ang, center=(x, y), rect_pts=box))
#     rect_mask = np.zeros((H, W), np.uint8)
#     if best["rect_pts"] is not None:
#         cv2.fillPoly(rect_mask, [best["rect_pts"]], 1)
#     rect_mask = rect_mask.astype(bool)
#     rect_area = rect_mask.sum() + 1e-6
#     overlap = float((rect_mask & mask_bool).sum())
#     overlap_ratio = overlap / rect_area
#     return rect_mask, best["angle"], best["center"], overlap_ratio


# def rectify_all_to_fixed_rects(masks_bool, rect_w, rect_h,
#                                angle_search_range=0.0, angle_step=1.0):
#     rect_masks, angles, centers, overlaps = [], [], [], []
#     for m in masks_bool:
#         base_ang = estimate_angle_deg(m)
#         rmask, ang, ctr, ov = fit_fixed_rect_to_mask(
#             m, rect_w, rect_h, base_ang,
#             angle_search_range=angle_search_range,
#             angle_step=angle_step
#         )
#         rect_masks.append(rmask)
#         angles.append(ang)
#         centers.append(ctr)
#         overlaps.append(ov)
#     if len(rect_masks) == 0:
#         return masks_bool, [], [], []
#     return np.stack(rect_masks, axis=0), angles, centers, overlaps


# def mask_y_top(m):
#     ys = np.where(m)[0]
#     return int(ys.min()) if ys.size else 10**9


# def r2_score(y_true, y_pred):
#     ss_res = np.sum((y_true - y_pred) ** 2)
#     ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
#     return 1.0 - ss_res / ss_tot


# # =========================================================
# #   seg-net → 2chマスク → CNN 推論
# # =========================================================

# def crop_and_predict_with_cnn(
#     img_bgr,
#     boxes,
#     model,
#     device,
#     v_stats,
#     seg_predictor,
#     seg_metadata,
#     save_dir,
#     crop_h=CROP_H,
#     crop_w=CROP_W,
#     signals_df=None,
#     row_idx=None,
# ):
#     """
#     - devnet の bbox をもとに center_crop (+ノイズ補完) を作る
#     - seg-net で上下梁をセグメントし，rect2mask で固定長方形化
#     - 上/下を 2ch ビームマスク (2, H, W) に変換
#     - 2ch CNN(ModuleControllerNet) で [V_left, V_right] を予測
#     - signals_df, row_idx があれば真値との比較&可視化
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     H, W, _ = img_bgr.shape

#     boxes_sorted = sorted(boxes, key=lambda b: b[1])
#     half_h = crop_h // 2
#     half_w = crop_w // 2

#     have_gt = (signals_df is not None) and (row_idx is not None)
#     if have_gt:
#         try:
#             row = signals_df.iloc[row_idx - 1]
#         except Exception:
#             have_gt = False
#             row = None
#     else:
#         row = None

#     v_left_mean, v_left_std, v_right_mean, v_right_std = v_stats

#     abs_errors_left = []
#     abs_errors_right = []
#     true_list = []
#     pred_list = []

#     for n, box in enumerate(boxes_sorted):
#         x1, y1, x2, y2 = [int(v) for v in box]
#         cx = int((x1 + x2) / 2)
#         cy = int((y1 + y2) / 2)

#         x1c, x2c = cx - half_w, cx + half_w
#         y1c, y2c = cy - half_h, cy + half_h

#         cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
#         mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

#         x1_src, y1_src = max(0, x1c), max(0, y1c)
#         x2_src, y2_src = min(W, x2c), min(H, y2c)

#         x1_dst = x1_src - x1c
#         y1_dst = y1_src - y1c
#         x2_dst = x1_dst + (x2_src - x1_src)
#         y2_dst = y1_dst + (y2_src - y1_src)

#         if x2_src > x1_src and y2_src > y1_src:
#             cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img_bgr[y1_src:y2_src, x1_src:x2_src]
#             mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

#         noise = sample_border_noise(crop_h, crop_w)
#         dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
#         alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
#         alpha_3 = alpha[..., None]
#         blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
#         blended = np.clip(blended, 0, 255).astype(np.uint8)

#         center_crop_path = os.path.join(save_dir, f"center_crop_{n:02d}.png")
#         cv2.imwrite(center_crop_path, blended)

#         # seg-net
#         seg_img = blended.copy()
#         outputs_seg = seg_predictor(seg_img)
#         instances_seg = outputs_seg["instances"].to("cpu")

#         if len(instances_seg) < 2:
#             print(f"  [WARN] seg-net: detected {len(instances_seg)} (<2) for module {n}, skip this module.")
#             continue

#         scores = instances_seg.scores.numpy()
#         masks_all = instances_seg.pred_masks.numpy().astype(bool)
#         top2_idx = np.argsort(-scores)[:2]
#         masks_two = masks_all[top2_idx]

#         rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
#             masks_two,
#             RECT_W, RECT_H,
#             angle_search_range=ANGLE_SEARCH_RANGE_DEG,
#             angle_step=ANGLE_SEARCH_STEP_DEG
#         )

#         if rect_masks.shape[0] != 2:
#             print(f"  [WARN] rect_masks not 2ch (got {rect_masks.shape[0]}), skip module {n}.")
#             continue

#         tops = [mask_y_top(rect_masks[i]) for i in range(2)]
#         idx_upper = int(np.argmin(tops))
#         idx_lower = 1 - idx_upper

#         upper = rect_masks[idx_upper].astype(np.float32)
#         lower = rect_masks[idx_lower].astype(np.float32)

#         mask_2ch = np.stack([upper, lower], axis=0)

#         vis = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
#         vis[upper > 0.5] = (0, 0, 255)
#         vis[lower > 0.5] = (255, 0, 0)
#         vis_path = os.path.join(save_dir, f"beam_mask_vis_{n:02d}.png")
#         cv2.imwrite(vis_path, vis)

#         x = torch.from_numpy(mask_2ch).unsqueeze(0).to(device)
#         with torch.no_grad():
#             pred_norm = model(x)
#         pred_norm = pred_norm.cpu().numpy()[0]

#         v_left_pred  = float(pred_norm[0] * v_left_std  + v_left_mean)
#         v_right_pred = float(pred_norm[1] * v_right_std + v_right_mean)

#         pred_list.append([v_left_pred, v_right_pred])

#         if have_gt:
#             v_left_true  = float(row[n])
#             v_right_true = float(row[n + 6])

#             true_list.append([v_left_true, v_right_true])
#             abs_errors_left.append(abs(v_left_pred - v_left_true))
#             abs_errors_right.append(abs(v_right_pred - v_right_true))

#     if len(pred_list) == 0:
#         return None, None

#     pred_arr = np.array(pred_list, dtype=np.float32)

#     true_arr = None
#     if len(true_list) > 0:
#         true_arr = np.array(true_list, dtype=np.float32)
#         modules = np.arange(true_arr.shape[0])

#         # 誤差バー
#         width = 0.35
#         if len(abs_errors_left) > 0:
#             plt.figure(figsize=(6, 4))
#             plt.bar(modules - width/2, abs_errors_left, width=width, label="Left |Pred-True|")
#             plt.bar(modules + width/2, abs_errors_right, width=width, label="Right |Pred-True|")
#             plt.xlabel("Module index (sorted by y)")
#             plt.ylabel("Absolute Voltage Error [V]")
#             plt.title("Voltage Error per Module")
#             plt.xticks(modules, [str(i) for i in modules])
#             plt.legend()
#             plt.grid(True, axis="y", alpha=0.3)
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, "voltage_error_bar.png"))
#             plt.close()

#         # 真値 vs 予測値（左 / 右）
#         width = 0.35
#         plt.figure(figsize=(6, 4))
#         plt.bar(modules - width/2, true_arr[:, 0], width=width, label="Left True")
#         plt.bar(modules + width/2, pred_arr[:, 0], width=width, label="Left Pred")
#         plt.xlabel("Module index (sorted by y)")
#         plt.ylabel("Voltage [V]")
#         plt.title("True vs Pred (Left Voltage per Module)")
#         plt.xticks(modules, [str(i) for i in modules])
#         plt.legend()
#         plt.grid(True, axis="y", alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, "voltage_bar_left_true_pred.png"))
#         plt.close()

#         plt.figure(figsize=(6, 4))
#         plt.bar(modules - width/2, true_arr[:, 1], width=width, label="Right True")
#         plt.bar(modules + width/2, pred_arr[:, 1], width=width, label="Right Pred")
#         plt.xlabel("Module index (sorted by y)")
#         plt.ylabel("Voltage [V]")
#         plt.title("True vs Pred (Right Voltage per Module)")
#         plt.xticks(modules, [str(i) for i in modules])
#         plt.legend()
#         plt.grid(True, axis="y", alpha=0.3)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, "voltage_bar_right_true_pred.png"))
#         plt.close()

#         # クリップ版 0–5V
#         pred_clip = np.clip(pred_arr, 0.0, 5.0)
#         plt.figure(figsize=(6, 4))
#         plt.bar(modules - width/2, true_arr[:, 0], width=width, label="Left True")
#         plt.bar(modules + width/2, pred_clip[:, 0], width=width, label="Left Pred (clipped 0–5V)")
#         plt.xlabel("Module index (sorted by y)")
#         plt.ylabel("Voltage [V]")
#         plt.title("True vs Pred (Left, clipped 0–5V)")
#         plt.xticks(modules, [str(i) for i in modules])
#         plt.legend()
#         plt.grid(True, axis="y", alpha=0.3)
#         plt.ylim(0.0, 5.0)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, "voltage_bar_left_true_pred_clipped.png"))
#         plt.close()

#         plt.figure(figsize=(6, 4))
#         plt.bar(modules - width/2, true_arr[:, 1], width=width, label="Right True")
#         plt.bar(modules + width/2, pred_clip[:, 1], width=width, label="Right Pred (clipped 0–5V)")
#         plt.xlabel("Module index (sorted by y)")
#         plt.ylabel("Voltage [V]")
#         plt.title("True vs Pred (Right, clipped 0–5V)")
#         plt.xticks(modules, [str(i) for i in modules])
#         plt.legend()
#         plt.grid(True, axis="y", alpha=0.3)
#         plt.ylim(0.0, 5.0)
#         plt.tight_layout()
#         plt.savefig(os.path.join(save_dir, "voltage_bar_right_true_pred_clipped.png"))
#         plt.close()

#     return true_arr, pred_arr


# # =========================================================
# #  Arduino / カメラ / ROI 合成ユーティリティ
# # =========================================================

# def select_image_via_dialog():
#     root = Tk()
#     root.withdraw()
#     file_path = filedialog.askopenfilename(
#         title="Select ROI image (1 module)",
#         filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
#     )
#     root.destroy()
#     if not file_path:
#         print("[INFO] No image selected.")
#         return None
#     return file_path


# def init_serial(port: str, baudrate: int):
#     print(f"[Serial] Opening {port}@{baudrate} ...")
#     ser = Serial(port, baudrate, timeout=1)
#     # Arduino から READY が来るまで待つ
#     while True:
#         line = ser.readline().decode(errors='ignore').strip()
#         if line:
#             print(f"[Serial] <- {line}")
#         if line == 'READY':
#             break
#     print("[Serial] Arduino READY")
#     return ser


# def send_voltages_to_arduino(ser, module_pred_volts):
#     """
#     module_pred_volts: shape (M,2) [v_left, v_right] （top→bottom）
#     Arduino には L1..L6, R1..R6 の12chを送る。
#     M < 6 の場合は残りを 0.0V とする。
#     """
#     if module_pred_volts is None or module_pred_volts.size == 0:
#         print("[Serial] No module voltages to send.")
#         return

#     M = module_pred_volts.shape[0]
#     L = [0.0] * 6
#     R = [0.0] * 6
#     for i in range(min(6, M)):
#         L[i] = float(module_pred_volts[i, 0])
#         R[i] = float(module_pred_volts[i, 1])

#     volts = L + R
#     # 0.0〜5.0 にクリップ
#     volts = [max(0.0, min(5.0, v)) for v in volts]

#     cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in volts) + '\n'
#     print(f"[Serial] -> {cmd.strip()}")
#     ser.write(cmd.encode())

#     while True:
#         resp = ser.readline().decode(errors='ignore').strip()
#         if resp:
#             print(f"[Serial] <- {resp}")
#         if resp == 'APPLIED':
#             break
#     print("[Serial] Voltages applied.")


# def reset_voltages_to_zero(ser, n_channels=12):
#     """
#     全チャンネルを 0.0V にして APPLIED を待つ
#     """
#     zeros = [0.0] * n_channels
#     cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in zeros) + '\n'
#     print(f"[Serial] -> {cmd.strip()}  (reset)")
#     ser.write(cmd.encode())
#     while True:
#         resp = ser.readline().decode(errors='ignore').strip()
#         if resp:
#             print(f"[Serial] <- {resp}")
#         if resp == 'APPLIED':
#             break
#     print("[Serial] All channels reset to 0.0V.")


# def capture_image_from_camera(index=0, width=1920, height=1080):
#     print(f"[Camera] Opening camera index {index} ...")
#     cap = cv2.VideoCapture(index)
#     if not cap.isOpened():
#         print("[Camera] Failed to open camera.")
#         return None

#     # USBカメラ側のフルフレーム解像度（ROI_CONFIG_FULL の座標系に合わせる）
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

#     for _ in range(5):
#         cap.grab()
#     ret, frame = cap.read()
#     cap.release()
#     if not ret:
#         print("[Camera] Failed to grab frame.")
#         return None

#     print(f"[Camera] captured frame shape: {frame.shape}")
#     return frame


# def load_roi_config(json_path: str):
#     with open(json_path, 'r') as f:
#         roi = json.load(f)
#     return roi["x"], roi["y"], roi["w"], roi["h"]


# def crop_with_roi(img, x, y, w, h):
#     H, W = img.shape[:2]

#     # ROI が完全に画面外なら空行列
#     if x >= W or y >= H:
#         return np.empty((0, 0, 3), dtype=img.dtype)

#     x1 = max(0, x)
#     y1 = max(0, y)
#     x2 = min(W, x + w)
#     y2 = min(H, y + h)

#     if x2 <= x1 or y2 <= y1:
#         return np.empty((0, 0, 3), dtype=img.dtype)

#     return img[y2:y1 if False else y1:y2, x1:x2].copy()  # just to emphasize slicing direction


# def crop_with_roi(img, x, y, w, h):
#     H, W = img.shape[:2]

#     if x >= W or y >= H:
#         return np.empty((0, 0, 3), dtype=img.dtype)

#     x1 = max(0, x)
#     y1 = max(0, y)
#     x2 = min(W, x + w)
#     y2 = min(H, y + h)

#     if x2 <= x1 or y2 <= y1:
#         return np.empty((0, 0, 3), dtype=img.dtype)

#     return img[y1:y2, x1:x2].copy()


# def make_red_tint(img_bgr):
#     red = img_bgr.copy()
#     red[:, :, 0] = 0   # B
#     red[:, :, 1] = 0   # G
#     return red


# def make_blue_tint(img_bgr):
#     blue = img_bgr.copy()
#     blue[:, :, 1] = 0  # G
#     blue[:, :, 2] = 0  # R
#     return blue


# def make_center_crops_from_boxes(img_bgr, boxes, save_dir, crop_h=CROP_H, crop_w=CROP_W):
#     """
#     devnet の bbox から center_crop を作って保存するだけの関数
#     （seg-net や電圧推論は行わない）
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     H, W, _ = img_bgr.shape

#     boxes_sorted = sorted(boxes, key=lambda b: b[1])
#     half_h = crop_h // 2
#     half_w = crop_w // 2

#     for n, box in enumerate(boxes_sorted):
#         x1, y1, x2, y2 = [int(v) for v in box]
#         cx = int((x1 + x2) / 2)
#         cy = int((y1 + y2) / 2)

#         x1c, x2c = cx - half_w, cx + half_w
#         y1c, y2c = cy - half_h, cy + half_h

#         cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
#         mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

#         x1_src, y1_src = max(0, x1c), max(0, y1c)
#         x2_src, y2_src = min(W, x2c), min(H, y2c)

#         x1_dst = x1_src - x1c
#         y1_dst = y1_src - y1c
#         x2_dst = x1_dst + (x2_src - x1_src)
#         y2_dst = y1_dst + (y2_src - y1_src)

#         if x2_src > x1_src and y2_src > y1_src:
#             cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img_bgr[y1_src:y2_src, x1_src:x2_src]
#             mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

#         noise = sample_border_noise(crop_h, crop_w)
#         dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
#         alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
#         alpha_3 = alpha[..., None]
#         blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
#         blended = np.clip(blended, 0, 255).astype(np.uint8)

#         out_path = os.path.join(save_dir, f"center_crop_cam_{n:02d}.png")
#         cv2.imwrite(out_path, blended)


# def save_roi_overlay(roi_image, captured_full, dev_predictor, save_dir):
#     """
#     - roi_image: ユーザが選んだ ROI 済み画像（切り取り不要）
#     - captured_full: USB カメラのフルフレーム画像
#     - roi_config_full.json の ROI で「captured_full だけ」切り取る
#       - 生のROI画像を roi_captured_raw.png として保存
#       - そのROI画像を devnet でモジュール分割 → center_crop_cam_XX.png 保存
#     - roi_image を赤，カメラ ROI 画像を青にして 50%/50% で合成
#     """
#     if not os.path.exists(ROI_CONFIG_FULL):
#         print(f"[ROI] roi_config_full.json not found: {ROI_CONFIG_FULL}. Skip overlay.")
#         return

#     x, y, w, h = load_roi_config(ROI_CONFIG_FULL)
#     roi_cap = crop_with_roi(captured_full, x, y, w, h)

#     if roi_cap.size == 0:
#         print("[ROI] roi_cap is empty. Check ROI config or camera resolution. Skip overlay.")
#         return

#     os.makedirs(save_dir, exist_ok=True)

#     # 1. ROIで切り出した「生」カメラ画像を保存
#     raw_path = os.path.join(save_dir, "roi_captured_raw.png")
#     cv2.imwrite(raw_path, roi_cap)
#     print(f"[ROI] Captured ROI raw image saved: {raw_path}")

#     # 2. devnet でROIカメラ画像をモジュール分割 → center_crop_cam_XX.png
#     outputs_cam = dev_predictor(roi_cap)
#     instances_cam = outputs_cam["instances"].to("cpu")
#     if len(instances_cam) == 0:
#         print("[ROI] No detections on captured ROI image. Skip center_crop_cam.")
#     else:
#         boxes_cam = instances_cam.pred_boxes.tensor.numpy()
#         center_dir = os.path.join(save_dir, "cam_center_crops")
#         make_center_crops_from_boxes(roi_cap, boxes_cam, center_dir, crop_h=CROP_H, crop_w=CROP_W)
#         print(f"[ROI] Camera center crops saved in: {center_dir}")

#     # 3. オーバーレイ画像作成（ユーザROI vs カメラROI）
#     h_o, w_o = roi_image.shape[:2]
#     roi_cap_resized = cv2.resize(roi_cap, (w_o, h_o))

#     red  = make_red_tint(roi_image)
#     blue = make_blue_tint(roi_cap_resized)

#     overlay = cv2.addWeighted(red, 0.5, blue, 0.5, 0)

#     cv2.imwrite(os.path.join(save_dir, "roi_original_red.png"), red)
#     cv2.imwrite(os.path.join(save_dir, "roi_captured_blue.png"), blue)
#     cv2.imwrite(os.path.join(save_dir, "roi_overlay_red_orig_blue_cap.png"), overlay)
#     print(f"[ROI] Overlay image saved in: {save_dir}")

#     # 4. center_crop の赤青合成を生成
#     input_center_dir = os.path.join(save_dir)
#     cam_center_dir   = os.path.join(save_dir, "cam_center_crops")
#     overlay_dir      = os.path.join(save_dir, "center_crop_overlays")
#     os.makedirs(overlay_dir, exist_ok=True)

#     input_crops = sorted([f for f in os.listdir(input_center_dir) if f.startswith("center_crop_") and f.endswith(".png")])
#     cam_crops   = sorted([f for f in os.listdir(cam_center_dir) if f.startswith("center_crop_cam_") and f.endswith(".png")])

#     for in_name, cam_name in zip(input_crops, cam_crops):
#         in_path  = os.path.join(input_center_dir, in_name)
#         cam_path = os.path.join(cam_center_dir, cam_name)
#         out_path = os.path.join(overlay_dir, f"overlay_{in_name.replace('center_crop_', 'center_crop_module')}")

#         img_red = make_red_tint(cv2.imread(in_path))
#         img_blue = make_blue_tint(cv2.imread(cam_path))

#         # サイズを合わせて合成
#         if img_red.shape[:2] != img_blue.shape[:2]:
#             img_blue = cv2.resize(img_blue, (img_red.shape[1], img_red.shape[0]))

#         overlay = cv2.addWeighted(img_red, 0.5, img_blue, 0.5, 0)
#         cv2.imwrite(out_path, overlay)

#     print(f"[ROI] Center crop overlay images saved in: {overlay_dir}")


# # =========================================================
# #   main
# # =========================================================

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # ===== モデル読み込み =====
#     model = ModuleControllerNet(feat_dim=128).to(device)
#     ckpt = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
#     model.load_state_dict(ckpt["model_state_dict"])
#     model.eval()

#     v_left_mean  = float(ckpt["v_left_mean"])
#     v_left_std   = float(ckpt["v_left_std"])
#     v_right_mean = float(ckpt["v_right_mean"])
#     v_right_std  = float(ckpt["v_right_std"])
#     v_stats = (v_left_mean, v_left_std, v_right_mean, v_right_std)

#     print("Voltage normalization stats:")
#     print(f"  left  mean={v_left_mean:.4f}, std={v_left_std:.4f}")
#     print(f"  right mean={v_right_mean:.4f}, std={v_right_std:.4f}")

#     # ===== devnet 設定 =====
#     try:
#         register_coco_instances("DEV_TRAIN", {}, DEVNET_TRAIN_JSON, DEVNET_TRAIN_IMAGES)
#     except Exception:
#         pass
#     cfg_dev = get_cfg()
#     cfg_dev.merge_from_file(
#         model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     )
#     cfg_dev.MODEL.WEIGHTS = DEVNET_WEIGHT
#     cfg_dev.MODEL.ROI_HEADS.NUM_CLASSES = 1
#     cfg_dev.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     cfg_dev.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#     cfg_dev.DATASETS.TEST = ()
#     dev_predictor = DefaultPredictor(cfg_dev)
#     dev_metadata = MetadataCatalog.get("DEV_TRAIN")

#     # ===== seg-net 設定 =====
#     try:
#         register_coco_instances("SEG_TRAIN", {}, SEG_TRAIN_JSON, SEG_TRAIN_IMAGES)
#     except Exception:
#         pass
#     cfg_seg = get_cfg()
#     cfg_seg.merge_from_file(
#         model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
#     )
#     cfg_seg.MODEL.WEIGHTS = SEG_WEIGHT
#     cfg_seg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#     cfg_seg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#     cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#     cfg_seg.DATASETS.TEST = ()
#     seg_predictor = DefaultPredictor(cfg_seg)
#     seg_metadata = MetadataCatalog.get("SEG_TRAIN")

#     # ===== ROI画像をユーザに選択させる =====
#     img_path = select_image_via_dialog()
#     if img_path is None:
#         return

#     img_bgr = cv2.imread(img_path)
#     if img_bgr is None:
#         print(f"[ERROR] Failed to read image: {img_path}")
#         return

#     img_name = os.path.splitext(os.path.basename(img_path))[0]
#     save_dir = os.path.join(RESULT_ROOT, f"selected_{img_name}")
#     os.makedirs(save_dir, exist_ok=True)

#     # ROI画像そのまま保存
#     cv2.imwrite(os.path.join(save_dir, f"{img_name}_roi_input.png"), img_bgr)

#     # ===== signals.csv がある場合だけ読み込んで GT 評価 =====
#     signals_df = None
#     row_idx = None
#     try:
#         parent_name = Path(img_path).parent.name
#         if parent_name.endswith("module_dataset_max_DAC"):
#             module_idx = int(parent_name[0])  # "1module_dataset_max_DAC" → 1 を想定
#             signals_path = os.path.join(BASE_DEVNET_DIR, f"{module_idx}module_dataset_max_DAC", "signals.csv")
#             if os.path.exists(signals_path):
#                 signals_df = pd.read_csv(signals_path, header=None)
#                 row_idx = int(img_name)  # ファイル名 "57.png" → row 57
#                 print(f"[INFO] Using signals.csv: {signals_path}, row={row_idx}")
#     except Exception as e:
#         print(f"[WARN] Could not prepare ground truth: {e}")
#         signals_df = None
#         row_idx = None

#     # ===== devnet でモジュール検出（選択画像） =====
#     outputs = dev_predictor(img_bgr)
#     instances = outputs["instances"].to("cpu")
#     if len(instances) == 0:
#         print(f"[INFO] No detection in {img_path}")
#         return

#     boxes = instances.pred_boxes.tensor.numpy()
#     masks = instances.pred_masks.numpy()
#     num_masks = masks.shape[0]
#     assigned_colors = generate_distinct_colors(num_masks)

#     vis = Visualizer(img_bgr[:, :, ::-1], metadata=dev_metadata, scale=1.0)
#     out_vis = vis.overlay_instances(
#         masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
#     )
#     result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")

#     cv2.imwrite(os.path.join(save_dir, f"{img_name}_devnet_vis.png"), result_img)

#     # ===== CNN で電圧予測（選択画像） =====
#     true_v, pred_v = crop_and_predict_with_cnn(
#         img_bgr=img_bgr,
#         boxes=boxes,
#         model=model,
#         device=device,
#         v_stats=v_stats,
#         seg_predictor=seg_predictor,
#         seg_metadata=seg_metadata,
#         save_dir=save_dir,
#         crop_h=CROP_H,
#         crop_w=CROP_W,
#         signals_df=signals_df,
#         row_idx=row_idx,
#     )

#     if pred_v is None:
#         print("[INFO] No valid module predictions collected.")
#         return

#     np.savetxt(os.path.join(save_dir, "pred_voltages_raw.txt"), pred_v, fmt="%.4f")

#     # ===== Arduino を開き、予測電圧を送信 =====
#     ser = None
#     try:
#         ser = init_serial(SERIAL_PORT, BAUDRATE)
#         send_voltages_to_arduino(ser, pred_v)

#         # ===== 5秒待機してカメラで撮影 =====
#         print("[INFO] Waiting 5 seconds before capturing camera image ...")
#         time.sleep(5.0)

#         cam_frame = capture_image_from_camera(CAMERA_INDEX)
#         if cam_frame is None:
#             print("[WARN] Camera capture failed, skip ROI overlay.")
#         else:
#             cv2.imwrite(os.path.join(save_dir, "captured_full.png"), cam_frame)
#             # ここでROI切り抜き + 生保存 + devnet分割 + center_crop_cam + overlay
#             save_roi_overlay(img_bgr, cam_frame, dev_predictor, save_dir)

#         # ★ 終了前に必ず 0.0V を送信
#         reset_voltages_to_zero(ser, n_channels=12)

#     finally:
#         if ser is not None:
#             try:
#                 ser.close()
#             except:
#                 pass

#     print(f"\n[INFO] All results saved in: {save_dir}")


# if __name__ == "__main__":
#     main()














"""
devnet_to_FF_arduino_roi_overlay.py

- 起動時にエクスプローラで「1モジュール ROI 画像」を1枚選択（※すでにROI済みの画像）
- 選択画像に対して:
    - devnet でモジュール検出
    - seg-net + rect2mask で2chマスクを生成
    - 2ch CNN(ModuleControllerNet) で [V_left, V_right] を予測
- 予測電圧を [0.0, 5.0] にクリップして Arduino に送信 (VOLT ... プロトコル)
- 電圧印加後 5 秒待ち、USBカメラで画像撮影
- roi_config_full.json の ROI で **USBカメラ画像だけ** を切り取り:
    - ROIで切り取った「生画像」を保存 (roi_captured_raw.png)
    - そのROI画像に対して devnet でモジュール検出し、
      center_crop_cam_00.png, center_crop_cam_01.png, ... を保存
    - さらに、ユーザ選択画像（そのまま）を赤、
      カメラ ROI 画像を青にして 50%/50% で合成して保存
- 終了前に必ず全12chを 0.0V にリセットしてからシリアルを閉じる
"""

import os
import random
import colorsys
import json
import time
from pathlib import Path
from tkinter import Tk, filedialog

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# detectron2 関係
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# Arduino シリアル
from serial import Serial


# =========================================================
#  パス設定
# =========================================================

BASE_DEVNET_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data"

MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_beam2ch_nocleansing_best.pth"

RESULT_ROOT = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_beam2ch_devnet_arduino3"
os.makedirs(RESULT_ROOT, exist_ok=True)

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


# =========================================================
# 2ch CNN モデル定義
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
#   seg-net → 2chマスク → CNN 推論
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
    - devnet の bbox をもとに center_crop (+ノイズ補完) を作る
    - seg-net で上下梁をセグメントし，rect2mask で固定長方形化
    - 上/下を 2ch ビームマスク (2, H, W) に変換
    - 2ch CNN(ModuleControllerNet) で [V_left, V_right] を予測
    - signals_df, row_idx があれば真値との比較&可視化
    """

    # ======== 保存用フォルダ設定 ========
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

        # ---- 修正版: targ_center_crops に保存 ----
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

    # --- 0〜5V クリップ棒グラフ 追加部分 ---
    if true_arr is not None and len(true_arr) > 0:
        pred_clip = np.clip(pred_arr, 0.0, 5.0)
        modules = np.arange(true_arr.shape[0])
        width = 0.35

        # 左
        plt.figure(figsize=(6, 4))
        plt.bar(modules - width/2, true_arr[:, 0], width=width, label="Left True")
        plt.bar(modules + width/2, pred_clip[:, 0], width=width, label="Left Pred (clipped 0–5V)")
        plt.xlabel("Module index")
        plt.ylabel("Voltage [V]")
        plt.title("True vs Pred (Left, clipped 0–5V)")
        plt.xticks(modules, [str(i) for i in modules])
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.ylim(0.0, 5.0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "voltage_bar_left_true_pred_clipped.png"))
        plt.close()

        # 右
        plt.figure(figsize=(6, 4))
        plt.bar(modules - width/2, true_arr[:, 1], width=width, label="Right True")
        plt.bar(modules + width/2, pred_clip[:, 1], width=width, label="Right Pred (clipped 0–5V)")
        plt.xlabel("Module index")
        plt.ylabel("Voltage [V]")
        plt.title("True vs Pred (Right, clipped 0–5V)")
        plt.xticks(modules, [str(i) for i in modules])
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        plt.ylim(0.0, 5.0)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "voltage_bar_right_true_pred_clipped.png"))
        plt.close()

        print(f"[Plot] Saved clipped voltage bar charts in {save_dir}")

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
    # Arduino から READY が来るまで待つ
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[Serial] <- {line}")
        if line == 'READY':
            break
    print("[Serial] Arduino READY")
    return ser


def send_voltages_to_arduino(ser, module_pred_volts):
    """
    module_pred_volts: shape (M,2) [v_left, v_right] （top→bottom）
    Arduino には L1..L6, R1..R6 の12chを送る。
    M < 6 の場合は残りを 0.0V とする。
    """
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
    # 0.0〜5.0 にクリップ
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
    """
    全チャンネルを 0.0V にして APPLIED を待つ
    """
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


def capture_image_from_camera(index=0, width=1920, height=1080):
    print(f"[Camera] Opening camera index {index} ...")
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print("[Camera] Failed to open camera.")
        return None

    # USBカメラ側のフルフレーム解像度（ROI_CONFIG_FULL の座標系に合わせる）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    for _ in range(5):
        cap.grab()
    ret, frame = cap.read()
    cap.release()
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

    # ROI が完全に画面外なら空行列
    if x >= W or y >= H:
        return np.empty((0, 0, 3), dtype=img.dtype)

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(W, x + w)
    y2 = min(H, y + h)

    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=img.dtype)

    return img[y2:y1 if False else y1:y2, x1:x2].copy()  # just to emphasize slicing direction


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


def make_center_crops_from_boxes(img_bgr, boxes, save_dir, crop_h=CROP_H, crop_w=CROP_W):
    """
    devnet の bbox から center_crop を作って保存するだけの関数
    （seg-net や電圧推論は行わない）
    """
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
    """
    - roi_image: ユーザが選んだ ROI 済み画像（切り取り不要）
    - captured_full: USB カメラのフルフレーム画像
    - roi_config_full.json の ROI で「captured_full だけ」切り取る
      - 生のROI画像を roi_captured_raw.png として保存
      - そのROI画像を devnet でモジュール分割 → center_crop_cam_XX.png 保存
    - roi_image を赤，カメラ ROI 画像を青にして 50%/50% で合成
    """
    if not os.path.exists(ROI_CONFIG_FULL):
        print(f"[ROI] roi_config_full.json not found: {ROI_CONFIG_FULL}. Skip overlay.")
        return

    x, y, w, h = load_roi_config(ROI_CONFIG_FULL)
    roi_cap = crop_with_roi(captured_full, x, y, w, h)

    if roi_cap.size == 0:
        print("[ROI] roi_cap is empty. Check ROI config or camera resolution. Skip overlay.")
        return

    os.makedirs(save_dir, exist_ok=True)

    # 1. ROIで切り出した「生」カメラ画像を保存
    raw_path = os.path.join(save_dir, "roi_captured_raw.png")
    cv2.imwrite(raw_path, roi_cap)
    print(f"[ROI] Captured ROI raw image saved: {raw_path}")

    # 2. devnet でROIカメラ画像をモジュール分割 → center_crop_cam_XX.png
    outputs_cam = dev_predictor(roi_cap)
    instances_cam = outputs_cam["instances"].to("cpu")
    if len(instances_cam) == 0:
        print("[ROI] No detections on captured ROI image. Skip center_crop_cam.")
    else:
        boxes_cam = instances_cam.pred_boxes.tensor.numpy()
        center_dir = os.path.join(save_dir, "cam_center_crops")
        make_center_crops_from_boxes(roi_cap, boxes_cam, center_dir, crop_h=CROP_H, crop_w=CROP_W)
        print(f"[ROI] Camera center crops saved in: {center_dir}")

    # 3. オーバーレイ画像作成（ユーザROI vs カメラROI）
    h_o, w_o = roi_image.shape[:2]
    roi_cap_resized = cv2.resize(roi_cap, (w_o, h_o))

    red  = make_red_tint(roi_image)
    blue = make_blue_tint(roi_cap_resized)

    overlay = cv2.addWeighted(red, 0.5, blue, 0.5, 0)

    cv2.imwrite(os.path.join(save_dir, "roi_original_red.png"), red)
    cv2.imwrite(os.path.join(save_dir, "roi_captured_blue.png"), blue)
    cv2.imwrite(os.path.join(save_dir, "roi_overlay_red_orig_blue_cap.png"), overlay)
    print(f"[ROI] Overlay image saved in: {save_dir}")

    # 4. center_crop の赤青合成を生成
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

        # サイズを合わせて合成
        if img_red.shape[:2] != img_blue.shape[:2]:
            img_blue = cv2.resize(img_blue, (img_red.shape[1], img_red.shape[0]))

        overlay = cv2.addWeighted(img_red, 0.5, img_blue, 0.5, 0)
        cv2.imwrite(out_path, overlay)

    print(f"[ROI] Center crop overlay images saved in: {overlay_dir}")


# =========================================================
#   main
# =========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ===== モデル読み込み =====
    model = ModuleControllerNet(feat_dim=128).to(device)
    ckpt = torch.load(MODEL_SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

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
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75
    cfg_seg.DATASETS.TEST = ()
    seg_predictor = DefaultPredictor(cfg_seg)
    seg_metadata = MetadataCatalog.get("SEG_TRAIN")

    # ===== ROI画像をユーザに選択させる =====
    img_path = select_image_via_dialog()
    if img_path is None:
        return

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        return

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_dir = os.path.join(RESULT_ROOT, f"selected_{img_name}")
    os.makedirs(save_dir, exist_ok=True)

    # ROI画像そのまま保存
    cv2.imwrite(os.path.join(save_dir, f"{img_name}_roi_input.png"), img_bgr)

    # ===== signals.csv がある場合だけ読み込んで GT 評価 =====
    signals_df = None
    row_idx = None
    try:
        img_p = Path(img_path)
        # .../6module_dataset_max_DAC/roi/xxx.png → module_dir = .../6module_dataset_max_DAC
        module_dir = img_p.parent.parent

        if module_dir.name.endswith("module_dataset_max_DAC"):
            # "6module_dataset_max_DAC" → 6 を想定
            module_idx = int(module_dir.name[0])
            signals_path = module_dir / "signals.csv"

            print(f"[DEBUG] module_dir = {module_dir}")
            print(f"[DEBUG] signals_path = {signals_path}")

            if signals_path.exists():
                signals_df = pd.read_csv(signals_path, header=None)
                # 画像ファイル名 "57.png" → img_name="57" → row 57
                try:
                    row_idx = int(img_name)
                except ValueError:
                    print(f"[WARN] img_name is not int: {img_name}. GT bars will be skipped.")
                    row_idx = None

                if row_idx is not None:
                    print(f"[INFO] Using signals.csv: {signals_path}, row={row_idx}")
            else:
                print(f"[WARN] signals.csv not found at: {signals_path}")
        else:
            print(f"[INFO] module_dir.name does not endwith 'module_dataset_max_DAC': {module_dir.name}")
    except Exception as e:
        print(f"[WARN] Could not prepare ground truth: {e}")
        signals_df = None
        row_idx = None


    # ===== devnet でモジュール検出（選択画像） =====
    outputs = dev_predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    if len(instances) == 0:
        print(f"[INFO] No detection in {img_path}")
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

    # ===== CNN で電圧予測（選択画像） =====
    true_v, pred_v = crop_and_predict_with_cnn(
        img_bgr=img_bgr,
        boxes=boxes,
        model=model,
        device=device,
        v_stats=v_stats,
        seg_predictor=seg_predictor,
        seg_metadata=seg_metadata,
        save_dir=save_dir,
        crop_h=CROP_H,
        crop_w=CROP_W,
        signals_df=signals_df,
        row_idx=row_idx,
    )

    if pred_v is None:
        print("[INFO] No valid module predictions collected.")
        return

    np.savetxt(os.path.join(save_dir, "pred_voltages_raw.txt"), pred_v, fmt="%.4f")

    # ===== Arduino を開き、予測電圧を送信 =====
    ser = None
    try:
        ser = init_serial(SERIAL_PORT, BAUDRATE)
        send_voltages_to_arduino(ser, pred_v)

        # ===== 5秒待機してカメラで撮影 =====
        print("[INFO] Waiting 5 seconds before capturing camera image ...")
        time.sleep(5.0)

        cam_frame = capture_image_from_camera(CAMERA_INDEX)
        if cam_frame is None:
            print("[WARN] Camera capture failed, skip ROI overlay.")
        else:
            cv2.imwrite(os.path.join(save_dir, "captured_full.png"), cam_frame)
            # ここでROI切り抜き + 生保存 + devnet分割 + center_crop_cam + overlay
            save_roi_overlay(img_bgr, cam_frame, dev_predictor, save_dir)

        # ★ 終了前に必ず 0.0V を送信
        reset_voltages_to_zero(ser, n_channels=12)

    finally:
        if ser is not None:
            try:
                ser.close()
            except:
                pass

    print(f"\n[INFO] All results saved in: {save_dir}")


if __name__ == "__main__":
    main()
