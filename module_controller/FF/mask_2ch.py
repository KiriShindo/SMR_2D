# -*- coding: utf-8 -*-
"""
detectron2_rect2mask_to_2ch.py

- ROI画像を Detectron2 でセグメント
- 2本の梁のマスクを rect2mask で細長い長方形にフィット
- y座標で「上梁 / 下梁」を決めて並べ替え
- 2チャネルマスク (2, H, W) を作成
    - ch0: 上梁
    - ch1: 下梁
- 背景黒 + 上赤 / 下青 のカラー画像も保存
- mask_relation.csv に幾何情報 (center, distance, angle_diff) を保存
"""

import os
import cv2
import math
import csv
import colorsys
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ======= 固定長方形サイズ・探索パラメータ =======
RECT_W = 40
RECT_H = 4
ANGLE_SEARCH_RANGE_DEG = 60.0
ANGLE_SEARCH_STEP_DEG  = 1.0

# ======= 出力時の色 (BGR) =======
UPPER_COLOR_BGR = (0, 0, 255)   # 上梁（赤）
LOWER_COLOR_BGR = (255, 0, 0)   # 下梁（青）


def generate_distinct_colors(n):
    # もう Visualizer は使わないが、残しておくならこう
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((r, g, b))
    return colors


def mask_y_top(m):
    ys = np.where(m)[0]
    return int(ys.min()) if ys.size else 10**9


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

    best = dict(score=-1,
                angle=base_angle_deg,
                center=(W // 2, H // 2),
                rect_pts=None)

    angles = [base_angle_deg]
    if angle_search_range > 1e-6 and angle_step > 0:
        offs = np.arange(-angle_search_range, angle_search_range + 1e-9, angle_step)
        angles = [base_angle_deg + a for a in offs]

    for ang in angles:
        kern, anchor = make_rotated_rect_kernel(rect_w, rect_h, ang)
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


def main():
    # ===== Detectron2設定 =====
    train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
    train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"

    try:
        register_coco_instances("Train", {}, train_json, train_images)
    except Exception:
        # すでに登録済みの場合など
        pass

    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ()

    predictor = DefaultPredictor(cfg)
    metadata  = MetadataCatalog.get("Train")

    # ===== 入出力パス =====
    in_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/silicon/roi_aug_rot_shift"
    out_dir = os.path.join(
        r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/silicon",
        "roi_aug_rot_shift_beammask_2ch"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 幾何情報のCSV
    csv_path = os.path.join(out_dir, "mask_relation.csv")

    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    print(f"Found {len(files)} images in {in_dir}")

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow([
            "filename",
            "center_upper_x", "center_upper_y",
            "center_lower_x", "center_lower_y",
            "distance",
            "angle_upper_deg", "angle_lower_deg",
            "angle_diff_deg"
        ])

        for idx, fname in enumerate(files, start=1):
            img_path = os.path.join(in_dir, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"[WARN] Failed to load: {img_path}")
                continue

            outputs = predictor(img_bgr)
            instances = outputs["instances"].to("cpu")
            if len(instances) < 2:
                print(f"[WARN] <2 instances: {fname}, skip.")
                continue

            scores = instances.scores.numpy()
            masks_all = instances.pred_masks.numpy().astype(bool)

            # スコア上位2つを取得
            top2_idx = np.argsort(-scores)[:2]
            masks_two = masks_all[top2_idx]

            # rect2mask で細長い長方形にフィット
            rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
                masks_two,
                RECT_W, RECT_H,
                angle_search_range=ANGLE_SEARCH_RANGE_DEG,
                angle_step=ANGLE_SEARCH_STEP_DEG
            )
            if len(centers) != 2 or rect_masks.shape[0] != 2:
                print(f"[WARN] rectification failed: {fname}")
                continue

            # ===== 上下を y座標で決めて並べ替え =====
            ys = [c[1] for c in centers]  # center = (x, y)
            order = np.argsort(ys)        # 小さいyが上

            rect_masks_sorted = rect_masks[order]                 # (2, H, W)
            angles_sorted     = [used_angles[i] for i in order]
            centers_sorted    = [centers[i]     for i in order]

            # 上 / 下 として展開
            (x_up, y_up), (x_low, y_low) = centers_sorted
            angle_up, angle_low = angles_sorted

            # 中心間距離
            distance = float(np.hypot(x_low - x_up, y_low - y_up))

            # 角度差（0〜90度に正規化）
            angle_diff = abs(angle_up - angle_low)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            # ===== 2chマスクを作成 (ch0=上, ch1=下) =====
            # bool -> float32 (0/1)
            mask_upper = rect_masks_sorted[0].astype(np.float32)
            mask_lower = rect_masks_sorted[1].astype(np.float32)
            mask_2ch   = np.stack([mask_upper, mask_lower], axis=0)   # (2, H, W)

            # ===== カラー可視化画像 (背景黒 + 上赤 / 下青) =====
            H, W = mask_upper.shape
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[mask_upper.astype(bool)] = UPPER_COLOR_BGR
            vis[mask_lower.astype(bool)] = LOWER_COLOR_BGR

            stem, ext = os.path.splitext(fname)
            out_img_path = os.path.join(out_dir, f"{stem}_beammask.png")
            out_npy_path = os.path.join(out_dir, f"{stem}_beam2ch.npy")

            cv2.imwrite(out_img_path, vis)
            np.save(out_npy_path, mask_2ch.astype(np.float32))

            # CSV 行を書き出し
            writer.writerow([
                fname,
                x_up,  y_up,
                x_low, y_low,
                distance,
                angle_up, angle_low,
                angle_diff
            ])

            print(f"[{idx}/{len(files)}] Saved: {out_img_path}, {out_npy_path}")

    print(f"\n✅ All done. CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
