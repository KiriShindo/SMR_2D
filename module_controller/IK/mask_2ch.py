# ### マスクのoverlay画像（元画像にマスクをかけた画像）も保存するようにした
# # -*- coding: utf-8 -*-
# """
# detectron2_rect2mask_to_2ch.py

# - ROI画像を Detectron2 でセグメント
# - 2本の梁のマスクを rect2mask で細長い長方形にフィット
# - y座標で「上梁 / 下梁」を決めて並べ替え
# - 2チャネルマスク (2, H, W) を作成
#     - ch0: 上梁
#     - ch1: 下梁
# - 背景黒 + 上赤 / 下青 のカラー画像も保存
# - 元画像にマスクを重ねた overlay 画像も保存
# - mask_relation.csv に幾何情報 (center, distance, angle_diff) を保存
# """

# import os
# import cv2
# import math
# import csv
# import colorsys
# import numpy as np

# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import MetadataCatalog

# # ======= 固定長方形サイズ・探索パラメータ =======
# RECT_W = 40
# RECT_H = 4
# ANGLE_SEARCH_RANGE_DEG = 60.0
# ANGLE_SEARCH_STEP_DEG  = 1.0

# # ======= 出力時の色 (BGR) =======
# UPPER_COLOR_BGR = (0, 0, 255)   # 上梁（赤）
# LOWER_COLOR_BGR = (255, 0, 0)   # 下梁（青）


# def generate_distinct_colors(n):
#     # Visualizer 用に残しているが、現在は未使用
#     colors = []
#     for i in range(n):
#         hue = i / max(n, 1)
#         r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
#         colors.append((r, g, b))
#     return colors


# def mask_y_top(m):
#     ys = np.where(m)[0]
#     return int(ys.min()) if ys.size else 10**9


# def estimate_angle_deg(mask_bool):
#     m8 = (mask_bool.astype(np.uint8)) * 255
#     cnts, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     if not cnts:
#         return 0.0
#     cnt = max(cnts, key=cv2.contourArea)
#     (_, _), (w, h), ang = cv2.minAreaRect(cnt)
#     # 縦長に見える場合は90度足す
#     if w < h:
#         ang += 90.0
#     # [-90, 90) に正規化
#     if ang >= 90:
#         ang -= 180
#     if ang < -90:
#         ang += 180
#     return float(ang)


# def make_rotated_rect_kernel(rect_w, rect_h, angle_deg):
#     diag = int(np.ceil(np.sqrt(rect_w**2 + rect_h**2))) + 4
#     kh = diag | 1  # 奇数に
#     kw = diag | 1
#     kern = np.zeros((kh, kw), np.uint8)
#     cx, cy = kw / 2.0, kh / 2.0
#     box = cv2.boxPoints(((cx, cy), (float(rect_w), float(rect_h)), float(angle_deg)))
#     box = np.int32(np.round(box))
#     cv2.fillPoly(kern, [box], 1)
#     return kern.astype(np.float32), (kw // 2, kh // 2)


# def fit_fixed_rect_to_mask(mask_bool, rect_w, rect_h, base_angle_deg,
#                            angle_search_range=0.0, angle_step=1.0):
#     """
#     与えられたマスクに対して、固定サイズの回転長方形をスコア最大になるようにフィットする
#     """
#     H, W = mask_bool.shape
#     mask_f = mask_bool.astype(np.float32)

#     best = dict(score=-1,
#                 angle=base_angle_deg,
#                 center=(W // 2, H // 2),
#                 rect_pts=None)

#     angles = [base_angle_deg]
#     if angle_search_range > 1e-6 and angle_step > 0:
#         offs = np.arange(-angle_search_range, angle_search_range + 1e-9, angle_step)
#         angles = [base_angle_deg + a for a in offs]

#     for ang in angles:
#         kern, anchor = make_rotated_rect_kernel(rect_w, rect_h, ang)
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


# def main():
#     # ===== Detectron2設定 =====
#     train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
#     train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"

#     try:
#         register_coco_instances("Train", {}, train_json, train_images)
#     except Exception:
#         # すでに登録済みの場合など
#         pass

#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(
#         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
#     ))
#     cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01/model_final.pth"
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#     cfg.MODEL.DEVICE = "cuda"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # 必要ならあとで下げる
#     cfg.DATASETS.TEST = ()

#     predictor = DefaultPredictor(cfg)
#     metadata  = MetadataCatalog.get("Train")

#     # ===== 入出力パス =====
#     in_dir = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon\aug_shift_500_10"
#     out_dir = os.path.join(
#         r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon",
#         "roi_aug_shift_500_10_beammask_2ch"
#     )
#     os.makedirs(out_dir, exist_ok=True)

#     # 幾何情報のCSV
#     csv_path = os.path.join(out_dir, "mask_relation.csv")

#     files = [f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     files.sort()
#     print(f"Found {len(files)} images in {in_dir}")

#     with open(csv_path, "w", newline="") as fcsv:
#         writer = csv.writer(fcsv)
#         writer.writerow([
#             "filename",
#             "center_upper_x", "center_upper_y",
#             "center_lower_x", "center_lower_y",
#             "distance",
#             "angle_upper_deg", "angle_lower_deg",
#             "angle_diff_deg"
#         ])

#         for idx, fname in enumerate(files, start=1):
#             img_path = os.path.join(in_dir, fname)
#             img_bgr = cv2.imread(img_path)
#             if img_bgr is None:
#                 print(f"[WARN] Failed to load: {img_path}")
#                 continue

#             outputs = predictor(img_bgr)
#             instances = outputs["instances"].to("cpu")
#             if len(instances) < 2:
#                 print(f"[WARN] <2 instances: {fname}, skip.")
#                 continue

#             scores = instances.scores.numpy()
#             masks_all = instances.pred_masks.numpy().astype(bool)

#             # スコア上位2つを取得
#             top2_idx = np.argsort(-scores)[:2]
#             masks_two = masks_all[top2_idx]

#             # rect2mask で細長い長方形にフィット
#             rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
#                 masks_two,
#                 RECT_W, RECT_H,
#                 angle_search_range=ANGLE_SEARCH_RANGE_DEG,
#                 angle_step=ANGLE_SEARCH_STEP_DEG
#             )
#             if len(centers) != 2 or rect_masks.shape[0] != 2:
#                 print(f"[WARN] rectification failed: {fname}")
#                 continue

#             # ===== 上下を y座標で決めて並べ替え =====
#             ys = [c[1] for c in centers]  # center = (x, y)
#             order = np.argsort(ys)        # 小さいyが上

#             rect_masks_sorted = rect_masks[order]                 # (2, H, W)
#             angles_sorted     = [used_angles[i] for i in order]
#             centers_sorted    = [centers[i]     for i in order]

#             # 上 / 下 として展開
#             (x_up, y_up), (x_low, y_low) = centers_sorted
#             angle_up, angle_low = angles_sorted

#             # 中心間距離
#             distance = float(np.hypot(x_low - x_up, y_low - y_up))

#             # 角度差（0〜90度に正規化）
#             angle_diff = abs(angle_up - angle_low)
#             if angle_diff > 90:
#                 angle_diff = 180 - angle_diff

#             # ===== 2chマスクを作成 (ch0=上, ch1=下) =====
#             mask_upper = rect_masks_sorted[0].astype(np.float32)
#             mask_lower = rect_masks_sorted[1].astype(np.float32)
#             mask_2ch   = np.stack([mask_upper, mask_lower], axis=0)   # (2, H, W)

#             # ===== カラー可視化画像 (背景黒 + 上赤 / 下青) =====
#             H, W = mask_upper.shape
#             vis = np.zeros((H, W, 3), dtype=np.uint8)
#             vis[mask_upper.astype(bool)] = UPPER_COLOR_BGR
#             vis[mask_lower.astype(bool)] = LOWER_COLOR_BGR

#             # ===== 元画像にマスクを重ねた確認画像 =====
#             overlay = img_bgr.copy()
#             alpha = 0.5  # マスクの透過率
#             upper_bool = mask_upper.astype(bool)
#             lower_bool = mask_lower.astype(bool)

#             overlay[upper_bool] = (
#                 overlay[upper_bool] * (1.0 - alpha)
#                 + np.array(UPPER_COLOR_BGR, dtype=np.float32) * alpha
#             )
#             overlay[lower_bool] = (
#                 overlay[lower_bool] * (1.0 - alpha)
#                 + np.array(LOWER_COLOR_BGR, dtype=np.float32) * alpha
#             )
#             overlay = np.clip(overlay, 0, 255).astype(np.uint8)

#             # ===== 出力パス =====
#             stem, ext = os.path.splitext(fname)
#             out_img_path      = os.path.join(out_dir, f"{stem}_beammask.png")
#             out_npy_path      = os.path.join(out_dir, f"{stem}_beam2ch.npy")
#             out_overlay_path  = os.path.join(out_dir, f"{stem}_overlay.png")

#             # 保存
#             cv2.imwrite(out_img_path, vis)
#             cv2.imwrite(out_overlay_path, overlay)
#             np.save(out_npy_path, mask_2ch.astype(np.float32))

#             # CSV 行を書き出し
#             writer.writerow([
#                 fname,
#                 x_up,  y_up,
#                 x_low, y_low,
#                 distance,
#                 angle_up, angle_low,
#                 angle_diff
#             ])

#             print(f"[{idx}/{len(files)}] Saved: {out_img_path}, {out_npy_path}, {out_overlay_path}")

#     print(f"\n✅ All done. CSV saved → {csv_path}")


# if __name__ == "__main__":
#     main()





### 上側マスクの角度が0°になるように回転させてから保存する
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
    # Visualizer 用に残しているが、現在は未使用
    colors = []
    for i in range(n):
        hue = i / max(n, 1)
        r, g, b = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
        colors.append((r, g, b))
    return colors


def estimate_angle_deg(mask_bool):
    """
    マスク領域の長軸方向を PCA で推定し，
    そのベクトルと画像 x 軸との角度を返す（度数法・[-90, 90] に正規化）。

    ここで 0° = 長軸が水平（画像の横端と平行）
           90° = 長軸が垂直
    """
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return 0.0

    pts = np.column_stack((xs, ys)).astype(np.float32)  # (N, 2) [x, y]
    mean = pts.mean(axis=0)
    pts_centered = pts - mean

    # 共分散行列 (2x2)
    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 固有値昇順

    # 長軸方向（最大固有値に対応する固有ベクトル）
    major_vec = eigvecs[:, np.argmax(eigvals)]  # [vx, vy]
    vx, vy = float(major_vec[0]), float(major_vec[1])

    # 画像の x 軸に対する角度
    ang = math.degrees(math.atan2(vy, vx))  # [-180, 180]

    # [-90, 90] に正規化（180 度反転は同じ向きとみなす）
    if ang <= -90.0:
        ang += 180.0
    elif ang > 90.0:
        ang -= 180.0

    return float(ang)


def make_rotated_rect_kernel(rect_w, rect_h, angle_deg):
    diag = int(np.ceil(np.sqrt(rect_w**2 + rect_h**2))) + 4
    kh = diag | 1  # 奇数に
    kw = diag | 1
    kern = np.zeros((kh, kw), np.uint8)
    cx, cy = kw / 2.0, kh / 2.0
    box = cv2.boxPoints(((cx, cy), (float(rect_w), float(rect_h)), float(angle_deg)))
    box = np.int32(np.round(box))
    cv2.fillPoly(kern, [box], 1)
    return kern.astype(np.float32), (kw // 2, kh // 2)


def fit_fixed_rect_to_mask(mask_bool, rect_w, rect_h, base_angle_deg,
                           angle_search_range=0.0, angle_step=1.0):
    """
    与えられたマスクに対して、固定サイズの回転長方形をスコア最大になるようにフィットする
    """
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
        base_ang = estimate_angle_deg(m)  # PCA ベースで長軸角度を取得
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
        return None, [], [], []
    return np.stack(rect_masks, axis=0), angles, centers, overlaps


def detect_and_rectify(img_bgr, predictor):
    """
    画像に対して detectron2 で2つのマスクを検出し、
    rect2maskで細長い長方形にフィットして、
    「上(小さいy) / 下(大きいy)」の順に並べて返す。

    戻り値:
        success: False のときは検出失敗
        rect_masks_sorted: (2, H, W) の bool 配列
        centers_sorted:    [(x_up, y_up), (x_low, y_low)]
        angles_sorted:     [angle_up_deg, angle_low_deg]
    """
    outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    if len(instances) < 2:
        return False, None, None, None

    scores = instances.scores.numpy()
    masks_all = instances.pred_masks.numpy().astype(bool)

    # スコア上位2つ
    top2_idx   = np.argsort(-scores)[:2]
    masks_two  = masks_all[top2_idx]

    # rect2mask
    rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
        masks_two,
        RECT_W, RECT_H,
        angle_search_range=ANGLE_SEARCH_RANGE_DEG,
        angle_step=ANGLE_SEARCH_STEP_DEG
    )
    if rect_masks is None or len(centers) != 2 or rect_masks.shape[0] != 2:
        return False, None, None, None

    # 上下を y 座標で並べ替え
    ys     = [c[1] for c in centers]      # center = (x, y)
    order  = np.argsort(ys)              # 小さい y が上
    rect_masks_sorted = rect_masks[order]
    centers_sorted    = [centers[i] for i in order]
    angles_sorted     = [used_angles[i] for i in order]

    return True, rect_masks_sorted, centers_sorted, angles_sorted


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
    in_dir = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon\aug_shift_1000_20"
    out_dir = os.path.join(
        r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\silicon",
        "roi_aug_shift_1000_20_beammask_2ch_upper0"
    )
    os.makedirs(out_dir, exist_ok=True)

    # 幾何情報のCSV（元画像座標系）
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

            H0, W0, _ = img_bgr.shape

            # =====================================================
            # ① 元画像で一度 detectron2 → 「生マスク」から角度を計算
            # =====================================================
            outputs1 = predictor(img_bgr)
            instances1 = outputs1["instances"].to("cpu")
            if len(instances1) < 2:
                print(f"[WARN] <2 instances (orig): {fname}")
                continue

            scores1 = instances1.scores.numpy()
            masks_all1 = instances1.pred_masks.numpy().astype(bool)

            # スコア上位2つ
            top2_idx1 = np.argsort(-scores1)[:2]
            masks_two1 = masks_all1[top2_idx1]

            # 各マスクの中心 & 角度（PCAベース）を計算
            centers_raw = []
            angles_raw  = []
            for m in masks_two1:
                ys, xs = np.nonzero(m)
                if xs.size == 0:
                    cx, cy = W0 / 2.0, H0 / 2.0
                else:
                    cx, cy = float(xs.mean()), float(ys.mean())
                centers_raw.append((cx, cy))
                angles_raw.append(estimate_angle_deg(m))

            # y が小さい方を「上」とみなす
            order1 = np.argsort([c[1] for c in centers_raw])  # 小さい y が上
            centers_sorted_raw1 = [centers_raw[i] for i in order1]
            angles_sorted_raw1  = [angles_raw[i]  for i in order1]

            (x_up, y_up), (x_low, y_low) = centers_sorted_raw1
            angle_up_orig  = angles_sorted_raw1[0]
            angle_low_orig = angles_sorted_raw1[1]

            # 中心間距離（元画像座標系）
            distance = float(math.hypot(x_low - x_up, y_low - y_up))

            # 角度差（0〜90度に正規化, 元画像座標系）
            angle_diff = abs(angle_up_orig - angle_low_orig)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            # =====================================================
            # ② 上ビームの長軸が 0°（水平）になるように元画像を回転
            #    → 例えば angle_up_orig=+3° なら rot_angle=-3° で回転
            # =====================================================
            rot_angle = angle_up_orig
            center_img = (W0 / 2.0, H0 / 2.0)
            M = cv2.getRotationMatrix2D(center_img, rot_angle, 1.0)

            img_rot = cv2.warpAffine(
                img_bgr, M, (W0, H0),
                flags=cv2.INTER_LINEAR,
                borderValue=(0, 0, 0)  # 新規領域は黒
            )

            # ===== デバッグ用に「回転前」と「回転後」の画像も保存 =====
            stem, ext = os.path.splitext(fname)
            debug_orig_path = os.path.join(out_dir, f"{stem}_orig.png")
            debug_rot_path  = os.path.join(out_dir, f"{stem}_rot.png")
            cv2.imwrite(debug_orig_path, img_bgr)
            cv2.imwrite(debug_rot_path, img_rot)


            # =====================================================
            # ③ 回転後の画像に対して、再度 detectron2 + rect2mask
            #     → ここから先のマスク / beammask / npy は「回転後」基準
            # =====================================================
            ok2, rect_masks_sorted_rot, centers_sorted_rot, angles_sorted_rot = \
                detect_and_rectify(img_rot, predictor)
            if not ok2:
                print(f"[WARN] detect_and_rectify (rot) failed: {fname}")
                continue

            # 回転後の上/下マスク（ここでも y が小さい方が「上」になるようにソート済み）
            mask_upper = rect_masks_sorted_rot[0].astype(np.float32)
            mask_lower = rect_masks_sorted_rot[1].astype(np.float32)
            H, W = mask_upper.shape

            # ===== 回転後画像にマスクを重ねたオーバーレイ =====
            overlay = img_rot.copy()
            alpha = 0.5
            upper_bool = mask_upper.astype(bool)
            lower_bool = mask_lower.astype(bool)

            overlay[upper_bool] = (
                overlay[upper_bool] * (1.0 - alpha)
                + np.array(UPPER_COLOR_BGR, dtype=np.float32) * alpha
            )
            overlay[lower_bool] = (
                overlay[lower_bool] * (1.0 - alpha)
                + np.array(LOWER_COLOR_BGR, dtype=np.float32) * alpha
            )
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)

            # ===== 2chマスク（回転後） =====
            mask_2ch = np.stack([mask_upper, mask_lower], axis=0)  # (2, H, W)

            # ===== beammask 可視化画像（回転後マスクから作成） =====
            vis = np.zeros((H, W, 3), dtype=np.uint8)
            vis[upper_bool] = UPPER_COLOR_BGR
            vis[lower_bool] = LOWER_COLOR_BGR

            # ===== 出力パス =====
            stem, ext = os.path.splitext(fname)
            out_img_path      = os.path.join(out_dir, f"{stem}_beammask.png")
            out_npy_path      = os.path.join(out_dir, f"{stem}_beam2ch.npy")
            out_overlay_path  = os.path.join(out_dir, f"{stem}_overlay.png")

            # ===== 保存（全部「回転後」の情報」） =====
            cv2.imwrite(out_img_path, vis)
            cv2.imwrite(out_overlay_path, overlay)
            np.save(out_npy_path, mask_2ch.astype(np.float32))

            # ===== CSV は「元画像座標系」の幾何情報を保存 =====
            writer.writerow([
                fname,
                x_up,  y_up,
                x_low, y_low,
                distance,
                angle_up_orig, angle_low_orig,
                angle_diff
            ])

            print(f"[{idx}/{len(files)}] Saved: {out_img_path}, {out_npy_path}, {out_overlay_path}")

    print(f"\n✅ All done. CSV saved → {csv_path}")


if __name__ == "__main__":
    main()
