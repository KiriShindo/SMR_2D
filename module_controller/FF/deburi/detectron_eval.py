## 梁マスク
import os
import cv2
import math
import csv
import colorsys
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ======= 固定長方形サイズ・探索パラメータ =======
RECT_W = 40
RECT_H = 4
ANGLE_SEARCH_RANGE_DEG = 60.0
ANGLE_SEARCH_STEP_DEG = 1.0

def generate_distinct_colors(n):
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
    best = dict(score=-1, angle=base_angle_deg, center=(W // 2, H // 2), rect_pts=None)
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
    # Detectron2設定
    train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet/annotations.json"
    train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet/"
    try:
        register_coco_instances("Train", {}, train_json, train_images)
    except Exception:
        pass
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet/prm01/model_final.pth"
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cuda"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ()
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get("Train")

    in_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/silicon/roi_aug_rot_shift"
    out_dir = os.path.join(
        r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/silicon",
        "roi_aug_rot_shift_rect2mask"
    )
    csv_path = os.path.join(out_dir, "mask_relation.csv")
    os.makedirs(out_dir, exist_ok=True)

    files = [f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    files.sort()
    print(f"Found {len(files)} images in {in_dir}")

    with open(csv_path, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["filename", "center1_x", "center1_y", "center2_x", "center2_y",
                         "distance", "angle1_deg", "angle2_deg", "angle_diff_deg"])

        for idx, fname in enumerate(files, start=1):
            img_path = os.path.join(in_dir, fname)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue

            outputs = predictor(img_bgr)
            instances = outputs["instances"].to("cpu")
            if len(instances) < 2:
                continue

            scores = instances.scores.numpy()
            masks_all = instances.pred_masks.numpy().astype(bool)
            top2_idx = np.argsort(-scores)[:2]
            masks_two = masks_all[top2_idx]

            rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
                masks_two,
                RECT_W, RECT_H,
                angle_search_range=ANGLE_SEARCH_RANGE_DEG,
                angle_step=ANGLE_SEARCH_STEP_DEG
            )
            if len(centers) != 2:
                continue

            (x1, y1), (x2, y2) = centers
            distance = float(np.hypot(x2 - x1, y2 - y1))
            angle1, angle2 = used_angles
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff

            colors = generate_distinct_colors(2)
            vis = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, scale=1.0)
            out_vis = vis.overlay_instances(
                masks=rect_masks,
                boxes=None,
                labels=None,
                assigned_colors=colors
            )
            result_bgr = out_vis.get_image()[:, :, ::-1].astype("uint8")

            out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_rect2mask.png")
            cv2.imwrite(out_path, result_bgr)

            writer.writerow([fname, x1, y1, x2, y2, distance, angle1, angle2, angle_diff])
            print(f"[{idx}/{len(files)}] Saved + logged: {fname}")

    print(f"/n✅ All done. CSV saved → {csv_path}")

if __name__ == "__main__":
    main()


#  梁を画像の端まで伸ばした時の4点の交点の座標を保存
# import os
# import cv2
# import math
# import csv
# import colorsys
# import numpy as np
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import MetadataCatalog

# # ======= 固定長方形サイズ・探索パラメータ =======
# RECT_W = 40
# RECT_H = 4
# ANGLE_SEARCH_RANGE_DEG = 60.0
# ANGLE_SEARCH_STEP_DEG = 1.0

# def generate_distinct_colors(n):
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
#     # 長辺をx方向に
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

# # ===== 画像境界との交点を計算 =====
# def compute_line_border_intersections(center, angle_deg, img_h, img_w):
#     """
#     center = (cx, cy)
#     angle_deg: 長軸方向の角度（estimate_angle_degと同じ定義, 画像座標系で）
#     画像の枠（x=0, x=img_w-1, y=0, y=img_h-1）との交点2点を返す。
#     戻り値: [(x1, y1), (x2, y2)] （float）
#     """
#     cx, cy = center
#     theta = math.radians(angle_deg)
#     dx = math.cos(theta)
#     dy = math.sin(theta)

#     # dx, dy がどちらもほぼ0みたいな異常ケースを防ぐ
#     if abs(dx) < 1e-8 and abs(dy) < 1e-8:
#         return [(cx, cy), (cx, cy)]

#     intersections = []

#     # 各境界との交点を計算（x=0, x=W-1, y=0, y=H-1）
#     # p = c + t * d
#     # x = cx + t dx, y = cy + t dy

#     # x = 0
#     if abs(dx) > 1e-8:
#         t = (0.0 - cx) / dx
#         y = cy + t * dy
#         if 0.0 <= y <= (img_h - 1):
#             intersections.append((t, 0.0, y))

#     # x = W-1
#     if abs(dx) > 1e-8:
#         t = ((img_w - 1) - cx) / dx
#         y = cy + t * dy
#         if 0.0 <= y <= (img_h - 1):
#             intersections.append((t, float(img_w - 1), y))

#     # y = 0
#     if abs(dy) > 1e-8:
#         t = (0.0 - cy) / dy
#         x = cx + t * dx
#         if 0.0 <= x <= (img_w - 1):
#             intersections.append((t, x, 0.0))

#     # y = H-1
#     if abs(dy) > 1e-8:
#         t = ((img_h - 1) - cy) / dy
#         x = cx + t * dx
#         if 0.0 <= x <= (img_w - 1):
#             intersections.append((t, x, float(img_h - 1)))

#     if len(intersections) < 2:
#         # 何かおかしいときは中心点を2つ返しておく
#         return [(cx, cy), (cx, cy)]

#     # tでソートして両端の2点を採用
#     intersections.sort(key=lambda v: v[0])
#     p1 = (intersections[0][1], intersections[0][2])
#     p2 = (intersections[-1][1], intersections[-1][2])
#     return [p1, p2]

# def main():
#     # Detectron2設定
#     train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset/annotations.json"
#     train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset/"
#     try:
#         register_coco_instances("Train", {}, train_json, train_images)
#     except Exception:
#         pass

#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset/prm01/model_final.pth"
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
#     cfg.MODEL.DEVICE = "cuda"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#     cfg.DATASETS.TEST = ()
#     predictor = DefaultPredictor(cfg)
#     metadata = MetadataCatalog.get("Train")

#     in_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/silicon/roi_aug_rot_shift_scale"
#     out_dir = os.path.join(
#         r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/silicon",
#         "roi_aug_rot_shift_scale_rect2mask_4point"
#     )
#     csv_path = os.path.join(out_dir, "mask_relation.csv")
#     os.makedirs(out_dir, exist_ok=True)

#     files = [f for f in os.listdir(in_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#     files.sort()
#     print(f"Found {len(files)} images in {in_dir}")

#     with open(csv_path, "w", newline="") as fcsv:
#         writer = csv.writer(fcsv)
#         writer.writerow([
#             "filename",
#             "center1_x", "center1_y", "center2_x", "center2_y",
#             "distance",
#             "angle1_deg", "angle2_deg", "angle_diff_deg",
#             "mask1_p1_x", "mask1_p1_y", "mask1_p2_x", "mask1_p2_y",
#             "mask2_p1_x", "mask2_p1_y", "mask2_p2_x", "mask2_p2_y"
#         ])

#         for idx, fname in enumerate(files, start=1):
#             img_path = os.path.join(in_dir, fname)
#             img_bgr = cv2.imread(img_path)
#             if img_bgr is None:
#                 continue

#             H, W = img_bgr.shape[:2]

#             outputs = predictor(img_bgr)
#             instances = outputs["instances"].to("cpu")
#             if len(instances) < 2:
#                 continue

#             scores = instances.scores.numpy()
#             masks_all = instances.pred_masks.numpy().astype(bool)
#             top2_idx = np.argsort(-scores)[:2]
#             masks_two = masks_all[top2_idx]

#             rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
#                 masks_two,
#                 RECT_W, RECT_H,
#                 angle_search_range=ANGLE_SEARCH_RANGE_DEG,
#                 angle_step=ANGLE_SEARCH_STEP_DEG
#             )
#             if len(centers) != 2:
#                 continue

#             # マスク1とマスク2（top2_idxの順）として扱う
#             (x1, y1), (x2, y2) = centers
#             angle1, angle2 = used_angles

#             # 重心距離＆角度差
#             distance = float(np.hypot(x2 - x1, y2 - y1))
#             angle_diff = abs(angle1 - angle2)
#             if angle_diff > 90:
#                 angle_diff = 180 - angle_diff

#             # 画像枠との交点を計算（各マスクにつき2点）
#             m1_p1, m1_p2 = compute_line_border_intersections((x1, y1), angle1, H, W)
#             m2_p1, m2_p2 = compute_line_border_intersections((x2, y2), angle2, H, W)

#             # 可視化：長方形マスク + 直線を重ねて保存
#             colors = generate_distinct_colors(2)
#             vis = Visualizer(img_bgr[:, :, ::-1], metadata=metadata, scale=1.0)
#             out_vis = vis.overlay_instances(
#                 masks=rect_masks,
#                 boxes=None,
#                 labels=None,
#                 assigned_colors=colors
#             )
#             result_bgr = out_vis.get_image()[:, :, ::-1].astype("uint8")

#             # 直線を描画（mask1, mask2）
#             c1 = (0, 255, 0)
#             c2 = (0, 0, 255)
#             cv2.line(
#                 result_bgr,
#                 (int(round(m1_p1[0])), int(round(m1_p1[1]))),
#                 (int(round(m1_p2[0])), int(round(m1_p2[1]))),
#                 c1, 2
#             )
#             cv2.line(
#                 result_bgr,
#                 (int(round(m2_p1[0])), int(round(m2_p1[1]))),
#                 (int(round(m2_p2[0])), int(round(m2_p2[1]))),
#                 c2, 2
#             )

#             out_path = os.path.join(out_dir, f"{os.path.splitext(fname)[0]}_rect2mask.png")
#             cv2.imwrite(out_path, result_bgr)

#             # CSV書き込み
#             writer.writerow([
#                 fname,
#                 x1, y1, x2, y2,
#                 distance,
#                 angle1, angle2, angle_diff,
#                 m1_p1[0], m1_p1[1], m1_p2[0], m1_p2[1],
#                 m2_p1[0], m2_p1[1], m2_p2[0], m2_p2[1]
#             ])

#             print(f"[{idx}/{len(files)}] Saved + logged: {fname}")

#     print(f"\n✅ All done. CSV saved → {csv_path}")

# if __name__ == "__main__":
#     main()
