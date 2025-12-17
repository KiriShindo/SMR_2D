# import os
# import cv2
# import colorsys
# import random
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import MetadataCatalog

# def generate_distinct_colors(n):
#     """
#     Generate n visually distinct colors using HSV space.
#     Returns list of RGB tuples normalized to [0,1] for Matplotlib.
#     """
#     colors = []
#     for i in range(n):
#         hue = i / n
#         r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
#         colors.append((r, g, b))
#     return colors

# def main():
#     # データセット登録
#     train_json   = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/annotations.json"
#     train_images = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/"
#     register_coco_instances("Train", {}, train_json, train_images)

#     # cfg 設定
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(
#         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.WEIGHTS                   = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/prm01/model_final.pth"
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES     = 1
#     cfg.MODEL.DEVICE                    = "cuda"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#     cfg.DATASETS.TEST = ()

#     predictor = DefaultPredictor(cfg)

#     # 画像読み込み
#     img_path = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/6module_dataset_max_DAC/roi/22.png"
#     img = cv2.imread(img_path)
#     if img is None:
#         raise FileNotFoundError(f"Image not found: {img_path}")

#     # 推論
#     outputs = predictor(img)
#     instances = outputs["instances"].to("cpu")
#     masks = instances.pred_masks.numpy()  # [N, H, W]

#     # 色生成（normalized RGB floats）
#     num_masks = masks.shape[0]
#     assigned_colors = generate_distinct_colors(num_masks)

#     # マスクのみ描画
#     metadata = MetadataCatalog.get("Train")
#     vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
#     out_vis = vis.overlay_instances(
#         masks=masks,
#         boxes=None,
#         labels=None,
#         assigned_colors=assigned_colors
#     )

#     # 結果表示・保存
#     result = out_vis.get_image()[:, :, ::-1].astype("uint8")
#     cv2.imshow("Distinct Colors Segmentation", result)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

#     out_path = "C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/result/module_masks_distinct.png"
#     cv2.imwrite(out_path, result)
#     print(f"Saved mask-only result with distinct colors to {out_path}")

# if __name__ == "__main__":
#     main()



# ### ランダムなモジュール画像から10枚ランダムに選択 ###
# import os
# import cv2
# import colorsys
# import random
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import MetadataCatalog

# def generate_distinct_colors(n):
#     """Generate n visually distinct colors using HSV space."""
#     colors = []
#     for i in range(n):
#         hue = i / n
#         r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
#         colors.append((r, g, b))
#     return colors


# def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
#     """
#     base_dir: C:/Users/.../SMR_control/devnet_data
#     n_folders: 1〜n_folders の範囲からランダムに選択
#     n_samples: 抽出する画像数
#     return: ファイルパスのリスト
#     """
#     random_paths = []
#     for _ in range(n_samples):
#         # ランダムな i を選ぶ
#         i = random.randint(1, n_folders)
#         roi_dir = os.path.join(base_dir, f"{i}module_dataset_max_DAC", "roi")

#         if not os.path.exists(roi_dir):
#             print(f"[WARN] Not found: {roi_dir}")
#             continue

#         # 拡張子フィルタ
#         files = [f for f in os.listdir(roi_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#         if not files:
#             print(f"[WARN] No images in {roi_dir}")
#             continue

#         # ランダム選択
#         chosen = random.choice(files)
#         random_paths.append(os.path.join(roi_dir, chosen))

#     return random_paths


# def main():
#     # ====== データセット登録 ======
#     train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/annotations.json"
#     train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/"
#     register_coco_instances("Train", {}, train_json, train_images)

#     # ====== cfg設定 ======
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(
#         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/prm01/model_final.pth"
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#     cfg.MODEL.DEVICE = "cuda"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#     cfg.DATASETS.TEST = ()
#     predictor = DefaultPredictor(cfg)

#     # ====== 評価画像をランダムに選択 ======
#     base_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data"
#     eval_images = get_random_eval_images(base_dir, n_folders=6, n_samples=10)
#     print("Randomly selected evaluation images:")
#     for p in eval_images:
#         print(" -", p)

#     metadata = MetadataCatalog.get("Train")

#     # ====== 推論ループ ======
#     for idx, img_path in enumerate(eval_images):
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[WARN] Image not found: {img_path}")
#             continue

#         outputs = predictor(img)
#         instances = outputs["instances"].to("cpu")
#         if len(instances) == 0:
#             print(f"[INFO] No detection in {img_path}")
#             continue

#         masks = instances.pred_masks.numpy()
#         num_masks = masks.shape[0]
#         assigned_colors = generate_distinct_colors(num_masks)

#         vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
#         out_vis = vis.overlay_instances(
#             masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
#         )

#         result = out_vis.get_image()[:, :, ::-1].astype("uint8")

#         out_path = os.path.join(
#             base_dir, "result", f"rand_eval_{idx:02d}.png"
#         )
#         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#         cv2.imwrite(out_path, result)
#         print(f"Saved: {out_path}")

#     print("✅ Finished 10 random evaluations.")


# if __name__ == "__main__":
#     main()





### モジュール画像切り取り ###
# import os
# import cv2
# import colorsys
# import random
# from detectron2.config import get_cfg
# from detectron2 import model_zoo
# from detectron2.engine import DefaultPredictor
# from detectron2.utils.visualizer import Visualizer
# from detectron2.data.datasets import register_coco_instances
# from detectron2.data import MetadataCatalog


# def generate_distinct_colors(n):
#     """Generate n visually distinct colors using HSV space."""
#     colors = []
#     for i in range(n):
#         hue = i / n
#         r, g, b = colorsys.hsv_to_rgb(hue, 1, 1)
#         colors.append((r, g, b))
#     return colors


# def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
#     """
#     base_dir: データルート (e.g., .../SMR_control/devnet_data)
#     n_folders: 1〜n_folders の範囲から選ぶ
#     n_samples: 抽出する画像数
#     return: [(img_path, module_idx)] のリスト
#     """
#     random_entries = []
#     for _ in range(n_samples):
#         i = random.randint(1, n_folders)
#         roi_dir = os.path.join(base_dir, f"{i}module_dataset_max_DAC", "roi")

#         if not os.path.exists(roi_dir):
#             print(f"[WARN] Not found: {roi_dir}")
#             continue

#         files = [f for f in os.listdir(roi_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
#         if not files:
#             print(f"[WARN] No images in {roi_dir}")
#             continue

#         chosen = random.choice(files)
#         img_path = os.path.join(roi_dir, chosen)
#         random_entries.append((img_path, i))  # ← モジュール番号も保持

#     return random_entries


# def crop_and_save(img, boxes, save_dir):
#     """
#     各bboxをcropして保存（上から順に番号付け）
#     img: ndarray (BGR)
#     boxes: Nx4 tensor (x1, y1, x2, y2)
#     save_dir: 出力フォルダ
#     """
#     os.makedirs(save_dir, exist_ok=True)
#     H, W, _ = img.shape

#     # --- y座標順（上→下）にソート ---
#     boxes_sorted = sorted(boxes, key=lambda b: b[1])  # y1が小さい順

#     for idx, box in enumerate(boxes_sorted):
#         x1, y1, x2, y2 = [int(v) for v in box]
#         x1 = max(0, min(x1, W - 1))
#         x2 = max(0, min(x2, W - 1))
#         y1 = max(0, min(y1, H - 1))
#         y2 = max(0, min(y2, H - 1))

#         crop = img[y1:y2, x1:x2]
#         out_path = os.path.join(save_dir, f"crop_{idx:02d}.png")
#         cv2.imwrite(out_path, crop)
#     print(f"Saved {len(boxes)} cropped images (sorted top→bottom) → {save_dir}")


# def main():
#     # ====== データセット登録 ======
#     train_json   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/annotations.json"
#     train_images = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/"
#     register_coco_instances("Train", {}, train_json, train_images)

#     # ====== cfg設定 ======
#     cfg = get_cfg()
#     cfg.merge_from_file(model_zoo.get_config_file(
#         "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
#     cfg.MODEL.WEIGHTS = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data/all_dataset/prm01/model_final.pth"
#     cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
#     cfg.MODEL.DEVICE = "cuda"
#     cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
#     cfg.DATASETS.TEST = ()
#     predictor = DefaultPredictor(cfg)

#     # ====== 評価画像をランダムに選択 ======
#     base_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devnet_data"
#     eval_entries = get_random_eval_images(base_dir, n_folders=6, n_samples=10)
#     print("Randomly selected evaluation images:")
#     for p, i in eval_entries:
#         print(f" - {p} (module {i})")

#     metadata = MetadataCatalog.get("Train")
#     result_root = os.path.join(base_dir, "result")
#     os.makedirs(result_root, exist_ok=True)

#     # ====== 推論ループ ======
#     for idx, (img_path, module_idx) in enumerate(eval_entries):
#         img = cv2.imread(img_path)
#         if img is None:
#             print(f"[WARN] Image not found: {img_path}")
#             continue

#         outputs = predictor(img)
#         instances = outputs["instances"].to("cpu")

#         if len(instances) == 0:
#             print(f"[INFO] No detection in {img_path}")
#             continue

#         boxes = instances.pred_boxes.tensor.numpy()
#         masks = instances.pred_masks.numpy()
#         num_masks = masks.shape[0]
#         assigned_colors = generate_distinct_colors(num_masks)

#         vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
#         out_vis = vis.overlay_instances(
#             masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
#         )
#         result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")

#         # ====== 保存フォルダ名に module番号を含める ======
#         img_name = os.path.splitext(os.path.basename(img_path))[0]
#         save_dir = os.path.join(result_root, f"module{module_idx}_{img_name}")
#         os.makedirs(save_dir, exist_ok=True)

#         # 可視化画像保存
#         vis_path = os.path.join(save_dir, f"{img_name}_vis.png")
#         cv2.imwrite(vis_path, result_img)
#         print(f"Saved visualization → {vis_path}")

#         # bbox 切り出し（上から順に保存）
#         crop_and_save(img, boxes, save_dir)

#     print("✅ Finished 10 random evaluations with top→bottom sorted crops.")


# if __name__ == "__main__":
#     main()






### モジュールコントローラの入力画像サイズにpadding（アスペクト比の維持）＋ノイズ補間 ###
# -*- coding: utf-8 -*-
"""
detectron2_bbox_crop_center_blend.py
- Detectron2を用いて全体画像からモジュールを検出
- bboxごとに切り出し（crop）＋中心50×62領域を生成
- はみ出し領域はノイズ + distance transform で自然に補完
- 結果はモジュール番号付きフォルダに保存
"""
import os
import cv2
import colorsys
import random
import numpy as np
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ===== 背景色の統計（BGR） =====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)   # 必ず更新
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)   # 必ず更新
BLEND_WIDTH = 5.0  # 3〜10で調整可


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


# ===== ランダム評価画像抽出 =====
def get_random_eval_images(base_dir, n_folders=6, n_samples=10):
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


# ===== bbox切り出し＋ブレンド補完 =====
def crop_and_save(img, boxes, save_dir, crop_h=62, crop_w=50):
    """
    各bboxをcropして保存（上から順に番号付け）
    - crop_XX.png        : 元のbboxをそのまま切り出した画像
    - center_crop_XX.png : bbox中心まわり62x50を、ノイズ + distance transform で補完した画像
    """
    os.makedirs(save_dir, exist_ok=True)
    H, W, _ = img.shape

    boxes_sorted = sorted(boxes, key=lambda b: b[1])  # y1が小さい順
    half_h = crop_h // 2
    half_w = crop_w // 2

    for idx, box in enumerate(boxes_sorted):
        x1, y1, x2, y2 = [int(v) for v in box]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # 中心基準の切り出し範囲
        x1c, x2c = cx - half_w, cx + half_w
        y1c, y2c = cy - half_h, cy + half_h

        cropped = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
        mask    = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # 元画像内での重なり部分を計算
        x1_src, y1_src = max(0, x1c), max(0, y1c)
        x2_src, y2_src = min(W, x2c), min(H, y2c)

        # ペースト先（cropped側）
        x1_dst = x1_src - x1c
        y1_dst = y1_src - y1c
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)

        if x2_src > x1_src and y2_src > y1_src:
            cropped[y1_dst:y2_dst, x1_dst:x2_dst] = img[y1_src:y2_src, x1_src:x2_src]
            mask[y1_dst:y2_dst, x1_dst:x2_dst] = 255

        # --- augment_random_affine_roi と同じ境界補完 ---
        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]

        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        # ---------------------------------------------

        # bbox全体のcrop（参考）
        full_crop = img[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]

        # 保存
        crop_path        = os.path.join(save_dir, f"crop_{idx:02d}.png")
        center_crop_path = os.path.join(save_dir, f"center_crop_{idx:02d}.png")
        cv2.imwrite(crop_path, full_crop)
        cv2.imwrite(center_crop_path, blended)

    print(f"Saved {len(boxes_sorted)} cropped + center-crop images → {save_dir}")


# ===== メイン処理 =====
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

    # ===== ランダム画像選択 =====
    base_dir = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data"
    eval_entries = get_random_eval_images(base_dir, n_folders=6, n_samples=10)
    print("Randomly selected evaluation images:")
    for p, i in eval_entries:
        print(f" - {p} (module {i})")

    metadata = MetadataCatalog.get("Train")
    result_root = os.path.join(base_dir, "result\devnet_only")
    os.makedirs(result_root, exist_ok=True)

    # ===== 推論ループ =====
    for idx, (img_path, module_idx) in enumerate(eval_entries):
        img = cv2.imread(img_path)
        if img is None:
            print(f"[WARN] Image not found: {img_path}")
            continue

        outputs = predictor(img)
        instances = outputs["instances"].to("cpu")

        if len(instances) == 0:
            print(f"[INFO] No detection in {img_path}")
            continue

        boxes = instances.pred_boxes.tensor.numpy()
        masks = instances.pred_masks.numpy()
        num_masks = masks.shape[0]
        assigned_colors = generate_distinct_colors(num_masks)

        vis = Visualizer(img[:, :, ::-1], metadata=metadata, scale=1.0)
        out_vis = vis.overlay_instances(
            masks=masks, boxes=None, labels=None, assigned_colors=assigned_colors
        )
        result_img = out_vis.get_image()[:, :, ::-1].astype("uint8")

        # 保存ディレクトリ（例: module3_22）
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_dir = os.path.join(result_root, f"module{module_idx}_{img_name}")
        os.makedirs(save_dir, exist_ok=True)

        vis_path = os.path.join(save_dir, f"{img_name}_vis.png")
        cv2.imwrite(vis_path, result_img)
        print(f"Saved visualization → {vis_path}")

        # bboxごとにcrop + blend
        crop_and_save(img, boxes, save_dir, crop_h=62, crop_w=50)

    print("✅ Finished 10 random evaluations with center-blended 62x50 crops.")


    # ===== center_crop 画像の集約保存 =====
    merged_dir = os.path.join(base_dir, "result", "merged_center_crops")
    os.makedirs(merged_dir, exist_ok=True)

    print(f"\nCollecting all center_crop_*.png into → {merged_dir}")

    # result_root配下の全フォルダを再帰的に探索
    idx = 1
    for root, dirs, files in os.walk(result_root):
        for f in sorted(files):
            if f.startswith("center_crop_") and f.lower().endswith(".png"):
                src_path = os.path.join(root, f)
                dst_path = os.path.join(merged_dir, f"{idx:04d}.png")
                cv2.imwrite(dst_path, cv2.imread(src_path))
                idx += 1

    print(f"✅ Collected {idx-1} center_crop images into {merged_dir}")



if __name__ == "__main__":
    main()
