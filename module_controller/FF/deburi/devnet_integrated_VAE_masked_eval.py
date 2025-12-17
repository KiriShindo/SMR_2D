# -*- coding: utf-8 -*-
"""
eval_vae_with_devnet_integrated.py

- 全体ROI画像を devnet (detectron2) でモジュール分割
- 各 bbox から center_crop (+ノイズ補完) を作成
- center_crop に対して seg-net を適用し，
  rect2mask コードと同じように細長い矩形マスク(rect_masks)を作成
- Visualizer + generate_distinct_colors でマスクを着色した画像を作り，
  その「マスク付きモジュール画像」を VAE に入力して電圧を予測
- 各元画像ごとに:
    - 元のroi画像
    - devnet の分割可視化画像
    - center_crop画像
    - segマスクがかかったモジュール画像
    - 再構成結果 (center_crop の recon グリッド)
    - モジュールごとの電圧誤差バー
- 全 center_crop の「サンプリングされた潜在 z」を集めて、
  FF_only の latent_z.npy から fit した PCA の上に
  module1_3, module4_8, module6_6 の3点だけを重ねてプロット
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
from FF_VAE_masked_train import (
    VAEControllerNet,
    LATENT_DIM,
)

# ===== パス設定 ======================================
BASE_DEVNET_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\devision_net\devnet_data"

# 1〜6 module 用 signals.csv & roi は
#   BASE_DEVNET_DIR / f"{i}module_dataset_max_DAC" / "roi"
#   BASE_DEVNET_DIR / f"{i}module_dataset_max_DAC" / "signals.csv"
# にある前提

# FF_only 側で保存した VAE 潜在（★muではなく z を読む）
FF_ONLY_LATENT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_vae_masked\FF_only"
FF_ONLY_LATENT_PATH = os.path.join(FF_ONLY_LATENT_DIR, "latent_z.npy")

# VAE モデルの重み
MODEL_SAVE_PATH = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\module_controller_vae_masked.pth"

# 今回の devnet 統合 VAE 評価結果
RESULT_ROOT = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\result_vae_masked\devnet_integrated"
os.makedirs(RESULT_ROOT, exist_ok=True)

# ---- devnet (分割ネット) 用 detectron2 設定 ----
DEVNET_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/annotations.json"
DEVNET_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/"
DEVNET_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/devision_net/devnet_data/all_dataset/prm01/model_final.pth"

# ---- seg-net (固定矩形マスク用) detectron2 設定 ----
SEG_TRAIN_JSON = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset/annotations.json"
SEG_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset/"
SEG_WEIGHT = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset/prm01/model_final.pth"

# ==== 背景ノイズ用統計（必要なら更新）====
MEAN_BGR = np.array([128.04, 131.38, 132.08], dtype=np.float32)
STD_BGR  = np.array([  6.56,   6.47,   5.51], dtype=np.float32)
BLEND_WIDTH = 5.0  # distanceTransform のスケール

# ===== rect2mask 用パラメータ =====
RECT_W = 40
RECT_H = 4
ANGLE_SEARCH_RANGE_DEG = 60.0
ANGLE_SEARCH_STEP_DEG = 1.0


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


# ===== rect2mask 系関数（rect2maskコードと同じ挙動） =====
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


# ===== center_crop + ノイズ補完 & segマスク → VAE推論 =====
def crop_and_predict_with_vae(
    img_bgr,
    boxes,
    module_idx,
    row_idx,
    signals_df,
    vae_model,
    device,
    transform,
    seg_predictor,
    seg_metadata,
    save_dir,
    crop_h=62,
    crop_w=50,
):
    """
    - devnet の bbox をもとに center_crop + ノイズ補完
    - 各 center_crop に対して seg-net を適用し，
      rect2mask コードと同じ色でマスクをかけた画像を生成
    - その「マスク付きモジュール画像」を VAE に入力して電圧予測
    - 真値電圧を signals_df から取得
    - 再構成グリッド & 電圧誤差バーを保存

    戻り値:
        all_mu: (num_modules, LATENT_DIM) の numpy
        all_z : (num_modules, LATENT_DIM) の numpy （※サンプリング z）
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
    orig_tensors = []   # VAEに実際に入れた「マスク付き」画像
    recon_tensors = []
    mus_list = []
    zs_list = []

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

        # ノイズ + distance transform で補完（従来通り）
        noise = sample_border_noise(crop_h, crop_w)
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
        alpha = np.clip(dist / BLEND_WIDTH, 0.0, 1.0).astype(np.float32)
        alpha_3 = alpha[..., None]
        blended = cropped.astype(np.float32) * alpha_3 + noise.astype(np.float32) * (1 - alpha_3)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # center_crop画像（マスクをかける前）も保存
        center_crop_path = os.path.join(save_dir, f"center_crop_{n:02d}.png")
        cv2.imwrite(center_crop_path, blended)

        # ==== seg-net でマスクをかける（rect2mask と同じ色の付け方）====
        seg_img = blended.copy()
        outputs_seg = seg_predictor(seg_img)
        instances_seg = outputs_seg["instances"].to("cpu")

        if len(instances_seg) >= 1:
            scores = instances_seg.scores.numpy()
            masks_all = instances_seg.pred_masks.numpy().astype(bool)
            topk = min(2, len(masks_all))
            topk_idx = np.argsort(-scores)[:topk]
            masks_two = masks_all[topk_idx]

            rect_masks, used_angles, centers, overlaps = rectify_all_to_fixed_rects(
                masks_two,
                RECT_W, RECT_H,
                angle_search_range=ANGLE_SEARCH_RANGE_DEG,
                angle_step=ANGLE_SEARCH_STEP_DEG
            )

            colors = generate_distinct_colors(rect_masks.shape[0])
            vis_seg = Visualizer(seg_img[:, :, ::-1], metadata=seg_metadata, scale=1.0)
            out_vis_seg = vis_seg.overlay_instances(
                masks=rect_masks,
                boxes=None,
                labels=None,
                assigned_colors=colors
            )
            seg_masked_bgr = out_vis_seg.get_image()[:, :, ::-1].astype("uint8")
        else:
            # セグが失敗したら，fallbackとして blended をそのまま使用
            print(f"  [WARN] seg-net: no mask for module {n}, use blended image.")
            seg_masked_bgr = blended

        # セグマスク付き画像を保存
        segmask_path = os.path.join(save_dir, f"center_crop_{n:02d}_segmask.png")
        cv2.imwrite(segmask_path, seg_masked_bgr)
        # ==============================================

        # ---- VAE へ入力（マスク付き画像を使う） ----
        img_rgb = cv2.cvtColor(seg_masked_bgr, cv2.COLOR_BGR2RGB)
        pil_img = T.functional.to_pil_image(img_rgb)
        pil_img = pil_img.resize((50, 62))  # (W=50, H=62) に揃える
        tensor = transform(pil_img).unsqueeze(0).to(device)  # (1,3,62,50)

        with torch.no_grad():
            x_recon, mu, logvar, pred = vae_model(tensor)

            # ★ mu, logvar から「サンプリングされた z」を生成
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std  # (1, LATENT_DIM)

        pred = pred.cpu().numpy()[0]     # [V_left, V_right]
        mu_np = mu.cpu().numpy()[0]      # (LATENT_DIM,)
        z_np  = z.cpu().numpy()[0]       # (LATENT_DIM,)
        mus_list.append(mu_np)
        zs_list.append(z_np)

        # 再構成画像ログ用（VAE に投げた画像基準）
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

        # 上段: orig(マスク付き), 下段: recon
        grid = make_grid(
            torch.cat([orig_batch, recon_batch], dim=0),
            nrow=len(orig_tensors),
            padding=2
        )
        grid_np = grid.permute(1, 2, 0).numpy()
        plt.figure(figsize=(len(orig_tensors) * 2, 4))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.title("Reconstruction (top: seg-masked input, bottom: recon)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "reconstruction_grid.png"))
        plt.close()

    # ---- 電圧誤差のバーグラフ ----
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

    # ===== devnet (分割ネット) detectron2 設定 =====
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

    # ===== seg-net detectron2 設定（rect2mask 用）=====
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
    all_mu_devnet = []        # （必要なら解析用）
    all_z_devnet = []         # Devnet 側の全 center_crop の z
    selected_z_devnet = []    # module1_3, 4_8, 6_6用

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

        # devnet 推論
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

        # 保存ディレクトリ（例: module3_22）
        save_dir = os.path.join(RESULT_ROOT, f"module{module_idx}_{img_name}")
        os.makedirs(save_dir, exist_ok=True)

        # 元ROI画像 & devnet分割可視化画像を保存
        roi_copy_path = os.path.join(save_dir, f"{img_name}_roi.png")
        vis_path = os.path.join(save_dir, f"{img_name}_vis.png")
        cv2.imwrite(roi_copy_path, img_bgr)
        cv2.imwrite(vis_path, result_img)

        print(f"Saved ROI + devnet visualization → {save_dir}")

        # center_crop + segマスク + VAE + 電圧誤差評価
        mu_dev, z_dev = crop_and_predict_with_vae(
            img_bgr=img_bgr,
            boxes=boxes,
            module_idx=module_idx,
            row_idx=row_idx,
            signals_df=signals_df,
            vae_model=vae_model,
            device=device,
            transform=transform,
            seg_predictor=seg_predictor,
            seg_metadata=seg_metadata,
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
        print("[INFO] No selected latent z found for PCA plot (module1_3, 4_8, 6_6).")
    else:
        # --- 各画像の z を 1 点に代表化（モジュール方向に平均） ---
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

            # --- オーバーレイ散布図（FF_only z: グレー, devnetの3点 z: 赤＋ラベル） ---
            plt.figure(figsize=(6, 5))
            plt.scatter(
                ff_z2d[:, 0], ff_z2d[:, 1],
                s=5, alpha=0.2, color="gray", label="FF_only (sampled z)"
            )
            plt.scatter(
                selected_z2d[:, 0], selected_z2d[:, 1],
                s=120, alpha=0.95, color="red", edgecolor="black",
                label="Devnet (3 forced samples, seg-masked, mean z)"
            )

            labels = ["1", "2", "3"]
            for (x, y, label) in zip(selected_z2d[:, 0], selected_z2d[:, 1], labels):
                plt.text(x + 0.02, y + 0.02, label, fontsize=10, fontweight="bold", color="red")

            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Latent PCA (sampled z: FF_only vs 3 Devnet-segmasked samples)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            overlay_path = os.path.join(RESULT_ROOT, "latent_z_pca_selected_overlay.png")
            plt.savefig(overlay_path)
            plt.close()
            print(f"Saved PCA overlay for selected devnet z → {overlay_path}")

            # 2D座標も保存（解析用）
            np.save(os.path.join(RESULT_ROOT, "latent_z_ff_only_pca2d.npy"), ff_z2d)
            np.save(os.path.join(RESULT_ROOT, "latent_z_selected3_devnet_segmasked_pca2d.npy"), selected_z2d)

            # 寄与率ログ（任意）
            explained = pca.explained_variance_ratio_
            pd.DataFrame({
                "PC": [1, 2],
                "explained_variance_ratio": explained,
            }).to_csv(
                os.path.join(RESULT_ROOT, "latent_z_pca_explained_variance_selected_segmasked.csv"),
                index=False
            )

    # ===== Devnet 側 z 全体のみで PCA 2D（おまけ） =====
    if len(all_z_devnet) > 0:
        all_z = np.concatenate(all_z_devnet, axis=0)  # (N_all_center_crops, D)
        try:
            pca_dev = PCA(n_components=2)
            z2d_dev = pca_dev.fit_transform(all_z)

            plt.figure(figsize=(6, 5))
            plt.scatter(z2d_dev[:, 0], z2d_dev[:, 1], s=5, alpha=0.6, color="tab:blue")
            plt.xlabel("PC1")
            plt.ylabel("PC2")
            plt.title("Latent Space (sampled z, Devnet-segmasked, PCA 2D)")
            plt.grid(True)
            plt.tight_layout()
            out_zpca_path = os.path.join(RESULT_ROOT, "latent_z_pca2d_devnet_segmasked.png")
            plt.savefig(out_zpca_path)
            plt.close()
            print(f"Saved PCA of sampled z (Devnet-segmasked) → {out_zpca_path}")

            # 2D座標 & 寄与率 & 生 z も保存
            np.save(os.path.join(RESULT_ROOT, "latent_z_devnet_segmasked_pca2d.npy"), z2d_dev)
            explained_dev = pca_dev.explained_variance_ratio_
            pd.DataFrame({
                "PC": [1, 2],
                "explained_variance_ratio": explained_dev,
            }).to_csv(
                os.path.join(RESULT_ROOT, "latent_z_pca_explained_variance_devnet_segmasked.csv"),
                index=False
            )
            np.save(os.path.join(RESULT_ROOT, "latent_z_devnet_segmasked.npy"), all_z)

        except Exception as e:
            print(f"[WARN] PCA(z, Devnet) failed: {e}")
    else:
        print("[INFO] No z collected for PCA (Devnet side).")

    print("Done.")


if __name__ == "__main__":
    main()
