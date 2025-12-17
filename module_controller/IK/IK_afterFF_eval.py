import os
import math
import csv

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.data.datasets import register_coco_instances
from detectron2.data import MetadataCatalog

# ==========================================================
# パス設定
# ==========================================================

SELECTED_ROOT = (
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025"
    r"\SMR_control\module_controller\FF\result_beam2ch_devnet_arduino3"
)

# 評価結果とデバッグ画像の出力ルート（★ここに全部出す）
RESULT_ROOT = (
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025"
    r"\SMR_control\module_controller\IK\afterFF"
)
os.makedirs(RESULT_ROOT, exist_ok=True)

SIGNALS_CSV_PATH = os.path.join(SELECTED_ROOT, "signals.csv")

IK_MODEL_PATH = (
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\ik_beam2ch_babbling_500_10_model.pth"
)

SEG_TRAIN_JSON   = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/annotations.json"
SEG_TRAIN_IMAGES = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/"
SEG_WEIGHTS_PATH = r"C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/seg_dataset_devnet_1module/prm01/model_final.pth"

# rect2mask パラメータ
RECT_W = 40
RECT_H = 4
ANGLE_SEARCH_RANGE_DEG = 60.0
ANGLE_SEARCH_STEP_DEG  = 1.0

# 2chマスクの可視化色 (BGR)
UPPER_COLOR_BGR = (0, 0, 255)   # 上梁（赤）
LOWER_COLOR_BGR = (255, 0, 0)   # 下梁（青）


# ==========================================================
# IKBeamNet 定義（train_ik_beam2ch.py と同じ）
# ==========================================================

class IKBeamEncoder(nn.Module):
    def __init__(self, in_ch: int = 6, feat_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1),
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
        self.fc  = nn.Linear(128, feat_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class VoltageHead(nn.Module):
    def __init__(self, in_dim: int = 128):
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


class IKBeamNet(nn.Module):
    def __init__(self, in_ch: int = 6, feat_dim: int = 128):
        super().__init__()
        self.encoder = IKBeamEncoder(in_ch=in_ch, feat_dim=feat_dim)
        self.head    = VoltageHead(in_dim=feat_dim)

    def forward(self, x):
        feat = self.encoder(x)
        out  = self.head(feat)
        return out


# ==========================================================
# detectron2 + rect2mask 関連
# ==========================================================

def estimate_angle_deg(mask_bool: np.ndarray) -> float:
    ys, xs = np.nonzero(mask_bool)
    if xs.size == 0:
        return 0.0
    pts = np.column_stack((xs, ys)).astype(np.float32)
    mean = pts.mean(axis=0)
    pts_centered = pts - mean
    cov = np.cov(pts_centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major_vec = eigvecs[:, np.argmax(eigvals)]
    vx, vy = float(major_vec[0]), float(major_vec[1])
    ang = math.degrees(math.atan2(vy, vx))
    if ang <= -90.0:
        ang += 180.0
    elif ang > 90.0:
        ang -= 180.0
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
        return None, [], [], []
    return np.stack(rect_masks, axis=0), angles, centers, overlaps


def detect_and_rectify(img_bgr, predictor):
    outputs = predictor(img_bgr)
    instances = outputs["instances"].to("cpu")
    if len(instances) < 2:
        return False, None, None, None

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
    if rect_masks is None or len(centers) != 2 or rect_masks.shape[0] != 2:
        return False, None, None, None

    ys = [c[1] for c in centers]
    order = np.argsort(ys)
    rect_masks_sorted = rect_masks[order]
    centers_sorted    = [centers[i] for i in order]
    angles_sorted     = [used_angles[i] for i in order]

    return True, rect_masks_sorted, centers_sorted, angles_sorted


def make_beammask_upper0_2ch(img_bgr, predictor):
    """
    入力画像を回転させず、そのままの座標系で 2ch ビームマスクを生成する。

    戻り値:
        success: bool
        mask_2ch: (2, H, W) float32
                  [0] 上ビーム, [1] 下ビーム
        img_out: 元画像（回転なし、デバッグ用）
    """
    ok, rect_masks_sorted, centers_sorted, angles_sorted = detect_and_rectify(img_bgr, predictor)
    if (not ok) or (rect_masks_sorted is None) or (rect_masks_sorted.shape[0] < 2):
        return False, None, None

    mask_upper = rect_masks_sorted[0].astype(np.float32)
    mask_lower = rect_masks_sorted[1].astype(np.float32)
    mask_2ch = np.stack([mask_upper, mask_lower], axis=0)

    img_out = img_bgr.copy()

    return True, mask_2ch.astype(np.float32), img_out


def beammask_to_color(mask_2ch):
    """
    2chマスク (2, H, W) を BGR の可視化画像 (H, W, 3) に変換。
    ch0: 上 (赤), ch1: 下 (青)
    """
    mask_upper = mask_2ch[0] > 0.5
    mask_lower = mask_2ch[1] > 0.5
    H, W = mask_2ch.shape[1:]
    vis = np.zeros((H, W, 3), dtype=np.uint8)
    vis[mask_upper] = UPPER_COLOR_BGR
    vis[mask_lower] = LOWER_COLOR_BGR
    return vis


# ==========================================================
# CSV 読み込み系
# ==========================================================

def load_signals_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            rows.append([float(v) for v in row])
    return rows  # rows[n][0..11]


def load_pred_voltages_raw(path):
    vals = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            L = float(parts[0])
            R = float(parts[1])
            vals.append((L, R))
    return np.array(vals, dtype=np.float32)  # (N,2)


# ==========================================================
# グラフ描画
# ==========================================================

def plot_delta_graph(mod_ids, v_i, v_targ, v_pred, title, save_path):
    v_i   = np.asarray(v_i, dtype=np.float32)
    v_t   = np.asarray(v_targ, dtype=np.float32)
    v_p   = np.asarray(v_pred, dtype=np.float32)

    dv_t = v_t - v_i
    dv_p = v_p - v_i

    x = np.arange(len(mod_ids))

    plt.figure(figsize=(6, 4))
    plt.bar(x, dv_p, width=0.6, color="blue", alpha=0.7, label="Pred ΔV")

    for xi, y in zip(x, dv_t):
        plt.hlines(
            y, xi - 0.4, xi + 0.4,
            colors="red", linestyles="dashed", linewidth=2.5,
            label="Target ΔV" if xi == 0 else None
        )

    plt.axhline(0.0, color="black", linewidth=1.0)

    plt.xticks(x, [str(m) for m in mod_ids])
    plt.xlabel("Module index")
    plt.ylabel("ΔVoltage (V) relative to time i")
    plt.title(title)
    plt.legend()
    plt.grid(True, axis="y", linestyle="--", alpha=0.5)
    plt.ylim(-5.5, 5.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  Saved plot: {save_path}")


# ==========================================================
# メイン
# ==========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # detectron2
    try:
        register_coco_instances("Train", {}, SEG_TRAIN_JSON, SEG_TRAIN_IMAGES)
    except Exception:
        pass

    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    )
    cfg.MODEL.WEIGHTS = SEG_WEIGHTS_PATH
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.DATASETS.TEST = ()

    predictor = DefaultPredictor(cfg)
    _ = MetadataCatalog.get("Train")

    # IKBeamNet
    model = IKBeamNet(in_ch=6, feat_dim=128).to(device)
    ckpt = torch.load(IK_MODEL_PATH, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded IK model from: {IK_MODEL_PATH}")

    # signals.csv
    signals_rows = load_signals_csv(SIGNALS_CSV_PATH)
    print(f"Loaded {len(signals_rows)} rows from signals.csv")

    # selected_[n] 一覧
    dirnames = [d for d in os.listdir(SELECTED_ROOT)
                if d.startswith("selected_") and os.path.isdir(os.path.join(SELECTED_ROOT, d))]
    dirnames.sort()

    for dirname in dirnames:
        sel_dir = os.path.join(SELECTED_ROOT, dirname)
        print(f"\n=== Evaluating folder: {dirname} ===")

        # 出力先フォルダ（元 selected_[n] は触らない）
        out_dir = os.path.join(RESULT_ROOT, dirname)
        os.makedirs(out_dir, exist_ok=True)

        # selected_1 → signals.csv の 0 行目, selected_2 → 1 行目, ...
        try:
            n_str = dirname.split("_")[-1]
            n_val = int(n_str)
            n_idx = n_val - 1
        except Exception:
            print(f"  [WARN] Could not parse index from {dirname}, skip.")
            continue

        if not (0 <= n_idx < len(signals_rows)):
            print(f"  [WARN] n_idx={n_idx} out of range for signals.csv, skip.")
            continue

        row = signals_rows[n_idx]
        if len(row) < 12:
            print(f"  [WARN] signals.csv row {n_idx} has < 12 columns, skip.")
            continue

        V_L_targ_full = np.array(row[0:6], dtype=np.float32)
        V_R_targ_full = np.array(row[6:12], dtype=np.float32)

        # 時刻 i の電圧（pred_voltages_raw.txt）
        pred_volt_path = os.path.join(sel_dir, "pred_voltages_raw.txt")
        if not os.path.exists(pred_volt_path):
            print(f"  [WARN] pred_voltages_raw.txt not found in {dirname}, skip.")
            continue
        V_i_all = load_pred_voltages_raw(pred_volt_path)
        if V_i_all.shape[0] == 0:
            print(f"  [WARN] pred_voltages_raw.txt empty in {dirname}, skip.")
            continue

        # center crop ディレクトリ
        cam_dir  = os.path.join(sel_dir, "cam_center_crops")
        targ_dir = os.path.join(sel_dir, "targ_center_crops")
        if not os.path.isdir(cam_dir) or not os.path.isdir(targ_dir):
            print(f"  [WARN] center crop dirs missing in {dirname}, skip.")
            continue

        def collect_mod_files(base_dir, prefix):
            mod_files = []
            for f in os.listdir(base_dir):
                if f.lower().endswith(".png") and f.startswith(prefix):
                    name = f[len(prefix):]
                    name = os.path.splitext(name)[0]
                    try:
                        m_idx = int(name)
                    except Exception:
                        continue
                    mod_files.append((m_idx, os.path.join(base_dir, f)))
            mod_files.sort(key=lambda t: t[0])
            return mod_files

        targ_mod_files = collect_mod_files(targ_dir, "center_crop_0")
        cam_mod_files  = collect_mod_files(cam_dir,  "center_crop_cam_0")

        if len(targ_mod_files) == 0 or len(cam_mod_files) == 0:
            print(f"  [WARN] no center_crop files in {dirname}, skip.")
            continue

        targ_mod_ids = [m for m, _ in targ_mod_files]
        cam_mod_ids  = [m for m, _ in cam_mod_files]
        common_mod_ids = sorted(set(targ_mod_ids) & set(cam_mod_ids))
        if len(common_mod_ids) == 0:
            print(f"  [WARN] no common module indices in {dirname}, skip.")
            continue

        print(f"  Modules found: {common_mod_ids}")

        V_L_i_list, V_R_i_list = [], []
        V_L_targ_list, V_R_targ_list = [], []
        V_L_pred_list, V_R_pred_list = [], []

        # モジュールごとに処理
        for m_idx in common_mod_ids:
            cam_path  = dict(cam_mod_files).get(m_idx, None)
            targ_path = dict(targ_mod_files).get(m_idx, None)
            if cam_path is None or targ_path is None:
                print(f"  [WARN] missing cam/targ file for module {m_idx} in {dirname}, skip this module.")
                continue

            img_cam  = cv2.imread(cam_path)
            img_targ = cv2.imread(targ_path)
            if img_cam is None or img_targ is None:
                print(f"  [WARN] failed to load images for module {m_idx} in {dirname}, skip this module.")
                continue

            # ---- beammask生成（回転なしで2chマスク生成） ----
            ok_cam,  mask_cam_2ch,  img_cam_rot  = make_beammask_upper0_2ch(img_cam,  predictor)
            ok_targ, mask_targ_2ch, img_targ_rot = make_beammask_upper0_2ch(img_targ, predictor)
            if (not ok_cam) or (not ok_targ):
                print(f"  [WARN] beammask generation failed for module {m_idx} in {dirname}, skip this module.")
                continue

            # ---- デバッグ画像保存（出力先 out_dir） ----
            # i: cam
            cam_orig_out = os.path.join(out_dir, f"mod{m_idx}_cam_orig.png")
            cam_rot_out  = os.path.join(out_dir, f"mod{m_idx}_cam_rot.png")
            cam_mask_out = os.path.join(out_dir, f"mod{m_idx}_cam_beammask.png")
            cv2.imwrite(cam_orig_out, img_cam)
            cv2.imwrite(cam_rot_out,  img_cam_rot)  # 実際には回転していない
            cv2.imwrite(cam_mask_out, beammask_to_color(mask_cam_2ch))

            # i+1: targ
            targ_orig_out = os.path.join(out_dir, f"mod{m_idx}_targ_orig.png")
            targ_rot_out  = os.path.join(out_dir, f"mod{m_idx}_targ_rot.png")
            targ_mask_out = os.path.join(out_dir, f"mod{m_idx}_targ_beammask.png")
            cv2.imwrite(targ_orig_out, img_targ)
            cv2.imwrite(targ_rot_out,  img_targ_rot)  # 実際には回転していない
            cv2.imwrite(targ_mask_out, beammask_to_color(mask_targ_2ch))

            # ---- 時刻 i の電圧（clip 付き） ----
            if m_idx >= V_i_all.shape[0]:
                print(f"  [WARN] V_i_all has only {V_i_all.shape[0]} rows, but module index {m_idx}, skip this module.")
                continue

            vL_i = float(V_i_all[m_idx, 0])
            vR_i = float(V_i_all[m_idx, 1])

            # 0〜5V に clip
            vL_i = max(0.0, min(5.0, vL_i))
            vR_i = max(0.0, min(5.0, vR_i))

            # ---- 時刻 i+1 のターゲット電圧 ----
            if m_idx >= V_L_targ_full.shape[0] or m_idx >= V_R_targ_full.shape[0]:
                print(f"  [WARN] signals.csv row has only {V_L_targ_full.shape[0]} modules, module index {m_idx}, skip.")
                continue
            vL_targ = float(V_L_targ_full[m_idx])
            vR_targ = float(V_R_targ_full[m_idx])

            # ---- 入力 x を作成 ----
            mask_cam_tensor  = torch.from_numpy(mask_cam_2ch).float()
            mask_targ_tensor = torch.from_numpy(mask_targ_2ch).float()

            _, H, W = mask_cam_tensor.shape
            q_map = torch.zeros((2, H, W), dtype=torch.float32)
            q_map[0, :, :] = vL_i / 5.0
            q_map[1, :, :] = vR_i / 5.0

            x = torch.cat([mask_cam_tensor, mask_targ_tensor, q_map], dim=0)
            x = x.unsqueeze(0).to(device)

            # ---- 予測 ----
            with torch.no_grad():
                y_hat = model(x)
            vL_pred = float(y_hat[0, 0].cpu().item())
            vR_pred = float(y_hat[0, 1].cpu().item())

            # 必要ならここで出力も clip できる（今はそのまま）：
            # vL_pred = max(0.0, min(5.0, vL_pred))
            # vR_pred = max(0.0, min(5.0, vR_pred))

            V_L_i_list.append(vL_i)
            V_R_i_list.append(vR_i)
            V_L_targ_list.append(vL_targ)
            V_R_targ_list.append(vR_targ)
            V_L_pred_list.append(vL_pred)
            V_R_pred_list.append(vR_pred)

            print(
                f"  Module {m_idx}: "
                f"L_i={vL_i:.3f}, L_targ={vL_targ:.3f}, L_pred={vL_pred:.3f} | "
                f"R_i={vR_i:.3f}, R_targ={vR_targ:.3f}, R_pred={vR_pred:.3f}"
            )

        if len(V_L_i_list) == 0:
            print(f"  [WARN] no valid modules evaluated in {dirname}, skip plotting.")
            continue

        # ※ common_mod_ids と V_L_i_list の長さが一致している前提だが、
        #   念のため len(V_L_i_list) に合わせて切る
        mod_ids = common_mod_ids[:len(V_L_i_list)]

        left_plot_path  = os.path.join(out_dir, "ik_beam2ch_delta_left.png")
        right_plot_path = os.path.join(out_dir, "ik_beam2ch_delta_right.png")

        plot_delta_graph(
            mod_ids,
            V_L_i_list,
            V_L_targ_list,
            V_L_pred_list,
            title=f"{dirname} - Left voltage Δ",
            save_path=left_plot_path
        )
        plot_delta_graph(
            mod_ids,
            V_R_i_list,
            V_R_targ_list,
            V_R_pred_list,
            title=f"{dirname} - Right voltage Δ",
            save_path=right_plot_path
        )

    print("\nAll evaluation finished.")


if __name__ == "__main__":
    main()
