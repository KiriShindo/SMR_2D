# -*- coding: utf-8 -*-
"""
eval_ik_beam2ch_all.py  （ランダムペア + 一定割合同orig_index版）

評価①:
  - 訓練データ & テストデータの全サンプルからランダムペアを構成し，
      ・Left: 真値 vs 予測値の散布図
      ・Right: 真値 vs 予測値の散布図
      ・Left: (pred - true) の符号付き誤差ヒストグラム
      ・Right: (pred - true) の符号付き誤差ヒストグラム
    → train/test 別々に出力（合計 8枚）

評価②:
  - 訓練データ & テストデータそれぞれから，
    ランダムペアを 10 個サンプルし，各ペアごとにフォルダを作成して，
      ・bar.png  （ΔVの棒グラフ: 真値=赤/オレンジ破線, 予測=青バー）
      ・i_raw.png   : マスク前画像 (anchor 側)
      ・j_raw.png   : マスク前画像 (partner 側)
      ・i_mask.png  : マスク後 (beammask) 画像 (anchor 側)
      ・j_mask.png  : マスク後 (beammask) 画像 (partner 側)
    を保存する。

前提:
  - train/test でそれぞれ signals_aug.csv, *_beam2ch.npy, *_beammask.png が存在すること
  - signals_aug.csv は augment_random_affine_babbling.py と同様の形式:
      filename, orig_index, Left_V, Right_V
"""

import os
import csv
import random
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn

# ==========================================================
# パス設定
# ==========================================================

# ---- 訓練側 ----
TRAIN_BEAMMASK_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\roi_aug_shift_1000_20_beammask_2ch_upper0"
)
TRAIN_AUG_IMG_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\aug_shift_1000_20"
)
TRAIN_CSV_PATH = TRAIN_AUG_IMG_DIR / "signals_aug.csv"

# ---- テスト側 ----
# TEST_BEAMMASK_DIR = Path(
#     r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
#     r"\module_controller\IK\1module_babbling_data\silicon\roi_aug_shift_500_10_beammask_2ch_upper0_test"
# )
# TEST_AUG_IMG_DIR = Path(
#     r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
#     r"\module_controller\IK\1module_babbling_data\silicon\aug_shift_500_10_test"
# )
# TEST_CSV_PATH = TEST_AUG_IMG_DIR / "signals_aug.csv"

# 学習済みモデル
MODEL_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\ik_beam2ch_babbling_1000_20_shift_upper0_randompair_model.pth"
)

# 出力ベースフォルダ
OUT_BASE_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\IK_only"
)
OUT_SCATTER_DIR = OUT_BASE_DIR / "scatter_hist"
OUT_EXAMPLES_DIR = OUT_BASE_DIR / "examples"
OUT_SCATTER_DIR.mkdir(parents=True, exist_ok=True)
OUT_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# ランダムペアにおける「同じ orig_index ペア」の割合
P_SAME = 0.1  # 10%

# 評価②で出すサンプル数
NUM_EXAMPLES = 10

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)

# ==========================================================
# モデル定義（学習時と同じ）
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
# サンプル読み込み & ランダムペア構成
# ==========================================================

def load_aug_samples(beammask_dir: Path, csv_aug_path: Path):
    """
    signals_aug.csv を読み込んで、各行を 1 サンプルとして扱う。
    戻り値: samples(list of dict)
      dict:
        - filename
        - orig_index
        - npy_path
        - vL, vR
    """
    beammask_dir = Path(beammask_dir)

    with open(csv_aug_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) == 0:
        raise RuntimeError(f"{csv_aug_path} にデータ行がありません。")

    samples = []
    for row in rows:
        if not all(k in row for k in ["filename", "orig_index", "Left_V", "Right_V"]):
            continue

        fname = row["filename"]
        stem  = os.path.splitext(fname)[0]
        npy_path = beammask_dir / f"{stem}_beam2ch.npy"
        if not npy_path.exists():
            # beammask がないデータは使わない
            continue

        try:
            orig_idx = int(row["orig_index"])
            vL = float(row["Left_V"])
            vR = float(row["Right_V"])
        except Exception:
            continue

        samples.append({
            "filename": fname,
            "orig_index": orig_idx,
            "npy_path": npy_path,
            "vL": vL,
            "vR": vR,
        })

    if len(samples) < 2:
        raise RuntimeError("有効なサンプルが 2 件未満のため、ペア構成ができません。")

    # orig_index -> [indices] の辞書も作っておく（ペア生成で利用）
    indices_by_orig = defaultdict(list)
    for idx, s in enumerate(samples):
        indices_by_orig[s["orig_index"]].append(idx)

    if len(indices_by_orig) < 1:
        raise RuntimeError("orig_index が 1 種類も見つかりません。")

    print(f"[INFO] load_aug_samples: 有効サンプル数 = {len(samples)}, uniq orig_index = {len(indices_by_orig)}")
    return samples, indices_by_orig


def build_random_pairs(samples, indices_by_orig, p_same: float, num_pairs: int = None):
    """
    ランダムに anchor/partner ペアを num_pairs 個生成する。

    - anchor: サンプル i
    - partner:
        確率 p_same   : 同じ orig_index の別サンプル（可能なら）
        確率 1-p_same : orig_index が異なるサンプル

    戻り値: pairs (list of dict)
      dict:
        - fname_i, fname_j
        - npy_i, npy_j
        - q_i  = (vL_i, vR_i)
        - q_j  = (vL_j, vR_j)
        - orig_i, orig_j
        - same_orig (bool)
    """
    N = len(samples)
    if num_pairs is None:
        num_pairs = N  # デフォルトはサンプル数と同じだけペアを作る

    pairs = []

    # anchor として使うインデックス（なるべく均等に使うため、周期的にまわす）
    anchor_indices = list(range(N))
    if num_pairs <= N:
        # num_pairs が小さい場合はランダムサブセット
        anchor_indices = random.sample(anchor_indices, num_pairs)
    else:
        # num_pairs が大きい場合は繰り返し使用
        times = num_pairs // N
        rest  = num_pairs % N
        anchor_indices = anchor_indices * times + random.sample(list(range(N)), rest)

    for idx in anchor_indices:
        item_i = samples[idx]
        orig_i = item_i["orig_index"]

        use_same = (random.random() < p_same)

        # --- 同じ orig_index から選ぶ場合 ---
        if use_same and len(indices_by_orig[orig_i]) >= 2:
            cand = indices_by_orig[orig_i]
            while True:
                j = random.choice(cand)
                if j != idx:
                    break
        else:
            # --- 異なる orig_index から選ぶ場合 ---
            while True:
                j = random.randint(0, N - 1)
                if samples[j]["orig_index"] != orig_i:
                    break

        item_j = samples[j]

        pairs.append({
            "fname_i": item_i["filename"],
            "fname_j": item_j["filename"],
            "npy_i":   item_i["npy_path"],
            "npy_j":   item_j["npy_path"],
            "q_i":     (item_i["vL"], item_i["vR"]),
            "q_j":     (item_j["vL"], item_j["vR"]),
            "orig_i":  orig_i,
            "orig_j":  item_j["orig_index"],
            "same_orig": (orig_i == item_j["orig_index"]),
        })

    print(f"[INFO] build_random_pairs: 作成ペア数 = {len(pairs)} (p_same={p_same})")
    return pairs

# ==========================================================
# 評価①: 散布図 & ヒスト
# ==========================================================

def eval_scatter_and_hist_for_split(
    tag: str,
    beammask_dir: Path,
    csv_path: Path,
    model: IKBeamNet,
    device,
    p_same: float,
):
    """
    tag: "train" or "test"

    ・全サンプル (N) からランダムペアを N 個構成し，
      Left/Right それぞれについて:
        - 真値 vs 予測値散布図
        - (pred - true) のヒスト
    """
    print(f"\n=== {tag.upper()} scatter & hist ===")

    samples, indices_by_orig = load_aug_samples(beammask_dir, csv_path)
    pairs = build_random_pairs(samples, indices_by_orig, p_same=p_same, num_pairs=len(samples))

    y_true_L = []
    y_pred_L = []
    y_true_R = []
    y_pred_R = []

    model.eval()
    with torch.no_grad():
        for s in pairs:
            # 入力テンソル作成
            mask_i = np.load(str(s["npy_i"])).astype(np.float32)
            mask_j = np.load(str(s["npy_j"])).astype(np.float32)

            vL_i, vR_i = s["q_i"]
            vL_true, vR_true = s["q_j"]

            mask_i_t = torch.from_numpy(mask_i)    # (2,H,W)
            mask_j_t = torch.from_numpy(mask_j)    # (2,H,W)

            _, H, W = mask_i_t.shape
            q_map = torch.zeros((2, H, W), dtype=torch.float32)
            q_map[0, :, :] = vL_i / 5.0
            q_map[1, :, :] = vR_i / 5.0

            x = torch.cat([mask_i_t, mask_j_t, q_map], dim=0).unsqueeze(0).to(device)

            y_hat = model(x)  # (1,2)
            vL_pred, vR_pred = y_hat.squeeze(0).cpu().numpy().tolist()

            y_true_L.append(vL_true)
            y_pred_L.append(vL_pred)
            y_true_R.append(vR_true)
            y_pred_R.append(vR_pred)

    y_true_L = np.array(y_true_L)
    y_pred_L = np.array(y_pred_L)
    y_true_R = np.array(y_true_R)
    y_pred_R = np.array(y_pred_R)

    # ---- Left: scatter ----
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_L, y_pred_L, alpha=0.3, s=5)
    min_v = min(y_true_L.min(), y_pred_L.min(), 0.0)
    max_v = max(y_true_L.max(), y_pred_L.max(), 5.0)
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.0)
    plt.xlabel("True V_L [V]")
    plt.ylabel("Pred V_L [V]")
    plt.title(f"{tag.capitalize()} Left: True vs Pred")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_SCATTER_DIR / f"{tag}_left_scatter.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

    # ---- Left: hist(error) ----
    err_L = y_pred_L - y_true_L
    plt.figure(figsize=(5, 4))
    plt.hist(err_L, bins=50, alpha=0.7)
    plt.axvline(0.0, color='k', linestyle='--')
    plt.xlabel("Pred - True [V]")
    plt.ylabel("Count")
    plt.title(f"{tag.capitalize()} Left: Error Histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_SCATTER_DIR / f"{tag}_left_error_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

    # ---- Right: scatter ----
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true_R, y_pred_R, alpha=0.3, s=5)
    min_v = min(y_true_R.min(), y_pred_R.min(), 0.0)
    max_v = max(y_true_R.max(), y_pred_R.max(), 5.0)
    plt.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1.0)
    plt.xlabel("True V_R [V]")
    plt.ylabel("Pred V_R [V]")
    plt.title(f"{tag.capitalize()} Right: True vs Pred")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_SCATTER_DIR / f"{tag}_right_scatter.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

    # ---- Right: hist(error) ----
    err_R = y_pred_R - y_true_R
    plt.figure(figsize=(5, 4))
    plt.hist(err_R, bins=50, alpha=0.7)
    plt.axvline(0.0, color='k', linestyle='--')
    plt.xlabel("Pred - True [V]")
    plt.ylabel("Count")
    plt.title(f"{tag.capitalize()} Right: Error Histogram")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = OUT_SCATTER_DIR / f"{tag}_right_error_hist.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Saved: {out_path}")

# ==========================================================
# 評価②: 10サンプルのバー + 画像保存
# ==========================================================

def plot_bar_and_save(v_i, v_true, v_pred, out_path):
    """
    v_i, v_true, v_pred: (2,) [L, R]
    - y=0 を「anchor の電圧」とみなした ΔV 表示。
      ΔV_true = V_partner_true - V_anchor
      ΔV_pred = V_partner_pred - V_anchor
    """
    vL_i, vR_i       = v_i
    vL_true, vR_true = v_true
    vL_pred, vR_pred = v_pred

    dL_true = vL_true - vL_i
    dR_true = vR_true - vR_i
    dL_pred = vL_pred - vL_i
    dR_pred = vR_pred - vR_i

    x = np.arange(2)  # 0:Left, 1:Right
    width = 0.4

    plt.figure(figsize=(5, 4))

    # 予測 ΔV を棒グラフ
    plt.bar(x, [dL_pred, dR_pred], width, label="Pred ΔV", alpha=0.7)

    # 真値 ΔV を赤/オレンジの点線
    plt.hlines(dL_true, x[0] - width/2, x[0] + width/2,
               linestyles="dashed", label="True ΔV (Left)", color="red")
    plt.hlines(dR_true, x[1] - width/2, x[1] + width/2,
               linestyles="dashed", label="True ΔV (Right)", color="orange")

    plt.axhline(0.0, color="k", linewidth=1.0)  # anchor 電圧を 0 とした基準線
    plt.xticks(x, ["Left", "Right"])
    plt.ylabel("ΔV [V] (partner - anchor)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_images_for_pair(pair, aug_img_dir: Path, beammask_dir: Path, out_dir: Path):
    """
    pair: build_random_pairs で作った 1 ペア(dict)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fname_i = pair["fname_i"]
    fname_j = pair["fname_j"]

    stem_i = os.path.splitext(fname_i)[0]
    stem_j = os.path.splitext(fname_j)[0]

    # マスク前画像（aug画像）
    img_i_path = aug_img_dir / fname_i
    img_j_path = aug_img_dir / fname_j

    # マスク後画像（beammask 可視化）
    mask_i_vis_path = beammask_dir / f"{stem_i}_beammask.png"
    mask_j_vis_path = beammask_dir / f"{stem_j}_beammask.png"

    def try_copy(src: Path, dst: Path, label: str):
        if src.exists():
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(dst), img)
            else:
                print(f"[WARN] 画像読み込み失敗 ({label}): {src}")
        else:
            print(f"[WARN] 画像ファイルが見つかりません ({label}): {src}")

    try_copy(img_i_path, out_dir / "i_raw.png", "i_raw")
    try_copy(img_j_path, out_dir / "j_raw.png", "j_raw")
    try_copy(mask_i_vis_path, out_dir / "i_mask.png", "i_mask")
    try_copy(mask_j_vis_path, out_dir / "j_mask.png", "j_mask")


def evaluate_examples_for_split(
    tag: str,
    beammask_dir: Path,
    aug_img_dir: Path,
    csv_path: Path,
    model: IKBeamNet,
    device,
    p_same: float,
):
    """
    tag: "train" or "test"
    ランダムペアから NUM_EXAMPLES 個サンプルして可視化。
    """
    print(f"\n=== {tag.upper()} examples ===")

    samples, indices_by_orig = load_aug_samples(beammask_dir, csv_path)
    pairs = build_random_pairs(samples, indices_by_orig, p_same=p_same, num_pairs=max(NUM_EXAMPLES * 2, NUM_EXAMPLES))

    # NUM_EXAMPLES 個だけ選ぶ
    if len(pairs) <= NUM_EXAMPLES:
        chosen = pairs
    else:
        chosen = random.sample(pairs, NUM_EXAMPLES)

    for i, pair in enumerate(chosen):
        print(f"[{tag}] example {i+1}/{len(chosen)}  (orig_i={pair['orig_i']}, orig_j={pair['orig_j']}, same={pair['same_orig']})")

        # 入力テンソル x を構成
        mask_i = np.load(str(pair["npy_i"])).astype(np.float32)
        mask_j = np.load(str(pair["npy_j"])).astype(np.float32)
        vL_i, vR_i = pair["q_i"]
        vL_true, vR_true = pair["q_j"]

        mask_i_t = torch.from_numpy(mask_i)    # (2,H,W)
        mask_j_t = torch.from_numpy(mask_j)    # (2,H,W)

        _, H, W = mask_i_t.shape
        q_map = torch.zeros((2, H, W), dtype=torch.float32)
        q_map[0, :, :] = vL_i / 5.0
        q_map[1, :, :] = vR_i / 5.0

        x = torch.cat([mask_i_t, mask_j_t, q_map], dim=0).unsqueeze(0).to(device)  # (1,6,H,W)

        # 推論
        model.eval()
        with torch.no_grad():
            y_hat = model(x)  # (1,2)
        vL_pred, vR_pred = y_hat.squeeze(0).cpu().numpy().tolist()

        # 出力フォルダ
        out_dir = OUT_EXAMPLES_DIR / f"{tag}_sample_{i:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1) バーグラフ
        bar_path = out_dir / "bar.png"
        plot_bar_and_save(
            v_i=(vL_i, vR_i),
            v_true=(vL_true, vR_true),
            v_pred=(vL_pred, vR_pred),
            out_path=bar_path
        )

        # 2) 画像（マスク前後）
        save_images_for_pair(pair, aug_img_dir, beammask_dir, out_dir)

    print(f"→ {tag} 側の例を {OUT_EXAMPLES_DIR} 配下に保存しました。")

# ==========================================================
# メイン
# ==========================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # モデルロード
    model = IKBeamNet(in_ch=6, feat_dim=128).to(device)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded model from: {MODEL_PATH}")

    # --- 評価①: 散布図 & ヒスト ---
    eval_scatter_and_hist_for_split(
        tag="train",
        beammask_dir=TRAIN_BEAMMASK_DIR,
        csv_path=TRAIN_CSV_PATH,
        model=model,
        device=device,
        p_same=P_SAME,
    )
    # eval_scatter_and_hist_for_split(
    #     tag="test",
    #     beammask_dir=TEST_BEAMMASK_DIR,
    #     csv_path=TEST_CSV_PATH,
    #     model=model,
    #     device=device,
    #     p_same=P_SAME,
    # )

    # --- 評価②: 10サンプルのバー + 画像 ---
    evaluate_examples_for_split(
        tag="train",
        beammask_dir=TRAIN_BEAMMASK_DIR,
        aug_img_dir=TRAIN_AUG_IMG_DIR,
        csv_path=TRAIN_CSV_PATH,
        model=model,
        device=device,
        p_same=P_SAME,
    )
    # evaluate_examples_for_split(
    #     tag="test",
    #     beammask_dir=TEST_BEAMMASK_DIR,
    #     aug_img_dir=TEST_AUG_IMG_DIR,
    #     csv_path=TEST_CSV_PATH,
    #     model=model,
    #     device=device,
    #     p_same=P_SAME,
    # )

    print("\n✅ All evaluation done.")


if __name__ == "__main__":
    main()
