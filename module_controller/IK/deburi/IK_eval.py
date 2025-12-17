# -*- coding: utf-8 -*-
"""
eval_ik_beam2ch_all.py

評価①:
  - 訓練データ & テストデータの全サンプルについて、
      ・Left: 真値 vs 予測値の散布図
      ・Right: 真値 vs 予測値の散布図
      ・Left: (pred - true) の符号付き誤差ヒストグラム
      ・Right: (pred - true) の符号付き誤差ヒストグラム
    → train/test 別々に出力（合計 8枚の図）

評価②:
  - 訓練データ & テストデータそれぞれから、隣接ペアサンプルをランダムに10個ずつ抽出し、
    各サンプルごとにフォルダを作り、その中に
      ・bar.png  （ΔVの棒グラフ: 真値=赤破線, 予測=青バー）
      ・i_raw.png   : マスク前画像 (時刻 i)
      ・ip1_raw.png : マスク前画像 (時刻 i+1)
      ・i_mask.png  : マスク後 (beammask) 画像 (時刻 i)
      ・ip1_mask.png: マスク後 (beammask) 画像 (時刻 i+1)
    を保存。

前提:
  - train/test でそれぞれ signals_aug.csv, *_beam2ch.npy, *_beammask.png が存在すること
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
    r"\module_controller\IK\1module_babbling_data\silicon\roi_aug_shift_500_10_beammask_2ch_upper0"
)
TRAIN_AUG_IMG_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\aug_shift_500_10"
)
TRAIN_CSV_PATH = TRAIN_AUG_IMG_DIR / "signals_aug.csv"

# ---- テスト側 ----
TEST_BEAMMASK_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\roi_aug_shift_500_10_beammask_2ch_upper0_test"
)
TEST_AUG_IMG_DIR = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\1module_babbling_data\silicon\aug_shift_500_10_test"
)
TEST_CSV_PATH = TEST_AUG_IMG_DIR / "signals_aug.csv"

# 学習済みモデル
MODEL_PATH = Path(
    r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control"
    r"\module_controller\IK\ik_beam2ch_babbling_500_10_shift_upper0_model.pth"
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
# サンプル構築（隣接ペア & 付随情報セット）
# ==========================================================

def build_samples(beammask_dir: Path, csv_aug_path: Path):
    """
    signals_aug.csv から orig_index ごとにグルーピングし、
    隣接ペア (k, k+1) について group_k × group_{k+1} の全組み合わせを返す。

    返り値: samples(list of dict)
      dict には以下を格納:
        - fname_i, fname_ip1
        - npy_i, npy_ip1
        - q_i  = (V_L^i,   V_R^i)
        - q_ip1= (V_L^{i+1}, V_R^{i+1})
    """
    beammask_dir = Path(beammask_dir)

    with open(csv_aug_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if len(rows) < 2:
        raise RuntimeError(f"{csv_aug_path} の行数が足りません。")

    groups = defaultdict(list)
    for row in rows:
        try:
            orig = int(row["orig_index"])
        except Exception:
            continue
        groups[orig].append(row)

    samples = []
    sorted_indices = sorted(groups.keys())

    for k in sorted_indices:
        k_next = k + 1
        if k_next not in groups:
            continue

        rows_k     = groups[k]
        rows_knext = groups[k_next]

        for row_i in rows_k:
            for row_ip1 in rows_knext:
                fname_i   = row_i["filename"]   # 例: "57_aug_0000.png"
                fname_ip1 = row_ip1["filename"] # 例: "58_aug_0003.png"

                stem_i   = os.path.splitext(fname_i)[0]
                stem_ip1 = os.path.splitext(fname_ip1)[0]

                npy_i   = beammask_dir / f"{stem_i}_beam2ch.npy"
                npy_ip1 = beammask_dir / f"{stem_ip1}_beam2ch.npy"

                if not npy_i.exists() or not npy_ip1.exists():
                    continue

                try:
                    q_i   = (float(row_i["Left_V"]),   float(row_i["Right_V"]))
                    q_ip1 = (float(row_ip1["Left_V"]), float(row_ip1["Right_V"]))
                except Exception:
                    continue

                samples.append({
                    "fname_i":   fname_i,
                    "fname_ip1": fname_ip1,
                    "npy_i":     npy_i,
                    "npy_ip1":   npy_ip1,
                    "q_i":       q_i,
                    "q_ip1":     q_ip1,
                })

    if len(samples) == 0:
        raise RuntimeError("有効な隣接ペアサンプルが見つかりません。")

    print(f"[INFO] build_samples: 有効な IK ペア数 = {len(samples)}")
    return samples

# ==========================================================
# 評価①: 散布図 & ヒスト
# ==========================================================

def eval_scatter_and_hist_for_split(
    tag: str,
    beammask_dir: Path,
    csv_path: Path,
    model: IKBeamNet,
    device
):
    """
    tag: "train" or "test"

    ・全サンプルについて推論
    ・Left/Right それぞれ:
        - 真値 vs 予測値散布図
        - (pred - true) ヒスト
    """
    print(f"\n=== {tag.upper()} scatter & hist ===")

    samples = build_samples(beammask_dir, csv_path)

    y_true_L = []
    y_pred_L = []
    y_true_R = []
    y_pred_R = []

    model.eval()
    with torch.no_grad():
        for s in samples:
            # 入力テンソル作成
            mask_i   = np.load(str(s["npy_i"])).astype(np.float32)
            mask_ip1 = np.load(str(s["npy_ip1"])).astype(np.float32)
            vL_i, vR_i = s["q_i"]
            vL_true, vR_true = s["q_ip1"]

            mask_i_t   = torch.from_numpy(mask_i)    # (2,H,W)
            mask_ip1_t = torch.from_numpy(mask_ip1)  # (2,H,W)

            _, H, W = mask_i_t.shape
            q_map = torch.zeros((2, H, W), dtype=torch.float32)
            q_map[0, :, :] = vL_i / 5.0
            q_map[1, :, :] = vR_i / 5.0

            x = torch.cat([mask_i_t, mask_ip1_t, q_map], dim=0).unsqueeze(0).to(device)

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
    - y=0 を「現在電圧 data[i]」とみなした ΔV 表示。
      ΔV_true = V^{i+1}_true - V^i
      ΔV_pred = V^{i+1}_pred - V^i
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

    plt.axhline(0.0, color="k", linewidth=1.0)  # 現在電圧を0とした基準線
    plt.xticks(x, ["Left", "Right"])
    plt.ylabel("ΔV [V] (next - current)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def save_images_for_pair(sample, aug_img_dir: Path, beammask_dir: Path, out_dir: Path):
    """
    sample: build_samples で作った1サンプル(dict)
    aug_img_dir: *_aug_*.png が入っているディレクトリ
    beammask_dir: *_beammask.png, *_beam2ch.npy が入っているディレクトリ
    out_dir: 保存先フォルダ
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fname_i   = sample["fname_i"]
    fname_ip1 = sample["fname_ip1"]

    stem_i   = os.path.splitext(fname_i)[0]
    stem_ip1 = os.path.splitext(fname_ip1)[0]

    # マスク前画像（aug画像）
    img_i_path   = aug_img_dir / fname_i
    img_ip1_path = aug_img_dir / fname_ip1

    # マスク後画像（beammask 可視化）
    mask_i_vis_path   = beammask_dir / f"{stem_i}_beammask.png"
    mask_ip1_vis_path = beammask_dir / f"{stem_ip1}_beammask.png"

    def try_copy(src: Path, dst: Path, label: str):
        if src.exists():
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(dst), img)
            else:
                print(f"[WARN] 画像読み込み失敗 ({label}): {src}")
        else:
            print(f"[WARN] 画像ファイルが見つかりません ({label}): {src}")

    try_copy(img_i_path,   out_dir / "i_raw.png",   "i_raw")
    try_copy(img_ip1_path, out_dir / "ip1_raw.png", "ip1_raw")
    try_copy(mask_i_vis_path,   out_dir / "i_mask.png",   "i_mask")
    try_copy(mask_ip1_vis_path, out_dir / "ip1_mask.png", "ip1_mask")


def evaluate_examples_for_split(
    tag: str,
    beammask_dir: Path,
    aug_img_dir: Path,
    csv_path: Path,
    model: IKBeamNet,
    device
):
    """
    tag: "train" or "test"
    """
    print(f"\n=== {tag.upper()} examples ===")
    samples = build_samples(beammask_dir, csv_path)

    # ランダムに NUM_EXAMPLES 個サンプル
    if len(samples) <= NUM_EXAMPLES:
        chosen = samples
    else:
        chosen = random.sample(samples, NUM_EXAMPLES)

    for i, sample in enumerate(chosen):
        print(f"[{tag}] example {i+1}/{len(chosen)}")

        # 入力テンソル x を構成
        mask_i   = np.load(str(sample["npy_i"])).astype(np.float32)
        mask_ip1 = np.load(str(sample["npy_ip1"])).astype(np.float32)
        vL_i, vR_i = sample["q_i"]
        vL_true, vR_true = sample["q_ip1"]

        mask_i_t   = torch.from_numpy(mask_i)    # (2,H,W)
        mask_ip1_t = torch.from_numpy(mask_ip1)  # (2,H,W)

        _, H, W = mask_i_t.shape
        q_map = torch.zeros((2, H, W), dtype=torch.float32)
        q_map[0, :, :] = vL_i / 5.0
        q_map[1, :, :] = vR_i / 5.0

        x = torch.cat([mask_i_t, mask_ip1_t, q_map], dim=0).unsqueeze(0).to(device)  # (1,6,H,W)

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
        save_images_for_pair(sample, aug_img_dir, beammask_dir, out_dir)

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
    )
    eval_scatter_and_hist_for_split(
        tag="test",
        beammask_dir=TEST_BEAMMASK_DIR,
        csv_path=TEST_CSV_PATH,
        model=model,
        device=device,
    )

    # --- 評価②: 10サンプルのバー + 画像 ---
    evaluate_examples_for_split(
        tag="train",
        beammask_dir=TRAIN_BEAMMASK_DIR,
        aug_img_dir=TRAIN_AUG_IMG_DIR,
        csv_path=TRAIN_CSV_PATH,
        model=model,
        device=device,
    )
    evaluate_examples_for_split(
        tag="test",
        beammask_dir=TEST_BEAMMASK_DIR,
        aug_img_dir=TEST_AUG_IMG_DIR,
        csv_path=TEST_CSV_PATH,
        model=model,
        device=device,
    )

    print("\n✅ All evaluation done.")


if __name__ == "__main__":
    main()
