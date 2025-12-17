# from PIL import Image
# import os

# IN_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset\normal\raw"
# OUT_GIF = os.path.join(IN_DIR, "preview_small.gif")

# # ===== 調整パラメータ =====
# FRAME_STEP = 1      # 何枚ごとに使うか（1なら全部, 2なら 1,3,5,...）
# SCALE = 0.5         # 解像度スケール（0.5 → 縦横半分）
# FPS = 10            # 表示フレームレート
# N_COLORS = 256      # GIFパレットの色数（256, 128, 64 くらいで調整）

# DURATION = int(1000 / FPS)  # ms

# # ===== 画像リスト取得（i.png の i でソート） =====
# files = sorted(
#     [f for f in os.listdir(IN_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))],
#     key=lambda x: int(os.path.splitext(x)[0])
# )

# # 間引き
# files = files[::FRAME_STEP]

# if not files:
#     raise RuntimeError("No images found.")

# frames = []
# for f in files:
#     path = os.path.join(IN_DIR, f)
#     img = Image.open(path).convert("RGB")

#     # 解像度縮小
#     if SCALE != 1.0:
#         w, h = img.size
#         img = img.resize((int(w * SCALE), int(h * SCALE)), Image.LANCZOS)

#     # パレット化（色数削減）
#     img = img.convert("P", palette=Image.ADAPTIVE, colors=N_COLORS)
#     frames.append(img)

# # GIF 保存（軽量化オプションつき）
# frames[0].save(
#     OUT_GIF,
#     save_all=True,
#     append_images=frames[1:],
#     duration=DURATION,
#     loop=0,
#     optimize=True
# )

# print("saved:", OUT_GIF)
# print("frames:", len(frames))





# from PIL import Image
# import os
# import re

# IN_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset\normal\roi_rot"
# OUT_GIF = os.path.join(IN_DIR, "1_rot.gif")

# FPS = 10
# DURATION = int(1000 / FPS)

# # 正規表現で angle を抽出する
# # 例:  1_rot_m090.png → -90
# #      1_rot_p015.png → +15
# angle_pattern = re.compile(r"^1_rot_([pm])(\d{3})\.(png|jpg|jpeg|bmp)$", re.IGNORECASE)

# files = []
# for f in os.listdir(IN_DIR):
#     match = angle_pattern.match(f)
#     if match:
#         sign, num, _ = match.groups()
#         angle = int(num)
#         if sign == "m":
#             angle = -angle
#         # angle とファイル名をセットで保持
#         files.append((angle, f))

# # 角度順にソート（-90 → ... → 0 → ... → +90）
# files_sorted = sorted(files, key=lambda x: x[0])

# print("Detected frames (angle, filename):")
# for angle, name in files_sorted:
#     print(angle, name)

# if not files_sorted:
#     raise RuntimeError("No matching files found starting with '1_rot_'.")

# # 画像読み込み
# frames = []
# for angle, name in files_sorted:
#     img = Image.open(os.path.join(IN_DIR, name)).convert("RGB")
#     frames.append(img)

# # GIF保存
# frames[0].save(
#     OUT_GIF,
#     save_all=True,
#     append_images=frames[1:],
#     duration=DURATION,
#     loop=0
# )

# print("Saved GIF:", OUT_GIF)




from PIL import Image
import os
# ==== 設定 ====
AUG_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\normal\roi_aug_random"
OUT_GIF = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\1module_dataset_max\normal\roi_aug_random\1_aug.gif"
# ==== 一番画像の50枚を読み込み ====
frames = []
for i in range(50):
    fname = f"1_aug_{i:04d}.png"
    path = os.path.join(AUG_DIR, fname)
    if not os.path.exists(path):
        print(f"[WARN] missing: {path}")
        continue
    img = Image.open(path).convert("RGB")
    frames.append(img)
# ==== GIFとして保存 ====
if len(frames) > 0:
    frames[0].save(
        OUT_GIF,
        save_all=True,
        append_images=frames[1:],
        duration=100,   # ms（100ms = 10fps）
        loop=0          # ループ無限
    )
    print(f"[OK] GIF saved to: {OUT_GIF}")
else:
    print("[ERROR] No frames loaded.")










