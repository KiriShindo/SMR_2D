# # -*- coding: utf-8 -*-
# """
# click_rgb_picker.py

# - エクスプローラから画像を1枚選択
# - ウィンドウに表示
# - 左クリックした画素の座標 (x, y) と BGR / RGB 値をコンソールに表示
# - q / Esc で終了
# """

# import os
# import cv2
# import tkinter as tk
# from tkinter import filedialog

# # グローバル参照用
# g_img = None
# g_window_name = "Click to inspect RGB (q / Esc to quit)"


# def select_image_via_dialog():
#     """
#     エクスプローラから画像ファイルを選択
#     """
#     root = tk.Tk()
#     root.withdraw()
#     root.update()
#     filetypes = [
#         ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
#         ("All files", "*.*")
#     ]
#     path = filedialog.askopenfilename(
#         title="Select an image",
#         filetypes=filetypes
#     )
#     root.destroy()
#     return path


# def mouse_callback(event, x, y, flags, param):
#     global g_img

#     if g_img is None:
#         return

#     if event == cv2.EVENT_LBUTTONDOWN:
#         H, W = g_img.shape[:2]
#         if 0 <= x < W and 0 <= y < H:
#             # OpenCVはBGR順
#             b, g, r = g_img[y, x]
#             print(f"(x={x}, y={y})  BGR=({b}, {g}, {r})  RGB=({r}, {g}, {b})")


# def main():
#     global g_img, g_window_name

#     img_path = select_image_via_dialog()
#     if not img_path:
#         print("[INFO] No image selected. Exit.")
#         return

#     g_img = cv2.imread(img_path)
#     if g_img is None:
#         print(f"[ERROR] Failed to read image: {img_path}")
#         return

#     H, W = g_img.shape[:2]
#     print(f"[INFO] Loaded image: {img_path} (W={W}, H={H})")
#     print("操作: 左クリックでRGB確認, q / Esc で終了")

#     cv2.namedWindow(g_window_name, cv2.WINDOW_NORMAL)
#     cv2.setMouseCallback(g_window_name, mouse_callback)

#     while True:
#         cv2.imshow(g_window_name, g_img)
#         key = cv2.waitKey(10) & 0xFF
#         if key in (ord('q'), 27):  # 'q' or ESC
#             break

#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()





# -*- coding: utf-8 -*-
"""
click_rgb_picker_with_stats.py

- エクスプローラから画像を1枚選択
- ウィンドウに表示
- 左クリックした画素の BGR を記録
- クリック回数が N_LIMIT に達したら、
  ・B, G, R の平均と標準偏差を計算して表示
  ・補完色サンプリング用の例コードを表示
- q / Esc で終了
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog
import numpy as np

# ===== 設定 =====
N_LIMIT = 100  # 何点クリックしたら統計を出すか
# =================

g_img = None
g_window_name = "Click to inspect RGB (q / Esc to quit)"
g_samples = []      # BGRサンプルを貯める
g_stats_done = False


def select_image_via_dialog():
    """
    エクスプローラから画像ファイルを選択
    """
    root = tk.Tk()
    root.withdraw()
    root.update()
    filetypes = [
        ("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff"),
        ("All files", "*.*")
    ]
    path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=filetypes
    )
    root.destroy()
    return path


def mouse_callback(event, x, y, flags, param):
    global g_img, g_samples, g_stats_done

    if g_img is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        H, W = g_img.shape[:2]
        if 0 <= x < W and 0 <= y < H:
            # OpenCVはBGR順
            b, g, r = g_img[y, x]
            g_samples.append((int(b), int(g), int(r)))

            n = len(g_samples)
            print(f"[{n}/{N_LIMIT}] (x={x}, y={y})  BGR=({b}, {g}, {r})  RGB=({r}, {g}, {b})")

            # 規定回数クリックしたら統計を出す
            if (not g_stats_done) and n >= N_LIMIT:
                g_stats_done = True
                print("\n===== 統計 (クリックした画素の BGR) =====")
                arr = np.array(g_samples, dtype=np.float32)  # shape: (N, 3)
                mean_bgr = arr.mean(axis=0)
                std_bgr  = arr.std(axis=0, ddof=1)  # 不偏標準偏差

                mb, mg, mr = mean_bgr
                sb, sg, sr = std_bgr

                print(f"mean BGR = ({mb:.2f}, {mg:.2f}, {mr:.2f})")
                print(f"std  BGR = ({sb:.2f}, {sg:.2f}, {sr:.2f})")

                print("\n===== 補完色サンプリング用の例（Pythonコード） =====")
                print("例えば、回転オーグメンテーションで borderValue に使う色を")
                print("この平均・分散からサンプリングするなら、こんな感じにできます：\n")

                # そのままコピペできるように出力
                print("import numpy as np")
                print()
                print(f"mean_bgr = np.array([{mb:.2f}, {mg:.2f}, {mr:.2f}], dtype=np.float32)")
                print(f"std_bgr  = np.array([{sb:.2f}, {sg:.2f}, {sr:.2f}], dtype=np.float32)")
                print()
                print("def sample_border_color():")
                print("    # ガウス分布からサンプリングして [0, 255] にクリップ")
                print("    c = np.random.normal(mean_bgr, std_bgr)")
                print("    c = np.clip(c, 0, 255).astype(np.uint8)")
                print("    # OpenCVはBGRなので、tupleで返す")
                print("    return (int(c[0]), int(c[1]), int(c[2]))")
                print()
                print("# warpAffine の例：")
                print("# cv2.warpAffine(img, M, (W, H),")
                print("#                 flags=cv2.INTER_LINEAR,")
                print("#                 borderMode=cv2.BORDER_CONSTANT,")
                print("#                 borderValue=sample_border_color())")
                print("============================================\n")


def main():
    global g_img, g_window_name

    img_path = select_image_via_dialog()
    if not img_path:
        print("[INFO] No image selected. Exit.")
        return

    g_img = cv2.imread(img_path)
    if g_img is None:
        print(f"[ERROR] Failed to read image: {img_path}")
        return

    H, W = g_img.shape[:2]
    print(f"[INFO] Loaded image: {img_path} (W={W}, H={H})")
    print(f"操作: 左クリックでRGB確認・サンプル収集 (目標 {N_LIMIT}点), q / Esc で終了")

    cv2.namedWindow(g_window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(g_window_name, mouse_callback)

    while True:
        cv2.imshow(g_window_name, g_img)
        key = cv2.waitKey(10) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
