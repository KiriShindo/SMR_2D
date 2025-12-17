# #最高解像度
# #!/usr/bin/env python3
# # -*- coding: utf-8 -*-

# import os
# import time
# import csv
# import cv2
# import av
# import threading
# import queue
# from serial import Serial

# # Windows のみキーボード検知
# if os.name == 'nt':
#     import msvcrt

# # --- 設定 ---
# SERIAL_PORT      = 'COM3'
# BAUDRATE         = 9600
# DEVICE_DSHOW     = 'video=HD Pro Webcam C920'
# VIDEO_SIZE       = '1920x1080'                  # ★ C920 の最高解像度想定
# OUT_DIR          = '1module_dataset_max/silicon/raw'
# FRAME_QUEUE_SIZE = 1
# # ------------------

# def kb_hit():
#     return msvcrt.kbhit() if os.name == 'nt' else False

# def get_char():
#     return msvcrt.getch().decode() if os.name == 'nt' else ''

# def generate_patterns():
#     """
#     左 (index 0), 右 (index 6)
#     0.0〜5.0 を 0.5 刻みで全組み合わせ → 121 通り
#     各パターン後に全ゼロを挿入
#     """
#     patterns = []
#     levels = [i * 0.5 for i in range(11)]  # 0.0〜5.0
#     base_zero = [0.0] * 12

#     for v_left in levels:
#         for v_right in levels:
#             pat = base_zero.copy()
#             pat[0] = v_left
#             pat[6] = v_right

#             patterns.append(pat)               # 印加パターン
#             patterns.append(base_zero.copy())  # 全ゼロ（リセット用）
#     return patterns

# def main():
#     os.makedirs(OUT_DIR, exist_ok=True)
#     csv_path = os.path.join(OUT_DIR, 'signals.csv')

#     with open(csv_path, 'w', newline='') as f:
#         pass  # 初期化だけ

#     ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
#     print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
#     while True:
#         line = ser.readline().decode().strip()
#         if line == 'READY':
#             print("Arduino ready.")
#             break

#     container = av.open(
#         format='dshow',
#         file=DEVICE_DSHOW,
#         options={'video_size': VIDEO_SIZE}
#     )

#     frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

#     def frame_reader():
#         for packet in container.demux(video=0):
#             for frame in packet.decode():
#                 img = frame.to_ndarray(format='bgr24')
#                 try:
#                     frame_queue.put_nowait(img)
#                 except queue.Full:
#                     _ = frame_queue.get_nowait()
#                     frame_queue.put_nowait(img)

#     reader_thread = threading.Thread(target=frame_reader, daemon=True)
#     reader_thread.start()

#     patterns = generate_patterns()
#     print("Starting sequence. Press 'q' to emergency stop./n")

#     img_index = 1
#     zero_captured = False  # ★ 最初の 0V だけ撮るためのフラグ

#     try:
#         for pat in patterns:

#             if kb_hit() and get_char().lower() == 'q':
#                 print("Emergency key → send 'q'")
#                 ser.write(b'q/n')
#                 ser.flush()
#                 break

#             # 電圧送信
#             cmd = 'VOLT ' + ','.join(f'{v:.1f}' for v in pat) + '/n'
#             ser.write(cmd.encode())

#             # APPLIED 待ち
#             while True:
#                 resp = ser.readline().decode().strip()
#                 if resp == 'APPLIED':
#                     print(f"[Step {img_index}] voltages applied.")
#                     break

#             time.sleep(2)

#             # ---------------------------------------------------
#             # 全て 0.0 の場合：
#             #   - まだ 0V を撮っていなければ → 撮る
#             #   - すでに撮っていれば         → リセット用としてスキップ
#             # ---------------------------------------------------
#             if all(v == 0.0 for v in pat):
#                 if zero_captured:
#                     print(f"[Skip] All-zero pattern (reset) → no capture / no save/n")
#                     continue
#                 else:
#                     print(f"[Info] First all-zero pattern → capture as 0V sample")
#                     zero_captured = True
#             # ---------------------------------------------------

#             # フレーム取得
#             try:
#                 img = frame_queue.get(timeout=1)
#             except queue.Empty:
#                 print(f"[Warning] no frame for step {img_index}")
#                 img_index += 1
#                 continue

#             # 保存
#             img_fn = os.path.join(OUT_DIR, f"{img_index}.png")
#             cv2.imwrite(img_fn, img)
#             print(f"[Capture] saved {img_index}.png")

#             # CSV 保存
#             with open(csv_path, 'a', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(pat)

#             img_index += 1
#             time.sleep(2)

#     finally:
#         ser.write(b'q/n')
#         ser.close()
#         container.close()
#         print("/nDone. All resources released.")

# if __name__ == '__main__':
#     main()




### DAC版 ###
import os
import time
import csv
import cv2
import av
import threading
import queue
from serial import Serial

# Windows のみキーボード検知
if os.name == 'nt':
    import msvcrt

# --- 設定 ---
SERIAL_PORT      = 'COM4'
BAUDRATE         = 9600
DEVICE_DSHOW     = 'video=HD Pro Webcam C920'
VIDEO_SIZE       = '1920x1080'                  # C920 の最高解像度想定
OUT_DIR          = 'C:/Users/ishiz/Documents/Akamine_workspace/master_thesis_2025/SMR_control/module_controller/FF/1module_dataset_max_DAC/none/raw'
FRAME_QUEUE_SIZE = 1

# DAC/Arduino 側との対応:
# Python から送る 12 要素の配列 pat[0..11] は、
#   (L1, L2, L3, L4, L5, L6, R1, R2, R3, R4, R5, R6)
# という UI 順と同じ並びで Arduino に送られ、
# Arduino の UI_TO_CH により
#   pat[0] → ch0 (DAC1 左)
#   pat[6] → ch3 (DAC1 右)
# となるようにしている。
LEFT_IDX  = 0   # 左アクチュエータ (グローバル ch0 に対応)
RIGHT_IDX = 6   # 右アクチュエータ (グローバル ch3 に対応)
NUM_CHANNELS = 12


def kb_hit():
    return msvcrt.kbhit() if os.name == 'nt' else False


def get_char():
    return msvcrt.getch().decode() if os.name == 'nt' else ''


def generate_patterns():
    """
    1モジュール分 (左: グローバル ch0, 右: グローバル ch3) のみを使用。

    Python 側では 12 要素の配列 pat[0..11] を作り、
      pat[LEFT_IDX]  = 左アクチュエータ用電圧
      pat[RIGHT_IDX] = 右アクチュエータ用電圧
    それ以外は 0.0V とする。

    左 (LEFT_IDX), 右 (RIGHT_IDX) について、
    0.0〜5.0 を 0.5 刻みで全組み合わせ → 121 通り。
    各パターンのあとに全ゼロパターンを挿入して、
    圧が残らないようにする。
    """
    patterns = []
    levels = [i * 0.5 for i in range(11)]  # 0.0〜5.0
    base_zero = [0.0] * NUM_CHANNELS

    for v_left in levels:
        for v_right in levels:
            pat = base_zero.copy()
            pat[LEFT_IDX]  = v_left
            pat[RIGHT_IDX] = v_right

            patterns.append(pat)               # 印加パターン
            patterns.append(base_zero.copy())  # 全ゼロ（リセット用）
    return patterns


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'signals.csv')

    # CSV 初期化
    with open(csv_path, 'w', newline='') as f:
        pass

    # シリアルオープン & READY 待ち
    ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[Serial] ← {line}")
        if line == 'READY':
            print("Arduino ready.")
            break

    # カメラオープン
    container = av.open(
        format='dshow',
        file=DEVICE_DSHOW,
        options={'video_size': VIDEO_SIZE}
    )

    frame_queue = queue.Queue(maxsize=FRAME_QUEUE_SIZE)

    def frame_reader():
        for packet in container.demux(video=0):
            for frame in packet.decode():
                img = frame.to_ndarray(format='bgr24')
                try:
                    frame_queue.put_nowait(img)
                except queue.Full:
                    _ = frame_queue.get_nowait()
                    frame_queue.put_nowait(img)

    reader_thread = threading.Thread(target=frame_reader, daemon=True)
    reader_thread.start()

    patterns = generate_patterns()
    print("Starting sequence. Press 'q' to emergency stop./n")

    img_index = 1
    zero_captured = False  # 最初の 0V だけ撮る

    try:
        for pat in patterns:

            # 非常停止
            if kb_hit() and get_char().lower() == 'q':
                print("Emergency key → send 'q'")
                ser.write(b'q/n')
                ser.flush()
                break

            # 電圧送信 (12 要素すべて送る)
            cmd = 'VOLT ' + ','.join(f'{v:.1f}' for v in pat) + '/n'
            print(f"[Serial] → {cmd.strip()}")
            ser.write(cmd.encode())

            # APPLIED 待ち
            while True:
                resp = ser.readline().decode(errors='ignore').strip()
                if resp:
                    print(f"[Serial] ← {resp}")
                if resp == 'APPLIED':
                    print(f"[Step {img_index}] voltages applied.")
                    break

            time.sleep(2)

            # 全て 0.0 の場合の扱い
            if all(v == 0.0 for v in pat):
                if zero_captured:
                    print(f"[Skip] All-zero pattern (reset) → no capture / no save/n")
                    continue
                else:
                    print(f"[Info] First all-zero pattern → capture as 0V sample")
                    zero_captured = True

            # フレーム取得
            try:
                img = frame_queue.get(timeout=1)
            except queue.Empty:
                print(f"[Warning] no frame for step {img_index}")
                img_index += 1
                continue

            # 画像保存
            img_fn = os.path.join(OUT_DIR, f"{img_index}.png")
            cv2.imwrite(img_fn, img)
            print(f"[Capture] saved {img_index}.png")

            # CSV 保存（12ch 分電圧をそのまま記録）
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pat)

            img_index += 1
            time.sleep(2)

    finally:
        # 終了時に q を送って全チャンネル 0V に
        try:
            ser.write(b'q/n')
        except Exception:
            pass

        try:
            ser.close()
        except Exception:
            pass

        try:
            container.close()
        except Exception:
            pass

        print("/nDone. All resources released.")


if __name__ == '__main__':
    main()
