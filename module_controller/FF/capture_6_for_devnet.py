#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import csv
import cv2
import av
import threading
import queue
import random
from serial import Serial

# Windows のみキーボード検知
if os.name == 'nt':
    import msvcrt

# --- 設定 ---
SERIAL_PORT      = 'COM4'   # シリアルポート
BAUDRATE         = 9600
DEVICE_DSHOW     = 'video=HD Pro Webcam C920'  # ffmpeg -list_devices で確認したデバイス名
VIDEO_SIZE       = '1920x1080'                 # WIDTHxHEIGHT
OUT_DIR          = r'C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\FF\6module_dataset_for_devnet'
FRAME_QUEUE_SIZE = 1

NUM_CHANNELS     = 12       # 出力チャネル数
NUM_SAMPLES      = 1000     # 取得するサンプル数
INTERVAL_SEC     = 1.0      # 次の信号に切り替えるまでの時間[s]
VALUE_STEP       = 0.5      # 電圧刻み幅
MIN_VOLT         = 0.0
MAX_VOLT         = 5.0
# ------------------

def kb_hit():
    return msvcrt.kbhit() if os.name == 'nt' else False

def get_char():
    return msvcrt.getch().decode() if os.name == 'nt' else ''

def random_pattern():
    """
    各チャネルごとに [0.0, 5.0] の 0.5 刻みからランダムに 1 つ選ぶ
    """
    possible_vals = [MIN_VOLT + i * VALUE_STEP
                     for i in range(int((MAX_VOLT - MIN_VOLT) / VALUE_STEP) + 1)]
    return [random.choice(possible_vals) for _ in range(NUM_CHANNELS)]

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'signals.csv')
    # CSV をヘッダなしで初期化
    with open(csv_path, 'w', newline='') as f:
        pass

    # シリアルポートオープン → READY 待ち
    ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line == 'READY':
            print("Arduino ready.")
            break

    # PyAV でカメラオープン（DirectShow）
    container = av.open(
        format='dshow',
        file=DEVICE_DSHOW,
        options={'video_size': VIDEO_SIZE}
    )

    # バックグラウンドでフレームを読み続けるスレッド
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

    print("Starting random sequence. Press 'q' to emergency stop.\n")

    img_index = 1
    try:
        for step in range(NUM_SAMPLES):
            # 緊急停止判定
            if kb_hit() and get_char().lower() == 'q':
                print("Emergency key → send 'q'")
                ser.write(b'q\n')
                ser.flush()
                break

            # ランダムパターン生成
            pat = random_pattern()

            # 電圧コマンド送信
            cmd = 'VOLT ' + ','.join(f'{v:.1f}' for v in pat) + '\n'
            ser.write(cmd.encode())

            # 'APPLIED' 応答待ち
            while True:
                resp = ser.readline().decode(errors='ignore').strip()
                if resp == 'APPLIED':
                    print(f"[Step {img_index}] voltages applied.")
                    break

            # 次の信号に切り替えるまでの時間を 1 秒にする
            # → 電圧印加後 1 秒待ってから撮影し、すぐ次のループへ
            time.sleep(INTERVAL_SEC)

            # 最新フレームを取得
            try:
                img = frame_queue.get(timeout=1)
            except queue.Empty:
                print(f"[Warning] no frame for step {img_index}")
                continue

            # 画像保存
            img_fn = os.path.join(OUT_DIR, f"{img_index}.png")
            cv2.imwrite(img_fn, img)
            print(f"[Capture] saved {img_index}.png")

            # CSV 追記
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(pat)

            img_index += 1

    finally:
        try:
            ser.write(b'q\n')
        except Exception:
            pass
        ser.close()
        container.close()
        print("\nDone. All resources released.")

if __name__ == '__main__':
    main()
