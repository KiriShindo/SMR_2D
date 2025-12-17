# -*- coding: utf-8 -*-
"""
motor_babbling_capture_resume.py
---------------------------------------------------------
左右2ch(0〜5V, 0.5刻み)で motor babbling を行い、
200ステップ分だけデータを取得して終了する。

特徴:
- 画像は 1.png, 2.png, ... と連番保存
- signals.csv に [Left_V, Right_V] を追記
- signals.csv が既に存在する場合は、
    - 最後の行の値を current_left/right として再開
    - 画像番号も続きからスタート
---------------------------------------------------------
"""

import os
import csv
import time
import random
import cv2
import av
import queue
import threading
from serial import Serial

# ====== 設定パラメータ ======
SERIAL_PORT = 'COM4'
BAUDRATE = 9600
DEVICE_DSHOW = 'video=HD Pro Webcam C920'
VIDEO_SIZE = '1920x1080'

OUT_DIR = r"C:\Users\ishiz\Documents\Akamine_workspace\master_thesis_2025\SMR_control\module_controller\IK\1module_babbling_data\none\raw"

FRAME_QUEUE_SIZE = 1
STEP_INTERVAL = 3.0      # 各ステップ後の待機時間 [s]
STEPS_PER_RUN = 500      # ★ 1回の実行で動かすステップ数

LEFT_IDX = 0
RIGHT_IDX = 6
NUM_CHANNELS = 12

LEVELS = [i * 0.5 for i in range(11)]  # 0.0, 0.5, ..., 5.0


def setup_output_dir():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'signals.csv')

    current_left = 0.0
    current_right = 0.0
    img_index = 1

    if os.path.exists(csv_path):
        # 既存CSVがあれば最後の行から再開
        with open(csv_path, 'r', newline='') as f:
            reader = list(csv.reader(f))
        if len(reader) > 1:
            # ヘッダを前提: 1行目はヘッダ, 2行目以降がデータ
            last = reader[-1]
            try:
                current_left = float(last[0])
                current_right = float(last[1])
            except Exception:
                current_left, current_right = 0.0, 0.0
            img_index = len(reader)  # ヘッダ1行 + データN行 → 次は N+1.png
            print(f"[Info] Resume from last state: L={current_left:.1f}V, R={current_right:.1f}V, next img_index={img_index}")
        else:
            # ヘッダしかない or 空に近い場合
            print("[Info] signals.csv has no data rows. Start from 0V.")
            img_index = 1
    else:
        # 新規作成: ヘッダを書いておく
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Left_V', 'Right_V'])
        print("[Info] Created new signals.csv. Start from 0V, img_index=1")

    return csv_path, current_left, current_right, img_index


def setup_camera():
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

    th = threading.Thread(target=frame_reader, daemon=True)
    th.start()

    return container, frame_queue


def setup_serial():
    ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
    print(f"Opening serial {SERIAL_PORT}@{BAUDRATE}…")
    while True:
        line = ser.readline().decode(errors='ignore').strip()
        if line:
            print(f"[Serial] ← {line}")
        if line == 'READY':
            print("Arduino ready.")
            break
    return ser


def send_voltage(ser, v_left, v_right):
    """左右の電圧を指定してArduinoに送信"""
    pat = [0.0] * NUM_CHANNELS
    pat[LEFT_IDX] = v_left
    pat[RIGHT_IDX] = v_right
    cmd = 'VOLT ' + ','.join(f'{v:.1f}' for v in pat) + '\n'
    print(f"[Serial] → {cmd.strip()}")
    ser.write(cmd.encode())
    ser.flush()
    return pat


def choose_next_level(current_left, current_right):
    """現在値から ±1.0V 以内になるように次のレベルを選ぶ"""
    while True:
        next_left = random.choice(LEVELS)
        next_right = random.choice(LEVELS)
        if abs(next_left - current_left) <= 1.0 and abs(next_right - current_right) <= 1.0:
            return next_left, next_right


def main():
    csv_path, current_left, current_right, img_index = setup_output_dir()
    ser = setup_serial()
    container, frame_queue = setup_camera()

    print(f"\nStart motor babbling: {STEPS_PER_RUN} steps this run.\n")

    try:
        for step in range(STEPS_PER_RUN):

            # 次の電圧レベルを選択（±1.0V以内の遷移）
            next_left, next_right = choose_next_level(current_left, current_right)
            print(f"[Step {step+1}/{STEPS_PER_RUN}] L={next_left:.1f}V, R={next_right:.1f}V")

            # 電圧送信
            send_voltage(ser, next_left, next_right)

            # 準静的になるまで待つ
            time.sleep(STEP_INTERVAL)

            # 画像キャプチャ
            try:
                frame = frame_queue.get(timeout=1.0)
                img_path = os.path.join(OUT_DIR, f"{img_index}.png")
                cv2.imwrite(img_path, frame)
                print(f"[Saved] {img_index}.png")
            except queue.Empty:
                print("[Warning] Frame not captured")

            # CSVに電圧を追記
            with open(csv_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([next_left, next_right])

            current_left, current_right = next_left, next_right
            img_index += 1

    except KeyboardInterrupt:
        print("\n[Info] KeyboardInterrupt. Stopping early...")

    finally:
        # 安全のため q を送って全チャネル 0V に戻す想定
        try:
            ser.write(b'q\n')
            ser.flush()
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

        print("\nDone. Resources released. You can rest the compressor and rerun this script to continue from where you left off.")


if __name__ == "__main__":
    main()
