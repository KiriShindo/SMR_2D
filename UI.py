import tkinter as tk
from PIL import Image, ImageTk
from serial import Serial

# --- 設定部分 --- #
SERIAL_PORT  = 'COM4'
BAUDRATE     = 9600
IMAGE_PATH   = 'UI.png'
POSITIONS = {
    'L1': (120,  50), 'L2': (120, 150), 'L3': (120, 250),
    'L4': (120, 350), 'L5': (120, 450), 'L6': (120, 550),
    'R1': (270,  50), 'R2': (270, 150), 'R3': (270, 250),
    'R4': (270, 350), 'R5': (270, 450), 'R6': (270, 550),
}
MUSCLE_NAMES = list(POSITIONS.keys())

class MuscleControllerApp:
    def __init__(self, root):
        self.root = root
        root.title("Muscle Voltage Controller")

        # ステータス表示ラベル
        self.status = tk.StringVar(value="Ready")
        status_label = tk.Label(root, textvariable=self.status, anchor='w')
        status_label.pack(fill='x', side='bottom')

        # シリアルポートを開いて READY を待つ
        try:
            self.ser = Serial(SERIAL_PORT, BAUDRATE, timeout=1)
            self.status.set(f"[Serial] Opening {SERIAL_PORT}@{BAUDRATE}...")
            while True:
                line = self.ser.readline().decode(errors='ignore').strip()
                if line:
                    self.status.set(f"[Serial] ← {line}")
                if line == 'READY':
                    break
            self.status.set("Arduino READY")
        except Exception as e:
            self.status.set(f"Serial Error: {e}")
            return

        # 画像 + キャンバス
        img = Image.open(IMAGE_PATH)
        self.tkimg = ImageTk.PhotoImage(img)
        canvas = tk.Canvas(root, width=self.tkimg.width(), height=self.tkimg.height())
        canvas.pack()
        canvas.create_image(0, 0, image=self.tkimg, anchor='nw')

        # テキストボックス
        self.entries = {}
        for name, (x, y) in POSITIONS.items():
            lbl = tk.Label(root, text=name, bg='white')
            canvas.create_window(x, y, window=lbl)
            ent = tk.Entry(root, width=3, font=('Arial',12), justify='center')
            ent.insert(0, "0.0")
            canvas.create_window(x+30, y, window=ent)
            self.entries[name] = ent

        # ボタン群
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="印加",   width=8, command=self.apply_voltages).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="リセット", width=8, command=self.reset_voltages).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="終了",   width=8, command=self.exit_app).grid( row=0, column=2, padx=5)

    def apply_voltages(self):
        try:
            volts = []
            for name in MUSCLE_NAMES:
                v = float(self.entries[name].get())
                v = max(0.0, min(5.0, v))
                volts.append(v)
            cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in volts) + '\n'
            self.status.set(f"[Serial] → {cmd.strip()}")
            self.ser.write(cmd.encode())
            while True:
                resp = self.ser.readline().decode(errors='ignore').strip()
                if resp:
                    self.status.set(f"[Serial] ← {resp}")
                if resp == 'APPLIED':
                    break
            self.status.set("Voltages applied.")
        except ValueError:
            self.status.set("Input Error: enter valid numbers.")

    def reset_voltages(self):
        zeros = [0.0] * len(MUSCLE_NAMES)
        cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in zeros) + '\n'
        self.status.set(f"[Serial] → {cmd.strip()}")
        self.ser.write(cmd.encode())
        while True:
            resp = self.ser.readline().decode(errors='ignore').strip()
            if resp:
                self.status.set(f"[Serial] ← {resp}")
            if resp == 'APPLIED':
                break
        self.status.set("All channels reset to 0.0V.")

    def exit_app(self):
        # リセット信号を送り、シリアル閉じて終了
        zeros = [0.0] * len(MUSCLE_NAMES)
        cmd = 'VOLT ' + ','.join(f"{v:.1f}" for v in zeros) + '\n'
        self.ser.write(cmd.encode())
        try:
            self.ser.close()
        except:
            pass
        self.root.destroy()

if __name__ == '__main__':
    root = tk.Tk()
    MuscleControllerApp(root)
    root.mainloop()
