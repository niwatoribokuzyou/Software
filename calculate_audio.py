import tkinter as tk
from tkinter import PhotoImage
import numpy as np
import librosa
import wave
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter

# ----------------------------
# フィルタ関数
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def volume_process(value):
    temp_value = (np.log10(value + 0.0001) + 1.7) * 33
    temp_value = min(100.0, max(0.0, temp_value))
    if temp_value <= 20: return "1"
    elif temp_value <= 40: return "2"
    elif temp_value <= 60: return "3"
    elif temp_value <= 80: return "4"
    else: return "5"

# ----------------------------
# GUI クラス
# ----------------------------
class DemoWindow(tk.Tk):
    def __init__(self, audio_file):
        super().__init__()
        self.title("Non-realtime Audio Viewer")
        self.geometry("800x600")
        self.canvas = tk.Canvas(self, width=800, height=600, bg="black")
        self.canvas.pack()
        
        # WAV 読み込み
        self.audio_data, self.sr = librosa.load(audio_file, sr=22050)
        self.length_sec = len(self.audio_data) / self.sr
        
        # 音量計算（全体平均）
        self.volume = np.mean(np.abs(self.audio_data))
        self.volume_level = volume_process(self.volume)
        
        # スペクトログラム表示
        self.plot_spectrogram()
        
        # 音量ラベル
        self.volume_label = tk.Label(self, text=f"Volume Level: {self.volume_level}", fg="yellow", bg="black", font=("Courier", 16))
        self.volume_label.pack(pady=10)
    
    def plot_spectrogram(self):
        plt.figure(figsize=(8,4))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(self.audio_data)), ref=np.max)
        plt.imshow(D, aspect='auto', origin='lower', cmap='magma')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("spectrogram.png")
        plt.close()
        self.spec_img = PhotoImage(file="spectrogram.png")
        self.canvas.create_image(400, 300, image=self.spec_img)

# ----------------------------
# 実行
# ----------------------------
if __name__ == "__main__":
    audio_file = "./answer_data/doa.mp3"  # 解析したい音声ファイル
    print("audio_file",audio_file)
    app = DemoWindow(audio_file)
    app.mainloop()
