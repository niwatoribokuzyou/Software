import librosa
import base64
from gpt_test import chat_with_gpt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio
from music_generater import generate_music
# wavファイルを読み込む
wav_path = "voice.wav"  # 読み込みたいwavファイルのパス
y, sr = librosa.load(wav_path, sr=None)  # sr=None で元のサンプリングレートを維持

# y が音声データ（numpy配列）なので decoded_audio の代わりに使える
caption = generate_audio_caption(y)
stt_data = transcribe_audio(y)
temperature = 27#夜ベッド
humidity = 60
pressure = 1012
illuminance = 10
# temperature = env_data.get("temperature", 25)
# pressure = env_data.get("pressure", 1013)
# humidity = env_data.get("humidity", 50)
# illuminance = env_data.get("lux", 500)

prompt = chat_with_gpt(stt_data, caption, temperature, humidity, pressure, illuminance)
print("Generated Prompt:", prompt)

# Sunoで音楽生成
music = generate_music(prompt, y)
