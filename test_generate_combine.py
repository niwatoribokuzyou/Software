from gpt_test import chat_with_gpt
from effb2_test import generate_audio_caption
from whisper_test import transcribe_audio
from generate import generates
from lets_music_with_beat import overlay_short_sounds_numpy
import soundfile as sf
import io
import base64

prompt = "Create a calm and soft jazz background music with a slow tempo, featuring piano and light saxophone. The music should evoke a relaxed and introspective atmosphere, suitable for a dimly lit room."
def combine_musics(prompt, kankyouonn,  shortsound_path = "./鍵を開ける1.mp3"):
    music, sr = generates(prompt)
    # combine_music = ?????(music, kankyouonn)
    
    # output_dir="./test/answer.wav"
    # combine_music = overlay_short_sounds_numpy(music, sr, shortsound_path, output_dir)
    
    return music


