from pydub import AudioSegment
import io
import torch
from transformers import AutoModel, PreTrainedTokenizerFast
import torchaudio


def generate_audio_caption(audio_bytes, model_name="wsntxxn/effb2-trm-audio-captioning", tokenizer_name="wsntxxn/audiocaps-simple-tokenizer") -> str:
    """
    音声ファイルからキャプションを生成してファイルに保存する関数

    Args:
        output_path (str): キャプションを保存するテキストファイルのパス
        model_name (str): HuggingFaceモデル名
        tokenizer_name (str): HuggingFaceトークナイザ名
    """
    input_path = "input_caption.wav"
    audio = AudioSegment.from_file(io.BytesIO(audio_bytes), format="mp3")
    audio.export(input_path, format="wav")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルとトークナイザ読み込み
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_name)

    # 音声読み込み
    wav, sr = torchaudio.load(input_path)
    wav = torchaudio.functional.resample(wav, sr, model.config.sample_rate)
    if wav.size(0) > 1:
        wav = wav.mean(0).unsqueeze(0)

    # 推論
    with torch.no_grad():
        word_idxs = model(audio=wav, audio_length=[wav.size(1)])

    caption = tokenizer.decode(word_idxs[0], skip_special_tokens=True)
    print("Caption:", caption)

    return caption



if __name__ == "__main__":
    with open("./voice.wav", "rb") as f:
        audio_bytes = f.read()
    
    generate_audio_caption(audio_bytes)