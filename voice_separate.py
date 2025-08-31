
import torch, torchaudio
from demucs.pretrained import get_model
from demucs.apply import apply_model
from demucs.audio import AudioFile


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model("demucs_2stems")  # vocals + accompaniment
model.to(device)

with AudioFile("./split_wavs/voice_chunk_2.wav") as f:
    wav = f.read(f.channels)

with torch.no_grad():
    sources = apply_model(model, wav, device=device, split=True)

torchaudio.save("vocals.wav", sources[0].cpu(), f.samplerate)
torchaudio.save("accompaniment.wav", sources[1].cpu(), f.samplerate)
