import torch
import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write

if __name__ == '__main__':
    # 加载音频文件
    waveform, sample_rate = torchaudio.load("uploads/recording.wav")
    # 将波形数据转换为张量
    waveform_tensor = torch.tensor(waveform)

    model = MusicGen.get_pretrained('facebook/musicgen-melody', device='cpu')
    model.set_generation_params(duration=8)  # generate 8 seconds.
    descriptions = ['Jazz']
    wav = model.generate_with_chroma(descriptions=descriptions,melody_wavs=waveform_tensor,melody_sample_rate=sample_rate,progress=True)  # generates 3 samples.

    # generates using the melody from the given audio and the provided descriptions.
    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        audio_write(f'{idx}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)