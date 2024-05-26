import io
import os
from threading import Thread

import numpy as np
import torch
import torchaudio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from flask import Flask, send_file
from flask import request
from scipy.io.wavfile import write
from transformers import MusicgenForConditionalGeneration, MusicgenProcessor, set_seed, AutoProcessor

from llm.MusicGenStreamer import MusicgenStreamer

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
GENERATE_FOLDER = 'generate'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATE_FOLDER'] = GENERATE_FOLDER

model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
processor = AutoProcessor.from_pretrained("facebook/musicgen-melody")

model.audio_encoder.config.sampling_rate = 44100
sampling_rate = model.audio_encoder.config.sampling_rate
frame_rate = model.audio_encoder.config.frame_rate

target_dtype = np.int16
max_range = np.iinfo(target_dtype).max


def generate_audio(text_prompt, audio_length_in_s=10.0, play_steps_in_s=2.0, seed=5):
    max_new_tokens = int(frame_rate * audio_length_in_s)
    play_steps = int(frame_rate * play_steps_in_s)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device != model.device:
        model.to(device)
        if device == "cuda:0":
            model.half()

    inputs = processor(
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    )

    streamer = MusicgenStreamer(model, device=device, play_steps=play_steps)

    generation_kwargs = dict(
        **inputs.to(device),
        streamer=streamer,
        max_new_tokens=max_new_tokens,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()

    set_seed(seed)
    first_chunk = True
    for new_audio in streamer:
        print(f"Sample of length: {round(new_audio.shape[0] / sampling_rate, 2)} seconds")
        new_audio = (new_audio * max_range).astype(np.int16)
        new_wav = ndarray_to_wav_bytes(new_audio, sampling_rate)
        # strip length information from first chunk header, remove headers entirely from subsequent chunks
        if first_chunk:
            new_wav = (
                    new_wav[:4] + b"\xFF\xFF\xFF\xFF" + new_wav[8:]
            )
            new_wav = (
                    new_wav[:40] + b"\xFF\xFF\xFF\xFF" + new_wav[44:]
            )
            first_chunk = False
        else:
            new_wav = new_wav[44:]
        yield new_wav


def ndarray_to_wav_bytes(ndarray, sample_rate):
    """
    Convert a numpy ndarray to WAV format as bytes array.

    Args:
    - ndarray: numpy ndarray containing audio data
    - sample_rate: sample rate of the audio data

    Yields:
    - bytes: WAV format data as bytes array
    """
    # Ensure the ndarray is of appropriate data type (int16)
    ndarray = np.int32(ndarray * 32767.0)

    # Write ndarray to WAV file in memory
    with io.BytesIO() as wav_bytes:
        write(wav_bytes, sample_rate, ndarray)
        return wav_bytes.getvalue()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['GENERATE_FOLDER']):
        os.makedirs(app.config['GENERATE_FOLDER'])


create_folder()


@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return {'error': 'No audio file'}

    audio_file = request.files['audio']

    if audio_file.filename == '':
        return {'error': 'No selected audio file'}

    if audio_file and allowed_file(audio_file.filename):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
        app.config['FILE_PATH'] = file_path
        audio_file.save(file_path)
        return {'message': 'Upload successful', 'filename': audio_file.filename}
    else:
        return {'error': 'Invalid file format'}


@app.route('/generate')
def generate():
    # 设置 WAV 文件的路径
    wav_file_path = app.config['FILE_PATH']

    # 加载音频文件
    waveform, sample_rate = torchaudio.load(wav_file_path)
    # 将波形数据转换为张量
    waveform_tensor = torch.tensor(waveform)

    model = MusicGen.get_pretrained('facebook/musicgen-melody', device='cpu')
    model.set_generation_params(duration=8)  # generate 8 seconds.
    descriptions = ['Jazz']
    wav = model.generate_with_chroma(descriptions=descriptions, melody_wavs=waveform_tensor,
                                     melody_sample_rate=sample_rate, progress=True)  # generates 3 samples.

    # generates using the melody from the given audio and the provided descriptions.
    for idx, one_wav in enumerate(wav):
        # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
        save_path = os.path.join(app.config['GENERATE_FOLDER'], f'{idx}')
        app.config['SAVE_PATH'] = save_path
        audio_write(save_path, one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

    generate_file_path = app.config['SAVE_PATH'] + ".wav"
    # 检查文件是否存在
    if not os.path.exists(generate_file_path):
        return "File not found", 404

    # 以二进制模式读取文件
    with open(generate_file_path, 'rb') as f:
        wav_data = f.read()

    # 返回 WAV 文件给客户端
    return send_file(
        io.BytesIO(wav_data),
        mimetype='audio/wav',
    )


@app.route('/streamGen')
def stream_gen():
    return generate_audio("An 80s driving pop song with heavy drums and synth pads in the background",
                          audio_length_in_s=10), {"Content-Type": 'audio/wav'}


@app.route('/streamCanvasMusic', methods=['POST'])
def stream_canvas_music():
    curves = request.form
    print(curves)
    return {"message": "success"}


if __name__ == '__main__':
    app.run(debug=True)
