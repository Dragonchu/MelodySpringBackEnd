import io
import os

import torch
import torchaudio
from audiocraft.data.audio import audio_write
from audiocraft.models import MusicGen
from flask import Flask, render_template, request, send_file

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
GENERATE_FOLDER = 'generate'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATE_FOLDER'] = GENERATE_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['GENERATE_FOLDER']):
        os.makedirs(app.config['GENERATE_FOLDER'])


create_folder()


@app.route('/')
def index():
    return render_template('index.html')


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


if __name__ == '__main__':
    app.run(debug=True)
