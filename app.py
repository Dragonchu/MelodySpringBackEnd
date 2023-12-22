import librosa
from flask import Flask, render_template, request, send_from_directory, jsonify
import os

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def create_upload_folder():
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])


create_upload_folder()


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
        audio_file.save(file_path)
        return {'message': 'Upload successful', 'filename': audio_file.filename}
    else:
        return {'error': 'Invalid file format'}


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/analyze', methods=['POST'])
def analyze():
    audio_path = 'uploads/recording.wav'
    y, sr = librosa.load(audio_path)

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr)
    pitch = pitches[magnitudes.argmax()]

    # 这里简单地将音高信息转换成字符表示
    notes = [f'{int(p)}' for p in pitch]

    return jsonify({'notes': notes})


if __name__ == '__main__':
    app.run(debug=True)
