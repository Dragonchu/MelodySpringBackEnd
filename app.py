import librosa
import numpy as np
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
    y, sr = librosa.load(path=audio_path)

    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=75, fmax=1600)

    # Ensure that the shapes of pitches and magnitudes are as expected
    if pitches.shape[0] == magnitudes.shape[0]:
        max_indices = np.argmax(magnitudes, axis=0)

        if max_indices.max() < pitches.shape[1]:
            pitch = pitches[max_indices, np.arange(pitches.shape[1])]

            # Convert pitch information to integer
            notes = [int(p) for p in pitch]
            return jsonify({'notes': notes})
        else:
            return jsonify({'error': f"Index {max_indices.max()} is out of bounds for pitches with size {pitches.shape[1]}"})
    else:
        return jsonify({'error': "Shapes of pitches and magnitudes do not match"})


if __name__ == '__main__':
    app.run(debug=True)
