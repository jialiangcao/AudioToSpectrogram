from flask import Flask, request, jsonify
import torch
import torchaudio
import struct
import numpy
from functools import wraps
import firebase_admin
from firebase_admin import credentials, auth
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Firebase
try:
    cred = credentials.ApplicationDefault()

    # Local testing only: 
    # cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        "projectId": "nsf-2131186-18936"
    })
    logging.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize Firebase Admin SDK: {e}")

def firebase_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        id_token = request.headers.get('Authorization')
        if not id_token or not id_token.startswith('Bearer '):
            logging.warning("Authorization header missing or malformed.")
            return jsonify({'error': 'Unauthorized'}), 401

        try:
            logging.info("Verifying Firebase ID Token.")
            decoded_token = auth.verify_id_token(id_token.split(' ')[1])
            logging.info(f"Token verified successfully for: {decoded_token.get('uid')}")
            request.environ['user'] = decoded_token
        except Exception as e:
            logging.error(f"Token verification failed: {e}")
            return jsonify({'error': 'Invalid token'}), 401

        return f(*args, **kwargs)
    return decorated_function

app = Flask(__name__)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)

# Constants
SR = 44100
AUDIO_N_MELS = 64
N_FFT = 1024
HOP_LENGTH = N_FFT // 4
SEGMENTS = 2
BYTE_SIZE = 2 # 8 for Float64/Double, 4 for 32 bit, 2 for 16
MAX_AUDIO_BYTES = SR*SEGMENTS*BYTE_SIZE

def unpack_audio(raw_audio_data):
    sample_count = len(raw_audio_data) // BYTE_SIZE
    # '<' for little-endian (Apple), 'd' for double
    float16_samples = struct.unpack(f'<{sample_count}e', raw_audio_data)
    # Convert back to Python float for processing
    return [float(sample) for sample in float16_samples]

# Divides data into segments
def split_audio(audio_data, num_segments = SEGMENTS):
    data_length = len(audio_data)
    segment_length = data_length // num_segments
    return [numpy.array(audio_data[n * segment_length: (n + 1) * segment_length]) for n in range(num_segments)]

# Creates mel spectrogram using torchaudio functions
def create_mel_spectrogram(waveform):
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=SR,
        n_mels=AUDIO_N_MELS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
    )(waveform)
    mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    mel_normalized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
    return mel_normalized.squeeze().tolist()

@app.route('/process_audio', methods=['POST'])
@firebase_required
def process_audio():
    try:
        raw_audio_data = request.get_data()
        if (len(raw_audio_data)>MAX_AUDIO_BYTES):
            return jsonify({"error": "Malformed data, length check failed"}), 400

        audio_data = unpack_audio(raw_audio_data)
        audio_segments = split_audio(audio_data)

        spectrograms = []
        # Appends each spectrogram to a list
        for segment in audio_segments:
            waveform = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
            spectrogram = create_mel_spectrogram(waveform)
            spectrograms.append(spectrogram)

        response = jsonify({"mel_spectrogram": spectrograms})
        return response, 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": "Internal server error creating spectrogram"}), 500
