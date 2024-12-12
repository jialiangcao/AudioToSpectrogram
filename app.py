from flask import Flask, request, jsonify
import torch
import torchaudio

app = Flask(__name__)

# Constants
SR = 44100
AUDIO_N_MELS = 64
N_FFT = 1024
HOP_LENGTH = N_FFT // 4

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        audio_array = request.json.get("audio")
        waveform = torch.tensor(audio_array, dtype=torch.float32).unsqueeze(0)

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_mels=AUDIO_N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )(waveform)

        mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        mel_normalized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        
        return jsonify({"mel_spectrogram": mel_normalized.squeeze().tolist()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
