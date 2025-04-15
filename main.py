from flask import Flask, request, jsonify
import torch
import torchaudio
import gzip
import io
import struct
import numpy

app = Flask(__name__)

# Constants
SR = 44100
AUDIO_N_MELS = 64
N_FFT = 1024
HOP_LENGTH = N_FFT // 4
segments = 2
byte_size = 8 # Representing 8 bytes for a float64 or Double size, must match type input type

# Converts gzipped data into raw values
def decompress_audio(compressed_audio_data):
    with gzip.GzipFile(fileobj=io.BytesIO(compressed_audio_data)) as f:
        audio_data = f.read()
    double_count = len(audio_data) // byte_size
    return struct.unpack(f'{double_count}d', audio_data)

# Divides data into segments
def split_audio(audio_data, num_segments = segments):
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
def process_audio():
    try:
        compressed_audio_data = request.get_data()
        audio_data = decompress_audio(compressed_audio_data)
        audio_segments = split_audio(audio_data)
        
        spectrograms = []
        # Appends each spectrogram to a list
        for segment in audio_segments:
            waveform = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)
            spectrogram = create_mel_spectrogram(waveform)
            spectrograms.append(spectrogram)
        
        # Return the list in a JSON serialized response
        response = jsonify({"mel_spectrogram": spectrograms})
        return response, 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500
