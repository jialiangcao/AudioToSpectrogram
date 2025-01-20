from flask import Flask, request, jsonify
import torch
import torchaudio
import gzip
import io
import struct
import numpy

print(numpy.__version__)
app = Flask(__name__)

SR = 44100
AUDIO_N_MELS = 64
N_FFT = 1024
HOP_LENGTH = N_FFT // 4

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        spectrograms = [None]*5
        compressedAudioData = request.get_data()
        
        with gzip.GzipFile(fileobj=io.BytesIO(compressedAudioData)) as f:
            audioData = f.read()

        doubleCount = len(audioData) // 8
        audioData = struct.unpack(f'{doubleCount}d', audioData)

        for n in range(5):
            dataLength = len(audioData)
            arrayLength = dataLength//5
            tempData = numpy.array(audioData[n * arrayLength: (n + 1) * arrayLength])
        
            waveform = torch.tensor(tempData, dtype=torch.float32).unsqueeze(0)

            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=SR,
                n_mels=AUDIO_N_MELS,
                n_fft=N_FFT,
                hop_length=HOP_LENGTH,
            )(waveform)

            mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            mel_normalized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            spectrograms[n] = mel_normalized.squeeze().tolist()
        
        response = jsonify({"mel_spectrogram": spectrograms})
        return response, 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500 