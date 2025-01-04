from flask import Flask, request, jsonify
import torch
import torchaudio
import gzip
import io
import struct

app = Flask(__name__)

# Constants
SR = 44100
AUDIO_N_MELS = 64
N_FFT = 1024
HOP_LENGTH = N_FFT // 4

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        # debug
        print("Received POST request to /process_audio")
        compressedAudioData = request.get_data()
        print(f"Received compressed binary audio data")
        
        print(f"Length of compressed data: {len(compressedAudioData)}")
        with gzip.GzipFile(fileobj=io.BytesIO(compressedAudioData)) as f:
            audioData = f.read()
            print(f"Length of uncompressed data: {len(audioData)}")

        doubleCount = len(audioData) // 8 # 8 for 64 bit double values
        audioData = struct.unpack(f'{doubleCount}d', audioData)
        waveform = torch.tensor(list(audioData), dtype=torch.float32).unsqueeze(0)
        # using float32 may be incorrect
        print(f"Shape of waveform tensor: {waveform.shape}")

        mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR,
            n_mels=AUDIO_N_MELS,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
        )(waveform)

        print(f"Shape of mel spectrogram: {mel_spec.shape}")

        mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
        print(f"Shape of mel_db: {mel_db.shape}")
        
        mel_normalized = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
        print(f"Shape of normalized mel spectrogram: {mel_normalized.shape}")
        
        response = jsonify({"mel_spectrogram": mel_normalized.squeeze().tolist()})
        print("Sending response")
        return response, 200
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)