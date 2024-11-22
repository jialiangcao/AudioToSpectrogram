from flask import Flask, request, jsonify
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np

app = Flask(__name__)

# Sample rate
sr = 44100
# Step size for the sliding window to slice the audio to 1 second long
step_size = 0.5
# Duration
duration = 1
# Mel bins
audio_n_mels = 64
# audio channels - stereo
audio_channels = 2
# This parameter defines the number of samples in each Fast Fourier Transform (FFT) window.
n_fft = 1024
# The `hop_length` defines how much the window moves (or "hops") forward between consecutive FFT calculations.
hop_length = 1024 // 4

@app.route('/process_audio', methods=['POST'])
def process_audio():
    try:
        audioRequest = request.json
        assert audioRequest is not None
        audioArray = audioRequest.get("audio")

        audio_np = np.array(audioArray, dtype=np.float32)
        waveform = torch.tensor(audio_np).unsqueeze(0)

        transform_mel_func = torchaudio.transforms.MelSpectrogram(
            sr,
            n_mels = audio_n_mels,
            n_fft = n_fft,
            hop_length = hop_length,
        )
        power_to_db_func = torchaudio.transforms.AmplitudeToDB(stype="power")

        mel_sig = transform_mel_func(waveform)
        mel_sig = power_to_db_func(mel_sig)

        return jsonify(mel_sig), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

class TorchAudioUtils:
    # Static method to read and slice an audio file into smaller segments
    @staticmethod
    def read_slice_audio(audio_path, start_idx, duration, step_size=0.5):
        """
        Parameters:
        - audio_path: str
            Path to the audio file to be loaded.
        - start_idx: int
            The starting index for slicing the audio. Determines the starting position of the slice.
        - duration: float
            Duration of the audio slice in seconds. Specifies the length of the extracted audio segment.
        - step_size: float, optional (default=0.5)
            Time interval in seconds between each slice. Used to calculate the sliding window step size.
        """
        try:
            # Load the audio file as a tensor and retrieve the sample rate (sr)
            audio_tensor, sr = torchaudio.load(audio_path)

            # Calculate the start and end times for slicing based on index, duration, and step size
            start = start_idx * step_size
            end = start + duration

            # Slice the audio tensor from the start to the end time (in seconds), adjusted by the sample rate
            audio_tensor = audio_tensor[:, int(start * sr) : int(end * sr)]

            # Return the sliced audio tensor and sample rate
            return audio_tensor, sr
        except Exception as e:
            # Catch any errors during loading or slicing and print an error message
            print(f"Error loading {audio_path}: {e}")
            return None, None

    # Static method to change the number of audio channels (rechanneling)
    @staticmethod
    def rechannel(sig, sr, new_channel):
        """
        Parameters:
        - sig: torch.Tensor
            The input audio signal as a tensor. The first dimension represents the number of channels.
        - sr: int
            The sample rate of the audio signal. Number of audio samples per second.
        - new_channel: int
            Target number of channels (1 for mono, 2 for stereo).
        """
        # If the signal already has the desired number of channels, return it unchanged
        if sig.shape[0] == new_channel:
            return sig, sr

        if new_channel == 1:
            # Convert from stereo to mono by selecting only the first channel (index 0)
            resig = sig[:1, :]
        else:
            # Convert from mono to stereo by duplicating the mono signal into both channels
            resig = torch.cat([sig, sig])

        # Return the rechanneled signal and the same sample rate
        return (resig, sr)

    # Static method to resize the audio signal to a fixed duration
    @staticmethod
    def resize_length(signal, sr, duration):
        """
        Parameters:
        - signal: torch.Tensor
            The input audio signal as a tensor. The first dimension is the number of channels.
        - sr: int
            The sample rate of the audio signal. Used to calculate the number of samples for the desired duration.
        - duration: float
            The desired duration in seconds to which the signal should be resized.
        """
        nrows, sig_length = signal.shape  # Get the number of channels and the length of the signal
        max_seconds = sr * duration  # Calculate the maximum length in samples for the given duration

        if sig_length == max_seconds:
            # If the signal length matches the target duration, return it as is
            return signal, sr
        elif sig_length > max_seconds:
            # If the signal is longer than the target duration, truncate it to the target length
            sig = signal[:, :max_seconds]
            return sig, sr
        else:
            # If the signal is shorter than the target duration, pad it with zeros at the end
            padding_num = max_seconds - sig_length
            padding_pattern = (0, padding_num, 0, 0)  # (padding_left, padding_right, padding_top, padding_bottom)
            pad_sig = torch.nn.functional.pad(signal, padding_pattern)  # Pad the signal to the target length
            return pad_sig, sr

    # Static method to resample the audio signal to a new sample rate
    @staticmethod
    def resample(signal, sr, newsr):
        """
        Parameters:
        - signal: torch.Tensor
            The input audio signal as a tensor. The first dimension is the number of channels.
        - sr: int
            The original sample rate of the audio signal (samples per second).
        - newsr: int
            The target sample rate to which the signal will be resampled.
        """
        # If the current sample rate matches the desired sample rate, return the signal unchanged
        if sr == newsr:
            return (signal, sr)
        num_channels = signal.shape[0]  # Get the number of audio channels

        # Resample the first channel of the signal to the new sample rate
        new_sig = torchaudio.transforms.Resample(sr, newsr)(signal[:1, :])

        if num_channels > 1:
            # If the signal has more than one channel, resample the second channel separately
            new_sig_2 = torchaudio.transforms.Resample(sr, newsr)(signal[1:, :])

            # Concatenate the resampled channels into a stereo signal
            resig = torch.cat([new_sig, new_sig_2])

            # Return the resampled stereo signal and the new sample rate
            return (resig, newsr)

        # Return the resampled mono signal and the new sample rate
        return (new_sig, newsr)

if __name__ == '__main__':
    app.run(debug=True)
