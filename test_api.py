import pytest
import json
import gzip
import io
import struct
import numpy as np
from main import app, decompress_audio, split_audio, create_mel_spectrogram

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def create_gzipped_audio_data(audio_data):
    double_data = struct.pack(f'{len(audio_data)}d', *audio_data)
    compressed_data = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed_data, mode='wb') as f:
        f.write(double_data)
    return compressed_data.getvalue()

def test_process_audio_success(client):
    audio_data = np.random.rand(44100 * 5)
    compressed_data = create_gzipped_audio_data(audio_data)

    response = client.post('/process_audio', data=compressed_data, content_type='application/octet-stream')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'mel_spectrogram' in data
    assert len(data['mel_spectrogram']) == 5
    assert len(data['mel_spectrogram'][0]) == 64
    assert len(data['mel_spectrogram'][0][0]) > 0

def test_process_audio_invalid_data(client):
    response = client.post('/process_audio', data=b'invalid data', content_type='application/octet-stream')

    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

def test_process_audio_empty_data(client):
    response = client.post('/process_audio', data=b'', content_type='application/octet-stream')

    assert response.status_code == 500
    data = json.loads(response.data)
    assert 'error' in data

def test_process_audio_wrong_method(client):
    response = client.get('/process_audio')

    assert response.status_code == 405

def test_process_audio_large_data(client):
    audio_data = np.random.rand(44100 * 60)
    compressed_data = create_gzipped_audio_data(audio_data)

    response = client.post('/process_audio', data=compressed_data, content_type='application/octet-stream')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'mel_spectrogram' in data
    assert len(data['mel_spectrogram']) == 5

def test_process_audio_small_data(client):
    audio_data = np.random.rand(int(44100 * 0.1))
    compressed_data = create_gzipped_audio_data(audio_data)

    response = client.post('/process_audio', data=compressed_data, content_type='application/octet-stream')

    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'mel_spectrogram' in data
    assert len(data['mel_spectrogram']) == 5

# Individual method tests

def test_decompress_audio():
    original_data = np.random.rand(1000)
    compressed_data = create_gzipped_audio_data(original_data)
    decompressed_data = decompress_audio(compressed_data)
    assert np.allclose(original_data, decompressed_data)

def test_split_audio():
    audio_data = np.random.rand(1000)
    segments = split_audio(audio_data, num_segments=5)
    assert len(segments) == 5
    assert all(len(segment) == 200 for segment in segments)
    assert np.allclose(np.concatenate(segments), audio_data)

def test_create_mel_spectrogram():
    import torch
    waveform = torch.randn(1, 44100)
    mel_spectrogram = create_mel_spectrogram(waveform)
    assert isinstance(mel_spectrogram, list)
    assert len(mel_spectrogram) == 64
    assert all(isinstance(row, list) for row in mel_spectrogram)

def test_process_audio_consistency(client):
    audio_data = np.random.rand(44100 * 5)
    compressed_data = create_gzipped_audio_data(audio_data)

    response1 = client.post('/process_audio', data=compressed_data, content_type='application/octet-stream')
    response2 = client.post('/process_audio', data=compressed_data, content_type='application/octet-stream')

    assert response1.status_code == 200
    assert response2.status_code == 200
    data1 = json.loads(response1.data)
    data2 = json.loads(response2.data)
    assert data1 == data2

def test_process_audio_different_sample_rates(client):
    for sr in [22050, 44100, 48000]:
        audio_data = np.random.rand(sr * 5)
        compressed_data = create_gzipped_audio_data(audio_data)

        response = client.post('/process_audio', data=compressed_data, content_type='application/octet-stream')

        assert response.status_code == 200
        data = json.loads(response.data)
        assert 'mel_spectrogram' in data
        assert len(data['mel_spectrogram']) == 5
        assert len(data['mel_spectrogram'][0]) == 64
