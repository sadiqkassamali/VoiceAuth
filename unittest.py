import datetime
import os
import tempfile
import uuid
import sqlite3
import shutil
import pytest
from unittest.mock import patch, MagicMock
from VoiceAuth import (  # Replace with the correct import path
    init_db,
    save_metadata,
    convert_to_wav,
    extract_features,
    predict_rf,
    predict_hf,
    get_file_metadata
)


# Test for database initialization
@patch('sqlite3.connect')
def test_db_initialization(mock_connect):
    mock_cursor = MagicMock()
    mock_connect.return_value.cursor.return_value = mock_cursor

    init_db()

    mock_connect.assert_called_once_with('DB/metadata.db')
    mock_cursor.execute.assert_called_with(
        '''CREATE TABLE IF NOT EXISTS file_metadata (
        uuid TEXT PRIMARY KEY,
        file_path TEXT,
        model_used TEXT,
        prediction_result TEXT,
        confidence REAL,
        timestamp TEXT,
        format TEXT,
        upload_count INTEGER DEFAULT 1
    )''')

# Test for feature extraction


@patch('joblib.load')
@patch('librosa.load')
def test_extract_features(mock_load, mock_joblib):
    mock_load.return_value = ([], 16000)
    mock_joblib.return_value = MagicMock()

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        features = extract_features(temp_file.name)
        assert features is not None
    finally:
        os.remove(temp_file.name)

# Test for converting audio to wav


@patch('pydub.AudioSegment.from_file')
def test_convert_to_wav(mock_from_file):
    mock_from_file.return_value = MagicMock()

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        wav_path = convert_to_wav(temp_file.name)
        assert wav_path.endswith(".wav")
    finally:
        os.remove(temp_file.name)

# Test for Random Forest prediction


@patch('joblib.load')
def test_predict_rf(mock_joblib):
    mock_model = MagicMock()
    mock_model.predict.return_value = [1]  # 1 means fake
    mock_model.predict_proba.return_value = [
        [0.3, 0.7]]  # Confidence of 70% fake
    mock_joblib.return_value = mock_model

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        is_fake, confidence = predict_rf(temp_file.name)
        assert is_fake is True  # Expecting 'fake' prediction
        assert confidence == 0.7  # Expected confidence of 70%
    finally:
        os.remove(temp_file.name)

# Test for Hugging Face prediction


@patch('transformers.pipeline')
def test_predict_hf(mock_pipeline):
    mock_pipeline.return_value = [{'label': 'fake', 'score': 0.85}]

    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.close()

    try:
        is_fake, confidence = predict_hf(temp_file.name)
        assert is_fake is True  # Expecting 'fake' prediction
        assert confidence == 0.85  # Expected confidence
    finally:
        os.remove(temp_file.name)

# Test for saving metadata


def test_save_metadata():
    file_uuid = str(uuid.uuid4())
    file_path = '/path/to/audio/file.wav'
    model_used = "Random Forest"
    prediction_result = "Fake"
    confidence = 0.8

    with patch('sqlite3.connect') as mock_connect:
        mock_cursor = MagicMock()
        mock_connect.return_value.cursor.return_value = mock_cursor

        already_seen = save_metadata(
            file_uuid,
            file_path,
            model_used,
            prediction_result,
            confidence)

        assert not already_seen  # Assert that it's a new entry
        mock_cursor.execute.assert_called_once_with(
            'INSERT INTO file_metadata (uuid, file_path, model_used, prediction_result, confidence, timestamp, format) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (file_uuid, file_path, model_used, prediction_result, confidence, str(datetime.datetime.now()), '.wav')
        )

        mock_cursor.fetchone.return_value = [1]  # Simulate file already seen
        already_seen = save_metadata(
            file_uuid,
            file_path,
            model_used,
            prediction_result,
            confidence)

        assert already_seen  # Assert that it's an existing file

# Test for getting file metadata


def test_get_file_metadata():
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(b"dummy data")
    temp_file.close()

    file_format, file_size, audio_length, bitrate = get_file_metadata(
        temp_file.name)

    assert file_format == '.wav'
    assert file_size > 0  # File size should be greater than 0
    assert audio_length > 0  # Length should be positive
    assert bitrate > 0  # Bitrate should be positive

    os.remove(temp_file.name)

# Test for run_thread method


@patch('os.remove')
def test_run_thread(mock_remove):
    # Placeholder for threading tests
    pass
