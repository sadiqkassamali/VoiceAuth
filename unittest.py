import unittest
from unittest.mock import patch, MagicMock
import os
import tempfile
import uuid
import sqlite3
import shutil
from VoiceAuth import (  # Replace with the correct import path
    init_db,
    save_metadata,
    convert_to_wav,
    extract_features,
    predict_rf,
    predict_hf,
    get_file_metadata
)


class TestAudioDetectionApp(unittest.TestCase):

    # Mocking sqlite3.connect to avoid actual DB changes
    @patch('sqlite3.connect')
    def test_db_initialization(self, mock_connect):
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
        mock_cursor.execute.assert_called_with(
            'CREATE TABLE IF NOT EXISTS file_metadata (uuid TEXT PRIMARY KEY, file_path TEXT, model_used TEXT, prediction_result TEXT, confidence REAL, timestamp TEXT, format TEXT, upload_count INTEGER DEFAULT 1)'
        )

    @patch('joblib.load')  # Mock loading model
    @patch('librosa.load')  # Mocking librosa.load to avoid file reading
    def test_extract_features(self, mock_load, mock_joblib):
        # Setup mock return values
        mock_load.return_value = ([], 16000)
        mock_joblib.return_value = MagicMock()

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            features = extract_features(temp_file.name)
            self.assertIsNotNone(features)
        finally:
            os.remove(temp_file.name)

    # Mock pydub AudioSegment to simulate file conversion
    @patch('pydub.AudioSegment.from_file')
    def test_convert_to_wav(self, mock_from_file):
        # Simulate the conversion
        mock_from_file.return_value = MagicMock()

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            wav_path = convert_to_wav(temp_file.name)
            self.assertTrue(wav_path.endswith(".wav"))
        finally:
            os.remove(temp_file.name)

    @patch('joblib.load')  # Mock the Random Forest model loading
    def test_predict_rf(self, mock_joblib):
        # Setup mock prediction and probability values
        mock_model = MagicMock()
        mock_model.predict.return_value = [1]  # 1 means fake
        mock_model.predict_proba.return_value = [
            [0.3, 0.7]]  # Confidence of 70% fake
        mock_joblib.return_value = mock_model

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            is_fake, confidence = predict_rf(temp_file.name)
            self.assertTrue(is_fake)  # Expecting 'fake' prediction
            self.assertEqual(confidence, 0.7)  # Expected confidence of 70%
        finally:
            os.remove(temp_file.name)

    @patch('transformers.pipeline')  # Mock Hugging Face pipeline
    def test_predict_hf(self, mock_pipeline):
        # Setup mock prediction from Hugging Face
        mock_pipeline.return_value = [{'label': 'fake', 'score': 0.85}]

        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.close()

        try:
            is_fake, confidence = predict_hf(temp_file.name)
            self.assertTrue(is_fake)  # Expecting 'fake' prediction
            self.assertEqual(confidence, 0.85)  # Expected confidence
        finally:
            os.remove(temp_file.name)

    def test_save_metadata(self):
        # Test metadata saving and updating the database
        file_uuid = str(uuid.uuid4())
        file_path = '/path/to/audio/file.wav'
        model_used = "Random Forest"
        prediction_result = "Fake"
        confidence = 0.8

        # Mock sqlite3 connection to avoid actual DB changes
        with patch('sqlite3.connect') as mock_connect:
            mock_cursor = MagicMock()
            mock_connect.return_value.cursor.return_value = mock_cursor

            already_seen = save_metadata(
                file_uuid,
                file_path,
                model_used,
                prediction_result,
                confidence)

            self.assertFalse(already_seen)  # Assert that it's a new entry
            mock_cursor.execute.assert_called_once_with(
                'INSERT INTO file_metadata (uuid, file_path, model_used, prediction_result, confidence, timestamp, format) VALUES (?, ?, ?, ?, ?, ?, ?)',
                (file_uuid, file_path, model_used, prediction_result, confidence, str(datetime.datetime.now()), '.wav')
            )

            # Simulate the case where the file has been uploaded before
            mock_cursor.fetchone.return_value = [1]  # File already seen once
            already_seen = save_metadata(
                file_uuid,
                file_path,
                model_used,
                prediction_result,
                confidence)
            # Assert that it was an existing file
            self.assertTrue(already_seen)

    def test_get_file_metadata(self):
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(b"dummy data")
        temp_file.close()

        file_format, file_size, audio_length, bitrate = get_file_metadata(
            temp_file.name)

        self.assertEqual(file_format, '.wav')
        self.assertGreater(file_size, 0)  # File size should be greater than 0
        self.assertGreater(audio_length, 0)  # Length should be positive
        self.assertGreater(bitrate, 0)  # Bitrate should be positive

        os.remove(temp_file.name)

    # Mock os.remove to prevent actual file deletion during tests
    @patch('os.remove')
    def test_run_thread(self, mock_remove):
        # Here you can test your threading functionality,
        # for now, it's skipped as this would require more advanced thread
        # handling.
        pass


if __name__ == "__main__":
    unittest.main()
