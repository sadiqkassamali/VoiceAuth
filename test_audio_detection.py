import pytest
import sqlite3
import os
import tempfile
import uuid
import shutil
from unittest.mock import patch, MagicMock
from transformers import pipeline
from VoiceAuth import init_db, save_metadata, convert_to_wav
import matplotlib
matplotlib.use('Agg')

# Test the database initialization function
def test_init_db():
    # Ensure database file exists
    db_path = 'DB/metadata.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    init_db()

    # Check if the table was created
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT name FROM sqlite_master WHERE type="table" AND name="file_metadata"')
    result = cursor.fetchone()
    conn.close()

    assert result is not None, "Table 'file_metadata' was not created in the database."

# Test save_metadata function
@pytest.fixture
def mock_file_metadata():
    return {
        "file_uuid": str(uuid.uuid4()),
        "file_path": "test_audio.wav",
        "model_used": "Random Forest",
        "prediction_result": "Real",
        "confidence": 0.98
    }

def test_save_metadata(mock_file_metadata):
    # Initialize database
    init_db()

    # Save metadata
    save_metadata(**mock_file_metadata)

    # Verify data was inserted into the database
    conn = sqlite3.connect('DB/metadata.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM file_metadata WHERE uuid = ?',
                   (mock_file_metadata["file_uuid"],))
    result = cursor.fetchone()
    conn.close()

    assert result is not None, "Metadata was not saved to the database."
    assert result[1] == mock_file_metadata["file_path"], "File path does not match the saved metadata."
    assert result[2] == mock_file_metadata["model_used"], "Model used does not match the saved metadata."
    assert result[3] == mock_file_metadata["prediction_result"], "Prediction result does not match the saved metadata."
    assert result[4] == mock_file_metadata["confidence"], "Confidence does not match the saved metadata."
    assert result[5] is not None, "Timestamp was not saved."

# Test convert_to_wav function
@pytest.mark.parametrize(
    "file_path, expected_extension",
    [
        ("test_audio.mp3", ".wav"),
        ("test_audio.ogg", ".wav"),
        ("test_audio.mp4", ".wav"),
        ("test_audio.wav", ".wav")
    ]
)
def test_convert_to_wav(file_path, expected_extension):
    # Create a temporary file for testing
    with tempfile.NamedTemporaryFile(suffix=file_path, delete=False) as temp_file:
        temp_file.write(b"fake audio content")

    try:
        # Run the function
        converted_file = convert_to_wav(temp_file.name)

        # Verify if the converted file has the correct extension
        assert converted_file.endswith(
            expected_extension), f"Expected {expected_extension}, got {converted_file[-4:]}"

    finally:
        # Clean up
        os.remove(temp_file.name)
        if os.path.exists(converted_file):
            os.remove(converted_file)

# Test logging and exception handling for invalid file format
def test_convert_to_wav_invalid_format():
    with pytest.raises(ValueError):
        # Testing an unsupported file format
        convert_to_wav("test_audio.txt")

# Mock the model loading function to avoid actual loading in tests
@patch('VoiceAuth.joblib.load')
def test_model_loading(mock_load):
    # Simulate a successful load
    mock_load.return_value = MagicMock()

    # Test if the model is loaded correctly
    try:
        # Assuming `rf_model` is loaded in the global context (replace
        # 'your_module' with the actual module name)
        from VoiceAuth import rf_model  # Trigger the loading of the model
        assert rf_model is not None, "Random Forest model did not load."
    except RuntimeError as e:
        pytest.fail(f"Model loading failed: {str(e)}")

# Test Hugging Face model loading using mock
@patch('VoiceAuth.pipeline')
def test_hugging_face_model_loading(mock_pipeline):
    # Simulate a successful pipeline initialization
    mock_pipeline.return_value = MagicMock()

    # Test if the Hugging Face model is loaded correctly
    pipe = pipeline(
        "audio-classification",
        model="MelodyMachine/Deepfake-audio-detection-V2")
    assert pipe is not None, "Hugging Face model did not load."

# Additional tests for database and file handling

def test_db_connection():
    # Ensure that the database connection is successful and handle cleanup
    db_path = 'DB/metadata.db'
    if os.path.exists(db_path):
        os.remove(db_path)

    init_db()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT count(*) FROM file_metadata')
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0, "Initial database is not empty."

# Cleanup test database after all tests
@pytest.fixture(scope='module', autouse=True)
def cleanup_db():
    yield
    # Cleanup database after tests
    if os.path.exists('DB/metadata.db'):
        os.remove('DB/metadata.db')

# Run the tests
if __name__ == '__main__':
    pytest.main()
