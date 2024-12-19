from transformers import pipeline
import pytest
from unittest.mock import MagicMock, patch
import uuid
import tempfile
import sqlite3
import shutil
import os
import matplotlib

from VideoAuthBackend import save_metadata
from VoiceAuthBackend import init_db, convert_to_wav

matplotlib.use("Agg")  # Ensure this is at the top


@pytest.fixture(scope="module", autouse=True)
def cleanup_db():
    """
    Fixture to clean up the database before and after tests.
    """
    db_path = "DB/metadata.db"
    if os.path.exists(db_path):
        os.remove(db_path)
    yield
    if os.path.exists(db_path):
        os.remove(db_path)


def test_init_db():
    db_path = "DB/metadata.db"
    init_db()

    # Check if the table was created
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT name FROM sqlite_master WHERE type="table" AND name="file_metadata"'
    )
    result = cursor.fetchone()
    conn.close()

    assert result is not None, "Table 'file_metadata' was not created in the database."


@pytest.fixture
def mock_file_metadata():
    return {
        "file_uuid": str(uuid.uuid4()),
        "file_path": "test_audio.wav",
        "model_used": "Random Forest",
        "prediction_result": "Real",
        "confidence": 0.98,
    }


def test_save_metadata(mock_file_metadata):
    init_db()
    save_metadata(**mock_file_metadata)

    conn = sqlite3.connect("DB/metadata.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM file_metadata WHERE uuid = ?",
                   (mock_file_metadata["file_uuid"],))
    result = cursor.fetchone()
    conn.close()

    assert result is not None, "Metadata was not saved to the database."
    assert result[1] == mock_file_metadata["file_path"], "File path mismatch."
    assert result[2] == mock_file_metadata["model_used"], "Model used mismatch."
    assert (
        result[3] == mock_file_metadata["prediction_result"]
    ), "Prediction result mismatch."
    assert result[4] == mock_file_metadata["confidence"], "Confidence mismatch."
    assert result[5] is not None, "Timestamp was not saved."


@pytest.mark.parametrize(
    "file_path, expected_extension",
    [
        ("test_audio.mp3", ".wav"),
        ("test_audio.ogg", ".wav"),
        ("test_audio.mp4", ".wav"),
        ("test_audio.wav", ".wav"),
    ],
)
def test_convert_to_wav(file_path, expected_extension):
    with tempfile.NamedTemporaryFile(suffix=file_path, delete=False) as temp_file:
        temp_file.write(b"fake audio content")

    try:
        converted_file = convert_to_wav(temp_file.name)
        assert converted_file.endswith(
            expected_extension
        ), f"Expected {expected_extension}, got {converted_file[-4:]}"
    finally:
        os.remove(temp_file.name)
        if os.path.exists(converted_file):
            os.remove(converted_file)


def test_convert_to_wav_invalid_format():
    with pytest.raises(ValueError):
        convert_to_wav("test_audio.txt")


@patch("VoiceAuth.joblib.load")
def test_model_loading(mock_load):
    mock_load.return_value = MagicMock()
    from VoiceAuthBackend import rf_model

    assert rf_model is not None, "Random Forest model did not load."


@patch("VoiceAuth.pipeline")
def test_hugging_face_model_loading(mock_pipeline):
    mock_pipeline.return_value = MagicMock()
    pipe = pipeline("audio-classification",
                    model="MelodyMachine/Deepfake-audio-detection-V2")
    assert pipe is not None, "Hugging Face model did not load."


def test_db_connection():
    db_path = "DB/metadata.db"
    init_db()

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT count(*) FROM file_metadata")
    count = cursor.fetchone()[0]
    conn.close()

    assert count == 0, "Initial database is not empty."


if __name__ == "__main__":
    pytest.main()
