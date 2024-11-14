import datetime
import logging
import os
import sqlite3
import sys
import tempfile
import threading
import warnings
from tkinter import messagebox

import joblib
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from tf_keras.src.utils import load_img
from transformers import pipeline

matplotlib.use("Agg")


def setup_logging(
        log_filename: str = "audio_detection.log") -> None:
    """Sets up logging to both file and console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(
                log_filename,
                mode="a"),
            logging.StreamHandler()],
    )


# Suppress TensorFlow deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

os.environ["PATH"] += os.pathsep + r"ffmpeg"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"
# Configuration settings
config = {"sample_rate": 16000, "n_mfcc": 40}

# Determine if running as a standalone executable
if getattr(sys, "frozen", False):
    # Running in a PyInstaller bundle
    base_path = os.path.dirname(sys._MEIPASS)
else:
    # Running as a script
    base_path = os.path.dirname(".")


def get_model_path(filename):
    """Get the absolute path to the model file, compatible with PyInstaller."""
    if getattr(sys, "frozen", False):
        # Running as a bundled executable
        base_path = os.path.dirname(sys._MEIPASS)
    else:
        # Running as a script
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, "dataset", filename)


# Load the Random Forest model
rf_model_path = get_model_path("deepfakevoice.joblib")
rf_model = None

try:
    print(f"Loading Random Forest model from {rf_model_path}...")
    rf_model = joblib.load(rf_model_path)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {rf_model_path}")
except Exception as e:
    raise RuntimeError("Error during loading models") from e

# Load Hugging Face model-melody
try:
    print("Loading Hugging Face model...")
    pipe = pipeline(
        "audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2")
    print("model-melody model loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")

# Load Hugging Face model-Gustking
try:
    print("Loading Hugging Face model...")
    pipe2 = pipeline(
        "audio-classification", model="Gustking/wav2vec2-large-xlsr-deepfake-audio-classification")
    print("Gustking model loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")


# Database initialization function
def init_db():
    # Get the path to the current directory or temporary
    # directory for PyInstaller
    if getattr(sys, "frozen", False):  # If running as a bundled app
        base_path = os.path.dirname(sys._MEIPASS)
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))

    db_path = os.path.join(base_path, "DB", "metadata.db")

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create the table if it doesn't exist
    cursor.execute(
        """
    CREATE TABLE IF NOT EXISTS file_metadata (
        uuid TEXT PRIMARY KEY,
        file_path TEXT,
        model_used TEXT,
        prediction_result TEXT,
        confidence REAL,
        timestamp TEXT,
        format TEXT,
        upload_count INTEGER DEFAULT 1
    )
    """
    )
    conn.commit()
    conn.close()


def save_metadata(
        file_uuid,
        file_path,
        model_used,
        prediction_result,
        confidence):
    conn = sqlite3.connect("DB/metadata.db")
    cursor = conn.cursor()

    # Check if the file's UUID already exists in the database
    cursor.execute(
        "SELECT upload_count FROM file_metadata WHERE uuid = ?",
        (file_uuid,
         ))
    result = cursor.fetchone()

    if result:
        # If the file exists, increment the upload_count
        new_count = result[0] + 1
        cursor.execute(
            "UPDATE file_metadata SET upload_count = ?, timestamp = ? WHERE uuid = ?",
            (new_count, str(datetime.datetime.now()), file_uuid),
        )
        already_seen = True
    else:
        # If the file doesn't exist, insert a new record with
        # upload_count = 1
        cursor.execute(
            """
            INSERT INTO file_metadata (uuid, file_path, model_used, prediction_result, confidence, timestamp, format)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                file_uuid,
                file_path,
                model_used,
                prediction_result,
                confidence,
                str(datetime.datetime.now()),
                os.path.splitext(file_path)[-1].lower(),
            ),
        )
        already_seen = False

    conn.commit()
    conn.close()
    return already_seen


# Call the database initialization at the start of the program
init_db()


# Convert various formats to WAV
def convert_to_wav(file_path):
    try:
        import moviepy.editor as mp
    except ImportError:
        raise Exception("Please install moviepy>=1.0.3 and retry")
    temp_wav_path = tempfile.mktemp(suffix=".wav")
    file_ext = os.path.splitext(file_path)[-1].lower()
    try:
        if file_ext in [
            ".mp3",
            ".ogg",
            ".wma",
            ".aac",
            ".flac",
            ".alac",
            ".aiff",
            ".m4a",
        ]:
            audio = AudioSegment.from_file(file_path)
            audio.export(temp_wav_path, format="wav")
        elif file_ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            video = mp.VideoFileClip(file_path)
            audio = video.audio
            audio.write_audiofile(
                temp_wav_path, codec="pcm_s16le")
        elif file_ext == ".wav":
            return file_path
        else:
            raise ValueError(
                f"Unsupported file format: {file_ext}")
        return temp_wav_path
    except Exception as e:
        logging.error(
            f"Error converting {file_path} to WAV: {e}")
        raise


# Feature extraction function for Random Forest model
def extract_features(file_path):
    wav_path = convert_to_wav(file_path)
    try:
        audio, sample_rate = librosa.load(
            wav_path, sr=config["sample_rate"])
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=config["n_mfcc"])
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_mean = mfccs_mean.reshape(1, -1)
        if wav_path != file_path:
            os.remove(wav_path)
        return mfccs_mean
    except Exception as e:
        raise RuntimeError(
            f"Error extracting features from {file_path}: {e}")


def predict_rf(file_path):
    """Predict using the Random Forest model."""
    if rf_model is None:
        raise ValueError("Random Forest model not loaded.")

    # Extract features from the audio file
    features = extract_features(file_path)

    # Ensure features are in the correct shape for prediction
    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    try:
        # Make predictions using the loaded Random Forest model
        prediction = rf_model.predict(features)
        confidence = rf_model.predict_proba(features)[0][1]
        is_fake = prediction[0] == 1
        return is_fake, confidence
    except Exception as e:
        messagebox.showerror(
            "Error", f"Error during prediction: {e}")
        raise RuntimeError(
            "Error during prediction: random forest") from e


def predict_hf(file_path):
    """Predict using the Hugging Face model."""
    try:
        # Run prediction using the Hugging Face pipeline
        prediction = pipe(file_path)

        # Extract the result and confidence score
        is_fake = prediction[0]["label"] == "fake"
        confidence = min(prediction[0]["score"], 0.99)

        return is_fake, confidence

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, 0.0

    except Exception as e:
        messagebox.showerror(
            "Error", f"Error during prediction: {e}")
        raise RuntimeError(
            "Error during prediction: melody") from e


def predict_hf2(file_path):
    """Predict using the Hugging Face model Gustking."""
    try:
        # Run prediction using the Hugging Face pipeline-Gustking
        prediction = pipe2(file_path)

        # Extract the result and confidence score
        is_fake = prediction[0]["label"] == "fake"
        confidence = min(prediction[0]["score"], 0.99)

        return is_fake, confidence

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, 0.0

    except Exception as e:
        messagebox.showerror(
            "Error", f"Error during prediction: {e}")
        raise RuntimeError(
            "Error during prediction: Gustking") from e


# Typewriter effect for logging
def typewriter_effect(text_widget, text):
    for i in range(len(text) + 1):
        text_widget.delete("1.0", "end")
        text_widget.insert("end", text[:i])
        text_widget.yview("end")
        text_widget.update()
        threading.Event().wait(0.01)


# Revised scoring labels
def get_score_label(confidence):
    if confidence > 0.90:
        return "Almost certainly real"
    elif confidence > 0.80:
        return "Probably real but with slight doubt"
    elif confidence > 0.65:
        return "High likelihood of being fake, use caution"
    else:
        return "Considered fake: Should...quality of audio does matter."


def get_file_metadata(file_path):
    """Extract metadata details such as file format, size, length, and
    bitrate."""
    file_size = os.path.getsize(
        file_path) / (1024 * 1024)  # Size in MB
    audio_length = librosa.get_duration(
        filename=file_path)  # Length in seconds
    # Bitrate in Mbps
    bitrate = (file_size * 8) / \
              audio_length if audio_length else 0
    file_format = os.path.splitext(file_path)[-1].lower()

    return file_format, file_size, audio_length, bitrate


def visualize_mfcc(temp_file_path):
    """Function to visualize MFCC features."""
    # Load the audio file
    audio_data, sr = librosa.load(temp_file_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    # Create a new figure for the MFCC plot
    plt.figure(figsize=(10, 4))
    plt.imshow(
        mfccs,
        aspect="auto",
        origin="lower",
        cmap="coolwarm")
    plt.title("MFCC Features")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB")

    # Save the plot to a file and show it
    plt.tight_layout()
    plt_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "mfcc_features.png")
    plt.savefig(plt_file_path)
    os.startfile(plt_file_path)


def create_mel_spectrogram(temp_file_path):
    audio_file = os.path.join(temp_file_path)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    y, sr = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel', cmap='inferno')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.savefig('melspectrogram.png')
    mel_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "melspectrogram.png")
    plt.savefig(mel_file_path)
    os.startfile(mel_file_path)
