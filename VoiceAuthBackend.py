import platform
import subprocess
from multiprocessing import freeze_support

from transformers import pipeline
from pydub import AudioSegment
import numpy as np
from sklearn.manifold import TSNE
from mutagen.wave import WAVE  # For WAV files
from mutagen.mp3 import MP3  # For MP3 files
import matplotlib.pyplot as plt
import matplotlib
import librosa
import joblib
import threading
import tempfile
import sys
import sqlite3
import shutil
import datetime
import logging
import os
import tensorflow_hub as hub
import PyQt5
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
TF_ENABLE_ONEDNN_OPTS=0
TF_CPP_MIN_LOG_LEVEL=2


freeze_support()
matplotlib.use("Agg")
# Check if running in a PyInstaller bundle
if getattr(sys, "frozen", False):
    # Add the ffmpeg path for the bundled executable
    base_path = sys._MEIPASS
    os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
else:
    # Add ffmpeg path for normal script execution
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"

def get_base_path():
    if getattr(sys, "frozen", False):  # Check if the app is running as a PyInstaller executable
        base_path = os.path.abspath(sys._MEIPASS)  # PyInstaller temp directory
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))  # Script's directory

    if logging.debug:
        print(f"Debug: Base path resolved to: {base_path} (OS: {os.name}, Platform: {sys.platform})")

    return base_path


def setup_logging(log_filename: str = "audio_detection.log") -> None:
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="a"),
            logging.StreamHandler(),
        ],
    )


config = {"sample_rate": 16000, "n_mfcc": 40}

# Determine if running as a standalone executable
if getattr(sys, "frozen", False):
    base_path = os.path.dirname(sys._MEIPASS)
else:
    base_path = os.path.dirname(".")


def get_model_path(filename):
    if getattr(sys, "frozen", False):
        base_path = sys._MEIPASS
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))

    # Ensure the dataset directory exists
    dataset_path = os.path.join(base_path, "dataset")
    os.makedirs(dataset_path, exist_ok=True)
    return os.path.join(dataset_path, filename)


# Load the Random Forest model
rf_model_path = get_model_path("deepfakevoice.joblib")


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
    pipe = pipeline("audio-classification",
                    model="MelodyMachine/Deepfake-audio-detection-V2")
    print("model-melody model loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")

from transformers import pipeline

try:
    pipe = pipeline("audio-classification", model="HyperMoon/wav2vec2-base-960h-finetuned-deepfake", framework="pt")
except Exception as e:
    print(f"Error loading pipeline: {e}")

db_path = None

def init_db():
    global db_path

    if getattr(sys, "frozen", False):  # If running as a bundled app
        base_path = sys._MEIPASS
        temp_dir = tempfile.mkdtemp()  # Create a temp directory
        db_path = os.path.join(temp_dir, "metadata.db")

        # Copy the database from the bundled resources if it doesn't exist
        if not os.path.exists(db_path):
            original_db = os.path.join(base_path, "DB", "metadata.db")
            shutil.copy(original_db, db_path)
    else:
        base_path = os.path.abspath(os.path.dirname(__file__))
        db_path = os.path.join(base_path, "DB", "metadata.db")

    logging.info(f"Using database path: {db_path}")

    # Ensure DB directory exists if running unbundled
    if not getattr(sys, "frozen", False):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    # Connect to the SQLite database and create table if not exists
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
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
    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        raise RuntimeError("Unable to open or create the database file") from e


def save_metadata(
        file_uuid,
        file_path,
        model_used,
        prediction_result,
        confidence):
    global db_path

    if db_path is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check if the file's UUID already exists in the database
        cursor.execute(
            "SELECT upload_count FROM file_metadata WHERE uuid = ?", (file_uuid,))
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
            # If the file doesn't exist, insert a new record with upload_count
            # = 1
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

    except sqlite3.Error as e:
        logging.error(f"SQLite error: {e}")
        return True  # Indicate an error occurred


VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"

# Load models
try:
    print("Loading VGGish model...")
    vggish_model = hub.load(VGGISH_MODEL_URL)
    print("VGGish model loaded successfully.")

    print("Loading YAMNet model...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    print("YAMNet model loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")

# YAMNet labels (you can download the label file if needed)
YAMNET_LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"


# Function to extract features and classify audio with YAMNet
def predict_yamnet(file_path):
    """
    Process audio and predict labels using the YAMNet model.
    :param audio_path: Path to the audio file
    :return: Tuple of (top_label, confidence)
    """
    try:
        # Load and preprocess the audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or unreadable.")

        # Run YAMNet model
        scores, embeddings, spectrogram = yamnet_model(audio)
        scores_np = scores.numpy()

        # Get the most confident label
        top_label_index = np.argmax(scores_np, axis=1)[0]
        top_label = yamnet_labels[top_label_index]
        confidence = scores_np[0, top_label_index]
        return top_label, confidence
    except Exception as e:
        raise RuntimeError(f"Error processing YAMNet prediction: {e}")


# Initialize the database when the script runs
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
            audio.write_audiofile(temp_wav_path, codec="pcm_s16le")
        elif file_ext == ".wav":
            return file_path
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return temp_wav_path
    except Exception as e:
        logging.error(f"Error converting {file_path} to WAV: {e}")
        raise


def load_yamnet_labels():
    import requests

    response = requests.get(YAMNET_LABELS_URL)
    return [line.split(",")[2].strip()
            for line in response.text.strip().split("\n")[1:]]


yamnet_labels = load_yamnet_labels()


# Function to extract features and make predictions with VGGish
def predict_vggish(file_path):
    """
    Process audio and predict embeddings using the VGGish model.
    :param audio_path: Path to the audio file
    :return: Embeddings (numpy array)
    """
    try:
        # Load and preprocess the audio
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or unreadable.")

        # Pad or truncate to 1 second (16000 samples)
        audio = (
            audio[:16000]
            if len(audio) > 16000
            else np.pad(audio, (0, max(0, 16000 - len(audio))))
        )

        # Run VGGish model to get embeddings
        embeddings = vggish_model(audio)
        return embeddings.numpy()
    except Exception as e:
        raise RuntimeError(f"Error processing VGGish prediction: {e}")


# Feature extraction function for Random Forest model
def extract_features(file_path):
    wav_path = convert_to_wav(file_path)
    try:
        audio, sample_rate = librosa.load(wav_path, sr=config["sample_rate"])
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=config["n_mfcc"])
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_mean = mfccs_mean.reshape(1, -1)
        if wav_path != file_path:
            os.remove(wav_path)
        return mfccs_mean
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {e}")


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
        logging.error(f"Error during prediction: Random Forest {e}")
        return None, None  # Return a safe fallback value


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
        logging.error(f"Error during prediction: Hugging Face {e}")
        return None, None  # Return a safe fallback value


def predict_hf2(file_path):
    """Predict using the Hugging Face model 960h."""
    try:
        # Run prediction using the Hugging Face pipeline-960h
        prediction = pipe2(file_path)

        # Extract the result and confidence score
        is_fake = prediction[0]["label"] == "fake"
        confidence = min(prediction[0]["score"], 0.99)

        return is_fake, confidence

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, 0.0

    except Exception as e:
        logging.error(f"Error during prediction: 960h {e}")
        return None, None  # Return a safe fallback value


def typewriter_effect(text_widget, text, typing_speed=0.05):
    if hasattr(text_widget, "delete") and hasattr(text_widget, "insert"):
        # Tkinter environment
        for i in range(len(text) + 1):
            text_widget.delete("1.0", "end")  # Clear the text area
            # Insert the progressively typed text
            text_widget.insert("end", text[:i])
            text_widget.yview("end")  # Scroll to the end
            text_widget.update()  # Update the widget
            threading.Event().wait(
                typing_speed
            )  # Wait for a bit before the next character
    else:
        pass


# Revised scoring labels
def get_score_label(confidence):
    if confidence > 0.90:
        return "Almost certainly real"
    elif confidence > 0.80:
        return "Probably real but with slight doubt"
    elif confidence > 0.65:
        return "High likelihood of being fake, use caution"
    else:
        return "Considered fake: quality of audio does matter, do check for false positive just in case.."


def get_file_metadata(file_path):
    # Get basic file properties
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
    file_format = os.path.splitext(file_path)[-1].lower()  # Get file extension

    # Load audio data using librosa
    y, sr = librosa.load(file_path, sr=None)
    audio_length = librosa.get_duration(y=y, sr=sr)  # Duration in seconds

    # Determine channels
    channels = 1 if len(y.shape) == 1 else y.shape[0]

    # Initialize metadata fields
    bitrate = None
    additional_metadata = {}

    # Use mutagen for specific metadata extraction
    if file_format == ".mp3":
        audio = MP3(file_path)
        bitrate = audio.info.bitrate / 1000  # Convert to Mbps
        additional_metadata = (
            {key: value for key, value in audio.tags.items()} if audio.tags else {}
        )
    elif file_format == ".wav":
        audio = WAVE(file_path)
        bitrate = (
                          audio.info.sample_rate * audio.info.bits_per_sample * audio.info.channels
                  ) / 1e6  # Mbps

    # Prepare metadata string
    metadata = (
        f"File Path: {file_path}\n"
        f"Format: {file_format[1:]}\n"
        f"Size (MB): {file_size:.2f}\n"
        f"Audio Length (s): {audio_length:.2f}\n"
        f"Sample Rate (Hz): {sr}\n"
        f"Channels: {channels}\n"
        f"Bitrate (Mbps): {bitrate:.2f}\n"
    )

    if additional_metadata:
        metadata += "Additional Metadata:\n"
        for key, value in additional_metadata.items():
            metadata += f"  {key}: {value}\n"

    additional_metadata = {"channels": channels, "sample_rate": sr}
    return file_format, file_size, audio_length, bitrate, additional_metadata


def visualize_mfcc(temp_file_path):
    """Function to visualize MFCC features."""
    # Load the audio file
    audio_data, sr = librosa.load(temp_file_path, sr=None)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    # Create a new figure for the MFCC plot
    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs, aspect="auto", origin="lower", cmap="coolwarm")
    plt.title("MFCC Features")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB")

    # Save the plot to a file and show it
    plt.tight_layout()
    plt_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "mfccfeatures.png")
    plt.savefig(plt_file_path)
    # Open the file based on the OS
    if platform.system() == "Windows":
        os.startfile(plt_file_path)
        return plt_file_path
    else:
        subprocess.run(["open", plt_file_path], check=True)
        return plt_file_path

def create_mel_spectrogram(temp_file_path):
    audio_file = os.path.join(temp_file_path)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)

    y, sr = librosa.load(audio_file)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    librosa.display.specshow(
        log_mel_spectrogram, sr=sr, x_axis="time", y_axis="mel", cmap="inferno"
    )
    plt.colorbar(format="%+2.0f dB")
    plt.title("Mel Spectrogram")
    plt.tight_layout()
    plt.savefig("melspectrogram.png")
    mel_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "melspectrogram.png")
    plt.savefig(mel_file_path)
    if platform.system() == "Windows":
        os.startfile(mel_file_path)
        return mel_file_path
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", mel_file_path], check=True)
        return mel_file_path
    else:  # Linux and others
        subprocess.run(["xdg-open", mel_file_path], check=True)
        return mel_file_path


# Function to visualize embeddings using t-SNE
def visualize_embeddings_tsne(file_path, output_path="tsne_visualization.png"):
    # Apply t-SNE to reduce dimensions to 2D
    embeddings = predict_vggish(file_path)

    # Check the number of samples (rows in embeddings)
    n_samples = embeddings.shape[0]

    # If there is only one sample, t-SNE cannot be performed
    if n_samples <= 1:
        print(
            f"t-SNE cannot be performed with only {n_samples} sample(s). Skipping visualization."
        )
        # Optionally, save a default plot or handle it however you want
        plt.figure(figsize=(10, 6))
        plt.text(
            0.5,
            0.5,
            "Not enough samples for t-SNE",
            fontsize=12,
            ha="center")
        plt.title("t-SNE Visualization of Audio Embeddings")
        plt.savefig(output_path)
        plt.close()
        os.startfile(output_path)
        return

    # Ensure perplexity is less than n_samples and set a valid float value
    # Ensure perplexity is less than n_samples
    perplexity = min(30, n_samples - 1)
    # Set a minimum valid value for perplexity
    perplexity = max(5.0, perplexity)

    # Apply t-SNE to reduce dimensions to 2D
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings)

    # Plot the reduced embeddings
    plt.figure(figsize=(10, 6))
    plt.scatter(
        reduced_embeddings[:, 0],
        reduced_embeddings[:, 1],
        c="blue",
        alpha=0.7,
        edgecolors="k",
    )
    plt.title("t-SNE Visualization of Audio Embeddings")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()

    # Save and show the plot
    plt.savefig(output_path)
    plt.close()
    # Open the file based on the OS
    if platform.system() == "Windows":
        os.startfile(output_path)
        return output_path
    else:  # macOS
        subprocess.run(["open", output_path], check=True)
        return output_path
