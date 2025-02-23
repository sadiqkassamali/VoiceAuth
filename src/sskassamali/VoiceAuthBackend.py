import csv
import platform
import subprocess
from multiprocessing import freeze_support
import librosa.display
import scipy
from pydub import AudioSegment
import moviepy as mp
import requests

from sklearn.manifold import TSNE
from mutagen.wave import WAVE
from mutagen.mp3 import MP3
import matplotlib
import librosa
import threading
import tempfile
import sys
import sqlite3
import shutil
import datetime
import logging
import os
from transformers import pipeline
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv

import matplotlib.pyplot as plt
from IPython.display import Audio
from scipy.io import wavfile

# Fix TensorFlow environment settings (Set them only once)
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
TF_ENABLE_ONEDNN_OPTS=1
TF_CPP_MIN_LOG_LEVEL=2
# Ensure correct matplotlib backend
matplotlib.use("Agg")
# Ensure freeze support is properly initialized (useful when using multiprocessing)
freeze_support()
# Handle PyInstaller frozen mode vs. normal script execution
if getattr(sys, "frozen", False):
    base_path = os.path.join(tempfile.gettempdir(), "VoiceAuth")  # Temp directory for frozen app
else:
    base_path = os.path.join(os.getcwd(), "VoiceAuth")  # Local directory for normal execution

# Ensure the base directory exists only once
os.makedirs(base_path, exist_ok=True)

# Set temp_dir and temp_file_path correctly
temp_dir = base_path
temp_file_path = os.path.join(temp_dir, "temp_audio_file")

# FFmpeg path setup
ffmpeg_path = os.path.join(base_path, "ffmpeg")

# Ensure FFmpeg directory exists before adding it to PATH
if os.path.exists(ffmpeg_path):
    os.environ["PATH"] += os.pathsep + ffmpeg_path

# Set up Librosa cache directory
librosa_cache_dir = os.path.join(tempfile.gettempdir(), "librosa")
os.makedirs(librosa_cache_dir, exist_ok=True)  # Ensure it exists
os.environ["LIBROSA_CACHE_DIR"] = librosa_cache_dir
YAMNET_LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
def setup_logging(log_filename: str = "audio_detection.log") -> None:
    """Sets up logging to a file and console."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_filename, mode="a"),
            logging.StreamHandler(),
        ],
    )

config = {"sample_rate": 16000, "n_mfcc": 40}

try:
    print("R-forest...")
    pipe3 = pipeline("audio-classification",
                    model="alexandreacff/sew-ft-fake-detection")
    print("R-forest model loaded successfully.")
except Exception as e:
    print(f"Error loading alexandreacff/sew-ft-fake-detection model: {e}")

try:
    print("MelodyMachine/Deepfake-audio-detection-V2...")
    pipe = pipeline("audio-classification",
                    model="alexandreacff/wav2vec2-large-ft-fake-detection")
    print("model-melody model loaded successfully.")
except Exception as e:
    print(f"Error loading alexandreacff/wav2vec2-large-ft-fake-detection model: {e}")

try:
    print("openai/WpythonW-large-v3...")

    pipe2 = pipeline("audio-classification", model="WpythonW/ast-fakeaudio-detector")
    print("openai/WpythonW-large-v3...")
except Exception as e:
    print(f"Error loading open AI pipeline: {e}")

db_path = None


def init_db():
    """Initializes the database and ensures it is properly set up."""
    global db_path

    # Use the existing base_path instead of undefined get_base_path()
    db_path = os.path.join(base_path, "DB", "metadata.db")
    db_dir = os.path.dirname(db_path)

    # Ensure DB directory exists only once
    os.makedirs(db_dir, exist_ok=True)

    # If running in a frozen PyInstaller bundle, use a temp location
    if getattr(sys, "frozen", False):
        temp_dir = os.path.join(tempfile.gettempdir(), "VoiceAuth")
        os.makedirs(temp_dir, exist_ok=True)
        temp_db_path = os.path.join(temp_dir, "metadata.db")

        original_db = db_path  # The original location inside the package

        # Only copy if original DB exists
        if os.path.exists(original_db) and not os.path.exists(temp_db_path):
            shutil.copy(original_db, temp_db_path)

        db_path = temp_db_path  # Use the temp DB path

    logging.info(f"Using database path: {db_path}")

    try:

        if not os.path.exists(db_path):
            logging.warning("Original database file not found. Creating a new one.")
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

        cursor.execute(
            "SELECT upload_count FROM file_metadata WHERE uuid = ?", (file_uuid,))
        result = cursor.fetchone()

        if result:

            new_count = result[0] + 1
            cursor.execute(
                "UPDATE file_metadata SET upload_count = ?, timestamp = ? WHERE uuid = ?",
                (new_count, str(datetime.datetime.now()), file_uuid),
            )
            already_seen = True
        else:

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
        return True

try:
    print("Loading VGGish model...")
    vggish_model =   hub.load('https://www.kaggle.com/models/google/vggish/TensorFlow2/vggish/1')
    print("VGGish model loaded successfully.")

    print("Loading YAMNet model...")
    yamnet_model =  hub.load('https://www.kaggle.com/models/google/yamnet/TensorFlow2/yamnet/1')
    print("YAMNet model loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")

def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names
model = hub.load('https://tfhub.dev/google/yamnet/1')
class_map_path = model.class_map_path().numpy()
class_names = class_names_from_csv(class_map_path)

def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = scipy.signal.resample(waveform, desired_length)

    return desired_sample_rate, waveform

def predict_yamnet(file_path):
    try:

        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or unreadable.")
        # wav_file_name = 'speech_whistling2.wav'
        wav_file_name = file_path
        sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
        sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
        Audio(wav_data, rate=sample_rate)
        # Show some basic information about the audio.
        duration = len(wav_data)/sample_rate
        scores, embeddings, spectrogram = yamnet_model(audio)
        scores_np = scores.numpy()
        spectrogram_np = spectrogram.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]
        waveform = wav_data / tf.int16.max
        plt.figure(figsize=(10, 6))

        # Plot the waveform.
        plt.subplot(3, 1, 1)
        plt.plot(waveform)
        plt.xlim([0, len(waveform)])

        # Plot the log-mel spectrogram (returned by the model).
        plt.subplot(3, 1, 2)
        plt.imshow(spectrogram_np.T, aspect='auto', interpolation='nearest', origin='lower')

        # Plot and label the model output scores for the top-scoring classes.
        mean_scores = np.mean(scores, axis=0)
        top_n = 10
        top_class_indices = np.argsort(mean_scores)[::-1][:top_n]
        plt.subplot(3, 1, 3)
        plt.imshow(scores_np[:, top_class_indices].T, aspect='auto', interpolation='nearest', cmap='gray_r')

        # patch_padding = (PATCH_WINDOW_SECONDS / 2) / PATCH_HOP_SECONDS
        # values from the model documentation
        patch_padding = (0.025 / 2) / 0.01
        plt.xlim([-patch_padding-0.5, scores.shape[0] + patch_padding-0.5])
        # Label the top_N classes.
        yticks = range(0, top_n, 1)
        plt.yticks(yticks, [class_names[top_class_indices[x]] for x in yticks])
        _ = plt.ylim(-0.5 + np.array([top_n, 0]))
        plt.tight_layout()
        plt_file_path = os.path.join(
        os.path.dirname(temp_file_path),"wave.png")
        plt.savefig(plt_file_path)
        if platform.system() == "Windows":
            os.startfile(plt_file_path)
        elif platform.system() == "Darwin":  # macOS
            subprocess.run(["open", plt_file_path], check=True)
        else:  # Linux/Unix
            subprocess.run(["xdg-open", plt_file_path], check=True)
        top_label_index = np.argmax(scores_np, axis=1)[0]
        top_label = yamnet_labels[top_label_index]
        confidence = scores_np[0, top_label_index]
        return top_label, confidence
    except Exception as e:
        raise RuntimeError(f"Error processing YAMNet prediction: {e}")


init_db()


def convert_to_wav(file_path):
    temp_wav_path = tempfile.mktemp(suffix=".wav")
    file_ext = os.path.splitext(file_path)[-1].lower()
    try:
        if file_ext in [".mp3", ".ogg", ".wma", ".aac", ".flac", ".alac", ".aiff", ".m4a"]:
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
    except Exception as e:
        logging.error(f"Error converting {file_path} to WAV: {e}")
        raise


def load_yamnet_labels():
    response = requests.get(YAMNET_LABELS_URL)
    if response.status_code == 200:
        return [line.split(",")[2].strip() for line in response.text.strip().split("\n")[1:]]
    else:
        logging.error("Failed to fetch YAMNet labels")
        return []


yamnet_labels = load_yamnet_labels()


def predict_vggish(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or unreadable.")
        audio = (
            audio[:16000]
            if len(audio) > 16000
            else np.pad(audio, (0, max(0, 16000 - len(audio))))
        )

        embeddings = vggish_model(audio)
        return embeddings.numpy()
    except Exception as e:
        raise RuntimeError(f"Error processing VGGish prediction: {e}")


def extract_features(file_path):
    wav_path = convert_to_wav(file_path)
    try:
        audio, sample_rate = librosa.load(wav_path, sr=config["sample_rate"])
        mfccs = librosa.feature.mfcc(
            y=audio, sr=sample_rate, n_mfcc=config["n_mfcc"])
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_mean = mfccs_mean.reshape(1, -1)
        if wav_path != file_path and os.path.exists(wav_path):
            os.remove(wav_path)
        return mfccs_mean
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {e}")


def predict_rf(file_path):
    """Predict using the R-Forest model (Hugging Face)."""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=16000)
        prediction = pipe3(audio_data)

        is_fake = prediction[0]["label"] == "fake"
        confidence = prediction[0]["score"]  # Keep full confidence scale (0 to 1)

        return is_fake, confidence

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None, None

    except Exception as e:
        logging.error(f"Error during prediction: R-forest {e}")
        return None, None


def predict_hf(file_path):
    """Predict using the Melody model (Hugging Face)."""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=16000)
        prediction = pipe(audio_data)

        is_fake = prediction[0]["label"] == "fake"
        confidence = prediction[0]["score"]

        return is_fake, confidence

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None, None

    except Exception as e:
        logging.error(f"Error during prediction: Melody {e}")
        return None, None


def predict_hf2(file_path):
    """Predict using the Hugging Face OpenAI model."""
    try:
        audio_data, sample_rate = librosa.load(file_path, sr=16000)
        prediction = pipe2(audio_data)

        is_fake = prediction[0]["label"] == "fake"
        confidence = prediction[0]["score"]

        return is_fake, confidence

    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None, None

    except Exception as e:
        logging.error(f"Error during prediction: OpenAi {e}")
        return None, None

def typewriter_effect(text_widget, text, typing_speed=0.009):
    if hasattr(text_widget, "delete") and hasattr(text_widget, "insert"):

        for i in range(len(text) + 1):
            text_widget.delete("1.0", "end")

            text_widget.insert("end", text[:i])
            text_widget.yview("end")
            text_widget.update()
            threading.Event().wait(
                typing_speed
            )
    else:
        pass


def get_score_label(is_fake, confidence):
    """Returns a result label based on model prediction and confidence scores."""
    if confidence is None or not isinstance(confidence, (int, float)):
        return "Invalid confidence value"

    if is_fake:  # If the model predicts Fake
        if confidence > 0.90:
            return "Almost certainly fake"
        elif confidence > 0.80:
            return "Probably fake but with slight doubt"
        elif confidence > 0.65:
            return "High likelihood of being fake, use caution"
        else:
            return "Possibly fake: confidence is low, double-check."

    else:  # If the model predicts Real
        if confidence > 0.90:
            return "Almost certainly real"
        elif confidence > 0.80:
            return "Probably real but with slight doubt"
        elif confidence > 0.65:
            return "High likelihood of being real, use caution"
        else:
            return "Possibly real: confidence is low, double-check."


def get_file_metadata(file_path):
    file_size = os.path.getsize(file_path) / (1024 * 1024)
    file_format = os.path.splitext(file_path)[-1].lower()

    y, sr = librosa.load(file_path, sr=None)
    audio_length = librosa.get_duration(y=y, sr=sr)

    channels = 1 if len(y.shape) == 1 else y.shape[0]

    bitrate = None
    additional_metadata = {}

    if file_format == ".mp3":
        audio = MP3(file_path)
        bitrate = audio.info.bitrate / 1000
        additional_metadata = (
            {key: value for key, value in audio.tags.items()} if audio.tags else {}
        )
    elif file_format == ".wav":
        audio = WAVE(file_path)
        bitrate = (
                          audio.info.sample_rate * audio.info.bits_per_sample * audio.info.channels
                  ) / 1e6

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

    audio_data, sr = librosa.load(temp_file_path, sr=None)

    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)

    plt.figure(figsize=(10, 4))
    plt.imshow(mfccs, aspect="auto", origin="lower", cmap="coolwarm")
    plt.title("MFCC Features")
    plt.ylabel("MFCC Coefficients")
    plt.xlabel("Time Frames")
    plt.colorbar(format="%+2.0f dB")

    plt.tight_layout()
    plt_file_path = os.path.join(
        os.path.dirname(temp_file_path),
        "mfccfeatures.png")
    plt.savefig(plt_file_path)

    if platform.system() == "Windows":
        os.startfile(plt_file_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", plt_file_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", plt_file_path], check=True)


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
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", mel_file_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", mel_file_path], check=True)


def visualize_embeddings_tsne(file_path, output_path="tsne_visualization.png"):
    embeddings = predict_vggish(file_path)

    n_samples = embeddings.shape[0]

    if n_samples <= 1:
        print(
            f"t-SNE cannot be performed with only {n_samples} sample(s). Skipping visualization."
        )

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

    perplexity = min(30, n_samples - 1)

    perplexity = max(5.0, perplexity)

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    reduced_embeddings = tsne.fit_transform(embeddings)

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

    plt.savefig(output_path)
    plt.close()

    if platform.system() == "Windows":
        os.startfile(output_path)
    elif platform.system() == "Darwin":  # macOS
        subprocess.run(["open", output_path], check=True)
    else:  # Linux/Unix
        subprocess.run(["xdg-open", output_path], check=True)
