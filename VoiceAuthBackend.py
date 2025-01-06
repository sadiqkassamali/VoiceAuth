import platform
import subprocess
from multiprocessing import freeze_support
import librosa.display
import numpy as np
import torch
from pydub import AudioSegment

from sklearn.manifold import TSNE
from mutagen.wave import WAVE  
from mutagen.mp3 import MP3  
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
from transformers import pipeline, Wav2Vec2Processor
import tensorflow as tf

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
TF_ENABLE_ONEDNN_OPTS=0
TF_CPP_MIN_LOG_LEVEL=2
freeze_support()
matplotlib.use("Agg")


def get_base_path():
    if getattr(sys, "frozen", False):  
        return r"\\tmp\\voiceauth"
    return os.path.abspath(os.path.dirname(__file__))  

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device set to use {device}")
device = "GPU" if tf.config.list_physical_devices("GPU") else "CPU"
print(f"Device set to use {device}")
def setup_environment():
    base_path = get_base_path()
    os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
    os.environ["LIBROSA_CACHE_DIR"] = os.path.join(base_path, "librosa_cache")
    if not os.path.exists(os.environ["LIBROSA_CACHE_DIR"]):
        os.makedirs(os.environ["LIBROSA_CACHE_DIR"])

setup_environment()


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


def get_model_path(filename):
    base_path = get_base_path()
    dataset_path = os.path.join(base_path, "dataset")
    os.makedirs(dataset_path, exist_ok=True)
    return os.path.join(dataset_path, filename)

rf_model_path = get_model_path("deepfakevoice.joblib")


try:
    print(f"Loading Random Forest model from {rf_model_path}...")
    rf_model = joblib.load(rf_model_path)
    print("Random Forest model loaded successfully.")
except FileNotFoundError:
    print(f"Model file not found at {rf_model_path}")
except Exception as e:
    raise RuntimeError("Error during loading models") from e



try:
    print("MelodyMachine/Deepfake-audio-detection-V2...")
    pipe = pipeline("audio-classification",
                    model="MelodyMachine/Deepfake-audio-detection-V2")
    print("model-melody model loaded successfully.")
except Exception as e:
    print(f"Error loading Hugging Face model: {e}")

try:
    print("facebook/wav2vec2-base-960h...")
    pipe2 = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    print("facebook/wav2vec2-base-960h.")
except Exception as e:
    print(f"Error loading FB pipeline: {e}")

db_path = None

def init_db():
    global db_path
    base_path = get_base_path()

    if getattr(sys, "frozen", False):  
        temp_dir = tempfile.mkdtemp()  
        db_path = os.path.join(temp_dir, "metadata.db")
        original_db = os.path.join(base_path, "DB", "metadata.db")
        if not os.path.exists(db_path):
            shutil.copy(original_db, db_path)
    else:  
        db_path = os.path.join(base_path, "DB", "metadata.db")
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    logging.info(f"Using database path: {db_path}")

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


VGGISH_MODEL_URL = "https://tfhub.dev/google/vggish/1"
YAMNET_MODEL_URL = "https://tfhub.dev/google/yamnet/1"


try:
    print("Loading VGGish model...")
    vggish_model = hub.load(VGGISH_MODEL_URL)
    print("VGGish model loaded successfully.")

    print("Loading YAMNet model...")
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    print("YAMNet model loaded successfully.")

except Exception as e:
    print(f"Error loading models: {e}")


YAMNET_LABELS_URL = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"



def predict_yamnet(file_path):

    try:
        
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        if len(audio) == 0:
            raise ValueError("Audio file is empty or unreadable.")

        
        scores, embeddings, spectrogram = yamnet_model(audio)
        scores_np = scores.numpy()

        
        top_label_index = np.argmax(scores_np, axis=1)[0]
        top_label = yamnet_labels[top_label_index]
        confidence = scores_np[0, top_label_index]
        return top_label, confidence
    except Exception as e:
        raise RuntimeError(f"Error processing YAMNet prediction: {e}")



init_db()



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
        if wav_path != file_path:
            os.remove(wav_path)
        return mfccs_mean
    except Exception as e:
        raise RuntimeError(f"Error extracting features from {file_path}: {e}")


def predict_rf(file_path):
    """Predict using the Random Forest model."""
    if rf_model is None:
        raise ValueError("Random Forest model not loaded.")


    features = extract_features(file_path)


    if len(features.shape) == 1:
        features = features.reshape(1, -1)

    try:

        prediction = rf_model.predict(features)
        confidence = rf_model.predict_proba(features)[0][1]
        is_fake = prediction[0] == 1
        return is_fake, confidence
    except Exception as e:
        logging.error(f"Error during prediction: Random Forest {e}")
        return None, None


def predict_hf(file_path):
    """Predict using the Hugging Face model."""
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(file_path, sr=16000)

        # Run the pipeline using the loaded audio waveform instead of the file path
        prediction = pipe(audio_data)

        is_fake = prediction[0]["label"] == "fake"
        confidence = min(prediction[0]["score"], 0.99)

        return is_fake, confidence

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, 0.0

    except Exception as e:
        logging.error(f"Error during prediction: Hugging Face {e}")
        return None, None


def predict_hf2(file_path):
    """Predict using the Hugging Face model 960h."""
    try:
        # Load the audio file
        audio_data, sample_rate = librosa.load(file_path, sr=16000)

        # Run the pipeline using the loaded audio waveform instead of the file path
        prediction = pipe2(audio_data)

        is_fake = prediction[0]["label"] == "fake"
        confidence = min(prediction[0]["score"], 0.99)

        return is_fake, confidence

    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None, 0.0

    except Exception as e:
        logging.error(f"Error during prediction: 960h {e}")
        return None, None


def typewriter_effect(text_widget, text, typing_speed=10):
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



def get_score_label(confidence):
    if confidence is None or not isinstance(confidence, (int, float)):
        return "Invalid confidence value"

    if confidence > 0.90:
        return "Almost certainly real"
    elif confidence > 0.80:
        return "Probably real but with slight doubt"
    elif confidence > 0.65:
        return "High likelihood of being fake, use caution"
    else:
        return "Considered fake: quality of audio does matter, do check for false positive just in case.."


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
    elif platform.system() == "Darwin":  
        subprocess.run(["open", mel_file_path], check=True)
        return mel_file_path
    else:  
        subprocess.run(["xdg-open", mel_file_path], check=True)
        return mel_file_path



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
        return output_path
    else:  
        subprocess.run(["open", output_path], check=True)
        return output_path
