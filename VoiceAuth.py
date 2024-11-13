import datetime
import logging
import os
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import uuid
import warnings
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import Menu, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText

import customtkinter as ctk
import joblib
import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pydub import AudioSegment
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

# Load Hugging Face model
try:
    print("Loading Hugging Face model...")
    pipe = pipeline(
        "audio-classification", model="MelodyMachine/Deepfake-audio-detection-V2")
    print("Hugging Face model loaded successfully.")
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
            "Error during prediction: random forest") from e


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
        return "High likelihood of being fake"
    else:
        return "Considered fake"


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


def select_file():
    file_paths = filedialog.askopenfilenames(
        filetypes=[
            ("Audio Files",
             "*.mp3;*.wav;*.ogg;*.flac;*.aac;*.m4a;*.mp4;*.mov;*.avi;*.mkv;*.webm",
             )])
    file_entry.delete(0, ctk.END)
    # Show multiple files
    file_entry.insert(0, ";".join(file_paths))


# Start prediction process in a new thread
def start_analysis():
    predict_button.configure(state="disabled")
    threading.Thread(target=run).start()  # Call run directly


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


def run():
    global confidence_label, result_entry, eta_label
    log_textbox.delete("1.0", "end")
    progress_bar.set(0)

    file_path = str(file_entry.get())
    # Generate a new UUID for this upload
    file_uuid = str(uuid.uuid4())

    # Create a temporary directory to store the file
    temp_dir = "temp_dir"
    temp_file_path = os.path.join(
        temp_dir, os.path.basename(file_path))

    # Move the uploaded file to the temporary directory
    os.makedirs(temp_dir, exist_ok=True)
    shutil.copy(file_path, temp_file_path)

    # Get audio length for initial ETA calculation
    audio_length = librosa.get_duration(path=temp_file_path)

    def update_progress(step, text="Processing...", eta=None):
        progress_bar.set(step)
        log_textbox.insert("end", f"{text}\n")
        log_textbox.yview("end")
        if eta is not None:
            eta_label.configure(
                text=f"Estimated Time: {eta:.2f} seconds"
            )

    def run_thread():
        predict_button.configure(state="normal")

    try:
        start_time = time.time()
        update_progress(0.1, "Starting analysis...")

        # Feature extraction
        extraction_start = time.time()
        update_progress(0.2, "Extracting features...")

        selected = selected_model.get()

        rf_is_fake = hf_is_fake = False
        rf_confidence = hf_confidence = 0.0

        # Define functions for model predictions
        def run_rf_model():
            return predict_rf(temp_file_path)

        def run_hf_model():
            return predict_hf(temp_file_path)

        if selected == "Both":
            # Run both models in parallel using
            # ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(run_rf_model): "Random Forest",
                    executor.submit(run_hf_model): "Hugging Face",
                }
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        if model_name == "Random Forest":
                            rf_is_fake, rf_confidence = future.result()
                        elif model_name == "Hugging Face":
                            hf_is_fake, hf_confidence = future.result()
                    except Exception as e:
                        print(
                            f"Error in {model_name} model: {e}")

            # Combine results
            combined_confidence = (
                                          rf_confidence + hf_confidence) / 2
            combined_result = rf_is_fake or hf_is_fake

        elif selected == "Random Forest":
            # Run only Random Forest model
            rf_is_fake, rf_confidence = run_rf_model()
            combined_confidence = rf_confidence
            combined_result = rf_is_fake

        elif selected == "Hugging Face":
            # Run only Hugging Face model
            hf_is_fake, hf_confidence = run_hf_model()
            combined_confidence = hf_confidence
            combined_result = hf_is_fake

        # Finalizing results
        update_progress(0.8, "Finalizing results...")
        total_time_taken = time.time() - start_time
        remaining_time = total_time_taken / \
                         (0.7) - total_time_taken
        update_progress(
            0.9,
            "Almost done...",
            eta=remaining_time)

        # Determine result text
        result_text = get_score_label(combined_confidence)
        confidence_label.configure(
            text=f"Confidence: {result_text} ({combined_confidence:.2f})"
        )
        result_label.configure(text=result_text)

        # Get file metadata
        file_format, file_size, audio_length, bitrate = get_file_metadata(
            temp_file_path)

        log_message = (
            f"File Path: {temp_file_path}\n"
            f"Format: {file_format}\n"
            f"Size (MB): {file_size:.2f}\n"
            f"Audio Length (s): {audio_length:.2f}\n"
            f"Bitrate (Mbps): {bitrate:.2f}\n"
        )

        if selected in ["Random Forest", "Both"]:
            log_message += f"RF Prediction: {'Fake' if rf_is_fake else 'Real'} (Confidence: {rf_confidence:.2f})\n"
        if selected in ["Hugging Face", "Both"]:
            log_message += f"HF Prediction: {'Fake' if hf_is_fake else 'Real'} (Confidence: {hf_confidence:.2f})\n"

        log_message += (
            f"Combined Confidence: {combined_confidence:.2f}\n"
            f"Result: {result_text}\n"
        )

        # Log using typewriter effect
        typewriter_effect(log_textbox, log_message)

        # Save metadata
        model_used = (selected if selected !=
                                  "Both" else "Random Forest and Hugging Face")
        prediction_result = "Fake" if combined_result else "Real"
        save_metadata(
            file_uuid,
            temp_file_path,
            model_used,
            prediction_result,
            combined_confidence,
        )

        already_seen = save_metadata(
            file_uuid,
            temp_file_path,
            model_used,
            prediction_result,
            combined_confidence,
        )

        file_status_label.configure(
            text="File already in database" if already_seen else "New file uploaded")

        visualize_mfcc(temp_file_path)

        update_progress(1.0, "Completed.")
        eta_label.configure(text="Estimated Time: Completed")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        raise RuntimeError("Error during processing") from e

    threading.Thread(target=run_thread, daemon=True).start()


# GUI setup
temp_dir = "temp_dir"
temp_file_path = os.path.join(temp_dir, os.path.basename("."))
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")
app = ctk.CTk()
app.title("VoiceAuth - Deepfake Audio and Voice Detector")
app.geometry("800x800")


def resource_path(relative_path):
    """Get absolute path to resource, works for development and PyInstaller."""
    try:
        # If the application is running as a PyInstaller bundle
        base_path = sys._MEIPASS
    except AttributeError:
        # If running as a script
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Load the image using the dynamic path
logo_image = ctk.CTkImage(Image.open(resource_path("images/bot2.png")), size=(128, 128))


def open_email():
    webbrowser.open("mailto:sadiqkassamali@gmail.com")


menu_bar = Menu(app)
contact_menu = Menu(menu_bar, tearoff=0)
contact_menu.add_command(label="For assistance: sadiqkassamali@gmail.com",command=open_email)
menu_bar.add_cascade(label="Contact", menu=contact_menu)

app.configure(menu=menu_bar)

header_label = ctk.CTkLabel(
    app,
    compound="top",
    justify=ctk.CENTER,
    image=logo_image,
    text="VoiceAuth",
    font=("Arial", 28, "bold"),
)
header_label.pack(pady=10)
sub_header_label = ctk.CTkLabel(
    app, text="Deepfake Audio and Voice Detector", font=(
        "Arial", 14, "bold"))
sub_header_label.pack(pady=5)

file_entry = ctk.CTkEntry(app, width=300)
file_entry.pack(pady=10)
# In your main function, call select_file like this:
select_button = ctk.CTkButton(
    app,
    text="Select Files",
    command=select_file)
select_button.pack(pady=5)

progress_bar = ctk.CTkProgressBar(app, width=300)
progress_bar.pack(pady=10)
progress_bar.set(0)

selected_model = ctk.StringVar(value="Both")
model_rf = ctk.CTkRadioButton(
    app,
    text="Random Forest",
    variable=selected_model,
    value="Random Forest")
model_hf = ctk.CTkRadioButton(
    app,
    text="Hugging Face",
    variable=selected_model,
    value="Hugging Face")
model_both = ctk.CTkRadioButton(
    app,
    text="Both",
    variable=selected_model,
    value="Both")
model_rf.pack()
model_hf.pack()
model_both.pack()

predict_button = ctk.CTkButton(
    app,
    text="Run Prediction",
    command=start_analysis,
    fg_color="green")
predict_button.pack(pady=20)

file_status_label = ctk.CTkLabel(
    app,
    text="",
    width=400,
    height=30,
    corner_radius=8)
file_status_label.pack(pady=10)

confidence_label = ctk.CTkLabel(
    app, text="Confidence: ", font=(
        "Arial", 14))
confidence_label.pack(pady=5)
result_entry = ctk.CTkEntry(app, width=200, state="readonly")
result_label = ctk.CTkLabel(app, text="", font=("Arial", 12))
result_label.pack(pady=10)

log_textbox = ScrolledText(
    app,
    height=8,
    bg="black",
    fg="lime",
    insertbackground="lime",
    wrap="word",
    font=("Arial", 13),
    relief="flat",
)
log_textbox.pack(padx=10, pady=10)

eta_label = ctk.CTkLabel(
    app, text="Estimated Time: ", font=(
        "Arial", 12))
eta_label.pack(pady=5)

try:
    import pyi_splash

    pyi_splash.update_text("Loading Voice Auth!")
    pyi_splash.update_text("Loading models!")
    pyi_splash.update_text("Installing database!")
    pyi_splash.update_text("Installing library!")
    pyi_splash.update_text("Almost done !")
    pyi_splash.close()
except:
    pass

try:
    app.mainloop()
except BaseException:
    f = open("app.log", "w")
    e = traceback.format_exc()
    f.write(str(e))
    f.close()
