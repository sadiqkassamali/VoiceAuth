import sys
import os
import time
import tempfile
import uuid
import logging
import datetime
import threading
import warnings
import shutil

import sqlite3

from tkinter import filedialog, messagebox, Menu
from tkinter.scrolledtext import ScrolledText

import customtkinter as ctk
import joblib
import librosa
import matplotlib.pyplot as plt
import numpy as np
from librosa.feature import mfcc
from moviepy.editor import VideoFileClip
from pydub import AudioSegment
from transformers import pipeline

# Set up logging
logging.basicConfig(filename='audio_detection.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Suppress TensorFlow deprecation warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

os.environ["PATH"] += os.pathsep + r"ffmpeg"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"
os.environ["LIBROSA_CACHE_DIR"] = "25"
# Configuration settings
config = {
    "sample_rate": 16000,
    "n_mfcc": 40
}

# Determine if running as a standalone executable
if getattr(sys, 'frozen', False):
    # Running in a PyInstaller bundle
    base_path = sys._MEIPASS
else:
    # Running as a script
    base_path = os.path.abspath(".")
# Load models

rf_model_path = os.path.join(base_path, 'dataset', 'deepfakevoice.joblib')
try:
    rf_model = joblib.load(rf_model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load the Random Forest model: {e}")

# Load Hugging Face model
pipe = pipeline(
    "audio-classification",
    model="MelodyMachine/Deepfake-audio-detection-V2")


# Database initialization function
def init_db():
    conn = sqlite3.connect('DB/metadata.db')
    cursor = conn.cursor()
    cursor.execute('''
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
    ''')
    conn.commit()
    conn.close()


# Save or update metadata in SQLite database
def save_metadata(
        file_uuid,
        file_path,
        model_used,
        prediction_result,
        confidence):
    conn = sqlite3.connect('DB/metadata.db')
    cursor = conn.cursor()
    cursor.execute(
        'SELECT upload_count FROM file_metadata WHERE uuid = ?', (file_uuid,))
    result = cursor.fetchone()
    already_seen = False

    if result:
        new_count = result[0] + 1
        cursor.execute(
            'UPDATE file_metadata SET upload_count = ?, timestamp = ? WHERE uuid = ?', (new_count, str(
                datetime.datetime.now()), file_uuid))
        already_seen = True
    else:
        cursor.execute('''
        INSERT INTO file_metadata (uuid, file_path, model_used, prediction_result, confidence, timestamp, format)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (file_uuid, file_path, model_used, prediction_result, confidence, str(datetime.datetime.now()),
              os.path.splitext(file_path)[-1].lower()))

    conn.commit()
    conn.close()
    return already_seen


# Call the database initialization at the start of the program
init_db()


# Convert various formats to WAV
def convert_to_wav(file_path):
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
                ".m4a"]:
            audio = AudioSegment.from_file(file_path)
            audio.export(temp_wav_path, format="wav")
        elif file_ext in [".mp4", ".mov", ".avi", ".mkv", ".webm"]:
            with VideoFileClip(file_path) as video:
                audio = video.audio
                audio.write_audiofile(temp_wav_path, codec="pcm_s16le")
        elif file_ext == ".wav":
            return file_path
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        return temp_wav_path
    except Exception as e:
        logging.error(f"Error converting {file_path} to WAV: {e}")
        # Log using typewriter effect
        typewriter_effect(log_textbox, log_message)

        # Save metadata
        model_used = "Random Forest and Hugging Face"
        prediction_result = "Fake" if combined_result else "Real"
        save_metadata(
            file_uuid,
            temp_file_path,
            model_used,
            prediction_result,
            combined_confidence)

        # Visualize MFCC features after predictions
        visualize_mfcc(temp_file_path)

        update_progress(1.0, "Completed.")
        eta_label.configure(text="Estimated Time: Completed")
    except Exception as e:
        logging.error(f"Error during processing: {e}")
        messagebox.showerror("Error", f"An error occurred: {e}")
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


threading.Thread(target=run_thread, daemon=True).start()


# GUI setup
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.title("Voice Auth - Deepfake Audio and Voice Detector")
app.geometry("800x700")

menu_bar = Menu(app)
contact_menu = Menu(menu_bar, tearoff=0)
contact_menu.add_command(label="For assistance: sadiqkassamali@gmail.com")
menu_bar.add_cascade(label="Contact", menu=contact_menu)
app.configure(menu=menu_bar)

header_label = ctk.CTkLabel(app, text="Voice Auth", font=("Arial", 20, "bold"))
header_label.pack(pady=20)
sub_header_label = ctk.CTkLabel(
    app,
    text="Deepfake Audio and Voice Detector",
    font=(
        "Arial",
         16))
sub_header_label.pack(pady=5)

file_entry = ctk.CTkEntry(app, width=300)
file_entry.pack(pady=10)
# In your main function, call select_file like this:
select_button = ctk.CTkButton(app, text="Select Files", command=select_file)
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

confidence_label = ctk.CTkLabel(app, text="Confidence: ", font=("Arial", 12))
confidence_label.pack(pady=5)
result_entry = ctk.CTkEntry(app, width=200, state='readonly')
result_entry.pack(pady=10)

log_textbox = ScrolledText(
    app,
    height=10,
    bg="black",
    fg="lime",
    insertbackground="lime",
    wrap="word")
log_textbox.pack(padx=10, pady=10)

eta_label = ctk.CTkLabel(app, text="Estimated Time: ", font=("Arial", 12))
eta_label.pack(pady=5)

app.mainloop()
