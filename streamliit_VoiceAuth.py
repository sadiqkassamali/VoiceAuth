import streamlit as st
import os
import shutil
import uuid
import time
import threading
import librosa
from VoiceAuthBackend import (predict_rf, predict_hf, get_score_label, get_file_metadata,
                              typewriter_effect, save_metadata, visualize_mfcc, create_mel_spectrogram, predict_hf2)

# Initialize Streamlit App
st.set_page_config(page_title="VoiceAuth - Deepfake Audio Detector", layout="centered")

# Configure the ffmpeg path
os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"

# Global variables
temp_dir = "temp_dir"
os.makedirs(temp_dir, exist_ok=True)

# Utility function to clean up temp directory on exit
def clean_temp_dir():
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)

# File selection and upload
st.title("VoiceAuth - Deepfake Audio Detector")
st.subheader("Select an audio file to analyze")

uploaded_files = st.file_uploader(
    "Upload Audio Files", type=["mp3", "wav", "ogg", "flac", "aac", "m4a", "mp4", "mov", "avi", "mkv", "webm"], accept_multiple_files=True
)

selected_model = st.radio(
    "Select Model",
    ("All", "Random Forest", "Melody", "960h"),
    index=0
)

progress_bar = st.progress(0)
log_area = st.empty()

# Function to log messages in the Streamlit interface
def update_log(message):
    log_area.text_area("Logs", value=message, height=200, key="log_area")

# Prediction function
def run_analysis(file_path):
    file_uuid = str(uuid.uuid4())
    temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))

    # Copy uploaded file to temp directory
    with open(temp_file_path, 'wb') as f:
        f.write(file_path.read())

    update_log("File uploaded successfully.")
    audio_length = librosa.get_duration(path=temp_file_path)

    def update_progress(step, text="Processing..."):
        progress_bar.progress(step)
        update_log(text)

    try:
        start_time = time.time()
        update_progress(0.1, "Starting analysis...")

        # Model predictions
        rf_is_fake = hf_is_fake = hf2_is_fake = False
        rf_confidence = hf_confidence = hf2_confidence = 0.0
        combined_confidence = 0.0

        # Functions for model predictions
        def run_rf_model():
            return predict_rf(temp_file_path)

        def run_hf_model():
            return predict_hf(temp_file_path)

        def run_hf2_model():
            return predict_hf2(temp_file_path)

        # Select and run the models
        if selected_model == "All":
            rf_is_fake, rf_confidence = run_rf_model()
            hf_is_fake, hf_confidence = run_hf_model()
            hf2_is_fake, hf2_confidence = run_hf2_model()
        elif selected_model == "Random Forest":
            rf_is_fake, rf_confidence = run_rf_model()
        elif selected_model == "Melody":
            hf_is_fake, hf_confidence = run_hf_model()
        elif selected_model == "960h":
            hf2_is_fake, hf2_confidence = run_hf2_model()

        # Calculate combined confidence
        confidences = [rf_confidence, hf_confidence, hf2_confidence]
        valid_confidences = [conf for conf in confidences if conf > 0]

        if valid_confidences:
            combined_confidence = sum(valid_confidences) / len(valid_confidences)
        else:
            combined_confidence = 0.0

        combined_result = rf_is_fake or hf_is_fake or hf2_is_fake
        result_text = get_score_label(combined_confidence)

        # Display results
        st.write(f"**Prediction Result**: {result_text}")
        st.write(f"**Confidence**: {combined_confidence:.2f}")

        # Get file metadata
        file_format, file_size, audio_length, bitrate = get_file_metadata(temp_file_path)

        metadata_log = (
            f"File Path: {temp_file_path}\n"
            f"Format: {file_format}\n"
            f"Size (MB): {file_size:.2f}\n"
            f"Audio Length (s): {audio_length:.2f}\n"
            f"Bitrate (Mbps): {bitrate:.2f}\n"
        )
        update_log(metadata_log)

        # Save metadata
        model_used = selected_model if selected_model != "All" else "Random Forest, Melody, and 960h"
        prediction_result = "Fake" if combined_result else "Real"
        save_metadata(file_uuid, temp_file_path, model_used, prediction_result, combined_confidence)

        visualize_mfcc(temp_file_path)
        create_mel_spectrogram(temp_file_path)
        update_progress(1.0, "Analysis completed!")

    except Exception as e:
        st.error(f"Error: {e}")

# Run the analysis when the user clicks the button
if uploaded_files:
    for uploaded_file in uploaded_files:
        st.write(f"Analyzing `{uploaded_file.name}`...")
        run_analysis(uploaded_file)

# Cleanup on exit
clean_temp_dir()
