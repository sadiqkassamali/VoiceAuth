import os
import shutil
import sys
import threading
import time
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed

import librosa
import streamlit as st
from PIL import Image
import tensorflow_hub as hub
from VoiceAuthBackend import (create_mel_spectrogram, get_file_metadata,
                              get_score_label, predict_hf, predict_hf2,
                              predict_rf, predict_vggish, predict_yamnet,
                              save_metadata, typewriter_effect,
                              visualize_embeddings_tsne, visualize_mfcc)

# Setup environment variables
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"
if getattr(sys, "frozen", False):
    base_path = sys._MEIPASS
    os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
else:
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")


# Helper function to update progress
def update_progress(progress, text="Processing...", eta=None):
    st.progress(progress)
    st.text(text)
    if eta is not None:
        st.text(f"Estimated Time: {eta:.2f} seconds")


# Streamlit application layout
st.set_page_config(
    page_title="VoiceAuth - Deepfake Audio and Voice Detector",
    page_icon="images/voiceauth.webp",
    layout="wide",
    initial_sidebar_state="auto",
)
st.title("VoiceAuth - Deepfake Audio and Voice Detector")
st.markdown("### Detect fake voices using deep learning models")
logo_image = Image.open("images/bot2.png")
st.image(logo_image, width=128)

# File uploader
uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=[
        "mp3",
        "wav",
        "ogg",
        "flac",
        "aac",
        "m4a",
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm",
    ],
)

# Model selection radio buttons
model_option = st.radio(
    "Select Model(s)",
    ("Random Forest",
     "Melody",
     "960h",
     "All"))

# Button to run prediction
if uploaded_file:
    if st.button("Run Prediction"):
        st.text("Processing...")
        progress = st.progress(0)

        file_uuid = str(uuid.uuid4())
        temp_dir = "temp_dir"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(
            temp_dir, os.path.basename(
                uploaded_file.name))

        # Save the uploaded file temporarily
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        audio_length = librosa.get_duration(path=temp_file_path)

        # Start processing
        start_time = time.time()

        update_progress(0.1, "Starting analysis...")

        # Feature extraction
        extraction_start = time.time()
        update_progress(0.2, "Extracting features...")

        selected = model_option

        rf_is_fake = hf_is_fake = hf2_is_fake = False
        rf_confidence = hf_confidence = hf2_confidence = 0.0
        combined_confidence = 0.0

        # Define functions for model predictions
        def run_rf_model():
            return predict_rf(temp_file_path)

        def run_hf_model():
            return predict_hf(temp_file_path)

        def run_hf2_model():
            return predict_hf2(temp_file_path)

        try:
            update_progress(0.4, "Running VGGish model...")
            embeddings = predict_vggish(temp_file_path)
            st.text(f"VGGish Embeddings: {embeddings[:5]}...\n")
        except Exception as e:
            st.text(f"VGGish model error: {e}")

        try:
            update_progress(0.5, "Running YAMNet model...")
            top_label, confidence = predict_yamnet(temp_file_path)
            st.text(
                f"YAMNet Prediction: {top_label} (Confidence: {confidence:.2f})\n")
        except Exception as e:
            st.text(f"YAMNet model error: {e}")

        # Parallel processing based on model selection
        if selected == "All":
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(run_rf_model): "Random Forest",
                    executor.submit(run_hf_model): "Melody",
                    executor.submit(run_hf2_model): "960h",
                }
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        if model_name == "Random Forest":
                            rf_is_fake, rf_confidence = future.result()
                        elif model_name == "Melody":
                            hf_is_fake, hf_confidence = future.result()
                        elif model_name == "960h":
                            hf2_is_fake, hf2_confidence = future.result()
                    except Exception as e:
                        st.text(f"Error in {model_name} model: {e}")

            confidences = [rf_confidence, hf_confidence, hf2_confidence]
            valid_confidences = [conf for conf in confidences if conf > 0]
            if valid_confidences:
                combined_confidence = sum(
                    valid_confidences) / len(valid_confidences)
            else:
                combined_confidence = 0.0

            combined_result = rf_is_fake or hf_is_fake or hf2_is_fake

        # Handle individual model predictions (Random Forest, Melody, 960h)
        elif selected == "Random Forest":
            rf_is_fake, rf_confidence = run_rf_model()
            combined_confidence = rf_confidence
            combined_result = rf_is_fake
        elif selected == "Melody":
            hf_is_fake, hf_confidence = run_hf_model()
            combined_confidence = hf_confidence
            combined_result = hf_is_fake
        elif selected == "960h":
            hf2_is_fake, hf2_confidence = run_hf2_model()
            combined_confidence = hf2_confidence
            combined_result = hf2_is_fake

        update_progress(0.8, "Finalizing results...")
        total_time_taken = time.time() - start_time
        remaining_time = total_time_taken / 0.7 - total_time_taken
        update_progress(0.9, "Almost done...", eta=remaining_time)

        # Display results
        result_text = get_score_label(combined_confidence)
        st.text(f"Confidence: {result_text} ({combined_confidence:.2f})")
        result_label = st.text(result_text)

        # Get file metadata
        file_format, file_size, audio_length, bitrate, additional_metadata = (
            get_file_metadata(temp_file_path)
        )
        st.text(
            f"File Format: {file_format}, Size: {file_size:.2f} MB, Audio Length: {audio_length:.2f} sec, Bitrate: {bitrate:.2f} Mbps"
        )

        log_message = (
            f"File Path: {temp_file_path}\n"
            f"Format: {file_format}\n"
            f"Size (MB): {file_size:.2f}\n"
            f"Audio Length (s): {audio_length:.2f}\n"
            f"Bitrate (Mbps): {bitrate:.2f}\n"
            f"Result: {result_text}\n"
        )

        # Log the result with the typewriter effect
        typewriter_effect(st, log_message)

        # Save metadata
        model_used = selected if selected != "All" else "Random Forest, Melody and 960h"
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

        st.text(f"File already in database: {already_seen}")

        visualize_mfcc(temp_file_path)
        create_mel_spectrogram(temp_file_path)
        visualize_embeddings_tsne(uploaded_file)
        update_progress(1.0, "Completed.")
        st.text("Time Taken: Completed")


# Open PayPal donation link
def open_donate():
    donate_url = "https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD"
    webbrowser.open(donate_url)


# Contact section
contact_expander = st.expander("Contact & Support")
contact_expander.markdown(
    "For assistance: [Email](mailto:sadiqkassamali@gmail.com)")
contact_expander.markdown(
    f"[Donate to Support](https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD)"
)
