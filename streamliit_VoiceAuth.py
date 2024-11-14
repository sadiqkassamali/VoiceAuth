import streamlit as st
import uuid
import os
import time
import shutil
import librosa
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from VoiceAuthBackend import predict_rf, predict_hf, predict_hf2, save_metadata, get_score_label, get_file_metadata, \
    visualize_mfcc, create_mel_spectrogram

# Logging Setup
log_filename = "audio_detection.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename, mode="a"), logging.StreamHandler()],
)

# Streamlit Page Configuration
st.set_page_config(page_title="VoiceAuth - Deepfake Audio Detector", layout="wide")
st.title("VoiceAuth - Deepfake Audio and Voice Detector")
st.subheader("Detect Deepfake Audio and Voices")

# File Upload
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "flac", "ogg"])

# Model Selection
model_option = st.radio("Choose Prediction Model", ("All", "Random Forest", "Melody", "Gustking"))

# Button to Run Predictions
run_button = st.button("Run Prediction")

# Progress Bar and Labels
progress_bar = st.progress(0)
confidence_label = st.empty()
log_box = st.text_area("Logs", height=200)
eta_label = st.empty()


def update_progress(progress, text="", eta=None):
    """Update the progress bar and logs."""
    progress_bar.progress(progress)
    if text:
        st.text(text)
    if eta:
        eta_label.text(f"Estimated Time Remaining: {eta:.2f} seconds")


def run_predictions(uploaded_file, model_option):
    """Handle the prediction process."""
    if not uploaded_file:
        st.error("Please upload a valid audio file.")
        return

    file_uuid = str(uuid.uuid4())
    temp_dir = "temp_dir"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    audio_length = librosa.get_duration(path=temp_file_path)

    start_time = time.time()

    # Functions for Model Predictions
    def run_rf_model():
        return predict_rf(temp_file_path)

    def run_hf_model():
        return predict_hf(temp_file_path)

    def run_hf2_model():
        return predict_hf2(temp_file_path)

    update_progress(0.1, "Starting analysis...")

    rf_is_fake = hf_is_fake = hf2_is_fake = False
    rf_confidence = hf_confidence = hf2_confidence = 0.0
    combined_confidence = 0.0

    try:
        # Run models based on user selection
        if model_option == "All":
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(run_rf_model): "Random Forest",
                    executor.submit(run_hf_model): "Melody",
                    executor.submit(run_hf2_model): "Gustking",
                }
                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        if model_name == "Random Forest":
                            rf_is_fake, rf_confidence = future.result()
                        elif model_name == "Melody":
                            hf_is_fake, hf_confidence = future.result()
                        elif model_name == "Gustking":
                            hf2_is_fake, hf2_confidence = future.result()
                    except Exception as e:
                        log_box.text(f"Error in {model_name} model: {e}")

        elif model_option == "Random Forest":
            rf_is_fake, rf_confidence = run_rf_model()
            combined_confidence = rf_confidence

        elif model_option == "Melody":
            hf_is_fake, hf_confidence = run_hf_model()
            combined_confidence = hf_confidence

        elif model_option == "Gustking":
            hf2_is_fake, hf2_confidence = run_hf2_model()
            combined_confidence = hf2_confidence

        # Calculate combined results if multiple models are used
        confidences = [rf_confidence, hf_confidence, hf2_confidence]
        valid_confidences = [conf for conf in confidences if conf > 0]

        if valid_confidences:
            combined_confidence = sum(valid_confidences) / len(valid_confidences)
        combined_result = rf_is_fake or hf_is_fake or hf2_is_fake

        update_progress(0.8, "Finalizing results...")
        total_time_taken = time.time() - start_time
        remaining_time = total_time_taken / 0.7 - total_time_taken
        update_progress(0.9, "Almost done...", eta=remaining_time)

        # Display Results
        result_text = get_score_label(combined_confidence)
        confidence_label.markdown(f"**Confidence**: {result_text} ({combined_confidence:.2f})")
        log_box.text(f"Combined Confidence: {combined_confidence:.2f}\nResult: {result_text}")

        # Get and display metadata
        file_format, file_size, audio_length, bitrate = get_file_metadata(temp_file_path)

        # Save metadata
        model_used = model_option if model_option != "All" else "All Models"
        prediction_result = "Fake" if combined_result else "Real"
        save_metadata(file_uuid, temp_file_path, model_used, prediction_result, combined_confidence)

        # Visualization (MFCC and Mel Spectrogram)
        visualize_mfcc(temp_file_path)
        create_mel_spectrogram(temp_file_path)

        update_progress(1.0, "Completed.")

    except Exception as e:
        st.error(f"An error occurred: {e}")
        logging.error(f"Error: {e}")


# Trigger prediction process when button is clicked
if run_button and uploaded_file:
    run_predictions(uploaded_file, model_option)
