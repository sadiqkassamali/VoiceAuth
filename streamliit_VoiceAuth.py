import streamlit as st
import uuid
import librosa
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from VoiceAuthBackend import predict_rf, predict_hf, save_metadata, visualize_mfcc, get_score_label, get_file_metadata, \
    create_mel_spectrogram

# Initialize the Streamlit interface
st.set_page_config(page_title="VoiceAuth - Deepfake Audio and Voice Detector", layout="wide")
st.title("VoiceAuth - Deepfake Audio and Voice Detector")
st.subheader("Detect Deepfake Audio and Voices")

# Upload file
uploaded_file = st.file_uploader("Upload your audio file", type=["mp3", "wav", "flac", "ogg"])

# Radio Buttons to select model
model_option = st.radio("Choose Prediction Model", ("Both", "Random Forest", "Hugging Face"))

# Button to run prediction
run_button = st.button("Run Prediction")

# Progress bar
progress_bar = st.progress(0)

# Confidence Label
confidence_label = st.empty()

# Function to update progress
def update_progress(progress, text="Processing..."):
    # Update progress bar
    progress_bar.progress(progress)

# Function to handle processing and prediction
def run_predictions(uploaded_file, model_option):
    # Generate a new UUID for this upload
    file_uuid = str(uuid.uuid4())

    # Create a temporary directory to store the file
    temp_dir = "temp_dir"
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)

    # Save the uploaded file
    os.makedirs(temp_dir, exist_ok=True)
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Get audio length for initial ETA calculation
    audio_length = librosa.get_duration(path=temp_file_path)

    # Function to update progress and ETA
    def update_eta(start_time):
        total_time_taken = time.time() - start_time
        remaining_time = total_time_taken / 0.7 - total_time_taken
        update_progress(0.9, f"Almost done... ETA: {remaining_time:.2f} seconds")

    # Run predictions based on selected model
    def run_rf_model():
        return predict_rf(temp_file_path)

    def run_hf_model():
        return predict_hf(temp_file_path)

    start_time = time.time()

    try:
        update_progress(0.1, "Starting analysis...")

        if model_option == "Both":
            # Run both models in parallel
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = {
                    executor.submit(run_rf_model): "Random Forest",
                    executor.submit(run_hf_model): "Hugging Face",
                }
                rf_is_fake = hf_is_fake = False
                rf_confidence = hf_confidence = 0.0

                for future in as_completed(futures):
                    model_name = futures[future]
                    try:
                        if model_name == "Random Forest":
                            rf_is_fake, rf_confidence = future.result()
                        elif model_name == "Hugging Face":
                            hf_is_fake, hf_confidence = future.result()
                    except Exception as e:
                        update_progress(0.5, f"Error in {model_name} model: {e}")

            # Combine results
            combined_confidence = (rf_confidence + hf_confidence) / 2
            combined_result = rf_is_fake or hf_is_fake

        elif model_option == "Random Forest":
            # Run only Random Forest model
            rf_is_fake, rf_confidence = run_rf_model()
            combined_confidence = rf_confidence
            combined_result = rf_is_fake

        elif model_option == "Hugging Face":
            # Run only Hugging Face model
            hf_is_fake, hf_confidence = run_hf_model()
            combined_confidence = hf_confidence
            combined_result = hf_is_fake

        # Finalizing results
        update_progress(0.8, "Finalizing results...")
        update_eta(start_time)

        result_text = get_score_label(combined_confidence)
        confidence_label.markdown(f"**Confidence**: {result_text} ({combined_confidence:.2f})")

        # Get file metadata
        file_format, file_size, audio_length, bitrate = get_file_metadata(temp_file_path)

        update_progress(1.0, "Completed.")  # Update completion progress

        # Save metadata
        model_used = model_option if model_option != "Both" else "Random Forest and Hugging Face"
        prediction_result = "Fake" if combined_result else "Real"
        save_metadata(file_uuid, temp_file_path, model_used, prediction_result, combined_confidence)

        # Visualize MFCC (if needed)
        visualize_mfcc(temp_file_path)
        create_mel_spectrogram(temp_file_path)

    except Exception as e:
        st.error(f"An error occurred: {e}")


# Run the prediction when button is pressed
if run_button and uploaded_file:
    run_predictions(uploaded_file, model_option)
