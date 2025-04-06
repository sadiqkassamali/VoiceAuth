import streamlit as st
import os
import tempfile
import logging
from voiceauthCore.core import predict_rf, predict_hf, predict_hf2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit UI
st.title("VoiceAuth - Deepfake Audio and Voice Detector")

# File uploader
uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav", "ogg", "flac", "aac", "m4a", "mp4", "mov", "avi", "mkv", "webm"])

# Model selection
model_options = ["All", "Random Forest", "Melody", "OpenAi"]
selected_model = st.radio("Select Model", model_options)

# Predict button
if st.button("Run Prediction"):
    if uploaded_file is not None:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            temp_file_path = tmp_file.name

        # Display progress bar
        progress_bar = st.progress(0)

        # Perform prediction
        try:
            progress_bar.progress(10)
            if selected_model == "All":
                rf_result = predict_rf(temp_file_path)
                hf_result = predict_hf(temp_file_path)
                hf2_result = predict_hf2(temp_file_path)
                # Combine results as needed
            elif selected_model == "Random Forest":
                result = predict_rf(temp_file_path)
            elif selected_model == "Melody":
                result = predict_hf(temp_file_path)
            elif selected_model == "OpenAi":
                result = predict_hf2(temp_file_path)
            progress_bar.progress(100)
            st.success("Prediction completed successfully.")
            # Display results
            st.write("Results:", result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error during prediction: {e}")
        finally:
            os.remove(temp_file_path)
    else:
        st.warning("Please upload an audio file before running prediction.")
