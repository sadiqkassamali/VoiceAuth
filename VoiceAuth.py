from multiprocessing import freeze_support

from PIL import Image
import matplotlib
import customtkinter as ctk

from customtkinter import *

from tkinter.scrolledtext import ScrolledText
from tkinter import Menu, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
import webbrowser
import uuid
import traceback
import time
import threading
import sys
import shutil
import logging
import os

from VoiceAuthBackend import (get_file_metadata,
                              get_score_label, predict_hf, predict_hf2,
                              predict_rf, predict_vggish, predict_yamnet,
                              save_metadata, typewriter_effect, visualize_mfcc, create_mel_spectrogram,
                              visualize_embeddings_tsne,
                              )

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
TF_ENABLE_ONEDNN_OPTS=0
TF_CPP_MIN_LOG_LEVEL=2

freeze_support()
matplotlib.use("TkAgg")
# Check if running in a PyInstaller bundle
if getattr(sys, "frozen", False):
    # Add the ffmpeg path for the bundled executable
    base_path = sys._MEIPASS
    os.environ["PATH"] += os.pathsep + os.path.join(base_path, "ffmpeg")
else:
    # Add ffmpeg path for normal script execution
    os.environ["PATH"] += os.pathsep + os.path.abspath("ffmpeg")
os.environ["LIBROSA_CACHE_DIR"] = "/tmp/librosa"


def setup_logging(log_filename: str = "audio_detection.log") -> None:
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


def run():
    global confidence_label, result_entry, eta_label
    log_textbox.delete("1.0", "end")
    progress_bar.set(0)

    file_path = str(file_entry.get())
    # Check if a valid file is selected
    if not file_path or not os.path.isfile(file_path):
        messagebox.showerror("Error", "Please select a valid audio file.")
        predict_button.configure(state="normal")  # Re-enable the button
        return
    # Generate a new UUID for this upload
    file_uuid = str(uuid.uuid4())

    temp_dir = "temp_dir"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, os.path.basename(file_path))

    try:
        # Copy the selected file to the temporary directory
        shutil.copy(file_path, temp_file_path)
    except Exception as e:
        messagebox.showerror("Error", f"Failed to copy the file: {e}")
        predict_button.configure(state="normal")  # Re-enable the button
        return

    import librosa
    try:
        audio_length = librosa.get_duration(path=temp_file_path)
    except Exception as e:
        print(f"Error loading audio file: {e}")

    def update_progress(step, text="Processing...", eta=None):
        progress_bar.set(step)
        log_textbox.insert("end", f"{text}\n")
        log_textbox.yview("end")
        if eta is not None:
            eta_label.configure(text=f"Estimated Time: {eta:.2f} seconds")

    def run_thread():
        predict_button.configure(state="normal")

    try:
        start_time = time.time()
        update_progress(0.1, "Starting analysis...")

        # Feature extraction
        extraction_start = time.time()
        update_progress(0.2, "Extracting features...")

        selected = selected_model.get()

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
            embeddings = run_vggish_model(temp_file_path)
            log_textbox.insert(
                "end", f"VGGish Embeddings: {embeddings[:5]}...\n")
        except Exception as e:
            log_textbox.insert("end", f"VGGish model error: {e}\n")

        try:
            update_progress(0.5, "Running YAMNet model...")
            top_label, confidence = run_yamnet_model(temp_file_path)
            log_textbox.insert(
                "end", f"YAMNet Prediction: {top_label} (Confidence: {confidence:.2f})\n", )
        except Exception as e:
            log_textbox.insert("end", f"YAMNet model error: {e}\n")

        if selected == "All":
            # Run All models in parallel using
            # ThreadPoolExecutor
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
                        print(f"Error in {model_name} model: {e}")

            confidences = [rf_confidence, hf_confidence, hf2_confidence]
            valid_confidences = [conf for conf in confidences if isinstance(conf, (int, float)) and conf > 0]
            if valid_confidences:
                combined_confidence = sum(
                    valid_confidences) / len(valid_confidences)
            else:
                combined_confidence = (
                    0.0  # Default if none of the models produced a valid result
                )

            combined_result = rf_is_fake or hf_is_fake or hf2_is_fake

        elif selected == "Random Forest":
            # Run only Random Forest model
            rf_is_fake, rf_confidence = run_rf_model()
            combined_confidence = rf_confidence
            combined_result = rf_is_fake

        elif selected == "Melody":
            # Run only Hugging Face model
            hf_is_fake, hf_confidence = run_hf_model()
            combined_confidence = hf_confidence
            combined_result = hf_is_fake

        elif selected == "960h":
            # Run only Hugging Face model
            hf2_is_fake, hf2_confidence = run_hf2_model()
            combined_confidence = hf2_confidence
            combined_result = hf2_is_fake

        # Finalizing results
        update_progress(0.8, "Finalizing results...")
        total_time_taken = time.time() - start_time
        remaining_time = total_time_taken / (0.7) - total_time_taken
        update_progress(0.9, "Almost done...", eta=remaining_time)

        # Determine result text
        result_text = get_score_label(combined_confidence)
        confidence_label.configure(
            text=f"Confidence: {result_text} ({combined_confidence:.2f})"
        )
        result_label.configure(text=result_text)

        # Get file metadata
        file_format, file_size, audio_length, bitrate, additional_metadata = (
            get_file_metadata(temp_file_path)
        )

        log_message = (
            f"File Path: {temp_file_path}\n"
            f"Format: {file_format}\n"
            f"Size (MB): {file_size:.2f}\n"
            f"Audio Length (s): {audio_length:.2f}\n"
            f"Bitrate (Mbps): {bitrate:.2f}\n"
            f"Result: {result_text}\n"
        )

        # Add Random Forest prediction if selected
        try:
            if selected in ["Random Forest", "All"]:
                log_message += f"RF Prediction: {'Fake' if rf_is_fake else 'Real'} (Confidence: {rf_confidence:.2f})\n"
        except NameError:
            log_message += "Random Forest model did not produce a result.\n"

        # Add Melody prediction if selected
        try:
            if selected in ["Melody", "All"]:
                log_message += f"Melody Prediction: {'Fake' if hf_is_fake else 'Real'} (Confidence: {hf_confidence:.2f})\n"
        except NameError:
            log_message += "Melody model did not produce a result.\n"

        # Add 960h prediction if selected
        try:
            if selected in ["960h", "All"]:
                log_message += f"960h Prediction: {'Fake' if hf2_is_fake else 'Real'} (Confidence: {hf2_confidence:.2f})\n"
        except NameError:
            log_message += "960h model did not produce a result.\n"

        # Calculate combined confidence only for models that succeeded
        valid_confidences = [
            conf for conf in [
                rf_confidence,
                hf_confidence,
                hf2_confidence] if conf > 0]
        if valid_confidences:
            combined_confidence = sum(
                valid_confidences) / len(valid_confidences)
            result_text = get_score_label(combined_confidence)
            log_message += (
                f"Combined Confidence: {combined_confidence:.2f}\n"
                f"Result: {result_text}\n"
            )
        else:
            log_message += "No valid predictions were made due to model failures.\n"

        # Log the result with the typewriter effect
        typewriter_effect(log_textbox, log_message)

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

        file_status_label.configure(
            text="File already in database" if already_seen else "New file uploaded")

        visualize_mfcc(temp_file_path)
        create_mel_spectrogram(temp_file_path)
        visualize_embeddings_tsne(file_path)
        update_progress(1.0, "Completed.")
        eta_label.configure(text="Estimated Time: Completed")

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        raise RuntimeError("Error during processing") from e

    threading.Thread(target=run_thread, daemon=True).start()


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


def open_donate():
    """Open PayPal donation link in the web browser."""
    donate_url = "https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD"
    webbrowser.open(donate_url)


# GUI setup
temp_dir = "temp_dir"
temp_file_path = os.path.join(temp_dir, os.path.basename("."))
if os.path.exists(temp_dir):
    shutil.rmtree(temp_dir, ignore_errors=True)
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")
app = ctk.CTk()
app.title("VoiceAuth - Deepfake Audio and Voice Detector")
app.geometry("900X900")


def resource_path(relative_path):
    """Get absolute path to resource, works for development and PyInstaller."""
    try:
        # If the application is running as a PyInstaller bundle
        base_path = sys._MEIPASS
    except AttributeError:
        # If running as a script
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


# Add VGGish and YAMNet to the prediction pipeline
def run_vggish_model(relative_path):
    """Run VGGish model on audio and return embeddings."""
    embeddings = predict_vggish(relative_path)
    # You can use embeddings for downstream tasks, such as classification
    return embeddings


def run_yamnet_model(relative_path):
    """Run YAMNet model on audio and return label and confidence."""
    top_label, confidence = predict_yamnet(relative_path)
    return top_label, confidence


# Load the image using the dynamic path
logo_image = ctk.CTkImage(
    Image.open(
        resource_path("images/bot2.png")),
    size=(
        128,
        128))


def open_email():
    webbrowser.open("mailto:sadiqkassamali@gmail.com")


menu_bar = Menu(app)
contact_menu = Menu(menu_bar, tearoff=0)
contact_menu.add_command(
    label="For assistance: sadiqkassamali@gmail.com", command=open_email
)
contact_menu.add_separator()
contact_menu.add_command(label="Donate to Support", command=open_donate)
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
    app, text="Deepfake Audio and Voice Detector", font=("Arial", 14, "bold")
)
sub_header_label.pack(pady=5)

file_entry = ctk.CTkEntry(app, width=300)
file_entry.pack(pady=10)
# In your main function, call select_file like this:
select_button = ctk.CTkButton(app, text="Select Files", command=select_file)
select_button.pack(pady=5)

progress_bar = ctk.CTkProgressBar(app, width=300)
progress_bar.pack(pady=10)
progress_bar.set(0)

selected_model = ctk.StringVar(value="All")
model_rf = ctk.CTkRadioButton(
    app, text="R Forest", variable=selected_model, value="Random Forest"
)
model_hf = ctk.CTkRadioButton(
    app, text="Melody", variable=selected_model, value="Melody"
)
model_hf2 = ctk.CTkRadioButton(
    app,
    text="960h",
    variable=selected_model,
    value="960h")
model_All = ctk.CTkRadioButton(
    app,
    text="All",
    variable=selected_model,
    value="All")
model_rf.pack(padx=5)
model_hf.pack()
model_hf2.pack()
model_All.pack()

predict_button = ctk.CTkButton(
    app, text="Run Prediction", command=start_analysis, fg_color="green"
)
predict_button.pack(pady=20)

file_status_label = ctk.CTkLabel(
    app,
    text="",
    width=400,
    height=30,
    corner_radius=8)
file_status_label.pack(pady=10)

confidence_label = ctk.CTkLabel(app, text="Confidence: ", font=("Arial", 14))
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
eta_label = ctk.CTkLabel(app, text="Time Taken: ", font=("Arial", 12))
eta_label.pack(pady=5)

try:
    app.mainloop()
    freeze_support()
except BaseException:
    f = open("app.log", "w", encoding="utf-8")
    e = traceback.format_exc()
    f.write(str(e))
    f.close()
