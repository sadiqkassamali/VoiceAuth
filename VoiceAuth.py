import logging
import os
import shutil
import sys
import threading
import time
import traceback
import uuid
import webbrowser
from concurrent.futures import ThreadPoolExecutor, as_completed
from tkinter import Menu, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import customtkinter as ctk
import librosa
import matplotlib
from PIL import Image

from VoiceAuthBackend import predict_rf, predict_hf, get_score_label, get_file_metadata, typewriter_effect, \
    save_metadata, visualize_mfcc

matplotlib.use("tkAgg")


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
