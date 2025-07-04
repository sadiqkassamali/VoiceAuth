from multiprocessing import freeze_support
from PIL import Image
import customtkinter as ctk
import tempfile
from tkinter.scrolledtext import ScrolledText
from tkinter import Menu, filedialog, messagebox
from concurrent.futures import ThreadPoolExecutor, as_completed
import webbrowser
import uuid
import time
import threading
import sys
import shutil
import logging
import os
import librosa

# Fixed imports to match backend structure
from voiceauthCore.core import (predict_hf, predict_hf2, predict_rf, predict_vggish, predict_yamnet, visualize_embeddings_tsne)
from voiceauthCore.database import save_metadata
from voiceauthCore.utils import (get_file_metadata, get_score_label, typewriter_effect, visualize_mfcc, create_mel_spectrogram)

freeze_support()

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Fixed resource path handling for compiled executables
def get_base_path():
    if getattr(sys, "frozen", False):
        return os.path.join(tempfile.gettempdir(), "VoiceAuth")
    else:
        return os.path.join(os.getcwd(), "VoiceAuth")

base_path = get_base_path()
os.makedirs(base_path, exist_ok=True)
temp_dir = base_path

def setup_logging(log_filename: str = "audio_detection.log") -> None:
    """Sets up logging to both file and console."""
    log_file_path = os.path.join(base_path, log_filename)
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file_path, mode="a"),
            logging.StreamHandler(),
        ],
    )

setup_logging()
logging.info("App starting...")

def convert_to_int(value):
    """Convert prediction result to integer for voting."""
    if isinstance(value, str):
        return 1 if value.lower() == "fake" else 0
    return 1 if value else 0

def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller."""
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def safe_predict(predict_func, file_path, model_name):
    """Safely execute prediction function with error handling."""
    try:
        result = predict_func(file_path)
        if result and isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], result[1]
        elif isinstance(result, str):
            return result, 0.5  # Default confidence for string results
        elif isinstance(result, bool):
            return "Fake" if result else "Real", 0.5
        else:
            logging.warning(f"{model_name} returned unexpected format: {result}")
            return "Error", 0.0
    except Exception as e:
        logging.error(f"{model_name} prediction failed: {e}")
        return "Error", 0.0

def run_vggish_model(file_path):
    """Run VGGish model on audio and return embeddings."""
    try:
        embeddings = predict_vggish(file_path)
        return embeddings
    except Exception as e:
        logging.error(f"VGGish model error: {e}")
        return None

def run_yamnet_model(file_path):
    """Run YAMNet model on audio and return label and confidence."""
    try:
        result = predict_yamnet(file_path)
        if isinstance(result, (tuple, list)) and len(result) >= 2:
            return result[0], result[1], result[1]  # label, class_name, confidence
        elif isinstance(result, (tuple, list)) and len(result) >= 3:
            return result[0], result[1], result[2]
        else:
            return "Unknown", "Unknown", 0.0
    except Exception as e:
        logging.error(f"YAMNet model error: {e}")
        return None, "Unknown", 0.0

def run_analysis():
    """Main analysis function that runs in a separate thread."""
    global confidence_label, result_label, eta_label, log_textbox, progress_bar, file_status_label

    try:
        log_textbox.delete("1.0", "end")
        progress_bar.set(0)

        file_path = str(file_entry.get())

        if not file_path or not os.path.isfile(file_path):
            messagebox.showerror("Error", "Please select a valid audio file.")
            return

        file_uuid = str(uuid.uuid4())
        temp_file_path = os.path.join(temp_dir, f"temp_{file_uuid}_{os.path.basename(file_path)}")

        try:
            shutil.copy(file_path, temp_file_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to copy the file: {e}")
            return

        logging.info(f"Temporary file path: {temp_file_path}")

        def update_progress(step, text="Processing...", eta=None):
            progress_bar.set(step)
            log_textbox.insert("end", f"{text}\n")
            log_textbox.yview("end")
            if eta is not None:
                eta_label.configure(text=f"Estimated Time: {eta:.2f} seconds")

        start_time = time.time()
        update_progress(0.1, "Starting analysis...")

        selected = selected_model.get()

        # Initialize variables
        rf_is_fake = hf_is_fake = hf2_is_fake = False
        rf_confidence = hf_confidence = hf2_confidence = 0.0

        def run_rf_model():
            return safe_predict(predict_rf, temp_file_path, "Random Forest")

        def run_hf_model():
            return safe_predict(predict_hf, temp_file_path, "Melody")

        def run_hf2_model():
            return safe_predict(predict_hf2, temp_file_path, "OpenAI")

        # Run VGGish model
        try:
            update_progress(0.3, "Running VGGish model...")
            embeddings = run_vggish_model(temp_file_path)
            if embeddings is not None:
                log_textbox.insert("end", f"VGGish Embeddings extracted successfully\n")
            else:
                log_textbox.insert("end", "VGGish model failed\n")
        except Exception as e:
            log_textbox.insert("end", f"VGGish model error: {e}\n")

        # Run YAMNet model
        try:
            update_progress(0.4, "Running YAMNet model...")
            top_label, inferred_class_name, confidence = run_yamnet_model(temp_file_path)
            if top_label is not None:
                log_textbox.insert("end", f"YAMNet Prediction: {inferred_class_name} (Confidence: {confidence:.2f})\n")
            else:
                log_textbox.insert("end", "YAMNet model failed\n")
        except Exception as e:
            log_textbox.insert("end", f"YAMNet model error: {e}\n")

        update_progress(0.5, "Running main models...")

        # Run selected models
        if selected == "All":
            with ThreadPoolExecutor(max_workers=3) as executor:
                futures = {
                    executor.submit(run_rf_model): "Random Forest",
                    executor.submit(run_hf_model): "Melody",
                    executor.submit(run_hf2_model): "OpenAI",
                }

                completed_count = 0
                for future in as_completed(futures):
                    model_name = futures[future]
                    completed_count += 1
                    progress = 0.5 + (completed_count * 0.1)
                    update_progress(progress, f"Completed {model_name} model...")

                    try:
                        result = future.result()
                        if model_name == "Random Forest":
                            rf_is_fake, rf_confidence = result
                        elif model_name == "Melody":
                            hf_is_fake, hf_confidence = result
                        elif model_name == "OpenAI":
                            hf2_is_fake, hf2_confidence = result

                        # Convert result for display
                        result_display = rf_is_fake if model_name == "Random Forest" else (hf_is_fake if model_name == "Melody" else hf2_is_fake)
                        result_conf = rf_confidence if model_name == "Random Forest" else (hf_confidence if model_name == "Melody" else hf2_confidence)

                        if isinstance(result_display, str):
                            display_label = result_display
                        else:
                            display_label = 'Fake' if result_display else 'Real'

                        log_textbox.insert("end", f"{model_name}: {display_label} (Confidence: {result_conf:.2f})\n")
                    except Exception as e:
                        log_textbox.insert("end", f"Error in {model_name} model: {e}\n")

        elif selected == "Random Forest":
            rf_is_fake, rf_confidence = run_rf_model()
        elif selected == "Melody":
            hf_is_fake, hf_confidence = run_hf_model()
        elif selected == "OpenAI":
            hf2_is_fake, hf2_confidence = run_hf2_model()

        update_progress(0.8, "Calculating results...")

        # Calculate combined results
        if selected == "All":
            # Voting system
            fake_votes = sum(convert_to_int(x) for x in [rf_is_fake, hf_is_fake, hf2_is_fake])
            real_votes = 3 - fake_votes
            combined_result = fake_votes > real_votes

            # Average confidence
            valid_confidences = [conf for conf in [rf_confidence, hf_confidence, hf2_confidence]
                                 if isinstance(conf, (int, float)) and conf > 0]
            combined_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
        else:
            # Single model result
            if selected == "Random Forest":
                combined_result = rf_is_fake
                combined_confidence = rf_confidence
            elif selected == "Melody":
                combined_result = hf_is_fake
                combined_confidence = hf_confidence
            elif selected == "OpenAI":
                combined_result = hf2_is_fake
                combined_confidence = hf2_confidence

        result_text = get_score_label(combined_result)

        # Update UI
        confidence_label.configure(text=f"Confidence: {result_text} ({combined_confidence:.2f})")
        result_label.configure(text=result_text)

        # Get file metadata
        try:
            file_format, file_size, audio_length, bitrate, additional_metadata = get_file_metadata(temp_file_path)

            log_message = (
                f"File Path: {file_path}\n"
                f"Format: {file_format}\n"
                f"Size (MB): {file_size:.2f}\n"
                f"Audio Length (s): {audio_length:.2f}\n"
                f"Bitrate (Mbps): {bitrate:.2f}\n"
                f"Final Result: {result_text}\n"
                f"Combined Confidence: {combined_confidence:.2f}\n"
            )

            typewriter_effect(log_textbox, log_message)

        except Exception as e:
            log_textbox.insert("end", f"Error getting file metadata: {e}\n")

        # Save to database - Fixed to match backend signature
        try:
            model_used = selected if selected != "All" else "Random Forest, Melody and OpenAI"

            # Convert combined_result to label for database
            if isinstance(combined_result, bool):
                combined_label = "Fake" if combined_result else "Real"
            elif isinstance(combined_result, str):
                combined_label = combined_result
            else:
                combined_label = "Unknown"

            # Create results dictionary for database
            results_dict = {
                "Random Forest": {"label": rf_is_fake, "confidence": rf_confidence},
                "Melody": {"label": hf_is_fake, "confidence": hf_confidence},
                "OpenAI": {"label": hf2_is_fake, "confidence": hf2_confidence}
            }

            already_seen = save_metadata(
                file_uuid=file_uuid,
                file_path=temp_file_path,
                model_used=model_used,
                results=results_dict,
                combined_label=combined_label,
                combined_confidence=combined_confidence,
                processing_time=time.time() - start_time
            )

            file_status_label.configure(
                text="File already in database" if not already_seen else "New file uploaded"
            )
        except Exception as e:
            log_textbox.insert("end", f"Database save error: {e}\n")

        # Generate visualizations
        try:
            update_progress(0.9, "Generating visualizations...")
            visualize_mfcc(temp_file_path)
            create_mel_spectrogram(temp_file_path)
            visualize_embeddings_tsne(temp_file_path)
        except Exception as e:
            log_textbox.insert("end", f"Visualization error: {e}\n")

        update_progress(1.0, "Analysis completed!")
        eta_label.configure(text="Estimated Time: Completed")

        total_time = time.time() - start_time
        log_textbox.insert("end", f"Total analysis time: {total_time:.2f} seconds\n")

    except Exception as e:
        error_msg = f"Critical error during analysis: {e}"
        logging.error(error_msg)
        messagebox.showerror("Error", error_msg)
    finally:
        # Clean up temporary file
        try:
            if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
        except Exception as e:
            logging.warning(f"Failed to clean up temporary file: {e}")

        # Re-enable button
        predict_button.configure(state="normal")

def select_file():
    """File selection dialog."""
    file_paths = filedialog.askopenfilenames(
        title="Select Audio Files",
        filetypes=[
            ("Audio Files", "*.mp3 *.wav *.ogg *.flac *.aac *.m4a *.mp4 *.mov *.avi *.mkv *.webm"),
            ("All Files", "*.*")
        ]
    )

    if file_paths:
        file_entry.delete(0, ctk.END)
        file_entry.insert(0, file_paths[0])  # Use only the first selected file

def start_analysis():
    """Start analysis in a separate thread."""
    if not file_entry.get().strip():
        messagebox.showerror("Error", "Please select an audio file first.")
        return

    predict_button.configure(state="disabled")
    analysis_thread = threading.Thread(target=run_analysis, daemon=True)
    analysis_thread.start()

def open_donate():
    """Open PayPal donation link in the web browser."""
    donate_url = "https://www.paypal.com/donate/?business=sadiqkassamali@gmail.com&no_recurring=0&item_name=Support+VoiceAuth+Development&currency_code=USD"
    webbrowser.open(donate_url)

def open_email():
    """Open email client."""
    webbrowser.open("mailto:sadiqkassamali@gmail.com")

def on_closing():
    """Handle application closing."""
    try:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception as e:
        logging.warning(f"Cleanup error: {e}")

    app.quit()
    app.destroy()

# Initialize GUI
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("dark-blue")

app = ctk.CTk()
app.title("VoiceAuth - Deepfake Audio and Voice Detector")
app.geometry("900x900")
app.protocol("WM_DELETE_WINDOW", on_closing)

# Load logo image with error handling
try:
    logo_image = ctk.CTkImage(
        Image.open(resource_path("images/bot2.png")),
        size=(128, 128)
    )
except Exception as e:
    logging.warning(f"Could not load logo image: {e}")
    logo_image = None

# Menu bar
menu_bar = Menu(app)
contact_menu = Menu(menu_bar, tearoff=0)
contact_menu.add_command(label="For assistance: sadiqkassamali@gmail.com", command=open_email)
contact_menu.add_separator()
contact_menu.add_command(label="Donate to Support", command=open_donate)
menu_bar.add_cascade(label="Contact", menu=contact_menu)
app.configure(menu=menu_bar)

# Header
if logo_image:
    header_label = ctk.CTkLabel(
        app,
        compound="top",
        justify=ctk.CENTER,
        image=logo_image,
        text="VoiceAuth",
        font=("Arial", 28, "bold"),
    )
else:
    header_label = ctk.CTkLabel(
        app,
        text="VoiceAuth",
        font=("Arial", 28, "bold"),
    )

header_label.pack(pady=10)

sub_header_label = ctk.CTkLabel(
    app, text="Deepfake Audio and Voice Detector", font=("Arial", 14, "bold")
)
sub_header_label.pack(pady=5)

# File selection
file_entry = ctk.CTkEntry(app, width=400, placeholder_text="Select an audio file...")
file_entry.pack(pady=10)

select_button = ctk.CTkButton(app, text="Select File", command=select_file)
select_button.pack(pady=5)

# Progress bar
progress_bar = ctk.CTkProgressBar(app, width=400)
progress_bar.pack(pady=10)
progress_bar.set(0)

# Model selection
model_frame = ctk.CTkFrame(app)
model_frame.pack(pady=10)

ctk.CTkLabel(model_frame, text="Select Model:", font=("Arial", 12, "bold")).pack(pady=5)

selected_model = ctk.StringVar(value="All")

models_frame = ctk.CTkFrame(model_frame)
models_frame.pack(pady=5)

model_rf = ctk.CTkRadioButton(models_frame, text="Random Forest", variable=selected_model, value="Random Forest")
model_hf = ctk.CTkRadioButton(models_frame, text="Melody", variable=selected_model, value="Melody")
model_hf2 = ctk.CTkRadioButton(models_frame, text="OpenAI", variable=selected_model, value="OpenAI")
model_all = ctk.CTkRadioButton(models_frame, text="All Models", variable=selected_model, value="All")

model_rf.pack(side="left", padx=10)
model_hf.pack(side="left", padx=10)
model_hf2.pack(side="left", padx=10)
model_all.pack(side="left", padx=10)

# Predict button
predict_button = ctk.CTkButton(
    app, text="Run Analysis", command=start_analysis, fg_color="green", height=40
)
predict_button.pack(pady=20)

# Results section
results_frame = ctk.CTkFrame(app)
results_frame.pack(pady=10, padx=20, fill="x")

file_status_label = ctk.CTkLabel(results_frame, text="", font=("Arial", 12))
file_status_label.pack(pady=5)

result_label = ctk.CTkLabel(results_frame, text="", font=("Arial", 16, "bold"))
result_label.pack(pady=5)

confidence_label = ctk.CTkLabel(results_frame, text="Confidence: ", font=("Arial", 14))
confidence_label.pack(pady=5)

# Log textbox
log_frame = ctk.CTkFrame(app)
log_frame.pack(pady=10, padx=20, fill="both", expand=True)

ctk.CTkLabel(log_frame, text="Analysis Log:", font=("Arial", 12, "bold")).pack(pady=(5, 0))

log_textbox = ScrolledText(
    log_frame,
    height=12,
    bg="#1a1a1a",
    fg="#00ff00",
    insertbackground="#00ff00",
    wrap="word",
    font=("Consolas", 10),
    relief="flat",
)
log_textbox.pack(padx=10, pady=5, fill="both", expand=True)

# ETA label
eta_label = ctk.CTkLabel(app, text="Estimated Time: ", font=("Arial", 12))
eta_label.pack(pady=5)

if __name__ == "__main__":
    freeze_support()
    app.mainloop()
