import logging
import os
import shutil
import sys
import traceback
import webbrowser
from tkinter import Menu
from tkinter.scrolledtext import ScrolledText
import customtkinter as ctk
import matplotlib
from PIL import Image
from VoiceAuthBackend import select_file, start_analysis

matplotlib.use("tkAgg")

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
    if __name__ == "__main__":
        app.mainloop()
except BaseException:
    f = open("app.log", "w")
    e = traceback.format_exc()
    f.write(str(e))
    f.close()
