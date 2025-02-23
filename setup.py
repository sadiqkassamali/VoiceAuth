from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(3000)  # Increase recursion limit if needed

# Base directory resolution
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src", "sskassamali")

# Function to validate file paths
def validate_file(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found - {path}")  # Logging instead of raising an error
        return False
    return True

# Define main scripts
main_script = os.path.join(SRC_DIR, "VoiceAuth.py")
backend_script = os.path.join(SRC_DIR, "VoiceAuthBackend.py")
exe_name = "VoiceAuth"

# Define dependencies and data files
include_files = [
    (os.path.join(SRC_DIR, "DB", "metadata.db"), "DB/metadata.db"),
    (os.path.join(SRC_DIR, "images", "bot2.png"), "images/bot2.png"),
    (os.path.join(SRC_DIR, "images", "splash.jpg"), "images/splash.jpg"),
    (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), "ffmpeg/ffmpeg.exe"),
    (os.path.join(SRC_DIR, "ffmpeg", "ffplay.exe"), "ffmpeg/ffplay.exe"),
    (os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe"), "ffmpeg/ffprobe.exe"),
]

# Filter out missing files to avoid build failures
include_files = [(src, dest) for src, dest in include_files if validate_file(src)]

# Define required packages
packages = [
    "tensorflow", "torch", "matplotlib", "transformers", "librosa", "moviepy", "sklearn",
    "customtkinter", "tensorflow_hub", "numpy", "py_splash", "joblib", "mutagen",
    "sympy", "keras", "tf_keras", "kivy", "kivymd", "plyer", "concurrent", "tkinter"
]

# Define build options
build_options = {
    "packages": packages,
    "include_files": include_files,
    "excludes": [],  # Exclude unused libraries
}

# Define executables
executables = [
    Executable(
        main_script,
        target_name=exe_name,
        icon=os.path.join(SRC_DIR, "images", "voiceauth.webp"),
    ),
    Executable(
        backend_script,
        target_name="VoiceAuthBackend",
    ),
]

# Setup configuration
setup(
    name=exe_name,
    version="1.0",
    description="Voice Authentication Application",
    packages=find_packages(where="src"),  # Finds packages inside 'src'
    package_dir={"": "src"},  # Maps package root to 'src'
    options={"build_exe": build_options},
    executables=executables,
)
