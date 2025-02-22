from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(3000)  # Increase recursion limit if needed

# Base directory resolution
BASE_DIR = os.path.abspath(os.getcwd())
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
    (os.path.join(SRC_DIR, "DB", "metadata.db"), os.path.join(SRC_DIR, "DB", "metadata.db")),
    (os.path.join(SRC_DIR, "images", "bot2.png"), os.path.join(SRC_DIR, "images", "bot2.png")),
    (os.path.join(SRC_DIR, "images", "splash.jpg"), os.path.join(SRC_DIR, "images", "splash.jpg")),
    (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe")),
    (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe")),
    (os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe"), os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe")),
]

# Build options
build_options = {
    "include_files": include_files,
}

# Define required packages
packages = [
    "tensorflow", "torch", "matplotlib", "transformers", "librosa", "moviepy", "sklearn",
    "customtkinter", "tensorflow_hub", "numpy", "py_splash", "joblib", "mutagen",
    "sympy", "keras", "tf_keras", "kivy", "kivymd", "plyer", "concurrent", "tkinter"
]

# Build options
build_options = {
    "include_files": include_files,
}

# Define executables
executables = [
    Executable(os.path.join(SRC_DIR, "VoiceAuth.py"), target_name="VoiceAuth"),
    Executable(os.path.join(SRC_DIR, "VoiceAuthBackend.py"), target_name="VoiceAuthBackend"),
]


# Setup configuration
setup(
    name=exe_name,
    version="1.0",
    description="Voice Authentication Application",
    packages=find_packages(where="src"),  # Finds packages inside 'src'
    package_dir={"": "src"},  # Maps package root to 'src'
    package_data={"sskassamali": ["*.py"]},  # Include all Python files
    include_package_data=True,
    options={"build_exe": build_options},
    executables=executables,
)
