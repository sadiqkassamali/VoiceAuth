from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(3000)  # Increase recursion limit if needed

# Base directory
BASE_DIR = os.path.abspath("src/sskassamali")

# Define main scripts
main_script = os.path.join(BASE_DIR, "VoiceAuth.py")
backend_script = os.path.join(BASE_DIR, "VoiceAuthBackend.py")
exe_name = "VoiceAuth"

# Define dependencies and data files
include_files = [
    (os.path.join(BASE_DIR, "DB", "metadata.db"), "DB/metadata.db"),
    (os.path.join(BASE_DIR, "images", "bot2.png"), "images/bot2.png"),
    (os.path.join(BASE_DIR, "images", "splash.jpg"), "images/splash.jpg"),
    (os.path.join(BASE_DIR, "ffmpeg", "ffmpeg.exe"), "ffmpeg/ffmpeg.exe"),
    (os.path.join(BASE_DIR, "ffmpeg", "ffplay.exe"), "ffmpeg/ffplay.exe"),
    (os.path.join(BASE_DIR, "ffmpeg", "ffprobe.exe"), "ffmpeg/ffprobe.exe"),
]

# Validate file paths exist
for src, dest in include_files:
    if not os.path.exists(src):
        raise FileNotFoundError(f"Required file '{src}' not found. Check path!")

# Define packages required
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
        icon=os.path.join(BASE_DIR, "images", "voiceauth.webp"),
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
