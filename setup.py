from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(3000)  # Increase recursion limit if needed

# Define main scripts
main_script = "VoiceAuth.py"
backend_script = "VoiceAuthBackend.py"
exe_name = "VoiceAuth"

# Define dependencies and data files
include_files = [
    ("DB/metadata.db", "DB/metadata.db"),
    ("images/bot2.png", "images/bot2.png"),
    ("images/splash.jpg", "images/splash.jpg"),
    ("ffmpeg/ffmpeg.exe", "ffmpeg/ffmpeg.exe"),
    ("ffmpeg/ffplay.exe", "ffmpeg/ffplay.exe"),
    ("ffmpeg/ffprobe.exe", "ffmpeg/ffprobe.exe"),
]

# Define packages required
packages = [
    "tensorflow", "torch", "matplotlib", "transformers", "librosa", "moviepy", "sklearn",
    "customtkinter", "tensorflow_hub", "numpy", "py_splash", "joblib", "mutagen",
    "sympy", "keras", "tf_keras", "kivy", "kivymd", "plyer", "concurrent",
]

# Define build options
build_options = {
    "packages": packages,
    "include_files": include_files,
    "excludes": ["tkinter"],  # Exclude unused libraries
}

# Define executables
executables = [
    Executable(main_script, target_name=exe_name, icon="images/voiceauth.webp"),
    Executable(backend_script, target_name="VoiceAuthBackend"),
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
