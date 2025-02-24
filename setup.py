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
    (src, dst) for src, dst in [
        (os.path.join(SRC_DIR, "DB", "metadata.db"), os.path.join("DB", "metadata.db")),
        (os.path.join(SRC_DIR, "images", "bot2.png"), os.path.join("images", "bot2.png")),
        (os.path.join(SRC_DIR, "images", "splash.jpg"), os.path.join("images", "splash.jpg")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), os.path.join("ffmpeg", "ffmpeg.exe")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffplay.exe"), os.path.join("ffmpeg", "ffplay.exe")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe"), os.path.join("ffmpeg", "ffprobe.exe")),
    ] if os.path.exists(src)
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

executables = [
    Executable(
        os.path.join(SRC_DIR, "VoiceAuth.py"),
        target_name="VoiceAuth.exe",
        base="Win32GUI"  # ✅ Hide console for GUI apps
    ),
    Executable(
        os.path.join(SRC_DIR, "VoiceAuthBackend.py"),
        target_name="VoiceAuthBackend.exe",
        base=None  # Keep console for backend
    ),

# Setup configuration
setup(
    name=exe_name,
    version="1.0",
    description="Voice Authentication Application",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"sskassamali": ["DB/*.db", "images/*.png", "images/*.jpg", "*.py"]}  # Ensure Python files are included
    include_package_data=True,
    options={"build_exe": {"include_files": include_files}},
    executables=executables,
)
