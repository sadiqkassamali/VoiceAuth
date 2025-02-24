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

# Define required packages
packages = [
    "tensorflow", "torch", "matplotlib", "transformers", "librosa", "moviepy", "sklearn",
    "customtkinter", "tensorflow_hub", "numpy", "py_splash", "joblib", "mutagen",
    "sympy", "keras", "tf_keras", "kivy", "kivymd", "plyer", "concurrent", "tkinter"
]

# MSI Options
msi_data = {
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "VoiceAuth",
         "TARGETDIR", "[TARGETDIR]VoiceAuth.exe", None, None, None, None, None, None, "TARGETDIR"),
    ]
}

bdist_msi_options = {
    "upgrade_code": "{12345678-1234-5678-1234-567812345678}",  # Change this for new versions
    "add_to_path": False,
    "install_icon": os.path.join(SRC_DIR, "images", "voiceauth.ico"),  # Set an icon for the installer
    "data": msi_data,
}

# Build options
build_options = {
    "include_files": include_files,
    "packages": packages,
}

base = None
if sys.platform == "win32":
    base = "Win32GUI"

# Define executables
executables = [
    Executable(
        main_script,
        target_name="VoiceAuth.exe",
        base=base,  # Ensures it's a windowed app (no console)
        icon=os.path.join(SRC_DIR, "images", "voiceauth.ico"),
    ),
    Executable(
        backend_script,
        target_name="VoiceAuthBackend.exe",
        base=None,
        icon=os.path.join(SRC_DIR, "images", "voiceauth_backend.ico"),
    ),
]

# Setup configuration
setup(
    name=exe_name,
    version="1.0",
    description="Voice Authentication Application",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={"sskassamali": ["DB/*.db", "images/*.png", "images/*.jpg"]},  # Include all necessary files
    include_package_data=True,
    options={
        "build_exe": build_options,
        "bdist_msi": bdist_msi_options,  # Add MSI build options
    },
    executables=executables,
)
