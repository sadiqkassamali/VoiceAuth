from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(3000)  # Increase recursion limit if needed

# Base directory resolution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src", "sskassamali")

# Function to validate file paths
def validate_file(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found - {path}")  # Logging instead of raising an error
        return False
    return True

# Define main scripts
main_script = os.path.join(BASE_DIR, "src", "sskassamali", "VoiceAuth.py")
backend_script = os.path.join(BASE_DIR, "src", "sskassamali", "VoiceAuthBackend.py")
exe_name = "VoiceAuth"


# Define dependencies and data files
include_files = []
file_paths = [
    ("DB", "metadata.db"),
    ("images", "bot2.png"),
    ("images", "splash.jpg"),
    ("ffmpeg", "ffmpeg.exe"),
    ("ffmpeg", "ffplay.exe"),
    ("ffmpeg", "ffprobe.exe"),
]

for folder, file in file_paths:
    src_path = os.path.join(SRC_DIR, folder, file)
    dst_path = os.path.join(folder, file)
    if validate_file(src_path):
        include_files.append((src_path, dst_path))

# Define required packages
packages = [
    "tensorflow", "torch", "matplotlib", "transformers", "librosa", "moviepy", "sklearn",
    "customtkinter", "tensorflow_hub", "numpy", "py_splash", "joblib", "mutagen",
    "sympy", "keras", "tf_keras", "kivy", "kivymd", "plyer", "concurrent"
]

# Define MSI data
msi_data = {
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "VoiceAuth",
         "TARGETDIR", "[TARGETDIR]VoiceAuth.exe", None, None, None, None, None, None, "TARGETDIR"),
    ],
    "ProgId": [
        ("Prog.Id", None, None, "Voice Authentication Application", "IconId", None),
    ],
    "Icon": [
        ("IconId", os.path.join(SRC_DIR, "images", "voiceauth.ico")),
    ],
}

# Build options
build_exe_options = {
    "excludes": ["tkinter"],
    "include_msvcr": True,  # Include C++ runtime
    "include_files": include_files,
    "packages": packages,
}

# MSI options
bdist_msi_options = {
    "upgrade_code": "{12345678-1234-5678-1234-567812345678}",  # Update for each version
    "add_to_path": False,
    "install_icon": os.path.join(SRC_DIR, "images", "voiceauth.ico"),  # Set an icon for the installer
    "data": msi_data,
}

# Define base for Windows GUI apps
base = "Win32GUI" if sys.platform == "win32" else None

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
    packages=find_packages(where="src"),  # Ensure correct package structure
    package_dir={"": "src"},
    include_package_data=True,
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options,  # Add MSI build options
    },
    executables=executables,
)
