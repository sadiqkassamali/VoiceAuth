from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(3000)  # Increase recursion limit if needed

# Base directory resolution
BASE_DIR = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(BASE_DIR, "src", "voiceAuth")

# Function to validate file paths
def validate_file(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found - {path}")  # Logging instead of raising an error
        return False
    return True

# Define main scripts
main_script = os.path.join(BASE_DIR, "src", "VoiceAuth", "VoiceAuth.py")
exe_name = "voiceAuth"


# Define dependencies and data files
include_files = [
    (src, dst) for src, dst in [
        (os.path.join(SRC_DIR, "images", "bot2.png"), os.path.join("images", "bot2.png")),
        (os.path.join(SRC_DIR, "images", "splash.jpg"), os.path.join("images", "splash.jpg")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), os.path.join("ffmpeg", "ffmpeg.exe")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffplay.exe"), os.path.join("ffmpeg", "ffplay.exe")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe"), os.path.join("ffmpeg", "ffprobe.exe")),
    ] if os.path.exists(src)
]

# Define required packages
packages = [
     "librosa", "moviepy", "voiceauthCore",
    "customtkinter",  "numpy", "py_splash", "mutagen",
    "kivy", "kivymd", "plyer"
]

# Define MSI data
msi_data = {
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "VoiceAuth",
         "TARGETDIR", "[TARGETDIR]VoiceAuth.exe", None, None, None, None, None, None, "TARGETDIR"),
    ],
    "Icon": [
        ("IconId", os.path.join(SRC_DIR, "images", "voiceauth.ico")),
    ],
}

# Build options
build_exe_options = {
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
    )
]

# Setup configuration
setup(
    name=exe_name,
    version="1.0",
    description="Voice Authentication Application",
    packages=find_packages('.'),  # Ensure correct package structure
    package_dir={"": "."},
    include_package_data=True,
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options,  # Add MSI build options
    },
    executables=executables,
)