from cx_Freeze import setup, Executable
import sys
import os
sys.setrecursionlimit(5000)
# Ensure cx_Freeze uses "all users" directory for installation
base = None
if sys.platform == "win32":
    base = "Win32GUI"

# Files and dependencies for the app
# List out additional files or directories if needed, such as your model
# files, icons, etc.
files = [
    "DB",
    "dataset",
    "images",
    "ffmpeg"
]


# cx_Freeze setup
setup(
    name="Voice Auth",
    version="1.0",
    description="Deepfake Audio and Voice Detector",
    author="Sadiq Kassamali | sadiqkssamali@gmail.com",
    options={
        "build_exe": {
            "packages": ["os", "numpy", "librosa", "joblib", "customtkinter", "transformers"],
            "include_files": files
        },
        "bdist_msi": {
            "all_users": True,  # Ensures installer applies to all users
            "add_to_path": False,

        }
    },
    # Target executable configuration
    executables=[
        Executable(
            script="VoiceAuth.py",
            base=base,
            icon="images/voiceauth.webp",

        )
    ]
)
