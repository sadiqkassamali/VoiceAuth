# invoke using:
#  python setup.py build

from cx_Freeze import setup, Executable

import sys
import glob
import os
import zlib
import shutil

# Remove the existing folders folder
shutil.rmtree("build", ignore_errors=True)
shutil.rmtree("dist", ignore_errors=True)
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
            "include_files": files,
            "optimize": 2,
            "include_msvcr": True,
            "excludes": ['FixTk', 'tcl', 'tk',
                         '_tkinter', 'tkinter',
                         'Tkinter', "PIL", "PyQt4",
                         "PyQt5", "pytest" 'matplotlib'],
        },
        "bdist_msi": {
            "all_users": True,  # Ensures installer applies to all users
            "add_to_path": False,

        }
    },

    executables=[
        Executable(
            "VoiceAuth.py",
            copyright="Copyright (C) 2024 VoiceAuth",
            base=base,
            icon="images/voiceauth.webp",
        )
    ]
)
