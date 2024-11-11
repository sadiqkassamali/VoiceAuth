import sys
from cx_Freeze import setup, Executable
import os
sys.setrecursionlimit(10000)
# Define base depending on OS
base = None
if sys.platform == "win32":
    base = "Win32GUI"

# Define the necessary files to include (datasets, images, etc.)
include_files = [('dataset/deepfakevoice.joblib', 'dataset'),
                 ('dataset', 'dataset'),
                 ('DB/metadata.db', 'DB'),
                 ('images', 'images'),
                 ('ffmpeg/*', 'ffmpeg'),
                 ('FAQ.txt', 'FAQ.txt'),
                 ('License.txt', 'License.txt'),
                 ('UserGuide.txt', 'UserGuide.txt'),

                 os.path.join('dataset', 'deepfakevoice.joblib'),  # Model file
                 os.path.join('images', 'voiceauth.webp'),  # WebP icon
                 'DB/metadata.db',  # Database file (if needed)
                 ]

# Define the executables
executables = [
    Executable(
        'VoiceAuth.py',  # Replace with the name of your main script
        base=base,
        icon=os.path.join('images', 'voiceauth.webp'),  # Set WebP icon
        # Output executable name
    )
]

# Setup function to build the application
setup(
    name="Voice Auth",
    version="1.0",
    description="Deepfake Audio and Voice Detection Application",
    options={
        'build_exe': {
            'include_files': include_files,  # Include all necessary files
            'packages': ['transformers', 'pydub', 'numpy', 'librosa', 'matplotlib', 'joblib', 'customtkinter',
                         'sqlite3', 'shutil', 'threading', 'datetime', 'logging', 'uuid', 'tempfile', 'time', 'os'],
            # Include necessary packages
            'excludes': ['tkinter'],  # Exclude unnecessary modules
            'optimize': 2,  # Level of optimization (e.g., bytecode optimization)
        }
    },
    executables=executables
)
