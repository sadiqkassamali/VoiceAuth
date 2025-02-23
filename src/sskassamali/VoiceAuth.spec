# -*- mode: python ; coding: utf-8 -*-
import sys
import os
import shutil
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
sys.setrecursionlimit(3000) 
# Define paths
main_script = "VoiceAuth.py"
backend_script = "VoiceAuthBackend.py"
exe_name = "VoiceAuth"

# Set the temporary directory
temp_dir = os.path.join(os.getenv("TEMP", "/tmp"), "VoiceAuth")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

# Collect submodules & data files from ML dependencies
tensorflow_data = collect_data_files("tensorflow")
torch_data = collect_data_files("torch")
matplotlib_data = collect_data_files("matplotlib")
transformers_data = collect_data_files("transformers")
librosa_data = collect_data_files("librosa")
moviepy_data = collect_data_files("moviepy")

binaries = [
    ("ffmpeg/ffmpeg.exe", "ffmpeg"),
    ("ffmpeg/ffplay.exe", "ffmpeg"),
    ("ffmpeg/ffprobe.exe", "ffmpeg"),
]

additional_data = [
    ("DB/metadata.db", "DB"),
    ("images/bot2.png", "images"),
    ("images/splash.jpg", "images"),  # Splash screen
]

hidden_imports = (
    collect_submodules("matplotlib")
    + collect_submodules("librosa")
    + collect_submodules("transformers")
    + collect_submodules("torch")
    + collect_submodules("sklearn")
    + collect_submodules("tkinter")
    + collect_submodules("customtkinter")
	+ collect_submodules("tensorflow_hub") 
	+ collect_submodules("numpy")
	+ collect_submodules("py_splash")
	+ collect_submodules("tkinter")
	+ collect_submodules("joblib")
	+ collect_submodules("mutagen")
	+ collect_submodules("tensorflow_intel")
	+ collect_submodules("sympy")
	+ collect_submodules("keras")
	+ collect_submodules("tf_keras")
)

# Analysis for the main frontend
a = Analysis(
    [main_script],
    pathex=["."],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Analysis for the backend
b = Analysis(
    [backend_script],
    pathex=["."],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# Bundle into a single .pyz archive
pyz = PYZ(a.pure + b.pure)

# Splash screen (Optional)
splash = Splash(
    "images/splash.jpg",
    binaries=a.binaries + b.binaries,
    datas=a.datas + b.datas,
    text_pos=(10, 50),
    text_size=14,
    text_color="white",
    minify_script=True,
    always_on_top=False,
)

# Create executable
exe = EXE(
    pyz,
    a.scripts + b.scripts,
    a.binaries + b.binaries,
    a.datas + b.datas,
    splash,
    splash.binaries,
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set False for GUI mode
    icon="images/voiceauth.webp",
)

