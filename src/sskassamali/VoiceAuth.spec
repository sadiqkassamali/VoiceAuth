# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE
from PyInstaller.building.splash import Splash  # Ensure this is supported
import pyi_splash  # Ensure this is installed

sys.setrecursionlimit(3000)

# Define paths
main_script = "src/sskassamali/VoiceAuth.py"
backend_script = "src/sskassamali/VoiceAuthBackend.py"
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
    ("src/sskassamali/ffmpeg/ffmpeg.exe", "ffmpeg"),
    ("src/sskassamali/ffmpeg/ffplay.exe", "ffmpeg"),
    ("src/sskassamali/ffmpeg/ffprobe.exe", "ffmpeg"),
]

additional_data = [
    ("src/sskassamali/DB/metadata.db", "DB"),
    ("src/sskassamali/images/bot2.png", "images"),
    ("src/sskassamali/images/splash.jpg", "images"),  # Splash screen
]

hidden_imports = (
    collect_submodules("matplotlib")
    + collect_submodules("librosa")
    + collect_submodules("transformers")
    + collect_submodules("torch")
    + collect_submodules("sklearn")
    + collect_submodules("customtkinter")
    + collect_submodules("tensorflow_hub")
    + collect_submodules("numpy")
    + collect_submodules("py_splash")
    + collect_submodules("joblib")
    + collect_submodules("mutagen")
    + collect_submodules("tensorflow_intel")
    + collect_submodules("sympy")
    + collect_submodules("keras")
    + collect_submodules("tf_keras")
)

# Splash screen configuration
splash = Splash(
    "src/sskassamali/images/splash.jpg",
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    text_pos=(10, 50),
    text_size=14,
    text_color="white",
    minify_script=True,
    always_on_top=False,
)

# Analysis for the main frontend
a = Analysis(
    [main_script],
    pathex=["."],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=["pyi_splash"],
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
    runtime_hooks=["pyi_splash"],
    excludes=[],
    noarchive=False,
)

# Bundle into a single .pyz archive
pyz = PYZ(a.pure + b.pure)

# Create executable with splash
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
    icon="src/sskassamali/images/voiceauth.webp",
)
