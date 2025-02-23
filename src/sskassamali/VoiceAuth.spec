# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE
from PyInstaller.building.splash import Splash  # Ensure this is supported
import py_splash  # Ensure this is installed

sys.setrecursionlimit(3000)

# Base project path
BASE_DIR = os.path.abspath("src/sskassamali")

# Define paths
main_script = os.path.join(BASE_DIR, "VoiceAuth.py")
backend_script = os.path.join(BASE_DIR, "VoiceAuthBackend.py")
exe_name = "VoiceAuth"

# Splash Image Fix: Ensure correct path
splash_image = os.path.join(BASE_DIR, "images", "splash.jpg")

# Verify the splash image exists
if not os.path.exists(splash_image):
    raise ValueError(f"Image file '{splash_image}' not found. Check path!")

# Collect submodules & data files from ML dependencies
tensorflow_data = collect_data_files("tensorflow")
torch_data = collect_data_files("torch")
matplotlib_data = collect_data_files("matplotlib")
transformers_data = collect_data_files("transformers")
librosa_data = collect_data_files("librosa")
moviepy_data = collect_data_files("moviepy")

inaries = [
    ("ffmpeg/ffmpeg.exe", "ffmpeg/ffmpeg.exe", "BINARY"),
    ("ffmpeg/ffplay.exe", "ffmpeg/ffplay.exe", "BINARY"),
    ("ffmpeg/ffprobe.exe", "ffmpeg/ffprobe.exe", "BINARY"),
]


additional_data = [
    (os.path.join(BASE_DIR, "DB", "metadata.db"), "DB"),
    (os.path.join(BASE_DIR, "images", "bot2.png"), "images"),
    (splash_image, "images"),  # Fix splash path
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
	+ collect_submodules("tkinter")
)

# ✅ Fix: Proper splash screen inclusion
splash = Splash(
    "images/splash.jpg",
    text_pos=(10, 50),
    text_size=14,
    text_color="white",
    minify_script=True,
    always_on_top=False,
)

# ✅ Fix: Main executable analysis
a = Analysis(
    [main_script],
    pathex=["."],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=["py_splash"],
    excludes=[],
    noarchive=False,
)

# ✅ Fix: Backend executable analysis
b = Analysis(
    [backend_script],
    pathex=["."],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=["py_splash"],
    excludes=[],
    noarchive=False,
)

# Bundle into a single .pyz archive
pyz = PYZ(a.pure + b.pure)

# ✅ Fix: Create final executable with splash
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
    icon=os.path.join(BASE_DIR, "images", "voiceauth.webp"),
)
