# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE
from PyInstaller.building.splash import Splash  # Ensure PyInstaller has splash support
import py_splash  # Ensure this is installed

sys.setrecursionlimit(3000)

# ✅ Base project path
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src", "sskassamali")

# ✅ Define script paths
main_script = os.path.join(SRC_DIR, "VoiceAuth.py")
backend_script = os.path.join(SRC_DIR, "VoiceAuthBackend.py")
exe_name = "VoiceAuth"

# ✅ Ensure Splash Image Exists
splash_image = os.path.join(SRC_DIR, "images", "splash.jpg")
if not os.path.exists(splash_image):
    raise FileNotFoundError(f"❌ ERROR: Splash image not found - {splash_image}")

# ✅ Collect necessary data files from dependencies
tensorflow_data = collect_data_files("tensorflow")
torch_data = collect_data_files("torch")
matplotlib_data = collect_data_files("matplotlib")
transformers_data = collect_data_files("transformers")
librosa_data = collect_data_files("librosa")
moviepy_data = collect_data_files("moviepy")

# ✅ Define binary dependencies (FFmpeg executables)
binaries = [
    (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), "ffmpeg"),
    (os.path.join(SRC_DIR, "ffmpeg", "ffplay.exe"), "ffmpeg"),
    (os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe"), "ffmpeg"),
]

# ✅ Define additional required files (metadata DB, images, splash)
additional_data = [
    (os.path.join(SRC_DIR, "DB", "metadata.db"), "DB"),
    (os.path.join(SRC_DIR, "images", "bot2.png"), "images"),
    (splash_image, "images"),  # ✅ Ensure splash is included
]

# ✅ Collect all hidden imports for dependencies
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

# ✅ Main executable analysis
a = Analysis(
    [main_script],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# ✅ Backend executable analysis
b = Analysis(
    [backend_script],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# ✅ Splash Screen Configuration
splash = Splash(
    image=splash_image,
    binaries=b.binaries,  # ✅ Ensuring splash is applied correctly
    text_pos=(10, 50),
    text_size=14,
    text_color="white",
    minify_script=True,
    always_on_top=True,
)

# ✅ Bundle into a single .pyz archive
pyz = PYZ(a.pure + b.pure)

# ✅ Create final executables
exe = EXE(
    pyz,
    a.scripts + b.scripts,
    a.binaries + b.binaries,
    a.datas + b.datas,
    splash,
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI mode
    icon=os.path.join(SRC_DIR, "images", "voiceauth.webp"),
)
