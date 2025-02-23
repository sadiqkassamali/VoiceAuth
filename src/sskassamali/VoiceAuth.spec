# -*- mode: python ; coding: utf-8 -*-
import sys
import os
from PyInstaller.utils.hooks import collect_submodules, collect_data_files
from PyInstaller.building.build_main import Analysis, PYZ, EXE
from PyInstaller.building.splash import Splash  # Ensure PyInstaller has splash support
import py_splash  # Ensure this is installed

sys.setrecursionlimit(3000)

# ✅ Function to validate file existence
def validate_file(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found - {path}")
        return False
    return True

# ✅ Base directory resolution
BASE_DIR = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(BASE_DIR, "src", "sskassamali")

# ✅ Define script paths
main_script = os.path.join(SRC_DIR, "VoiceAuth.py")
backend_script = os.path.join(SRC_DIR, "VoiceAuthBackend.py")
exe_name = "VoiceAuth"

# ✅ Ensure Splash Image Exists
splash_image = os.path.join(SRC_DIR, "images", "splash.jpg")
if not validate_file(splash_image):
    raise FileNotFoundError(f"❌ ERROR: Splash image not found - {splash_image}")

# ✅ Collect necessary data files
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

additional_data = [
    (os.path.join(SRC_DIR, "DB", "metadata.db"), "DB"),
    (os.path.join(SRC_DIR, "images", "bot2.png"), "images"),
    (splash_image, "images"),  # ✅ Ensures splash is included
]

# ✅ Filter out missing files to avoid build errors
additional_data = [(src, dest) for src, dest in additional_data if validate_file(src)]

# ✅ Ensure all collected data is included
datas = tensorflow_data + torch_data + matplotlib_data + transformers_data + librosa_data + moviepy_data + additional_data

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

# ✅ Frontend executable analysis
a = Analysis(
    [main_script],
    pathex=[BASE_DIR],
    binaries=binaries,
    datas=datas,
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
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# ✅ Splash Screen Configuration (Only for Frontend)
splash = Splash(
    image=splash_image,
    text_pos=(10, 50),
    text_size=14,
    text_color="white",
    minify_script=True,
    always_on_top=True,
)

# ✅ Bundle into a single .pyz archive
pyz_a = PYZ(a.pure)
pyz_b = PYZ(b.pure)

# ✅ Create final executables
exe_frontend = EXE(
    pyz_a,
    a.scripts,
    a.binaries,
    a.datas,
    name="VoiceAuth",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # GUI mode
    icon=os.path.join(SRC_DIR, "images", "voiceauth.webp"),
    splash=splash,  # ✅ Apply splash only to frontend
)

exe_backend = EXE(
    pyz_b,
    b.scripts,
    b.binaries,
    b.datas,
    name="VoiceAuthBackend",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # ✅ Backend runs in console mode
    icon=os.path.join(SRC_DIR, "images", "voiceauth_backend.webp"),
)