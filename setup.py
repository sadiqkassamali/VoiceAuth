from cx_Freeze import setup, Executable
import os
import sys
from setuptools import find_packages

sys.setrecursionlimit(3000)  # Prevent RecursionError during cx_Freeze packaging

# Base directories
BASE_DIR = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(BASE_DIR, "src", "voiceAuth")

# Validate if file exists
def safe_include(src, dst):
    if os.path.exists(src):
        return (src, dst)
    else:
        print(f"⚠️ File not found, skipping: {src}")
        return None

# Main script
main_script = os.path.join(SRC_DIR, "VoiceAuth.py")

# Executable metadata
exe_name = "VoiceAuth"

# Data files to bundle with executable
include_files = list(filter(None, [
    safe_include(os.path.join(BASE_DIR, "src", "voiceAuth", "images", "bot2.png"),     os.path.join("images", "bot2.png")),
    safe_include(os.path.join(BASE_DIR, "src", "voiceAuth", "images", "splash.jpg"),   os.path.join("images", "splash.jpg")),
    safe_include(os.path.join(BASE_DIR, "src", "voiceAuth", "images", "img.png"),      os.path.join("images", "img.png")),
    safe_include(os.path.join(BASE_DIR, "src", "voiceAuth", "images", "voiceauth.ico"),os.path.join("images", "voiceauth.ico")),
]))

# Required packages
packages = [
    "librosa", "moviepy", "customtkinter", "numpy", "py_splash", "mutagen", "joblib",
    "matplotlib", "torch", "pandas", "keras", "tf_keras",
    "scipy", "torchvision", "voiceauthCore", "tokenizers", "tensorflow-intel"
]

# Build options
build_exe_options = {
    "include_msvcr": True,
    "include_files": include_files,
    "packages": packages,
    "optimize": 2
}

# MSI metadata
bdist_msi_options = {
    "upgrade_code": "{12345678-1234-5678-1234-567812345678}",
    "add_to_path": False,
    "install_icon": os.path.join(SRC_DIR, "images", "voiceauth.ico"),
    "data": {
        "Shortcut": [
            ("DesktopShortcut", "DesktopFolder", exe_name,
             "TARGETDIR", "[TARGETDIR]VoiceAuth.exe", None, None, None, None, None, None, "TARGETDIR")
        ]
    }
}

# Define GUI executable
executables = [
    Executable(
        script=main_script,

        icon=os.path.join(SRC_DIR, "images", "voiceauth.ico")
    )
]

setup(
    name=exe_name,
    version="1.2.11",
    description="Voice Authentication Application",
    packages=find_packages(where="."),
    package_dir={"": "."},
    include_package_data=True,
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options
    },
    executables=executables
)
