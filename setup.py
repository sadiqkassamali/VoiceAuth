from cx_Freeze import setup, Executable
from setuptools import find_packages
import sys
import os

sys.setrecursionlimit(5000)  # Increase recursion limit if needed

# Base directory resolution
BASE_DIR = os.path.abspath(os.getcwd())
SRC_DIR = os.path.join(BASE_DIR, "src", "voiceAuth")

# Function to validate file paths
def validate_file(path):
    if not os.path.exists(path):
        print(f"⚠️ Warning: File not found - {path}")  # Logging instead of raising an error
        return False
    return True
exclude_files = { "service_2.json.gz.*", "paginators_1.json.*" ,"service_2.json.gz.*", "endpoint_rule_set_1.json.gz.*"}
# Define main scripts
main_script = os.path.join(SRC_DIR, "VoiceAuth.py")
exe_name = "VoiceAuth"


# Define dependencies and data files
include_files = [
    (src, dst) for src, dst in [
        (os.path.join(SRC_DIR, "images", "bot2.png"), os.path.join("images", "bot2.png")),
        (os.path.join(SRC_DIR, "images", "splash.jpg"), os.path.join("images", "splash.jpg")),
        (os.path.join(SRC_DIR, "images", "img.jpg"), os.path.join("images", "img.jpg")),
        (os.path.join(SRC_DIR, "images", "voiceauth.ico"), os.path.join("images", "voiceauth.ico")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffmpeg.exe"), os.path.join("ffmpeg", "ffmpeg.exe")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffplay.exe"), os.path.join("ffmpeg", "ffplay.exe")),
        (os.path.join(SRC_DIR, "ffmpeg", "ffprobe.exe"), os.path.join("ffmpeg", "ffprobe.exe")),
    ] if os.path.exists(src)
]

# Define required packages
packages = [
     "librosa", "moviepy",
    "customtkinter",  "numpy", "py_splash", "mutagen", "joblib", "scipy",
    "kivy", "kivymd", "plyer", "numpy","sklearn", "matplotlib", "torch",
]

# Define MSI data
msi_data = {
    "Shortcut": [
        ("DesktopShortcut", "DesktopFolder", "VoiceAuth",
         "TARGETDIR", "[TARGETDIR]VoiceAuth.exe", None, None, None, None, None, None, "TARGETDIR"),
    ]
}

# Build options
build_exe_options = {
    "include_msvcr": True,  # Include C++ runtime
    "include_files": include_files,
    "packages": packages,
    "optimize": 2,  # Optimize bytecode to reduce size
    "excludes": [
        "service_2.json.gz.*", "paginators_1.json.*" ,"service_2.json.gz.*", "endpoint_rule_set_1.json.gz.*"]
}

# MSI options
bdist_msi_options = {
    "upgrade_code": "{12345678-1234-5678-1234-567812345678}",  # Update for each version
    "add_to_path": False,
    "install_icon": os.path.join(SRC_DIR, "images", "voiceauth.ico"),  # Set an icon for the installer
    "data": msi_data,
}

base = "Win32GUI"

executables = [
    Executable(
        main_script,
        target_name="VoiceAuth.exe",
        base=base,
        icon=os.path.join(SRC_DIR, "images", "voiceauth.ico"),
    )
]

executables = [
    Executable(os.path.join(SRC_DIR, "VoiceAuth.py"), base=base)
]
setup(
    name=exe_name,
    version="1.0",
    description="Voice Authentication Application",
    packages=find_packages('.'), 
    package_dir={"": "."},
    include_package_data=True,
    options={
        "build_exe": build_exe_options,
        "bdist_msi": bdist_msi_options,  # Add MSI build options
    },
    executables=executables,
)