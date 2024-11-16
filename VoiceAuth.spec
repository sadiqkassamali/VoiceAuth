# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['VoiceAuth.py'],
    pathex=['.'],  # Add current directory to the path
    binaries=[],
    datas=[
        ('dataset/deepfakevoice.joblib', 'dataset'),
        ('DB/metadata.db', 'DB'),
        ('images/bot2.png', 'images'),
        ('images/splash.jpg', 'images'),
        ('images/voiceauth.webp', 'images'),
        ("ffmpeg/ffmpeg.exe", "./ffmpeg"),
        ("ffmpeg/ffplay.exe", "./ffmpeg"),
        ("ffmpeg/ffprobe.exe", "./ffmpeg")

    ],
    hiddenimports=[
        "matplotlib",
        "ctypes",
        "librosa",
        "transformers",
        "joblib",
        "sklearn",
        "sklearn.ensemble",
        "sklearn.ensemble._forest",
        "sklearn.tree",
        "sklearn.neighbors",
        "sklearn.preprocessing",
        "sklearn.utils",
        "scipy.sparse"
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

# Splash screen configuration (if needed)
splash = Splash(
    'images/splash.jpg',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(10, 50),
    text_size=14,
    text_color="red",
    minify_script=True,
    always_on_top=False,
)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    splash,
    splash.binaries,
    [],
    name='VoiceAuth',
    debug=True,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='images/voiceauth.webp',
)
