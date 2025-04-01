# -*- mode: python ; coding: utf-8 -*-

import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# Collect all data from voiceauthcore or custom model folders
voiceauthcore_data = collect_data_files('voiceauthCore')
customtkinter_data = collect_data_files('customtkinter')
tensorflow_data = collect_data_files('tensorflow')
numpy_data = collect_data_files('numpy')


# Hidden imports
hidden = [
    'PIL._tkinter_finder',
    'cv2',
    'pydub',
    'wave',
    'scipy',
    'numpy', 'keras',
    'sklearn.ensemble._forest',
    'sklearn.tree._tree',
    'tensorflow',
    'tensorflow.lite',
    'customtkinter',
    'matplotlib.backends.backend_tkagg',
    'matplotlib.pyplot',
    'openai',
    'soundfile',
    'pyaudio',
    'pytz',
    'librosa',
    'numba',
    'decorator',
    'pkg_resources.py2_warn',  # sometimes needed for older libraries
]

a = Analysis(
    ['src\\voiceAuth\\VoiceAuth.py'],
    pathex=[],
    binaries=collect_dynamic_libs('numpy')
             + collect_dynamic_libs('tensorflow')
             + collect_dynamic_libs('librosa'),
    datas=voiceauthcore_data + customtkinter_data + tensorflow_data + numpy_data,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=2,
)

pyz = PYZ(a.pure)

splash = Splash(
    'src/voiceAuth/images/splash.jpg',  # your edited white image
    binaries=a.binaries,
    datas=a.datas,
    text_pos=(50, 50),
	text_color='white',		# optional, can show text if text is not embedded in image
    text_size=16,
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
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['src\\voiceAuth\\images\\voiceauth.ico'],
)
