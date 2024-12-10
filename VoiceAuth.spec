# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['VoiceAuth.py'],
    pathex=[],
    binaries=[],
    datas=[('./*', '.'), ('dataset/deepfakevoice.joblib', 'dataset'), ('DB/metadata.db', 'DB'), ('images/bot2.png', 'images')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)
pyz = PYZ(a.pure)
splash = Splash(
    'images/splash.jpg',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    text_color="red"
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
    [('O', None, 'OPTION')],
    name='VoiceAuth',
    debug=False,
    bootloader_ignore_signals=False,
    strip=True,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['images\\voiceauth.webp'],
)
