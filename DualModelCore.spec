# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['dualmodel', 'dualmodel.api_server', 'dualmodel.config', 'dualmodel.gate', 'dualmodel.interrupter', 'dualmodel.local_model', 'dualmodel.cloud_model', 'dualmodel.pipeline', 'dualmodel.utils', 'uvicorn', 'fastapi', 'openai', 'modelscope', 'pydantic', 'starlette']
hiddenimports += collect_submodules('dualmodel')


a = Analysis(
    ['run_app.py'],
    pathex=[],
    binaries=[],
    datas=[('dualmodel', 'dualmodel')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='DualModelCore',
    debug=False,
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
)
