# IM Desktop Viewer (Windows)

Offline desktop app (PySide6) to parse `.im` (XML) RF files and plot f0 metrics vs Pout:
- Gt(dB), AM/PM offset (deg), Drain Efficiency (%), Input Return Loss (dB)
- Optional **S1/S3 overlay** (auto-detect or manual select).

## Quick start (from source)
```bash
pip install -r requirements_desktop.txt
python im_desktop_app.py
```

## One-file EXE build (local)
```bat
py -m venv venv
call venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements_desktop.txt pyinstaller
pyinstaller --noconfirm --clean --onefile --windowed ^
  --name "IMDesktop" ^
  --collect-all PySide6 ^
  --collect-all matplotlib ^
  im_desktop_app.py
```
Result: `dist\IMDesktop.exe` (ship **just this file** to your team).

## GitHub Actions (automatic EXE build)
- Push this repo to GitHub.
- On every push to `main`, CI builds a Windows EXE and uploads it as an artifact.
- If you push a tag like `v1.0.0`, CI builds and **attaches the EXE to a GitHub Release**.

See `.github/workflows/build-windows.yml`.

## Notes
- App works fully offline; no internet required at runtime.
- If a tester sees a Visual C++ runtime error, install the **Microsoft VC++ 2015–2022 x64 Redistributable** once.
- The EXE self-extracts to `%TEMP%`; users need write access there.
