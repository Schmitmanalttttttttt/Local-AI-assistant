@echo off
echo ============================================
echo   AI File Assistant - Setup ^& Launcher
echo ============================================
echo.

REM Check Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

echo [1/3] Installing dependencies...
pip install ^
    faster-whisper ^
    numpy ^
    requests ^
    keyboard ^
    psutil ^
    pystray ^
    Pillow ^
    kokoro ^
    sounddevice ^
    pyaudio ^
    pycaw ^
    comtypes ^
    opencv-python ^
    deepface ^
    tf-keras ^
    ultralytics ^
    --quiet

if errorlevel 1 (
    echo ERROR: Failed to install one or more dependencies.
    pause
    exit /b 1
)

echo [2/3] Dependencies installed.
echo [3/3] Launching AI Assistant...
echo.
echo TIP: The app will appear in your system tray.
echo      Press Ctrl+Shift+Space to toggle the popup.
echo      Close this window to stop the assistant.
echo.

cd /d "%~dp0"

python assistant.py

pause
