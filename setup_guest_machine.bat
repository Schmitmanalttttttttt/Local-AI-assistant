@echo off
title Jarvis AI - Guest Machine Setup
chcp 65001 >nul 2>&1

:: ============================================================
::  Auto-escalate to Administrator
::  If not already admin, relaunch this script elevated via UAC.
:: ============================================================
net session >nul 2>&1
if %errorlevel% equ 0 goto :IS_ADMIN

echo.
echo  Requesting administrator privileges...
echo  Please click Yes in the UAC prompt that appears.
echo.
powershell -NoProfile -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
exit /b

:IS_ADMIN
:: ============================================================
::  Running as Administrator — continue with setup
:: ============================================================
echo.
echo ============================================================
echo   Jarvis AI  -  Guest / Worker Machine Setup
echo ============================================================
echo.

:: ── Check Python ─────────────────────────────────────────────
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  Python not found on this machine.
    echo  Downloading Python 3.12 installer...
    echo.
    powershell -NoProfile -Command ^
        "$url='https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe';" ^
        "$out='%TEMP%\python_setup.exe';" ^
        "Write-Host '  Downloading...';" ^
        "Invoke-WebRequest -Uri $url -OutFile $out -UseBasicParsing;" ^
        "Write-Host '  Installing Python (this may take a minute)...';" ^
        "Start-Process $out -ArgumentList '/quiet InstallAllUsers=1 PrependPath=1 Include_pip=1' -Wait;" ^
        "Remove-Item $out -Force -ErrorAction SilentlyContinue;" ^
        "Write-Host '  Python installed.'"
    if errorlevel 1 (
        echo.
        echo  ERROR: Python installation failed.
        echo  Please install Python 3.10+ from https://python.org then re-run this script.
        pause
        exit /b 1
    )
    :: Reload PATH so python is available in this session
    for /f "tokens=*" %%i in ('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\"Path\",\"Machine\")"') do set "PATH=%%i;%PATH%"
    python --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo  Python was installed but is not on PATH yet.
        echo  Please CLOSE this window and re-run setup_guest_machine.bat.
        pause
        exit /b 1
    )
)

:: ── Run the Python setup script ───────────────────────────────
cd /d "%~dp0"
python setup_guest_machine.py

if errorlevel 1 (
    echo.
    echo  Setup script exited with an error.
    echo  Check the messages above for details.
    pause
    exit /b 1
)

exit /b 0
