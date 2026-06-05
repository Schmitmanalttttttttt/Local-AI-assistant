#!/usr/bin/env python3
"""
Jarvis AI – Guest / Worker Machine Setup
=========================================
Run this on the SECONDARY PC that will serve as an additional Ollama worker.
It does NOT install the full Jarvis assistant — just Ollama + the required
models + firewall + optional auto-start, so the host PC can offload inference.

Launched automatically by setup_guest_machine.bat (admin-elevated).
Can also be run directly: python setup_guest_machine.py
"""

import ctypes
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

# ── Color helpers (Windows ANSI) ──────────────────────────────────────────────
os.system("")  # enable ANSI on Windows console

def c(text, code): return f"\033[{code}m{text}\033[0m"
def green(t):  return c(t, "92")
def yellow(t): return c(t, "93")
def red(t):    return c(t, "91")
def cyan(t):   return c(t, "96")
def bold(t):   return c(t, "1")

# ── Admin guard ───────────────────────────────────────────────────────────────
def is_admin():
    try:
        return ctypes.windll.shell32.IsUserAnAdmin()
    except Exception:
        return False

if not is_admin():
    print(yellow("Not running as administrator — re-launching with elevated privileges..."))
    ctypes.windll.shell32.ShellExecuteW(
        None, "runas", sys.executable,
        f'"{os.path.abspath(__file__)}"',
        None, 1
    )
    sys.exit(0)

# ── Utilities ─────────────────────────────────────────────────────────────────
REQUIRED_MODELS = [
    ("llama3.2:3b",       "~2.0 GB",  "fast chat + fallback"),
    ("qwen3:4b",          "~2.6 GB",  "main command model"),
    ("deepseek-r1:8b",    "~4.9 GB",  "reasoning model"),
    ("nomic-embed-text",  "~274 MB",  "memory embeddings"),
]
OPTIONAL_MODELS = [
    ("llama3.1:70b",  "~40 GB",  "ultra deep thinking — needs 64 GB RAM"),
]

def ask(prompt: str, *, default: str = "y") -> str:
    hint = "[Y/n]" if default == "y" else "[y/N]"
    while True:
        val = input(f"  {prompt} {hint}: ").strip().lower()
        if not val:
            return default
        if val in ("y", "n", "yes", "no"):
            return val[0]

def run(cmd: str, **kw) -> int:
    return subprocess.run(cmd, shell=True, **kw).returncode

def run_out(cmd: str) -> str:
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()

def sep(title: str = ""):
    width = 60
    if title:
        pad = (width - len(title) - 2) // 2
        print(f"\n{cyan('─' * pad)} {bold(title)} {cyan('─' * pad)}")
    else:
        print(cyan("─" * width))

def ok(msg):  print(f"  {green('[OK]')}  {msg}")
def warn(msg): print(f"  {yellow('[!!]')}  {msg}")
def info(msg): print(f"  {cyan('-->')}  {msg}")

# ── Banner ────────────────────────────────────────────────────────────────────
print()
print(bold(cyan("╔══════════════════════════════════════════════════════════╗")))
print(bold(cyan("║      Jarvis AI  —  Guest / Worker Machine Setup          ║")))
print(bold(cyan("╚══════════════════════════════════════════════════════════╝")))
print()
print("  This PC will act as an Ollama inference worker.")
print("  The host PC (running Jarvis) will send requests here.")
print()
print(yellow("  Models to install  (~10 GB total, more if optional selected):"))
for name, size, desc in REQUIRED_MODELS:
    print(f"    • {name:<22}  {size:<10}  {desc}")
print()
input("  Press Enter to begin setup, or Ctrl+C to cancel...")
print()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 1 — Python packages (minimal, for this script and monitoring)
# ═════════════════════════════════════════════════════════════════════════════
sep("1 / 6  —  Python packages")
pkgs = ["requests", "psutil"]
info(f"Installing: {', '.join(pkgs)}")
rc = run(f'"{sys.executable}" -m pip install --quiet --upgrade ' + " ".join(pkgs))
if rc == 0:
    ok("Packages ready.")
else:
    warn("pip install had errors — continuing anyway.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 2 — Install / verify Ollama
# ═════════════════════════════════════════════════════════════════════════════
sep("2 / 6  —  Ollama installation")

ollama_path = shutil.which("ollama")
if ollama_path:
    ver = run_out("ollama --version")
    ok(f"Ollama already installed: {ver}")
else:
    info("Ollama not found — downloading installer (~70 MB)...")
    installer = Path(os.environ.get("TEMP", ".")) / "OllamaSetup.exe"
    try:
        url = "https://ollama.com/download/OllamaSetup.exe"
        print(f"  Downloading from {url}")
        urllib.request.urlretrieve(url, installer,
            reporthook=lambda n, bs, ts: print(
                f"\r  {n * bs // (1024*1024):>3}/{ts // (1024*1024):>3} MB", end=""))
        print()
        info("Running Ollama installer (silent)...")
        run(f'"{installer}" /S')
        time.sleep(6)
        installer.unlink(missing_ok=True)
        # Refresh PATH in this process
        new_path = run_out('powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable(\'Path\', \'Machine\')"')
        if new_path:
            os.environ["PATH"] = new_path + ";" + os.environ.get("PATH", "")
        ok("Ollama installed.")
    except Exception as exc:
        warn(f"Download failed: {exc}")
        warn("Please install Ollama manually from https://ollama.com/download")
        input("  Press Enter once Ollama is installed to continue...")

# Ensure Ollama serve is running
sep("    Starting Ollama service")
try:
    import requests as _req
    _req.get("http://localhost:11434", timeout=3)
    ok("Ollama service already running.")
except Exception:
    info("Starting 'ollama serve' in background...")
    subprocess.Popen(["ollama", "serve"],
                     creationflags=subprocess.CREATE_NEW_CONSOLE,
                     close_fds=True)
    for i in range(12):
        time.sleep(1)
        try:
            import requests as _req
            _req.get("http://localhost:11434", timeout=2)
            ok("Ollama service started.")
            break
        except Exception:
            print(f"\r  Waiting... ({i+1}/12)", end="")
    else:
        warn("Ollama may not have started — check manually with: ollama serve")
    print()

# ═════════════════════════════════════════════════════════════════════════════
# STEP 3 — Pull required models
# ═════════════════════════════════════════════════════════════════════════════
sep("3 / 6  —  AI models")

# Get already-installed models
try:
    import requests as _req
    resp = _req.get("http://localhost:11434/api/tags", timeout=5)
    installed = {m["name"] for m in resp.json().get("models", [])}
except Exception:
    installed = set()

for model, size, desc in REQUIRED_MODELS:
    if model in installed:
        ok(f"{model:<22}  already installed")
    else:
        print(f"\n  {cyan('Pulling')} {bold(model)} ({size} — {desc})")
        rc = run(f"ollama pull {model}")
        if rc == 0:
            ok(f"{model} ready.")
        else:
            warn(f"Failed to pull {model} — run 'ollama pull {model}' manually.")

# Optional models
for model, size, desc in OPTIONAL_MODELS:
    print()
    if ask(f"Install {bold(model)} ({size} — {desc})?", default="n") == "y":
        rc = run(f"ollama pull {model}")
        if rc == 0:
            ok(f"{model} ready.")
        else:
            warn(f"Failed to pull {model}.")
    else:
        info(f"Skipping {model}.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 4 — Windows Firewall
# ═════════════════════════════════════════════════════════════════════════════
sep("4 / 6  —  Windows Firewall")
print("  The host PC needs to reach port 11434 on this machine.")
print("  This will add an inbound firewall rule for TCP port 11434.")
print()
if ask("Allow inbound connections on port 11434?") == "y":
    # Remove existing rule first to avoid duplicates
    run('netsh advfirewall firewall delete rule name="Ollama AI Server" >nul 2>&1')
    rc = run(
        'netsh advfirewall firewall add rule '
        'name="Ollama AI Server" dir=in action=allow '
        'protocol=TCP localport=11434 description="Jarvis AI Ollama worker"'
    )
    if rc == 0:
        ok("Firewall rule added: port 11434 open for inbound TCP.")
    else:
        warn("Failed to add firewall rule — try running as Administrator.")
else:
    warn("Firewall rule SKIPPED — the host PC may not be able to connect!")
    warn("Add it manually: netsh advfirewall firewall add rule "
         'name="Ollama AI Server" dir=in action=allow protocol=TCP localport=11434')

# ═════════════════════════════════════════════════════════════════════════════
# STEP 5 — Auto-start on Windows boot
# ═════════════════════════════════════════════════════════════════════════════
sep("5 / 6  —  Auto-start on boot")
print("  When Windows starts, Ollama should already be running")
print("  so the host PC can send requests immediately.")
print()
if ask("Start Ollama automatically when Windows boots?") == "y":
    # Use HKCU registry Run key (current user, no password required)
    import winreg
    try:
        key_path = r"Software\Microsoft\Windows\CurrentVersion\Run"
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                            winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, "OllamaAIServer", 0, winreg.REG_SZ,
                              "ollama serve")
        ok("Registry startup entry created (HKCU\\...\\Run → 'ollama serve').")
        info("Ollama will start automatically when you log in.")
    except Exception as exc:
        warn(f"Could not write registry: {exc}")
        # Fallback: Task Scheduler
        info("Trying Task Scheduler fallback...")
        task_xml = """\
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo><Description>Jarvis AI Ollama worker</Description></RegistrationInfo>
  <Triggers><BootTrigger><Enabled>true</Enabled></BootTrigger></Triggers>
  <Principals>
    <Principal id="Author">
      <UserId>BUILTIN\\Administrators</UserId>
      <LogonType>S4U</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
  </Settings>
  <Actions Context="Author">
    <Exec><Command>ollama</Command><Arguments>serve</Arguments></Exec>
  </Actions>
</Task>"""
        xml_path = Path(os.environ.get("TEMP", ".")) / "ollama_task.xml"
        xml_path.write_text(task_xml, encoding="utf-16")
        rc = run(f'schtasks /Create /TN "Ollama AI Server" /XML "{xml_path}" /F /RU SYSTEM')
        xml_path.unlink(missing_ok=True)
        if rc == 0:
            ok("Task Scheduler entry created — Ollama starts at boot (SYSTEM).")
        else:
            warn("Could not create auto-start. Start Ollama manually before using it.")
else:
    info("Auto-start skipped. Remember to run 'ollama serve' before the host connects.")

# ═════════════════════════════════════════════════════════════════════════════
# STEP 6 — Verify and show connection info
# ═════════════════════════════════════════════════════════════════════════════
sep("6 / 6  —  Verification")

# Verify Ollama API
try:
    import requests as _req
    resp = _req.get("http://localhost:11434/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    ok(f"Ollama API responding. Models installed: {len(models)}")
    for m in models:
        print(f"    • {m}")
except Exception as exc:
    warn(f"Could not reach Ollama API: {exc}")
    warn("Run 'ollama serve' in a terminal to start it manually.")

# Find all non-loopback IPs
print()
info("This machine's network addresses (add one to Jarvis Settings → Ollama Hosts):")
try:
    hostname = socket.gethostname()
    addrs = socket.getaddrinfo(hostname, None)
    seen = set()
    for addr in addrs:
        ip = addr[4][0]
        if ip not in seen and not ip.startswith("127.") and ":" not in ip:
            seen.add(ip)
            print(f"    {green(ip + ':11434')}")
except Exception:
    pass
# Also get all adapters via PowerShell for thoroughness
ips_ps = run_out(
    "powershell -NoProfile -Command "
    "\"Get-NetIPAddress -AddressFamily IPv4 | "
    "Where-Object { $_.IPAddress -ne '127.0.0.1' } | "
    "Select-Object -ExpandProperty IPAddress\""
)
for ip in ips_ps.splitlines():
    ip = ip.strip()
    if ip and ip not in seen:
        print(f"    {green(ip + ':11434')}")

print()
sep()
print(bold(green("  Setup complete!")))
print()
print("  Next steps on the HOST machine:")
print("   1. Open Jarvis → Settings (⚙) → Ollama Hosts")
print("   2. Click '+ Add' and enter this machine's IP:port shown above")
print("   3. Check the checkbox to activate it")
print("   4. Set GPU layers (default 999 = full GPU,")
print("      lower if you want CPU+RAM overflow, 0 = CPU only)")
print("   5. Click Save")
print()
input("  Press Enter to exit...")
