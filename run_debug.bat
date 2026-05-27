@echo off
echo ============================================
echo   AI File Assistant  ^|  DEV MODE
echo ============================================
echo.

cd /d "%~dp0"
if not exist logs mkdir logs

REM Get timestamp for log filename
powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss" > "%TEMP%\jv_ts.tmp"
set /p DEVTS=<"%TEMP%\jv_ts.tmp"
del "%TEMP%\jv_ts.tmp" 2>nul
set DEVLOG=logs\dev_%DEVTS%.log
set CONVLOG=logs\conversation_%DEVTS%.log

echo [DEV] Diagnostic log  : %DEVLOG%
echo [DEV] Conversation log: %CONVLOG%
echo.

REM =====================================================================
REM  STEP 1/4  -  Install / update dependencies  (verbose, no --quiet)
REM  Mirrors run.bat but shows every pip line so failures are obvious
REM =====================================================================
echo [1/3] Installing / verifying dependencies...
echo       (verbose - all pip output shown)
echo.
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
    comtypes

if errorlevel 1 (
    echo.
    echo ERROR: pip install failed. Fix the errors above before continuing.
    pause
    exit /b 1
)

echo.
echo [2/3] Dependencies OK. Running diagnostics...
echo.

REM =====================================================================
REM  BLOCK 1  -  System / Python / Packages / CUDA / Ollama
REM  Note: Log'string' (no space) is a PS parse error - always Log 'string'
REM =====================================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "function Log([string]$m){Write-Host $m;Add-Content '%DEVLOG%' $m -Encoding ascii};" ^
    "Log '================================================================';" ^
    "Log '  DEV DIAGNOSTIC REPORT - JARVIS AI ASSISTANT';" ^
    "Log '================================================================';" ^
    "Log ('  Started       : ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'));" ^
    "Log ('  Diagnostic log: %DEVLOG%');" ^
    "Log ('  Conv log      : %CONVLOG%');" ^
    "Log '  Flags         : PYTHONDEVMODE + all warnings + faulthandler + tracemalloc=5';" ^
    "Log '';" ^
    "Log '--- 1. SYSTEM -------------------------------------------------';" ^
    "Log ('  OS      : ' + [Environment]::OSVersion.VersionString);" ^
    "Log ('  Arch    : ' + $env:PROCESSOR_ARCHITECTURE);" ^
    "Log ('  Cores   : ' + $env:NUMBER_OF_PROCESSORS);" ^
    "Log ('  User    : ' + $env:USERNAME + ' @ ' + $env:COMPUTERNAME);" ^
    "Log ('  Dir     : ' + (Get-Location).Path);" ^
    "Log ('  TEMP    : ' + $env:TEMP);" ^
    "try { $ram=(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory; Log ('  RAM     : '+[math]::Round($ram/1GB,1)+' GB') } catch {};" ^
    "try { $d=Get-PSDrive C; Log ('  Disk C: : '+[math]::Round($d.Free/1GB,1)+' GB free of '+[math]::Round(($d.Free+$d.Used)/1GB,1)+' GB') } catch {};" ^
    "Log '';" ^
    "Log '--- 2. PYTHON -------------------------------------------------';" ^
    "$pv=(python --version 2>&1); Log ('  Version    : '+$pv);" ^
    "$pe=(python -c 'import sys;print(sys.executable)' 2>$null); Log ('  Executable : '+$pe);" ^
    "$pf=(python -c 'import sys;print(sys.version.replace(chr(10),chr(32)))' 2>$null); Log ('  Full build : '+$pf);" ^
    "$pp=(pip --version 2>&1); Log ('  pip        : '+$pp);" ^
    "Log '';" ^
    "Log '--- 3. PACKAGES -----------------------------------------------';" ^
    "foreach ($p in @('faster-whisper','numpy','requests','keyboard','psutil','pystray','Pillow','kokoro','sounddevice','pyaudio','pycaw','comtypes','torch','huggingface-hub')) {" ^
    "  $raw=(pip show $p 2>$null | Where-Object { $_ -like 'Version:*' } | Select-Object -First 1);" ^
    "  if ($raw) { Log ('  [OK]  '+$p.PadRight(18)+'  v'+($raw -replace 'Version:\s*','')) }" ^
    "  else { Log ('  [!!]  '+$p.PadRight(18)+'  NOT INSTALLED  -->  pip install '+$p) }" ^
    "};" ^
    "Log '';" ^
    "Log '--- 4. CUDA / GPU ---------------------------------------------';" ^
    "$cu=(python -c 'import torch;print(torch.cuda.is_available(),torch.cuda.device_count(),torch.__version__)' 2>$null);" ^
    "if ($LASTEXITCODE -eq 0) { Log ('  torch : '+$cu) } else { Log '  torch not installed - Whisper runs CPU/int8 (expected)' };" ^
    "Log '';" ^
    "Log '--- 5. OLLAMA CONNECTIVITY ------------------------------------';" ^
    "try { $r=Invoke-WebRequest -Uri 'http://localhost:11434' -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop; Log ('  [OK]  Ollama reachable - HTTP '+$r.StatusCode) } catch { Log ('  [!!]  Ollama NOT reachable: '+$_.Exception.Message); Log '        All AI responses will fail. Start Ollama first.' };" ^
    "Log '';" ^
    "Log '--- 6. OLLAMA MODELS ------------------------------------------';" ^
    "Log '  Required: qwen2.5:1.5b  qwen2.5:7b  deepseek-r1:7b  nomic-embed-text';" ^
    "try {" ^
    "  $mj=Invoke-WebRequest -Uri 'http://localhost:11434/api/tags' -UseBasicParsing -TimeoutSec 5 -ErrorAction Stop;" ^
    "  $ml=($mj.Content | ConvertFrom-Json).models | ForEach-Object { $_.name };" ^
    "  Log ('  Installed ('+@($ml).Count+' total): '+($ml -join ',  '));" ^
    "  foreach ($req in @('qwen2.5:1.5b','qwen2.5:7b','deepseek-r1:7b','nomic-embed-text')) { if ($ml -contains $req) { Log ('  [OK]  '+$req) } else { Log ('  [!!]  '+$req+' MISSING  -->  ollama pull '+$req) } }" ^
    "} catch { Log '  [!!]  Could not query model list (Ollama not running?)' };" ^
    "Log ''"

REM =====================================================================
REM  BLOCK 2  -  Config / Memory / Caches / Hotkeys / Ports / Perms
REM  Key rule: always put a space between Log and the quoted string
REM =====================================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "function Log([string]$m){Write-Host $m;Add-Content '%DEVLOG%' $m -Encoding ascii};" ^
    "Log '';" ^
    "Log '--- 7. CONFIG FILE --------------------------------------------';" ^
    "$cf=[Environment]::GetFolderPath('UserProfile')+'\.ai_assistant_config.json';" ^
    "Log ('  Path: '+$cf);" ^
    "if (Test-Path $cf) {" ^
    "  Log '  [OK]  Found. Contents:';" ^
    "  (Get-Content $cf) | ForEach-Object { Log ('    '+$_) }" ^
    "} else {" ^
    "  Log '  [ ]   Not found - defaults on first run (hotkey=ctrl+shift+space)';" ^
    "};" ^
    "Log '';" ^
    "Log '--- 8. MEMORY DIRECTORY ---------------------------------------';" ^
    "$md=Join-Path (Get-Location).Path 'memory';" ^
    "Log ('  Path: '+$md);" ^
    "if (Test-Path $md) {" ^
    "  Log '  [OK]  Exists. Files:';" ^
    "  Get-ChildItem $md -File | ForEach-Object { Log ('    '+$_.Name.PadRight(28)+[math]::Round($_.Length/1KB,1)+' KB  '+$_.LastWriteTime) };" ^
    "  $ff=Join-Path $md 'feedback.json';" ^
    "  if (Test-Path $ff) { try { $fb=Get-Content $ff -Raw|ConvertFrom-Json; Log ('    feedback entries     : '+$fb.interactions.Count) } catch { Log '    feedback.json : parse error' } };" ^
    "  $ef=Join-Path $md 'explicit_memory.json';" ^
    "  if (Test-Path $ef) { try { $em=Get-Content $ef -Raw|ConvertFrom-Json; Log ('    explicit memories    : '+$em.Count) } catch { Log '    explicit_memory.json : parse error' } };" ^
    "  $pf=Join-Path $md 'playbooks.json';" ^
    "  if (Test-Path $pf) { try { $pb=Get-Content $pf -Raw|ConvertFrom-Json; Log ('    playbooks            : '+$pb.Count) } catch { Log '    playbooks.json : parse error' } }" ^
    "} else {" ^
    "  Log '  [ ]   Not found - will be created on first run';" ^
    "};" ^
    "Log '';" ^
    "Log '--- 9. WHISPER + KOKORO CACHE ---------------------------------';" ^
    "Log '  Whisper: small.en, cpu, int8  (~244MB first run)';" ^
    "Log '  Kokoro : hexgrad/Kokoro-82M   (~350MB first TTS use)';" ^
    "$hf=[Environment]::GetFolderPath('UserProfile')+'\.cache\huggingface\hub';" ^
    "if (Test-Path $hf) {" ^
    "  $wd=Get-ChildItem $hf -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match 'whisper|faster' };" ^
    "  if ($wd) { Log '  [OK]  Whisper cached:'; $wd | ForEach-Object { Log ('    '+$_.FullName) } } else { Log '  [ ]   Whisper not cached - will download on first run' };" ^
    "  $kd=Get-ChildItem $hf -Recurse -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '[Kk]okoro' };" ^
    "  if ($kd) { Log '  [OK]  Kokoro cached:'; $kd | ForEach-Object { Log ('    '+$_.FullName) } } else { Log '  [ ]   Kokoro not cached - will download on first TTS use' }" ^
    "} else {" ^
    "  Log '  [ ]   HuggingFace cache not found - both models download on first run';" ^
    "};" ^
    "Log '';" ^
    "Log '--- 10. HOTKEY CONFLICT SCAN ----------------------------------';" ^
    "$hotkey='ctrl+shift+space';" ^
    "$cf2=[Environment]::GetFolderPath('UserProfile')+'\.ai_assistant_config.json';" ^
    "if (Test-Path $cf2) { try { $hk=(Get-Content $cf2 -Raw|ConvertFrom-Json).hotkey; if ($hk) { $hotkey=$hk } } catch {} };" ^
    "Log ('  Configured hotkey: '+$hotkey);" ^
    "foreach ($r in @('powertoys','autohotkey','ahk','keypirinha','launchy','flow.launcher','wox','ueli','executor','cerebro')) {" ^
    "  if (Get-Process -Name ('*'+$r+'*') -ErrorAction SilentlyContinue) { Log ('  [!!]  '+$r+' is running - may intercept the hotkey') } else { Log ('  [  ]  '+$r) }" ^
    "};" ^
    "Log '';" ^
    "Log '--- 11. PORTS -------------------------------------------------';" ^
    "try {" ^
    "  $t=New-Object Net.Sockets.TcpClient; $a=$t.BeginConnect('127.0.0.1',11434,$null,$null); $ok=$a.AsyncWaitHandle.WaitOne(500,$false); $t.Close();" ^
    "  if ($ok) { Log '  [OK]  :11434 OPEN (Ollama listening)' } else { Log '  [!!]  :11434 CLOSED - Ollama not listening' }" ^
    "} catch { Log '  [!!]  :11434 CLOSED - Ollama not listening' };" ^
    "Log '';" ^
    "Log '--- 12. PERMISSIONS + ENVIRONMENT ----------------------------';" ^
    "$admin=([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator);" ^
    "if ($admin) { Log '  [OK]  Running as Administrator' } else { Log '  [ ]   Not admin - hotkey may fail' };" ^
    "if (-not $admin) { Log '        Re-run as Admin if Ctrl+Shift+Space does nothing.' };" ^
    "try { 'x'|Set-Content (Join-Path $env:TEMP 'jv_wtest.tmp') -ErrorAction Stop; Remove-Item (Join-Path $env:TEMP 'jv_wtest.tmp') -Force -ErrorAction SilentlyContinue; Log ('  [OK]  TEMP writable: '+$env:TEMP) } catch { Log ('  [!!]  TEMP not writable - TTS WAV playback will FAIL: '+$env:TEMP) };" ^
    "Log ('  PS version : '+$PSVersionTable.PSVersion.ToString());" ^
    "Log ''"

REM =====================================================================
REM  BLOCK 3  -  NEW: assistant.py / audio / env / processes / ports / logs
REM =====================================================================
powershell -NoProfile -ExecutionPolicy Bypass -Command ^
    "function Log([string]$m){Write-Host $m;Add-Content '%DEVLOG%' $m -Encoding ascii};" ^
    "Log '--- 13. ASSISTANT.PY FILE CHECK ------------------------------';" ^
    "$ap=Join-Path (Get-Location).Path 'assistant.py';" ^
    "if (Test-Path $ap) {" ^
    "  $fi=Get-Item $ap;" ^
    "  $lc=@(Get-Content $ap -ErrorAction SilentlyContinue).Count;" ^
    "  Log ('  [OK]  Found: '+$ap);" ^
    "  Log ('    Size     : '+[math]::Round($fi.Length/1KB,1)+' KB');" ^
    "  Log ('    Lines    : '+$lc);" ^
    "  Log ('    Modified : '+$fi.LastWriteTime);" ^
    "  $syn=(python -c 'import ast; ast.parse(open(""assistant.py"",encoding=""utf-8"").read()); print(""OK"")' 2>&1);" ^
    "  if ($LASTEXITCODE -eq 0) { Log '    Syntax   : [OK]  No parse errors' } else { Log ('    Syntax   : [!!] Parse error - check assistant.py') }" ^
    "} else {" ^
    "  Log '  [!!]  assistant.py NOT FOUND - cannot launch!';" ^
    "  Log ('        Expected at: '+$ap);" ^
    "  Log '        Ensure run_debug.bat and assistant.py are in the same folder.';" ^
    "};" ^
    "Log '';" ^
    "Log '--- 14. AUDIO DEVICES (WMI) ----------------------------------';" ^
    "try {" ^
    "  $devs=Get-CimInstance Win32_SoundDevice -ErrorAction Stop;" ^
    "  if ($devs) { $devs | ForEach-Object { Log ('  '+$_.Name.PadRight(42)+' Status: '+$_.Status) } }" ^
    "  else { Log '  (no audio devices found via WMI)' }" ^
    "} catch { Log '  [!!]  WMI audio device query failed' };" ^
    "Log '';" ^
    "Log '--- 15. WINDOWS AUDIO SERVICES -------------------------------';" ^
    "foreach ($sn in @('Audiosrv','AudioEndpointBuilder','RtkAudioUniversalService')) {" ^
    "  $svc=Get-Service -Name $sn -ErrorAction SilentlyContinue;" ^
    "  if ($svc) { if ($svc.Status -eq 'Running') { Log ('  [OK]  '+$sn+' : Running') } else { Log ('  [!!]  '+$sn+' : '+$svc.Status+' - mic/TTS may fail') } } else { Log ('  [  ]  '+$sn+' not found') }" ^
    "};" ^
    "Log '';" ^
    "Log '--- 16. PYTHON-RELEVANT ENV VARS -----------------------------';" ^
    "foreach ($v in @('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','CONDA_DEFAULT_ENV','CONDA_PREFIX','HF_HOME','HUGGINGFACE_HUB_CACHE','TRANSFORMERS_CACHE','OLLAMA_HOST','OLLAMA_MODELS','CUDA_VISIBLE_DEVICES','CUDA_HOME')) {" ^
    "  $val=[Environment]::GetEnvironmentVariable($v);" ^
    "  if ($val) { Log ('  [SET] '+$v.PadRight(26)+' = '+$val) } else { Log ('  [   ] '+$v.PadRight(26)+' (not set)') }" ^
    "};" ^
    "Log '';" ^
    "Log '--- 17. CONFLICTING AI / AUDIO PROCESSES --------------------';" ^
    "Log '  (processes that may compete for mic, GPU, or hotkeys)';" ^
    "foreach ($r in @('ollama','whisper','vosk','dragon','cortana','speechruntime','audacity','obs64','obs32','discord','teams','zoom','slack')) {" ^
    "  $p=Get-Process -Name ('*'+$r+'*') -ErrorAction SilentlyContinue;" ^
    "  if ($p) { Log ('  [RUN] '+$r.PadRight(20)+' PID: '+($p | Select-Object -First 1 -ExpandProperty Id)) } else { Log ('  [   ] '+$r) }" ^
    "};" ^
    "Log '';" ^
    "Log '--- 18. LOCALHOST PORT SCAN ----------------------------------';" ^
    "foreach ($port in @(11434, 8080, 5000, 7860, 3000, 8000)) {" ^
    "  try { $t=New-Object Net.Sockets.TcpClient; $a=$t.BeginConnect('127.0.0.1',$port,$null,$null); $ok=$a.AsyncWaitHandle.WaitOne(300,$false); $t.Close();" ^
    "    if ($ok) { Log ('  [OPEN] :'+$port) } else { Log ('  [    ] :'+$port+' closed') }" ^
    "  } catch { Log ('  [    ] :'+$port+' closed') }" ^
    "};" ^
    "Log '';" ^
    "Log '--- 19. RECENT LOG FILES IN ./logs/ --------------------------';" ^
    "$ld=Join-Path (Get-Location).Path 'logs';" ^
    "if (Test-Path $ld) {" ^
    "  Get-ChildItem $ld -File | Sort-Object LastWriteTime -Descending | Select-Object -First 10 | ForEach-Object { Log ('  '+$_.Name.PadRight(40)+[math]::Round($_.Length/1KB,1)+' KB  '+$_.LastWriteTime) }" ^
    "} else { Log '  (no logs directory yet)' };" ^
    "Log '';" ^
    "Log '================================================================';" ^
    "Log '  END OF DIAGNOSTICS';" ^
    "Log '================================================================';" ^
    "Log ''"



REM =====================================================================
REM  STEP 4/4  -  Set debug flags and launch
REM  No PowerShell pipe - tkinter/GUI apps don't work through Tee-Object.
REM  We launch _dev_launcher.py which loads and runs assistant.py via
REM  exec(), identical in behaviour to "python assistant.py".
REM  After the assistant exits, a session-end marker is appended.
REM =====================================================================
echo ================================================================
echo   [3/3] Launching assistant.py
echo   TIP:  App appears in system tray.  Close this window to stop.
echo   LOGS: %DEVLOG%
echo   CONV: %CONVLOG%
echo ================================================================
echo.

set PYTHONWARNINGS=all
set PYTHONUTF8=1
set PYTHONFAULTHANDLER=1
set PYTHONDEVMODE=1
set PYTHONASYNCIODEBUG=1
set PYTHONTRACEMALLOC=5

python -u -W all -X dev assistant.py

powershell -NoProfile -Command ^
    "Add-Content '%CONVLOG%' ('[' + (Get-Date -Format 'HH:mm:ss') + '] [====] SESSION END ====') -Encoding UTF8" 2>nul

echo.
pause
