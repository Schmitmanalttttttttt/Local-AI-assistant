Jarvis AI Assistant

Like the one from Iron Man, except real, local, and a little bit vibe-coded. Consider this early access — like Subnautica 2, except shittily written. An official release is coming eventually. Right now the goal is to get it to a point where it actually feels like an assistant before anything gets called "done."

⚠️ Warning: This assistant has deep system access. It can execute commands, manage files, control browsers, access hardware stats, and interact with your system. Run only on a machine you own and trust.

A fully local AI assistant for Windows with offline voice control, browser automation, file management, camera vision, object detection, face recognition, scheduling, memory, and multi-model AI routing. No subscriptions, no cloud dependency after setup.

Features

Offline voice input via Whisper small.en
Kokoro-82M text-to-speech with selectable voices
File management (read / write / move / delete / copy)
Browser automation via Playwright
App launcher and script runner
System controls (volume, processes, hardware stats, uptime, battery)
Task scheduling and reminders
Persistent memory + semantic search embeddings
Webcam vision with YOLOv8 object detection
Face recognition + DeepFace analysis
Optional screen observer using a vision model
Automatic 3-tier LLM routing for fast / normal / deep reasoning tasks
Spotify integration (requires a Spotify Developer account — see below)
Multi-machine inference routing — link a second PC for extra processing power


Camera & Vision

YOLOv8 nano real-time object detection
30 FPS webcam display with threaded detection
CPU-only inference to avoid GPU conflicts
Object / face detection mode selector
"Who" button stores known faces locally
Scene description support via Ollama vision models


Multi-machine inference
You can link a second computer as a remote Ollama inference node. This is highly recommended if you want to run the 70B deep think model (DeepSeek R1 Distill 70B) — it's too big for a single consumer GPU.
Processing is split into four routing groups:
GroupPurposeChat commandsNormal conversation and assistant tasksCameraVision / object detection inferenceScheduled tasks / overflowBackground tasks and overflow from other groupsDefaultHandles everything — use this if you're not sure how to configure the others
You can configure which machines handle which groups in the settings UI. If you just want things to work, pick Default.

Requirements
Hardware
I'm running this on an ASUS ProArt P16 with an RTX 4060 and a Ryzen AI 9 HX 370. If you're at roughly that level you should be fine. Anything lower will just be slower — use your judgment.

Windows 10 or 11 (64-bit)
Python 3.9+
16 GB RAM recommended
GPU optional but strongly recommended for faster LLM responses

For the 70B deep think model, a second machine with its own GPU is highly recommended. See the multi-machine section above.
Software
Install Ollama: https://ollama.com/download/windows
Ollama must be running before you launch Jarvis.

Required Ollama models
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
ollama pull llama3.2-vision
Model roles
ModelRoleqwen2.5:1.5bFast / simple repliesqwen2.5:7bStandard assistant tasksdeepseek-r1:7bComplex reasoningnomic-embed-textMemory embeddingsllama3.2-visionCamera and screen understanding
Optional (requires a second machine with enough VRAM):
ollama pull deepseek-r1:70b

Installation
Automatic (recommended)
run.bat
Manual dependencies
pip install faster-whisper numpy requests keyboard psutil pystray Pillow kokoro sounddevice pyaudio pycaw comtypes opencv-python deepface tf-keras ultralytics spotipy
If Kokoro fails:
pip install git+https://github.com/hexgrad/kokoro.git

First-run downloads
These are downloaded automatically on first launch and cached locally. After that, the assistant works fully offline.

Whisper small.en (~244 MB)
Kokoro-82M (~350 MB)
YOLOv8 nano (~6 MB)


Spotify setup
Spotify integration requires a free Spotify Developer account.

Go to https://developer.spotify.com/dashboard and create an app
Set the redirect URI to http://localhost:8888/callback
Copy your Client ID and Client Secret into Jarvis settings

This requirement may be removed in a future update.

File structure
assistant.py        Main assistant
run.bat             Main launcher
run_debug.bat       Debug launcher
scripts/            Voice-triggerable scripts
memory/             Memory, tasks, faces, playbooks
logs/               Debug logs

Debugging
Run run_debug.bat for a full diagnostic pass:

Package / version checks
Ollama connectivity
Audio device checks
Cache status
Hotkey conflicts
Port and permission checks


Limitations

Mostly AI-generated codebase — expect bugs
Windows-only for now
DeepFace estimates are approximate (age, gender, emotion)
Ollama must already be running before launch
Hotkeys may require administrator privileges
Camera inference intentionally runs on CPU to avoid GPU conflicts
Multi-machine processing selector is present but not yet fully tested
Spotify requires a developer account for now


Changelog
June 2025

Refreshed UI — cleaner layout
Improved facial recognition accuracy
Performance: persistent SQLite connection (saves 5–10ms per query)
Performance: synchronous=NORMAL + cache_size=-20000 (~5x faster writes)
Performance: vectorized cosine similarity
Performance: mtime file cache with cache invalidation on save
Performance: script/camera bypass before embedding (saves ~100ms)
Added Spotify integration
Added secondary PC inference node support
Added multi-machine / single-machine processing selector (experimental — not fully tested)
Added processing group routing (chat / camera / scheduled / default)
Removed boot-up time display from console (was inaccurate)
General code cleanup
