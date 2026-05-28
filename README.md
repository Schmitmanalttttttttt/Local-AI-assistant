# Jarvis AI Assistant (like the one from Iron Man lol :3)

Going to do an official release maybe soon-ish? I don't know. I want to get it somewhere where there's a lot more ease and capabilities that are beyond what it currently can do. I want it to be like an actual assistant before I actually make any kind of official launch. Consider this early access. (Like Subnautica 2, except really shittily written and little bit vibe coded.)

⚠️ Warning: This assistant has deep system access. It can execute commands, manage files, control browsers, access hardware stats, and interact with your system. Run only on a machine you own and trust.

A fully local AI assistant for Windows with offline voice control, browser automation, file management, camera vision, object detection, face recognition, scheduling, memory, and multi-model AI routing. No subscriptions, no cloud dependency after setup.

#Features

* Offline voice input via Whisper small.en
* Kokoro-82M text-to-speech with selectable voices
* File management (read/write/move/delete/copy)
* Browser automation via Playwright
* App launcher and script runner
* System controls (volume, processes, hardware stats, uptime, battery)
* Task scheduling and reminders
* Persistent memory + semantic search embeddings
* Webcam vision with YOLOv8 object detection
* Face recognition + DeepFace analysis
* Optional screen observer using a vision model
* Automatic 3-tier LLM routing for fast/normal/deep reasoning tasks

#Camera & Vision

* YOLOv8 nano real-time object detection
* 30 FPS webcam display with threaded detection
* CPU-only inference to avoid GPU conflicts
* Object/face detection mode selector
* “Who” button stores known faces locally
* Scene description support via Ollama vision models

#Requirements

Somebody needs to educate me, but I'm using a laptop RTX 4070 game-ready drivers, and I'm using a Ryzen 9 AI 370. If you're equal to that, you should be fine. Anything lower than that, it'll just be slower. Use at your own risk, I guess. 

#Hardware

* Windows 10/11 (64-bit)
* Python 3.9+
* 16 GB RAM recommended
* GPU optional but strongly recommended for faster LLM responses

#Software

Install Ollama:
https://ollama.com/download/windows

Ollama must be running before launching Jarvis.

#Required Ollama Models

```bash
ollama pull qwen2.5:1.5b
ollama pull qwen2.5:7b
ollama pull deepseek-r1:7b
ollama pull nomic-embed-text
ollama pull llama3.2-vision
```

#Model Roles

* qwen2.5:1.5b → fast/simple replies
* qwen2.5:7b → standard assistant tasks
* deepseek-r1:7b → complex reasoning
* nomic-embed-text → memory embeddings
* llama3.2-vision → camera/screen understanding

#Installation

#Automatic (Recommended)

Run:

```bash
run.bat
```

### Manual Dependencies

```bash
pip install faster-whisper numpy requests keyboard psutil pystray Pillow kokoro sounddevice pyaudio pycaw comtypes opencv-python deepface tf-keras ultralytics
```

If Kokoro fails:

```bash
pip install git+https://github.com/hexgrad/kokoro.git
```

#First-Run Downloads

* Whisper small.en (~244 MB)
* Kokoro-82M (~350 MB)
* YOLOv8 nano (~6 MB)

All downloads are cached locally. After setup, the assistant works fully offline.

#File Structure

```text
assistant.py      - Main assistant
run.bat           - Main launcher
run_debug.bat     - Debug launcher
scripts/          - Voice-triggerable scripts
memory/           - Memory, tasks, faces, playbooks
logs/             - Debug logs
```

#Debugging

Use `run_debug.bat` for diagnostics:

* Package/version checks
* Ollama connectivity
* Audio device checks
* Cache status
* Hotkey conflicts
* Port + permission checks

#Limitations

* Mostly AI-generated codebase — expect bugs
* Windows-only currently
* DeepFace estimates are approximate
* Ollama must already be running
* Hotkeys may require administrator privileges
* Camera inference intentionally runs on CPU
