"""
Jarvis AI File & Browser Assistant - Windows popup app with system tray
Press Ctrl+Shift+Space to open. Type or speak your commands.

Self-contained: run this file directly with `python assistant.py`
Dependencies are installed automatically on first launch.
"""

# ── Must be set before ANY import that could load TensorFlow's native DLL ────
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"]         = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]          = "3"
os.environ["GLOG_minloglevel"]              = "3"
os.environ["ABSL_MIN_LOG_LEVEL"]            = "3"
os.environ["TF_ENABLE_DEPRECATION_WARNINGS"]= "0"

# ── Self-Bootstrap (runs before anything else) ───────────────────────────────
import sys
import subprocess

_REQUIRED_PACKAGES = [
    # (pip package name,  import name to test)
    ("faster-whisper",   "faster_whisper"),
    ("numpy",            "numpy"),
    ("requests",         "requests"),
    ("keyboard",         "keyboard"),
    ("psutil",           "psutil"),
    ("pystray",          "pystray"),
    ("Pillow",           "PIL"),
    ("kokoro",           "kokoro"),
    ("sounddevice",      "sounddevice"),
    ("pyaudio",          "pyaudio"),
    ("pycaw",            "pycaw"),
    ("comtypes",         "comtypes"),
    ("opencv-python",    "cv2"),
    ("deepface",         "deepface"),
    ("tf-keras",         "tf_keras"),
    ("ultralytics",      "ultralytics"),
    ("chromadb",         "chromadb"),
]

# Packages that trigger heavy side-effects on import (TF loading, etc.)
# Check via find_spec (presence only) rather than actually importing them
_SPEC_CHECK_ONLY = {"tf_keras", "deepface"}

def _bootstrap():
    import importlib.util
    missing = []
    for pip_name, import_name in _REQUIRED_PACKAGES:
        try:
            if import_name in _SPEC_CHECK_ONLY:
                if importlib.util.find_spec(import_name) is None:
                    raise ImportError(import_name)
            else:
                __import__(import_name)
        except ImportError:
            missing.append(pip_name)

    if not missing:
        return  # all good — continue normally

    print("=" * 52)
    print("  Jarvis AI Assistant — First-Run Setup")
    print("=" * 52)
    print(f"\nInstalling {len(missing)} missing package(s):")
    for p in missing:
        print(f"  • {p}")
    print()

    pip_cmd = [sys.executable, "-m", "pip", "install", "--quiet"] + missing
    result = subprocess.run(pip_cmd)

    if result.returncode != 0:
        print("\n❌ Installation failed. Check your internet connection and try again.")
        sys.exit(1)

    print("\n✅ All dependencies installed. Restarting...\n")
    # Re-exec so newly installed packages are importable
    subprocess.run([sys.executable] + sys.argv)
    sys.exit(0)

_bootstrap()
import time as _time
_BOOT_START = _time.perf_counter()
# ─────────────────────────────────────────────────────────────────────────────

import json
import shutil
import threading
import subprocess
import requests
import queue
import time
import logging
import warnings
import ctypes
import base64
import sqlite3
import uuid
from io import BytesIO
import numpy as np

# Silence TF Python-level deprecation noise.
# Root-logger filters don't apply to records that propagate up from child
# loggers, so the filter must live on each TF-family logger directly.
class _BlockTFNoise(logging.Filter):
    _terms = ("sparse_softmax", "tf.losses", "deprecated", "is deprecated")
    def filter(self, record):
        if record.levelno < logging.ERROR:
            return False
        msg = record.getMessage()
        return not any(t in msg for t in self._terms)

for _ln in ("tensorflow", "tensorflow.python", "tensorflow.python.util",
            "absl", "tf_keras"):
    _lg = logging.getLogger(_ln)
    _lg.setLevel(logging.ERROR)
    _lg.addFilter(_BlockTFNoise())
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tf_keras")
warnings.filterwarnings("ignore", message=".*sparse_softmax_cross_entropy.*")
warnings.filterwarnings("ignore", message=r".*tf\.losses.*")
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")
import tkinter as tk
from tkinter import scrolledtext, filedialog, ttk, messagebox
from pathlib import Path
from datetime import datetime

# -- Voice & Offline AI Engines ----------------------------------------------
from faster_whisper import WhisperModel

# Silence noisy startup warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*dropout.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weight_norm.*", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*repo_id.*", category=UserWarning)

print("🤖 Connecting to Local Whisper Matrix...")
print("📥 Loading Whisper small.en model... (first run downloads ~244 MB, then cached offline)")

whisper_model = WhisperModel("small.en", device="cpu", compute_type="int8")
print("✅ Whisper small.en loaded. Systems Nominal.")

# -- TTS: Kokoro offline synthesis → WAV → shell-start playback --------------
_tts_queue: queue.Queue = queue.Queue()
_tts_pipeline = None
_tts_thread = None
_tts_stop_event = threading.Event()  # set to interrupt current playback on mute
_inference_stop = threading.Event()   # set to cancel current model request
_inference_response = None            # active streaming response; close() to abort immediately

_KOKORO_VOICES = {
    "British Male — George (Default)": "bm_george",
    "British Male — Lewis":            "bm_lewis",
    "British Female — Emma":           "bf_emma",
    "British Female — Isabella":       "bf_isabella",
    "American Male — Adam":            "am_adam",
    "American Male — Michael":         "am_michael",
    "American Female — Bella":         "af_bella",
    "American Female — Nicole":        "af_nicole",
    "American Female — Sarah":         "af_sarah",
    "American Female — Sky":           "af_sky",
}

def _lang_code_for_voice(voice_id: str) -> str:
    return 'a' if voice_id.startswith('a') else 'b'

def _tts_worker():
    """Stream each Kokoro audio chunk directly to sounddevice as it is synthesized.
    First audio plays almost immediately instead of waiting for full synthesis + PowerShell."""
    global _tts_pipeline
    import sounddevice as sd
    from kokoro import KPipeline
    current_lang = None

    while True:
        item = _tts_queue.get()
        if item is None:
            break
        text, app_ref = item
        if app_ref and getattr(app_ref, 'tts_muted', False):
            continue
        try:
            voice = _tts_voice
            needed_lang = _lang_code_for_voice(voice)
            if needed_lang != current_lang:
                print(f"[TTS] Initializing Kokoro pipeline (lang_code='{needed_lang}')...")
                _tts_pipeline = KPipeline(lang_code=needed_lang, repo_id='hexgrad/Kokoro-82M')
                current_lang = needed_lang
                print(f"✅ Kokoro TTS ready — voice '{voice}' active.")
            print(f"[TTS] Speaking: {text[:60]}...")
            _tts_stop_event.clear()

            play_kw = {"samplerate": 24000}
            if _sound_device_idx is not None:
                play_kw["device"] = _sound_device_idx

            for _, _, audio in _tts_pipeline(text, voice=voice, speed=1.0):
                if _tts_stop_event.is_set():
                    sd.stop()
                    break
                if app_ref and getattr(app_ref, 'tts_muted', False):
                    sd.stop()
                    break
                chunk = np.asarray(audio, dtype=np.float32)
                sd.play(chunk, **play_kw)
                # Poll based on chunk duration so mute/stop can interrupt
                deadline = time.time() + len(chunk) / 24000.0 + 0.05
                while time.time() < deadline:
                    if _tts_stop_event.is_set() or (app_ref and getattr(app_ref, 'tts_muted', False)):
                        sd.stop()
                        break
                    time.sleep(0.02)

        except Exception as e:
            print(f"[TTS] Kokoro synthesis error: {e}")
            current_lang = None

def _start_tts_thread():
    global _tts_thread
    _tts_thread = threading.Thread(target=_tts_worker, daemon=True)
    _tts_thread.start()

_start_tts_thread()

# -- Optional UI / System Hotkey Deps ----------------------------------------
try:
    import keyboard
    KEYBOARD_OK = True
except ImportError:
    KEYBOARD_OK = False

try:
    import psutil
    PSUTIL_OK = True
except ImportError:
    PSUTIL_OK = False
    print("⚠️ psutil not installed — hardware commands limited. Run: pip install psutil")

try:
    import pystray
    from PIL import Image, ImageDraw
    TRAY_OK = True
except ImportError:
    TRAY_OK = False

# -- Config -----------------------------------------------------------------
CONFIG_FILE = Path.home() / ".ai_assistant_config.json"
CHAT_MODEL   = "llama3.2:3b"     # fast tier — simple chat and greetings
OLLAMA_MODEL = "qwen3:4b"        # main tier — commands and normal queries
REASON_MODEL = "deepseek-r1:8b"  # deep tier — complex multi-step reasoning (override below if ultra-deep enabled)
EMBED_MODEL  = "nomic-embed-text"
ULTRA_DEEP_THINKING_MODEL = "llama3.1:70b"  # ultra tier — requires ~64GB RAM

_active_model = OLLAMA_MODEL  # tracks which model handled the last request

# Keywords that signal a complex, multi-step command → route to reasoning model
_COMPLEX_SIGNALS = {
    # Multi-step indicators
    "and then", "after that", "after you", "once you", "then do", "then go",
    "first", "next then", "finally", "step by step", "one by one", "in order",
    "in sequence", "sequentially", "stage by stage", "phase by phase",
    # Effort/reasoning indicators
    "figure out", "work out", "try to", "find a way", "figure it out",
    "try different", "keep trying", "try again", "attempt", "see if you can",
    "can you figure", "can you work out", "can you find", "can you try",
    # Automation/multi-action
    "automatically", "auto", "on your own", "by yourself", "without me",
    "navigate and", "open and", "click and", "go to and", "search and",
    "multiple", "several", "a series of", "a bunch of", "a few things",
    # Complex task phrasing
    "make sure", "ensure that", "verify that", "check that",
    "i need you to", "i want you to", "please go ahead and",
    "do all of", "take care of", "handle", "deal with",
}

_TRIVIAL_SIGNALS = {
    # Greetings
    "hello", "hi", "hey", "howdy", "sup", "what's up", "whats up", "yo",
    # How are you / small talk
    "how are you", "how are ya", "you doing", "you good", "you ok",
    "what's new", "whats new", "what's going on", "how's it going",
    # Simple identity / capability questions
    "who are you", "what are you", "what can you do", "what do you do",
    "are you there", "you there", "you awake", "you alive",
    # Acknowledgements
    "ok", "okay", "got it", "sure", "alright", "sounds good", "cool",
    "thanks", "thank you", "thx", "ty", "cheers", "appreciate it",
    "good job", "nice", "perfect", "great", "awesome", "amazing",
    # Simple yes/no
    "yes", "no", "yep", "nope", "yeah", "nah",
    # Sign-off
    "bye", "goodbye", "see you", "later", "good night", "good morning",
}

def _is_trivial(text: str) -> bool:
    lower = text.strip().lower().rstrip("!.?")
    if lower in _TRIVIAL_SIGNALS:
        return True
    if any(lower.startswith(kw) for kw in ("hi ", "hey ", "hello ", "thanks ", "thank you")):
        return True
    if len(text.split()) <= 4 and not any(c in lower for c in ("open", "go to", "search", "find", "list", "show", "run", "create", "delete", "move", "copy")):
        return True
    return False

def _is_complex(text: str) -> bool:
    lower = text.lower()
    if lower.count(" then ") >= 1:
        return True
    if any(kw in lower for kw in _COMPLEX_SIGNALS):
        return True
    if len(text.split()) > 25:
        return True
    return False

def get_embedding(text: str) -> list:
    try:
        r = requests.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text}, timeout=8)
        return r.json().get("embedding", [])
    except Exception:
        return []

def cosine_similarity(a: list, b: list) -> float:
    if not a or not b:
        return 0.0
    va, vb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

def load_config():
    if CONFIG_FILE.exists():
        with open(CONFIG_FILE) as f:
            cfg = json.load(f)
        cfg.pop("openai_api_key", None)
        return cfg
    return {"watched_folder": "", "hotkey": "ctrl+shift+space", "ollama_host": "localhost", "ollama_hosts": ["localhost"]}

def save_config(cfg):
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)

config = load_config()
_ollama_host = config.get("ollama_host", "localhost")
OLLAMA_URL = f"http://{_ollama_host}:11434/api/chat"
OLLAMA_EMBED_URL = f"http://{_ollama_host}:11434/api/embeddings"
_tts_voice = config.get("tts_voice", "bm_george")
CHAT_MEMORY = []
_SESSION_ID = str(uuid.uuid4())  # unique ID for this app launch

# Apply ultra-deep thinking model if enabled
if config.get("ultra_deep_thinking", False):
    REASON_MODEL = ULTRA_DEEP_THINKING_MODEL

# Always use system default — clear any previously saved override
_sound_device_idx = None
if "sound_device" in config:
    del config["sound_device"]
    save_config(config)

# -- Hardware Access Layer ---------------------------------------------------
def get_system_info() -> str:
    lines = []
    if PSUTIL_OK:
        cpu = psutil.cpu_percent(interval=0.5)
        ram = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        lines.append(f"CPU: {cpu:.1f}%")
        lines.append(f"RAM: {ram.percent:.1f}% used ({ram.used//1024//1024:,} MB / {ram.total//1024//1024:,} MB)")
        lines.append(f"Disk C:\\: {disk.percent:.1f}% used ({disk.used//1024//1024//1024:.1f} GB / {disk.total//1024//1024//1024:.1f} GB)")
        try:
            bat = psutil.sensors_battery()
            if bat:
                status = "charging" if bat.power_plugged else "discharging"
                lines.append(f"Battery: {bat.percent:.0f}% ({status})")
        except Exception:
            pass
        boot = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot
        h, m = divmod(int(uptime.total_seconds()) // 60, 60)
        lines.append(f"Uptime: {h}h {m}m")
        net = psutil.net_io_counters()
        lines.append(f"Network: ↑{net.bytes_sent//1024//1024} MB sent  ↓{net.bytes_recv//1024//1024} MB received")
    else:
        lines.append("psutil not installed — run: pip install psutil")
    return "🖥️ System Info:\n" + "\n".join(f"  {l}" for l in lines)

def list_processes(sort_by: str = "cpu") -> str:
    if not PSUTIL_OK:
        return "❌ psutil not installed."
    procs = []
    for p in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
        try:
            procs.append(p.info)
        except Exception:
            pass
    key = 'memory_percent' if sort_by == "ram" else 'cpu_percent'
    procs.sort(key=lambda x: x.get(key, 0) or 0, reverse=True)
    lines = [f"{'PID':>6}  {'CPU%':>5}  {'RAM%':>5}  Name"]
    for p in procs[:15]:
        lines.append(f"  {p['pid']:>6}  {(p.get('cpu_percent') or 0):>5.1f}  {(p.get('memory_percent') or 0):>5.1f}  {p['name']}")
    return "⚙️ Top Processes:\n" + "\n".join(lines)

def kill_process_by_name(name: str) -> str:
    if not PSUTIL_OK:
        return "❌ psutil not installed."
    killed = []
    for p in psutil.process_iter(['pid', 'name']):
        try:
            if name.lower() in p.info['name'].lower():
                p.kill()
                killed.append(f"{p.info['name']} (PID {p.info['pid']})")
        except Exception:
            pass
    return f"💀 Killed: {', '.join(killed)}" if killed else f"❌ No process found matching '{name}'"

def list_audio_devices() -> str:
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_out = sd.default.device[1]
        lines = [f"Current audio device: [{_sound_device_idx if _sound_device_idx is not None else default_out}]", "Available output devices:"]
        for i, d in enumerate(devices):
            if d['max_output_channels'] > 0:
                marker = " ◄ ACTIVE" if i == (_sound_device_idx or default_out) else ""
                lines.append(f"  [{i}] {d['name']}{marker}")
        lines.append('\nTo switch: tell Jarvis "set audio device to [number]"')
        return "\n".join(lines)
    except Exception as e:
        return f"❌ Could not query audio devices: {e}"

def set_audio_device(index: int) -> str:
    global _sound_device_idx
    _sound_device_idx = index
    config["sound_device"] = index
    save_config(config)
    return f"🔊 Audio output switched to device [{index}]. Takes effect immediately."

def get_volume() -> int:
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        return int(volume.GetMasterVolumeLevelScalar() * 100)
    except Exception:
        return -1

def set_volume(level: int) -> str:
    try:
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        from comtypes import CLSCTX_ALL
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        level = max(0, min(100, level))
        volume.SetMasterVolumeLevelScalar(level / 100.0, None)
        return f"🔊 Volume set to {level}%"
    except Exception as e:
        return f"❌ Volume control failed: {e}"

# -- Feedback & Learning System ----------------------------------------------
MEMORY_DIR = Path(__file__).parent / "memory"
MEMORY_DIR.mkdir(exist_ok=True)
FEEDBACK_FILE = MEMORY_DIR / "feedback.json"
FEEDBACK_LOG  = MEMORY_DIR / "feedback_log.txt"
EXPLICIT_MEMORY_FILE = MEMORY_DIR / "explicit_memory.json"
PLAYBOOK_FILE        = MEMORY_DIR / "playbooks.json"
SCHEDULED_FILE       = MEMORY_DIR / "scheduled_tasks.json"
SCRIPTS_FILE         = MEMORY_DIR / "scripts.json"
CHAT_HISTORY_DB      = MEMORY_DIR / "chat_history.db"
SCRIPTS_DIR          = Path(__file__).parent / "scripts"
SCRIPTS_DIR.mkdir(exist_ok=True)
KNOWN_FACES_DIR      = MEMORY_DIR / "known_faces"
KNOWN_FACES_DIR.mkdir(exist_ok=True)
LAST_INTERACTION: dict = {"text": "", "raw": ""}
_offline_mode: bool = False
_ui_app = None          # set after AssistantApp is constructed

# -- Chat History (SQLite) ---------------------------------------------------
class ChatHistoryDB:
    """Persistent, session-aware chat log with semantic retrieval.

    Storage: one SQLite file in memory/.
    Retrieval: cosine similarity over float32 embedding BLOBs — no extra deps.
    Save: background thread so it never blocks the UI.
    """

    def __init__(self, db_path: Path):
        self._path = str(db_path)
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self._path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self):
        with self._lock, self._conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chat_history (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id  TEXT    NOT NULL,
                    role        TEXT    NOT NULL,
                    content     TEXT    NOT NULL,
                    timestamp   REAL    NOT NULL,
                    embedding   BLOB
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_session   ON chat_history(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON chat_history(timestamp)")

    def save_exchange(self, user_text: str, assistant_text: str, user_emb: list):
        """Save a Q+A pair. Embedding is stored for the combined pair (better recall)."""
        combined = f"user: {user_text[:300]}\nassistant: {assistant_text[:300]}"
        emb_blob = None
        if user_emb:
            emb_blob = np.array(user_emb, dtype=np.float32).tobytes()

        now = time.time()
        def _write():
            with self._lock, self._conn() as conn:
                conn.execute(
                    "INSERT INTO chat_history(session_id,role,content,timestamp,embedding) VALUES(?,?,?,?,?)",
                    (_SESSION_ID, "user", user_text[:2000], now, emb_blob)
                )
                conn.execute(
                    "INSERT INTO chat_history(session_id,role,content,timestamp,embedding) VALUES(?,?,?,?,?)",
                    (_SESSION_ID, "assistant", assistant_text[:2000], now + 0.001, None)
                )
        threading.Thread(target=_write, daemon=True).start()

    def get_relevant_history(self, query_emb: list, top_k: int = 3,
                             exclude_session: str = None) -> str:
        """Return top-K semantically relevant past exchanges as a context string."""
        if not query_emb:
            return ""
        try:
            q_vec = np.array(query_emb, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm == 0:
                return ""

            with self._lock, self._conn() as conn:
                # Only fetch rows that have embeddings (user turns)
                rows = conn.execute(
                    "SELECT id, content, timestamp, embedding FROM chat_history "
                    "WHERE role='user' AND embedding IS NOT NULL "
                    "ORDER BY timestamp DESC LIMIT 500"
                ).fetchall()

            if not rows:
                return ""

            scored = []
            for row_id, content, ts, blob in rows:
                vec = np.frombuffer(blob, dtype=np.float32)
                norm = np.linalg.norm(vec)
                if norm == 0:
                    continue
                sim = float(np.dot(q_vec, vec) / (q_norm * norm))
                if sim > 0.30:
                    scored.append((sim, ts, row_id, content))

            if not scored:
                return ""

            scored.sort(key=lambda x: x[0], reverse=True)
            top = scored[:top_k]

            # For each matched user turn, fetch the assistant reply that follows it
            lines = ["Relevant past conversations:"]
            with self._lock, self._conn() as conn:
                for _, ts, row_id, user_content in top:
                    reply_row = conn.execute(
                        "SELECT content FROM chat_history WHERE role='assistant' "
                        "AND timestamp > ? ORDER BY timestamp ASC LIMIT 1",
                        (ts,)
                    ).fetchone()
                    reply = reply_row[0][:300] if reply_row else "(no reply)"
                    lines.append(f"  Q: {user_content[:200]}")
                    lines.append(f"  A: {reply}")

            result = "\n".join(lines)
            # Hard cap at 3,200 chars so it never floods the prompt
            return result[:3200]
        except Exception:
            return ""

    def get_sessions(self) -> list:
        """Return list of (session_id, first_timestamp, message_count) for UI browsing."""
        try:
            with self._lock, self._conn() as conn:
                return conn.execute(
                    "SELECT session_id, MIN(timestamp), COUNT(*) FROM chat_history "
                    "GROUP BY session_id ORDER BY MIN(timestamp) DESC LIMIT 100"
                ).fetchall()
        except Exception:
            return []

    def get_session_messages(self, session_id: str) -> list:
        """Return all (role, content, timestamp) rows for a session."""
        try:
            with self._lock, self._conn() as conn:
                return conn.execute(
                    "SELECT role, content, timestamp FROM chat_history "
                    "WHERE session_id=? ORDER BY timestamp ASC",
                    (session_id,)
                ).fetchall()
        except Exception:
            return []

    def reset(self):
        """Wipe all stored history. Opens and explicitly closes a connection so no lock lingers."""
        with self._lock:
            conn = self._conn()
            try:
                conn.execute("DELETE FROM chat_history")
                conn.commit()
                conn.execute("VACUUM")
                conn.commit()
            except Exception:
                pass
            finally:
                conn.close()

chat_db = ChatHistoryDB(CHAT_HISTORY_DB)
_memory_warned = False  # show the memory-size warning at most once per session

def _wipe_all_memory():
    """Delete all persistent memory without touching the directory itself.
    Avoids Windows file-lock issues that shutil.rmtree triggers on open SQLite files."""
    global CHAT_MEMORY, _memory_warned
    # 1. Clear the SQLite chat history (connection is explicitly closed inside reset())
    try:
        chat_db.reset()
    except Exception:
        pass
    # 2. Delete individual memory files (leaves the directory intact)
    for f in [EXPLICIT_MEMORY_FILE, FEEDBACK_FILE, FEEDBACK_LOG, PLAYBOOK_FILE,
              SCHEDULED_FILE, SCRIPTS_FILE]:
        try:
            if f.exists():
                f.unlink()
        except Exception:
            pass
    # 3. Wipe chroma_db sub-folder if present
    chroma_dir = MEMORY_DIR / "chroma_db"
    if chroma_dir.exists():
        try:
            shutil.rmtree(chroma_dir)
        except Exception:
            pass
    # 4. Wipe known faces
    if KNOWN_FACES_DIR.exists():
        try:
            shutil.rmtree(KNOWN_FACES_DIR)
        except Exception:
            pass
    KNOWN_FACES_DIR.mkdir(exist_ok=True)
    # 5. Reset in-memory state
    CHAT_MEMORY = []
    _memory_warned = False

_POSITIVE = {
    # Direct praise
    "good job", "great job", "nice job", "perfect job", "amazing job", "awesome job",
    "well done", "good work", "great work", "nice work", "excellent work",
    "good one", "great one", "nice one",
    # Correctness
    "correct", "that's correct", "that is correct", "you're correct", "you are correct",
    "that's right", "that is right", "you're right", "you are right", "exactly right",
    "yes that's it", "that's it", "yep that's it", "yeah that's it",
    "yes exactly", "exactly", "precisely", "spot on", "bang on", "dead on",
    "you got it", "nailed it", "bingo", "yes", "yep", "yeah",
    # Success confirmation
    "that worked", "it worked", "that's working", "it's working", "works perfectly",
    "perfect", "excellent", "fantastic", "brilliant", "amazing", "awesome", "great",
    "love it", "love that", "nice", "good", "that's what i wanted", "that's what i needed",
    "keep it up", "good response", "great response", "thumbs up",
}
_NEGATIVE = {
    # Direct criticism
    "bad job", "terrible job", "awful job", "horrible job", "poor job",
    "bad work", "poor work", "sloppy work",
    # Wrongness
    "wrong", "that's wrong", "that is wrong", "you're wrong", "you are wrong",
    "incorrect", "that's incorrect", "that is incorrect", "not correct",
    "not right", "that's not right", "that is not right",
    "that's not it", "that is not it", "not what i wanted", "not what i asked",
    "not what i meant", "that's not what i meant", "that's not what i wanted",
    # Failure confirmation
    "that didn't work", "it didn't work", "that failed", "it failed", "didn't work",
    "doesn't work", "not working", "broken", "fail", "failed",
    "wrong answer", "wrong response", "bad response", "bad answer",
    "no not that", "no that's wrong", "nope", "no", "thumbs down",
    "you messed up", "messed up", "try again", "not quite", "almost but no",
}

def detect_verbal_feedback(text: str):
    lower = text.lower().strip().rstrip(".").rstrip("!")
    # Exact match or the entire utterance starts with a feedback phrase
    if lower in _POSITIVE or any(lower.startswith(p) for p in _POSITIVE):
        return "positive"
    if lower in _NEGATIVE or any(lower.startswith(p) for p in _NEGATIVE):
        return "negative"
    # Short utterances (≤6 words) — check if a feedback phrase is contained
    if len(lower.split()) <= 6:
        if any(p in lower for p in _POSITIVE):
            return "positive"
        if any(p in lower for p in _NEGATIVE):
            return "negative"
    return None

def load_feedback() -> dict:
    if FEEDBACK_FILE.exists():
        try:
            with open(FEEDBACK_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {"interactions": []}

def save_feedback_entry(rating: int):
    if not LAST_INTERACTION["text"]:
        return
    timestamp = datetime.now().isoformat()
    embedding = get_embedding(LAST_INTERACTION["text"]) if rating > 0 else []
    entry = {
        "text": LAST_INTERACTION["text"],
        "raw": LAST_INTERACTION["raw"],
        "rating": rating,
        "timestamp": timestamp,
        "embedding": embedding,
    }
    data = load_feedback()
    data["interactions"].append(entry)
    data["interactions"] = data["interactions"][-300:]
    try:
        with open(FEEDBACK_FILE, "w") as f:
            json.dump(data, f, indent=2)
    except Exception:
        pass
    sign = "+" if rating > 0 else ""
    label = f"{sign}{rating}"
    try:
        with open(FEEDBACK_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n[{timestamp}] Rating: {label}\n")
            f.write(f"Command : {LAST_INTERACTION['text']}\n")
            f.write(f"Response: {LAST_INTERACTION['raw'][:200]}\n")
            f.write("-" * 60 + "\n")
    except Exception:
        pass

def get_learned_context(query: str = "", emb: list = None) -> str:
    data = load_feedback()
    good = [i for i in data["interactions"] if i.get("rating", 0) > 0 and i.get("raw")]
    if not good:
        return ""

    if query:
        # Semantic search: embed the query and rank by similarity
        query_emb = emb or get_embedding(query)
        if query_emb:
            scored = [
                (cosine_similarity(query_emb, i.get("embedding", [])), i)
                for i in good if i.get("embedding")
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            examples = [i for score, i in scored[:3] if score > 0.25]
            if examples:
                lines = ["Semantically similar past successful commands:"]
                for ex in examples:
                    lines.append(f'  "{ex["text"]}" → {ex["raw"][:140]}')
                return "\n".join(lines)

    # Fallback: recency-based (no embeddings yet or nomic not installed)
    seen, examples = set(), []
    for entry in reversed(good):
        key = entry["text"].lower()[:40]
        if key not in seen:
            seen.add(key)
            examples.append(entry)
        if len(examples) >= 3:
            break
    lines = ["Past successful commands (use only as loose guidance — context may differ):"]
    for ex in reversed(examples):
        lines.append(f'  "{ex["text"]}" → {ex["raw"][:140]}')
    return "\n".join(lines)

# -- Explicit Memory System --------------------------------------------------
def load_explicit_memories() -> list:
    if EXPLICIT_MEMORY_FILE.exists():
        try:
            with open(EXPLICIT_MEMORY_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_explicit_memory(fact: str):
    memories = load_explicit_memories()
    memories.append({
        "fact": fact,
        "embedding": get_embedding(fact),
        "timestamp": datetime.now().isoformat(),
    })
    memories = memories[-500:]
    try:
        with open(EXPLICIT_MEMORY_FILE, "w") as f:
            json.dump(memories, f, indent=2)
    except Exception:
        pass

def get_relevant_memories(query: str, emb: list = None) -> str:
    memories = load_explicit_memories()
    if not memories:
        return ""

    query_emb = emb or get_embedding(query)
    if query_emb:
        scored = [
            (cosine_similarity(query_emb, m.get("embedding", [])), m)
            for m in memories if m.get("embedding")
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [m for score, m in scored[:4] if score > 0.3]
    else:
        # Keyword overlap fallback when embeddings unavailable
        q_words = set(query.lower().split())
        def _overlap(m):
            return len(q_words & set(m["fact"].lower().split()))
        ranked = sorted(memories, key=_overlap, reverse=True)
        top = [m for m in ranked[:4] if _overlap(m) > 0]

    if not top:
        return ""
    lines = ["Relevant things Schmit has asked you to remember:"]
    for m in top:
        lines.append(f"  - {m['fact']}")
    return "\n".join(lines)

# -- Playbook (Macro Learning) -----------------------------------------------
def load_playbooks() -> list:
    if PLAYBOOK_FILE.exists():
        try:
            with open(PLAYBOOK_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def save_playbook(description: str, actions: list):
    playbooks = load_playbooks()
    playbooks.append({
        "description": description,
        "actions": actions,
        "embedding": get_embedding(description),
        "created": datetime.now().isoformat(),
        "used_count": 0,
    })
    playbooks = playbooks[-200:]
    try:
        with open(PLAYBOOK_FILE, "w") as f:
            json.dump(playbooks, f, indent=2)
    except Exception:
        pass

def get_relevant_playbooks(query: str, emb: list = None) -> str:
    playbooks = load_playbooks()
    if not playbooks:
        return ""
    query_emb = emb or get_embedding(query)
    if query_emb:
        scored = [(cosine_similarity(query_emb, p.get("embedding", [])), p)
                  for p in playbooks if p.get("embedding")]
        scored.sort(key=lambda x: x[0], reverse=True)
        top = [p for score, p in scored[:2] if score > 0.4]
    else:
        top = []
    if not top:
        return ""
    lines = ["Playbooks from similar past tasks (use as a starting point):"]
    for p in top:
        steps = " → ".join(
            f"{a['action']}({list(a.get('args', {}).values())[0] if a.get('args') else ''})"
            for a in p["actions"][:6]
        )
        lines.append(f'  "{p["description"]}" → {steps}')
    return "\n".join(lines)

# -- Script Trigger System ---------------------------------------------------
def load_scripts() -> dict:
    if SCRIPTS_FILE.exists():
        try:
            with open(SCRIPTS_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_script(trigger: str, path: str) -> str:
    scripts = load_scripts()
    scripts[trigger.lower().strip()] = path
    try:
        with open(SCRIPTS_FILE, "w") as f:
            json.dump(scripts, f, indent=2)
        return f"✅ Script saved: say '{trigger}' to run {Path(path).name}"
    except Exception as e:
        return f"❌ Could not save script: {e}"

def remove_script_entry(trigger: str) -> str:
    scripts = load_scripts()
    key = trigger.lower().strip()
    if key in scripts:
        del scripts[key]
        with open(SCRIPTS_FILE, "w") as f:
            json.dump(scripts, f, indent=2)
        return f"✅ Removed script trigger: '{trigger}'"
    return f"❌ No script found for '{trigger}'"

def _script_normalize(s: str) -> str:
    import re
    return re.sub(r'[\s\-_.,!?\'\"]+', '', s.lower())

def _camel_to_words(s: str) -> str:
    import re
    spaced = re.sub(r'([A-Z][a-z]+)', r' \1', re.sub(r'([A-Z]+)(?=[A-Z][a-z])', r' \1', s))
    return spaced.strip().lower()

def get_folder_scripts() -> dict:
    """Auto-discover scripts in scripts/ folder. Registers CamelCase and underscore variants."""
    found = {}
    for p in SCRIPTS_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in (".bat", ".cmd", ".ps1", ".py"):
            path = str(p)
            t1 = p.stem.lower().replace("_", " ").replace("-", " ")
            t2 = _camel_to_words(p.stem)
            for t in {t1, t2}:
                if t.strip():
                    found[t.strip()] = path
    return found

def match_script(text: str):
    import difflib
    scripts = load_scripts()
    scripts.update(get_folder_scripts())
    lower = text.lower().strip()

    # 1. Exact match
    if lower in scripts:
        return scripts[lower]

    # 2. Normalized — ignore all spaces (schoolwifilogin == school wifi login)
    norm_input = _script_normalize(lower)
    for trigger, path in scripts.items():
        if _script_normalize(trigger) == norm_input:
            return path

    # 3. All trigger words present somewhere in the input
    input_words = set(lower.split())
    best_overlap, best_path = 0, None
    for trigger, path in scripts.items():
        trig_words = set(trigger.split())
        if not trig_words:
            continue
        overlap = len(trig_words & input_words) / len(trig_words)
        if overlap == 1.0:
            return path
        if overlap > best_overlap:
            best_overlap, best_path = overlap, path

    # 4. Fuzzy similarity — catches typos and close misses
    triggers = list(scripts.keys())
    if triggers:
        matches = difflib.get_close_matches(lower, triggers, n=1, cutoff=0.55)
        if matches:
            return scripts[matches[0]]

    # 5. Best partial word-overlap above 60%
    if best_overlap >= 0.6 and best_path:
        return best_path

    return None

def run_script(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return f"❌ Script not found: {path}"
    ext = p.suffix.lower()
    try:
        if ext in (".bat", ".cmd"):
            result = subprocess.run(["cmd", "/c", str(p)], capture_output=True, text=True, timeout=60,
                                    creationflags=subprocess.CREATE_NO_WINDOW)
        elif ext == ".ps1":
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(p)],
                capture_output=True, text=True, timeout=60,
                creationflags=subprocess.CREATE_NO_WINDOW)
        elif ext == ".py":
            result = subprocess.run([sys.executable, str(p)], capture_output=True, text=True, timeout=60)
        else:
            return f"❌ Unsupported script type: {ext}. Use .bat, .cmd, .ps1, or .py"
        out = (result.stdout or result.stderr or "(no output)").strip()
        return f"🚀 Ran {p.name}:\n{out[:600]}"
    except subprocess.TimeoutExpired:
        return f"⏱️ Script timed out after 60s: {p.name}"
    except Exception as e:
        return f"❌ Script error: {e}"

# -- Task Scheduler ----------------------------------------------------------
def load_scheduled_tasks() -> list:
    if SCHEDULED_FILE.exists():
        try:
            with open(SCHEDULED_FILE) as f:
                return json.load(f)
        except Exception:
            pass
    return []

def _save_scheduled_tasks(tasks: list):
    try:
        with open(SCHEDULED_FILE, "w") as f:
            json.dump(tasks, f, indent=2)
    except Exception:
        pass

def add_scheduled_task(description: str, interval_minutes: int) -> str:
    tasks = load_scheduled_tasks()
    task_id = f"task_{int(time.time())}"
    next_run = (datetime.now() + __import__("datetime").timedelta(minutes=interval_minutes)).isoformat()
    tasks.append({
        "id": task_id,
        "description": description,
        "interval_minutes": interval_minutes,
        "created": datetime.now().isoformat(),
        "last_run": None,
        "next_run": next_run,
    })
    _save_scheduled_tasks(tasks)
    return task_id

def cancel_scheduled_task(task_id: str) -> bool:
    tasks = load_scheduled_tasks()
    before = len(tasks)
    tasks = [t for t in tasks if t["id"] != task_id]
    _save_scheduled_tasks(tasks)
    return len(tasks) < before

class TaskScheduler:
    def __init__(self):
        self._thread = None
        self._app_ref = None

    def start(self, app_ref):
        self._app_ref = app_ref
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while True:
            time.sleep(30)
            try:
                tasks = load_scheduled_tasks()
                now = datetime.now().isoformat()
                changed = False
                for task in tasks:
                    if task.get("next_run", "") <= now:
                        print(f"[Scheduler] Running: {task['description']}")
                        response = process_message(task["description"])
                        task["last_run"] = datetime.now().isoformat()
                        task["next_run"] = (
                            datetime.now() +
                            __import__("datetime").timedelta(minutes=task["interval_minutes"])
                        ).isoformat()
                        changed = True
                        if self._app_ref:
                            msg = f"⏰ Scheduled task ran: {task['description'][:60]}\n{response}"
                            self._app_ref.root.after(0, lambda m=msg: self._app_ref._append_message("system", m))
                if changed:
                    _save_scheduled_tasks(tasks)
            except Exception as e:
                print(f"[Scheduler] Error: {e}")

task_scheduler = TaskScheduler()

# -- Vocal Response System ---------------------------------------------------
def speak(text, app_instance=None):
    # Respect explicit mute flag — voice_mode no longer gates TTS output
    if app_instance and getattr(app_instance, 'tts_muted', False):
        print("[TTS] Muted — skipping speech.")
        return
    if not text or not text.strip():
        print("[TTS] Empty text — skipping.")
        return
    # Self-heal: restart TTS thread if it crashed
    if _tts_thread is not None and not _tts_thread.is_alive():
        print("[TTS] ⚠️ Thread died — restarting...")
        _start_tts_thread()
    # Replace any pending-but-not-yet-started item with the newest response.
    # The item currently being synthesised/played is already dequeued and unaffected.
    while not _tts_queue.empty():
        try: _tts_queue.get_nowait()
        except queue.Empty: break
    _tts_queue.put((text, app_instance))

def mute_jarvis_instantly():
    while not _tts_queue.empty():
        try: _tts_queue.get_nowait()
        except queue.Empty: break
    _tts_stop_event.set()  # interrupt any active playback wait

# -- File engine -------------------------------------------------------------
class FileEngine:
    def __init__(self):
        self.folder = Path(config.get("watched_folder", "")) if config.get("watched_folder") else None

    def set_folder(self, path: str):
        self.folder = Path(path)
        config["watched_folder"] = path
        save_config(config)

    def _resolve(self, name: str) -> Path:
        """Resolve a filename relative to the watched folder and ensure directory safety.
        Automatically checks for correct file extensions if the user omits them.
        """
        p = Path(name)
        
        # If the path is absolute, handle it directly
        if p.is_absolute():
            resolved_path = p.resolve()
        elif self.folder:
            resolved_path = (self.folder / name).resolve()
        else:
            resolved_path = p.resolve()

        # SMART EXTENSION DETECTOR: If the file doesn't exist and has no extension (or user omitted it)
        if not resolved_path.exists() and resolved_path.suffix == "":
            if self.folder and self.folder.exists():
                # Scan the folder for a file that matches the name regardless of its extension
                base_name = resolved_path.name.lower()
                for item in self.folder.iterdir():
                    if item.is_file() and item.stem.lower() == base_name:
                        print(f"🎯 Smart Extension Match Found: Automatically detected '{item.suffix}' for {item.name}")
                        return item.resolve()
                        
        return resolved_path

    def list_files(self, subdir: str = "") -> str:
        root = self._resolve(subdir) if subdir else self.folder
        if not root or not root.exists():
            return "❌ Watched folder not set or doesn't exist."
        items = sorted(root.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
        lines = []
        for item in items:
            prefix = "📁" if item.is_dir() else "📄"
            size = f"  ({item.stat().st_size:,} bytes)" if item.is_file() else ""
            lines.append(f"  {prefix} {item.name}{size}")
        return f"📂 Contents of {root}:\n" + ("\n".join(lines) if lines else "  (empty)")

    def move_file(self, src: str, dst: str) -> str:
        s = self._resolve(src)
        d = self._resolve(dst)
        if not s.exists():
            return f"❌ Source not found: {s}"
        if d.is_dir():
            d = d / s.name
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            if s.is_file():
                cmd = ["robocopy", str(s.parent), str(d.parent), s.name, "/MOV", "/MT:32", "/R:1", "/W:1"]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode <= 7: return f"✅ High-speed move complete: {s.name} → {d}"
            elif s.is_dir():
                cmd = ["robocopy", str(s), str(d), "/MOVE", "/E", "/MT:32", "/R:1", "/W:1"]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"✅ High-speed folder move complete: {d}"
        except Exception:
            shutil.move(str(s), str(d))
            return f"✅ Moved {s.name} → {d} (Standard I/O fallback)"

    def copy_file(self, src: str, dst: str) -> str:
        s = self._resolve(src)
        d = self._resolve(dst)
        if not s.exists():
            return f"❌ Source not found: {s}"
        if d.is_dir():
            d = d / s.name
        d.parent.mkdir(parents=True, exist_ok=True)
        try:
            if s.is_file():
                cmd = ["robocopy", str(s.parent), str(d.parent), s.name, "/MT:32", "/R:1", "/W:1"]
                result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                if result.returncode <= 7: return f"✅ High-speed copy complete: {s.name} → {d}"
            elif s.is_dir():
                cmd = ["robocopy", str(s), str(d), "/E", "/MT:32", "/R:1", "/W:1"]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return f"✅ High-speed folder copy complete: {d}"
        except Exception:
            shutil.copy2(str(s), str(d))
            return f"✅ Copied {s.name} → {d} (Standard I/O fallback)"

    def delete_file(self, name: str) -> str:
        p = self._resolve(name)
        if not p.exists(): return f"❌ Not found: {p}"
        if p.is_dir(): shutil.rmtree(p)
        else: p.unlink()
        return f"🗑️ Deleted {p.name}"

    def rename_file(self, src: str, new_name: str) -> str:
        s = self._resolve(src)
        if not s.exists(): return f"❌ Not found: {s}"
        clean_name = Path(new_name).name
        d = s.parent / clean_name
        s.rename(d)
        return f"✅ Renamed {s.name} → {clean_name}"

    def create_folder(self, name: str) -> str:
        p = self._resolve(name)
        p.mkdir(parents=True, exist_ok=True)
        return f"✅ Created folder: {p}"

    def write_file(self, name: str, content: str) -> str:
        p = self._resolve(name)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"✅ Data dump successfully written to: {p.name}"
        except Exception as e:
            return f"❌ Failed to write file data: {e}"

    def read_file(self, name: str) -> str:
        p = self._resolve(name)
        if not p.exists(): return f"❌ File not found: {p}"
        if p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(p)
                text_content = [page.extract_text() for page in reader.pages if page.extract_text()]
                combined_text = "\n".join(text_content)
                return f"[Content of PDF: {p.name}]\n{combined_text[:100000]}" if combined_text.strip() else f"⚠️ Empty PDF: {p.name}"
            except Exception as e: return f"❌ Could not parse PDF: {e}"
        if p.stat().st_size > 500_000: return "⚠️ File too large to read."
        try: return p.read_text(encoding="utf-8", errors="replace")
        except Exception as e: return f"❌ Could not read file: {e}"

    def search_files(self, pattern: str) -> str:
        clean = pattern.replace("*", "").strip()
        if not clean:
            return "❓ Search query blank."
        words = [w.lower() for w in clean.split() if len(w) > 1]
        home = Path.home()
        search_dirs = []
        for name in ("Desktop", "Downloads", "Documents"):
            p = home / name
            if p.exists():
                search_dirs.append(p)
        for od in (home / "OneDrive" / "Desktop", home / "OneDrive"):
            if od.exists():
                search_dirs.append(od)
                break
        if self.folder and self.folder.exists():
            search_dirs.append(self.folder)
        seen, matches = set(), []
        for d in search_dirs:
            try:
                for f in d.rglob("*"):
                    if not f.is_file() or str(f) in seen:
                        continue
                    nl = f.name.lower()
                    if any(w in nl for w in words):
                        seen.add(str(f))
                        matches.append(f)
                        if len(matches) >= 30:
                            break
            except (PermissionError, OSError):
                pass
        if not matches:
            return f"🔍 No files matching '{clean}' found in Desktop, Downloads, Documents, or watched folder."
        lines = [f"  📄 {m}" for m in matches[:20]]
        return f"🔍 Found {len(matches)} match(es):\n" + "\n".join(lines)

    def get_folder_summary(self) -> str:
        if not self.folder or not self.folder.exists(): return "No watched folder set."
        files = list(self.folder.rglob("*"))
        total = len([f for f in files if f.is_file()])
        exts = {}
        for f in files:
            if f.is_file(): exts[f.suffix.lower()] = exts.get(f.suffix.lower(), 0) + 1
        ext_summary = ", ".join(f"{e or 'no ext'}×{c}" for e, c in sorted(exts.items(), key=lambda x: -x[1])[:8])
        return f"Watched folder: {self.folder}\nTotal files: {total}\nTypes: {ext_summary}"

    def open_application(self, app_name: str) -> str:
        app = app_name.lower().strip()
        try:
            _browser_kw = {
                "chrome", "browser", "internet", "web", "online", "website", "site",
                "google", "search", "chatgpt", "gpt", "openai", "chat gpt",
                "youtube", "yt", "video", "videos", "watch",
                "docs", "google docs", "document", "spreadsheet", "sheets",
                "gmail", "email", "mail", "inbox",
                "reddit", "twitch", "stream", "streaming",
                "facebook", "instagram", "twitter", "x.com",
                "navigate to", "go to", "open up", "pull up",
            }
            if any(kw in app for kw in _browser_kw):
                url_map = {
                    "chatgpt": "https://chat.openai.com",
                    "gpt": "https://chat.openai.com",
                    "openai": "https://chat.openai.com",
                    "youtube": "https://www.youtube.com",
                    "yt": "https://www.youtube.com",
                    "docs": "https://docs.google.com",
                    "sheets": "https://sheets.google.com",
                    "gmail": "https://mail.google.com",
                    "mail": "https://mail.google.com",
                    "reddit": "https://www.reddit.com",
                    "twitch": "https://www.twitch.tv",
                    "facebook": "https://www.facebook.com",
                    "instagram": "https://www.instagram.com",
                    "twitter": "https://www.twitter.com",
                }
                url = next((v for k, v in url_map.items() if k in app), "https://www.google.com")
                return run_browser_action("browse_navigate", {"url": url})
            elif "explorer" in app or "files" in app:
                subprocess.Popen(["explorer", str(self.folder) if self.folder else "explorer.exe"])
                return "📁 Opening File Explorer..."
            elif "notepad" in app: subprocess.Popen(["notepad.exe"])
            elif "calculator" in app or "calc" in app: subprocess.Popen(["calc.exe"])
            else:
                shortcut = _find_app_shortcut(app)
                if shortcut:
                    try:
                        os.startfile(shortcut)
                        return f"🚀 Launched: {Path(shortcut).stem}"
                    except Exception as e:
                        return f"❌ Found but couldn't launch '{Path(shortcut).stem}': {e}"
                subprocess.Popen(f'start "" "{app_name}"', shell=True)
            return f"🚀 Launched shortcut: '{app}'"
        except Exception as e: return f"❌ Failed to launch: {e}"

    def run_system_command(self, command: str) -> str:
        try:
            result = subprocess.run(["cmd", "/c", command], capture_output=True, text=True, timeout=15, shell=True)
            output = result.stdout.strip() or result.stderr.strip()
            return f"✅ Executed Output:\n{output}" if result.returncode == 0 else f"⚠️ Error {result.returncode}:\n{output}"
        except Exception as e: return f"❌ Failed: {e}"

def _find_app_shortcut(name: str) -> str:
    """Fuzzy-search Start Menu and Desktop for a .lnk or .exe matching name."""
    name_lower = name.lower()
    search_dirs = [
        Path(os.environ.get("APPDATA", "")) / "Microsoft/Windows/Start Menu",
        Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft/Windows/Start Menu",
        Path.home() / "Desktop",
        Path(os.environ.get("PUBLIC", "C:\\Users\\Public")) / "Desktop",
    ]
    for d in search_dirs:
        if not d.exists():
            continue
        for ext in ("*.lnk", "*.exe"):
            for p in d.rglob(ext):
                if name_lower in p.stem.lower():
                    return str(p)
    return ""

file_engine = FileEngine()

# -- Browser Agent (Playwright Custom Directory Overhaul) -------------------
PLAYWRIGHT_OK = False
try:
    from playwright.sync_api import sync_playwright
    PLAYWRIGHT_OK = True
except ImportError:
    pass

JARVIS_BROWSER_DATA = str(Path.home() / "AppData" / "Local" / "JarvisBrowser" / "User Data")
_DEFAULT_PROFILE = "Default"

class BrowserAgent:
    def __init__(self, profile: str = _DEFAULT_PROFILE):
        self.profile = profile
        self._pw = None
        self._browser = None
        self._page = None
        self.task_queue = queue.Queue()
        self.browser_thread = None

    def start_worker(self):
        if self.browser_thread and self.browser_thread.is_alive(): return
        self.browser_thread = threading.Thread(target=self._browser_loop, daemon=True)
        self.browser_thread.start()

    def _browser_loop(self):
        try:
            self._start()
        except Exception as e:
            print(f"[Browser] Startup failed: {e}")
            # Drain any tasks already queued so execute() doesn't hang
            time.sleep(0.2)
            while True:
                try:
                    _, _, rq = self.task_queue.get_nowait()
                    rq.put(f"❌ Browser failed to start: {e}")
                except queue.Empty:
                    break
            return

        while True:
            func, args, result_queue = self.task_queue.get()
            try:
                result = func(*args)
            except Exception:
                try:
                    self.close()
                    self._start()
                    result = func(*args)
                except Exception as e:
                    result_queue.put(f"❌ Browser error: {e}")
                    continue
            result_queue.put(result)

    def execute(self, func, *args):
        self.start_worker()
        result_queue = queue.Queue()
        self.task_queue.put((func, args, result_queue))
        try:
            return result_queue.get(timeout=90)
        except queue.Empty:
            return "❌ Browser timed out — it may have failed to start."

    def _start(self):
        if self._page:
            return
        if not PLAYWRIGHT_OK:
            raise RuntimeError("Playwright not installed.")
        Path(JARVIS_BROWSER_DATA).mkdir(parents=True, exist_ok=True)
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch_persistent_context(
            user_data_dir=JARVIS_BROWSER_DATA,
            headless=False,
            args=[
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars',
                '--no-first-run',
                '--no-default-browser-check',
            ],
        )
        pages = self._browser.pages
        self._page = pages[0] if pages else self._browser.new_page()

    def _ensure(self):
        if not self._browser or not self._page or self._page.is_closed():
            self.close()
            self._start()

    def navigate(self, url: str) -> str:
        self._ensure()
        if not url.startswith("http"): url = "https://" + url
        self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
        title = self._page.title().strip()
        if not title:
            from urllib.parse import urlparse
            title = urlparse(self._page.url).netloc
        return f"🌐 Opened: {title}"

    def search_web(self, query: str) -> str:
        self._ensure()
        try:
            self._page.goto("https://www.google.com", wait_until="domcontentloaded")
            search_box = self._page.locator('textarea[name="q"]').first
            search_box.fill(query)
            self._page.keyboard.press("Enter")
            self._page.wait_for_load_state("networkidle")
            return f"🌐 Searched Google for: {query}"
        except Exception as e: return f"❌ Search failed: {e}"

    _KEY_MAP = {
        "escape": "Escape", "esc": "Escape",
        "enter": "Enter", "return": "Enter",
        "tab": "Tab", "backspace": "Backspace",
        "delete": "Delete", "space": "Space",
        "up": "ArrowUp", "down": "ArrowDown",
        "left": "ArrowLeft", "right": "ArrowRight",
        "f5": "F5", "f11": "F11", "f12": "F12",
        "ctrl+a": "Control+a", "ctrl+c": "Control+c",
        "ctrl+v": "Control+v", "ctrl+z": "Control+z",
        "ctrl+s": "Control+s",
    }

    def press_key(self, key: str) -> str:
        self._ensure()
        try:
            normalized = self._KEY_MAP.get(key.lower().strip(), key)
            self._page.keyboard.press(normalized)
            return f"⌨️ Pressed: {normalized}"
        except Exception as e:
            return f"❌ Key press failed: {e}"

    _DOC_BODY_KEYWORDS = {
        # Body / content area
        "document body", "doc body", "page body", "body", "canvas",
        "the document", "the doc", "the page", "document area", "page area",
        "content area", "writing area", "text area", "typing area", "editor area",
        "the editor", "editor", "main area", "main section", "main content",
        "the content", "page content", "doc content", "text body",
        # Middle / center phrasing
        "middle", "the middle", "center", "the center",
        "middle of document", "middle of the document",
        "middle of page", "middle of the page",
        "center of document", "center of the document",
        "center of page", "center of the page",
        "in the document", "into the document", "in the doc",
        # Typing context phrasing
        "type here", "write here", "start typing", "begin typing",
        "where i write", "where text goes", "where i type",
    }

    def _click_document_body(self) -> str:
        """Click the canvas body of a document editor (Google Docs, etc.)."""
        try:
            # Google Docs canvas
            canvas = self._page.locator('.kix-canvas-tile-content').first
            if canvas.count() > 0:
                box = canvas.bounding_box()
                if box:
                    self._page.mouse.click(box['x'] + box['width'] / 2, box['y'] + box['height'] / 3)
                    time.sleep(0.3)
                    return "🖱️ Clicked document body"
            # Generic fallback: click viewport center
            vp = self._page.viewport_size or {"width": 800, "height": 600}
            self._page.mouse.click(vp['width'] / 2, vp['height'] / 2)
            time.sleep(0.3)
            return "🖱️ Clicked page center"
        except Exception as e:
            return f"❌ Document body click failed: {e}"

    def type_at_cursor(self, text: str) -> str:
        """Type at the current cursor position using keyboard events.
        Required for canvas editors like Google Docs where fill() won't work."""
        self._ensure()
        try:
            self._page.keyboard.type(text, delay=20)
            return "⌨️ Typed successfully"
        except Exception as e:
            return f"❌ Typing failed: {e}"

    def click(self, selector_or_text: str) -> str:
        self._ensure()
        target = selector_or_text.strip()
        target_lower = target.lower()

        # If target is a keyboard key name, press it instead of clicking
        if target_lower in self._KEY_MAP:
            return self.press_key(target)

        # Document body / canvas click — use mouse coordinates, not DOM
        if target_lower in self._DOC_BODY_KEYWORDS or "document body" in target_lower or "middle of" in target_lower:
            return self._click_document_body()

        def _try(loc):
            try:
                loc.scroll_into_view_if_needed(timeout=2000)
                loc.click(timeout=5000)
                return True
            except Exception:
                return False

        # Role-based strategies (most reliable for buttons/links)
        for role in ("button", "link", "tab", "menuitem", "option", "checkbox", "radio"):
            if _try(self._page.get_by_role(role, name=target, exact=False).first):
                return "🖱️ Clicked successfully"

        # Text-based (case-insensitive)
        if _try(self._page.get_by_text(target, exact=False).first):
            return "🖱️ Clicked successfully"

        # Label / placeholder
        if _try(self._page.get_by_label(target, exact=False).first):
            return "🖱️ Clicked successfully"

        # CSS/XPath selector as last resort
        try:
            if _try(self._page.locator(target).first):
                return "🖱️ Clicked successfully"
        except Exception:
            pass

        return f"❌ Could not find anything to click: {target}"

    def type_text(self, selector: str, text: str) -> str:
        self._ensure()
        # Build candidate selectors — start with what the LLM gave us, then smart fallbacks
        hints = selector.lower() if selector else ""
        candidates = []
        if selector and hints not in ("auto", ""):
            candidates.append(selector)
        if "email" in hints or not hints:
            candidates.append('input[type="email"]:visible')
        if "password" in hints or not hints:
            candidates.append('input[type="password"]:visible')
        if "search" in hints:
            candidates.append('input[type="search"]:visible')
            candidates.append('textarea[name="q"]:visible')
        candidates += [
            'input[type="text"]:visible',
            'input:not([type="hidden"]):not([type="submit"]):not([type="button"]):not([type="checkbox"]):visible',
            'textarea:visible',
            '[contenteditable="true"]:visible',
        ]
        for sel in candidates:
            try:
                loc = self._page.locator(sel).first
                if loc.count() > 0:
                    loc.click(timeout=2000)
                    loc.fill(text, timeout=5000)
                    return f"⌨️ Typed successfully"
            except Exception:
                continue
        return "❌ No visible text input found on the page"

    def get_page_text(self) -> str:
        self._ensure()
        try: return self._page.inner_text("body")[:3000]
        except Exception: return ""

    def get_page_context(self) -> str:
        """Return page title, URL, and visible interactive elements for LLM context."""
        try:
            if not self._page or self._page.is_closed():
                return ""
            title = self._page.title().strip()
            url = self._page.url
            elements = self._page.evaluate("""() => {
                const seen = new Set();
                const out = [];
                const nodes = document.querySelectorAll(
                    'button, a[href], input:not([type="hidden"]), textarea, select, ' +
                    '[role="button"], [role="link"], [role="tab"], [role="menuitem"], ' +
                    '[contenteditable="true"]'
                );
                for (const el of nodes) {
                    const r = el.getBoundingClientRect();
                    if (r.width < 1 || r.height < 1) continue;
                    const label = (
                        el.getAttribute('aria-label') ||
                        el.getAttribute('title') ||
                        el.getAttribute('placeholder') ||
                        el.textContent?.trim() ||
                        el.getAttribute('value') || ''
                    ).replace(/\\s+/g, ' ').trim().slice(0, 60);
                    if (!label || seen.has(label)) continue;
                    seen.add(label);
                    const tag = el.tagName.toLowerCase();
                    const type = el.getAttribute('type') || el.getAttribute('role') || '';
                    out.push(`[${tag}${type ? ':' + type : ''}] ${label}`);
                    if (out.length >= 40) break;
                }
                return out;
            }""")
            lines = [f"Browser page: {title}", f"URL: {url}", "Visible interactive elements:"]
            lines.extend(f"  {e}" for e in (elements or []))
            return "\n".join(lines)
        except Exception:
            return ""

    def screenshot(self, filename: str = "screenshot.png") -> str:
        self._ensure()
        p = Path.home() / "Desktop" / filename
        self._page.screenshot(path=str(p))
        return f"📸 Screenshot saved: {p}"

    def run_task(self, task: str) -> str:
        self._ensure()
        step_log = []
        for step_num in range(12):
            page_text = self.get_page_text()
            prompt = f"Task: {task}\nURL: {self._page.url}\nText: {page_text}\nLog: {step_log}\nNext action JSON:"
            raw = ask_local_ai(prompt)
            try:
                cmd_data = json.loads(raw.replace("```json", "").replace("```", "").strip())
            except Exception: break
            act = cmd_data.get("action")
            if act == "navigate": result = self.navigate(cmd_data.get("url", ""))
            elif act == "click": result = self.click(cmd_data.get("target", ""))
            elif act == "type": result = self.type_text(cmd_data.get("selector", "input"), cmd_data.get("text", ""))
            elif act == "done": return "\n".join(step_log) + f"\nSummary: {cmd_data.get('summary')}"
            else: break
            step_log.append(f"Step {step_num+1}: {result}")
        return "\n".join(step_log)

    def close(self):
        if self._browser: self._browser.close()
        if self._pw: self._pw.stop()
        self._page = self._browser = self._pw = None

browser_agent = BrowserAgent()

def run_browser_action(action: str, args: dict) -> str:
    try:
        if action == "browse_navigate": return browser_agent.execute(browser_agent.navigate, args.get("url", ""))
        elif action == "browse_search": return browser_agent.execute(browser_agent.search_web, args.get("query", ""))
        elif action == "browse_click": return browser_agent.execute(browser_agent.click, args.get("target", ""))
        elif action == "browse_type": return browser_agent.execute(browser_agent.type_text, args.get("selector", ""), args.get("text", ""))
        elif action == "browse_type_cursor": return browser_agent.execute(browser_agent.type_at_cursor, args.get("text", ""))
        elif action == "browse_screenshot": return browser_agent.execute(browser_agent.screenshot, args.get("filename", "screenshot.png"))
        elif action == "browse_read": return browser_agent.execute(browser_agent.get_page_text)
        elif action == "browse_key": return browser_agent.execute(browser_agent.press_key, args.get("key", ""))
        elif action == "browse_task": return browser_agent.execute(browser_agent.run_task, args.get("task", ""))
        return f"❌ Unknown browser action: {action}"
    except Exception as e: return f"❌ Browser error: {e}"

# -- Online check & web lookup -----------------------------------------------
def check_online() -> bool:
    if _offline_mode:
        return False
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except Exception:
        return False

def web_lookup(query: str) -> str:
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": 1, "skip_disambig": 1},
            timeout=8,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        data = r.json()
        abstract = data.get("AbstractText", "").strip()
        if abstract:
            return f"🔍 {abstract[:600]}"
        topics = [t.get("Text", "") for t in data.get("RelatedTopics", [])[:5] if isinstance(t, dict) and t.get("Text")]
        return ("🔍 " + " | ".join(topics[:3]))[:600] if topics else "🔍 No results found for that query."
    except Exception as e:
        return f"❌ Web lookup failed: {e}"

# -- Enhanced Web Search Tools -----------------------------------------------
def web_search_ddg(query: str) -> str:
    """Two-stage DDG search: JSON instant-answer API first, HTML scrape as fallback."""
    import re
    print(f"[WebSearch] Query: {query}")

    def _clean(s):
        s = re.sub(r"<[^>]+>", "", s)
        return re.sub(r"\s+", " ",
            s.replace("&amp;", "&").replace("&quot;", '"')
             .replace("&#x27;", "'").replace("&gt;", ">").replace("&lt;", "<")
        ).strip()

    # ── Stage 1: DDG JSON instant-answer API ─────────────────────────────────
    try:
        r = requests.get(
            "https://api.duckduckgo.com/",
            params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            headers={"User-Agent": "Mozilla/5.0 (compatible; JarvisAI/1.0)"},
            timeout=10,
        )
        data = r.json()
        parts = []

        answer = data.get("Answer", "").strip()
        if answer:
            parts.append(f"✅ {_clean(answer)}")

        abstract = data.get("AbstractText", "").strip()
        src = data.get("AbstractSource", "")
        if abstract:
            prefix = f"{src}: " if src else ""
            parts.append(f"📌 {prefix}{abstract[:500]}")

        definition = data.get("Definition", "").strip()
        if definition and definition not in abstract:
            parts.append(f"📖 {definition[:300]}")

        topics = []
        for t in data.get("RelatedTopics", []):
            if isinstance(t, dict):
                if t.get("Text"):
                    topics.append(f"• {t['Text'][:180]}")
                elif t.get("Topics"):
                    for sub in t["Topics"][:2]:
                        if sub.get("Text"):
                            topics.append(f"• {sub['Text'][:180]}")
            if len(topics) >= 5:
                break
        if topics:
            parts.append("Related:\n" + "\n".join(topics[:5]))

        if parts:
            print(f"[WebSearch] DDG JSON: {len(parts)} section(s) found.")
            return f'🔍 Results for "{query}":\n' + "\n\n".join(parts)
    except Exception as e:
        print(f"[WebSearch] DDG JSON error: {e}")

    # ── Stage 2: DDG HTML scrape ──────────────────────────────────────────────
    print("[WebSearch] JSON empty — trying HTML scrape...")
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        })
        hr = session.post(
            "https://html.duckduckgo.com/html/",
            data={"q": query, "kl": "us-en"},
            timeout=12,
        )
        # DDG HTML structure uses <a class="result__a"> and <a class="result__snippet">
        titles   = re.findall(r'class="result__a"\s[^>]*>(.*?)</a>',   hr.text, re.DOTALL)
        snippets = re.findall(r'class="result__snippet"\s[^>]*>(.*?)</a>', hr.text, re.DOTALL)
        results  = []
        for t, s in zip(titles[:5], snippets[:5]):
            ct, cs = _clean(t), _clean(s)
            if ct and cs:
                results.append(f"• {ct}: {cs}")
        if results:
            print(f"[WebSearch] HTML scrape: {len(results)} results.")
            return f'🔍 Results for "{query}":\n' + "\n".join(results)
        print(f"[WebSearch] HTML scrape: 0 results. Response length={len(hr.text)}")
    except Exception as e:
        print(f"[WebSearch] HTML scrape error: {e}")

    return f'🔍 No results found for "{query}". Try more specific terms.'

def wiki_lookup(query: str) -> str:
    """Wikipedia REST API — fast, clean encyclopedia summaries."""
    print(f"[Wikipedia] Query: {query}")
    try:
        search_r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": query,
                    "format": "json", "srlimit": 1},
            timeout=8,
        )
        results = search_r.json().get("query", {}).get("search", [])
        if not results:
            return f"📖 No Wikipedia article found for '{query}'."
        title = results[0]["title"]
        summ_r = requests.get(
            f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
            timeout=8,
        )
        extract = summ_r.json().get("extract", "").strip()
        if extract:
            return f"📖 Wikipedia — {title}:\n{extract[:900]}"
        return f"📖 No summary available for '{title}'."
    except Exception as e:
        return f"❌ Wikipedia lookup failed: {e}"

def fetch_url(url: str) -> str:
    """Fetch any webpage and return its readable text content."""
    import re
    try:
        if not url.startswith("http"):
            url = "https://" + url
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        r = requests.get(url, headers=headers, timeout=14)
        text = r.text
        text = re.sub(r"<script[^>]*>.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>",  " ", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&[a-zA-Z#0-9]+;", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "⚠️ Could not extract readable text from that page."
        if len(text) > 4000:
            text = text[:4000] + "…[truncated]"
        return f"🌐 Content from {url}:\n{text}"
    except Exception as e:
        return f"❌ Could not fetch URL: {e}"

# -- Camera Engine (Vision / Face / Object Recognition) ---------------------

# Lazy-load DeepFace on first use (heavy library with pre-trained models)
_deepface_module = None

def _get_deepface():
    global _deepface_module
    if _deepface_module is None:
        print("[DeepFace] Loading face recognition models (first run ~1-2 min, then cached)...")
        import sys as _sys, io as _io
        _old_err = _sys.stderr
        _sys.stderr = _io.StringIO()
        try:
            import deepface.DeepFace as _df
            _deepface_module = _df
        finally:
            _sys.stderr = _old_err
        print("[DeepFace] ✅ Models loaded.")
    return _deepface_module

class CameraEngine:
    """Manages webcam capture, YOLO object detection, DeepFace analysis, and known-face recognition."""
    YOLO_MODEL = "yolov8n.pt"   # nano — fastest, ~6 MB download on first use

    def __init__(self):
        self._cap          = None
        self._running      = False
        self._thread       = None
        self._detect_thread = None
        self._frame        = None           # latest raw BGR frame
        self._lock         = threading.Lock()
        self._yolo         = None
        self._detect_mode  = "both"         # "objects" | "faces" | "both" | "none"
        self._ui_cb        = None           # called with PIL.Image for live display
        # Detection results written by _detection_loop, read by display loop
        self._det_lock     = threading.Lock()
        self._yolo_boxes   = []             # [(x1,y1,x2,y2,label,conf), ...]
        self._face_rects   = []             # [(x,y,w,h), ...]
        self._yolo_lock    = threading.Lock()  # prevents concurrent YOLO inference

    # ── Lifecycle ──────────────────────────────────────────────────────────
    def start(self, ui_callback=None) -> str:
        if self._running:
            return "📷 Camera is already running."
        try:
            import cv2
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                return "❌ Could not open camera. Make sure a webcam is connected."
            self._cap  = cap
            self._ui_cb = ui_callback
            self._running = True
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            self._detect_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self._detect_thread.start()
            return "📷 Camera started."
        except Exception as e:
            return f"❌ Camera error: {e}"

    def stop(self):
        self._running = False
        if self._cap:
            try: self._cap.release()
            except Exception: pass
            self._cap = None

    # ── Main capture loop ──────────────────────────────────────────────────
    def _loop(self):
        try:
            self._loop_inner()
        except Exception as e:
            print(f"[Camera] ❌ Loop crashed: {e}")
            import traceback; traceback.print_exc()

    def _loop_inner(self):
        """Display-only loop — runs at 30fps. Detection runs in _detection_loop thread."""
        import cv2
        from PIL import Image as _PILImg

        while self._running:
            ret, frame = self._cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self._lock:
                self._frame = frame.copy()

            if self._ui_cb:
                try:
                    display = cv2.resize(frame, (640, 480))
                    # Read latest detection results from the detection thread
                    with self._det_lock:
                        yolo_boxes = list(self._yolo_boxes)
                        face_rects = list(self._face_rects)
                    if self._detect_mode in ("objects", "both"):
                        for (x1, y1, x2, y2, label, conf) in yolo_boxes:
                            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 140, 0), 2)
                            cv2.putText(display, f"{label} {conf:.2f}", (x1, y1 - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 140, 0), 1)
                    if self._detect_mode in ("faces", "both"):
                        for rect in face_rects:
                            x, y, w, h = rect[0], rect[1], rect[2], rect[3]
                            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 220, 100), 2)
                            cv2.putText(display, "Face", (x, y - 6),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)
                    rgb  = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    pimg = _PILImg.fromarray(rgb)
                    self._ui_cb(pimg)
                except Exception:
                    pass

            time.sleep(0.033)   # ~30 fps

    def _detection_loop(self):
        """Runs YOLO + Haar detection as fast as hardware allows, independent of display."""
        import cv2
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        while self._running:
            if self._detect_mode == "none":
                time.sleep(0.1)
                continue

            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue

            new_yolo  = []
            new_faces = []

            if self._detect_mode in ("objects", "both") and self._yolo is not None:
                try:
                    with self._yolo_lock:
                        res = self._yolo(frame, verbose=False, conf=0.4, device='cpu')
                    for box in res[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        label = res[0].names[int(box.cls[0])]
                        conf  = float(box.conf[0])
                        new_yolo.append((x1, y1, x2, y2, label, conf))
                except Exception:
                    pass

            if self._detect_mode in ("faces", "both"):
                try:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                    new_faces = list(faces) if len(faces) > 0 else []
                except Exception:
                    pass

            with self._det_lock:
                self._yolo_boxes = new_yolo
                self._face_rects = new_faces

    # ── Helpers ────────────────────────────────────────────────────────────
    def _get_frame(self):
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def _load_yolo(self):
        if self._yolo is None:
            print("[Camera] Loading YOLOv8 nano model (downloads ~6 MB on first use)...")
            from ultralytics import YOLO
            self._yolo = YOLO(self.YOLO_MODEL)
            print("[Camera] YOLO ready.")

    def _frame_to_b64(self, frame) -> str:
        import cv2
        from PIL import Image as _PILImg
        from io import BytesIO
        rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img  = _PILImg.fromarray(rgb)
        img.thumbnail((1024, 768))
        buf  = BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    # ── Public actions ─────────────────────────────────────────────────────
    def describe_scene(self) -> str:
        frame = self._get_frame()
        if frame is None:
            return "❌ Camera not started. Open the camera window first."
        b64   = self._frame_to_b64(frame)
        model = _detect_vision_model()
        if not model:
            return "❌ No vision model found. Run: ollama pull moondream"
        desc  = _ask_vision(b64, model,
            "Describe everything visible in this image in full detail: people, objects, text, setting.",
            max_tokens=500)
        return f"👁️ {desc}" if desc else "❌ Vision model returned no description."

    def detect_objects(self) -> str:
        frame = self._get_frame()
        if frame is None:
            return "❌ Camera not started."
        try:
            self._load_yolo()
            with self._yolo_lock:
                results = self._yolo(frame, verbose=False, conf=0.35)
            names   = results[0].names
            boxes   = results[0].boxes
            if boxes is None or len(boxes) == 0:
                return "👁️ No objects detected in the current frame."
            counts = {}
            for box in boxes:
                label = names[int(box.cls[0])]
                counts[label] = counts.get(label, 0) + 1
            items = sorted(counts.items(), key=lambda x: -x[1])
            return "📦 Objects detected:\n" + "\n".join(f"  • {v}× {k}" for k, v in items)
        except Exception as e:
            return f"❌ Object detection error: {e}"

    def analyze_face(self) -> str:
        frame = self._get_frame()
        if frame is None:
            return "❌ Camera not started."
        try:
            DeepFace = _get_deepface()
            result = DeepFace.analyze(frame, actions=["emotion", "age", "gender"],
                                      enforce_detection=False, silent=True)
            if isinstance(result, list):
                result = result[0]
            emotion = result.get("dominant_emotion", "unknown")
            age     = result.get("age", "?")
            gender  = result.get("dominant_gender", result.get("gender", "?"))
            raw_emo = result.get("emotion", {})
            top3    = sorted(raw_emo.items(), key=lambda x: -x[1])[:3]
            emo_str = "  ".join(f"{e}: {v:.0f}%" for e, v in top3)
            return (f"😊 Face Analysis:\n"
                    f"  Estimated age : ~{age}\n"
                    f"  Gender        : {gender}\n"
                    f"  Mood          : {emotion}\n"
                    f"  Emotions      : {emo_str}")
        except Exception as e:
            return f"❌ Face analysis error: {e}"

    def remember_face(self, name: str) -> str:
        import cv2
        if not self._running:
            return "❌ Camera not started — open the camera window first."
        # Wait up to 2 seconds for the first frame to arrive after camera start
        deadline = time.time() + 2.0
        while time.time() < deadline:
            frame = self._get_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        else:
            return "❌ Camera started but no frame received yet — try again."
        person_dir = KNOWN_FACES_DIR / name.lower().replace(" ", "_")
        person_dir.mkdir(exist_ok=True)
        idx  = len(list(person_dir.glob("*.jpg"))) + 1
        path = person_dir / f"face_{idx}.jpg"
        cv2.imwrite(str(path), frame)
        return f"✅ Face saved for '{name}'. I'll recognize you next time."

    def identify_face(self) -> str:
        import cv2, tempfile
        if not self._running:
            return "❌ Camera not started — open the camera window first."
        deadline = time.time() + 2.0
        while time.time() < deadline:
            frame = self._get_frame()
            if frame is not None:
                break
            time.sleep(0.05)
        else:
            return "❌ Camera started but no frame received yet — try again."
        subdirs = [d for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()]
        if not subdirs:
            return "❌ No saved faces yet. Say 'remember my face as [name]' to train me."
        try:
            DeepFace = _get_deepface()
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
            cv2.imwrite(tmp_path, frame)
            result = DeepFace.find(img_path=tmp_path,
                                   db_path=str(KNOWN_FACES_DIR),
                                   enforce_detection=False, silent=True)
            try: os.unlink(tmp_path)
            except OSError: pass
            if result and len(result) > 0 and not result[0].empty:
                identity_path = result[0].iloc[0]["identity"]
                name = Path(identity_path).parent.name.replace("_", " ").title()
                return f"👤 I recognize: {name}"
            return "👤 Face not recognized — not in my saved faces."
        except Exception as e:
            return f"❌ Face identification error: {e}"

    def list_known_faces(self) -> str:
        names = [d.name.replace("_", " ").title()
                 for d in KNOWN_FACES_DIR.iterdir() if d.is_dir()]
        if names:
            return "👤 Known faces:\n" + "\n".join(f"  • {n}" for n in sorted(names))
        return "👤 No saved faces yet."

    def forget_face(self, name: str) -> str:
        import shutil
        person_dir = KNOWN_FACES_DIR / name.lower().replace(" ", "_")
        if person_dir.exists():
            shutil.rmtree(person_dir)
            return f"🗑️ Forgotten: {name}"
        return f"❌ No saved face for '{name}'."

camera_engine = CameraEngine()

# -- Screen Observer (Deep Think / Proactive Mode) ---------------------------

def _get_active_window_title() -> str:
    try:
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return ""


def _screenshot_b64() -> str:
    try:
        from PIL import ImageGrab
        img = ImageGrab.grab(all_screens=True)
        w, h = img.size
        scale = min(1920 / w, 1080 / h, 1.0)
        img = img.resize((int(w * scale), int(h * scale)))
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=60)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

def _detect_vision_model() -> str:
    try:
        r = requests.get(f"http://{_ollama_host}:11434/api/tags", timeout=3)
        for m in r.json().get("models", []):
            name = m.get("name", "")
            if any(vm in name for vm in ("llava", "moondream", "minicpm", "bakllava")):
                return name
    except Exception:
        pass
    return ""

_VISION_BACKGROUND_Q = (
    "Describe what is on screen in 2-3 sentences. "
    "Include the active application name, any visible text or document content, "
    "and what the user appears to be working on."
)
_VISION_DETAIL_Q = (
    "Describe everything visible on screen in full detail. "
    "Include: the active application, all readable text (especially document or file contents), "
    "any error messages, UI elements, and what the user is doing."
)

def _ask_vision(b64: str, model: str, question: str = _VISION_BACKGROUND_Q, max_tokens: int = 300) -> str:
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": question, "images": [b64]}],
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": max_tokens},
        }
        print(f"[Vision] Asking {model}...")
        r = requests.post(OLLAMA_URL, json=payload, timeout=90)
        data = r.json()
        content = data.get("message", {}).get("content", "").strip()
        if not content:
            err = data.get("error", "")
            print(f"[Vision] Empty response. Error: {err!r}  Full: {str(data)[:200]}")
        else:
            print(f"[Vision] {len(content)} chars received.")
        return content
    except Exception as e:
        print(f"[Vision] ❌ {e}")
        return ""

class ScreenObserver:
    INTERVAL = 5  # seconds between captures

    def __init__(self):
        self.active = False
        self._thread = None
        self._vision_model = ""
        self._observations: list = []

    def start(self, app_ref):
        if self.active:
            return
        self.active = True
        self._vision_model = _detect_vision_model()
        self._thread = threading.Thread(target=self._loop, args=(app_ref,), daemon=True)
        self._thread.start()

    def stop(self):
        self.active = False

    def get_context(self) -> str:
        if not self._observations:
            return ""
        lines = ["Screen activity (background observations):"] + [f"  - {o}" for o in self._observations[-4:]]
        return "\n".join(lines)

    def _capture(self) -> str:
        parts = []
        win = _get_active_window_title()
        if win:
            parts.append(f"Active window: {win}")
        if self._vision_model:
            b64 = _screenshot_b64()
            if b64:
                desc = _ask_vision(b64, self._vision_model)
                if desc:
                    parts.append(f"Screen: {desc}")
        else:
            try:
                import pytesseract
                from PIL import ImageGrab
                img = ImageGrab.grab(all_screens=True)
                w, h = img.size
                scale = min(1920 / w, 1080 / h, 1.0)
                img = img.resize((int(w * scale), int(h * scale)))
                text = pytesseract.image_to_string(img)
                lines = [l.strip() for l in text.splitlines() if len(l.strip()) > 8][:6]
                if lines:
                    parts.append("Screen text: " + " | ".join(lines))
            except Exception:
                pass
        return " | ".join(parts)

    def capture_now(self) -> str:
        obs = self._capture()
        if obs:
            self._observations.append(obs)
            if len(self._observations) > 12:
                self._observations.pop(0)
        return obs

    def capture_now_with_question(self, question: str = _VISION_DETAIL_Q) -> str:
        parts = []
        win = _get_active_window_title()
        if win:
            parts.append(f"Active window: {win}")
        if self._vision_model:
            b64 = _screenshot_b64()
            if b64:
                answer = _ask_vision(b64, self._vision_model, question, max_tokens=600)
                if answer:
                    parts.append(answer)
        obs = " | ".join(parts)
        if obs:
            self._observations.append(obs)
            if len(self._observations) > 12:
                self._observations.pop(0)
        return obs

    def _loop(self, app_ref):
        tick = 0
        while self.active:
            obs = self._capture()
            if obs:
                self._observations.append(obs)
                if len(self._observations) > 12:
                    self._observations.pop(0)
                tick += 1
                if tick % 8 == 0:
                    self._generate_thought(app_ref)
            time.sleep(self.INTERVAL)

    def _generate_thought(self, app_ref):
        ctx = self.get_context()
        if not ctx:
            return
        try:
            prompt = (
                f"You are Jarvis, silently observing Schmit's screen.\n{ctx}\n\n"
                "In plain English (NOT JSON), write 1-2 sentences: what Schmit is doing and one "
                "specific, useful observation or tip based on exactly what you can see. "
                "Be concrete — mention specific apps, file names, or text if visible. "
                "If there is truly nothing notable, reply with only a period."
            )
            payload = {
                "model": OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200},
            }
            r = requests.post(OLLAMA_URL, json=payload, timeout=20)
            thought = r.json().get("message", {}).get("content", "").strip()
            if thought and thought != "." and len(thought) > 8:
                if app_ref:
                    app_ref.root.after(0, lambda t=thought: app_ref._show_thought(t))
        except Exception:
            pass

screen_observer = ScreenObserver()

# -- Local LLM Prompt Engineering --------------------------------------------
def ask_local_ai(prompt: str, context: str = "") -> str:
    global CHAT_MEMORY
    system_prompt = """You are Jarvis, a highly sophisticated, intelligent file system and browser automation agent running on Windows.
You are speaking to your administrator, Schmit. Always address them or refer to them as "Schmit" when appropriate.

CRITICAL LANGUAGE REQUIREMENT: You must speak, think, and respond EXCLUSIVELY in English. Under no circumstances are you to output Chinese characters or any other language.

You MUST respond in JSON format ONLY. Do not include any text, code blocks, or explanations outside the raw JSON object.

Supported actions:
- list (Args: None)
- move (Args: "src", "dst")
- copy (Args: "src", "dst")
- delete (Args: "name")
- rename (Args: "src", "new_name")
- mkdir (Args: "name")
- read (Args: "name")
- write (Args: "name", "content")
- search (Args: "query")
- setfolder (Args: "path")
- open_app (Args: "app_name") — Use ONLY for non-browser desktop apps: notepad, calculator, explorer, etc. NEVER use open_app for Chrome, websites, or any URL. Use browse_navigate instead.
- run_cmd (Args: "command")
- system_info (Args: none) — CPU%, RAM%, disk, battery, uptime, network stats
- list_processes (Args: "sort_by" = "cpu" or "ram") — top running processes
- kill_process (Args: "name") — kill a process by name
- list_audio_devices (Args: none) — list all audio output devices with their index numbers
- set_audio_device (Args: "index") — switch audio output to device number N
- get_volume (Args: none) — get current system volume
- set_volume (Args: "level") — set system volume 0-100
- chat (Args: "message" - Use this for general logic, questions, or conversation)
- browse_navigate (Args: "url") — Use for ALL browser and website requests: "open Chrome", "go to X", "open ChatGPT", "open Google", etc. Always include https://.
- browse_search (Args: "query")
- browse_click (Args: "target") — target can be plain English like "Sign in", "Next", "Submit", "Accept all". Prefer visible button or link text over CSS selectors.
- browse_type (Args: "selector", "text") — selector can be "email", "password", "search", "auto", or a CSS selector. Use "auto" to type into the first visible input on the page. Do NOT use this for Google Docs document body.
- browse_type_cursor (Args: "text") — Type text at the current cursor position using raw keyboard events. Use this INSTEAD of browse_type whenever you are typing inside a document editor body (Google Docs, Word Online, etc.). It types wherever the cursor already is — no element needed.
- browse_screenshot (Args: "filename")
- browse_read (Args: none)
- browse_key (Args: "key") — Press a keyboard key: "Escape", "Enter", "Tab", "Backspace", "Delete", "ArrowUp", "ArrowDown", "ctrl+z", "ctrl+s", etc.
- browse_task (Args: "task")
- web_lookup (Args: "query") — Search the web for information. Only use when NOT in offline mode.
- schedule_task (Args: "description", "interval_minutes") — schedule a task to repeat every N minutes in the background. Use when Schmit says "every X minutes do Y", "remind me to check X", "repeat this task every N minutes".
- list_tasks (Args: none) — list all currently scheduled background tasks with their IDs and intervals.
- cancel_task (Args: "task_id") — cancel a scheduled task by its ID.
- remember (Args: "fact") — store a fact in long-term memory when Schmit says "remember that...", "don't forget...", or asks you to remember anything. Write the fact as a clear, self-contained sentence.
- look_at_screen (Args: "question") — takes a screenshot of ALL monitors, a vision model describes everything visible, then you receive that description and decide what action to take. Use whenever Schmit references something on screen: "click the next button", "look at my screen and do X", "read what's on screen", "click that button", etc. Set "question" to the full intent including any action (e.g. "Click the Next button if visible", "Read all text in the Notepad window", "Click the Sign In button on the page"). Only works when Screen Share is active. NEVER use file read/list/search for on-screen content.
- set_voice (Args: "voice_id") — change your own TTS voice. Available voice IDs: bm_george (British Male, default), bm_lewis (British Male), bf_emma (British Female), bf_isabella (British Female), am_adam (American Male), am_michael (American Male), af_bella (American Female), af_nicole (American Female), af_sarah (American Female), af_sky (American Female). Pick the closest match to what Schmit describes.
- web_search (Args: "query") — search the web via DuckDuckGo and return top results with titles and snippets. Use for any "search for", "look up", "find out about", "what is", "who is", or current-events questions. Faster than browser.
- wiki_lookup (Args: "query") — look up a topic on Wikipedia. Best for factual knowledge: people, places, history, science, concepts. Returns a clean encyclopedia summary.
- fetch_url (Args: "url") — download and read the full text of any webpage. Use when Schmit gives a specific URL to read, or when you need to read an article or page in full.
- add_script (Args: "trigger", "path") — save a script shortcut: when Schmit says [trigger], run the script at [path]. Supports .bat, .cmd, .ps1, .py files. Use when Schmit says "when I say X run Y", "save a script", or "add a shortcut".
- list_scripts (Args: none) — show all saved script triggers and their file paths.
- remove_script (Args: "trigger") — delete a saved script trigger mapping.
- run_script (Args: "trigger") — run a saved script by its trigger phrase.
- camera_describe (Args: none) — take a snapshot from the webcam and describe everything visible using the vision model. Use when Schmit asks "what do you see?", "describe my room", "look at the camera".
- camera_objects (Args: none) — detect and list all objects visible in the webcam frame using YOLO. Use when Schmit asks "what objects can you see?", "what's in front of me?", "detect objects".
- camera_face (Args: none) — analyze the face in the webcam frame: emotion, estimated age, gender. Use for "how do I look?", "what's my mood?", "analyze my face".
- camera_remember (Args: "name") — save the current face as a known person. Use when Schmit says "remember my face as [name]", "save my face", "learn who I am".
- camera_identify (Args: none) — identify who is in the webcam frame by comparing to saved faces. Use for "who am I?", "do you know me?", "recognize me".
- camera_faces_list (Args: none) — list all saved known faces.
- camera_forget (Args: "name") — delete a saved face by name.

Profile notes: "my profile", "your profile", "profile 19", "eve" all refer to Profile 19 (the main/default profile).
URL shortcuts: "chatgpt" → https://chat.openai.com, "youtube" → https://www.youtube.com, "google docs" or "docs" → https://docs.google.com, "gmail" → https://mail.google.com, "reddit" → https://www.reddit.com, "google" or "chrome" → https://www.google.com.

SPEECH DICTION: Schmit may dictate punctuation verbally. "dash" = "-", "dot" = ".", "slash" = "/", "underscore" = "_", "at" = "@", "hash" = "#", "colon" = ":". Example: "name dash test dot doc" = "name-test.doc". Apply this when interpreting file names, URLs, or any text to type.

BROWSER CONTEXT AWARENESS: When the context includes "Browser page:" and "Visible interactive elements:", you are looking at a live web page. Use the listed element labels to choose the correct browse_click or browse_type target. Always prefer clicking or typing using the exact label text shown in the elements list. If you need to dismiss something or cancel, use browse_key with "Escape".

GOOGLE DOCS / DOCUMENT EDITORS: The document body in Google Docs is a canvas — it is NOT a text input. Rules:
1. To rename/title the document: use browse_type with selector "title" or the visible title input label.
2. To type content INTO the document body: ALWAYS use browse_click with target "document body" first, then immediately follow with browse_type_cursor. Never use browse_type for document body content.
3. Never confuse the document title box (top of page, small input) with the document body (large canvas area in the center).
Example sequence for typing in a doc: [{"action":"browse_click","args":{"target":"document body"}},{"action":"browse_type_cursor","args":{"text":"Your text here"}}]

CRITICAL: Every response MUST be valid JSON with an "action" key.
For MULTI-STEP tasks (navigating + typing + submitting, etc.) emit MULTIPLE JSON objects in a single response, one per line — the executor runs them in order automatically. Example:
{"action": "browse_navigate", "args": {"url": "https://chat.openai.com"}}
{"action": "browse_click", "args": {"target": "Send a message"}}
{"action": "browse_type", "args": {"selector": "auto", "text": "hello"}}
{"action": "browse_key", "args": {"key": "Enter"}}
Always complete the FULL sequence for any browser task — never stop after one step and wait.
For greetings, chitchat, or anything not a file/browser operation, always use:
{"action": "chat", "args": {"message": "<your reply here as Jarvis>"}}
Never respond with plain text. Never omit the "action" key.

""" + (
    "OFFLINE MODE IS ON — do NOT use browse_*, web_lookup, look_at_screen, or any internet action. All local file/system actions (list, search, read, write, move, copy, delete, rename, mkdir, run_cmd, open_app, system_info, remember) work normally. Use them freely for local tasks."
    if _offline_mode else
    "ONLINE MODE — internet access is available. All actions including browse_* and web_lookup are available."
)

    try:
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(CHAT_MEMORY)
        messages.append(
            {
                "role": "user",
                "content": f"Current Folder Context:\n{context}\n\nUser Request: {prompt}",
            }
        )

        if _is_complex(prompt):
            model = REASON_MODEL
        elif _is_trivial(prompt):
            model = CHAT_MODEL
        else:
            model = OLLAMA_MODEL
        global _active_model
        _active_model = model
        print(f"🤖 Model: {model}")

        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "format": "json",
                "temperature": 0.0,
                "num_gpu": 999,      # force all layers onto VRAM
            },
        }
        def _call(mdl, timeout_s):
            global _inference_response
            payload["model"] = mdl
            r = requests.post(OLLAMA_URL, json=payload, timeout=timeout_s, stream=True)
            r.raise_for_status()
            _inference_response = r
            try:
                chunks = []
                for raw_line in r.iter_lines():
                    if _inference_stop.is_set():
                        r.close()
                        return None  # cancelled
                    if not raw_line:
                        continue
                    chunk = json.loads(raw_line)
                    delta = chunk.get("message", {}).get("content", "")
                    if delta:
                        chunks.append(delta)
                    if chunk.get("done"):
                        break
                return {"message": {"content": "".join(chunks)}}
            finally:
                _inference_response = None

        data = None
        fallback_notice = ""
        try:
            data = _call(model, 120)
        except Exception:
            if model != CHAT_MODEL:
                fallback_notice = f"⚠️ {model} timed out — used {CHAT_MODEL} instead."
                print(fallback_notice)
                _active_model = CHAT_MODEL
                if _ui_app is not None:
                    _ui_app.root.after(0, lambda: _ui_app.model_label.config(text=f"🤖 {CHAT_MODEL} (fallback)"))
                try:
                    data = _call(CHAT_MODEL, 60)
                except Exception:
                    return f'{{"action": "chat", "args": {{"message": "❌ Both models timed out. Try a shorter request."}}}}'
            else:
                return f'{{"action": "chat", "args": {{"message": "❌ Model timed out. Ollama may be busy — please try again."}}}}'

        if data is None:
            return f'{{"action": "chat", "args": {{"message": "⏹ Request cancelled."}}}}'

        if "message" in data:
            content = data["message"]["content"].strip()
            # Strip deepseek-r1's internal <think>...</think> reasoning block
            if "<think>" in content:
                import re
                content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
            if fallback_notice and _ui_app is not None:
                _ui_app.root.after(0, lambda n=fallback_notice: _ui_app._append_message("system", n))
            return content
        return str(data)
    except Exception as e:
        return f'{{"action": "chat", "args": {{"message": "❌ Local AI error: {str(e)}"}}}}'


def process_message(text: str) -> str:
    global CHAT_MEMORY, LAST_INTERACTION
    text = text.strip()
    if not text:
        return ""

    context = file_engine.get_folder_summary()

    # Always embed the query — needed for chat history retrieval + existing memory search
    _query_emb = get_embedding(text)

    # Inject semantically relevant explicit memories
    memories = get_relevant_memories(text, emb=_query_emb)
    if memories:
        context += "\n\n" + memories

    # Inject relevant playbooks (macro sequences for similar past tasks)
    playbooks = get_relevant_playbooks(text, emb=_query_emb)
    if playbooks:
        context += "\n\n" + playbooks

    # Inject learned patterns from past good interactions
    learned = get_learned_context(text, emb=_query_emb)
    if learned:
        context += "\n\n" + learned

    # Inject semantically relevant past chat history (top-3, capped at 3,200 chars)
    past = chat_db.get_relevant_history(_query_emb)
    if past:
        context += "\n\n" + past

    # Warn once per session when accumulated memory is getting large
    global _memory_warned
    if len(context) > 10_000 and not _memory_warned and _ui_app is not None:
        _memory_warned = True
        _ui_app.root.after(0, _ui_app._show_memory_warning)

    # Cap total injected context so the model input stays manageable
    if len(context) > 10_000:
        context = context[:10_000] + "\n[... context trimmed ...]"

    # Inject live browser page context only if a browser page is already open
    if browser_agent._page and not browser_agent._page.is_closed():
        try:
            page_ctx = browser_agent.execute(browser_agent.get_page_context)
            if page_ctx:
                context += "\n\n" + page_ctx
        except Exception:
            pass

    # Inject screen observations when Deep Think mode is active
    if screen_observer.active:
        screen_ctx = screen_observer.get_context()
        if screen_ctx:
            context += "\n\n" + screen_ctx

    # Script trigger — bypass LLM entirely for exact/partial phrase matches
    script_path = match_script(text)
    if script_path:
        result = run_script(script_path)
        LAST_INTERACTION["text"] = text
        LAST_INTERACTION["raw"] = result
        CHAT_MEMORY.append({"role": "user", "content": text[:600]})
        CHAT_MEMORY.append({"role": "assistant", "content": result[:600]})
        if len(CHAT_MEMORY) > 8:
            CHAT_MEMORY = CHAT_MEMORY[-8:]
        return result

    # Direct camera command intercepts — bypass LLM to avoid timeout on short phrases
    import re as _re
    _tl = text.lower().strip().strip("'\"")
    _rem = _re.search(r"remember (?:my face|me) as (.+)", _tl)
    if _rem:
        name = _rem.group(1).strip().strip("'\"")
        if _ui_app:
            _ui_app.root.after(0, lambda: _ui_app._ensure_camera_open())
        return camera_engine.remember_face(name)
    if _re.search(r"\b(who am i|recognize me|identify me|do you know me)\b", _tl):
        if _ui_app:
            _ui_app.root.after(0, lambda: _ui_app._ensure_camera_open())
        return camera_engine.identify_face()

    raw = ask_local_ai(text, context)

    # ADVANCED EXTRACTOR: Isolates true JSON boundaries even if Qwen speaks before the brackets
    raw_clean = raw.replace("```json", "").replace("```", "").strip()
    if "{" in raw_clean:
        raw_clean = raw_clean[raw_clean.find("{"):] # Crop off any prefix hallucinations instantly

    results = []
    decoder = json.JSONDecoder()
    pos = 0
    executed_any = False
    _session_actions: list = []  # track browse actions this turn for playbook saving

    while pos < len(raw_clean):
        while pos < len(raw_clean) and raw_clean[pos].isspace():
            pos += 1
        if pos >= len(raw_clean):
            break

        try:
            data, next_pos = decoder.raw_decode(raw_clean, pos)
            pos = next_pos
            action = data.get("action")
            args = data.get("args", {})
            if not isinstance(args, dict):
                args = {"message": str(args)} if args else {}

            if not action:
                fallback_text = (
                    args.get("message")
                    or data.get("message")
                    or data.get("response")
                    or data.get("text")
                    or data.get("content")
                )
                results.append(str(fallback_text) if fallback_text else "🤖 No response.")
                continue

            # Block all internet actions when offline mode is on
            _internet_actions = {"web_lookup", "browse_navigate", "browse_search", "browse_click",
                                  "browse_type", "browse_type_cursor", "browse_key", "browse_read",
                                  "browse_screenshot", "browse_task", "look_at_screen"}
            if _offline_mode and (action in _internet_actions or action.startswith("browse_")):
                # If it's a web_lookup and nothing else ran yet, redirect to local file search
                if action == "web_lookup" and not results:
                    query = args.get("query", text)
                    results.append(file_engine.search_files(query))
                elif not results:
                    # Only show the offline warning if no other action already produced output
                    results.append("📵 Offline mode is on — toggle it off to use internet features.")
                executed_any = True

            elif action == "chat":
                results.append(args.get("message", ""))
            elif action == "setfolder":
                path = args.get("path")
                if path:
                    file_engine.set_folder(path)
                    results.append(f"✅ Watching folder: {path}")
                executed_any = True
            elif action == "list":
                results.append(file_engine.list_files())
                executed_any = True
            elif action == "move":
                results.append(file_engine.move_file(args["src"], args["dst"]))
                executed_any = True
            elif action == "copy":
                results.append(file_engine.copy_file(args["src"], args["dst"]))
                executed_any = True
            elif action == "delete":
                results.append(file_engine.delete_file(args["name"]))
                executed_any = True
            elif action == "rename":
                results.append(file_engine.rename_file(args["src"], args["new_name"]))
                executed_any = True
            elif action == "mkdir":
                results.append(file_engine.create_folder(args["name"]))
                executed_any = True
            elif action == "read":
                results.append(file_engine.read_file(args["name"]))
                executed_any = True
            elif action == "write":
                results.append(file_engine.write_file(args["name"], args["content"]))
                executed_any = True
            elif action == "search":
                results.append(file_engine.search_files(args["query"]))
                executed_any = True
            elif action == "open_app":
                results.append(file_engine.open_application(args["app_name"]))
                executed_any = True
            elif action == "run_cmd":
                results.append(file_engine.run_system_command(args.get("command", "")))
                executed_any = True
            elif action == "system_info":
                results.append(get_system_info())
                executed_any = True
            elif action == "list_processes":
                results.append(list_processes(args.get("sort_by", "cpu")))
                executed_any = True
            elif action == "kill_process":
                results.append(kill_process_by_name(args["name"]))
                executed_any = True
            elif action == "list_audio_devices":
                results.append(list_audio_devices())
                executed_any = True
            elif action == "set_audio_device":
                results.append(set_audio_device(int(args["index"])))
                executed_any = True
            elif action == "get_volume":
                vol = get_volume()
                results.append(f"🔊 Current volume: {vol}%" if vol >= 0 else "❌ Could not read volume.")
                executed_any = True
            elif action == "set_volume":
                results.append(set_volume(int(args["level"])))
                executed_any = True
            elif action == "web_lookup":
                if check_online():
                    results.append(web_lookup(args.get("query", "")))
                else:
                    results.append("📡 Offline — cannot search the web right now.")
                executed_any = True
            elif action.startswith("browse_"):
                result = run_browser_action(action, args)
                results.append(result)
                _session_actions.append({"action": action, "args": args})
                executed_any = True
            elif action == "look_at_screen":
                if not screen_observer.active:
                    results.append("📺 Screen Share is not active — enable it first so I can see your screen.")
                else:
                    user_question = args.get("question", "Describe what is on screen and what Schmit is doing.")
                    # Stage 1: moondream — detailed visual description of every monitor
                    visual_desc = screen_observer.capture_now_with_question()
                    if visual_desc:
                        # Stage 2: qwen — decide what action to take based on the description
                        action_prompt = (
                            f"You are Jarvis. Schmit asked: \"{user_question}\"\n\n"
                            f"A vision model just described everything currently on screen:\n{visual_desc}\n\n"
                            "Based on what is visible and what Schmit asked, respond with the correct JSON action.\n"
                            "Available actions:\n"
                            "  browse_click (args: target) — click a button or link by its visible text or description\n"
                            "  browse_navigate (args: url) — open a URL\n"
                            "  browse_type (args: selector, text) — type into an input field\n"
                            "  browse_key (args: key) — press a keyboard key e.g. Enter, ctrl+t, Escape\n"
                            "  browse_read (no args) — read the current page text\n"
                            "  browse_search (args: query) — search Google\n"
                            "  run_cmd (args: command) — run a Windows shell command\n"
                            "  chat (args: message) — reply with a description or answer\n"
                            "RESPOND WITH A SINGLE JSON OBJECT ONLY. No plain text outside the JSON."
                        )
                        try:
                            r2 = requests.post(OLLAMA_URL, json={
                                "model": OLLAMA_MODEL,
                                "messages": [{"role": "user", "content": action_prompt}],
                                "stream": False,
                                "options": {"temperature": 0.1, "num_predict": 300},
                            }, timeout=25)
                            raw2 = r2.json().get("message", {}).get("content", "").strip()
                            # Parse and dispatch the sub-action
                            raw2 = raw2.replace("```json", "").replace("```", "").strip()
                            if "{" in raw2:
                                raw2 = raw2[raw2.find("{"):]
                            try:
                                sub = json.JSONDecoder().raw_decode(raw2)[0]
                                sub_action = sub.get("action", "")
                                sub_args = sub.get("args", {}) or {}
                                if sub_action == "chat":
                                    results.append(f"👁️ {sub_args.get('message', visual_desc)}")
                                elif sub_action.startswith("browse_"):
                                    results.append(run_browser_action(sub_action, sub_args))
                                elif sub_action == "run_cmd":
                                    results.append(file_engine.run_system_command(sub_args.get("command", "")))
                                else:
                                    results.append(f"👁️ {raw2 or visual_desc}")
                            except Exception:
                                results.append(f"👁️ {raw2 or visual_desc}")
                        except Exception:
                            results.append(f"👁️ {visual_desc}")
                    else:
                        results.append("👁️ I wasn't able to capture your screen right now.")
                executed_any = True
            elif action == "schedule_task":
                desc     = args.get("description", "").strip()
                interval = int(args.get("interval_minutes", 10))
                if desc:
                    tid = add_scheduled_task(desc, interval)
                    results.append(f"⏰ Scheduled: \"{desc}\" every {interval} min (ID: {tid})")
                else:
                    results.append("❌ No task description provided.")
                executed_any = True
            elif action == "list_tasks":
                tasks = load_scheduled_tasks()
                if tasks:
                    lines = ["⏰ Scheduled tasks:"]
                    for t in tasks:
                        lines.append(f"  [{t['id']}] every {t['interval_minutes']}min — {t['description'][:60]}")
                    results.append("\n".join(lines))
                else:
                    results.append("⏰ No scheduled tasks.")
                executed_any = True
            elif action == "cancel_task":
                tid = args.get("task_id", "").strip()
                if cancel_scheduled_task(tid):
                    results.append(f"✅ Cancelled task {tid}.")
                else:
                    results.append(f"❌ Task '{tid}' not found.")
                executed_any = True
            elif action == "remember":
                fact = args.get("fact", "").strip()
                if fact:
                    save_explicit_memory(fact)
                    results.append(f"🧠 Got it, I'll remember: {fact}")
                else:
                    results.append("❌ Nothing to remember — fact was empty.")
                executed_any = True
            elif action == "web_search":
                results.append(web_search_ddg(args.get("query", "")))
                executed_any = True
            elif action == "wiki_lookup":
                results.append(wiki_lookup(args.get("query", "")))
                executed_any = True
            elif action == "fetch_url":
                results.append(fetch_url(args.get("url", "")))
                executed_any = True
            elif action == "add_script":
                trigger = args.get("trigger", "").strip()
                path    = args.get("path", "").strip()
                if trigger and path:
                    results.append(save_script(trigger, path))
                else:
                    results.append("❌ Need both 'trigger' and 'path' to save a script.")
                executed_any = True
            elif action == "list_scripts":
                saved  = load_scripts()
                folder = get_folder_scripts()
                lines  = []
                if saved:
                    lines.append("📜 Saved triggers (memory/scripts.json):")
                    for t, p in saved.items():
                        lines.append(f"  • say \"{t}\" → {p}")
                if folder:
                    lines.append(f"📁 Drop-in scripts (scripts/ folder):")
                    for t, p in folder.items():
                        lines.append(f"  • say \"{t}\" → {Path(p).name}")
                if lines:
                    results.append("\n".join(lines))
                else:
                    results.append(f"📜 No scripts found.\n• Drop .bat/.ps1/.py files into the scripts/ folder — they're auto-detected.\n• Or say \"when I say X, run C:\\path\\to\\script.bat\" to map a trigger manually.")
                executed_any = True
            elif action == "remove_script":
                results.append(remove_script_entry(args.get("trigger", "")))
                executed_any = True
            elif action == "run_script":
                trigger = args.get("trigger", "").strip()
                scripts = load_scripts()
                path = scripts.get(trigger.lower())
                if path:
                    results.append(run_script(path))
                else:
                    results.append(f"❌ No script found for trigger '{trigger}'. Use 'list scripts' to see saved ones.")
                executed_any = True
            elif action == "camera_describe":
                results.append(camera_engine.describe_scene())
                executed_any = True
            elif action == "camera_objects":
                if _ui_app:
                    _ui_app.root.after(0, lambda: _ui_app._ensure_camera_open())
                results.append(camera_engine.detect_objects())
                executed_any = True
            elif action == "camera_face":
                if _ui_app:
                    _ui_app.root.after(0, lambda: _ui_app._ensure_camera_open())
                results.append(camera_engine.analyze_face())
                executed_any = True
            elif action == "camera_remember":
                name = args.get("name", "").strip()
                if name:
                    if _ui_app:
                        _ui_app.root.after(0, lambda: _ui_app._ensure_camera_open())
                    results.append(camera_engine.remember_face(name))
                else:
                    results.append("❌ Please provide a name: 'remember my face as [name]'")
                executed_any = True
            elif action == "camera_identify":
                if _ui_app:
                    _ui_app.root.after(0, lambda: _ui_app._ensure_camera_open())
                results.append(camera_engine.identify_face())
                executed_any = True
            elif action == "camera_faces_list":
                results.append(camera_engine.list_known_faces())
                executed_any = True
            elif action == "camera_forget":
                results.append(camera_engine.forget_face(args.get("name", "")))
                executed_any = True
            elif action == "set_voice":
                global _tts_voice
                requested = args.get("voice_id", "").strip().lower()
                all_ids = list(_KOKORO_VOICES.values())
                match = None
                if requested in all_ids:
                    match = requested
                if not match:
                    for vid in all_ids:
                        if requested in vid or vid in requested:
                            match = vid
                            break
                if not match:
                    for name, vid in _KOKORO_VOICES.items():
                        if requested in name.lower():
                            match = vid
                            break
                if match:
                    _tts_voice = match
                    config["tts_voice"] = match
                    save_config(config)
                    friendly = next(k for k, v in _KOKORO_VOICES.items() if v == match)
                    results.append(f"✅ Voice changed to {friendly}.")
                else:
                    options = ", ".join(_KOKORO_VOICES.keys())
                    results.append(f"❌ Couldn't match voice '{args.get('voice_id')}'. Options: {options}")
                executed_any = True
            else:
                results.append(f"❌ Unknown action: {action}")
        except json.JSONDecodeError:
            if not executed_any:
                # Try to salvage a "message" value from malformed JSON before
                # showing raw output (llama3.2 sometimes leaves unescaped quotes)
                import re as _re2
                _m = _re2.search(r'"message"\s*:\s*"((?:[^"\\]|\\.)*)"|"message"\s*:\s*\'((?:[^\'\\]|\\.)*)\'', raw_clean)
                if _m:
                    return (_m.group(1) or _m.group(2)).replace("\\n", "\n")
                return raw
            break

    # Save a playbook if ≥2 browser actions succeeded this turn
    if len(_session_actions) >= 2 and LAST_INTERACTION.get("text"):
        threading.Thread(
            target=save_playbook,
            args=(LAST_INTERACTION["text"], _session_actions),
            daemon=True
        ).start()

    final_output = "\n\n".join(results) if results else raw

    CHAT_MEMORY.append({"role": "user", "content": text[:600]})
    CHAT_MEMORY.append({"role": "assistant", "content": raw[:600]})
    if len(CHAT_MEMORY) > 8:
        CHAT_MEMORY = CHAT_MEMORY[-8:]

    LAST_INTERACTION["text"] = text
    LAST_INTERACTION["raw"] = raw

    # Persist exchange to chat history DB (background thread — never blocks UI)
    chat_db.save_exchange(text, final_output, _query_emb)

    return final_output

# -- Custom Tkinter Interface ------------------------------------------------
class AssistantApp:
    def __init__(self):
        self.root = None
        self.visible = False
        self._busy = False
        self._msg_queue: list[str] = []
        self._build_window()
        self._apply_startup_defaults()
    def _build_window(self):
        self.root = tk.Tk()
        self.root.title("Jarvis AI")
        self.root.geometry("1160x700")
        self.root.configure(bg="#0f0f16")
        self.root.protocol("WM_DELETE_WINDOW", self.hide)
        self.root.withdraw()
        self.main_frame = tk.Frame(self.root, bg="#0f0f16", bd=1, highlightbackground="#222235", highlightthickness=1)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        header = tk.Frame(self.main_frame, bg="#161623", height=54)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(header, text="🎩 JARVIS System Core", font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#161623").pack(side=tk.LEFT, padx=16, pady=14)
        folder_label = config.get("watched_folder", "") or "No folder set"
        self.folder_btn = tk.Button(header, text="📁 " + (Path(folder_label).name if folder_label != "No folder set" else "Select Directory"),
                                    font=("Segoe UI Semibold", 9), fg="#9a9ab0", bg="#1e1e30", bd=0, cursor="hand2", padx=10, pady=4, command=self.pick_folder)
        self.folder_btn.pack(side=tk.LEFT, padx=12, pady=10)
        tk.Button(header, text="✕", font=("Segoe UI", 12), fg="#666680", bg="#161623", bd=0, cursor="hand2", command=self.hide).pack(side=tk.RIGHT, padx=16)
        tk.Button(header, text="⚙", font=("Segoe UI", 13), fg="#666680", bg="#161623", bd=0, cursor="hand2", command=self.open_settings).pack(side=tk.RIGHT, padx=4)
        self.model_label = tk.Label(header, text=f"🤖 {_active_model}", font=("Segoe UI", 9), fg="#5c5c80", bg="#161623")
        self.model_label.pack(side=tk.RIGHT, padx=12)
        chat_container = tk.Frame(self.main_frame, bg="#0f0f16")
        chat_container.pack(fill=tk.BOTH, expand=True, padx=16, pady=(12, 0))
        self.chat = scrolledtext.ScrolledText(chat_container, wrap=tk.WORD, state=tk.DISABLED, bg="#06060a", fg="#d1d1e0", font=("Consolas", 11), insertbackground="#5c5cff", relief=tk.FLAT, padx=14, pady=14, highlightthickness=1, highlightbackground="#1c1c28")
        self.chat.pack(fill=tk.BOTH, expand=True)
        self.chat.tag_config("user", foreground="#5ce1e6", font=("Consolas", 11, "bold"))
        self.chat.tag_config("assistant", foreground="#e1e1ea")
        self.chat.tag_config("system", foreground="#5c5c70", font=("Consolas", 10, "italic"))
        self.chat.tag_config("thought", foreground="#7060a8", font=("Consolas", 10, "italic"))
        # Queue strip — hidden when empty, shows pending messages as pill chips
        self.queue_strip = tk.Frame(self.main_frame, bg="#0f0f16")
        self._queue_label = tk.Label(self.queue_strip, text="Queued:", font=("Segoe UI", 8), fg="#5c5c80", bg="#0f0f16")
        self._queue_label.pack(side=tk.LEFT, padx=(16, 6))
        self._queue_chips_frame = tk.Frame(self.queue_strip, bg="#0f0f16")
        self._queue_chips_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        # queue_strip is NOT packed here — it appears via _refresh_queue_strip when needed
        self.input_container = tk.Frame(self.main_frame, bg="#0f0f16")
        input_container = self.input_container
        input_container.pack(fill=tk.X, padx=16, pady=(12, 12))
        self.input_var = tk.StringVar()
        self.entry = tk.Entry(input_container, textvariable=self.input_var, font=("Segoe UI", 11), bg="#06060a", fg="#ffffff", insertbackground="#5c5cff", relief=tk.FLAT, highlightthickness=1, highlightbackground="#1c1c28")
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=10, padx=(0, 10))
        self.entry.bind("<Return>", self.send)
        self.entry.bind("<Escape>", lambda e: self.hide())
        self.send_btn = tk.Button(input_container, text="Send", font=("Segoe UI Semibold", 10), bg="#4747b2", fg="#ffffff", relief=tk.FLAT, cursor="hand2", padx=20, pady=8, command=self.send)
        self.send_btn.pack(side=tk.RIGHT)
        # -- Speech Framework Toggle Variable Configurations --
        self.mic_mode = "off"   # "off" | "auto" | "ptt"
        self.voice_mode = False  # True only when mic_mode == "auto"
        self.tts_muted = True   # voice muted by default
        self._ptt_active = False
        self.camera_btn = tk.Button(input_container, text="📷 Cam", font=("Segoe UI Semibold", 10), bg="#222235", fg="#9a9ab0", relief=tk.FLAT, cursor="hand2", padx=12, pady=8, command=self.toggle_camera)
        self.camera_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.websearch_btn = tk.Button(input_container, text="🔍", font=("Segoe UI Semibold", 10), bg="#0f2a3a", fg="#44bbee", relief=tk.FLAT, cursor="hand2", padx=12, pady=8, command=self.web_search_send)
        self.websearch_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.online_btn = tk.Button(input_container, text="🌐 Online", font=("Segoe UI Semibold", 10), bg="#1a3a1a", fg="#44cc66", relief=tk.FLAT, cursor="hand2", padx=12, pady=8, command=self.toggle_online_mode)
        self.online_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.think_btn = tk.Button(input_container, text="📺 Screen", font=("Segoe UI Semibold", 10), bg="#222235", fg="#9a9ab0", relief=tk.FLAT, cursor="hand2", padx=12, pady=8, command=self.toggle_think_mode)
        self.think_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.mute_btn = tk.Button(input_container, text="🔇 Muted", font=("Segoe UI Semibold", 10), bg="#222235", fg="#ff5555", relief=tk.FLAT, cursor="hand2", padx=12, pady=8, command=self.toggle_tts)
        self.mute_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.mic_btn = tk.Button(input_container, text="🎤 Off", font=("Segoe UI Semibold", 10), bg="#222235", fg="#9a9ab0", relief=tk.FLAT, cursor="hand2", padx=12, pady=8, command=self.toggle_voice_mode)
        self.mic_btn.pack(side=tk.RIGHT, padx=(0, 6))
        self.ptt_container = tk.Frame(self.main_frame, bg="#0f0f16")
        self.ptt_container.pack(fill=tk.X, padx=16, pady=0)
        self.ptt_btn = tk.Button(self.ptt_container, text="🎙  Hold to Talk — Release to Send",
                                  font=("Segoe UI Semibold", 11), bg="#7d47b2", fg="#ffffff",
                                  relief=tk.FLAT, cursor="hand2", padx=20, pady=10)
        self.ptt_btn.bind("<ButtonPress-1>", self._ptt_press)
        self.ptt_btn.bind("<ButtonRelease-1>", self._ptt_release)
        # ptt_btn starts hidden — shown only in push-to-talk mode

        feedback_bar = tk.Frame(self.main_frame, bg="#0f0f16")
        feedback_bar.pack(fill=tk.X, padx=16, pady=(0, 4))
        tk.Label(feedback_bar, text="Rate:", font=("Segoe UI", 8), fg="#404054", bg="#0f0f16").pack(side=tk.LEFT)
        tk.Label(feedback_bar, text="👎", font=("Segoe UI", 10), bg="#0f0f16", fg="#ff5555").pack(side=tk.LEFT, padx=(6, 2))
        _rating_colors = {
            -5: "#8b0000", -4: "#bb2222", -3: "#cc4444",
            -2: "#cc6633", -1: "#aa8833",
             0: "#444455",
             1: "#3388aa",  2: "#22aaaa",  3: "#11bb88",
             4: "#33ccbb",  5: "#5ce1e6",
        }
        for r in range(-5, 6):
            val = r
            tk.Button(
                feedback_bar, text=str(r), font=("Consolas", 8, "bold"),
                bg=_rating_colors[r], fg="#ffffff", bd=0, cursor="hand2",
                width=2, pady=1, relief=tk.FLAT,
                command=lambda v=val: self._give_feedback(v)
            ).pack(side=tk.LEFT, padx=1)
        tk.Label(feedback_bar, text="👍", font=("Segoe UI", 10), bg="#0f0f16", fg="#5ce1e6").pack(side=tk.LEFT, padx=(2, 0))
        self.status_var = tk.StringVar(value="System Active  •  Offline Whisper Array Enabled")
        tk.Label(self.main_frame, textvariable=self.status_var, font=("Segoe UI", 8, "italic"), fg="#404054", bg="#0f0f16", anchor="w").pack(fill=tk.X, padx=18, pady=(0, 8))
        self._append_message("system", "⚡ Local Jarvis Framework Initialized. Speech Processing shifted 100% locally via Faster-Whisper.")

    def _apply_startup_defaults(self):
        target_mode = config.get("default_mic_mode", "off")
        modes = ["off", "auto", "ptt"]
        if target_mode in modes and target_mode != "off":
            for _ in range(modes.index(target_mode)):
                self.toggle_voice_mode()
        if not config.get("tts_muted_default", True):
            self.toggle_tts()  # flip from default-muted to unmuted
        task_scheduler.start(self)

    def _show_memory_warning(self):
        win = tk.Toplevel(self.root)
        win.title("Memory Getting Large")
        win.configure(bg="#1a1a2e")
        win.resizable(False, False)
        win.grab_set()

        # Centre over the main window
        self.root.update_idletasks()
        rx, ry = self.root.winfo_x(), self.root.winfo_y()
        rw, rh = self.root.winfo_width(), self.root.winfo_height()
        win.geometry(f"440x260+{rx + rw//2 - 220}+{ry + rh//2 - 130}")

        tk.Label(win, text="⚠️  Memory Overload Warning",
                 font=("Segoe UI Semibold", 13), fg="#ffcc44", bg="#1a1a2e"
                 ).pack(pady=(22, 6))

        tk.Label(win,
                 text=(
                     "Your assistant's saved memory has grown large enough\n"
                     "to cause slow responses and unpredictable behaviour.\n\n"
                     "Clearing it now will keep Jarvis fast and stable.\n"
                     "You may want to copy anything important first."
                 ),
                 font=("Segoe UI", 10), fg="#c8c8e0", bg="#1a1a2e",
                 justify=tk.CENTER
                 ).pack(padx=24)

        btn_row = tk.Frame(win, bg="#1a1a2e")
        btn_row.pack(pady=22)

        def do_delete():
            global CHAT_MEMORY, _memory_warned, _deepface_module
            _wipe_all_memory()
            # Clear DeepFace cache from memory if loaded
            try:
                if _deepface_module is not None and hasattr(_deepface_module, 'representation_cache'):
                    _deepface_module.representation_cache.clear()
            except Exception:
                pass
            _deepface_module = None
            win.destroy()
            self._append_message("system", "🗑️ All memory cleared.")

        def do_keep():
            win.destroy()
            self._append_message("system",
                "⚠️ Memory kept. Performance may degrade over time.")

        tk.Button(btn_row, text="🗑️  Delete Memory",
                  font=("Segoe UI Semibold", 10), bg="#5a1a1a", fg="#ff8888",
                  relief=tk.FLAT, cursor="hand2", padx=16, pady=7,
                  command=do_delete
                  ).pack(side=tk.LEFT, padx=(0, 12))

        tk.Button(btn_row, text="Keep It for Now",
                  font=("Segoe UI", 10), bg="#222235", fg="#9a9ab0",
                  relief=tk.FLAT, cursor="hand2", padx=16, pady=7,
                  command=do_keep
                  ).pack(side=tk.LEFT)

    def _append_message(self, role: str, text: str):
        self.chat.config(state=tk.NORMAL)
        if role == "user":
            self.chat.insert(tk.END, "\nSchmit\n", "user")
            self.chat.insert(tk.END, text + "\n", "assistant")
        elif role == "assistant":
            self.chat.insert(tk.END, "\nJarvis\n", "user")
            self.chat.insert(tk.END, text + "\n", "assistant")
        elif role == "system":
            self.chat.insert(tk.END, text + "\n", "system")
        elif role == "thought":
            self.chat.insert(tk.END, text + "\n", "thought")
        self.chat.config(state=tk.DISABLED)
        self.chat.see(tk.END)

    def _show_thought(self, thought: str):
        self._append_message("thought", f"💭 {thought}")

    def toggle_think_mode(self):
        if not screen_observer.active:
            screen_observer.start(self)
            vm = screen_observer._vision_model
            vision_note = f" · Vision model: {vm}" if vm else " · Text/window mode (no vision model found in Ollama)"
            self.think_btn.config(text="📺 Screen ON", bg="#7d47b2", fg="#ffffff")
            self._append_message("system", f"📺 Screen Share active{vision_note} — watching your screen every {ScreenObserver.INTERVAL}s.")
        else:
            screen_observer.stop()
            self.think_btn.config(text="📺 Screen", bg="#222235", fg="#9a9ab0")
            self._append_message("system", "📺 Screen Share disabled.")

    def toggle_camera(self):
        if camera_engine._running:
            camera_engine.stop()
            self.camera_btn.config(text="📷 Cam", bg="#222235", fg="#9a9ab0")
            self._append_message("system", "📷 Camera stopped.")
            if hasattr(self, '_cam_window') and self._cam_window and self._cam_window.winfo_exists():
                self._cam_window.destroy()
            self._cam_window = None
        else:
            self._open_camera_window()

    def _ensure_camera_open(self):
        if not camera_engine._running:
            self._open_camera_window()

    def _open_camera_window(self):
        if hasattr(self, '_cam_window') and self._cam_window and self._cam_window.winfo_exists():
            self._cam_window.lift()
            return
        win = tk.Toplevel(self.root)
        win.title("Jarvis Camera")
        win.geometry("720x580")
        win.configure(bg="#0f0f16")
        self._cam_window = win

        # ── Controls bar ──────────────────────────────────────────────────
        ctrl = tk.Frame(win, bg="#161623", height=44)
        ctrl.pack(fill=tk.X)
        ctrl.pack_propagate(False)

        tk.Label(ctrl, text="Detect:", font=("Segoe UI", 9), fg="#666680", bg="#161623").pack(side=tk.LEFT, padx=(12, 4))
        mode_var = tk.StringVar(value="both")
        for label, val in [("Objects", "objects"), ("Faces", "faces"), ("Both", "both"), ("Off", "none")]:
            tk.Radiobutton(ctrl, text=label, variable=mode_var, value=val,
                           bg="#161623", fg="#9a9ab0", selectcolor="#222235",
                           activebackground="#161623", activeforeground="#fff",
                           font=("Segoe UI", 9),
                           command=lambda v=mode_var: setattr(camera_engine, '_detect_mode', v.get())
                           ).pack(side=tk.LEFT, padx=4)

        for btn_text, action in [("📸 Describe", "describe"), ("😊 Face", "face"), ("📦 Objects", "objects"), ("👤 Who?", "identify")]:
            tk.Button(ctrl, text=btn_text, font=("Segoe UI", 9), bg="#222235", fg="#9a9ab0",
                      relief=tk.FLAT, cursor="hand2", padx=8, pady=3,
                      command=lambda a=action: threading.Thread(target=self._cam_action, args=(a,), daemon=True).start()
                      ).pack(side=tk.RIGHT, padx=4, pady=6)

        # ── Live feed ─────────────────────────────────────────────────────
        self._cam_label = tk.Label(win, bg="#000000", cursor="crosshair")
        self._cam_label.pack(fill=tk.BOTH, expand=True)

        # ── Status bar ────────────────────────────────────────────────────
        self._cam_status_var = tk.StringVar(value="Starting camera…")
        tk.Label(win, textvariable=self._cam_status_var,
                 font=("Segoe UI", 8, "italic"), fg="#404054", bg="#0f0f16"
                 ).pack(pady=(2, 6))

        def on_close():
            camera_engine.stop()
            self.camera_btn.config(text="📷 Cam", bg="#222235", fg="#9a9ab0")
            self._cam_window = None
            win.destroy()

        win.protocol("WM_DELETE_WINDOW", on_close)

        _pending = [False]   # one-element list so the closure can mutate it

        def ui_callback(pil_img):
            # Called from camera background thread — do NO Tkinter work here.
            # ImageTk.PhotoImage must be created on the main thread; doing it
            # here (old code) caused random UI freezes and corruption.
            if not win.winfo_exists():
                return
            if _pending[0]:          # drop frame if a UI update is already queued
                return
            _pending[0] = True
            # Capture status text while still in background thread (cheap)
            status = f"Live  •  {pil_img.width}×{pil_img.height}  •  mode: {camera_engine._detect_mode}"
            def _update():
                _pending[0] = False
                if not win.winfo_exists():
                    return
                try:
                    from PIL import ImageTk
                    photo = ImageTk.PhotoImage(pil_img)   # main thread — safe
                    self._cam_label.config(image=photo)
                    self._cam_label.image = photo         # keep reference alive
                    self._cam_status_var.set(status)
                except Exception:
                    pass
            win.after(0, _update)

        result = camera_engine.start(ui_callback=ui_callback)
        self._append_message("system", result)
        self.camera_btn.config(text="📷 Cam ON", bg="#1a3a1a", fg="#44cc66")

    def _cam_action(self, action: str):
        if action == "describe":
            result = camera_engine.describe_scene()
        elif action == "face":
            result = camera_engine.analyze_face()
        elif action == "objects":
            result = camera_engine.detect_objects()
        elif action == "identify":
            result = camera_engine.identify_face()
        else:
            result = "Unknown camera action."
        self.root.after(0, lambda: self._append_message("assistant", result))
        speak(result, self)

    def toggle_online_mode(self):
        global _offline_mode
        _offline_mode = not _offline_mode
        if _offline_mode:
            self.online_btn.config(text="✈️ Offline", bg="#3a1a1a", fg="#ff7777")
            self.websearch_btn.config(bg="#2a1a0a", fg="#cc8833")
            self.websearch_btn.config(text="📁")
            self._append_message("system", "✈️ Offline mode ON — 🔍 button now searches your PC files.")
        else:
            self.online_btn.config(text="🌐 Online", bg="#1a3a1a", fg="#44cc66")
            self.websearch_btn.config(bg="#0f2a3a", fg="#44bbee")
            self.websearch_btn.config(text="🔍")
            self._append_message("system", "🌐 Online mode — 🔍 button now searches the web.")

    def web_search_send(self):
        if self._busy:
            return
        text = self.input_var.get().strip()
        if not text:
            hint = "💡 Type a filename or keyword, then click 🔍 to search your PC." if _offline_mode else "💡 Type something in the box first, then click 🔍 to search."
            self._append_message("system", hint)
            return
        self.input_var.set("")
        self._busy = True
        self.send_btn.config(text="Stop", bg="#aa2222", fg="#ffffff", command=self._cancel_request)
        self.websearch_btn.config(state=tk.DISABLED)

        if _offline_mode:
            self._append_message("user", f"📁 {text}")
            self.status_var.set("Searching your PC...")

            def run_local():
                result = file_engine.search_files(text)
                self.root.after(0, lambda: self._on_web_search_response(result))

            threading.Thread(target=run_local, daemon=True).start()
        else:
            self._append_message("user", f"🔍 {text}")
            self.status_var.set("Searching the web...")

            def run_web():
                ddg  = web_search_ddg(text)
                wiki = wiki_lookup(text)
                parts = [ddg]
                if wiki and "No Wikipedia article" not in wiki and "failed" not in wiki.lower():
                    parts.append(wiki)
                self.root.after(0, lambda: self._on_web_search_response("\n\n".join(parts)))

            threading.Thread(target=run_web, daemon=True).start()

    def _on_web_search_response(self, text: str):
        self._append_message("assistant", text)
        self.websearch_btn.config(state=tk.NORMAL)
        self.entry.focus()
        self._busy = False
        self.send_btn.config(text="Send", bg="#4747b2", fg="#ffffff", command=self.send)
        if self._msg_queue:
            next_msg = self._msg_queue.pop(0)
            self._refresh_queue_strip()
            self._start_request(next_msg)
        else:
            self.status_var.set("System Active")

    def send(self, event=None):
        text = self.input_var.get().strip()
        if not text:
            return
        self.input_var.set("")
        if self._busy:
            self._msg_queue.append(text)
            self._refresh_queue_strip()
            return
        self._start_request(text)

    def _start_request(self, text: str):
        self._busy = True
        _inference_stop.clear()
        self._append_message("user", text)
        self.status_var.set("Analyzing execution vectors...")
        self.send_btn.config(text="Stop", bg="#aa2222", fg="#ffffff", command=self._cancel_request)

        def run():
            response = process_message(text)
            self.root.after(0, lambda: self._on_response(response))

        threading.Thread(target=run, daemon=True).start()

    def _cancel_request(self):
        global _inference_response
        _inference_stop.set()
        if _inference_response is not None:
            try:
                _inference_response.close()
            except Exception:
                pass
        self.status_var.set("Cancelling...")

    def _refresh_queue_strip(self):
        for w in self._queue_chips_frame.winfo_children():
            w.destroy()
        if not self._msg_queue:
            self.queue_strip.pack_forget()
            return
        self.queue_strip.pack(fill=tk.X, padx=16, pady=(4, 0), before=self.input_container)
        for idx, msg in enumerate(self._msg_queue):
            i = idx
            chip = tk.Frame(self._queue_chips_frame, bg="#1c1c30", padx=0, pady=0)
            chip.pack(side=tk.LEFT, padx=(0, 4), pady=2)
            label_text = msg if len(msg) <= 28 else msg[:25] + "..."
            tk.Label(chip, text=label_text, font=("Segoe UI", 8), fg="#9090c0", bg="#1c1c30", padx=6, pady=2).pack(side=tk.LEFT)
            tk.Button(chip, text="×", font=("Segoe UI", 8, "bold"), fg="#666680", bg="#1c1c30",
                      bd=0, cursor="hand2", padx=4, pady=2,
                      command=lambda x=i: self._remove_queued(x)).pack(side=tk.LEFT, padx=(0, 3))

    def _remove_queued(self, idx: int):
        if 0 <= idx < len(self._msg_queue):
            self._msg_queue.pop(idx)
            self._refresh_queue_strip()

    def _on_response(self, text: str):
        self._append_message("assistant", text)
        self.model_label.config(text=f"🤖 {_active_model}")
        speak(text, self)
        self.entry.focus()
        self._busy = False
        self.send_btn.config(text="Send", bg="#4747b2", fg="#ffffff", command=self.send)
        if self._msg_queue:
            next_msg = self._msg_queue.pop(0)
            self._refresh_queue_strip()
            self._start_request(next_msg)
        else:
            self.status_var.set("System Active")

    def _give_feedback(self, rating: int):
        threading.Thread(target=save_feedback_entry, args=(rating,), daemon=True).start()
        if rating > 0:
            icon = "👍"
            desc = {5: "Perfect", 4: "Great", 3: "Good", 2: "Okay", 1: "Passable"}.get(rating, "Positive")
        elif rating < 0:
            icon = "👎"
            desc = {-1: "Off-target", -2: "Wrong", -3: "Bad", -4: "Very bad", -5: "Completely wrong"}.get(rating, "Negative")
        else:
            icon = "—"
            desc = "Neutral"
        sign = "+" if rating > 0 else ""
        self._append_message("system", f"{icon} Rating {sign}{rating} ({desc}) recorded.")
        self.status_var.set("Feedback saved")
        self.root.after(2000, lambda: self.status_var.set("System Active"))

    def toggle_tts(self):
        self.tts_muted = not self.tts_muted
        if self.tts_muted:
            self.mute_btn.config(text="🔇 Muted", bg="#222235", fg="#ff5555")
            mute_jarvis_instantly()
        else:
            _tts_stop_event.clear()
            self.mute_btn.config(text="🔊 Voice", bg="#1e1e30", fg="#5ce1e6")

    def toggle_voice_mode(self):
        _modes = ["off", "auto", "ptt"]
        self.mic_mode = _modes[(_modes.index(self.mic_mode) + 1) % len(_modes)]
        self.voice_mode = (self.mic_mode == "auto")
        if self.mic_mode == "off":
            self.mic_btn.config(text="🎤 Off", bg="#222235", fg="#9a9ab0")
            self.ptt_btn.pack_forget()
            self.status_var.set("System Active")
        elif self.mic_mode == "auto":
            self.mic_btn.config(text="🎤 Live", bg="#ff5555", fg="#ffffff")
            self.ptt_btn.pack_forget()
            self.status_var.set("🎤 Listening — speak anytime")
            threading.Thread(target=self.listen_to_schmit, daemon=True).start()
        elif self.mic_mode == "ptt":
            self.mic_btn.config(text="🎤 PTT", bg="#7d47b2", fg="#ffffff")
            self.ptt_btn.pack(fill=tk.X, pady=(4, 0))
            self.status_var.set("🎙 Push-to-Talk active — hold button to speak")

    def listen_to_schmit(self):
        """Continuous listening loop — every utterance is sent directly as a command."""
        import pyaudio
        import wave

        CHUNK = 1024
        RATE = 16000
        SPEECH_THRESHOLD = 500       # RMS level that counts as speech
        SILENCE_LIMIT = int(RATE / CHUNK * 1.2)  # ~1.2 s of silence ends an utterance
        PRE_ROLL = 8                 # chunks to keep before speech fires (captures word starts)

        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except Exception as init_err:
            print(f"Failed to initialize local microphone hardware: {init_err}")
            self.status_var.set("❌ Microphone Hardware Initialization Error")
            self.root.after(0, self.toggle_voice_mode)
            return

        self.status_var.set("🎤 Listening — speak anytime")
        rolling = []  # pre-roll buffer so word beginnings aren't clipped

        while self.mic_mode == "auto":
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception:
                time.sleep(0.01)
                continue

            rms = np.frombuffer(data, dtype=np.int16).std()
            rolling.append(data)
            if len(rolling) > PRE_ROLL:
                rolling.pop(0)

            if rms <= SPEECH_THRESHOLD:
                continue

            # Speech detected — record until silence
            self.status_var.set("👂 Hearing you...")
            frames = list(rolling)
            rolling.clear()
            silent_chunks = 0

            while self.mic_mode == "auto":
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                except Exception:
                    time.sleep(0.01)
                    continue
                frames.append(data)
                rms = np.frombuffer(data, dtype=np.int16).std()
                if rms > SPEECH_THRESHOLD:
                    silent_chunks = 0
                else:
                    silent_chunks += 1
                    if silent_chunks >= SILENCE_LIMIT:
                        break

            if self.mic_mode != "auto":
                break

            # Transcribe
            self.status_var.set("💭 Thinking...")
            tmp = f"cmd_{threading.get_ident()}.wav"
            wf = wave.open(tmp, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

            segments, _ = whisper_model.transcribe(tmp, beam_size=5, vad_filter=True, language="en")
            command_text = " ".join(seg.text for seg in segments).strip()
            try: os.remove(tmp)
            except OSError: pass

            if command_text and self.mic_mode == "auto":
                self.root.after(0, lambda cmd=command_text: self.execute_voice_command(cmd))
                while self.send_btn['state'] == tk.DISABLED and self.mic_mode == "auto":
                    time.sleep(0.2)

            self.status_var.set("🎤 Listening — speak anytime")

        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            pass

    def execute_voice_command(self, text):
        if not text.strip() or self.mic_mode == "off":
            return
        self._append_message("user", text)
        self.send_btn.config(state=tk.DISABLED)

        def run():
            if self.mic_mode == "auto" and not self.voice_mode:
                # auto mode was switched off mid-processing
                self.root.after(0, lambda: self.send_btn.config(state=tk.NORMAL))
                return
            response = process_message(text)
            self.root.after(0, lambda: self._on_response(response))

        threading.Thread(target=run, daemon=True).start()

    def _ptt_press(self, event=None):
        if self.mic_mode != "ptt" or self._ptt_active:
            return
        self._ptt_active = True
        self.ptt_btn.config(text="🔴  Recording...", bg="#ff5555")
        self.status_var.set("🔴 Recording — release to send")
        threading.Thread(target=self._ptt_record, daemon=True).start()

    def _ptt_release(self, event=None):
        self._ptt_active = False
        if self.mic_mode == "ptt":
            self.ptt_btn.config(text="🎙  Hold to Talk — Release to Send", bg="#7d47b2")
            self.status_var.set("🎙 Push-to-Talk active — hold button to speak")

    def _ptt_record(self):
        import pyaudio
        import wave
        CHUNK = 1024
        RATE = 16000
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
        except Exception as e:
            print(f"[PTT] Mic init failed: {e}")
            self._ptt_active = False
            self.root.after(0, lambda: self.ptt_btn.config(text="🎙  Hold to Talk — Release to Send", bg="#7d47b2"))
            return

        frames = []
        while self._ptt_active:
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
            except Exception:
                break

        try:
            stream.stop_stream()
            stream.close()
            p.terminate()
        except Exception:
            pass

        if not frames:
            return

        self.root.after(0, lambda: self.status_var.set("💭 Transcribing..."))

        tmp = f"ptt_{threading.get_ident()}.wav"
        command_text = ""
        try:
            wf = wave.open(tmp, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            segments, _ = whisper_model.transcribe(tmp, beam_size=5, vad_filter=True, language="en")
            command_text = " ".join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"[PTT] Transcribe error: {e}")
        finally:
            try: os.remove(tmp)
            except OSError: pass

        if command_text and self.mic_mode == "ptt":
            self.root.after(0, lambda cmd=command_text: self.execute_voice_command(cmd))
        else:
            self.root.after(0, lambda: self.status_var.set("🎙 Push-to-Talk active — hold button to speak"))

    def pick_folder(self):
        folder = filedialog.askdirectory(title="Select Target Folder Context")
        if folder:
            file_engine.set_folder(folder)
            short = Path(folder).name
            self.folder_btn.config(text=f"📁 {short}")
            self._append_message("system", f"Context switched: {folder}")

    def open_settings(self):
        win = tk.Toplevel(self.root)
        win.title("Global Settings")
        win.configure(bg="#0f0f16")
        win.resizable(True, True)
        win.grab_set()

        main_layer = tk.Frame(win, bg="#0f0f16", bd=1, highlightbackground="#222235", highlightthickness=1)
        main_layer.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        main_layer.columnconfigure(1, weight=1)

        _lbl = lambda row, text: tk.Label(
            main_layer, text=text, font=("Segoe UI", 10), fg="#9a9ab0", bg="#0f0f16"
        ).grid(row=row, column=0, padx=16, pady=10, sticky="e")

        # ── Row 0: Hotkey ──────────────────────────────────────────────────────
        _lbl(0, "Hotkey:")
        hk_var = tk.StringVar(value=config.get("hotkey", "ctrl+shift+space"))
        tk.Entry(main_layer, textvariable=hk_var, font=("Consolas", 10),
                 bg="#06060a", fg="#ffffff", insertbackground="#5c5cff",
                 relief=tk.FLAT, width=20, highlightbackground="#1c1c28", highlightthickness=1
                 ).grid(row=0, column=1, padx=6, pady=10, ipady=6, sticky="w")

        # ── Row 1: TTS Voice ───────────────────────────────────────────────────
        _lbl(1, "TTS Voice:")
        voice_names = list(_KOKORO_VOICES.keys())
        current_id = config.get("tts_voice", "bm_george")
        current_name = next((k for k, v in _KOKORO_VOICES.items() if v == current_id), voice_names[0])
        voice_var = tk.StringVar(value=current_name)

        style = ttk.Style(win)
        style.theme_use("default")
        style.configure("Dark.TCombobox",
                        fieldbackground="#06060a", background="#1e1e30",
                        foreground="#ffffff", selectbackground="#222235",
                        selectforeground="#ffffff", arrowcolor="#9a9ab0")
        style.map("Dark.TCombobox", fieldbackground=[("readonly", "#06060a")])

        voice_combo = ttk.Combobox(main_layer, textvariable=voice_var, values=voice_names,
                                   state="readonly", width=32, font=("Segoe UI", 9),
                                   style="Dark.TCombobox")
        voice_combo.grid(row=1, column=1, padx=6, pady=10, sticky="w")

        # ── Row 2: Mic default ─────────────────────────────────────────────────
        _lbl(2, "Mic default:")
        mic_frame = tk.Frame(main_layer, bg="#0f0f16")
        mic_frame.grid(row=2, column=1, padx=6, pady=10, sticky="w")
        mic_mode_var = tk.StringVar(value=config.get("default_mic_mode", "off"))
        for value, label in [("off", "Off"), ("auto", "Always On"), ("ptt", "Push-to-Talk")]:
            tk.Radiobutton(
                mic_frame, text=label, variable=mic_mode_var, value=value,
                bg="#0f0f16", fg="#9a9ab0", selectcolor="#222235",
                activebackground="#0f0f16", activeforeground="#ffffff",
                font=("Segoe UI", 9)
            ).pack(side=tk.LEFT, padx=(0, 14))

        # ── Row 3: Voice on startup ────────────────────────────────────────────
        _lbl(3, "Voice on startup:")
        voice_on_var = tk.BooleanVar(value=not config.get("tts_muted_default", True))
        tk.Checkbutton(
            main_layer, text="Enable Jarvis voice on startup",
            variable=voice_on_var,
            bg="#0f0f16", fg="#9a9ab0", selectcolor="#222235",
            activebackground="#0f0f16", activeforeground="#ffffff",
            font=("Segoe UI", 9)
        ).grid(row=3, column=1, padx=6, pady=10, sticky="w")

        # ── Row 4: Clear Memory ────────────────────────────────────────────────
        _lbl(4, "Memory:")

        def clear_memory():
            if not messagebox.askyesno(
                "Clear Memory",
                "Are you sure?\n\nThis will permanently delete all stored assistant memory and learned data.",
                icon="warning", parent=win
            ):
                return
            _wipe_all_memory()
            self._append_message("system", "🗑️ All memory cleared.")

        tk.Button(
            main_layer, text="🗑️ Clear All Memory",
            font=("Segoe UI", 9), bg="#3a1a1a", fg="#ff7777",
            relief=tk.FLAT, cursor="hand2", padx=10, pady=4,
            command=clear_memory
        ).grid(row=4, column=1, padx=6, pady=10, sticky="w")

        # ── Row 5: Ultra Deep Thinking ─────────────────────────────────────────
        _lbl(5, "AI Model Tier:")
        
        def configure_ultra_deep():
            ultra_win = tk.Toplevel(win)
            ultra_win.title("Ultra Deep Thinking Configuration")
            ultra_win.geometry("450x280")
            ultra_win.configure(bg="#0f0f16")
            ultra_win.resizable(False, False)
            ultra_win.grab_set()
            
            content = tk.Frame(ultra_win, bg="#0f0f16")
            content.pack(fill=tk.BOTH, expand=True, padx=16, pady=16)
            
            title_lbl = tk.Label(content, text="🧠 Ultra Deep Thinking Model", 
                                font=("Segoe UI", 12, "bold"), fg="#ffffff", bg="#0f0f16")
            title_lbl.pack(pady=(0, 12))
            
            model_frame = tk.Frame(content, bg="#1a1a2e", relief=tk.FLAT, bd=1, highlightbackground="#222235", highlightthickness=1)
            model_frame.pack(fill=tk.X, pady=8)
            
            tk.Label(model_frame, text="Model:", font=("Segoe UI", 10, "bold"), fg="#9a9ab0", bg="#1a1a2e").pack(anchor="w", padx=10, pady=6)
            tk.Label(model_frame, text="llama3.1:70b", font=("Consolas", 10), fg="#5cff5c", bg="#1a1a2e").pack(anchor="w", padx=20, pady=(0, 6))
            
            req_frame = tk.Frame(content, bg="#1a1a2e", relief=tk.FLAT, bd=1, highlightbackground="#222235", highlightthickness=1)
            req_frame.pack(fill=tk.X, pady=8)
            
            tk.Label(req_frame, text="⚠️  System Requirements:", font=("Segoe UI", 10, "bold"), fg="#ffaa00", bg="#1a1a2e").pack(anchor="w", padx=10, pady=(6, 0))
            tk.Label(req_frame, text="• RAM: ~64 GB minimum", font=("Segoe UI", 9), fg="#c8c8e0", bg="#1a1a2e").pack(anchor="w", padx=20, pady=2)
            tk.Label(req_frame, text="• Disk: ~50 GB for model download", font=("Segoe UI", 9), fg="#c8c8e0", bg="#1a1a2e").pack(anchor="w", padx=20, pady=2)
            tk.Label(req_frame, text="• Will require application restart", font=("Segoe UI", 9), fg="#c8c8e0", bg="#1a1a2e").pack(anchor="w", padx=20, pady=(2, 6))
            
            btn_frame = tk.Frame(content, bg="#0f0f16")
            btn_frame.pack(fill=tk.X, pady=12)
            
            current_enabled = config.get("ultra_deep_thinking", False)
            
            def enable_ultra_deep():
                config["ultra_deep_thinking"] = True
                save_config(config)
                ultra_win.destroy()
                messagebox.showinfo(
                    "Restart Required",
                    "Ultra Deep Thinking enabled!\n\n"
                    "Please restart the application to apply this change and start using llama3.1:70b for complex reasoning tasks.\n\n"
                    "Make sure your system has at least 64GB of available RAM.",
                    parent=win
                )
            
            def disable_ultra_deep():
                config["ultra_deep_thinking"] = False
                save_config(config)
                ultra_win.destroy()
                messagebox.showinfo(
                    "Ultra Deep Thinking Disabled",
                    "Ultra Deep Thinking has been disabled.\n\n"
                    "Complex reasoning will use the default deepseek-r1:8b model on restart.",
                    parent=win
                )
            
            if current_enabled:
                tk.Button(btn_frame, text="✓ Ultra Deep Thinking: ON",
                         font=("Segoe UI", 10, "bold"), bg="#1a5a1a", fg="#5cff5c",
                         relief=tk.FLAT, cursor="hand2", padx=12, pady=6,
                         command=disable_ultra_deep).pack(side=tk.LEFT, padx=4)
            else:
                tk.Button(btn_frame, text="Enable Ultra Deep Thinking",
                         font=("Segoe UI", 10), bg="#3a2a1a", fg="#ffaa00",
                         relief=tk.FLAT, cursor="hand2", padx=12, pady=6,
                         command=enable_ultra_deep).pack(side=tk.LEFT, padx=4)
            
            tk.Button(btn_frame, text="Close",
                     font=("Segoe UI", 9), bg="#222235", fg="#9a9ab0",
                     relief=tk.FLAT, cursor="hand2", padx=12, pady=6,
                     command=ultra_win.destroy).pack(side=tk.LEFT, padx=4)
        
        ultra_status = config.get("ultra_deep_thinking", False)
        ultra_text = "🧠 Ultra Deep (llama3.1:70b)" if ultra_status else "Standard (deepseek-r1:8b)"
        tk.Button(
            main_layer, text=ultra_text,
            font=("Segoe UI", 9), bg="#2a3a4a" if ultra_status else "#1a2a3a", fg="#5cff5c" if ultra_status else "#9a9ab0",
            relief=tk.FLAT, cursor="hand2", padx=10, pady=4,
            command=configure_ultra_deep
        ).grid(row=5, column=1, padx=6, pady=10, sticky="w")

        # ── Row 6: Ollama Hosts ────────────────────────────────────────────────
        tk.Label(main_layer, text="Ollama Hosts:", font=("Segoe UI", 10),
                 fg="#9a9ab0", bg="#0f0f16"
                 ).grid(row=6, column=0, padx=16, pady=(10, 0), sticky="ne")

        hosts_panel = tk.Frame(main_layer, bg="#0f0f16")
        hosts_panel.grid(row=6, column=1, padx=6, pady=(8, 4), sticky="ew")

        lb_wrap = tk.Frame(hosts_panel, bg="#0f0f16")
        lb_wrap.pack(fill=tk.X)

        hosts_lb = tk.Listbox(lb_wrap, font=("Consolas", 10), bg="#06060a", fg="#d1d1e0",
                              selectbackground="#222235", selectforeground="#5ce1e6",
                              relief=tk.FLAT, highlightbackground="#1c1c28", highlightthickness=1,
                              height=4, width=34)
        hosts_lb.pack(side=tk.LEFT, fill=tk.X, expand=True)
        _lb_sb = tk.Scrollbar(lb_wrap, orient=tk.VERTICAL, command=hosts_lb.yview)
        _lb_sb.pack(side=tk.RIGHT, fill=tk.Y)
        hosts_lb.config(yscrollcommand=_lb_sb.set)

        _all_hosts = list(config.get("ollama_hosts", [config.get("ollama_host", "localhost")]))

        def _refresh_lb():
            hosts_lb.delete(0, tk.END)
            active = config.get("ollama_host", "localhost")
            for h in _all_hosts:
                label = f"  [ACTIVE]  {h}" if h == active else f"           {h}"
                hosts_lb.insert(tk.END, label)

        _refresh_lb()

        action_row = tk.Frame(hosts_panel, bg="#0f0f16")
        action_row.pack(fill=tk.X, pady=(4, 0))

        def _set_active():
            sel = hosts_lb.curselection()
            if not sel:
                return
            config["ollama_host"] = _all_hosts[sel[0]]
            _refresh_lb()
            conn_lbl.config(text=f"Active: {config['ollama_host']} — press Save to apply.", fg="#5ce1e6")

        def _remove_host():
            sel = hosts_lb.curselection()
            if not sel or len(_all_hosts) <= 1:
                return
            idx = sel[0]
            removed = _all_hosts.pop(idx)
            if removed == config.get("ollama_host", "localhost"):
                config["ollama_host"] = _all_hosts[0]
            _refresh_lb()

        tk.Button(action_row, text="Set Active", font=("Segoe UI", 9),
                  bg="#1a2a4a", fg="#5c9aff", relief=tk.FLAT, cursor="hand2",
                  padx=6, pady=3, command=_set_active).pack(side=tk.LEFT, padx=(0, 4))
        tk.Button(action_row, text="Remove", font=("Segoe UI", 9),
                  bg="#3a1a1a", fg="#ff7777", relief=tk.FLAT, cursor="hand2",
                  padx=6, pady=3, command=_remove_host).pack(side=tk.LEFT)

        add_row = tk.Frame(hosts_panel, bg="#0f0f16")
        add_row.pack(fill=tk.X, pady=(6, 0))

        new_host_var = tk.StringVar()
        new_host_entry = tk.Entry(add_row, textvariable=new_host_var, font=("Consolas", 10),
                                  bg="#06060a", fg="#ffffff", insertbackground="#5c5cff",
                                  relief=tk.FLAT, width=18, highlightbackground="#1c1c28",
                                  highlightthickness=1)
        new_host_entry.pack(side=tk.LEFT, ipady=4, padx=(0, 4))

        def _add_host(event=None):
            h = new_host_var.get().strip()
            if h and h not in _all_hosts:
                _all_hosts.append(h)
                _refresh_lb()
            new_host_var.set("")

        new_host_entry.bind("<Return>", _add_host)
        tk.Button(add_row, text="+ Add", font=("Segoe UI", 9),
                  bg="#2a4a2a", fg="#5cff5c", relief=tk.FLAT, cursor="hand2",
                  padx=6, pady=3, command=_add_host).pack(side=tk.LEFT)

        test_row = tk.Frame(hosts_panel, bg="#0f0f16")
        test_row.pack(fill=tk.X, pady=(6, 0))

        conn_lbl = tk.Label(test_row, text="Select a host, then click Test.",
                            font=("Segoe UI", 8), fg="#5c5c70", bg="#0f0f16", anchor="w")
        conn_lbl.pack(side=tk.LEFT, fill=tk.X, expand=True)

        def _test_connection():
            sel = hosts_lb.curselection()
            if not sel:
                conn_lbl.config(text="Select a host first.", fg="#ffaa00")
                return
            host = _all_hosts[sel[0]]
            conn_lbl.config(text=f"Testing {host}...", fg="#9a9ab0")
            win.update_idletasks()

            def _do_test():
                try:
                    r = requests.get(f"http://{host}:11434/api/tags", timeout=5)
                    if r.status_code == 200:
                        models = [m.get("name", "") for m in r.json().get("models", [])]
                        snippet = ", ".join(models[:4]) or "(no models)"
                        win.after(0, lambda: conn_lbl.config(
                            text=f"Connected: {snippet}", fg="#5cff5c"))
                    else:
                        win.after(0, lambda: conn_lbl.config(
                            text=f"HTTP {r.status_code} from {host}", fg="#ffaa00"))
                except requests.exceptions.ConnectionError:
                    win.after(0, lambda: conn_lbl.config(
                        text=f"Unreachable: {host}", fg="#ff5555"))
                except requests.exceptions.Timeout:
                    win.after(0, lambda: conn_lbl.config(
                        text=f"Timed out: {host}", fg="#ff5555"))
                except Exception as exc:
                    win.after(0, lambda: conn_lbl.config(
                        text=f"Error: {exc}", fg="#ff5555"))

            threading.Thread(target=_do_test, daemon=True).start()

        tk.Button(test_row, text="Test Connection", font=("Segoe UI", 9),
                  bg="#0f2a3a", fg="#44bbee", relief=tk.FLAT, cursor="hand2",
                  padx=8, pady=3, command=_test_connection).pack(side=tk.RIGHT)

        # ── Save ───────────────────────────────────────────────────────────────
        def save():
            global _tts_voice, _ollama_host, OLLAMA_URL, OLLAMA_EMBED_URL
            config["hotkey"] = hk_var.get().strip()
            config["tts_voice"] = _KOKORO_VOICES.get(voice_var.get(), "bm_george")
            config["default_mic_mode"] = mic_mode_var.get()
            config["tts_muted_default"] = not voice_on_var.get()
            config["ollama_hosts"] = list(_all_hosts)
            if config.get("ollama_host") not in _all_hosts:
                config["ollama_host"] = _all_hosts[0] if _all_hosts else "localhost"
            _tts_voice = config["tts_voice"]
            _ollama_host = config["ollama_host"]
            OLLAMA_URL = f"http://{_ollama_host}:11434/api/chat"
            OLLAMA_EMBED_URL = f"http://{_ollama_host}:11434/api/embeddings"
            save_config(config)
            win.destroy()
            self._append_message("system",
                f"⚙️ Settings saved. Active Ollama host: {_ollama_host}. Voice change is live. Mic/startup defaults apply on next launch. Hotkey requires restart.")

        tk.Button(main_layer, text="Save State", font=("Segoe UI Semibold", 9),
                  bg="#4747b2", fg="white", relief=tk.FLAT, padx=16, pady=6,
                  cursor="hand2", command=save
                  ).grid(row=7, column=1, pady=14, padx=6, sticky="e")

        win.update_idletasks()
        win.geometry(f"540x{win.winfo_reqheight() + 12}")

    def show(self):
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()
        self.entry.focus()
        self.visible = True

    def hide(self):
        self.root.withdraw()
        self.visible = False

    def toggle(self):
        if self.visible:
            self.hide()
        else:
            self.show()

    def run(self):
        self.root.mainloop()


# -- Hotkey registration -----------------------------------------------------


def register_hotkey(app: AssistantApp):
    if not KEYBOARD_OK:
        print("⚠️ 'keyboard' library not installed — hotkey disabled. Run: pip install keyboard")
        return
    hotkey = config.get("hotkey", "ctrl+shift+space")
    try:
        keyboard.add_hotkey(hotkey, lambda: app.root.after(0, app.toggle))
        print(f"✅ Hotkey registered: {hotkey}")
    except Exception as e:
        print(f"⚠️ Could not register hotkey: {e}")


# -- System tray -------------------------------------------------------------


def make_tray_icon():
    img = Image.new("RGB", (64, 64), color="#161623")
    d = ImageDraw.Draw(img)
    d.ellipse([8, 8, 56, 56], fill="#4747b2", outline="#5c5cff", width=2)
    d.text((18, 18), "AI", fill="white")
    return img


def start_tray(app: AssistantApp):
    if not TRAY_OK:
        print("⚠️ pystray/Pillow not installed — system tray disabled.")
        return

    icon = pystray.Icon(
        "Jarvis",
        make_tray_icon(),
        "Jarvis",
        menu=pystray.Menu(
            pystray.MenuItem("Open", lambda: app.root.after(0, app.show), default=True),
            pystray.MenuItem("Quit", lambda: (icon.stop(), app.root.after(0, app.root.quit))),
        )
    )
    threading.Thread(target=icon.run, daemon=True).start()


# -- Entry point -------------------------------------------------------------


if __name__ == "__main__":
    app = AssistantApp()
    _ui_app = app
    register_hotkey(app)
    start_tray(app)

    _boot_ms = (_time.perf_counter() - _BOOT_START) * 1000
    print(f"Boot time: {_boot_ms / 1000:.3f}s ({_boot_ms:.0f}ms)")

    app.show()
    app.run()
