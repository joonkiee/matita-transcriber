#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matita Transcriber — a local whisper.cpp app for macOS

Install:     /usr/bin/python3 ~/Downloads/stt_app.py --install
             Runs quietly in the background (starts at login, no Terminal window)
             and puts a Matita app on your Desktop and in your Applications folder.
Uninstall:   /usr/bin/python3 ~/whisper-app/stt_app.py --uninstall

Everything runs on your Mac. Transcripts are saved to ~/Documents/Transcripts.
"""
import array
import base64
import hashlib
import io
import json
import os
import platform
import plistlib
import queue
import re
import shutil
import signal
import socket
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.parse
import uuid
import webbrowser
import zipfile
from xml.sax.saxutils import escape as xml_escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "3.21.0"
REPO = "joonkiee/matita-transcriber"   # GitHub "owner/name", used for update checks
# Set by Matita.app (the downloadable Mac app), which carries its own Python, whisper-cli, ffmpeg and
# the Matita Audio helper. Without it, Matita runs the older way: a LaunchAgent plus Homebrew tools.
BUNDLE = os.environ.get("MATITA_APP_BUNDLE") or None
BUNDLE_BIN = os.path.join(BUNDLE, "Contents", "Resources", "bin") if BUNDLE else None
NAME = "Matita"
PORT = 8765
HOME = os.path.expanduser("~")
MODEL_DIR = os.path.join(HOME, "whisper-models")
DEFAULT_MODEL = "ggml-large-v3-turbo.bin"
OUT_DIR = os.path.join(HOME, "Documents", "Transcripts")
APP_DIR = os.path.join(HOME, "whisper-app")
LABEL = "com.local.matita"
OLD_LABELS = ["com.local.transcriber"]          # the app was called Transcriber before 3.0
AGENT_PLIST = os.path.join(HOME, "Library", "LaunchAgents", LABEL + ".plist")
OLD_PLISTS = [os.path.join(HOME, "Library", "LaunchAgents", l + ".plist") for l in OLD_LABELS]
APP_COPIES = [os.path.join(HOME, "Applications", "Matita.app"), os.path.join(HOME, "Desktop", "Matita.app")]
SERVICE_APP = os.path.join(APP_DIR, "Matita Service.app")   # the background service; this gets Full Disk Access
OLD_APPS = [os.path.join(HOME, "Applications", "Transcriber.app"), os.path.join(HOME, "Desktop", "Transcriber.app"),
            os.path.join(HOME, "Applications", "Transcriber Launcher.app"),
            os.path.join(APP_DIR, "Transcriber Service.app"), os.path.join(APP_DIR, "Transcriber.app")]
URL = "http://127.0.0.1:%d/" % PORT
LANGS = ("en", "auto", "ko", "fr", "es", "de", "zh", "ja")
MODEL_URL = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/"
SPK_ENV = os.path.join(APP_DIR, "speakers-env")
SPK_PY = os.path.join(SPK_ENV, "bin", "python3")
SPK_DIR = os.path.join(MODEL_DIR, "speakers")
SEG_MODEL = os.path.join(SPK_DIR, "segmentation.onnx")
EMB_MODEL = os.path.join(SPK_DIR, "embedding.onnx")          # older installs: 3D-Speaker ERes2Net
EMB2_MODEL = os.path.join(SPK_DIR, "titanet-small.onnx")     # NeMo TitaNet: fewer mix-ups, about twice as fast
EMB2_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
            "nemo_en_titanet_small.onnx")
PK_DIR = os.path.join(MODEL_DIR, "parakeet-v3")     # NVIDIA Parakeet TDT 0.6B v3 (int8), run by sherpa-onnx
PK_ARCHIVE = "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8"
PK_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/%s.tar.bz2" % PK_ARCHIVE
PK_FILES = ("encoder.int8.onnx", "decoder.int8.onnx", "joiner.int8.onnx", "tokens.txt")
PK_VAD_URL = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
PK_LANGS = ("en", "fr", "es", "de")                 # of Matita's languages, the ones Parakeet v3 knows
SEG_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/"
           "sherpa-onnx-pyannote-segmentation-3-0.tar.bz2")
EMB_URL = ("https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/"
           "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx")

if BUNDLE:          # the app's own tools first; Homebrew only for extras such as Ollama
    os.environ["PATH"] = BUNDLE_BIN + ":/usr/bin:/bin:/usr/sbin:/sbin:/opt/homebrew/bin:/usr/local/bin"
else:
    os.environ["PATH"] = "/opt/homebrew/bin:/usr/local/bin:" + os.environ.get("PATH", "")


def find_bin(*names):
    for n in names:
        p = shutil.which(n)
        if p:
            return p
    return None


CAFFEINATE = find_bin("caffeinate")
OSASCRIPT = find_bin("osascript")
CURL = find_bin("curl") or "/usr/bin/curl"


def refresh_bins():
    """Look for the tools again (they may have been installed while the app was running)."""
    global FFMPEG, FFPROBE, WHISPER, BREW
    FFMPEG = find_bin("ffmpeg")
    FFPROBE = find_bin("ffprobe")
    WHISPER = find_bin("whisper-cli", "whisper-cpp")
    BREW = find_bin("brew")


refresh_bins()


CONFIG = os.path.join(APP_DIR, "config.json")
DEFAULT_OUT = os.path.join(HOME, "Documents", "Transcripts")


def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


CFG_LOCK = threading.Lock()


def atomic_write(path, data):
    """Write a file so it is never left half-written (temp file, fsync, rename)."""
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp-")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def save_config(cfg):
    atomic_write(CONFIG, json.dumps(cfg, ensure_ascii=False, indent=2).encode("utf-8"))


def edit_config(fn):
    """Read-modify-write the settings file safely from any thread."""
    with CFG_LOCK:
        cfg = load_config()
        fn(cfg)
        save_config(cfg)
        return cfg


def writable_dir(d):
    try:
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".write-test")
        open(probe, "w").close()
        os.remove(probe)
        os.listdir(d)
        return True
    except OSError:
        return False


OUT_FALLBACK = {"from": None}      # set when the chosen folder couldn't be used
READY = threading.Event()          # set once the save folder is known


def pick_out_dir():
    """The folder you chose; otherwise ~/Documents/Transcripts; ~/Transcripts if macOS blocks Documents."""
    chosen = load_config().get("out_dir") or DEFAULT_OUT
    for d in [chosen, DEFAULT_OUT, os.path.join(HOME, "Transcripts")]:
        if writable_dir(d):
            OUT_FALLBACK["from"] = None if d == chosen else chosen.replace(HOME, "~", 1)
            return d
    OUT_FALLBACK["from"] = chosen.replace(HOME, "~", 1)
    return os.path.join(HOME, "Transcripts")


def set_out_dir(path):
    """Switch the save folder, remember it, and show the transcripts already in it."""
    global OUT_DIR
    path = os.path.abspath(os.path.expanduser(path.strip())) if path and path.strip() else DEFAULT_OUT
    if not writable_dir(path):
        raise ValueError("Matita can't save to that folder. Pick a folder you can write to.")
    def change(cfg):
        if path == DEFAULT_OUT:
            cfg.pop("out_dir", None)
        else:
            cfg["out_dir"] = path
    edit_config(change)
    OUT_DIR = path
    OUT_FALLBACK["from"] = None
    with LOCK:
        for j in [j for j in ORDER if JOBS.get(j, {}).get("library")]:
            JOBS.pop(j, None)
            ORDER.remove(j)
    load_library()
    return path


# ---------------------------------------------------------------- quality modes
# Every mode meets the same floor: a large-class Whisper model at 8-bit or better, beam search,
# Silero voice-activity detection (skips silence and stops "phantom" text), the temperature
# fallback kept on, and repeated-line removal afterwards. The modes only trade speed for the
# last ~1 point of word error rate.
VAD_MODEL = "ggml-silero-v5.1.2.bin"
VAD_URL = "https://huggingface.co/ggml-org/whisper-vad/resolve/main/" + VAD_MODEL
MODES = {
    "parakeet": {"label": "Parakeet · Fastest", "model": "parakeet-v3", "beam": 0, "size": "640 MB",
                 "about": "NVIDIA Parakeet: several times faster than Whisper, with accuracy close to Turbo. "
                          "English, French, Spanish and German; other languages use Whisper Turbo. Hints aren't used."},
    "fast":     {"label": "Turbo 8-bit · Fast", "model": "ggml-large-v3-turbo-q8_0.bin", "beam": 2,
                 "size": "870 MB",
                 "about": "Compressed Turbo with a lighter search. Quickest, and only slightly less accurate."},
    "balanced": {"label": "Turbo · Recommended", "model": "ggml-large-v3-turbo.bin", "beam": 5, "size": "1.6 GB",
                 "about": "Nearly as accurate as Large v3 at several times the speed. Right for most recordings."},
    "best":     {"label": "Large v3 · Most accurate", "model": "ggml-large-v3.bin", "beam": 5, "size": "3.1 GB",
                 "about": "Whisper's biggest model. Slowest; worth it for noisy audio, strong accents or names that matter."},
}
DEFAULT_QUALITY = "balanced"
# Turbo matches large-v3 on these languages in published benchmarks; elsewhere (German, Chinese,
# and automatic detection, which may land on a language where turbo breaks down) it loses
# noticeably, so every mode uses the full large-v3 model there.
TURBO_OK = ("en", "es", "fr", "ja", "ko")


_HELP = {}


def whisper_help():
    """whisper-cli's --help text (cached), to skip flags an older Homebrew build doesn't know."""
    if WHISPER not in _HELP:
        try:
            r = subprocess.run([WHISPER, "--help"], capture_output=True, text=True, timeout=20, errors="replace")
            _HELP[WHISPER] = r.stdout + r.stderr
        except (OSError, subprocess.SubprocessError):
            _HELP[WHISPER] = ""
    return _HELP[WHISPER]


def whisper_has(flag):
    h = whisper_help()
    return not h or re.search(r"(^|[\s,])" + re.escape(flag) + r"([\s,]|$)", h, re.M) is not None


def choose_config(quality, lang):
    """The model file and beam size for a quality mode and language."""
    m = MODES.get(quality) or MODES[DEFAULT_QUALITY]
    if quality == "parakeet":
        if lang in PK_LANGS:
            return m["model"], 0
        m = MODES["balanced"]                     # Parakeet doesn't know this language
    model, beam = m["model"], m["beam"]
    if "turbo" in model and lang not in TURBO_OK:
        model = MODES["best"]["model"]          # same beam as the chosen mode
    return model, beam


def clean_quality(q):
    return q if q in MODES else DEFAULT_QUALITY


DL_LOCK = threading.Lock()


def ensure_file(url, dest, on_progress=None, jid=None):
    """Download a model file once, even if several jobs ask for it at the same time."""
    with DL_LOCK:
        if os.path.isfile(dest) and os.path.getsize(dest) > 100_000:
            return
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        total = remote_size(url)
        part = dest + ".part"
        if jid and state_of(jid) == "cancelled":    # cancelled while waiting for another download
            return                                  # (the caller sees the cancel and stops)
        p = subprocess.Popen([CURL, "-L", "--fail", "-s", "--speed-limit", "1024", "--speed-time", "60", "-o", part, url])
        DOWNLOADS.add(p.pid)
        if jid:
            with LOCK:
                PROCS[jid] = p                     # so Cancel can stop the download
        t0, seen = time.time(), []
        while p.poll() is None:
            if total and os.path.exists(part):
                got = os.path.getsize(part)
                if on_progress:
                    on_progress(int(min(99, 100 * got / total)))
                if jid:                             # size, speed and time left for the first-run download
                    seen = (seen + [(time.time(), got)])[-20:]           # last ~10 s for a steady speed
                    rate = (seen[-1][1] - seen[0][1]) / max(0.5, seen[-1][0] - seen[0][0]) if len(seen) > 1 else 0
                    update(jid, dl={"got": got, "total": total,
                                    "eta": int((total - got) / rate) if rate > 1000 and time.time() - t0 > 3 else None})
            time.sleep(0.5)
        DOWNLOADS.discard(p.pid)
        if jid:
            with LOCK:
                PROCS.pop(jid, None)
        if p.returncode != 0:
            try:
                os.remove(part)
            except OSError:
                pass
            raise RuntimeError("Couldn't download %s. Check your internet connection and try again."
                               % os.path.basename(dest))
        os.replace(part, dest)


def drop_repeats(segs):
    """Remove Whisper's 'stuck' loops: the same line three or more times in a row is kept once.
    (A line said twice stays as it is; people do repeat themselves.)"""
    key = lambda seg: re.sub(r"\W+", "", seg["text"]).lower()
    out, i = [], 0
    while i < len(segs):
        j = i
        while j + 1 < len(segs) and key(segs[j + 1]) and key(segs[j + 1]) == key(segs[i]):
            j += 1
        if j - i + 1 >= 3:
            first = dict(segs[i])
            first["end"] = segs[j]["end"]
            out.append(first)
        else:
            out.extend(segs[i:j + 1])
        i = j + 1
    return out


def write_plain_files(base, segs):
    txt = "\n".join(s["text"] for s in segs) + "\n"
    srt = "".join("%d\n%s --> %s\n%s\n\n" % (i, hms(s["start"], True), hms(s["end"], True), s["text"])
                  for i, s in enumerate(segs, 1))
    atomic_write(base + ".txt", txt.encode("utf-8"))
    atomic_write(base + ".srt", srt.encode("utf-8"))


def parakeet_ready():
    return all(os.path.isfile(os.path.join(PK_DIR, f)) for f in PK_FILES + ("silero_vad.onnx",)) and sherpa_ready() \
        and os.path.isfile(os.path.join(SPK_ENV, ".sherpa-ok"))


def sherpa_ready():
    return os.path.isfile(SPK_PY) and (os.path.isfile(os.path.join(SPK_ENV, ".sherpa-ok")) or os.path.isfile(SEG_MODEL))


def list_models():
    if not os.path.isdir(MODEL_DIR):
        return []
    return (["parakeet-v3"] if parakeet_ready() else []) + sorted(
        f for f in os.listdir(MODEL_DIR)
        if f.startswith("ggml-") and f.endswith(".bin")
        and os.path.getsize(os.path.join(MODEL_DIR, f)) > 10_000_000
    )


# ---------------------------------------------------------------- jobs
JOBS = {}
ORDER = []
PROCS = {}
SIDE = {}          # speaker identification running alongside the transcription
AWAKE = {}         # caffeinate while the page is recording (and while a job runs)
RUNNING = threading.Event()   # cleared while the queue is paused
RUNNING.set()
PAUSE = {"since": None}
DOWNLOADS = set()             # curl processes; downloads keep going while paused


def stop_proc(p):
    """Terminate a process even if it is paused (a stopped process can't act on SIGTERM until continued)."""
    if p and p.poll() is None:
        try:
            p.terminate()
            os.kill(p.pid, signal.SIGCONT)
        except OSError:
            pass


def register(jid, p, table=None):
    """Track a job's process. Done under LOCK together with the pause/cancel checks, so a Pause,
    Resume or Cancel arriving at the same moment can't leave the process frozen or running."""
    table = PROCS if table is None else table
    with LOCK:
        table[jid] = p
        if (JOBS.get(jid) or {}).get("state") == "cancelled":
            stop_proc(p)
        elif not RUNNING.is_set() and p.poll() is None:
            try:
                os.kill(p.pid, signal.SIGSTOP)
            except OSError:
                pass


# ---------------------------------------------------------------- keep going with the lid closed (optional)
# macOS always sleeps when the lid closes, even with caffeinate. `pmset -a disablesleep 1` stops that; it
# needs admin rights, so a one-time setup adds a sudoers rule that allows exactly these two commands.
# Sleep is only disabled while there is work, and restored as soon as the queue is empty or paused.
SUDO = "/usr/bin/sudo"
PMSET = "/usr/bin/pmset"
SUDOERS_FILE = "/etc/sudoers.d/matita-lid"
OLD_SUDOERS = "/etc/sudoers.d/transcriber-lid"     # same rule, from before the rename
LID = {"disabled": None, "idle_since": None, "failed": 0}   # disabled: what we last set (None = unknown)
LID_LOCK = threading.Lock()


def lid_mode():
    m = load_config().get("lid")
    return m if m in ("sleep", "plugged", "always") else "sleep"


def lid_ready():
    """True when the sudoers rule is installed and works without a password (checked by running the
    harmless 'sleep allowed' command; the lid loop turns it back off if work is running)."""
    if not (os.path.isfile(SUDOERS_FILE) or os.path.isfile(OLD_SUDOERS)):
        return False
    try:
        ok = subprocess.run([SUDO, "-n", PMSET, "-a", "disablesleep", "0"],
                            capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False
    if ok:
        with LID_LOCK:
            LID["disabled"], LID["failed"] = False, 0
    return ok


def lid_restore():
    """Make sure the Mac can sleep again (at start-up and on any exit)."""
    if os.path.isfile(SUDOERS_FILE) or os.path.isfile(OLD_SUDOERS):
        try:
            subprocess.run([SUDO, "-n", PMSET, "-a", "disablesleep", "0"], capture_output=True, timeout=10)
        except Exception:
            pass
    LID["disabled"] = False


def on_ac_power():
    try:
        r = subprocess.run([find_bin("pmset") or PMSET, "-g", "batt"], capture_output=True, text=True, timeout=10)
        return "AC Power" in r.stdout
    except (OSError, subprocess.SubprocessError):
        return True                                # desktops have no battery


def lid_set(disable):
    with LID_LOCK:
        if LID["disabled"] == disable:
            return
        if disable and time.time() - LID["failed"] < 300:
            return                                 # broken permission: don't retry (and log) every 5 s
        if disable:
            LID["disabled"] = True                 # set first, so an exit right now still restores sleep
        try:
            r = subprocess.run([SUDO, "-n", PMSET, "-a", "disablesleep", "1" if disable else "0"],
                               capture_output=True, timeout=15)
            ok = r.returncode == 0
        except (OSError, subprocess.SubprocessError):
            ok = False
        if ok:
            LID["disabled"] = disable
        elif disable:
            LID["disabled"], LID["failed"] = None, time.time()


def lid_setup():
    """Ask once for the admin password (macOS dialog) and install the narrow sudoers rule."""
    import getpass
    user = getpass.getuser()
    if not re.match(r"^[A-Za-z0-9._-]+$", user):
        raise RuntimeError("Your account name has characters this setup doesn't support.")
    rule = "%s ALL=(root) NOPASSWD: %s -a disablesleep 1, %s -a disablesleep 0" % (user, PMSET, PMSET)
    # Everything happens as root in a root-owned temp file, so nothing the user's account can write
    # is ever installed into /etc/sudoers.d.
    sh = ("umask 077; f=$(/usr/bin/mktemp /private/var/root/matita-lid.XXXXXX) || exit 1; "
          "printf '%s\\n' '{rule}' > \"$f\" && /usr/sbin/visudo -cf \"$f\" && "
          "/usr/bin/install -m 0440 -o root -g wheel \"$f\" '{dst}'; rc=$?; /bin/rm -f \"$f\"; exit $rc"
          ).format(rule=rule, dst=SUDOERS_FILE)
    script = 'do shell script "%s" with administrator privileges with prompt "%s"' % (
        sh.replace("\\", "\\\\").replace('"', '\\"'),
        "Matita wants to keep transcribing when your MacBook lid is closed.")
    try:
        r = subprocess.run([OSASCRIPT or "/usr/bin/osascript", "-e", script], capture_output=True, text=True,
                           timeout=300, errors="replace")
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError("Couldn't show the macOS password dialog: %s" % e)
    if r.returncode != 0:
        if "-128" in r.stderr or "canceled" in r.stderr.lower():
            raise RuntimeError("Setup was cancelled.")
        raise RuntimeError("Couldn't finish setup: " + r.stderr.strip()[-200:])
    if not lid_ready():
        raise RuntimeError("The setting was installed but doesn't work yet. Try again.")


def lid_loop():
    """Every few seconds: disable lid sleep only while there's work, the queue isn't paused and
    (if chosen) the Mac is plugged in. Restore normal sleep as soon as any of that stops."""
    while True:
        try:
            lid_tick()
        except Exception:
            pass
        time.sleep(5)


def lid_tick():
    mode = lid_mode()
    with LOCK:
        busy = any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS)
    want = (mode != "sleep" and busy and RUNNING.is_set()
            and (mode == "always" or on_ac_power()))
    if want:
        LID["idle_since"] = None
        lid_set(True)
    elif LID["disabled"]:
        # a short grace period after the queue empties, so gaps between files don't flicker;
        # paused, unplugged or switched off: restore sleep right away
        LID["idle_since"] = LID["idle_since"] or time.time()
        if busy or time.time() - LID["idle_since"] >= 15:
            lid_set(False)
            LID["idle_since"] = None


def keep_awake(on):
    """Hold a caffeinate only while there's work and the queue isn't paused (decided under the
    lock, so a late call can't leave one running). -w ends it if Matita itself stops."""
    with LOCK:
        busy = any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS)
        old = AWAKE.pop("job", None)
        if on and busy and RUNNING.is_set() and CAFFEINATE:
            AWAKE["job"] = subprocess.Popen([CAFFEINATE, "-i", "-w", str(os.getpid())])
    if old and old.poll() is None:
        old.terminate()


def set_paused(on):
    """Pause or resume transcription: running whisper/ffmpeg/speaker processes are frozen in place
    (SIGSTOP) and continue exactly where they were (SIGCONT). Nothing new starts while paused."""
    with LOCK:
        if on == (not RUNNING.is_set()):
            return
        procs = [x for x in list(PROCS.values()) + list(SIDE.values())
                 if x and x.poll() is None and x.pid not in DOWNLOADS]
        if on:
            RUNNING.clear()
            PAUSE["since"] = time.time()
        else:
            gap = time.time() - (PAUSE["since"] or time.time())
            PAUSE["since"] = None
            for j in JOBS.values():                 # keep time estimates honest
                if j["state"] in ACTIVE and j["state"] != "downloading" and j.get("started"):
                    j["started"] += gap
            RUNNING.set()
        for x in procs:                             # signal under the lock (os.kill doesn't block)
            try:
                os.kill(x.pid, signal.SIGSTOP if on else signal.SIGCONT)
            except OSError:
                pass
    keep_awake(not on)                              # let the Mac sleep while paused
LOCK = threading.Lock()
Q = queue.Queue()
BATCH = {"done": 0, "failed": 0}
ACTIVE = ("queued", "downloading", "converting", "transcribing", "speakers")
PUBLIC_KEYS = ("id", "name", "state", "progress", "error", "created", "started",
               "finished", "lang", "model", "quality", "note", "preview", "duration", "words", "library", "has_srt", "summary",
               "summary_ai", "speakers", "speaker_count", "warning", "auto_titled", "recorded", "dl")

TS_RE = re.compile(r"-->\s*(\d+):(\d+):(\d+)\.(\d+)\]")
PROG_RE = re.compile(r"progress\s*=\s*(\d+)%")
SEG_LINE = re.compile(r"^\[\d+:\d+:\d+\.\d+\s*-->\s*\d+:\d+:\d+\.\d+\]\s*(.*)$")


def update(jid, **kw):
    with LOCK:
        if jid in JOBS:
            JOBS[jid].update(kw)


def state_of(jid):
    with LOCK:
        return JOBS.get(jid, {}).get("state")


def duration_of(path):
    if not FFPROBE:
        return None
    try:
        r = subprocess.run(
            [FFPROBE, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=nw=1:nk=1", path],
            capture_output=True, text=True, timeout=60)
        return float(r.stdout.strip())
    except Exception:
        return None


def unique_base(stem):
    base = os.path.join(OUT_DIR, stem)
    cand, i = base, 2
    taken = lambda c: any(os.path.exists(c + e) for e in (".txt", ".srt", ".json", ".md", ".docx")) \
        or os.path.exists(sidecar_path(c)) or bool(files_of(c))
    while taken(cand):
        cand = "%s (%d)" % (base, i)
        i += 1
    return cand


def count_words(text):
    return len(text.split())


STOP = set("""a about above after again against all also am an and any are aren't as at be because been before being
below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't down during each few for
from further get got had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself
his how how's i i'd i'll i'm i've if in into is isn't it it's its itself just let's like me more most mustn't my myself no
nor not of off on once only or other ought our ours ourselves out over own really right same say said says she she'd she'll
she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they
they'd they'll they're they've thing things think this those through to too um uh under until up us very was wasn't we
we'd we'll we're we've were weren't what what's when when's where where's which while who who's whom why why's will with
won't would wouldn't yeah yes you you'd you'll you're you've your yours yourself yourselves okay ok kind sort actually
basically mean going gonna know want wanna lot lots well also maybe still even much many one two way go come make made
something anything everything need see look also really today now gonna got get gets sets lets goes takes makes
used uses using said tell told feel felt yeah everyone everybody next thanks thank great good
welcome hello hey""".split())
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]+|[\uac00-\ud7a3]{2,}")


def summarize(text, limit=170):
    """A short, extractive summary: the sentence that best represents the transcript, plus keywords."""
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return "No speech was found in this recording.", []
    sentences = [x.strip() for x in re.split(r"(?<=[.!?。])\s+", text) if x.strip()]
    words = [w.lower() for w in WORD_RE.findall(text)]
    content = [w for w in words if w not in STOP and len(w) > 2]
    if not content:
        return (text[:limit].rsplit(" ", 1)[0] + "…") if len(text) > limit else text, []
    freq = {}
    for w in content:
        freq[w] = freq.get(w, 0) + 1
    top = max(freq.values())
    lead = max(freq, key=freq.get) if top > 1 else None
    scored = []
    for i, sen in enumerate(sentences):
        ws = [w.lower() for w in WORD_RE.findall(sen)]
        cw = [w for w in ws if w in freq]
        if len(ws) < 6:
            continue
        score = sum(freq[w] / top for w in set(cw)) / (len(ws) ** 0.6)
        if lead and lead in cw:
            score *= 1.4                                              # mentions the main topic
        score *= 1.2 if i < max(3, len(sentences) // 4) else 1.0      # openings often state the topic
        scored.append((score, i, sen))
    if not scored:
        scored = [(0, i, sen) for i, sen in enumerate(sentences)]
    scored.sort(reverse=True)
    picked = [scored[0]]
    if len(scored[0][2]) < 80 and len(scored) > 1:
        picked.append(scored[1])
    summary = " ".join(sen for _, _, sen in sorted(picked, key=lambda x: x[1]))
    if len(summary) > limit:
        summary = summary[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"
    keywords = [w for w, n in sorted(freq.items(), key=lambda kv: -kv[1]) if n > 1 and not w.isdigit()][:5]
    return summary, keywords


def srt_segments(path):
    segs = []
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            raw = fh.read().strip()
    except OSError:
        return segs
    for block in re.split(r"\n\s*\n", raw):
        lines = block.strip().splitlines()
        if len(lines) >= 3 and "-->" in lines[1]:
            stamp = lines[1].split("-->")[0].strip().replace(".", ",")
            start = stamp.split(",")[0]
            m = re.match(r"(\d+):(\d+):(\d+)(?:,(\d+))?", stamp)
            sec = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
                   + int((m.group(4) or "0").ljust(3, "0")[:3]) / 1000.0) if m else 0.0
            text = " ".join(l.strip() for l in lines[2:]).strip()
            if text:
                segs.append({"t": start, "s": sec, "text": text})
    return segs


# ---------------------------------------------------------------- background setup tasks
TASKS = {k: {"state": "idle", "progress": 0, "message": ""} for k in ("tools", "model", "speakers", "update")}
UPDATE = {"latest": None, "available": False, "checked": 0}


def task(name, **kw):
    with LOCK:
        TASKS[name].update(kw)


def run_logged(cmd, name, timeout=None):
    """Run a command, showing its latest output line as the task message."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, errors="replace")
    last = ""
    for line in p.stdout:
        line = line.strip()
        if line:
            last = line
            task(name, message=line[-140:])
    p.wait(timeout)
    return p.returncode, last


def remote_size(url):
    try:
        r = subprocess.run([CURL, "-sIL", "--max-time", "20", url], capture_output=True, text=True)
        sizes = re.findall(r"(?im)^content-length:\s*(\d+)", r.stdout)
        return int(sizes[-1]) if sizes else 0
    except Exception:
        return 0


def download(url, dest, name, weight=1.0, base=0.0):
    """Download with curl (it uses the Mac's own certificates), reporting progress into TASKS[name]."""
    total = remote_size(url)
    part = dest + ".part"
    p = subprocess.Popen([CURL, "-L", "--fail", "-s", "--speed-limit", "1024", "--speed-time", "60", "-o", part, url])
    while p.poll() is None:
        if total and os.path.exists(part):
            pct = base + weight * 100 * os.path.getsize(part) / total
            task(name, progress=int(min(99, pct)))
        time.sleep(0.5)
    if p.returncode != 0:
        try:
            os.remove(part)
        except OSError:
            pass
        raise RuntimeError("Download failed. Check your internet connection and try again.")
    os.replace(part, dest)


def setup_tools():
    task("tools", state="running", progress=0, message="Starting Homebrew…")
    try:
        if not BREW:
            raise RuntimeError("Homebrew isn't installed. Install it from brew.sh, then try again.")
        code, last = run_logged([BREW, "install", "whisper-cpp", "ffmpeg"], "tools")
        refresh_bins()
        if code != 0 or not (WHISPER and FFMPEG):
            raise RuntimeError("Homebrew couldn't finish: " + last)
        task("tools", state="done", progress=100, message="whisper.cpp and ffmpeg are installed.")
    except Exception as e:
        task("tools", state="error", message=str(e))


def setup_model():
    task("model", state="running", progress=0, message="Downloading the speech model (about 1.6 GB)…")
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)
        with DL_LOCK:                             # don't race a job downloading the same file
            download(MODEL_URL + DEFAULT_MODEL, os.path.join(MODEL_DIR, DEFAULT_MODEL), "model")
        ensure_file(VAD_URL, os.path.join(MODEL_DIR, VAD_MODEL))
        task("model", state="done", progress=100, message="Speech model downloaded.")
    except Exception as e:
        task("model", state="error", message=str(e))


def speakers_ready():
    return all(os.path.isfile(p) for p in (SPK_PY, SEG_MODEL)) and (os.path.isfile(EMB2_MODEL) or os.path.isfile(EMB_MODEL))


SHERPA_LOCK = threading.Lock()
SHERPA_VERSION = "2.0.1"


def sherpa_current():
    return os.path.isfile(SPK_PY) and subprocess.run(
        [SPK_PY, "-c", "import sys, numpy, sherpa_onnx; sys.exit(sherpa_onnx.__version__ != %r)" % SHERPA_VERSION],
        capture_output=True).returncode == 0


def install_sherpa(say, log="speakers"):
    """The private Python environment with sherpa-onnx (speaker identification and Parakeet)."""
    with SHERPA_LOCK:                               # speakers setup and a first Parakeet job may both ask
        _install_sherpa(say, log)


def _install_sherpa(say, log):
    py = sys.executable if BUNDLE else ("/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable)
    healthy = os.path.exists(SPK_PY) and subprocess.run(
        [SPK_PY, "-c", "import sys"], capture_output=True).returncode == 0
    if not healthy:
        say("Creating a private Python environment…", 3)
        r = subprocess.run([py, "-m", "venv", "--clear", SPK_ENV], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("Couldn't create the Python environment: " + r.stderr.strip()[-200:])
    if not (healthy and sherpa_current()):
        say("Installing the speech engine (sherpa-onnx)…", 10)
        code, last = run_logged([SPK_PY, "-m", "pip", "install", "--quiet", "--disable-pip-version-check",
                                 "sherpa-onnx==" + SHERPA_VERSION, "numpy"], log)
        if code != 0:
            raise RuntimeError("Couldn't install the speech engine: " + last)
    open(os.path.join(SPK_ENV, ".sherpa-ok"), "w").close()


PK_LOCK = threading.Lock()


def ensure_parakeet(jid):
    """Install Parakeet the first time it's used: the engine, the model (640 MB) and the voice detector."""
    with PK_LOCK:
        if parakeet_ready():
            return
        note = lambda msg, pct=0: update(jid, state="downloading", note=msg, progress=pct, started=time.time(), error=None)
        if not sherpa_ready() or not sherpa_current():   # older speaker installs may have an older engine
            install_sherpa(lambda msg, pct: note("Installing Parakeet (one time)", pct), "parakeet")
        os.makedirs(PK_DIR, exist_ok=True)
        if not all(os.path.isfile(os.path.join(PK_DIR, f)) for f in PK_FILES):
            arc = os.path.join(MODEL_DIR, PK_ARCHIVE + ".tar.bz2")
            note("Downloading Parakeet (one time, 640 MB)")
            ensure_file(PK_URL, arc, lambda pct: update(jid, progress=pct), jid)
            if state_of(jid) == "cancelled":
                return
            note("Unpacking Parakeet…", 99)
            with tarfile.open(arc, "r:bz2") as t:
                for m in t.getmembers():
                    name = os.path.basename(m.name)
                    if name in PK_FILES and m.isfile():
                        with t.extractfile(m) as src, open(os.path.join(PK_DIR, name + ".part"), "wb") as out:
                            shutil.copyfileobj(src, out)
                        os.replace(os.path.join(PK_DIR, name + ".part"), os.path.join(PK_DIR, name))
            try:
                os.remove(arc)
            except OSError:
                pass
            if not all(os.path.isfile(os.path.join(PK_DIR, f)) for f in PK_FILES):
                raise RuntimeError("The Parakeet download was incomplete. Try again.")
        if not os.path.isfile(os.path.join(PK_DIR, "silero_vad.onnx")):
            ensure_file(PK_VAD_URL, os.path.join(PK_DIR, "silero_vad.onnx"), None, jid)


def upgrade_voice_model():
    """Installs from before 3.18 used a voice model trained on Chinese speakers; fetch the better one."""
    if speakers_ready() and not os.path.isfile(EMB2_MODEL):
        try:
            ensure_file(EMB2_URL, EMB2_MODEL)
        except Exception:
            pass


def setup_speakers(echo=False):
    """Install the speaker-identification add-on: a private Python environment plus two small models."""
    def say(msg, pct):
        task("speakers", message=msg, progress=pct)
        if echo:
            print(msg)
    task("speakers", state="running", progress=0, message="")
    try:
        install_sherpa(say, "speakers")
        os.makedirs(SPK_DIR, exist_ok=True)
        if not os.path.isfile(SEG_MODEL):
            say("Downloading the speech-turn model…", 40)
            tmp = tempfile.mkdtemp()
            try:
                arc = os.path.join(tmp, "seg.tar.bz2")
                download(SEG_URL, arc, "speakers", weight=0.15, base=40)
                with tarfile.open(arc, "r:bz2") as t:
                    member = next(m for m in t.getmembers() if m.name.endswith("/model.onnx"))
                    with t.extractfile(member) as src, open(SEG_MODEL + ".part", "wb") as out:
                        shutil.copyfileobj(src, out)
                os.replace(SEG_MODEL + ".part", SEG_MODEL)   # only a complete file counts as installed
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
        if not os.path.isfile(EMB2_MODEL):
            say("Downloading the voice model…", 55)
            download(EMB2_URL, EMB2_MODEL, "speakers", weight=0.4, base=55)
        say("Checking everything works…", 97)
        r = subprocess.run([SPK_PY, "-c", "import sherpa_onnx, numpy"], capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("The speaker engine didn't load: " + r.stderr.strip()[-200:])
        task("speakers", state="done", progress=100, message="Speaker identification is ready.")
        if echo:
            print("Speaker identification is ready.")
        return True
    except Exception as e:
        task("speakers", state="error", message=str(e))
        if echo:
            print("Speaker identification wasn't set up: %s" % e)
        return False


# ---------------------------------------------------------------- updates
def parse_version(v):
    return tuple(int(x) for x in re.findall(r"\d+", v or "0")[:3])


def fetch_latest_source():
    url = "https://raw.githubusercontent.com/%s/main/matita.py" % REPO
    r = subprocess.run([CURL, "-fsSL", "--max-time", "30", url], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("Couldn't reach GitHub.")
    src = r.stdout.decode("utf-8")
    m = re.search(r'^VERSION = "([^"]+)"', src, re.M)
    if not m:
        raise RuntimeError("The file on GitHub doesn't look like Matita.")
    return m.group(1), src


def latest_release():
    """(version, page URL, .dmg URL) of the newest Matita.app on GitHub Releases."""
    r = subprocess.run([CURL, "-fsSL", "--max-time", "30", "-H", "Accept: application/vnd.github+json",
                        "https://api.github.com/repos/%s/releases/latest" % REPO], capture_output=True)
    if r.returncode != 0:
        raise RuntimeError("Couldn't reach GitHub.")
    d = json.loads(r.stdout.decode("utf-8"))
    dmg = next((a.get("browser_download_url") for a in d.get("assets") or [] if str(a.get("name", "")).endswith(".dmg")), None)
    return str(d.get("tag_name") or "").lstrip("v"), d.get("html_url"), dmg


def check_update():
    if REPO.startswith("YOUR_" + "GITHUB"):      # not set up for a GitHub repo yet
        return
    if BUNDLE:
        try:
            latest, page, dmg = latest_release()
            UPDATE.update(latest=latest or None, page=page, dmg=dmg, checked=time.time(),
                          available=bool(latest) and parse_version(latest) > parse_version(VERSION))
        except Exception:
            UPDATE.update(checked=time.time())
        return
    try:
        latest, _ = fetch_latest_source()
        UPDATE.update(latest=latest, available=parse_version(latest) > parse_version(VERSION), checked=time.time())
    except Exception:
        UPDATE.update(checked=time.time())


def update_loop():
    while True:
        check_update()
        time.sleep(12 * 3600)


def apply_update():
    if BUNDLE:      # the app updates by downloading the new .dmg
        url = UPDATE.get("dmg") or UPDATE.get("page") or "https://github.com/%s/releases/latest" % REPO
        subprocess.run(["open", url])
        task("update", state="done", progress=100, message="Downloading the new Matita. Open it and drag Matita to Applications.")
        return
    task("update", state="running", progress=10, message="Downloading the new version…")
    try:
        with LOCK:
            busy = any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS)
        if busy:
            raise RuntimeError("Wait until the recordings in progress are finished, then update.")
        latest, src = fetch_latest_source()
        if parse_version(latest) <= parse_version(VERSION):
            raise RuntimeError("You already have the latest version.")
        compile(src, "matita.py", "exec")          # never install a broken file
        with LOCK:
            if any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS):
                raise RuntimeError("Wait until the recordings in progress are finished, then update.")
        target = os.path.abspath(__file__)
        backup = target + ".bak"
        shutil.copyfile(target, backup)
        atomic_write(target, src.encode("utf-8"))
        if os.environ.get("XPC_SERVICE_NAME") == LABEL:
            task("update", state="done", progress=100, message="Updated to %s. Restarting…" % latest)
            # Restart with the new code; if it isn't answering 25 seconds later, put the old version back.
            script = ('sleep 1; launchctl kickstart -k "gui/$(id -u)/{label}"; sleep 25; '
                      'if ! curl -s -o /dev/null --max-time 3 "{url}api/info"; then '
                      'cp "{bak}" "{target}"; launchctl kickstart -k "gui/$(id -u)/{label}"; fi'
                      ).format(label=LABEL, url=URL, bak=backup, target=target)
            subprocess.Popen(["/bin/sh", "-c", script], start_new_session=True)
        else:
            task("update", state="error",
                 message="Updated to %s. Quit Matita in Terminal (Control-C) and start it again to finish." % latest)
    except Exception as e:
        task("update", state="error", message=str(e))


# ---------------------------------------------------------------- speakers
def sidecar_path(base):
    return os.path.join(os.path.dirname(base), ".transcriber", os.path.basename(base) + ".json")


UNSURE_P = 0.5          # words Whisper gave less than this probability are underlined for checking


def unsure_path(base):
    return os.path.join(os.path.dirname(base), ".transcriber", os.path.basename(base) + ".unsure.json")


def unsure_words(segs, lang):
    """Words the model wasn't confident about: [[word, start seconds, probability], ...]."""
    sample = "".join(seg.get("text", "") for seg in segs[:40])
    spaced = lang not in ("ja", "zh") and not (     # Japanese and Chinese: each token is a "word"
        lang == "auto" and sample and sample.count(" ") < len(sample) * 0.04)
    words = []
    for seg in segs:
        cur = None
        for tok in seg.get("tokens", []):
            tt, a, p = tok[0], tok[1], (tok[3] if len(tok) > 3 else 1.0)
            if not re.search(r"\w", tt):            # punctuation: keep the text, ignore its probability
                if cur:
                    cur[0] += tt
                continue
            if cur is None or not spaced or tt.startswith(" "):
                if cur:
                    words.append(cur)
                cur = [tt.strip(), a, p]
            else:
                cur[0] += tt
                cur[2] = min(cur[2], p)
        if cur:
            words.append(cur)
    out = []
    for w, a, p in words:
        w = re.sub(r"^\W+|\W+$", "", w)
        if w and p < UNSURE_P:
            out.append([w, round(a, 2), round(p, 2)])
    return out


def audio_path(base):
    return os.path.join(os.path.dirname(base), ".transcriber", os.path.basename(base) + ".m4a")


def peaks_of(base, bins=1200):
    """A small loudness outline of the recording for the reader's waveform (cached next to the audio)."""
    src = audio_path(base)
    cache = os.path.join(os.path.dirname(base), ".transcriber", os.path.basename(base) + ".peaks.json")
    try:
        if os.path.getmtime(cache) >= os.path.getmtime(src):
            with open(cache, encoding="utf-8") as fh:
                return json.load(fh)
    except (OSError, ValueError):
        pass
    rate = 2000
    r = subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-i", src, "-ac", "1", "-ar", str(rate),
                        "-f", "s16le", "-"], capture_output=True, timeout=600)
    pcm = array.array("h")
    pcm.frombytes(r.stdout[:len(r.stdout) // 2 * 2])
    if sys.byteorder == "big":
        pcm.byteswap()
    n = len(pcm)
    if not n:
        return {"duration": 0, "peaks": []}
    step = max(1, n // bins)
    peaks = []
    for i in range(0, n, step):
        chunk = pcm[i:i + step]
        peaks.append(max(max(chunk), -min(chunk)) / 32768.0)
    top = max(peaks) or 1.0
    data = {"duration": n / float(rate), "peaks": [round(min(1.0, p / top), 3) for p in peaks]}
    try:
        atomic_write(cache, json.dumps(data).encode("utf-8"))
    except OSError:
        pass
    return data


CLOCK = {}


def mac_clock():
    """The Mac's own time format: whether 24-hour time is forced, and the region (e.g. en_FR)."""
    if not CLOCK:
        def read(key):
            try:
                r = subprocess.run(["defaults", "read", "-g", key], capture_output=True, text=True, timeout=5)
                return r.stdout.strip() if r.returncode == 0 else ""
            except (OSError, subprocess.SubprocessError):
                return ""
        force24, force12 = read("AppleICUForce24HourTime"), read("AppleICUForce12HourTime")
        CLOCK.update(h24=True if force24 == "1" else (False if force12 == "1" else None),
                     locale=read("AppleLocale"))
    return CLOCK


APPEAR = {"t": 0, "dark": None}


def mac_dark():
    """True when macOS is in Dark mode right now (also follows Auto)."""
    if time.time() - APPEAR["t"] > 2:
        try:
            r = subprocess.run(["defaults", "read", "-g", "AppleInterfaceStyle"], capture_output=True, text=True, timeout=5)
            APPEAR["dark"] = r.returncode == 0 and r.stdout.strip().lower() == "dark"
        except (OSError, subprocess.SubprocessError):
            APPEAR["dark"] = None
        APPEAR["t"] = time.time()
    return APPEAR["dark"]


def load_unsure(base):
    try:
        with open(unsure_path(base), encoding="utf-8") as fh:
            return [w for w in json.load(fh) if isinstance(w, list) and len(w) == 3]
    except (OSError, ValueError):
        return []


def save_audio(wav, base):
    """Keep a small copy of the audio (AAC, about 0.4 MB a minute) so lines can be played back."""
    dest = audio_path(base)
    tmp = dest + ".part"
    try:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        r = subprocess.run([FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", wav, "-c:a", "aac",
                            "-b:a", "48k", "-ac", "1", "-movflags", "+faststart", "-f", "mp4", tmp],
                           capture_output=True, timeout=3600)
        if r.returncode == 0:
            os.replace(tmp, dest)
            return True
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def load_sidecar(base):
    try:
        with open(sidecar_path(base), encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    # only a real speaker file counts (a "Talk.peaks" transcript would otherwise read Talk's waveform cache)
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        return None
    return data


def save_sidecar(base, data):
    atomic_write(sidecar_path(base), json.dumps(data, ensure_ascii=False).encode("utf-8"))


def hms(sec, ms=False):
    total = int(round(max(0.0, sec) * 1000))          # whole milliseconds first, so 1.9996 s -> 00:00:02,000
    h, rem = divmod(total, 3600000)
    m, rem = divmod(rem, 60000)
    s, milli = divmod(rem, 1000)
    if ms:
        return "%02d:%02d:%02d,%03d" % (h, m, s, milli)
    return "%02d:%02d:%02d" % (h, m, s)


def whisper_segments(path):
    """Segments (and word-level tokens) from whisper.cpp's full JSON output."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            data = json.loads(fh.read(), strict=False)
    except (OSError, ValueError):
        return []
    segs = []
    for seg in data.get("transcription", []):
        text = (seg.get("text") or "").strip()
        if not text:
            continue
        off = seg.get("offsets", {})
        toks = []
        for t in seg.get("tokens", []):
            tt = t.get("text", "")
            if tt.startswith("[_") or not tt.strip():
                continue
            o = t.get("offsets", {})
            toks.append((tt, o.get("from", 0) / 1000.0, o.get("to", 0) / 1000.0, float(t.get("p", 1.0))))
        start, end = off.get("from", 0) / 1000.0, off.get("to", 0) / 1000.0
        # With voice-activity detection on, whisper.cpp maps segment times back to the original
        # recording but leaves word times on the silence-removed timeline. Stretch the words
        # back over their segment so speaker labels line up with the real audio.
        if toks:
            p0, p1 = min(t[1] for t in toks), max(t[2] for t in toks)
            if p1 > p0 and end > start:
                k = (end - start) / (p1 - p0)
                toks = [(tt, start + (a - p0) * k, start + (b - p0) * k, pr) for tt, a, b, pr in toks]
        segs.append({"start": start, "end": end, "text": text, "tokens": toks})
    return segs


def speaker_at(a, b, diar):
    """The speaker who talks the most between a and b (or the nearest one)."""
    best, best_ov = None, 0.0
    for s, e, spk in diar:
        ov = min(b, e) - max(a, s)
        if ov > best_ov:
            best, best_ov = spk, ov
    if best is None and diar:
        mid = (a + b) / 2
        best = min(diar, key=lambda d: min(abs(mid - d[0]), abs(mid - d[1])))[2]
    return best


def assign_speakers(segs, diar):
    out = []
    for seg in segs:
        toks = seg["tokens"]
        joined = "".join(t[0] for t in toks).strip()
        labels = [speaker_at(t[1], max(t[2], t[1] + 0.01), diar) for t in toks]
        # smooth one-token blips
        for i in range(1, len(labels) - 1):
            if labels[i - 1] == labels[i + 1] != labels[i]:
                labels[i] = labels[i - 1]
        if toks and len(set(labels)) > 1 and joined == seg["text"]:
            run = None
            for tok, lab in zip(toks, labels):
                if run and run["speaker"] == lab:
                    run["text"] += tok[0]
                    run["end"] = tok[2]
                else:
                    run = {"speaker": lab, "start": tok[1], "end": tok[2], "text": tok[0]}
                    out.append(run)
        else:
            out.append({"speaker": speaker_at(seg["start"], seg["end"], diar),
                        "start": seg["start"], "end": seg["end"], "text": seg["text"]})
    order = {}
    for o in out:
        o["text"] = o["text"].strip()
        if o["speaker"] not in order:
            order[o["speaker"]] = len(order) + 1
        o["speaker"] = order[o["speaker"]]
    return [o for o in out if o["text"]]


def turns_of(data):
    """Consecutive segments by the same person, merged into one turn."""
    turns = []
    for seg in data.get("segments", []):
        if turns and turns[-1]["speaker"] == seg["speaker"]:
            turns[-1]["text"] += " " + seg["text"]
            turns[-1]["end"] = seg["end"]
        else:
            turns.append(dict(seg))
    return turns


def speaker_name(data, spk):
    return data.get("names", {}).get(str(spk)) or "Speaker %d" % spk


def write_speaker_files(base, data):
    """Rewrite the .txt and .srt so they include speaker names."""
    txt = "\n\n".join("%s: %s" % (speaker_name(data, t["speaker"]), t["text"]) for t in turns_of(data)) + "\n"
    srt = "".join("%d\n%s --> %s\n%s: %s\n\n" % (i, hms(seg["start"], True), hms(seg["end"], True),
                                                speaker_name(data, seg["speaker"]), seg["text"])
                  for i, seg in enumerate(data["segments"], 1))
    atomic_write(base + ".txt", txt.encode("utf-8"))
    atomic_write(base + ".srt", srt.encode("utf-8"))
    os.utime(sidecar_path(base), None)          # mark the files as written by Matita


def plain_text(job):
    """Transcript text without speaker labels (for summaries and word counts)."""
    data = load_sidecar(job["txt"][:-4]) if job.get("txt") else None
    if data:
        return " ".join(s["text"] for s in data.get("segments", []))
    try:
        with open(job["txt"], encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except (OSError, KeyError, TypeError):
        return ""


def diarize_main(wav, out, n, threads=1):
    """Runs inside the speaker-identification environment (it has sherpa-onnx and numpy)."""
    import wave
    import numpy as np
    import sherpa_onnx
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    # thresholds tuned on a labelled five-speaker mix (TitaNet: 30% vs 36% of speech mislabelled)
    emb, thr = (EMB2_MODEL, 0.7) if os.path.isfile(EMB2_MODEL) else (EMB_MODEL, 0.9)
    cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=SEG_MODEL),
            num_threads=threads),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=emb, num_threads=threads),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=n if n > 0 else -1, threshold=thr),
        min_duration_on=0.3, min_duration_off=0.5)
    sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def cb(done, total):
        print("progress = %d%%" % int(100 * done / max(1, total)), flush=True)
        return 0
    res = sd.process(x, callback=cb).sort_by_start_time()
    with open(out, "w") as fh:
        json.dump([[r.start, r.end, r.speaker] for r in res], fh)


def parakeet_main(wav, out, threads=4):
    """Runs inside the sherpa-onnx environment: Silero VAD cuts the audio into speech chunks,
    Parakeet transcribes each one. Prints whisper.cpp-style lines so the app can show progress."""
    import wave
    import math
    import numpy as np
    import sherpa_onnx
    f = lambda n: os.path.join(PK_DIR, n)
    rec = sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f("encoder.int8.onnx"), decoder=f("decoder.int8.onnx"), joiner=f("joiner.int8.onnx"),
        tokens=f("tokens.txt"), num_threads=threads, model_type="nemo_transducer", decoding_method="greedy_search")
    with wave.open(wav) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    vc = sherpa_onnx.VadModelConfig()
    vc.silero_vad.model = f("silero_vad.onnx")
    vc.silero_vad.min_silence_duration = 0.4
    vc.silero_vad.min_speech_duration = 0.25
    vc.silero_vad.max_speech_duration = 20
    vc.sample_rate = 16000
    vad = sherpa_onnx.VoiceActivityDetector(vc, buffer_size_in_seconds=60)
    ws = vc.silero_vad.window_size
    total = max(1, len(x))
    segs = []

    def stamp(t):
        return "%02d:%02d:%06.3f" % (t // 3600, t % 3600 // 60, t % 60)

    def recognize(chunk, start, depth=0):
        st = rec.create_stream()
        st.accept_waveform(16000, chunk)
        rec.decode_stream(st)
        r = st.result
        lp = list(getattr(r, "ys_log_probs", []) or [])
        du = list(getattr(r, "durations", []) or [])
        toks = []
        for i, t in enumerate(r.tokens):
            a = start + r.timestamps[i]
            b = start + (r.timestamps[i] + du[i] if i < len(du) and du[i] > 0 else
                         (r.timestamps[i + 1] if i + 1 < len(r.timestamps) else len(chunk) / 16000.0))
            toks.append([t, round(a, 3), round(b, 3), round(math.exp(lp[i]), 4) if i < len(lp) else 1.0])
        if depth >= 2:
            return toks
        # The voice detector says this is all speech, so a long stretch with no words means the model
        # skipped it (it can, after a sudden change of voice or loudness): transcribe that part again alone.
        end = start + len(chunk) / 16000.0
        edges = [start] + [v for t in toks for v in (t[1], t[2])] + [end]
        out, last = [], 0
        for k in range(0, len(edges) - 1, 2):
            g0, g1 = edges[k], edges[k + 1]
            if g1 - g0 > 1.0:
                a, b = max(start, g0 - 0.15), min(end, g1 + 0.15)
                extra = recognize(chunk[int((a - start) * 16000):int((b - start) * 16000)], a, depth + 1)
                extra = [t for t in extra if g0 - 0.05 <= t[1] < g1 - 0.08]   # not the word right after the gap
                out += toks[last:k // 2] + extra
                last = k // 2
        return out + toks[last:] if out else toks

    def decode(chunk, start):
        toks = recognize(chunk, start)
        # one line per sentence (or every ~15 s), like Whisper's segments
        cur = []
        for k, t in enumerate(toks):
            cur.append(t)
            nxt = toks[k + 1][0] if k + 1 < len(toks) else " "
            end_sentence = re.search(r"[.?!…。？！]$", t[0].strip() or "x") and nxt.startswith(" ")
            if k + 1 == len(toks) or (end_sentence and cur[-1][2] - cur[0][1] > 1.2) or \
                    (cur[-1][2] - cur[0][1] > 15 and nxt.startswith(" ")):
                text = re.sub(r"([.?!])\1+", r"\1", "".join(c[0] for c in cur).strip())
                if text:
                    seg = {"start": cur[0][1], "end": max(cur[-1][2], cur[0][1] + 0.2), "text": text, "tokens": cur}
                    segs.append(seg)
                    print("[%s --> %s]  %s" % (stamp(seg["start"]), stamp(seg["end"]), text), flush=True)
                cur = []

    def drain():
        while not vad.empty():
            s = vad.front
            chunk, start = np.array(s.samples, dtype=np.float32), s.start / 16000.0   # copy before pop
            vad.pop()
            decode(chunk, start)

    for i in range(0, len(x), ws):
        vad.accept_waveform(x[i:i + ws])
        drain()
        if i % (ws * 200) == 0:
            print("progress = %d%%" % int(100 * i / total), flush=True)
    vad.flush()
    drain()
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(segs, fh, ensure_ascii=False)


def spk_threads():
    # Whisper runs on the GPU at the same time, so leave it some CPU cores.
    return max(2, min(8, (os.cpu_count() or 4) // 2))


class Diarizer:
    """Speaker identification, started as soon as the audio is ready so it runs while Whisper
    transcribes (Whisper uses the GPU, this uses the CPU)."""

    def __init__(self, jid, wav, n):
        self.jid, self.out, self.progress, self.tail = jid, wav + ".speakers.json", 0, []
        self.p = subprocess.Popen([SPK_PY, os.path.abspath(__file__), "--diarize", wav, self.out, str(n),
                                   str(spk_threads())],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                  errors="replace")
        register(jid, self.p, SIDE)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        for line in self.p.stdout:
            self.tail = (self.tail + [line])[-8:]
            m = PROG_RE.search(line)
            if m:
                self.progress = min(99, int(m.group(1)))

    def result(self):
        """Wait for it to finish (showing progress) and return [(start, end, speaker), ...]."""
        while self.p.poll() is None:
            update(self.jid, progress=self.progress)
            time.sleep(0.5)
        self.reader.join(5)
        with LOCK:
            SIDE.pop(self.jid, None)
        if self.p.returncode != 0 or not os.path.isfile(self.out):
            raise RuntimeError("Speaker identification failed.\n" + "".join(self.tail).strip()[-300:])
        with open(self.out) as fh:
            return [tuple(d) for d in json.load(fh)]


# ---------------------------------------------------------------- AI assist (optional)
# Two ways to get an AI summary and suggested fixes for misheard words:
#  * Ollama: an open model running on this Mac (nothing leaves the computer)
#  * Claude API: Anthropic's models, with the user's own API key (the text is sent, never the audio)
OLLAMA_URL = "http://127.0.0.1:11434"
CLAUDE_URL = "https://api.anthropic.com/v1/messages"
CLAUDE_MODELS = [("claude-haiku-5-5", "Haiku 5.5 · fastest, lowest cost"),
                 ("claude-sonnet-5-5", "Sonnet 5.5 · recommended"),
                 ("claude-opus-5-5", "Opus 5.5 · most capable, highest cost")]
DEFAULT_CLAUDE = "claude-sonnet-5-5"
OLLAMA_SUGGESTED = "gemma4:12b"
KEY_FILE = os.path.join(APP_DIR, "claude-api-key")
AI_RUNS = {}            # job id -> {"kind", "progress", "error"} while an AI task runs


def ai_path(base):
    return os.path.join(os.path.dirname(base), ".transcriber", os.path.basename(base) + ".ai.json")


def load_ai(base):
    try:
        with open(ai_path(base), encoding="utf-8") as fh:
            d = json.load(fh)
            return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_ai(base, data):
    atomic_write(ai_path(base), json.dumps(data, ensure_ascii=False).encode("utf-8"))


TITLE_PARTS = ("date", "time", "topic", "keywords", "original")
DATE_STYLES = {"iso": "%Y-%m-%d", "dmy": "%-d %b %Y", "compact": "%y%m%d"}
TITLE_SEPS = (" ", " · ", " – ", "_")
DEFAULT_FORMAT = {"parts": [{"k": "date", "on": True}, {"k": "time", "on": True}, {"k": "topic", "on": True},
                            {"k": "keywords", "on": False}, {"k": "original", "on": False}],
                  "date": "iso", "sep": " "}


def clean_format(f):
    f = f if isinstance(f, dict) else {}
    seen, parts = set(), []
    for p in f.get("parts") if isinstance(f.get("parts"), list) else []:
        if isinstance(p, dict) and p.get("k") in TITLE_PARTS and p["k"] not in seen:
            seen.add(p["k"]); parts.append({"k": p["k"], "on": bool(p.get("on"))})
    parts += [dict(p, on=p["on"] if not parts else False) for p in DEFAULT_FORMAT["parts"] if p["k"] not in seen]
    if not any(p["on"] for p in parts):
        parts = [dict(p) for p in DEFAULT_FORMAT["parts"]]
    return {"parts": parts, "date": f.get("date") if f.get("date") in DATE_STYLES else "iso",
            "sep": f.get("sep") if f.get("sep") in TITLE_SEPS else " "}


def compose_title(fmt, when, topic, keywords, original):
    t = time.localtime(when)
    style = DATE_STYLES[fmt["date"]]
    date = time.strftime(style.replace("%-d", str(t.tm_mday)), t)
    hour = "%d%s" % (t.tm_hour % 12 or 12, "am" if t.tm_hour < 12 else "pm")
    sep = fmt["sep"]
    vals = {"date": date, "time": hour, "topic": topic or "", "keywords": ("-" if sep == "_" else ", ").join(keywords[:3]),
            "original": original or ""}
    out = []
    for p in fmt["parts"]:
        if p["on"] and vals[p["k"]]:
            if p["k"] == "time" and out and out[-1][0] == "date" and sep != "_":
                out[-1] = ("date", out[-1][1] + " " + vals["time"])       # "2025-10-31 2pm" reads as one unit
                continue
            out.append((p["k"], vals[p["k"]]))
    return clean_title(sep.join(v for _, v in out), limit=215)          # room for " (2)"


def ai_config():
    c = load_config().get("ai") or {}
    return {"provider": c.get("provider") if c.get("provider") in ("off", "ollama", "claude") else "off",
            "titles": c.get("titles") if c.get("titles") in ("choose", "auto", "off") else "choose",
            "title_format": clean_format(c.get("title_format")),
            "ollama_model": c.get("ollama_model") or "",
            "claude_model": c.get("claude_model") if c.get("claude_model") in dict(CLAUDE_MODELS) else DEFAULT_CLAUDE}


def claude_key():
    try:
        with open(KEY_FILE, encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def save_claude_key(key):
    if not key:
        try:
            os.remove(KEY_FILE)
        except OSError:
            pass
        return
    os.makedirs(APP_DIR, exist_ok=True)
    fd = os.open(KEY_FILE + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(key)
    os.replace(KEY_FILE + ".tmp", KEY_FILE)
    os.chmod(KEY_FILE, 0o600)


def ollama_models():
    """Installed Ollama models, or None if Ollama isn't running."""
    import urllib.request
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/tags", timeout=2) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception:
        return None
    return [{"name": m.get("name", ""), "size": m.get("size", 0)} for m in data.get("models", [])
            if m.get("name") and "embed" not in m.get("name", "")]


def ollama_chat(model, system, user, want_json=False):
    import urllib.request
    body = {"model": model, "stream": False,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "options": {"num_ctx": 8192, "temperature": 0.2}}
    if want_json:
        body["format"] = "json"
    req = urllib.request.Request(OLLAMA_URL + "/api/chat", data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            data = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        msg = getattr(e, "read", lambda: b"")()
        raise RuntimeError("Ollama didn't answer: " + (msg.decode("utf-8", "replace")[:200] or str(e)))
    text = (data.get("message") or {}).get("content", "")
    return re.sub(r"(?s)<think>.*?</think>", "", text).strip()


def claude_request(path, body=None, key=None):
    """Call the Claude API with curl (it uses the Mac's own certificates). The key goes in a private
    header file, never on the command line where other programs could see it."""
    key = key or claude_key()
    if not key:
        raise RuntimeError("Add your Claude API key in AI assist first.")
    hdir = tempfile.mkdtemp(prefix="stt-ai-")
    hfile = os.path.join(hdir, "h")
    try:
        fd = os.open(hfile, os.O_WRONLY | os.O_CREAT, 0o600)
        with os.fdopen(fd, "w") as fh:
            fh.write("x-api-key: %s\nanthropic-version: 2023-06-01\ncontent-type: application/json\n" % key)
        cmd = [CURL, "-sS", "--max-time", "600", "-H", "@" + hfile]
        if body is not None:
            cmd += ["--data-binary", "@-"]
        r = subprocess.run(cmd + ["https://api.anthropic.com" + path], capture_output=True, text=True,
                           input=json.dumps(body) if body is not None else None, errors="replace")
    except OSError as e:
        raise RuntimeError("Couldn't run curl to reach the Claude API: %s" % e)
    finally:
        shutil.rmtree(hdir, ignore_errors=True)
    if r.returncode != 0:
        raise RuntimeError("Couldn't reach the Claude API. Check your internet connection. " + r.stderr.strip()[-160:])
    try:
        data = json.loads(r.stdout)
    except ValueError:
        raise RuntimeError("Unexpected answer from the Claude API.")
    if data.get("type") == "error" or "error" in data and isinstance(data["error"], dict):
        err = data.get("error") or {}
        kind = err.get("type", "")
        if kind == "authentication_error":
            raise RuntimeError("The Claude API key was rejected. Check it in AI assist.")
        raise RuntimeError("Claude API: " + (err.get("message") or kind or "error"))
    return data


def claude_chat(model, system, user, max_tokens=4000):
    data = claude_request("/v1/messages", {"model": model, "max_tokens": max_tokens, "system": system,
                                           "messages": [{"role": "user", "content": user}]})
    return "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text").strip()


def ai_chat(cfg, system, user, want_json=False):
    if cfg["provider"] == "ollama":
        if not cfg["ollama_model"]:
            raise RuntimeError("Choose an Ollama model in AI assist first.")
        return ollama_chat(cfg["ollama_model"], system, user, want_json)
    if cfg["provider"] == "claude":
        return claude_chat(cfg["claude_model"], system, user)
    raise RuntimeError("AI assist is off. Turn it on in Settings.")


def chunks_of(text, limit):
    """Split a transcript into pieces of at most about `limit` characters (characters, not words, so
    Japanese, Chinese and Korean are measured fairly), at line breaks where possible."""
    lines = []
    for line in text.splitlines():
        while len(line) > limit:                   # a single very long line (no breaks)
            lines.append(line[:limit])
            line = line[limit:]
        lines.append(line)
    out, cur, n = [], [], 0
    for line in lines:
        k = len(line) + 1
        if cur and n + k > limit:
            out.append("\n".join(cur))
            cur, n = [], 0
        cur.append(line)
        n += k
    if cur:
        out.append("\n".join(cur))
    return [c for c in out if c.strip()] or [text]


SUMMARY_SYSTEM = (
    "You summarize transcripts of recordings (lectures, meetings, interviews, voice notes). "
    "Write in the same language as the transcript, but keep the label 'Title:' in English. Start with three lines, "
    "each 'Title: ' followed by a short, "
    "specific title for the recording (at most 8 words, no quotes, no date); make the three different in angle "
    "(e.g. the topic, the main outcome, who or what it was about). Then use Markdown with these sections, leaving out any "
    "that don't apply: '## Overview' (one paragraph of 4-6 sentences: what it was about and what came out of it), "
    "'## Outline' (the main topics in the order they came up; each bullet starts with a short **bold topic** followed by "
    "one or two sentences), '## Key takeaways' (bullets), '## Decisions' (bullets), "
    "'## Action items' (bullets; include who, if said). The transcript came from speech recognition and "
    "may contain misheard words: read past them using context, but never invent facts, names or numbers.")
NOTES_SYSTEM = (
    "You take notes on one part of a longer transcript. Write in the transcript's language. Give concise "
    "bullet notes that keep names, numbers, decisions and action items. No introduction.")
FIX_SYSTEM = (
    "You proofread speech-recognition output. Find words the recognizer probably MISHEARD: wrong names, "
    "wrong technical terms, sound-alike words that make no sense in context. Do NOT change style, grammar, "
    "filler words or punctuation, and do not rephrase. Only suggest a fix when you are fairly confident. "
    "Answer with JSON only, in this shape: {\"fixes\": [{\"find\": \"...\", \"replace\": \"...\", \"why\": \"...\"}]}. "
    "\"find\" must be copied EXACTLY from the transcript (2-6 words, so it is unique enough); \"replace\" is the "
    "same text with only the misheard part corrected; \"why\" is a short reason in the transcript's language. "
    "At most 25 fixes. If nothing needs fixing, return {\"fixes\": []}.")


def ai_progress(jid, **kw):
    with LOCK:
        if jid in AI_RUNS:
            AI_RUNS[jid].update(kw)


_T_LABEL = r"(?:titles?|제목|タイトル)(?:\s*\d)?\s*\**\s*[:：]\s*\**"
_T_LIST = r"(?:\d+[.)]|[-*•])"


def split_titles(out):
    """Peel up to three title ideas off the top of a summary reply.
    Accepts "Title: …" lines (numbered, bulleted, bold, 제목/タイトル, full-width colon) or a
    bare "Titles:" header followed by a list. Stops at anything that looks like the summary."""
    titles = []
    hdr = re.match(r"\s*(?:#+\s*)?\**\s*(?:titles|title ideas|제목들?|タイトル)\s*\**\s*[:：]?\s*\**[ \t]*\n", out, re.I)
    if hdr:
        out = out[hdr.end():]
    line = (r"\s*(?:#+\s*)?(?:" + _T_LIST + r"\s*)?\**\s*(?:" + _T_LABEL + r")" + ("?" if hdr else "")
            + r"[ \t]*(.+?)[ \t]*(?:\n|$)")
    while len(titles) < 3:
        m = re.match(line, out, re.I)
        if not m or m.group(1).lstrip().startswith("#"):
            break
        head = out[:m.start(1)]
        if hdr and not re.search(_T_LIST + "|" + _T_LABEL, head, re.I):   # after a header, only list items
            break
        t = clean_title(m.group(1).strip().strip('"“”*'), limit=120)
        if t and t.lower() not in {x.lower() for x in titles}:
            titles.append(t)
        out = out[m.end():].lstrip()
    out = re.sub(r"\A(?:\s*(?:" + _T_LIST + r"\s*)?\**\s*" + _T_LABEL + r"[ \t]*(?:\n|\Z))+", "", out, flags=re.I)  # empty labels
    return titles, out.lstrip()


def run_ai(jid, kind, cfg):
    with LOCK:
        job = dict(JOBS.get(jid) or {})
    try:
        if not job.get("txt"):
            raise RuntimeError("This transcript is no longer in the list.")
        base = job["txt"][:-4]
        with open(job["txt"], encoding="utf-8", errors="replace") as fh:
            text = fh.read().strip()
        size = 4500 if cfg["provider"] == "ollama" else 150000        # characters; local models have small context
        parts = chunks_of(text, size)
        model = cfg["ollama_model"] if cfg["provider"] == "ollama" else dict(CLAUDE_MODELS)[cfg["claude_model"]].split(" ·")[0]
        if kind == "summary":
            if len(parts) == 1:
                ai_progress(jid, progress="Writing the summary")
                out = ai_chat(cfg, SUMMARY_SYSTEM, "Transcript:\n\n" + text)
            else:
                notes = []
                for i, p in enumerate(parts, 1):
                    ai_progress(jid, progress="Reading part %d of %d" % (i, len(parts)))
                    notes.append(ai_chat(cfg, NOTES_SYSTEM, "Part %d of %d:\n\n%s" % (i, len(parts), p)))
                ai_progress(jid, progress="Writing the summary")
                out = ai_chat(cfg, SUMMARY_SYSTEM, "These are notes on consecutive parts of one transcript. "
                              "Summarize the whole recording from them.\n\n" + "\n\n".join(notes))
            out = (out or "").strip()
            titles, out = split_titles(out)
            found = bool(titles)
            fmt = cfg.get("title_format") or clean_format(None)
            kws = summarize(plain_text(job) or text)[1] if text else []      # without "Speaker 1:" labels
            orig = os.path.basename(job.get("txt") or "")[:-4]
            made = []
            for t in (titles[:3] or [""]):
                c = compose_title(fmt, recorded_at(job), t, kws, orig)
                if c and c not in made:
                    made.append(c)
            titles = made
            title = titles[0] if titles else None
            if not out:
                raise RuntimeError("The model returned an empty summary. Try again or pick another model.")
            with LOCK:                          # the transcript may have been renamed while the model was working
                gone = jid not in JOBS
                base = (JOBS.get(jid, {}).get("txt") or base + ".txt")[:-4]
            if gone:
                with LOCK:
                    AI_RUNS.pop(jid, None)
                return
            with LOCK:
                data = load_ai(base)
                data["summary"] = {"text": out, "by": model, "created": time.time(), "title": title, "titles": titles}
                save_ai(base, data)
            prev = ai_preview(base)
            with LOCK:
                if prev and jid in JOBS:
                    JOBS[jid].update(summary=prev, summary_ai=True)
        else:
            hints = []
            if job.get("prompt"):
                hints.append("Names and terms the speaker expected: " + job["prompt"])
            unsure = sorted({w[0] for w in load_unsure(base)})
            if unsure:
                hints.append("Words the recognizer was unsure about (check these first): " + ", ".join(unsure[:150]))
            fixes, seen = [], set()
            for i, p in enumerate(parts, 1):
                ai_progress(jid, progress="Checking part %d of %d" % (i, len(parts)) if len(parts) > 1 else "Checking the transcript")
                raw = ai_chat(cfg, FIX_SYSTEM, "\n".join(hints) + "\n\nTranscript:\n\n" + p, want_json=True)
                m = re.search(r"(?s)\{.*\}", raw)
                try:
                    items = json.loads(m.group(0)).get("fixes", []) if m else []
                except (ValueError, AttributeError):
                    items = []
                for it in items if isinstance(items, list) else []:
                    if not isinstance(it, dict):
                        continue
                    f, r = str(it.get("find") or "").strip(), str(it.get("replace") or "").strip()
                    if not f or not r or f == r or f in seen or f not in text:
                        continue                     # only fixes that point at real text
                    seen.add(f)
                    fixes.append({"find": f, "replace": r, "why": str(it.get("why") or "")[:200],
                                  "count": len(re.findall(fix_pattern(f), text))})
            with LOCK:
                if jid not in JOBS or not JOBS[jid].get("txt"):   # trashed or removed meanwhile
                    AI_RUNS.pop(jid, None)
                    return
                base = JOBS[jid]["txt"][:-4]
                data = load_ai(base)
                data["fixes"] = fixes
                data["fixes_by"] = model
                data["fixes_checked"] = time.time()
                save_ai(base, data)
        with LOCK:
            AI_RUNS.pop(jid, None)
        if kind == "summary" and found and any(p["k"] == "topic" and p["on"] for p in fmt["parts"]):
            maybe_auto_rename(jid, title)
    except Exception as e:
        with LOCK:
            if jid in AI_RUNS:
                AI_RUNS[jid].update(error=str(e)[:400], kind=kind, done=True)


def start_ai(jid, kind):
    cfg = ai_config()
    if cfg["provider"] == "off":
        return False
    with LOCK:
        job = JOBS.get(jid) or {}
        if not job.get("txt") or (jid in AI_RUNS and not AI_RUNS[jid].get("done")):
            return False
        AI_RUNS[jid] = {"kind": kind, "progress": "Starting", "started": time.time()}
    threading.Thread(target=run_ai, args=(jid, kind, cfg), daemon=True).start()
    return True


def auto_title_soon(jid):
    """A new recording with a name like "New Recording 23": write its summary right away, which also names it."""
    if ai_config()["titles"] != "auto":
        return
    with LOCK:
        job = dict(JOBS.get(jid) or {})
    if job.get("txt") and is_generic(os.path.basename(job["txt"])[:-4]) and not load_ai(job["txt"][:-4]).get("summary"):
        start_ai(jid, "summary")


def maybe_auto_rename(jid, title):
    """Use the AI's title when the current name says nothing (and the user hasn't named it themselves)."""
    with LOCK:
        job = dict(JOBS.get(jid) or {})
    if not title or not job.get("txt") or ai_config()["titles"] != "auto":
        return
    base = job["txt"][:-4]
    if not is_generic(os.path.basename(base)) or load_ai(base).get("keep_name"):
        return
    old = os.path.basename(base)
    if is_generic(title) or clean_title(title) == old:      # nothing better to offer
        return
    new = None
    for cand in (title, title + " (2)", title + " (3)"):
        try:
            new = rename_transcript(jid, cand)
            break
        except ValueError as e:
            if "already has that name" not in str(e):
                return
    if not new or new == old:
        return
    with LOCK:
        nb = (JOBS.get(jid, {}).get("txt") or ".txt")[:-4]
        if nb:
            d = load_ai(nb); d["keep_name"] = True; save_ai(nb, d)
    update(jid, auto_titled={"from": old, "to": new, "at": time.time()})


def fix_pattern(find):
    pre = r"(?<!\w)" if re.match(r"\w", find) else ""
    post = r"(?!\w)" if re.search(r"\w$", find) else ""
    return pre + re.escape(find) + post


def apply_fix(job, fix):
    """Replace a misheard phrase everywhere in the transcript. Returns how many places changed."""
    base = job["txt"][:-4]
    pat = re.compile(fix_pattern(fix["find"]))
    sub = lambda t: pat.subn(lambda m: fix["replace"], t)
    side = load_sidecar(base)
    try:
        edited = side is not None and os.path.getmtime(job["txt"]) > os.path.getmtime(sidecar_path(base)) + 2
    except OSError:
        edited = False
    if side and not edited:
        # Speaker transcripts: fix the source segments, then rebuild .txt/.srt from them.
        n = 0
        for seg in side.get("segments", []):
            seg["text"], k = sub(seg["text"])
            n += k
        if n:
            save_sidecar(base, side)
            write_speaker_files(base, side)
        return n
    # Plain transcripts (or ones edited outside Matita): change the files directly.
    n = 0
    for path in (job["txt"], job.get("srt") or base + ".srt"):
        try:
            with open(path, encoding="utf-8", errors="replace") as fh:
                old = fh.read()
        except OSError:
            continue
        new, k = sub(old)
        if k:
            atomic_write(path, new.encode("utf-8"))
            n = max(n, k)
    return n


# ---------------------------------------------------------------- search across all transcripts
SEARCH_CACHE = {}       # transcript path -> (signature, [(start seconds or None, text), ...])


def searchable_lines(job):
    """The lines of a transcript with their start times (speaker turns when speakers were found)."""
    base = job["txt"][:-4]
    srt = job.get("srt") or base + ".srt"
    sig = []
    for path in (job["txt"], srt, sidecar_path(base)):
        try:
            sig.append(os.path.getmtime(path))
        except OSError:
            sig.append(None)
    hit = SEARCH_CACHE.get(job["txt"])
    if hit and hit[0] == sig:
        return hit[1]
    side = load_sidecar(base)
    if side:
        lines = [(t["start"], "%s: %s" % (speaker_name(side, t["speaker"]), t["text"])) for t in turns_of(side)]
    else:
        lines = [(x["s"], x["text"]) for x in srt_segments(srt)]
    if not lines:
        try:
            with open(job["txt"], encoding="utf-8", errors="replace") as fh:
                lines = [(None, l.strip()) for l in fh if l.strip()]
        except OSError:
            lines = []
    SEARCH_CACHE[job["txt"]] = (sig, lines)
    return lines


def search_all(q, per=4, limit=60):
    q = q.strip().lower()
    if len(q) < 2:
        return []
    with LOCK:
        done = [dict(JOBS[j]) for j in ORDER if j in JOBS and JOBS[j]["state"] == "done" and JOBS[j].get("txt")]
    done.sort(key=lambda j: -(j.get("finished") or 0))
    out = []
    for job in done:
        hits = []
        for start, text in searchable_lines(job):
            k = text.lower().find(q)
            if k < 0:
                continue
            a = max(0, k - 90)
            snippet = ("…" if a else "") + text[a:k + len(q) + 130] + ("…" if k + len(q) + 130 < len(text) else "")
            hits.append({"s": start, "t": (lambda t: t[3:] if t.startswith("00:") else t)(hms(start)) if start is not None else "",
                         "text": snippet})
        if hits:
            out.append({"id": job["id"], "name": job["name"], "count": len(hits), "hits": hits[:per]})
            if len(out) >= limit:
                break
    return out


# ---------------------------------------------------------------- editing a line in the reader
STALE = "This line changed since you opened it. Close the transcript and open it again."


def edit_plain(job, start, old, new):
    """Transcripts without speakers: change one subtitle block, then the same line in the .txt."""
    base = job["txt"][:-4]
    srt = job.get("srt") or base + ".srt"
    with open(srt, encoding="utf-8", errors="replace") as fh:
        blocks = re.split(r"\n\s*\n", fh.read().strip())
    done = False
    nth, at = 0, None                               # which subtitle (counting only real ones) was edited
    for i, block in enumerate(blocks):
        lines = block.strip().splitlines()
        if len(lines) < 3 or "-->" not in lines[1]:
            continue
        nth += 1
        if done:
            continue
        m = re.match(r"(\d+):(\d+):(\d+)[,.](\d+)", lines[1].strip())
        sec = (int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
               + int(m.group(4)[:3].ljust(3, "0")) / 1000.0) if m else -1
        text = " ".join(l.strip() for l in lines[2:]).strip()
        if abs(sec - start) < 0.6 and text == old:
            blocks[i] = "\n".join(lines[:2] + [new])
            done, at = True, nth - 1
    if not done:
        raise ValueError(STALE)
    atomic_write(srt, ("\n\n".join(b.strip() for b in blocks) + "\n").encode("utf-8"))
    with open(job["txt"], encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    lines = txt.split("\n")
    body = [k for k, l in enumerate(lines) if l.strip()]
    if len(body) == nth and at is not None and lines[body[at]].strip() == old:
        lines[body[at]] = lines[body[at]].replace(old, new)    # one line per subtitle: change the same one
    else:
        if old in txt:
            lines = txt.replace(old, new, 1).split("\n")
        else:                                       # fall back to rebuilding the text from the subtitles
            lines = [x["text"] for x in srt_segments(srt)] + [""]
    atomic_write(job["txt"], "\n".join(lines).encode("utf-8"))


def edit_turn(base, side, start, old, new):
    """Transcripts with speakers: change the segments behind one speaker turn, then rebuild the files."""
    segs = side.get("segments", [])
    i = next((k for k, sg in enumerate(segs) if abs(sg["start"] - start) < 0.01
              and (k == 0 or segs[k - 1]["speaker"] != sg["speaker"])), None)
    if i is None:
        raise ValueError(STALE)
    j = i
    while j + 1 < len(segs) and segs[j + 1]["speaker"] == segs[i]["speaker"]:
        j += 1
    part = segs[i:j + 1]
    if " ".join(sg["text"] for sg in part) != old:
        raise ValueError(STALE)
    # Find what changed; if it sits inside one segment, only that segment changes (keeps timings).
    p = 0
    while p < min(len(old), len(new)) and old[p] == new[p]:
        p += 1
    q = 0
    while q < min(len(old), len(new)) - p and old[-1 - q] == new[-1 - q]:
        q += 1
    pos = 0
    for sg in part:
        a, b = pos, pos + len(sg["text"])
        if a <= p and len(old) - q <= b:
            sg["text"] = sg["text"][:p - a] + new[p:len(new) - q] + sg["text"][len(old) - q - a:]
            break
        pos = b + 1
    else:                                           # spans segments: keep one segment for the turn
        part[0]["text"], part[0]["end"] = new, part[-1]["end"]
        del segs[i + 1:j + 1]
    side["segments"] = [sg for sg in segs if sg["text"].strip()]
    save_sidecar(base, side)
    write_speaker_files(base, side)


# ---------------------------------------------------------------- the queue survives restarts
QUEUE_DIR = os.path.join(APP_DIR, "queue")
QUEUE_FILE = os.path.join(APP_DIR, "queue.json")
QUEUE_KEYS = ("id", "name", "src", "workdir", "lang", "quality", "model", "prompt", "speakers",
              "duration", "created", "vm_uid", "recorded", "mix_src", "mix_delay")


def new_workdir():
    """Uploaded files wait here (not in a temporary folder), so they're still there after a restart."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    return tempfile.mkdtemp(prefix="stt-", dir=QUEUE_DIR)


def queue_snapshot():
    with LOCK:
        return [{k: JOBS[j].get(k) for k in QUEUE_KEYS} for j in ORDER
                if j in JOBS and JOBS[j]["state"] in ACTIVE]


def save_queue(last=[None]):
    data = json.dumps(queue_snapshot(), ensure_ascii=False)
    if data != last[0]:
        try:
            atomic_write(QUEUE_FILE, data.encode("utf-8"))
            last[0] = data
        except OSError:
            pass


STARTED = time.time()
RESTORED = threading.Event()                        # until set, queue.json still holds the last session's queue


def queue_loop():
    while True:
        save_queue()
        time.sleep(3)


def restore_queue():
    """Put back whatever was waiting or in progress when Matita last stopped. Work in progress
    starts again from the beginning of that recording."""
    try:
        with open(QUEUE_FILE, encoding="utf-8") as fh:
            saved = json.load(fh)
    except (OSError, ValueError):
        saved = []
    keep = set()
    for e in saved if isinstance(saved, list) else []:
        if not isinstance(e, dict) or not e.get("id") or not e.get("src") or not os.path.isfile(e["src"]):
            continue
        with LOCK:
            if e["id"] in JOBS:
                continue
            JOBS[e["id"]] = dict(e, state="queued", progress=0)
            ORDER.append(e["id"])
        keep.add(os.path.abspath(e.get("workdir") or ""))
        Q.put(e["id"])
    with LOCK:                                      # uploads that arrived while Matita was starting up
        keep |= {os.path.abspath(j.get("workdir") or "") for j in JOBS.values()}
    if os.path.isdir(QUEUE_DIR):                    # leftovers nobody is waiting for
        for d in os.listdir(QUEUE_DIR):
            path = os.path.abspath(os.path.join(QUEUE_DIR, d))
            try:
                fresh = os.path.getmtime(path) >= STARTED - 5
            except OSError:
                continue
            if path not in keep and not fresh:
                shutil.rmtree(path, ignore_errors=True)
    RESTORED.set()
    return len(keep)


# ---------------------------------------------------------------- export
def export_markdown(job, text, data):
    title = os.path.splitext(job["name"])[0]
    meta = " · ".join(m for m in [
        clock_str(job["duration"]) + " of audio" if job.get("duration") else "",
        "%s words" % job["words"] if job.get("words") is not None else "",
        time.strftime("%d %b %Y", time.localtime(job.get("finished") or time.time()))] if m)
    lines = ["# " + title, "", "*" + meta + "*", ""]
    ai_sum = (load_ai(job["txt"][:-4]).get("summary") or {}).get("text") if job.get("txt") else None
    if ai_sum:
        lines += [ai_sum.replace("## ", "### "), "", "---", ""]
    elif job.get("summary"):
        lines += ["> " + job["summary"], ""]
    if data:
        for t in turns_of(data):
            lines += ["**%s** · %s  " % (speaker_name(data, t["speaker"]), hms(t["start"])), t["text"], ""]
    else:
        for p in text.splitlines():
            if p.strip():
                lines += [p.strip(), ""]
    return "\n".join(lines).strip() + "\n"


def clock_str(sec):
    sec = int(round(sec or 0))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return "%d:%02d:%02d" % (h, m, s) if h else "%d:%02d" % (m, s)


def export_docx(job, text, data):
    def run(t, bold=False, italic=False, color=None, size=None):
        props = ""
        if bold:
            props += "<w:b/>"
        if italic:
            props += "<w:i/>"
        if color:
            props += '<w:color w:val="%s"/>' % color
        if size:
            props += '<w:sz w:val="%d"/><w:szCs w:val="%d"/>' % (size, size)
        return '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (
            "<w:rPr>%s</w:rPr>" % props if props else "", xml_escape(t))

    def para(*runs, after=160):
        return '<w:p><w:pPr><w:spacing w:after="%d"/></w:pPr>%s</w:p>' % (after, "".join(runs))

    title = os.path.splitext(job["name"])[0]
    meta = " · ".join(m for m in [
        clock_str(job["duration"]) + " of audio" if job.get("duration") else "",
        "%s words" % job["words"] if job.get("words") is not None else "",
        time.strftime("%d %b %Y", time.localtime(job.get("finished") or time.time()))] if m)
    body = [para(run(title, bold=True, size=40), after=80), para(run(meta, color="6E6E73", size=20), after=240)]
    if job.get("summary"):
        body.append(para(run(job["summary"], italic=True, color="3A3A3C"), after=280))
    if data:
        for t in turns_of(data):
            body.append(para(run(speaker_name(data, t["speaker"]), bold=True),
                             run("   " + hms(t["start"]), color="8E8E93", size=18), after=40))
            body.append(para(run(t["text"]), after=200))
    else:
        for p in text.splitlines():
            if p.strip():
                body.append(para(run(p.strip())))
    doc = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
           '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body>%s'
           '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1440" w:right="1440" w:bottom="1440" '
           'w:left="1440" w:header="708" w:footer="708" w:gutter="0"/></w:sectPr></w:body></w:document>') % "".join(body)
    styles = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
              '<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:docDefaults>'
              '<w:rPrDefault><w:rPr><w:rFonts w:ascii="Helvetica Neue" w:hAnsi="Helvetica Neue" w:eastAsia="Apple SD Gothic Neo" '
              'w:cs="Helvetica Neue"/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:rPrDefault>'
              '<w:pPrDefault><w:pPr><w:spacing w:line="300" w:lineRule="auto"/></w:pPr></w:pPrDefault></w:docDefaults></w:styles>')
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                   '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                   '<Default Extension="xml" ContentType="application/xml"/>'
                   '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                   '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
                   '</Types>')
        z.writestr("_rels/.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                   '</Relationships>')
        z.writestr("word/_rels/document.xml.rels",
                   '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                   '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                   '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
                   '</Relationships>')
        z.writestr("word/document.xml", doc)
        z.writestr("word/styles.xml", styles)
    return buf.getvalue()


EXPORTS = {"txt": ("text/plain; charset=utf-8", ".txt"), "srt": ("text/plain; charset=utf-8", ".srt"),
           "md": ("text/markdown; charset=utf-8", ".md"),
           "docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", ".docx")}


def build_export(job, fmt):
    """Returns (bytes, filename, content type) for a finished transcript."""
    base = job["txt"][:-4]
    ctype, ext = EXPORTS.get(fmt, EXPORTS["txt"])
    name = os.path.basename(base) + ext
    if fmt in ("txt", "srt"):
        path = job.get(fmt)
        if not path or not os.path.isfile(path):
            raise FileNotFoundError
        with open(path, "rb") as fh:
            return fh.read(), name, ctype
    with open(job["txt"], encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    data = load_sidecar(base)
    if fmt == "md":
        return export_markdown(job, text, data).encode("utf-8"), name, ctype
    return export_docx(job, text, data), name, ctype


def srt_duration(path):
    """Length of a recording, read from the last timestamp in its .srt file."""
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            ends = re.findall(r"--> (\d+):(\d+):(\d+)[,.](\d+)", fh.read())
    except OSError:
        return None
    if not ends:
        return None
    h, m, s_, ms = (int(x) for x in ends[-1])
    return h * 3600 + m * 60 + s_ + ms / 1000.0


def notify(title, message):
    if not OSASCRIPT:
        return
    try:
        subprocess.run([OSASCRIPT, "-e", "display notification %s with title %s"
                        % (json.dumps(message), json.dumps(title))], timeout=10)
    except Exception:
        pass


WORKING = {"id": None}                             # the job the worker is inside right now (even if cancelled)
BUSY_MSG = "Still stopping the last step. Try again in a few seconds."


def run_job(jid):
    with LOCK:
        if JOBS.get(jid, {}).get("state") != "queued":
            return
        WORKING["id"] = jid
    keep_awake(True)                               # keep the Mac awake meanwhile (not while paused)
    try:
        _run_job(jid)
    finally:
        with LOCK:
            WORKING["id"] = None
        keep_awake(False)
        with LOCK:
            side = SIDE.pop(jid, None)
        stop_proc(side)


def _run_job(jid):
    with LOCK:
        job = dict(JOBS.get(jid) or {})
    if not job:
        return
    workdir = job["workdir"]
    outbase = None
    try:
        model, beam = choose_config(job.get("quality"), job["lang"])
        pk = model == MODES["parakeet"]["model"]
        need = [] if pk else [(MODEL_URL + model, os.path.join(MODEL_DIR, model), model),
                              (VAD_URL, os.path.join(MODEL_DIR, VAD_MODEL), VAD_MODEL)]
        if pk:
            ensure_parakeet(jid)
            if state_of(jid) == "cancelled":
                return
        for url, dest, fname in need:
            if not (os.path.isfile(dest) and os.path.getsize(dest) > 100_000):
                mode = next((m for m in MODES.values() if m["model"] == fname), None)
                label = ("Downloading %s (one time, %s)" % (mode["label"].split(" · ")[0], mode["size"])
                         if mode else "Downloading the voice detector (one time)")
                update(jid, state="downloading", progress=0, note=label, started=time.time(), error=None)
                ensure_file(url, dest, lambda pct: update(jid, progress=pct), jid)
                if state_of(jid) == "cancelled":
                    return
        update(jid, state="converting", started=time.time(), progress=0, error=None, note=None, model=model, dl=None)
        wav = os.path.join(workdir, "audio.wav")
        mix = job.get("mix_src") if job.get("mix_src") and os.path.isfile(job["mix_src"]) else None
        if mix and is_silent(mix):
            mix = None
            update(jid, warning="Other apps' sound was silent, so only your microphone was transcribed. If you were "
                                "in a call, allow Matita Audio in System Settings → Privacy & Security → "
                                "Screen & System Audio Recording.")
        if mix:                                     # microphone + what the Mac played, lined up in time
            d = float(job.get("mix_delay") or 0)
            mic_ms, sys_ms = int(max(0, -d) * 1000), int(max(0, d) * 1000)
            conv_cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", job["src"], "-i", mix, "-filter_complex",
                        "[0:a]aformat=channel_layouts=mono,aresample=16000,adelay=%d:all=1[a];"
                        "[1:a]aformat=channel_layouts=mono,aresample=16000,adelay=%d:all=1[b];"
                        "[a][b]amix=inputs=2:duration=longest:normalize=0,alimiter=limit=0.95[o]" % (mic_ms, sys_ms),
                        "-map", "[o]", "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav]
        else:
            conv_cmd = [FFMPEG, "-nostdin", "-loglevel", "error", "-y", "-i", job["src"],
                        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", wav]
        conv = subprocess.Popen(
            conv_cmd,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, errors="replace")
        register(jid, conv)
        _, err = conv.communicate()
        with LOCK:
            PROCS.pop(jid, None)
        if state_of(jid) == "cancelled":
            return
        if conv.returncode != 0:
            raise RuntimeError("Couldn't read this file as audio. " + (err or "").strip()[-200:])

        dur = job.get("duration") or duration_of(wav)
        want = job.get("speakers", "off")
        diarizer = None
        if want != "off" and speakers_ready():
            try:
                diarizer = Diarizer(jid, wav, 0 if want == "auto" else int(want))
            except Exception:
                update(jid, warning="Speakers couldn't be identified for this recording, so the transcript has no speaker names.")
        os.makedirs(OUT_DIR, exist_ok=True)
        outbase = unique_base(os.path.splitext(job["name"])[0])
        update(jid, outbase=outbase)            # so a rename elsewhere can't take this name meanwhile
        pk_out = os.path.join(workdir, "parakeet.json")
        cmd = [SPK_PY, os.path.abspath(__file__), "--parakeet", wav, pk_out,
               str(spk_threads() if diarizer else max(2, min(8, (os.cpu_count() or 4) - 1)))] if pk else \
              [WHISPER, "-m", os.path.join(MODEL_DIR, model), "-f", wav,
               "-l", job["lang"], "-otxt", "-osrt", "-oj", "-ojf", "-of", outbase, "-pp",
               "-bs", str(beam), "-bo", str(beam)]       # beam search; temperature fallback stays on
        if not pk and whisper_has("--suppress-nst"):
            cmd += ["-sns"]                               # no "[Music]"-style sound tags
        if not pk and whisper_has("--vad"):               # older whisper.cpp builds lack these
            cmd += ["--vad", "-vm", os.path.join(MODEL_DIR, VAD_MODEL),
                    "-vsd", "1000", "-vp", "300"]         # keep pauses under 1 s inside speech
        if job.get("prompt") and not pk:
            cmd += ["--prompt", job["prompt"]]
            if whisper_has("--carry-initial-prompt"):
                cmd += ["--carry-initial-prompt"]

        update(jid, state="transcribing", duration=dur)
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             text=True, bufsize=1, errors="replace")
        register(jid, p)

        tail, said = [], []
        for line in p.stdout:
            tail = (tail + [line])[-30:]
            seg = SEG_LINE.match(line)
            if seg and seg.group(1).strip() and (not said or said[-1] != seg.group(1).strip()):
                said = (said + [seg.group(1).strip()])[-3:]
                update(jid, preview=" ".join(said)[-260:])
            m = PROG_RE.search(line)
            pct = None
            if m:
                pct = int(m.group(1))
            else:
                m = TS_RE.search(line)
                if m and dur:
                    h, mi, s, ms = m.groups()
                    t = int(h) * 3600 + int(mi) * 60 + int(s) + int(ms) / 1000.0
                    pct = int(t / dur * 100)
            if pct is not None:
                pct = max(0, min(99, pct))
                with LOCK:
                    if pct > JOBS[jid].get("progress", 0):
                        JOBS[jid]["progress"] = pct
        p.wait()
        with LOCK:
            PROCS.pop(jid, None)

        if state_of(jid) == "cancelled":
            for ext in (".txt", ".srt", ".json"):
                try:
                    os.remove(outbase + ext)
                except OSError:
                    pass
            return
        if pk and p.returncode == 0:
            try:
                with open(pk_out, encoding="utf-8") as fh:
                    pk_segs = json.load(fh)
                write_plain_files(outbase, pk_segs)
                with open(outbase + ".json", "w", encoding="utf-8") as fh:   # same shape as whisper.cpp's
                    json.dump({"transcription": [{"text": x["text"], "offsets": {"from": int(x["start"] * 1000),
                               "to": int(x["end"] * 1000)}, "tokens": [{"text": t[0], "p": t[3],
                               "offsets": {"from": int(t[1] * 1000), "to": int(t[2] * 1000)}} for t in x["tokens"]]}
                               for x in pk_segs]}, fh, ensure_ascii=False)
            except (OSError, ValueError, KeyError, IndexError):
                pass
        if p.returncode != 0 or not os.path.isfile(outbase + ".txt"):
            raise RuntimeError("Transcription stopped unexpectedly.\n" + "".join(tail[-6:]).strip())

        segs = whisper_segments(outbase + ".json")
        cleaned = drop_repeats(segs)
        if len(cleaned) < len(segs):
            segs = cleaned
            write_plain_files(outbase, segs)
        try:
            os.remove(outbase + ".json")
        except OSError:
            pass
        save_audio(wav, outbase)                  # runs while speaker identification finishes
        if state_of(jid) == "cancelled":
            raise RuntimeError("cancelled")
        try:
            atomic_write(unsure_path(outbase), json.dumps(unsure_words(segs, job["lang"]),
                                                          ensure_ascii=False).encode("utf-8"))
        except OSError:
            pass

        speaker_count = 0
        if diarizer and segs:
            update(jid, state="speakers", progress=diarizer.progress)
            try:
                diar = diarizer.result()
            except Exception:
                if state_of(jid) == "cancelled":
                    raise
                diar = None
                update(jid, warning="Speakers couldn't be identified for this recording, so the transcript has no speaker names.")
            if state_of(jid) == "cancelled":
                raise RuntimeError("cancelled")
            if diar:
                data = {"version": 1, "names": {}, "segments": assign_speakers(segs, diar)}
                speaker_count = len({x["speaker"] for x in data["segments"]})
                save_sidecar(outbase, data)
                write_speaker_files(outbase, data)

        job_now = {"txt": outbase + ".txt"}
        text = plain_text(job_now).strip()
        if os.path.abspath(outbase + ".txt") in hidden_paths():
            set_hidden(remove=[outbase + ".txt"])
        save_meta(outbase, recorded=recorded_at(job))
        update(jid, state="done", progress=100, preview=None, words=count_words(text), summary=summarize(text)[0],
               recorded=recorded_at(job),
               name=os.path.basename(outbase) + ".txt", txt=outbase + ".txt", srt=outbase + ".srt", speaker_count=speaker_count,
               has_srt=os.path.isfile(outbase + ".srt"), finished=time.time())
        if job.get("vm_uid"):
            edit_config(lambda c: (c.setdefault("vm_done", {}).__setitem__(job["vm_uid"], outbase + ".txt"),
                                   c.setdefault("vm_failed", {}).pop(job["vm_uid"], None)))
        try:
            auto_title_soon(jid)
        except Exception:
            pass
        BATCH["done"] += 1
        shutil.rmtree(workdir, ignore_errors=True)
    except Exception as e:
        if state_of(jid) == "cancelled" and outbase:
            for ext in (".txt", ".srt", ".json"):
                try:
                    os.remove(outbase + ext)
                except OSError:
                    pass
            for extra in (sidecar_path(outbase), audio_path(outbase), unsure_path(outbase), ai_path(outbase)):
                try:
                    os.remove(extra)
                except OSError:
                    pass
        if state_of(jid) != "cancelled":
            update(jid, state="error", error=str(e), finished=time.time())
            BATCH["failed"] += 1
            if job.get("vm_uid"):
                edit_config(lambda c: c.setdefault("vm_failed", {}).__setitem__(job["vm_uid"], time.time()))


def worker():
    while True:
        jid = Q.get()
        RUNNING.wait()                              # paused: don't start the next recording
        try:
            run_job(jid)
        except Exception as e:                      # never let one file stop the queue
            update(jid, state="error", error="Unexpected problem: %s" % e, finished=time.time())
        finally:
            Q.task_done()
        with LOCK:
            busy = any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS)
        if not busy and (BATCH["done"] or BATCH["failed"]):
            d, f = BATCH["done"], BATCH["failed"]
            msg = "%d file%s transcribed" % (d, "" if d == 1 else "s")
            if f:
                msg += ", %d failed" % f
            notify("Matita", msg)
            BATCH["done"] = BATCH["failed"] = 0


# ---------------------------------------------------------------- Voice Memos
VM_DIRS = [os.path.join(HOME, "Library", "Group Containers", "group.com.apple.VoiceMemos.shared", "Recordings"),
           os.path.join(HOME, "Library", "Application Support", "com.apple.voicememos", "Recordings")]
VM_PLACEHOLDER = 10000          # files smaller than this haven't been downloaded from iCloud yet
FDA_URL = "x-apple.systempreferences:com.apple.preference.security?Privacy_AllFiles"


def vm_dir():
    """The Voice Memos folder on this Mac, or None. Raises PermissionError if macOS blocks access."""
    blocked = False
    for d in VM_DIRS:
        try:
            names = os.listdir(d)
        except FileNotFoundError:
            continue
        except PermissionError:
            blocked = True
            continue
        if "CloudRecordings.db" in names:
            return d
    if blocked:
        raise PermissionError
    return None


def responsible_app():
    """What the person has to add under Full Disk Access for the app to read Voice Memos."""
    if BUNDLE:
        return {"kind": "app", "name": "Matita", "path": BUNDLE}
    if os.environ.get("XPC_SERVICE_NAME") in [LABEL] + OLD_LABELS and (os.environ.get("MATITA_SERVICE_APP")
                                                                    or os.environ.get("TRANSCRIBER_SERVICE_APP")):
        return {"kind": "app", "name": "Matita Service", "path": SERVICE_APP}
    if os.environ.get("XPC_SERVICE_NAME") == LABEL:          # older installs: Python runs the service directly
        exe = ""
        try:
            exe = subprocess.run(["ps", "-o", "comm=", "-p", str(os.getpid())],
                                 capture_output=True, text=True).stdout.strip()
        except Exception:
            pass
        m = re.match(r"(.*?\.app)/Contents/MacOS/", exe) if exe.startswith("/") else None
        if m:
            path = m.group(1)                                   # e.g. .../Python3.framework/.../Python.app
        else:
            bundle = os.path.join(getattr(sys, "base_prefix", sys.prefix), "Resources", "Python.app")
            path = bundle if os.path.isdir(bundle) else os.path.realpath(sys.executable)
        return {"kind": "python", "name": os.path.basename(path), "path": path}
    return {"kind": "terminal", "name": "Terminal", "path": "/System/Applications/Utilities/Terminal.app"}


def vm_query(db):
    """Run the recordings query; copy the database first if it's busy."""
    def run(conn):
        cols = {r[1] for r in conn.execute("PRAGMA table_info(ZCLOUDRECORDING)")}
        title = "ZENCRYPTEDTITLE" if "ZENCRYPTEDTITLE" in cols else ("ZCUSTOMLABEL" if "ZCUSTOMLABEL" in cols else "NULL")
        uid = "ZUNIQUEID" if "ZUNIQUEID" in cols else "ZPATH"
        dur = "ZDURATION" if "ZDURATION" in cols else "NULL"
        where = "ZPATH IS NOT NULL" + (" AND ZEVICTIONDATE IS NULL" if "ZEVICTIONDATE" in cols else "")
        return conn.execute("SELECT %s, ZPATH, %s, ZDATE, %s FROM ZCLOUDRECORDING WHERE %s ORDER BY ZDATE DESC"
                            % (uid, title, dur, where)).fetchall()
    try:
        conn = sqlite3.connect("file:%s?mode=ro" % urllib.parse.quote(db), uri=True, timeout=5)
        try:
            return run(conn)
        finally:
            conn.close()
    except sqlite3.OperationalError:
        tmp = tempfile.mkdtemp()
        try:
            for ext in ("", "-wal"):
                if os.path.exists(db + ext):
                    shutil.copyfile(db + ext, os.path.join(tmp, "db.sqlite" + ext))
            conn = sqlite3.connect(os.path.join(tmp, "db.sqlite"))
            try:
                return run(conn)
            finally:
                conn.close()
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


def safe_title(t, fallback="Voice memo"):
    """A name that is safe as a macOS file name: no slashes or colons, not hidden, under 200 bytes."""
    t = re.sub(r"[/:\\\x00-\x1f]", "-", t or "").strip().lstrip(". ")
    b = t.encode("utf-8")[:200]
    t = b.decode("utf-8", "ignore").strip()
    return t or fallback


def list_voice_memos():
    """{"status": ok|needs_access|not_found|error, "memos": [...]}"""
    try:
        d = vm_dir()
    except PermissionError:
        return {"status": "needs_access", "app": responsible_app()}
    if not d:
        return {"status": "not_found"}
    try:
        rows = vm_query(os.path.join(d, "CloudRecordings.db"))
    except PermissionError:
        return {"status": "needs_access", "app": responsible_app()}
    except sqlite3.DatabaseError as e:
        if "authoriz" in str(e) or "unable to open" in str(e):
            return {"status": "needs_access", "app": responsible_app()}
        return {"status": "error", "message": "Couldn't read the Voice Memos library: %s" % e}
    cfg = load_config()
    done, failed = cfg.get("vm_done", {}), cfg.get("vm_failed", {})
    with LOCK:
        queued = {JOBS[j].get("vm_uid") for j in ORDER if j in JOBS and JOBS[j]["state"] in ACTIVE}
    memos = []
    for uid, rel, title, zdate, dur in rows:
        path = os.path.join(d, os.path.basename(rel or ""))
        try:
            st = os.stat(path)
            size, mtime = st.st_size, st.st_mtime
        except OSError:
            size, mtime = 0, 0
        uid = str(uid)
        memos.append({
            "uid": uid, "title": (title or "").strip() or "Voice memo", "ext": os.path.splitext(path)[1].lstrip(".").lower(),
            "file": os.path.basename(path),
            "date": (zdate or 0) + 978307200, "duration": dur, "local": size >= VM_PLACEHOLDER, "mtime": mtime,
            "done": uid in done and (done[uid] == "trashed" or os.path.exists(done[uid])), "failed": uid in failed, "queued": uid in queued,
        })
    return {"status": "ok", "memos": memos, "folder": d.replace(HOME, "~", 1), "auto": cfg.get("vm_auto", {})}


def enqueue_voice_memo(uid, opts, auto=False):
    """Copy one memo into a work folder and queue it, named after its title."""
    if not auto:                                # picked by hand: auto-import may take it again later
        edit_config(lambda c: c.get("vm_skip", {}).pop(str(uid), None) if isinstance(c.get("vm_skip"), dict) else None)
    d = vm_dir()
    if not d:
        raise RuntimeError("Voice Memos library not found.")
    rows = [r for r in vm_query(os.path.join(d, "CloudRecordings.db")) if str(r[0]) == str(uid)]
    if not rows:
        raise RuntimeError("That recording is no longer in Voice Memos.")
    _, rel, title, zdate, dur = rows[0]
    src_file = os.path.join(d, os.path.basename(rel))
    if not os.path.exists(src_file) or os.path.getsize(src_file) < VM_PLACEHOLDER:
        raise RuntimeError("This recording is still in iCloud. Play it once in Voice Memos to download it.")
    return queue_local(src_file, safe_title(title), opts, duration=dur, vm_uid=str(uid),
                       recorded=(zdate + 978307200) if zdate else None)


def queue_local(path, title, opts, duration=None, **extra):
    """Copy a file on this Mac into a work folder (it may be on a card that gets ejected) and queue it."""
    ext = os.path.splitext(path)[1].lower()
    workdir = new_workdir()
    src = os.path.join(workdir, "input" + ext)
    try:
        shutil.copyfile(path, src)
    except BaseException:
        shutil.rmtree(workdir, ignore_errors=True)
        raise
    jid = uuid.uuid4().hex[:10]
    job = {"id": jid, "name": (title or "Recording") + ext, "state": "queued", "progress": 0, "created": time.time(),
           "lang": opts.get("lang", "en"), "quality": clean_quality(opts.get("quality")),
           "model": choose_config(opts.get("quality"), opts.get("lang", "en"))[0],
           "prompt": opts.get("prompt", ""), "speakers": opts.get("speakers", "off"),
           "src": src, "workdir": workdir, "duration": duration or duration_of(src)}
    job.update(extra)
    with LOCK:
        JOBS[jid] = job
        ORDER.append(jid)
    Q.put(jid)
    return jid


# ---------------------------------------------------------------- Markdown notes (Obsidian)
# One note per transcript, in a folder you choose (an Obsidian vault works well): YAML properties,
# the AI summary, [[links]] to related recordings, and the transcript. Rewritten when the transcript,
# its summary or its name changes, but never over a note you've edited yourself.
NOTE_SIG = {}                                       # transcript path -> what its note was last built from
NOTE_KW = {}                                        # transcript path -> (mtime, keywords)


def notes_cfg():
    n = load_config().get("notes")
    n = n if isinstance(n, dict) else {}
    d = n.get("dir") if isinstance(n.get("dir"), str) and n.get("dir") else os.path.join(OUT_DIR, "Notes")
    return {"on": bool(n.get("on")), "dir": d}


def note_name(base):
    """Obsidian can't link to names with # | ^ [ ]."""
    return re.sub(r"[#|^\[\]]", "-", os.path.basename(base))


def _sha(b):
    return hashlib.sha1(b).hexdigest()


def note_tag(k):
    t = re.sub(r"[^\w/-]+", "-", (k or "").strip().lower()).strip("-/")
    return t if t and not t.isdigit() else ""


def note_keywords(txt):
    try:
        m = os.path.getmtime(txt)
    except OSError:
        return []
    c = NOTE_KW.get(txt)
    if c and c[0] == m:
        return c[1]
    try:
        kws = summarize(plain_text({"txt": txt}))[1]
    except Exception:
        kws = []
    NOTE_KW[txt] = (m, kws)
    return kws


def build_note(job, related):
    base = job["txt"][:-4]
    title = os.path.basename(base)
    with open(job["txt"], encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    data = load_sidecar(base)
    y = lambda v: json.dumps(v, ensure_ascii=False)     # JSON strings and lists are valid YAML
    when = recorded_at(job)
    fm = ["---", "title: " + y(title), "date: " + time.strftime("%Y-%m-%dT%H:%M", time.localtime(when))]
    if job.get("duration"):
        fm.append("duration: " + y(clock_str(job["duration"])))
    if job.get("words") is not None:
        fm.append("words: %d" % job["words"])
    if job.get("lang") and job.get("lang") != "auto":
        fm.append("language: " + y(job["lang"]))
    if data:
        fm.append("speakers: " + y([speaker_name(data, k) for k in sorted({x["speaker"] for x in data["segments"]})]))
    tags = []
    for t in ["matita"] + [note_tag(k) for k in note_keywords(job["txt"])[:6]]:
        if t and t not in tags:
            tags.append(t)
    fm.append("tags: " + y(tags))
    if job.get("summary"):
        fm.append("summary: " + y(job["summary"]))
    fm += ["transcript: " + y(job["txt"].replace(HOME, "~", 1)), "generated_by: matita", "---", ""]
    lines = ["# " + title, ""]
    ai_sum = (load_ai(base).get("summary") or {}).get("text")
    if ai_sum:
        ai_sum = re.sub(r"\A\s*#+\s*(?:summary|요약|résumé)\s*\n", "", ai_sum, flags=re.I)   # we add our own heading
        lines += ["## Summary", "", re.sub(r"(?m)^(#{1,5}) ", lambda m: "#" * min(6, len(m.group(1)) + 1) + " ", ai_sum).strip(), ""]
    elif job.get("summary"):
        lines += ["> " + job["summary"], ""]
    if related:
        lines += ["## Related", ""] + ["- [[%s]] · %s" % (n, why) for n, why in related] + [""]
    lines += ["## Transcript", ""]
    if data:
        for t in turns_of(data):
            lines += ["**%s** · %s  " % (speaker_name(data, t["speaker"]), hms(t["start"])), t["text"], ""]
    else:
        lines += [p.strip() + "\n" for p in text.splitlines() if p.strip()]
    return ("\n".join(fm + lines).strip() + "\n").encode("utf-8")


def note_owned(base, d=None):
    """The note for this transcript, if Matita wrote it and nobody has changed it since."""
    meta = load_meta(base)
    path = os.path.join(d or meta.get("note_dir") or notes_cfg()["dir"], note_name(base) + ".md")
    try:
        with open(path, "rb") as fh:
            ok = meta.get("note_hash") and _sha(fh.read()) == meta["note_hash"]
    except OSError:
        return None
    return path if ok else None


def write_note(job, related, d):
    base = job["txt"][:-4]
    path = os.path.join(d, note_name(base) + ".md")
    body = build_note(job, related)
    meta = load_meta(base)
    if os.path.exists(path):
        try:
            with open(path, "rb") as fh:
                cur = fh.read()
        except OSError:
            return "error"
        if cur == body:
            if meta.get("note_hash") != _sha(body):
                save_meta(base, note_hash=_sha(body), note_dir=d)
            return "same"
        if meta.get("note_hash") != _sha(cur):                                # you edited it, or it's a note of your own: leave it
            return "kept"
    atomic_write(path, body)
    save_meta(base, note_hash=_sha(body), note_dir=d)
    return "written"


def related_of(jobs):
    """For each transcript: recordings from the same day, then ones sharing at least two keywords."""
    facts = []
    for j in jobs:
        kws = {k.lower() for k in note_keywords(j["txt"])}
        t = recorded_at(j)
        facts.append((note_name(j["txt"][:-4]), time.localtime(t)[:3], t, kws))
    out = {}
    for i, (name, day, t, kws) in enumerate(facts):
        topic, same = [], []
        for k, (n2, d2, t2, k2) in enumerate(facts):
            if k == i:
                continue
            shared = sorted(kws & k2)
            if len(shared) >= 2:
                topic.append((-len(shared), abs(t - t2), n2, "also about " + ", ".join(shared[:3])))
            elif d2 == day:
                same.append((abs(t - t2), n2, "same day"))
        topic.sort()
        same.sort()
        out[jobs[i]["txt"]] = [(n, why) for *_, n, why in topic[:4]] + [(n, why) for _, n, why in same[:3]]
    return out


def notes_pass(force=False):
    nc = notes_cfg()
    if not nc["on"]:
        return 0
    d = nc["dir"]
    try:
        os.makedirs(d, exist_ok=True)
    except OSError:
        return 0
    with LOCK:
        jobs = [dict(JOBS[j]) for j in ORDER if j in JOBS and JOBS[j].get("state") == "done" and JOBS[j].get("txt")
                and not (AI_RUNS.get(j) and not AI_RUNS[j].get("done"))]
    jobs = [j for j in jobs if os.path.isfile(j["txt"])]
    rel = related_of(jobs)
    n = 0
    for j in jobs:
        base = j["txt"][:-4]
        sig = [d, j.get("name"), j.get("duration"), j.get("words"), rel.get(j["txt"])]
        for f in (j["txt"], sidecar_path(base), os.path.join(os.path.dirname(base), ".transcriber",
                                                            os.path.basename(base) + ".ai.json")):
            try:
                sig.append(os.path.getmtime(f))
            except OSError:
                sig.append(None)
        if not force and NOTE_SIG.get(j["txt"]) == sig:
            continue
        try:
            if write_note(j, rel.get(j["txt"]), d) == "written":
                n += 1
        except Exception:
            pass
        NOTE_SIG[j["txt"]] = sig
    return n


def notes_loop():
    while True:
        time.sleep(6)
        try:
            notes_pass()
        except Exception:
            pass


def trashable(base):
    """A transcript's files for the Trash. A note you edited stays, even when notes live next to the transcripts."""
    note, md = note_owned(base), base + ".md"
    here = os.path.abspath(load_meta(base).get("note_dir") or notes_cfg()["dir"]) == os.path.abspath(os.path.dirname(base))
    return [p for p, _, _ in files_of(base) if not (here and p == md and p != note)] + \
           ([note] if note and note != md else [])


def note_follow_rename(old_base, new_base):
    """Move the note along with its transcript (keeps your edits; Matita rewrites it only if it's untouched)."""
    meta = load_meta(new_base)                      # the meta file has already moved
    d = meta.get("note_dir")
    if not d:
        return
    a = os.path.join(d, note_name(old_base) + ".md")
    b = os.path.join(d, note_name(new_base) + ".md")
    if os.path.isfile(a) and not os.path.exists(b):
        try:
            os.rename(a, b)
        except OSError:
            pass


# ---------------------------------------------------------------- watch folders
# Any folder (a recorder's SD card, Zoom's recordings folder, a Dropbox folder): new audio or video
# files that appear there are copied and transcribed. Files are remembered by path and size, so a
# card that's plugged in again isn't transcribed twice.
MEDIA_RE = re.compile(r"\.(m4a|mp3|wav|aac|flac|ogg|opus|mp4|mov|m4v|qta|caf|aif|aiff|webm|wma|amr)$", re.I)
WATCH_PENDING = {}                                  # path -> (size, mtime) seen once, waiting to stop changing
WATCH_STATE = {}                                    # folder -> status for the settings screen


def watch_folders():
    w = load_config().get("watch")
    return [f for f in (w if isinstance(w, list) else []) if isinstance(f, dict) and isinstance(f.get("path"), str)]


def media_in(root, depth=3, cap=5000):
    skip = {os.path.abspath(QUEUE_DIR), os.path.abspath(notes_cfg()["dir"])}
    out = []

    def go(d, n):
        try:
            it = os.scandir(d)
        except OSError:
            return
        with it:
            for e in it:
                if len(out) >= cap or e.name.startswith("."):
                    continue
                try:
                    if e.is_dir(follow_symlinks=False):
                        if n < depth and os.path.abspath(e.path) not in skip:
                            go(e.path, n + 1)
                    elif e.is_file() and MEDIA_RE.search(e.name):
                        st = e.stat()
                        out.append((e.path, st.st_size, st.st_mtime, getattr(st, "st_birthtime", None)))
                except OSError:
                    continue
    go(root, 0)
    return out


def check_watch_path(path):
    path = os.path.abspath(os.path.expanduser((path or "").strip()))
    if not os.path.isdir(path):
        raise ValueError("That folder doesn't exist. Connect the drive or pick another folder.")
    if path in ("/", HOME, "/Volumes", "/Users", os.path.join(HOME, "Library")):
        raise ValueError("That's too big to watch. Pick the folder your recordings go into.")
    try:
        d = vm_dir()
    except PermissionError:                         # no Full Disk Access: still recognise the folder by its path
        d = next((x for x in VM_DIRS if os.path.isdir(x)), None)
    if d and (path == d or path.startswith(d + os.sep)):
        raise ValueError("Voice Memos already has its own switch: turn on “Transcribe new memos automatically”.")
    if path == os.path.abspath(QUEUE_DIR) or path.startswith(os.path.abspath(QUEUE_DIR) + os.sep):
        raise ValueError("That's Matita's own work folder. Pick another folder.")
    return path


def watch_add(path, opts):
    """Start watching; what's already there is left alone (watch_existing() queues it on request)."""
    path = check_watch_path(path)
    if any(os.path.abspath(f["path"]) == path for f in watch_folders()):
        raise ValueError("Matita is already watching that folder.")
    base = {p: [size, mt, "b"] for p, size, mt, _ in media_in(path)}

    def change(c):
        c["watch"] = [f for f in watch_folders()] + [{"path": path, "opts": clean_opts(opts), "added": time.time()}]
        seen = c.get("watch_seen") if isinstance(c.get("watch_seen"), dict) else {}
        for p, v in base.items():
            seen.setdefault(p, v)
        c["watch_seen"] = seen
    edit_config(change)
    WATCH_STATE[path] = {"status": "ok", "checked": time.time()}
    return path, len(base)


def watch_existing(path):
    """Also transcribe the files that were in the folder when you started watching it."""
    path = os.path.abspath(path)
    pre = path + os.sep
    n = [0]

    def change(c):
        seen = c.get("watch_seen") if isinstance(c.get("watch_seen"), dict) else {}
        for p in [p for p, v in seen.items() if p.startswith(pre) and isinstance(v, list) and v[2:3] == ["b"]]:
            seen.pop(p)
            n[0] += 1
    edit_config(change)
    return n[0]


def watch_remove(path):
    path = os.path.abspath(path)
    edit_config(lambda c: c.__setitem__("watch", [f for f in watch_folders() if os.path.abspath(f["path"]) != path]))
    WATCH_STATE.pop(path, None)


WATCH_LOCK = threading.Lock()


def watch_pass():
    if not WATCH_LOCK.acquire(blocking=False):      # a pass is already running
        return 0
    try:
        return _watch_pass()
    finally:
        WATCH_LOCK.release()


def _remember_seen(items):
    def f(c):
        if not isinstance(c.get("watch_seen"), dict):
            c["watch_seen"] = {}
        c["watch_seen"].update(items)
    edit_config(f)


def _watch_pass():
    folders = watch_folders()
    if not folders:
        return 0
    seen = load_config().get("watch_seen")
    seen = seen if isinstance(seen, dict) else {}
    done, queued = {}, 0
    for f in folders:
        root = os.path.abspath(f["path"])
        if not os.path.isdir(root):
            WATCH_STATE[root] = {"status": "offline", "checked": time.time()}
            continue
        waiting = 0
        for p, size, mt, bt in media_in(root):
            old = seen.get(p)
            if (isinstance(old, list) and old and old[0] == size and old[2:3] != ["failed"]) or p in done or size < 2048:
                continue                            # transcribed (or skipped) before, or still empty
            if WATCH_PENDING.get(p) != (size, mt) or time.time() - mt < 8:
                WATCH_PENDING[p] = (size, mt)       # still being copied or recorded: look again next time
                waiting += 1
                continue
            WATCH_PENDING.pop(p, None)
            try:
                queue_local(p, clean_title(os.path.splitext(os.path.basename(p))[0]) or "Recording",
                            f.get("opts") or {}, recorded=bt or mt, watch_src=p)
                done[p] = [size, mt]
                queued += 1
            except Exception:
                done[p] = [size, mt, "failed"]
            _remember_seen({p: done[p]})            # right away, so a restart mid-pass can't queue it twice
        WATCH_STATE[root] = {"status": "ok", "waiting": waiting, "checked": time.time()}
    return queued


def watch_loop():
    time.sleep(5)
    while True:
        try:
            watch_pass()
        except Exception:
            pass
        time.sleep(15)


def folders_info():
    nc = notes_cfg()
    out = []
    for f in watch_folders():
        root = os.path.abspath(f["path"])
        st = WATCH_STATE.get(root) or {}
        out.append({"path": root, "show": root.replace(HOME, "~", 1),
                    "status": "offline" if not os.path.isdir(root) else "ok", "waiting": st.get("waiting", 0)})
    return {"notes": {"on": nc["on"], "dir": nc["dir"], "show": nc["dir"].replace(HOME, "~", 1)}, "watch": out}


def choose_folder(prompt, start):
    """macOS folder picker. Returns a path, or None if cancelled. Raises ValueError if it can't open."""
    if not OSASCRIPT:
        raise ValueError("The folder picker isn't available. Paste a folder path instead.")
    start = start if os.path.isdir(start) else HOME
    script = ['-e', 'on run argv', '-e', 'activate', '-e',
              'POSIX path of (choose folder with prompt (item 1 of argv) default location (POSIX file (item 2 of argv)))',
              '-e', 'end run']
    try:
        r = subprocess.run([OSASCRIPT] + script + [prompt, start], capture_output=True, text=True, timeout=600)
    except subprocess.TimeoutExpired:
        return None
    if r.returncode != 0:
        if "-128" in r.stderr:
            return None
        raise ValueError("Couldn't open the folder picker. Paste a folder path instead.")
    return r.stdout.strip().rstrip("/") or "/"


def clean_opts(o):
    o = o or {}
    sp = str(o.get("speakers") or "off")
    return {"lang": o.get("lang") if o.get("lang") in LANGS else "en",
            "quality": clean_quality(o.get("quality")),
            "prompt": str(o.get("prompt") or "")[:500],
            "speakers": sp if (sp in ("off", "auto") or sp.isdigit()) and (sp == "off" or speakers_ready()) else "off"}


def vm_auto_loop():
    """Every two minutes, queue new Voice Memos recorded after auto-transcribe was turned on."""
    while True:
        time.sleep(120)
        try:
            cfg = load_config()
            auto, skipped = cfg.get("vm_auto", {}), cfg.get("vm_skip", {})
            if not auto.get("enabled"):
                continue
            res = list_voice_memos()
            if res["status"] != "ok":
                continue
            for m in res["memos"]:
                if (m["date"] >= auto.get("since", 0) and m["local"] and not m["done"]
                        and not m["queued"] and not m["failed"] and m["uid"] not in skipped
                        and (m["duration"] or 0) > 0 and time.time() - m["mtime"] > 90):
                    # skip recordings that are still being recorded or downloaded from iCloud
                    try:
                        real = duration_of(os.path.join(vm_dir(), "%s" % m.get("file", "")))
                    except Exception:
                        real = None
                    if real is not None and abs(real - m["duration"]) > max(2.0, 0.05 * m["duration"]):
                        continue
                    try:
                        enqueue_voice_memo(m["uid"], auto.get("opts", {}), auto=True)
                    except Exception:
                        edit_config(lambda c, u=m["uid"]: c.setdefault("vm_failed", {}).__setitem__(u, time.time()))
        except Exception:
            pass


def hidden_paths():
    """Transcripts the user took off the list (the files stay in the folder)."""
    h = load_config().get("hidden")
    return set(h) if isinstance(h, list) else set()


def set_hidden(add=(), remove=()):
    def change(c):
        h = set(c.get("hidden") or [])
        h |= {os.path.abspath(p) for p in add}
        h -= {os.path.abspath(p) for p in remove}
        h = {p for p in h if os.path.exists(p) or not os.path.isdir(os.path.dirname(p))}   # keep ones on an unplugged drive
        c["hidden"] = sorted(h)
    edit_config(change)


# ---------------------------------------------------------------- rename, trash
MAIN_EXTS = (".txt", ".srt", ".json", ".md", ".docx", ".vtt")
SIDE_EXTS = (".json", ".unsure.json", ".m4a", ".peaks.json", ".ai.json", ".meta.json")      # in .transcriber/


def meta_path(base):
    return os.path.join(os.path.dirname(base), ".transcriber", os.path.basename(base) + ".meta.json")


def load_meta(base):
    try:
        with open(meta_path(base), encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_meta(base, **kw):
    d = load_meta(base)
    d.update(kw)
    try:
        os.makedirs(os.path.dirname(meta_path(base)), exist_ok=True)
        atomic_write(meta_path(base), json.dumps(d).encode("utf-8"))
    except OSError:
        pass


def recorded_at(job):
    """When the recording itself was made (file date, Voice Memo date); falls back to when it was added."""
    t = job.get("recorded")
    if not t and job.get("txt"):
        t = load_meta(job["txt"][:-4]).get("recorded")
        if not t:
            try:
                st = os.stat(job["txt"])
                t = getattr(st, "st_birthtime", None) or st.st_mtime
            except OSError:
                t = None
    for v in (t, job.get("created")):
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if 946684800 < v < time.time() + 86400:
            return v
    return time.time()


def stamp(ts):
    """Date and the hour only: "2025-10-31 2pm"."""
    t = time.localtime(ts)
    h = t.tm_hour % 12 or 12
    return "%s %d%s" % (time.strftime("%Y-%m-%d", t), h, "am" if t.tm_hour < 12 else "pm")


_DATE = r"\d{4}[-._]?\d{2}[-._]?\d{2}(?:[\s_T-]*(?:at\s*)?(?:\d{1,2}\s*(?:am|pm)|\d{1,2}[.:h-]?\d{2}(?:[.:-]?\d{2})?))?"
GENERIC_NAME = re.compile(r"""^(?:
    (?:new\s+)?(?:recording|record|rec|audio|voice\s*memo|memo|untitled|meeting|track|sound|clip|video|movie|
       screen\s*recording|zoom|call|new|녹음|새로운\s*녹음|음성\s*메모)(?:[\s_-]*(?:""" + _DATE + r"""|\d+))?(?:\s*\(\d+\))?
  | """ + _DATE + r"""(?:[\s_-]*\d+)?(?:\s*\(\d+\))?
  | (?:img|vid|rec|aud|dsc|mov|dji|gopr)[\s_-]*\d+(?:\s*\(\d+\))?
  | gmt\d{8}-\d{6}.*
  | [\d\s_.()-]+
)$""", re.I | re.X)


def is_generic(name):
    stem = os.path.splitext(name or "")[0] if (name or "").lower().endswith((".txt", ".m4a", ".mp3", ".wav", ".mov", ".mp4", ".qta", ".caf")) else (name or "")
    return bool(GENERIC_NAME.match(stem.strip()))


def files_of(base):
    """Every file that belongs to one transcript: (path, kind, ext)."""
    d, b = os.path.dirname(base), os.path.basename(base)
    out = [(base + e, "main", e) for e in MAIN_EXTS if os.path.isfile(base + e)]
    out += [(os.path.join(d, ".transcriber", b + e), "side", e) for e in SIDE_EXTS
            if os.path.isfile(os.path.join(d, ".transcriber", b + e))]
    return out


def clean_title(name, limit=228):
    name = name if isinstance(name, str) else ""
    name = re.sub(r"[/:\x00-\x1f]", "-", name.strip()).strip(" .")
    name = re.sub(r"\.(meta|ai|unsure|peaks)$", r" \1", name, flags=re.I)   # would clash with our side files
    b = name.encode("utf-8")                     # APFS allows 255 bytes; leave room for ".unsure.json" etc.
    if len(b) > limit:
        name = b[:limit].decode("utf-8", "ignore").rstrip(" .")
    return name


def rename_transcript(jid, name):
    name = clean_title(name)
    if not name:
        raise ValueError("Type a name first.")
    with LOCK:
        job = JOBS.get(jid)
        if not job or job.get("state") != "done" or not job.get("txt"):
            raise ValueError("This transcript can't be renamed right now.")
        run = AI_RUNS.get(jid)
        if run and not run.get("done"):
            raise ValueError("The AI is still working on this transcript. Rename it when it's done.")
        old = job["txt"][:-4]
        new = os.path.join(os.path.dirname(old), name)
        if new == old:
            return name
        busy = {os.path.abspath(j.get("outbase") or "").lower() for j in JOBS.values() if j.get("state") in ACTIVE}
        if new.lower() != old.lower() and (files_of(new) or os.path.exists(sidecar_path(new)) or os.path.abspath(new).lower() in busy):
            raise ValueError("Another transcript already has that name.")
        moved = []
        try:
            for path, kind, ext in files_of(old):
                dest = (new + ext) if kind == "main" else os.path.join(os.path.dirname(new), ".transcriber", name + ext)
                os.rename(path, dest)
                moved.append((path, dest))
        except OSError:
            for a, b in reversed(moved):                # put everything back the way it was
                try:
                    os.rename(b, a)
                except OSError:
                    pass
            raise ValueError("Couldn't rename the files. Check that the folder is writable.")
        job.update(name=name + ".txt", txt=new + ".txt", srt=new + ".srt")   # the list shows the name without ".txt"
    if os.path.abspath(old + ".txt") in hidden_paths():
        set_hidden(remove=[old + ".txt"])
    note_follow_rename(old, new)

    def follow(c):                                  # Voice Memos already done keep pointing at the right file
        for uid, path in list((c.get("vm_done") or {}).items()):
            if os.path.abspath(path) == os.path.abspath(old + ".txt"):
                c["vm_done"][uid] = new + ".txt"
    edit_config(follow)
    return name


TRASH = {}                                          # jid -> {"job", "timer"}: waiting a few seconds for Undo
REMOVED = {}                                        # jid -> job, so "Remove from list" can be undone
TRASH_JS = ("function run(argv){ObjC.import('Foundation');var fm=$.NSFileManager.defaultManager,bad=[];"
            "argv.forEach(function(p){if(!fm.trashItemAtURLResultingItemURLError($.NSURL.fileURLWithPath(p),null,null))bad.push(p);});"
            "return bad.join('\\n');}")


def _trash_paths(paths, timeout=60):
    """Move files to the Trash; returns the ones that couldn't be moved."""
    if not paths:
        return []
    if not OSASCRIPT:
        return paths
    try:
        r = subprocess.run([OSASCRIPT, "-l", "JavaScript", "-e", TRASH_JS] + paths,
                           capture_output=True, text=True, timeout=timeout)
        return [l for l in r.stdout.strip().splitlines() if l] if r.returncode == 0 else paths
    except (OSError, subprocess.SubprocessError):
        return paths


def _trashed(entry):
    txt = os.path.abspath(entry["job"]["txt"])

    def mark(c):                                    # a trashed Voice Memo must not be transcribed again
        for uid, path in list((c.get("vm_done") or {}).items()):
            if os.path.abspath(path) == txt:
                c["vm_done"][uid] = "trashed"
    edit_config(mark)


def trash_now(jid):
    with LOCK:
        entry = TRASH.get(jid)
        if entry:
            entry["busy"] = True                    # too late for Undo from here on
    if not entry:
        return
    bad = _trash_paths(trashable(entry["job"]["txt"][:-4]))
    with LOCK:
        if TRASH.pop(jid, None) is None:            # undone meanwhile
            return
        if bad:                                     # couldn't trash: keep it, and show it again
            JOBS[jid] = entry["job"]
            if jid not in ORDER:
                ORDER.append(jid)
            JOBS[jid]["warning"] = "Couldn't move this to the Trash. You can delete it in Finder."
    if not bad:
        _trashed(entry)


def trash_later(jid, delay=8.0):
    with LOCK:
        job = JOBS.get(jid)
        if not job or job.get("state") != "done" or not job.get("txt"):
            return False
        run = AI_RUNS.get(jid)
        if run and not run.get("done"):            # the AI would write its notes back after the file is gone
            return False
        JOBS.pop(jid, None)
        if jid in ORDER:
            ORDER.remove(jid)
        t = threading.Timer(delay, trash_now, args=(jid,))
        t.daemon = True
        TRASH[jid] = {"job": job, "timer": t}
    t.start()
    return True


def trash_undo(jid):
    with LOCK:
        entry = TRASH.get(jid)
        if not entry or entry.get("busy"):
            return False
        TRASH.pop(jid, None)
        entry["timer"].cancel()
        JOBS[jid] = entry["job"]
        if jid not in ORDER:
            ORDER.append(jid)
    return True


def flush_trash():
    """On quit, finish whatever was waiting for Undo: one quick call for everything."""
    with LOCK:
        entries = list(TRASH.values())
        TRASH.clear()
    for e in entries:
        e["timer"].cancel()
    paths = [p for e in entries for p in trashable(e["job"]["txt"][:-4])]
    if _trash_paths(paths, timeout=8) == []:
        for e in entries:
            _trashed(e)


def ai_preview(base):
    """First lines of the AI summary, for the list (None if there isn't one)."""
    data = load_ai(base)
    summ = data.get("summary") if isinstance(data, dict) else None
    text = summ.get("text") if isinstance(summ, dict) else None
    if not isinstance(text, str) or not text.strip():
        return None
    text = text.strip()
    m = re.search(r"##\s*(?:Summary|Overview)\s*\n+(.+?)(\n\s*\n|\n##|$)", text, re.S)
    para = (m.group(1) if m else re.sub(r"^#.*$", "", text, flags=re.M)).strip()
    para = re.sub(r"[*_`#>]", "", " ".join(para.split()))
    return summarize(para)[0] if para else None


def load_library():
    """Show transcripts already in ~/Documents/Transcripts so past work stays visible
    (except ones the user removed from the list)."""
    if not os.path.isdir(OUT_DIR):
        return
    hidden = hidden_paths()
    items = []
    with LOCK:
        known = {os.path.abspath(JOBS[j]["txt"]) for j in ORDER if j in JOBS and JOBS[j].get("txt")}
        known |= {os.path.abspath(e["job"]["txt"]) for e in TRASH.values()}     # on their way to the Trash
        known |= {os.path.abspath(j["outbase"] + ".txt") for j in JOBS.values()  # still being finished
                  if j.get("outbase") and j.get("state") in ACTIVE}
    for f in os.listdir(OUT_DIR):
        if not f.endswith(".txt") or f.startswith("."):
            continue
        path = os.path.join(OUT_DIR, f)
        if os.path.abspath(path) in known or os.path.abspath(path) in hidden:
            continue
        base = path[:-4]
        try:
            mtime = os.path.getmtime(path)
            body = plain_text({"txt": path})
            words = count_words(body)
            summary = summarize(body)[0]
            ai_sum = ai_preview(base)
            side = load_sidecar(base)
            spk = len({x["speaker"] for x in side["segments"]}) if side else 0
            dur = srt_duration(base + ".srt")
        except Exception:                       # one odd file must never stop the app from starting
            continue
        jid = "lib" + hashlib.md5(path.encode("utf-8")).hexdigest()[:10]
        items.append({"id": jid, "name": f, "state": "done", "progress": 100,
                      "created": mtime, "finished": mtime, "txt": path, "srt": base + ".srt",
                      "has_srt": os.path.isfile(base + ".srt"), "words": words, "summary": ai_sum or summary,
                      "summary_ai": bool(ai_sum),
                      "speaker_count": spk, "duration": dur,
                      "recorded": recorded_at({"txt": path, "created": mtime}),
                      "library": True})
    items.sort(key=lambda x: x["created"])
    with LOCK:
        for it in items:
            JOBS[it["id"]] = it
            ORDER.append(it["id"])


# ---------------------------------------------------------------- http
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send_bytes(self, body, ctype, code=200, extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def send_file_range(self, path, ctype):
        """Serve a file with HTTP range support, so the audio player can jump to any point."""
        size = os.path.getsize(path)
        start, end, code = 0, size - 1, 200
        m = re.match(r"bytes=(\d*)-(\d*)$", (self.headers.get("Range") or "").strip())
        if m and (m.group(1) or m.group(2)):
            if m.group(1):
                start = int(m.group(1))
                end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
            else:
                start = max(0, size - int(m.group(2)))
            if start > end or start >= size:
                return self.send_bytes(b"", ctype, 416, {"Content-Range": "bytes */%d" % size})
            code = 206
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Cache-Control", "no-store")
        if code == 206:
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        self.end_headers()
        try:
            with open(path, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(1 << 16, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass                                          # the player moved on; that's normal

    def send_json(self, obj, code=200):
        self.send_bytes(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                        "application/json; charset=utf-8", code)

    def attachment(self, data, filename, ctype):
        self.send_bytes(data, ctype, extra={
            "Content-Disposition": "attachment; filename*=UTF-8''" + urllib.parse.quote(filename)})

    ALLOWED_HOSTS = ("127.0.0.1:%d" % PORT, "localhost:%d" % PORT)

    def trusted(self, post=False):
        """Only this page may use the app: no other website in the browser, no DNS rebinding."""
        if self.headers.get("Host", "") not in self.ALLOWED_HOSTS:
            return False
        if not post and urllib.parse.urlparse(self.path).path in ("/", "/favicon.svg", "/icon.png",
                                                                   "/apple-touch-icon.png", "/favicon.ico"):
            return True                                  # opening the page from a link is fine
        origin = self.headers.get("Origin")
        if origin and origin.replace("http://", "", 1) not in self.ALLOWED_HOSTS:
            return False
        if self.headers.get("Sec-Fetch-Site") == "cross-site":
            return False
        if post and self.headers.get("X-Transcriber") != "1":
            return False
        return True

    def do_GET(self):
        if not self.trusted():
            return self.send_json({"error": "forbidden"}, 403)
        u = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(u.query)
        arg = lambda k, d="": (qs.get(k) or [d])[0]

        if u.path == "/":
            return self.send_bytes(PAGE.replace("__VER__", VERSION).encode("utf-8"), "text/html; charset=utf-8")

        if u.path == "/favicon.svg":
            return self.send_bytes(FAVICON_SVG.encode("utf-8"), "image/svg+xml")

        if u.path in ("/icon.png", "/apple-touch-icon.png", "/favicon.ico"):
            return self.send_bytes(base64.b64decode(ICON_B64), "image/png")

        if u.path == "/api/info":
            if not (WHISPER and FFMPEG):
                refresh_bins()
            models = list_models()
            return self.send_json({
                "ffmpeg": bool(FFMPEG), "whisper": bool(WHISPER), "models": models,
                "default_model": DEFAULT_MODEL if DEFAULT_MODEL in models else (models[0] if models else None),
                "modes": [{"id": k, "label": v["label"], "size": v["size"], "about": v["about"],
                           "ready": v["model"] in models, "langs": list(PK_LANGS) if k == "parakeet" else None}
                          for k, v in MODES.items()],
                "default_quality": DEFAULT_QUALITY,
                "out_dir": OUT_DIR.replace(HOME, "~", 1),
                "out_dir_default": os.path.abspath(OUT_DIR) == DEFAULT_OUT,
                "out_dir_fallback": OUT_FALLBACK["from"], "starting": not READY.is_set(),
                "brew": bool(BREW), "speakers_ready": speakers_ready(),
                "lid": lid_mode(), "lid_active": bool(LID["disabled"]),
                "hidden": len([p for p in hidden_paths() if os.path.exists(p)]),
                "clock24": mac_clock().get("h24"), "locale": mac_clock().get("locale"),
                "version": VERSION, "update": dict(UPDATE), "tasks": json.loads(json.dumps(TASKS)),
                "capture": capture_ready(), "capture_build": dict(CAPTURE_BUILD), "app": bool(BUNDLE),
                "capture_os": sys.platform == "darwin" and mac_version() >= (14, 2),
            })


        if u.path == "/api/vm":
            return self.send_json(list_voice_memos())

        if u.path == "/api/folders":
            return self.send_json(folders_info())

        if u.path == "/api/jobs":
            with LOCK:
                jobs = [{k: JOBS[j].get(k) for k in PUBLIC_KEYS} for j in ORDER if j in JOBS]
            return self.send_json({"jobs": jobs, "now": time.time(), "paused": not RUNNING.is_set()})

        if u.path == "/api/appearance":
            return self.send_json({"dark": mac_dark() if sys.platform == "darwin" else None})

        if u.path == "/api/text":
            with LOCK:
                job = dict(JOBS.get(arg("id"), {}))
            if not job.get("txt") or not os.path.isfile(job["txt"]):
                return self.send_json({"error": "This transcript file was moved or deleted."}, 404)
            with open(job["txt"], encoding="utf-8", errors="replace") as fh:
                text = fh.read().strip()
            data = load_sidecar(job["txt"][:-4])
            summary, keywords = summarize(plain_text(job))
            turns = None
            if data:
                turns = [{"speaker": t["speaker"], "name": speaker_name(data, t["speaker"]),
                          "t": hms(t["start"]), "s": t["start"], "text": t["text"]} for t in turns_of(data)]
            return self.send_json({"text": text, "segments": srt_segments(job.get("srt", "")),
                                   "summary": summary, "keywords": keywords, "turns": turns,
                                   "audio": os.path.isfile(audio_path(job["txt"][:-4])),
                                   "unsure": load_unsure(job["txt"][:-4]),
                                   "names": {str(k): speaker_name(data, k) for k in sorted({x["speaker"] for x in data["segments"]})} if data else {}})

        if u.path == "/api/search":
            return self.send_json({"results": search_all(arg("q"))})

        if u.path == "/api/ai/info":
            cfg = ai_config()
            key = claude_key()
            return self.send_json({
                "provider": cfg["provider"], "titles": cfg["titles"], "title_format": cfg["title_format"],
                "ollama": {"models": ollama_models(), "model": cfg["ollama_model"], "suggested": OLLAMA_SUGGESTED},
                "claude": {"has_key": bool(key), "hint": key[-4:] if key else "", "model": cfg["claude_model"],
                           "models": [{"id": k, "label": v} for k, v in CLAUDE_MODELS]}})

        if u.path == "/api/ai":
            with LOCK:
                job = dict(JOBS.get(arg("id"), {}))
                run = dict(AI_RUNS.get(arg("id")) or {})
            if not job.get("txt"):
                return self.send_json({"error": "not found"}, 404)
            data = load_ai(job["txt"][:-4])
            cfg = ai_config()
            return self.send_json({"summary": data.get("summary"), "fixes": data.get("fixes"),
                                   "fixes_by": data.get("fixes_by"), "run": run or None,
                                   "provider": cfg["provider"]})

        if u.path == "/api/peaks":
            with LOCK:
                job = dict(JOBS.get(arg("id"), {}))
            base = job["txt"][:-4] if job.get("txt") else ""
            if not base or not os.path.isfile(audio_path(base)):
                return self.send_json({"error": "not found"}, 404)
            try:
                return self.send_json(peaks_of(base))
            except (OSError, subprocess.SubprocessError):
                return self.send_json({"duration": 0, "peaks": []})

        if u.path == "/api/audio":
            with LOCK:
                job = dict(JOBS.get(arg("id"), {}))
            path = audio_path(job["txt"][:-4]) if job.get("txt") else ""
            if not path or not os.path.isfile(path):
                return self.send_json({"error": "not found"}, 404)
            return self.send_file_range(path, "audio/mp4")

        if u.path == "/api/download":
            with LOCK:
                job = dict(JOBS.get(arg("id"), {}))
            try:
                data, name, ctype = build_export(job, arg("kind", "txt"))
            except (OSError, KeyError, TypeError):
                return self.send_json({"error": "not found"}, 404)
            return self.attachment(data, name, ctype)

        if u.path == "/api/download-all":
            with LOCK:
                paths = []
                for j in ORDER:
                    job = JOBS.get(j, {})
                    if job.get("state") == "done":
                        paths += [p for p in (job.get("txt"), job.get("srt")) if p and os.path.isfile(p)]
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                for p in paths:
                    z.write(p, os.path.basename(p))
            name = time.strftime("transcripts-%Y%m%d-%H%M.zip")
            return self.attachment(buf.getvalue(), name, "application/zip")

        self.send_json({"error": "not found"}, 404)

    def do_POST(self):
        if not self.trusted(post=True):
            return self.send_json({"error": "forbidden"}, 403)
        u = urllib.parse.urlparse(self.path)
        try:
            length = max(0, int(self.headers.get("Content-Length") or 0))
        except ValueError:
            return self.send_json({"error": "bad request"}, 400)

        if u.path == "/api/upload":
            hdr = lambda k: urllib.parse.unquote(self.headers.get(k) or "")
            raw = os.path.basename(hdr("X-Filename")) or "recording"
            stem, ext = os.path.splitext(raw)
            name = safe_title(stem, "recording") + ext.lower()[:8]
            lang = hdr("X-Lang") if hdr("X-Lang") in LANGS else "en"
            quality = clean_quality(hdr("X-Quality"))
            model = choose_config(quality, lang)[0]
            prompt = hdr("X-Prompt").strip()[:500]
            speakers = hdr("X-Speakers") or "off"
            if speakers not in ("off", "auto") and not speakers.isdigit():
                speakers = "off"
            try:
                recorded = int(hdr("X-Recorded") or 0) / 1000.0
            except ValueError:
                recorded = 0
            if not (946684800 < recorded < time.time() + 86400):
                recorded = None

            workdir = new_workdir()
            src = os.path.join(workdir, "input" + os.path.splitext(name)[1].lower())
            remaining = length
            try:
                with open(src, "wb") as fh:
                    while remaining > 0:
                        chunk = self.rfile.read(min(1 << 20, remaining))
                        if not chunk:
                            break
                        fh.write(chunk)
                        remaining -= len(chunk)
            except OSError:
                remaining = -1
            if remaining != 0 or length == 0:            # upload was cut off or empty
                shutil.rmtree(workdir, ignore_errors=True)
                return self.send_json({"error": "The upload didn't finish. Add the file again."}, 400)
            dur = duration_of(src)                       # slow; keep it outside the lock
            extra = {}
            cap, cap_t0 = capture_take(hdr("X-Capture")) if hdr("X-Capture") else (None, None)
            if hdr("X-Capture") and not cap:
                extra = {"warning": "Matita couldn't record other apps' sound, so only your microphone was transcribed."}
            if cap:
                try:
                    mic_t0 = int(hdr("X-Mic-Start") or 0) / 1000.0
                    dest = os.path.join(workdir, "system" + os.path.splitext(cap)[1])
                    shutil.move(cap, dest)
                    extra = {"mix_src": dest, "mix_delay": round(cap_t0 - mic_t0, 3) if mic_t0 else 0.0}
                    dur = max(dur or 0, duration_of(dest) or 0) or dur
                except (OSError, ValueError):
                    extra = {}

            jid = uuid.uuid4().hex[:10]
            with LOCK:
                JOBS[jid] = {"id": jid, "name": name, "state": "queued", "progress": 0,
                             "created": time.time(), "lang": lang, "model": model, "quality": quality,
                             "prompt": prompt, "src": src, "workdir": workdir, "speakers": speakers,
                             "duration": dur, "recorded": recorded, **extra}
                ORDER.append(jid)
            Q.put(jid)
            return self.send_json({"id": jid})

        body = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(body or b"{}")
        except ValueError:
            data = {}
        if not isinstance(data, dict):
            data = {}
        jid = data.get("id") if isinstance(data.get("id"), str) else None

        if u.path == "/api/cancel":
            ids = [jid] if jid else list(ORDER)
            skip = []
            with LOCK:
                for j in ids:
                    job = JOBS.get(j)
                    if job and job["state"] in ACTIVE:
                        job.update(state="cancelled", finished=time.time())
                        if job.get("vm_uid"):
                            skip.append(job["vm_uid"])
                        for p in (PROCS.get(j), SIDE.get(j)):
                            stop_proc(p)
                left = any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS)
            if skip:                                 # auto-import must not bring a cancelled memo back
                edit_config(lambda c: [c.setdefault("vm_skip", {}).__setitem__(x, time.time()) for x in skip])
            if not left:
                set_paused(False)                    # nothing left to resume: don't stay paused
            return self.send_json({"ok": True})

        if u.path == "/api/retry":
            if jid and WORKING["id"] == jid:
                return self.send_json({"error": BUSY_MSG}, 409)
            with LOCK:
                job = JOBS.get(jid)
                ok = bool(job and job["state"] in ("error", "cancelled")
                          and os.path.isfile(job.get("src", "")))
                if ok:
                    job.update(state="queued", progress=0, error=None, preview=None,
                               started=None, finished=None, created=time.time())
                    ORDER.remove(jid)
                    ORDER.append(jid)
            if ok:
                Q.put(jid)
                return self.send_json({"ok": True})
            return self.send_json({"error": "The original file is no longer available. Drop it in again."}, 400)

        if u.path == "/api/unhide-one":
            with LOCK:
                job = REMOVED.pop(jid, None)
                if job and jid not in JOBS:
                    JOBS[jid] = job
                    ORDER.append(jid)
            if job:
                set_hidden(remove=[job["txt"]])
            return self.send_json({"ok": bool(job)})

        if u.path == "/api/rename":
            try:
                name = rename_transcript(jid, data.get("name"))
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            with LOCK:
                base = (JOBS.get(jid, {}).get("txt") or ".txt")[:-4]
                if base:
                    d = load_ai(base); d["keep_name"] = True; save_ai(base, d)
                if jid in JOBS:
                    JOBS[jid].pop("auto_titled", None)
            return self.send_json({"ok": True, "name": name})

        if u.path == "/api/trash":
            if WORKING["id"] == jid:
                return self.send_json({"error": BUSY_MSG}, 409)
            with LOCK:
                run = AI_RUNS.get(jid)
                ai_busy = bool(run and not run.get("done"))
            if ai_busy:
                return self.send_json({"error": "The AI is still working on this transcript. Try again when it's done."}, 409)
            if not trash_later(jid):
                return self.send_json({"error": "This transcript can't be moved to the Trash right now."}, 400)
            return self.send_json({"ok": True})

        if u.path == "/api/capture/start":
            try:
                token, t0 = capture_start()
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json({"id": token, "started": t0})

        if u.path == "/api/capture/ctl":
            what = data.get("do") if data.get("do") in ("alive", "pause", "resume", "mute", "unmute", "stop") else "alive"
            return self.send_json({"ok": capture_ctl(str(data.get("id") or ""), what)})

        if u.path == "/api/capture/discard":
            path, _ = capture_take(str(data.get("id") or ""))
            if path:
                try:
                    os.remove(path)
                except OSError:
                    pass
            return self.send_json({"ok": True})

        if u.path == "/api/notes":
            if data.get("choose"):
                try:
                    picked = choose_folder("Choose a folder for Markdown notes (an Obsidian vault works well)",
                                           notes_cfg()["dir"])
                except ValueError as e:
                    return self.send_json({"error": str(e)}, 400)
                if picked is None:
                    return self.send_json({"cancelled": True})
                data["dir"] = picked
            new = {}
            if "dir" in data:
                d = os.path.abspath(os.path.expanduser(str(data.get("dir") or "").strip())) if data.get("dir") else ""
                if d and not writable_dir(d):
                    return self.send_json({"error": "Matita can't write to that folder. Pick another one."}, 400)
                new["dir"] = d or None
            if "on" in data:
                new["on"] = bool(data["on"])
            edit_config(lambda c: c.__setitem__("notes", dict(c.get("notes") if isinstance(c.get("notes"), dict) else {}, **new)))
            NOTE_SIG.clear()
            threading.Thread(target=notes_pass, daemon=True).start()
            return self.send_json(folders_info())

        if u.path == "/api/watch/add":
            path = data.get("path")
            if not path:
                try:
                    path = choose_folder("Choose a folder to watch for new recordings", os.path.join(HOME, "Desktop"))
                except ValueError as e:
                    return self.send_json({"error": str(e)}, 400)
                if path is None:
                    return self.send_json({"cancelled": True})
            try:
                path, existing = watch_add(path, data.get("opts") or {})
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json(dict(folders_info(), added=path, show=path.replace(HOME, "~", 1), existing=existing))

        if u.path == "/api/watch/existing":
            n = watch_existing(str(data.get("path") or ""))
            threading.Thread(target=watch_pass, daemon=True).start()
            return self.send_json({"ok": True, "count": n})

        if u.path == "/api/watch/remove":
            watch_remove(str(data.get("path") or ""))
            return self.send_json(folders_info())

        if u.path == "/api/trash/undo":
            return self.send_json({"ok": trash_undo(jid)})

        if u.path == "/api/remove":
            ids = [jid] if jid else [j for j in ORDER if JOBS.get(j, {}).get("state") not in ACTIVE]
            if data.get("done_only"):
                ids = [j for j in ids if JOBS.get(j, {}).get("state") == "done"]
            if jid and WORKING["id"] == jid:
                return self.send_json({"error": BUSY_MSG}, 409)
            gone = []
            with LOCK:
                for j in ids:
                    job = JOBS.get(j)
                    if not job or job["state"] in ACTIVE or WORKING["id"] == j:
                        continue
                    if job.get("workdir"):
                        shutil.rmtree(job["workdir"], ignore_errors=True)
                    if job["state"] == "done" and job.get("txt"):
                        gone.append(job["txt"])          # remember, so it stays off the list next time
                        if jid:
                            REMOVED[j] = dict(job)       # for Undo
                    JOBS.pop(j, None)
                    if j in ORDER:
                        ORDER.remove(j)
            if gone:
                set_hidden(add=gone)
            return self.send_json({"ok": True, "removed": len(gone)})

        if u.path == "/api/unhide":
            with LOCK:
                REMOVED.clear()
            set_hidden(remove=list(hidden_paths()))
            load_library()
            return self.send_json({"ok": True})

        if u.path == "/api/choose-folder":
            try:
                picked = choose_folder("Choose where to save transcripts", OUT_DIR)
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            if picked is None:
                return self.send_json({"cancelled": True})
            try:
                path = set_out_dir(picked)
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json({"out_dir": path.replace(HOME, "~", 1)})

        if u.path == "/api/save-as":
            with LOCK:
                job = dict(JOBS.get(jid, {}))
            fmt = data.get("kind") if data.get("kind") in EXPORTS else "txt"
            try:
                content, default, _ = build_export(job, fmt)
            except (OSError, KeyError, TypeError):
                return self.send_json({"error": "This transcript file was moved or deleted."}, 404)
            if not OSASCRIPT:
                return self.send_json({"error": "no dialog"}, 400)
            start = os.path.join(HOME, "Desktop") if os.path.isdir(os.path.join(HOME, "Desktop")) else HOME
            script = ['-e', 'on run argv', '-e', 'activate', '-e',
                      'POSIX path of (choose file name with prompt "Save transcript as" default name (item 1 of argv) '
                      'default location (POSIX file (item 2 of argv)))', '-e', 'end run']
            try:
                r = subprocess.run([OSASCRIPT] + script + [default, start], capture_output=True, text=True, timeout=600)
            except subprocess.TimeoutExpired:
                return self.send_json({"cancelled": True})
            if r.returncode != 0:
                if "-128" in r.stderr:
                    return self.send_json({"cancelled": True})
                return self.send_json({"error": "no dialog"}, 400)          # the page downloads it instead
            dest = r.stdout.strip()
            if not dest.lower().endswith(EXPORTS[fmt][1]):
                # the name was changed after the dialog's own overwrite check, so don't replace anything
                stem, ext = os.path.splitext(dest)[0], EXPORTS[fmt][1]
                dest, i = stem + ext, 2
                while os.path.exists(dest):
                    dest, i = "%s (%d)%s" % (stem, i, ext), i + 1
            try:
                with open(dest, "wb") as fh:
                    fh.write(content)
            except OSError:
                return self.send_json({"error": "no dialog"}, 400)          # e.g. no permission there: download instead
            return self.send_json({"path": dest.replace(HOME, "~", 1)})

        if u.path == "/api/lid":
            mode = data.get("mode") if data.get("mode") in ("sleep", "plugged", "always") else "sleep"
            if mode != "sleep" and not lid_ready():
                try:
                    lid_setup()                          # shows the macOS password dialog
                except RuntimeError as e:
                    return self.send_json({"error": str(e)}, 400)
            edit_config(lambda c: c.__setitem__("lid", mode))
            if mode == "sleep":
                lid_set(False)
            return self.send_json({"ok": True, "mode": mode})

        if u.path == "/api/edit":
            with LOCK:
                job = dict(JOBS.get(jid, {}))
            old, new = str(data.get("old") or "").strip(), " ".join(str(data.get("new") or "").split())
            try:
                start = float(data.get("s"))
            except (TypeError, ValueError):
                start = None
            if not job.get("txt") or not os.path.isfile(job["txt"]) or start is None:
                return self.send_json({"error": "This transcript file was moved or deleted."}, 404)
            if not new:
                return self.send_json({"error": "A line can't be empty."}, 400)
            if new == old:
                return self.send_json({"ok": True})
            base = job["txt"][:-4]
            try:
                with LOCK:
                    side = load_sidecar(base)
                    if side:
                        try:
                            edited = os.path.getmtime(job["txt"]) > os.path.getmtime(sidecar_path(base)) + 2
                        except OSError:
                            edited = False
                        if edited:
                            return self.send_json({"error": "This transcript was edited outside Matita. "
                                                            "Edit it there instead, so your changes aren't lost."}, 409)
                        edit_turn(base, side, start, old, new)
                    else:
                        edit_plain(job, start, old, new)
            except ValueError as e:
                return self.send_json({"error": str(e)}, 409)
            except OSError as e:
                return self.send_json({"error": "Couldn't save the change: %s" % e}, 500)
            text = plain_text({"txt": job["txt"]}).strip()
            update(jid, words=count_words(text), summary=summarize(text)[0])
            return self.send_json({"ok": True})

        if u.path == "/api/pause":
            set_paused(bool(data.get("on")))
            return self.send_json({"ok": True, "paused": not RUNNING.is_set()})

        if u.path == "/api/awake":
            # Keep the Mac from sleeping while the page records from the microphone.
            with LOCK:
                old = AWAKE.pop("rec", None)
            if old and old.poll() is None:
                old.terminate()
            if data.get("on") and CAFFEINATE:
                with LOCK:
                    AWAKE["rec"] = subprocess.Popen([CAFFEINATE, "-i", "-w", str(os.getpid()), "-t", "600"])  # page renews it
            return self.send_json({"ok": True})

        if u.path == "/api/ai/settings":
            prov = data.get("provider")
            if "claude_key" in data:
                key = str(data.get("claude_key") or "").strip()
                if key:
                    if not re.match(r"^[A-Za-z0-9_\-]{20,300}$", key):
                        return self.send_json({"error": "That doesn't look like a Claude API key."}, 400)
                    try:
                        claude_request("/v1/models", key=key)      # free check that the key works
                    except RuntimeError as e:
                        return self.send_json({"error": str(e)}, 400)
                save_claude_key(key)

            def change(c):
                ai = c.setdefault("ai", {})
                if prov in ("off", "ollama", "claude"):
                    ai["provider"] = prov
                if isinstance(data.get("ollama_model"), str):
                    ai["ollama_model"] = data["ollama_model"][:120]
                if data.get("claude_model") in dict(CLAUDE_MODELS):
                    ai["claude_model"] = data["claude_model"]
                if data.get("titles") in ("choose", "auto", "off"):
                    ai["titles"] = data["titles"]
                if isinstance(data.get("title_format"), dict):
                    ai["title_format"] = clean_format(data["title_format"])
            edit_config(change)
            return self.send_json({"ok": True})

        if u.path == "/api/ai/run":
            kind = data.get("kind") if data.get("kind") in ("summary", "fixes") else "summary"
            cfg = ai_config()
            with LOCK:
                job = dict(JOBS.get(jid, {}))
                ok_file = bool(job.get("txt")) and os.path.isfile(job["txt"])
                busy = jid in AI_RUNS and not AI_RUNS[jid].get("done")
                if ok_file and not busy and cfg["provider"] != "off":
                    AI_RUNS[jid] = {"kind": kind, "progress": "Starting", "started": time.time()}
            if not ok_file:
                return self.send_json({"error": "This transcript file was moved or deleted."}, 404)
            if cfg["provider"] == "off":
                return self.send_json({"error": "Turn on AI assist in Settings first."}, 400)
            if busy:
                return self.send_json({"error": "AI is already working on this transcript."}, 409)
            threading.Thread(target=run_ai, args=(jid, kind, cfg), daemon=True).start()
            return self.send_json({"ok": True})

        if u.path in ("/api/ai/apply", "/api/ai/dismiss"):
            with LOCK:                               # one lock: a rename can't slip in between
                job = dict(JOBS.get(jid, {}))
                if not job.get("txt"):
                    return self.send_json({"error": "not found"}, 404)
                base = job["txt"][:-4]
                ai = load_ai(base)
                fixes = ai.get("fixes") or []
                idx = range(len(fixes)) if data.get("all") else [i for i, f in enumerate(fixes)
                                                                if f.get("find") == data.get("find")][:1]
                picked = [fixes[i] for i in idx]
                changed = 0
                if u.path == "/api/ai/apply":
                    for f in picked:
                        changed += apply_fix(job, f)
                ai["fixes"] = [f for i, f in enumerate(fixes) if i not in set(idx)]
                save_ai(base, ai)
            if u.path == "/api/ai/apply" and picked and not changed:
                return self.send_json({"ok": True, "changed": 0, "note": "That text has already changed, or it "
                                       "runs across two speakers' lines, so it couldn't be fixed automatically."})
            if changed:
                text = plain_text({"txt": job["txt"]}).strip()
                update(jid, words=count_words(text), summary=summarize(text)[0])
            return self.send_json({"ok": True, "changed": changed})

        if u.path == "/api/rename-speaker":
            with LOCK:
                job = dict(JOBS.get(jid, {}))
            base = (job.get("txt") or "")[:-4]
            side = load_sidecar(base) if base else None
            name = str(data.get("name") or "").strip()[:60]
            if not side:
                return self.send_json({"error": "This transcript has no speakers."}, 400)
            try:
                edited = os.path.getmtime(base + ".txt") > os.path.getmtime(sidecar_path(base)) + 2
            except OSError:
                edited = False
            if edited:
                return self.send_json({"error": "This transcript was edited outside Matita, so renaming here "
                                                "would overwrite your changes. Rename the speaker in your editor instead."}, 409)
            with LOCK:
                cur = (JOBS.get(jid, {}).get("txt") or "")[:-4]
                if cur != base:                            # renamed meanwhile: don't recreate the old files
                    return self.send_json({"error": "This transcript was just renamed. Try again."}, 409)
                side = load_sidecar(base) or side          # re-read inside the lock so a parallel edit isn't lost
                names = side.setdefault("names", {})
                if name:
                    names[str(data.get("speaker"))] = name
                else:
                    names.pop(str(data.get("speaker")), None)
                save_sidecar(base, side)
                write_speaker_files(base, side)
            return self.send_json({"ok": True})

        if u.path == "/api/vm/import":
            opts = clean_opts(data.get("opts"))
            added, errors = 0, []
            for uid in data.get("uids") or []:
                try:
                    enqueue_voice_memo(uid, opts)
                    added += 1
                except PermissionError:
                    return self.send_json({"error": "needs_access"}, 403)
                except Exception as e:
                    errors.append(str(e))
            return self.send_json({"added": added, "errors": errors})

        if u.path == "/api/vm/auto":
            on = bool(data.get("enabled"))
            def change(c):
                if on:
                    c["vm_auto"] = {"enabled": True, "since": time.time(), "opts": clean_opts(data.get("opts"))}
                else:
                    c["vm_auto"] = {"enabled": False}
            edit_config(change)
            return self.send_json({"ok": True})

        if u.path == "/api/vm/open-settings":
            subprocess.run(["open", FDA_URL])
            return self.send_json({"ok": True})

        if u.path == "/api/vm/reveal-app":
            subprocess.run(["open", "-R", responsible_app()["path"]])
            return self.send_json({"ok": True})

        if u.path == "/api/restart":
            # macOS only applies a new Full Disk Access permission after the app restarts
            with LOCK:
                busy = any(JOBS[j]["state"] in ACTIVE for j in ORDER if j in JOBS)
            if busy:
                return self.send_json({"error": "Wait until the recordings in progress are finished, then restart."}, 400)
            if os.environ.get("XPC_SERVICE_NAME") == LABEL:
                subprocess.Popen(["/bin/sh", "-c", "sleep 1; launchctl kickstart -k gui/$(id -u)/%s" % LABEL],
                                 start_new_session=True)
                return self.send_json({"ok": True})
            if BUNDLE:
                return self.send_json({"error": "Quit Matita (⌘Q) and open it again, so macOS applies the permission."}, 400)
            return self.send_json({"error": "Quit Matita in Terminal (Control-C) and start it again."}, 400)

        if u.path == "/api/setup":
            what = data.get("what")
            target = {"tools": setup_tools, "model": setup_model, "speakers": setup_speakers}.get(what)
            if not target:
                return self.send_json({"error": "unknown"}, 400)
            if TASKS[what]["state"] != "running":
                threading.Thread(target=target, daemon=True).start()
            return self.send_json({"ok": True})

        if u.path == "/api/update":
            if TASKS["update"]["state"] != "running":
                threading.Thread(target=apply_update, daemon=True).start()
            return self.send_json({"ok": True})

        if u.path == "/api/check-update":
            check_update()
            return self.send_json({"update": dict(UPDATE)})

        if u.path == "/api/set-folder":
            try:
                path = set_out_dir(str(data.get("path") or ""))
            except ValueError as e:
                return self.send_json({"error": str(e)}, 400)
            return self.send_json({"out_dir": path.replace(HOME, "~", 1)})

        if u.path == "/api/reveal":
            with LOCK:
                path = JOBS.get(jid, {}).get("txt")
            if path and os.path.isfile(path):
                subprocess.run(["open", "-R", path])
            else:
                os.makedirs(OUT_DIR, exist_ok=True)
                subprocess.run(["open", OUT_DIR])
            return self.send_json({"ok": True})

        self.send_json({"error": "not found"}, 404)


# ---------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matita</title>
<script>try{var t=localStorage.getItem("stt.theme"),o=localStorage.getItem("stt.os");if(t!=="light"&&t!=="dark")t=o;if(t==="light"||t==="dark")document.documentElement.setAttribute("data-theme",t);}catch(e){}</script>
<link rel="icon" type="image/svg+xml" href="/favicon.svg?v=__VER__">
<link rel="icon" type="image/png" sizes="512x512" href="/icon.png?v=__VER__">
<link rel="apple-touch-icon" href="/apple-touch-icon.png?v=__VER__">
<meta name="apple-mobile-web-app-title" content="Matita">
<meta name="theme-color" content="#ececee">
<style>
:root{color-scheme:light dark}
:root[data-theme="light"]{color-scheme:light}
:root[data-theme="dark"]{color-scheme:dark}
:root{
  --mint:#ececee;        /* page backdrop */
  --ink:#1d1d1f;
  --green:#50a65c;
  --green-ink:#2f7a3b;
  --cream:#ffffff;       /* panels */
  --row:#f5f5f7;         /* rows inside panels */
  --rule:#e3e3e8;
  --soft:#6e6e73;
  --new:#c2338a;
  --fail:#d70015;
  --mark:#fde68a;
  --hero:#1d1d1f;
  --hover:#f2f2f4;       /* hovered line in the reader */
  --playing:#e6f3e8;     /* line being played */
  --unsure-bg:#fdf1dc;   /* hovered unsure word */
  --tray:#fafafb;        /* AI box */
  --pick:#eef7ef;        /* chosen option */
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){ --mint:#000000; --cream:#1c1c1e; --row:#2c2c2e; --rule:#3a3a3c; --ink:#f5f5f7; --soft:#98989d;
    --green-ink:#6fd17d; --new:#ff6fbf; --fail:#ff6961; --mark:#6b5a12; --hero:#2c2c2e;
    --hover:#2c2c2e; --playing:#1f3a24; --unsure-bg:#3d2f12; --tray:#232325; --pick:#1d3321; }
}
:root[data-theme="dark"]{ --mint:#000000; --cream:#1c1c1e; --row:#2c2c2e; --rule:#3a3a3c; --ink:#f5f5f7; --soft:#98989d;
    --green-ink:#6fd17d; --new:#ff6fbf; --fail:#ff6961; --mark:#6b5a12; --hero:#2c2c2e;
    --hover:#2c2c2e; --playing:#1f3a24; --unsure-bg:#3d2f12; --tray:#232325; --pick:#1d3321; }

*{box-sizing:border-box}
[hidden]{display:none!important}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--mint);color:var(--ink);min-height:100vh;
  font:15px/1.47 -apple-system,BlinkMacSystemFont,"SF Pro Text","Helvetica Neue",sans-serif;letter-spacing:-.01em;-webkit-font-smoothing:antialiased;font-variant-numeric:tabular-nums}
button,input,select{font:inherit;color:inherit}
:focus-visible{outline:3px solid rgba(80,166,92,.45);outline-offset:1px;border-radius:10px}
.wrap{max-width:900px;margin:0 auto;padding:22px 24px 48px}
@media (max-width:700px){.wrap{padding:14px 14px 36px}}

/* ---------- top bar ---------- */
.top{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:18px}
.brand{display:flex;align-items:center;gap:10px;font:600 19px -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.02em}
.brand svg{width:38px;height:38px;flex:none}
.btn{display:inline-flex;align-items:center;gap:8px;border-radius:99px;border:2px solid var(--ink);background:transparent;
  padding:7px 16px;font-weight:600;font-size:14px;cursor:pointer;text-decoration:none;color:var(--ink);white-space:nowrap}
.btn:hover{background:var(--ink);color:var(--cream)}
.btn.go{background:var(--green);border-color:var(--green);color:#fff}
.btn.go:hover{filter:brightness(.94)}
.btn.dark{background:var(--ink);color:var(--cream)}
.btn.dark:hover{opacity:.88}
.btn[disabled]{opacity:.35;pointer-events:none}
.btn svg{flex:none}
.link{background:none;border:0;padding:0;cursor:pointer;font-weight:600;font-size:13.5px;color:var(--green-ink);
  text-decoration:underline;text-decoration-thickness:2px;text-underline-offset:3px}
.link.muted{color:var(--soft);font-weight:500}
.muted{color:var(--soft)}

/* ---------- panels ---------- */
.panel{background:var(--cream);border-radius:22px;padding:22px 24px 10px}
@media (max-width:700px){.panel{padding:20px 18px;border-radius:24px}}
.first{display:grid;grid-template-columns:minmax(0,7fr) minmax(0,5fr);gap:18px}
@media (max-width:900px){.first{grid-template-columns:1fr}}
.stack{display:flex;flex-direction:column;gap:22px;margin-top:22px}
/* drop area */
.drop{background:var(--cream);border-radius:22px;padding:34px 30px 20px;text-align:center}
.drop h1{font:600 28px/1.15 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.025em;margin:0 0 24px}
.drop .lead{margin:6px 0 22px;color:var(--soft);font-size:15px}
.drop .cta{display:flex;justify-content:center;align-items:center;gap:14px;flex-wrap:wrap}
.drop .grp{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.drop .sep{width:1px;height:26px;background:var(--rule);margin:0 6px}
.count{color:var(--soft);font-weight:500;margin-left:4px}
.btn.soft{background:var(--row);border-color:var(--row);color:var(--ink)}
.btn.soft:hover{background:var(--rule);border-color:var(--rule);color:var(--ink)}
.btn.rec{background:transparent;border:2px solid var(--rule);color:var(--ink);padding-left:16px;padding-right:18px}
.btn.rec:hover{background:transparent;border-color:var(--fail);color:var(--ink)}
.recdot{width:10px;height:10px;border-radius:50%;background:var(--fail);box-shadow:0 0 0 3px color-mix(in srgb,var(--fail) 18%,transparent)}
.chiprow{display:flex;justify-content:center;flex-wrap:wrap;gap:4px 6px;margin-top:24px;padding-top:14px;border-top:1px solid var(--rule)}
.opt{display:inline-flex;align-items:center;gap:6px;border:0;background:transparent;border-radius:9px;padding:4px 4px 4px 10px;font:inherit;font-size:13px;color:var(--soft);cursor:pointer}
.opt:hover,.opt:focus-within{background:var(--row)}
.opt select{appearance:none;-webkit-appearance:none;background:transparent;border:0;min-height:0;padding:2px 18px 2px 0;font:500 13px/1.3 inherit;color:var(--ink);cursor:pointer;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='8' height='5'%3E%3Cpath d='M0 0l4 5 4-5z' fill='%236e6e73'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 4px center;width:auto;box-shadow:none}
.opt select:focus{outline:none;box-shadow:none}
.opt select{field-sizing:content}
.opt select:disabled{opacity:.5}
#hintChip{padding-right:10px} #hintChip b{color:var(--ink);font-weight:500;max-width:200px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.hintpop{max-width:520px;margin:12px auto 0;text-align:left;background:var(--row);border-radius:14px;padding:14px 16px}
.hintpop label{font-size:13px;font-weight:600}
.hintpop input{width:100%;margin-top:6px;background:var(--cream)}
.hintpop .hint{margin:8px 0 0}
.drop .spk-note:empty{display:none}
.drop .spk-note{justify-content:center;margin-top:6px}
.tb{display:inline-flex;align-items:center;gap:6px;height:34px;padding:0 10px;border-radius:99px;border:0;background:transparent;color:var(--soft);font:500 13px/1 inherit;cursor:pointer}
.tb:hover{background:var(--cream);color:var(--ink)}
.tb.on{color:var(--green-ink)}
/* transcript list */
#library{gap:0}
.row.lib{grid-template-columns:minmax(0,1fr) auto;background:transparent;border-radius:0;padding:14px 4px;border-top:1px solid var(--rule);align-items:start;overflow:visible}
.row.lib:first-child{border-top:0}
.row.lib:hover{background:transparent}
.row.lib:hover .name{color:var(--green-ink)}
.row.lib .sum{-webkit-line-clamp:1;opacity:1;color:var(--soft);font-size:14px;margin-top:2px}
.row.lib .meta{font-size:12.5px;color:var(--soft);opacity:.8;margin-top:3px}
.row.lib .lr{text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:6px}
.row.lib .when{font-size:12.5px;color:var(--soft)}
.row.lib .acts{margin-top:0;opacity:0;transition:opacity .15s}
.row.lib:hover .acts,.row.lib:focus-within .acts{opacity:1}
/* reader: player bar */
.pbar{display:flex;align-items:center;gap:12px;padding:12px 22px;border-top:1px solid var(--hair);background:var(--glass)}
.pbar .pp{width:36px;height:36px;border-radius:50%;border:0;background:var(--ink);color:var(--cream);display:grid;place-items:center;cursor:pointer;flex:none}
.pbar .pt{font-size:12.5px;color:var(--soft);font-variant-numeric:tabular-nums;min-width:38px;text-align:center;flex:none}
.pbar .wave{flex:1;height:38px;position:relative;cursor:pointer;touch-action:none;border-radius:6px}
.pbar .wave canvas{width:100%;height:100%;display:block}
.pbar .rate{height:28px;padding:0 10px;border-radius:8px;border:0;background:var(--fill);color:var(--ink);font:600 12.5px/1 inherit;cursor:pointer;flex:none;font-variant-numeric:tabular-nums}
/* speakers */
.turn .who{align-items:center;gap:8px}
.turn .av{width:22px;height:22px;border-radius:50%;background:var(--c);color:#fff;display:grid;place-items:center;font:700 10px/1 -apple-system,BlinkMacSystemFont,sans-serif;letter-spacing:.02em;flex:none}
.turn .who .nm{color:var(--ink)}
/* drag and drop: the drop card lights up */
.drop{transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease}
body.dragging .drop{transform:scale(1.012);border-color:rgba(80,166,92,.75);
  box-shadow:0 0 0 4px rgba(80,166,92,.18), 0 24px 60px -24px rgba(48,130,62,.55)}
body.dragging .drop h1{color:var(--green-ink)}
/* motion */
dialog[open]{animation:dlgIn .18s cubic-bezier(.2,.8,.2,1)}
dialog[open]::backdrop{animation:fadeIn .18s ease}
dialog.closing{animation:dlgOut .13s ease-in forwards}
@keyframes dlgIn{from{opacity:0;transform:scale(.97) translateY(6px)}to{opacity:1;transform:none}}
@keyframes dlgOut{to{opacity:0;transform:scale(.98)}}
@keyframes fadeIn{from{opacity:0}}
@media (prefers-reduced-motion: reduce){dialog[open],dialog.closing,.row.lib.enter{animation:none}}
.row.lib .when{font-variant-numeric:tabular-nums}
/* in-progress rows, same rhythm as the transcript list */
.row.qrow{grid-template-columns:minmax(0,1fr) auto;background:transparent;border-radius:0;padding:14px 4px;border-top:1px solid var(--hair);align-items:start;overflow:visible}
.row.qrow:first-child{border-top:0}
.row.qrow .lr{display:flex;flex-direction:column;align-items:flex-end;gap:6px}
.row.qrow .pct{font-size:13px;font-weight:600;color:var(--soft);font-variant-numeric:tabular-nums}
.row.qrow .acts{margin-top:0}
.track{height:3px;border-radius:2px;background:var(--fill);margin:9px 0 2px;overflow:hidden;max-width:520px}
.track i{display:block;height:100%;background:var(--green);border-radius:2px;transition:width .6s ease}
.row.qrow .live{padding:6px 0 0;font-size:13.5px;-webkit-mask-image:none;mask-image:none;opacity:.65;font-style:italic}
.row.qrow .err{padding:6px 0 0}
#queue{gap:0}
.row.lib.enter{animation:enter .45s ease-out}
@keyframes enter{from{opacity:0;transform:translateY(-4px)}to{opacity:1;transform:none}}
.quiet-empty{margin:18px 0 26px;text-align:center;color:var(--soft);font-size:14px}
/* settings: grouped list like System Settings */
.glist{background:var(--fill);border-radius:12px;margin-bottom:14px;overflow:hidden}
.grow{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:12px 14px}
.grow + .grow{border-top:1px solid var(--hair)}
.gl{display:flex;flex-direction:column;min-width:0}
.gl b{font-weight:500;font-size:14px}
.gl span{font-size:12.5px;color:var(--soft);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.gnote{padding:0 14px 12px;font-size:12.5px;color:var(--soft);margin-top:-4px}
.gnote:empty{display:none}
.btn.sm{height:30px;padding:0 13px;font-size:13px;flex:none}
select.gsel{width:auto;min-height:30px;padding:4px 28px 4px 10px;font-size:13px;background-color:var(--cream);flex:none}
.more{display:block;width:100%;margin:6px 0 10px;padding:10px;border:0;border-radius:12px;background:transparent;color:var(--soft);
  font:500 13.5px/1.2 inherit;cursor:pointer;text-align:center}
.more:hover{background:var(--fill);color:var(--ink)}
.newdot{display:inline-block;width:7px;height:7px;border-radius:50%;background:var(--green);margin-right:8px;vertical-align:2px}
#filter{width:220px;min-height:36px}
h2{font:700 24px/1.1 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.022em;margin:0}
.ph{display:flex;justify-content:space-between;align-items:flex-start;gap:14px;flex-wrap:wrap;margin-bottom:18px}
.ph .sub{color:var(--soft);font-size:14px;margin-top:6px}
.ph .tools{display:flex;align-items:center;gap:12px;flex-wrap:wrap}

/* ---------- hero ---------- */
.hero{background:var(--hero);color:#f5f5f7;border-radius:30px;padding:36px 36px 32px;min-height:300px;
  display:flex;flex-direction:column;justify-content:space-between;gap:28px}
.hero h1{font:700 clamp(38px,4.6vw,58px)/1.04 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.035em;margin:0;max-width:11em}
.hero h1 .g{color:var(--green)}
.hero .lead{color:#a1a1a6;font-size:17px;line-height:1.45;margin:14px 0 0;max-width:34ch;letter-spacing:-.015em}
.hero .foot{display:flex;align-items:flex-end;justify-content:space-between;gap:20px;flex-wrap:wrap}
.hero .cta{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.hero .note{font-size:13.5px;color:#a1a1a6;max-width:24ch;line-height:1.35}
.stat{text-align:right;margin-left:auto}
.stat b{display:block;font:700 44px/1 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.03em}
.stat span{display:block;color:#a1a1a6;font-size:13px;margin-top:4px;white-space:nowrap}
.stat.idle b{color:#6e6e73}

/* ---------- settings ---------- */
.field{margin-bottom:16px}
.field:last-child{margin-bottom:0}
.field > label,.field > .lbl{display:block;font-weight:600;font-size:14px;margin-bottom:6px}
.field small{display:block;color:var(--soft);font-size:12.5px;margin-top:5px}
select,input[type=text],input[type=search]{width:100%;background:var(--row);border:2px solid transparent;border-radius:14px;padding:9px 12px;min-height:44px}
select:focus,input:focus{outline:none;border-color:var(--green)}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.where{display:flex;align-items:center;gap:10px;background:var(--row);border-radius:14px;padding:9px 12px;min-height:44px}
.where svg{flex:none;color:var(--green-ink)}
.where span{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font-size:14px}

/* ---------- rows (habit-list style) ---------- */
.rows{display:grid;gap:10px}
.rows.cols{grid-template-columns:repeat(2,minmax(0,1fr))}
@media (max-width:900px){.rows.cols{grid-template-columns:1fr}}
.row{display:grid;grid-template-columns:42px minmax(0,1fr) auto;align-items:center;gap:14px;background:var(--row);border-radius:18px;padding:12px 0 12px 12px;position:relative;overflow:hidden}
.row.open{cursor:pointer}
.row.open:hover{background:color-mix(in srgb,var(--row) 70%,var(--rule))}
.ic{width:42px;height:42px;border-radius:12px;background:var(--cream);display:grid;place-items:center;color:var(--ink)}
.name{font-weight:600;font-size:15.5px;line-height:1.3;overflow-wrap:anywhere}
.sum{font-size:13.5px;line-height:1.4;color:var(--ink);opacity:.82;margin-top:3px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;max-width:62ch}
.status{display:flex;align-items:center;gap:6px;font-size:13px;margin-top:2px;color:var(--soft);flex-wrap:wrap}
.status .s{display:inline-flex;align-items:center;gap:4px;font-weight:600}
.s.done{color:var(--green-ink)} .s.new{color:var(--new)} .s.run{color:var(--green-ink)} .s.fail{color:var(--fail)} .s.wait{color:var(--soft)}
.acts{display:flex;gap:14px;flex-wrap:wrap;margin-top:6px}
.acts a,.acts button{background:none;border:0;padding:0;font-size:13px;font-weight:600;color:var(--ink);cursor:pointer;text-decoration:underline;
  text-decoration-color:var(--rule);text-decoration-thickness:2px;text-underline-offset:3px}
.acts a:hover,.acts button:hover{text-decoration-color:var(--green)}
.acts .quiet{color:var(--soft);font-weight:500}
.cell{min-width:86px;align-self:stretch;display:flex;flex-direction:column;align-items:center;justify-content:center;
  border-left:2px solid var(--rule);padding:0 14px;text-align:center}
.cell b{font-size:18px;font-weight:700;line-height:1.1}
.cell span{font-size:12px;color:var(--soft)}
.bar{position:absolute;left:0;bottom:0;height:5px;background:var(--green);transition:width .6s}
.err{grid-column:2/4;color:var(--fail);font-size:12.5px;white-space:pre-wrap;padding-right:14px}
.live{grid-column:2/4;font-size:13.5px;line-height:1.45;color:var(--ink);opacity:.7;padding:0 14px 6px 0;
  display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden;
  -webkit-mask-image:linear-gradient(90deg,transparent 0,#000 28px);mask-image:linear-gradient(90deg,transparent 0,#000 28px)}
u.unsure{text-decoration:underline dotted #c27c0e;text-decoration-thickness:2px;text-underline-offset:4px;cursor:pointer;border-radius:4px}
u.unsure:hover{background:var(--unsure-bg)}
.rec{text-align:center;padding:10px 0 6px}
.rec .clock{font:600 54px/1 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.rec .state{display:inline-flex;align-items:center;gap:8px;font-size:13.5px;font-weight:600;color:var(--fail);margin-bottom:14px}
.rec .state i{width:9px;height:9px;border-radius:50%;background:var(--fail);animation:blink 1.2s infinite}
.rec .state.paused{color:var(--soft)} .rec .state.paused i{background:var(--soft);animation:none}
@keyframes blink{50%{opacity:.25}}
.meter{display:flex;align-items:center;justify-content:center;gap:3px;height:46px;margin:18px 0 6px}
.meter span{width:4px;border-radius:2px;background:var(--green);height:4px;transition:height .08s}
.rec .sub{font-size:13px;color:var(--soft);margin-top:8px}
.rec .btns{justify-content:center;margin-top:22px}

.empty{background:var(--row);border-radius:18px;padding:22px 24px}
.empty b{display:block;font-size:16px;font-weight:600}
.empty p{margin:4px 0 0;color:var(--soft);font-size:14px}

.topr{display:flex;align-items:center;gap:10px}
.qpill{display:inline-flex;align-items:center;gap:8px;background:#1d1d1f;color:#f5f5f7;border-radius:99px;padding:8px 16px;font-size:13.5px;font-weight:600;white-space:nowrap}
.dot{width:9px;height:9px;border-radius:50%;background:#6c7064}
.dot.on{background:#50a65c;box-shadow:0 0 0 4px rgba(80,166,92,.3)}
.qpill[hidden]{display:none}
@media (max-width:560px){.qpill{display:none!important}}

/* ---------- drop overlay ---------- */
.overlay{position:fixed;inset:0;z-index:30;background:rgba(29,29,31,.86);backdrop-filter:blur(8px);-webkit-backdrop-filter:blur(8px);display:none;place-items:center;color:#fff;text-align:center}
.overlay.on{display:grid}

.overlay p{font-size:17px;color:#a1a1a6;margin:12px 0 0}

/* ---------- dialogs ---------- */
dialog{border:0;padding:0;border-radius:30px;width:min(860px,calc(100vw - 24px));max-height:calc(100vh - 40px);background:var(--cream);color:var(--ink);
  box-shadow:0 40px 90px -30px rgba(0,0,0,.5)}
dialog.small{width:min(560px,calc(100vw - 24px))}
dialog::backdrop{background:rgba(35,35,35,.55)}
.dlg{display:flex;flex-direction:column;max-height:calc(100vh - 40px)}
.d-head{padding:26px 28px 16px}
.d-top{display:flex;justify-content:space-between;gap:16px;align-items:flex-start}
.d-head h3{margin:0;font:700 clamp(22px,2.6vw,28px)/1.15 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.025em;overflow-wrap:anywhere}
.d-meta{color:var(--soft);font-size:13.5px;margin-top:8px}
.d-sum{font-size:15px;line-height:1.45;margin-top:12px;max-width:70ch}
.chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
.aibox{border-top:2px solid var(--rule);padding:16px 28px;max-height:40vh;overflow:auto;background:var(--tray)}
.aibox .ah{display:flex;align-items:center;justify-content:space-between;gap:12px;margin-bottom:6px}
.aibox .ah b{font-size:13px;letter-spacing:.04em;text-transform:uppercase;color:var(--soft)}
.aibox .md{font-size:14.5px;line-height:1.55;max-width:72ch}
.aibox .md h4{font-size:14.5px;margin:12px 0 2px}
.aibox .md ul{margin:4px 0;padding-left:20px}
.aibox .md p{margin:6px 0}
.aibox .busy{display:flex;align-items:center;gap:10px;font-size:14px;color:var(--soft)}
.spin{width:16px;height:16px;border-radius:50%;border:2.5px solid var(--rule);border-top-color:var(--green);animation:sp .9s linear infinite;flex:none}
@keyframes sp{to{transform:rotate(360deg)}}
.fix{display:grid;grid-template-columns:1fr auto;gap:4px 14px;padding:10px 0;border-bottom:1px solid var(--rule);font-size:14.5px}
.fix:last-child{border-bottom:0}
.fix .chg del{color:var(--fail);text-decoration-thickness:1.5px} .fix .chg ins{color:var(--green-ink);text-decoration:none;font-weight:600}
.fix .why{grid-column:1;font-size:12.5px;color:var(--soft)}
.fix .fa{grid-row:1/3;grid-column:2;display:flex;gap:12px;align-items:center}
.aiopts{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}
.aiopt{text-align:left;background:var(--row);border:2px solid transparent;border-radius:16px;padding:14px 14px 12px;cursor:pointer;font:inherit;color:inherit}
.aiopt b{display:block;font-size:15px}
.aiopt span{display:block;font-size:12.5px;color:var(--soft);margin-top:3px;line-height:1.35}
.aiopt[aria-pressed="true"]{border-color:var(--green);background:var(--pick)}
.aidet{margin-top:16px;font-size:14px;line-height:1.5}
.aidet table{border-collapse:collapse;width:100%;margin:10px 0 14px;font-size:13.5px}
.aidet td{padding:6px 0;border-bottom:1px solid var(--rule);vertical-align:top}
.aidet td:first-child{color:var(--soft);width:30%;padding-right:12px}
.aidet code{background:var(--row);border-radius:6px;padding:2px 6px;font-size:13px}
.aidet input,.aidet select{width:100%}
.okline{color:var(--green-ink);font-weight:600} .badline{color:var(--fail);font-weight:600}
@media (max-width:640px){.aiopts{grid-template-columns:1fr}}
.chip{background:var(--row);border-radius:99px;padding:3px 10px;font-size:12.5px;color:var(--soft)}
.x{width:38px;height:38px;border-radius:50%;border:0;background:var(--row);cursor:pointer;display:grid;place-items:center;flex:none}
.x:hover{background:var(--rule)}
.d-bar{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin-top:18px}
.seg{display:flex;background:var(--row);border-radius:99px;padding:3px}
.seg button{border:0;background:transparent;border-radius:99px;padding:6px 14px;font-weight:600;font-size:13.5px;cursor:pointer;color:var(--soft)}
.seg button[aria-pressed=true]{background:var(--ink);color:var(--cream)}
.glist .seg button{padding:5px 12px;font-size:13px}
.d-bar input{flex:1;min-width:170px}
.hits{font-size:13px;color:var(--soft)}
.d-body{overflow:auto;padding:6px 28px 26px;border-top:2px solid var(--rule)}
.d-body .text{max-width:68ch;font-size:16px;line-height:1.65}
.d-body p{margin:18px 0 0}
.d-body .line{display:grid;grid-template-columns:70px 1fr;gap:12px;margin-top:12px;max-width:72ch;font-size:15.5px;line-height:1.6}
.d-body .ts{font-size:12.5px;line-height:2;color:var(--green-ink);font-weight:600}
mark{background:var(--mark);color:inherit;border-radius:3px;padding:0 1px}
.d-foot{display:flex;gap:12px;justify-content:space-between;align-items:center;flex-wrap:wrap;padding:16px 28px 22px;border-top:2px solid var(--rule)}
.d-foot .acts{margin:0}
.d-body.plain{border-top:0;padding-top:0}
.pathbox{display:flex;align-items:center;gap:10px;background:var(--row);border-radius:16px;padding:14px 16px;font-weight:600;overflow-wrap:anywhere}
.pathbox svg{flex:none;color:var(--green-ink)}
.btns{display:flex;gap:10px;flex-wrap:wrap;margin-top:14px}
.paste{margin-top:22px;padding-top:18px;border-top:2px solid var(--rule)}
.paste .lbl{font-weight:600;font-size:14px;margin-bottom:6px}
.paste .inline{display:flex;gap:10px}
.paste .inline input{flex:1}
.formerr{color:var(--fail);font-size:13px;margin-top:10px;min-height:1em}

/* ---------- update banner + setup ---------- */
.banner{display:flex;align-items:center;justify-content:space-between;gap:14px;flex-wrap:wrap;background:var(--ink);color:var(--cream);
  border-radius:18px;padding:12px 14px 12px 20px;margin-bottom:18px;font-size:14.5px}
.banner .btn{border-color:var(--cream);color:var(--cream)}
.banner .btn.go{border-color:var(--green)}
.banner .link{color:#a1a1a6}
.setup{margin-bottom:18px}
.steps{display:grid;gap:10px}
.step{display:grid;grid-template-columns:34px 1fr auto;gap:14px;align-items:center;background:var(--row);border-radius:18px;padding:14px 16px}
.step .num{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;background:var(--cream);font-weight:700}
.step.ok .num{background:var(--green);color:#fff}
.step b{display:block;font-weight:600}
.step p{margin:2px 0 0;color:var(--soft);font-size:13.5px}
.step code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--cream);border-radius:8px;padding:6px 8px;display:block;margin-top:8px;overflow-wrap:anywhere}
.mini{height:5px;background:var(--rule);border-radius:99px;overflow:hidden;margin-top:8px;max-width:420px}
.mini i{display:block;height:100%;background:var(--green);transition:width .5s}
.spk-note{font-size:12.5px;color:var(--soft);margin-top:6px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
/* speakers in reader */
.turn{margin-top:16px;max-width:72ch}
.d-body.playable [data-s]{cursor:pointer;border-radius:10px;margin-left:-10px;padding:2px 10px;transition:background .15s}
.d-body.playable [data-s]:hover{background:var(--hover)}
.d-body.playable [data-s].playing{background:var(--playing)}
.player{display:flex;align-items:center;gap:10px;font-size:13px;color:var(--soft)}
.player button{width:34px;height:34px;border-radius:50%;border:0;background:var(--ink);color:var(--cream);display:grid;place-items:center;cursor:pointer;flex:none}
.player button:disabled{opacity:.35;cursor:default}
.player b{font-variant-numeric:tabular-nums;color:var(--ink);font-weight:600}
.player .rate{width:auto;height:28px;padding:0 9px;border-radius:8px;background:var(--row);color:var(--ink);font:600 12.5px/1 inherit;font-variant-numeric:tabular-nums}
.d-body [data-s]{position:relative}
.d-body .ed{position:absolute;top:4px;right:6px;opacity:0;border:0;background:var(--cream);color:var(--soft);font:600 12px/1 inherit;
  padding:5px 8px;border-radius:7px;cursor:pointer;box-shadow:0 0 0 1px var(--rule)}
.d-body [data-s]:hover .ed,.d-body .ed:focus{opacity:1}
.d-body textarea.edit{width:100%;font:inherit;line-height:1.55;color:var(--ink);background:var(--cream);border:2px solid var(--green);
  border-radius:10px;padding:8px 10px;resize:vertical;min-height:64px;display:block}
.edbar{display:flex;gap:12px;align-items:center;margin-top:6px;font-size:12.5px;color:var(--soft);grid-column:1/-1;flex-wrap:wrap}
.d-body .flash{animation:flash 1.8s ease-out}
@keyframes flash{0%,30%{background:var(--playing)}100%{background:transparent}}
.thits{margin:0 0 18px}
.thits .th{font-size:13px;font-weight:600;color:var(--soft);margin:0 2px 8px}
.thit{background:var(--row);border-radius:16px;padding:12px 16px;margin-bottom:8px}
.thit .tn{font-weight:600;font-size:14.5px}.thit .tn span{color:var(--soft);font-weight:500}
.thit button{display:grid;grid-template-columns:52px 1fr;gap:10px;width:100%;text-align:left;background:none;border:0;padding:6px 0 0;
  font:inherit;font-size:14px;line-height:1.45;color:var(--ink);cursor:pointer}
.thit button:hover .tx2{text-decoration:underline;text-decoration-color:var(--rule);text-underline-offset:3px}
.thit .tt2{color:var(--green-ink);font-weight:600;font-size:12.5px;padding-top:1px}
.turn .who{display:flex;align-items:baseline;gap:10px;margin-bottom:2px}
.turn .nm{background:none;border:0;padding:0;font:600 14px -apple-system,BlinkMacSystemFont,sans-serif;cursor:pointer;color:var(--c)}
.turn .nm:hover{text-decoration:underline;text-underline-offset:3px}
.turn .tt{font-size:12px;color:var(--soft)}
.turn .tx{font-size:16px;line-height:1.6}
.turn input{font:600 14px -apple-system,sans-serif;padding:3px 8px;min-height:0;width:180px;border-radius:8px}
.hint{font-size:12.5px;color:var(--soft);margin-top:14px}
/* format picker */
.fmts{display:grid;gap:8px}
.fmt{display:flex;justify-content:space-between;align-items:center;gap:12px;width:100%;text-align:left;background:var(--row);border:2px solid transparent;
  border-radius:14px;padding:12px 14px;cursor:pointer}
.fmt:hover{border-color:var(--green)}
.fmt b{display:block;font-weight:600}
.fmt span{color:var(--soft);font-size:13px}
.fmt em{font-style:normal;font-size:12.5px;color:var(--soft);white-space:nowrap}
/* voice memos */
.btn.ghost-light{border-color:#5a5a5f;color:#f5f5f7}
.btn.ghost-light:hover{background:#f5f5f7;color:#1d1d1f}
dialog.wide{width:min(760px,calc(100vw - 24px))}
.vm-tools{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:16px}
.vm-tools input[type=search]{flex:1;min-width:180px}
.vm-list{display:grid;gap:6px;padding-top:14px}
.vm{display:grid;grid-template-columns:24px 1fr auto;gap:12px;align-items:center;background:var(--row);border-radius:14px;padding:11px 14px;cursor:pointer}
.vm.off{opacity:.55;cursor:default}
.vm input{width:18px;height:18px;accent-color:var(--green);margin:0}
.vm b{display:block;font-weight:600;overflow-wrap:anywhere}
.vm span{font-size:13px;color:var(--soft)}
.badge{font-size:12px;font-weight:600;border-radius:99px;padding:2px 9px;white-space:nowrap;background:var(--cream);color:var(--soft)}
.badge.ok{color:var(--green-ink)}
.badge.fail{color:var(--fail)}
.switch{display:flex;align-items:center;gap:10px;font-size:14px;cursor:pointer}
.switch input{width:38px;height:22px;appearance:none;-webkit-appearance:none;background:var(--rule);border-radius:99px;position:relative;cursor:pointer;margin:0;transition:background .2s}
.switch input::after{content:"";position:absolute;top:2px;left:2px;width:18px;height:18px;border-radius:50%;background:#fff;transition:transform .2s;box-shadow:0 1px 2px rgba(0,0,0,.25)}
.switch input:checked{background:var(--green)}
.switch input:checked::after{transform:translateX(16px)}
.perm ol{margin:14px 0 0;padding:0;list-style:none;display:grid;gap:10px;counter-reset:s}
.perm li{display:grid;grid-template-columns:30px 1fr auto;gap:12px;align-items:center;background:var(--row);border-radius:14px;padding:12px 14px;counter-increment:s}
.perm li::before{content:counter(s);width:30px;height:30px;border-radius:50%;background:var(--cream);display:grid;place-items:center;font-weight:700}
.perm li p{margin:2px 0 0;font-size:13px;color:var(--soft)}
.perm code{font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace;background:var(--cream);border-radius:6px;padding:1px 6px;overflow-wrap:anywhere}
.warnline{font-size:12.5px;color:var(--fail);margin-top:6px;line-height:1.4}
.foot-note{text-align:center;color:var(--soft);font-size:12.5px;margin-top:22px}
.toast{position:fixed;left:50%;bottom:28px;transform:translate(-50%,14px);opacity:0;background:var(--green);color:#fff;
  padding:10px 18px;border-radius:99px;font-size:14px;font-weight:600;transition:.2s;pointer-events:none;z-index:40;max-width:calc(100vw - 32px);text-align:center}
.toast.on{opacity:1;transform:translate(-50%,0)}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}

/* ======================================================================
   Glass layer: translucent surfaces over a soft colour wash (macOS-style).
   Placed last so it refines everything above without changing behaviour.
   ====================================================================== */
:root{
  --wash:radial-gradient(1100px 620px at 8% -8%, rgba(80,166,92,.20), transparent 62%),
         radial-gradient(900px 600px at 102% 4%, rgba(90,140,235,.16), transparent 58%),
         radial-gradient(900px 700px at 55% 112%, rgba(190,140,230,.13), transparent 62%),
         #f2f2f6;
  --glass:rgba(255,255,255,.58);
  --glass-strong:rgba(252,252,254,.80);
  --glass-edge:rgba(255,255,255,.75);
  --hair:rgba(60,60,67,.12);
  --fill:rgba(120,120,128,.12);
  --fill-2:rgba(120,120,128,.20);
  --lift:0 1px 0 rgba(255,255,255,.6) inset, 0 1px 2px rgba(0,0,0,.04), 0 12px 32px -12px rgba(0,0,0,.12);
  --ring:0 0 0 3px rgba(80,166,92,.30);
  --row:var(--fill);
  --rule:var(--hair);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --wash:radial-gradient(1100px 620px at 8% -8%, rgba(80,166,92,.075), transparent 62%),
           radial-gradient(900px 600px at 102% 4%, rgba(90,120,220,.07), transparent 58%),
           radial-gradient(900px 700px at 55% 112%, rgba(150,110,210,.05), transparent 62%),
           #0f0f11;
    --glass:rgba(38,38,42,.55);
    --glass-strong:rgba(34,34,38,.84);
    --glass-edge:rgba(255,255,255,.08);
    --hair:rgba(235,235,245,.10);
    --fill:rgba(118,118,128,.20);
    --fill-2:rgba(118,118,128,.30);
    --lift:0 1px 0 rgba(255,255,255,.05) inset, 0 12px 32px -12px rgba(0,0,0,.6);
    --row:var(--fill);
    --rule:var(--hair);
  }
}
:root[data-theme="dark"]{
    --wash:radial-gradient(1100px 620px at 8% -8%, rgba(80,166,92,.075), transparent 62%),
           radial-gradient(900px 600px at 102% 4%, rgba(90,120,220,.07), transparent 58%),
           radial-gradient(900px 700px at 55% 112%, rgba(150,110,210,.05), transparent 62%),
           #0f0f11;
    --glass:rgba(38,38,42,.55);
    --glass-strong:rgba(34,34,38,.84);
    --glass-edge:rgba(255,255,255,.08);
    --hair:rgba(235,235,245,.10);
    --fill:rgba(118,118,128,.20);
    --fill-2:rgba(118,118,128,.30);
    --lift:0 1px 0 rgba(255,255,255,.05) inset, 0 12px 32px -12px rgba(0,0,0,.6);
    --row:var(--fill);
    --rule:var(--hair);
  }

body{background:var(--wash);background-attachment:fixed;letter-spacing:normal}
.glass,.drop,.panel,.top{background:var(--glass);-webkit-backdrop-filter:blur(28px) saturate(170%);backdrop-filter:blur(28px) saturate(170%);
  border:1px solid var(--glass-edge);box-shadow:var(--lift)}

/* header: a floating toolbar */
.top{position:sticky;top:12px;z-index:20;padding:8px 10px 8px 12px;border-radius:18px;margin-bottom:18px}
.brand{font-size:17px;gap:9px}
.brand svg{width:28px;height:28px;border-radius:23%;filter:drop-shadow(0 .5px .5px rgba(0,0,0,.22)) drop-shadow(0 2px 4px rgba(0,0,0,.12))}
.tb{height:32px;border-radius:10px}
.tb:hover{background:var(--fill);color:var(--ink)}

/* surfaces */
.drop,.panel{border-radius:24px}
.drop{padding:36px 30px 18px}
.drop h1{font-size:30px;font-weight:650;letter-spacing:-.022em;margin-bottom:26px}
h2{font-size:21px;font-weight:650;letter-spacing:-.018em}
.count{font-weight:500}

/* buttons: Apple-style capsules, one height everywhere */
.btn{height:38px;padding:0 17px;border:0;border-radius:99px;font-size:14px;font-weight:600;letter-spacing:-.005em;gap:7px;
  background:var(--fill);color:var(--ink);transition:background .15s, transform .1s, filter .15s}
.btn:hover{background:var(--fill-2);color:var(--ink)}
.btn:active{transform:scale(.98)}
.btn:focus-visible{outline:none;box-shadow:var(--ring)}
.btn.go{background:var(--green);color:#fff;box-shadow:0 1px 0 rgba(255,255,255,.25) inset, 0 6px 16px -8px rgba(48,130,62,.7)}
.btn.go:hover{background:var(--green);filter:brightness(1.05)}
.btn.soft{background:var(--fill)}
.btn.soft:hover{background:var(--fill-2)}
.btn.dark{background:var(--ink);color:var(--cream)}
.btn.rec{background:transparent;border:1px solid var(--hair);box-shadow:0 1px 2px rgba(0,0,0,.04)}
.btn.rec:hover{background:var(--fill);border-color:var(--hair)}
.drop .sep{height:22px;background:var(--hair)}
.chiprow{border-top:1px solid var(--hair);margin-top:26px;padding-top:12px}
.opt{border-radius:8px}
.opt:hover,.opt:focus-within{background:var(--fill)}

/* fields */
html{scroll-padding-top:84px}
.drop,.panel,.banner{scroll-margin-top:84px}
select,input[type=text],input[type=search],input[type=password]{background:var(--fill);border:0;border-radius:10px;min-height:38px;padding:8px 12px;color:var(--ink);
  transition:box-shadow .15s, background .15s}
select:focus,input:focus{box-shadow:var(--ring);border:0}
.hintpop{background:var(--fill);border:0}
.hintpop input{background:var(--cream)}
.where{background:var(--fill);border-radius:12px}

/* lists */
.row{background:var(--fill)}
.row.lib{background:transparent;border-top:1px solid var(--hair);border-radius:0;transition:background .12s}
.row.lib:hover,.row.lib:focus-visible{background:var(--fill);border-radius:12px;outline:none}
.row.lib:hover + .row.lib{border-top-color:transparent}
.thit{background:var(--fill)}
.bar{height:3px;border-radius:2px}

/* dialogs: frosted sheets */
dialog{background:var(--glass-strong);-webkit-backdrop-filter:blur(40px) saturate(180%);backdrop-filter:blur(40px) saturate(180%);
  border:1px solid var(--glass-edge);border-radius:24px;box-shadow:0 30px 80px -24px rgba(0,0,0,.35), 0 0 0 .5px rgba(0,0,0,.08)}
dialog::backdrop{background:rgba(20,20,24,.22);-webkit-backdrop-filter:blur(6px);backdrop-filter:blur(6px)}
.d-body{border-top:1px solid var(--hair)}
.d-foot{border-top:1px solid var(--hair)}
.d-head h3{font-weight:650;letter-spacing:-.02em}
.x{background:var(--fill)}
.aibox{background:transparent;border-top:1px solid var(--hair)}

/* drag-and-drop overlay */
.overlay{background:rgba(242,242,246,.55);-webkit-backdrop-filter:blur(26px) saturate(160%);backdrop-filter:blur(26px) saturate(160%);color:var(--ink)}
.ov-card{display:grid;justify-items:center;gap:10px;padding:56px 72px;border-radius:32px;border:2px dashed rgba(80,166,92,.55);
  background:var(--glass-strong);box-shadow:0 30px 80px -30px rgba(0,0,0,.35)}
.ov-card svg{color:var(--green)}
.ov-title{font:650 34px/1.15 -apple-system,BlinkMacSystemFont,"SF Pro Display",sans-serif;letter-spacing:-.02em}
.ov-card p{margin:0;font-size:15px;color:var(--soft);letter-spacing:normal}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]) .overlay{background:rgba(12,12,14,.55)}
}
:root[data-theme="dark"] .overlay{background:rgba(12,12,14,.55)}


/* toast: small dark glass pill */
.toast{background:rgba(30,30,32,.82);-webkit-backdrop-filter:blur(20px) saturate(160%);backdrop-filter:blur(20px) saturate(160%);
  color:#fff;border:1px solid rgba(255,255,255,.08);box-shadow:0 12px 30px -10px rgba(0,0,0,.4);font-weight:500}
.qpill{background:var(--fill);color:var(--ink)}

/* setup card and banner follow the same surfaces */
.banner{background:var(--glass);color:var(--ink);-webkit-backdrop-filter:blur(28px);backdrop-filter:blur(28px);border:1px solid var(--glass-edge)}
.foot-note{color:var(--soft)}

/* ---------- 3.5: compact hero, list menu, reading column, focus, skeleton, toast ---------- */
.glyph{flex:none}
.drop.compact{padding:14px 16px 14px 18px;display:flex;align-items:center;justify-content:space-between;gap:10px 14px;flex-wrap:wrap;text-align:left}
.drop.compact h1{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);margin:0}
.drop.compact .cta{gap:6px;justify-content:flex-start;flex-wrap:nowrap}
.drop.compact .grp{gap:6px;flex-wrap:nowrap}
.drop.compact .btn{height:34px;padding:0 13px;font-size:13.5px;gap:6px}
.drop.compact .sep,.drop.compact .long{display:none}
.drop.compact .chiprow{margin:0;padding:0;border:0;gap:0;justify-content:flex-end;flex-wrap:nowrap}
.drop.compact .opt{gap:4px;padding:4px 0 4px 8px}
.drop.compact .opt select{padding-right:16px}
.drop.compact .hintpop,.drop.compact .spk-note{flex-basis:100%}
.drop.compact .spk-note{justify-content:flex-start;margin-top:0}
body.dragging .drop.compact{padding:30px 30px 18px;display:block;text-align:center}
body.dragging .drop.compact h1{position:static;width:auto;height:auto;clip:auto;margin:0 0 18px}
body.dragging .drop.compact .cta{justify-content:center}
body.dragging .drop.compact .chiprow{justify-content:center;margin-top:18px}

.mwrap{position:relative}
.more-btn{width:34px;height:34px;padding:0;display:grid;place-items:center;border:0;background:transparent;color:var(--soft);cursor:pointer}
.more-btn[aria-expanded="true"]{background:var(--fill);color:var(--ink)}
.menu{position:absolute;right:0;top:40px;z-index:25;min-width:190px;padding:5px;border-radius:14px;background:color-mix(in srgb, var(--cream) 97%, transparent);
  -webkit-backdrop-filter:blur(30px) saturate(180%);backdrop-filter:blur(30px) saturate(180%);border:1px solid var(--glass-edge);
  box-shadow:0 18px 40px -14px rgba(0,0,0,.35), 0 0 0 .5px rgba(0,0,0,.06);animation:fadeIn .12s ease-out}
.menu button{display:flex;align-items:center;gap:10px;width:100%;border:0;background:transparent;color:var(--ink);font:500 14px/1 inherit;
  padding:9px 10px;border-radius:9px;cursor:pointer;text-align:left}
.menu button svg{color:var(--soft)}
.menu button:hover:not(:disabled),.menu button:focus-visible{background:var(--green);color:#fff;outline:none;box-shadow:none}
.menu button:hover:not(:disabled) svg,.menu button:focus-visible svg{color:#fff}
.menu button:disabled{opacity:.4;cursor:default}

/* speakers view: a share-of-talk bar, then each person's lines together */
.spkview{max-width:calc(66ch + 82px)}
.sbar{display:flex;gap:2px;height:8px;border-radius:99px;overflow:hidden;margin:18px 0 4px}
.sbar span{min-width:3px}
.sgrp{margin-top:22px;padding:14px 16px 12px;border-radius:14px;background:color-mix(in srgb,var(--c) 7%,transparent);border-left:3px solid var(--c)}
.sgh{display:flex;align-items:center;gap:8px;flex-wrap:wrap;margin-bottom:4px}
.sgh .av{width:22px;height:22px;border-radius:50%;background:var(--c);color:#fff;display:grid;place-items:center;font:700 10px/1 var(--ui,-apple-system),sans-serif;flex:none}
.sgh .nm{background:none;border:0;padding:0;font:600 14px var(--ui,-apple-system),sans-serif;cursor:pointer;color:var(--c)}
.sgh .nm:hover{text-decoration:underline;text-underline-offset:3px}
.sgh input{font:600 14px var(--ui,-apple-system),sans-serif;padding:3px 8px;min-height:0;width:180px;border-radius:8px}
.sgm{font-size:12px;color:var(--soft)}
.sgrp .line{margin-top:6px}
/* reading column: ~65 characters, centred in the sheet */
.d-body{font-size:16px}
.d-body .text{max-width:66ch;font-size:16px;line-height:1.7;text-wrap:pretty}
.d-body .text p{margin:0 0 1em}
.d-body .text p:first-child{margin-top:18px}
.d-body .line{max-width:calc(66ch + 82px);margin-top:4px;line-height:1.65;font-size:15.5px}
.d-body .line:first-child{margin-top:14px}
.turn{max-width:calc(66ch + 32px);margin-top:26px}
.turn:first-child{margin-top:18px}
.turn .who{align-items:center;margin-bottom:4px}
.turn .tx{margin-left:10px;padding-left:20px;border-left:2px solid color-mix(in srgb,var(--c) 45%,transparent);line-height:1.7;text-wrap:pretty}
.turn .tt{font-variant-numeric:tabular-nums}

/* focus: one soft green ring everywhere, never Chrome's blue */
:focus{outline:none}
:focus-visible{outline:2px solid rgba(80,166,92,.6);outline-offset:2px;border-radius:10px}
.btn:focus-visible,.tb:focus-visible,.x:focus-visible,.seg button:focus-visible,.opt:focus-within{outline:none;box-shadow:0 0 0 3px rgba(80,166,92,.35)}
.row.lib:focus-visible{box-shadow:inset 0 0 0 2px rgba(80,166,92,.55)}
dialog:focus-visible{outline:none}
input,select,textarea{accent-color:var(--green)}
::selection{background:rgba(80,166,92,.28)}

/* skeleton rows while the list loads */
.skel{display:grid;gap:22px;padding:18px 8px 10px}
.skel i{display:block;height:12px;border-radius:6px;background:linear-gradient(90deg,var(--fill) 0%,var(--fill-2) 40%,var(--fill) 80%);background-size:300% 100%;
  animation:shimmer 1.3s ease-in-out infinite}
.skel i:nth-child(1){width:58%}.skel i:nth-child(2){width:72%}.skel i:nth-child(3){width:46%}
@keyframes shimmer{0%{background-position:100% 0}100%{background-position:0 0}}

/* 3.6: list groups, richer rows, row menu, reader nav + rename, AI dot */
.grp-h{font-size:12.5px;font-weight:650;letter-spacing:.01em;color:var(--soft);padding:16px 8px 6px;text-transform:none}
.grp-h:first-child{padding-top:4px}
.grp-h + .row.lib{border-top-color:transparent}
.row.lib .meta{display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.row.lib .meta span{display:inline-flex;align-items:center;gap:5px}
.row.lib .meta svg{opacity:.75;flex:none}
.row.lib .sum.ai{color:var(--ink);opacity:.86}
.row.lib .aib{color:var(--green);margin-right:6px;display:inline-block;vertical-align:-1px}
.row.lib .when{font-variant-numeric:tabular-nums}
.rmore{width:30px;padding:0!important;display:inline-grid;place-items:center}
.row.lib.menu-open{background:var(--fill);border-radius:12px}
.row.lib.menu-open .acts{opacity:1;pointer-events:auto}
.rowmenu{position:fixed;right:auto;top:0;z-index:45}
.menu .msep{height:1px;background:var(--hair);margin:4px 8px}
.menu button.danger{color:var(--fail)}
.menu button.danger svg{color:var(--fail)}
.menu button.danger:hover,.menu button.danger:focus-visible{background:var(--fail);color:#fff}
.menu button.danger:hover svg,.menu button.danger:focus-visible svg{color:#fff}
.d-top{align-items:flex-start}
.d-title{flex:1;min-width:0}
.navs{display:flex;gap:6px;margin-left:auto}
.navs .x:disabled{opacity:.35;cursor:default}
#rName{cursor:text;border-radius:8px;margin-left:-6px;padding:0 6px}
#rName:hover{background:var(--fill)}
#rName:has(input){background:none}
#rName input.rn{font:inherit;letter-spacing:inherit;color:inherit;width:100%;background:var(--fill);border:0;border-radius:8px;padding:0 6px;margin:0 -6px;min-height:0;box-shadow:var(--ring)}
.sugt{display:flex;align-items:baseline;gap:8px;flex-wrap:wrap;font-size:13.5px;color:var(--soft);margin:2px 0 8px}
.sugt b{color:var(--ink);font-weight:600}
.toast .tact{border:0;background:none;color:#8fe39b;font:650 14px/1 inherit;margin-left:14px;padding:4px 2px;cursor:pointer}
.toast .tact:hover{text-decoration:underline;text-underline-offset:3px}
.toast.has-act{pointer-events:auto}
.vh{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);white-space:nowrap}
#aiOpen{position:relative;width:34px;padding:0;justify-content:center}
.aidot{position:absolute;right:5px;top:6px;width:7px;height:7px;border-radius:50%;background:var(--soft);opacity:.5;box-shadow:0 0 0 2px var(--glass-strong)}
#aiOpen.on .aidot{background:var(--green);opacity:1}
#aiOpen.on{color:var(--ink)}

/* ======================================================================
   3.7 — calmer: glass only on the floating toolbar (Apple: "avoid glass on glass,
   keep content non-glass"), flat content on a quiet canvas, rows instead of cards.
   ====================================================================== */
:root{--canvas:#f6f6f7;--dash:rgba(60,60,67,.24)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--canvas:#151517;--dash:rgba(235,235,245,.2)}}
:root[data-theme="dark"]{--canvas:#151517;--dash:rgba(235,235,245,.2)}
body{background:var(--canvas)}

/* content surfaces: no glass, no shadow */
.panel,.drop{background:transparent;-webkit-backdrop-filter:none;backdrop-filter:none;box-shadow:none;border:0}
.panel{padding:0 4px;border-radius:0}
.stack{gap:34px;margin-top:30px}
.ph{margin:0 4px 6px;align-items:center}
h2{font-size:17px;font-weight:650;letter-spacing:-.012em}
.count{color:var(--soft);font-weight:500;margin-left:4px}
.setup{background:var(--cream);border:1px solid var(--hair);border-radius:18px;padding:20px 22px}
.banner{-webkit-backdrop-filter:none;backdrop-filter:none;background:var(--cream);border:1px solid var(--hair)}
#filter{background:var(--fill);min-height:32px;height:32px;padding:6px 12px;font-size:13.5px;width:220px}

/* the drop zone: a dashed outline is the whole affordance */
.drop{border:1.5px dashed var(--dash);border-radius:20px}
.drop.compact{position:relative;padding:12px 12px 12px 14px;gap:12px}
.dz{display:none}
.drop.compact .dz{display:flex;align-items:center;gap:12px;min-width:0;flex:1}
.dzmark{width:38px;height:38px;border-radius:11px;display:grid;place-items:center;flex:none;color:var(--green);
  background:color-mix(in srgb, var(--green) 10%, transparent)}
.dzt{display:flex;flex-direction:column;gap:1px;min-width:0}
.dzt b{font-size:14px;font-weight:600;color:var(--ink);letter-spacing:-.005em}
.optsum{display:inline-flex;align-items:center;gap:5px;border:0;background:none;padding:0;font:13px/1.3 inherit;color:var(--soft);cursor:pointer;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:100%;text-align:left}
.optsum:hover,.optsum[aria-expanded="true"]{color:var(--ink)}
.optsum svg{flex:none;transition:transform .15s}
.optsum[aria-expanded="true"] svg{transform:rotate(180deg)}
.drop.compact .cta{flex:none}
.drop.compact .opts{display:none}
.drop.compact .opts.open{display:block;position:absolute;left:12px;top:calc(100% + 8px);z-index:30;width:340px;padding:8px;border-radius:16px;
  background:var(--glass-strong);-webkit-backdrop-filter:blur(30px) saturate(180%);backdrop-filter:blur(30px) saturate(180%);
  border:1px solid var(--glass-edge);box-shadow:0 20px 50px -18px rgba(0,0,0,.35), 0 0 0 .5px rgba(0,0,0,.06);animation:fadeIn .12s ease-out}
.drop.compact .opts .chiprow{display:grid;gap:2px;justify-content:stretch;margin:0;padding:0;border:0}
.drop.compact .opts .opt{display:flex;justify-content:space-between;padding:8px 10px;font-size:13.5px;border-radius:10px}
.drop.compact .opts .opt select{font-size:13.5px}
.drop.compact .opts .hintpop{margin:6px 2px 2px}
.drop.compact .opts .spk-note{padding:4px 10px}
.drop.compact .btn.rec{background:transparent}
.drop:hover{border-color:color-mix(in srgb, var(--green) 45%, var(--dash))}
body.dragging .drop{border-style:solid;border-color:var(--green);background:color-mix(in srgb, var(--green) 6%, transparent);box-shadow:none;transform:none}
body.dragging .drop.compact .dz,body.dragging .drop.compact .cta,body.dragging .drop.compact .opts{display:none}
body.dragging .drop.compact h1{margin:6px 0}
body.dragging .drop.compact{padding:26px}

/* rows: hairlines on the canvas; actions appear on hover */
.row.lib{padding-left:10px;padding-right:10px;margin:0 -6px}
.row.lib:hover,.row.lib:focus-visible{background:color-mix(in srgb, var(--ink) 4.5%, transparent)}
.grp-h{padding:20px 4px 6px;font-size:12px;letter-spacing:.02em;color:var(--soft)}
.grp-h:first-child{padding-top:8px}
.row.lib .name{font-size:15px}
.row.qrow{padding:14px 4px}
.foot-note{margin-top:40px;opacity:.8}

/* toolbar: no box; once the page scrolls it gets a full-width frosted strip and a hairline (macOS style) */
.top{position:sticky;top:0;margin:-22px 0 14px;padding:16px 4px 12px;border:0;border-radius:0;background:transparent;box-shadow:none;
  -webkit-backdrop-filter:none;backdrop-filter:none;transition:background .2s, box-shadow .2s}
.top.scrolled{background:color-mix(in srgb, var(--canvas) 86%, transparent);-webkit-backdrop-filter:blur(20px) saturate(160%);backdrop-filter:blur(20px) saturate(160%);
  box-shadow:0 0 0 100vmax color-mix(in srgb, var(--canvas) 86%, transparent);clip-path:inset(0 -100vmax)}
.top::after{content:"";position:absolute;left:-100vmax;right:-100vmax;bottom:0;height:1px;background:var(--hair);opacity:0;transition:opacity .2s}
.top.scrolled::after{opacity:1}
html{scroll-padding-top:70px}

/* ---------- 3.8 ---------- */
/* header: just the icon */
.brand svg{width:26px;height:26px}
/* two-line rows; actions replace the time/meta on hover (like Mail) */
.row.lib{padding-top:11px;padding-bottom:11px}
.row.lib .name{line-height:1.3}
.row.lib .sum{margin-top:3px;font-size:13.5px}
.row.lib .lr{position:relative;gap:4px;min-width:150px;padding-top:1px}
.row.lib .meta{margin:0;gap:10px;font-size:12px;justify-content:flex-end;flex-wrap:nowrap;white-space:nowrap}
.row.lib .when,.row.lib .meta{transition:opacity .12s}
.row.lib .acts{position:absolute;right:0;top:50%;transform:translateY(-50%);gap:4px;flex-wrap:nowrap;pointer-events:none}
.row.lib:hover .acts,.row.lib:focus-within .acts,.row.lib.menu-open .acts{pointer-events:auto}
.row.lib:hover .when,.row.lib:hover .meta,.row.lib:focus-within .when,.row.lib:focus-within .meta,.row.lib.menu-open .when,.row.lib.menu-open .meta{opacity:0}
.row.lib .acts button{text-decoration:none;font-weight:600;font-size:12.5px;color:var(--ink);padding:5px 9px;border-radius:8px;background:color-mix(in srgb, var(--ink) 6%, transparent)}
.row.lib .acts button:hover{background:color-mix(in srgb, var(--ink) 11%, transparent)}
.row.lib .acts .rmore{width:28px;padding:5px 0!important}
/* in progress: the first group of the list */
.qblock{margin-bottom:6px}
.qh{display:flex;align-items:baseline;gap:8px;padding:8px 4px 6px;font-size:12px;color:var(--soft)}
.qh .qt{font-weight:650;color:var(--green-ink);letter-spacing:.02em}
.qh .qsub{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.qh .qtools{display:flex;gap:14px}
.qh .link{font-size:12.5px}
.qblock + .thits + .rows .grp-h:first-child,.qblock:not([hidden]) ~ #library .grp-h:first-child{padding-top:20px}
.row.qrow{margin:0 -6px;padding:11px 10px;border-radius:0}
/* 4. one primary action; the others are quiet */
.drop.compact .grp{display:contents}
.drop.compact #choose{order:1}.drop.compact #vmOpen{order:2}.drop.compact #recOpen{order:3}
.drop.compact #vmOpen,.drop.compact #recOpen{background:transparent;border:0;box-shadow:none;color:var(--soft);padding:0 10px;font-weight:550}
.drop.compact #vmOpen:hover,.drop.compact #recOpen:hover{background:var(--fill);color:var(--ink)}
.drop.compact #recOpen .recdot{width:8px;height:8px}
/* 5. reader and other sheets: opaque, fewer lines */
dialog{background:var(--cream);-webkit-backdrop-filter:none;backdrop-filter:none;border:1px solid var(--hair)}
.d-foot{border-top:0;padding-top:10px}
.pbar{background:transparent}
.aibox{border-top:1px solid var(--hair)}

/* ---------- 3.9: monochrome, warm grays, ⌘K ---------- */
:root{--canvas:#f7f7f5;--soft:#6f6e6a;--hair:rgba(55,50,40,.085);--fill:rgba(120,114,100,.10);--fill-2:rgba(120,114,100,.17);--dash:rgba(55,50,40,.2);
  --ring-c:rgba(25,25,25,.22)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--canvas:#161615;--soft:#9b9993;--hair:rgba(240,235,225,.075);--fill:rgba(150,145,135,.14);
  --fill-2:rgba(150,145,135,.22);--dash:rgba(240,235,225,.17);--cream:#1d1d1c;--ring-c:rgba(255,255,255,.28)}}
:root[data-theme="dark"]{--canvas:#161615;--soft:#9b9993;--hair:rgba(240,235,225,.075);--fill:rgba(150,145,135,.14);
  --fill-2:rgba(150,145,135,.22);--dash:rgba(240,235,225,.17);--cream:#1d1d1c;--ring-c:rgba(255,255,255,.28)}
/* one primary colour: ink. Green is kept for state (live, progress, new, recording) */
.btn.go,.btn.go:hover{background:var(--ink);color:var(--canvas);box-shadow:0 1px 2px rgba(0,0,0,.14);filter:none}
.btn.go:hover{opacity:.88}
.banner .btn.go{border-color:var(--ink)}
.link{color:var(--ink)}
.tb.on,.row.lib:hover .name{color:inherit}
:focus-visible{outline-color:var(--ring-c)}
.btn:focus-visible,.tb:focus-visible,.x:focus-visible,.seg button:focus-visible,.opt:focus-within,.ckbtn:focus-visible{box-shadow:0 0 0 3px var(--ring-c)}
.row.lib:focus-visible{box-shadow:inset 0 0 0 1.5px var(--ring-c)}
select:focus,input:focus{box-shadow:0 0 0 3px var(--ring-c)}
input,select,textarea{accent-color:var(--ink)}
::selection{background:color-mix(in srgb, var(--ink) 15%, transparent)}
.menu button:hover:not(:disabled),.menu button:focus-visible{background:var(--fill-2);color:var(--ink)}
.menu button:hover:not(:disabled) svg,.menu button:focus-visible svg{color:var(--ink)}
.row.lib .aib{color:var(--soft)}
.d-body .ts,.thit .tt2{color:var(--soft);font-weight:500}
.fmt:hover,.aiopt[aria-pressed="true"]{border-color:var(--ink)}
.aiopt[aria-pressed="true"]{background:var(--fill)}
.switch input:checked{background:var(--ink)}
.d-body textarea.edit{border-color:var(--ink)}
.where svg,.pathbox svg{color:var(--soft)}
.fix .chg ins{color:var(--ink)}
/* no decoration */
.dzmark{width:auto;height:auto;background:none;color:var(--soft);border-radius:0}
.drop:hover{border-color:color-mix(in srgb, var(--ink) 28%, transparent)}
body.dragging .drop{border-color:var(--ink);background:color-mix(in srgb, var(--ink) 3.5%, transparent)}
body.dragging .drop h1{color:var(--ink)}
.chips{gap:0}
.chips .chip{background:none;padding:0;border-radius:0;font-size:12.5px;color:var(--soft)}
.chips .chip + .chip::before{content:"·";margin:0 7px;opacity:.6}
kbd{font:500 11px/1 ui-monospace,SFMono-Regular,Menlo,monospace;letter-spacing:0;padding:3px 5px;border-radius:5px;background:var(--fill);color:var(--soft);border:0}
.btn kbd.kb{background:color-mix(in srgb, var(--canvas) 18%, transparent);color:inherit;opacity:.7;margin-left:2px;padding:2px 5px}
.drop:not(.compact) .btn kbd.kb{display:none}
/* header search button */
.ckbtn{display:inline-flex;align-items:center;gap:8px;height:32px;padding:0 6px 0 11px;border:0;border-radius:9px;background:var(--fill);color:var(--soft);
  font:500 13px/1 inherit;cursor:pointer;min-width:190px}
.ckbtn span{flex:1;text-align:left}
.ckbtn:hover{background:var(--fill-2);color:var(--ink)}
.topr{gap:6px}
/* command menu */
dialog.cmdk{width:min(620px,calc(100vw - 32px));margin-top:12vh;max-height:min(520px,72vh);padding:0;border-radius:16px;overflow:hidden;
  display:flex;flex-direction:column;background:color-mix(in srgb, var(--cream) 92%, transparent);-webkit-backdrop-filter:blur(30px) saturate(170%);backdrop-filter:blur(30px) saturate(170%);
  box-shadow:0 30px 80px -20px rgba(0,0,0,.45), 0 0 0 .5px rgba(0,0,0,.1)}
dialog.cmdk:not([open]){display:none}
dialog.cmdk::backdrop{background:rgba(10,10,10,.18);-webkit-backdrop-filter:none;backdrop-filter:none}
.ck-in{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--hair);color:var(--soft)}
.ck-in input{flex:1;background:none;border:0;box-shadow:none!important;min-height:0;padding:2px 0;font-size:16px;color:var(--ink)}
.ck-list{overflow:auto;padding:6px;flex:1}
.ck-sec{font-size:11.5px;font-weight:600;color:var(--soft);padding:10px 10px 4px}
.ck-item{display:flex;align-items:center;gap:11px;padding:8px 10px;border-radius:9px;cursor:pointer;color:var(--ink);font-size:14px}
.ck-item svg{flex:none;color:var(--soft)}
.ck-item.on{background:var(--fill-2)}
.ck-item.on svg{color:var(--ink)}
.ck-l{flex:1;min-width:0;display:flex;flex-direction:column;gap:2px}
.ck-t{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ck-s{font-size:12.5px;color:var(--soft);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ck-m{font-size:12px;color:var(--soft);font-variant-numeric:tabular-nums;flex:none}
.ck-empty{padding:28px 12px;text-align:center;color:var(--soft);font-size:14px}
.ck-foot{display:flex;gap:14px;align-items:center;padding:9px 14px;border-top:1px solid var(--hair);font-size:12px;color:var(--soft)}
.ck-foot kbd{margin-right:3px}
.ck-sp{flex:1}
@media (max-width:640px){.ckbtn{min-width:0}.ckbtn span,.ckbtn kbd{display:none}}

/* ---------- 3.9: type scale + numerals, HUD toasts ---------- */
:root{--mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,monospace}
#filter{display:none}
/* numbers, times and durations in the mono face, a touch smaller so they sit with the text */
.row.lib .when,.row.lib .meta span,.d-body .ts,.turn .tt,.pbar .pt,.ck-m,.row.qrow .pct,.thit .tt2,.d-meta{font-family:var(--mono);letter-spacing:-.01em}
.row.lib .when{font-size:11.5px}
.row.lib .meta{font-size:11px}
.d-body .ts,.turn .tt,.thit .tt2{font-size:11.5px}
.pbar .pt,.ck-m,.row.qrow .pct{font-size:11.5px}
.d-meta{font-size:12px}
/* a short type scale: 22 / 17 / 15 / 13.5 / 12 */
h2{font-size:17px;letter-spacing:-.015em}
.row.lib .name{font-size:14.5px;font-weight:600;letter-spacing:-.006em}
.row.lib .sum{font-size:13.5px}
.grp-h,.qh{font-size:12px}
.d-head h3{font-size:22px;letter-spacing:-.02em}
/* toasts: small HUD that scales in */
.toast[popover]{padding:8px 14px;font-size:13px;font-weight:500;border-radius:12px;bottom:24px;
  transform:translate(-50%,6px) scale(.96);transition:opacity .16s ease, transform .2s cubic-bezier(.2,.9,.3,1.2)}
.toast.on{transform:translate(-50%,0) scale(1)}
.toast .tact{font-size:13px;color:#fff;text-decoration:underline;text-underline-offset:3px;text-decoration-color:rgba(255,255,255,.4)}
mark{background:color-mix(in srgb, #f5c400 32%, transparent)}

/* ---------- 3.9.1: green is the brand again; more air ---------- */
:root{--brand:#3f9a52;--brand-ink:#2f7d40;--brand-tint:rgba(63,154,82,.10)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--brand:#4cb262;--brand-ink:#79d38a;--brand-tint:rgba(76,178,98,.14)}}
:root[data-theme="dark"]{--brand:#4cb262;--brand-ink:#79d38a;--brand-tint:rgba(76,178,98,.14)}
/* where green lives: the primary action, the drop zone, focus, live state, AI, selection */
.btn.go,.btn.go:hover{background:var(--brand);color:#fff;box-shadow:0 1px 0 rgba(255,255,255,.18) inset, 0 4px 12px -6px color-mix(in srgb, var(--brand) 70%, transparent)}
.btn.go:hover{opacity:1;filter:brightness(1.05)}
.btn kbd.kb{background:rgba(255,255,255,.18);color:#fff;opacity:.85}
.dzmark{width:40px;height:40px;border-radius:12px;background:var(--brand-tint);color:var(--brand-ink)}
.drop:hover{border-color:color-mix(in srgb, var(--brand) 45%, var(--dash))}
body.dragging .drop{border-color:var(--brand);background:var(--brand-tint)}
body.dragging .drop h1{color:var(--brand-ink)}
.row.lib .aib{color:var(--brand)}
:root{--ring-c:color-mix(in srgb, var(--brand) 38%, transparent)}
::selection{background:color-mix(in srgb, var(--brand) 24%, transparent)}
input,select,textarea{accent-color:var(--brand)}
.switch input:checked{background:var(--brand)}
.aiopt[aria-pressed="true"]{border-color:var(--brand);background:var(--brand-tint)}
.d-body textarea.edit{border-color:var(--brand)}
.d-body .ts,.thit .tt2{color:var(--brand-ink)}
.fix .chg ins{color:var(--brand-ink)}
.qh .qt{color:var(--brand-ink)}
.ck-item.on svg{color:var(--brand-ink)}

/* more air: roomier rows, larger gaps, a calmer rhythm */
.wrap{padding-top:28px}
.top{padding:20px 4px 14px;margin-bottom:22px}
.drop.compact{padding:16px 16px 16px 18px;gap:16px}
.dzt{gap:3px}
.dzt b{font-size:14.5px}
.stack{margin-top:44px}
.ph{margin-bottom:10px}
h2{font-size:18px}
.grp-h{padding:26px 4px 8px}
.grp-h:first-child{padding-top:10px}
.row.lib{padding-top:16px;padding-bottom:16px;border-top-color:color-mix(in srgb, var(--hair) 80%, transparent)}
.row.lib .name{font-size:15px;line-height:1.35}
.row.lib .sum{font-size:14px;margin-top:5px;line-height:1.45}
.row.lib .lr{gap:6px}
.row.lib .when{font-size:12px}
.row.lib .meta{font-size:11.5px;gap:12px}
.row.qrow{padding:16px 10px}
.qh{padding:10px 4px 8px}
.foot-note{margin-top:56px}

/* ---------- 3.9.2: the drop zone is the hero — about a third of the window ---------- */
.drop.compact{flex-direction:column;justify-content:center;align-items:center;text-align:center;gap:0;
  min-height:clamp(230px, 33vh, 360px);padding:28px 24px;border-radius:24px}
.drop.compact .dz,.drop.compact .dzt{display:contents}
.drop.compact .dzmark{order:1;width:52px;height:52px;border-radius:16px;margin-bottom:14px}
.drop.compact .dzmark svg{width:24px;height:24px}
.drop.compact .dzt b{order:2;font-size:22px;font-weight:650;letter-spacing:-.02em;margin-bottom:6px}
.drop.compact .cta{order:4;margin-top:20px;justify-content:center;flex-wrap:wrap}
.drop.compact .optsum{order:3;font-size:13.5px}
.drop.compact .btn{height:38px;padding:0 16px;font-size:14px}
.drop.compact #vmOpen,.drop.compact #recOpen{padding:0 12px}
.drop.compact .opts.open{left:50%;transform:translateX(-50%);width:340px;text-align:left}
.drop.compact .opts.open{background:color-mix(in srgb, var(--cream) 98%, transparent)}
:root:not([data-theme="dark"]) .drop{--dash:rgba(55,50,40,.26)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .drop{--dash:rgba(240,235,225,.17)}}
body.dragging .drop.compact{min-height:clamp(230px, 33vh, 360px);padding:28px 24px;display:flex}
body.dragging .drop.compact .dzmark,body.dragging .drop.compact .dzt b,body.dragging .drop.compact .optsum{display:none}
body.dragging .drop.compact h1{font-size:26px}
.stack{margin-top:40px}

/* ---------- 3.10: glass hero, two columns (add on the left, settings on the right) ---------- */
body{background:
  radial-gradient(520px 260px at 28% 150px, rgba(63,154,82,.16), transparent 70%),
  radial-gradient(480px 260px at 78% 190px, rgba(80,120,230,.12), transparent 70%),
  var(--canvas)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body{background:
  radial-gradient(520px 260px at 28% 150px, rgba(76,178,98,.13), transparent 70%),
  radial-gradient(480px 260px at 78% 190px, rgba(90,120,230,.10), transparent 70%), var(--canvas)}}
:root[data-theme="dark"] body{background:
  radial-gradient(520px 260px at 28% 150px, rgba(76,178,98,.13), transparent 70%),
  radial-gradient(480px 260px at 78% 190px, rgba(90,120,230,.10), transparent 70%), var(--canvas)}
.drop.compact{display:grid;grid-template-columns:minmax(0,1.65fr) minmax(250px,1fr);align-items:stretch;padding:0;gap:0;text-align:left;
  min-height:clamp(230px, 33vh, 360px);border:1px solid rgba(255,255,255,.7);border-radius:24px;
  background:rgba(255,255,255,.55);-webkit-backdrop-filter:blur(30px) saturate(170%);backdrop-filter:blur(30px) saturate(170%);
  box-shadow:0 1px 0 rgba(255,255,255,.8) inset, 0 1px 2px rgba(0,0,0,.04), 0 20px 50px -24px rgba(30,40,30,.22)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .drop.compact{background:rgba(40,40,38,.5);border-color:rgba(255,255,255,.08);
  box-shadow:0 1px 0 rgba(255,255,255,.05) inset, 0 20px 50px -24px rgba(0,0,0,.6)}}
:root[data-theme="dark"] .drop.compact{background:rgba(40,40,38,.5);border-color:rgba(255,255,255,.08);box-shadow:0 1px 0 rgba(255,255,255,.05) inset, 0 20px 50px -24px rgba(0,0,0,.6)}
.drop.compact:hover{border-color:rgba(255,255,255,.7)}
.drop.compact .dzl{display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;padding:28px 24px}
.drop.compact .optsum{display:none}
.drop.compact .dzt b{margin-bottom:4px}
.drop.compact .dzl .dzsub{display:block}
.dzsub{display:none;font-size:13.5px;color:var(--soft);order:3}
.drop.compact .cta{margin-top:20px}
/* right column: always-visible settings */
.opts-h{display:none}
.drop.compact .opts,.drop.compact .opts.open{display:flex;flex-direction:column;position:static;transform:none;width:auto;top:auto;left:auto;animation:none;
  background:transparent;-webkit-backdrop-filter:none;backdrop-filter:none;border:0;border-left:1px solid var(--hair);border-radius:0;box-shadow:none;padding:18px;justify-content:center}
.drop.compact .opts-h{display:block;font-size:12px;font-weight:600;color:var(--soft);padding:0 10px 8px}
.drop.compact .opts .chiprow{display:flex;flex-direction:column;gap:1px}
.drop.compact .opts .opt{width:100%;justify-content:space-between;padding:8px 10px;font-size:13.5px;border-radius:10px;color:var(--soft)}
.drop.compact .opts .opt b,.drop.compact .opts .opt select{color:var(--ink);font-weight:500}
.drop.compact .opts .opt:hover{background:var(--fill)}
.aiopt-row b::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--soft);opacity:.5;margin-right:7px;vertical-align:1.5px}
.aiopt-row.on b::before{background:var(--brand);opacity:1}
.drop:not(.compact) .aiopt-row{display:none}
/* dragging: the whole card lights up, the left side says what will happen */
body.dragging .drop.compact{display:grid;padding:0;border-color:var(--brand);box-shadow:0 0 0 4px var(--brand-tint), 0 20px 50px -24px rgba(30,60,30,.3)}
body.dragging .drop.compact .dzl{background:var(--brand-tint);border-radius:23px 0 0 23px}
body.dragging .drop.compact .opts{display:flex;text-align:left}
body.dragging .drop.compact h1{position:static;width:auto;height:auto;clip:auto;margin:0;font-size:24px;color:var(--brand-ink)}
@media (max-width:720px){.drop.compact{grid-template-columns:1fr}.drop.compact .opts{border-left:0;border-top:1px solid var(--hair)}}

/* ---------- 3.10.2: drop moment — a living, translucent green ---------- */
@property --spin{syntax:"<angle>";inherits:false;initial-value:0deg}
.drop.compact{isolation:isolate}
.drop.compact::before,.drop.compact::after{content:"";position:absolute;pointer-events:none;opacity:0;transition:opacity .35s ease}
/* aurora: soft green light drifting under the glass */
.drop.compact::before{inset:-30%;z-index:-1;filter:blur(38px) saturate(130%);
  background:
    radial-gradient(38% 42% at 30% 40%, rgba(76,190,110,.55), transparent 70%),
    radial-gradient(34% 40% at 70% 60%, rgba(40,170,140,.45), transparent 70%),
    radial-gradient(30% 36% at 55% 25%, rgba(150,220,120,.40), transparent 70%),
    radial-gradient(40% 40% at 20% 80%, rgba(60,150,230,.22), transparent 70%)}
/* light running round the edge */
.drop.compact::after{inset:0;border-radius:inherit;padding:1.5px;
  background:conic-gradient(from var(--spin), rgba(76,190,110,0) 0deg, rgba(110,220,140,.95) 70deg, rgba(76,190,110,0) 150deg,
    rgba(76,190,110,0) 210deg, rgba(150,230,170,.8) 280deg, rgba(76,190,110,0) 340deg);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude}
@keyframes aurora{0%{transform:translate(-6%,-4%) rotate(0deg) scale(1)}50%{transform:translate(5%,4%) rotate(14deg) scale(1.08)}100%{transform:translate(-3%,6%) rotate(-10deg) scale(1.02)}}
@keyframes spin{to{--spin:360deg}}
@keyframes bob{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes breathe{0%,100%{opacity:.82}50%{opacity:1}}
body.dragging .drop.compact{overflow:hidden;border-color:rgba(255,255,255,.55);
  background:rgba(255,255,255,.38);box-shadow:0 1px 0 rgba(255,255,255,.8) inset, 0 0 0 6px rgba(76,178,98,.10), 0 30px 70px -28px rgba(30,120,60,.45)}
body.dragging .drop.compact::before{opacity:1;animation:aurora 7s ease-in-out infinite alternate}
body.dragging .drop.compact::after{opacity:1;animation:spin 4.5s linear infinite}
body.dragging .drop.compact .dzl{background:none}
body.dragging .drop.compact .dz{display:contents}
body.dragging .drop.compact .dzt b{display:none}
body.dragging .drop.compact .dzmark{display:grid;order:1;width:60px;height:60px;border-radius:18px;margin-bottom:16px;color:#fff;
  background:linear-gradient(160deg, #6fd08a, #2f9a50);box-shadow:0 1px 0 rgba(255,255,255,.45) inset, 0 12px 28px -10px rgba(30,140,70,.7);
  animation:bob 2.4s ease-in-out infinite}
body.dragging .drop.compact .dzmark svg{width:28px;height:28px}
body.dragging .drop.compact h1{order:2;font-size:24px;font-weight:650;letter-spacing:-.02em;color:#1f6b33;animation:breathe 2.4s ease-in-out infinite}
body.dragging .drop.compact .opts{opacity:.5;transition:opacity .3s}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact{background:rgba(30,40,32,.42);border-color:rgba(140,230,160,.18);
  box-shadow:0 1px 0 rgba(255,255,255,.06) inset, 0 0 0 6px rgba(76,178,98,.10), 0 30px 70px -28px rgba(0,0,0,.7)}
  :root:not([data-theme="light"]) body.dragging .drop.compact h1{color:#a6ebb4}}
:root[data-theme="dark"] body.dragging .drop.compact{background:rgba(30,40,32,.42);border-color:rgba(140,230,160,.18);
  box-shadow:0 1px 0 rgba(255,255,255,.06) inset, 0 0 0 6px rgba(76,178,98,.10), 0 30px 70px -28px rgba(0,0,0,.7)}
:root[data-theme="dark"] body.dragging .drop.compact h1{color:#a6ebb4}
@media (prefers-reduced-motion:reduce){body.dragging .drop.compact::before,body.dragging .drop.compact::after,
  body.dragging .drop.compact .dzmark,body.dragging .drop.compact h1{animation:none}}

/* ---------- 3.10.3: drop moment, refined ---------- */
.dz-glow{display:none}
body.dragging .drop.compact .dz-glow{display:block;position:absolute;inset:-4px;border-radius:28px;pointer-events:none;z-index:0;filter:blur(10px);opacity:.9}
.dz-glow i{position:absolute;inset:0;border-radius:inherit;padding:5px;
  background:conic-gradient(from var(--spin), rgba(76,190,110,0) 0deg, rgba(110,225,145,1) 70deg, rgba(76,190,110,0) 150deg,
    rgba(76,190,110,0) 210deg, rgba(140,230,200,.9) 280deg, rgba(76,190,110,0) 340deg);
  -webkit-mask:linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);-webkit-mask-composite:xor;mask-composite:exclude;
  animation:spin 4.5s linear infinite}
body.dragging .drop.compact{overflow:visible;transform:scale(1.006);transition:transform .35s cubic-bezier(.2,.9,.3,1.15), box-shadow .3s, background .3s;
  background-image:linear-gradient(180deg, rgba(255,255,255,.42), rgba(255,255,255,0) 42%), url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .09 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>");
  background-blend-mode:normal, overlay}
body.dragging .drop.compact::before{inset:0;border-radius:inherit;overflow:hidden;
  background:
    radial-gradient(42% 52% at 28% 42%, rgba(70,195,110,.62), transparent 70%),
    radial-gradient(36% 46% at 72% 58%, rgba(30,175,150,.48), transparent 70%),
    radial-gradient(30% 40% at 52% 22%, rgba(170,230,130,.42), transparent 70%),
    radial-gradient(44% 44% at 18% 85%, rgba(70,140,235,.22), transparent 70%),
    radial-gradient(30% 34% at 88% 18%, rgba(120,220,190,.30), transparent 70%);
  background-size:140% 140%;filter:blur(30px) saturate(135%);animation:aurora2 9s ease-in-out infinite alternate}
@keyframes aurora2{0%{background-position:0% 0%}33%{background-position:60% 30%}66%{background-position:30% 80%}100%{background-position:100% 50%}}
/* icon: a small pane of green glass, with ripples */
body.dragging .drop.compact .dzmark{position:relative;background:linear-gradient(160deg,#8fe3a6 0%,#43b064 52%,#2a8848 100%);
  box-shadow:0 1px 0 rgba(255,255,255,.6) inset, 0 -1px 0 rgba(0,0,0,.12) inset, 0 14px 30px -12px rgba(30,140,70,.75), 0 0 0 1px rgba(255,255,255,.25)}
body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{content:"";position:absolute;inset:0;border-radius:inherit;
  border:1.5px solid rgba(90,205,125,.55);animation:ripple 2.6s cubic-bezier(.2,.6,.3,1) infinite}
body.dragging .drop.compact .dzmark::after{animation-delay:1.3s}
@keyframes ripple{0%{transform:scale(1);opacity:.7}100%{transform:scale(1.9);opacity:0}}
body.dragging .drop.compact .dzmark svg{filter:drop-shadow(0 1px 1px rgba(0,60,20,.35))}
/* title: green into teal */
body.dragging .drop.compact h1{background:linear-gradient(95deg,#1d6a33 10%,#1c8a6c 90%);-webkit-background-clip:text;background-clip:text;color:transparent;
  font-size:25px;letter-spacing:-.022em}
body.dragging .drop.compact .opts{opacity:.4;filter:saturate(.6)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact{
  background-image:linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,0) 42%), url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .09 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>")}
  :root:not([data-theme="light"]) body.dragging .drop.compact h1{background-image:linear-gradient(95deg,#b9f2c6 10%,#93e4d3 90%)}}
:root[data-theme="dark"] body.dragging .drop.compact{background-image:linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,0) 42%), url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='180' height='180'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='.9' numOctaves='2' stitchTiles='stitch'/><feColorMatrix values='0 0 0 0 0  0 0 0 0 0  0 0 0 0 0  0 0 0 .09 0'/></filter><rect width='100%' height='100%' filter='url(%23n)'/></svg>")}
:root[data-theme="dark"] body.dragging .drop.compact h1{background-image:linear-gradient(95deg,#b9f2c6 10%,#93e4d3 90%)}
@media (prefers-reduced-motion:reduce){.dz-glow i,body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{animation:none}}

div.quiet-empty{display:flex;flex-direction:column;align-items:center;gap:14px;text-align:center}
div.quiet-empty p{margin:0}

/* ---------- 3.10.5 ---------- */
/* nothing wider than the window; no scroll on the first screen */
html,body{overflow-x:clip}
body{min-height:0}
.top::after{display:none}
.top.scrolled{box-shadow:none;clip-path:none}
.wrap{padding-bottom:24px}
.foot-note{margin-top:28px;margin-bottom:18px}
.drop.compact{min-height:clamp(220px, 33vh, 360px)}
/* drop: the mark stays where it is and comes alive; nothing jumps */
body.dragging .drop.compact .dz{display:contents}
body.dragging .drop.compact h1{position:absolute;width:1px;height:1px;overflow:hidden;clip:rect(0 0 0 0);margin:0;animation:none}
body.dragging .drop.compact .dzt b{display:block}
body.dragging .drop.compact .cta{display:flex;visibility:hidden;opacity:0}
.drop.compact .cta{transition:opacity .25s ease, visibility .25s}
.drop.compact .dzmark{position:relative;transition:box-shadow .45s ease, color .45s ease, transform .6s cubic-bezier(.2,.8,.2,1)}
body.dragging .drop.compact .dzmark{width:52px;height:52px;border-radius:16px;margin-bottom:14px;color:#fff;background:var(--brand-tint);
  box-shadow:inset 0 0 0 30px #3fa95c, inset 0 1px 0 rgba(255,255,255,.45), 0 10px 26px -10px rgba(40,150,80,.55), 0 0 0 6px rgba(76,178,98,.12);
  animation:float 3.2s ease-in-out infinite}
body.dragging .drop.compact .dzmark svg{width:24px;height:24px;filter:none}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-3px)}}
body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{border:1px solid rgba(90,205,125,.45);animation:halo 3.2s cubic-bezier(.25,.6,.3,1) infinite}
body.dragging .drop.compact .dzmark::after{animation-delay:1.6s}
@keyframes halo{0%{transform:scale(1);opacity:.55}100%{transform:scale(1.55);opacity:0}}
.drop.compact .dzt b{transition:color .35s ease}
body.dragging .drop.compact .dzt b{background:linear-gradient(95deg,#1d6a33 10%,#1c8a6c 90%);-webkit-background-clip:text;background-clip:text;color:transparent}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#b9f2c6 10%,#93e4d3 90%)}}
:root[data-theme="dark"] body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#b9f2c6 10%,#93e4d3 90%)}
body.dragging .drop.compact{transform:none;min-height:clamp(220px, 33vh, 360px);text-align:left}
body.dragging .drop.compact .chiprow{margin-top:0;justify-content:stretch}
@media (prefers-reduced-motion:reduce){body.dragging .drop.compact .dzmark{animation:none}}

/* ---------- 3.11: reader — summary first, no scrollbars ---------- */
dialog,dialog *{scrollbar-width:none}
dialog ::-webkit-scrollbar,dialog::-webkit-scrollbar{display:none;width:0;height:0}
html:has(dialog[open]){overflow:hidden}
button{-webkit-user-select:none;user-select:none}
.aiview{max-width:66ch;padding:18px 0 8px}
.aiview .aih{display:flex;justify-content:space-between;align-items:center;gap:12px;font-size:12px;color:var(--soft);margin-bottom:6px}
.aiview .aih .acts{margin:0;gap:4px}
.aiview .aih .acts button{text-decoration:none;font-size:12.5px;font-weight:600;color:var(--soft);padding:4px 8px;border-radius:7px}
.aiview .aih .acts button:hover{background:var(--fill);color:var(--ink)}
.aiview .md{font-size:15.5px;line-height:1.7;max-width:66ch}
.aiview .md h4{font-size:12.5px;font-weight:650;letter-spacing:.02em;color:var(--brand-ink);margin:26px 0 6px;text-transform:none}
.aiview .md h4:first-child{margin-top:8px}
.aiview .md p{margin:0 0 10px}
.aiview .md ul{margin:4px 0 8px;padding-left:20px}
.aiview .md li{margin:6px 0}
.aiview .md li::marker{color:var(--soft)}
.aiview .busy{display:flex;align-items:center;gap:10px;color:var(--soft);font-size:14.5px;padding:24px 0 6px}
.aiview .hint{margin-top:6px}
.aiview .sugt{margin:2px 0 6px}
#vAI[aria-pressed="true"]{background:var(--brand);color:#fff}

/* ---------- 3.11.1: colour only — deep forest green (reference: #1F4D35 + mint) ---------- */
:root{--brand:#1f4d35;--brand-ink:#1f4d35;--brand-tint:#e3f0e6;--green:#2f7a4c;--green-ink:#1f5a3a;
  --ring-c:color-mix(in srgb, #1f4d35 35%, transparent)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--brand:#2c6b49;--brand-ink:#9fd8b0;--brand-tint:rgba(80,170,115,.16);
  --green:#4fa86f;--green-ink:#9fd8b0;--ring-c:rgba(140,215,165,.4)}}
:root[data-theme="dark"]{--brand:#2c6b49;--brand-ink:#9fd8b0;--brand-tint:rgba(80,170,115,.16);--green:#4fa86f;--green-ink:#9fd8b0;--ring-c:rgba(140,215,165,.4)}
.btn.go,.btn.go:hover{box-shadow:0 1px 0 rgba(255,255,255,.12) inset, 0 6px 14px -8px rgba(20,60,40,.55)}
.btn.go:hover{filter:brightness(1.12)}
.dzmark{color:var(--brand-ink)}
/* drop moment: same motion, deeper greens */
body.dragging .drop.compact .dzmark{box-shadow:inset 0 0 0 30px var(--brand), inset 0 1px 0 rgba(255,255,255,.3), 0 10px 26px -10px rgba(20,70,45,.6), 0 0 0 6px rgba(31,77,53,.10)}
body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{border-color:rgba(31,77,53,.4)}
body.dragging .drop.compact{border-color:rgba(31,77,53,.35)}
body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#173d29 10%,#2a6b4a 90%)}
body.dragging .drop.compact::before{background:
    radial-gradient(42% 52% at 28% 42%, rgba(52,140,88,.50), transparent 70%),
    radial-gradient(36% 46% at 72% 58%, rgba(30,110,80,.40), transparent 70%),
    radial-gradient(30% 40% at 52% 22%, rgba(140,205,120,.36), transparent 70%),
    radial-gradient(44% 44% at 18% 85%, rgba(31,77,53,.30), transparent 70%),
    radial-gradient(30% 34% at 88% 18%, rgba(110,190,150,.28), transparent 70%);background-size:140% 140%}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#b9e6c6 10%,#8fd3a5 90%)}
  :root:not([data-theme="light"]) body.dragging .drop.compact{border-color:rgba(140,215,165,.2)}}
:root[data-theme="dark"] body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#b9e6c6 10%,#8fd3a5 90%)}
:root[data-theme="dark"] body.dragging .drop.compact{border-color:rgba(140,215,165,.2)}
.dz-glow i{background:conic-gradient(from var(--spin), rgba(31,77,53,0) 0deg, rgba(60,150,95,.95) 70deg, rgba(31,77,53,0) 150deg,
    rgba(31,77,53,0) 210deg, rgba(120,200,150,.85) 280deg, rgba(31,77,53,0) 340deg)}
.drop.compact::after{background:conic-gradient(from var(--spin), rgba(31,77,53,0) 0deg, rgba(70,160,105,.95) 70deg, rgba(31,77,53,0) 150deg,
    rgba(31,77,53,0) 210deg, rgba(130,205,155,.8) 280deg, rgba(31,77,53,0) 340deg)}
/* the soft light behind the card, a shade deeper */
body{background:radial-gradient(520px 260px at 28% 150px, rgba(31,95,60,.13), transparent 70%),
  radial-gradient(480px 260px at 78% 190px, rgba(90,160,120,.10), transparent 70%), var(--canvas)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body{background:radial-gradient(520px 260px at 28% 150px, rgba(60,140,95,.14), transparent 70%),
  radial-gradient(480px 260px at 78% 190px, rgba(60,120,90,.10), transparent 70%), var(--canvas)}}
:root[data-theme="dark"] body{background:radial-gradient(520px 260px at 28% 150px, rgba(60,140,95,.14), transparent 70%),
  radial-gradient(480px 260px at 78% 190px, rgba(60,120,90,.10), transparent 70%), var(--canvas)}
::selection{background:color-mix(in srgb, #1f4d35 20%, transparent)}

/* ---------- 3.12: a voice of its own — serif titles, paper, olive and lime ---------- */
/* Newsreader (SIL Open Font License 1.1, see docs/Newsreader-OFL.txt), embedded so it works offline */
@font-face{font-family:"Matita Serif";font-style:normal;font-weight:500;font-display:swap;src:url(data:font/woff2;base64,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) format("woff2")}
:root{--serif:"Matita Serif","New York",ui-serif,Charter,"Iowan Old Style",Georgia,serif}
.wm{font:500 21px/1 var(--serif);letter-spacing:-.015em;color:var(--ink)}
.brand{gap:10px}
.drop.compact .dzt b,body.dragging .drop.compact .dzt b{font-family:var(--serif);font-weight:500;font-size:28px;letter-spacing:-.02em;line-height:1.15}
h2,#lTitle{font-family:var(--serif);font-weight:500;font-size:24px;letter-spacing:-.018em}
.d-head h3,#rName{font-family:var(--serif);font-weight:500;font-size:30px;letter-spacing:-.022em;line-height:1.15}
dialog h3,.d-top h3{font-family:var(--serif);font-weight:500;letter-spacing:-.02em}
#rName input.rn{font-family:var(--serif)}

/* palette: warm paper, deep olive, a lime accent */
:root{--canvas:#f4f2ea;--cream:#fcfbf6;--ink:#1d1c17;--soft:#6f6c61;--hair:rgba(60,55,35,.10);--fill:rgba(110,100,60,.09);--fill-2:rgba(110,100,60,.16);
  --brand:#3e5a1f;--brand-ink:#3e5a1f;--brand-tint:#e8eecb;--lime:#d9e66e;--lime-ink:#2d3a0d;--green:#5f8a2c;--green-ink:#3e5a1f;
  --ring-c:color-mix(in srgb, #3e5a1f 35%, transparent);--dash:rgba(60,55,35,.24)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--canvas:#151511;--cream:#1d1c17;--ink:#f1efe4;--soft:#a29e8e;
  --hair:rgba(240,235,210,.08);--fill:rgba(170,160,120,.13);--fill-2:rgba(170,160,120,.21);
  --brand:#cfe06a;--brand-ink:#d3e27e;--brand-tint:rgba(207,224,106,.14);--lime:#cfe06a;--lime-ink:#1d2208;--green:#a6c94b;--green-ink:#d3e27e;
  --ring-c:rgba(207,224,106,.45);--dash:rgba(240,235,210,.18)}}
:root[data-theme="dark"]{--canvas:#151511;--cream:#1d1c17;--ink:#f1efe4;--soft:#a29e8e;
  --hair:rgba(240,235,210,.08);--fill:rgba(170,160,120,.13);--fill-2:rgba(170,160,120,.21);
  --brand:#cfe06a;--brand-ink:#d3e27e;--brand-tint:rgba(207,224,106,.14);--lime:#cfe06a;--lime-ink:#1d2208;--green:#a6c94b;--green-ink:#d3e27e;
  --ring-c:rgba(207,224,106,.45);--dash:rgba(240,235,210,.18)}
/* the primary button: olive on paper, lime in the dark */
.btn.go,.btn.go:hover{background:var(--brand);color:#fbfaf2;box-shadow:0 1px 0 rgba(255,255,255,.14) inset, 0 6px 16px -8px rgba(50,70,20,.55)}
.btn kbd.kb{background:rgba(255,255,255,.16);color:inherit}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .btn.go{color:var(--lime-ink);box-shadow:0 6px 18px -8px rgba(207,224,106,.45)}
  :root:not([data-theme="light"]) .btn kbd.kb{background:rgba(0,0,0,.12)}}
:root[data-theme="dark"] .btn.go{color:var(--lime-ink);box-shadow:0 6px 18px -8px rgba(207,224,106,.45)}
:root[data-theme="dark"] .btn kbd.kb{background:rgba(0,0,0,.12)}
/* lime as the accent: drop mark, AI tab, new dot, AI dot */
.dzmark{background:var(--lime);color:var(--lime-ink)}
#vAI[aria-pressed="true"]{background:var(--lime);color:var(--lime-ink)}
.newdot{background:var(--green)}
.aiopt-row.on b::before,#aiOpen.on .aidot{background:var(--green)}
.row.lib .aib{color:var(--green)}
.aiview .md h4{color:var(--brand-ink)}
.qh .qt{color:var(--brand-ink)}
::selection{background:color-mix(in srgb, var(--lime) 55%, transparent)}
/* glass card on warm light */
body{background:radial-gradient(560px 280px at 26% 150px, rgba(217,230,110,.28), transparent 70%),
  radial-gradient(500px 280px at 80% 190px, rgba(62,90,31,.10), transparent 70%), var(--canvas)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body{background:radial-gradient(560px 280px at 26% 150px, rgba(207,224,106,.10), transparent 70%),
  radial-gradient(500px 280px at 80% 190px, rgba(110,140,60,.10), transparent 70%), var(--canvas)}}
:root[data-theme="dark"] body{background:radial-gradient(560px 280px at 26% 150px, rgba(207,224,106,.10), transparent 70%),
  radial-gradient(500px 280px at 80% 190px, rgba(110,140,60,.10), transparent 70%), var(--canvas)}
.drop.compact{background:rgba(253,252,246,.62);border-color:rgba(255,255,255,.8)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .drop.compact{background:rgba(36,35,28,.55);border-color:rgba(255,250,220,.07)}}
:root[data-theme="dark"] .drop.compact{background:rgba(36,35,28,.55);border-color:rgba(255,250,220,.07)}
/* drop moment in olive and lime */
body.dragging .drop.compact .dzmark{box-shadow:inset 0 0 0 30px var(--brand), inset 0 1px 0 rgba(255,255,255,.3), 0 10px 26px -10px rgba(50,70,20,.55), 0 0 0 6px rgba(217,230,110,.25);color:#fbfaf2}
body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{border-color:rgba(120,150,40,.45)}
body.dragging .drop.compact{border-color:rgba(120,150,40,.35)}
body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#2f4614 10%,#5f7f22 90%)}
body.dragging .drop.compact::before{background:
    radial-gradient(42% 52% at 28% 42%, rgba(200,222,90,.55), transparent 70%),
    radial-gradient(36% 46% at 72% 58%, rgba(90,130,40,.38), transparent 70%),
    radial-gradient(30% 40% at 52% 22%, rgba(235,240,150,.50), transparent 70%),
    radial-gradient(44% 44% at 18% 85%, rgba(62,90,31,.28), transparent 70%),
    radial-gradient(30% 34% at 88% 18%, rgba(170,205,90,.32), transparent 70%);background-size:140% 140%}
.dz-glow i{background:conic-gradient(from var(--spin), rgba(120,150,40,0) 0deg, rgba(190,215,80,.95) 70deg, rgba(120,150,40,0) 150deg,
    rgba(120,150,40,0) 210deg, rgba(150,190,70,.85) 280deg, rgba(120,150,40,0) 340deg)}
.drop.compact::after{background:conic-gradient(from var(--spin), rgba(120,150,40,0) 0deg, rgba(185,210,80,.95) 70deg, rgba(120,150,40,0) 150deg,
    rgba(120,150,40,0) 210deg, rgba(150,190,70,.8) 280deg, rgba(120,150,40,0) 340deg)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact .dzmark{color:var(--lime-ink)}
  :root:not([data-theme="light"]) body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#e6efaa 10%,#cfe06a 90%)}}
:root[data-theme="dark"] body.dragging .drop.compact .dzmark{color:var(--lime-ink)}
:root[data-theme="dark"] body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#e6efaa 10%,#cfe06a 90%)}

/* ---------- 3.12.1: paper stays; the green goes back to the earlier fresh green ---------- */
:root{--brand:#3f9a52;--brand-ink:#2f7d40;--brand-tint:rgba(63,154,82,.10);--green:#50a65c;--green-ink:#2f7a3b;
  --ring-c:color-mix(in srgb, #3f9a52 38%, transparent)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--brand:#4cb262;--brand-ink:#79d38a;--brand-tint:rgba(76,178,98,.14);
  --green:#4cb262;--green-ink:#79d38a;--ring-c:rgba(121,211,138,.42)}}
:root[data-theme="dark"]{--brand:#4cb262;--brand-ink:#79d38a;--brand-tint:rgba(76,178,98,.14);--green:#4cb262;--green-ink:#79d38a;--ring-c:rgba(121,211,138,.42)}
.btn.go,.btn.go:hover{background:var(--brand);color:#fff;box-shadow:0 1px 0 rgba(255,255,255,.18) inset, 0 4px 12px -6px color-mix(in srgb, var(--brand) 70%, transparent)}
.btn.go:hover{filter:brightness(1.05)}
.btn kbd.kb{background:rgba(255,255,255,.18);color:#fff}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .btn.go{color:#fff}:root:not([data-theme="light"]) .btn kbd.kb{background:rgba(255,255,255,.18)}}
:root[data-theme="dark"] .btn.go{color:#fff}
:root[data-theme="dark"] .btn kbd.kb{background:rgba(255,255,255,.18)}
.dzmark{background:var(--brand-tint);color:var(--brand-ink)}
#vAI[aria-pressed="true"]{background:var(--brand);color:#fff}
::selection{background:color-mix(in srgb, var(--brand) 24%, transparent)}
body{background:radial-gradient(560px 280px at 26% 150px, rgba(63,154,82,.15), transparent 70%),
  radial-gradient(500px 280px at 80% 190px, rgba(80,120,230,.09), transparent 70%), var(--canvas)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body{background:radial-gradient(560px 280px at 26% 150px, rgba(76,178,98,.12), transparent 70%),
  radial-gradient(500px 280px at 80% 190px, rgba(90,120,230,.08), transparent 70%), var(--canvas)}}
:root[data-theme="dark"] body{background:radial-gradient(560px 280px at 26% 150px, rgba(76,178,98,.12), transparent 70%),
  radial-gradient(500px 280px at 80% 190px, rgba(90,120,230,.08), transparent 70%), var(--canvas)}
/* drop moment in the fresh green */
body.dragging .drop.compact .dzmark{box-shadow:inset 0 0 0 30px #3fa95c, inset 0 1px 0 rgba(255,255,255,.45), 0 10px 26px -10px rgba(40,150,80,.55), 0 0 0 6px rgba(76,178,98,.12);color:#fff}
body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{border-color:rgba(90,205,125,.45)}
body.dragging .drop.compact{border-color:rgba(76,178,98,.4)}
body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#1d6a33 10%,#1c8a6c 90%)}
body.dragging .drop.compact::before{background:
    radial-gradient(42% 52% at 28% 42%, rgba(70,195,110,.62), transparent 70%),
    radial-gradient(36% 46% at 72% 58%, rgba(30,175,150,.48), transparent 70%),
    radial-gradient(30% 40% at 52% 22%, rgba(170,230,130,.42), transparent 70%),
    radial-gradient(44% 44% at 18% 85%, rgba(70,140,235,.22), transparent 70%),
    radial-gradient(30% 34% at 88% 18%, rgba(120,220,190,.30), transparent 70%);background-size:140% 140%}
.dz-glow i{background:conic-gradient(from var(--spin), rgba(76,190,110,0) 0deg, rgba(110,225,145,1) 70deg, rgba(76,190,110,0) 150deg,
    rgba(76,190,110,0) 210deg, rgba(140,230,200,.9) 280deg, rgba(76,190,110,0) 340deg)}
.drop.compact::after{background:conic-gradient(from var(--spin), rgba(76,190,110,0) 0deg, rgba(110,220,140,.95) 70deg, rgba(76,190,110,0) 150deg,
    rgba(76,190,110,0) 210deg, rgba(150,230,170,.8) 280deg, rgba(76,190,110,0) 340deg)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact .dzmark{color:#fff}
  :root:not([data-theme="light"]) body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#b9f2c6 10%,#93e4d3 90%)}
  :root:not([data-theme="light"]) body.dragging .drop.compact{border-color:rgba(140,230,160,.2)}}
:root[data-theme="dark"] body.dragging .drop.compact .dzmark{color:#fff}
:root[data-theme="dark"] body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#b9f2c6 10%,#93e4d3 90%)}
:root[data-theme="dark"] body.dragging .drop.compact{border-color:rgba(140,230,160,.2)}

/* ---------- 3.12.3: header scrolls away; wordmark in Geist (SIL OFL, docs/Geist-OFL.txt) ---------- */

.top,.top.scrolled{position:relative;top:auto;background:transparent;-webkit-backdrop-filter:none;backdrop-filter:none;box-shadow:none}
.wm{font:600 19px/1 "Geist",-apple-system,BlinkMacSystemFont,sans-serif;letter-spacing:-.045em;color:var(--ink)}
.brand{gap:9px}

/* ======================================================================
   3.13 — one typeface, one deep green, no decoration, a clear hierarchy
   ====================================================================== */
@font-face{font-family:"Geist";font-style:normal;font-weight:400;font-display:swap;src:url(data:font/woff2;base64,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) format("woff2")}
@font-face{font-family:"Geist";font-style:normal;font-weight:500;font-display:swap;src:url(data:font/woff2;base64,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) format("woff2")}
@font-face{font-family:"Geist";font-style:normal;font-weight:600;font-display:swap;src:url(data:font/woff2;base64,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) format("woff2")}
:root{--ui:"Geist",-apple-system,BlinkMacSystemFont,"Helvetica Neue",sans-serif}
body,button,input,select,textarea{font-family:var(--ui)}
body{font-feature-settings:"tnum" 1;letter-spacing:-.006em}
/* numbers, times and paths: the same face, tabular — no second mono font */
.row.lib .when,.row.lib .meta span,.d-body .ts,.turn .tt,.pbar .pt,.ck-m,.row.qrow .pct,.thit .tt2,.d-meta{font-family:var(--ui);letter-spacing:0;font-variant-numeric:tabular-nums}
.row.lib .when{font-size:12.5px}
.row.lib .meta{font-size:12px}
.d-meta{font-size:12.5px}
/* titles: Geist everywhere; serif only for the recording's title in the reader */
.wm{font-family:var(--ui)}
.drop.compact .dzt b,body.dragging .drop.compact .dzt b{font-family:var(--ui);font-weight:600;font-size:24px;letter-spacing:-.028em;line-height:1.2}
h2,#lTitle{font-family:var(--ui);font-weight:600;font-size:15px;letter-spacing:-.01em}
#lTitle .count{font-size:15px;font-weight:500}
dialog h3,.d-top h3{font-family:var(--ui);font-weight:600;letter-spacing:-.02em}
#rName{font-family:var(--serif);font-weight:500;font-size:30px;letter-spacing:-.022em}
.d-head h3:not(#rName){font-family:var(--ui);font-weight:600;font-size:22px;letter-spacing:-.02em}
.stack{margin-top:36px}
.ph{margin-bottom:4px}

/* colour: one deep green, used sparingly; no glows */
:root{--brand:#1f4d35;--brand-ink:#1f4d35;--brand-tint:rgba(31,77,53,.08);--green:#2f7a4c;--green-ink:#1f4d35;
  --ring-c:rgba(31,77,53,.30)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--brand:#2c6b49;--brand-ink:#9fd3ae;--brand-tint:rgba(120,200,150,.10);
  --green:#5fae7c;--green-ink:#9fd3ae;--ring-c:rgba(159,211,174,.38)}}
:root[data-theme="dark"]{--brand:#2c6b49;--brand-ink:#9fd3ae;--brand-tint:rgba(120,200,150,.10);--green:#5fae7c;--green-ink:#9fd3ae;--ring-c:rgba(159,211,174,.38)}
.btn.go,.btn.go:hover{background:var(--brand);color:#fff;box-shadow:none;filter:none}
.btn.go:hover{background:color-mix(in srgb, var(--brand) 88%, #000)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .btn.go:hover{background:color-mix(in srgb, var(--brand) 88%, #fff)}}
:root[data-theme="dark"] .btn.go:hover{background:color-mix(in srgb, var(--brand) 88%, #fff)}
#vAI[aria-pressed="true"]{background:var(--brand);color:#fff}
::selection{background:rgba(31,77,53,.16)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .btn.go{color:#fff;box-shadow:none}}
:root[data-theme="dark"] .btn.go{color:#fff;box-shadow:none}

/* no decoration: flat paper, a card with one thin line, an icon without a tile */
body,:root[data-theme="dark"] body{background:var(--canvas)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body{background:var(--canvas)}}
.drop.compact{background:var(--cream);-webkit-backdrop-filter:none;backdrop-filter:none;border:1px solid var(--hair);box-shadow:none}
.drop.compact:hover{border-color:color-mix(in srgb, var(--ink) 18%, transparent)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) .drop.compact{background:var(--cream);border-color:var(--hair);box-shadow:none}}
:root[data-theme="dark"] .drop.compact{background:var(--cream);border-color:var(--hair);box-shadow:none}
.dzmark{width:auto;height:auto;background:none;border-radius:0;color:var(--brand-ink);margin-bottom:12px}
.dzmark svg{width:26px;height:26px}
.drop.compact .dzmark{width:52px;height:52px;display:grid;place-items:center;margin-bottom:6px}
.drop.compact .opts .opt{color:var(--soft)}
.drop.compact .opts{border-left-color:var(--hair)}
.ckbtn{background:transparent;border:1px solid var(--hair)}
.ckbtn:hover{background:var(--fill)}

/* drop moment: same motion, deep green, no muddy glass */
body.dragging .drop.compact{background:var(--cream);border-color:color-mix(in srgb, var(--brand) 40%, transparent);box-shadow:none;background-image:none}
body.dragging .drop.compact .dzmark{width:52px;height:52px;border-radius:16px;margin-bottom:6px;background:none;color:#fff;
  box-shadow:inset 0 0 0 30px var(--brand), 0 10px 24px -12px rgba(20,60,40,.5)}
body.dragging .drop.compact .dzmark svg{width:24px;height:24px}
body.dragging .drop.compact .dzmark::before,body.dragging .drop.compact .dzmark::after{border-color:color-mix(in srgb, var(--brand) 45%, transparent)}
body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#173d29 10%,#2f7a4c 90%)}
body.dragging .drop.compact::before{background:
    radial-gradient(42% 52% at 28% 42%, rgba(60,150,95,.30), transparent 70%),
    radial-gradient(36% 46% at 72% 58%, rgba(31,110,80,.22), transparent 70%),
    radial-gradient(30% 40% at 52% 22%, rgba(150,210,140,.26), transparent 70%),
    radial-gradient(44% 44% at 18% 85%, rgba(31,77,53,.16), transparent 70%);background-size:140% 140%}
.dz-glow i{background:conic-gradient(from var(--spin), rgba(31,77,53,0) 0deg, rgba(60,150,95,.8) 70deg, rgba(31,77,53,0) 150deg,
    rgba(31,77,53,0) 210deg, rgba(110,190,140,.7) 280deg, rgba(31,77,53,0) 340deg)}
.drop.compact::after{background:conic-gradient(from var(--spin), rgba(31,77,53,0) 0deg, rgba(60,150,95,.9) 70deg, rgba(31,77,53,0) 150deg,
    rgba(31,77,53,0) 210deg, rgba(110,190,140,.75) 280deg, rgba(31,77,53,0) 340deg)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#cdebd6 10%,#9fd3ae 90%)}}
:root[data-theme="dark"] body.dragging .drop.compact .dzt b{background-image:linear-gradient(95deg,#cdebd6 10%,#9fd3ae 90%)}

/* ---------- 3.14.1: finishing the 3.13 clean-up everywhere ---------- */
/* dark drag: flat card like light (older dark glass rules were more specific) */
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body.dragging .drop.compact{background:var(--cream);background-image:none;box-shadow:none;
  border-color:color-mix(in srgb, var(--brand) 55%, transparent)}}
:root[data-theme="dark"] body.dragging .drop.compact{background:var(--cream);background-image:none;box-shadow:none;border-color:color-mix(in srgb, var(--brand) 55%, transparent)}
/* one typeface: no stray monospace or hard-coded system font */
kbd,.rec .clock,.turn .nm,.turn .av,.turn input{font-family:var(--ui)}
kbd{font-size:11px;font-weight:500}
/* visible selection and focus in both themes */
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) ::selection{background:rgba(159,211,174,.28)}}
:root[data-theme="dark"] ::selection{background:rgba(159,211,174,.28)}
:root{--ring-c:rgba(31,77,53,.55)}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--ring-c:rgba(159,211,174,.5)}}
:root[data-theme="dark"]{--ring-c:rgba(159,211,174,.5)}
#rName input.rn{box-shadow:0 0 0 3px var(--ring-c)}
/* the header no longer sticks, so nothing needs to scroll under it */
html{scroll-padding-top:0}
.drop,.panel,.banner{scroll-margin-top:16px}

/* ---------- 3.15: title ideas + the setting for them ---------- */
.tideas{margin:4px 0 18px;border:1px solid var(--hair);border-radius:12px;overflow:hidden}
.tideas .tih{font-size:12px;font-weight:600;color:var(--soft);padding:10px 14px 6px}
.tidea{display:flex;align-items:center;justify-content:space-between;gap:12px;width:100%;border:0;border-top:1px solid var(--hair);background:none;
  padding:10px 14px;font:500 14px/1.35 var(--ui);color:var(--ink);text-align:left;cursor:pointer}
.tidea span{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tidea em{font-style:normal;font-size:12px;font-weight:600;color:var(--soft);flex:none}
button.tidea:hover{background:var(--fill)}
button.tidea:hover em{color:var(--brand-ink)}
.tidea.on{cursor:default}
.tidea.on em{color:var(--brand-ink)}
.tidea.on em::before{content:"✓ "}
#titleRow select{max-width:240px}
#titleRow.off{opacity:.45}
#titleRow.off select{pointer-events:none}

/* title format builder */
.tfrow{display:block}
.tf{display:grid;gap:10px}
.tfh{display:flex;flex-direction:column;gap:2px}
.tfh b{font-weight:600}
.tfh span{font-size:12.5px;color:var(--soft)}
.tfparts{list-style:none;margin:0;padding:0;display:grid;gap:4px}
.tfparts li{display:flex;align-items:center;gap:10px;padding:7px 8px 7px 10px;border-radius:9px;background:var(--cream);border:1px solid var(--hair);
  font-size:13.5px;color:var(--soft);cursor:grab;user-select:none}
.tfparts li.on{color:var(--ink)}
.tfparts li.drag{opacity:.5}
.tfparts .grip{color:var(--soft);opacity:.6;display:grid}
.tfparts label{flex:1;display:flex;align-items:center;gap:8px;cursor:pointer}
.tfparts input{width:15px;height:15px;margin:0;accent-color:var(--brand)}
.tfparts .mv{display:flex;gap:2px}
.tfparts .mv button{width:24px;height:24px;border:0;border-radius:6px;background:transparent;color:var(--soft);cursor:pointer;font:600 12px/1 var(--ui)}
.tfparts .mv button:hover:not(:disabled){background:var(--fill);color:var(--ink)}
.tfparts .mv button:disabled{opacity:.3;cursor:default}
.tfopts{display:flex;gap:14px;flex-wrap:wrap;font-size:13px;color:var(--soft)}
.tfopts label{display:flex;align-items:center;gap:8px}
.tfopts select{min-height:30px;padding:4px 10px;font-size:13px}
.tfprev{display:flex;gap:10px;align-items:baseline;font-size:12.5px;color:var(--soft);padding:8px 10px;border-radius:9px;background:var(--brand-tint)}
.tfprev b{color:var(--ink);font-weight:600;font-size:13.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
#tfRow.off{opacity:.45}
#tfRow.off li{cursor:default}


/* notes + watch folders */
.gact{display:flex;align-items:center;gap:12px;flex:none}
.recapps{justify-content:center;margin:14px auto 0;text-align:left;max-width:340px}
.recapps span{display:flex;flex-direction:column;line-height:1.3}
.recapps b{font-weight:500;font-size:13.5px}
.recapps em{font-style:normal;font-size:12px;color:var(--soft)}
.recapps.on em{color:var(--brand)}
.wlist{list-style:none;margin:0;padding:0 14px 6px}
.wlist li{display:flex;align-items:center;gap:12px;padding:10px 0;border-top:1px solid var(--hair);font-size:13px}
.wlist .wp{flex:1;min-width:0;display:flex;flex-direction:column}
.wlist .wp b{font-weight:500;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.wlist .wp span{font-size:12px;color:var(--soft)}
.wlist .wp span.off{color:var(--fail)}
.wlist .wp span::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--brand);margin-right:6px;vertical-align:1px}
.wlist .wp span.off::before{background:var(--fail)}
.wlist .wp span.wait::before{background:var(--soft)}

/* 3.19 design pass */
/* 3. one rule for the settings column: menus ▾, things that open a panel › */
.drop.compact .opts #hintChip b,.drop.compact .opts #aiChip b{padding-right:18px;background:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='6' height='10'%3E%3Cpath d='M1 1l4 4-4 4' fill='none' stroke='%236e6e73' stroke-width='1.6' stroke-linecap='round' stroke-linejoin='round'/%3E%3C/svg%3E") no-repeat right 5px center}
.drop.compact .opts .opt b,.drop.compact .opts .opt select{text-align:right}
/* 5. previous / next read as buttons */
.navs .x{background:var(--fill);color:var(--ink)}
.navs .x:hover:not(:disabled){background:var(--row)}
.navs .x:disabled{opacity:.3}
/* 6. settings sections */
.settabs{display:flex;margin:2px 24px 14px;align-self:flex-start;width:max-content}
#setDlg .glist.off-pane{display:none}
/* 8. the drop mark gets a soft disc so the empty card has a centre */
body:not(.dragging) .drop.compact .dzmark{width:56px;height:56px;border-radius:50%;background:color-mix(in srgb, var(--brand) 9%, transparent);margin-bottom:10px}
body:not(.dragging) .drop.compact .dzmark svg{width:24px;height:24px}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]) body:not(.dragging) .drop.compact .dzmark,:root:not([data-theme="light"]) .start i{background:color-mix(in srgb, var(--brand-ink) 14%, transparent)}}
:root[data-theme="dark"] body:not(.dragging) .drop.compact .dzmark,:root[data-theme="dark"] .start i{background:color-mix(in srgb, var(--brand-ink) 14%, transparent)}
/* 9. list rhythm: no dividers, a soft highlight on hover */
.row.lib{border-top:0;border-radius:12px}
.row.lib + .row.lib{margin-top:2px}
/* 10. first run */
.starts{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0 6px}
.start{display:flex;flex-direction:column;align-items:flex-start;gap:6px;text-align:left;padding:18px;border:1px solid var(--hair);border-radius:14px;background:transparent;color:var(--ink);font:inherit;cursor:pointer;transition:background .15s,border-color .15s}
.start:hover,.start:focus-visible{background:var(--fill);border-color:transparent}
.start i{width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:color-mix(in srgb, var(--brand) 9%, transparent);color:var(--brand-ink);margin-bottom:6px}
.start b{font-weight:600;font-size:14px}
.start span{font-size:12.5px;color:var(--soft);line-height:1.4}
@media (max-width:720px){.starts{grid-template-columns:1fr}}
/* wordmark: Montserrat SemiBold (SIL OFL 1.1, docs/Montserrat-OFL.txt), only the letters of "Matita" */
@font-face{font-family:"Matita Logo";font-weight:600;font-display:block;src:url(data:font/woff2;base64,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) format("woff2")}
.wm,.brand .wm{font:600 19px/1 "Matita Logo","Montserrat",-apple-system,BlinkMacSystemFont,sans-serif;letter-spacing:-.01em;color:var(--ink)}
.brand svg{filter:none;box-shadow:none}            /* flat logo, no shadow */
/* toast lives in the top layer (popover) so it shows above dialogs */
.toast[popover]{inset:auto auto 24px 50%;margin:0;border:1px solid rgba(255,255,255,.08);overflow:visible}
.toast[popover]:not(:popover-open){display:none}
@media (prefers-reduced-motion:reduce){.skel i{animation:none}}
</style>
</head>
<body>
<div class="wrap">

  <header class="top">
    <div class="brand">
      <svg viewBox="0 0 100 100" aria-hidden="true"><rect width="100" height="100" rx="23" fill="#1f4d35"/><path d="M27 70V31L50 56L73 31V70" fill="none" stroke="#a6e8ae" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/></svg>
      <span class="wm">Matita</span>
    </div>
    <div class="topr">
      <button class="tb" id="aiOpen" title="AI assist" hidden>
        <svg class="glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 3l1.8 4.6L18.5 9.4l-4.7 1.8L12 16l-1.8-4.8L5.5 9.4l4.7-1.8z"/><path d="M19 15l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z"/></svg>
        <span class="aidot" aria-hidden="true"></span><span class="vh" id="aiState">AI off</span></button>
      <button class="tb" id="setOpen" title="Settings" aria-label="Settings">
        <svg class="glyph" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 1 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 1 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 1 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 1 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/></svg></button>
    </div>
  </header>

  <div class="banner" id="banner" hidden>
    <span id="bannerText"></span>
    <span style="display:flex;gap:10px;align-items:center">
      <button class="link" id="bannerLater">Later</button>
      <button class="btn go" id="bannerGo">Update now</button>
    </span>
  </div>

  <section class="panel setup" id="setupCard" hidden aria-labelledby="setupTitle">
    <div class="ph"><div><h2 id="setupTitle">Finish setting up</h2></div></div>
    <div class="steps">
      <div class="step" id="stepTools"><div class="num">1</div><div id="stepToolsBody"></div><div id="stepToolsAct"></div></div>
      <div class="step" id="stepModel"><div class="num">2</div><div id="stepModelBody"></div><div id="stepModelAct"></div></div>
    </div>
  </section>

  <section class="drop compact" id="hero" aria-labelledby="heroTitle">
    <div class="dzl">
    <h1 id="heroTitle">Drop recordings here</h1>
    <div class="dz">
      <span class="dzmark" aria-hidden="true"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v10M7.5 9.5L12 14l4.5-4.5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg></span>
      <div class="dzt"><b>Drop audio or video here</b>
        <button class="optsum" id="optSum" aria-expanded="false" aria-controls="opts" title="Transcription settings"><span id="optSumText">English · Turbo</span><svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 4.5l3 3 3-3"/></svg></button></div>
    </div>
    <div class="cta">
      <div class="grp" role="group" aria-label="Add recordings you already have">
        <button class="btn go" id="choose">
          <svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M12 5v14M5 12h14"/></svg>
          Add files</button>
        <button class="btn soft" id="vmOpen">
          <svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="2" width="12" height="20" rx="3"/><path d="M10.5 4.8h3"/><path d="M9.5 12v2M12 10v6M14.5 11.2v3.6"/></svg>
          <span><span class="long">From iPhone </span>Voice Memos</span></button>
      </div>
      <span class="sep" aria-hidden="true"></span>
      <button class="btn rec" id="recOpen"><span class="recdot" aria-hidden="true"></span><span>Record<span class="long"> now</span></span></button>
    </div>
    <input type="file" id="file" multiple hidden accept="audio/*,video/*,.m4a,.mp3,.wav,.aac,.flac,.ogg,.opus,.mp4,.mov,.m4v,.qta,.caf,.aif,.aiff,.webm">
    </div>
    <span class="dz-glow" aria-hidden="true"><i></i></span>
    <div class="opts" id="opts">
    <div class="chiprow" aria-label="Transcription settings">
      <label class="opt"><span>Language</span>
        <select id="lang">
          <option value="en">English</option><option value="auto">Detect automatically</option>
          <option value="ko">Korean</option><option value="fr">French</option><option value="es">Spanish</option>
          <option value="de">German</option><option value="zh">Chinese</option><option value="ja">Japanese</option>
        </select></label>
      <label class="opt"><span>Model</span>
        <select id="quality"><option value="balanced" selected>Turbo · Recommended</option></select></label>
      <label class="opt"><span>Speakers</span>
        <select id="speakers">
          <option value="off">Off</option>
          <option value="auto">Auto</option>
          <option value="2">2 people</option><option value="3">3 people</option><option value="4">4 people</option>
          <option value="5">5 people</option><option value="6">6 people</option>
        </select></label>
      <button class="opt" id="hintChip" aria-expanded="false" aria-controls="hintPop"><span>Hints</span><b id="hintVal">None</b></button>
      <button class="opt aiopt-row" id="aiChip"><span>AI assist</span><b id="aiChipVal">Off</b></button>
    </div>
    <div class="hintpop" id="hintPop" hidden>
      <label for="prompt">Words that are easy to mishear</label>
      <input type="text" id="prompt" placeholder="e.g. people's names, places, company or product names, jargon" autocomplete="off">
      <p class="hint">Write them the way they should be spelled, separated by commas. For example: <i>Prof. Chandon, Fontainebleau, Roland Berger, EBITDA</i>. Leave it empty if nothing special comes up.</p>
    </div>
    <div class="spk-note" id="spkNote"></div>
    </div>
  </section>

  <div class="stack">
    <section class="panel" aria-labelledby="lTitle">
      <div class="ph">
        <div><h2 id="lTitle">Transcripts <span class="count" id="lSub"></span></h2></div>
        <div class="tools">
          <input type="search" id="filter" placeholder="Search all transcripts" aria-label="Search transcript names and text">
          <button class="ckbtn" id="ckOpen" aria-label="Search and commands" title="Search and commands"><svg class="glyph" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg><span>Search transcripts</span><kbd>⌘K</kbd></button>
          <div class="mwrap">
            <button class="tb more-btn" id="listMore" aria-label="More" aria-haspopup="menu" aria-expanded="false"><svg class="glyph" width="18" height="18" viewBox="0 0 24 24" fill="currentColor"><circle cx="5.5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="18.5" cy="12" r="1.7"/></svg></button>
            <div class="menu" id="listMenu" role="menu" hidden>
              <button role="menuitem" id="zipAll" disabled><svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v11M7 10l5 5 5-5"/><path d="M5 19h14"/></svg>Download all</button>
              <button role="menuitem" id="clearList" disabled><svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/></svg>Clear list…</button>
            </div>
          </div>
        </div>
      </div>
      <div class="qblock" id="queuePanel" hidden>
        <div class="qh"><span class="qt" id="qTitle">In progress</span><span class="qsub" id="qSub"></span>
          <span class="qtools"><button class="link muted" id="pauseAll">Pause</button><button class="link muted" id="cancelAll">Cancel all</button></span></div>
        <div class="rows" id="queue"></div>
      </div>
      <div class="thits" id="textHits" hidden></div>
      <div class="rows" id="library"><div class="skel" aria-hidden="true"><i></i><i></i><i></i></div></div>
    </section>
  </div>
</div>

<p class="foot-note" id="footNote"></p>
<div class="overlay" id="overlay" aria-hidden="true"><div class="ov-card"><svg width="40" height="40" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 15V3M7 8l5-5 5 5"/><path d="M4 15v3a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-3"/></svg><div class="ov-title">Drop to transcribe</div><p>Audio and video files, or whole folders.</p></div></div>

<dialog id="reader" aria-labelledby="rName">
  <div class="dlg">
    <div class="d-head">
      <div class="d-top">
        <div class="d-title"><h3 id="rName" title="Click to rename"></h3><div class="d-meta" id="rMeta"></div><div class="d-sum" id="rSum"></div><div class="chips" id="rKeys"></div></div>
        <div class="navs"><button class="x" id="rPrev" aria-label="Previous transcript" title="Previous (⌘[)"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 5l-7 7 7 7"/></svg></button><button class="x" id="rNext" aria-label="Next transcript" title="Next (⌘])"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M9 5l7 7-7 7"/></svg></button></div>
        <button class="x" id="rClose" aria-label="Close"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M2 2l10 10M12 2L2 12"/></svg></button>
      </div>
      <div class="d-bar">
        <div class="seg" role="group" aria-label="View">
          <button id="vAI" aria-pressed="false" hidden>AI summary</button>
          <button id="vTime" aria-pressed="true">Transcript</button>
          <button id="vSpk" aria-pressed="false" hidden>Speakers</button>
        </div>
        <input type="search" id="rFind" placeholder="Find in transcript" aria-label="Find in transcript">
        <span class="hits" id="rHits"></span>
      </div>
    </div>
    <div class="aibox" id="aiBox" hidden></div>
    <div class="d-body" id="rBody"></div>
    <div class="pbar" id="rPlayer" hidden>
      <button class="pp" id="pPlay" aria-label="Play"></button>
      <span class="pt" id="pTime">0:00</span>
      <div class="wave" id="pWave" role="slider" aria-label="Position in the recording" aria-valuemin="0" aria-valuemax="0" aria-valuenow="0" tabindex="0"><canvas id="pCanvas"></canvas></div>
      <span class="pt" id="pDur">0:00</span>
      <button class="rate" id="pRate" title="Playback speed">1×</button>
      <span id="pHint" hidden></span>
    </div>
    <div class="d-foot">
      <div class="acts">
        <button id="rReveal">Show in Finder</button>
        <button id="aiSum" hidden>AI summary</button>
        <button id="aiFix" hidden>Suggest fixes</button>
      </div>
      <div style="display:flex;gap:10px">
        <button class="btn" id="rSave">Save as…</button>
        <button class="btn go" id="rCopy">Copy text</button>
      </div>
    </div>
  </div>
</dialog>

<dialog id="saveDlg" class="small" aria-labelledby="sTitle">
  <div class="dlg">
    <div class="d-head">
      <div class="d-top"><div><h3 id="sTitle">Save location</h3></div>
        <button class="x" id="sClose" aria-label="Close"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M2 2l10 10M12 2L2 12"/></svg></button></div>
    </div>
    <div class="d-body plain">
      <div class="pathbox">
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg>
        <span id="sPath">~/Documents/Transcripts</span>
      </div>
      <div class="btns">
        <button class="btn go" id="sChoose">Choose folder</button>
        <button class="btn" id="sOpen">Open in Finder</button>
      </div>
      <div style="margin-top:12px"><button class="link muted" id="sReset" hidden>Go back to Documents/Transcripts</button></div>
      <div class="paste">
        <label class="lbl" for="sInput">Or paste a folder path</label>
        <div class="inline"><input type="text" id="sInput" placeholder="~/Dropbox/Lecture notes"><button class="btn dark" id="sSave">Use this folder</button></div>
        <div class="formerr" id="sErr" role="alert"></div>
      </div>
    </div>
  </div>
</dialog>

<dialog id="vmDlg" class="wide" aria-labelledby="vmTitle">
  <div class="dlg">
    <div class="d-head">
      <div class="d-top"><div><h3 id="vmTitle">Voice Memos</h3><div class="d-meta" id="vmMeta"></div></div>
        <button class="x" id="vmClose" aria-label="Close"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M2 2l10 10M12 2L2 12"/></svg></button></div>
      <div class="vm-tools" id="vmTools" hidden>
        <input type="search" id="vmFind" placeholder="Search titles" aria-label="Search Voice Memos">
        <button class="link muted" id="vmPickNew">Select all new</button>
        <button class="link muted" id="vmPickNone">Clear</button>
      </div>
    </div>
    <div class="d-body" id="vmBody"></div>
    <div class="d-foot" id="vmFoot" hidden>
      <label class="switch"><input type="checkbox" id="vmAuto"> Transcribe new memos automatically</label>
      <button class="btn go" id="vmGo" disabled>Transcribe</button>
    </div>
  </div>
</dialog>

<dialog id="fmtDlg" class="small" aria-labelledby="fTitle">
  <div class="dlg">
    <div class="d-head"><div class="d-top"><div><h3 id="fTitle">Save as</h3><div class="d-meta" id="fName"></div></div>
      <button class="x" id="fClose" aria-label="Close"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M2 2l10 10M12 2L2 12"/></svg></button></div></div>
    <div class="d-body plain">
      <div class="fmts">
        <button class="fmt" data-fmt="docx"><div><b>Word document</b></div><em>.docx</em></button>
        <button class="fmt" data-fmt="md"><div><b>Markdown</b></div><em>.md</em></button>
        <button class="fmt" data-fmt="txt"><div><b>Plain text</b></div><em>.txt</em></button>
        <button class="fmt" data-fmt="srt" id="fSrt"><div><b>Subtitles</b></div><em>.srt</em></button>
      </div>
    </div>
  </div>
</dialog>

<dialog id="setDlg" class="small" aria-labelledby="setTitle">
  <div class="dlg">
    <div class="d-head"><div class="d-top"><div><h3 id="setTitle">Settings</h3></div>
      <button class="x" id="setClose" aria-label="Close"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M2 2l10 10M12 2L2 12"/></svg></button></div></div>
    <div class="settabs seg" role="tablist" aria-label="Settings sections" id="setTabs">
      <button role="tab" data-pane="general" aria-selected="true">General</button><button role="tab" data-pane="library" aria-selected="false">Library</button><button role="tab" data-pane="ai" aria-selected="false">AI</button>
    </div>
    <div class="d-body plain">
      <div class="glist" data-pane="general">
        <div class="grow">
          <div class="gl"><b>Appearance</b></div>
          <div class="seg" role="group" aria-label="Appearance" id="themeSeg">
            <button data-theme-set="system">System</button><button data-theme-set="light">Light</button><button data-theme-set="dark">Dark</button>
          </div>
        </div>
      </div>
      <div class="glist" data-pane="library">
        <div class="grow">
          <div class="gl"><b>Save transcripts in</b><span id="whereText">~/Documents/Transcripts</span></div>
          <button class="btn soft sm" id="whereChange">Change…</button>
        </div>
        <div class="gnote warnline" id="whereWarn" hidden></div>
      </div>
      <div class="glist" id="folderList" data-pane="library">
        <div class="grow">
          <div class="gl"><b>Markdown notes</b><span id="notesText">Off</span></div>
          <div class="gact"><button class="btn soft sm" id="notesDir" hidden>Folder…</button>
            <label class="switch"><input type="checkbox" id="notesOn" aria-label="Markdown notes"></label></div>
        </div>
        <div class="gnote">A note for each recording with its summary, tags and [[links]] to related recordings. Point it at a folder in your Obsidian vault.</div>
        <div class="grow">
          <div class="gl"><b>Watch folders</b><span>New recordings there are transcribed automatically</span></div>
          <button class="btn soft sm" id="watchAdd">Add folder…</button>
        </div>
        <ul class="wlist" id="watchList" hidden></ul>
      </div>
      <div class="glist" data-pane="library" id="hiddenList" hidden>
        <div class="grow" id="hiddenRow">
          <div class="gl"><b>Removed from the list</b><span id="hiddenText"></span></div>
          <button class="btn soft sm" id="unhide">Show again</button>
        </div>
      </div>
      <div class="glist" data-pane="general">
        <div class="grow">
          <div class="gl"><b>When the lid is closed</b></div>
          <select id="lid" class="gsel">
            <option value="sleep">Pause until I open it</option>
            <option value="plugged">Keep going when plugged in</option>
            <option value="always">Keep going, even on battery</option>
          </select>
        </div>
        <div class="gnote" id="lidNote"></div>
      </div>
      <div class="glist" data-pane="ai">
        <div class="grow">
          <div class="gl"><b>AI assist</b><span id="aiState2">Off</span></div>
          <button class="btn soft sm" id="aiOpen2">Change…</button>
        </div>
        <div class="grow" id="titleRow">
          <div class="gl"><b>Recording titles</b><span id="titleNote">AI suggests titles with each summary</span></div>
          <select id="titleMode" aria-label="Recording titles">
            <option value="choose">Suggest, I'll choose</option>
            <option value="auto">Rename untitled automatically</option>
            <option value="off">Don't suggest titles</option>
          </select>
        </div>
        <div class="grow tfrow" id="tfRow">
          <div class="tf">
            <div class="tfh"><b>Title format</b><span>Tick what to include, drag to change the order.</span></div>
            <ol class="tfparts" id="tfParts"></ol>
            <div class="tfopts">
              <label>Date <select id="tfDate"><option value="iso">2025-10-31</option><option value="dmy">31 Oct 2025</option><option value="compact">251031</option></select></label>
              <label>Separator <select id="tfSep"><option value=" ">Space</option><option value=" · ">·</option><option value=" – ">–</option><option value="_">_</option></select></label>
            </div>
            <div class="tfprev"><span>Preview</span><b id="tfPreview"></b></div>
          </div>
        </div>
      </div>
    </div>
  </div>
</dialog>

<dialog id="aiDlg" class="wide" aria-labelledby="aiTitle">
  <div class="dlg">
    <div class="d-head"><div class="d-top"><div><h3 id="aiTitle">AI assist</h3>
      <div class="d-meta">Optional. Adds an AI summary and suggested fixes for misheard words to each transcript. You choose when to run it, and fixes are only applied when you accept them.</div></div>
      <button class="x" id="aiClose" aria-label="Close"><svg width="14" height="14" viewBox="0 0 14 14" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"><path d="M2 2l10 10M12 2L2 12"/></svg></button></div></div>
    <div class="d-body plain">
      <div class="aiopts" role="group" aria-label="AI provider">
        <button class="aiopt" data-prov="off"><b>Off</b><span>Transcripts only. Nothing else is installed or sent.</span></button>
        <button class="aiopt" data-prov="ollama"><b>On this Mac</b><span>Ollama runs a model on your Mac. Private, free, slower.</span></button>
        <button class="aiopt" data-prov="claude"><b>Claude API</b><span>Anthropic's models, with your API key. Best quality, fast, paid per use.</span></button>
      </div>
      <div class="aidet" id="aiDet"></div>
    </div>
  </div>
</dialog>

<dialog id="recDlg" class="small" aria-labelledby="recTitle">
  <div class="dlg">
    <div class="d-head"><div class="d-top"><div><h3 id="recTitle">Recording</h3><div class="d-meta">Uses the language, model and speaker settings on the main page.</div></div></div></div>
    <div class="d-body plain">
      <div class="rec">
        <div class="state" id="recState"><i></i><span>Recording</span></div>
        <div class="clock" id="recClock">0:00</div>
        <div class="meter" id="recMeter" aria-hidden="true"></div>
        <div class="sub">Keep this window open until you stop. Your Mac won't sleep while recording.</div>
        <label class="switch recapps" id="recAppsRow" hidden><input type="checkbox" id="recApps">
          <span><b>Include sound from other apps</b><em id="recAppsNote">For calls: records the other people too</em></span></label>
        <div class="btns">
          <button class="btn" id="recPause">Pause</button>
          <button class="btn go" id="recStop">Stop and transcribe</button>
        </div>
        <div class="btns" style="margin-top:10px"><button class="link muted" id="recDiscard">Discard recording</button></div>
      </div>
    </div>
  </div>
</dialog>

<div class="toast" id="toast" role="status" aria-live="polite" popover="manual"></div>
<dialog id="cmdk" class="cmdk" aria-label="Search and commands">
  <div class="ck-in"><svg class="glyph" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/></svg><input id="ckq" type="text" placeholder="Search transcripts or type a command" autocomplete="off" spellcheck="false" role="combobox" aria-controls="ckList" aria-expanded="true"><kbd>esc</kbd></div>
  <div class="ck-list" id="ckList" role="listbox"></div>
  <div class="ck-foot"><span><kbd>↑</kbd><kbd>↓</kbd> move</span><span><kbd>↵</kbd> open</span><span class="ck-sp"></span><span><kbd>⌘K</kbd> toggle</span></div>
</dialog>
<div class="menu rowmenu" id="rowMenu" role="menu" hidden>
  <button role="menuitem" data-rm="rename"><svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 20h4L19 9l-4-4L4 16z"/><path d="M13.5 6.5l4 4"/></svg>Rename…</button>
  <button role="menuitem" data-rm="remove"><svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12s3.5-6 9-6 9 6 9 6-3.5 6-9 6-9-6-9-6z"/><path d="M4 4l16 16"/></svg>Remove from list</button>
  <div class="msep"></div>
  <button role="menuitem" data-rm="trash" class="danger"><svg class="glyph" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/></svg>Move to Trash</button>
</div>

<script>
const $ = s => document.querySelector(s);
// Sheets fade out quickly when closed (Esc still closes at once).
(() => { const shut = HTMLDialogElement.prototype.close;
  HTMLDialogElement.prototype.close = function(v){
    if (!this.open || this.classList.contains("closing") || matchMedia("(prefers-reduced-motion: reduce)").matches) return shut.call(this, v);
    this.classList.add("closing");
    this._shutT = setTimeout(() => { this._shutT = null; this.classList.remove("closing"); shut.call(this, v); }, 130);
  }; })();
// Opening a sheet with the mouse shouldn't light up a focus ring on its close button.
(() => { const open = HTMLDialogElement.prototype.showModal;
  HTMLDialogElement.prototype.showModal = function(){
    if (this._shutT){ clearTimeout(this._shutT); this._shutT = null; this.classList.remove("closing"); }   // reopened mid-fade: stay open
    if (!this.open) open.call(this);
    const a = document.activeElement; if (a && a.classList && a.classList.contains("x")) a.blur(); }; })();
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const AUDIO = /\.(m4a|mp3|wav|aac|flac|ogg|opus|mp4|mov|m4v|qta|caf|aif|aiff|webm|wma|amr)$/i;
const post = (url, data) => fetch(url, {method:"POST", body: JSON.stringify(data || {}), headers: {"X-Transcriber": "1"}});
let jobs = [], info = null, uploading = {done:0,total:0}, timer = null, current = null, view = "time";

function store(k, v){ try{ localStorage.setItem("stt."+k, v); }catch(e){} }
function load(k){ try{ return localStorage.getItem("stt."+k); }catch(e){ return null; } }
let toastT;
function toast(msg, act){
  const t = $("#toast"); t.textContent = msg;
  t.classList.toggle("has-act", !!act);
  if (act){ const b = document.createElement("button"); b.className = "tact"; b.textContent = act.label;
    b.onclick = () => { clearTimeout(toastT); t.classList.remove("on"); try{ t.hidePopover(); }catch(e){} act.fn(); }; t.append(b); }
  const top = [...document.querySelectorAll("dialog[open]:not(.closing)")].pop();
  (top || document.body).append(t);                 // inside the open sheet, so its buttons aren't inert
  const anchor = top && top.id === "reader" && (!$("#rPlayer").hidden ? $("#rPlayer") : $("#reader .d-foot"));
  t.style.bottom = anchor ? Math.max(28, innerHeight - anchor.getBoundingClientRect().top + 12) + "px" : "";   // sit above the player bar
  try{ if (t.matches(":popover-open")) t.hidePopover(); t.showPopover(); }catch(e){}   // above any open dialog
  requestAnimationFrame(() => t.classList.add("on"));
  clearTimeout(toastT);
  toastT = setTimeout(() => { t.classList.remove("on"); setTimeout(() => { if (!t.classList.contains("on")) try{ t.hidePopover(); }catch(e){} }, 220); }, act ? 6000 : Math.min(3200, 1400 + msg.length * 25));
}
const plural = (n, w, pl) => `${n.toLocaleString()} ${n===1 ? w : (pl || w+"s")}`;
const bare = n => n.replace(/\.[^.]+$/, "");
function clock(sec){
  if (sec == null || !isFinite(sec)) return "";
  sec = Math.round(sec); const h=Math.floor(sec/3600), m=Math.floor(sec%3600/60), s=sec%60;
  return h ? `${h}:${String(m).padStart(2,"0")}:${String(s).padStart(2,"0")}` : `${m}:${String(s).padStart(2,"0")}`;
}
function lengthCell(sec){
  if (!sec) return ["–",""];
  if (sec < 60) return [Math.round(sec), "sec"];
  if (sec < 3600) return [Math.round(sec/60), "min"];
  return [(sec/3600).toFixed(1).replace(/\.0$/,""), "hr"];
}
function roughly(sec){
  if (!isFinite(sec) || sec <= 0) return "";
  if (sec < 60) return "under a minute left";
  const m = Math.round(sec/60);
  return m < 60 ? `about ${m} min left` : `about ${Math.floor(m/60)} h ${m%60} min left`;
}
function when(ts){
  // Mac-style dates: "Today 17:09", "Yesterday 09:30", "Tuesday", "28 Oct", "28 Oct 2025".
  const d = new Date(ts*1000), now = new Date();
  const loc = info && info.locale ? info.locale.split("@")[0].replace("_", "-") : undefined;
  const h12 = info && info.clock24 === true ? false : (info && info.clock24 === false ? true : undefined);
  const fmt = o => { try{ return d.toLocaleString(loc, o); }catch(e){ return d.toLocaleString(undefined, o); } };
  const time = fmt({hour:"numeric", minute:"2-digit", hour12: h12});
  const day = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 86400000);
  if (diff === 0) return "Today " + time;
  if (diff === 1) return "Yesterday " + time;
  if (diff > 1 && diff < 7) return fmt({weekday:"long"});
  return fmt(d.getFullYear() === now.getFullYear() ? {day:"numeric", month:"short"} : {day:"numeric", month:"short", year:"numeric"});
}
function dayDiff(ts){ const d = new Date(ts*1000), now = new Date(), day = x => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  return Math.round((day(now) - day(d)) / 86400000); }
function fmtAt(ts, o){ const d = new Date(ts*1000), loc = info && info.locale ? info.locale.split("@")[0].replace("_", "-") : undefined;
  try{ return d.toLocaleString(loc, o); }catch(e){ return d.toLocaleString(undefined, o); } }
const whenOf = j => j.recorded || j.finished || 0;      // when it was recorded (falls back to when it was transcribed)
function dayGroup(ts){
  const k = dayDiff(ts), d = new Date(ts*1000);
  if (k <= 0) return "Today"; if (k === 1) return "Yesterday"; if (k < 7) return fmtAt(ts, {weekday:"long"});
  return fmtAt(ts, d.getFullYear() === new Date().getFullYear() ? {month:"long"} : {month:"long", year:"numeric"});
}
function rowWhen(ts){
  const h12 = info && info.clock24 === true ? false : (info && info.clock24 === false ? true : undefined);
  return dayDiff(ts) < 7 ? fmtAt(ts, {hour:"numeric", minute:"2-digit", hour12: h12}) : fmtAt(ts, {day:"numeric", month:"short"});
}
const SI = {
  file:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 4v10M7.5 9.5L12 14l4.5-4.5"/><path d="M4 15v3a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-3"/></svg>',
  phone:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="2" width="12" height="20" rx="3"/><path d="M9.5 12v2M12 10v6M14.5 11.2v3.6"/></svg>',
  mic:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/></svg>'};
const MI = {
  len:'<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
  ppl:'<svg width="14" height="13" viewBox="0 0 26 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="9" cy="8" r="3.5"/><path d="M2.5 20c.6-3.6 3.2-5.5 6.5-5.5s5.9 1.9 6.5 5.5"/><circle cx="18.5" cy="9" r="2.8"/><path d="M17.5 14.6c3 .1 5.2 1.9 5.8 5"/></svg>',
  ai:'<svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M12 2l2.2 5.8L20 10l-5.8 2.2L12 18l-2.2-5.8L4 10l5.8-2.2z"/></svg>'
};
const I = {
  doc:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linejoin="round" stroke-linecap="round"><path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4M9 12h6M9 16h4"/></svg>',
  mic:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="3" width="6" height="11" rx="3"/><path d="M5.5 11a6.5 6.5 0 0 0 13 0M12 17.5V21"/></svg>',
  clock:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></svg>',
  warn:'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5v5.5M12 16.5v.2"/></svg>'
};

/* ---------- settings + save location ---------- */
let infoTimer = null, updating = false, infoBusy = false;
async function loadInfo(){
  if (infoBusy) return;
  infoBusy = true;
  try{ await loadInfoInner(); } finally { infoBusy = false; }
}
function setHTML(el, html){ if (el._html === html) return false; el._html = html; el.innerHTML = html; return true; }
async function loadInfoInner(){
  clearTimeout(infoTimer);
  try{ info = await (await fetch("/api/info")).json(); }catch(e){ infoTimer = setTimeout(loadInfo, 3000); return; }
  const sel = $("#quality"), prev = (sel.dataset.ready && sel.value) || load("quality") || info.default_quality;
  sel.dataset.ready = "1";
  if (document.activeElement !== sel)
    setHTML(sel, (info.modes||[]).map(m => `<option value="${esc(m.id)}" data-full="${esc(m.label)}" data-short="${esc(m.label.split(" · ")[0])}">${esc(m.label.split(" · ")[0])}</option>`).join(""));
  sel.value = (info.modes||[]).some(m => m.id === prev) ? prev : info.default_quality;
  modelHint();
  showFolder(info.out_dir, info.out_dir_default);
  const fb = info.out_dir_fallback;
  $("#whereWarn").hidden = !fb;
  if (fb) $("#whereWarn").textContent = `macOS didn't allow access to ${fb}, so transcripts are saved here instead. Choose a folder with Change, or allow access in System Settings → Privacy & Security.`;
  renderSetup(); renderSpeakers(); renderUpdate(); if (!aiInfo) loadAI();
  if (!lidBusy) $("#lid").value = info.lid || "sleep";
  fitSelects();
  lidNote();
  if (jobs.length === 0 || info.hidden) renderLibrary(Date.now()/1000);      // the empty list mentions hidden ones
  if (setHTML($("#footNote"), `Matita ${esc(info.version)} · <button class="link muted" id="chk">Check for updates</button>`)) $("#chk").onclick = async () => { const r = await (await post("/api/check-update")).json(); info.update = r.update;
    renderUpdate(true); };
  const busy = Object.values(info.tasks).some(t => t.state === "running") || updating || info.starting;
  const ready = info.whisper && info.ffmpeg && info.models.length;
  clearTimeout(infoTimer);
  infoTimer = setTimeout(loadInfo, busy ? 1000 : (ready ? 60000 : 4000));
}
function stepHtml(ok, title, desc, extra){
  return `<b>${title}</b><p>${desc}</p>${extra||""}`;
}
function renderSetup(){
  const tools = info.whisper && info.ffmpeg, model = info.models.length > 0, card = $("#setupCard");
  card.hidden = tools && model;
  $("#choose").disabled = !(tools && model); $("#vmOpen").disabled = !(tools && model); $("#recOpen").disabled = !(tools && model);
  if (card.hidden) return;
  const tt = info.tasks.tools, tm = info.tasks.model;
  $("#stepTools").classList.toggle("ok", tools);
  $("#stepModel").classList.toggle("ok", model);
  let body, act = "";
  if (tools){ body = stepHtml(1, "whisper.cpp and ffmpeg are installed", "The transcription engine and the audio converter."); }
  else if (tt.state === "running"){ body = stepHtml(0, "Installing whisper.cpp and ffmpeg…", esc(tt.message || "This can take a few minutes.")); }
  else if (!info.brew){
    body = stepHtml(0, "Install Homebrew first", "Homebrew installs the transcription engine. Open Terminal, paste this line and press Return, then come back here:",
      `<code>/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"</code>`);
    act = `<button class="btn" data-copy='/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'>Copy</button>`;
  } else {
    body = stepHtml(0, "Install whisper.cpp and ffmpeg", (tt.state === "error" ? `<span style="color:var(--fail)">${esc(tt.message)}</span> ` : "") + "The transcription engine and the audio converter, installed with Homebrew.");
    act = `<button class="btn go" data-setup="tools">${tt.state === "error" ? "Try again" : "Install"}</button>`;
  }
  $("#stepToolsBody").innerHTML = body; $("#stepToolsAct").innerHTML = act;
  act = "";
  if (model){ body = stepHtml(1, "Speech model downloaded", "large-v3-turbo, the best balance of speed and accuracy."); }
  else if (tm.state === "running"){ body = stepHtml(0, `Downloading the speech model… ${tm.progress}%`, "About 1.6 GB. You can keep this window open while it downloads.",
      `<div class="mini"><i style="width:${tm.progress}%"></i></div>`); }
  else { body = stepHtml(0, "Download the speech model", (tm.state === "error" ? `<span style="color:var(--fail)">${esc(tm.message)}</span> ` : "") + "About 1.6 GB, saved to ~/whisper-models.");
    act = `<button class="btn go" data-setup="model">${tm.state === "error" ? "Try again" : "Download"}</button>`; }
  $("#stepModelBody").innerHTML = body; $("#stepModelAct").innerHTML = act;
}
const SPK_OPTS = `<option value="off">Off</option><option value="auto">Auto</option>
  <option value="2">2 people</option><option value="3">3 people</option><option value="4">4 people</option>
  <option value="5">5 people</option><option value="6">6 people</option>`;
function renderSpeakers(){
  const sel = $("#speakers"), note = $("#spkNote"), t = info.tasks.speakers;
  note.textContent = "";
  if (info.speakers_ready){
    if (sel.dataset.mode !== "ready"){ sel.innerHTML = SPK_OPTS; sel.dataset.mode = "ready"; sel.value = load("speakers") || "auto"; }
    sel.disabled = false;
  } else if (t.state === "running"){
    sel.dataset.mode = "running"; sel.disabled = true;
    sel.innerHTML = `<option>Setting up… ${t.progress || 0}%</option>`;
  } else {
    if (sel.dataset.mode !== "setup"){
      sel.innerHTML = `<option value="off">Off</option><option value="__setup" data-full="Set up speaker detection (60 MB download)…" data-short="Set up…">Set up speaker detection (60 MB download)…</option>`;
      sel.dataset.mode = "setup";
    }
    sel.disabled = false; sel.value = "off";
    if (t.state === "error") note.innerHTML = `<span style="color:var(--fail)">Speaker setup didn't finish: ${esc(t.message)}</span>`;
  }
  fitSelects();
}
$("#speakers").addEventListener("change", async e => {
  if (e.target.value !== "__setup") return;
  e.target.value = "off";
  await post("/api/setup", {what: "speakers"}); toast("Setting up speaker detection. It takes a minute or two."); loadInfo();
});
function renderUpdate(fromCheck){
  const u = info.update || {}, b = $("#banner"), t = info.tasks.update;
  if (t.state === "running" || updating){ b.hidden = false; $("#bannerText").textContent = t.message || "Updating…"; $("#bannerGo").hidden = true; $("#bannerLater").hidden = true; return; }
  if (t.state === "error"){ b.hidden = false; $("#bannerText").textContent = "Update failed: " + t.message; $("#bannerGo").hidden = false; $("#bannerGo").textContent = "Try again"; $("#bannerLater").hidden = false; return; }
  const skip = load("skip");
  b.hidden = !(u.available && (fromCheck || skip !== u.latest));
  $("#bannerGo").hidden = false; $("#bannerLater").hidden = false; $("#bannerGo").textContent = info.app ? "Download" : "Update now";
  if (!b.hidden) $("#bannerText").textContent = `Matita ${u.latest} is available. You have ${info.version}.`;
  else if (fromCheck) toast(u.checked ? (u.latest ? "You have the latest version." : "Couldn't check for updates right now.") : "Update checks aren't set up for this copy.");
}
$("#bannerLater").onclick = () => { store("skip", (info.update||{}).latest || ""); $("#banner").hidden = true; };
$("#bannerGo").onclick = async () => {
  if (info.app){ await post("/api/update"); $("#banner").hidden = true;
    return toast("Downloading the new Matita. Open the file and drag Matita to Applications to replace this one."); }
  const target = (info.update||{}).latest; updating = true; await post("/api/update"); renderUpdate();
  const started = Date.now();
  const wait = async () => {
    try{ const i = await (await fetch("/api/info")).json();
      if (i.version === target){ location.reload(); return; }
      if (i.tasks.update.state === "error"){ updating = false; info = i; renderUpdate(); return; }
    }catch(e){}
    if (Date.now() - started < 90000) setTimeout(wait, 1500); else { updating = false; toast("Update is taking long. Open Matita again from your Desktop."); }
  };
  setTimeout(wait, 1500);
};
document.addEventListener("click", async e => {
  const st = e.target.closest("[data-setup]");
  if (st){ await post("/api/setup", {what: st.dataset.setup}); loadInfo(); }
  const cp = e.target.closest("[data-copy]");
  if (cp){ await navigator.clipboard.writeText(cp.dataset.copy); toast("Copied. Paste it into Terminal."); }
});
function showFolder(path, isDefault){
  $("#whereText").textContent = path; $("#sPath").textContent = path;
  $("#sReset").hidden = !!isDefault;
}
["lang","prompt"].forEach(id => { const el=$("#"+id), v=load(id); if (v!==null) el.value=v; });
hintVal();
/* The first time Matita opens in its own window, size it to fit the content and centre it.
   After that the browser remembers whatever size the user picks. */
(function fitWindow(){
  if (!window.toolbar || window.toolbar.visible) return;            // a normal browser tab: leave it
  if (load("win") === "1") return;
  const w = Math.min(940, screen.availWidth - 40), h = Math.min(860, screen.availHeight - 40);
  try{
    window.resizeTo(w, h);
    window.moveTo((screen.availLeft || 0) + Math.round((screen.availWidth - w) / 2),
                  (screen.availTop || 0) + Math.round((screen.availHeight - h) / 2));
    store("win", "1");
  }catch(e){}
})();
/* Menus show a short value when closed ("Turbo") and the full description when open. */
function shortLabels(sel, full){
  [...sel.options].forEach(o => { if (o.dataset.full) o.textContent = full ? o.dataset.full : o.dataset.short; });
  if (!full) fitSelects();
}
document.querySelectorAll(".opt select").forEach(sel => {
  ["mousedown","focus","keydown"].forEach(ev => sel.addEventListener(ev, () => shortLabels(sel, true)));
  ["change","blur"].forEach(ev => sel.addEventListener(ev, () => shortLabels(sel, false)));
});
/* size each setting menu to its chosen option, so the row stays compact */
const measure = document.createElement("canvas").getContext("2d");
function fitSelects(){
  if (typeof optSummary === "function" && $("#optSumText")) optSummary();
  document.querySelectorAll(".opt select").forEach(sel => {
    const o = sel.options[sel.selectedIndex]; if (!o) return;
    const cs = getComputedStyle(sel); measure.font = `${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    sel.style.width = Math.ceil(measure.measureText(o.text).width + 22) + "px";
  });
}
document.querySelectorAll(".opt select").forEach(sel => sel.addEventListener("change", fitSelects));
function optSummary(){
  const t = id => { const s = $("#"+id), o = s && s.options[s.selectedIndex]; return o ? (o.dataset.short || o.textContent) : ""; };
  const sp = $("#speakers").value;
  const parts = [t("lang"), t("quality"), sp === "off" || sp === "__setup" ? "No speakers" : sp === "auto" ? "Speakers: auto" : sp + " speakers"];
  if ($("#prompt").value.trim()) parts.push("Hints");
  $("#optSumText").textContent = parts.filter(Boolean).join(" · ");
}
function optsOpen(on, kb){
  const sb = $("#optSum");                      // drop the panel right under the settings line
  $("#opts").style.top = (sb.offsetTop + sb.offsetHeight + 8) + "px";
  $("#opts").classList.toggle("open", on); $("#optSum").setAttribute("aria-expanded", on);
  if (on){ requestAnimationFrame(fitSelects); const f = $("#opts select"); if (f && kb) f.focus(); }
}
$("#optSum").onclick = e => { e.stopPropagation(); optsOpen(!$("#opts").classList.contains("open"), e.detail === 0); };
document.addEventListener("click", e => { if ($("#opts").classList.contains("open") && !e.target.closest("#opts, #optSum")) optsOpen(false); });
document.addEventListener("keydown", e => { if (e.key === "Escape" && $("#opts").classList.contains("open") && $("#hero").classList.contains("compact")){ optsOpen(false); $("#optSum").focus(); } });
["lang","quality","speakers","prompt"].forEach(id => { $("#"+id).addEventListener("change", optSummary); $("#"+id).addEventListener("input", optSummary); });
fitSelects();
if (!$("#lang").value) $("#lang").value = "en";
["lang","quality","prompt","speakers"].forEach(id => $("#"+id).addEventListener("change", e => { if (e.target.value !== "__setup") store(id, e.target.value); }));
function modelHint(){
  const m = ((info && info.modes) || []).find(x => x.id === $("#quality").value);
  const lang = $("#lang").value, turboOk = ["en","es","fr","ja","ko"].includes(lang);
  let t = m ? m.about + (m.ready ? "" : ` Downloads ${m.size} the first time.`) : "";
  if (m && m.id === "parakeet"){ if (!(m.langs || []).includes(lang)) t += " This language isn't one of them, so Whisper Turbo is used."; }
  else if (m && m.id !== "best" && !turboOk) t += " For this language it uses Large v3, which is more accurate here.";
  $("#quality").title = t; $("#quality").closest(".opt").title = t;      // shown on hover, not on the page
}
["lang","quality"].forEach(id => $("#"+id).addEventListener("change", modelHint));

async function folderResult(r){
  const d = await r.json();
  if (d.cancelled) return;
  if (!r.ok){ $("#sErr").textContent = d.error; return; }
  $("#sErr").textContent = ""; $("#sInput").value = "";
  showFolder(d.out_dir, d.out_dir === "~/Documents/Transcripts");
  toast("New transcripts will be saved to " + d.out_dir); poll();
}
const openSave = () => { $("#sErr").textContent = ""; $("#saveDlg").showModal(); };
$("#whereChange").onclick = openSave;
$("#sClose").onclick = () => $("#saveDlg").close();
$("#sChoose").onclick = async () => { $("#sChoose").disabled = true; const hint = dialogHint();
  try{ await folderResult(await post("/api/choose-folder")); } finally { clearTimeout(hint); $("#sChoose").disabled = false; } };
$("#sOpen").onclick = () => post("/api/reveal",{});
$("#sReset").onclick = async () => folderResult(await post("/api/set-folder",{path:""}));
$("#sSave").onclick = async () => {
  const p = $("#sInput").value.trim();
  if (!p){ $("#sErr").textContent = "Paste a folder path first, for example ~/Dropbox/Lecture notes"; return; }
  folderResult(await post("/api/set-folder",{path:p}));
};
$("#sInput").addEventListener("keydown", e => { if (!e.isComposing && e.keyCode !== 229 && e.key==="Enter") $("#sSave").click(); });

/* ---------- adding files ---------- */
async function filesFromDrop(dt){
  const entries = [...(dt.items||[])].map(i => i.webkitGetAsEntry && i.webkitGetAsEntry()).filter(Boolean);
  if (!entries.length) return [...dt.files];
  const out = [];
  async function walk(en){
    if (en.isFile){ out.push(await new Promise((res,rej)=>en.file(res,rej))); }
    else if (en.isDirectory){
      const r = en.createReader(); let batch;
      do { batch = await new Promise((res,rej)=>r.readEntries(res,rej)); for (const c of batch) await walk(c); } while (batch.length);
    }
  }
  for (const en of entries) await walk(en);
  return out;
}
async function enqueue(list){
  const all = [...list], files = all.filter(f => AUDIO.test(f.name)), skipped = all.length - files.length;
  if (!files.length){ toast(skipped ? "No audio or video files in what you dropped." : "No files added."); return; }
  uploading.total += files.length; render();
  let added = 0;
  for (const f of files){
    try{
      const r = await fetch("/api/upload", {method:"POST", body:f, headers:{
        "X-Transcriber": "1",
        "X-Filename": encodeURIComponent(f.name), "X-Lang": encodeURIComponent($("#lang").value),
        "X-Recorded": String(f.lastModified || 0),
        "X-Quality": encodeURIComponent($("#quality").value), "X-Prompt": encodeURIComponent($("#prompt").value),
        "X-Speakers": $("#speakers").disabled ? "off" : $("#speakers").value,
        ...(f._cap ? {"X-Capture": f._cap.id, "X-Mic-Start": String(f._cap.mic)} : {})}});
      if (r.ok) added++; else toast("Couldn't add " + f.name + ". Try again.");
    }catch(e){ toast("Couldn't add " + f.name); }
    uploading.done++; poll();
  }
  toast(`Added ${plural(added,"recording")}` + (skipped ? `, skipped ${skipped} that aren't audio` : ""));
  uploading.total -= files.length; uploading.done -= files.length;      // other batches may still be uploading
  if (uploading.total <= 0) uploading = {done:0,total:0};
  $("#queuePanel").scrollIntoView({behavior:"smooth", block:"start"});
  return added;
}
$("#choose").onclick = () => $("#file").click();
$("#file").addEventListener("change", e => { enqueue(e.target.files); e.target.value=""; });
let depth = 0; const ov = $("#overlay");
const isFileDrag = e => [...(e.dataTransfer?.types||[])].includes("Files");
function dragOn(on){
  document.body.classList.toggle("dragging", on);
  const tb = $(".dzt b"); if (tb) tb.textContent = on ? "Drop to transcribe" : "Drop audio or video here";
  $("#heroTitle").textContent = on ? "Drop to transcribe" : "Drop recordings here";
  if (on){ const r = $("#hero").getBoundingClientRect(); if (r.bottom < 80 || r.top > innerHeight - 80) window.scrollTo({top: 0, behavior: "smooth"}); }
}
document.addEventListener("dragenter", e => { if (!isFileDrag(e)) return; e.preventDefault(); if (depth++ === 0) dragOn(true); });
document.addEventListener("dragleave", () => { if (--depth<=0){ depth=0; dragOn(false); }});
document.addEventListener("dragover", e => e.preventDefault());
document.addEventListener("drop", async e => { e.preventDefault(); depth=0; dragOn(false); if (isFileDrag(e)) enqueue(await filesFromDrop(e.dataTransfer)); });

/* ---------- lid closed ---------- */
let lidBusy = false;
function lidNote(){
  const m = $("#lid").value;
  $("#lidNote").textContent = m === "sleep"
    ? "Your Mac sleeps as usual. Work pauses and continues when you open it."
    : (info && info.lid_active ? "Active now: your Mac stays awake with the lid closed until the queue is done. " : "")
      + "Only while something is transcribing. Leave it on a desk, not in a bag: it can get warm."
      + (m === "always" ? " Uses the battery quickly." : "");
}
$("#lid").addEventListener("change", async e => {
  const want = e.target.value, before = (info && info.lid) || "sleep";
  lidBusy = true; e.target.disabled = true;
  if (want !== "sleep") toast("macOS may ask for your password once. Look for the dialog (it can open behind this window).");
  try{
    const r = await post("/api/lid", {mode: want}); const d = await r.json();
    if (!r.ok){ toast(d.error || "Couldn't change this setting."); e.target.value = before; }
    else { if (info) info.lid = d.mode; toast(want === "sleep" ? "Your Mac will sleep normally when closed." : "Done. Transcribing continues with the lid closed."); }
  } catch(err){ toast("Couldn't change this setting."); e.target.value = before; }
  finally { lidBusy = false; e.target.disabled = false; lidNote(); }
});

/* ---------- AI assist ---------- */
let aiInfo = null;
async function loadAI(){
  try{ aiInfo = await (await fetch("/api/ai/info")).json(); }catch(e){ return; }
  const p = aiInfo.provider;
  const claudeLabel = (aiInfo.claude.models.find(m => m.id === aiInfo.claude.model) || {}).label || "";
  $("#aiState").textContent = p === "ollama" ? `On this Mac · ${aiInfo.ollama.model || "no model chosen"}`
    : p === "claude" ? `Claude API · ${claudeLabel.split(" ·")[0]}${aiInfo.claude.has_key ? "" : " · key missing"}` : "Off";
  $("#aiState").textContent = p === "off" ? "AI off" : p === "ollama" ? "AI · this Mac" : "AI · Claude";
  $("#aiOpen").classList.toggle("on", p !== "off");
  $("#aiChipVal").textContent = p === "off" ? "Off" : p === "ollama" ? (aiInfo.ollama.model || "This Mac") : "Claude " + (claudeLabel.split(" ·")[0] || "");
  $("#aiChip").classList.toggle("on", p !== "off");
  const tm = $("#titleMode"); tm.value = aiInfo.titles || "choose"; tm.disabled = p === "off";
  $("#titleRow").classList.toggle("off", p === "off");
  const tfOff = p === "off" || tm.value === "off";
  $("#tfRow").classList.toggle("off", tfOff);
  if (!tfT) tfFmt = JSON.parse(JSON.stringify(aiInfo.title_format || TF_DEFAULT));   // keep an edit that is still saving
  renderTF();                                       // draws checkboxes and the edge arrows disabled as needed
  $("#tfRow").querySelectorAll("select").forEach(x => x.disabled = tfOff);
  $("#titleNote").textContent = p === "off" ? "Turn on AI assist to use this"
    : tm.value === "auto" ? "Names like “New Recording 23” get the first idea; others show ideas to pick from"
    : tm.value === "off" ? "No title ideas" : "Pick one of three ideas in the AI summary";
  $("#aiOpen").title = "AI assist · " + (p === "ollama" ? "on this Mac" + (aiInfo.ollama.model ? " (" + aiInfo.ollama.model + ")" : "")
    : p === "claude" ? "Claude API" + (aiInfo.claude.has_key ? "" : " (key missing)") : "off");
  $("#aiState2").textContent = p === "ollama" ? `On this Mac · ${aiInfo.ollama.model || "no model chosen"}`
    : p === "claude" ? `Claude API · ${claudeLabel.split(" ·")[0]}` : "Off";
}
function gb(bytes){ return bytes ? (bytes/1e9).toFixed(1) + " GB" : ""; }
function drawAIDialog(){
  if (!aiInfo){ $("#aiDet").innerHTML = `<p class="badline">Couldn't read the AI settings. Close this and try again.</p>`; return; }
  const p = aiInfo.provider;
  document.querySelectorAll(".aiopt").forEach(b => b.setAttribute("aria-pressed", b.dataset.prov === p));
  let h = "";
  if (p === "off"){
    h = `<p>AI assist is off. Transcripts work exactly as before. Pick <b>On this Mac</b> to keep everything private, or <b>Claude API</b> for the best results.</p>`;
  } else if (p === "ollama"){
    const ms = aiInfo.ollama.models;
    h = `<p><b>How it works.</b> <a href="https://ollama.com" target="_blank" rel="noopener">Ollama</a> is a free app that runs open language models on your Mac. Matita sends it the transcript text over your Mac's own network loopback; nothing leaves your computer.</p>
      <table>
        <tr><td>Disk space</td><td>Ollama itself about 0.5 GB, plus the model: <b>Gemma 4 12B</b> (recommended) about 8 GB; <b>Gemma 4 E4B</b> about 5 GB for Macs with 8 GB of memory.</td></tr>
        <tr><td>Memory</td><td>16 GB or more for Gemma 4 12B. While it works, the model uses most of that memory, so other apps may slow down.</td></tr>
        <tr><td>Speed</td><td>An hour-long recording takes roughly 1–4 minutes for a summary and a few minutes for fixes on Apple silicon. Longer on older or 8 GB Macs.</td></tr>
        <tr><td>Quality</td><td>Good, readable summaries. Catches obvious misheard words but misses more subtle ones (names, jargon) than Claude. Works in many languages, including Korean.</td></tr>
        <tr><td>Cost</td><td>Free.</td></tr>
      </table>`;
    if (ms === null){
      h += `<p class="badline">Ollama isn't running.</p>
        <ol style="margin:6px 0 10px;padding-left:20px"><li>Install Ollama from <a href="https://ollama.com/download" target="_blank" rel="noopener">ollama.com/download</a> and open it.</li>
        <li>In Terminal, download the model: <code>ollama pull ${esc(aiInfo.ollama.suggested)}</code> <button class="link" id="aiCopyPull">Copy</button></li>
        <li>Come back and click <button class="link" id="aiRecheck">Check again</button></li></ol>`;
    } else if (!ms.length){
      h += `<p class="okline">Ollama is running, but no model is downloaded yet.</p>
        <p>In Terminal, run <code>ollama pull ${esc(aiInfo.ollama.suggested)}</code> <button class="link" id="aiCopyPull">Copy</button>, then <button class="link" id="aiRecheck">check again</button>.</p>`;
    } else {
      h += `<p class="okline">Ollama is running.</p>
        <label class="lbl" for="aiOllamaModel">Model</label>
        <select id="aiOllamaModel">${ms.map(m => `<option value="${esc(m.name)}">${esc(m.name)}${m.size ? " · " + gb(m.size) : ""}</option>`).join("")}</select>
        <p class="hint">Bigger models write better summaries but need more memory and time. To add another: <code>ollama pull ${esc(aiInfo.ollama.suggested)}</code></p>`;
    }
  } else {
    const c = aiInfo.claude;
    h = `<p><b>How it works.</b> Matita sends the transcript <b>text</b> (never the audio) to Anthropic's Claude API with your own API key, and shows the answer here. Your key is stored only on this Mac, in a file only your account can read.</p>
      <table>
        <tr><td>Disk space</td><td>None.</td></tr>
        <tr><td>Speed</td><td>Usually 5–30 seconds, even for long recordings.</td></tr>
        <tr><td>Quality</td><td>The strongest option: better at spotting misheard names and terms from context, and at summarizing long or mixed-language recordings.</td></tr>
        <tr><td>Cost</td><td>Billed by Anthropic per use. An hour of audio is roughly 10,000–15,000 words of text, which usually costs cents rather than dollars per run (Haiku least, Opus most). Check current prices in the Claude Console.</td></tr>
        <tr><td>Privacy</td><td>The text leaves your Mac and is handled under Anthropic's API terms. Keep this Off for confidential recordings, or use On this Mac.</td></tr>
      </table>
      <label class="lbl" for="aiKey">API key</label>
      ${c.has_key ? `<p class="okline" style="margin:4px 0 8px">Key saved (ends in ${esc(c.hint)}). <button class="link muted" id="aiKeyRemove">Remove</button></p>` : ""}
      <div style="display:flex;gap:10px"><input type="password" id="aiKey" placeholder="${c.has_key ? "Paste a new key to replace it" : "sk-ant-…"}" autocomplete="off" spellcheck="false"><button class="btn" id="aiKeySave">Save</button></div>
      <p class="hint">Create a key at <a href="https://console.anthropic.com/settings/keys" target="_blank" rel="noopener">console.anthropic.com</a> (Settings → API keys). Add a small credit balance there first.</p>
      <label class="lbl" for="aiClaudeModel" style="margin-top:14px;display:block">Model</label>
      <select id="aiClaudeModel">${c.models.map(m => `<option value="${esc(m.id)}">${esc(m.label)}</option>`).join("")}</select>`;
  }
  $("#aiDet").innerHTML = h;
  const om = $("#aiOllamaModel");
  if (om){
    if (!aiInfo.ollama.model || !aiInfo.ollama.models.some(m => m.name === aiInfo.ollama.model)){
      const pick = aiInfo.ollama.models.find(m => m.name.startsWith(aiInfo.ollama.suggested)) || aiInfo.ollama.models[0];
      om.value = pick.name; saveAI({ollama_model: pick.name});
    } else om.value = aiInfo.ollama.model;
    om.onchange = () => saveAI({ollama_model: om.value});
  }
  const cm = $("#aiClaudeModel"); if (cm){ cm.value = aiInfo.claude.model; cm.onchange = () => saveAI({claude_model: cm.value}); }
  const cp = $("#aiCopyPull"); if (cp) cp.onclick = () => navigator.clipboard.writeText("ollama pull " + aiInfo.ollama.suggested).then(() => toast("Copied. Paste it into Terminal."));
  const rc = $("#aiRecheck"); if (rc) rc.onclick = async () => { await loadAI(); drawAIDialog(); };
  const ks = $("#aiKeySave"); if (ks) ks.onclick = async () => {
    const key = $("#aiKey").value.trim(); if (!key) return toast("Paste your API key first.");
    ks.disabled = true; ks.textContent = "Checking…";
    const r = await post("/api/ai/settings", {claude_key: key}); const d = await r.json();
    ks.disabled = false; ks.textContent = "Save";
    if (!r.ok) return toast(d.error || "Couldn't save the key.");
    toast("Key saved and working."); await loadAI(); drawAIDialog();
  };
  const kr = $("#aiKeyRemove"); if (kr) kr.onclick = async () => { await post("/api/ai/settings", {claude_key: ""}); toast("Key removed."); await loadAI(); drawAIDialog(); };
}
async function saveAI(change){ await post("/api/ai/settings", change); await loadAI(); }
document.querySelectorAll(".aiopt").forEach(b => b.onclick = async () => { await saveAI({provider: b.dataset.prov}); drawAIDialog(); if (current) refreshAIButtons(); });
$("#setOpen").onclick = () => {
  const n = (info && info.hidden) || 0;
  $("#hiddenList").hidden = !n; $("#hiddenText").textContent = `${plural(n, "transcript")}. The files are still in your folder.`;
  setPane(load("setPane") || "general");
  drawFolders(); loadFolders();
  $("#setDlg").showModal();
};

/* Markdown notes + watch folders */
let folders = null;
async function loadFolders(){
  try{ folders = await (await fetch("/api/folders")).json(); }catch(e){ return; }
  drawFolders();
}
function drawFolders(){
  if (!folders) return;
  const n = folders.notes;
  $("#notesOn").checked = n.on; $("#notesText").textContent = n.on ? n.show : "Off"; $("#notesText").title = n.on ? n.dir : "";
  $("#notesDir").hidden = !n.on;
  $("#watchList").hidden = !folders.watch.length;
  $("#watchList").innerHTML = folders.watch.map((f, i) => `<li><div class="wp"><b title="${esc(f.path)}">${esc(f.show)}</b>
    <span class="${f.status === "offline" ? "off" : f.waiting ? "wait" : ""}">${f.status === "offline" ? "Not connected. Plug the drive in to pick up new files" : f.waiting ? "Picking up a new file…" : "Watching"}</span></div>
    <button class="btn soft sm" data-unwatch="${i}" aria-label="Stop watching ${esc(f.show)}">Remove</button></li>`).join("");
}
async function notesPost(body){
  const hint = body.choose ? dialogHint() : null;
  try{
    const r = await post("/api/notes", body), d = await r.json().catch(() => ({}));
    if (d.cancelled) return null;
    if (!r.ok){ toast(d.error || "Couldn't change that."); return null; }
    folders = d; drawFolders(); return d;
  } finally { clearTimeout(hint); }
}
$("#notesOn").onchange = async e => {
  const d = await notesPost({on: e.target.checked});
  if (!d){ drawFolders(); return; }
  if (d.notes.on) toast(`Writing a note for each transcript to ${d.notes.show}`, {label: "Change folder", fn: () => $("#notesDir").click()});
};
$("#notesDir").onclick = async () => {
  $("#notesDir").disabled = true;
  try{ const d = await notesPost({choose: true}); if (d) toast("Notes will be written to " + d.notes.show); }
  finally { $("#notesDir").disabled = false; }
};
$("#watchAdd").onclick = async () => {
  $("#watchAdd").disabled = true; const hint = dialogHint();
  try{
    const r = await post("/api/watch/add", {opts: currentOpts()}), d = await r.json().catch(() => ({}));
    if (d.cancelled) return;
    if (!r.ok) return toast(d.error || "Couldn't watch that folder.");
    folders = d; drawFolders();
    if (d.existing) toast(`Watching ${d.show}. ${plural(d.existing, "recording")} already there.`, {label: "Transcribe them too", fn: async () => {
      const x = await (await post("/api/watch/existing", {path: d.added})).json().catch(() => ({}));
      toast(x.count ? `Adding ${plural(x.count, "recording")} from ${d.show}…` : "Nothing left to add."); setTimeout(poll, 1500); }});
    else toast(`Watching ${d.show}. New recordings there will be transcribed.`);
  } finally { clearTimeout(hint); $("#watchAdd").disabled = false; }
};
$("#watchList").addEventListener("click", async e => {
  const b = e.target.closest("[data-unwatch]"); if (!b || !folders) return;
  const f = folders.watch[+b.dataset.unwatch]; if (!f) return;
  b.disabled = true;
  folders = await (await post("/api/watch/remove", {path: f.path})).json().catch(() => folders);
  drawFolders(); toast(`Stopped watching ${f.show}.`);
});
function setPane(name){
  if (!["general", "library", "ai"].includes(name)) name = "general";
  document.querySelectorAll("#setTabs [data-pane]").forEach(b => { b.setAttribute("aria-selected", b.dataset.pane === name);
    b.setAttribute("aria-pressed", b.dataset.pane === name); b.tabIndex = b.dataset.pane === name ? 0 : -1; });
  document.querySelectorAll("#setDlg .glist[data-pane]").forEach(g => g.classList.toggle("off-pane", g.dataset.pane !== name));
  store("setPane", name);
}
$("#setTabs").addEventListener("click", e => { const b = e.target.closest("[data-pane]"); if (b) setPane(b.dataset.pane); });
$("#setTabs").addEventListener("keydown", e => {
  if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
  const tabs = [...$("#setTabs").children], k = tabs.findIndex(b => b.getAttribute("aria-selected") === "true");
  const n = tabs[(k + (e.key === "ArrowRight" ? 1 : tabs.length - 1)) % tabs.length]; setPane(n.dataset.pane); n.focus(); e.preventDefault();
});
$("#unhide").onclick = async () => { await post("/api/unhide"); $("#hiddenList").hidden = true; toast("They're back on the list."); loadInfo(); poll(); };
$("#setClose").onclick = () => $("#setDlg").close();
let osDark = load("os") === "dark" ? true : load("os") === "light" ? false : null;            // what macOS shows; Chrome's own appearance setting can make the media query lie
function applyTheme(t){
  if (t === "light" || t === "dark") document.documentElement.setAttribute("data-theme", t);
  else {
    t = "system";
    if (osDark === null) document.documentElement.removeAttribute("data-theme");
    else document.documentElement.setAttribute("data-theme", osDark ? "dark" : "light");
  }
  store("theme", t);
  document.querySelectorAll("[data-theme-set]").forEach(b => b.setAttribute("aria-pressed", b.dataset.themeSet === t));
  if (typeof drawWave === "function") requestAnimationFrame(drawWave);
}
document.querySelectorAll("[data-theme-set]").forEach(b => b.onclick = () => applyTheme(b.dataset.themeSet));
applyTheme(load("theme") || "system");
async function followMac(){
  try{
    const d = (await (await fetch("/api/appearance")).json()).dark;
    if (d === true || d === false){
      const changed = d !== osDark; osDark = d; store("os", d ? "dark" : "light");
      if (changed && (load("theme") || "system") === "system") applyTheme("system");
    }
  }catch(e){}
}
followMac(); setInterval(followMac, 4000);
addEventListener("focus", followMac);
function backdropClose(id, guard){
  const d = $("#"+id); let downOn = null;
  d.addEventListener("pointerdown", e => { downOn = e.target; });
  d.addEventListener("click", e => { if (e.target === d && downOn === d && (!guard || guard())) d.close(); downOn = null; });
}
backdropClose("setDlg");
$("#aiOpen2").onclick = () => { $("#setDlg").close(); $("#aiOpen").click(); };
function hintVal(){ const v = $("#prompt").value.trim(); $("#hintVal").textContent = v ? (v.length > 28 ? v.slice(0, 26) + "…" : v) : "None"; }
$("#hintChip").onclick = () => {
  const pop = $("#hintPop"); pop.hidden = !pop.hidden; $("#hintChip").setAttribute("aria-expanded", !pop.hidden);
  if (!pop.hidden) $("#prompt").focus();
};
$("#prompt").addEventListener("input", hintVal);
$("#prompt").addEventListener("keydown", e => { if (e.isComposing || e.keyCode === 229) return; if (e.key === "Enter" || e.key === "Escape"){ e.preventDefault(); $("#hintPop").hidden = true; $("#hintChip").setAttribute("aria-expanded", false); $("#hintChip").focus(); } });
$("#aiChip").onclick = () => $("#aiOpen").click();
/* title format builder */
const TF_LABELS = {date: "Date", time: "Hour", topic: "Topic (AI)", keywords: "Keywords", original: "Original name"};
const TF_DEFAULT = {parts: [{k:"date",on:true},{k:"time",on:true},{k:"topic",on:true},{k:"keywords",on:false},{k:"original",on:false}], date: "iso", sep: " "};
let tfFmt = JSON.parse(JSON.stringify(TF_DEFAULT)), tfT = null, tfDrag = null;
function tfCompose(f){
  const d = new Date(), M = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"], p2 = n => String(n).padStart(2, "0");
  const date = f.date === "dmy" ? `${d.getDate()} ${M[d.getMonth()]} ${d.getFullYear()}` : f.date === "compact" ? `${String(d.getFullYear()).slice(2)}${p2(d.getMonth()+1)}${p2(d.getDate())}` : `${d.getFullYear()}-${p2(d.getMonth()+1)}-${p2(d.getDate())}`;
  const hour = `${d.getHours() % 12 || 12}${d.getHours() < 12 ? "am" : "pm"}`;
  const v = {date, time: hour, topic: "Quarterly revenue review", keywords: ["revenue","churn","pricing"].join(f.sep === "_" ? "-" : ", "), original: "New Recording 23"};
  const out = [];
  for (const p of f.parts){ if (!p.on) continue;
    if (p.k === "time" && out.length && out[out.length-1][0] === "date" && f.sep !== "_"){ out[out.length-1][1] += " " + v.time; continue; }
    out.push([p.k, v[p.k]]); }
  return out.map(x => x[1]).join(f.sep);
}
function renderTF(){
  const off = $("#tfRow") && $("#tfRow").classList.contains("off");
  $("#tfParts").innerHTML = tfFmt.parts.map((p, i) => `<li draggable="true" data-k="${p.k}" class="${p.on ? "on" : ""}">
    <span class="grip" aria-hidden="true"><svg width="10" height="14" viewBox="0 0 10 14" fill="currentColor"><circle cx="3" cy="3" r="1.2"/><circle cx="7" cy="3" r="1.2"/><circle cx="3" cy="7" r="1.2"/><circle cx="7" cy="7" r="1.2"/><circle cx="3" cy="11" r="1.2"/><circle cx="7" cy="11" r="1.2"/></svg></span>
    <label><input type="checkbox" ${p.on ? "checked" : ""} ${off ? "disabled" : ""} data-on="${i}"> ${TF_LABELS[p.k]}</label>
    <span class="mv"><button data-mv="-1" data-i="${i}" aria-label="Move ${TF_LABELS[p.k]} up" ${i === 0 || off ? "disabled" : ""}>↑</button><button data-mv="1" data-i="${i}" aria-label="Move ${TF_LABELS[p.k]} down" ${i === tfFmt.parts.length - 1 || off ? "disabled" : ""}>↓</button></span></li>`).join("");
  $("#tfDate").value = tfFmt.date; $("#tfSep").value = tfFmt.sep;
  $("#tfPreview").textContent = tfCompose(tfFmt) || "—";
}
function tfSave(){
  renderTF(); clearTimeout(tfT);
  tfT = setTimeout(async () => {
    const snap = JSON.parse(JSON.stringify(tfFmt));
    try{ await post("/api/ai/settings", {title_format: snap}); if (aiInfo) aiInfo.title_format = snap; }
    finally{ if (JSON.stringify(snap) === JSON.stringify(tfFmt)) tfT = null; }
  }, 250);
}
$("#tfParts").addEventListener("change", e => {
  const i = e.target.dataset.on; if (i == null) return;
  tfFmt.parts[+i].on = e.target.checked;
  if (!tfFmt.parts.some(p => p.on)){ tfFmt.parts[+i].on = true; toast("Keep at least one part."); }
  tfSave();
  const again = $(`#tfParts [data-on="${i}"]`); if (again) again.focus();
});
$("#tfParts").addEventListener("click", e => {
  const b = e.target.closest("[data-mv]"); if (!b) return;
  const i = +b.dataset.i, j = i + (+b.dataset.mv); if (j < 0 || j >= tfFmt.parts.length) return;
  [tfFmt.parts[i], tfFmt.parts[j]] = [tfFmt.parts[j], tfFmt.parts[i]]; tfSave();
  const again = $(`#tfParts [data-mv="${b.dataset.mv}"][data-i="${j}"]`); if (again && !again.disabled) again.focus();
});
$("#tfParts").addEventListener("dragstart", e => { const li = e.target.closest("li"); if (!li || $("#tfRow").classList.contains("off")) return e.preventDefault();
  tfDrag = li; li.classList.add("drag"); e.dataTransfer.effectAllowed = "move"; e.dataTransfer.setData("text/plain", li.dataset.k); });
$("#tfParts").addEventListener("dragover", e => { if (!tfDrag) return; e.preventDefault();
  const li = e.target.closest("li"); if (!li || li === tfDrag) return;
  const r = li.getBoundingClientRect(); li.parentNode.insertBefore(tfDrag, e.clientY < r.top + r.height / 2 ? li : li.nextSibling); });
$("#tfParts").addEventListener("drop", e => { if (!tfDrag) return; e.preventDefault();
  const order = [...$("#tfParts").children].map(li => li.dataset.k);
  tfFmt.parts.sort((a, b) => order.indexOf(a.k) - order.indexOf(b.k));
  tfDrag.classList.remove("drag"); tfDrag = null; tfSave(); });
$("#tfParts").addEventListener("dragend", () => { if (!tfDrag) return;    // dropped outside: put things back
  tfDrag.classList.remove("drag"); tfDrag = null; renderTF(); });
$("#tfDate").onchange = e => { tfFmt.date = e.target.value; tfSave(); };
$("#tfSep").onchange = e => { tfFmt.sep = e.target.value; tfSave(); };
$("#titleMode").onchange = async e => { await post("/api/ai/settings", {titles: e.target.value}); await loadAI(); if (view === "ai") drawReader(); };
$("#aiOpen").onclick = async () => { await loadAI(); drawAIDialog(); $("#aiDlg").showModal(); };
$("#aiClose").onclick = () => $("#aiDlg").close();
backdropClose("aiDlg");
$("#aiDlg").addEventListener("close", () => { if (current) refreshAIButtons(); });

/* AI in the reader */
let aiDoc = null, aiTimer = null, aiShow = null;
function mdLite(t){   // a small, safe Markdown subset: headings, bullets, bold
  const inline = x => esc(x).replace(/\*\*(.+?)\*\*/g, "<b>$1</b>");
  let out = "", inList = false;
  for (const raw of t.split("\n")){
    const line = raw.trim();
    const li = line.match(/^[-*•]\s+(.*)/);
    if (li){ if (!inList){ out += "<ul>"; inList = true; } out += `<li>${inline(li[1])}</li>`; continue; }
    if (inList){ out += "</ul>"; inList = false; }
    const hd = line.match(/^#{1,4}\s+(.*)/);
    if (hd) out += `<h4>${inline(hd[1])}</h4>`; else if (line) out += `<p>${inline(line)}</p>`;
  }
  return out + (inList ? "</ul>" : "");
}
const aiOn = () => !!(aiInfo && aiInfo.provider !== "off");
function refreshAIButtons(){
  const on = aiOn();
  $("#aiSum").hidden = true;                     // the summary is a view now (first tab)
  $("#aiFix").hidden = !on; $("#vAI").hidden = !on;
  if (!on && view === "ai" && $("#reader").open) setView("time");
}
/* the AI summary view */
const aiAuto = new Set();
function titleIdeas(){
  const s = aiDoc && aiDoc.summary; if (!s) return [];
  return (Array.isArray(s.titles) && s.titles.length ? s.titles : (s.title ? [s.title] : [])).filter(t => typeof t === "string" && t);
}
function aiViewHTML(){
  const d = aiDoc;
  const busy = (msg, more) => `<div class="aiview"><div class="busy"><span class="spin"></span>${esc(msg)}…</div>${more ? `<p class="hint">${more}</p>` : ""}</div>`;
  if (!d) return busy("Loading");
  const running = d.run && !d.run.done && d.run.kind === "summary";
  if (running) return busy(d.run.progress || "Writing the summary", "Long recordings take a minute or two. You can read the transcript meanwhile.");
  if (d.summary){
    const ideas = titleIdeas(), now = current ? bare(current.name) : "";
    const sug = ideas.length && (aiInfo && aiInfo.titles) !== "off"
      ? `<div class="tideas"><div class="tih">Title ideas</div>${ideas.map((t, i) => t === now
          ? `<div class="tidea on"><span>${esc(t)}</span><em>Current</em></div>`
          : `<button class="tidea" data-title="${i}"><span>${esc(t)}</span><em>Use</em></button>`).join("")}</div>` : "";
    return `<div class="aiview"><div class="aih"><span>Summary · ${esc(d.summary.by || "AI")}</span><span class="acts"><button data-ai="copy">Copy</button><button data-ai="redo">Write again</button></span></div>${sug}<div class="md">${mdLite(d.summary.text)}</div></div>`;
  }
  if (d.run && d.run.error && d.run.kind === "summary")
    return `<div class="aiview"><p class="badline">${esc(d.run.error)}</p><button class="btn soft" data-ai="redo">Try again</button></div>`;
  return busy("Starting the summary");
}
async function ensureSummary(){
  if (!current || !aiOn() || !aiDoc) return;
  const id = current.id;
  if (aiDoc.summary || (aiDoc.run && !aiDoc.run.done) || (aiDoc.run && aiDoc.run.error) || aiAuto.has(id)) return;
  if (aiInfo.provider === "claude"){                  // paid: skip transcripts you only flick past with ⌘[ / ⌘]
    aiAuto.add(id);
    await new Promise(ok => setTimeout(ok, 1500));
    if (!current || current.id !== id || view !== "ai" || !$("#reader").open){ aiAuto.delete(id); return; }
  } else aiAuto.add(id);
  const r = await post("/api/ai/run", {id, kind: "summary"});
  if (!current || current.id !== id || !aiDoc) return;
  if (r.status === 409){ loadAIDoc(); return; }        // already running (e.g. naming a new recording): just follow it
  if (!r.ok){ const d = await r.json().catch(() => ({})); aiDoc.run = {kind: "summary", done: true, error: d.error || "Couldn't start the summary."}; if (view === "ai") drawReader(); return; }
  loadAIDoc();
}
async function loadAIDoc(){
  if (!current) return;
  const id = current.id;
  let d;
  try{ d = await (await fetch("/api/ai?id=" + encodeURIComponent(id))).json(); }catch(e){ return; }
  if (!current || current.id !== id) return;                // another transcript is open now
  aiDoc = d;
  drawAIBox();
  if (view === "ai"){ drawReader(); ensureSummary(); }
  clearTimeout(aiTimer);
  if (aiDoc.run && !aiDoc.run.done && $("#reader").open) aiTimer = setTimeout(loadAIDoc, 1500);
  else if (aiDoc.run && aiDoc.run.error && aiShow === aiDoc.run.kind){ toast(aiDoc.run.error); }
}
function drawAIBox(){
  const box = $("#aiBox"), d = aiDoc;
  if (!d || !aiShow){ box.hidden = true; return; }
  const running = d.run && !d.run.done && d.run.kind === aiShow;
  let h = "";
  if (aiShow === "summary"){
    h = `<div class="ah"><b>AI summary${d.summary && !running ? " · " + esc(d.summary.by) : ""}</b><div class="acts" style="margin:0">
      ${d.summary && !running ? `<button id="aiCopySum">Copy</button><button id="aiRedo">Write again</button>` : ""}<button class="quiet" id="aiHide">Hide</button></div></div>`;
    if (running) h += `<div class="busy"><span class="spin"></span>${esc(d.run.progress || "Working")}…</div>`;
    else {
      if (d.run && d.run.error && d.run.kind === "summary") h += `<p class="badline">${esc(d.run.error)}</p>`;
      if (d.summary && d.summary.title && current && d.summary.title !== bare(current.name))
        h += `<div class="sugt">Suggested title <b>${esc(d.summary.title)}</b><button class="link" data-title="1">Use this title</button></div>`;
      if (d.summary) h += `<div class="md">${mdLite(d.summary.text)}</div>`;
    }
  } else {
    const fx = d.fixes || [];
    h = `<div class="ah"><b>Suggested fixes${d.fixes_by && !running ? " · " + esc(d.fixes_by) : ""}</b><div class="acts" style="margin:0">
      ${fx.length && !running ? `<button id="aiAcceptAll">Accept all</button>` : ""}${!running ? `<button id="aiRedo">Check again</button>` : ""}<button class="quiet" id="aiHide">Hide</button></div></div>`;
    if (running) h += `<div class="busy"><span class="spin"></span>${esc(d.run.progress || "Working")}…</div>`;
    else if (d.run && d.run.error && d.run.kind === "fixes" && !fx.length) h += `<p class="badline">${esc(d.run.error)}</p>`;
    else if (!fx.length) h += `<p class="hint" style="margin:6px 0">${d.fixes ? "No more suggestions. The transcript looks right to the model." : ""}</p>`;
    else h += fx.map((f, i) => `<div class="fix"><div class="chg"><del>${esc(f.find)}</del> → <ins>${esc(f.replace)}</ins>${f.count > 1 ? ` <span class="muted">(${f.count} places)</span>` : ""}</div>
        <div class="why">${esc(f.why)}</div>
        <div class="fa acts" style="margin:0"><button data-fix="${i}" data-do="apply">Accept</button><button class="quiet" data-fix="${i}" data-do="dismiss">Dismiss</button></div></div>`).join("");
  }
  box.innerHTML = h; box.hidden = false;
  const on = (id, fn) => { const el = $("#"+id); if (el) el.onclick = fn; };
  on("aiHide", () => { aiShow = null; drawAIBox(); });
  on("aiRedo", () => runAI(aiShow, true));
  on("aiCopySum", () => navigator.clipboard.writeText(d.summary.text).then(() => toast("Summary copied.")));
  on("aiAcceptAll", () => fixAct({all: true}, "apply"));
}
async function runAI(kind, again){
  if (!current) return;
  if (kind === "summary"){
    if (again) aiAuto.add(current.id);              // this call starts it; don't let the view start a second one
    setView("ai");
    if (again){ aiDoc && (aiDoc.summary = null); drawReader(); const r = await post("/api/ai/run", {id: current.id, kind}); if (!r.ok) toast((await r.json().catch(() => ({}))).error || "Couldn't start."); loadAIDoc(); }
    return;
  }
  aiShow = kind;
  const have = aiDoc && (kind === "summary" ? aiDoc.summary : aiDoc.fixes);
  if (have && !again){ drawAIBox(); return; }
  const r = await post("/api/ai/run", {id: current.id, kind});
  if (!r.ok){ toast((await r.json()).error || "Couldn't start."); }
  loadAIDoc();
}
async function fixAct(sel, act){
  if (!current) return;
  let r, d = {};
  try{ r = await post(act === "apply" ? "/api/ai/apply" : "/api/ai/dismiss", Object.assign({id: current.id}, sel)); d = await r.json().catch(() => ({})); }catch(e){}
  if (!r || !r.ok){ drawAIBox(); return toast(d.error || "Couldn't change the transcript."); }
  if (act === "apply"){
    toast(d.changed ? `Fixed in ${plural(d.changed, "place")}.` : (d.note || "That text wasn't found anymore."));
    try{ doc = await getText(current.id); drawReader(); }catch(e){}
  }
  await loadAIDoc();
}
$("#aiSum").onclick = () => runAI("summary");
$("#aiFix").onclick = () => runAI("fixes");
$("#aiBox").addEventListener("click", async e => {
  if (e.target.closest("[data-title]") && current && aiDoc && aiDoc.summary){
    if (renaming) return;
    const id = current.id, name = await renameTo(id, aiDoc.summary.title);
    if (name && current && current.id === id){ $("#rName").textContent = name; drawAIBox(); }
    return;
  }
  const b = e.target.closest("[data-fix]"); if (!b || !aiDoc || !aiDoc.fixes) return;
  const f = aiDoc.fixes[+b.dataset.fix]; if (!f) return;
  $("#aiBox").querySelectorAll("button").forEach(x => x.disabled = true);    // no double clicks
  fixAct({find: f.find}, b.dataset.do);
});
$("#reader").addEventListener("close", () => { clearTimeout(aiTimer); aiShow = null; $("#aiBox").hidden = true; });

/* ---------- recording from the microphone ---------- */
let rec = null;
for (let i = 0; i < 28; i++) $("#recMeter").appendChild(document.createElement("span"));
let recStarting = false;
async function startRecording(){
  if (rec || recStarting) return;
  recStarting = true;
  try{ await startRecordingInner(); } finally { recStarting = false; }
}
async function startRecordingInner(){
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia || !window.MediaRecorder)
    return toast("This browser can't record audio. Open Matita in Chrome.");
  let stream;
  try{ stream = await navigator.mediaDevices.getUserMedia({audio:{echoCancellation:false, noiseSuppression:false, autoGainControl:true}}); }
  catch(e){
    return toast(e && e.name === "NotAllowedError"
      ? "Microphone access is blocked. Allow it in the browser's address-bar settings, and for your browser in System Settings → Privacy & Security → Microphone."
      : "Couldn't find a microphone.");
  }
  const type = ["audio/webm;codecs=opus","audio/webm","audio/mp4"].find(t => MediaRecorder.isTypeSupported(t)) || "";
  let mr;
  try{ mr = new MediaRecorder(stream, type ? {mimeType:type, audioBitsPerSecond:96000} : undefined); }
  catch(e){ stream.getTracks().forEach(t => t.stop()); return toast("Couldn't start recording."); }
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const an = ctx.createAnalyser(); an.fftSize = 64; ctx.createMediaStreamSource(stream).connect(an);
  rec = {mr, stream, ctx, an, chunks:[], type: mr.mimeType || type, started: Date.now(), t0: Date.now(), elapsed: 0, lock: null, discard: false};
  mr.ondataavailable = e => { if (e.data && e.data.size) rec.chunks.push(e.data); };
  mr.onstop = finishRecording;
  mr.start(1000);
  try{ rec.lock = await navigator.wakeLock.request("screen"); }catch(e){}
  post("/api/awake", {on: true}).catch(() => {});
  rec.awake = setInterval(() => { post("/api/awake", {on: true}).catch(() => {});        // keep-awake lasts 10 min
    if (rec && rec.cap) post("/api/capture/ctl", {id: rec.cap, do: "alive"}).catch(() => {}); }, 5 * 60 * 1000);
  $("#recAppsRow").hidden = !(info && info.capture);
  $("#recApps").checked = !!(info && info.capture) && load("recApps") === "1"; drawRecApps();
  if ($("#recApps").checked) capStart();
  $("#recState").classList.remove("paused"); $("#recState span").textContent = "Recording"; $("#recPause").textContent = "Pause";
  $("#recDlg").showModal(); recTick();
}
function recSeconds(){ return rec ? rec.elapsed + (rec.mr.state === "recording" ? (Date.now() - rec.started)/1000 : 0) : 0; }
function recTick(){
  if (!rec) return;
  $("#recClock").textContent = mmss(recSeconds());
  const data = new Uint8Array(rec.an.frequencyBinCount); rec.an.getByteTimeDomainData(data);
  let peak = 0; for (const v of data) peak = Math.max(peak, Math.abs(v - 128));
  const level = rec.mr.state === "recording" ? Math.min(1, peak / 64) : 0;
  [...$("#recMeter").children].forEach((b, i) => {
    const shape = 1 - Math.abs(i - 13.5) / 14;                 // taller in the middle
    b.style.height = Math.max(4, 44 * level * (0.35 + 0.65 * shape) * (0.75 + 0.25 * Math.random())) + "px";
  });
  requestAnimationFrame(recTick);
}
function stopRecording(discard){
  if (!rec) return;
  rec.discard = discard;
  if (rec.mr.state !== "inactive") rec.mr.stop(); else finishRecording();
}
/* other apps' sound (calls), recorded by Matita Audio alongside the microphone */
function drawRecApps(){
  const on = $("#recApps").checked, live = rec && rec.cap;
  $("#recAppsRow").classList.toggle("on", !!(on && live));
  $("#recAppsNote").textContent = !on ? "For calls: records the other people too"
    : live ? "Recording your Mac's sound as well" : "Starting…";
}
async function capStart(){
  const r0 = rec; if (!r0 || r0.cap || r0.capBusy) return;
  r0.capBusy = true; drawRecApps();
  try{
    const r = await post("/api/capture/start"), d = await r.json().catch(() => ({}));
    if (!r.ok){ toast(d.error || "Couldn't record other apps' sound."); $("#recApps").checked = false; return; }
    if (rec !== r0){ post("/api/capture/ctl", {id: d.id, do: "stop"}).then(() => post("/api/capture/discard", {id: d.id})); return; }
    r0.cap = d.id;
    // the microphone file has no audio for pauses so far: leave them out when lining the two up
    r0.capPausedMs = (r0.pausedMs || 0) + (r0.mr.state === "paused" && r0.pausedAt ? Date.now() - r0.pausedAt : 0);
    if (r0.mr.state === "paused") post("/api/capture/ctl", {id: d.id, do: "pause"});
    if (!$("#recApps").checked) post("/api/capture/ctl", {id: d.id, do: "mute"});
  } finally { r0.capBusy = false; drawRecApps(); }
}
$("#recApps").onchange = e => {
  store("recApps", e.target.checked ? "1" : "0");
  if (!rec) return;
  if (e.target.checked){ if (rec.cap) post("/api/capture/ctl", {id: rec.cap, do: "unmute"}); else capStart(); }
  else if (rec.cap) post("/api/capture/ctl", {id: rec.cap, do: "mute"});      // silence from here on; timing stays lined up
  drawRecApps();
};
function finishRecording(){
  const r = rec; if (!r) return; rec = null;
  r.stream.getTracks().forEach(t => t.stop()); try{ r.ctx.close(); }catch(e){}
  if (r.lock) r.lock.release().catch(()=>{});
  clearInterval(r.awake);
  post("/api/awake", {on: false}).catch(() => {});
  $("#recDlg").close();
  const capDone = r.cap ? post("/api/capture/ctl", {id: r.cap, do: "stop"}).catch(() => {}) : Promise.resolve();
  if (r.discard || !r.chunks.length){
    if (r.cap) capDone.then(() => post("/api/capture/discard", {id: r.cap}));
    toast(r.discard ? "Recording discarded." : "Nothing was recorded."); return;
  }
  const ext = /mp4/.test(r.type) ? ".m4a" : ".webm";
  const d = new Date(r.t0), pad = n => String(n).padStart(2, "0");       // named after when it started
  const name = `Recording ${d.getFullYear()}-${pad(d.getMonth()+1)}-${pad(d.getDate())} ${pad(d.getHours())}.${pad(d.getMinutes())}${ext}`;
  const file = new File(r.chunks, name, {type: r.type || "audio/webm", lastModified: r.t0});
  if (r.cap) file._cap = {id: r.cap, mic: r.t0 + (r.capPausedMs || 0)};
  capDone.then(() => enqueue([file])).then(n => { if (!n) keepRecording(file); }, () => keepRecording(file));
}
function keepRecording(file){
  // The upload failed: don't lose the audio. Save it to Downloads so it can be added later.
  const a = document.createElement("a"); a.href = URL.createObjectURL(file); a.download = file.name;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 60000);
  setTimeout(() => toast("Couldn't add the recording, so it was saved to your Downloads folder. Add it from there."), 2700);
}
$("#recOpen").onclick = startRecording;
$("#recStop").onclick = () => stopRecording(false);
$("#recDiscard").onclick = () => { if (recSeconds() < 5 || confirm("Discard this recording?")) stopRecording(true); };
$("#recPause").onclick = () => {
  if (!rec) return;
  if (rec.cap) post("/api/capture/ctl", {id: rec.cap, do: rec.mr.state === "recording" ? "pause" : "resume"});
  if (rec.mr.state === "recording") rec.pausedAt = Date.now();
  else if (rec.mr.state === "paused" && rec.pausedAt){ rec.pausedMs = (rec.pausedMs || 0) + Date.now() - rec.pausedAt; rec.pausedAt = null; }
  if (rec.mr.state === "recording"){ rec.elapsed += (Date.now() - rec.started)/1000; rec.mr.pause();
    $("#recPause").textContent = "Resume"; $("#recState").classList.add("paused"); $("#recState span").textContent = "Paused"; }
  else if (rec.mr.state === "paused"){ rec.started = Date.now(); rec.mr.resume();
    $("#recPause").textContent = "Pause"; $("#recState").classList.remove("paused"); $("#recState span").textContent = "Recording"; }
};
$("#recDlg").addEventListener("cancel", e => e.preventDefault());
document.addEventListener("visibilitychange", async () => {     // the screen lock is dropped when hidden
  if (rec && document.visibilityState === "visible"){ try{ rec.lock = await navigator.wakeLock.request("screen"); }catch(e){} }
});     // Esc mustn't throw a recording away
window.addEventListener("beforeunload", e => { if (rec){ e.preventDefault(); e.returnValue = ""; } });

/* ---------- rendering ---------- */
const isActive = j => ["queued","downloading","converting","transcribing","speakers"].includes(j.state);
function speedRatio(){
  const d = jobs.filter(j => j.state==="done" && !j.library && j.duration && j.started && j.finished);
  const work = d.reduce((a,j)=>a+(j.finished-j.started),0), audio = d.reduce((a,j)=>a+j.duration,0);
  return audio ? work/audio : null;
}
function remainingFor(j, now, ratio){
  if (j.state==="transcribing" && j.progress>3 && j.started){ const el = now - j.started; return el/(j.progress/100) - el; }
  if (j.state==="queued" && j.duration && ratio) return j.duration*ratio;
  return null;
}

function renderQueue(now){
  const q = jobs.filter(j => isActive(j) || j.state==="error" || j.state==="cancelled");
  const running = jobs.filter(isActive), ratio = speedRatio();
  const waiting = running.length + Math.max(0, uploading.total-uploading.done);
  $("#queuePanel").hidden = !q.length && !uploading.total;
  $("#cancelAll").hidden = !running.length;
  $("#pauseAll").hidden = !running.length && !paused;
  $("#pauseAll").textContent = paused ? "Resume" : "Pause";
  $("#pauseAll").classList.toggle("go", paused);
  let left = 0, known = true;
  running.forEach(j => { const r = remainingFor(j, now, ratio); if (r==null) known=false; else left+=r; });

  const parts = [];
  if (uploading.total) parts.push(`Adding ${Math.min(uploading.done+1,uploading.total)} of ${uploading.total}.`);
  if (running.length && paused) parts.push(`Paused with ${plural(running.length,"recording")} to go. Resume continues exactly where it stopped.`);
  else if (running.length) parts.push(`${plural(running.length,"recording")} to go` + (known && left>0 ? `, ${roughly(left)}.` : "."));
  else if (q.length) parts.push("Nothing running. Try the files below again or remove them.");
  $("#qSub").textContent = parts.join(" ");


  setHTML($("#queue"), q.map(j => {
    const [num, unit] = lengthCell(j.duration);
    let icon = I.clock, status = "", acts = "", extra = "", cell = `<b>${num}</b><span>${unit}</span>`;
    if (j.state==="queued"){ status = `<span class="s wait">Waiting</span><span>Starts after the one before it.</span>`; acts = `<button data-act="cancel" data-id="${j.id}">Cancel</button>`; }
    if (j.state==="speakers"){
      icon = I.mic; status = `<span class="s run">Identifying speakers</span>`;
      acts = `<button data-act="cancel" data-id="${j.id}">Cancel</button>`;
      cell = `<b>${j.progress||0}%</b><span>done</span>`;
      extra = `<div class="bar" style="width:${j.progress||0}%"></div>`;
    }
    if (j.state==="downloading"){
      const d = j.dl, mb = n => n >= 1e9 ? (n/1e9).toFixed(1) + " GB" : Math.round(n/1e6) + " MB";
      const left = d && d.eta != null ? (d.eta < 60 ? "less than a minute left" : `about ${Math.ceil(d.eta/60)} min left`) : "";
      const sizes = d && d.total ? `${mb(d.got)} of ${mb(d.total)}` : "";
      const tip = j.lang === "en" && !String(j.model || "").startsWith("parakeet") && /Downloading (Turbo|Large)/.test(j.note || "")
        ? " · Tip: Parakeet is a smaller, faster download for English" : "";
      icon = I.clock; status = `<span class="s run">${esc(j.note || "Downloading the model")}${sizes ? " · " + sizes : ""}${left ? " · " + left : ""}${esc(tip)}</span>`;
      acts = `<button data-act="cancel" data-id="${j.id}">Cancel</button>`;
      cell = `<b>${j.progress||0}%</b><span>downloaded</span>`;
      extra = `<div class="bar" style="width:${j.progress||0}%"></div>`;
    }
    if (j.state==="converting"){ icon = I.mic; status = `<span class="s run">Preparing audio</span>`; acts = `<button data-act="cancel" data-id="${j.id}">Cancel</button>`; extra = `<div class="bar" style="width:2%"></div>`; }
    if (j.state==="transcribing"){
      icon = I.mic; const r = remainingFor(j, now, ratio);
      status = `<span class="s run">Transcribing</span>${r ? `<span>${roughly(r)}</span>` : ""}`;
      acts = `<button data-act="cancel" data-id="${j.id}">Cancel</button>`;
      cell = `<b>${j.progress||0}%</b><span>done</span>`;
      extra = (j.preview ? `<div class="live">${esc(j.preview)}</div>` : "") + `<div class="bar" style="width:${j.progress||0}%"></div>`;
    }
    if (j.state==="error"){ icon = I.warn; status = `<span class="s fail">Didn't finish</span>`;
      acts = `<button data-act="retry" data-id="${j.id}">Try again</button><button class="quiet" data-act="remove" data-id="${j.id}">Remove</button>`;
      extra = `<div class="err">${esc(j.error)}</div>`; }
    if (j.state==="cancelled"){ icon = I.warn; status = `<span class="s wait">Cancelled</span>`;
      acts = `<button data-act="retry" data-id="${j.id}">Start again</button><button class="quiet" data-act="remove" data-id="${j.id}">Remove</button>`; }
    if (paused && ["converting","transcribing","speakers"].includes(j.state))
      status = `<span class="s wait">Paused</span>${j.progress ? `<span>at ${j.progress}%</span>` : ""}`;
    const pct = ["transcribing","speakers","downloading"].includes(j.state) ? `${j.progress||0}%` : (j.duration ? clock(j.duration) : "");
    const barW = j.state==="converting" ? 2 : (["transcribing","speakers","downloading"].includes(j.state) ? (j.progress||0) : null);
    const live = j.state==="transcribing" && j.preview ? `<div class="live">${esc(j.preview)}</div>` : "";
    const err = j.state==="error" ? `<div class="err">${esc(j.error)}</div>` : "";
    return `<article class="row qrow${j.state==="error" ? " bad" : ""}">
      <div class="lm"><div class="name">${esc(bare(j.name))}</div><div class="status">${status}</div>
        ${barW != null ? `<div class="track"><i style="width:${barW}%"></i></div>` : ""}${live}${err}</div>
      <div class="lr"><div class="pct">${esc(pct)}</div><div class="acts">${acts}</div></div></article>`;
  }).join(""));
}

let showOlder = false, libSeen = null;
let libOrder = [];
const opened = new Set();
function renderLibrary(now){
  const term = $("#filter").value.trim().toLowerCase();
  const done = jobs.filter(j => j.state==="done").sort((a,b)=>whenOf(b)-whenOf(a));      // newest recording first
  const RECENT = 7 * 86400;
  const matches = term ? done.filter(j => (j.name + " " + (j.summary||"")).toLowerCase().includes(term)) : done;
  const recent = matches.filter(j => !j.finished || now - j.finished < RECENT || now - whenOf(j) < RECENT);
  const older = matches.length - recent.length;
  // Show the last 7 days by default (at least a few items, so the list never looks empty).
  const shown = (term || showOlder) ? matches : (recent.length >= 3 ? recent : matches.slice(0, Math.max(3, recent.length)));
  const hiddenOlder = matches.length - shown.length;
  const words = done.reduce((a,j)=>a+(j.words||0),0);
  $("#zipAll").disabled = !done.length; $("#clearList").disabled = !done.length; $("#listMore").hidden = !done.length;
  $("#zipAll").closest(".tools").hidden = !done.length && !$("#filter").value;     // nothing to search or save yet
  $("#lSub").textContent = done.length ? done.length : "";
  const lib = $("#library");
  if (!done.length){
    const hid = info && info.hidden ? info.hidden : 0;
    setHTML(lib, hid
      ? `<div class="quiet-empty"><p>Your list is empty. ${plural(hid, "transcript")} you removed earlier ${hid === 1 ? "is" : "are"} still in your folder.</p><button class="btn soft" id="unhideAll">Show ${hid === 1 ? "it" : "them"} again</button></div>`
      : `<div class="starts" role="list">
          <button class="start" role="listitem" data-start="choose"><i>${SI.file}</i><b>Add a recording</b><span>Drop a file anywhere, or press ⌘O</span></button>
          <button class="start" role="listitem" data-start="vmOpen"><i>${SI.phone}</i><b>From your iPhone</b><span>Pick Voice Memos, or transcribe new ones automatically</span></button>
          <button class="start" role="listitem" data-start="recOpen"><i>${SI.mic}</i><b>${info && info.capture ? "Record a lecture or call" : "Record a lecture"}</b><span>${info && info.capture ? "Your microphone, plus the other people in a call" : "With your Mac's microphone"}</span></button>
        </div><p class="quiet-empty">Transcripts stay on this Mac, in ${esc((info && info.out_dir) || "~/Documents/Transcripts")}.</p>`);
    lib.querySelectorAll("[data-start]").forEach(b => b.onclick = () => $("#" + b.dataset.start).click());
    const ua = $("#unhideAll"); if (ua) ua.onclick = async () => { ua.disabled = true;
      try{ const r = await post("/api/unhide"); toast(r.ok ? "They're back on the list." : "Couldn't bring them back. Try again."); }
      catch(err){ toast("Couldn't reach Matita. Try again."); }
      finally{ ua.disabled = false; loadInfo(); poll(); } };
    return;
  }
  if (!shown.length){ setHTML(lib, term && $("#textHits").hidden ? `<div class="empty"><b>No match for “${esc(term)}”</b><p>Try another word.</p></div>` : ""); return; }
  libOrder = shown.map(j => j.id);
  let lastGroup = null;
  setHTML(lib, shown.map(j => {
    const enter = libSeen && !libSeen.has(j.id) ? " enter" : "";
    const fresh = !j.library && !opened.has(j.id);
    const g = whenOf(j) ? dayGroup(whenOf(j)) : "";
    const head = g !== lastGroup ? `<div class="grp-h">${esc(g)}</div>` : ""; lastGroup = g;
    const bits = [j.duration ? `<span>${MI.len}${esc(clock(j.duration))}</span>` : "",
      j.speaker_count ? `<span title="${esc(plural(j.speaker_count, "speaker"))}">${MI.ppl}${j.speaker_count}</span>` : ""].filter(Boolean).join("");
    return head + `<article class="row lib open${enter}" data-open="${j.id}" tabindex="0" aria-label="Read ${esc(bare(j.name))}">
      <div class="lm"><div class="name">${fresh ? `<span class="newdot" title="New"></span>` : ""}${esc(bare(j.name))}</div>
        ${j.summary ? `<div class="sum${j.summary_ai ? " ai" : ""}">${j.summary_ai ? `<span class="aib" title="AI summary">${MI.ai}</span>` : ""}${esc(j.summary)}</div>` : ""}
        ${j.warning ? `<div class="warnline">${esc(j.warning)}</div>` : ""}</div>
      <div class="lr"><div class="when" title="Recorded ${esc(fmtAt(whenOf(j), {dateStyle:"medium", timeStyle:"short"}))}">${whenOf(j) ? esc(rowWhen(whenOf(j))) : ""}</div><div class="meta">${bits}</div>
        <div class="acts">
          <button data-act="copy" data-id="${j.id}">Copy</button>
          <button data-act="save" data-id="${j.id}">Save…</button>
          <button class="rmore" data-act="rowmenu" data-id="${j.id}" aria-label="More for ${esc(bare(j.name))}" aria-haspopup="menu"><svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><circle cx="5.5" cy="12" r="1.7"/><circle cx="12" cy="12" r="1.7"/><circle cx="18.5" cy="12" r="1.7"/></svg></button>
        </div></div></article>`;
  }).join("") + (hiddenOlder > 0
      ? `<button class="more" id="showOlder">Show ${plural(hiddenOlder, "older transcript")}</button>`
      : (showOlder && !term && older > 0 ? `<button class="more" id="showOlder">Show only the last 7 days</button>` : "")));
  libSeen = new Set([...(libSeen || []), ...done.map(j => j.id)]);
  const so = $("#showOlder"); if (so) so.onclick = () => { showOlder = !showOlder; renderLibrary(Date.now()/1000); };
}
function heroMode(){}
function render(now){ now = now || Date.now()/1000; heroMode(); renderQueue(now); renderLibrary(now); }

/* ---------- actions ---------- */
let fmtFor = null;
function pickFormat(id){
  const j = jobs.find(x => x.id===id); if (!j) return;
  fmtFor = id; $("#fName").textContent = bare(j.name); $("#fSrt").hidden = !j.has_srt;
  $("#fmtDlg").showModal();
}
document.querySelectorAll(".fmt").forEach(b => b.onclick = () => { $("#fmtDlg").close(); store("fmt", b.dataset.fmt); saveAs(fmtFor, b.dataset.fmt); });
$("#fClose").onclick = () => $("#fmtDlg").close();
function dialogHint(){
  return setTimeout(() => toast("A Mac dialog opened. If you don't see it, look behind this window."), 1800);
}
async function saveAs(id, kind){
  const hint = dialogHint();
  const r = await post("/api/save-as", {id, kind}), d = await r.json();
  clearTimeout(hint);
  if (d.cancelled) return;
  if (!r.ok){
    if (d.error === "no dialog"){ location.href = `/api/download?id=${id}&kind=${kind}`; return; }   // fallback: browser download
    toast(d.error || "Couldn't save."); return;
  }
  toast("Saved to " + d.path);
}
async function getText(id){
  const r = await fetch("/api/text?id="+id), d = await r.json();
  if (!r.ok) throw new Error(d.error || "Couldn't open this transcript.");
  return d;
}
async function copyText(id, withTimes){
  try{
    const d = await getText(id);
    const txt = withTimes && d.segments.length ? d.segments.map(s => `[${s.t}] ${s.text}`).join("\n") : d.text;
    await navigator.clipboard.writeText(txt); toast("Copied");
  }catch(e){ toast(e.message); }
}
document.addEventListener("click", async e => {
  if (e.target.closest("[data-stop]")) return;
  const b = e.target.closest("[data-act]");
  if (b){
    e.stopPropagation(); const id = b.dataset.id, act = b.dataset.act;
    if (act==="rowmenu") return openRowMenu(b, id);
    if (act==="copy") return copyText(id, false);
    if (act==="save") return pickFormat(id);
    if (act==="cancel"){ const j = jobs.find(x => x.id === id); if (j){ j.state = "cancelled"; render(); } toast("Cancelled"); await post("/api/cancel",{id}); }
    if (act==="retry"){ const r = await post("/api/retry",{id}); if (!r.ok) toast((await r.json()).error); }
    if (act==="remove"){ const r = await post("/api/remove",{id}); if (!r.ok) return toast((await r.json().catch(() => ({}))).error || "Couldn't remove it."); toast("Removed from the list. The file is still in the folder. Bring it back in Settings."); }
    return poll();
  }
  const row = e.target.closest("[data-open]");
  if (row) openReader(row.dataset.open);
});
document.addEventListener("keydown", e => {
  const row = e.target.closest && e.target.closest("[data-open]");
  if (row && e.target===row && (e.key==="Enter"||e.key===" ")){ e.preventDefault(); openReader(row.dataset.open); }
  if ((e.metaKey||e.ctrlKey) && e.key==="f" && $("#reader").open){ e.preventDefault(); $("#rFind").focus(); }
});
$("#pauseAll").onclick = async () => {
  $("#pauseAll").disabled = true;
  try{
    const r = await (await post("/api/pause", {on: !paused})).json();
    paused = !!r.paused; toast(paused ? "Paused. Your Mac can sleep; Resume picks up where it stopped." : "Resumed.");
  }catch(e){ toast("Couldn't reach Matita. Try again."); }
  finally{ $("#pauseAll").disabled = false; poll(); }
};
$("#cancelAll").onclick = async () => { await post("/api/cancel",{}); toast("Cancelled everything in progress"); poll(); };
$("#clearList").onclick = async () => {
  if (!confirm("Take every transcript off this list? The files stay in your folder, and you can bring them back in Settings.")) return;
  await post("/api/remove",{done_only: true}); toast("List cleared. Your files are still in the folder."); poll(); };
$("#zipAll").onclick = () => { location.href = "/api/download-all"; };
/* per-row ⋯ menu */
let rowMenuId = null;
function openRowMenu(btn, id){
  const m = $("#rowMenu");
  if (!m.hidden && rowMenuId === id){ closeRowMenu(); return; }
  rowMenuId = id; m.hidden = false;
  const r = btn.getBoundingClientRect(), w = m.offsetWidth, h = m.offsetHeight;
  m.style.left = Math.max(8, Math.min(innerWidth - w - 8, r.right - w)) + "px";
  m.style.top = (r.bottom + h + 8 > innerHeight ? r.top - h - 6 : r.bottom + 6) + "px";
  btn.closest(".row").classList.add("menu-open");
  m.querySelector("button").focus();
}
function closeRowMenu(){ $("#rowMenu").hidden = true; rowMenuId = null; document.querySelectorAll(".row.menu-open").forEach(r => r.classList.remove("menu-open")); }
document.addEventListener("click", e => { if (!$("#rowMenu").hidden && !e.target.closest("#rowMenu, [data-act=rowmenu]")) closeRowMenu(); });
window.addEventListener("scroll", () => { if (!$("#rowMenu").hidden) closeRowMenu(); }, {passive: true});
document.addEventListener("keydown", e => {
  if ($("#rowMenu").hidden) return;
  if (e.key === "Escape"){ const id = rowMenuId; closeRowMenu(); const b = document.querySelector(`[data-act=rowmenu][data-id="${id}"]`); if (b) b.focus(); }
  if (e.key === "ArrowDown" || e.key === "ArrowUp"){
    const items = [...$("#rowMenu").querySelectorAll("button")]; e.preventDefault();
    const i = items.indexOf(document.activeElement); items[(i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length].focus();
  }
});
$("#rowMenu").addEventListener("click", async e => {
  const b = e.target.closest("[data-rm]"); if (!b) return;
  const id = rowMenuId; closeRowMenu(); if (!id) return;
  const j = jobs.find(x => x.id === id), name = j ? bare(j.name) : "";
  if (b.dataset.rm === "rename") return openReader(id, {rename: true});
  if (b.dataset.rm === "remove"){
    jobs = jobs.filter(x => x.id !== id); render();                 // show it gone at once
    const r = await post("/api/remove", {id}); if (!r.ok) return toast((await r.json().catch(() => ({}))).error || "Couldn't remove it.");
    toast("Removed from the list. The file is still in the folder.", {label: "Undo", fn: async () => { await post("/api/unhide-one", {id}); poll(); }});
    return poll();
  }
  if (b.dataset.rm === "trash") return trashIt(id, name);
});
async function trashIt(id, name){
  const r = await post("/api/trash", {id});
  if (!r.ok) return toast((await r.json().catch(() => ({}))).error || "Couldn't move it to the Trash.");
  if ($("#reader").open && current && current.id === id) $("#reader").close();
  jobs = jobs.filter(x => x.id !== id); render();
  toast(`Moved “${name}” to the Trash.`, {label: "Undo", fn: async () => {
    const d = await (await post("/api/trash/undo", {id})).json().catch(() => ({}));
    if (!d.ok) toast("It's already in the Trash. Put it back from the Trash in Finder."); poll(); }});
  poll();
}
function listMenu(open){
  const m = $("#listMenu"), b = $("#listMore"); m.hidden = !open; b.setAttribute("aria-expanded", open);
  if (open){ const f = m.querySelector("button:not(:disabled)"); if (f) f.focus(); }
}
$("#listMore").onclick = e => { e.stopPropagation(); listMenu($("#listMenu").hidden); };
$("#listMenu").addEventListener("click", e => { if (e.target.closest("button")) listMenu(false); });
document.addEventListener("click", e => { if (!$("#listMenu").hidden && !e.target.closest(".mwrap")) listMenu(false); });
document.addEventListener("keydown", e => {
  if ($("#listMenu").hidden) return;
  if (e.key === "Escape"){ listMenu(false); $("#listMore").focus(); }
  if (e.key === "ArrowDown" || e.key === "ArrowUp"){
    const items = [...$("#listMenu").querySelectorAll("button:not(:disabled)")]; if (!items.length) return;
    e.preventDefault(); const i = items.indexOf(document.activeElement);
    items[(i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length].focus();
  }
});
$("#filter").addEventListener("input", () => renderLibrary(Date.now()/1000));

/* ---------- voice memos ---------- */
let vm = null, vmSel = new Set();
function currentOpts(){
  return {lang: $("#lang").value, quality: $("#quality").value, prompt: $("#prompt").value,
          speakers: $("#speakers").disabled ? "off" : $("#speakers").value};
}
function fmtDate(ts){
  const d = new Date(ts*1000);
  return d.toLocaleDateString([], {year: d.getFullYear() === new Date().getFullYear() ? undefined : "numeric", month:"short", day:"numeric"})
    + " " + d.toLocaleTimeString([], {hour:"2-digit", minute:"2-digit"});
}
async function loadVM(){
  $("#vmBody").innerHTML = `<p class="hint" style="margin:18px 0">Reading your Voice Memos…</p>`;
  try{ vm = await (await fetch("/api/vm")).json(); }catch(e){ vm = {status:"error", message:"Matita isn't running."}; }
  drawVM();
}
function drawVM(){
  const body = $("#vmBody"), ok = vm.status === "ok";
  $("#vmTools").hidden = !ok; $("#vmFoot").hidden = !ok;
  body.classList.toggle("plain", !ok);
  if (vm.status === "needs_access"){
    const app = vm.app || {}, py = app.kind === "python" || app.kind === "app";
    body.innerHTML = `<div class="perm">
      <p style="margin:0;font-size:15px;line-height:1.5">macOS keeps Voice Memos private, so Matita needs <b>Full Disk Access</b> to read them. It only reads your recordings and never changes or uploads them.</p>
      <ol>
        <li><div><b>Open Full Disk Access settings</b><p>System Settings opens on the right page.</p></div><button class="btn" id="vmSettings">Open</button></li>
        <li><div><b>Add ${app.kind === "python" ? "Python" : esc(app.name)}</b><p>${py
          ? `Click Show in Finder, then drag <code>${esc(app.name)}</code> into the Full Disk Access list and turn it on.`
          : `Turn on <code>${esc(app.name)}</code> in the list, or click + to add it.`}</p></div>
          ${py ? `<button class="btn" id="vmReveal">Show in Finder</button>` : "<span></span>"}</li>
        ${info && info.app ? `<li><div><b>Quit Matita and open it again</b><p>macOS applies the permission after Matita restarts. Press ⌘Q, then open Matita from Applications.</p></div><span></span></li>`
          : `<li><div><b>Restart Matita</b><p>macOS applies the permission after a restart. It takes a few seconds.</p></div><button class="btn go" id="vmRestart">Restart</button></li>`}
      </ol>${vm.retried ? `<p class="hint">Still blocked? Make sure the switch next to ${app.kind === "python" ? "Python" : esc(app.name)} is on in Full Disk Access.${app.name === "Matita Service" ? " (The Matita app on your Desktop doesn't need it.)" : ""} If you just reinstalled Matita, remove it from the list with the − button, add it again, then restart.</p>` : ""}</div>`;
    $("#vmSettings").onclick = () => post("/api/vm/open-settings");
    if ($("#vmReveal")) $("#vmReveal").onclick = () => post("/api/vm/reveal-app");
    if ($("#vmRestart")) $("#vmRestart").onclick = restartForVM;
    return;
  }
  if (vm.status === "not_found"){
    body.innerHTML = `<div class="empty" style="margin-top:6px"><b>No Voice Memos found on this Mac</b>
      <p>Turn on iCloud for Voice Memos on your iPhone and Mac (Settings → your name → iCloud → Voice Memos), then open the Voice Memos app on this Mac once so your recordings sync.</p></div>`;
    return;
  }
  if (!ok){ body.innerHTML = `<div class="empty"><b>Couldn't read Voice Memos</b><p>${esc(vm.message || "")}</p></div>`; return; }

  $("#vmMeta").textContent = plural(vm.memos.length,"recording");
  $("#vmAuto").checked = !!(vm.auto && vm.auto.enabled);
  const term = $("#vmFind").value.trim().toLowerCase();
  const list = vm.memos.filter(m => !term || m.title.toLowerCase().includes(term));
  if (!list.length){ body.innerHTML = `<div class="empty" style="margin-top:14px"><b>${vm.memos.length ? "No match" : "No recordings yet"}</b><p>${vm.memos.length ? "Try a different word." : "Record something in Voice Memos and it will show up here."}</p></div>`; updateVMGo(); return; }
  body.innerHTML = `<div class="vm-list">` + list.map(m => {
    const can = m.local && !m.queued;
    let badge = "";
    if (m.queued) badge = `<span class="badge">In the queue</span>`;
    else if (!m.local) badge = `<span class="badge">In iCloud only</span>`;
    else if (m.done) badge = `<span class="badge ok">✓ Transcribed</span>`;
    else if (m.failed) badge = `<span class="badge fail">Didn't finish</span>`;
    const meta = [fmtDate(m.date), m.duration ? clock(m.duration) : ""].filter(Boolean).join(" · ");
    return `<label class="vm ${can ? "" : "off"}" title="${!m.local ? "Play it once in the Voice Memos app to download it from iCloud." : ""}">
      <input type="checkbox" data-uid="${esc(m.uid)}" ${can ? "" : "disabled"} ${vmSel.has(m.uid) ? "checked" : ""}>
      <div><b>${esc(m.title)}</b><span>${esc(meta)}</span></div>${badge}</label>`;
  }).join("") + `</div>`;
  updateVMGo();
}
function updateVMGo(){
  const n = vmSel.size, b = $("#vmGo");
  b.disabled = !n; b.textContent = n ? `Transcribe ${plural(n,"recording")}` : "Transcribe";
}
async function restartForVM(){
  const r = await post("/api/restart");
  if (!r.ok){ toast((await r.json()).error); return; }
  $("#vmBody").innerHTML = `<p class="hint" style="margin:18px 0">Restarting Matita…</p>`;
  const t0 = Date.now();
  const again = async () => {
    try{ await fetch("/api/info"); await loadVM(); if (vm.status === "needs_access"){ vm.retried = true; drawVM(); } return; }catch(e){}
    if (Date.now() - t0 < 30000) setTimeout(again, 1000);
  };
  setTimeout(again, 2500);
}
$("#vmOpen").onclick = () => { vmSel = new Set(); $("#vmFind").value = ""; $("#vmDlg").showModal(); loadVM(); };
$("#vmClose").onclick = () => $("#vmDlg").close();
$("#vmFind").addEventListener("input", drawVM);
$("#vmBody").addEventListener("change", e => {
  const c = e.target.closest("input[data-uid]"); if (!c) return;
  c.checked ? vmSel.add(c.dataset.uid) : vmSel.delete(c.dataset.uid); updateVMGo();
});
$("#vmPickNew").onclick = () => { (vm.memos||[]).forEach(m => { if (m.local && !m.done && !m.queued) vmSel.add(m.uid); }); drawVM(); };
$("#vmPickNone").onclick = () => { vmSel.clear(); drawVM(); };
$("#vmGo").onclick = async () => {
  $("#vmGo").disabled = true;
  const r = await post("/api/vm/import", {uids: [...vmSel], opts: currentOpts()}), d = await r.json();
  if (r.status === 403){ vm = {status:"needs_access", app:(vm||{}).app}; return loadVM(); }
  $("#vmDlg").close();
  toast(`Added ${plural(d.added,"recording")} from Voice Memos` + (d.errors && d.errors.length ? `. ${d.errors[0]}` : ""));
  poll(); setTimeout(() => $("#queuePanel").scrollIntoView({behavior:"smooth", block:"start"}), 400);
};
$("#vmAuto").onchange = async e => {
  await post("/api/vm/auto", {enabled: e.target.checked, opts: currentOpts()});
  toast(e.target.checked ? "New Voice Memos will be transcribed automatically, using the current settings." : "Automatic transcription is off.");
};

/* ---------- reader ---------- */
let doc = null;
let hits = 0;
function hl(text){   // escape text, marking matches of the search term (works on raw text, not on escaped HTML)
  const term = $("#rFind").value.trim();
  if (!term) return esc(text);
  const low = text.toLowerCase(), t = term.toLowerCase();
  let out = "", i = 0, k;
  while ((k = low.indexOf(t, i)) !== -1){
    out += esc(text.slice(i, k)) + "<mark>" + esc(text.slice(k, k + t.length)) + "</mark>";
    i = k + t.length; hits++;
  }
  return out + esc(text.slice(i));
}
function initials(name, n){
  const m = /^Speaker (\d+)$/.exec(name || "");
  if (m) return "S" + m[1];
  const w = String(name || "").trim().split(/\s+/).filter(Boolean);
  return ((w[0] || "?")[0] + (w.length > 1 ? w[w.length - 1][0] : "")).toUpperCase();
}
function markUnsure(text, from, to){
  // Underline words Whisper wasn't sure about (only when not searching, so the two marks don't clash).
  if ($("#rFind").value.trim() || !doc.unsure || !doc.unsure.length || from == null) return hl(text);
  const lows = doc.unsure.filter(w => w[1] >= from - 0.3 && (to == null || w[1] < to - 0.3));
  const cjk = current && ["ja","zh"].includes(current.lang);
  const rx = w => new RegExp((cjk ? "" : "(?<![\\p{L}\\p{N}])") + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + (cjk ? "" : "(?![\\p{L}\\p{N}])"), "gu");
  let out = "", i = 0;
  for (const [w, t, p] of lows){
    const re = rx(w); re.lastIndex = i; const m = re.exec(text); if (!m) continue;
    const k = m.index;
    out += esc(text.slice(i, k)) + `<u class="unsure" data-w="${t}" title="Possibly misheard (${Math.round(p*100)}% sure). Click to hear it.">${esc(w)}</u>`;
    i = k + w.length;
  }
  return out + esc(text.slice(i));
}
function editDirty(){ return !!(editing && editing.ta.value.trim() !== editing.item.text); }
function okToLeaveEdit(){ return !editDirty() || confirm("Discard the change to this line?"); }
function drawReader(){
  editing = null;
  if (!doc) return;
  let html; hits = 0;
  if (view === "ai") html = aiViewHTML();
  else if (doc.turns && view === "spk") html = speakersHTML();
  else if (doc.turns){
    const starts = doc.turns.map(t => t.s);
    html = doc.turns.map((t, i) => `<div class="turn"${t.s!=null ? ` data-s="${t.s}"` : ""} style="--c:${SPK[(t.speaker-1)%SPK.length]}"><div class="who">
      <span class="av" aria-hidden="true">${esc(initials(t.name, t.speaker))}</span><button class="nm" data-spk="${t.speaker}" title="Rename">${esc(t.name)}</button><span class="tt">${esc(t.t.replace(/^00:/,""))}</span></div>
      <div class="tx">${markUnsure(t.text, t.s, starts[i+1])}</div><button class="ed" data-ed="${i}">Edit</button></div>`).join("");
  } else if (doc.segments.length){
    html = doc.segments.map((s, i) => `<div class="line" data-s="${s.s||0}"><span class="ts">${esc(s.t.replace(/^00:/,""))}</span><span class="tx">${markUnsure(s.text, s.s, (doc.segments[i+1]||{}).s)}</span><button class="ed" data-ed="${i}">Edit</button></div>`).join("");
  } else {
    const paras = doc.text.split(/\n+/).map(p=>p.trim()).filter(Boolean);
    html = `<div class="text">${paras.map(p => `<p>${hl(p)}</p>`).join("") || "<p>This transcript is empty.</p>"}</div>`;
  }
  const term = $("#rFind").value.trim();
  $("#rBody").innerHTML = html;
  const canPlay = !!doc.audio && view !== "ai";
  $("#rBody").classList.toggle("playable", canPlay);
  $("#rPlayer").hidden = !doc.audio;
  $("#pPlay").disabled = !doc.audio;
  const nUnsure = (doc.unsure||[]).length;
  if (!doc.audio) $("#pHint").textContent = "Playback works for recordings transcribed with version 1.5 or later.";
  else if (audio.paused && !audio.currentTime) $("#pHint").textContent = nUnsure
    ? `Click a line to hear it. ${plural(nUnsure, "dotted word")} may be misheard.` : "Click a line to hear it.";
  lastRow = null; markPlaying();
  $("#rHits").textContent = term ? (hits ? plural(hits,"match","matches") : "No matches") : "";
  const first = $("#rBody mark"); if (first) first.scrollIntoView({block:"center"});
}
/* everything each person said, one person at a time, with how much of the talking was theirs */
function speakersHTML(){
  const T = doc.turns, end = wave.duration || (current && current.duration) || 0;
  const by = new Map();
  T.forEach((t, i) => {
    const next = i + 1 < T.length ? T[i + 1].s : end;
    const len = t.s != null && next != null ? Math.max(0, next - t.s) : 0;
    if (!by.has(t.speaker)) by.set(t.speaker, {speaker: t.speaker, name: t.name, idx: [], secs: 0});
    const g = by.get(t.speaker); g.idx.push(i); g.secs += len;
  });
  const groups = [...by.values()].sort((a, b) => b.secs - a.secs || a.speaker - b.speaker);
  const total = groups.reduce((n, g) => n + g.secs, 0) || 1;
  const col = g => SPK[(g.speaker - 1) % SPK.length];
  const pct = g => Math.round(100 * g.secs / total);
  const bar = `<div class="sbar" aria-hidden="true">${groups.map(g => `<span style="flex:${g.secs || 1};background:${col(g)}"></span>`).join("")}</div>`;
  const starts = T.map(t => t.s);
  return `<div class="spkview">${bar}` + groups.map(g => `<section class="sgrp" style="--c:${col(g)}">
    <div class="sgh"><span class="av" aria-hidden="true">${esc(initials(g.name, g.speaker))}</span><button class="nm" data-spk="${g.speaker}" title="Rename">${esc(g.name)}</button>
      <span class="sgm">${pct(g)}% of the talking · ${plural(g.idx.length, "turn")}${g.secs ? " · " + clock(g.secs) : ""}</span></div>
    ${g.idx.map(i => { const t = T[i]; return `<div class="line"${t.s != null ? ` data-s="${t.s}"` : ""}><span class="ts">${esc((t.t || "").replace(/^00:/, ""))}</span><span class="tx">${markUnsure(t.text, t.s, starts[i+1])}</span><button class="ed" data-ed="${i}">Edit</button></div>`; }).join("")}
  </section>`).join("") + `</div>`;
}
let readerSeq = 0;
async function openReader(id, opts){
  const j = jobs.find(x => x.id===id); if (!j) return;
  if ($("#reader").open && !okToLeaveEdit()) return;
  const seq = ++readerSeq; let d;
  try{ d = await getText(id); }catch(e){ return toast(e.message); }
  if (seq !== readerSeq) return;                              // a newer one was opened meanwhile
  doc = d;
  current = j; opened.add(id);
  audio.pause();
  const k = libOrder.indexOf(id);
  $("#rPrev").disabled = k <= 0; $("#rNext").disabled = k < 0 || k >= libOrder.length - 1;
  if (doc.audio){ audio.src = `/api/audio?id=${encodeURIComponent(id)}&v=${j.finished||""}`; }
  else { audio.removeAttribute("src"); audio.load(); }
  $("#pTime").textContent = "0:00"; $("#pDur").textContent = j.duration ? mmss(j.duration) : "0:00"; playIcon();
  wave = {peaks: [], duration: j.duration || 0, id};
  if (doc.audio) fetch("/api/peaks?id=" + encodeURIComponent(id)).then(r => r.json()).then(d => {
    if (!current || current.id !== id) return;
    wave.peaks = d.peaks || []; if (d.duration) wave.duration = d.duration;
    $("#pDur").textContent = mmss(wave.duration); drawWave();
  }).catch(() => {});
  requestAnimationFrame(drawWave);
  aiDoc = null; aiShow = null; $("#aiBox").hidden = true; refreshAIButtons(); loadAIDoc();
  $("#rName").textContent = bare(j.name);
  $("#rMeta").textContent = [whenOf(j) ? "Recorded " + when(whenOf(j)) : "", j.duration ? clock(j.duration)+" of audio" : "", j.words!=null ? plural(j.words,"word") : ""].filter(Boolean).join(" · ");
  // 4. the one-line summary only helps when it says something the opening lines don't
  const norm = t => String(t || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, " ").trim();
  const ns = norm(doc.summary), nt = norm(doc.text);
  const dup = !!ns && (nt.length < ns.length * 3 || nt.slice(0, ns.length + 40).includes(ns.slice(0, 120)));   // short, or just the opening
  $("#rSum").textContent = doc.summary || ""; $("#rSum").hidden = !doc.summary || dup;
  $("#rKeys").innerHTML = (doc.keywords||[]).map(k => `<span class="chip">${esc(k)}</span>`).join("");
  const many = !!(doc.turns && new Set(doc.turns.map(t => t.speaker)).size > 1);
  $("#vSpk").hidden = !many;
  $("#rFind").value = (opts && opts.term) || "";
  setView(opts && (opts.at != null || opts.term) ? "time" : aiOn() ? "ai" : (load("view")==="spk" && many ? "spk" : "time"));
  $("#reader").showModal();
  if (opts && opts.rename) startRename();
  if (opts && opts.at != null){
    const rows = [...document.querySelectorAll("#rBody [data-s]")];
    const row = rows.filter(r => +r.dataset.s <= opts.at + 0.05).pop() || rows[0];
    if (row){ row.scrollIntoView({block:"center"}); row.classList.add("flash"); }
  } else if (!(opts && opts.term)) $("#rBody").scrollTop = 0;
}
function setView(v){ view = v; if (v !== "ai") store("view", v);
  $("#vSpk").setAttribute("aria-pressed", v==="spk"); $("#vTime").setAttribute("aria-pressed", v==="time"); $("#vAI").setAttribute("aria-pressed", v==="ai");
  drawReader(); if (v === "ai") ensureSummary(); }
/* playback in the timestamps view */
const audio = new Audio(); audio.preload = "none";
let lastRow = null;
const PLAY = `<svg width="12" height="14" viewBox="0 0 12 14"><path d="M1 1.2v11.6c0 .6.7 1 1.2.6l9-5.8c.5-.3.5-.9 0-1.2l-9-5.8C1.7.2 1 .6 1 1.2z" fill="currentColor"/></svg>`;
const PAUSE = `<svg width="12" height="14" viewBox="0 0 12 14" fill="currentColor"><rect x="1" y="1" width="3.6" height="12" rx="1"/><rect x="7.4" y="1" width="3.6" height="12" rx="1"/></svg>`;
function playIcon(){ $("#pPlay").innerHTML = audio.paused ? PLAY : PAUSE; $("#pPlay").setAttribute("aria-label", audio.paused ? "Play" : "Pause"); }
function mmss(t){ t = Math.max(0, Math.floor(t||0)); const h = Math.floor(t/3600), m = Math.floor(t%3600/60), s = t%60;
  return (h ? h+":"+String(m).padStart(2,"0") : m) + ":" + String(s).padStart(2,"0"); }
function playFrom(sec){
  const go = () => { audio.currentTime = Math.max(0, sec - 0.1); audio.play().catch(() => toast("Couldn't play the audio.")); };
  if (audio.readyState >= 1) go(); else { audio.addEventListener("loadedmetadata", go, {once:true}); audio.load(); }
}
function markPlaying(){
  if (!$("#rBody").classList.contains("playable")) return;
  const rows = document.querySelectorAll("#rBody [data-s]"), t = audio.currentTime + 0.05;
  let row = null;
  if (audio.currentTime > 0) for (const r of rows){ if (+r.dataset.s <= t) row = r; else break; }
  if (row === lastRow) return;
  if (lastRow) lastRow.classList.remove("playing");
  if (row){ row.classList.add("playing"); if (!audio.paused) row.scrollIntoView({block:"nearest", behavior:"smooth"}); }
  lastRow = row;
}
audio.addEventListener("timeupdate", () => { $("#pTime").textContent = mmss(audio.currentTime); markPlaying(); drawWave();
  const w = $("#pWave"), dur = wave.duration || audio.duration || 0;
  w.setAttribute("aria-valuemax", Math.round(dur)); w.setAttribute("aria-valuenow", Math.round(audio.currentTime));
  w.setAttribute("aria-valuetext", mmss(audio.currentTime) + " of " + mmss(dur)); });

/* waveform in the player bar, coloured by who is speaking */
let wave = {peaks: [], duration: 0, id: null};
const SPK = ["#4f9a5e", "#4a78c2", "#c0803e", "#8e66b8", "#c05d7a"];
function speakerAt(t){
  if (!doc || !doc.turns) return 0;
  let k = 0;
  for (let i = 0; i < doc.turns.length; i++){ if ((doc.turns[i].s || 0) <= t) k = doc.turns[i].speaker; else break; }
  return k;
}
function drawWave(){
  const c = $("#pCanvas"); if (!c || $("#rPlayer").hidden) return;
  const box = c.parentElement.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  if (!box.width) return;
  if (c.width !== Math.round(box.width * dpr)){ c.width = Math.round(box.width * dpr); c.height = Math.round(box.height * dpr); }
  const g = c.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); g.clearRect(0, 0, box.width, box.height);
  const n = Math.max(1, Math.floor(box.width / 3)), dur = wave.duration || audio.duration || 0;
  const pos = dur ? Math.min(1, audio.currentTime / dur) : 0;
  const neutral = getComputedStyle(document.documentElement).getPropertyValue("--soft").trim() || "#888";
  for (let i = 0; i < n; i++){
    let v = 0.08;
    if (wave.peaks.length){
      const a = Math.floor(i * wave.peaks.length / n), b = Math.max(a + 1, Math.floor((i + 1) * wave.peaks.length / n));
      for (let k = a; k < b; k++) v = Math.max(v, wave.peaks[k] || 0);
    }
    const h = Math.max(2, Math.pow(v, 0.7) * box.height * 0.92), x = i * 3, y = (box.height - h) / 2;
    const spk = speakerAt(dur * (i + 0.5) / n);
    g.globalAlpha = (i + 0.5) / n <= pos ? 1 : 0.32;
    g.fillStyle = spk ? SPK[(spk - 1) % SPK.length] : neutral;
    g.beginPath(); g.roundRect ? g.roundRect(x, y, 2, h, 1) : g.rect(x, y, 2, h); g.fill();
  }
  g.globalAlpha = 1;
  if (dur){ g.fillStyle = getComputedStyle(document.documentElement).getPropertyValue("--ink").trim() || "#000";
    g.fillRect(Math.round(pos * box.width) - 1, 0, 2, box.height); }
}
function seekAt(e){
  const box = $("#pWave").getBoundingClientRect(), dur = wave.duration || audio.duration || 0;
  if (!dur) return;
  playFrom(Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) * dur + 0.1);
}
let scrubbing = false;
$("#pWave").addEventListener("pointerdown", e => { scrubbing = true; $("#pWave").setPointerCapture(e.pointerId); seekAt(e); });
$("#pWave").addEventListener("pointermove", e => { if (scrubbing){ const box = $("#pWave").getBoundingClientRect(), dur = wave.duration || audio.duration || 0;
  if (dur && audio.readyState >= 1) audio.currentTime = Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) * dur; } });
$("#pWave").addEventListener("pointerup", () => { scrubbing = false; });
window.addEventListener("resize", () => drawWave());
try{ matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => requestAnimationFrame(drawWave)); }catch(e){}
["pointercancel","lostpointercapture"].forEach(ev => $("#pWave").addEventListener(ev, () => { scrubbing = false; }));
$("#pWave").addEventListener("keydown", e => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" || !doc || !doc.audio) return;
  e.preventDefault(); e.stopPropagation();
  if (audio.readyState >= 1) audio.currentTime = Math.max(0, audio.currentTime + (e.key === "ArrowLeft" ? -5 : 5)); else playFrom(0);
});
["play","pause","ended"].forEach(ev => audio.addEventListener(ev, () => { playIcon(); if (ev==="play") $("#pHint").textContent = ""; }));
audio.addEventListener("error", () => { if (audio.getAttribute("src")) toast("Couldn't load the audio for this transcript."); });
$("#pPlay").onclick = () => { if (!doc || !doc.audio) return;
  if (audio.paused && view !== "time" && !$("#vTime").disabled && !editing){ setView("time"); toast("Following along in Timestamps."); }
  if (audio.paused){ if (audio.readyState >= 1) audio.play().catch(()=>{}); else playFrom(0); } else audio.pause(); };
$("#rBody").addEventListener("click", e => {
  if (!doc || !doc.audio || view!=="time") return;
  if (e.target.closest(".nm, input, textarea, button, a, .edbar")) return;
  if (String(window.getSelection() || "")) return;          // selecting text, not playing
  const u = e.target.closest("u.unsure"); if (u){ playFrom(Math.max(0, +u.dataset.w - 1.2)); return; }
  const row = e.target.closest("[data-s]"); if (row) playFrom(+row.dataset.s);
});
$("#reader").addEventListener("close", () => audio.pause());
/* editing a line */
let editing = null;
$("#rBody").addEventListener("click", e => {
  const b = e.target.closest(".ed"); if (!b || !doc || editing) return;
  const i = +b.dataset.ed, row = b.closest("[data-s]"), item = doc.turns ? doc.turns[i] : doc.segments[i];
  if (!item || !row) return;
  audio.pause();
  const tx = row.querySelector(".tx"), ta = document.createElement("textarea");
  ta.className = "edit"; ta.value = item.text; ta.setAttribute("aria-label", "Edit this line");
  const bar = document.createElement("div"); bar.className = "edbar";
  bar.innerHTML = `<button class="btn go" style="padding:5px 14px;font-size:13px">Save</button><button class="link muted">Cancel</button><span>Enter saves · Esc cancels${doc.audio ? " · ⌥Space plays this line" : ""}</span>`;
  tx.replaceWith(ta); b.hidden = true; row.appendChild(bar);
  editing = {row, i, ta, item};
  ta.focus(); ta.setSelectionRange(ta.value.length, ta.value.length);
  const done = async save => {
    if (!editing || editing.saving) return;
    if (save && ta.value.trim() && ta.value.trim() !== item.text){
      editing.saving = true; bar.querySelector(".btn").disabled = true;
      let r, d = {};
      try{ r = await post("/api/edit", {id: current.id, s: item.s, old: item.text, new: ta.value}); d = await r.json().catch(() => ({})); }catch(err){}
      if (!r || !r.ok){ toast(d.error || "Couldn't save the change."); if (editing) editing.saving = false; bar.querySelector(".btn").disabled = false; return; }
      toast("Saved.");
      try{ doc = await getText(current.id); }catch(err){}
    }
    editing = null; drawReader();
    const again = document.querySelector(`#rBody .ed[data-ed="${i}"]`); if (again && again.closest("[data-s]")) again.closest("[data-s]").classList.add("flash");
  };
  bar.querySelector(".btn").onclick = () => done(true);
  bar.querySelector(".link").onclick = () => done(false);
  ta.addEventListener("keydown", ev => {
    if (ev.isComposing || ev.keyCode === 229) return;          // Korean/Japanese input still composing
    if (ev.key === "Enter" && !ev.shiftKey){ ev.preventDefault(); done(true); }
    else if (ev.key === "Escape"){ ev.preventDefault(); ev.stopPropagation(); done(false); }
    else if (ev.code === "Space" && ev.altKey && doc.audio){ ev.preventDefault(); playFrom(item.s || 0); }
  });
});
$("#reader").addEventListener("cancel", e => { if (editing || renaming) e.preventDefault(); });   // Esc while editing
/* previous / next transcript, in the order of the list */
function navReader(step){
  if (!current) return;
  const k = libOrder.indexOf(current.id), id = libOrder[k + step];
  if (k < 0 || !id || !okToLeaveEdit()) return;
  openReader(id);
}
$("#rPrev").onclick = () => navReader(-1);
$("#rNext").onclick = () => navReader(1);
document.addEventListener("keydown", e => {
  if (!$("#reader").open || !(e.metaKey || e.ctrlKey) || (e.key !== "[" && e.key !== "]")) return;
  if (e.target.closest && e.target.closest("input, textarea")) return;
  e.preventDefault(); navReader(e.key === "[" ? -1 : 1);
});
/* renaming the transcript: click its title */
let renaming = false;
async function renameTo(id, v){
  const r = await post("/api/rename", {id, name: v}); const d = await r.json().catch(() => ({}));
  if (!r.ok){ toast(d.error || "Couldn't rename."); return null; }
  for (const x of [jobs.find(y => y.id === id), current && current.id === id ? current : null]) if (x) x.name = d.name + ".txt";
  render(); toast("Renamed."); poll(); return d.name;
}
function startRename(){
  if (!current || renaming) return;
  const h = $("#rName"), old = bare(current.name), id = current.id;
  const inp = document.createElement("input"); inp.type = "text"; inp.className = "rn"; inp.value = old; inp.setAttribute("aria-label", "Transcript name");
  h.textContent = ""; h.appendChild(inp); inp.focus(); inp.select(); renaming = true;
  let fin = false;
  const done = async save => {
    if (fin) return; fin = true; renaming = false;
    const v = inp.value.trim(); h.textContent = old;
    if (!save || !v || v === old) return;
    const name = await renameTo(id, v);
    if (name && current && current.id === id){ h.textContent = name; drawAIBox(); }
  };
  inp.addEventListener("keydown", ev => {
    if (ev.isComposing || ev.keyCode === 229) return;
    if (ev.key === "Enter"){ ev.preventDefault(); done(true); }
    else if (ev.key === "Escape"){ ev.preventDefault(); ev.stopPropagation(); done(false); }
  });
  inp.addEventListener("blur", () => done(true));
}
$("#rName").addEventListener("click", e => { if (!e.target.closest("input")) startRename(); });

/* keyboard: space plays/pauses, arrows skip 5 s */
document.addEventListener("keydown", e => {
  if (!$("#reader").open || !doc || !doc.audio || editing) return;
  if (e.target.closest && e.target.closest("input, textarea, select, button")) return;   // a focused button handles Space itself
  if (e.key === " "){ e.preventDefault(); $("#pPlay").click(); }
  else if (e.key === "ArrowLeft" || e.key === "ArrowRight"){
    e.preventDefault();
    const go = () => { audio.currentTime = Math.max(0, audio.currentTime + (e.key === "ArrowLeft" ? -5 : 5)); };
    if (audio.readyState >= 1) go(); else playFrom(0);
  }
});
const RATES = [1, 1.25, 1.5, 2];
function setRate(r){ audio.playbackRate = r; audio.defaultPlaybackRate = r; $("#pRate").textContent = (r === 1 ? "1" : r) + "×"; store("rate", r); }
$("#pRate").onclick = () => setRate(RATES[(RATES.indexOf(audio.playbackRate) + 1) % RATES.length] || 1);
setRate(RATES.includes(+load("rate")) ? +load("rate") : 1);
audio.addEventListener("loadedmetadata", () => { audio.playbackRate = audio.defaultPlaybackRate; });

/* search inside every transcript */
let searchT = null, searchSeq = 0;
function markTerm(text, term){
  const low = text.toLowerCase(), t = term.toLowerCase(); let out = "", i = 0, k;
  while (t && (k = low.indexOf(t, i)) !== -1){ out += esc(text.slice(i, k)) + "<mark>" + esc(text.slice(k, k + t.length)) + "</mark>"; i = k + t.length; }
  return out + esc(text.slice(i));
}
async function searchText(){
  const term = $("#filter").value.trim(), box = $("#textHits"), seq = ++searchSeq;
  if (term.length < 2){ box.hidden = true; box.innerHTML = ""; return; }
  let res;
  try{ res = (await (await fetch("/api/search?q=" + encodeURIComponent(term))).json()).results || []; }catch(e){ return; }
  if (seq !== searchSeq) return;                                    // a newer search is on its way
  if (!res.length){ box.hidden = true; box.innerHTML = ""; renderLibrary(Date.now()/1000); return; }
  const total = res.reduce((a, r) => a + r.count, 0);
  box.innerHTML = `<div class="th">“${esc(term)}” appears ${plural(total, "time")} in ${plural(res.length, "transcript")}</div>` +
    res.map(r => `<div class="thit"><div class="tn">${esc(bare(r.name))} <span>· ${plural(r.count, "match", "matches")}</span></div>` +
      r.hits.map(h => `<button data-id="${esc(r.id)}" data-s="${h.s == null ? "" : h.s}"><span class="tt2">${esc(h.t)}</span><span class="tx2">${markTerm(h.text, term)}</span></button>`).join("") +
      (r.count > r.hits.length ? `<button data-id="${esc(r.id)}" data-s=""><span></span><span class="muted">Show all ${r.count} in this transcript…</span></button>` : "") + `</div>`).join("");
  box.hidden = false; renderLibrary(Date.now()/1000);
}
$("#filter").addEventListener("input", () => { clearTimeout(searchT); searchT = setTimeout(searchText, 250); });
$("#textHits").addEventListener("click", e => {
  const b = e.target.closest("button[data-id]"); if (!b) return;
  openReader(b.dataset.id, {at: b.dataset.s === "" ? null : +b.dataset.s, term: $("#filter").value.trim()});
});

$("#vSpk").onclick = () => okToLeaveEdit() && setView("spk");
$("#vTime").onclick = () => okToLeaveEdit() && setView("time");
$("#vAI").onclick = () => okToLeaveEdit() && setView("ai");
$("#rFind").addEventListener("input", () => { if (editing) return; if (view === "ai" && $("#rFind").value.trim()) setView("time"); else drawReader(); });
$("#rClose").onclick = () => okToLeaveEdit() && $("#reader").close();
$("#reader").addEventListener("close", () => { editing = null; current = null; });
$("#rCopy").onclick = () => current && copyText(current.id, false);
$("#rSave").onclick = () => current && pickFormat(current.id);
$("#rBody").addEventListener("click", e => {
  const b = e.target.closest(".nm"); if (!b || !current) return;
  const inp = document.createElement("input"); inp.type = "text"; inp.value = b.textContent; inp.setAttribute("aria-label","Speaker name");
  b.replaceWith(inp); inp.focus(); inp.select();
  let done = false;
  const finish = async (save) => {
    if (done) return; done = true;
    if (save && inp.value.trim() !== b.textContent){
      const id = current.id;
      const rr = await post("/api/rename-speaker", {id, speaker: b.dataset.spk, name: inp.value.trim()});
      if (!rr.ok){ toast((await rr.json().catch(() => ({}))).error || "Couldn't rename."); if (current) drawReader(); return; }
      toast("Renamed. The saved transcript was updated too.");
      if (!current || current.id !== id) return;
      const keepTerm = $("#rFind").value; try{ doc = await getText(id); }catch(e){} $("#rFind").value = keepTerm;
    }
    if (current) drawReader();
  };
  inp.addEventListener("keydown", ev => { if (ev.isComposing || ev.keyCode === 229) return; if (ev.key==="Enter") finish(true); if (ev.key==="Escape"){ ev.preventDefault(); finish(false); } });
  inp.addEventListener("blur", () => finish(true));
});
backdropClose("reader", okToLeaveEdit); ["saveDlg","fmtDlg","vmDlg"].forEach(id => backdropClose(id));
$("#rBody").addEventListener("click", async e => {
  const a = e.target.closest("[data-ai]"), t = e.target.closest("[data-title]");
  if (t && current && aiDoc && aiDoc.summary){
    if (renaming || t.disabled) return;
    const pick = titleIdeas()[+t.dataset.title]; if (!pick) return;
    t.disabled = true;
    const id = current.id, name = await renameTo(id, pick);
    if (name && current && current.id === id){ $("#rName").textContent = name; drawReader(); }
    else t.disabled = false;
    return;
  }
  if (!a || !current) return;
  if (a.dataset.ai === "copy" && aiDoc && aiDoc.summary){ try{ await navigator.clipboard.writeText(aiDoc.summary.text); toast("Summary copied."); }catch(err){ toast("Couldn't copy."); } }
  if (a.dataset.ai === "redo"){ aiAuto.delete(current.id); if (aiDoc) aiDoc.run = null; runAI("summary", true); }
});
$("#rReveal").onclick = () => current && post("/api/reveal",{id:current.id});


/* ---------- ⌘K: search and commands ---------- */
const CKI = {
  add:'<path d="M12 5v14M5 12h14"/>', vm:'<rect x="6" y="2" width="12" height="20" rx="3"/><path d="M9.5 12v2M12 10v6M14.5 11.2v3.6"/>',
  rec:'<circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2.5" fill="currentColor"/>', find:'<circle cx="11" cy="11" r="6.5"/><path d="M16 16l4.5 4.5"/>',
  set:'<circle cx="12" cy="12" r="3"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M5.3 18.7l2.1-2.1M16.6 7.4l2.1-2.1"/>',
  ai:'<path d="M12 3l1.8 4.6L18.5 9.4l-4.7 1.8L12 16l-1.8-4.8L5.5 9.4l4.7-1.8z"/>', sun:'<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
  moon:'<path d="M20 14.5A8 8 0 0 1 9.5 4a8 8 0 1 0 10.5 10.5z"/>', dl:'<path d="M12 4v11M7 10l5 5 5-5M5 19h14"/>',
  doc:'<path d="M6 3h8l4 4v14H6z"/><path d="M14 3v4h4"/>', copy:'<rect x="8" y="8" width="12" height="12" rx="2"/><path d="M16 8V5a1 1 0 0 0-1-1H5a1 1 0 0 0-1 1v10a1 1 0 0 0 1 1h3"/>',
  pen:'<path d="M4 20h4L19 9l-4-4L4 16z"/>', trash:'<path d="M5 7h14M10 7V5h4v2M7 7l1 12h8l1-12"/>', folder:'<path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>',
  text:'<path d="M4 6h16M4 12h16M4 18h10"/>'
};
const ckIcon = k => `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${CKI[k]||CKI.doc}</svg>`;
function ckCommands(){
  const c = [];
  if ($("#reader").open && current){
    const id = current.id;
    c.push({sec:"This transcript", icon:"copy", label:"Copy text", run:() => copyText(id, view==="time")},
           {sec:"This transcript", icon:"pen", label:"Rename…", run:() => startRename()},
           ...(aiOn() ? [{sec:"This transcript", icon:"ai", label:"AI summary", run:() => runAI("summary")},
                         {sec:"This transcript", icon:"ai", label:"Suggest fixes", run:() => runAI("fixes")}] : []),
           {sec:"This transcript", icon:"folder", label:"Show in Finder", run:() => post("/api/reveal",{id})},
           {sec:"This transcript", icon:"trash", label:"Move to Trash", run:() => trashIt(id, bare(current.name))});
  }
  c.push({icon:"add", label:"Add files…", kb:"⌘O", run:() => $("#choose").click()},
         {icon:"vm", label:"Import from iPhone Voice Memos", run:() => $("#vmOpen").click()},
         {icon:"rec", label:"Record now", run:() => $("#recOpen").click()},
         {icon:"ai", label:"AI assist settings", run:() => $("#aiOpen").click()},
         {icon:"set", label:"Settings", kb:"⌘,", run:() => $("#setOpen").click()},
         {icon:"sun", label:"Appearance: Light", run:() => applyTheme("light")},
         {icon:"moon", label:"Appearance: Dark", run:() => applyTheme("dark")},
         {icon:"set", label:"Appearance: Match the system", run:() => applyTheme("system")},
         {icon:"dl", label:"Download all transcripts", run:() => { location.href = "/api/download-all"; }});
  return c.map(x => Object.assign({sec: "Commands"}, x));
}
function ckScore(text, q){
  const t = text.toLowerCase(); if (!q) return 1;
  const i = t.indexOf(q); if (i >= 0) return 3 - (i === 0 ? 0 : (/\s/.test(t[i-1]) ? 0.5 : 1));
  let k = 0; for (const ch of t){ if (ch === q[k]) k++; if (k === q.length) return 0.5; }
  return 0;
}
let ckItems = [], ckSel = 0, ckSeq = 0, ckT = null, ckHits = [];
function ckRender(){
  const q = $("#ckq").value.trim().toLowerCase();
  const done = jobs.filter(j => j.state === "done").sort((a,b) => (b.finished||0) - (a.finished||0));
  let docs = q ? done.map(j => ({j, s: Math.max(ckScore(bare(j.name), q) * 2, ckScore(j.summary || "", q) > 1 ? 1 : 0)})).filter(x => x.s > 0)
                  .sort((a,b) => b.s - a.s).slice(0, 6).map(x => x.j) : done.slice(0, 5);
  const items = docs.map(j => ({sec: q ? "Transcripts" : "Recent", icon:"doc", label: bare(j.name), meta: j.finished ? rowWhen(j.finished) : "", run: () => openReader(j.id, q ? {term: $("#ckq").value.trim()} : undefined)}));
  const cmds = ckCommands().map(c => Object.assign(c, {s: ckScore(c.label, q)})).filter(c => c.s > 0);
  if (q) cmds.sort((a,b) => b.s - a.s);
  const hits = q.length >= 2 ? ckHits.slice(0, 5) : [];
  const hitItems = hits.map(h => ({sec:"In the text", icon:"text", label: bare(h.name), sub: h.text, meta: h.t, term: q, run: () => openReader(h.id, {at: h.s, term: $("#ckq").value.trim()})}));
  ckItems = (q ? [...cmds.filter(c => c.sec !== "Commands" || c.s >= 2), ...items, ...hitItems, ...cmds.filter(c => c.sec === "Commands" && c.s < 2)]
               : [...cmds.filter(c => c.sec === "This transcript"), ...items, ...cmds.filter(c => c.sec === "Commands")]);
  ckSel = Math.min(ckSel, Math.max(0, ckItems.length - 1));
  let last = null;
  $("#ckList").innerHTML = ckItems.length ? ckItems.map((it, i) => {
    const head = it.sec !== last ? `<div class="ck-sec">${esc(it.sec)}</div>` : ""; last = it.sec;
    return head + `<div class="ck-item${i === ckSel ? " on" : ""}" role="option" aria-selected="${i === ckSel}" data-i="${i}">${ckIcon(it.icon)}
      <span class="ck-l"><span class="ck-t">${it.term ? markTerm(it.label, "") : esc(it.label)}</span>${it.sub ? `<span class="ck-s">${markTerm(it.sub, it.term)}</span>` : ""}</span>
      ${it.kb ? `<kbd>${esc(it.kb)}</kbd>` : it.meta ? `<span class="ck-m">${esc(it.meta)}</span>` : ""}</div>`;
  }).join("") : `<div class="ck-empty">Nothing found for “${esc($("#ckq").value.trim())}”</div>`;
  const on = $("#ckList .ck-item.on"); if (on) on.scrollIntoView({block:"nearest"});
}
async function ckSearchText(){
  const q = $("#ckq").value.trim(), seq = ++ckSeq;
  if (q.length < 2){ ckHits = []; return ckRender(); }
  let res = [];
  try{ res = (await (await fetch("/api/search?q=" + encodeURIComponent(q))).json()).results || []; }catch(e){}
  if (seq !== ckSeq) return;
  ckHits = res.flatMap(r => r.hits.map(h => ({id: r.id, name: r.name, s: h.s, t: h.t, text: h.text})));
  ckRender();
}
function ckOpen(){
  const d = $("#cmdk"); if (d.open){ d.close(); return; }
  $("#ckq").value = ""; ckSel = 0; ckHits = []; ckRender(); d.showModal(); $("#ckq").focus();
}
function ckRun(i){ const it = ckItems[i]; if (!it) return; const d = $("#cmdk");
  d.addEventListener("close", () => setTimeout(() => it.run(), 0), {once: true}); d.close(); }
$("#ckOpen").onclick = ckOpen;
$("#ckq").addEventListener("input", () => { ckSel = 0; ckRender(); clearTimeout(ckT); ckT = setTimeout(ckSearchText, 160); });
$("#ckq").addEventListener("keydown", e => {
  if (e.isComposing || e.keyCode === 229) return;
  if (e.key === "ArrowDown" || e.key === "ArrowUp"){ e.preventDefault(); if (!ckItems.length) return;
    ckSel = (ckSel + (e.key === "ArrowDown" ? 1 : -1) + ckItems.length) % ckItems.length; ckRender(); }
  else if (e.key === "Enter"){ e.preventDefault(); ckRun(ckSel); }
});
$("#ckList").addEventListener("mousemove", e => { const it = e.target.closest(".ck-item"); if (it && +it.dataset.i !== ckSel){ ckSel = +it.dataset.i; ckRender(); } });
$("#ckList").addEventListener("click", e => { const it = e.target.closest(".ck-item"); if (it) ckRun(+it.dataset.i); });
backdropClose("cmdk");
document.addEventListener("keydown", e => {
  const mod = e.metaKey || e.ctrlKey; if (!mod || e.altKey) return;
  const k = e.key.toLowerCase();
  if (k === "k"){ e.preventDefault(); if (!($("#recDlg") && $("#recDlg").open)) ckOpen(); return; }
  const modal = [...document.querySelectorAll("dialog")].some(d => d.open);
  if (k === "f" && !e.shiftKey){ e.preventDefault(); if ($("#reader").open) $("#rFind").focus(); else if (!modal) ckOpen(); return; }
  if (modal) return;
  if (k === "o" && !e.shiftKey){ e.preventDefault(); $("#choose").click(); }
  else if (e.key === ","){ e.preventDefault(); $("#setOpen").click(); }
});

/* ---------- polling ---------- */
let polling = false, pollAgain = false, paused = false, prevStates = null;
const pageStart = Date.now() / 1000, titledSeen = new Set();
async function poll(){
  if (polling){ pollAgain = true; return; }      // one request at a time; run once more when it finishes
  polling = true; clearTimeout(timer);
  try{
    const r = await (await fetch("/api/jobs")).json();
    const was = prevStates; prevStates = Object.fromEntries(r.jobs.map(j => [j.id, j.state]));
    jobs = r.jobs; paused = !!r.paused; render(r.now);
    const named = [];                                 // named by AI just now: say so, with Undo
    for (const j of jobs){
      const a = j.auto_titled; if (!a || a.at < pageStart || titledSeen.has(j.id + a.at)) continue;
      titledSeen.add(j.id + a.at); named.push([j, a]);
      if (current && current.id === j.id && !renaming){ current.name = j.name; $("#rName").textContent = bare(j.name); if (view === "ai") drawReader(); }
    }
    if (named.length){
      const undo = async () => { for (const [j, a] of named){ const n = await renameTo(j.id, a.from);
        if (n && current && current.id === j.id && !renaming) $("#rName").textContent = n; } };
      toast(named.length === 1 ? `Named “${named[0][1].to}”` : `Named ${named.length} recordings`, {label: "Undo", fn: undo});
    }
    if (was && !named.length){
      const ready = jobs.filter(j => j.state === "done" && was[j.id] && was[j.id] !== "done");
      if (ready.length === 1) toast(`“${bare(ready[0].name)}” is ready.`, {label: "Open", fn: () => openReader(ready[0].id)});
      else if (ready.length > 1) toast(`${ready.length} transcripts are ready.`, {label: "Show", fn: () => $("#library").scrollIntoView({behavior: "smooth", block: "start"})});
    }
    clearTimeout(timer);
    timer = setTimeout(poll, (jobs.some(isActive) || uploading.total) ? 1000 : 4000);
  }catch(e){
    clearTimeout(timer);
    toast("Matita isn't running. Open Matita from your Desktop or Applications.");
    timer = setTimeout(poll, 5000);
  } finally {
    polling = false;
    if (pollAgain){ pollAgain = false; poll(); }
  }
}
loadInfo(); poll();
</script>
</body>
</html>
"""


FAVICON_SVG = '<svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><rect width="100" height="100" rx="23" fill="#1f4d35"/><path d="M27 70V31L50 56L73 31V70" fill="none" stroke="#a6e8ae" stroke-width="9" stroke-linecap="round" stroke-linejoin="round"/></svg>'

ICON_B64 = "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"


# ---------------------------------------------------------------- main
def port_in_use():
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", PORT)) == 0


def wait_for_server(seconds=15):
    end = time.time() + seconds
    while time.time() < end:
        if port_in_use():
            return True
        time.sleep(0.3)
    return False


def launchctl(*args):
    return subprocess.run(["launchctl"] + list(args), capture_output=True, text=True)


def stop_agent():
    uid = os.getuid()
    for label, plist in [(LABEL, AGENT_PLIST)] + list(zip(OLD_LABELS, OLD_PLISTS)):
        if launchctl("bootout", "gui/%d/%s" % (uid, label)).returncode != 0 and os.path.isfile(plist):
            launchctl("unload", plist)


LAUNCH_SCRIPT = r"""#!/bin/bash
# Matita: make sure the background service is running, then open it in its own window.
URL="%(url)s"
U=$(id -u)
if ! curl -s -o /dev/null --max-time 1 "${URL}api/info"; then
  launchctl kickstart "gui/$U/%(label)s" >/dev/null 2>&1 \
    || launchctl bootstrap "gui/$U" "%(plist)s" >/dev/null 2>&1 \
    || launchctl load -w "%(plist)s" >/dev/null 2>&1
  for i in $(seq 1 80); do curl -s -o /dev/null --max-time 1 "${URL}api/info" && break; sleep 0.25; done
  if ! curl -s -o /dev/null --max-time 1 "${URL}api/info"; then
    choice=$(osascript -e 'button returned of (display dialog "Matita could not start in the background.\n\nOpen Login Items and make sure Matita Service is allowed, then try again." buttons {"Cancel", "Open Login Items"} default button 2 with icon caution)' 2>/dev/null)
    [ "$choice" = "Open Login Items" ] && open "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
    exit 1
  fi
fi
# A Chromium browser can show the app in its own window, without tabs or an address bar.
for b in "Google Chrome" "Microsoft Edge" "Brave Browser" "Chromium"; do
  for dir in "/Applications" "$HOME/Applications"; do
    if [ -d "$dir/$b.app" ]; then
      open -na "$dir/$b.app" --args --app="$URL" --window-size=940,860
      exit 0
    fi
  done
done
open "$URL"
"""


SERVICE_C = r"""
// Matita background service. It starts Python and stays its parent, so macOS attributes
// permissions (like Full Disk Access for Voice Memos) to this app instead of to Python.
#include <errno.h>
#include <limits.h>
#include <signal.h>
#include <spawn.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>
extern char **environ;
static pid_t child = 0;
static void forward(int sig) { if (child > 0) kill(child, sig); }
int main(void) {
    const char *home = getenv("HOME");
    char script[PATH_MAX];
    snprintf(script, sizeof script, "%%s/whisper-app/stt_app.py", home ? home : "");
    char *args[] = {"%(py)s", script, "--serve", NULL};
    signal(SIGTERM, forward); signal(SIGINT, forward); signal(SIGHUP, forward);
    if (posix_spawn(&child, "%(py)s", NULL, NULL, args, environ) != 0) { perror("posix_spawn"); return 1; }
    int status = 0;
    while (waitpid(child, &status, 0) < 0) { if (errno != EINTR) return 1; }
    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
"""


CAPTURE_APP = os.path.join(APP_DIR, "Matita Audio.app")   # records other apps' sound (calls) for meetings
CAPTURE_EXE = os.path.join(CAPTURE_APP, "Contents", "MacOS", "Matita Audio")
if BUNDLE:                                          # Matita.app ships it, signed with the app
    CAPTURE_EXE = os.path.join(BUNDLE, "Contents", "MacOS", "matita-audio")
CAPTURE_SWIFT = r"""
// Matita Audio: records what the Mac plays (Zoom, Teams, Meet in a browser...) with a Core Audio
// process tap (macOS 14.2+), so a recorded meeting has the other people as well as your microphone.
// Usage: "Matita Audio" <out.m4a>. Control files next to it: .stop (finish), .pause (skip audio while
// present, to match a paused microphone), .mute (write silence, so the timing stays lined up), .alive (touched by Matita; if it goes stale for 15 minutes, recording stops on its own).
// Progress is written to <out>.status: "started <unix ms>", "error <message>", "done".
import Foundation
import AudioToolbox
import AVFoundation
import CoreAudio

let outPath = CommandLine.arguments.count > 1 ? CommandLine.arguments[1] : ""
func status(_ line: String) {
    let url = URL(fileURLWithPath: outPath + ".status")
    let data = (line + "\n").data(using: .utf8)!
    if let h = try? FileHandle(forWritingTo: url) { h.seekToEndOfFile(); h.write(data); try? h.close() }
    else { try? data.write(to: url) }
}
struct Fail: Error { let msg: String }

func prop<T>(_ obj: AudioObjectID, _ sel: AudioObjectPropertySelector, _ initial: T) throws -> T {
    var addr = AudioObjectPropertyAddress(mSelector: sel, mScope: kAudioObjectPropertyScopeGlobal,
                                          mElement: kAudioObjectPropertyElementMain)
    var size: UInt32 = 0
    var err = AudioObjectGetPropertyDataSize(obj, &addr, 0, nil, &size)
    if err != noErr { throw Fail(msg: "property size \(sel): \(err)") }
    var value = initial
    err = withUnsafeMutablePointer(to: &value) { AudioObjectGetPropertyData(obj, &addr, 0, nil, &size, $0) }
    if err != noErr { throw Fail(msg: "property \(sel): \(err)") }
    return value
}

@available(macOS 14.2, *)
final class Recorder {
    var tapID = AudioObjectID(kAudioObjectUnknown)
    var aggID = AudioObjectID(kAudioObjectUnknown)
    var procID: AudioDeviceIOProcID?
    var file: AVAudioFile?
    var started = false
    var paused = false
    var muted = false
    let queue = DispatchQueue(label: "matita.audio", qos: .userInitiated)

    func start(_ out: URL) throws {
        let tap = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
        tap.uuid = UUID()
        tap.muteBehavior = .unmuted
        var err = AudioHardwareCreateProcessTap(tap, &tapID)
        if err != noErr { throw Fail(msg: "macOS didn't allow recording other apps (\(err)). Allow Matita Audio in System Settings → Privacy & Security → Screen & System Audio Recording, then try again.") }
        let outDev: AudioDeviceID = try prop(AudioObjectID(kAudioObjectSystemObject),
                                             kAudioHardwarePropertyDefaultSystemOutputDevice, AudioDeviceID(kAudioObjectUnknown))
        let outUID: CFString = try prop(outDev, kAudioDevicePropertyDeviceUID, "" as CFString)
        var asbd: AudioStreamBasicDescription = try prop(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription())
        let desc: [String: Any] = [
            kAudioAggregateDeviceNameKey: "Matita Audio",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outUID as String,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outUID as String]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: tap.uuid.uuidString]],
        ]
        err = AudioHardwareCreateAggregateDevice(desc as CFDictionary, &aggID)
        if err != noErr { throw Fail(msg: "Couldn't create the capture device (\(err)).") }
        guard let format = AVAudioFormat(streamDescription: &asbd) else { throw Fail(msg: "Unknown audio format.") }
        let settings: [String: Any] = [AVFormatIDKey: kAudioFormatMPEG4AAC, AVSampleRateKey: format.sampleRate,
                                       AVNumberOfChannelsKey: format.channelCount, AVEncoderBitRateKey: 96000]
        file = try AVAudioFile(forWriting: out, settings: settings, commonFormat: .pcmFormatFloat32,
                               interleaved: format.isInterleaved)
        err = AudioDeviceCreateIOProcIDWithBlock(&procID, aggID, queue) { [weak self] _, input, _, _, _ in
            guard let self = self, let f = self.file else { return }
            if !self.started {
                self.started = true
                status("started \(Int64(Date().timeIntervalSince1970 * 1000))")
            }
            if self.paused { return }
            guard let buf = AVAudioPCMBuffer(pcmFormat: format, bufferListNoCopy: input, deallocator: nil) else { return }
            if self.muted, let z = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: buf.frameLength) {
                z.frameLength = buf.frameLength
                for b in UnsafeMutableAudioBufferListPointer(z.mutableAudioBufferList) {
                    if let d = b.mData { memset(d, 0, Int(b.mDataByteSize)) }
                }
                try? f.write(from: z)
            } else {
                try? f.write(from: buf)
            }
        }
        if err != noErr { throw Fail(msg: "Couldn't start capturing (\(err)).") }
        err = AudioDeviceStart(aggID, procID)
        if err != noErr { throw Fail(msg: "Couldn't start capturing (\(err)).") }
    }

    func stop() {
        if aggID != AudioObjectID(kAudioObjectUnknown) {
            AudioDeviceStop(aggID, procID)
            if let p = procID { AudioDeviceDestroyIOProcID(aggID, p) }
            AudioHardwareDestroyAggregateDevice(aggID)
        }
        if tapID != AudioObjectID(kAudioObjectUnknown) { AudioHardwareDestroyProcessTap(tapID) }
        queue.sync { self.file = nil }            // closes the file and finishes the m4a
    }
}

if outPath.isEmpty { exit(2) }
if #available(macOS 14.2, *) {
    let rec = Recorder()
    do { try rec.start(URL(fileURLWithPath: outPath)) }
    catch let e as Fail { status("error " + e.msg); exit(1) }
    catch { status("error \(error)"); exit(1) }
    let began = Date()
    let fm = FileManager.default
    func finish() { rec.stop(); status("done"); exit(0) }
    let timer = DispatchSource.makeTimerSource(queue: .main)
    timer.schedule(deadline: .now() + 0.25, repeating: 0.25)
    timer.setEventHandler {
        let p = fm.fileExists(atPath: outPath + ".pause"), m = fm.fileExists(atPath: outPath + ".mute")
        rec.queue.async { rec.paused = p; rec.muted = m }
        var stale = Date().timeIntervalSince(began) > 6 * 3600
        if let a = try? fm.attributesOfItem(atPath: outPath + ".alive"), let m = a[.modificationDate] as? Date,
           Date().timeIntervalSince(m) > 900 { stale = true }
        if fm.fileExists(atPath: outPath + ".stop") || stale { finish() }
    }
    timer.resume()
    signal(SIGTERM, SIG_IGN)
    let term = DispatchSource.makeSignalSource(signal: SIGTERM, queue: .main)
    term.setEventHandler { finish() }
    term.resume()
    dispatchMain()
} else {
    status("error Recording other apps needs macOS 14.2 or later.")
    exit(1)
}
"""


def mac_version():
    try:
        return tuple(int(x) for x in (platform.mac_ver()[0] or "0").split(".")[:2])
    except ValueError:
        return (0,)


def capture_ready():
    return sys.platform == "darwin" and mac_version() >= (14, 2) and os.access(CAPTURE_EXE, os.X_OK)


CAPTURE_BUILD = {"state": None, "error": None}


def build_capture_app():
    """Compile Matita Audio with the Swift compiler from Apple's Command Line Tools (Homebrew needs them too).
    Rebuilt only when the source changes, so the recording permission survives reinstalls."""
    if sys.platform != "darwin" or mac_version() < (14, 2):
        return None
    if BUNDLE:
        return CAPTURE_EXE if os.access(CAPTURE_EXE, os.X_OK) else None
    stamp = hashlib.sha256(CAPTURE_SWIFT.encode("utf-8")).hexdigest()
    stamp_file = os.path.join(CAPTURE_APP, "Contents", "Resources", "source.sha256")
    try:
        with open(stamp_file) as fh:
            if fh.read().strip() == stamp and os.access(CAPTURE_EXE, os.X_OK):
                return CAPTURE_EXE
    except OSError:
        pass
    swiftc = find_bin("swiftc") or ("/usr/bin/swiftc" if os.path.exists("/usr/bin/swiftc") else None)
    if not swiftc:
        CAPTURE_BUILD.update(state="error", error="The Swift compiler isn't installed (xcode-select --install).")
        return None
    CAPTURE_BUILD.update(state="building", error=None)
    tmp = tempfile.mkdtemp()
    try:
        src = os.path.join(tmp, "main.swift")
        with open(src, "w") as fh:
            fh.write(CAPTURE_SWIFT)
        out = os.path.join(tmp, "capture")
        r = subprocess.run([swiftc, "-O", "-o", out, src], capture_output=True, text=True, timeout=600)
        if r.returncode != 0:
            CAPTURE_BUILD.update(state="error", error=(r.stderr or r.stdout).strip()[-1500:])
            try:
                with open(os.path.join(APP_DIR, "capture-build.log"), "w") as fh:
                    fh.write(r.stderr + r.stdout)
            except OSError:
                pass
            return None
        shutil.rmtree(CAPTURE_APP, ignore_errors=True)
        os.makedirs(os.path.dirname(CAPTURE_EXE))
        os.makedirs(os.path.dirname(stamp_file))
        shutil.copyfile(out, CAPTURE_EXE)
        os.chmod(CAPTURE_EXE, 0o755)
        with open(os.path.join(CAPTURE_APP, "Contents", "Info.plist"), "wb") as fh:
            plistlib.dump({
                "CFBundleName": "Matita Audio", "CFBundleDisplayName": "Matita Audio",
                "CFBundleIdentifier": LABEL + ".audio", "CFBundleExecutable": "Matita Audio",
                "CFBundleIconFile": "AppIcon", "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": VERSION, "CFBundleVersion": VERSION,
                "LSUIElement": True, "LSMinimumSystemVersion": "14.2",
                "NSAudioCaptureUsageDescription": "Matita records the sound your Mac plays, such as the other "
                                                  "people in a call, so it can transcribe the whole meeting. "
                                                  "Nothing leaves your Mac.",
            }, fh)
        icns = os.path.join(APP_COPIES[0], "Contents", "Resources", "AppIcon.icns")
        if os.path.isfile(icns):
            shutil.copyfile(icns, os.path.join(CAPTURE_APP, "Contents", "Resources", "AppIcon.icns"))
        with open(stamp_file, "w") as fh:
            fh.write(stamp)
        subprocess.run(["codesign", "--force", "--sign", "-", CAPTURE_APP], capture_output=True)
        CAPTURE_BUILD.update(state="done", error=None)
        return CAPTURE_EXE
    except Exception as e:
        CAPTURE_BUILD.update(state="error", error=str(e)[:300])
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


CAPTURES = {}                                       # token -> {"path", "started"}
CAP_DIR = os.path.join(QUEUE_DIR, "capture")


def _cap_status(path):
    try:
        with open(path + ".status", encoding="utf-8") as fh:
            return fh.read().splitlines()
    except OSError:
        return []


def capture_start():
    if not capture_ready():
        raise ValueError("Recording other apps' sound needs macOS 14.2 or later.")
    os.makedirs(CAP_DIR, exist_ok=True)
    token = uuid.uuid4().hex[:12]
    path = os.path.join(CAP_DIR, token + ".m4a")
    open(path + ".alive", "w").close()
    if BUNDLE:      # a child of Matita.app, so macOS asks on behalf of Matita
        subprocess.Popen([CAPTURE_EXE, path], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    else:
        r = subprocess.run(["open", "-n", "-g", "-a", CAPTURE_APP, "--args", path], capture_output=True, text=True)
        if r.returncode != 0:
            raise ValueError("Couldn't start Matita Audio. " + (r.stderr or "").strip()[-200:])
    for _ in range(60):                             # the first time, macOS may be showing a permission prompt
        lines = _cap_status(path)
        err = next((x[6:] for x in lines if x.startswith("error ")), None)
        if err:
            raise ValueError(err)
        st = next((x for x in lines if x.startswith("started ")), None)
        if st:
            CAPTURES[token] = {"path": path, "started": int(st.split()[1]) / 1000.0}
            return token, CAPTURES[token]["started"]
        time.sleep(0.25)
    CAPTURES[token] = {"path": path, "started": None}   # still waiting (permission prompt?); keep going
    return token, None


def capture_ctl(token, what):
    c = CAPTURES.get(token)
    if not c:
        return False
    path = c["path"]
    if what == "alive":
        open(path + ".alive", "w").close()
    elif what == "pause":
        open(path + ".pause", "w").close()
    elif what == "resume":
        try:
            os.remove(path + ".pause")
        except OSError:
            pass
    elif what == "mute":
        open(path + ".mute", "w").close()
    elif what == "unmute":
        try:
            os.remove(path + ".mute")
        except OSError:
            pass
    elif what == "stop":
        open(path + ".stop", "w").close()
        for _ in range(40):
            if "done" in _cap_status(path):
                break
            time.sleep(0.25)
        if c.get("started") is None:
            st = next((x for x in _cap_status(path) if x.startswith("started ")), None)
            c["started"] = int(st.split()[1]) / 1000.0 if st else None
    return True


def capture_take(token):
    """Hand the finished system-audio file to a job: (path, unix start time) or (None, None)."""
    c = CAPTURES.pop(token, None)
    if not c:
        return None, None
    path = c["path"]
    for ext in (".stop", ".pause", ".mute", ".alive", ".status"):
        try:
            os.remove(path + ext)
        except OSError:
            pass
    if not os.path.isfile(path) or os.path.getsize(path) < 4096 or c.get("started") is None:
        try:
            os.remove(path)
        except OSError:
            pass
        return None, None
    return path, c["started"]


def is_silent(path):
    """True if a file has no sound at all (macOS gives silence when the permission is missing)."""
    if not FFMPEG:
        return False
    r = subprocess.run([FFMPEG, "-nostdin", "-hide_banner", "-i", path, "-af", "volumedetect", "-f", "null", "-"],
                       capture_output=True, text=True, errors="replace", timeout=900)
    m = re.search(r"max_volume:\s*(-?[\d.]+|-inf) dB", r.stderr)
    return bool(m) and (m.group(1) == "-inf" or float(m.group(1)) < -70)


def build_service_app(py):
    """Compile the small service app. Returns its executable, or None if it can't be built.
    It is only rebuilt when its source changes, so a Full Disk Access permission survives reinstalls."""
    src = SERVICE_C % {"py": py}
    stamp = hashlib.sha256(src.encode("utf-8")).hexdigest()
    exe = os.path.join(SERVICE_APP, "Contents", "MacOS", "Matita Service")
    stamp_file = os.path.join(SERVICE_APP, "Contents", "Resources", "source.sha256")
    try:
        with open(stamp_file) as fh:
            if fh.read().strip() == stamp and os.access(exe, os.X_OK):
                return exe
    except OSError:
        pass
    cc = find_bin("clang", "cc")
    if not cc:
        return None
    tmp = tempfile.mkdtemp()
    try:
        c_file = os.path.join(tmp, "service.c")
        with open(c_file, "w") as fh:
            fh.write(src)
        out = os.path.join(tmp, "service")
        if subprocess.run([cc, "-O2", "-o", out, c_file], capture_output=True).returncode != 0:
            return None
        shutil.rmtree(SERVICE_APP, ignore_errors=True)
        os.makedirs(os.path.dirname(exe))
        os.makedirs(os.path.dirname(stamp_file))
        shutil.copyfile(out, exe)
        os.chmod(exe, 0o755)
        with open(os.path.join(SERVICE_APP, "Contents", "Info.plist"), "wb") as fh:
            plistlib.dump({
                "CFBundleName": "Matita Service", "CFBundleDisplayName": "Matita Service",
                "CFBundleIdentifier": LABEL + ".service", "CFBundleExecutable": "Matita Service",
                "CFBundleIconFile": "AppIcon", "CFBundlePackageType": "APPL",
                "CFBundleShortVersionString": VERSION, "CFBundleVersion": VERSION,
                "LSUIElement": True, "LSBackgroundOnly": True,
            }, fh)
        icns = os.path.join(APP_COPIES[0], "Contents", "Resources", "AppIcon.icns")
        if os.path.isfile(icns):
            shutil.copyfile(icns, os.path.join(SERVICE_APP, "Contents", "Resources", "AppIcon.icns"))
        with open(stamp_file, "w") as fh:
            fh.write(stamp)
        try:
            subprocess.run(["codesign", "--force", "--sign", "-", SERVICE_APP], capture_output=True)
        except OSError:
            pass
        return exe
    except OSError:
        return None
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def ours(bundle):
    """True if a .app at this path was made by this installer (or doesn't exist)."""
    if not os.path.exists(bundle):
        return True
    try:
        with open(os.path.join(bundle, "Contents", "Info.plist"), "rb") as fh:
            bid = plistlib.load(fh).get("CFBundleIdentifier", "")
            return any(bid.startswith(l) for l in [LABEL] + OLD_LABELS)
    except Exception:
        return False


def build_app(path):
    """A tiny Mac app (no Terminal window) that starts the service if needed and opens Matita."""
    if not ours(path):
        print("Skipped %s: another app with that name is already there." % path.replace(HOME, "~", 1))
        return False
    shutil.rmtree(path, ignore_errors=True)
    macos = os.path.join(path, "Contents", "MacOS")
    res = os.path.join(path, "Contents", "Resources")
    os.makedirs(macos)
    os.makedirs(res)
    with open(os.path.join(path, "Contents", "Info.plist"), "wb") as fh:
        plistlib.dump({
            "CFBundleName": "Matita", "CFBundleDisplayName": "Matita",
            "CFBundleIdentifier": LABEL + ".app", "CFBundleExecutable": "matita",
            "CFBundleIconFile": "AppIcon", "CFBundlePackageType": "APPL",
            "CFBundleShortVersionString": VERSION, "CFBundleVersion": VERSION, "LSUIElement": True,
            "LSMinimumSystemVersion": "12.0",
        }, fh)
    exe = os.path.join(macos, "matita")
    with open(exe, "w") as fh:
        fh.write(LAUNCH_SCRIPT % {"url": URL, "label": LABEL, "plist": AGENT_PLIST})
    os.chmod(exe, 0o755)
    # App icon (.icns) built with the Mac's own sips and iconutil
    try:
        tmp = tempfile.mkdtemp()
        png = os.path.join(tmp, "icon.png")
        with open(png, "wb") as fh:
            fh.write(base64.b64decode(ICON_B64))
        iconset = os.path.join(tmp, "AppIcon.iconset")
        os.makedirs(iconset)
        for size in (16, 32, 128, 256):
            for scale in (1, 2):
                px = size * scale
                name = "icon_%dx%d%s.png" % (size, size, "@2x" if scale == 2 else "")
                subprocess.run(["sips", "-z", str(px), str(px), png, "--out", os.path.join(iconset, name)],
                               capture_output=True)
        subprocess.run(["cp", png, os.path.join(iconset, "icon_512x512.png")])
        subprocess.run(["iconutil", "-c", "icns", iconset, "-o", os.path.join(res, "AppIcon.icns")],
                       capture_output=True)
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["touch", path])                 # make Finder pick up the new icon
    except Exception:
        pass
    return True


def install():
    # /usr/bin/python3 survives Xcode updates; the real path behind it can move
    py = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else (sys.executable or "python3")
    os.makedirs(APP_DIR, exist_ok=True)
    dst = os.path.join(APP_DIR, "stt_app.py")
    if os.path.abspath(__file__) != os.path.abspath(dst):
        shutil.copyfile(__file__, dst)

    # Build everything first, so a problem here never leaves you without a running Matita.
    for path in APP_COPIES:
        try:
            build_app(path)
        except OSError as e:
            print("Couldn't create %s (%s). You can still open Matita from Applications."
                  % (path.replace(HOME, "~", 1), e.strerror or e))
    service = build_service_app(py)
    if mac_version() >= (14, 2):
        print("Building Matita Audio (for recording calls)…")
        if not build_capture_app():
            print("Skipped: %s" % (CAPTURE_BUILD.get("error") or "unknown error").splitlines()[-1][:200])

    stop_agent()
    # stop an older copy that was started by hand from Terminal (only if it really is Matita)
    if port_in_use():
        r = subprocess.run(["lsof", "-ti", "tcp:%d" % PORT, "-sTCP:LISTEN"], capture_output=True, text=True)
        for pid in r.stdout.split():
            if not pid.isdigit() or int(pid) == os.getpid():
                continue
            cmd = subprocess.run(["ps", "-o", "command=", "-p", pid], capture_output=True, text=True).stdout
            if "stt_app.py" in cmd or "Transcriber" in cmd or "Matita" in cmd:
                subprocess.run(["kill", pid], capture_output=True)
        time.sleep(1)
    env = {"PATH": "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"}
    if service:
        env["MATITA_SERVICE_APP"] = "1"
    os.makedirs(os.path.dirname(AGENT_PLIST), exist_ok=True)
    with open(AGENT_PLIST, "wb") as fh:
        plistlib.dump({
            "Label": LABEL,
            "ProgramArguments": [service] if service else [py, dst, "--serve"],
            "RunAtLoad": True,
            "KeepAlive": True,
            "ThrottleInterval": 10,
            "ProcessType": "Interactive",
            "EnvironmentVariables": env,
            "StandardOutPath": os.path.join(APP_DIR, "matita.log"),
            "StandardErrorPath": os.path.join(APP_DIR, "matita.log"),
        }, fh)
    # right after a bootout, bootstrap can briefly fail ("error 5"), so retry for a few seconds
    for attempt in range(8):
        if launchctl("bootstrap", "gui/%d" % os.getuid(), AGENT_PLIST).returncode == 0:
            break
        time.sleep(1)
    else:
        launchctl("load", "-w", AGENT_PLIST)

    # Matita is now Matita: remove the old apps, launcher and service (only ones this installer made)
    for path in OLD_APPS:
        if os.path.exists(path) and ours(path):
            shutil.rmtree(path, ignore_errors=True)
    for path in OLD_PLISTS + [os.path.join(HOME, "Desktop", "Transcriber.command")]:
        if os.path.isfile(path):
            os.remove(path)

    if wait_for_server():
        print("Installed. Matita runs in the background and starts when you log in.")
        print("From now on, just double-click Matita on your Desktop (it's also in Applications).")
        print("You can close this Terminal window.\n")
        subprocess.run(["open", APP_COPIES[0]])
    else:
        print("Installed, but the background service didn't start.")
        print("If macOS asked about Matita Service, allow it, then double-click Matita on your Desktop.")
        print("If it still won't start: System Settings > General > Login Items, turn on Matita Service.")
        print("Details are in ~/whisper-app/matita.log")


def uninstall():
    stop_agent()
    rules = [f for f in (SUDOERS_FILE, OLD_SUDOERS) if os.path.isfile(f)]
    if rules:
        subprocess.run([SUDO, "-n", PMSET, "-a", "disablesleep", "0"], capture_output=True)
    for path in (AGENT_PLIST,):
        if os.path.isfile(path):
            os.remove(path)
    for path in APP_COPIES + [SERVICE_APP, CAPTURE_APP] + OLD_APPS:
        if ours(path):
            shutil.rmtree(path, ignore_errors=True)
    for path in OLD_PLISTS:
        if os.path.isfile(path):
            os.remove(path)
    print("Removed the background service and the Matita app.")
    print("Your transcripts and models were left untouched.")
    print("If you added it to the Dock with Safari, drag that icon out of the Dock to remove it.")
    if rules:
        print("To also remove the lid-closed permission, run:  sudo rm -f " + " ".join(rules))


def retire_agent():
    """First start of Matita.app on a Mac that had the older install: stop and remove the background
    service and its launchers (your transcripts, settings and models stay where they are)."""
    if not (BUNDLE and os.path.isfile(AGENT_PLIST)):
        return
    try:
        stop_agent()
        os.remove(AGENT_PLIST)
    except OSError:
        pass
    for path in APP_COPIES + [SERVICE_APP]:
        try:
            with open(os.path.join(path, "Contents", "Info.plist"), "rb") as fh:
                ours = str(plistlib.load(fh).get("CFBundleIdentifier", "")).startswith(LABEL)
        except Exception:
            ours = False
        if ours and os.path.abspath(path) != os.path.abspath(BUNDLE):
            shutil.rmtree(path, ignore_errors=True)
    for _ in range(40):                             # give the old server a moment to let go of the port
        if not port_in_use():
            break
        time.sleep(0.25)


def serve(open_browser):
    global OUT_DIR
    retire_agent()
    if port_in_use():
        if open_browser:
            print("Matita is already running. Opening it in your browser.")
            webbrowser.open(URL)
            return
        print("Port %d is already in use." % PORT)
        sys.exit(1)

    missing = []
    if not WHISPER:
        missing.append("whisper.cpp  ->  brew install whisper-cpp")
    if not FFMPEG:
        missing.append("ffmpeg       ->  brew install ffmpeg")
    if not list_models():
        missing.append("a model      ->  download a ggml-*.bin file into ~/whisper-models")
    if missing:
        print("Missing:\n  " + "\n  ".join(missing) + "\n")

    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)   # answer right away, even while starting up

    def start_up():
        # picking the folder can wait on a macOS privacy prompt, so it must not block the server
        global OUT_DIR
        OUT_DIR = pick_out_dir()
        READY.set()
        load_library()
        restore_queue()
        threading.Thread(target=worker, daemon=True).start()
        threading.Thread(target=queue_loop, daemon=True).start()
        threading.Thread(target=update_loop, daemon=True).start()
        threading.Thread(target=vm_auto_loop, daemon=True).start()
        threading.Thread(target=notes_loop, daemon=True).start()
        threading.Thread(target=upgrade_voice_model, daemon=True).start()
        threading.Thread(target=build_capture_app, daemon=True).start()
        threading.Thread(target=watch_loop, daemon=True).start()
        threading.Thread(target=lid_loop, daemon=True).start()
    threading.Thread(target=lid_restore, daemon=True).start()     # undo anything left from a crash
    threading.Thread(target=start_up, daemon=True).start()

    def stop(*_):
        if RESTORED.is_set():                       # otherwise queue.json still holds work we haven't loaded
            save_queue()
        flush_trash()
        lid_restore()
        os._exit(0)
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(URL)).start()
        print("Matita is running at", URL)
        print("Keep this window open while you use it. Closing it quits Matita.")
    else:
        print(time.strftime("%Y-%m-%d %H:%M"), "Matita service started at", URL)
    sys.stdout.flush()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        lid_restore()


def main():
    if "--diarize" in sys.argv:
        i = sys.argv.index("--diarize")
        extra = sys.argv[i + 4:i + 5]
        return diarize_main(sys.argv[i + 1], sys.argv[i + 2], int(sys.argv[i + 3]),
                            int(extra[0]) if extra and extra[0].isdigit() else 1)
    if "--parakeet" in sys.argv:
        i = sys.argv.index("--parakeet")
        return parakeet_main(sys.argv[i + 1], sys.argv[i + 2], int(sys.argv[i + 3]))
    if "--setup-speakers" in sys.argv:
        sys.exit(0 if setup_speakers(echo=True) else 1)
    if "--print-capture-swift" in sys.argv:        # used by the app build
        return sys.stdout.write(CAPTURE_SWIFT)
    if "--write-icon" in sys.argv:
        with open(sys.argv[sys.argv.index("--write-icon") + 1], "wb") as fh:
            fh.write(base64.b64decode(ICON_B64))
        return
    if "--version" in sys.argv:
        return print(VERSION)
    if "--install" in sys.argv:
        return install()
    if "--uninstall" in sys.argv:
        return uninstall()
    serve(open_browser="--serve" not in sys.argv)


if __name__ == "__main__":
    main()
