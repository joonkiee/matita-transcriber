#!/bin/bash
# Matita Transcriber installer for macOS
# Usage:  curl -fsSL https://raw.githubusercontent.com/joonkiee/matita-transcriber/main/install.sh | bash
set -euo pipefail

REPO="joonkiee/matita-transcriber"
BRANCH="main"
MODEL_NAME="${MATITA_MODEL:-${TRANSCRIBER_MODEL:-ggml-large-v3-turbo.bin}}"
MODEL_DIR="$HOME/whisper-models"
MODEL_URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$MODEL_NAME"
APP_URL="https://raw.githubusercontent.com/$REPO/$BRANCH/matita.py"

bold() { printf "\033[1m%s\033[0m\n" "$1"; }
step() { printf "\n\033[1;32m==>\033[0m \033[1m%s\033[0m\n" "$1"; }
fail() { printf "\n\033[1;31mError:\033[0m %s\n" "$1" >&2; exit 1; }

[ "$(uname -s)" = "Darwin" ] || fail "Matita only runs on macOS."
if [ "$(uname -m)" != "arm64" ]; then
  echo "Note: this Mac has an Intel chip. Matita works, but transcription will be much slower than on Apple Silicon."
fi

bold "Installing Matita Transcriber"
echo "This takes about 5–15 minutes, mostly downloading the speech model (about 1.7 GB in total)."

# 1. Homebrew
step "Checking Homebrew"
if ! command -v brew >/dev/null 2>&1; then
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$p" ] && eval "$("$p" shellenv)"; done
fi
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew isn't installed. Installing it now (it will ask for your Mac password)."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" </dev/tty
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do [ -x "$p" ] && eval "$("$p" shellenv)"; done
  command -v brew >/dev/null 2>&1 || fail "Homebrew didn't install. See https://brew.sh and run this installer again."
else
  echo "Homebrew is installed."
fi

# 2. whisper.cpp and ffmpeg
step "Installing whisper.cpp and ffmpeg"
brew install whisper-cpp ffmpeg

# 3. Speech model
step "Downloading the speech model ($MODEL_NAME)"
mkdir -p "$MODEL_DIR"
if [ -f "$MODEL_DIR/$MODEL_NAME" ] && [ "$(stat -f%z "$MODEL_DIR/$MODEL_NAME")" -gt 100000000 ]; then
  echo "Already downloaded."
else
  curl -L --fail --progress-bar -o "$MODEL_DIR/$MODEL_NAME.part" "$MODEL_URL" || fail "Couldn't download the model. Check your internet connection and try again."
  mv "$MODEL_DIR/$MODEL_NAME.part" "$MODEL_DIR/$MODEL_NAME"
fi
VAD="$MODEL_DIR/ggml-silero-v5.1.2.bin"      # small voice-activity model (under 1 MB)
[ -f "$VAD" ] || { curl -L --fail -s -o "$VAD.part" "https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v5.1.2.bin" && mv "$VAD.part" "$VAD"; } || true

# 4. The app (and the optional speaker add-on)
step "Installing the app"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then   # run as a file from the repo, not piped from curl
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
if [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/matita.py" ]; then
  cp "$SCRIPT_DIR/matita.py" "$TMP/stt_app.py"          # running from a cloned repo
else
  curl -fsSL -o "$TMP/stt_app.py" "$APP_URL" || fail "Couldn't download the app from $APP_URL"
fi
if [ "${MATITA_SPEAKERS:-${TRANSCRIBER_SPEAKERS:-1}}" != "0" ]; then
  step "Setting up speaker identification (about 60 MB)"
  /usr/bin/python3 "$TMP/stt_app.py" --setup-speakers || echo "Skipped for now. You can set it up later from the app's Settings."
fi
# Apple's built-in Python avoids problems with other Python installs on the Mac
step "Starting Matita"
/usr/bin/python3 "$TMP/stt_app.py" --install

bold ""
bold "Done! Matita is open in your browser."
echo "From now on, just double-click Matita on your Desktop. You can close Terminal."
