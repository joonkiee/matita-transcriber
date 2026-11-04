#!/bin/bash
# Removes Matita's background service and launcher. Transcripts and models are kept.
set -e
if [ -f "$HOME/whisper-app/stt_app.py" ]; then
  /usr/bin/python3 "$HOME/whisper-app/stt_app.py" --uninstall || true
fi
rm -rf "$HOME/whisper-app"
if [ -f /etc/sudoers.d/matita-lid ] || [ -f /etc/sudoers.d/transcriber-lid ]; then
  echo "Removing the lid-closed permission (macOS will ask for your password)…"
  sudo /usr/bin/pmset -a disablesleep 0 || true
  sudo rm -f /etc/sudoers.d/matita-lid /etc/sudoers.d/transcriber-lid || true
fi
echo
echo "To also free up disk space, you can delete:"
echo "  ~/whisper-models                       (speech, Parakeet and speaker models)"
echo "  brew uninstall whisper-cpp ffmpeg      (if nothing else uses them)"
echo "Your transcripts in ~/Documents/Transcripts were not touched."
