**Download:**
- **Mac:** `Matita-3.22.1.dmg` (Apple Silicon, macOS 14 or later)
- **Windows (beta):** `Matita-3.22.1-setup.exe` (64-bit Windows 10 or 11)

### Fixed
- **Matita sometimes wouldn't open on a Mac** after it had quit unexpectedly or been updated: an earlier copy of its background server could keep running and hold its port, so the window stayed on "Starting Matita…". The server now stops by itself when its app goes away, and Matita clears any leftover copy when it starts.

### New in 3.22
- **Matita for Windows (beta):** add recordings, record from the microphone, name speakers, AI summaries, watch folders and exports. Recording other apps' sound (calls) and iPhone Voice Memos are Mac-only for now.

### First time installing?
**Mac:** Open the .dmg and drag Matita into Applications. The first time, macOS says it can't check the app for malware, because Matita isn't signed with a paid Apple Developer ID. Click **Done**, then **System Settings → Privacy & Security → Open Anyway**.

**Windows:** Run the installer. If Windows shows "Windows protected your PC", click **More info**, then **Run anyway**.

Your transcripts, settings and models stay when you update.
