#!/bin/bash
# Builds Matita.app and Matita-<version>.dmg on an Apple Silicon Mac (or a GitHub macos-14 runner).
# Everything the app needs goes inside it: Python, whisper-cli (Metal), ffmpeg/ffprobe and Matita Audio.
#   bash scripts/build-app.sh            -> dist/Matita.app and dist/Matita-<version>.dmg
# Needs: Xcode Command Line Tools (clang, swiftc), cmake, curl. Caches downloads/builds in .build/.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
WHISPER_TAG="${WHISPER_TAG:-v1.9.5}"
FFMPEG_VER="${FFMPEG_VER:-7.1.1}"
PY_MINOR="${PY_MINOR:-3.12}"
MIN_MACOS=14.0
B="$ROOT/.build"; DIST="$ROOT/dist"
mkdir -p "$B" "$DIST"
VERSION=$(sed -n 's/^VERSION = "\(.*\)"/\1/p' matita.py)
echo "== Matita $VERSION"
export MACOSX_DEPLOYMENT_TARGET=$MIN_MACOS

# ---------------------------------------------------------------- whisper.cpp (static, Metal shaders embedded)
if [ ! -x "$B/whisper/bin/whisper-cli" ]; then
  echo "== whisper.cpp $WHISPER_TAG"
  rm -rf "$B/whisper.cpp"
  git clone -q --depth 1 --branch "$WHISPER_TAG" https://github.com/ggml-org/whisper.cpp "$B/whisper.cpp"
  cmake -S "$B/whisper.cpp" -B "$B/whisper.cpp/build" -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DGGML_NATIVE=OFF \
    -DGGML_OPENMP=OFF -DWHISPER_SDL2=OFF -DWHISPER_BUILD_TESTS=OFF -DWHISPER_BUILD_SERVER=OFF \
    -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=$MIN_MACOS >/dev/null
  cmake --build "$B/whisper.cpp/build" --config Release -j "$(sysctl -n hw.ncpu)" --target whisper-cli
  mkdir -p "$B/whisper/bin"
  cp "$B/whisper.cpp/build/bin/whisper-cli" "$B/whisper/bin/"
fi

# ---------------------------------------------------------------- ffmpeg (static, LGPL, only macOS system libraries)
if [ ! -x "$B/ffmpeg/bin/ffmpeg" ]; then
  echo "== ffmpeg $FFMPEG_VER"
  curl -fsSL "https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz" -o "$B/ffmpeg.tar.xz"
  rm -rf "$B/ffmpeg-src"; mkdir -p "$B/ffmpeg-src"
  tar -xJf "$B/ffmpeg.tar.xz" -C "$B/ffmpeg-src" --strip-components 1
  ( cd "$B/ffmpeg-src" && ./configure --prefix="$B/ffmpeg" --arch=arm64 --cc=clang \
      --extra-cflags="-mmacosx-version-min=$MIN_MACOS" --extra-ldflags="-mmacosx-version-min=$MIN_MACOS" \
      --enable-static --disable-shared --disable-autodetect --enable-audiotoolbox --enable-videotoolbox \
      --enable-zlib --disable-doc --disable-ffplay --disable-network --disable-debug >/dev/null \
    && make -j "$(sysctl -n hw.ncpu)" >/dev/null && make install >/dev/null )
fi

# ---------------------------------------------------------------- Python (python-build-standalone)
if [ ! -x "$B/python/bin/python3" ]; then
  echo "== Python $PY_MINOR"
  AUTH=(); [ -n "${GITHUB_TOKEN:-}" ] && AUTH=(-H "Authorization: Bearer $GITHUB_TOKEN")
  URL=$(curl -fsSL "${AUTH[@]}" https://api.github.com/repos/astral-sh/python-build-standalone/releases/latest \
    | python3 -c "import json,sys,re; a=json.load(sys.stdin)['assets']; print(next(x['browser_download_url'] for x in a if re.fullmatch(r'cpython-$PY_MINOR\.\d+\+\d+-aarch64-apple-darwin-install_only_stripped\.tar\.gz', x['name'])))")
  echo "   $URL"
  curl -fsSL "$URL" -o "$B/python.tar.gz"
  rm -rf "$B/python"; mkdir -p "$B/pyx"; tar -xzf "$B/python.tar.gz" -C "$B/pyx"; mv "$B/pyx/python" "$B/python"; rm -rf "$B/pyx"
  # what Matita never needs: tests, IDLE, Tk, ensurepip's wheels stay (venv needs them)
  rm -rf "$B/python/lib/python$PY_MINOR/test" "$B/python/lib/python$PY_MINOR/idlelib" "$B/python/lib/python$PY_MINOR/tkinter" \
         "$B/python/lib/python$PY_MINOR/turtledemo" "$B/python/lib/tcl"* "$B/python/lib/tk"* "$B/python/lib/itcl"* "$B/python/lib/thread"*
fi

# ---------------------------------------------------------------- the app
echo "== Matita.app"
APP="$DIST/Matita.app"; C="$APP/Contents"
rm -rf "$APP"; mkdir -p "$C/MacOS" "$C/Resources/bin"
swiftc -O -target arm64-apple-macos$MIN_MACOS app/Matita.swift -o "$C/MacOS/Matita"
python3 matita.py --print-capture-swift > "$B/MatitaAudio.swift"
swiftc -O -target arm64-apple-macos$MIN_MACOS "$B/MatitaAudio.swift" -o "$C/MacOS/matita-audio"
cp matita.py "$C/Resources/matita.py"
cp -R "$B/python" "$C/Resources/python"
cp "$B/whisper/bin/whisper-cli" "$B/ffmpeg/bin/ffmpeg" "$B/ffmpeg/bin/ffprobe" "$C/Resources/bin/"
cp LICENSE "$C/Resources/LICENSE.txt"
cp docs/*-OFL.txt "$C/Resources/" 2>/dev/null || true
cat > "$C/Resources/THIRD-PARTY.txt" <<EOF
Matita.app includes:
- whisper.cpp $WHISPER_TAG (MIT) https://github.com/ggml-org/whisper.cpp
- FFmpeg $FFMPEG_VER (LGPL 2.1, built without GPL parts) https://ffmpeg.org  source: https://ffmpeg.org/releases/ffmpeg-$FFMPEG_VER.tar.xz
- Python $PY_MINOR from python-build-standalone (PSF License and others) https://github.com/astral-sh/python-build-standalone
EOF

# icon: from the PNG inside matita.py
python3 matita.py --write-icon "$B/icon.png"
rm -rf "$B/AppIcon.iconset"; mkdir "$B/AppIcon.iconset"
for s in 16 32 128 256; do
  sips -z $s $s "$B/icon.png" --out "$B/AppIcon.iconset/icon_${s}x${s}.png" >/dev/null
  sips -z $((s*2)) $((s*2)) "$B/icon.png" --out "$B/AppIcon.iconset/icon_${s}x${s}@2x.png" >/dev/null
done
cp "$B/icon.png" "$B/AppIcon.iconset/icon_512x512.png"
iconutil -c icns "$B/AppIcon.iconset" -o "$C/Resources/AppIcon.icns"

cat > "$C/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Matita</string>
  <key>CFBundleDisplayName</key><string>Matita</string>
  <key>CFBundleIdentifier</key><string>com.local.matita.app</string>
  <key>CFBundleExecutable</key><string>Matita</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>$VERSION</string>
  <key>CFBundleVersion</key><string>$VERSION</string>
  <key>LSMinimumSystemVersion</key><string>$MIN_MACOS</string>
  <key>LSApplicationCategoryType</key><string>public.app-category.productivity</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSHumanReadableCopyright</key><string>MIT License · github.com/joonkiee/matita-transcriber</string>
  <key>NSMicrophoneUsageDescription</key><string>Matita records lectures and meetings with your microphone so it can transcribe them. Recordings stay on your Mac.</string>
  <key>NSAudioCaptureUsageDescription</key><string>Matita records the sound your Mac plays, such as the other people in a call, so it can transcribe the whole meeting. Nothing leaves your Mac.</string>
  <key>NSAppleEventsUsageDescription</key><string>Matita uses Finder to move transcripts you delete to the Trash.</string>
</dict></plist>
EOF

# every binary signed (ad hoc), inside out; Gatekeeper still asks once because there's no Developer ID
find "$C/Resources" -type f \( -perm -u+x -o -name '*.dylib' -o -name '*.so' \) -print0 | while IFS= read -r -d '' f; do
  if file "$f" | grep -q 'Mach-O'; then codesign --force --sign - --timestamp=none "$f" >/dev/null; fi
done
codesign --force --sign - --timestamp=none "$C/MacOS/matita-audio"
codesign --force --sign - --timestamp=none "$APP"
codesign --verify --deep --strict "$APP"

# nothing may point at Homebrew or the build machine
for f in "$C/MacOS/Matita" "$C/MacOS/matita-audio" "$C/Resources/bin/"*; do
  if otool -L "$f" | tail -n +2 | grep -vE '^\s*(/usr/lib/|/System/)'; then echo "!! $f links outside macOS"; exit 1; fi
done
du -sh "$APP"

# ---------------------------------------------------------------- disk image
echo "== dmg"
rm -rf "$B/dmg"; mkdir -p "$B/dmg"; cp -R "$APP" "$B/dmg/"; ln -s /Applications "$B/dmg/Applications"
DMG="$DIST/Matita-$VERSION.dmg"; rm -f "$DMG"
hdiutil create -quiet -volname "Matita $VERSION" -srcfolder "$B/dmg" -fs HFS+ -format UDZO -imagekey zlib-level=9 "$DMG"
du -sh "$DMG"
echo "$DMG"
