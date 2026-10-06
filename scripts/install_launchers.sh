#!/bin/zsh
# Install two ways to open Hand Synth without a terminal `cd`:
#   1. ~/Applications/Hand Synth.app  (double-click, Spotlight, Dock; its own camera permission)
#   2. ~/.local/bin/hand-synth        (a command that works from any terminal)
# Both run this checkout. Re-run after moving the repo. Usage: scripts/install_launchers.sh
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
APP="$HOME/Applications/Hand Synth.app"
BIN="$HOME/.local/bin"
LOG="$HOME/Library/Logs/HandSynth.log"

echo "Repo: $REPO"

# 1. Python environment in the repo (Python 3.12; mediapipe has no wheels for newer)
if [[ ! -x "$REPO/.venv/bin/python" ]]; then
  echo "Creating .venv ..."
  if command -v uv >/dev/null; then
    uv venv -q -p 3.12 "$REPO/.venv"
    uv pip install -q -p "$REPO/.venv/bin/python" -e "$REPO"
  else
    python3.12 -m venv "$REPO/.venv"
    "$REPO/.venv/bin/pip" install -q -e "$REPO"
  fi
fi

# 2. App bundle
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources" "$(dirname "$LOG")"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>Hand Synth</string>
  <key>CFBundleDisplayName</key><string>Hand Synth</string>
  <key>CFBundleIdentifier</key><string>local.handsynth.app</string>
  <key>CFBundleVersion</key><string>1.0</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>hand-synth</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSCameraUsageDescription</key><string>Hand Synth reads your hand positions from the camera to play chords.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/hand-synth" <<LAUNCH
#!/bin/zsh
# Launched by Finder/Dock: run the instrument from the repo, logging to ~/Library/Logs/HandSynth.log.
cd "$REPO" || exit 1
export PYTHONUNBUFFERED=1          # log lines appear as they happen
{ echo; echo "=== \$(date) ==="; } >> "$LOG"
exec "$REPO/.venv/bin/python" -m hand_synth.main "\$@" >> "$LOG" 2>&1
LAUNCH
chmod +x "$APP/Contents/MacOS/hand-synth"

ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
"$REPO/.venv/bin/python" "$REPO/scripts/make_icon.py" "$ICONSET/icon_512x512@2x.png"
for s in 16 32 128 256 512; do
  sips -z $s $s "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${s}x${s}.png" >/dev/null
  sips -z $((s * 2)) $((s * 2)) "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${s}x${s}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns"

codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true   # ad-hoc: lets macOS remember the camera grant
touch "$APP"                                                        # refresh Finder's icon cache

# 3. Terminal command
mkdir -p "$BIN"
ln -sf "$REPO/.venv/bin/hand-synth" "$BIN/hand-synth"

echo "Installed:"
echo "  $APP   (open from Spotlight: Hand Synth)"
echo "  $BIN/hand-synth   (run: hand-synth [--key G] [--ui minimal])"
echo "  Logs: $LOG"
