#!/bin/zsh
# Usage: LOOP_ARGS="-still" tools/loop_mac.sh TAG clip [clip ...]
# Builds FocusBench (macOS, same engines/shaders as the phone app) and runs the clips, saving STILL frames only (every 6th frame, JPEG). No video, no phone.
cd "$(dirname $0)/.."
TAG=$1; shift; CLIPS=("$@")
EXTRA=${LOOP_ARGS:--still}
if [ -z "$NOBUILD" ]; then
  xcodebuild -project Focus.xcodeproj -scheme FocusBench -configuration Release -destination 'platform=macOS,arch=arm64' build 2>&1 | grep -E "error:|BUILD FAILED" | sort -u | head -n 8
fi
BIN=$(ls -d ~/Library/Developer/Xcode/DerivedData/Focus-*/Build/Products/Release/FocusBench.app | head -n 1)/Contents/MacOS/FocusBench
ROOT="$PWD/bench_root_$TAG"; mkdir -p "$ROOT/still"
for c in $CLIPS; do ln -sf "$PWD/synth/$c.mp4" "$ROOT/still/$c.mp4"; [ -f "$PWD/synth/${c}_poses.txt" ] && ln -sf "$PWD/synth/${c}_poses.txt" "$ROOT/still/${c}_poses.txt"; done
OUT="$PWD/out_still/$TAG"; rm -rf "$OUT"; mkdir -p "$OUT"
rm -rf "$ROOT/still_out"
"$BIN" -videos ${=EXTRA} -stills 6 -root "$ROOT" -indir still -only ${(j:,:)CLIPS} > "$OUT/log.txt" 2>&1
cp "$ROOT/still_out/status.txt" "$OUT/status.txt" 2>/dev/null; cp "$ROOT/still_out/progress.txt" "$OUT/progress.txt" 2>/dev/null
for c in $CLIPS; do cp -R "$ROOT/still_out/${c}_focus_frames" "$OUT/${c}_frames" 2>/dev/null; done
cat "$OUT/status.txt" | cut -c1-200
