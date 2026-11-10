#!/bin/zsh
# Usage: tools/run_videos.sh [clip ...]   (iPhone unlocked + connected, app installed)
# Copies clips to the phone, runs the live engines on them, pulls the MP4s back and opens them.
D=A3D6F69C-8877-56F3-8B2E-20EDC8D0B7DB; B=com.tohraan.focus-dr
SRC="$HOME/Downloads/video mock"; OUT="$HOME/gdg-ideathon/out"; mkdir -p "$OUT"
CLIPS=("$@"); [ ${#CLIPS[@]} -eq 0 ] && CLIPS=(IMG_5362 IMG_5363 IMG_5364 IMG_5365)
for c in $CLIPS; do
  xcrun devicectl device copy to --device $D --domain-type appDataContainer --domain-identifier $B \
    --source "$SRC/$c.MOV" --destination "Documents/in/$c.MOV" >/dev/null 2>&1 || echo "copy failed: $c"
done
xcrun devicectl device process launch --device $D --terminate-existing $B -- -videos >/dev/null 2>&1
sleep 8
for i in $(seq 1 180); do sleep 5
  xcrun devicectl device info processes --device $D 2>&1 | grep -q "Focus.app/Focus" || break
done
xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B \
  --source Documents/out/status.txt --destination "$OUT/status.txt" >/dev/null 2>&1
cat "$OUT/status.txt"
for c in $CLIPS; do
  for f in ${c}_focus.mp4 ${c}_frame.jpg; do
    xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B \
      --source Documents/out/$f --destination "$OUT/$f" >/dev/null 2>&1
  done
  [ -f "$OUT/${c}_focus.mp4" ] && open "$OUT/${c}_focus.mp4"
done
