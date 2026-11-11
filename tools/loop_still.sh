#!/bin/zsh
# Usage: tools/loop_still.sh TAG [clip ...]
# Runs synthetic clips through both engines on the phone, saving STILL frames only (every 6th frame as JPEG, no video).
# Pulls them to out_still/TAG/<clip>_frames/. Needs the iPhone unlocked + connected, app installed.
D=A3D6F69C-8877-56F3-8B2E-20EDC8D0B7DB; B=com.tohraan.focus-dr
TAG=${1:-run}; shift; CLIPS=("$@"); [ ${#CLIPS[@]} -eq 0 ] && CLIPS=(synthA_random synthB_exposure synthC_shadow)
EXTRA=${LOOP_ARGS:--still}
OUT="$HOME/gdg-ideathon/out_still/$TAG"; mkdir -p "$OUT"
for c in $CLIPS; do
  xcrun devicectl device copy to --device $D --domain-type appDataContainer --domain-identifier $B \
    --source "$HOME/gdg-ideathon/synth/$c.mp4" --destination "Documents/still/$c.mp4" >/dev/null 2>&1 || echo "copy failed: $c"
done
xcrun devicectl device process launch --device $D --terminate-existing $B -- -videos ${=EXTRA} -stills 6 -indir still -only ${(j:,:)CLIPS} >/dev/null 2>&1
sleep 6
for i in $(seq 1 360); do sleep 5
  xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B --source Documents/still_out/status.txt --destination "$OUT/status.txt" >/dev/null 2>&1
  grep -q "^DONE" "$OUT/status.txt" 2>/dev/null && break
done
xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B --source Documents/still_out/status.txt --destination "$OUT/status.txt" >/dev/null 2>&1
xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B --source Documents/still_out/progress.txt --destination "$OUT/progress.txt" >/dev/null 2>&1
for c in $CLIPS; do
  xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B \
    --source Documents/still_out/${c}_focus_frames --destination "$OUT/${c}_frames" >/dev/null 2>&1
done
cat "$OUT/status.txt"
