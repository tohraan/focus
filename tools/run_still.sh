#!/bin/zsh
# Usage: tools/run_still.sh TAG [clip ...]   runs synth clips (still camera) on the phone, pulls results to out_still/TAG/
# iPhone unlocked + connected, app installed. TAG names this run (e.g. baseline, s2).
D=A3D6F69C-8877-56F3-8B2E-20EDC8D0B7DB; B=com.tohraan.focus-dr
TAG=${1:-run}; shift; CLIPS=("$@"); [ ${#CLIPS[@]} -eq 0 ] && CLIPS=(synthA_random synthB_exposure synthC_shadow)
OUT="$HOME/gdg-ideathon/out_still/$TAG"; mkdir -p "$OUT"
for c in $CLIPS; do
  xcrun devicectl device copy to --device $D --domain-type appDataContainer --domain-identifier $B \
    --source "$HOME/gdg-ideathon/synth/$c.mp4" --destination "Documents/still/$c.mp4" >/dev/null 2>&1 || echo "copy failed: $c"
done
xcrun devicectl device process launch --device $D --terminate-existing $B -- -videos -still -indir still >/dev/null 2>&1
sleep 8
for i in $(seq 1 240); do sleep 5
  xcrun devicectl device info processes --device $D 2>&1 | grep -q "Focus.app/Focus" || break
done
for f in status.txt ${^CLIPS}_focus.mp4; do
  xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B \
    --source Documents/still_out/$f --destination "$OUT/$f" >/dev/null 2>&1
done
cat "$OUT/status.txt"
