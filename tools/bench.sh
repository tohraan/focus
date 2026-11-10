#!/bin/zsh
# Usage: tools/bench.sh   (iPhone unlocked, connected). Runs each model/unit combo as its own launch.
D=A3D6F69C-8877-56F3-8B2E-20EDC8D0B7DB; B=com.tohraan.focus-dr
OUT=${1:-/private/tmp/claude-501/-Users-tohraan/c872b4cf-cfe5-454e-94fb-c8dc5028c1cc/scratchpad/b}; mkdir -p $OUT
for m in ${MODELS:-Yolo11sSeg Yolo11mSeg}; do for u in ${UNITS:-cpu+ane cpu+gpu}; do
  f=bench_${m}_${u}.txt; rm -f $OUT/$f
  xcrun devicectl device process launch --device $D --terminate-existing $B -bench $m $u >/dev/null 2>&1
  for i in $(seq 1 40); do sleep 4
    xcrun devicectl device copy from --device $D --domain-type appDataContainer --domain-identifier $B --source Documents/$f --destination $OUT/$f >/dev/null 2>&1
    grep -q DONE $OUT/$f 2>/dev/null && break
  done
  grep -v DONE $OUT/$f 2>/dev/null | cut -c1-230 || echo "$m [$u] no result"
done; done
