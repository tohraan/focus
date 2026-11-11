#!/bin/zsh
# Usage: tools/run_all.sh ITER   (builds once, runs all 7 clips in 4 parallel groups on the Mac, scores them, prints a table)
cd "$(dirname $0)/.."
IT=$1
xcodebuild -project Focus.xcodeproj -scheme FocusBench -configuration Release -destination 'platform=macOS,arch=arm64' build 2>&1 | grep -E "error:|BUILD FAILED" | sort -u | head -n 8
export NOBUILD=1
LOOP_ARGS="-still" tools/loop_mac.sh ${IT}_g1 synthA_random synthB_exposure > /dev/null 2>&1 &
LOOP_ARGS="-still" tools/loop_mac.sh ${IT}_g2 synthC_shadow synthD_coldstart > /dev/null 2>&1 &
LOOP_ARGS="-still" tools/loop_mac.sh ${IT}_g3 synthE_crowd_hold synthF_lowlight > /dev/null 2>&1 &
LOOP_ARGS="-nowarm" tools/loop_mac.sh ${IT}_g4 synthH2_pan > /dev/null 2>&1 &
wait
OUT=out_still/$IT; rm -rf $OUT; mkdir -p $OUT
for g in g1 g2 g3 g4; do for d in out_still/${IT}_$g/*_frames; do cp -R $d $OUT/; done; done
python3 tools/synth/metrics_stills.py $IT synthA_random synthB_exposure synthC_shadow synthD_coldstart synthE_crowd_hold synthF_lowlight synthH2_pan 2>&1 | python3 -c "
import sys,json; d=json.load(sys.stdin)
print('%-18s %-9s %7s %9s %7s %8s %7s' % ('clip','engine','leak%','hole_mae','bg_mae','flicker','score'))
for c,v in d.items():
    if c=='MEAN_SCORE': print('MEAN SCORE (lower is better):', v); continue
    for e,m in v.items(): print('%-18s %-9s %7.2f %9.2f %7.2f %8.3f %7.2f' % (c, e, m['leak_pct'], m['hole_mae'], m['bg_mae'], m['flicker'], m['score']))
"
