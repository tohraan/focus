"""Scores a run of still frames (no video). usage: metrics_stills.py TAG [clip ...]
Reads out_still/TAG/<clip>_frames/{mem,kf}_NNNN.jpg, synth/<clip>_{truth.png,masks.npz}.
Per engine: leak_pct (person pixels Vision still finds in the output / true person pixels), hole_mae (error vs truth inside the person region),
bg_mae (error vs truth outside it: lower = cleaner), flicker (change between consecutive sampled frames outside the person), score (lower is better)."""
import sys, os, json, glob, subprocess, numpy as np, cv2
ROOT = os.path.expanduser('~/gdg-ideathon')
import os
LEAK = os.environ.get('LEAK_BIN', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'leak_img'))
tag = sys.argv[1]; clips = sys.argv[2:] or ['synthA_random', 'synthB_exposure', 'synthC_shadow']
res = {}
for c in clips:
    d = f'{ROOT}/out_still/{tag}/{c}_frames'
    if not os.path.isdir(d): print('missing', d); continue
    pan = os.path.exists(f'{ROOT}/synth/{c}_offsets.json') or os.path.exists(f'{ROOT}/synth/{c}_poses.txt')
    posed = os.path.exists(f'{ROOT}/synth/{c}_poses.txt')
    if pan:
        canvas = cv2.imread(f'{ROOT}/synth/{c}_canvas.png').astype(np.float32)
        H, W = 768, 1080
        if posed:
            L = open(f'{ROOT}/synth/{c}_poses.txt').read().strip().split('\n'); Kf = float(L[0].split()[1])
            Rs = [np.array(list(map(float, l.split()[1:]))).reshape(3, 3).T for l in L[1:]]
            def truth_at(i):
                xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
                dc = np.stack([(xs - W / 2) / Kf, -(ys - H / 2) / Kf, -np.ones_like(xs)], -1); dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
                d = dc @ Rs[i].T; CH, CW = canvas.shape[:2]
                u = (Kf * d[..., 0] / -d[..., 2] + CW / 2).astype(np.float32); v = (CH / 2 - Kf * d[..., 1] / -d[..., 2]).astype(np.float32)
                return cv2.remap(canvas, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        else:
            offs = json.load(open(f'{ROOT}/synth/{c}_offsets.json'))
    else:
        truth = cv2.imread(f'{ROOT}/synth/{c}_truth.png').astype(np.float32); H, W = truth.shape[:2]
    masks = np.load(f'{ROOT}/synth/{c}_masks.npz')['masks']
    out = {}
    for eng, pre in [('memory', 'mem'), ('keyframes', 'kf')]:
        files = sorted(glob.glob(f'{d}/{pre}_*.jpg'))
        idx = [int(os.path.basename(f)[len(pre) + 1:-4]) for f in files]
        leak_lines = subprocess.run([LEAK, d], capture_output=True, text=True).stdout.split('\n') if eng == 'memory' else leak_lines
        leakmap = {l.split()[0]: int(l.split()[1]) for l in leak_lines if l.strip()}
        lk, hm, bm, fl, prev = [], [], [], [], None
        rv, gn, eg = [], [], []
        always = np.ones(masks.shape[1:], bool)         # pixels masked in EVERY frame so far = never observed (still clips only)
        always_by_frame = []
        for k in range(len(masks)):
            always = always & masks[k]; always_by_frame.append(always.copy())
        for f, i in zip(files, idx):
            im = cv2.imread(f).astype(np.float32)
            if im.shape[:2] != (H, W): im = cv2.resize(im, (W, H))
            m = cv2.resize(masks[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
            area = int(m.sum())
            tr = (truth_at(i) if posed else canvas[offs[i][1]:offs[i][1] + H, offs[i][0]:offs[i][0] + W]) if pan else truth
            e = np.abs(im - tr).mean(2)
            if area > 3000:
                lk.append((leakmap.get(os.path.basename(f), 0), area)); hm.append(e[m].mean())
                if not pan:
                    nv = cv2.resize(always_by_frame[i].astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
                    if (m & ~nv).sum() > 500: rv.append(e[m & ~nv].mean())
                    if (m & nv).sum() > 500: gn.append(e[m & nv].mean())
            bm.append(e[~m].mean())
            if area > 3000:
                dist = cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 3); band = (dist > 0) & (dist <= 14)
                if band.sum() > 200: eg.append(e[band].mean())
            sig = im - tr                                   # signed error vs truth: camera motion cancels, only real flicker remains
            if prev is not None: fl.append(np.abs(sig - prev[0]).mean(2)[~(m | prev[1])].mean())
            prev = (sig, m)
        r = {'leak_pct': round(100 * sum(a for a, _ in lk) / max(1, sum(b for _, b in lk)), 2),
             'hole_mae': round(float(np.mean(hm)), 2), 'reveal_mae': round(float(np.mean(rv)), 2) if rv else None, 'gen_mae': round(float(np.mean(gn)), 2) if gn else None, 'bg_mae': round(float(np.mean(bm)), 2), 'edge_mae': round(float(np.mean(eg)), 2) if eg else None, 'flicker': round(float(np.mean(fl)), 3)}
        core = r['reveal_mae'] if r['reveal_mae'] is not None else r['hole_mae']
        r['score'] = round(r['leak_pct'] + core / 2 + (r['gen_mae'] or 0) / 12 + 2 * r['bg_mae'] + 10 * r['flicker'], 2)
        out[eng] = r
    res[c] = out
tot = {e: round(float(np.mean([res[c][e]['score'] for c in res])), 2) for e in ('memory', 'keyframes')}
res['MEAN_SCORE'] = tot
print(json.dumps(res, indent=1))
json.dump(res, open(f'{ROOT}/out_still/{tag}/metrics.json', 'w'), indent=1)
