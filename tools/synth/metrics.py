"""Metrics for a still-camera run. usage: metrics.py TAG [clip ...]  (reads out_still/TAG/<clip>_focus.mp4, synth/<clip>_{truth.png,masks.npz})"""
import sys, os, json, subprocess, numpy as np, cv2
ROOT = os.path.expanduser('~/gdg-ideathon'); LEAK = '/private/tmp/claude-501/-Users-tohraan/2adc22dc-94de-4ecc-a367-edaeae43e5c6/scratchpad/leak'
tag = sys.argv[1]; clips = sys.argv[2:] or ['synthA_random', 'synthB_exposure', 'synthC_shadow']
res = {}
for c in clips:
    vid = f'{ROOT}/out_still/{tag}/{c}_focus.mp4'
    truth = cv2.imread(f'{ROOT}/synth/{c}_truth.png'); H, W = truth.shape[:2]
    masks = np.load(f'{ROOT}/synth/{c}_masks.npz')['masks']
    pw = int(W * (960 / max(W, H))) // 2 * 2; ph = int(H * (960 / max(W, H))) // 2 * 2
    tp = cv2.resize(truth, (pw, ph), interpolation=cv2.INTER_AREA).astype(np.float32)
    # decode frames
    p = subprocess.run(['ffmpeg', '-v', 'error', '-i', vid, '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], capture_output=True)
    ow = pw * 3; oh = len(p.stdout) // (ow * 3 * masks.shape[0]) if False else None
    info = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=height,nb_frames', '-of', 'json', vid], capture_output=True, text=True).stdout)['streams'][0]
    oh = int(info['height']); n = len(p.stdout) // (ow * oh * 3)
    fr = np.frombuffer(p.stdout, np.uint8)[:n * ow * oh * 3].reshape(n, oh, ow, 3)
    leak = np.array([list(map(int, l.split()))[1:] for l in subprocess.run([LEAK, vid, str(pw), str(ph)], capture_output=True, text=True).stdout.strip().splitlines()])
    mk = np.stack([cv2.resize(m.astype(np.uint8), (pw, ph), interpolation=cv2.INTER_NEAREST) > 0 for m in masks[:n]])
    inarea = mk.reshape(n, -1).sum(1)
    r = {}
    for k, name in [(1, 'memory'), (2, 'keyframes')]:
        pan = fr[:, :ph, k * pw:(k + 1) * pw].astype(np.float32)
        ev = inarea > 2000
        r[name] = {
            'leak_pct': round(100 * float(leak[:n, k - 1][ev].sum()) / float(inarea[ev].sum()), 2),
            'hole_mae': round(float(np.mean([np.abs(pan[t] - tp)[mk[t]].mean() for t in np.where(ev)[0]])), 2),
            'flicker': round(float(np.mean([np.abs(pan[t] - pan[t - 1])[~(mk[t] | mk[t - 1])].mean() for t in range(1, n)])), 3)}
    r['frames'] = int(n)
    res[c] = r
print(json.dumps(res, indent=1))
os.makedirs(f'{ROOT}/out_still/{tag}', exist_ok=True)
json.dump(res, open(f'{ROOT}/out_still/{tag}/metrics.json', 'w'), indent=1)
