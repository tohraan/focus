"""Synthetic still-camera clips with known truth.
Background = people-free ceiling/wall crop of a real classroom frame; people = Vision cut-outs from the real clips,
moved on random paths with jumps and a stand-still hold. Truth plate and per-frame person masks are saved."""
import cv2, numpy as np, glob, subprocess, sys, os
rng = np.random.default_rng(7)
ROOT = os.path.expanduser('~/gdg-ideathon/synth'); os.makedirs(ROOT, exist_ok=True)
FPS, SECS = 30, 12
bg = cv2.imread(f'{ROOT}/frames/5364_1.2.png')[0:768, 0:1080]
H, W = bg.shape[:2]

# person cut-outs: connected components of the matte, big enough, not touching the frame border
people = []
for f in sorted(glob.glob(f'{ROOT}/cut/*.png')):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    a = (im[..., 3] > 128).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(a)
    for i in range(1, n):
        x, y, w, h, ar = st[i]
        if ar < 20000 or h < 250 or x <= 2 or x + w >= im.shape[1] - 2: continue
        crop = im[y:y + h, x:x + w].copy()
        crop[..., 3] = np.where(lab[y:y + h, x:x + w] == i, crop[..., 3], 0)
        people.append(crop)
print('cutouts', len(people), file=sys.stderr)

def sprite(i, height):
    p = people[i % len(people)]
    s = height / p.shape[0]
    return cv2.resize(p, (max(8, int(p.shape[1] * s)), height), interpolation=cv2.INTER_AREA)

def paste(frame, mask, sp, cx, cy):
    h, w = sp.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h)          # (cx, cy) = feet position
    xs, ys = max(0, x0), max(0, y0); xe, ye = min(W, x0 + w), min(H, y0 + h)
    if xe <= xs or ye <= ys: return
    s = sp[ys - y0:ye - y0, xs - x0:xe - x0]
    a = s[..., 3:4].astype(np.float32) / 255
    frame[ys:ye, xs:xe] = (frame[ys:ye, xs:xe] * (1 - a) + s[..., :3] * a).astype(np.uint8)
    mask[ys:ye, xs:xe] |= (s[..., 3] > 60)

def path(n, hold=None):
    """feet positions: random walk with occasional jumps; `hold` = (start, end) frames standing still."""
    x, y = rng.uniform(0.25, 0.75) * W, rng.uniform(0.80, 0.98) * H
    vx, vy = rng.normal(0, 3), rng.normal(0, 1.5)
    out, jump_t = [], -999
    for t in range(n):
        if hold and hold[0] <= t < hold[1]: out.append((x, y, 0)); continue
        if rng.random() < 0.02: vx, vy = rng.normal(0, 7), rng.normal(0, 2.5)
        x += vx; y += vy
        if x < 0.1 * W or x > 0.9 * W: vx = -vx
        if y < 0.7 * H or y > 0.99 * H: vy = -vy
        x = float(np.clip(x, 0.08 * W, 0.92 * W)); y = float(np.clip(y, 0.7 * H, 0.99 * H))
        if rng.random() < 0.012 and t - jump_t > 25: jump_t = t
        dt = t - jump_t
        lift = 4 * 220 * (dt / 20) * (1 - dt / 20) if 0 <= dt <= 20 else 0       # 20-frame parabola, 220 px
        out.append((x, y - lift, lift))
    return out

def make(name, people_n=2, hold_person0=(90, 240), exposure=False, shadow=False):
    n = FPS * SECS
    paths = [path(n, hold_person0 if i == 0 else None) for i in range(people_n)]
    sps = [sprite(i * 3, int(H * rng.uniform(0.55, 0.75))) for i in range(people_n)]
    vid = f'{ROOT}/{name}.mp4'
    ff = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                           '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', vid], stdin=subprocess.PIPE)
    masks = np.zeros((n, H // 4, W // 4), bool)
    for t in range(n):
        f = bg.copy(); m = np.zeros((H, W), bool)
        for pi in range(people_n):
            x, y, lift = paths[pi][t]
            if shadow:
                sh = np.zeros((H, W), np.float32)
                cv2.ellipse(sh, (int(x), int(y + lift)), (int(sps[pi].shape[1] * 0.55), 28), 0, 0, 360, 1, -1)
                sh = cv2.GaussianBlur(sh, (0, 0), 14) * 0.45
                f = (f * (1 - sh[..., None])).astype(np.uint8)
            paste(f, m, sps[pi], x, y)
        if exposure and 240 <= t < 270:            # 1 s auto-exposure dip of 12% (whole frame)
            f = (f.astype(np.float32) * (1 - 0.12 * (t - 240) / 30)).astype(np.uint8)
        elif exposure and t >= 270:
            f = (f.astype(np.float32) * 0.88).astype(np.uint8)
        f = np.clip(f.astype(np.float32) + rng.normal(0, 1.2, f.shape), 0, 255).astype(np.uint8)   # sensor noise
        masks[t] = cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA) > 0
        ff.stdin.write(f.tobytes())
    ff.stdin.close(); ff.wait()
    np.savez_compressed(f'{ROOT}/{name}_masks.npz', masks=masks)
    cv2.imwrite(f'{ROOT}/{name}_truth.png', bg)
    print('wrote', vid, file=sys.stderr)

if __name__ == '__main__':
    make('synthA_random')
    make('synthB_exposure', exposure=True)
    make('synthC_shadow', shadow=True)
