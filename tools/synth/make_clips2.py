"""Harder fixtures (test inputs only, short). D cold start (person in frame from frame 0), E never-revealed person + crowd,
F low light (noise, 50 Hz flicker), H1 handheld pan with people (camera moves over a larger canvas; truth by offsets)."""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, os.path.dirname(__file__))
import make_clips as M
from make_clips import *   # bg, people, sprite, paste, path, rng, W, H, FPS, ROOT

def writer(vid, w, h):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', str(FPS), '-i', '-',
                             '-c:v', 'libx264', '-crf', '10', '-pix_fmt', 'yuv420p', vid], stdin=subprocess.PIPE)

def make_still(name, secs, people_n, holds, noise=1.2, flicker=0.0):
    n = FPS * secs
    paths = [path(n, holds.get(i)) for i in range(people_n)]
    sps = [sprite(i * 3 + 1, int(H * rng.uniform(0.55, 0.75))) for i in range(people_n)]
    ff = writer(f'{ROOT}/{name}.mp4', W, H)
    masks = np.zeros((n, H // 4, W // 4), bool)
    for t in range(n):
        f = bg.copy(); m = np.zeros((H, W), bool)
        for pi in range(people_n):
            x, y, lift = paths[pi][t]
            paste(f, m, sps[pi], x, y)
        g = 1 + flicker * np.sin(2 * np.pi * 100 * t / FPS / 3.3)       # aliased 100 Hz mains flicker
        f = np.clip(f.astype(np.float32) * g + rng.normal(0, noise, f.shape), 0, 255).astype(np.uint8)
        masks[t] = cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA) > 0
        ff.stdin.write(f.tobytes())
    ff.stdin.close(); ff.wait()
    np.savez_compressed(f'{ROOT}/{name}_masks.npz', masks=masks); cv2.imwrite(f'{ROOT}/{name}_truth.png', bg)
    print('wrote', name, file=sys.stderr)

def make_pan(name, secs, people_n):
    n = FPS * secs
    canvas = cv2.resize(cv2.imread(f'{ROOT}/frames/5364_1.2.png')[0:768, 0:1080], None, fx=1.6, fy=1.6, interpolation=cv2.INTER_CUBIC)
    CH, CW = canvas.shape[:2]
    cx0, cy0 = (CW - W) // 2, (CH - H) // 2
    paths = [path(n) for _ in range(people_n)]
    sps = [sprite(i * 3 + 2, int(H * rng.uniform(0.5, 0.7))) for i in range(people_n)]
    ff = writer(f'{ROOT}/{name}.mp4', W, H)
    masks = np.zeros((n, H // 4, W // 4), bool); offs = []
    ph = rng.uniform(0, 6.28, 4)
    for t in range(n):
        ox = int(cx0 + 280 * np.sin(2 * np.pi * t / n * 1.3 + ph[0]) + 40 * np.sin(t * 0.21 + ph[1]))
        oy = int(cy0 + 160 * np.sin(2 * np.pi * t / n * 0.9 + ph[2]) + 25 * np.sin(t * 0.17 + ph[3]))
        ox = int(np.clip(ox, 0, CW - W)); oy = int(np.clip(oy, 0, CH - H)); offs.append((ox, oy))
        f = canvas[oy:oy + H, ox:ox + W].copy(); m = np.zeros((H, W), bool)
        for pi in range(people_n):
            x, y, lift = paths[pi][t]; paste(f, m, sps[pi], x, y)
        f = np.clip(f.astype(np.float32) + rng.normal(0, 1.2, f.shape), 0, 255).astype(np.uint8)
        masks[t] = cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA) > 0
        ff.stdin.write(f.tobytes())
    ff.stdin.close(); ff.wait()
    np.savez_compressed(f'{ROOT}/{name}_masks.npz', masks=masks)
    cv2.imwrite(f'{ROOT}/{name}_canvas.png', canvas); json.dump(offs, open(f'{ROOT}/{name}_offsets.json', 'w'))
    print('wrote', name, file=sys.stderr)

if __name__ == '__main__' and len(sys.argv) == 1:
    make_still('synthD_coldstart', 8, 2, {0: (0, 90)})
    make_still('synthE_crowd_hold', 8, 3, {0: (0, 240)})
    make_still('synthF_lowlight', 8, 2, {0: (60, 150)}, noise=5.0, flicker=0.03)
    make_pan('synthH_pan', 8, 2)


def rot_y(a): c, s_ = np.cos(a), np.sin(a); return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]])
def rot_x(a): c, s_ = np.cos(a), np.sin(a); return np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]])

def pan_maps(R, canvas_shape, f, W=1080, H=768):
    """remap coordinates: for each frame pixel, where it lands in the canvas (reference camera, R_ref = I). Same maths as the shader."""
    CH, CW = canvas_shape[:2]
    xs, ys = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    dc = np.stack([(xs - W / 2) / f, -(ys - H / 2) / f, -np.ones_like(xs)], -1)
    dc /= np.linalg.norm(dc, axis=-1, keepdims=True)
    d = dc @ R.T                                   # world dir = R @ dc
    u = f * d[..., 0] / -d[..., 2] + CW / 2
    v = CH / 2 - f * d[..., 1] / -d[..., 2]
    return u.astype(np.float32), v.astype(np.float32)

def make_pan2(name, secs, people_n, yaw_deg=16, pitch_deg=9):
    n = FPS * secs
    f = (W / 2) / np.tan(np.deg2rad(63) / 2)
    base = cv2.imread(f'{ROOT}/frames/5364_1.2.png')[0:768, 0:1080]
    canvas = cv2.resize(base, None, fx=2.6, fy=2.6, interpolation=cv2.INTER_CUBIC)
    paths = [path(n) for _ in range(people_n)]
    sps = [sprite(i * 3 + 2, int(H * rng.uniform(0.5, 0.7))) for i in range(people_n)]
    ff = writer(f'{ROOT}/{name}.mp4', W, H)
    masks = np.zeros((n, H // 4, W // 4), bool); poses = []
    ph = rng.uniform(0, 6.28, 4)
    for t in range(n):
        yaw = np.deg2rad(yaw_deg * np.sin(2 * np.pi * t / n * 1.3 + ph[0]) + 2.5 * np.sin(t * 0.21 + ph[1]))
        pitch = np.deg2rad(pitch_deg * np.sin(2 * np.pi * t / n * 0.9 + ph[2]) + 1.5 * np.sin(t * 0.17 + ph[3]))
        R = rot_y(yaw) @ rot_x(pitch); poses.append(R)
        u, v = pan_maps(R, canvas.shape, f)
        fr = cv2.remap(canvas, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        m = np.zeros((H, W), bool)
        for pi in range(people_n):
            x, y, lift = paths[pi][t]; paste(fr, m, sps[pi], x, y)
        fr = np.clip(fr.astype(np.float32) + rng.normal(0, 1.2, fr.shape), 0, 255).astype(np.uint8)
        masks[t] = cv2.resize(m.astype(np.uint8), (W // 4, H // 4), interpolation=cv2.INTER_AREA) > 0
        ff.stdin.write(fr.tobytes())
    ff.stdin.close(); ff.wait()
    np.savez_compressed(f'{ROOT}/{name}_masks.npz', masks=masks); cv2.imwrite(f'{ROOT}/{name}_canvas.png', canvas)
    with open(f'{ROOT}/{name}_poses.txt', 'w') as fh:
        fh.write(f'K {f} {f} {W / 2} {H / 2}\n')
        for i, R in enumerate(poses): fh.write(f'{i} ' + ' '.join(str(x) for x in R.T.reshape(-1)) + '\n')   # columns first
    print('wrote', name, file=sys.stderr)

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == 'pan2':
    make_pan2('synthH2_pan', 8, 2)
