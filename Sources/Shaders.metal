#include <metal_stdlib>
using namespace metal;

constexpr sampler linClamp(filter::linear, address::clamp_to_edge);
constexpr sampler linPano(filter::linear, address::repeat);

struct RenderUniforms {
    float3x3 viewToImage;   // viewport uv -> camera image uv
    float3x3 camToWorld;
    float fx, fy, cx, cy;
    float2 viewSize;
    float2 imageSize;
    int mode;               // 0 processed, 1 original, 2 split, 3 mask overlay
    float splitX;
    float4 gain;            // per-channel brightness match, memory -> live exposure
    int still;              // 1 = still-camera background plate is valid this frame
};

struct PanoUniforms {
    float3x3 camToWorld;
    float fx, fy, cx, cy;
    float2 imageSize;
    float maskRadius;       // in image uv units
    float blend;
};

struct VOut { float4 pos [[position]]; float2 uv; };

vertex VOut fullscreenVS(uint vid [[vertex_id]]) {
    float2 p = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(p * 2.0 - 1.0, 0, 1);
    o.uv = float2(p.x, 1.0 - p.y);
    return o;
}

static float3 ycbcr(texture2d<float> Y, texture2d<float> C, float2 uv) {
    float y = Y.sample(linClamp, uv).r;
    float2 c = C.sample(linClamp, uv).rg - 0.5;
    return saturate(float3(y + 1.402 * c.y,
                           y - 0.344136 * c.x - 0.714136 * c.y,
                           y + 1.772 * c.x));
}

static float2 dirToPano(float3 d) {
    float lon = atan2(d.x, -d.z);
    float lat = asin(clamp(d.y, -1.0, 1.0));
    return float2(lon / (2.0 * M_PI_F) + 0.5, 0.5 - lat / M_PI_F);
}
static float3 panoToDir(float2 uv) {
    float lon = (uv.x - 0.5) * 2.0 * M_PI_F;
    float lat = (0.5 - uv.y) * M_PI_F;
    return float3(cos(lat) * sin(lon), sin(lat), -cos(lat) * cos(lon));
}

// YOLO texture (320x240, RG): R = people Vision missed (shown), G = all detections (memory veto only).
static float yoloVeto(texture2d<float> yolo, float2 uv) {
    float y = 0;
    float2 t = float2(1.5 / 320.0, 1.5 / 240.0);
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            y = max(y, yolo.sample(linClamp, uv + float2(dx, dy) * t).g);
    return y;
}

kernel void updatePano(texture2d<float> Y [[texture(0)]],
                       texture2d<float> C [[texture(1)]],
                       texture2d<float> mask [[texture(2)]],
                       texture2d<float, access::read_write> pano [[texture(3)]],
                       texture2d<float> yolo [[texture(4)]],
                       constant PanoUniforms& u [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint W = pano.get_width(), H = pano.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float3 dw = panoToDir((float2(gid) + 0.5) / float2(W, H));
    float3 dc = transpose(u.camToWorld) * dw;
    if (dc.z > -0.05) return;
    float px = u.fx * (dc.x / -dc.z) + u.cx;
    float py = u.cy - u.fy * (dc.y / -dc.z);
    float2 uv = float2(px, py) / u.imageSize;
    if (uv.x < 0.02 || uv.x > 0.98 || uv.y < 0.02 || uv.y > 0.98) return;
    float r = u.maskRadius;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            if (mask.sample(linClamp, uv + float2(dx, dy) * r).r > 0.1) return;
    if (yoloVeto(yolo, uv) > 0.3) return;                       // a person only YOLO saw: never learn it
    float3 c = ycbcr(Y, C, uv);
    float4 old = pano.read(gid);
    float3 n = old.a > 0.9 ? mix(old.rgb, c, u.blend) : c;      // a=1 observed, 0.6 generated (real data replaces it)
    pano.write(float4(n, 1.0), gid);
}

// Exposure match: compares live vs remembered colour on a sparse grid of unmasked, known pixels.
kernel void gainStats(texture2d<float> Y [[texture(0)]],
                      texture2d<float> C [[texture(1)]],
                      texture2d<float> mask [[texture(2)]],
                      texture2d<float> pano [[texture(3)]],
                      texture2d<float> yolo [[texture(4)]],
                      constant PanoUniforms& u [[buffer(0)]],
                      device atomic_uint* stats [[buffer(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    float2 uv = (float2(gid) + 0.5) / float2(64, 48) * 0.9 + 0.05;
    if (mask.sample(linClamp, uv).r > 0.05 || yoloVeto(yolo, uv) > 0.05) return;
    float2 p = uv * u.imageSize;
    float3 dc = normalize(float3((p.x - u.cx) / u.fx, -(p.y - u.cy) / u.fy, -1.0));
    float4 mem = pano.sample(linPano, dirToPano(u.camToWorld * dc));
    if (mem.a < 0.9) return;
    float3 live = ycbcr(Y, C, uv);
    atomic_fetch_add_explicit(&stats[0], uint(live.r * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[1], uint(live.g * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[2], uint(live.b * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[3], uint(mem.r * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[4], uint(mem.g * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[5], uint(mem.b * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[6], 1u, memory_order_relaxed);
}

kernel void premultFill(texture2d<float> Y [[texture(0)]],
                        texture2d<float> C [[texture(1)]],
                        texture2d<float> mask [[texture(2)]],
                        texture2d<float> yolo [[texture(3)]],
                        texture2d<float, access::write> outTex [[texture(4)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint W = outTex.get_width(), H = outTex.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);
    float m = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            m = max(m, mask.sample(linClamp, uv + float2(dx, dy) * (3.0 / float2(W, H))).r);
    m = max(m, yolo.sample(linClamp, uv).r);
    float w = 1.0 - smoothstep(0.1, 0.4, m);
    outTex.write(float4(ycbcr(Y, C, uv) * w, w), gid);
}

static float3 fillColor(texture2d<float> fill, float2 uv) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    float3 col = float3(0.5);
    for (int L = 1; L <= 8; L++) {
        float4 t = fill.sample(s, uv, level(float(L)));
        if (t.a > 0.25) { col = t.rgb / t.a; break; }
    }
    return col;
}


// ================= Still-camera background plate =================
// Camera does not move: keep a per-pixel clean background (plate) learned from pixels that are outside every person
// mask AND stable over time. Foreground = person masks, plus anything that differs from the plate next to a person
// (arms/legs the matte missed, shadows). Works at 1/3 resolution in image space (no pose needed).
constant float3 LW = float3(0.299, 0.587, 0.114);

struct StillU {
    float4 gain;            // live/plate brightness ratio per channel (smoothed on CPU)
    float2 imageSize;       // full-res camera image
    float learn;            // 0 while the table is shaking: use the plate, do not update it
};

static float3 liveAt(texture2d<float> Y, texture2d<float> C, float2 uv, float2 px) {
    float3 a = ycbcr(Y, C, uv + float2(-0.25, -0.25) * px) + ycbcr(Y, C, uv + float2(0.25, -0.25) * px)
             + ycbcr(Y, C, uv + float2(-0.25, 0.25) * px) + ycbcr(Y, C, uv + float2(0.25, 0.25) * px);
    return a * 0.25;
}

static float max3(float3 v) { return max(v.x, max(v.y, v.z)); }

// flags (r8Uint): 1 person (shown), 2 person (blocks learning), 4 big difference from plate, 8 shadow-like, 16 moving
kernel void stillClassify(texture2d<float> Y [[texture(0)]],
                          texture2d<float> C [[texture(1)]],
                          texture2d<float> mask [[texture(2)]],
                          texture2d<float> yolo [[texture(3)]],
                          texture2d<float> plate [[texture(4)]],
                          texture2d<float> lastT [[texture(5)]],
                          texture2d<uint, access::write> cls [[texture(6)]],
                          constant StillU& u [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    uint W = plate.get_width(), H = plate.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 px = 1.0 / float2(W, H);
    float2 uv = (float2(gid) + 0.5) * px;
    float3 live = liveAt(Y, C, uv, px);

    float show = 0, learn = 0;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            if (mask.sample(linClamp, uv + float2(dx, dy) * 3.0 / u.imageSize).r > 0.3) show = 1;
    if (yolo.sample(linClamp, uv).r > 0.4) show = 1;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            if (mask.sample(linClamp, uv + float2(dx, dy) * 7.0 / u.imageSize).r > 0.1) learn = 1;
    if (yoloVeto(yolo, uv) > 0.3) learn = 1;
    learn = max(learn, show);

    float3 last = lastT.read(gid).rgb;
    bool moving = max3(abs(live - last)) > 0.06;

    float4 pl = plate.read(gid);
    float3 padj = pl.rgb * u.gain.rgb;
    float ad = max3(abs(live - padj));
    float lL = dot(live, LW), lP = dot(padj, LW);
    bool known = pl.a > 0.5;
    bool big = known && ad > 0.14;
    float3 cl = live / max(lL, 1e-3), cp = padj / max(lP, 1e-3);
    bool shadow = known && lP > 0.08 && lL < lP * 0.94 && lL > lP * 0.30 && ad > 0.04 && max3(abs(cl - cp)) < 0.18;

    uint f = (show > 0 ? 1u : 0u) | (learn > 0 ? 2u : 0u) | (big ? 4u : 0u) | (shadow ? 8u : 0u) | (moving ? 16u : 0u);
    cls.write(uint4(f, 0, 0, 0), gid);
}

kernel void stillResolve(texture2d<uint> cls [[texture(0)]],
                         texture2d<float> lastT [[texture(1)]],
                         texture2d<float> fgPrev [[texture(2)]],
                         texture2d<float, access::write> fgRaw [[texture(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    int W = int(cls.get_width()), H = int(cls.get_height());
    if (int(gid.x) >= W || int(gid.y) >= H) return;
    uint own = cls.read(gid).r;
    bool show = own & 1u, big = own & 4u, shadow = own & 8u, moving = own & 16u;
    float run = lastT.read(gid).a;
    bool fg = show;
    if (!fg && (big || shadow)) {
        // distance (plate px) to the nearest shown person pixel, searched on rings; previous foreground also counts for 'big'
        float nearD = 99;
        for (int r = 4; r <= 20; r += 4) {
            for (int i = 0; i < 8; i++) {
                float a = float(i) * 0.785398;
                int2 q = clamp(int2(gid) + int2(round(cos(a) * float(r)), round(sin(a) * float(r))), int2(0), int2(W - 1, H - 1));
                if (cls.read(uint2(q)).r & 1u) { nearD = min(nearD, float(r)); }
            }
            if (nearD < 99) break;
        }
        bool nearPrev = false;
        for (int i = 0; i < 8; i++) {
            float a = float(i) * 0.785398;
            int2 q = clamp(int2(gid) + int2(round(cos(a) * 3.0), round(sin(a) * 3.0)), int2(0), int2(W - 1, H - 1));
            if (fgPrev.read(uint2(q)).r > 0.5) nearPrev = true;
        }
        if (big && run < 90 && (nearD <= 12 || nearPrev)) fg = true;      // matte missed part of the body
        else if (big && moving) fg = true;                                // moving thing the matte missed entirely
        else if (shadow && nearD <= 20) fg = true;                        // shadow next to a person
    }
    fgRaw.write(fg ? 1.0 : 0.0, gid);
}

kernel void stillCommit(texture2d<float> Y [[texture(0)]],
                        texture2d<float> C [[texture(1)]],
                        texture2d<float> fgRaw [[texture(2)]],
                        texture2d<uint> cls [[texture(3)]],
                        texture2d<float, access::read_write> lastT [[texture(4)]],
                        texture2d<float, access::read_write> plate [[texture(5)]],
                        texture2d<float, access::write> fgOut [[texture(6)]],
                        texture2d<float, access::write> disp [[texture(7)]],
                        texture2d<float, access::write> off [[texture(8)]],
                        constant StillU& u [[buffer(0)]],
                        device atomic_uint* stats [[buffer(1)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint lane [[thread_index_in_simdgroup]]) {
    int W = int(plate.get_width()), H = int(plate.get_height());
    bool inside = int(gid.x) < W && int(gid.y) < H;
    float3 statLive = 0, statPlate = 0; float statN = 0, statK = 0;
    if (inside) {
        float2 px = 1.0 / float2(W, H);
        float2 uv = (float2(gid) + 0.5) * px;
        float3 live = liveAt(Y, C, uv, px);
        float cnt = 0;
        for (int dy = -2; dy <= 2; dy++)
            for (int dx = -2; dx <= 2; dx++) {
                int2 q = clamp(int2(gid) + int2(dx, dy), int2(0), int2(W - 1, H - 1));
                cnt += fgRaw.read(uint2(q)).r;
            }
        bool fgAny = cnt > 0.5;
        fgOut.write(smoothstep(0.05, 0.5, cnt / 25.0), gid);

        uint own = cls.read(gid).r;
        bool learnBlock = (own & 2u) || (own & 16u) || fgAny;
        float4 lv = lastT.read(gid);
        float run = (own & 16u) ? 0.0 : min(lv.a + 1.0, 255.0);
        float3 g = clamp(u.gain.rgb, 0.6, 1.6);
        float4 pl = plate.read(gid);
        if (!learnBlock && u.learn > 0.5) {
            if (pl.a < 0.01 || fabs(pl.a - 0.51) < 0.01) {                       // never seen, or only generated (MI-GAN): real data replaces it
                if (run >= 8) pl = float4(saturate(live / g), 0.3);                 // first sight of this background pixel
            } else if (run >= 4) {
                bool absorb = (own & 4u) && run >= 90;                             // scene really changed (moved chair)
                float rate = absorb ? 0.25 : 0.03;
                pl.rgb += rate * (saturate(live / g) - pl.rgb);
                pl.a = min(1.0, pl.a + 0.08);
            }
            plate.write(pl, gid);
        }
        lastT.write(float4(live, run), gid);
        float3 pd = saturate(pl.rgb * g);
        disp.write(float4(pd, pl.a), gid);
        bool valid = pl.a > 0.5 && !fgAny && !(own & 2u) && !(own & 16u);
        off.write(valid ? float4(clamp(live - pd, -0.12, 0.12), 1.0) : float4(0.0), gid);
        if (pl.a > 0.5) statK = 1;
        if (valid && run >= 4) { statLive = live; statPlate = pl.rgb; statN = 1; }
    }
    // exposure statistics: one atomic per SIMD group
    float3 sl = simd_sum(statLive), sp = simd_sum(statPlate); float sn = simd_sum(statN), sk = simd_sum(statK);
    if (lane == 0 && sk > 0) atomic_fetch_add_explicit(&stats[7], uint(sk), memory_order_relaxed);
    if (lane == 0 && sn > 0) {
        atomic_fetch_add_explicit(&stats[0], uint(sl.x * 256.0), memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[1], uint(sl.y * 256.0), memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[2], uint(sl.z * 256.0), memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[3], uint(sp.x * 256.0), memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[4], uint(sp.y * 256.0), memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[5], uint(sp.z * 256.0), memory_order_relaxed);
        atomic_fetch_add_explicit(&stats[6], uint(sn), memory_order_relaxed);
    }
}

kernel void stillClear(texture2d<float, access::write> a [[texture(0)]],
                       texture2d<float, access::write> b [[texture(1)]],
                       texture2d<float, access::write> c [[texture(2)]],
                       texture2d<float, access::write> d [[texture(3)]],
                       texture2d<float, access::write> e [[texture(4)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= a.get_width() || gid.y >= a.get_height()) return;
    a.write(float4(0), gid); b.write(float4(0), gid); c.write(float4(0), gid); d.write(float4(0), gid); e.write(float4(0), gid);
}


// ---- MI-GAN fill for background the plate has never seen (person region with unknown plate) ----
// Builds the 512x512 inputs (image 0..255 CHW, mask 255 = keep / 0 = hole) from the live frame, the plate and the foreground mask.
kernel void miganPrep(texture2d<float> Y [[texture(0)]],
                      texture2d<float> C [[texture(1)]],
                      texture2d<float> disp [[texture(2)]],
                      texture2d<float> fg [[texture(3)]],
                      texture2d<float, access::write> holeTex [[texture(4)]],
                      device float* imgBuf [[buffer(0)]],
                      device float* maskBuf [[buffer(1)]],
                      device atomic_uint* count [[buffer(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint N = 512;
    if (gid.x >= N || gid.y >= N) return;
    float2 uv = (float2(gid) + 0.5) / float(N);
    float3 live = ycbcr(Y, C, uv);
    float f = fg.sample(linClamp, uv).r;
    float4 p = disp.sample(linClamp, uv);
    bool inPerson = f > 0.08;
    bool hole = inPerson && p.a < 0.5;
    float3 col = (inPerson && !hole) ? p.rgb : live;
    uint i = gid.y * N + gid.x;
    imgBuf[i] = col.r * 255.0; imgBuf[N * N + i] = col.g * 255.0; imgBuf[2 * N * N + i] = col.b * 255.0;
    maskBuf[i] = hole ? 0.0 : 255.0;
    holeTex.write(hole ? 1.0 : 0.0, gid);
    if (hole) atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}

// Writes the generated pixels into the plate (confidence 0.51 = generated; real observations replace them).
kernel void miganCommit(texture2d<float> holeTex [[texture(0)]],
                        texture2d<float> gen [[texture(1)]],
                        texture2d<float, access::read_write> plate [[texture(2)]],
                        constant float4& gain [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]]) {
    uint W = plate.get_width(), H = plate.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);
    constexpr sampler nearest(filter::nearest, address::clamp_to_edge);
    if (holeTex.sample(nearest, uv).r < 0.5) return;
    float4 pl = plate.read(gid);
    if (pl.a >= 0.5) return;
    float3 g = clamp(gain.rgb, 0.6, 1.6);
    float3 c = gen.sample(linClamp, uv).rgb;
    plate.write(float4(saturate(c / g), 0.51), gid);
}

// Offset diffusion: low-frequency (live - plate) measured on visible background, spread into the hole so no seam shows.
static float3 stillOffset(texture2d<float> off, float2 uv) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    for (int L = 0; L <= 6; L++) {
        float4 t = off.sample(s, uv, level(float(L)));
        if (t.a > 0.15) return t.rgb / t.a;
    }
    return float3(0);
}

// Replacement colour for a masked pixel when the still plate is valid; falls back to `engineRepl` where the plate has not seen the pixel.
static float3 stillReplacement(texture2d<float> disp, texture2d<float> off, float2 uv, float3 engineRepl) {
    float4 p = disp.sample(linClamp, uv);
    float k = smoothstep(0.35, 0.5, p.a);
    float3 plateCol = saturate(p.rgb + stillOffset(off, uv));
    return mix(engineRepl, plateCol, k);
}

fragment float4 compositeFS(VOut in [[stage_in]],
                            texture2d<float> Y [[texture(0)]],
                            texture2d<float> C [[texture(1)]],
                            texture2d<float> mask [[texture(2)]],
                            texture2d<float> pano [[texture(3)]],
                            texture2d<float> yolo [[texture(4)]],
                            texture2d<float> fill [[texture(5)]],
                            texture2d<float> stDisp [[texture(8)]],
                            texture2d<float> stOff [[texture(9)]],
                            texture2d<float> stFg [[texture(10)]],
                            constant RenderUniforms& u [[buffer(0)]]) {
    float2 iuv = (u.viewToImage * float3(in.uv, 1.0)).xy;
    float3 live = ycbcr(Y, C, iuv);
    float m = mask.sample(linClamp, iuv).r;
    float2 t = 3.0 / u.imageSize;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            m = max(m, mask.sample(linClamp, iuv + float2(dx, dy) * t * 2.0).r);
    m = max(m, yolo.sample(linClamp, iuv).r);          // only people Vision missed, no dilation
    m = smoothstep(0.25, 0.6, m);
    if (u.still != 0) m = max(m, stFg.sample(linClamp, iuv).r);

    float3 processed = live;
    if (m > 0.0) {
        float2 p = iuv * u.imageSize;
        float3 dc = normalize(float3((p.x - u.cx) / u.fx, -(p.y - u.cy) / u.fy, -1.0));
        float4 mem = pano.sample(linPano, dirToPano(u.camToWorld * dc));
        float3 repl = mem.a > 0.5 ? saturate(mem.rgb * u.gain.rgb) : fillColor(fill, iuv);
        if (u.still != 0) repl = stillReplacement(stDisp, stOff, iuv, repl);
        processed = mix(live, repl, m);
    }

    float3 outc;
    if (u.mode == 1) outc = live;
    else if (u.mode == 3) outc = mix(live, float3(1, 0.1, 0.1), m * 0.55);
    else if (u.mode == 2) outc = in.uv.x < u.splitX ? live : processed;
    else outc = processed;
    if (u.mode == 2 && fabs(in.uv.x - u.splitX) < 0.002) outc = float3(1);
    return float4(outc, 1.0);
}

// ---- AutoFill (LaMa) support ----
// Builds the 512x512 LaMa inputs: image (CHW float, remembered colour under people where known) and hole mask
// (person region whose background was never observed).
kernel void prepLama(texture2d<float> Y [[texture(0)]],
                     texture2d<float> C [[texture(1)]],
                     texture2d<float> mask [[texture(2)]],
                     texture2d<float> yolo [[texture(3)]],
                     texture2d<float> pano [[texture(4)]],
                     texture2d<float, access::write> holeTex [[texture(5)]],
                     device float* imgBuf [[buffer(0)]],
                     device float* maskBuf [[buffer(1)]],
                     constant PanoUniforms& u [[buffer(2)]],
                     constant float4& gain [[buffer(3)]],
                     device atomic_uint* count [[buffer(4)]],
                     uint2 gid [[thread_position_in_grid]]) {
    const uint N = 512;
    if (gid.x >= N || gid.y >= N) return;
    float2 uv = (float2(gid) + 0.5) / float(N);
    float p = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            p = max(p, mask.sample(linClamp, uv + float2(dx, dy) * (3.0 / 512.0)).r);
    p = max(p, yolo.sample(linClamp, uv).r);
    float3 live = ycbcr(Y, C, uv);
    float2 pp = uv * u.imageSize;
    float3 dc = normalize(float3((pp.x - u.cx) / u.fx, -(pp.y - u.cy) / u.fy, -1.0));
    float4 mem = pano.sample(linPano, dirToPano(u.camToWorld * dc));
    bool known = mem.a > 0.5;
    bool hole = p > 0.3 && !known;
    float3 col = (p > 0.3 && known) ? saturate(mem.rgb * gain.rgb) : live;
    uint idx = gid.y * N + gid.x;
    imgBuf[idx] = col.r; imgBuf[N * N + idx] = col.g; imgBuf[2 * N * N + idx] = col.b;
    maskBuf[idx] = hole ? 1.0 : 0.0;
    holeTex.write(float4(hole ? 1.0 : 0.0, 0, 0, 0), gid);
    if (hole) atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}


// Same as prepLama but in MI-GAN's convention: image 0..255, mask 255 = keep / 0 = hole.
kernel void prepMigan(texture2d<float> Y [[texture(0)]],
                      texture2d<float> C [[texture(1)]],
                      texture2d<float> mask [[texture(2)]],
                      texture2d<float> yolo [[texture(3)]],
                      texture2d<float> pano [[texture(4)]],
                      texture2d<float, access::write> holeTex [[texture(5)]],
                      device float* imgBuf [[buffer(0)]],
                      device float* maskBuf [[buffer(1)]],
                      constant PanoUniforms& u [[buffer(2)]],
                      constant float4& gain [[buffer(3)]],
                      device atomic_uint* count [[buffer(4)]],
                      uint2 gid [[thread_position_in_grid]]) {
    const uint N = 512;
    if (gid.x >= N || gid.y >= N) return;
    float2 uv = (float2(gid) + 0.5) / float(N);
    float p = 0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++)
            p = max(p, mask.sample(linClamp, uv + float2(dx, dy) * (3.0 / 512.0)).r);
    p = max(p, yolo.sample(linClamp, uv).r);
    float3 live = ycbcr(Y, C, uv);
    float2 pp = uv * u.imageSize;
    float3 dc = normalize(float3((pp.x - u.cx) / u.fx, -(pp.y - u.cy) / u.fy, -1.0));
    float4 mem = pano.sample(linPano, dirToPano(u.camToWorld * dc));
    bool known = mem.a > 0.5;
    bool hole = p > 0.3 && !known;
    float3 col = (p > 0.3 && known) ? saturate(mem.rgb * gain.rgb) : live;
    uint idx = gid.y * N + gid.x;
    imgBuf[idx] = col.r * 255.0; imgBuf[N * N + idx] = col.g * 255.0; imgBuf[2 * N * N + idx] = col.b * 255.0;
    maskBuf[idx] = hole ? 0.0 : 255.0;
    holeTex.write(float4(hole ? 1.0 : 0.0, 0, 0, 0), gid);
    if (hole) atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}

// Writes the LaMa result into still-unknown memory at the pose the request was made from. alpha 0.6 = generated.
kernel void writeGenerated(texture2d<float, access::read_write> pano [[texture(0)]],
                           texture2d<float> hole [[texture(1)]],
                           texture2d<float> gen [[texture(2)]],
                           constant PanoUniforms& u [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    uint W = pano.get_width(), H = pano.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float3 dw = panoToDir((float2(gid) + 0.5) / float2(W, H));
    float3 dc = transpose(u.camToWorld) * dw;
    if (dc.z > -0.05) return;
    float px = u.fx * (dc.x / -dc.z) + u.cx;
    float py = u.cy - u.fy * (dc.y / -dc.z);
    float2 uv = float2(px, py) / u.imageSize;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) return;
    if (hole.sample(linClamp, uv).r < 0.5) return;
    float4 old = pano.read(gid);
    if (old.a > 0.5) return;
    pano.write(float4(gen.sample(linClamp, uv).rgb, 0.6), gid);
}

// =====================  R2: pose-aware keyframes + plane homography  =====================
struct KFUniforms {
    float3x3 H[4];          // current working-res pixel -> keyframe working-res pixel, per candidate
    float4 viewDist;        // viewpoint dissimilarity of each candidate (smaller = better source)
    int4 slice;             // keyframe slice per candidate, -1 = unused
    float2 curSize;
    float conf;             // 0 = distrust keyframes (fall back to soft estimate)
    float pad;
};
struct PlaneUniforms {
    float3x3 H[6];          // homographies for the 6 plane-depth hypotheses (best keyframe only)
    int slice;
    float2 curSize;
};
constant float2 KFSIZE = float2(960.0, 720.0);
constexpr sampler kfS(filter::linear, address::clamp_to_edge);

struct Cand { float3 col; float valid; float vd; };
static Cand kfCandidate(constant KFUniforms& k, texture2d_array<float> kfColor, texture2d_array<float> kfMask, float2 pc) {
    Cand best; best.col = float3(0); best.valid = 0.0; best.vd = 1e9;
    for (int i = 0; i < 4; i++) {
        int sl = k.slice[i];
        if (sl < 0) continue;
        float3 q = k.H[i] * float3(pc, 1.0);
        if (q.z <= 1e-4) continue;
        float2 uv = q.xy / q.z / KFSIZE;
        if (uv.x < 0.02 || uv.x > 0.98 || uv.y < 0.02 || uv.y > 0.98) continue;
        if (kfMask.sample(kfS, uv, uint(sl)).r > 0.1) continue;            // a person was here in that keyframe
        if (k.viewDist[i] < best.vd) { best.col = kfColor.sample(kfS, uv, uint(sl)).rgb; best.vd = k.viewDist[i]; best.valid = 1.0; }
    }
    return best;
}

kernel void storeKeyframe(texture2d<float> Y [[texture(0)]],
                          texture2d<float> C [[texture(1)]],
                          texture2d<float> mask [[texture(2)]],
                          texture2d<float> yolo [[texture(3)]],
                          texture2d_array<float, access::write> outColor [[texture(4)]],
                          texture2d_array<float, access::write> outMask [[texture(5)]],
                          constant uint& slice [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= 960 || gid.y >= 720) return;
    float2 uv = (float2(gid) + 0.5) / KFSIZE;
    float m = 0.0;
    for (int dy = -2; dy <= 2; dy++)
        for (int dx = -2; dx <= 2; dx++) {
            float2 o = float2(dx, dy) * (6.0 / KFSIZE);                 // dilate ~12 px: people never leak into memory
            m = max(m, mask.sample(linClamp, uv + o).r);
            m = max(m, yolo.sample(linClamp, uv + o).g);
        }
    outColor.write(float4(ycbcr(Y, C, uv), 1.0), gid, slice);
    outMask.write(float4(m > 0.08 ? 1.0 : 0.0, 0, 0, 0), gid, slice);
}

// Scores 6 plane-depth hypotheses on background pixels that touch a person region (the ring around the hole).
kernel void planeCost(texture2d<float> Y [[texture(0)]],
                      texture2d<float> C [[texture(1)]],
                      texture2d<float> mask [[texture(2)]],
                      texture2d<float> yolo [[texture(3)]],
                      texture2d_array<float> kfColor [[texture(4)]],
                      texture2d_array<float> kfMask [[texture(5)]],
                      constant PlaneUniforms& pu [[buffer(0)]],
                      device atomic_uint* stats [[buffer(1)]],
                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x >= 64 || gid.y >= 48 || gid.z >= 6) return;
    float2 uv = (float2(gid.xy) + 0.5) / float2(64, 48) * 0.9 + 0.05;
    if (mask.sample(linClamp, uv).r > 0.05 || yolo.sample(linClamp, uv).g > 0.05) return;
    bool ring = false;
    for (int k = 0; k < 8; k++) {
        float a = float(k) * 0.785398;
        float2 o = float2(cos(a), sin(a)) * 0.05;
        if (mask.sample(linClamp, uv + o).r > 0.3 || yolo.sample(linClamp, uv + o).g > 0.3) { ring = true; break; }
    }
    if (!ring) return;
    float3 q = pu.H[gid.z] * float3(uv * KFSIZE, 1.0);
    if (q.z <= 1e-4) return;
    float2 kuv = q.xy / q.z / KFSIZE;
    if (kuv.x < 0.02 || kuv.x > 0.98 || kuv.y < 0.02 || kuv.y > 0.98) return;
    if (kfMask.sample(kfS, kuv, uint(pu.slice)).r > 0.1) return;
    float3 live = ycbcr(Y, C, uv);
    float3 kc = kfColor.sample(kfS, kuv, uint(pu.slice)).rgb;
    float cost = dot(abs(live - kc), float3(0.3333));
    atomic_fetch_add_explicit(&stats[gid.z], uint(cost * 1000.0), memory_order_relaxed);
    atomic_fetch_add_explicit(&stats[8 + gid.z], 1u, memory_order_relaxed);
}

// (live - warped keyframe) on the ring around holes, at low res; mip pull diffuses it into the hole.
kernel void diffPrep(texture2d<float> Y [[texture(0)]],
                     texture2d<float> C [[texture(1)]],
                     texture2d<float> mask [[texture(2)]],
                     texture2d<float> yolo [[texture(3)]],
                     texture2d_array<float> kfColor [[texture(4)]],
                     texture2d_array<float> kfMask [[texture(5)]],
                     texture2d<float, access::write> outTex [[texture(6)]],
                     constant KFUniforms& k [[buffer(0)]],
                     uint2 gid [[thread_position_in_grid]]) {
    uint W = outTex.get_width(), H = outTex.get_height();
    if (gid.x >= W || gid.y >= H) return;
    float2 uv = (float2(gid) + 0.5) / float2(W, H);
    if (max(mask.sample(linClamp, uv).r, yolo.sample(linClamp, uv).r) > 0.05) { outTex.write(float4(0), gid); return; }
    bool near = false;
    for (int i = 0; i < 8; i++) {
        float a = float(i) * 0.785398;
        float2 o = float2(cos(a), sin(a)) * 0.03;
        if (mask.sample(linClamp, uv + o).r > 0.3 || yolo.sample(linClamp, uv + o).r > 0.3) { near = true; break; }
    }
    if (!near) { outTex.write(float4(0), gid); return; }
    Cand c = kfCandidate(k, kfColor, kfMask, uv * KFSIZE);
    if (c.valid < 0.5) { outTex.write(float4(0), gid); return; }
    float3 d = clamp(ycbcr(Y, C, uv) - c.col, -0.25, 0.25);
    outTex.write(float4(d, 1.0), gid);
}

static float3 diffusedOffset(texture2d<float> diff, float2 uv) {
    constexpr sampler s(filter::linear, mip_filter::linear, address::clamp_to_edge);
    for (int L = 1; L <= 7; L++) {
        float4 t = diff.sample(s, uv, level(float(L)));
        if (t.a > 0.15) return t.rgb / t.a;
    }
    return float3(0);
}

fragment float4 compositeKF(VOut in [[stage_in]],
                            texture2d<float> Y [[texture(0)]],
                            texture2d<float> C [[texture(1)]],
                            texture2d<float> mask [[texture(2)]],
                            texture2d<float> yolo [[texture(3)]],
                            texture2d<float> fill [[texture(4)]],
                            texture2d<float> diff [[texture(5)]],
                            texture2d_array<float> kfColor [[texture(6)]],
                            texture2d_array<float> kfMask [[texture(7)]],
                            texture2d<float> stDisp [[texture(8)]],
                            texture2d<float> stOff [[texture(9)]],
                            texture2d<float> stFg [[texture(10)]],
                            texture2d<float> genPano [[texture(11)]],
                            constant RenderUniforms& u [[buffer(0)]],
                            constant KFUniforms& k [[buffer(1)]]) {
    float2 iuv = (u.viewToImage * float3(in.uv, 1.0)).xy;
    float3 live = ycbcr(Y, C, iuv);
    float m = mask.sample(linClamp, iuv).r;
    float2 t = 3.0 / u.imageSize;
    for (int dy = -1; dy <= 1; dy++)
        for (int dx = -1; dx <= 1; dx++)
            m = max(m, mask.sample(linClamp, iuv + float2(dx, dy) * t * 2.0).r);
    m = max(m, yolo.sample(linClamp, iuv).r);
    m = smoothstep(0.25, 0.6, m);
    if (u.still != 0) m = max(m, stFg.sample(linClamp, iuv).r);

    float3 processed = live;
    if (m > 0.0) {
        float3 soft = fillColor(fill, iuv);
        float3 repl = soft;
        {   // generated (MI-GAN) world memory beats the blurry estimate where no keyframe sees the pixel
            float2 pp = iuv * u.imageSize;
            float3 dcg = normalize(float3((pp.x - u.cx) / u.fx, -(pp.y - u.cy) / u.fy, -1.0));
            float4 gm = genPano.sample(linPano, dirToPano(u.camToWorld * dcg));
            if (gm.a > 0.5) repl = saturate(gm.rgb * u.gain.rgb);
        }
        if (k.conf > 0.5) {
            Cand c = kfCandidate(k, kfColor, kfMask, iuv * KFSIZE);
            if (c.valid > 0.5) repl = saturate(c.col + diffusedOffset(diff, iuv));
        }
        if (u.still != 0) repl = stillReplacement(stDisp, stOff, iuv, repl);
        processed = mix(live, repl, m);
    }
    float3 outc;
    if (u.mode == 1) outc = live;
    else if (u.mode == 3) outc = mix(live, float3(1, 0.1, 0.1), m * 0.55);
    else if (u.mode == 2) outc = in.uv.x < u.splitX ? live : processed;
    else outc = processed;
    if (u.mode == 2 && fabs(in.uv.x - u.splitX) < 0.002) outc = float3(1);
    return float4(outc, 1.0);
}
