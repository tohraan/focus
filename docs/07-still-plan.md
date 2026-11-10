# 07 Still-camera background layer

Plan: `~/.claude/plans/humming-dancing-russell.md` (steps S0-S10). Branch `still-layer`.
Rule: no video renders (offline bench included) until the user says so.

## S0 baseline (2026-11-10, iPhone 15, Release build, 360 frames each, 1080x768)
Metrics from `tools/synth/metrics.py baseline`. leak_pct = person pixels Vision still finds in the output / input person pixels (lower is better).
hole_mae = mean abs error vs clean truth inside the person region (0-255 scale). flicker = mean frame-to-frame change in background.

| clip | engine | leak % | hole MAE | flicker |
|---|---|---|---|---|
| A random | Memory | 2.72 | 56.3 | 0.369 |
| A random | Keyframes | 0.39 | 35.5 | 0.252 |
| B exposure | Memory | 0.25 | 39.4 | 0.284 |
| B exposure | Keyframes | 0.45 | 40.1 | 0.276 |
| C shadow | Memory | 10.15 | 46.2 | 0.377 |
| C shadow | Keyframes | 3.25 | 37.4 | 0.349 |

Speed: about 125 ms/frame for both engines plus the offline path (not the live path).
Confirmed in code: Keyframes stores 1 keyframe on a still camera ("kf 1" in every status line).
Caveats: the synthetic people are cut-out seated torsos from the classroom clips on a ceiling background, so it is a proxy. The real test is the user's own propped-phone clip at S10.

## Log

### 2026-11-10 S1-S8 written in code (compile-checked only, NOT run; no renders per user)
- `StillPlate.swift` + kernels `stillClassify/stillResolve/stillCommit/stillClear` in Shaders.metal: per-pixel plate at 1/3 res in image space, learned from pixels outside every person mask (dilated ~14 px) and stable for 8+ frames; difference mask (big difference next to a person, or moving and big), shadow rule (darker, same chroma, within 20 plate px of a person), exposure gain from non-foreground pixels, offset diffusion (mip) for seamless edges, absorb after 3 s of persistent change (moved object).
- Camera state in `Renderer.updateStill`: moving (reset) / frozen (small shake or brief tracking loss: keep plate, stop learning) / still (learn + use). Resets on > 1 deg or 3 cm in 1 s, or > 1.5 deg / 2 cm from the anchor, or tracking lost > 2 s.
- Both composites (`compositeFS`, `compositeKF`) use the plate first where its confidence is > 0.5, else the engine's own fill. Keyframes also refreshes a keyframe every 2 s when still.
- "Capture background" now clears everything and shows a banner with progress and % learned (still) or views saved (moving); HUD line shows camera still/moving and % learned.
- Known limits: person who never moves from the first frame is not revealed (plate has nothing there; engine fill is used); a matte that misses a still person for 3+ s gets absorbed; a TV/monitor showing video is frozen to the plate.

### 2026-11-10 MI-GAN generative fill (compile-checked, installed, NOT run on device)
- Model: MI-GAN 512 (MIT, Picsart; ONNX from `andraniksargsyan/migan` on Hugging Face), converted ONNX -> torch (onnx2torch) -> Core ML fp16, 14 MB, stored in `ModelsStash/MIGAN512.mlpackage` and copied to `Sources/Models/` (gitignored like the other models). Conversion recipe: fix input to 1x3x512x512, onnxsim, turn uint8 casts into float, float 0..255 in/out, mask 255 = keep, 0 = hole.
- Checked on one still frame (person removed on a 512x512 squash): plausible desk/wall/screen fill. Core ML vs ONNX: mean abs diff 0.03 on CPU+GPU (23 ms on the Mac GPU), but WRONG on the Neural Engine (mean diff 15, 120 inside the hole), so the app uses CPU+GPU.
- App: `MiganFill.swift`, kernels `miganPrep` / `miganCommit`. When the still plate is active, person pixels the plate has never seen are generated and written into the plate with confidence 0.51 (generated); real observations replace them immediately. Stale results are dropped if the plate was reset meanwhile.

### 2026-11-10 Handheld path: MI-GAN replaces LaMa (compile-checked, installed, NOT run on device)
- `AutoFill` (world-anchored generated memory, the moving-camera fill) now runs MI-GAN 512 on CPU+GPU via new kernel `prepMigan` (0..255 image, 255 = keep / 0 = hole) instead of LaMa. LaMa kernel and model remain in the repo, unused.
- It now runs for BOTH engines whenever the still plate is NOT active (before: Memory engine only). The still plate has its own MI-GAN path.
- Keyframes composite (`compositeKF`) now uses the generated world memory (panorama alpha > 0.5) before the blurry soft fill where no keyframe sees the pixel. Binding: panorama at fragment texture 11.
- Known cost/risk: for the Keyframes engine the panorama only holds generated pixels, so the generator also fills regions that keyframes could fill with real pixels; those are ignored by the composite (keyframe wins). Slight wasted GPU work, no visual effect expected.
- Next for handheld: mask temporal stabiliser (from `891c4f6`, branch `experimental-1200`), per-pixel depth (Depth Anything V2 small, Core ML, about 34 ms on iPhone 15 Pro Max per Apple's card) to replace the single-plane keyframe warp.
