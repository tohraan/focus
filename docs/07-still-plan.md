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
- `StillPlate.swift` + kernels `stillClassify/stillResolve/stillCommit/stillClear` in Shaders.metal: per-pixel plate (originally 1/3 res image space; since R5 a 4096x2048 equirectangular canvas), learned from pixels outside every person mask (dilated ~14 px) and stable for 8+ frames; difference mask (big difference next to a person, or moving and big), shadow rule (darker, same chroma, within 20 plate px of a person), exposure gain from non-foreground pixels, offset diffusion (mip) for seamless edges, absorb after 3 s of persistent change (moved object).
- Camera state in `Renderer.updateStill`: moving (anchor reset only; the plate itself resets on tracking lost > 2 s, world re-base, or walking > 0.8 m) / frozen (small shake or brief tracking loss: keep plate, stop learning) / still (learn + use). Resets on > 1 deg or 3 cm in 1 s, or > 1.5 deg / 2 cm from the anchor, or tracking lost > 2 s.
- Both composites (`compositeFS`, `compositeKF`) use the plate first where its confidence passes smoothstep(0.35, 0.5), else the engine's own fill. Keyframes also refreshes a keyframe every 2 s when still.
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

### 2026-11-10 R3 mask flicker fix (compile-checked, installed, NOT run on device)
- `MaskStabilizer.swift` + kernels `maskStabilize` / `stabClear`: the Vision matte and both YOLO channels are smoothed over time in image space at 1/4 resolution. Instant attack (0.9 per 30 fps frame), slow release (0.25 per frame while the camera is still, about 4 frames; 0.6 while it moves, about 2 frames), rates converted to the real frame time. Ping-pong textures.
- Applied at the top of `Renderer.encodeScene`, so the composite, the still plate, the keyframe store, the panorama update and the generated fill all see the smoothed masks. Camera counts as moving if the still detector says so or speed > 10 deg/s or > 0.1 m/s.
- "Smooth edges" switch in the app (on by default) turns it off for A/B comparison; the "Mask" view shows the smoothed mask.
- Trade-off to check: a short fade-out trail (about 4 frames) behind a fast-moving person on a still camera; it shows the background fill, so it should be hard to see.
- Tuning knobs: `attack30`, `releaseStill30`, `releaseMoving30` in `MaskStabilizer`.

### 2026-11-10 Fix: still layer never activated on a propped phone (found in the device log; compile-checked, installed, NOT run on device)
- Evidence: `metrics.csv` from the user's live run: tracking was "limited" for all of the last 600 frames (screenshot said "limited (initializing)"). ARKit world tracking needs motion to initialise, so a phone on a table never reaches "normal", and the still layer (and both engines) were gated on "normal".
- Fix: the still-camera check now accepts ARKit `.limited(.initializing)` and `.limited(.insufficientFeatures)` (`FrameData.trackingUsable`). In those states the pose has valid rotation (gyro) but no translation, so an image-based guard was added: if over 30% of the visible background differs strongly from the plate for 4 frames in a row, the plate is reset (camera slid or lights changed). Stats slots 8/9 of the plate stats buffer.
- Still gated on normal tracking: Memory panorama update, keyframe store, generated world memory (handheld paths).

### 2026-11-10 User feedback: still ~5/10 ("capture background is very bad"), handheld "just breaks"
Hypotheses (no data yet) and what was changed (compile-checked, installed, NOT run):
- Handheld: engines (panorama update, keyframes, generated memory) were gated on ARKit "normal" tracking, so while ARKit sat in "initializing" (rotation only) nothing was processed at all. They now run on "usable" tracking (normal, initializing or insufficient features; rotation-only pose when translation is unknown). When tracking becomes normal the world frame can be re-based, so keyframes and the panorama are cleared once.
- Still: learning made more conservative against contamination: first-sight needs 10 stable frames (was 8), learning-protection radius around the person matte is about 18 px (was 14).
- Diagnostics so the next report can be checked against data (stills and text, no video): `Documents/still_log.txt` (every 0.5 s: tracking, camera state 0/1/2, still active, plate coverage, gain, reset count, fps) and `Documents/snap/NN_<tag>_{live,plate,conf,fg}.png` at plate resolution. Snapshots are saved automatically 6 s into a capture and when it reports "captured", or by hand with "Save debug snapshot" in the controls panel.
- Pull with: `xcrun devicectl device copy from --device <id> --domain-type appDataContainer --domain-identifier com.tohraan.focus-dr --source Documents/snap --destination <dir>` (same for `Documents/still_log.txt`).


## 2026-11-10 (evening) Autonomous loop on the Mac bench (no phone, no video)
Bench: `FocusBench` (macOS target, same engines and shaders) + `tools/run_all.sh ITER` (7 synthetic clips, 4 parallel groups, scores printed by `tools/table.py`). Still frames only. Metrics (`tools/synth/metrics_stills.py`): leak% (person pixels Vision still finds), reveal (error in person-region pixels that WERE visible earlier; should be about 2), gen (error where the background was never seen: only a plausible guess is possible), bg (error outside the person), flicker, score (lower is better). Ground truth known (synthetic clips from real classroom frames; handheld clip uses true camera rotations).

Bench bugs found and fixed (they distorted every earlier number): Core Image colour management brightened every frame by ~8-10 levels; the generator timer carried over between clips so the 2nd clip of each run never got generated fill.

Real bugs found by the loop:
1. Exposure gain estimate drifted 1.00 -> 1.18 in 12 s (feedback into the plate): plate drifted brighter. Fixed by storing raw camera values in the plate (reveal error 8.9 -> 2.1).
2. Generated fill for a huge hole came out flat grey: now the border colour mismatch is measured and diffused into the hole (push-pull); gen error 26 -> 9 on the cold-start clip.
3. Plate now lives on a world-anchored canvas (rotation only, canvas 3x the image at plate resolution): handheld uses the same learning/shadow/fill machinery. Handheld fill error 6.9 -> 1.65, leak 0.2% -> 0.

Mean score: corrected baseline 31.6 -> 24.9 (A 16.9, B 31.4, C 19.6, D 17.0, E 17.4, F 55.9, handheld 16.4; B and F are floor-limited by exposure change and sensor noise).

Tried and reverted (made scores worse): plate at 1/2 resolution, faster first learning, wider person region/foreground dilation, global tone scaling of generated fill, shadow-like faster absorption.
Ablations (earlier, with the gain bug present): mask stabiliser slightly worse on synthetic masks (no mask-edge flicker in the metric), engines alone beat the plate until the gain bug was fixed.

### 2026-11-10 First live data (phone log) and fix
Live log (handheld, ARKit "good"): the plate was reset 41 times in ~60 s (coverage kept falling to 0) and fps was 15 (2 once). Causes (inferred, not yet re-measured live): the camera-moved guard (30% strong difference for 4 frames) and the 0.12 m distance limit fire constantly in a hand because translation parallax makes the rotation-only plate misalign; repeated resets also re-trigger heavy generated fill (CPU seam correction + GPU) every 0.6 s.
Changes: no reset on a short misalignment. If >25% of the visible background differs strongly, the plate is not used that frame (engines' own fill shows instead) and learning pauses; wiped only after 1.5 s of continuous misalignment. Distance limit 0.12 -> 0.5 m, angle 40 -> 55 deg. Generated fill only while the plate can learn, every 1.2 s on the phone. (Plate divisor was removed in R5: the canvas is a fixed 4096x2048.) Log now prints the last reset reason and misalignment. Bench unchanged (no regression).

### 2026-11-10 Review of the user's screen recording (ScreenRecording 21-32-27, 1:03, handheld in a bedroom) + Gemini's frame-by-frame read
Evidence frames (seconds): 0.3 invented picture (neon/wardrobe pattern with torn edge) that is not in the room; 0.8 and 1 large flat grey slabs with jagged edges over the wall; 3.2 objects on the bed cut by jagged patches; 6.4 grey ghost shape beside the chair; 32.6, 35.2, 49.3, 55 the seated woman is FULLY VISIBLE (not removed); 52.6 scrambled/duplicated patches on the cabinet; 58.6 and 38 shredded curtain edge; HUD says "camera: still" the whole time although the phone is being moved; background % jumps 0 <-> 80.
Causes found / suspected: (1) HUD/banner used "plate in use" as "camera still" (my bug, fixed); (2) generated fill runs on huge holes (> 40% of the frame) and invents scenery (now skipped); (3) difference-mask growth (big difference / moving / shadow) floods when the plate is misaligned, which removes furniture and clothes (now only while the camera is steady and the plate aligned; unseeded "moving" rule off while panning); (4) person NOT removed in several frames: seated woman from behind, dim light, chair in front: detector miss (Vision matte + YOLO union) — needs real raw frames to tune; (5) background memory cannot hold in a hand because the plate is rotation-only and translation parallax breaks it; real fix = per-pixel depth (Depth Anything V2 small) so views can be re-projected.
Added: "Auto debug stills (every 2 s)" and "Save debug snapshot" now save, at 640x480: rawlive, rawmask (r = Vision matte before smoothing, g = YOLO extras, b = YOLO all detections), plus plate live/plate/conf/fg. Pull `Documents/snap`.

## R5 (2026-11-10): full-sphere canvas
Plate canvas is now an equirectangular map (4096x2048, world-fixed, rotation only) instead of a 3x rectilinear canvas. Removes the "moved far from start" reset on angle (it fired 10x in 25 s of handheld on the phone); only a walk of more than 0.8 m resets. Dumps are 1024x512. Bench (stills only): mean 24.78 / 24.72 (was 24.9), no clip regressed. NOT yet verified on the phone (fps, memory, 4096x2048 rgba16f cost).
