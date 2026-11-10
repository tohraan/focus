# FOCUS v2 Plan — "people are not noticeable" (2026-11-10)

Goal: any field of view, seated or walking people, phone rotating or moving, removal that is hard to notice.
Research caveat: model facts below come from search summaries, not primary pages. Every latency/recall number must be MEASURED on the iPhone 15 (task P0).

## Diagnosis of v1 (from on-device screenshots, 29-30 FPS, seg 12-24 ms, latency 39-67 ms measured)
1. Mask misses small/distant/seated people; edges cut through hands/objects.
2. Missed people are written into memory -> ghosts/smears.
3. 4096x2048 panorama = ~11 px/deg vs ~30 px/deg camera -> blocky blur.
4. Seated people never move -> background behind them is never observed -> nothing to reveal.
5. Rotation-only memory breaks under translation (parallax).

## Target architecture
```
ARFrame -> MaskEngine -> union mask (stabilized, edge-refined)
        -> KeyframeStore (sharp, pose-tagged, person-free pixels only, commit rule)
        -> Reprojector (pose + depth) : pick best 1-3 keyframes per output region
        -> HoleFiller: regions no keyframe covers -> cached inpainted "synthetic keyframe"
        -> Compositor: observed > generated > hatch, exposure/grain matched, feathered
Async (not on display path): Inpainter (on-device MI-GAN/LaMa CoreML, optional Mac offload)
```
Key trick for temporal stability: never inpaint per frame. Inpaint once per hole into a synthetic keyframe, then reproject it like any observation. Generated pixels stay flagged "generated".

## Tasks (ordered; each = deliverable / verify)
| # | Task | Verify (done when) |
|---|---|---|
| P0 | Benchmark harness: Vision seg balanced vs accurate, instance mask, human-rect veto; MI-GAN + LaMa CoreML (256/512, FP16) latency/thermal over 5 min; Depth Anything V2 S latency. Numbers to CSV. | measured table committed to docs/ |
| P1 | Replay tooling: opt-in local clip recorder on device (see Decisions) + offline replay of clip + saved poses on the Mac, so tuning does not need the user in class | same clip gives identical output on phone-side and Mac-side metrics |
| P2 | MaskEngine: union(person seg [accurate if budget allows], instance masks), temporal hysteresis + dilation, Metal guided-filter edge refine | recall on 30 hand-labeled frames; flicker metric (mask area jitter) down vs v1 |
| P3 | Commit rule: pixel enters memory only after N consecutive person-free frames AND human-rect veto clear AND tracking normal; sticky "was-person" map | no ghost in 5-min classroom replay |
| P4 | KeyframeStore + Reprojector (rotation homography first): camera-res keyframes, pose-proximity selection, multi-keyframe blend, exposure gain match | sharpness (edge SSIM) vs v1 panorama on replay |
| P5 | Translation: Depth Anything V2 S relative depth scaled to ARKit feature points; depth reprojection; plane fallback | lateral-move replay: seam/misalignment px error vs P4 |
| P6 | HoleFiller: coverage map -> crop hole+context -> MI-GAN/LaMa -> synthetic keyframe, refresh when coverage changes; grain/exposure match | seated-person replay has no hatching; fill flagged generated |
| P7 | Optional Mac offload for higher-quality fill (LaMa/diffusion) over Tailscale, async only | fill swaps in without stalling display |
| P8 | Integration pass + 5-min thermal/FPS run, testing report with real numbers, demo script | acceptance below |

Critical path: P0 -> P1 -> P2/P3 -> P4 -> P6. P5, P7 are quality upgrades.

## Acceptance
- >= 24 FPS sustained 5 min, latency reported (not estimated).
- Single/multiple people, walking and seated: no visible person silhouette in Focus mode on replay clips at normal viewing; residual artifacts documented.
- Honest limits: filled regions are estimates; large translation or fast motion degrades; never claim see-through.

## Decisions needed from user
1. Debug clips: recording classmates, even locally on-device, is a privacy call. OK to record short opt-in clips stored only on the phone/Mac, deleted after tuning? (Alternative: tune only live, much slower.)
2. Licensing: Ultralytics YOLO-seg is AGPL, RVM is GPL. Plan avoids both unless P2 recall is insufficient. If needed, is open-sourcing the repo acceptable?
3. Mac offload (P7) allowed over Tailscale, async only?
4. MI-GAN CoreML port license is unclear; I would convert from the MIT upstream myself (needs Python torch/coremltools, ~2-3 GB disk). OK?
