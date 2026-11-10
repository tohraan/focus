# FOCUS v3 — plan (2026-11-10), written before building, then red-teamed

## Goal
Any people in view disappear, natural, ~30 FPS, ~50 ms latency, while the phone rotates AND translates,
people sit still or walk. Revert point per revision. Base = tag `good-1105` (user-confirmed best).

## What each past design can/can't do (evidence from device recordings)
| Design | Works | Fails |
|---|---|---|
| Rotation-only panorama memory (11:05) | still/rotating camera, sharp | translation: torn/misaligned patches (parallax) |
| Soft push-pull fill | never crashes | blurry blobs, no texture |
| LaMa 512 (quantized) | - | purple/dark fills |
| LaMa 800 | right colours | 1.7-10 s per fill, starves GPU (FPS 11-18), grey silhouettes |

## Chosen architecture: pose-aware keyframes + plane-induced homography
1. **KeyframeStore**: up to 24 sharp keyframes (960x720 RGBA, texture array) each with its own *dilated person mask*,
   ARKit pose and intrinsics. Insert only when tracking == normal, frame not blurred, and camera has moved
   >= 6 cm or >= 8 deg from every stored keyframe. Evict the keyframe most redundant with a neighbour.
   (Replaces per-direction "commit" logic: a person can never leak into memory, each keyframe knows its own people.)
2. **Fill for the current frame**: for each candidate keyframe (top 4 by viewpoint similarity) compute
   H = K_k (R + t n^T / D) K_cur^-1 for a fronto-parallel plane at distance D. Per hole pixel pick the nearest-view keyframe
   where the keyframe mask is clear. Handles translation (parallax) for locally planar surroundings.
3. **Plane distance D by plane sweep**: 6 hypotheses (0.5-6 m) scored on the ring of background around the hole against
   the best keyframe (GPU), smoothed in log-depth. Evidence-based, no ARKit feature points needed.
4. **Seam/exposure hiding**: diffuse (live - warped) from the ring into the hole (push-pull on the difference image) and
   add it to the warped keyframe. Brightness/colour match and invisible seams without explicit gain.
5. **Unseen regions**: soft estimate now (flagged), patch-based synthesis later (R5). No LaMa in the live path.
6. **Mask**: unchanged from 11:05 (Vision + YOLO extras only). Temporal smoothing added later in its own revision with A/B.

## Revisions (each must beat the last on recording + metrics, else revert)
- R1 = tag good-1105 (done).
- R2 = keyframes + homography + sweep + difference diffusion, behind an Engine switch (Memory | Keyframes) for A/B on phone.
- R3 = mask temporal stabiliser (flicker), A/B against R2.
- R4 = per-pixel depth (Depth Anything 28 ms ANE, scaled by the sweep) replaces the single plane.
- R5 = patch-based fill for never-seen areas; thermal/FPS tuning.

## Red team (questions I asked the plan, with answers)
1. *Single plane is wrong for deep scenes -> torn patches again?* Yes, partly. Mitigations: nearest-view selection,
   difference diffusion hides small misalignment, photometric gate: if ring error is high the keyframe is rejected and
   soft fill is used, so failure is blur, never tearing. Real fix = R4 depth.
2. *Plane sweep unreliable in textureless rooms?* Low-texture also means low error everywhere -> D ambiguous but harmless
   (nothing to misalign). Prior = 2 m, smoothed.
3. *ARKit pose drift / world re-base?* Learned the hard way at 11:30. Keyframes only inserted/used while tracking == normal;
   clear the store on session reset/interruption. Never learn in "initializing".
4. *Missed person leaks into a keyframe?* YOLO extras already in the mask; keyframe mask dilated; photometric gate catches
   residue; a bad keyframe is evicted when its ring error is consistently worst.
5. *Static seated person never revealed?* Only fixable by a keyframe taken from another angle (scan) or generation.
   Honest limit; soft fill meanwhile, labelled "estimated".
6. *Cost?* Per frame: <=4 H matrices on CPU, one fragment pass with 4 array samples, 240x180 diffusion pass, sweep at
   10 Hz on a 64x48 grid. Keyframe insert <=3/s. Memory ~66 MB + masks. Must measure; budget 30 FPS.
7. *Is there a simpler thing that is as good?* Optical-flow chaining fails for long-lived static occluders and needs a flow
   per frame; LaMa/diffusion are too slow/ugly. Pose+plane reuses the strongest signal we have (ARKit pose).
8. *How do we know it's better?* A/B Engine switch on the same phone session; Gemini QA prompt + my frame review;
   device log (FPS, latency); unit test of the homography maths on Mac against synthetic geometry.
