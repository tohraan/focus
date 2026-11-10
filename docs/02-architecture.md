# FOCUS — Architecture

```
ARSession (ARWorldTrackingConfiguration, rotation+translation, intrinsics)
   | ARFrame @ ~30-60 fps: capturedImage (YCbCr), camera.transform, intrinsics, trackingState
   v
FrameCoordinator (serial, drops frames when busy)
   |--> PersonSegmenter (Vision, balanced, async)  -> mask (R8) 
   |       MaskStabilizer: dilate + temporal smoothing (EMA)
   |--> Pose: R_world_from_cam, K
   v
SceneMemory (Metal, equirect panorama 2048x1024 RGBA + confidence)
   update:  for each live pixel with mask==0 -> splat color into panorama at ray direction
   query:   for each output pixel -> ray -> panorama sample (+known flag)
   v
Compositor (Metal fragment shader)
   out = mask>0 and known ? memory : live      (feathered edge)
   unknown (mask>0 and !known) -> dimmed blur + "UNKNOWN" hatching (never presented as real)
   v
MTKView (processed)  |  split/toggle original vs processed
   v
HUD: Focus toggle, original/processed, tracking status, FPS, latency, Capture-background, safety notice
```

## Key design
- One design covers Stage A and B: in Stage A the pose is static so the panorama is just the screen; with rotation, the same ray lookup `ray = R * K^-1 * pixel` keeps memory aligned. Translation is ignored (documented, shown by tracking status/quality indicator).
- Memory update is gated: only pixels with mask==0, dilated margin, and tracking state `.normal`.
- Metrics: capture-to-display latency via frame timestamp, segmentation ms, composite ms, FPS - all logged to the HUD and a CSV for the testing report.
- Privacy: frames stay in memory; nothing written or uploaded; no identification.

## Interfaces (Swift)
- `PersonSegmenter.segment(pixelBuffer) async -> MTLTexture(r8)`
- `SceneMemory.update(frame, mask, pose)` / `SceneMemory.texture`
- `Compositor.render(live, mask, memory, pose, drawable)`
- `Metrics` (ring buffers, CSV export)
