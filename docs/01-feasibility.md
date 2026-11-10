# FOCUS — Feasibility Report (2026-11-10)

Status labels: **Verified** = source fetched; **Unverified** = not confirmed, must be measured on device.

## Environment (inspected)
Mac M4 / 16 GB, Xcode 26.6, Swift 6.3, Python 3.14 (cv2, onnxruntime). Target: iPhone 15 (iPhone15,4), iOS 26.6.1, Developer Mode on. **No LiDAR on iPhone 15** (Apple spec page lists none; ARKit sceneDepth is LiDAR-only). No Android tooling.

## Findings
| Area | Finding | Source |
|---|---|---|
| Segmentation | Vision `VNGeneratePersonSegmentationRequest` (iOS 15+, fast/balanced/accurate, stateful for video) gives ONE merged person matte | developer.apple.com/documentation/vision/vngeneratepersonsegmentationrequest |
| | `VNGeneratePersonInstanceMaskRequest` (iOS 17+) gives per-person masks; "max 4" is third-party only | Apple docs; charleswiltgen.github.io/Axiom/reference/vision-ref |
| | ARKit `segmentationBuffer` exists (iOS 13+), merged matte; device support must be checked at runtime | Apple docs |
| | MediaPipe selfie segmenter: selfie-oriented, only Pixel 6 latency published | developers.google.com/edge/mediapipe |
| | YOLO-seg CoreML: **AGPL-3.0**; published 4.9 ms is iPhone 17 Pro, not iPhone 15 | github.com/ultralytics/ultralytics |
| Pose | `ARCamera.transform / intrinsics / trackingState` per frame | Apple ARCamera docs |
| Depth | Depth Anything V2 Small CoreML, Apache-2.0, relative (not metric) depth; ~34 ms on iPhone 15 Pro Max (garbled table) | huggingface.co/apple/coreml-depth-anything-v2-small |
| Reconstruction | ProPainter is non-commercial (S-Lab) and PyTorch/GPU-heavy; E2FGVI stale since 2023. Neither realistic on-device real-time | their GitHub repos |
| Prior art | jasonmayes/Real-Time-Person-Removal (Apache-2.0, TF.js, **static camera only**) shows the Stage-A idea works. No maintained ARKit/Metal sample found | github.com/jasonmayes/Real-Time-Person-Removal |
| Browser | getUserMedia needs HTTPS; iOS gyro needs a user-gesture permission; no ARKit pose, no intrinsics | MDN |

No latency number for any option on iPhone 15 is published. **All performance claims in this project come from our own on-device measurements.**

## Approaches compared
| | Fully on-device (A) | Phone + desktop (B) | Hybrid (C) |
|---|---|---|---|
| Latency | lowest, no network | network + encode | mixed |
| Reliability for demo | high (no Wi-Fi dependency) | low (venue network) | low |
| Model access | Vision/CoreML built in | any Python model | any |
| Build effort | one Swift codebase | two codebases + sync | most |
| Verdict | **Chosen** | fallback only | rejected |

## Decision
Native iOS, fully on-device. Vision person segmentation, ARKit pose, Metal compositing, a **rotation-based world-anchored background panorama** as scene memory. Heavy inpainting (Priority 4) is out of scope; unknown regions are shown honestly, not faked.

## Risks
1. Rotation-only reprojection is wrong under translation/near objects (parallax). Documented limit; Stage C not promised.
2. Segmentation edges leak a halo of person pixels -> dilate mask before compositing.
3. Auto-exposure shifts make remembered pixels mismatch live ones -> per-region gain correction later.
4. Thermal throttling on sustained use -> measure over 5 min.
5. Vision mask resolution/latency unknown -> benchmark first (Task 3).
6. Stationary people who never moved get baked into memory. Mitigation: record a clean background first (Stage A flow).

## Unresolved dependencies
- Signing team in Xcode (user has Apple ID; need to select team on first build).
- iPhone must be unlocked/trusted for on-device runs (user is in class; build verification until then).
- Actual segmentation latency on iPhone 15.

## Safety (product constraint)
This filters the *screen*, not vision. A removed person is still physically there. The app shows a persistent "Do not walk while using" notice. No face recognition, no recording, no upload.
