# FOCUS

**Look at the room. Not at the crowd.** FOCUS removes people from a live camera view in real time, on the phone, so a student can see the board, the page or the space without the visual noise of everyone in it.

GDG Ideathon · Ideaverse · BITS Pilani, Dubai Campus

![Swift](https://img.shields.io/badge/Swift-6.3-F05138?logo=swift&logoColor=fff)
![Metal](https://img.shields.io/badge/Metal-GPU-8E8E93?logo=apple&logoColor=fff)
![ARKit](https://img.shields.io/badge/ARKit-pose-0A84FF)
![Core ML](https://img.shields.io/badge/Core%20ML-on--device-34C759)
![iOS](https://img.shields.io/badge/iOS-26-000?logo=apple)
![Privacy](https://img.shields.io/badge/data%20stored-none-F2B84B)

**Jump to:** [Why](#why-it-matters) · [How it works](#how-it-works) · [Demo](#demo) · [Hardware concept](#hardware-concept-glasses) · [Status](#prototype-status-measured-not-claimed) · [Run it](#run-it) · [Limits](#honest-limits)

![Pipeline](docs/img/pipeline.png)

## Why it matters

Crowded lecture halls, libraries and labs are visually busy. For students prone to sensory overload, ADHD or social anxiety, constant movement and faces make it hard to hold attention on the task. Today the only options are to leave or to put up with it. FOCUS gives a third option: dim the crowd, keep the room.

> This filters the *screen*, not vision. A removed person is still physically there, so the app shows a persistent "Do not walk while using" notice.

## How it works

| Step | What happens | Tech |
| --- | --- | --- |
| 1. See | Camera frame, camera pose and intrinsics every frame | ARKit |
| 2. Find people | Person mask, plus extra detections for small or distant people | Vision, YOLO11 segmentation (Core ML) |
| 3. Remember the room | A still-camera background plate when the phone is steady; pose-aware keyframes with plane-induced homography when it moves | Metal compute kernels |
| 4. Fill | Real remembered pixels first. Areas never seen are generated once and flagged as generated | MI-GAN 512 (Core ML, CPU+GPU) |
| 5. Show | Feathered composite, with an original / processed toggle | Metal fragment shader |

Design rules we kept: every performance number is measured on the device, never assumed; generated pixels are never presented as real; nothing is recorded, uploaded or identified.

## Demo

- **Physical demo:** we present FOCUS live on an iPhone, as close to the intended wearable (glasses) experience as we can get it.
- **Videos:** _to be added here._
- **Hosted link:** none. It is a native iOS app.

## Hardware concept (glasses)

The long-term form is a pair of glasses so the effect follows your gaze. We designed the frame and a light-blocking shield in parametric CAD (FreeCAD and Blender, from one spec). **It is a concept: no camera, display or optics are modelled or claimed, and nothing here is manufacturing-ready.**

![Assembly](hardware/renders/assembly_3q.jpg)
![Exploded](hardware/renders/exploded.jpg)

Blueprints: [`hardware/blueprints/`](hardware/blueprints) · spec: [`hardware/specifications/`](hardware/specifications)

## Prototype status (measured, not claimed)

Measured on iPhone 15 (A16), iOS 26.6.1:

| Metric | Result |
| --- | --- |
| Live pipeline, first version (device recordings) | 29-30 FPS, segmentation 12-24 ms, capture-to-display latency 39-67 ms |
| Depth Anything V2 Small, Neural Engine | 28.2 ms median |
| LaMa 512 | 499 ms on GPU, hangs on the Neural Engine, so not used live |
| Still-camera test bench (synthetic clips, 360 frames) | Keyframes engine: person leak 0.39% vs 2.72% for the memory engine on random motion |

Newer work (still-camera plate, MI-GAN fill, keyframe engine) is **compile-checked and bench-tested offline but not yet re-run on the phone.** Details and the full log: [`docs/`](docs).

## Run it

Needs a Mac with Xcode 26, an iPhone with Developer Mode, and a free Apple ID for signing.

```bash
brew install xcodegen
xcodegen generate          # builds Focus.xcodeproj from project.yml
open Focus.xcodeproj       # pick your team, run on a device (the camera is not available in the simulator)
```

Core ML models are not committed (size and licensing). Place `MIGAN512.mlpackage` and `Yolo11sSeg.mlpackage` in `Sources/Models/`; the conversion recipe for MI-GAN is in [`docs/07-still-plan.md`](docs/07-still-plan.md). Note YOLO11 weights are AGPL-3.0, so check the license before redistributing.

## Honest limits

- A person who never moves from the first frame is not revealed on a steady camera; the fill is generated and flagged.
- Moving the phone sideways breaks the rotation-only assumption; the keyframe engine mitigates it for mostly flat surroundings.
- A TV or monitor playing video is frozen to its remembered image.
- Sustained thermal behaviour over 5 minutes is not yet measured.
- No audio, and the glasses are a concept only.

## Docs

[01 Feasibility](docs/01-feasibility.md) · [02 Architecture](docs/02-architecture.md) · [03 Plan](docs/03-plan.md) · [04 v2 plan](docs/04-v2-plan.md) · [05 Benchmark](docs/05-p0-benchmark.md) · [06 v3 plan](docs/06-v3-plan.md) · [07 Still-camera layer](docs/07-still-plan.md)
