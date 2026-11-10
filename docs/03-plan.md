# FOCUS — Implementation Plan

| # | Task | Deliverable | Depends | Verify / done when |
|---|---|---|---|---|
| 1 | Xcode project scaffold (SwiftUI + MTKView), signing | builds for device | - | `xcodebuild` succeeds for generic iOS device |
| 2 | ARSession camera + Metal preview + FPS/latency HUD | live feed | 1 | on phone: feed + FPS shown (needs user) |
| 3 | Vision person segmentation + overlay + benchmark | mask overlay, ms logged | 2 | measured ms on iPhone 15 recorded |
| 4 | Background capture + removal (Stage A) | people replaced by clean bg | 3 | Test 1/2 pass on device |
| 5 | Original/processed toggle + Focus Mode UI | UI | 4 | manual |
| 6 | Pose-aligned panorama memory (Stage B) | rotation-stable removal | 4 | Test 3 |
| 7 | Mask stabilizer, halo dilation, exposure gain | fewer artifacts | 4 | visual + flicker metric |
| 8 | Unknown-region handling | honest hatching | 6 | Test 6 |
| 9 | Test report with measured numbers + demo script | docs | 4-8 | real numbers only |
| - | Audio | only when user says video is polished | | |

Critical path: 1 -> 2 -> 3 -> 4. Tasks 1 and logic unit tests can be verified without the phone; camera tasks need the user's phone.
