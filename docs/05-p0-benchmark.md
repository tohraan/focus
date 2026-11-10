# P0 Benchmark — iPhone 15 (A16), iOS 26.6.1, measured 2026-11-10, thermal state nominal (0)
Synthetic gray input; latency is independent of content. 20 iterations after 3 warm-ups + 1 first-predict.

| Model | Units | Load | First predict | Median | p95 |
|---|---|---|---|---|---|
| Depth Anything V2 Small F16 (518x392) | CPU+GPU | 0.5 s | 250 ms | 121 ms | 125 ms |
| | CPU+Neural Engine | 9.0 s | 35 ms | **28.2 ms** | 28.5 ms |
| | All | 9.8 s | 30 ms | 28.2 ms | 28.6 ms |
| LaMa 512 (Dadm-n Core ML) | CPU+GPU | 2.5 s | 2.5 s | **499 ms** | 506 ms |
| | CPU+Neural Engine | **never finished loading (>90 s)** | | | |
| | All | **never finished loading (>90 s)** | | | |

Takeaways: depth fits the live budget on the ANE (28 ms). LaMa works only on GPU at ~0.5 s -> background hole-fill at ~2 Hz, never per-frame. LaMa on the ANE hangs on load; do not use it.
Not yet measured: MI-GAN, Vision instance/accurate segmentation, YOLO-seg.
Reproduce: tools/bench.sh (phone unlocked).
