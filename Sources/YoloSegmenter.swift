import Vision
import CoreML

/// YOLO11-seg person detector (AGPL model; private/hackathon use). Catches small, distant and
/// turned-away people that Vision's person matte misses. Output is a coarse union mask (W/4 x H/4).
final class YoloSegmenter {
    let request: VNCoreMLRequest
    let inW: Float = 1280, inH: Float = 960

    init?(model name: String) {
        guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc") else { return nil }
        let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
        guard let ml = try? MLModel(contentsOf: url, configuration: cfg),
              let vm = try? VNCoreMLModel(for: ml) else { return nil }
        request = VNCoreMLRequest(model: vm)
        request.imageCropAndScaleOption = .scaleFill   // full 4:3 sensor frame -> 1280x960, no letterbox
    }

    /// Overlapping 60% tiles (top-left origin): each is scaled up 1.67x so far-away people are big enough to detect.
    static let tiles: [(x: Float, y: Float)] = [(0, 0), (0.4, 0), (0, 0.4), (0.4, 0.4)]
    static let tileSize: Float = 0.6
    private var tileMask: [YoloDecode.Result?] = [nil, nil, nil, nil]
    private var tileCursor = 0

    /// Runs the request on one tile; returns its decoded result, or nil.
    private func runTile(_ k: Int, _ pb: CVPixelBuffer, vision: YoloDecode.VisionMask?) -> YoloDecode.Result? {
        let t = Self.tiles[k], sz = Self.tileSize
        request.regionOfInterest = CGRect(x: CGFloat(t.x), y: CGFloat(1 - t.y - sz), width: CGFloat(sz), height: CGFloat(sz))  // Vision: origin bottom-left
        defer { request.regionOfInterest = CGRect(x: 0, y: 0, width: 1, height: 1) }
        try? handler.perform([request], on: pb, orientation: .up)
        var v: YoloDecode.VisionMask?
        if let vm = vision {   // crop the Vision matte to the tile so "already covered" is judged in tile coordinates
            let x0 = Int(t.x * Float(vm.w)), y0 = Int(t.y * Float(vm.h)), cw = Int(sz * Float(vm.w)), ch = Int(sz * Float(vm.h))
            var d = [UInt8](repeating: 0, count: cw * ch)
            for y in 0..<ch { for x in 0..<cw { d[y * cw + x] = vm.data[min(vm.h - 1, y0 + y) * vm.w + min(vm.w - 1, x0 + x)] } }
            v = YoloDecode.VisionMask(data: d, w: cw, h: ch)
        }
        return result(vision: v)
    }

    /// Full-frame result plus the stored tile results (max).
    private func combined(_ full: YoloDecode.Result?) -> YoloDecode.Result? {
        guard var r = full else { return nil }
        for (k, tr) in tileMask.enumerated() {
            if let tr = tr { YoloDecode.merge(&r, tile: tr, ox: Self.tiles[k].x, oy: Self.tiles[k].y, tw: Self.tileSize, th: Self.tileSize) }
        }
        return r
    }

    private let handler = VNSequenceRequestHandler()
    private let queue = DispatchQueue(label: "focus.yolo", qos: .userInitiated)
    private let lock = NSLock()
    private var busy = false
    private var latest: YoloDecode.Result?
    private(set) var lastMs = 0.0

    /// Non-blocking: starts a detection if idle and returns the most recent finished result.
    var syncMode = false      // offline video tests: run inline and return this frame's own result

    func submit(_ pb: CVPixelBuffer, vision: YoloDecode.VisionMask?) -> YoloDecode.Result? {
        if syncMode {
            let t = ProcessInfo.processInfo.systemUptime
            try? handler.perform([request], on: pb, orientation: .up)
            lastMs = (ProcessInfo.processInfo.systemUptime - t) * 1000
            let full = result(vision: vision)
            for k in 0..<4 { tileMask[k] = runTile(k, pb, vision: vision) }
            lastMs = (ProcessInfo.processInfo.systemUptime - t) * 1000
            return combined(full)
        }
        lock.lock()
        let start = !busy
        if start { busy = true }
        let r = latest
        lock.unlock()
        if start {
            queue.async { [self] in
                let t = ProcessInfo.processInfo.systemUptime
                try? handler.perform([request], on: pb, orientation: .up)
                let full = result(vision: vision)
                // one tile per cycle, round robin: every tile refreshes about every 4 cycles
                let k = tileCursor; tileCursor = (tileCursor + 1) % 4
                tileMask[k] = runTile(k, pb, vision: vision)
                let res = combined(full)
                lock.lock(); latest = res; busy = false; lastMs = (ProcessInfo.processInfo.systemUptime - t) * 1000; lock.unlock()
            }
        }
        return r
    }

    /// Call after `request` was performed.
    func result(vision: YoloDecode.VisionMask? = nil) -> YoloDecode.Result? {
        guard let obs = request.results as? [VNCoreMLFeatureValueObservation] else { return nil }
        var det: MLMultiArray?, pro: MLMultiArray?
        for o in obs { if let a = o.featureValue.multiArrayValue { if a.shape.count == 3 { det = a } else { pro = a } } }
        guard let d = det, let p = pro else { return nil }
        return YoloDecode.decode(det: d, protos: p, inW: inW, inH: inH, vision: vision)
    }
}
