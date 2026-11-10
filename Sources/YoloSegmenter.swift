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
            return result(vision: vision)
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
                let res = result(vision: vision)
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
