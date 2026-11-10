import CoreML
import CoreVideo
import Foundation

/// P0: measures real Core ML latency on this device. Writes Documents/bench.txt.
enum Benchmark {
    /// One model x one compute-unit config per launch; hard watchdog so a hang is recorded, not waited on.
    static func runOne(model: String, units label: String, timeout: Double = 90, warmup: Int = 3, iters: Int = 20) -> String {
        let u: MLComputeUnits = label == "cpu+gpu" ? .cpuAndGPU : label == "cpu+ane" ? .cpuAndNeuralEngine : label == "cpu" ? .cpuOnly : .all
        let file = "bench_\(model)_\(label).txt"
        flush("RUNNING \(model) [\(label)]\n", file)
        DispatchQueue.global().asyncAfter(deadline: .now() + timeout) {
            flush("TIMEOUT after \(Int(timeout)) s (load or predict never finished) \(model) [\(label)]\nDONE\n", file)
            exit(0)
        }
        guard let url = Bundle.main.url(forResource: model, withExtension: "mlmodelc") else {
            let r = "MODEL NOT IN BUNDLE: \(model)\nDONE\n"; flush(r, file); return r
        }
        let r = one(url, label, u, warmup, iters) + "thermal \(ProcessInfo.processInfo.thermalState.rawValue)\nDONE\n"
        flush(r, file)
        return r
    }

    private static func flush(_ s: String, _ name: String) {
        let f = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent(name)
        try? s.write(to: f, atomically: true, encoding: .utf8)
    }

    private static func one(_ url: URL, _ label: String, _ units: MLComputeUnits, _ warmup: Int, _ iters: Int) -> String {
        let name = url.deletingPathExtension().lastPathComponent
        let cfg = MLModelConfiguration(); cfg.computeUnits = units
        let tLoad = CFAbsoluteTimeGetCurrent()
        guard let model = try? MLModel(contentsOf: url, configuration: cfg) else { return "\(name) [\(label)] LOAD FAILED\n" }
        let loadMs = (CFAbsoluteTimeGetCurrent() - tLoad) * 1000
        var feats: [String: MLFeatureValue] = [:]
        var desc = ""
        for (k, d) in model.modelDescription.inputDescriptionsByName {
            if let ic = d.imageConstraint {
                var pb: CVPixelBuffer?
                CVPixelBufferCreate(nil, ic.pixelsWide, ic.pixelsHigh, ic.pixelFormatType, nil, &pb)
                if let pb { fill(pb); feats[k] = MLFeatureValue(pixelBuffer: pb) }
                desc += " \(k)=img\(ic.pixelsWide)x\(ic.pixelsHigh)"
            } else if let mc = d.multiArrayConstraint, let a = try? MLMultiArray(shape: mc.shape, dataType: mc.dataType) {
                feats[k] = MLFeatureValue(multiArray: a)
                desc += " \(k)=arr\(mc.shape.map { $0.intValue })"
            }
        }
        guard let input = try? MLDictionaryFeatureProvider(dictionary: feats) else { return "\(name) [\(label)] BAD INPUT\(desc)\n" }
        do {
            let tFirst = CFAbsoluteTimeGetCurrent()
            _ = try model.prediction(from: input)
            let firstMs = (CFAbsoluteTimeGetCurrent() - tFirst) * 1000
            for _ in 0..<warmup { _ = try model.prediction(from: input) }
            var ts: [Double] = []
            for _ in 0..<iters {
                let t = CFAbsoluteTimeGetCurrent()
                _ = try model.prediction(from: input)
                ts.append((CFAbsoluteTimeGetCurrent() - t) * 1000)
            }
            ts.sort()
            return String(format: "%@ [%@] load %.0f ms | first-predict %.0f ms | median %.1f | p95 %.1f | min %.1f | n=%d |%@\n",
                          name, label, loadMs, firstMs, ts[ts.count / 2], ts[Int(Double(ts.count) * 0.95)], ts[0], ts.count, desc)
        } catch {
            return "\(name) [\(label)] PREDICT FAILED: \(error.localizedDescription) |\(desc)\n"
        }
    }

    private static func fill(_ pb: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(pb, [])
        if let p = CVPixelBufferGetBaseAddress(pb) { memset(p, 128, CVPixelBufferGetDataSize(pb)) }
        CVPixelBufferUnlockBaseAddress(pb, [])
    }
}
