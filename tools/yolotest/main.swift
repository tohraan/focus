import Vision
import CoreML
import AppKit

// Mac-side check of YoloDecode against the Python reference. Usage: yolo_decode_test <model.mlpackage> <img>...
let args = CommandLine.arguments
let compiled = try MLModel.compileModel(at: URL(fileURLWithPath: args[1]))
let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndNeuralEngine
let vm = try VNCoreMLModel(for: MLModel(contentsOf: compiled, configuration: cfg))
for path in args.dropFirst(2) {
    let req = VNCoreMLRequest(model: vm); req.imageCropAndScaleOption = .scaleFill
    let h = VNImageRequestHandler(url: URL(fileURLWithPath: path))
    let t = CFAbsoluteTimeGetCurrent()
    try h.perform([req])
    let ms = (CFAbsoluteTimeGetCurrent() - t) * 1000
    let obs = (req.results as? [VNCoreMLFeatureValueObservation]) ?? []
    var det: MLMultiArray?, pro: MLMultiArray?
    for o in obs { if let a = o.featureValue.multiArrayValue { if a.shape.count == 3 { det = a } else { pro = a } } }
    guard let d = det, let p = pro else { print(path, "no outputs"); continue }
    let t2 = CFAbsoluteTimeGetCurrent()
    let r = YoloDecode.decode(det: d, protos: p, inW: 1280, inH: 960)
    let dms = (CFAbsoluteTimeGetCurrent() - t2) * 1000
    print(String(format: "%@ persons=%d scores=%@ model %.0f ms decode %.1f ms",
                 (path as NSString).lastPathComponent, r.count, r.scores.map { String(format: "%.2f", $0) }.joined(separator: ","), ms, dms))
    let name = "/private/tmp/claude-501/-Users-tohraan/c872b4cf-cfe5-454e-94fb-c8dc5028c1cc/scratchpad/yt/mask_\((path as NSString).lastPathComponent).pgm"
    var out = Data("P5\n\(r.w) \(r.h)\n255\n".utf8); out.append(contentsOf: r.mask)
    try out.write(to: URL(fileURLWithPath: name))
}
