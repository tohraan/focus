import Vision
import CoreVideo

struct MaskOutput {
    var vision: CVPixelBuffer?         // fine matte, sensor orientation
    var yolo: YoloDecode.Result?       // coarse union mask of detected people
    var visionMs = 0.0, yoloMs = 0.0
}

/// Union of Apple Vision person segmentation (fine edges) and YOLO-seg (recall on small/turned-away people).
final class PersonSegmenter {
    private let request: VNGeneratePersonSegmentationRequest
    private let handler = VNSequenceRequestHandler()
    let yolo: YoloSegmenter?
    var useYolo = true

    init(quality: VNGeneratePersonSegmentationRequest.QualityLevel = .balanced, yoloModel: String? = "Yolo11sSeg") {
        request = VNGeneratePersonSegmentationRequest()
        request.qualityLevel = quality
        request.outputPixelFormat = kCVPixelFormatType_OneComponent8
        yolo = yoloModel.flatMap { YoloSegmenter(model: $0) }
    }

    private static func copy(_ pb: CVPixelBuffer) -> YoloDecode.VisionMask? {
        CVPixelBufferLockBaseAddress(pb, .readOnly); defer { CVPixelBufferUnlockBaseAddress(pb, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(pb) else { return nil }
        let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb), bpr = CVPixelBufferGetBytesPerRow(pb)
        var d = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h { memcpy(&d[y * w], base + y * bpr, w) }
        return YoloDecode.VisionMask(data: d, w: w, h: h)
    }

    func segment(_ pb: CVPixelBuffer) -> MaskOutput {
        var out = MaskOutput()
        let t0 = ProcessInfo.processInfo.systemUptime
        try? handler.perform([request], on: pb, orientation: .up)
        out.vision = (request.results?.first as? VNPixelBufferObservation)?.pixelBuffer
        let t1 = ProcessInfo.processInfo.systemUptime
        out.visionMs = (t1 - t0) * 1000
        if useYolo, let y = yolo {
            out.yolo = y.submit(pb, vision: out.vision.flatMap { Self.copy($0) })      // async: uses latest finished result, never blocks the frame loop
            out.yoloMs = y.lastMs
        }
        return out
    }
}
