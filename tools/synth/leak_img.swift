import Vision
import CoreImage
import Foundation
// usage: leak_img dir  -> "<filename> <person pixels>" per jpg, measured by Vision on the OUTPUT stills
let dir = URL(fileURLWithPath: CommandLine.arguments[1])
let files = (try! FileManager.default.contentsOfDirectory(atPath: dir.path)).filter { $0.hasSuffix(".jpg") }.sorted()
let req = VNGeneratePersonSegmentationRequest(); req.qualityLevel = .balanced; req.outputPixelFormat = kCVPixelFormatType_OneComponent8
for f in files {
    autoreleasepool {
        try? VNImageRequestHandler(url: dir.appendingPathComponent(f)).perform([req])
        var c = 0
        if let m = (req.results?.first as? VNPixelBufferObservation)?.pixelBuffer {
            CVPixelBufferLockBaseAddress(m, .readOnly)
            let w = CVPixelBufferGetWidth(m), h = CVPixelBufferGetHeight(m), bpr = CVPixelBufferGetBytesPerRow(m)
            let base = CVPixelBufferGetBaseAddress(m)!.assumingMemoryBound(to: UInt8.self)
            for y in 0..<h { for x in 0..<w where base[y * bpr + x] > 127 { c += 1 } }
            CVPixelBufferUnlockBaseAddress(m, .readOnly)
            let img = CIImage(contentsOf: dir.appendingPathComponent(f))!
            c = Int(Double(c) * Double(img.extent.width * img.extent.height) / Double(w * h))
        }
        print(f, c)
    }
}
