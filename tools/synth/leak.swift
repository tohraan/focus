import AVFoundation
import Vision
import CoreImage
// usage: leak in.mp4 pw ph  -> one line per frame: "<frame> <p1 person px> <p2 person px>" (panel 1 = Memory, 2 = Keyframes), measured by Vision on the OUTPUT
let a = CommandLine.arguments
let pw = Int(a[2])!, ph = Int(a[3])!
let asset = AVAsset(url: URL(fileURLWithPath: a[1]))
let tr = asset.tracks(withMediaType: .video)[0]
let rd = try! AVAssetReader(asset: asset)
let out = AVAssetReaderTrackOutput(track: tr, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
rd.add(out); rd.startReading()
let ci = CIContext()
let req = VNGeneratePersonSegmentationRequest(); req.qualityLevel = .balanced; req.outputPixelFormat = kCVPixelFormatType_OneComponent8
var n = 0
while let s = out.copyNextSampleBuffer() {
    autoreleasepool {
        guard let pb = CMSampleBufferGetImageBuffer(s) else { return }
        let full = CIImage(cvPixelBuffer: pb)
        let H = Int(full.extent.height)
        var counts: [Int] = []
        for p in 1...2 {
            let crop = full.cropped(to: CGRect(x: p * pw, y: H - ph, width: pw, height: ph)).transformed(by: CGAffineTransform(translationX: CGFloat(-p * pw), y: CGFloat(-(H - ph))))
            var dst: CVPixelBuffer?
            CVPixelBufferCreate(nil, pw, ph, kCVPixelFormatType_32BGRA, nil, &dst)
            ci.render(crop, to: dst!)
            try? VNImageRequestHandler(cvPixelBuffer: dst!).perform([req])
            var c = 0
            if let m = (req.results?.first as? VNPixelBufferObservation)?.pixelBuffer {
                CVPixelBufferLockBaseAddress(m, .readOnly)
                let mw = CVPixelBufferGetWidth(m), mh = CVPixelBufferGetHeight(m), bpr = CVPixelBufferGetBytesPerRow(m)
                let base = CVPixelBufferGetBaseAddress(m)!.assumingMemoryBound(to: UInt8.self)
                for y in 0..<mh { for x in 0..<mw where base[y * bpr + x] > 127 { c += 1 } }
                CVPixelBufferUnlockBaseAddress(m, .readOnly)
                c = Int(Double(c) * Double(pw * ph) / Double(mw * mh))      // normalise to panel pixels
            }
            counts.append(c)
        }
        print(n, counts[0], counts[1]); n += 1
    }
}
