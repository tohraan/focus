import AVFoundation
import CoreImage
import UIKit
import simd

/// Runs the live engines (Memory and Keyframes) on a video file instead of the camera and writes
/// an MP4: Original | Memory | Keyframes. Pose is rotation-only, estimated from the footage (see VideoPose).
final class VideoRunner {
    let renderer = Renderer()
    private let ci = CIContext(options: [.useSoftwareRenderer: false])

    private static func orientation(_ t: CGAffineTransform) -> CGImagePropertyOrientation {
        switch (Int(round(t.a)), Int(round(t.b)), Int(round(t.c)), Int(round(t.d))) {
        case (0, 1, -1, 0): return .right
        case (0, -1, 1, 0): return .left
        case (-1, 0, 0, -1): return .down
        default: return .up
        }
    }

    private func makeNV12Pool(_ w: Int, _ h: Int) -> CVPixelBufferPool? {
        var pool: CVPixelBufferPool?
        let attrs: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_420YpCbCr8BiPlanarFullRange,
            kCVPixelBufferWidthKey as String: w, kCVPixelBufferHeightKey as String: h,
            kCVPixelBufferIOSurfacePropertiesKey as String: [:], kCVPixelBufferMetalCompatibilityKey as String: true]
        CVPixelBufferPoolCreate(nil, nil, attrs as CFDictionary, &pool)
        return pool
    }

    /// Decode every frame as an upright NV12 buffer. `body` gets (buffer, pts seconds).
    private func decode(_ url: URL, body: (CVPixelBuffer, Double) -> Void) throws -> (w: Int, h: Int, fps: Double) {
        let asset = AVAsset(url: url)
        guard let track = asset.tracks(withMediaType: .video).first else { throw NSError(domain: "video", code: 1) }
        let o = Self.orientation(track.preferredTransform)
        let rd = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        rd.add(out); rd.startReading()
        var pool: CVPixelBufferPool?
        var size = (w: 0, h: 0)
        while let s = out.copyNextSampleBuffer() {
            autoreleasepool {
                guard let src = CMSampleBufferGetImageBuffer(s) else { return }
                var img = CIImage(cvPixelBuffer: src).oriented(o)
                img = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
                let w = Int(img.extent.width), h = Int(img.extent.height)
                if pool == nil { pool = makeNV12Pool(w, h); size = (w, h) }
                var dst: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, pool!, &dst)
                guard let d = dst else { return }
                CVBufferSetAttachment(d, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4, .shouldPropagate)
                ci.render(img, to: d, bounds: img.extent, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
                body(d, CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(s)))
            }
        }
        return (size.w, size.h, Double(track.nominalFrameRate))
    }

    private func label(_ text: String, width: Int) -> CIImage {
        let r = UIGraphicsImageRenderer(size: CGSize(width: width, height: 34))
        let img = r.image { c in
            UIColor.black.withAlphaComponent(0.6).setFill(); c.fill(CGRect(x: 0, y: 0, width: width, height: 34))
            (text as NSString).draw(at: CGPoint(x: 10, y: 6), withAttributes: [.font: UIFont.boldSystemFont(ofSize: 18), .foregroundColor: UIColor.white])
        }
        return CIImage(cgImage: img.cgImage!)
    }

    /// Returns a one-line report.
    func run(input: URL, output: URL, dumpFrame: URL?) throws -> String {
        // 1. pose track
        var poses: [simd_float4x4] = [], times: [Double] = []
        var vp: VideoPose?
        let still = CommandLine.arguments.contains("-still")      // camera on a table: identity pose, no warm-up loop
        let info = try decode(input) { pb, t in
            if vp == nil { vp = VideoPose(width: Float(CVPixelBufferGetWidth(pb)), height: Float(CVPixelBufferGetHeight(pb))) }
            poses.append(still ? matrix_identity_float4x4 : vp!.step(pb)); times.append(t)
        }
        guard let vp, !poses.isEmpty else { throw NSError(domain: "video", code: 2) }
        let W = info.w, H = info.h
        let big = Double(max(W, H)), sc = 960.0 / big
        let pw = Int(Double(W) * sc) / 2 * 2, ph = Int(Double(H) * sc) / 2 * 2
        let labelH = 34
        let outW = pw * 3, outH = ph + labelH

        // 2. writer
        try? FileManager.default.removeItem(at: output)
        let wr = try AVAssetWriter(outputURL: output, fileType: .mp4)
        let inp = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: outW, AVVideoHeightKey: outH,
            AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 14_000_000]])
        let ad = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: inp, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA, kCVPixelBufferWidthKey as String: outW, kCVPixelBufferHeightKey as String: outH])
        wr.add(inp); wr.startWriting(); wr.startSession(atSourceTime: .zero)

        let labels = [label("ORIGINAL", width: pw), label("MEMORY 11:05 + LaMa", width: pw), label("KEYFRAMES (R2)", width: pw)]
        renderer.prepareOffline()
        let texMem = renderer.makeTarget(width: W, height: H), texKF = renderer.makeTarget(width: W, height: H)
        let Kmat = vp.K
        let duration = (times.last ?? 0) - (times.first ?? 0) + 1.0 / max(info.fps, 1)
        let t0run = Date()

        var idx = 0, pass = 0, written = 0, failed = 0
        var dumped = false, why = ""
        // pass 0 = warm-up (memory learns, nothing written); pass 1 = written
        for p in (still ? 1..<2 : 0..<2) {
            pass = p; idx = 0
            _ = try decode(input) { pb, t in
                guard idx < poses.count else { return }
                let pose = poses[idx], ts = t + Double(pass) * duration
                renderer.engine = 0
                let okA = renderer.processOffline(pb: pb, pose: pose, K: Kmat, ts: ts, target: texMem)
                renderer.engine = 1
                let okB = renderer.processOffline(pb: pb, pose: pose, K: Kmat, ts: ts, target: texKF)
                if !(okA && okB) { failed += 1 }
                if pass == 1 {
                    if !dumped, let u = dumpFrame { dumped = true; try? UIImage(ciImage: CIImage(cvPixelBuffer: pb)).jpegData(compressionQuality: 0.8)?.write(to: u) }
                    var ob: CVPixelBuffer?
                    if let pl = ad.pixelBufferPool { CVPixelBufferPoolCreatePixelBuffer(nil, pl, &ob) } else { why = "no writer pool: \(wr.error.map { "\($0)" } ?? "nil")" }
                    if let ob {
                        func panel(_ img: CIImage, _ i: Int) -> CIImage {
                            img.transformed(by: CGAffineTransform(scaleX: CGFloat(pw) / img.extent.width, y: CGFloat(ph) / img.extent.height))
                               .transformed(by: CGAffineTransform(translationX: CGFloat(i * pw), y: CGFloat(labelH)))
                        }
                        func fromTex(_ t: MTLTexture) -> CIImage { (CIImage(mtlTexture: t, options: nil) ?? CIImage.empty()).oriented(.downMirrored) }
                        var comp = panel(CIImage(cvPixelBuffer: pb), 0).composited(over: CIImage(color: .black).cropped(to: CGRect(x: 0, y: 0, width: outW, height: outH)))
                        comp = panel(fromTex(texMem), 1).composited(over: comp)
                        comp = panel(fromTex(texKF), 2).composited(over: comp)
                        for (i, l) in labels.enumerated() { comp = l.transformed(by: CGAffineTransform(translationX: CGFloat(i * pw), y: 0)).composited(over: comp) }
                        ci.render(comp, to: ob)
                        while !inp.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
                        ad.append(ob, withPresentationTime: CMTime(seconds: t - (times.first ?? 0), preferredTimescale: 600))
                        written += 1
                    }
                }
                idx += 1
            }
        }
        inp.markAsFinished()
        let sem = DispatchSemaphore(value: 0); wr.finishWriting { sem.signal() }; sem.wait()
        if wr.status != .completed { throw wr.error ?? NSError(domain: "video", code: 3) }
        return "\(input.lastPathComponent) still=\(still) \(String(format: "%.0f", Date().timeIntervalSince(t0run) * 1000 / Double(max(written, 1)))) ms/frame: \(W)x\(H) \(poses.count) frames, wrote \(written), failed \(failed) \(why), kf \(renderer.kfStatus) | \(renderer.autoStatus)"
    }
}

struct VideoBatch {
    /// Processes every .MOV in Documents/in -> Documents/out, writing a status line per clip.
    static func runAll() {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let args = CommandLine.arguments
        let sub = args.firstIndex(of: "-indir").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }   // -indir NAME => Documents/NAME -> Documents/NAME_out
        let inDir = docs.appendingPathComponent(sub ?? "in"), outDir = docs.appendingPathComponent(sub.map { "\($0)_out" } ?? "out")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let clips = ((try? FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { ["mov", "mp4"].contains($0.pathExtension.lowercased()) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        var log = ""
        let runner = VideoRunner()
        for c in clips {
            let name = c.deletingPathExtension().lastPathComponent
            let t0 = Date()
            do {
                let line = try runner.run(input: c, output: outDir.appendingPathComponent("\(name)_focus.mp4"),
                                          dumpFrame: outDir.appendingPathComponent("\(name)_frame.jpg"))
                log += "OK \(line) in \(Int(Date().timeIntervalSince(t0))) s\n"
            } catch { log += "FAIL \(name): \(error)\n" }
            try? (log + "PROGRESS\n").write(to: outDir.appendingPathComponent("status.txt"), atomically: true, encoding: .utf8)
        }
        try? (log + "DONE\n").write(to: outDir.appendingPathComponent("status.txt"), atomically: true, encoding: .utf8)
    }
}
