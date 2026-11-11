import AVFoundation
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import simd

/// Runs the live engines (Memory and Keyframes) on a video file instead of the camera and writes
/// an MP4: Original | Memory | Keyframes. Pose is rotation-only, estimated from the footage (see VideoPose).
final class VideoRunner {
    let renderer = Renderer()
    private let ci = CIContext(options: [.useSoftwareRenderer: false, .workingColorSpace: NSNull(), .outputColorSpace: NSNull()])   // no colour management: bytes in = bytes out

    private static func orientation(_ t: CGAffineTransform) -> CGImagePropertyOrientation {
        switch (Int(round(t.a)), Int(round(t.b)), Int(round(t.c)), Int(round(t.d))) {
        case (0, 1, -1, 0): return .right
        case (0, -1, 1, 0): return .left
        case (-1, 0, 0, -1): return .down
        default: return .up
        }
    }

    /// BGRA -> full-range BT.601 NV12 on the CPU, the exact inverse of the shader's `ycbcr()`. (Core Image applies a brightness curve here.)
    static func convertToNV12(_ src: CVPixelBuffer, into dst: CVPixelBuffer) {
        CVPixelBufferLockBaseAddress(src, .readOnly); CVPixelBufferLockBaseAddress(dst, [])
        defer { CVPixelBufferUnlockBaseAddress(src, .readOnly); CVPixelBufferUnlockBaseAddress(dst, []) }
        let w = CVPixelBufferGetWidth(src), h = CVPixelBufferGetHeight(src)
        let sb = CVPixelBufferGetBaseAddress(src)!.assumingMemoryBound(to: UInt8.self), sbpr = CVPixelBufferGetBytesPerRow(src)
        let yb = CVPixelBufferGetBaseAddressOfPlane(dst, 0)!.assumingMemoryBound(to: UInt8.self), ybpr = CVPixelBufferGetBytesPerRowOfPlane(dst, 0)
        let cb = CVPixelBufferGetBaseAddressOfPlane(dst, 1)!.assumingMemoryBound(to: UInt8.self), cbpr = CVPixelBufferGetBytesPerRowOfPlane(dst, 1)
        for y in 0..<h {
            for x in 0..<w {
                let o = y * sbpr + x * 4
                let b = Float(sb[o]), g = Float(sb[o + 1]), r = Float(sb[o + 2])
                yb[y * ybpr + x] = UInt8(max(0, min(255, (0.299 * r + 0.587 * g + 0.114 * b).rounded())))
            }
        }
        for y in 0..<(h / 2) {
            for x in 0..<(w / 2) {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for dy in 0..<2 { for dx in 0..<2 { let o = (2 * y + dy) * sbpr + (2 * x + dx) * 4; b += Float(sb[o]); g += Float(sb[o + 1]); r += Float(sb[o + 2]) } }
                r /= 4; g /= 4; b /= 4
                cb[y * cbpr + x * 2] = UInt8(max(0, min(255, (128 - 0.168736 * r - 0.331264 * g + 0.5 * b).rounded())))
                cb[y * cbpr + x * 2 + 1] = UInt8(max(0, min(255, (128 + 0.5 * r - 0.418688 * g - 0.081312 * b).rounded())))
            }
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
                var img = CIImage(cvPixelBuffer: src, options: [.colorSpace: NSNull()]).oriented(o)
                img = img.transformed(by: CGAffineTransform(translationX: -img.extent.minX, y: -img.extent.minY))
                let w = Int(img.extent.width), h = Int(img.extent.height)
                if pool == nil { pool = makeNV12Pool(w, h); size = (w, h) }
                var dst: CVPixelBuffer?
                CVPixelBufferPoolCreatePixelBuffer(nil, pool!, &dst)
                guard let d = dst else { return }
                CVBufferSetAttachment(d, kCVImageBufferYCbCrMatrixKey, kCVImageBufferYCbCrMatrix_ITU_R_601_4, .shouldPropagate)
                CVBufferSetAttachment(d, kCVImageBufferColorPrimariesKey, kCVImageBufferColorPrimaries_ITU_R_709_2, .shouldPropagate)
                CVBufferSetAttachment(d, kCVImageBufferTransferFunctionKey, kCVImageBufferTransferFunction_sRGB, .shouldPropagate)   // else Core Image encodes with the 709 curve and everything comes out ~10 levels brighter
                var tmp: CVPixelBuffer?
                CVPixelBufferCreate(nil, w, h, kCVPixelFormatType_32BGRA, [kCVPixelBufferIOSurfacePropertiesKey as String: [:]] as CFDictionary, &tmp)
                guard let t = tmp else { return }
                ci.render(img, to: t, bounds: img.extent, colorSpace: nil)   // orientation only (BGRA -> BGRA), no colour conversion
                VideoRunner.convertToNV12(t, into: d)
                body(d, CMTimeGetSeconds(CMSampleBufferGetPresentationTimeStamp(s)))
            }
        }
        return (size.w, size.h, Double(track.nominalFrameRate))
    }

    private func label(_ text: String, width: Int) -> CIImage {
        CIImage(color: CIColor(red: 0, green: 0, blue: 0, alpha: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: width, height: 34))
    }

    static func writeJPEG(_ img: CGImage, to url: URL, quality: Double = 0.93) {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { return }
        CGImageDestinationAddImage(dest, img, [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        CGImageDestinationFinalize(dest)
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
        var KfromFile: simd_float3x3?
        let poseFile = input.deletingPathExtension().appendingPathExtension("poses.txt")
        let altPoseFile = input.deletingLastPathComponent().appendingPathComponent(input.deletingPathExtension().lastPathComponent + "_poses.txt")
        if let txt = try? String(contentsOf: FileManager.default.fileExists(atPath: altPoseFile.path) ? altPoseFile : poseFile, encoding: .utf8) {
            // ground-truth camera rotations (camera -> world) for synthetic handheld clips: what ARKit's gyro would deliver
            var loaded: [simd_float4x4] = []
            for line in txt.split(separator: "\n") {
                let t = line.split(separator: " ")
                if t.first == "K", t.count >= 5, let fx = Float(t[1]), let fy = Float(t[2]), let cx = Float(t[3]), let cy = Float(t[4]) {
                    KfromFile = simd_float3x3(columns: (SIMD3(fx, 0, 0), SIMD3(0, fy, 0), SIMD3(cx, cy, 1)))
                } else if t.count == 10 {
                    let v = t[1...].compactMap { Float($0) }
                    if v.count == 9 { loaded.append(simd_float4x4(columns: (SIMD4(v[0], v[1], v[2], 0), SIMD4(v[3], v[4], v[5], 0), SIMD4(v[6], v[7], v[8], 0), SIMD4(0, 0, 0, 1)))) }
                }
            }
            if loaded.count >= poses.count { poses = Array(loaded.prefix(poses.count)) }
        }

        let W = info.w, H = info.h
        if ProcessInfo.processInfo.environment["FOCUS_DUMPPOSE"] != nil {      // bench debugging: camera->world rotation per frame
            var t = "K \(vp.K[0][0]) \(vp.K[1][1]) \(vp.K[2][0]) \(vp.K[2][1])\n"
            for (i, m) in poses.enumerated() { t += "\(i) " + [m.columns.0, m.columns.1, m.columns.2].map { "\($0.x) \($0.y) \($0.z)" }.joined(separator: " ") + "\n" }
            try? t.write(to: output.deletingLastPathComponent().appendingPathComponent(output.deletingPathExtension().lastPathComponent + "_poses.txt"), atomically: true, encoding: .utf8)
        }
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
        let argv = CommandLine.arguments
        let stillsEvery: Int? = argv.firstIndex(of: "-stills").flatMap { $0 + 1 < argv.count ? Int(argv[$0 + 1]) : nil }   // -stills N: no video, save every Nth frame as JPEG
        let toVideo = stillsEvery == nil
        let framesDir = output.deletingLastPathComponent().appendingPathComponent(output.deletingPathExtension().lastPathComponent + "_frames")
        if toVideo { wr.add(inp); wr.startWriting(); wr.startSession(atSourceTime: .zero) }
        else { try? FileManager.default.removeItem(at: framesDir); try? FileManager.default.createDirectory(at: framesDir, withIntermediateDirectories: true) }

        let labels = [label("ORIGINAL", width: pw), label("MEMORY 11:05 + LaMa", width: pw), label("KEYFRAMES (R2)", width: pw)]
        renderer.prepareOffline()
        let texMem = renderer.makeTarget(width: W, height: H), texKF = renderer.makeTarget(width: W, height: H)
        let Kmat = KfromFile ?? vp.K
        let duration = (times.last ?? 0) - (times.first ?? 0) + 1.0 / max(info.fps, 1)
        let t0run = Date()

        var idx = 0, pass = 0, written = 0, failed = 0
        var dumped = false, why = ""
        let progURL = output.deletingLastPathComponent().appendingPathComponent("progress.txt")
        var lastProg = Date()
        // pass 0 = warm-up (memory learns, nothing written); pass 1 = written
        let warm = !still && !CommandLine.arguments.contains("-nowarm")      // -nowarm: honest single pass, no pre-learning loop
        for p in (warm ? 0..<2 : 1..<2) {
            pass = p; idx = 0
            _ = try decode(input) { pb, t in
                guard idx < poses.count else { return }
                let pose = poses[idx], ts = t + Double(pass) * duration
                renderer.engine = 0
                let okA = renderer.processOffline(pb: pb, pose: pose, K: Kmat, ts: ts, target: texMem)
                renderer.engine = 1
                let okB = renderer.processOffline(pb: pb, pose: pose, K: Kmat, ts: ts, target: texKF)
                if !(okA && okB) { failed += 1 }
                if pass == 1, !toVideo {
                    if idx % (stillsEvery ?? 6) == 0 {
                        func save(_ t: MTLTexture, _ name: String) {
                            // raw BGRA bytes -> sRGB JPEG, no Core Image colour conversion
                            let w = t.width, h = t.height
                            var bytes = [UInt8](repeating: 0, count: w * h * 4)
                            t.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
                            guard let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                                      bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue),
                                  let img = ctx.makeImage() else { return }
                            VideoRunner.writeJPEG(img, to: framesDir.appendingPathComponent(String(format: "%@_%04d.jpg", name, idx)))
                        }
                        save(texMem, "mem"); save(texKF, "kf")
                        written += 1
                    }
                } else if pass == 1 {
                    if !dumped, let u = dumpFrame { dumped = true; if let cg = ci.createCGImage(CIImage(cvPixelBuffer: pb), from: CIImage(cvPixelBuffer: pb).extent) { VideoRunner.writeJPEG(cg, to: u, quality: 0.8) } }
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
                if idx % 10 == 0 {
                    let now = Date()
                    let line = String(format: "%@ frame %d  %.0f ms/frame (last 10)  %@\n", input.lastPathComponent, idx, now.timeIntervalSince(lastProg) * 100, renderer.kfStatus + " | " + renderer.miganLast)
                    lastProg = now
                    if let h = try? FileHandle(forWritingTo: progURL) { h.seekToEndOfFile(); h.write(Data(line.utf8)); try? h.close() }
                    else { try? line.write(to: progURL, atomically: true, encoding: .utf8) }
                }
            }
        }
        if toVideo {
            inp.markAsFinished()
            let sem = DispatchSemaphore(value: 0); wr.finishWriting { sem.signal() }; sem.wait()
            if wr.status != .completed { throw wr.error ?? NSError(domain: "video", code: 3) }
        }
        return "\(input.lastPathComponent) still=\(still) \(String(format: "%.0f", Date().timeIntervalSince(t0run) * 1000 / Double(max(written, 1)))) ms/frame: \(W)x\(H) \(poses.count) frames, wrote \(written), failed \(failed) \(why), kf \(renderer.kfStatus) | \(renderer.autoStatus)"
    }
}

extension VideoRunner {
    /// -colortest: source RGB vs NV12 bytes vs what the shader renders in "original" mode, at a few pixels.
    func colorTest(_ url: URL) throws {
        var done = false
        _ = try decode(url) { pb, _ in
            if done { return }; done = true
            let w = CVPixelBufferGetWidth(pb), h = CVPixelBufferGetHeight(pb)
            let tex = renderer.makeTarget(width: w, height: h)
            renderer.prepareOffline()
            _ = renderer.processOffline(pb: pb, pose: matrix_identity_float4x4, K: simd_float3x3(columns: (SIMD3(900, 0, 0), SIMD3(0, 900, 0), SIMD3(Float(w) / 2, Float(h) / 2, 1))), ts: 0, target: tex, mode: 1)
            var bytes = [UInt8](repeating: 0, count: w * h * 4)
            tex.getBytes(&bytes, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
            CVPixelBufferLockBaseAddress(pb, .readOnly)
            let yb = CVPixelBufferGetBaseAddressOfPlane(pb, 0)!.assumingMemoryBound(to: UInt8.self), ybpr = CVPixelBufferGetBytesPerRowOfPlane(pb, 0)
            for (x, y) in [(100, 100), (500, 300), (900, 600), (300, 700)] {
                let o = (y * w + x) * 4
                print("px (\(x),\(y)) NV12 Y \(yb[y * ybpr + x])  rendered(original) BGR \(bytes[o]),\(bytes[o + 1]),\(bytes[o + 2])")
            }
            CVPixelBufferUnlockBaseAddress(pb, .readOnly)
        }
    }
}

struct VideoBatch {
    /// Processes every .MOV in Documents/in -> Documents/out, writing a status line per clip.
    static func runAll() {
        if let i = CommandLine.arguments.firstIndex(of: "-colortest"), i + 1 < CommandLine.arguments.count { try? VideoRunner().colorTest(URL(fileURLWithPath: CommandLine.arguments[i + 1])); return }
        let args = CommandLine.arguments
        let docs = args.firstIndex(of: "-root").flatMap { $0 + 1 < args.count ? URL(fileURLWithPath: args[$0 + 1]) : nil }
            ?? FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let sub = args.firstIndex(of: "-indir").flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }   // -indir NAME => Documents/NAME -> Documents/NAME_out
        let inDir = docs.appendingPathComponent(sub ?? "in"), outDir = docs.appendingPathComponent(sub.map { "\($0)_out" } ?? "out")
        try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let clips = ((try? FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)) ?? [])
            .filter { ["mov", "mp4"].contains($0.pathExtension.lowercased()) }.sorted { $0.lastPathComponent < $1.lastPathComponent }
            .filter { c in                                       // -only a,b,c : just these clips (by file name without extension)
                guard let i = args.firstIndex(of: "-only"), i + 1 < args.count else { return true }
                return args[i + 1].split(separator: ",").map(String.init).contains(c.deletingPathExtension().lastPathComponent)
            }
        var log = ""
        try? "RUNNING\n".write(to: outDir.appendingPathComponent("status.txt"), atomically: true, encoding: .utf8)
        try? FileManager.default.removeItem(at: outDir.appendingPathComponent("progress.txt"))
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
