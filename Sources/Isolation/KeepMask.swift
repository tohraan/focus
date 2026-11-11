import Foundation
import Vision
import CoreImage
import CoreGraphics
import CoreVideo

/// What survives Isolation Mode: the user's hands (tracked live) plus everything between the viewer and the far edge of
/// the scanned table, i.e. desk, laptop, notebook and the space in between.  Everything farther away becomes the world.
final class KeepMask {
    struct Config {
        var reachMetres: Float = 1.0
        var useDepth = true
        var canvas = CGSize(width: 540, height: 720)
    }
    var config = Config()
    let ctx: CIContext

    private let hands = VNDetectHumanHandPoseRequest()
    private var prevKeep: CGImage?
    private(set) var lastHandCount = 0

    init(ctx: CIContext = CIContext()) {
        self.ctx = ctx
        hands.maximumHandCount = 2
    }

    func resetHistory() { prevKeep = nil }

    /// `depth` is Float32 metres, already oriented like `orientation` maps the camera image.
    /// Hand presence only (no mask built): what the scan's hands step needs, at a fraction of the cost of `keep`.
    func detectHandsOnly(for pb: CVPixelBuffer, orientation: CGImagePropertyOrientation) {
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: orientation, options: [:]).perform([hands])
        lastHandCount = hands.results?.count ?? 0
    }

    func keep(for pb: CVPixelBuffer, orientation: CGImagePropertyOrientation, depth: CVPixelBuffer?, volume: CGImage? = nil) -> CIImage {
        let W = config.canvas.width, H = config.canvas.height
        let canvasRect = CGRect(origin: .zero, size: config.canvas)
        try? VNImageRequestHandler(cvPixelBuffer: pb, orientation: orientation, options: [:]).perform([hands])

        // hands: capsules along the finger chains
        var k = CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: canvasRect)
        let hs = hands.results ?? []
        lastHandCount = hs.count
        if !hs.isEmpty {
            let g = bitmap(W, H) { c in
                for h in hs {
                    guard let pts = try? h.recognizedPoints(.all) else { continue }
                    let good = pts.values.filter { $0.confidence > 0.25 }
                    guard good.count >= 5 else { continue }
                    let xs = good.map { $0.location.x * W }, ys = good.map { $0.location.y * H }
                    let r = max(8, hypot(xs.max()! - xs.min()!, ys.max()! - ys.min()!) * 0.16)
                    c.setFillColor(CGColor(gray: 1, alpha: 1)); c.setStrokeColor(CGColor(gray: 1, alpha: 1))
                    c.setLineWidth(r * 1.6); c.setLineCap(.round); c.setLineJoin(.round)
                    let chains: [[VNHumanHandPoseObservation.JointName]] = [
                        [.wrist, .thumbCMC, .thumbMP, .thumbIP, .thumbTip], [.wrist, .indexMCP, .indexPIP, .indexDIP, .indexTip],
                        [.wrist, .middleMCP, .middlePIP, .middleDIP, .middleTip], [.wrist, .ringMCP, .ringPIP, .ringDIP, .ringTip],
                        [.wrist, .littleMCP, .littlePIP, .littleDIP, .littleTip]]
                    for ch in chains {
                        let p = ch.compactMap { pts[$0] }.filter { $0.confidence > 0.25 }.map { CGPoint(x: $0.location.x * W, y: $0.location.y * H) }
                        guard p.count >= 2 else { continue }
                        c.beginPath(); c.move(to: p[0]); p.dropFirst().forEach { c.addLine(to: $0) }; c.strokePath()
                    }
                    for p in good { c.fillEllipse(in: CGRect(x: p.location.x * W - r, y: p.location.y * H - r, width: 2 * r, height: 2 * r)) }
                }
            }
            k = CIImage(cgImage: g)
        }

        // everything closer than `reach` (table, laptop, notebook, and the space between)
        if config.useDepth, let d = depth {
            var di = CIImage(cvPixelBuffer: d, options: [.colorSpace: NSNull()])
            let e = di.extent
            di = di.transformed(by: CGAffineTransform(scaleX: W / e.width, y: H / e.height)).cropped(to: canvasRect)
            func ch(_ scale: CGFloat, _ bias: CGFloat) -> CIImage {
                di.applyingFilter("CIColorMatrix", parameters: [
                    "inputRVector": CIVector(x: scale, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: scale, y: 0, z: 0, w: 0),
                    "inputBVector": CIVector(x: scale, y: 0, z: 0, w: 0), "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0),
                    "inputBiasVector": CIVector(x: bias, y: bias, z: bias, w: 1)])
                    .applyingFilter("CIColorClamp", parameters: ["inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0), "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)])
            }
            let t = CGFloat(config.reachMetres), kk: CGFloat = 14
            let near = ch(-kk, t * kk)               // 1 inside reach, soft edge ~7 cm
            let valid = ch(40, -6)                   // depth holes (0) are not "near"
            let nearK = near.applyingFilter("CIMultiplyCompositing", parameters: [kCIInputBackgroundImageKey: valid])
            k = nearK.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: k])
        }

        // scanned table box (sensor-oriented raster) -> same orientation as the camera, then merged
        if let v = volume {
            var vi = CIImage(cgImage: v).oriented(orientation)
            let e = vi.extent
            vi = vi.transformed(by: CGAffineTransform(translationX: -e.minX, y: -e.minY)).transformed(by: CGAffineTransform(scaleX: W / e.width, y: H / e.height)).cropped(to: canvasRect)
            k = vi.applyingFilter("CIMaximumCompositing", parameters: [kCIInputBackgroundImageKey: k])
        }

        // feather, then blend with the previous mask to stop flicker
        k = k.clampedToExtent().applyingGaussianBlur(sigma: 2.5).cropped(to: canvasRect)
            .applyingFilter("CIColorMatrix", parameters: [
                "inputRVector": CIVector(x: 2.5, y: 0, z: 0, w: 0), "inputGVector": CIVector(x: 0, y: 2.5, z: 0, w: 0), "inputBVector": CIVector(x: 0, y: 0, z: 2.5, w: 0),
                "inputAVector": CIVector(x: 0, y: 0, z: 0, w: 0), "inputBiasVector": CIVector(x: -0.75, y: -0.75, z: -0.75, w: 1)])
            .applyingFilter("CIColorClamp", parameters: ["inputMinComponents": CIVector(x: 0, y: 0, z: 0, w: 0), "inputMaxComponents": CIVector(x: 1, y: 1, z: 1, w: 1)])
        if let p = prevKeep {
            k = k.applyingFilter("CIDissolveTransition", parameters: [kCIInputTargetImageKey: CIImage(cgImage: p), kCIInputTimeKey: 0.3]).cropped(to: canvasRect)
        }
        let out = ctx.createCGImage(k, from: canvasRect, format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        prevKeep = out
        return out.map { CIImage(cgImage: $0) } ?? k
    }

    private func bitmap(_ w: CGFloat, _ h: CGFloat, _ draw: (CGContext) -> Void) -> CGImage {
        let c = CGContext(data: nil, width: Int(w), height: Int(h), bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        c.setFillColor(CGColor(gray: 0, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: w, height: h)); draw(c)
        return c.makeImage()!
    }
}

/// Table scan: median of a far-edge depth percentile over the lower part of the view.
enum TableReach {
    /// Far edge (metres) of whatever fills the lower 60% of the depth map, or nil if too little valid depth.
    static func farEdge(depth d: CVPixelBuffer, percentile: Double = 0.85) -> Float? {
        CVPixelBufferLockBaseAddress(d, .readOnly); defer { CVPixelBufferUnlockBaseAddress(d, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(d) else { return nil }
        let w = CVPixelBufferGetWidth(d), h = CVPixelBufferGetHeight(d), bpr = CVPixelBufferGetBytesPerRow(d)
        var v = [Float](); v.reserveCapacity(w * h / 16)
        var y = Int(Double(h) * 0.4)
        while y < h {
            let row = (base + y * bpr).assumingMemoryBound(to: Float.self)
            var x = 0
            while x < w { let m = row[x]; if m.isFinite && m > 0.15 && m < 6 { v.append(m) }; x += 4 }
            y += 4
        }
        guard v.count > 40 else { return nil }
        v.sort()
        return v[min(v.count - 1, Int(Double(v.count) * percentile))]
    }
    static func reach(fromFarEdges e: [Float], margin: Float = 0.2) -> Float {
        guard !e.isEmpty else { return 1.0 }
        let m = e.sorted()[e.count / 2] + margin
        return min(2.0, max(0.5, m))
    }
}

/// Final composite: camera where `keep` is white, the world elsewhere.  No glow, no rim.
enum IsolationCompositor {
    static func compose(camera: CIImage, keep: CIImage, world: CIImage) -> CIImage {
        camera.applyingFilter("CIBlendWithRedMask", parameters: [kCIInputBackgroundImageKey: world.clampedToExtent(), kCIInputMaskImageKey: keep]).cropped(to: camera.extent)
    }
}
