import Foundation
import CoreGraphics
import simd

/// The scanned table as a box: the detected horizontal surface (world-space outline) extruded upward.
/// Projected into the camera it covers the table, a laptop or notebook standing on it, hands over it, and (because every
/// ray to the table crosses it) the space between the viewer and those objects.  Needs no LiDAR: only ARKit plane detection + pose.
struct TableVolume {
    var base: [SIMD3<Float>]      // world-space outline of the plane (metres)
    var height: Float = 0.45      // tall enough for a laptop screen

    /// Convex hull of the projected box, in sensor-image pixels (landscape, origin top-left).
    func screenHull(camera: simd_float4x4, fx: Float, fy: Float, cx: Float, cy: Float, near: Float = 0.05) -> [CGPoint] {
        guard base.count >= 3 else { return [] }
        let inv = camera.inverse
        func cam(_ p: SIMD3<Float>) -> SIMD3<Float> { let v = inv * SIMD4(p, 1); return SIMD3(v.x, v.y, v.z) }
        let lo = base.map(cam), hi = base.map { cam($0 + SIMD3(0, height, 0)) }
        var edges: [(SIMD3<Float>, SIMD3<Float>)] = []
        for i in 0..<lo.count {
            let j = (i + 1) % lo.count
            edges.append((lo[i], lo[j])); edges.append((hi[i], hi[j])); edges.append((lo[i], hi[i]))
        }
        var pts = [CGPoint]()
        for (a, b) in edges {
            // keep the part in front of the camera (camera looks down -Z)
            let ia = a.z < -near, ib = b.z < -near
            if !ia && !ib { continue }
            var p = a, q = b
            if !ia { let t = (-near - a.z) / (b.z - a.z); p = a + (b - a) * t }
            if !ib { let t = (-near - a.z) / (b.z - a.z); q = a + (b - a) * t }
            for v in [p, q] { let d = -v.z; pts.append(CGPoint(x: Double(cx + fx * v.x / d), y: Double(cy - fy * v.y / d))) }
        }
        return TableVolume.hull(pts)
    }

    /// Monotone-chain convex hull.
    static func hull(_ pts: [CGPoint]) -> [CGPoint] {
        let p = pts.sorted { $0.x == $1.x ? $0.y < $1.y : $0.x < $1.x }
        guard p.count > 2 else { return p }
        func cross(_ o: CGPoint, _ a: CGPoint, _ b: CGPoint) -> CGFloat { (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x) }
        var h = [CGPoint]()
        for q in p { while h.count >= 2 && cross(h[h.count - 2], h[h.count - 1], q) <= 0 { h.removeLast() }; h.append(q) }
        let t = h.count + 1
        for q in p.reversed().dropFirst() { while h.count >= t && cross(h[h.count - 2], h[h.count - 1], q) <= 0 { h.removeLast() }; h.append(q) }
        h.removeLast()
        return h
    }

    /// White-on-black mask in sensor orientation, `canvas` pixels wide/high (same aspect as the sensor image).
    func raster(camera: simd_float4x4, intrinsics K: simd_float3x3, imageSize: CGSize, canvas: CGSize) -> CGImage? {
        let hull = screenHull(camera: camera, fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1])
        guard hull.count >= 3 else { return nil }
        let sx = canvas.width / imageSize.width, sy = canvas.height / imageSize.height
        let c = CGContext(data: nil, width: Int(canvas.width), height: Int(canvas.height), bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        c.setFillColor(CGColor(gray: 0, alpha: 1)); c.fill(CGRect(origin: .zero, size: canvas))
        c.setFillColor(CGColor(gray: 1, alpha: 1))
        // image y runs down, CGContext y runs up
        c.beginPath()
        for (i, p) in hull.enumerated() {
            let q = CGPoint(x: p.x * sx, y: canvas.height - p.y * sy)
            if i == 0 { c.move(to: q) } else { c.addLine(to: q) }
        }
        c.closePath(); c.fillPath()
        return c.makeImage()
    }
}
