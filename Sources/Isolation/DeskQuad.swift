import Foundation
import simd

/// Pure geometry for the tap-the-corners desk flow (no ARKit, so it is unit-testable).
enum DeskQuad {
    /// Where a camera ray meets the horizontal plane y = planeY (nil if parallel or behind the origin).
    static func rayPlane(origin o: SIMD3<Float>, direction d: SIMD3<Float>, planeY: Float) -> SIMD3<Float>? {
        guard abs(d.y) > 1e-4 else { return nil }
        let t = (planeY - o.y) / d.y
        guard t > 0 else { return nil }
        return o + d * t
    }

    /// Corners sorted by angle around their centre (xz plane), so the polygon never self-intersects.
    static func ordered(_ pts: [SIMD3<Float>]) -> [SIMD3<Float>] {
        guard pts.count > 2 else { return pts }
        let c = pts.reduce(SIMD3<Float>(), +) / Float(pts.count)
        return pts.sorted { atan2($0.z - c.z, $0.x - c.x) < atan2($1.z - c.z, $1.x - c.x) }
    }

    /// Height above the desk of the thing the user tapped: the point on the vertical line through the quad centre
    /// that is closest to the tap ray.  Clamped to a sane range.
    static func height(rayOrigin o: SIMD3<Float>, rayDirection d: SIMD3<Float>, corners: [SIMD3<Float>], planeY: Float) -> Float? {
        guard !corners.isEmpty else { return nil }
        let c = corners.reduce(SIMD3<Float>(), +) / Float(corners.count)
        let v = SIMD3<Float>(0, 1, 0), w0 = o - SIMD3(c.x, planeY, c.z)
        let a = simd_dot(d, d), b = simd_dot(d, v), dd = simd_dot(d, w0), e = simd_dot(v, w0)
        let den = a - b * b
        guard den > 1e-6 else { return nil }
        let s = (a * e - b * dd) / den
        return min(1.2, max(0.1, s))
    }

    /// Mask reach for the depth path: the farthest corner (horizontal) plus a margin.
    static func reach(corners: [SIMD3<Float>], camera: SIMD3<Float>) -> Float {
        let far = corners.map { simd_length(SIMD2($0.x - camera.x, $0.z - camera.z)) }.max() ?? 1
        return min(2.5, max(0.3, far + 0.2))
    }

    /// Fixed-height assumption for desks where plane detection never fires.
    static func assumedDesk(cameraPosition c: SIMD3<Float>, forward f: SIMD3<Float>) -> [SIMD3<Float>] { DeskRect.assumed(cameraPosition: c, forward: f).corners }
}

/// An adjustable rectangle on a horizontal plane: the assumed desk the user can trim (width, depth, level) or re-fit.
struct DeskRect {
    static let dropBelowCamera: Float = 0.55
    var center: SIMD3<Float>          // centre of the rectangle; y = plane height
    var right: SIMD3<Float>           // unit, horizontal
    var forward: SIMD3<Float>         // unit, horizontal
    var width: Float                  // along `right`
    var depth: Float                  // along `forward`

    /// Four corners in outline order.
    var corners: [SIMD3<Float>] {
        let r = right * (width / 2), f = forward * (depth / 2)
        return [center - r - f, center + r - f, center + r + f, center - r + f]
    }

    /// 0.7 x 0.45 m, 0.55 m below the camera, centred where the camera points when tilted down (else 0.6 m ahead).
    static func assumed(cameraPosition c: SIMD3<Float>, forward f: SIMD3<Float>, width: Float = 0.7, depth: Float = 0.45) -> DeskRect {
        var fh = SIMD3<Float>(f.x, 0, f.z)
        fh = simd_length(fh) > 1e-4 ? simd_normalize(fh) : SIMD3(0, 0, -1)
        let right = SIMD3<Float>(-fh.z, 0, fh.x)             // camera-right on the horizontal plane
        let y = c.y - dropBelowCamera
        var ctr = c + fh * 0.6
        if f.y < -0.15 {                                     // looking down: the desk spot under the crosshair
            let t = (y - c.y) / f.y
            let hit = c + f * t
            let dist = simd_length(SIMD2(hit.x - c.x, hit.z - c.z))
            if dist > 0.3 && dist < 1.5 { ctr = hit }
        }
        return DeskRect(center: SIMD3(ctr.x, y, ctr.z), right: right, forward: fh, width: width, depth: depth)
    }

    /// Rectangle from two opposite corners, aligned to this rectangle's axes (the camera heading when assumed).
    func fitted(a: SIMD3<Float>, b: SIMD3<Float>) -> DeskRect {
        let d = b - a
        var r = self
        r.center = SIMD3((a.x + b.x) / 2, center.y, (a.z + b.z) / 2)
        r.width = max(0.15, abs(simd_dot(d, right)))
        r.depth = max(0.15, abs(simd_dot(d, forward)))
        return r
    }
}
