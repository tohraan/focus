import Vision
import simd
import CoreVideo

/// Rotation-only camera track from plain video: frame-to-frame homography -> R = K^-1 H K, accumulated.
/// Output convention: ARKit camera->world rotation (x right, y up, looks down -z), world = first frame's camera.
final class VideoPose {
    let K: simd_float3x3
    private var prev: CVPixelBuffer?
    private(set) var R = matrix_identity_float3x3     // camera(now)->world, ARKit convention
    init(width: Float, height: Float, hfovDeg: Float = 63) {
        let f = (width / 2) / tan(hfovDeg * .pi / 360)
        K = simd_float3x3(columns: (SIMD3(f, 0, 0), SIMD3(0, f, 0), SIMD3(width / 2, height / 2, 1)))
    }

    /// Vision's warp for (prev -> cur); nil if registration failed.
    static func warp(from a: CVPixelBuffer, to b: CVPixelBuffer) -> simd_float3x3? {
        let req = VNHomographicImageRegistrationRequest(targetedCVPixelBuffer: b)
        let h = VNImageRequestHandler(cvPixelBuffer: a, options: [:])
        try? h.perform([req])
        return (req.results?.first as? VNImageHomographicAlignmentObservation)?.warpTransform
    }

    /// Rotation (image convention, y down, z forward) best explaining H on a grid of rays (Wahba, polar iteration).
    /// Returns the rotation and its mean pixel residual.
    static func rotation(H: simd_float3x3, K: simd_float3x3) -> (R: simd_float3x3, err: Float) {
        let w = K[2][0] * 2, h = K[2][1] * 2
        var B = simd_float3x3(0)
        var pairs: [(SIMD3<Float>, SIMD3<Float>)] = []
        for iy in 0..<5 { for ix in 0..<5 {
            let p = SIMD3<Float>(w * (0.1 + 0.2 * Float(ix)), h * (0.1 + 0.2 * Float(iy)), 1)
            var q = H * p; q /= q.z
            let a = simd_normalize(K.inverse * p), b = simd_normalize(K.inverse * q)
            B += simd_float3x3(columns: (b * a.x, b * a.y, b * a.z))
            pairs.append((p, q))
        } }
        var Q = B
        for _ in 0..<20 { Q = (Q + Q.inverse.transpose) * 0.5 }
        if Q.determinant < 0 { Q = Q * -1 }
        let Hm = K * Q * K.inverse
        var e: Float = 0
        for (p, q) in pairs { var r = Hm * p; r /= r.z; e += simd_distance(SIMD2(r.x, r.y), SIMD2(q.x, q.y)) }
        return (Q, e / Float(pairs.count))
    }

    func step(_ pb: CVPixelBuffer) -> simd_float4x4 {
        defer { prev = pb }
        if let p = prev, let H = Self.warp(from: p, to: pb) {
            let Rimg = Self.rotation(H: H, K: K).R
            let f = simd_float3x3(diagonal: SIMD3(1, -1, -1))
            R = R * (f * Rimg.transpose * f)   // H maps prev->cur rays, so cur->prev is the transpose
        }
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4(R.columns.0, 0); m.columns.1 = SIMD4(R.columns.1, 0); m.columns.2 = SIMD4(R.columns.2, 0)
        return m
    }
}
