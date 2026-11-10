import simd

/// Geometry for pose-aware keyframe reprojection (ARKit conventions: camera x right, y up, looks down -z).
enum Reproject {
    static let flip = simd_float4x4(diagonal: SIMD4<Float>(1, -1, -1, 1))   // ARKit camera <-> image convention (y down, z forward)

    static func K(fx: Float, fy: Float, cx: Float, cy: Float) -> simd_float3x3 {
        simd_float3x3(columns: (SIMD3(fx, 0, 0), SIMD3(0, fy, 0), SIMD3(cx, cy, 1)))
    }

    /// Homography mapping a pixel of the CURRENT view to the pixel of keyframe `key` that sees the same point of a
    /// fronto-parallel plane z = D (current camera, image convention). Poses are ARKit camera->world transforms.
    static func homography(curToWorld: simd_float4x4, keyToWorld: simd_float4x4,
                           Kcur: simd_float3x3, Kkey: simd_float3x3, D: Float) -> simd_float3x3 {
        let M = flip * keyToWorld.inverse * curToWorld * flip          // current image-cam -> key image-cam
        let R = simd_float3x3(SIMD3(M.columns.0.x, M.columns.0.y, M.columns.0.z),
                              SIMD3(M.columns.1.x, M.columns.1.y, M.columns.1.z),
                              SIMD3(M.columns.2.x, M.columns.2.y, M.columns.2.z))
        let t = SIMD3(M.columns.3.x, M.columns.3.y, M.columns.3.z)
        // n = (0,0,1): outer product t * n^T only fills the third column
        let tn = simd_float3x3(columns: (SIMD3<Float>(0, 0, 0), SIMD3<Float>(0, 0, 0), t))
        return Kkey * (R + tn * (1 / D)) * Kcur.inverse
    }

    static func angleDeg(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let fa = -SIMD3(a.columns.2.x, a.columns.2.y, a.columns.2.z), fb = -SIMD3(b.columns.2.x, b.columns.2.y, b.columns.2.z)
        return acos(max(-1, min(1, simd_dot(fa, fb)))) * 180 / .pi
    }

    /// Viewpoint dissimilarity between two cameras (smaller = better source): degrees of rotation + cm of translation.
    static func viewDistance(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
        let fa = -SIMD3(a.columns.2.x, a.columns.2.y, a.columns.2.z), fb = -SIMD3(b.columns.2.x, b.columns.2.y, b.columns.2.z)
        let ang = acos(max(-1, min(1, simd_dot(fa, fb)))) * 180 / .pi
        let pa = SIMD3(a.columns.3.x, a.columns.3.y, a.columns.3.z), pb = SIMD3(b.columns.3.x, b.columns.3.y, b.columns.3.z)
        return ang + simd_distance(pa, pb) * 100
    }
}
