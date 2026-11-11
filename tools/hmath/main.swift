import simd
import Foundation

func pose(pos: SIMD3<Float>, yawDeg: Float, pitchDeg: Float = 0) -> simd_float4x4 {
    let qy = simd_quatf(angle: yawDeg * .pi / 180, axis: SIMD3(0, 1, 0)), qx = simd_quatf(angle: pitchDeg * .pi / 180, axis: SIMD3(1, 0, 0))
    var m = simd_float4x4(qy * qx); m.columns.3 = SIMD4(pos, 1); return m
}
let Kc = Reproject.K(fx: 700, fy: 700, cx: 480, cy: 360), Kk = Reproject.K(fx: 700, fy: 700, cx: 480, cy: 360)
var worst: Float = 0
for (dx, yaw, D) in [(0.0, 0.0, 1.0), (0.10, 5.0, 1.0), (0.25, -8.0, 2.0), (-0.3, 12.0, 0.7)] as [(Float, Float, Float)] {
    let Tc = pose(pos: SIMD3(0.2, 0.1, 0.3), yawDeg: 3), Tk = pose(pos: SIMD3(0.2 + dx, 0.1, 0.3), yawDeg: 3 + yaw, pitchDeg: 2)
    let H = Reproject.homography(curToWorld: Tc, keyToWorld: Tk, Kcur: Kc, Kkey: Kk, D: D)
    var err: Float = 0
    for _ in 0..<200 {
        let X = SIMD3<Float>(Float.random(in: -0.6...0.6) * D, Float.random(in: -0.4...0.4) * D, D)   // on plane z=D (image conv)
        let pc = Kc * X; let uc = SIMD2(pc.x / pc.z, pc.y / pc.z)
        let Xw = Tc * Reproject.flip * SIMD4(X, 1)                    // to world
        let Xk = Reproject.flip * (Tk.inverse * Xw)                  // to key image-cam
        let pk = Kk * SIMD3(Xk.x, Xk.y, Xk.z); let truth = SIMD2(pk.x / pk.z, pk.y / pk.z)
        let h = H * SIMD3(uc.x, uc.y, 1); let got = SIMD2(h.x / h.z, h.y / h.z)
        err = max(err, simd_distance(truth, got))
    }
    print(String(format: "baseline %.2fm yaw %.0f D %.1f -> max reprojection error %.5f px", dx, yaw, D, err)); worst = max(worst, err)
}
print(worst < 0.01 ? "PASS" : "FAIL")
