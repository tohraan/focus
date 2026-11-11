import Foundation
import simd

/// Pure maths for seating the glasses on a face (testable without ARKit).
enum FaceFit {
    /// Eye point inside the glasses model (ARKit face space, metres, model origin = centre of the front face).
    /// ARKit's eye anchor is the eyeball centre.  The CAD head reference (hardware/export/v3/head_ref.stl) puts the
    /// cornea at 26 mm behind the frame front at the pupil column (x = ±32), and the eyeball centre is ~12 mm further back.
    static let modelEyeL = SIMD3<Float>(-0.0325, 0.000, -0.038)   // x=-32.5 mm, up=0, z=-38 mm
    static let modelEyeR = SIMD3<Float>( 0.0325, 0.000, -0.038)

    /// Model-to-face transform: translates eye midpoint, yaws/rolls with the eye axis, scales by IPD ratio (clamped).
    static func transform(eyeL: SIMD3<Float>, eyeR: SIMD3<Float>, headUp: SIMD3<Float>, headForward: SIMD3<Float>) -> (m: simd_float4x4, scale: Float) {
        let ipd = simd_length(eyeR - eyeL)
        let modelIPD = simd_length(modelEyeR - modelEyeL)
        let s = min(1.15, max(0.85, ipd / modelIPD))
        let mid = (eyeL + eyeR) / 2
        // Orthonormal frame from the eye axis (x) and the head's up.
        let x = simd_normalize(eyeR - eyeL)
        let z0 = simd_normalize(headForward)
        let y = simd_normalize(simd_cross(z0, x))
        let z = simd_normalize(simd_cross(x, y))
        _ = headUp
        let modelMid = (modelEyeL + modelEyeR) / 2 * s
        let R = simd_float3x3(x, y, z)
        let t = mid - R * modelMid
        var m = matrix_identity_float4x4
        m.columns.0 = SIMD4(x * s, 0); m.columns.1 = SIMD4(y * s, 0); m.columns.2 = SIMD4(z * s, 0)
        m.columns.3 = SIMD4(t, 1)
        return (m, s)
    }
}

/// Guided face scan: needs a steady face seen at centre, left and right before the glasses latch on.
struct ScanState {
    enum Phase: Equatable { case searching, scanning, fitting, latched }
    private(set) var phase: Phase = .searching
    private(set) var coverage: [Bool] = [false, false, false]   // left, centre, right
    private(set) var steadyFrames = 0
    private(set) var ipdSamples: [Float] = []
    private var lostFrames = 0
    private var fitFrames = 0
    static let fitDuration = 18

    var progress: Float {
        switch phase {
        case .searching: return 0
        case .scanning: return Float(coverage.filter { $0 }.count) / 3 * 0.9
        case .fitting, .latched: return 1
        }
    }
    /// 0...1 progress of the latch-on animation (valid in .fitting, 1 once latched).
    var fitProgress: Float { phase == .latched ? 1 : (phase == .fitting ? Float(fitFrames) / Float(Self.fitDuration) : 0) }
    var medianIPD: Float? { ipdSamples.isEmpty ? nil : ipdSamples.sorted()[ipdSamples.count / 2] }

    /// yaw in radians (face turning), tracked = face currently tracked, ipd metres.
    mutating func update(tracked: Bool, yaw: Float, ipd: Float) {
        switch phase {
        case .searching:
            if tracked { steadyFrames += 1; if steadyFrames >= 8 { phase = .scanning; steadyFrames = 0 } } else { steadyFrames = 0 }
        case .scanning:
            if !tracked { lostFrames += 1; if lostFrames > 30 { self = ScanState() }; return }
            lostFrames = 0
            ipdSamples.append(ipd); if ipdSamples.count > 90 { ipdSamples.removeFirst() }
            let bin = yaw < -0.17 ? 0 : (yaw > 0.17 ? 2 : 1)
            coverage[bin] = true
            if coverage.allSatisfy({ $0 }) { phase = .fitting; fitFrames = 0 }
        case .fitting:
            fitFrames += 1; if fitFrames >= Self.fitDuration { phase = .latched }
        case .latched:
            if !tracked { lostFrames += 1; if lostFrames > 90 { self = ScanState() } } else { lostFrames = 0 }
        }
    }
}
