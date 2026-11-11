import XCTest
import SceneKit
import simd
@testable import Focus

final class TryOnTests: XCTestCase {
    func eyes(ipd: Float, mid: SIMD3<Float> = SIMD3(0, 0.01, 0.03)) -> (SIMD3<Float>, SIMD3<Float>) {
        (mid - SIMD3(ipd / 2, 0, 0), mid + SIMD3(ipd / 2, 0, 0))
    }

    func testFitSeatsModelEyesOnRealEyes() {
        let (l, r) = eyes(ipd: 0.063)
        let f = FaceFit.transform(eyeL: l, eyeR: r, headUp: SIMD3(0, 1, 0), headForward: SIMD3(0, 0, 1))
        let mL = f.m * SIMD4(FaceFit.modelEyeL, 1), mR = f.m * SIMD4(FaceFit.modelEyeR, 1)
                XCTAssertEqual(f.scale, 0.063 / 0.065, accuracy: 1e-4)
        XCTAssertLessThan(abs((mL.x + mR.x) / 2 - (l.x + r.x) / 2), 1e-5)
        XCTAssertLessThan(abs((mL.y + mR.y) / 2 - (l.y + r.y) / 2), 1e-5)
        XCTAssertLessThan(abs((mL.z + mR.z) / 2 - (l.z + r.z) / 2), 1e-5)
    }

    func testScaleIsClamped() {
        let (l, r) = eyes(ipd: 0.12)
        XCTAssertEqual(FaceFit.transform(eyeL: l, eyeR: r, headUp: SIMD3(0, 1, 0), headForward: SIMD3(0, 0, 1)).scale, 1.15, accuracy: 1e-5)
        let (a, b) = eyes(ipd: 0.04)
        XCTAssertEqual(FaceFit.transform(eyeL: a, eyeR: b, headUp: SIMD3(0, 1, 0), headForward: SIMD3(0, 0, 1)).scale, 0.85, accuracy: 1e-5)
    }

    func testFitFollowsHeadRoll() {
        let a: Float = 0.3   // roll about z
        let l = SIMD3<Float>(-0.0315 * cos(a), -0.0315 * sin(a), 0), r = -l
        let f = FaceFit.transform(eyeL: l, eyeR: r, headUp: SIMD3(0, 1, 0), headForward: SIMD3(0, 0, 1))
        let xAxis = SIMD3(f.m.columns.0.x, f.m.columns.0.y, f.m.columns.0.z) / f.scale
        XCTAssertEqual(atan2(xAxis.y, xAxis.x), a, accuracy: 1e-4)
    }

    func testScanNeedsAllThreePoses() {
        var s = ScanState()
        for _ in 0..<10 { s.update(tracked: true, yaw: 0, ipd: 0.063) }
        XCTAssertEqual(s.phase, .scanning)
        for _ in 0..<60 { s.update(tracked: true, yaw: 0, ipd: 0.063) }
        XCTAssertEqual(s.phase, .scanning, "front pose alone must not finish the scan")
        s.update(tracked: true, yaw: -0.3, ipd: 0.063)
        XCTAssertEqual(s.phase, .scanning)
        s.update(tracked: true, yaw: 0.3, ipd: 0.063)
        XCTAssertEqual(s.phase, .fitting)
        for _ in 0..<ScanState.fitDuration { s.update(tracked: true, yaw: 0, ipd: 0.063) }
        XCTAssertEqual(s.phase, .latched)
        XCTAssertEqual(s.progress, 1)
        XCTAssertEqual(s.medianIPD ?? 0, 0.063, accuracy: 1e-5)
    }

    func testScanResetsAfterLosingFace() {
        var s = ScanState()
        for _ in 0..<10 { s.update(tracked: true, yaw: 0, ipd: 0.06) }
        for _ in 0..<40 { s.update(tracked: false, yaw: 0, ipd: 0) }
        XCTAssertEqual(s.phase, .searching)
    }

    func testAllGlassesPartsLoadWithRealisticSize() throws {
        let node = try XCTUnwrap(GlassesNode.bundled(), "STL resources missing from the app bundle")
        XCTAssertEqual(node.childNodes.count, 19)
        let (mn, mx) = node.boundingBox
        let w = Float(mx.x - mn.x) * 1000, h = Float(mx.y - mn.y) * 1000, d = Float(mx.z - mn.z) * 1000
        XCTAssertEqual(w, 146, accuracy: 4)      // 146 mm wide
        XCTAssertEqual(h, 56, accuracy: 3)       // shield spans 56 mm
        XCTAssertEqual(d, 157, accuracy: 4)      // front to temple tips
        XCTAssertGreaterThan(Float(mx.z), 0)     // front face is at z=0 and nothing sticks out in front of it
        XCTAssertLessThan(Float(mx.z), 0.003)    // only the camera lenses poke ~2 mm past the front face
    }
}
