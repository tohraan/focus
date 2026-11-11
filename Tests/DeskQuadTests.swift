import XCTest
import simd
@testable import Focus

final class DeskQuadTests: XCTestCase {
    func testRayHitsPlaneBelowCamera() {
        let p = DeskQuad.rayPlane(origin: SIMD3(0, 0, 0), direction: normalize(SIMD3(0, -1, -1)), planeY: -0.5)
        XCTAssertNotNil(p)
        XCTAssertEqual(p!.y, -0.5, accuracy: 1e-5); XCTAssertEqual(p!.z, -0.5, accuracy: 1e-5)
        XCTAssertNil(DeskQuad.rayPlane(origin: .zero, direction: SIMD3(0, 1, -1), planeY: -0.5))      // looking up
        XCTAssertNil(DeskQuad.rayPlane(origin: .zero, direction: SIMD3(1, 0, 0), planeY: -0.5))       // parallel
    }

    func testOrderingRemovesSelfIntersection() {
        let pts: [SIMD3<Float>] = [SIMD3(0, 0, 0), SIMD3(1, 0, 1), SIMD3(1, 0, 0), SIMD3(0, 0, 1)]   // bow-tie order
        let o = DeskQuad.ordered(pts)
        var area: Float = 0
        for i in 0..<4 { let a = o[i], b = o[(i + 1) % 4]; area += a.x * b.z - b.x * a.z }
        XCTAssertEqual(abs(area) / 2, 1, accuracy: 1e-5)          // full unit square, not a bow-tie (area 0)
    }

    func testHeightFromTapAboveDeskCentre() {
        let corners: [SIMD3<Float>] = [SIMD3(-0.3, -0.7, -0.5), SIMD3(0.3, -0.7, -0.5), SIMD3(0.3, -0.7, -1.1), SIMD3(-0.3, -0.7, -1.1)]
        // a point 0.4 m above the centre (0, -0.3, -0.8), seen from the origin
        let dir = normalize(SIMD3<Float>(0, -0.3, -0.8))
        let h = DeskQuad.height(rayOrigin: .zero, rayDirection: dir, corners: corners, planeY: -0.7)
        XCTAssertEqual(h!, 0.4, accuracy: 0.01)
        XCTAssertEqual(DeskQuad.height(rayOrigin: .zero, rayDirection: normalize(SIMD3(0, 0.8, -0.8)), corners: corners, planeY: -0.7)!, 1.2, accuracy: 0.01)   // clamped
    }

    func testReachIsFarthestCornerPlusMargin() {
        let c: [SIMD3<Float>] = [SIMD3(0, -0.7, -0.5), SIMD3(0, -0.7, -1.0)]
        XCTAssertEqual(DeskQuad.reach(corners: c, camera: .zero), 1.2, accuracy: 1e-4)
        XCTAssertEqual(DeskQuad.reach(corners: [SIMD3(0, 0, -9)], camera: .zero), 2.5)
    }

    func testAssumedDeskIsSmallAndUnderTheCrosshair() {
        let r = DeskRect.assumed(cameraPosition: SIMD3(1, 1.4, 2), forward: normalize(SIMD3(0, -0.5, -1)))
        XCTAssertEqual(r.width, 0.7, accuracy: 1e-6); XCTAssertEqual(r.depth, 0.45, accuracy: 1e-6)
        // ray down at slope 0.5 reaches 0.55 m below the camera after 1.1 m forward... horizontally 0.55/0.5 = 1.1
        XCTAssertEqual(r.center.y, 0.85, accuracy: 1e-5); XCTAssertEqual(r.center.z, 2 - 1.1, accuracy: 0.01); XCTAssertEqual(r.center.x, 1, accuracy: 1e-4)
        XCTAssertEqual(r.corners.count, 4)
        for p in r.corners { XCTAssertEqual(abs(p.x - 1), 0.35, accuracy: 1e-4); XCTAssertEqual(p.y, 0.85, accuracy: 1e-5) }
    }

    func testFitFromTwoOppositeCorners() {
        let base = DeskRect.assumed(cameraPosition: SIMD3(0, 1, 0), forward: SIMD3(0, -0.4, -1))
        let f = base.fitted(a: SIMD3(-0.4, 0.45, -0.5), b: SIMD3(0.5, 0.45, -1.1))
        XCTAssertEqual(f.width, 0.9, accuracy: 1e-5); XCTAssertEqual(f.depth, 0.6, accuracy: 1e-5)
        XCTAssertEqual(f.center.x, 0.05, accuracy: 1e-5); XCTAssertEqual(f.center.z, -0.8, accuracy: 1e-5)
    }
}
