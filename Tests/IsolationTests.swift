import XCTest
import CoreImage
import simd
@testable import Focus

final class IsolationTests: XCTestCase {
    let ctx = CIContext()
    func px(_ img: CIImage, _ x: Int, _ y: Int) -> [UInt8] {
        var b = [UInt8](repeating: 0, count: 4)
        ctx.render(img, toBitmap: &b, rowBytes: 4, bounds: CGRect(x: x, y: y, width: 1, height: 1), format: .RGBA8, colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
        return b
    }
    func solid(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat, _ size: CGSize) -> CIImage {
        CIImage(color: CIColor(red: r, green: g, blue: b)).cropped(to: CGRect(origin: .zero, size: size))
    }

    func testCompositorKeepsCameraInsideMaskAndWorldOutside() {
        let size = CGSize(width: 200, height: 200)
        let cam = solid(1, 0, 0, size), world = solid(0, 0, 1, size)
        let keep = solid(1, 1, 1, CGSize(width: 80, height: 80)).transformed(by: CGAffineTransform(translationX: 60, y: 60))
            .composited(over: solid(0, 0, 0, size))
        let out = IsolationCompositor.compose(camera: cam, keep: keep, world: world)
        XCTAssertGreaterThan(px(out, 100, 100)[0], 200)      // inside: camera (red)
        XCTAssertLessThan(px(out, 100, 100)[2], 80)
        XCTAssertGreaterThan(px(out, 5, 5)[2], 200)          // outside: world (blue)
        XCTAssertLessThan(px(out, 5, 5)[0], 80)
    }

    func testWorldCoversFrameForEveryTheme() {
        for t in IsolationWorld.Theme.allCases {
            let w = IsolationWorld(size: CGSize(width: 180, height: 240), theme: t)
            let f = w.frame(time: 1.0)
            XCTAssertEqual(f.extent.width, 180, accuracy: 1); XCTAssertEqual(f.extent.height, 240, accuracy: 1)
            XCTAssertGreaterThan(Int(px(f.clampedToExtent(), 90, 200)[2]) + Int(px(f.clampedToExtent(), 90, 200)[1]), 60, "scene must not be black (\(t.name))")
        }
    }

    func testWorldIsStaticAndNotFlashy() {
        let w = IsolationWorld(size: CGSize(width: 180, height: 240))
        let a = ctx.createCGImage(w.frame(time: 0), from: CGRect(x: 0, y: 0, width: 180, height: 240))!
        let b = ctx.createCGImage(w.frame(time: 5), from: CGRect(x: 0, y: 0, width: 180, height: 240))!
        XCTAssertEqual(a.dataProvider!.data as Data?, b.dataProvider!.data as Data?)
    }

    func testTableReachUsesFarEdgeWithMargin() {
        var dpb: CVPixelBuffer!
        CVPixelBufferCreate(nil, 64, 48, kCVPixelFormatType_DepthFloat32, nil, &dpb)
        CVPixelBufferLockBaseAddress(dpb, [])
        let p = CVPixelBufferGetBaseAddress(dpb)!.assumingMemoryBound(to: Float.self)
        for i in 0..<(64 * 48) { p[i] = 0.4 + Float(i % 64) / 64 * 0.5 }     // 0.4 ... 0.9 m
        CVPixelBufferUnlockBaseAddress(dpb, [])
        let e = TableReach.farEdge(depth: dpb)
        XCTAssertNotNil(e)
        XCTAssertEqual(e!, 0.83, accuracy: 0.08)
        XCTAssertEqual(TableReach.reach(fromFarEdges: [e!, e!, e!]), e! + 0.2, accuracy: 1e-4)
        XCTAssertEqual(TableReach.reach(fromFarEdges: [5]), 2.0)   // clamped
        XCTAssertEqual(TableReach.reach(fromFarEdges: []), 1.0)
    }

    /// Synthetic scene: blank camera frame (no people/hands), bottom 30% of depth is 0.5 m, the rest 3 m.
    func testNearDepthIsKeptAndFarIsReplaced() throws {
        var pb: CVPixelBuffer!
        CVPixelBufferCreate(nil, 320, 240, kCVPixelFormatType_32BGRA, nil, &pb)
        ctx.render(solid(0.4, 0.4, 0.4, CGSize(width: 320, height: 240)), to: pb)
        var dpb: CVPixelBuffer!
        CVPixelBufferCreate(nil, 80, 60, kCVPixelFormatType_DepthFloat32, nil, &dpb)
        CVPixelBufferLockBaseAddress(dpb, [])
        let p = CVPixelBufferGetBaseAddress(dpb)!.assumingMemoryBound(to: Float.self)
        for y in 0..<60 { for x in 0..<80 { p[y * 80 + x] = y >= 42 ? 0.5 : 3.0 } }   // memory rows: top first
        CVPixelBufferUnlockBaseAddress(dpb, [])
        let km = KeepMask(ctx: ctx); km.config.canvas = CGSize(width: 160, height: 120); km.config.reachMetres = 1.0
        var keep = km.keep(for: pb, orientation: .up, depth: dpb)
        for _ in 0..<4 { keep = km.keep(for: pb, orientation: .up, depth: dpb) }
        XCTAssertGreaterThan(px(keep, 80, 10)[0], 200, "near band (bottom of image) kept")
        XCTAssertLessThan(px(keep, 80, 100)[0], 40, "far region replaced")
    }

    func testMissingDepthKeepsNothingOnBlankFrame() {
        var pb: CVPixelBuffer!
        CVPixelBufferCreate(nil, 320, 240, kCVPixelFormatType_32BGRA, nil, &pb)
        ctx.render(solid(0.4, 0.4, 0.4, CGSize(width: 320, height: 240)), to: pb)
        let km = KeepMask(ctx: ctx); km.config.canvas = CGSize(width: 160, height: 120); km.config.reachMetres = 1.0
        let keep = km.keep(for: pb, orientation: .up, depth: nil)
        XCTAssertLessThan(px(keep, 80, 60)[0], 40)
    }
}

final class TableVolumeTests: XCTestCase {
    // identity camera at the origin looking down -Z; a 1 m x 1 m table 0.5 m below, 0.5...1.5 m ahead
    let K = simd_float3x3(columns: (SIMD3(1000, 0, 0), SIMD3(0, 1000, 0), SIMD3(960, 720, 1)))
    let table = TableVolume(base: [SIMD3(-0.5, -0.5, -0.5), SIMD3(0.5, -0.5, -0.5), SIMD3(0.5, -0.5, -1.5), SIMD3(-0.5, -0.5, -1.5)])

    func px(_ img: CGImage, _ x: Int, _ y: Int) -> UInt8 {
        var b = [UInt8](repeating: 0, count: 4)
        CIContext().render(CIImage(cgImage: img), toBitmap: &b, rowBytes: 4, bounds: CGRect(x: x, y: y, width: 1, height: 1), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        return b[0]
    }

    func testTableAndLaptopRegionKeptSkyAndSidesReplaced() throws {
        let m = try XCTUnwrap(table.raster(camera: matrix_identity_float4x4, intrinsics: K, imageSize: CGSize(width: 1920, height: 1440), canvas: CGSize(width: 320, height: 240)))
        // table centre: ~(0, -0.5, -1) -> v = 720 + 500 = 1220 px -> canvas row from top 203 (CG y flipped: 240-203=37)
        XCTAssertGreaterThan(px(m, 160, 40), 200, "table surface is kept")
        // laptop on the table at 0.3 m height, 1 m away: v = 720 - 1000*(-0.2)/1... y=-0.2 -> v=920 -> inside hull
        XCTAssertGreaterThan(px(m, 160, 240 - 153), 200, "space above the table up to laptop height is kept")
        XCTAssertLessThan(px(m, 160, 240 - 30), 40, "far wall/sky above the box is replaced")
        XCTAssertLessThan(px(m, 5, 120), 40, "far left is replaced")
    }

    func testTableMostlyBehindCameraStillProducesMask() {
        var behind = table
        behind.base = [SIMD3(-0.5, -0.5, 0.5), SIMD3(0.5, -0.5, 0.5), SIMD3(0.5, -0.5, -1.0), SIMD3(-0.5, -0.5, -1.0)]
        let hull = behind.screenHull(camera: matrix_identity_float4x4, fx: 1000, fy: 1000, cx: 960, cy: 720)
        XCTAssertGreaterThanOrEqual(hull.count, 3)
        XCTAssertTrue(hull.allSatisfy { $0.x.isFinite && $0.y.isFinite })
    }

    func testCameraMotionMovesTheMask() {
        var cam = matrix_identity_float4x4; cam.columns.3 = SIMD4(0.4, 0, 0, 1)      // camera moved right
        let a = table.screenHull(camera: matrix_identity_float4x4, fx: 1000, fy: 1000, cx: 960, cy: 720)
        let b = table.screenHull(camera: cam, fx: 1000, fy: 1000, cx: 960, cy: 720)
        let ma: Double = a.map { Double($0.x) }.reduce(0, +) / Double(a.count)
        let mb: Double = b.map { Double($0.x) }.reduce(0, +) / Double(b.count)
        XCTAssertLessThan(mb, ma)   // scene shifts left
    }
}
