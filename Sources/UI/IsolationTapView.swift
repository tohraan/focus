import SwiftUI
import ARKit
import MetalKit

/// Isolation Mode (rear camera only), tap flow: tap the 4 corners of your desk on the live view, then (optionally) tap the
/// top of what stands on it to set the height. Everything inside that box stays real (hands, desk, laptop, notebook, the
/// space between); the rest becomes the moon world. The mask is a polygon built from the taps, so it is ready the moment
/// the 4th corner lands. Corners are world-anchored (they stay put when the phone moves) and can be dragged.
final class IsolationTapEngine: NSObject, ObservableObject, ARSessionDelegate, MTKViewDelegate {
    enum Step { case tap, active }

    @Published var step = Step.tap
    @Published var prompt = "Tap the 1st corner of your desk"
    @Published var handles = [CGPoint]()           // corner positions in view coordinates, tap order
    @Published var heightPending = false
    @Published var fps = 0.0
    @Published var message: String?
    @Published var tableHeight: Double = 0.25 { didSet { lock2.lock(); height = Float(tableHeight); lock2.unlock(); rebuild() } }
    @Published var theme = IsolationWorld.Theme.moon { didSet { world.setTheme(theme) } }
    /// Desk level nudge (m): moves the whole desk plane and corners up or down when the assumed level is off.
    @Published var deskNudge: Double = 0 { didSet {
        let d = Float(deskNudge - oldValue)
        lock2.lock(); if let y = planeY { planeY = y + d }; corners = corners.map { SIMD3($0.x, $0.y + d, $0.z) }; lock2.unlock()
        rebuild() } }
    @Published var reach: Double = 1.0 { didSet { mask.config.reachMetres = Float(reach) } }
    @Published var hasDepth = true

    private let session = ARSession()
    private let queue = DispatchQueue(label: "isolation.frames", qos: .userInitiated)
    private let device = MTLCreateSystemDefaultDevice()!
    private lazy var cmdQueue = device.makeCommandQueue()!
    private lazy var ctx = CIContext(mtlDevice: device, options: [.cacheIntermediates: false])
    private lazy var mask = KeepMask(ctx: ctx)
    private let world = IsolationWorld(size: CGSize(width: 720, height: 960))
    private var busy = false
    private var latest: CIImage?
    private let lock = NSLock()
    private var t0 = CACurrentMediaTime(), frames = 0

    // shared between the main thread (taps) and the frame queue; guarded by lock2
    private let lock2 = NSLock()
    private var lastFrame: ARFrame?
    private var corners = [SIMD3<Float>]()
    private var planeY: Float?
    private var table: TableVolume?
    private var height: Float = 0.25
    private var viewSize = CGSize.zero
    private var stepInternal = Step.tap
    private var hadHandles = false      // frame queue only

    var supported: Bool { ARWorldTrackingConfiguration.isSupported }

    func setViewSize(_ s: CGSize) { lock2.lock(); viewSize = s; lock2.unlock() }

    func attach(_ v: MTKView) {
        v.device = device; v.delegate = self; v.framebufferOnly = false
        v.colorPixelFormat = .bgra8Unorm; v.preferredFramesPerSecond = 30
    }

    func start() {
        session.delegate = self; session.delegateQueue = queue
        UIApplication.shared.isIdleTimerDisabled = true
        run()
    }
    func stop() { session.pause(); UIApplication.shared.isIdleTimerDisabled = false }

    private func run() {
        message = nil
        let c = ARWorldTrackingConfiguration()
        c.planeDetection = [.horizontal]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { c.frameSemantics.insert(.smoothedSceneDepth); hasDepth = true }
        else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { c.frameSemantics.insert(.sceneDepth); hasDepth = true }
        else { hasDepth = false }       // no LiDAR: the tapped box does the work
        session.run(c, options: [.resetTracking, .removeExistingAnchors])
    }

    /// Back to the first corner.
    func restart() {
        deskNudge = 0
        lock2.lock(); corners = []; planeY = nil; table = nil; stepInternal = .tap; lock2.unlock()
        mask.resetHistory()
        step = .tap; heightPending = false; handles = []; prompt = Self.cornerPrompt(0); message = nil
        run()
    }

    private static func cornerPrompt(_ n: Int) -> String {
        ["Tap the 1st corner of your desk", "Tap the 2nd corner", "Tap the 3rd corner", "Tap the last corner"][min(n, 3)]
    }

    // MARK: taps (main thread)

    /// Ray through a point in view coordinates, in world space.
    private func ray(_ p: CGPoint, frame f: ARFrame, viewSize vs: CGSize) -> ARRaycastQuery? {
        guard vs.width > 0, vs.height > 0 else { return nil }
        let n = CGPoint(x: p.x / vs.width, y: p.y / vs.height)
        let img = n.applying(f.displayTransform(for: .portrait, viewportSize: vs).inverted())
        return f.raycastQuery(from: img, allowing: .estimatedPlane, alignment: .horizontal)
    }

    func tap(at p: CGPoint) {
        lock2.lock(); let f = lastFrame, vs = viewSize, cs = corners, py = planeY, pending = heightPending; lock2.unlock()
        guard let f, let q = ray(p, frame: f, viewSize: vs) else { return }
        if cs.count >= 4 {
            guard pending, let py, let h = DeskQuad.height(rayOrigin: q.origin, rayDirection: q.direction, corners: cs, planeY: py) else { return }
            tableHeight = Double(h)             // rebuilds via didSet
            heightPending = false; prompt = ""
            return
        }
        var point: SIMD3<Float>?
        var y = py
        if let py {
            point = DeskQuad.rayPlane(origin: q.origin, direction: q.direction, planeY: py)
        } else if let hit = session.raycast(q).first {
            let t = hit.worldTransform.columns.3
            point = SIMD3(t.x, t.y, t.z); y = t.y
        } else {
            // ARKit sees no surface (black mat, glare): assume a horizontal desk 0.55 m below the camera
            y = q.origin.y - 0.55
            point = DeskQuad.rayPlane(origin: q.origin, direction: q.direction, planeY: y!)
        }
        guard let point, let y else { message = "Tilt the phone down toward the desk and tap again."; return }
        message = nil
        lock2.lock(); planeY = y; corners.append(point); let n = corners.count; lock2.unlock()
        if n == 4 { finishCorners() } else { prompt = Self.cornerPrompt(n) }
    }

    func moveCorner(_ i: Int, to p: CGPoint) {
        lock2.lock(); let f = lastFrame, vs = viewSize, py = planeY, n = corners.count; lock2.unlock()
        guard i < n, let f, let py, let q = ray(p, frame: f, viewSize: vs),
              let pt = DeskQuad.rayPlane(origin: q.origin, direction: q.direction, planeY: py) else { return }
        lock2.lock(); corners[i] = pt; lock2.unlock()
        rebuild()
    }

    func skipHeight() { heightPending = false; prompt = "" }

    private func finishCorners() {
        lock2.lock()
        let cam = lastFrame.map { SIMD3($0.camera.transform.columns.3.x, $0.camera.transform.columns.3.y, $0.camera.transform.columns.3.z) } ?? .zero
        let r = DeskQuad.reach(corners: corners, camera: cam)
        lock2.unlock()
        rebuild()
        mask.config.reachMetres = r; reach = Double(r)
        lock2.lock(); stepInternal = .active; lock2.unlock()
        step = .active; heightPending = true
        prompt = "Tap the top of your laptop or notebooks to set the height"
    }

    private func rebuild() {
        lock2.lock()
        if corners.count >= 4 { var t = TableVolume(base: DeskQuad.ordered(corners)); t.height = height; table = t }
        lock2.unlock()
        mask.resetHistory()
    }

    // MARK: ARSessionDelegate
    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        lock2.lock(); lastFrame = frame; let cs = corners, tbl = table, st = stepInternal, vs = viewSize; lock2.unlock()
        if (!cs.isEmpty || hadHandles) && vs.width > 0 {
            hadHandles = !cs.isEmpty
            let inv = frame.camera.transform.inverse
            let pts = cs.map { c -> CGPoint in
                let v = inv * SIMD4(c, 1)
                guard v.z < -0.05 else { return CGPoint(x: -1000, y: -1000) }     // behind the camera
                return frame.camera.projectPoint(c, orientation: .portrait, viewportSize: vs)
            }
            DispatchQueue.main.async { self.handles = pts }
        }

        lock.lock(); if busy { lock.unlock(); return }; busy = true; lock.unlock()
        defer { lock.lock(); busy = false; lock.unlock() }

        let orient: CGImagePropertyOrientation = .right
        let W: CGFloat = 720, H: CGFloat = 960
        var camImg = CIImage(cvPixelBuffer: frame.capturedImage).oriented(orient)
        camImg = camImg.transformed(by: CGAffineTransform(scaleX: W / camImg.extent.width, y: H / camImg.extent.height))

        let shown: CIImage
        if st == .active, let t = tbl {
            var depthPB: CVPixelBuffer?
            if hasDepth, let d = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap { depthPB = IsolationTapEngine.orient(d, ctx: ctx, orientation: orient) }
            let res = frame.camera.imageResolution
            let volume = t.raster(camera: frame.camera.transform, intrinsics: frame.camera.intrinsics, imageSize: res, canvas: CGSize(width: 320, height: 240))
            let keep = mask.keep(for: frame.capturedImage, orientation: orient, depth: depthPB, volume: volume)
            let k = keep.transformed(by: CGAffineTransform(scaleX: W / keep.extent.width, y: H / keep.extent.height))
            shown = IsolationCompositor.compose(camera: camImg, keep: k, world: world.frame())
        } else { shown = camImg }
        lock.lock(); latest = shown; lock.unlock()

        frames += 1
        let now = CACurrentMediaTime()
        if now - t0 > 1 { let f = Double(frames) / (now - t0); frames = 0; t0 = now; DispatchQueue.main.async { self.fps = f } }
    }

    /// Orient a depth map like the camera image (float preserved).
    private static func orient(_ d: CVPixelBuffer, ctx: CIContext, orientation: CGImagePropertyOrientation) -> CVPixelBuffer? {
        let img = CIImage(cvPixelBuffer: d, options: [.colorSpace: NSNull()]).oriented(orientation)
        var out: CVPixelBuffer?
        CVPixelBufferCreate(nil, Int(img.extent.width), Int(img.extent.height), kCVPixelFormatType_DepthFloat32, [kCVPixelBufferMetalCompatibilityKey: true] as CFDictionary, &out)
        guard let o = out else { return nil }
        ctx.render(img, to: o, bounds: img.extent, colorSpace: nil)
        return o
    }

    func session(_ session: ARSession, didFailWithError error: Error) { DispatchQueue.main.async { self.message = error.localizedDescription } }

    // MARK: MTKViewDelegate
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}
    func draw(in view: MTKView) {
        lock.lock(); let img = latest; lock.unlock()
        guard let img, let dr = view.currentDrawable, let cb = cmdQueue.makeCommandBuffer() else { return }
        let ds = view.drawableSize
        let s = max(ds.width / img.extent.width, ds.height / img.extent.height)
        let tx = (ds.width - img.extent.width * s) / 2, ty = (ds.height - img.extent.height * s) / 2
        let fitted = img.transformed(by: CGAffineTransform(scaleX: s, y: s).translatedBy(x: tx / s, y: ty / s))
        ctx.render(fitted, to: dr.texture, commandBuffer: cb, bounds: CGRect(origin: .zero, size: ds), colorSpace: CGColorSpaceCreateDeviceRGB())
        cb.present(dr); cb.commit()
    }
}

struct IsolationTapMetal: UIViewRepresentable {
    let engine: IsolationTapEngine
    func makeUIView(context: Context) -> MTKView { let v = MTKView(); engine.attach(v); return v }
    func updateUIView(_ v: MTKView, context: Context) {}
}

/// Corner handles and the outline of the quad; handles are draggable.
struct QuadOverlay: View {
    @ObservedObject var e: IsolationTapEngine
    var body: some View {
        let pts = e.handles
        ZStack {
            if pts.count >= 2 {
                Path { p in
                    let c = CGPoint(x: pts.map(\.x).reduce(0, +) / CGFloat(pts.count), y: pts.map(\.y).reduce(0, +) / CGFloat(pts.count))
                    let o = pts.sorted { atan2($0.y - c.y, $0.x - c.x) < atan2($1.y - c.y, $1.x - c.x) }
                    p.move(to: o[0]); o.dropFirst().forEach { p.addLine(to: $0) }
                    if pts.count >= 4 { p.closeSubpath() }
                }.stroke(Color.white.opacity(0.85), style: StrokeStyle(lineWidth: 2, dash: pts.count >= 4 ? [] : [6, 4]))
            }
            ForEach(pts.indices, id: \.self) { i in
                Circle().fill(Color.orange).frame(width: 26, height: 26)
                    .overlay(Circle().stroke(.white, lineWidth: 3))
                    .overlay(Text("\(i + 1)").font(.caption2.bold()).foregroundStyle(.white))
                    .padding(14).contentShape(Circle())                                  // bigger touch target
                    .position(pts[i])
                    .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("iso")).onChanged { e.moveCorner(i, to: $0.location) })
            }
        }
    }
}

struct IsolationTapScreen: View {
    var back: () -> Void = {}
    @StateObject private var e = IsolationTapEngine()
    @State private var showControls = false
    var body: some View {
        ZStack {
            if e.supported { IsolationTapMetal(engine: e).ignoresSafeArea() } else { Color.black.ignoresSafeArea() }
            GeometryReader { g in
                ZStack {
                    Color.clear.contentShape(Rectangle())
                        .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("iso")).onEnded { v in
                            if hypot(v.translation.width, v.translation.height) < 12 { e.tap(at: v.location) }
                        })
                    QuadOverlay(e: e)
                }
                .coordinateSpace(name: "iso")
                .onAppear { e.setViewSize(g.size) }
                .onChange(of: g.size) { _, s in e.setViewSize(s) }
            }.ignoresSafeArea()

            VStack(spacing: 8) {
                if !e.prompt.isEmpty {
                    HStack(spacing: 10) {
                        Image(systemName: e.step == .tap ? "hand.tap" : "arrow.up.and.down").font(.title3)
                        Text(e.prompt).font(.headline).multilineTextAlignment(.leading)
                        Spacer(minLength: 0)
                        if e.heightPending { Button("Skip") { e.skipHeight() }.buttonStyle(.bordered).controlSize(.small) }
                    }
                    .padding(12).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14)).padding(.trailing, 56)
                }
                if let m = e.message { Text(m).font(.footnote).padding(8).background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 10)).padding(.trailing, 56) }
                Spacer()
                if showControls {
                    VStack(spacing: 10) {
                        Picker("World", selection: $e.theme) { ForEach(IsolationWorld.Theme.allCases, id: \.self) { Text($0.name).tag($0) } }.pickerStyle(.segmented)
                        HStack { Text("Height").font(.caption); Slider(value: $e.tableHeight, in: 0.15...0.9); Text(String(format: "%.2f m", e.tableHeight)).font(.caption.monospacedDigit()) }
                        HStack { Text("Desk level").font(.caption); Slider(value: $e.deskNudge, in: -0.3...0.3); Text(String(format: "%+.2f m", e.deskNudge)).font(.caption.monospacedDigit()) }
                        if e.hasDepth { HStack { Text("Reach").font(.caption); Slider(value: $e.reach, in: 0.3...2.5); Text(String(format: "%.1f m", e.reach)).font(.caption.monospacedDigit()) } }
                    }
                    .padding(12).background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 16)).padding(.trailing, 56)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                HStack {
                    Button { e.restart() } label: { Label("Reset", systemImage: "arrow.counterclockwise").font(.footnote.weight(.semibold)) }
                        .buttonStyle(.bordered).controlSize(.small).tint(.white)
                    Button { back() } label: { Label("Auto scan", systemImage: "viewfinder").font(.footnote.weight(.semibold)) }
                        .buttonStyle(.bordered).controlSize(.small).tint(.white)
                    Spacer()
                    Button { withAnimation { showControls.toggle() } } label: {
                        Image(systemName: showControls ? "chevron.down" : "slider.horizontal.3").font(.system(size: 15, weight: .semibold))
                            .frame(width: 34, height: 34).background(.black.opacity(0.5), in: Circle())
                    }.padding(.trailing, 56)
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8).foregroundStyle(.white)
        }
        .onAppear { e.start() }
        .onDisappear { e.stop() }
    }
}
