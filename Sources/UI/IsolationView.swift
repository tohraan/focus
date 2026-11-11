import SwiftUI
import ARKit
import MetalKit

/// Isolation Mode (rear camera only), guided: 1) bring your hands into view, 2) look over your table/laptop/notebook,
/// 3) everything between you and the far edge of the table stays (hands, desk, devices, the space between); the rest becomes a calm natural world.
final class IsolationEngine: NSObject, ObservableObject, ARSessionDelegate, MTKViewDelegate {
    enum Step { case hands, table, active }

    /// What the guide overlay shows during the scan.
    struct ScanHint: Equatable {
        var icon = "hand.raised"
        var arrow = 0                       // -1 move left, +1 move right, 0 none
        var bins = [Bool](repeating: false, count: 3)   // sweep coverage, left to right
        var current = 1                     // bin the phone points at now
        var elapsed = 0.0
        var eta: Double?                    // seconds left, nil until estimable
    }

    @Published var step = Step.hands
    @Published var progress = 0.0
    @Published var prompt = "Bring your hands into view"
    @Published var fps = 0.0
    @Published var message: String?
    @Published var tableHeight: Double = 0.25 { didSet { table?.height = Float(tableHeight) } }
    // the assumed desk, trimmed live (only when the desk is assumed, not when ARKit found a real plane)
    @Published var deskAdjustable = false
    @Published var deskWidth: Double = 0.7 { didSet { applyDesk() } }
    @Published var deskDepth: Double = 0.45 { didSet { applyDesk() } }
    @Published var deskLevel: Double = 0 { didSet { applyDesk() } }
    @Published var outline = [CGPoint]()           // quad outline in view coordinates while adjusting
    @Published var fitting = false
    @Published var fitPrompt = ""
    @Published var scan = ScanHint()
    @Published var theme = IsolationWorld.Theme.moon { didSet { world.setTheme(theme) } }
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
    private var handFrames = 0, tableFrames = 0
    private var edges = [Float]()
    private var planeArea: Float = 0
    private var candidate: (area: Float, outline: [SIMD3<Float>])?
    private var table: TableVolume?
    private var camY: Float = 0
    private let planeStableSecs = 0.4   // wall-clock: table plane in view this long
    private let handsNeed = 4           // hand detections (checked every 2nd frame)
    private let handsGiveUpSecs = 1.5   // the hands step never holds the scan longer than this
    private let fallbackAfterSecs = 4.0 // no plane by then: offer the fallbacks (buttons)
    private let autoAssumeSecs = 3.0    // seconds in the table step with no plane before the desk is assumed automatically
    private let hardCapSecs = 10.0      // the scan never runs longer than this
    private var tableStart: Double?
    private let lock2 = NSLock()
    private var lastFrame: ARFrame?
    private var viewSize = CGSize.zero
    private var desk: DeskRect?
    private var deskBaseY: Float = 0
    private var suppressApply = false
    private var fitPoints = [SIMD3<Float>]()
    var outlineOn = false                          // set by the view while controls are open or fitting
    private var planeSince: Double?
    private var forcedDesk = false
    private var lastCam = matrix_identity_float4x4
    private var frameIdx = 0, procMs = 0.0, procN = 0
    private var marks = Set<String>()
    @Published var needFallback = false
    private var scanStart = CACurrentMediaTime()
    private var baseYaw: Float?
    private var cover = [Bool](repeating: false, count: 3)

    var supported: Bool { ARWorldTrackingConfiguration.isSupported }
    func setViewSize(_ s: CGSize) { lock2.lock(); viewSize = s; lock2.unlock() }

    func attach(_ v: MTKView) {
        v.device = device; v.delegate = self; v.framebufferOnly = false
        v.colorPixelFormat = .bgra8Unorm; v.preferredFramesPerSecond = 30
    }

    func start() {
        session.delegate = self; session.delegateQueue = queue
        UIApplication.shared.isIdleTimerDisabled = true
        scanStart = CACurrentMediaTime(); marks = []
        run()
    }
    func stop() { session.pause(); UIApplication.shared.isIdleTimerDisabled = false }

    private func run() {
        message = nil
        let c = ARWorldTrackingConfiguration()
        c.planeDetection = [.horizontal]
        if ARWorldTrackingConfiguration.supportsFrameSemantics(.smoothedSceneDepth) { c.frameSemantics.insert(.smoothedSceneDepth); hasDepth = true }
        else if ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth) { c.frameSemantics.insert(.sceneDepth); hasDepth = true }
        else { hasDepth = false }       // no LiDAR: the scanned table box does the work
        session.run(c, options: [.resetTracking, .removeExistingAnchors])
    }

    /// Back to step 1.
    func restart() {
        handFrames = 0; tableFrames = 0; edges = []; planeArea = 0; candidate = nil; table = nil; mask.resetHistory()
        baseYaw = nil; cover = [Bool](repeating: false, count: 3); scanStart = CACurrentMediaTime(); marks = []
        planeSince = nil; forcedDesk = false; frameIdx = 0; tableStart = nil
        lock2.lock(); desk = nil; fitPoints = []; lock2.unlock()
        step = .hands
        DispatchQueue.main.async { self.step = .hands; self.progress = 0; self.prompt = "Hold your hands in view"; self.scan = ScanHint(); self.needFallback = false; self.deskAdjustable = false; self.fitting = false; self.outline = [] }
        run()
    }

    /// Logs a scan milestone once, with seconds since the scan started (read it with devicectl --console).
    private func mark(_ name: String) {
        guard marks.insert(name).inserted else { return }
        NSLog("[iso] %.2fs %@", CACurrentMediaTime() - scanStart, name)
    }

    /// Fallback for desks where plane detection never fires (black mats, glare): a desk plane from gravity,
    /// 0.55 m below the camera and 0.25-0.95 m ahead of it.
    private func assumeDesk(_ reason: String) {
        let t = lastCam
        let pos = SIMD3<Float>(t.columns.3.x, t.columns.3.y, t.columns.3.z)
        let fwd = -SIMD3<Float>(t.columns.2.x, t.columns.2.y, t.columns.2.z)
        let d = DeskRect.assumed(cameraPosition: pos, forward: fwd)
        lock2.lock(); desk = d; deskBaseY = d.center.y; lock2.unlock()
        candidate = (1.0, d.corners); planeArea = 1.0; forcedDesk = true
        DispatchQueue.main.async {
            self.suppressApply = true; self.deskWidth = 0.7; self.deskDepth = 0.45; self.deskLevel = 0; self.suppressApply = false
            self.needFallback = false; self.deskAdjustable = true
        }
        mark("assumed_desk (\(reason))")
    }

    /// Slider changes: rebuild the box from the trimmed rectangle.
    private func applyDesk() {
        guard !suppressApply else { return }
        let w = Float(deskWidth), dp = Float(deskDepth), lv = Float(deskLevel)
        queue.async {
            self.lock2.lock()
            guard var d = self.desk else { self.lock2.unlock(); return }
            d.width = w; d.depth = dp; d.center.y = self.deskBaseY + lv; self.desk = d
            self.lock2.unlock()
            var t = TableVolume(base: d.corners); t.height = Float(self.tableHeight); self.table = t
            self.mask.resetHistory()
        }
    }

    /// "Fit to desk": two taps on opposite corners, on the assumed plane.
    func startFit() { fitPoints = []; fitting = true; fitPrompt = "Tap two opposite corners of your desk (1 of 2)" }
    func cancelFit() { fitting = false; fitPoints = [] }
    func fitTap(at p: CGPoint) {
        lock2.lock(); let f = lastFrame, vs = viewSize, d = desk; lock2.unlock()
        guard fitting, let f, let d, vs.width > 0, let q = f.rayThrough(viewPoint: p, viewSize: vs),
              let hit = DeskQuad.rayPlane(origin: q.origin, direction: q.direction, planeY: d.center.y) else { return }
        fitPoints.append(hit)
        if fitPoints.count < 2 { fitPrompt = "Tap the opposite corner (2 of 2)"; return }
        let nd = d.fitted(a: fitPoints[0], b: fitPoints[1])
        lock2.lock(); desk = nd; lock2.unlock()
        fitting = false; fitPoints = []
        suppressApply = true; deskWidth = Double(nd.width); deskDepth = Double(nd.depth); suppressApply = false
        applyDesk()
    }

    /// Button version, from the main thread.
    func useAssumedDesk() { queue.async { self.assumeDesk("button") } }

    // MARK: ARSessionDelegate
    func session(_ session: ARSession, didAdd anchors: [ARAnchor]) { track(anchors) }
    func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) { track(anchors) }
    private func track(_ anchors: [ARAnchor]) {
        guard step != .active else { return }
        for a in anchors {
            guard let p = a as? ARPlaneAnchor, p.alignment == .horizontal else { continue }
            // a table is 0.15-1.0 m below the camera (the floor is farther, ceilings are above)
            let below = camY - p.transform.columns.3.y
            guard below > 0.15 && below < 1.0 else { continue }
            let area = p.planeExtent.width * p.planeExtent.height
            guard area > (candidate?.area ?? 0) * 0.85 else { continue }
            let outline = p.geometry.boundaryVertices.map { v -> SIMD3<Float> in let w = p.transform * SIMD4(v.x, 0, v.z, 1); return SIMD3(w.x, w.y, w.z) }
            if outline.count >= 3 { candidate = (area, outline); planeArea = area }
        }
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        lock2.lock(); lastFrame = frame; let vs = viewSize, dk = desk; lock2.unlock()
        if outlineOn, let dk, vs.width > 0 {
            let inv = frame.camera.transform.inverse
            let pts = dk.corners.map { c -> CGPoint in
                guard (inv * SIMD4(c, 1)).z < -0.05 else { return CGPoint(x: -1000, y: -1000) }
                return frame.camera.projectPoint(c, orientation: .portrait, viewportSize: vs)
            }
            DispatchQueue.main.async { self.outline = pts }
        } else if !outline.isEmpty && !outlineOn { DispatchQueue.main.async { self.outline = [] } }
        lock.lock(); if busy { lock.unlock(); return }; busy = true; lock.unlock()
        defer { lock.lock(); busy = false; lock.unlock() }

        let tFrame = CACurrentMediaTime()
        let orient: CGImagePropertyOrientation = .right
        camY = frame.camera.transform.columns.3.y; lastCam = frame.camera.transform
        let W: CGFloat = 720, H: CGFloat = 960
        var camImg = CIImage(cvPixelBuffer: frame.capturedImage).oriented(orient)
        camImg = camImg.transformed(by: CGAffineTransform(scaleX: W / camImg.extent.width, y: H / camImg.extent.height))
        frameIdx += 1

        // camera heading: yaw > 0 = turned right; pitch < 0 = looking down
        let fwd = -frame.camera.transform.columns.2
        let yaw = atan2(fwd.x, -fwd.z)
        let pitch = asin(max(-1, min(1, fwd.y)))
        let planeOK = (candidate?.area ?? 0) > 0.2
        let el = tFrame - scanStart
        if step != .active {                                  // the table is learned while the hands step runs too
            if planeOK { if planeSince == nil { planeSince = tFrame; mark("plane_first") } } else { planeSince = nil }
            if !planeOK && el > fallbackAfterSecs && !needFallback { DispatchQueue.main.async { self.needFallback = true }; mark("no_plane_fallback_offered") }
        }
        let planeStable = planeSince.map { tFrame - $0 >= planeStableSecs } ?? false
        if planeStable { mark("plane_stable") }

        var keep: CIImage?
        switch step {
        case .hands:
            // no mask is built during the scan: only hand presence, every 2nd frame
            if frameIdx % 2 == 0 { mask.detectHandsOnly(for: frame.capturedImage, orientation: orient) }
            handFrames = mask.lastHandCount > 0 ? handFrames + 1 : max(0, handFrames - 1)
            publishScan(prompt: "Hold your hands in view over the desk", icon: "hand.raised", arrow: 0, current: 1, planeStable: planeStable)
            if handFrames >= handsNeed || el >= handsGiveUpSecs { mark(handFrames >= handsNeed ? "hands_done" : "hands_skipped"); baseYaw = nil; advance(.table, "Look at your desk") }
        case .table:
            if tableStart == nil { tableStart = tFrame }
            var depthPB: CVPixelBuffer?
            if hasDepth, let d = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap { depthPB = IsolationEngine.orient(d, ctx: ctx, orientation: orient) }
            if let d = depthPB, let e = TableReach.farEdge(depth: d) { edges.append(e) }
            if baseYaw == nil { baseYaw = yaw }
            var d = yaw - baseYaw!
            while d > .pi { d -= 2 * .pi }; while d < -.pi { d += 2 * .pi }
            let cur = Int((d * 180 / .pi / 8).rounded()) + 1
            if (0...2).contains(cur) { cover[cur] = true }
            let missing = cover.indices.filter { !cover[$0] }
            if missing.isEmpty { mark("sweep_done") }
            var arrow = 0
            if let m = missing.min(by: { abs($0 - cur) < abs($1 - cur) }) { arrow = m > cur ? 1 : -1 }
            let text: String, icon: String
            if !planeOK {
                if pitch > -0.2 { text = "Tilt the phone down toward your desk"; icon = "arrow.down" }
                else { text = "Move the phone slowly to find the desk"; icon = "viewfinder" }
            } else if arrow != 0 && !forcedDesk {
                text = arrow > 0 ? "Sweep slowly to the right" : "Sweep slowly to the left"; icon = arrow > 0 ? "arrow.right" : "arrow.left"
            } else { text = "Hold steady for a moment"; icon = "checkmark.circle" }
            publishScan(prompt: text, icon: icon, arrow: planeOK && !forcedDesk ? arrow : 0, current: cur, planeStable: planeStable)
            // never wait on a desk ARKit cannot see: sweep done or 3 s in the table step with no plane, or 10 s in total
            if !planeOK && !forcedDesk && (missing.isEmpty || tFrame - (tableStart ?? tFrame) >= autoAssumeSecs || el >= hardCapSecs) {
                assumeDesk(missing.isEmpty ? "sweep done, no plane" : (el >= hardCapSecs ? "10 s cap" : "3 s no plane"))
            } else if planeOK && !forcedDesk && el >= hardCapSecs { forcedDesk = true; mark("cap_with_plane") }
            if (forcedDesk || (planeStable && missing.isEmpty)), let c = candidate {
                var t = TableVolume(base: c.outline); t.height = Float(tableHeight)
                table = t
                let r = forcedDesk ? 1.0 : Double(TableReach.reach(fromFarEdges: edges))
                DispatchQueue.main.async { self.reach = r }
                mask.config.reachMetres = Float(r)
                mark("active")
                advance(.active, "")
            }
        case .active:
            var depthPB: CVPixelBuffer?
            if hasDepth, let d = (frame.smoothedSceneDepth ?? frame.sceneDepth)?.depthMap { depthPB = IsolationEngine.orient(d, ctx: ctx, orientation: orient) }
            var volume: CGImage?
            if let t = table {
                let res = frame.camera.imageResolution
                volume = t.raster(camera: frame.camera.transform, intrinsics: frame.camera.intrinsics, imageSize: res, canvas: CGSize(width: 320, height: 240))
            }
            keep = mask.keep(for: frame.capturedImage, orientation: orient, depth: hasDepth ? depthPB : nil, volume: volume)
        }

        let shown: CIImage
        if step == .active, let keep {
            let k = keep.transformed(by: CGAffineTransform(scaleX: W / keep.extent.width, y: H / keep.extent.height))
            shown = IsolationCompositor.compose(camera: camImg, keep: k, world: world.frame())
        } else { shown = camImg }
        lock.lock(); latest = shown; lock.unlock()

        if step != .active {                                  // per-frame cost while scanning, logged once a second
            procMs += (CACurrentMediaTime() - tFrame) * 1000; procN += 1
            if procN >= 30 { NSLog("[iso] %.2fs scan frame avg %.1f ms (%d frames)", CACurrentMediaTime() - scanStart, procMs / Double(procN), procN); procMs = 0; procN = 0 }
        }
        frames += 1
        let now = CACurrentMediaTime()
        if now - t0 > 1 { let f = Double(frames) / (now - t0); frames = 0; t0 = now; DispatchQueue.main.async { self.fps = f } }
    }

    /// Overall scan progress: hands 25%, table found 35%, sweep coverage 40%. ETA from the pace so far.
    private func publishScan(prompt s: String, icon: String, arrow: Int, current: Int, planeStable: Bool) {
        let hands = min(1, Double(handFrames) / Double(handsNeed))
        let plane = planeStable ? 1.0 : (planeSince.map { min(1, (CACurrentMediaTime() - $0) / planeStableSecs) } ?? 0)
        let sweep = Double(cover.filter { $0 }.count) / Double(cover.count)
        let p = 0.25 * hands + 0.35 * plane + (step == .table ? 0.4 * sweep : 0)
        let el = CACurrentMediaTime() - scanStart
        let eta: Double? = p > 0.08 ? min(30, el * (1 - p) / p) : nil
        let h = ScanHint(icon: icon, arrow: arrow, bins: cover, current: current, elapsed: el, eta: eta)
        DispatchQueue.main.async { self.progress = p; self.prompt = s; self.scan = h }
    }
    private func advance(_ s: Step, _ prompt: String) { DispatchQueue.main.async { self.step = s; self.progress = 0; self.prompt = prompt }; step = s }

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

struct IsolationMetal: UIViewRepresentable {
    let engine: IsolationEngine
    func makeUIView(context: Context) -> MTKView { let v = MTKView(); engine.attach(v); return v }
    func updateUIView(_ v: MTKView, context: Context) {}
}

/// Guided scan by default; "Tap corners" switches to the manual tap flow.
struct IsolationScreen: View {
    @State private var tapMode = false
    var body: some View {
        if tapMode { IsolationTapScreen(back: { tapMode = false }) } else { IsolationScanScreen(useTap: { tapMode = true }) }
    }
}

struct IsolationScanScreen: View {
    var useTap: () -> Void = {}
    @StateObject private var e = IsolationEngine()
    @State private var showControls = false
    private var timeText: String {
        let el = String(format: "%.0fs", e.scan.elapsed)
        if let eta = e.scan.eta { return String(format: "%@ · about %.0fs left", el, max(1, eta.rounded())) }
        return el
    }
    var body: some View {
        ZStack {
            if e.supported { IsolationMetal(engine: e).ignoresSafeArea() } else { Color.black.ignoresSafeArea() }
            // outline of the keep area while adjusting, and the tap layer for "Fit to desk"
            GeometryReader { g in
                ZStack {
                    Color.clear.contentShape(Rectangle()).allowsHitTesting(e.fitting)
                        .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named("iso")).onEnded { v in e.fitTap(at: v.location) })
                    if (showControls || e.fitting) && e.outline.count == 4 && e.step == .active {
                        Path { p in p.move(to: e.outline[0]); e.outline.dropFirst().forEach { p.addLine(to: $0) }; p.closeSubpath() }
                            .stroke(Color.white.opacity(0.8), lineWidth: 1.5).allowsHitTesting(false)
                    }
                }
                .coordinateSpace(name: "iso")
                .onAppear { e.setViewSize(g.size) }
                .onChange(of: g.size) { _, sz in e.setViewSize(sz) }
            }.ignoresSafeArea()
            VStack(spacing: 8) {
                if e.fitting {
                    HStack { Image(systemName: "hand.tap").font(.title3); Text(e.fitPrompt).font(.headline); Spacer(minLength: 0)
                             Button("Cancel") { e.cancelFit() }.buttonStyle(.bordered).controlSize(.small) }
                        .padding(12).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14)).padding(.trailing, 56)
                }
                if e.step != .active {
                    VStack(spacing: 8) {
                        HStack {
                            Text(e.step == .hands ? "Scanning · step 1 of 2" : "Scanning · step 2 of 2").font(.caption).foregroundStyle(.white.opacity(0.7))
                            Spacer()
                            Text(timeText).font(.caption.monospacedDigit()).foregroundStyle(.white.opacity(0.85))
                        }
                        HStack(spacing: 8) {
                            Image(systemName: e.scan.icon).font(.title3)
                            Text(e.prompt).font(.headline).multilineTextAlignment(.leading)
                            Spacer(minLength: 0)
                        }
                        ProgressView(value: e.progress).tint(.white)
                        if e.step == .table { SweepStrip(bins: e.scan.bins, current: e.scan.current) }
                        if e.needFallback {
                            Text("Can't find the desk surface.").font(.footnote).foregroundStyle(.white.opacity(0.8))
                            HStack {
                                Button("Assume a desk") { e.useAssumedDesk() }.buttonStyle(.borderedProminent).controlSize(.small)
                                Button("Tap corners") { useTap() }.buttonStyle(.bordered).controlSize(.small)
                            }
                        }
                    }
                    .padding(12).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14)).padding(.trailing, 56)
                }
                if let m = e.message { Text(m).font(.footnote).padding(8).background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 10)).padding(.trailing, 56) }
                Spacer()
                if showControls {
                    VStack(spacing: 10) {
                        Picker("World", selection: $e.theme) { ForEach(IsolationWorld.Theme.allCases, id: \.self) { Text($0.name).tag($0) } }.pickerStyle(.segmented)
                        HStack { Text("Height").font(.caption); Slider(value: $e.tableHeight, in: 0.15...0.9); Text(String(format: "%.2f m", e.tableHeight)).font(.caption.monospacedDigit()) }
                        if e.deskAdjustable {
                            HStack { Text("Width").font(.caption); Slider(value: $e.deskWidth, in: 0.2...1.6); Text(String(format: "%.2f m", e.deskWidth)).font(.caption.monospacedDigit()) }
                            HStack { Text("Depth").font(.caption); Slider(value: $e.deskDepth, in: 0.2...1.2); Text(String(format: "%.2f m", e.deskDepth)).font(.caption.monospacedDigit()) }
                            HStack { Text("Desk level").font(.caption); Slider(value: $e.deskLevel, in: -0.3...0.3); Text(String(format: "%+.2f m", e.deskLevel)).font(.caption.monospacedDigit()) }
                        }
                        if e.hasDepth { HStack { Text("Reach").font(.caption); Slider(value: $e.reach, in: 0.3...2.5); Text(String(format: "%.1f m", e.reach)).font(.caption.monospacedDigit()) } }
                        HStack { Button("Scan again") { e.restart() }.buttonStyle(.bordered).controlSize(.small)
                                 Button("Tap corners") { useTap() }.buttonStyle(.bordered).controlSize(.small) }
                    }
                    .padding(12).background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 16)).padding(.trailing, 56)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                HStack {
                    if e.deskAdjustable && e.step == .active && !e.fitting {
                        Button { e.startFit() } label: { Label("Fit to desk", systemImage: "rectangle.dashed").font(.footnote.weight(.semibold)) }
                            .buttonStyle(.borderedProminent).controlSize(.small)
                    }
                    Spacer()
                    Button { withAnimation { showControls.toggle() }; e.outlineOn = showControls } label: {
                        Image(systemName: showControls ? "chevron.down" : "slider.horizontal.3").font(.system(size: 15, weight: .semibold))
                            .frame(width: 34, height: 34).background(.black.opacity(0.5), in: Circle())
                    }.padding(.trailing, 56)
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8).foregroundStyle(.white)
        }
        .overlay {
            if e.step == .table && e.scan.arrow != 0 {
                Image(systemName: e.scan.arrow > 0 ? "chevron.right.2" : "chevron.left.2")
                    .font(.system(size: 64, weight: .bold)).foregroundStyle(.white.opacity(0.9))
                    .shadow(radius: 6).frame(maxWidth: .infinity, alignment: e.scan.arrow > 0 ? .trailing : .leading)
                    .padding(.horizontal, 70).allowsHitTesting(false)
            }
        }
        .onChange(of: e.fitting) { _, f in e.outlineOn = f || showControls }
        .onAppear { e.start() }
        .onDisappear { e.stop() }
    }
}

/// Sweep coverage: five segments left to right; filled = already seen, ring = where the phone points now.
struct SweepStrip: View {
    let bins: [Bool]; let current: Int
    var body: some View {
        HStack(spacing: 4) {
            ForEach(bins.indices, id: \.self) { i in
                RoundedRectangle(cornerRadius: 4)
                    .fill(bins[i] ? Color.green.opacity(0.9) : Color.white.opacity(0.18))
                    .frame(height: 10)
                    .overlay(RoundedRectangle(cornerRadius: 4).stroke(i == current ? Color.white : Color.clear, lineWidth: 2))
            }
        }
    }
}

extension ARFrame {
    /// World-space ray through a point in view coordinates (portrait, aspect-fill viewport).
    func rayThrough(viewPoint p: CGPoint, viewSize vs: CGSize) -> ARRaycastQuery? {
        let n = CGPoint(x: p.x / vs.width, y: p.y / vs.height)
        let img = n.applying(displayTransform(for: .portrait, viewportSize: vs).inverted())
        return raycastQuery(from: img, allowing: .estimatedPlane, alignment: .horizontal)
    }
}
