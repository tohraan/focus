import ARKit
import MetalKit
import SwiftUI
import AVFoundation

struct RenderUniforms {
    var viewToImage = matrix_identity_float3x3
    var camToWorld = matrix_identity_float3x3
    var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
    var viewSize = SIMD2<Float>(1, 1)
    var imageSize = SIMD2<Float>(1, 1)
    var mode: Int32 = 0
    var splitX: Float = 0.5
    var gain = SIMD4<Float>(1, 1, 1, 1)
    var still: Int32 = 0
}

struct PanoUniforms {
    var camToWorld = matrix_identity_float3x3
    var fx: Float = 0, fy: Float = 0, cx: Float = 0, cy: Float = 0
    var imageSize = SIMD2<Float>(1, 1)
    var maskRadius: Float = 0.004
    var blend: Float = 0.15
}

private struct FrameData {
    var yTex: MTLTexture, cTex: MTLTexture, mTex: MTLTexture, yoloTex: MTLTexture
    var keep: [CVMetalTexture]
    var pose: simd_float4x4
    var camToWorld: simd_float3x3
    var intrinsics: simd_float3x3
    var imageSize: SIMD2<Float>
    var displayInverse: simd_float3x3
    var timestamp: TimeInterval
    var trackingNormal: Bool
}

final class Renderer: NSObject, ObservableObject, ARSessionDelegate, MTKViewDelegate {
    // HUD state (main thread)
    @Published var fps = 0.0
    @Published var segMs = 0.0
    @Published var latencyMs = 0.0
    @Published var tracking = "starting"
    @Published var message: String? = nil
    @Published var autoStatus = "autofill: idle"
    @Published var stillStatus = "camera: moving"
    @Published var captureTitle: String? = nil        // banner shown after "Capture background"
    @Published var captureDetail = ""
    @Published var captureProgress = 0.0

    // controls (benign races: plain values read by render thread)
    var mode: Int32 = 0
    var splitX: Float = 0.5

    static let panoW = 4096, panoH = 2048

    private let device: MTLDevice
    private let queue: MTLCommandQueue
    private let compositePSO: MTLRenderPipelineState
    private let updatePSO: MTLComputePipelineState
    private var pano: MTLTexture
    private let statsBuf: MTLBuffer
    private let gainPSO: MTLComputePipelineState
    private var gain = SIMD4<Float>(1, 1, 1, 1)
    private var autoFill: AutoFill?
    var engine = 1                       // 0 = 11:05 panorama memory, 1 = R2 pose-aware keyframes
    private var kf: KeyframeFill?
    private let compositeKFPSO: MTLRenderPipelineState
    private var kfResetFlag = false
    private var plate: StillPlate?
    private var migan: MiganFill?
    // still-camera detector: moved > 5 mm or 0.3 deg over the last second => moving; drift > 3 cm or 2 deg from the anchor => moving
    private var poseHist: [(t: Double, pose: simd_float4x4)] = []
    private var stillAnchor: simd_float4x4?
    private var stillSince = 0.0
    private var stillActive = false
    private var lastNormalT = 0.0
    private var captureStart: Double?
    private var captureDoneAt: Double?
    private var emptyYolo: MTLTexture
    private let fillTex: MTLTexture
    private let fillPSO: MTLComputePipelineState
    private var texCache: CVMetalTextureCache!
    private let session = ARSession()
    private let segmenter = PersonSegmenter()
    private let segQueue = DispatchQueue(label: "focus.seg", qos: .userInteractive)
    private let lock = NSLock()
    private var busy = false
    private var current: FrameData?
    private weak var view: MTKView?
    private var viewSize = CGSize(width: 390, height: 844)
    private var frameCount = 0
    private var lastFpsT = ProcessInfo.processInfo.systemUptime
    private var csv: FileHandle?
    private var segMsLast = 0.0

    override init() {
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue()!
        let lib = device.makeDefaultLibrary()!
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "fullscreenVS")
        d.fragmentFunction = lib.makeFunction(name: "compositeFS")
        d.colorAttachments[0].pixelFormat = .bgra8Unorm
        compositePSO = try! device.makeRenderPipelineState(descriptor: d)
        let dk = MTLRenderPipelineDescriptor()
        dk.vertexFunction = lib.makeFunction(name: "fullscreenVS")
        dk.fragmentFunction = lib.makeFunction(name: "compositeKF")
        dk.colorAttachments[0].pixelFormat = .bgra8Unorm
        compositeKFPSO = try! device.makeRenderPipelineState(descriptor: dk)
        updatePSO = try! device.makeComputePipelineState(function: lib.makeFunction(name: "updatePano")!)
        fillPSO = try! device.makeComputePipelineState(function: lib.makeFunction(name: "premultFill")!)
        let fd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 480, height: 360, mipmapped: true)
        fd.usage = [.shaderRead, .shaderWrite]; fd.storageMode = .private
        fillTex = device.makeTexture(descriptor: fd)!
        pano = Renderer.makePano(device)
        statsBuf = device.makeBuffer(length: 32, options: .storageModeShared)!
        gainPSO = try! device.makeComputePipelineState(function: lib.makeFunction(name: "gainStats")!)
        emptyYolo = Renderer.makeYolo(device, [UInt8](repeating: 0, count: 320 * 240 * 2), 320, 240)
        super.init()
        CVMetalTextureCacheCreate(nil, nil, device, nil, &texCache)
        kf = KeyframeFill(device: device, library: lib)
        plate = StillPlate(device: device, library: lib)
        migan = MiganFill(device: device, queue: queue, library: lib)
        migan?.onStatus = { [weak self] t in DispatchQueue.main.async { self?.autoStatus = t } }
        autoFill = AutoFill(device: device, queue: queue, library: lib)
        autoFill?.onStatus = { [weak self] t in DispatchQueue.main.async { self?.autoStatus = t } }
        setupCSV()
    }

    private static func makeYolo(_ device: MTLDevice, _ bytes: [UInt8], _ w: Int, _ h: Int) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg8Unorm, width: w, height: h, mipmapped: false)
        let t = device.makeTexture(descriptor: td)!
        t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: bytes, bytesPerRow: w * 2)
        return t
    }

    private static func makePano(_ device: MTLDevice, _ fmt: MTLPixelFormat = .rgba8Unorm) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: panoW, height: panoH, mipmapped: false)
        td.usage = [.shaderRead, .shaderWrite]
        td.storageMode = .private
        let t = device.makeTexture(descriptor: td)!
        return t
    }

    /// Forget everything observed so far (the "capture background" button restarts learning).
    func resetMemory() {
        kf?.reset()
        plate?.reset()
        poseHist.removeAll(); stillAnchor = nil; stillActive = false; lastNormalT = 0
        lock.lock(); defer { lock.unlock() }
        pano = Renderer.makePano(device)   // zero-filled => alpha 0 = unknown
    }

    /// The "Capture background" button: clears everything and shows what is happening until the background is learned.
    func beginCapture() {
        resetMemory()
        captureStart = ProcessInfo.processInfo.systemUptime; captureDoneAt = nil
        DispatchQueue.main.async {
            self.captureTitle = "Capturing background"
            self.captureDetail = "Memory cleared. Hold the phone still or prop it up, then move around so every part of the room is seen without you."
            self.captureProgress = 0
        }
    }

    /// Called every frame from the GPU completion handler: drives the banner and the HUD line.
    private func updateCaptureUI() {
        let now = ProcessInfo.processInfo.systemUptime
        let cov = Double(plate?.coverage ?? 0)
        let still = stillActive
        var title: String? = nil, detail = "", prog = 0.0
        if let t0 = captureStart {
            let el = now - t0
            if still {
                prog = cov
                if cov >= 0.9 || captureDoneAt != nil {
                    if captureDoneAt == nil { captureDoneAt = now }
                    title = "Background captured"; detail = String(format: "%.0f%% of the scene learned. You can move freely now.", cov * 100); prog = 1
                } else {
                    title = "Capturing background"
                    detail = String(format: "Camera is still. %.0f%% of the scene learned. Keep moving around; spots you stand on are filled once you step away.", cov * 100)
                }
            } else {
                prog = min(1, Double(kf?.count ?? 0) / 6.0)
                title = "Capturing background"
                detail = "Camera is moving. Pan slowly across the room (views saved: \(kf?.count ?? 0)). Prop the phone still to learn the background faster."
                if (kf?.count ?? 0) >= 6 && captureDoneAt == nil { captureDoneAt = now }
                if captureDoneAt != nil { title = "Background captured"; detail = "\(kf?.count ?? 0) views saved."; prog = 1 }
            }
            if let d = captureDoneAt, now - d > 2.5 { captureStart = nil; captureDoneAt = nil; title = nil }
            else if el > 30 { captureStart = nil; title = nil }
        }
        let st = still ? String(format: "camera: still · background %.0f%%", cov * 100) : "camera: moving"
        DispatchQueue.main.async {
            self.stillStatus = st
            self.captureTitle = title; if title != nil { self.captureDetail = detail; self.captureProgress = prog }
        }
    }

    /// Camera state from the ARKit pose (identity pose offline => always still).
    /// 0 = moving or lost (plate invalid), 1 = frozen (small shake or brief tracking dropout: keep using the plate, stop learning), 2 = still (learn + use).
    private func updateStill(pose: simd_float4x4, t: Double, normal: Bool) -> Int {
        func dist(_ a: simd_float4x4, _ b: simd_float4x4) -> Float {
            simd_distance(SIMD3(a.columns.3.x, a.columns.3.y, a.columns.3.z), SIMD3(b.columns.3.x, b.columns.3.y, b.columns.3.z))
        }
        guard normal else {                                   // jumping next to the table can make ARKit drop to 'limited' for a moment
            if t - lastNormalT > 2.0 { poseHist.removeAll(); stillAnchor = nil; return 0 }
            return stillAnchor == nil ? 0 : 1
        }
        lastNormalT = t
        poseHist.append((t, pose))
        while poseHist.count > 1, t - poseHist[1].t >= 1.0 { poseHist.removeFirst() }
        let ref = poseHist[0].pose
        let da = Reproject.angleDeg(ref, pose), dd = dist(ref, pose)
        let moved = da > 1.0 || dd > 0.03                     // a real move: nothing learned so far lines up any more
        if let a = stillAnchor, Reproject.angleDeg(a, pose) > 1.5 || dist(a, pose) > 0.02 { poseHist.removeAll(); stillAnchor = nil; return 0 }
        if moved { poseHist.removeAll(); stillAnchor = nil; return 0 }
        if stillAnchor == nil { stillAnchor = pose; stillSince = t }
        if da > 0.3 || dd > 0.005 { return 1 }                // shake: keep the plate, pause learning
        return t - stillSince >= 0.5 ? 2 : 1
    }

    private func setupCSV() {
        let url = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0].appendingPathComponent("metrics.csv")
        FileManager.default.createFile(atPath: url.path, contents: Data("t,seg_ms,latency_ms,tracking\n".utf8))
        csv = try? FileHandle(forWritingTo: url)
        csv?.seekToEndOfFile()
    }

    func attach(_ v: MTKView) {
        view = v
        v.device = device
        v.colorPixelFormat = .bgra8Unorm
        v.framebufferOnly = true
        v.isPaused = true
        v.enableSetNeedsDisplay = false
        v.delegate = self
    }

    func start() {
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .denied, .restricted:
            DispatchQueue.main.async { self.message = "Camera access denied. Enable it in Settings > FOCUS." }
            return
        default: break
        }
        guard ARWorldTrackingConfiguration.isSupported else {
            message = "ARKit world tracking is not supported on this device."
            return
        }
        session.delegate = self
        session.delegateQueue = DispatchQueue(label: "focus.ar")
        let cfg = ARWorldTrackingConfiguration()
        cfg.worldAlignment = .gravity
        session.run(cfg, options: [.resetTracking, .removeExistingAnchors])
    }

    func stop() { session.pause() }

    // MARK: ARSessionDelegate
    func session(_ session: ARSession, didFailWithError error: Error) {
        DispatchQueue.main.async { self.message = "Camera/AR error: \(error.localizedDescription)" }
    }
    func sessionWasInterrupted(_ session: ARSession) {
        DispatchQueue.main.async { self.tracking = "interrupted" }
    }
    func sessionInterruptionEnded(_ session: ARSession) {
        kf?.reset()
        session.run(session.configuration ?? ARWorldTrackingConfiguration(), options: [.resetTracking])
    }

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        lock.lock()
        if busy { lock.unlock(); return }
        busy = true
        lock.unlock()

        let size = viewSize
        let pb = frame.capturedImage
        let cam = frame.camera
        let ts = frame.timestamp
        let disp = frame.displayTransform(for: .portrait, viewportSize: size).inverted()
        let trackText: String
        switch cam.trackingState {
        case .normal: trackText = "good"
        case .notAvailable: trackText = "unavailable"
        case .limited(let r): trackText = "limited (\(r))"
        }
        let normal = cam.trackingState == .normal
        if case .limited(let why) = cam.trackingState, why == .relocalizing { kfResetFlag = true }   // world frame may have jumped
        let m = cam.transform
        let c2w = simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        let K = cam.intrinsics
        let isz = SIMD2<Float>(Float(CVPixelBufferGetWidth(pb)), Float(CVPixelBufferGetHeight(pb)))

        segQueue.async { [weak self] in
            guard let self else { return }
            let t0 = ProcessInfo.processInfo.systemUptime
            let mo = self.segmenter.segment(pb)
            guard let maskPB = mo.vision,
                  let fd = self.makeFrame(pb: pb, mask: maskPB, yolo: mo.yolo, pose: m, c2w: c2w, K: K, isz: isz,
                                          disp: disp, ts: ts, normal: normal) else {
                self.release(); return
            }
            self.segMsLast = (ProcessInfo.processInfo.systemUptime - t0) * 1000
            DispatchQueue.main.async {
                self.tracking = trackText
                self.segMs = self.segMsLast
                self.current = fd
                self.view?.draw()
            }
        }
    }

    private func release() { lock.lock(); busy = false; lock.unlock() }

    private func tex(_ pb: CVPixelBuffer, _ fmt: MTLPixelFormat, plane: Int) -> (MTLTexture, CVMetalTexture)? {
        var out: CVMetalTexture?
        let w = plane < 0 ? CVPixelBufferGetWidth(pb) : CVPixelBufferGetWidthOfPlane(pb, plane)
        let h = plane < 0 ? CVPixelBufferGetHeight(pb) : CVPixelBufferGetHeightOfPlane(pb, plane)
        let r = CVMetalTextureCacheCreateTextureFromImage(nil, texCache, pb, nil, fmt, w, h, max(plane, 0), &out)
        guard r == kCVReturnSuccess, let o = out, let t = CVMetalTextureGetTexture(o) else { return nil }
        return (t, o)
    }

    private func makeFrame(pb: CVPixelBuffer, mask: CVPixelBuffer, yolo: YoloDecode.Result?, pose: simd_float4x4, c2w: simd_float3x3, K: simd_float3x3,
                           isz: SIMD2<Float>, disp: CGAffineTransform, ts: TimeInterval, normal: Bool) -> FrameData? {
        guard let (y, ky) = tex(pb, .r8Unorm, plane: 0),
              let (c, kc) = tex(pb, .rg8Unorm, plane: 1),
              let (m, km) = tex(mask, .r8Unorm, plane: -1) else { return nil }
        let d = simd_float3x3(SIMD3(Float(disp.a), Float(disp.b), 0),
                              SIMD3(Float(disp.c), Float(disp.d), 0),
                              SIMD3(Float(disp.tx), Float(disp.ty), 1))
        let yt = yolo.map { Renderer.makeYolo(device, $0.mask, $0.w, $0.h) } ?? emptyYolo
        return FrameData(yTex: y, cTex: c, mTex: m, yoloTex: yt, keep: [ky, kc, km], pose: pose, camToWorld: c2w, intrinsics: K,
                         imageSize: isz, displayInverse: d, timestamp: ts, trackingNormal: normal)
    }

    // MARK: MTKViewDelegate
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        viewSize = view.bounds.size
    }

    func draw(in view: MTKView) {
        guard let fd = current,
              let drawable = view.currentDrawable,
              let rpd = view.currentRenderPassDescriptor,
              let cb = queue.makeCommandBuffer() else { release(); return }
        current = nil
        viewSize = view.bounds.size
        let K = fd.intrinsics
        let panoTex = pano
        encodeScene(fd: fd, cb: cb, rpd: rpd, viewSize: viewSize)
        cb.present(drawable)
        finishDraw(fd: fd, cb: cb, panoTex: panoTex, offline: false)
    }

    /// Everything that turns one FrameData into pixels (memory update, fill, keyframes, composite). Shared by live and offline paths.
    private func encodeScene(fd: FrameData, cb: MTLCommandBuffer, rpd: MTLRenderPassDescriptor, viewSize: CGSize) {
        let K = fd.intrinsics
        let panoTex = pano
        if kfResetFlag { kf?.reset(); kfResetFlag = false }
        let camState = updateStill(pose: fd.pose, t: fd.timestamp, normal: fd.trackingNormal)
        let nowStill = camState > 0
        if nowStill, let pl = plate {
            pl.encode(cb: cb, y: fd.yTex, c: fd.cTex, mask: fd.mTex, yolo: fd.yoloTex, imageSize: fd.imageSize, learn: camState == 2)
        } else if stillActive { plate?.reset() }          // camera moved or tracking lost for good: the plate is no longer valid
        stillActive = nowStill
        if engine == 0, fd.trackingNormal, let enc = cb.makeComputeCommandEncoder() {
            var pu = PanoUniforms(camToWorld: fd.camToWorld, fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                  imageSize: fd.imageSize, maskRadius: 4.0 / fd.imageSize.x, blend: 0.15)
            enc.setComputePipelineState(updatePSO)
            enc.setTexture(fd.yTex, index: 0); enc.setTexture(fd.cTex, index: 1)
            enc.setTexture(fd.mTex, index: 2); enc.setTexture(panoTex, index: 3)
            enc.setTexture(fd.yoloTex, index: 4)
            enc.setBytes(&pu, length: MemoryLayout<PanoUniforms>.stride, index: 0)
            let tg = MTLSize(width: 16, height: 16, depth: 1)
            enc.dispatchThreads(MTLSize(width: Renderer.panoW, height: Renderer.panoH, depth: 1), threadsPerThreadgroup: tg)
            enc.endEncoding()
        }

        if engine == 0, fd.trackingNormal, let enc = cb.makeComputeCommandEncoder() {
            memset(statsBuf.contents(), 0, 32)
            var pu = PanoUniforms(camToWorld: fd.camToWorld, fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1], imageSize: fd.imageSize)
            enc.setComputePipelineState(gainPSO)
            enc.setTexture(fd.yTex, index: 0); enc.setTexture(fd.cTex, index: 1)
            enc.setTexture(fd.mTex, index: 2); enc.setTexture(panoTex, index: 3); enc.setTexture(fd.yoloTex, index: 4)
            enc.setBytes(&pu, length: MemoryLayout<PanoUniforms>.stride, index: 0)
            enc.setBuffer(statsBuf, offset: 0, index: 1)
            enc.dispatchThreads(MTLSize(width: 64, height: 48, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            enc.endEncoding()
        }
        if let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(fillPSO)
            enc.setTexture(fd.yTex, index: 0); enc.setTexture(fd.cTex, index: 1)
            enc.setTexture(fd.mTex, index: 2); enc.setTexture(fd.yoloTex, index: 3); enc.setTexture(fillTex, index: 4)
            enc.dispatchThreads(MTLSize(width: 480, height: 360, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        if let blit = cb.makeBlitCommandEncoder() { blit.generateMipmaps(for: fillTex); blit.endEncoding() }

        var kfU = KFUniforms()
        if engine == 1, let kf = kf {
            kf.encode(cb: cb, y: fd.yTex, c: fd.cTex, mask: fd.mTex, yolo: fd.yoloTex, camToWorld: fd.pose, K: K,
                      imageSize: fd.imageSize, tracking: fd.trackingNormal, time: fd.timestamp, still: stillActive)
            kfU = kf.uniforms
        }
        var ru = RenderUniforms(viewToImage: fd.displayInverse, camToWorld: fd.camToWorld,
                                fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1],
                                viewSize: SIMD2(Float(viewSize.width), Float(viewSize.height)),
                                imageSize: fd.imageSize, mode: mode, splitX: splitX, gain: gain,
                                still: stillActive && plate != nil ? 1 : 0)
        let stD = stillActive ? plate?.disp : nil, stO = stillActive ? plate?.off : nil, stF = stillActive ? plate?.fg : nil
        let dummyTex = plate?.dummy ?? emptyYolo
        let enc = cb.makeRenderCommandEncoder(descriptor: rpd)!
        enc.setFragmentTexture(stD ?? dummyTex, index: 8); enc.setFragmentTexture(stO ?? dummyTex, index: 9); enc.setFragmentTexture(stF ?? dummyTex, index: 10)
        if engine == 1, let kf = kf {
            enc.setRenderPipelineState(compositeKFPSO)
            enc.setFragmentTexture(fd.yTex, index: 0); enc.setFragmentTexture(fd.cTex, index: 1)
            enc.setFragmentTexture(fd.mTex, index: 2); enc.setFragmentTexture(fd.yoloTex, index: 3)
            enc.setFragmentTexture(fillTex, index: 4); enc.setFragmentTexture(kf.diffTex, index: 5)
            enc.setFragmentTexture(kf.colorArr, index: 6); enc.setFragmentTexture(kf.maskArr, index: 7)
            enc.setFragmentTexture(panoTex, index: 11)
            enc.setFragmentBytes(&ru, length: MemoryLayout<RenderUniforms>.stride, index: 0)
            enc.setFragmentBytes(&kfU, length: MemoryLayout<KFUniforms>.stride, index: 1)
        } else {
            enc.setRenderPipelineState(compositePSO)
            enc.setFragmentTexture(fd.yTex, index: 0); enc.setFragmentTexture(fd.cTex, index: 1)
            enc.setFragmentTexture(fd.mTex, index: 2); enc.setFragmentTexture(panoTex, index: 3)
            enc.setFragmentTexture(fd.yoloTex, index: 4); enc.setFragmentTexture(fillTex, index: 5)
            enc.setFragmentBytes(&ru, length: MemoryLayout<RenderUniforms>.stride, index: 0)
        }
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
    }

    private func finishDraw(fd: FrameData, cb: MTLCommandBuffer, panoTex: MTLTexture, offline: Bool) {
        let K = fd.intrinsics
        let keep = fd.keep
        let (yT, cT, mT, oT) = (fd.yTex, fd.cTex, fd.mTex, fd.yoloTex)
        let keepObjs: [AnyObject] = keep.map { $0 as AnyObject }
        let c2w = fd.camToWorld, isz = fd.imageSize, fdTracking = fd.trackingNormal
        let ts = fd.timestamp
        let track = fd.trackingNormal ? "normal" : "limited"
        cb.addCompletedHandler { [weak self] _ in
            _ = keep
            guard let self else { return }
            let st = self.statsBuf.contents().assumingMemoryBound(to: UInt32.self)
            if st[6] > 40 {
                let n = Float(st[6])
                func g(_ l: UInt32, _ m: UInt32) -> Float { min(1.6, max(0.6, Float(l) / max(Float(m), 1))) }
                let target = SIMD4<Float>(g(st[0], st[3]), g(st[1], st[4]), g(st[2], st[5]), 1)
                self.gain += (target - self.gain) * 0.35      // smoothed so lighting changes never flicker
                _ = n
            }
            self.kf?.finishSweep()
            if self.engine == 1, self.frameCount % 10 == 0, let k = self.kf { let t = k.status; DispatchQueue.main.async { self.autoStatus = t } }
            if !offline, self.stillActive, let pl = self.plate, let mg = self.migan {
                mg.request(MiganFill.Frame(y: yT, c: cT, keep: keepObjs), plate: pl)
            }
            if !offline, !self.stillActive, fdTracking, let af = self.autoFill {
                af.request(AutoFill.Frame(y: yT, c: cT, mask: mT, yolo: oT, keep: keepObjs, camToWorld: c2w,
                                          fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1], imageSize: isz),
                           pano: panoTex, gain: self.gain, stillCurrent: { [weak self] t in self?.pano === t })
            }
            let now = ProcessInfo.processInfo.systemUptime
            let lat = (now - ts) * 1000
            self.frameCount += 1
            self.csv?.write(Data(String(format: "%.3f,%.1f,%.1f,%@\n", now, self.segMsLast, lat, track).utf8))
            let dt = now - self.lastFpsT
            if dt >= 1.0 {
                let f = Double(self.frameCount) / dt
                self.frameCount = 0; self.lastFpsT = now
                DispatchQueue.main.async { self.fps = f }
            }
            DispatchQueue.main.async { self.latencyMs = lat }
            self.updateCaptureUI()
            if !offline { self.release() }
        }
        cb.commit()
    }

    // MARK: Offline (video-file test) path -- same segmenter, same engines, same shaders as live
    /// Process one already-upright NV12 frame with a given pose and render it into `target` (bgra8). Blocks until done.
    /// `mode` 0 = processed, 1 = original.
    func processOffline(pb: CVPixelBuffer, pose: simd_float4x4, K: simd_float3x3, ts: TimeInterval, target: MTLTexture, mode: Int32 = 0) -> Bool {
        let isz = SIMD2<Float>(Float(CVPixelBufferGetWidth(pb)), Float(CVPixelBufferGetHeight(pb)))
        let m = pose
        let c2w = simd_float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z),
                                SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z),
                                SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z))
        let mo = segmenter.segment(pb)
        guard let maskPB = mo.vision,
              let fd = makeFrame(pb: pb, mask: maskPB, yolo: mo.yolo, pose: m, c2w: c2w, K: K, isz: isz,
                                 disp: .identity, ts: ts, normal: true),
              let cb = queue.makeCommandBuffer() else { return false }
        let rpd = MTLRenderPassDescriptor()
        rpd.colorAttachments[0].texture = target
        rpd.colorAttachments[0].loadAction = .dontCare
        rpd.colorAttachments[0].storeAction = .store
        let saved = self.mode; self.mode = mode; defer { self.mode = saved }
        encodeScene(fd: fd, cb: cb, rpd: rpd, viewSize: CGSize(width: Double(isz.x), height: Double(isz.y)))
        finishDraw(fd: fd, cb: cb, panoTex: pano, offline: true)
        cb.waitUntilCompleted()
        if stillActive, let pl = plate, let mg = migan {
            mg.offlineClock = ts
            mg.request(MiganFill.Frame(y: fd.yTex, c: fd.cTex, keep: fd.keep.map { $0 as AnyObject }), plate: pl)
        }
        // same LaMa fill into the memory as the live path, run inline on video time (about once per second of video)
        if !stillActive, let af = autoFill {
            af.offlineClock = ts
            af.request(AutoFill.Frame(y: fd.yTex, c: fd.cTex, mask: fd.mTex, yolo: fd.yoloTex, keep: fd.keep.map { $0 as AnyObject }, camToWorld: fd.camToWorld,
                                      fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1], imageSize: fd.imageSize),
                       pano: pano, gain: gain, stillCurrent: { [weak self] t in self?.pano === t })
        }
        return true
    }

    /// Fresh state for a new clip (both engines' memories are cleared; the caller alternates `engine` per frame).
    func prepareOffline() {
        segmenter.yolo?.syncMode = true
        for _ in 0..<200 where !((autoFill?.modelReady ?? true) && (migan?.modelReady ?? true)) { Thread.sleep(forTimeInterval: 0.1) }   // LaMa preloads asynchronously
        resetMemory(); gain = SIMD4<Float>(1, 1, 1, 1)
    }

    func makeTarget(width: Int, height: Int) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .bgra8Unorm, width: width, height: height, mipmapped: false)
        td.usage = [.renderTarget, .shaderRead]; td.storageMode = .shared
        return device.makeTexture(descriptor: td)!
    }
    var kfStatus: String { kf?.status ?? "" }
}
