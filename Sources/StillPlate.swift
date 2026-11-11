import Metal
import simd
import Foundation

/// A value that the render thread and a background queue both touch. Reads and writes are serialised by a lock.
final class Guarded<T> {
    private var v: T
    private let lock = NSLock()
    init(_ v: T) { self.v = v }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return v }
        set { lock.lock(); v = newValue; lock.unlock() }
    }
}

/// Still-camera background plate (see the stillClassify / stillResolve / stillCommit kernels in Shaders.metal).
/// World-anchored equirectangular canvas (4096x2048, rotation only); a still camera is just the case where the rotation never changes. The caller decides when the camera is still.
/// Mirror of `CanvasU` in Shaders.metal (144 bytes).
struct CanvasUniforms {
    var Rt = matrix_identity_float3x3          // current camera -> world
    var Rref = matrix_identity_float3x3        // canvas camera -> world
    var fx: Float = 1, fy: Float = 1, cx: Float = 0, cy: Float = 0
    var imageSize = SIMD2<Float>(1, 1)
    var canvasSize = SIMD2<Float>(1, 1)
    var cf = SIMD4<Float>(1, 1, 0, 0)
    static let identity = CanvasUniforms()
}

final class StillPlate {
    private struct StillU { var gain = SIMD4<Float>(1, 1, 1, 1); var imageSize = SIMD2<Float>(1, 1); var learn: Float = 1; var ringScale: Float = 1 }

        /// Canvas = equirectangular map of every direction (about 11 px per degree): memory of the whole room, no "moved too far" reset.
    static let canvasW = 4096, canvasH = 2048
    private let device: MTLDevice
    private let classifyPSO: MTLComputePipelineState, resolvePSO: MTLComputePipelineState
    private let commitPSO: MTLComputePipelineState, clearPSO: MTLComputePipelineState
    private let statsBuf: MTLBuffer
    private var w = 0, h = 0           // canvas size in plate px
    private(set) var refRot = matrix_identity_float3x3
    private(set) var refPos = SIMD3<Float>(0, 0, 0)
    private(set) var lastCanvas = CanvasUniforms.identity
    var anchored: Bool { !needsClear }
    private(set) var plate: MTLTexture!
    private var last: MTLTexture!, cls: MTLTexture!, fgRaw: MTLTexture!
    private(set) var fg: MTLTexture!, disp: MTLTexture!, off: MTLTexture!
    let dummy: MTLTexture
    private(set) var gain = SIMD3<Float>(1, 1, 1)
    private var needsClear = true
    /// Bumped on every reset so asynchronous writers (MI-GAN) can tell their result is stale.
    private let genBox = Guarded(0)
    /// Bumped on every reset; the background fill compares it to drop stale results.
    var generation: Int { genBox.value }
    private(set) var resetCount = 0
    /// Share of the frame whose background the plate knows (0...1). Updated one frame late.
    private(set) var coverage: Float = 0
    /// True when most of the visible background suddenly differs from the plate for several frames: the camera moved (or the lights changed).
    private(set) var backgroundChanged = false
    private var changeStreak = 0
    func clearChangeFlag() { backgroundChanged = false; changeStreak = 0 }

    init?(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        guard let a = library.makeFunction(name: "stillClassify"), let b = library.makeFunction(name: "stillResolve"),
              let c = library.makeFunction(name: "stillCommit"), let d = library.makeFunction(name: "stillClear"),
              let pa = try? device.makeComputePipelineState(function: a), let pb = try? device.makeComputePipelineState(function: b),
              let pc = try? device.makeComputePipelineState(function: c), let pd = try? device.makeComputePipelineState(function: d),
              let sb = device.makeBuffer(length: 16 * 4, options: .storageModeShared) else { return nil }
        classifyPSO = pa; resolvePSO = pb; commitPSO = pc; clearPSO = pd; statsBuf = sb
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        td.usage = [.shaderRead]
        dummy = device.makeTexture(descriptor: td)!
    }

    /// Angle (degrees) between the canvas orientation and a camera orientation.
    func angleFromRef(_ camRot: simd_float3x3) -> Float {
        let a = -refRot.columns.2, b = -camRot.columns.2
        return acos(max(-1, min(1, simd_dot(a, b)))) * 180 / .pi
    }

    private(set) var lastResetReason = "start"
    /// Share of the visible, non-person background that differs strongly from the plate (previous frame). High = the plate no longer lines up.
    private(set) var misalignFrac: Float = 0
    func reset(_ why: String = "manual") { lastResetReason = why; misalignFrac = 0; needsClear = true; coverage = 0; gain = SIMD3<Float>(1, 1, 1); genBox.value += 1; resetCount += 1 }

    private func tex(_ fmt: MTLPixelFormat, mips: Bool = false) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: mips)
        td.usage = [.shaderRead, .shaderWrite]; td.storageMode = .private
        return device.makeTexture(descriptor: td)!
    }

    private func ensure(_ imageSize: SIMD2<Float>) {
        let nw = StillPlate.canvasW, nh = StillPlate.canvasH
        if nw == w && nh == h && plate != nil { return }
        w = nw; h = nh
        plate = tex(.rgba16Float); last = tex(.rgba16Float); cls = tex(.r8Uint); fgRaw = tex(.r8Unorm)
        fg = tex(.r8Unorm); disp = tex(.rgba16Float); off = tex(.rgba16Float, mips: true)
        needsClear = true
    }

    /// Reads the previous frame's statistics: smoothed exposure gain (live / plate) and plate coverage.
    private func absorbStats() {
        let st = statsBuf.contents().assumingMemoryBound(to: UInt32.self)
        if st[6] > 200 {
            func g(_ l: UInt32, _ p: UInt32) -> Float { min(1.6, max(0.6, Float(l) / max(Float(p), 1))) }
            let target = SIMD3<Float>(g(st[0], st[3]), g(st[1], st[4]), g(st[2], st[5]))
            gain += (target - gain) * 0.35
        }
        if w > 0 { coverage = min(1, Float(st[7]) / max(Float(st[10]), 1)) }       // known pixels / pixels in the current view
        misalignFrac = st[9] > 3000 ? Float(st[8]) / Float(st[9]) : 0
        if st[9] > 3000 && Float(st[8]) / Float(st[9]) > 0.3 { changeStreak += 1 } else { changeStreak = 0 }
        if changeStreak >= 4 { backgroundChanged = true }
    }

    private lazy var dumpPSO: MTLComputePipelineState? = {
        guard let lib = device.makeDefaultLibrary(), let f = lib.makeFunction(name: "stillDump") else { return nil }
        return try? device.makeComputePipelineState(function: f)
    }()

    /// Diagnostics: copies live frame, plate colour, plate confidence and foreground mask into shared RGBA8 textures (plate resolution).
    func encodeDump(cb: MTLCommandBuffer, y: MTLTexture, c: MTLTexture) -> [MTLTexture]? {
        var cu = lastCanvas
        guard w > 0, let pso = dumpPSO, let disp, let fg else { return nil }
        var outs: [MTLTexture] = []
        for _ in 0..<4 {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1024, height: 512, mipmapped: false)
            td.usage = [.shaderWrite, .shaderRead]; td.storageMode = .shared
            outs.append(device.makeTexture(descriptor: td)!)
        }
        guard let e = cb.makeComputeCommandEncoder() else { return nil }
        e.setComputePipelineState(pso)
        e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(disp, index: 2); e.setTexture(fg, index: 3)
        for i in 0..<4 { e.setTexture(outs[i], index: 4 + i) }
        e.setBytes(&cu, length: MemoryLayout<CanvasUniforms>.stride, index: 0)
        e.dispatchThreads(MTLSize(width: 1024, height: 512, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        e.endEncoding()
        return outs
    }

    /// Encodes the three passes. The frame's textures must be the camera planes and the person masks.
    func encode(cb: MTLCommandBuffer, y: MTLTexture, c: MTLTexture, mask: MTLTexture, yolo: MTLTexture, imageSize: SIMD2<Float>, learn: Bool, camRot: simd_float3x3, camPos: SIMD3<Float>, K: simd_float3x3) {
        absorbStats()
        ensure(imageSize)
        memset(statsBuf.contents(), 0, 16 * 4)
        let grid = MTLSize(width: w, height: h, depth: 1), tg = MTLSize(width: 16, height: 16, depth: 1)
        if needsClear, let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(clearPSO)
            e.setTexture(plate, index: 0); e.setTexture(last, index: 1); e.setTexture(fg, index: 2)
            e.setTexture(disp, index: 3); e.setTexture(off, index: 4)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
            needsClear = false
            refRot = camRot; refPos = camPos                            // the canvas is anchored to the orientation of this first frame
        }
        let divisorEff = K[0][0] * 2 * .pi / Float(w)                     // camera pixels per canvas pixel at the centre of the view
        var cu = CanvasUniforms(Rt: camRot, Rref: refRot, fx: K[0][0], fy: K[1][1], cx: K[2][0], cy: K[2][1], imageSize: imageSize,
                                canvasSize: SIMD2<Float>(Float(w), Float(h)), cf: SIMD4<Float>(1, 1, 0, 0))
        lastCanvas = cu
        var u = StillU(gain: SIMD4<Float>(gain.x, gain.y, gain.z, 1), imageSize: imageSize, learn: learn ? 1 : 0, ringScale: 3 / divisorEff)
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(classifyPSO)
            e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(mask, index: 2); e.setTexture(yolo, index: 3)
            e.setTexture(plate, index: 4); e.setTexture(last, index: 5); e.setTexture(cls, index: 6)
            e.setBytes(&u, length: MemoryLayout<StillU>.stride, index: 0)
            e.setBytes(&cu, length: MemoryLayout<CanvasUniforms>.stride, index: 2)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(resolvePSO)
            e.setTexture(cls, index: 0); e.setTexture(last, index: 1); e.setTexture(fg, index: 2); e.setTexture(fgRaw, index: 3)
            e.setBytes(&u, length: MemoryLayout<StillU>.stride, index: 0)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(commitPSO)
            e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(fgRaw, index: 2); e.setTexture(cls, index: 3)
            e.setTexture(last, index: 4); e.setTexture(plate, index: 5); e.setTexture(fg, index: 6)
            e.setTexture(disp, index: 7); e.setTexture(off, index: 8)
            e.setBytes(&u, length: MemoryLayout<StillU>.stride, index: 0)
            e.setBuffer(statsBuf, offset: 0, index: 1)
            e.setBytes(&cu, length: MemoryLayout<CanvasUniforms>.stride, index: 2)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        if let b = cb.makeBlitCommandEncoder() { b.generateMipmaps(for: off); b.endEncoding() }
    }
}
