import Metal
import simd

/// Still-camera background plate (see the stillClassify / stillResolve / stillCommit kernels in Shaders.metal).
/// Works in image space at 1/3 resolution; needs no camera pose. The caller decides when the camera is still.
final class StillPlate {
    private struct StillU { var gain = SIMD4<Float>(1, 1, 1, 1); var imageSize = SIMD2<Float>(1, 1); var learn: Float = 1 }

    private let device: MTLDevice
    private let classifyPSO: MTLComputePipelineState, resolvePSO: MTLComputePipelineState
    private let commitPSO: MTLComputePipelineState, clearPSO: MTLComputePipelineState
    private let statsBuf: MTLBuffer
    private var w = 0, h = 0
    private(set) var plate: MTLTexture!
    private var last: MTLTexture!, cls: MTLTexture!, fgRaw: MTLTexture!
    private(set) var fg: MTLTexture!, disp: MTLTexture!, off: MTLTexture!
    let dummy: MTLTexture
    private(set) var gain = SIMD3<Float>(1, 1, 1)
    private var needsClear = true
    /// Bumped on every reset so asynchronous writers (MI-GAN) can tell their result is stale.
    private(set) var generation = 0
    /// Share of the frame whose background the plate knows (0...1). Updated one frame late.
    private(set) var coverage: Float = 0

    init?(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        guard let a = library.makeFunction(name: "stillClassify"), let b = library.makeFunction(name: "stillResolve"),
              let c = library.makeFunction(name: "stillCommit"), let d = library.makeFunction(name: "stillClear"),
              let pa = try? device.makeComputePipelineState(function: a), let pb = try? device.makeComputePipelineState(function: b),
              let pc = try? device.makeComputePipelineState(function: c), let pd = try? device.makeComputePipelineState(function: d),
              let sb = device.makeBuffer(length: 8 * 4, options: .storageModeShared) else { return nil }
        classifyPSO = pa; resolvePSO = pb; commitPSO = pc; clearPSO = pd; statsBuf = sb
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        td.usage = [.shaderRead]
        dummy = device.makeTexture(descriptor: td)!
    }

    func reset() { needsClear = true; coverage = 0; gain = SIMD3<Float>(1, 1, 1); generation += 1 }

    private func tex(_ fmt: MTLPixelFormat, mips: Bool = false) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: mips)
        td.usage = [.shaderRead, .shaderWrite]; td.storageMode = .private
        return device.makeTexture(descriptor: td)!
    }

    private func ensure(_ imageSize: SIMD2<Float>) {
        let nw = max(64, Int(imageSize.x / 3)), nh = max(64, Int(imageSize.y / 3))
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
        if w > 0 { coverage = min(1, Float(st[7]) / Float(w * h)) }
    }

    /// Encodes the three passes. The frame's textures must be the camera planes and the person masks.
    func encode(cb: MTLCommandBuffer, y: MTLTexture, c: MTLTexture, mask: MTLTexture, yolo: MTLTexture, imageSize: SIMD2<Float>, learn: Bool) {
        absorbStats()
        ensure(imageSize)
        memset(statsBuf.contents(), 0, 8 * 4)
        let grid = MTLSize(width: w, height: h, depth: 1), tg = MTLSize(width: 16, height: 16, depth: 1)
        if needsClear, let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(clearPSO)
            e.setTexture(plate, index: 0); e.setTexture(last, index: 1); e.setTexture(fg, index: 2)
            e.setTexture(disp, index: 3); e.setTexture(off, index: 4)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
            needsClear = false
        }
        var u = StillU(gain: SIMD4<Float>(gain.x, gain.y, gain.z, 1), imageSize: imageSize, learn: learn ? 1 : 0)
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(classifyPSO)
            e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(mask, index: 2); e.setTexture(yolo, index: 3)
            e.setTexture(plate, index: 4); e.setTexture(last, index: 5); e.setTexture(cls, index: 6)
            e.setBytes(&u, length: MemoryLayout<StillU>.stride, index: 0)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(resolvePSO)
            e.setTexture(cls, index: 0); e.setTexture(last, index: 1); e.setTexture(fg, index: 2); e.setTexture(fgRaw, index: 3)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(commitPSO)
            e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(fgRaw, index: 2); e.setTexture(cls, index: 3)
            e.setTexture(last, index: 4); e.setTexture(plate, index: 5); e.setTexture(fg, index: 6)
            e.setTexture(disp, index: 7); e.setTexture(off, index: 8)
            e.setBytes(&u, length: MemoryLayout<StillU>.stride, index: 0)
            e.setBuffer(statsBuf, offset: 0, index: 1)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        if let b = cb.makeBlitCommandEncoder() { b.generateMipmaps(for: off); b.endEncoding() }
    }
}
