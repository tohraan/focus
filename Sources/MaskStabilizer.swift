import Metal
import simd

/// Temporal smoothing of the person masks (Vision matte + YOLO extras), in image space.
/// A person pixel turns on at once (no leak) but turns off slowly, so single-frame drop-outs at the edge no longer flicker.
/// While the camera or the person moves fast the release is quicker so no trail is left behind.
final class MaskStabilizer {
    private let device: MTLDevice
    private let pso: MTLComputePipelineState, clearPSO: MTLComputePipelineState
    private var w = 0, h = 0
    private var maskTex: [MTLTexture] = [], yoloTex: [MTLTexture] = []
    private var cur = 0
    private var needsClear = true
    private(set) var mask: MTLTexture?, yolo: MTLTexture?

    /// Per-frame blend factors at 30 fps; converted to the real frame time.
    var attack30: Float = 0.9
    var releaseStill30: Float = 0.25       // camera still: person pixels fade over about 4 frames
    var yoloHoldStill30: Float = 0.06      // YOLO channel (far / missed people) on a still camera: ~0.5 s hold, so a missed frame never flickers
    var yoloHoldMoving30: Float = 0.35
    var releaseMoving30: Float = 0.6       // camera moving: about 2 frames

    init?(device: MTLDevice, library: MTLLibrary) {
        self.device = device
        guard let a = library.makeFunction(name: "maskStabilize"), let b = library.makeFunction(name: "stabClear"),
              let pa = try? device.makeComputePipelineState(function: a), let pb = try? device.makeComputePipelineState(function: b) else { return nil }
        pso = pa; clearPSO = pb
    }

    func reset() { needsClear = true }

    private func make(_ fmt: MTLPixelFormat) -> MTLTexture {
        let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: fmt, width: w, height: h, mipmapped: false)
        td.usage = [.shaderRead, .shaderWrite]; td.storageMode = .private
        return device.makeTexture(descriptor: td)!
    }

    /// Encodes one step; afterwards `mask` / `yolo` hold the stabilised masks to use in place of the raw ones.
    func encode(cb: MTLCommandBuffer, rawMask: MTLTexture, rawYolo: MTLTexture, imageSize: SIMD2<Float>, dt: Double, moving: Bool) {
        let nw = max(64, Int(imageSize.x / 4)), nh = max(64, Int(imageSize.y / 4))
        if nw != w || nh != h || maskTex.isEmpty {
            w = nw; h = nh
            maskTex = [make(.r8Unorm), make(.r8Unorm)]; yoloTex = [make(.rg8Unorm), make(.rg8Unorm)]
            needsClear = true
        }
        let grid = MTLSize(width: w, height: h, depth: 1), tg = MTLSize(width: 16, height: 16, depth: 1)
        if needsClear {
            for i in 0..<2 {
                if let e = cb.makeComputeCommandEncoder() {
                    e.setComputePipelineState(clearPSO)
                    e.setTexture(maskTex[i], index: 0); e.setTexture(yoloTex[i], index: 1)
                    e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
                }
            }
            needsClear = false
        }
        let steps = Float(min(max(dt, 1.0 / 120), 0.2) * 30)           // how many 30 fps frames this step covers
        func rate(_ r30: Float) -> Float { 1 - pow(1 - r30, steps) }
        var rates = SIMD4<Float>(rate(attack30), rate(moving ? releaseMoving30 : releaseStill30), rate(moving ? yoloHoldMoving30 : yoloHoldStill30), 0)
        let prev = cur, next = 1 - cur
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(pso)
            e.setTexture(rawMask, index: 0); e.setTexture(rawYolo, index: 1)
            e.setTexture(maskTex[prev], index: 2); e.setTexture(yoloTex[prev], index: 3)
            e.setTexture(maskTex[next], index: 4); e.setTexture(yoloTex[next], index: 5)
            e.setBytes(&rates, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            e.dispatchThreads(grid, threadsPerThreadgroup: tg); e.endEncoding()
        }
        cur = next
        mask = maskTex[next]; yolo = yoloTex[next]
    }
}
