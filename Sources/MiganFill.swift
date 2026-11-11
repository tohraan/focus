import Metal
import CoreML
import simd

/// Generative fill (MI-GAN 512, a phone-sized inpainting network) for background that the still plate has never seen,
/// e.g. behind someone who stood in one place from the first frame. Runs on its own queue, never blocks display,
/// writes the result into the plate with confidence 0.51 ("generated"); real observations replace it.
/// Core ML runs on CPU+GPU: the Neural Engine produces wrong pixels for this model (measured).
final class MiganFill {
    private let device: MTLDevice, queue: MTLCommandQueue
    private let prepPSO: MTLComputePipelineState, commitPSO: MTLComputePipelineState
    private let imgBuf: MTLBuffer, maskBuf: MTLBuffer, countBuf: MTLBuffer
    private let holeTex: MTLTexture, genTex: MTLTexture
    private let q = DispatchQueue(label: "focus.migan", qos: .utility)
    private var model: MLModel?
    private let busyBox = Guarded(false), lastRunBox = Guarded(0.0)       // touched by the render thread and the fill queue
    private var busy: Bool { get { busyBox.value } set { busyBox.value = newValue } }
    private var lastRun: Double { get { lastRunBox.value } set { lastRunBox.value = newValue } }
    var minInterval = 0.6
    var enabled = true
    func resetTiming() { lastRun = -1e9; busy = false }      // new clip in the offline bench
    var offlineClock: Double?          // video-file tests: run inline on video time
    var modelReady: Bool { model != nil }
    var onStatus: ((String) -> Void)?
    private var fills = 0
    private let N = 512

    struct Frame { var y: MTLTexture, c: MTLTexture; var keep: [AnyObject] }

    init?(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) {
        self.device = device; self.queue = queue
        guard let a = library.makeFunction(name: "miganPrep"), let b = library.makeFunction(name: "miganCommit"),
              let pa = try? device.makeComputePipelineState(function: a), let pb = try? device.makeComputePipelineState(function: b),
              let i = device.makeBuffer(length: 3 * 512 * 512 * 4, options: .storageModeShared),
              let m = device.makeBuffer(length: 512 * 512 * 4, options: .storageModeShared),
              let c = device.makeBuffer(length: 16, options: .storageModeShared) else { return nil }
        prepPSO = pa; commitPSO = pb; imgBuf = i; maskBuf = m; countBuf = c
        let hd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 512, height: 512, mipmapped: false)
        hd.usage = [.shaderRead, .shaderWrite]; hd.storageMode = .private
        holeTex = device.makeTexture(descriptor: hd)!
        let gd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 512, height: 512, mipmapped: false)
        gd.storageMode = .shared
        genTex = device.makeTexture(descriptor: gd)!
        q.async { [weak self] in
            guard let self, let url = Bundle.main.url(forResource: "MIGAN512", withExtension: "mlmodelc") else { return }
            let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU
            self.model = try? MLModel(contentsOf: url, configuration: cfg)
        }
    }

    /// Smooth interpolation of sparse samples (push-pull pyramid): `val` (3 channels) known where `w` > 0, filled everywhere else.
    private func pushPull(_ val: [Float], _ w: [Float], _ n: Int) -> [Float] {
        var vs = [val], ws = [w], sizes = [n]
        var cur = n
        while cur > 2 {
            let nx = cur / 2
            var v2 = [Float](repeating: 0, count: nx * nx * 3), w2 = [Float](repeating: 0, count: nx * nx)
            let pv = vs[vs.count - 1], pw = ws[ws.count - 1]
            for y in 0..<nx { for x in 0..<nx {
                var sw: Float = 0, a: Float = 0, b: Float = 0, c: Float = 0
                for dy in 0..<2 { for dx in 0..<2 {
                    let j = (2 * y + dy) * cur + (2 * x + dx), wj = pw[j]
                    sw += wj; a += wj * pv[j * 3]; b += wj * pv[j * 3 + 1]; c += wj * pv[j * 3 + 2]
                } }
                if sw > 0 { let i = y * nx + x; w2[i] = min(1, sw); v2[i * 3] = a / sw; v2[i * 3 + 1] = b / sw; v2[i * 3 + 2] = c / sw }
            } }
            vs.append(v2); ws.append(w2); sizes.append(nx); cur = nx
        }
        var f = vs[vs.count - 1]
        for l in stride(from: vs.count - 2, through: 0, by: -1) {
            let sz = sizes[l], sc = sizes[l + 1]
            var out = [Float](repeating: 0, count: sz * sz * 3)
            for y in 0..<sz { for x in 0..<sz {
                let fx = (Float(x) + 0.5) / 2 - 0.5, fy = (Float(y) + 0.5) / 2 - 0.5
                let x0 = max(0, min(sc - 1, Int(floor(fx)))), y0 = max(0, min(sc - 1, Int(floor(fy))))
                let x1 = min(sc - 1, x0 + 1), y1 = min(sc - 1, y0 + 1)
                let tx = max(0, min(1, fx - Float(x0))), ty = max(0, min(1, fy - Float(y0)))
                let i = y * sz + x, wl = ws[l][i]
                for c in 0..<3 {
                    let top = f[(y0 * sc + x0) * 3 + c] * (1 - tx) + f[(y0 * sc + x1) * 3 + c] * tx
                    let bot = f[(y1 * sc + x0) * 3 + c] * (1 - tx) + f[(y1 * sc + x1) * 3 + c] * tx
                    out[i * 3 + c] = wl * vs[l][i * 3 + c] + (1 - wl) * (top * (1 - ty) + bot * ty)
                }
            } }
            f = out
        }
        return f
    }

    /// The generator's tone can drift (a huge hole comes out flat grey). Measure the colour mismatch along the hole border
    /// (generated pixel vs the known pixels beside it) and spread it smoothly into the hole: seamless border, generated detail kept.
    private func seamCorrect(_ rgba: inout [UInt8]) {
        let n = N, nn = N * N
        let mk = maskBuf.contents().assumingMemoryBound(to: Float.self), im = imgBuf.contents().assumingMemoryBound(to: Float.self)
        var val = [Float](repeating: 0, count: nn * 3), w = [Float](repeating: 0, count: nn)
        var count = 0
        for y in 3..<(n - 3) { for x in 3..<(n - 3) {
            let i = y * n + x
            guard mk[i] < 127, mk[i - 1] >= 127 || mk[i + 1] >= 127 || mk[i - n] >= 127 || mk[i + n] >= 127 else { continue }
            var s0: Float = 0, s1: Float = 0, s2: Float = 0, c: Float = 0
            for dy in -3...3 { for dx in -3...3 {
                let j = (y + dy) * n + (x + dx)
                if mk[j] >= 127 { s0 += im[j]; s1 += im[nn + j]; s2 += im[2 * nn + j]; c += 1 }
            } }
            if c >= 4 {
                val[i * 3] = s0 / c - Float(rgba[i * 4]); val[i * 3 + 1] = s1 / c - Float(rgba[i * 4 + 1]); val[i * 3 + 2] = s2 / c - Float(rgba[i * 4 + 2])
                w[i] = 1; count += 1
            }
        } }
        guard count > 40 else { return }
        let f = pushPull(val, w, n)
        for i in 0..<nn where mk[i] < 127 {
            for c in 0..<3 { rgba[i * 4 + c] = UInt8(max(0, min(255, Float(rgba[i * 4 + c]) + max(-110, min(110, f[i * 3 + c]))))) }
        }
    }

    func request(_ f: Frame, plate: StillPlate) {
        let now = offlineClock ?? ProcessInfo.processInfo.systemUptime
        guard enabled, !busy, now - lastRun > minInterval, model != nil,
              let plateTex = plate.plate, let disp = plate.disp, let fg = plate.fg else { return }
        busy = true; lastRun = now
        let gen0 = plate.generation
        var canvas = plate.lastCanvas                                    // mapping at request time; the plate is world-anchored so the result stays valid
        let gain = SIMD4<Float>(plate.gain.x, plate.gain.y, plate.gain.z, 1)
        let work = { [self] in
            defer { busy = false; lastRun = offlineClock ?? ProcessInfo.processInfo.systemUptime }
            let t0 = ProcessInfo.processInfo.systemUptime
            memset(countBuf.contents(), 0, 16)
            guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return }
            e.setComputePipelineState(prepPSO)
            e.setTexture(f.y, index: 0); e.setTexture(f.c, index: 1); e.setTexture(disp, index: 2); e.setTexture(fg, index: 3)
            e.setTexture(holeTex, index: 4)
            e.setBuffer(imgBuf, offset: 0, index: 0); e.setBuffer(maskBuf, offset: 0, index: 1); e.setBuffer(countBuf, offset: 0, index: 2)
            e.setBytes(&canvas, length: MemoryLayout<CanvasUniforms>.stride, index: 3)
            e.dispatchThreads(MTLSize(width: N, height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let holes = countBuf.contents().load(as: UInt32.self)
            guard holes > 400, holes < UInt32(N * N * 2 / 5), let model else { return }   // under 0.15%: not worth it; over 40%: too little context, the generator invents scenery

            do {
                let img = try MLMultiArray(dataPointer: imgBuf.contents(), shape: [1, 3, 512, 512], dataType: .float32,
                                           strides: [786432, 262144, 512, 1], deallocator: nil)
                let msk = try MLMultiArray(dataPointer: maskBuf.contents(), shape: [1, 1, 512, 512], dataType: .float32,
                                           strides: [262144, 262144, 512, 1], deallocator: nil)
                let inp = try MLDictionaryFeatureProvider(dictionary: ["image": MLFeatureValue(multiArray: img),
                                                                        "mask": MLFeatureValue(multiArray: msk)])
                let out = try model.prediction(from: inp)
                guard let arr = out.featureValue(for: "result")?.multiArrayValue else { return }
                var rgba = [UInt8](repeating: 255, count: N * N * 4)
                let st = arr.strides.map { $0.intValue }
                func px(_ c: Int, _ y: Int, _ x: Int) -> Float {
                    let i = c * st[1] + y * st[2] + x * st[3]
                    return arr.dataType == .float16 ? Float(arr.dataPointer.assumingMemoryBound(to: Float16.self)[i])
                                                    : arr.dataPointer.assumingMemoryBound(to: Float.self)[i]
                }
                for y in 0..<N { for x in 0..<N {
                    let o = (y * N + x) * 4
                    rgba[o] = UInt8(max(0, min(255, px(0, y, x))))
                    rgba[o + 1] = UInt8(max(0, min(255, px(1, y, x))))
                    rgba[o + 2] = UInt8(max(0, min(255, px(2, y, x))))
                } }
                seamCorrect(&rgba)
                genTex.replace(region: MTLRegionMake2D(0, 0, N, N), mipmapLevel: 0, withBytes: rgba, bytesPerRow: N * 4)
            } catch { onStatus?("fill error"); return }

            // the camera may have moved while we were generating: then the plate was reset and this result is stale
            guard plate.generation == gen0, let cb2 = queue.makeCommandBuffer(), let e2 = cb2.makeComputeCommandEncoder() else { return }
            var g = gain
            e2.setComputePipelineState(commitPSO)
            e2.setTexture(holeTex, index: 0); e2.setTexture(genTex, index: 1); e2.setTexture(plateTex, index: 2)
            e2.setBytes(&g, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            e2.setBytes(&canvas, length: MemoryLayout<CanvasUniforms>.stride, index: 1)
            e2.dispatchThreads(MTLSize(width: plateTex.width, height: plateTex.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
            fills += 1
            _ = f.keep
            onStatus?(String(format: "generated fill: %d · %.0f ms · %d px", fills, (ProcessInfo.processInfo.systemUptime - t0) * 1000, holes))
        }
        if offlineClock != nil { q.sync(execute: work) } else { q.async(execute: work) }
    }
}
