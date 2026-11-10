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
    private var busy = false
    private var lastRun = 0.0
    var minInterval = 0.6
    var enabled = true
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

    func request(_ f: Frame, plate: StillPlate) {
        let now = offlineClock ?? ProcessInfo.processInfo.systemUptime
        guard enabled, !busy, now - lastRun > minInterval, model != nil,
              let plateTex = plate.plate, let disp = plate.disp, let fg = plate.fg else { return }
        busy = true; lastRun = now
        let gen0 = plate.generation
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
            e.dispatchThreads(MTLSize(width: N, height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let holes = countBuf.contents().load(as: UInt32.self)
            guard holes > 400, let model else { return }               // under 0.15% of the frame: nothing worth generating

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
                genTex.replace(region: MTLRegionMake2D(0, 0, N, N), mipmapLevel: 0, withBytes: rgba, bytesPerRow: N * 4)
            } catch { onStatus?("fill error"); return }

            // the camera may have moved while we were generating: then the plate was reset and this result is stale
            guard plate.generation == gen0, let cb2 = queue.makeCommandBuffer(), let e2 = cb2.makeComputeCommandEncoder() else { return }
            var g = gain
            e2.setComputePipelineState(commitPSO)
            e2.setTexture(holeTex, index: 0); e2.setTexture(genTex, index: 1); e2.setTexture(plateTex, index: 2)
            e2.setBytes(&g, length: MemoryLayout<SIMD4<Float>>.stride, index: 0)
            e2.dispatchThreads(MTLSize(width: plateTex.width, height: plateTex.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
            fills += 1
            _ = f.keep
            onStatus?(String(format: "generated fill: %d · %.0f ms · %d px", fills, (ProcessInfo.processInfo.systemUptime - t0) * 1000, holes))
        }
        if offlineClock != nil { q.sync(execute: work) } else { q.async(execute: work) }
    }
}
