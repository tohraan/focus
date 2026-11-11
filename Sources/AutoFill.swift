import Metal
import CoreML
import simd

/// Background inpainting of never-observed regions behind people. Runs MI-GAN (was LaMa) on its own queue only when a hole
/// exists, writes the result into the world-anchored memory (flagged generated), never blocks display.
final class AutoFill {
    private let device: MTLDevice, queue: MTLCommandQueue
    private let prepPSO: MTLComputePipelineState, writePSO: MTLComputePipelineState
    private let imgBuf: MTLBuffer, maskBuf: MTLBuffer, countBuf: MTLBuffer
    private let holeTex: MTLTexture, genTex: MTLTexture
    private let q = DispatchQueue(label: "focus.autofill", qos: .utility)
    private var model: MLModel?
    private let busyBox = Guarded(false), lastRunBox = Guarded(0.0)       // touched by the render thread and the fill queue
    private var busy: Bool { get { busyBox.value } set { busyBox.value = newValue } }
    private var lastRun: Double { get { lastRunBox.value } set { lastRunBox.value = newValue } }
    var minInterval = 1.0
    var enabled = true
    func resetTiming() { lastRun = -1e9; busy = false }      // new clip in the offline bench
    var offlineClock: Double?      // video-file tests: run inline, on video time, so results are deterministic
    var modelReady: Bool { model != nil }
    var onStatus: ((String) -> Void)?
    private var fills = 0
    private let N = 512

    init?(device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary) {
        self.device = device; self.queue = queue
        guard let a = library.makeFunction(name: "prepMigan"), let b = library.makeFunction(name: "writeGenerated"),
              let pa = try? device.makeComputePipelineState(function: a),
              let pb = try? device.makeComputePipelineState(function: b),
              let i = device.makeBuffer(length: 3 * 512 * 512 * 4, options: .storageModeShared),
              let m = device.makeBuffer(length: 512 * 512 * 4, options: .storageModeShared),
              let c = device.makeBuffer(length: 16, options: .storageModeShared) else { return nil }
        prepPSO = pa; writePSO = pb; imgBuf = i; maskBuf = m; countBuf = c
        let hd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 512, height: 512, mipmapped: false)
        hd.usage = [.shaderRead, .shaderWrite]; hd.storageMode = .private
        holeTex = device.makeTexture(descriptor: hd)!
        let gd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 512, height: 512, mipmapped: false)
        gd.storageMode = .shared
        genTex = device.makeTexture(descriptor: gd)!
        q.async { [weak self] in   // preload so the first fill doesn't stall
            guard let self, let url = Bundle.main.url(forResource: "MIGAN512", withExtension: "mlmodelc") else { return }
            let cfg = MLModelConfiguration(); cfg.computeUnits = .cpuAndGPU     // MI-GAN gives wrong pixels on the Neural Engine (measured)
            self.model = try? MLModel(contentsOf: url, configuration: cfg)
        }
    }

    struct Frame {
        var y: MTLTexture, c: MTLTexture, mask: MTLTexture, yolo: MTLTexture
        var keep: [AnyObject]
        var camToWorld: simd_float3x3
        var fx: Float, fy: Float, cx: Float, cy: Float
        var imageSize: SIMD2<Float>
    }

    func request(_ f: Frame, pano: MTLTexture, gain: SIMD4<Float>, stillCurrent: @escaping (MTLTexture) -> Bool) {
        let now = offlineClock ?? ProcessInfo.processInfo.systemUptime
        guard enabled, !busy, now - lastRun > minInterval, model != nil else { return }
        busy = true; lastRun = now
        let run: (@escaping () -> Void) -> Void = { [self] work in offlineClock != nil ? q.sync(execute: work) : q.async(execute: work) }
        run { [self] in
            defer { busy = false; lastRun = offlineClock ?? ProcessInfo.processInfo.systemUptime }
            let t0 = ProcessInfo.processInfo.systemUptime
            memset(countBuf.contents(), 0, 16)
            var pu = PanoUniforms(camToWorld: f.camToWorld, fx: f.fx, fy: f.fy, cx: f.cx, cy: f.cy, imageSize: f.imageSize)
            var g = gain
            guard let cb = queue.makeCommandBuffer(), let e = cb.makeComputeCommandEncoder() else { return }
            e.setComputePipelineState(prepPSO)
            e.setTexture(f.y, index: 0); e.setTexture(f.c, index: 1); e.setTexture(f.mask, index: 2)
            e.setTexture(f.yolo, index: 3); e.setTexture(pano, index: 4); e.setTexture(holeTex, index: 5)
            e.setBuffer(imgBuf, offset: 0, index: 0); e.setBuffer(maskBuf, offset: 0, index: 1)
            e.setBytes(&pu, length: MemoryLayout<PanoUniforms>.stride, index: 2)
            e.setBytes(&g, length: MemoryLayout<SIMD4<Float>>.stride, index: 3)
            e.setBuffer(countBuf, offset: 0, index: 4)
            e.dispatchThreads(MTLSize(width: N, height: N, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding(); cb.commit(); cb.waitUntilCompleted()
            let holes = countBuf.contents().load(as: UInt32.self)
            guard holes > 3000, let model else { return }       // < ~1% of frame: nothing worth filling

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
            } catch { onStatus?("autofill error"); return }

            guard stillCurrent(pano), let cb2 = queue.makeCommandBuffer(), let e2 = cb2.makeComputeCommandEncoder() else { return }
            e2.setComputePipelineState(writePSO)
            e2.setTexture(pano, index: 0); e2.setTexture(holeTex, index: 1); e2.setTexture(genTex, index: 2)
            e2.setBytes(&pu, length: MemoryLayout<PanoUniforms>.stride, index: 0)
            e2.dispatchThreads(MTLSize(width: pano.width, height: pano.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
            fills += 1
            _ = f.keep
            onStatus?(String(format: "autofill: %d fills · last %.0f ms", fills, (ProcessInfo.processInfo.systemUptime - t0) * 1000))
        }
    }
}
