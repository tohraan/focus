import Metal
import simd

struct KFUniforms {
    var H: (simd_float3x3, simd_float3x3, simd_float3x3, simd_float3x3) =
        (matrix_identity_float3x3, matrix_identity_float3x3, matrix_identity_float3x3, matrix_identity_float3x3)
    var viewDist = SIMD4<Float>(repeating: 1e9)
    var slice = SIMD4<Int32>(repeating: -1)
    var curSize = SIMD2<Float>(960, 720)
    var conf: Float = 1
    var pad: Float = 0
}

struct PlaneUniforms {
    var H: (simd_float3x3, simd_float3x3, simd_float3x3, simd_float3x3, simd_float3x3, simd_float3x3) =
        (matrix_identity_float3x3, matrix_identity_float3x3, matrix_identity_float3x3,
         matrix_identity_float3x3, matrix_identity_float3x3, matrix_identity_float3x3)
    var slice: Int32 = 0
    var curSize = SIMD2<Float>(960, 720)
}

/// R2 engine: sharp pose-tagged keyframes (each with its own dilated person mask), reprojected into the current view
/// through a plane-induced homography whose depth is found by a GPU plane sweep. Seams/exposure hidden by diffusing
/// (live - warped) from the background ring into the hole. See docs/06-v3-plan.md.
final class KeyframeFill {
    static let W = 960, H = 720, N = 24
    static let sweepDepths: [Float] = [0.5, 0.8, 1.2, 2.0, 3.5, 6.0]

    let colorArr: MTLTexture, maskArr: MTLTexture, diffTex: MTLTexture
    private let storePSO: MTLComputePipelineState, planePSO: MTLComputePipelineState, diffPSO: MTLComputePipelineState
    private let statsBuf: MTLBuffer
    private var poses = [simd_float4x4?](repeating: nil, count: KeyframeFill.N)
    private var Ks = [simd_float3x3](repeating: matrix_identity_float3x3, count: KeyframeFill.N)
    private var stamps = [Double](repeating: 0, count: KeyframeFill.N)
    private(set) var count = 0
    private var lastInsertT = 0.0, lastSweepT = 0.0, lastT = 0.0
    private var lastPose: simd_float4x4?
    private var sweepPending = false
    private(set) var depthLog = log(Float(2.0))
    private(set) var conf: Float = 1
    private(set) var uniforms = KFUniforms()
    var status: String { String(format: "kf %d · D %.1f m · %@", count, exp(depthLog), conf > 0.5 ? "warp ok" : "soft") }

    init?(device: MTLDevice, library: MTLLibrary) {
        guard let a = library.makeFunction(name: "storeKeyframe"), let b = library.makeFunction(name: "planeCost"),
              let c = library.makeFunction(name: "diffPrep"),
              let pa = try? device.makeComputePipelineState(function: a), let pb = try? device.makeComputePipelineState(function: b),
              let pc = try? device.makeComputePipelineState(function: c),
              let sb = device.makeBuffer(length: 64, options: .storageModeShared) else { return nil }
        storePSO = pa; planePSO = pb; diffPSO = pc; statsBuf = sb
        let cd = MTLTextureDescriptor()
        cd.textureType = .type2DArray; cd.pixelFormat = .rgba8Unorm; cd.width = Self.W; cd.height = Self.H
        cd.arrayLength = Self.N; cd.usage = [.shaderRead, .shaderWrite]; cd.storageMode = .private
        let md = MTLTextureDescriptor()
        md.textureType = .type2DArray; md.pixelFormat = .r8Unorm; md.width = Self.W; md.height = Self.H
        md.arrayLength = Self.N; md.usage = [.shaderRead, .shaderWrite]; md.storageMode = .private
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 240, height: 180, mipmapped: true)
        dd.usage = [.shaderRead, .shaderWrite]; dd.storageMode = .private
        guard let ca = device.makeTexture(descriptor: cd), let ma = device.makeTexture(descriptor: md),
              let dt = device.makeTexture(descriptor: dd) else { return nil }
        colorArr = ca; maskArr = ma; diffTex = dt
    }

    func reset() {
        poses = [simd_float4x4?](repeating: nil, count: Self.N); count = 0; lastPose = nil
        uniforms = KFUniforms(); sweepPending = false
    }

    /// Encodes keyframe insertion, the plane sweep and the difference pass for this frame; fills `uniforms` for the composite.
    func encode(cb: MTLCommandBuffer, y: MTLTexture, c: MTLTexture, mask: MTLTexture, yolo: MTLTexture,
                camToWorld: simd_float4x4, K: simd_float3x3, imageSize: SIMD2<Float>, tracking: Bool, time: Double, still: Bool = false) {
        let sx = Float(Self.W) / imageSize.x, sy = Float(Self.H) / imageSize.y
        let Kw = Reproject.K(fx: K[0][0] * sx, fy: K[1][1] * sy, cx: K[2][0] * sx, cy: K[2][1] * sy)
        guard tracking else { uniforms = KFUniforms(); uniforms.conf = 0; lastPose = nil; return }   // never use/learn while not "normal"

        // camera angular speed -> skip blurred frames
        var speed: Float = 0
        if let lp = lastPose, time > lastT { speed = Reproject.angleDeg(lp, camToWorld) / Float(time - lastT) }
        lastPose = camToWorld; lastT = time

        // insertion
        var inserted = -1
        var minVD = Float.greatestFiniteMagnitude
        for i in 0..<Self.N { if let p = poses[i] { minVD = min(minVD, Reproject.viewDistance(p, camToWorld)) } }
        if speed < 60, time - lastInsertT > 0.3, count == 0 || minVD > 8 || (still && time - lastInsertT > 2.0) {   // a still camera refreshes by age: never one stale view
            var slot = count
            if count >= Self.N {                       // evict the keyframe most redundant with a neighbour (older first)
                var bestScore = Float.greatestFiniteMagnitude
                slot = 0
                for i in 0..<Self.N {
                    var nn = Float.greatestFiniteMagnitude
                    for j in 0..<Self.N where j != i { if let a = poses[i], let b = poses[j] { nn = min(nn, Reproject.viewDistance(a, b)) } }
                    let score = nn + Float(stamps[i] * 1e-6)
                    if score < bestScore { bestScore = score; slot = i }
                }
            } else { count += 1 }
            if let e = cb.makeComputeCommandEncoder() {
                var s = UInt32(slot)
                e.setComputePipelineState(storePSO)
                e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(mask, index: 2); e.setTexture(yolo, index: 3)
                e.setTexture(colorArr, index: 4); e.setTexture(maskArr, index: 5)
                e.setBytes(&s, length: 4, index: 0)
                e.dispatchThreads(MTLSize(width: Self.W, height: Self.H, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                e.endEncoding()
            }
            poses[slot] = camToWorld; Ks[slot] = Kw; stamps[slot] = time; lastInsertT = time; inserted = slot
        }

        // candidates: nearest viewpoints (excluding the one just stored from this very frame)
        var cands: [(Int, Float)] = []
        for i in 0..<Self.N where i != inserted { if let p = poses[i] { let d = Reproject.viewDistance(p, camToWorld); if d < 60 { cands.append((i, d)) } } }
        cands.sort { $0.1 < $1.1 }
        let D = exp(depthLog)
        var u = KFUniforms()
        u.conf = conf
        var hs: [simd_float3x3] = []
        for (n, cnd) in cands.prefix(4).enumerated() {
            hs.append(Reproject.homography(curToWorld: camToWorld, keyToWorld: poses[cnd.0]!, Kcur: Kw, Kkey: Ks[cnd.0], D: D))
            u.viewDist[n] = cnd.1; u.slice[n] = Int32(cnd.0)
        }
        while hs.count < 4 { hs.append(matrix_identity_float3x3) }
        u.H = (hs[0], hs[1], hs[2], hs[3])
        uniforms = u
        guard let best = cands.first else { return }

        // plane sweep on the best keyframe (~10 Hz)
        if !sweepPending, time - lastSweepT > 0.1, let e = cb.makeComputeCommandEncoder() {
            memset(statsBuf.contents(), 0, 64)
            var pu = PlaneUniforms()
            let H6 = Self.sweepDepths.map { Reproject.homography(curToWorld: camToWorld, keyToWorld: poses[best.0]!, Kcur: Kw, Kkey: Ks[best.0], D: $0) }
            pu.H = (H6[0], H6[1], H6[2], H6[3], H6[4], H6[5]); pu.slice = Int32(best.0)
            e.setComputePipelineState(planePSO)
            e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(mask, index: 2); e.setTexture(yolo, index: 3)
            e.setTexture(colorArr, index: 4); e.setTexture(maskArr, index: 5)
            e.setBytes(&pu, length: MemoryLayout<PlaneUniforms>.stride, index: 0)
            e.setBuffer(statsBuf, offset: 0, index: 1)
            e.dispatchThreads(MTLSize(width: 64, height: 48, depth: 6), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            e.endEncoding()
            sweepPending = true; lastSweepT = time
        }

        // seam / exposure diffusion image
        if let e = cb.makeComputeCommandEncoder() {
            e.setComputePipelineState(diffPSO)
            e.setTexture(y, index: 0); e.setTexture(c, index: 1); e.setTexture(mask, index: 2); e.setTexture(yolo, index: 3)
            e.setTexture(colorArr, index: 4); e.setTexture(maskArr, index: 5); e.setTexture(diffTex, index: 6)
            var uu = uniforms
            e.setBytes(&uu, length: MemoryLayout<KFUniforms>.stride, index: 0)
            e.dispatchThreads(MTLSize(width: 240, height: 180, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            e.endEncoding()
        }
        if let b = cb.makeBlitCommandEncoder() { b.generateMipmaps(for: diffTex); b.endEncoding() }
    }

    /// Call from the command buffer's completion handler.
    func finishSweep() {
        guard sweepPending else { return }
        sweepPending = false
        let st = statsBuf.contents().assumingMemoryBound(to: UInt32.self)
        var best = -1; var bestCost = Float.greatestFiniteMagnitude
        for i in 0..<6 where st[8 + i] >= 12 {
            let c = Float(st[i]) / 1000 / Float(st[8 + i])
            if c < bestCost { bestCost = c; best = i }
        }
        guard best >= 0 else { return }
        depthLog += 0.25 * (log(Self.sweepDepths[best]) - depthLog)
        conf += 0.3 * ((bestCost < 0.12 ? 1 : 0) - conf)       // photometric gate: bad alignment -> soft fill, never torn patches
    }
}
