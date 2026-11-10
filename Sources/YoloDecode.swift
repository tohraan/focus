import CoreML
import Foundation

/// Decodes YOLO11-seg raw outputs into a person-union mask at prototype resolution.
/// det:    [1, 116, N]  rows: cx,cy,w,h (input px), 80 class scores (person = 0), 32 mask coeffs
/// protos: [1, 32, H/4, W/4]
enum YoloDecode {
    struct Result { var mask: [UInt8]; var w: Int; var h: Int; var count: Int; var scores: [Float] }  // mask interleaved RG: R = people Vision missed (display), G = all detections (memory veto)

    /// Vision matte copied out so YOLO can judge which detections Vision already covers.
    struct VisionMask { var data: [UInt8]; var w: Int; var h: Int }

    static func decode(det: MLMultiArray, protos: MLMultiArray, inW: Float, inH: Float, vision: VisionMask? = nil,
                       scoreThr: Float = 0.2, nmsIoU: Float = 0.5, maxDet: Int = 40) -> Result {
        let C = det.shape[1].intValue, N = det.shape[2].intValue
        let ds = det.strides.map { $0.intValue }
        let PC = protos.shape[1].intValue, PH = protos.shape[2].intValue, PW = protos.shape[3].intValue
        let ps = protos.strides.map { $0.intValue }
        precondition(C == 4 + 80 + PC)

        let detRead = reader(det), protoRead = reader(protos)
        // 1. candidates
        var cand: [(score: Float, i: Int)] = []
        for i in 0..<N {
            let s = detRead(ds[1] * 4 + ds[2] * i)
            if s > scoreThr { cand.append((s, i)) }
        }
        cand.sort { $0.score > $1.score }
        // 2. NMS
        func box(_ i: Int) -> (Float, Float, Float, Float) {
            let cx = detRead(ds[2] * i), cy = detRead(ds[1] + ds[2] * i)
            let w = detRead(ds[1] * 2 + ds[2] * i), h = detRead(ds[1] * 3 + ds[2] * i)
            return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        }
        var kept: [(score: Float, i: Int, b: (Float, Float, Float, Float))] = []
        for c in cand {
            let b = box(c.i)
            var ok = true
            for k in kept {
                let ix = max(0, min(b.2, k.b.2) - max(b.0, k.b.0)), iy = max(0, min(b.3, k.b.3) - max(b.1, k.b.1))
                let inter = ix * iy
                let u = (b.2 - b.0) * (b.3 - b.1) + (k.b.2 - k.b.0) * (k.b.3 - k.b.1) - inter
                if u > 0 && inter / u > nmsIoU { ok = false; break }
            }
            if ok { kept.append((c.score, c.i, b)); if kept.count >= maxDet { break } }
        }
        // 3. masks (only inside each box). G = all detections; R = only detections Vision mostly missed.
        var mask = [UInt8](repeating: 0, count: PW * PH * 2)
        let sx = Float(PW) / inW, sy = Float(PH) / inH
        for k in kept {
            var coef = [Float](repeating: 0, count: PC)
            for c in 0..<PC { coef[c] = detRead(ds[1] * (84 + c) + ds[2] * k.i) }
            let x0 = max(0, Int(k.b.0 * sx)), x1 = min(PW - 1, Int(k.b.2 * sx) + 1)
            let y0 = max(0, Int(k.b.1 * sy)), y1 = min(PH - 1, Int(k.b.3 * sy) + 1)
            if x1 < x0 || y1 < y0 { continue }
            var pts: [Int] = []
            var covered = 0
            for y in y0...y1 {
                for x in x0...x1 {
                    var acc: Float = 0
                    let base = ps[2] * y + ps[3] * x
                    for c in 0..<PC { acc += coef[c] * protoRead(ps[1] * c + base) }
                    if acc > 0 {
                        pts.append(y * PW + x)
                        if let v = vision {
                            let vx = min(v.w - 1, x * v.w / PW), vy = min(v.h - 1, y * v.h / PH)
                            if v.data[vy * v.w + vx] > 100 { covered += 1 }
                        }
                    }
                }
            }
            let missed = vision == nil || Float(covered) < 0.35 * Float(max(pts.count, 1))
            for i in pts { mask[i * 2 + 1] = 255; if missed { mask[i * 2] = 255 } }
        }
        return Result(mask: mask, w: PW, h: PH, count: kept.count, scores: kept.map { $0.score })
    }

    private static func reader(_ a: MLMultiArray) -> (Int) -> Float {
        let p = a.dataPointer
        switch a.dataType {
        case .float16: let q = p.assumingMemoryBound(to: Float16.self); return { Float(q[$0]) }
        case .float32: let q = p.assumingMemoryBound(to: Float.self); return { q[$0] }
        default: fatalError("unsupported dtype")
        }
    }
}
