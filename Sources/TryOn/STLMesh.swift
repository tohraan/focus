import Foundation
import SceneKit
import simd

/// Binary-STL loader (millimetres, CAD axes) -> SCNGeometry in ARKit face space (metres, +Y up, +Z out of the face).
/// CAD axes: X across, Y backwards from the front face, Z up.  ARKit: (x, y, z) = (X, Z, -Y) * 0.001.
enum STLMesh {
    static func load(url: URL, crease: Float = 35) -> SCNGeometry? {
        guard let d = try? Data(contentsOf: url, options: .mappedIfSafe), d.count > 84 else { return nil }
        let n = Int(d.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: 80, as: UInt32.self) })
        guard d.count >= 84 + n * 50 else { return nil }
        var tri = [SIMD3<Float>](); tri.reserveCapacity(n * 3)
        d.withUnsafeBytes { p in
            for i in 0..<n {
                let o = 84 + i * 50 + 12
                for k in 0..<3 {
                    let x = p.loadUnaligned(fromByteOffset: o + k * 12, as: Float.self)
                    let y = p.loadUnaligned(fromByteOffset: o + k * 12 + 4, as: Float.self)
                    let z = p.loadUnaligned(fromByteOffset: o + k * 12 + 8, as: Float.self)
                    tri.append(SIMD3(x, z, -y) * 0.001)
                }
            }
        }
        return geometry(triangles: tri, crease: crease)
    }

    /// Smooth normals inside `crease` degrees, hard edges beyond it.
    static func geometry(triangles tri: [SIMD3<Float>], crease: Float) -> SCNGeometry {
        let n = tri.count / 3
        var fn = [SIMD3<Float>](repeating: .zero, count: n)
        for i in 0..<n {
            let c = simd_cross(tri[i*3+1] - tri[i*3], tri[i*3+2] - tri[i*3])   // area-weighted
            fn[i] = c
        }
        struct Key: Hashable { let x: Int32, y: Int32, z: Int32 }
        func key(_ v: SIMD3<Float>) -> Key { Key(x: Int32((v.x * 20000).rounded()), y: Int32((v.y * 20000).rounded()), z: Int32((v.z * 20000).rounded())) }
        var groups = [Key: [Int]](); groups.reserveCapacity(n)
        for i in 0..<n { for k in 0..<3 { groups[key(tri[i*3+k]), default: []].append(i) } }
        let cs = cos(crease * .pi / 180)
        var nor = [SIMD3<Float>](repeating: .zero, count: n * 3)
        for i in 0..<n {
            let mine = simd_normalize(fn[i] + SIMD3(repeating: 1e-20))
            for k in 0..<3 {
                var acc = SIMD3<Float>.zero
                for j in groups[key(tri[i*3+k])]! {
                    let o = simd_normalize(fn[j] + SIMD3(repeating: 1e-20))
                    if simd_dot(o, mine) >= cs { acc += fn[j] }
                }
                nor[i*3+k] = simd_normalize(acc + SIMD3(repeating: 1e-20))
            }
        }
        let vs = SCNGeometrySource(vertices: tri.map { SCNVector3($0.x, $0.y, $0.z) })
        let ns = SCNGeometrySource(normals: nor.map { SCNVector3($0.x, $0.y, $0.z) })
        let idx = (0..<Int32(n * 3)).map { $0 }
        let el = SCNGeometryElement(indices: idx, primitiveType: .triangles)
        return SCNGeometry(sources: [vs, ns], elements: [el])
    }
}
