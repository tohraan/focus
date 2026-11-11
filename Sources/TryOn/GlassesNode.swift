import Foundation
import SceneKit
#if canImport(UIKit)
import UIKit
typealias PlatformColor = UIColor
#else
import AppKit
typealias PlatformColor = NSColor
#endif

/// The FOCUS v3 glasses (same CAD meshes as the launch film) as a SceneKit node in ARKit face space.
enum GlassesNode {
    private static let parts: [(String, String)] = [
        ("front_frame", "frame"), ("temple_L", "frame"), ("temple_R", "frame"),
        ("lens_L", "display"), ("lens_R", "display"),
        ("shield", "shield"), ("foam", "foam"),
        ("camera_housing_L", "ring"), ("camera_housing_R", "ring"),
        ("camera_lens_L", "camglass"), ("camera_lens_R", "camglass"),
        ("nose_pad_L", "silicone"), ("nose_pad_R", "silicone"),
        ("hinge_pin_L", "hinge"), ("hinge_pin_R", "hinge"),
        ("indicator_window_L", "led"), ("indicator_window_R", "led"),
        ("magnet_temple_L", "steel"), ("magnet_temple_R", "steel"),
    ]

    private static func mat(_ c: PlatformColor, rough: CGFloat, metal: CGFloat = 0, emission: PlatformColor? = nil, alpha: CGFloat = 1) -> SCNMaterial {
        let m = SCNMaterial()
        m.lightingModel = .physicallyBased
        m.diffuse.contents = c; m.roughness.contents = rough; m.metalness.contents = metal
        if let e = emission { m.emission.contents = e }
        m.isDoubleSided = true
        if alpha < 1 { m.transparency = alpha; m.transparencyMode = .dualLayer; m.writesToDepthBuffer = false }
        return m
    }

    static func materials() -> [String: SCNMaterial] {
        func c(_ r: CGFloat, _ g: CGFloat, _ b: CGFloat) -> PlatformColor { PlatformColor(red: r, green: g, blue: b, alpha: 1) }
        return [
            // soft matte black polymer: charcoal albedo + high roughness so it reads as plastic, not a silhouette
            "frame": mat(c(0.085, 0.085, 0.092), rough: 0.78),
            "foam": mat(c(0.06, 0.06, 0.065), rough: 0.95),
            // smoked, semi-transparent wrap-around shield and clear near-eye lenses (you can see the eyes through them)
            "shield": mat(c(0.10, 0.12, 0.14), rough: 0.25, alpha: 0.38),
            "display": mat(c(0.55, 0.70, 0.85), rough: 0.05, alpha: 0.16),
            "camglass": mat(c(0.01, 0.01, 0.03), rough: 0.05, emission: c(0.03, 0.02, 0.10)),
            "ring": mat(c(0.07, 0.07, 0.08), rough: 0.55, metal: 0.2),
            "hinge": mat(c(0.12, 0.12, 0.13), rough: 0.4, metal: 1),
            "steel": mat(c(0.62, 0.62, 0.65), rough: 0.35, metal: 1),
            "silicone": mat(c(0.78, 0.8, 0.85), rough: 0.3, alpha: 0.7),
            "led": mat(c(0.7, 0.88, 1), rough: 0.4, emission: c(0.4, 0.6, 0.8)),
        ]
    }

    /// `dir` = folder holding the .stl files.  Returns nil only if nothing loads.
    static func build(dir: URL) -> SCNNode? {
        let root = SCNNode(); root.name = "focus_glasses"
        let M = materials()
        var count = 0
        for (name, mk) in parts {
            guard let g = STLMesh.load(url: dir.appendingPathComponent(name + ".stl")) else { continue }
            g.materials = [M[mk]!]
            let n = SCNNode(geometry: g); n.name = name
            if mk == "shield" || mk == "display" || mk == "silicone" { n.renderingOrder = 20 }   // translucent parts draw last
            root.addChildNode(n); count += 1
        }
        return count > 0 ? root : nil
    }

    static func bundled() -> SCNNode? {
        for base in [Bundle.main.resourceURL, Bundle.main.resourceURL?.appendingPathComponent("glasses")].compactMap({ $0 }) {
            if FileManager.default.fileExists(atPath: base.appendingPathComponent("front_frame.stl").path) { return build(dir: base) }
        }
        return nil
    }
}
