import SwiftUI
import ARKit
import SceneKit

/// Scan -> fit -> latch.  ARKit TrueDepth face tracking; the FOCUS glasses snap onto the eyes/nose bridge.
final class TryOnModel: ObservableObject {
    @Published var phase: ScanState.Phase = .searching
    @Published var coverage: [Bool] = [false, false, false]
    @Published var progress: Float = 0
    @Published var supported = ARFaceTrackingConfiguration.isSupported
    @Published var resetToken = 0
    @Published var message: String?
    var title: String {
        switch phase {
        case .searching: return "Look at the camera"
        case .scanning: return coverage.allSatisfy { $0 } ? "Got it" : "Slowly turn your head left, then right"
        case .fitting: return "Fitting FOCUS to your face…"
        case .latched: return "FOCUS is on"
        }
    }
}

struct TryOnView: UIViewRepresentable {
    @ObservedObject var model: TryOnModel

    func makeCoordinator() -> Coordinator { Coordinator(model) }

    func makeUIView(context: Context) -> ARSCNView {
        let v = ARSCNView()
        v.delegate = context.coordinator
        v.automaticallyUpdatesLighting = false      // own soft studio environment: matte plastic keeps its shape
        v.scene.lightingEnvironment.contents = Coordinator.studioEnvironment()
        v.scene.lightingEnvironment.intensity = 1.4
        v.rendersCameraGrain = false
        context.coordinator.view = v
        context.coordinator.run()
        return v
    }

    func updateUIView(_ v: ARSCNView, context: Context) {
        if context.coordinator.lastToken != model.resetToken { context.coordinator.lastToken = model.resetToken; context.coordinator.reset() }
    }

    static func dismantleUIView(_ v: ARSCNView, coordinator: Coordinator) { v.session.pause() }

    final class Coordinator: NSObject, ARSCNViewDelegate {
        let model: TryOnModel
        weak var view: ARSCNView?
        var lastToken = 0
        private var scan = ScanState()
        private var glasses: SCNNode?
        private var smoothL: SIMD3<Float>?, smoothR: SIMD3<Float>?
        private var smoothT: simd_float4x4?
        private var haptics = UINotificationFeedbackGenerator()

        init(_ m: TryOnModel) { model = m }

        /// Soft grey gradient used as the reflection/ambient environment (bright overhead, darker below).
        static func studioEnvironment() -> UIImage {
            let sz = CGSize(width: 256, height: 128)
            return UIGraphicsImageRenderer(size: sz).image { ctx in
                let g = CGGradient(colorsSpace: CGColorSpaceCreateDeviceRGB(), colors: [UIColor(white: 0.95, alpha: 1).cgColor, UIColor(white: 0.55, alpha: 1).cgColor, UIColor(white: 0.25, alpha: 1).cgColor] as CFArray, locations: [0, 0.5, 1])!
                ctx.cgContext.drawLinearGradient(g, start: .zero, end: CGPoint(x: 0, y: sz.height), options: [])
            }
        }

        func run() {
            guard ARFaceTrackingConfiguration.isSupported else { return }
            let c = ARFaceTrackingConfiguration()
            c.maximumNumberOfTrackedFaces = 1
            c.isLightEstimationEnabled = true
            view?.session.run(c, options: [.resetTracking, .removeExistingAnchors])
        }

        func reset() {
            scan = ScanState(); smoothL = nil; smoothR = nil; smoothT = nil
            glasses?.isHidden = true
            publish()
            run()
        }

        private func publish() {
            let s = scan
            DispatchQueue.main.async { [model] in
                model.phase = s.phase; model.coverage = s.coverage; model.progress = s.progress
            }
        }

        func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
            guard anchor is ARFaceAnchor, let dev = (renderer as? ARSCNView)?.device else { return nil }
            let node = SCNNode()
            // Invisible face mesh: writes depth only, so the temples tuck behind the cheeks/ears.
            if let geo = ARSCNFaceGeometry(device: dev) {
                let m = SCNMaterial(); m.colorBufferWriteMask = []; m.isDoubleSided = false
                geo.materials = [m]
                let occ = SCNNode(geometry: geo); occ.name = "occluder"; occ.renderingOrder = -1
                node.addChildNode(occ)
            }
            if glasses == nil { glasses = GlassesNode.bundled() }
            if let g = glasses {
                g.removeFromParentNode(); g.isHidden = true; g.renderingOrder = 10
                node.addChildNode(g)
            } else {
                DispatchQueue.main.async { [model] in model.message = "Glasses model missing from the app bundle." }
            }
            return node
        }

        func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
            guard let face = anchor as? ARFaceAnchor else { return }
            if let occ = node.childNode(withName: "occluder", recursively: false)?.geometry as? ARSCNFaceGeometry { occ.update(from: face.geometry) }
            guard let g = glasses else { return }

            // Eye positions in face space (sorted so "left" is the smaller x).
            var a = SIMD3<Float>(face.leftEyeTransform.columns.3.x, face.leftEyeTransform.columns.3.y, face.leftEyeTransform.columns.3.z)
            var b = SIMD3<Float>(face.rightEyeTransform.columns.3.x, face.rightEyeTransform.columns.3.y, face.rightEyeTransform.columns.3.z)
            if a.x > b.x { swap(&a, &b) }
            let k: Float = 0.35
            smoothL = smoothL.map { $0 + (a - $0) * k } ?? a
            smoothR = smoothR.map { $0 + (b - $0) * k } ?? b
            let ipd = simd_length(smoothR! - smoothL!)

            var yaw: Float = 0
            if let cam = (renderer as? ARSCNView)?.session.currentFrame?.camera.transform {
                let local = simd_mul(cam.inverse, face.transform)
                yaw = atan2(local.columns.2.x, local.columns.2.z)
            }
            let before = scan.phase
            scan.update(tracked: face.isTracked, yaw: yaw, ipd: ipd)
            if scan.phase != before || scan.phase == .scanning { publish() }
            if before == .fitting && scan.phase == .latched { DispatchQueue.main.async { self.haptics.notificationOccurred(.success) } }

            // Use the median IPD from the scan for scale stability, live eyes for pose.
            let fit = FaceFit.transform(eyeL: smoothL!, eyeR: smoothR!, headUp: SIMD3(0, 1, 0), headForward: SIMD3(0, 0, 1))
            var m = fit.m
            if scan.phase == .fitting {
                // fly in from slightly in front and above, bigger, easing onto the face
                let t = scan.fitProgress, e = 1 - pow(1 - t, 3)
                let off = SIMD3<Float>(0, 0.035, 0.16) * (1 - e)
                let s = 1 + 0.5 * (1 - e)
                m.columns.0 *= SIMD4<Float>(s, s, s, 1)
                m.columns.1 *= SIMD4<Float>(s, s, s, 1)
                m.columns.2 *= SIMD4<Float>(s, s, s, 1)
                m.columns.3 += SIMD4(off, 0)
                g.opacity = CGFloat(min(1, t * 3))
            } else { g.opacity = 1 }
            g.simdTransform = m
            g.isHidden = !(scan.phase == .fitting || scan.phase == .latched) || !face.isTracked
        }

        func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
            scan.update(tracked: false, yaw: 0, ipd: 0); publish()
        }

        func session(_ session: ARSession, didFailWithError error: Error) {
            DispatchQueue.main.async { [model] in model.message = error.localizedDescription }
        }
    }
}

struct TryOnScreen: View {
    @StateObject private var model = TryOnModel()
    var body: some View {
        ZStack {
            if model.supported {
                TryOnView(model: model).ignoresSafeArea()
            } else {
                Color.black.ignoresSafeArea()
                Text("Face tracking needs an iPhone or iPad with a TrueDepth camera.").multilineTextAlignment(.center).padding(.leading, 24).padding(.trailing, 80)
            }
            VStack(spacing: 10) {
                VStack(spacing: 8) {
                    Text(model.title).font(.headline)
                    ProgressView(value: Double(model.progress)).tint(model.phase == .latched ? .green : .white)
                    HStack(spacing: 14) {
                        ForEach(0..<3, id: \.self) { i in
                            Label(["Left", "Front", "Right"][i], systemImage: model.coverage[i] ? "checkmark.circle.fill" : "circle")
                                .font(.caption).foregroundStyle(model.coverage[i] ? .green : .white.opacity(0.7))
                        }
                    }
                    if let m = model.message { Text(m).font(.caption).foregroundStyle(.red) }
                }
                .padding(12).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 14))
                Spacer()
                if model.phase == .latched {
                    Button { model.resetToken += 1 } label: { Label("Scan again", systemImage: "arrow.triangle.2.circlepath") }
                        .buttonStyle(.borderedProminent).controlSize(.small)
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8).foregroundStyle(.white)
            .animation(.easeInOut(duration: 0.2), value: model.phase)
        }
    }
}
