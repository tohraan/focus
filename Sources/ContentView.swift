import SwiftUI
import MetalKit

struct MetalView: UIViewRepresentable {
    let renderer: Renderer
    func makeUIView(context: Context) -> MTKView {
        let v = MTKView()
        renderer.attach(v)
        return v
    }
    func updateUIView(_ uiView: MTKView, context: Context) {}
}

struct ContentView: View {
    @StateObject private var r = Renderer()
    @State private var focusOn = true
    @State private var view = 0   // 0 processed, 1 split, 2 mask
    @State private var engine = 1

    private func apply() {
        if !focusOn { r.mode = 1 }
        else { r.mode = [0, 2, 3][view] }
    }

    var body: some View {
        ZStack {
            MetalView(renderer: r).ignoresSafeArea()
            VStack(spacing: 8) {
                HStack {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(String(format: "%.0f FPS  ·  seg %.0f ms  ·  latency %.0f ms", r.fps, r.segMs, r.latencyMs))
                        Text("Tracking: \(r.tracking)")
                        Text(r.stillStatus)
                        Text(r.autoStatus)
                    }
                    .font(.system(size: 12, design: .monospaced))
                    .padding(8).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 10))
                    Spacer()
                }
                if let t = r.captureTitle {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack { Image(systemName: r.captureProgress >= 1 ? "checkmark.circle.fill" : "scope"); Text(t).font(.headline) }
                        ProgressView(value: min(max(r.captureProgress, 0), 1))
                        Text(r.captureDetail).font(.footnote).fixedSize(horizontal: false, vertical: true)
                    }
                    .padding(12).background(.blue.opacity(0.85), in: RoundedRectangle(cornerRadius: 14))
                    .transition(.opacity)
                }
                if let m = r.message {
                    Text(m).font(.footnote).padding(8).background(.red.opacity(0.8), in: RoundedRectangle(cornerRadius: 10))
                }
                Spacer()
                VStack(spacing: 10) {
                    Text("Filtered view only. People you can't see are still there — don't walk while using this.")
                        .font(.caption2).multilineTextAlignment(.center).foregroundStyle(.white.opacity(0.8))
                    if focusOn {
                        Picker("View", selection: $view) {
                            Text("Processed").tag(0); Text("Split").tag(1); Text("Mask").tag(2)
                        }.pickerStyle(.segmented)
                    }
                    Picker("Engine", selection: $engine) { Text("Memory (11:05)").tag(0); Text("Keyframes").tag(1) }
                        .pickerStyle(.segmented)
                    HStack {
                        Toggle("Focus Mode", isOn: $focusOn).toggleStyle(.switch).fixedSize()
                        Spacer()
                        Button("Capture background") {
                            UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                            r.beginCapture()
                        }
                            .buttonStyle(.borderedProminent)
                    }
                }
                .padding(12).background(.black.opacity(0.55), in: RoundedRectangle(cornerRadius: 16))
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .foregroundStyle(.white)
        }
        .onAppear { apply(); r.start() }
        .onChange(of: engine) { r.engine = $0; r.resetMemory() }
        .animation(.easeInOut(duration: 0.2), value: r.captureTitle)
        .onChange(of: focusOn) { _ in apply() }
        .onChange(of: view) { _ in apply() }
    }
}
