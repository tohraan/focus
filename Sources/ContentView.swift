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
    @State private var smooth = true
    @State private var autoStills = false
    @State private var showControls = false
    @State private var showDetails = false

    private func apply() {
        if !focusOn { r.mode = 1 }
        else { r.mode = [0, 2, 3][view] }
    }

    var body: some View {
        ZStack {
            MetalView(renderer: r).ignoresSafeArea()
            VStack(spacing: 8) {
                // Top: one small status pill. Tap it for the full numbers.
                HStack {
                    Button { withAnimation { showDetails.toggle() } } label: {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(String(format: "Focus Mode · %.0f FPS · ", r.fps) + r.stillStatus)
                            if showDetails {
                                Text(String(format: "seg %.0f ms · latency %.0f ms", r.segMs, r.latencyMs))
                                Text("Tracking: \(r.tracking)")
                                Text(r.autoStatus)
                            }
                        }
                        .font(.system(size: 11, design: .monospaced))
                        .multilineTextAlignment(.leading)
                        .padding(.horizontal, 8).padding(.vertical, 5)
                        .background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                    }
                    .buttonStyle(.plain)
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

                // Bottom: two small buttons. The full controls open only on request.
                if showControls {
                    VStack(spacing: 10) {
                        if focusOn {
                            Picker("View", selection: $view) {
                                Text("Processed").tag(0); Text("Split").tag(1); Text("Mask").tag(2)
                            }.pickerStyle(.segmented)
                        }
                        Picker("Engine", selection: $engine) { Text("Memory (11:05)").tag(0); Text("Keyframes").tag(1) }
                            .pickerStyle(.segmented)
                        Toggle("Focus Mode", isOn: $focusOn).toggleStyle(.switch)
                        Toggle("Smooth edges", isOn: $smooth).toggleStyle(.switch)
                        Toggle("Auto debug stills (every 2 s)", isOn: $autoStills).toggleStyle(.switch)
                        Button("Save debug snapshot") { r.requestSnapshot("manual") }.buttonStyle(.bordered).controlSize(.small)
                        Text("Filtered view only. People you can't see are still there, so don't walk while using this.")
                            .font(.caption2).multilineTextAlignment(.center).foregroundStyle(.white.opacity(0.7))
                    }
                    .padding(12).background(.black.opacity(0.6), in: RoundedRectangle(cornerRadius: 16))
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
                HStack {
                    Button {
                        UIImpactFeedbackGenerator(style: .medium).impactOccurred()
                        r.beginCapture()
                    } label: { Label("Capture background", systemImage: "scope").font(.footnote.weight(.semibold)) }
                        .buttonStyle(.borderedProminent).controlSize(.small).fixedSize()
                    Spacer()
                    Button { withAnimation { showControls.toggle() } } label: {
                        Image(systemName: showControls ? "chevron.down" : "slider.horizontal.3")
                            .font(.system(size: 15, weight: .semibold)).frame(width: 34, height: 34)
                            .background(.black.opacity(0.5), in: Circle())
                    }
                }
            }
            .padding(.horizontal, 12).padding(.vertical, 8)
            .foregroundStyle(.white)
        }
        .onAppear { apply(); r.start() }
        .onDisappear { r.stop() }
        .onChange(of: engine) { r.engine = $0; r.resetMemory() }
        .animation(.easeInOut(duration: 0.2), value: r.captureTitle)
        .onChange(of: focusOn) { _ in apply() }
        .onChange(of: smooth) { r.stabilizeMasks = $0 }
        .onChange(of: autoStills) { r.autoSnap = $0 }
        .onChange(of: view) { _ in apply() }
    }
}
