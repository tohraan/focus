import SwiftUI

@main
struct FocusApp: App {
    var body: some Scene {
        WindowGroup {
            if let i = CommandLine.arguments.firstIndex(of: "-bench"), CommandLine.arguments.count > i + 2 {
                BenchView(model: CommandLine.arguments[i + 1], units: CommandLine.arguments[i + 2])
            }
            else if CommandLine.arguments.contains("-videos") { VideoBatchView() }
            else { ContentView().preferredColorScheme(.dark) }
        }
    }
}

struct BenchView: View {
    let model: String
    let units: String
    @State private var text = "Running benchmark…"
    var body: some View {
        ScrollView { Text(text).font(.system(size: 10, design: .monospaced)).padding() }
            .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
            .task { text = await Task.detached { Benchmark.runOne(model: model, units: units) }.value }
    }
}

struct VideoBatchView: View {
    @State private var text = "Processing videos…"
    var body: some View {
        Text(text).font(.system(size: 14, design: .monospaced)).padding()
            .onAppear { UIApplication.shared.isIdleTimerDisabled = true }
            .task {
                await Task.detached(priority: .userInitiated) { VideoBatch.runAll() }.value
                text = "Done."
                exit(0)
            }
    }
}
