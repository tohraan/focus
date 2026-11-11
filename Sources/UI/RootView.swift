import SwiftUI

enum AppMode: Int, CaseIterable {
    case focus, tryOn, isolation
    var icon: String { ["person.slash", "eyeglasses", "sparkles.rectangle.stack"][rawValue] }
    var label: String { ["Focus", "Try-On", "Isolate"][rawValue] }
}

struct RootView: View {
    @State private var mode: AppMode = CommandLine.arguments.contains("-tryon") ? .tryOn : (CommandLine.arguments.contains("-isolate") ? .isolation : .focus)
    var body: some View {
        ZStack(alignment: .trailing) {
            switch mode {
            case .focus: ContentView()
            case .tryOn: TryOnScreen()
            case .isolation: IsolationScreen()
            }
            // Mode rail: a slim vertical switcher on the trailing edge, clear of the corner controls.
            VStack(spacing: 6) {
                ForEach(AppMode.allCases, id: \.self) { m in
                    Button { mode = m } label: {
                        VStack(spacing: 2) { Image(systemName: m.icon).font(.system(size: 16)); Text(m.label).font(.system(size: 9, weight: .semibold)) }
                            .frame(width: 46, height: 46)
                            .background(mode == m ? Color.white.opacity(0.28) : Color.clear, in: RoundedRectangle(cornerRadius: 12))
                            .foregroundStyle(.white)
                    }
                }
            }
            .padding(5).background(.black.opacity(0.5), in: RoundedRectangle(cornerRadius: 16))
            .padding(.trailing, 6)
        }
        .preferredColorScheme(.dark)
    }
}
