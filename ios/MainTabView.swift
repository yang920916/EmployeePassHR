import SwiftUI

struct MainTabView: View {
    @State private var tab: Int = 0
    var body: some View {
        ZStack(alignment: .bottom) {
            TabView(selection: $tab) {
                ClockPage().tag(0)
                CalendarPage().tag(1)
                ProfilePage().tag(2)
            }
            HStack(spacing: 32) {
                TabButton(icon: "clock", title: "打卡", tag: 0, sel: $tab)
                TabButton(icon: "calendar", title: "行事曆", tag: 1, sel: $tab)
                TabButton(icon: "person", title: "基本資料", tag: 2, sel: $tab)
            }
            .padding(.horizontal, 24).padding(.vertical, 12)
            .background(.ultraThinMaterial).clipShape(Capsule())
            .padding(.bottom, 12)
            .shadow(radius: 10)
        }
    }
}

struct TabButton: View { let icon: String; let title: String; let tag: Int; @Binding var sel: Int
    var body: some View {
        Button { sel = tag } label: {
            VStack { Image(systemName: icon); Text(title).font(.caption2) }
        }.tint(sel==tag ? .blue : .secondary)
    }
}
