import SwiftUI

@main
struct EmployeePassHRApp: App {
    @StateObject var api = APIClient()
    var body: some Scene {
        WindowGroup {
            if api.token == nil { AuthView().environmentObject(api) }
            else { MainTabView().environmentObject(api) }
        }
    }
}
