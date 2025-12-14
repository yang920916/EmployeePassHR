import SwiftUI

struct AuthView: View {
    @EnvironmentObject var api: APIClient
    @State private var email = "alice@corp"
    @State private var password = "123456"
    @State private var error: String?

    var body: some View {
        VStack(spacing: 16) {
            Text("員工通 HR").font(.largeTitle).bold()
            TextField("Email", text: $email).textInputAutocapitalization(.never).textFieldStyle(.roundedBorder)
            SecureField("密碼", text: $password).textFieldStyle(.roundedBorder)
            Button(action: login) { Text("登入").frame(maxWidth: .infinity) }
                .buttonStyle(.borderedProminent)
            if let e = error { Text(e).foregroundStyle(.red) }
        }
        .padding()
    }
    func login() {
        Task { do { try await api.login(email: email, password: password) } catch { self.error = "登入失敗" } }
    }
}
