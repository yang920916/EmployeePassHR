import Foundation
import CoreLocation
import UIKit

@MainActor
final class APIClient: ObservableObject {
    @Published var token: String? = nil
    @Published var me: User? = nil
    let base = URL(string: "http://127.0.0.1:8000")!

    func login(email: String, password: String) async throws {
        struct Req: Codable { let email: String; let password: String }
        struct Res: Codable { let token: String; let user: User }
        let url = base.appending(path: "/auth/login")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.httpBody = try JSONEncoder().encode(Req(email: email, password: password))
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        let r = try JSONDecoder().decode(Res.self, from: data)
        token = r.token; me = r.user
    }

    func headers() -> [String:String] { ["Authorization":"Bearer \(token ?? "")"] }

    func getConfig() async throws -> Config {
        let url = base.appending(path: "/config")
        let (data, _) = try await URLSession.shared.data(from: url)
        return try JSONDecoder().decode(Config.self, from: data)
    }

    struct PunchReq: Codable { let type: String?; let lat: Double?; let lng: Double?; let qr: String; let photo_b64: String? }

    func punch(qr: String, type: String? = nil, loc: CLLocation?, photo: UIImage?) async throws -> PunchResult {
        let url = base.appending(path: "/attendance/punch")
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        let lat = loc?.coordinate.latitude
        let lng = loc?.coordinate.longitude
        var photoB64: String? = nil
        if let img = photo, let data = img.jpegData(compressionQuality: 0.85) {
            photoB64 = "data:image/jpeg;base64," + data.base64EncodedString()
        }
        let body = PunchReq(type: type, lat: lat, lng: lng, qr: qr, photo_b64: photoB64)
        req.httpBody = try JSONEncoder().encode(body)
        let (data, resp) = try await URLSession.shared.data(for: req)
        guard (resp as? HTTPURLResponse)?.statusCode == 200 else { throw URLError(.badServerResponse) }
        return try JSONDecoder().decode(PunchResult.self, from: data)
    }

    func history() async throws -> [TimeLog] {
        let url = base.appending(path: "/attendance/history")
        var req = URLRequest(url: url)
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        let (data, _) = try await URLSession.shared.data(for: req)
        return try JSONDecoder().decode([TimeLog].self, from: data)
    }

    func payroll(month: String) async throws -> PayrollPreview {
        var comps = URLComponents(url: base.appending(path: "/payroll/preview"), resolvingAgainstBaseURL: false)!
        comps.queryItems = [URLQueryItem(name: "month", value: month)]
        var req = URLRequest(url: comps.url!)
        headers().forEach { req.setValue($1, forHTTPHeaderField: $0) }
        let (data, _) = try await URLSession.shared.data(for: req)
        return try JSONDecoder().decode(PayrollPreview.self, from: data)
    }

    func logout() { token = nil; me = nil }
}
