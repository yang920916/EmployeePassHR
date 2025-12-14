import Foundation

struct User: Codable { let id: Int; let name: String; let role: String; let dept: String }
struct Config: Codable { let office_lat: Double; let office_lng: Double; let radius_m: Double }
struct PunchResult: Codable { let ok: Bool; let ts: String; let type: String }
struct TimeLog: Codable, Identifiable { let id: Int; let ts: String; let type: String; let lat: Double?; let lng: Double?; let photo_path: String? }
struct PayrollPreview: Codable { let month: String; let minutes: Int; let hours: Double; let gross_demo: Int }
