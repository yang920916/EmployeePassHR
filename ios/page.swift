import SwiftUI
import AVFoundation

struct QRScannerView: UIViewRepresentable {
    class Coordinator: NSObject, AVCaptureMetadataOutputObjectsDelegate {
        var parent: QRScannerView
        init(_ parent: QRScannerView) { self.parent = parent }
        func metadataOutput(_ output: AVCaptureMetadataOutput, didOutput metadataObjects: [AVMetadataObject], from connection: AVCaptureConnection) {
            guard let obj = metadataObjects.first as? AVMetadataMachineReadableCodeObject,
                  obj.type == .qr, let str = obj.stringValue else { return }
            parent.onFound(str)
        }
    }

    var onFound: (String) -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeUIView(context: Context) -> UIView {
        let view = UIView()
        let session = AVCaptureSession()
        guard let device = AVCaptureDevice.default(for: .video), let input = try? AVCaptureDeviceInput(device: device) else { return view }
        session.addInput(input)
        let output = AVCaptureMetadataOutput(); session.addOutput(output)
        output.setMetadataObjectsDelegate(context.coordinator, queue: DispatchQueue.main)
        output.metadataObjectTypes = [.qr]
        let preview = AVCaptureVideoPreviewLayer(session: session)
        preview.videoGravity = .resizeAspectFill
        preview.frame = UIScreen.main.bounds
        view.layer.addSublayer(preview)
        session.startRunning()
        return view
    }

    func updateUIView(_ uiView: UIView, context: Context) {}
}

struct CameraCaptureView: UIViewControllerRepresentable {
    var onCapture: (UIImage?) -> Void

    func makeUIViewController(context: Context) -> UIImagePickerController {
        let picker = UIImagePickerController()
        picker.sourceType = .camera
        picker.cameraCaptureMode = .photo
        picker.delegate = context.coordinator
        return picker
    }

    func updateUIViewController(_ uiViewController: UIImagePickerController, context: Context) {}

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    final class Coordinator: NSObject, UINavigationControllerDelegate, UIImagePickerControllerDelegate {
        let parent: CameraCaptureView
        init(_ parent: CameraCaptureView) { self.parent = parent }

        func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
            parent.onCapture(nil)
        }

        func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
            let image = (info[.editedImage] ?? info[.originalImage]) as? UIImage
            parent.onCapture(image)
        }
    }
}

struct ClockPage: View {
    @EnvironmentObject var api: APIClient
    @StateObject var loc = LocationService()
    @State private var showScanner = false
    @State private var showCamera = false
    @State private var scannedQR: String?
    @State private var shootingFor: String?
    @State private var toast: String?
    @State private var captured: UIImage?

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                Text("在公司電腦螢幕上掃描 QR 以打卡，並可自拍存證").foregroundStyle(.secondary)
                Button { showScanner = true } label: {
                    Label("掃描打卡 QR", systemImage: "qrcode.viewfinder").font(.title3)
                }.buttonStyle(.borderedProminent)

                if let t = toast { Text(t).foregroundStyle(.green) }

                if let l = loc.last {
                    Text(String(format: "定位：%.5f, %.5f", l.coordinate.latitude, l.coordinate.longitude))
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    Text("定位中…").font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
            }
            .padding()
            .navigationTitle("打卡")
            .sheet(isPresented: $showScanner) {
                QRScannerView { qr in
                    scannedQR = qr
                    showScanner = false
                    showCamera = true  // 掃到後開相機
                }.ignoresSafeArea()
            }
            .sheet(isPresented: $showCamera) {
                CameraCaptureView { img in
                    captured = img
                    showCamera = false
                    Task { await punch() }
                }.ignoresSafeArea()
            }
        }
    }

    func punch() async {
        guard let qr = scannedQR else { toast = "未取得 QR"; return }
        do {
            let result = try await api.punch(qr: qr, type: nil, loc: loc.last, photo: captured)
            toast = "成功：\(result.type) @ \(result.ts)"
            captured = nil; scannedQR = nil
        } catch {
            toast = "打卡失敗"
        }
    }
}

struct CalendarPage: View {
    let events = [
        ("人事部月會", "2025-11-20", "會議室 A"),
        ("員工大會", "2025-12-05", "多功能廳"),
        ("年度考核起跑", "2025-12-10", "HR Portal")
    ]
    var body: some View {
        List(events, id: \.0) { e in
            VStack(alignment: .leading) {
                Text(e.0).font(.headline)
                Text(e.1 + (e.2.isEmpty ? "" : " · " + e.2)).foregroundStyle(.secondary)
            }
        }
        .navigationTitle("行事曆")
    }
}

struct ProfilePage: View {
    @EnvironmentObject var api: APIClient

    var body: some View {
        NavigationStack {
            List {
                Section("個人資訊") {
                    Text("姓名：\(api.me?.name ?? "-")")
                    Text("部門：\(api.me?.dept ?? "-")")
                    Text("資歷：—")
                    Text("角色：\(api.me?.role ?? "-")")
                    // 例：只有 ADMIN 才顯示管理入口
                    if api.me?.role == "ADMIN" {
                        NavigationLink("管理員面板（Demo）") { AdminPanel() }
                    }
                }
                Section("功能") {
                    NavigationLink("薪資預覽") { SalaryPreviewPage() }
                    NavigationLink("歷史打卡") { HistoryPage() }
                }
                Section { Button("登出", role: .destructive) { api.logout() } }
            }
            .navigationTitle("基本資料")
        }
    }
}

struct AdminPanel: View {
    var body: some View {
        List {
            Label("審核請假（待做）", systemImage: "person.badge.clock")
            Label("匯出月報（待做）", systemImage: "doc.plaintext")
        }.navigationTitle("管理員面板")
    }
}

struct SalaryPreviewPage: View {
    @EnvironmentObject var api: APIClient
    @State private var month: String = {
        let f = DateFormatter(); f.dateFormat = "yyyy-MM"; return f.string(from: Date())
    }()
    @State private var data: PayrollPreview?

    var body: some View {
        Form {
            Section("月份") {
                TextField("YYYY-MM", text: $month).keyboardType(.numbersAndPunctuation)
                Button("載入") { Task { try? await load() } }
            }
            if let d = data {
                Section("預覽") {
                    Text("總時數：\(d.hours, specifier: "%.2f") 小時")
                    Text("Demo 淨額：$\(d.gross_demo)")
                }
            }
        }
        .navigationTitle("薪資預覽")
        .onAppear { Task { try? await load() } }
    }

    func load() async throws { data = try await api.payroll(month: month) }
}

struct HistoryPage: View {
    @EnvironmentObject var api: APIClient
    @State private var logs: [TimeLog] = []
    var body: some View {
        List(logs) { l in
            HStack(alignment: .top, spacing: 12) {
                if let p = l.photo_path, let url = URL(string: api.base.appending(path: "/uploads/\(p)").absoluteString) {
                    AsyncImage(url: url) { img in img.resizable().scaledToFill() } placeholder: { Color.gray.opacity(0.2) }
                        .frame(width: 56, height: 56).clipShape(RoundedRectangle(cornerRadius: 8))
                }
                VStack(alignment: .leading) {
                    Text(l.type).font(.headline)
                    Text(l.ts).foregroundStyle(.secondary)
                    if let lat = l.lat, let lng = l.lng {
                        Text(String(format: "%.5f, %.5f", lat, lng)).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .navigationTitle("歷史打卡")
        .onAppear { Task { logs = (try? await api.history()) ?? [] } }
    }
}