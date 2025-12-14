import Foundation
import CoreLocation

final class LocationService: NSObject, ObservableObject, CLLocationManagerDelegate {
    @Published var last: CLLocation?
    private let mgr = CLLocationManager()
    override init() {
        super.init()
        mgr.delegate = self
        mgr.requestWhenInUseAuthorization()
        mgr.startUpdatingLocation()
    }
    func locationManager(_ manager: CLLocationManager, didUpdateLocations locations: [CLLocation]) { last = locations.last }
}
