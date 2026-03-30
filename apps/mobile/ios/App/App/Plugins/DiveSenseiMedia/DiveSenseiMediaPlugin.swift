import Foundation
import Capacitor
import AVFoundation
import Photos
import PhotosUI
import UIKit

@objc(DiveSenseiMediaPlugin)
public final class DiveSenseiMediaPlugin: CAPPlugin, CAPBridgedPlugin {
    public let identifier = "DiveSenseiMediaPlugin"
    public let jsName = "DiveSenseiMedia"
    public let pluginMethods: [CAPPluginMethod] = [
        CAPPluginMethod(name: "pickSourceVideo", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getSourceAvailability", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "repairSource", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "createSession", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "listSessions", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "startAnalysis", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getJob", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "observeJob", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "cancelJob", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getSessionManifest", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "saveDecision", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "listDecisions", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "getReviewProxy", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "startExport", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "listExports", returnType: CAPPluginReturnPromise),
        CAPPluginMethod(name: "deleteSession", returnType: CAPPluginReturnPromise)
    ]

    private let sourceStore = MobileSourceStore()

    @objc public func pickSourceVideo(_ call: CAPPluginCall) {
        guard let presenter = bridge?.viewController else {
            call.reject("Bridge view controller unavailable.", "bridge_unavailable")
            return
        }

        let picker = SourceVideoPickerController(sourceStore: sourceStore) { result in
            switch result {
            case .success(let source):
                if let source {
                    call.resolve([
                        "cancelled": false,
                        "source": source.asDictionary
                    ])
                } else {
                    call.resolve(["cancelled": true])
                }
            case .failure(let error as DiveSenseiPluginError):
                call.reject(error.message, error.code)
            case .failure(let error):
                call.reject(error.localizedDescription, "pick_source_failed")
            }
        }

        picker.present(from: presenter)
    }

    @objc public func getSourceAvailability(_ call: CAPPluginCall) {
        guard let sourceRef = call.getString("sourceRef"), !sourceRef.isEmpty else {
            call.reject("sourceRef is required.", "invalid_source_ref")
            return
        }

        do {
            let response = try sourceStore.availability(sourceRef: sourceRef)
            call.resolve(response.asDictionary)
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "source_availability_failed")
        }
    }

    @objc public func repairSource(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }

        guard let presenter = bridge?.viewController else {
            call.reject("Bridge view controller unavailable.", "bridge_unavailable")
            return
        }

        let picker = SourceVideoPickerController(sourceStore: sourceStore) { [weak self] result in
            guard let self else { return }
            switch result {
            case .success(let source):
                guard let source else {
                    call.resolve([
                        "repaired": false,
                        "availability": "missing"
                    ])
                    return
                }
                self.notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
                    "sessionId": sessionId,
                    "status": "created",
                    "updatedAt": ISO8601DateFormatter().string(from: Date())
                ])
                call.resolve([
                    "repaired": true,
                    "source": source.asDictionary,
                    "availability": source.availability.rawValue
                ])
            case .failure(let error as DiveSenseiPluginError):
                call.reject(error.message, error.code)
            case .failure(let error):
                call.reject(error.localizedDescription, "repair_source_failed")
            }
        }

        picker.present(from: presenter)
    }

    @objc public func createSession(_ call: CAPPluginCall) {
        call.resolve([
            "sessionId": "sess_stub_0001",
            "status": "created",
            "createdAt": ISO8601DateFormatter().string(from: Date())
        ])
    }

    @objc public func listSessions(_ call: CAPPluginCall) {
        call.resolve(["sessions": []])
    }

    @objc public func startAnalysis(_ call: CAPPluginCall) {
        let sessionId = call.getString("sessionId") ?? "sess_stub_0001"
        let jobId = "job_stub_0001"
        let now = ISO8601DateFormatter().string(from: Date())

        notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
            "sessionId": sessionId,
            "status": "analyzing",
            "updatedAt": now
        ])

        notifyListeners("DiveSenseiMedia.jobProgress", data: [
            "jobId": jobId,
            "sessionId": sessionId,
            "kind": "analysis",
            "phase": "detecting",
            "status": "running",
            "progress": 0.05,
            "message": "Stub native analysis started.",
            "updatedAt": now
        ])

        call.resolve([
            "jobId": jobId,
            "sessionId": sessionId,
            "kind": "analysis",
            "status": "running"
        ])
    }

    @objc public func getJob(_ call: CAPPluginCall) {
        call.resolve([
            "job": [
                "jobId": call.getString("jobId") ?? "job_stub_0001",
                "sessionId": "sess_stub_0001",
                "kind": "analysis",
                "phase": "detecting",
                "status": "running",
                "progress": 0.05,
                "message": "Stub job.",
                "updatedAt": ISO8601DateFormatter().string(from: Date())
            ]
        ])
    }

    @objc public func observeJob(_ call: CAPPluginCall) {
        call.resolve()
    }

    @objc public func cancelJob(_ call: CAPPluginCall) {
        call.resolve(["cancelled": false])
    }

    @objc public func getSessionManifest(_ call: CAPPluginCall) {
        call.reject("Manifest generation is not implemented yet.", "manifest_unimplemented")
    }

    @objc public func saveDecision(_ call: CAPPluginCall) {
        call.reject("Decision persistence is not implemented yet.", "decision_unimplemented")
    }

    @objc public func listDecisions(_ call: CAPPluginCall) {
        call.resolve(["decisions": []])
    }

    @objc public func getReviewProxy(_ call: CAPPluginCall) {
        call.resolve([
            "proxy": [
                "sessionId": call.getString("sessionId") ?? "sess_stub_0001",
                "status": "failed",
                "updatedAt": ISO8601DateFormatter().string(from: Date()),
                "playerBackend": "html_video"
            ]
        ])
    }

    @objc public func startExport(_ call: CAPPluginCall) {
        call.reject("Export is not implemented yet.", "export_unimplemented")
    }

    @objc public func listExports(_ call: CAPPluginCall) {
        call.resolve(["exports": []])
    }

    @objc public func deleteSession(_ call: CAPPluginCall) {
        call.resolve(["deleted": false])
    }
}

private struct DiveSenseiPluginError: Error {
    let code: String
    let message: String
}

private enum SourceOrigin: String {
    case photos
    case files
}

private enum SourceAvailability: String {
    case available
    case needsDownload = "needs_download"
    case missing
    case permissionDenied = "permission_denied"
    case unsupported
}

private struct StoredSource: Codable {
    let sourceRef: String
    let origin: SourceOrigin
    let displayName: String
    let availability: SourceAvailability
    let durationSeconds: Double?
    let fileSizeBytes: Int64?
    let canPersist: Bool
    let assetLocalIdentifier: String?
    let lastResolvedAt: String?

    var asDictionary: [String: Any] {
        [
            "sourceRef": sourceRef,
            "origin": origin.rawValue,
            "displayName": displayName,
            "availability": availability.rawValue,
            "durationSeconds": durationSeconds as Any,
            "fileSizeBytes": fileSizeBytes as Any,
            "canPersist": canPersist
        ]
    }
}

private struct SourceAvailabilityResponse {
    let sourceRef: String
    let availability: SourceAvailability
    let origin: SourceOrigin?
    let displayName: String?
    let durationSeconds: Double?
    let fileSizeBytes: Int64?
    let lastResolvedAt: String?

    var asDictionary: [String: Any] {
        [
            "sourceRef": sourceRef,
            "availability": availability.rawValue,
            "origin": origin?.rawValue as Any,
            "displayName": displayName as Any,
            "durationSeconds": durationSeconds as Any,
            "fileSizeBytes": fileSizeBytes as Any,
            "lastResolvedAt": lastResolvedAt as Any
        ]
    }
}

private final class MobileSourceStore {
    private let userDefaults = UserDefaults.standard
    private let key = "divesensei.mobile.sources"

    func save(_ source: StoredSource) throws {
      var all = try loadAll()
      all[source.sourceRef] = source
      let data = try JSONEncoder().encode(all)
      userDefaults.set(data, forKey: key)
    }

    func get(sourceRef: String) throws -> StoredSource? {
      try loadAll()[sourceRef]
    }

    func availability(sourceRef: String) throws -> SourceAvailabilityResponse {
      guard let source = try get(sourceRef: sourceRef) else {
        throw DiveSenseiPluginError(code: "source_not_found", message: "Source was not found.")
      }

      if let assetId = source.assetLocalIdentifier {
        let fetchResult = PHAsset.fetchAssets(withLocalIdentifiers: [assetId], options: nil)
        if fetchResult.firstObject == nil {
          return SourceAvailabilityResponse(
            sourceRef: source.sourceRef,
            availability: .missing,
            origin: source.origin,
            displayName: source.displayName,
            durationSeconds: source.durationSeconds,
            fileSizeBytes: source.fileSizeBytes,
            lastResolvedAt: source.lastResolvedAt
          )
        }
      }

      return SourceAvailabilityResponse(
        sourceRef: source.sourceRef,
        availability: source.availability,
        origin: source.origin,
        displayName: source.displayName,
        durationSeconds: source.durationSeconds,
        fileSizeBytes: source.fileSizeBytes,
        lastResolvedAt: ISO8601DateFormatter().string(from: Date())
      )
    }

    private func loadAll() throws -> [String: StoredSource] {
      guard let data = userDefaults.data(forKey: key) else {
        return [:]
      }
      return try JSONDecoder().decode([String: StoredSource].self, from: data)
    }
}

private final class SourceVideoPickerController: NSObject, PHPickerViewControllerDelegate {
    private let sourceStore: MobileSourceStore
    private let completion: (Result<StoredSource?, Error>) -> Void
    private var retainedPicker: PHPickerViewController?

    init(
      sourceStore: MobileSourceStore,
      completion: @escaping (Result<StoredSource?, Error>) -> Void
    ) {
      self.sourceStore = sourceStore
      self.completion = completion
    }

    func present(from presenter: UIViewController) {
      var configuration = PHPickerConfiguration(photoLibrary: .shared())
      configuration.filter = .videos
      configuration.selectionLimit = 1
      let picker = PHPickerViewController(configuration: configuration)
      picker.delegate = self
      retainedPicker = picker
      presenter.present(picker, animated: true)
    }

    func picker(_ picker: PHPickerViewController, didFinishPicking results: [PHPickerResult]) {
      picker.dismiss(animated: true)
      defer { retainedPicker = nil }

      guard let result = results.first else {
        completion(.success(nil))
        return
      }

      let assetId = result.assetIdentifier
      let displayName = "Picked Video"
      let sourceRef = "src_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())"
      let source = StoredSource(
        sourceRef: sourceRef,
        origin: .photos,
        displayName: displayName,
        availability: .available,
        durationSeconds: nil,
        fileSizeBytes: nil,
        canPersist: true,
        assetLocalIdentifier: assetId,
        lastResolvedAt: ISO8601DateFormatter().string(from: Date())
      )

      do {
        try sourceStore.save(source)
        completion(.success(source))
      } catch {
        completion(.failure(error))
      }
    }
}
