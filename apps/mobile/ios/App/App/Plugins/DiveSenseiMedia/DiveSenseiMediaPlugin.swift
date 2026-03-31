import AVFoundation
import Capacitor
import Foundation
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

    private lazy var store = try! MobilePersistenceStore()
    private let jobStore = MobileJobStore()
    private var activePicker: SourceVideoPickerController?

    @objc public func pickSourceVideo(_ call: CAPPluginCall) {
        guard let presenter = bridge?.viewController else {
            call.reject("Bridge view controller unavailable.", "bridge_unavailable")
            return
        }

        let picker = SourceVideoPickerController(store: store) { [weak self] result in
            self?.activePicker = nil
            switch result {
            case .success(let source):
                if let source {
                    call.resolve([
                        "cancelled": false,
                        "source": source.asSummaryPayload().asDictionary
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

        activePicker = picker
        picker.present(from: presenter)
    }

    @objc public func getSourceAvailability(_ call: CAPPluginCall) {
        guard let sourceRef = call.getString("sourceRef"), !sourceRef.isEmpty else {
            call.reject("sourceRef is required.", "invalid_source_ref")
            return
        }

        do {
            let response = try store.sourceAvailability(sourceRef: sourceRef)
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

        let picker = SourceVideoPickerController(store: store) { [weak self] result in
            guard let self else { return }
            self.activePicker = nil

            switch result {
            case .success(let source):
                guard let source else {
                    call.resolve([
                        "repaired": false,
                        "availability": SourceAvailability.missing.rawValue
                    ])
                    return
                }

                do {
                    let response = try self.store.repairSessionSource(sessionId: sessionId, source: source)
                    self.notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
                        "sessionId": sessionId,
                        "status": SessionStatus.created.rawValue,
                        "updatedAt": response.updatedAt
                    ])
                    call.resolve(response.asDictionary)
                } catch let error as DiveSenseiPluginError {
                    call.reject(error.message, error.code)
                } catch {
                    call.reject(error.localizedDescription, "repair_source_failed")
                }
            case .failure(let error as DiveSenseiPluginError):
                call.reject(error.message, error.code)
            case .failure(let error):
                call.reject(error.localizedDescription, "repair_source_failed")
            }
        }

        activePicker = picker
        picker.present(from: presenter)
    }

    @objc public func createSession(_ call: CAPPluginCall) {
        guard let sourceRef = call.getString("sourceRef"), !sourceRef.isEmpty else {
            call.reject("sourceRef is required.", "invalid_source_ref")
            return
        }
        guard let profile = SessionProfile(rawValue: call.getString("profile") ?? "") else {
            call.reject("profile is required.", "invalid_profile")
            return
        }
        guard let detectorId = DetectorId(rawValue: call.getString("detectorId") ?? "") else {
            call.reject("detectorId is required.", "invalid_detector_id")
            return
        }

        do {
            let response = try store.createSession(
                sourceRef: sourceRef,
                sessionName: call.getString("sessionName"),
                profile: profile,
                detectorId: detectorId
            )
            notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
                "sessionId": response.sessionId,
                "status": response.status.rawValue,
                "updatedAt": response.createdAt
            ])
            call.resolve(response.asDictionary)
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "create_session_failed")
        }
    }

    @objc public func listSessions(_ call: CAPPluginCall) {
        do {
            let sessions = try store.listSessions()
            call.resolve(["sessions": sessions.map(\.jsonObject)])
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "list_sessions_failed")
        }
    }

    @objc public func startAnalysis(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }

        do {
            _ = try store.requireSession(sessionId: sessionId)
            let now = DateTimestamp.isoNow()
            let job = jobStore.createAnalysisJob(sessionId: sessionId, startedAt: now)
            _ = try store.updateSessionStatus(sessionId: sessionId, status: .analyzing, updatedAt: now)

            notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
                "sessionId": sessionId,
                "status": SessionStatus.analyzing.rawValue,
                "updatedAt": now
            ])
            notifyListeners("DiveSenseiMedia.jobProgress", data: job.asDictionary)

            DispatchQueue.main.asyncAfter(deadline: .now() + 0.35) { [weak self] in
                guard let self else { return }
                let completedAt = DateTimestamp.isoNow()
                do {
                    _ = try self.store.promoteSessionToReviewReady(sessionId: sessionId, updatedAt: completedAt)
                    let completedJob = self.jobStore.completeJob(
                        jobId: job.jobId,
                        phase: .reviewReady,
                        status: .completed,
                        progress: 1,
                        message: "Placeholder native analysis completed.",
                        finishedAt: completedAt
                    )
                    self.notifyListeners("DiveSenseiMedia.jobProgress", data: completedJob.asDictionary)
                    self.notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
                        "sessionId": sessionId,
                        "status": SessionStatus.reviewReady.rawValue,
                        "updatedAt": completedAt
                    ])
                } catch {
                    let failedJob = self.jobStore.completeJob(
                        jobId: job.jobId,
                        phase: .failed,
                        status: .failed,
                        progress: 1,
                        message: error.localizedDescription,
                        finishedAt: completedAt
                    )
                    self.notifyListeners("DiveSenseiMedia.jobProgress", data: failedJob.asDictionary)
                }
            }

            call.resolve([
                "jobId": job.jobId,
                "sessionId": sessionId,
                "kind": JobKind.analysis.rawValue,
                "status": JobStatus.running.rawValue
            ])
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "start_analysis_failed")
        }
    }

    @objc public func getJob(_ call: CAPPluginCall) {
        guard let jobId = call.getString("jobId"), !jobId.isEmpty else {
            call.reject("jobId is required.", "invalid_job_id")
            return
        }

        guard let job = jobStore.get(jobId: jobId) else {
            call.reject("Job was not found.", "job_not_found")
            return
        }

        call.resolve(["job": job.jsonObject])
    }

    @objc public func observeJob(_ call: CAPPluginCall) {
        call.resolve()
    }

    @objc public func cancelJob(_ call: CAPPluginCall) {
        guard let jobId = call.getString("jobId"), !jobId.isEmpty else {
            call.reject("jobId is required.", "invalid_job_id")
            return
        }

        let cancelled = jobStore.cancel(jobId: jobId)
        call.resolve(["cancelled": cancelled])
    }

    @objc public func getSessionManifest(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }

        do {
            let manifest = try store.getSessionManifest(sessionId: sessionId)
            call.resolve(["manifest": manifest.jsonObject])
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "manifest_failed")
        }
    }

    @objc public func saveDecision(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }
        guard let detectionId = call.getString("detectionId"), !detectionId.isEmpty else {
            call.reject("detectionId is required.", "invalid_detection_id")
            return
        }
        guard let label = ReviewDecisionLabel(rawValue: call.getString("label") ?? "") else {
            call.reject("label is required.", "invalid_decision_label")
            return
        }

        do {
            let decision = try store.saveDecision(
                sessionId: sessionId,
                detectionId: detectionId,
                label: label,
                notes: call.getString("notes") ?? ""
            )
            if let session = try? store.requireSession(sessionId: sessionId) {
                notifyListeners("DiveSenseiMedia.sessionUpdated", data: [
                    "sessionId": session.sessionId,
                    "status": session.status.rawValue,
                    "updatedAt": session.updatedAt
                ])
            }
            call.resolve(["decision": decision.jsonObject])
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "save_decision_failed")
        }
    }

    @objc public func listDecisions(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }

        do {
            let decisions = try store.listDecisions(sessionId: sessionId)
            call.resolve(["decisions": decisions.map(\.jsonObject)])
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "list_decisions_failed")
        }
    }

    @objc public func getReviewProxy(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }

        store.getOrCreateReviewProxy(sessionId: sessionId, bridge: bridge) { result in
            DispatchQueue.main.async {
                switch result {
                case .success(let proxy):
                    self.notifyListeners("DiveSenseiMedia.reviewProxyUpdated", data: proxy.asDictionary)
                    call.resolve(["proxy": proxy.jsonObject])
                case .failure(let error as DiveSenseiPluginError):
                    call.reject(error.message, error.code)
                case .failure(let error):
                    call.reject(error.localizedDescription, "review_proxy_failed")
                }
            }
        }
    }

    @objc public func startExport(_ call: CAPPluginCall) {
        call.reject("Export is not implemented yet.", "export_unimplemented")
    }

    @objc public func listExports(_ call: CAPPluginCall) {
        call.resolve(["exports": []])
    }

    @objc public func deleteSession(_ call: CAPPluginCall) {
        guard let sessionId = call.getString("sessionId"), !sessionId.isEmpty else {
            call.reject("sessionId is required.", "invalid_session_id")
            return
        }

        do {
            let deleted = try store.deleteSession(sessionId: sessionId, deleteExports: call.getBool("deleteExports") ?? false)
            call.resolve(["deleted": deleted])
        } catch let error as DiveSenseiPluginError {
            call.reject(error.message, error.code)
        } catch {
            call.reject(error.localizedDescription, "delete_session_failed")
        }
    }
}

private struct DiveSenseiPluginError: Error {
    let code: String
    let message: String
}

private protocol JSONObjectEncodable: Encodable {}

private extension JSONObjectEncodable {
    var jsonObject: Any {
        do {
            let data = try JSONEncoder.divesensei.encode(self)
            return try JSONSerialization.jsonObject(with: data)
        } catch {
            return [:]
        }
    }

    var asDictionary: [String: Any] {
        jsonObject as? [String: Any] ?? [:]
    }
}

private extension JSONEncoder {
    static var divesensei: JSONEncoder {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.withoutEscapingSlashes]
        return encoder
    }
}

private extension JSONDecoder {
    static var divesensei: JSONDecoder { JSONDecoder() }
}

private enum SourceOrigin: String, Codable {
    case photos
    case files
}

private enum SourceAvailability: String, Codable {
    case available
    case needsDownload = "needs_download"
    case missing
    case permissionDenied = "permission_denied"
    case unsupported
}

private enum ReviewDecisionLabel: String, Codable {
    case keep
    case reject
    case unsure
}

private enum JobKind: String, Codable {
    case analysis
    case export
}

private enum JobStatus: String, Codable {
    case queued
    case running
    case completed
    case failed
    case cancelled
}

private enum SessionProfile: String, Codable {
    case longSession = "long-session"
    case reviewed
}

private enum DetectorId: String, Codable {
    case audioV1Heuristic = "audio_v1_heuristic"
    case audioV2PcenClassifier = "audio_v2_pcen_classifier"
    case audioV2HybridVideo = "audio_v2_hybrid_video"
}

private enum SessionStatus: String, Codable {
    case created
    case analyzing
    case reviewPending = "review_pending"
    case reviewReady = "review_ready"
    case exporting
    case completeWithErrors = "complete_with_errors"
    case failed
    case deleted
}

private enum JobPhase: String, Codable {
    case sourceResolving = "source_resolving"
    case sourceDownloading = "source_downloading"
    case audioDecode = "audio_decode"
    case detecting
    case manifestWriting = "manifest_writing"
    case proxyGenerating = "proxy_generating"
    case reviewReady = "review_ready"
    case exportPreparing = "export_preparing"
    case exporting
    case savingToLibrary = "saving_to_library"
    case completed
    case failed
    case cancelled
}

private enum ProxyStatus: String, Codable {
    case pending
    case ready
    case failed
}

private enum PlayerBackend: String, Codable {
    case htmlVideo = "html_video"
    case nativeAvPlayer = "native_avplayer"
}

private enum ConfidenceLevel: String, Codable {
    case high
    case medium
    case low
}

private struct StoredSourceRecord: Codable, JSONObjectEncodable {
    let sourceRef: String
    let origin: SourceOrigin
    let displayName: String
    let durationSeconds: Double?
    let fileSizeBytes: Int64?
    let canPersist: Bool
    let assetLocalIdentifier: String?
    let createdAt: String
    let lastResolvedAt: String

    func asSummaryPayload(availability: SourceAvailability = .available) -> SourceSummaryPayload {
        SourceSummaryPayload(
            sourceRef: sourceRef,
            origin: origin,
            displayName: displayName,
            availability: availability,
            durationSeconds: durationSeconds,
            fileSizeBytes: fileSizeBytes,
            canPersist: canPersist
        )
    }
}

private struct SourceSummaryPayload: Codable, JSONObjectEncodable {
    let sourceRef: String
    let origin: SourceOrigin
    let displayName: String
    let availability: SourceAvailability
    let durationSeconds: Double?
    let fileSizeBytes: Int64?
    let canPersist: Bool
}

private struct SourceAvailabilityResponsePayload: Codable, JSONObjectEncodable {
    let sourceRef: String
    let availability: SourceAvailability
    let origin: SourceOrigin?
    let displayName: String?
    let durationSeconds: Double?
    let fileSizeBytes: Int64?
    let lastResolvedAt: String?
}

private struct StoredSessionRecord: Codable, JSONObjectEncodable {
    let sessionId: String
    var sessionName: String
    var sourceRef: String
    let profile: SessionProfile
    let detectorId: DetectorId
    var status: SessionStatus
    var candidateCount: Int
    var keptCount: Int
    var rejectCount: Int
    var unsureCount: Int
    var exportCount: Int
    let createdAt: String
    var updatedAt: String
    var lastOpenedAt: String?
    let manifestFileName: String
    var reviewProxyFileName: String?
}

private struct CreateSessionResponsePayload: Codable, JSONObjectEncodable {
    let sessionId: String
    let status: SessionStatus
    let createdAt: String
}

private struct SessionLibraryItemPayload: Codable, JSONObjectEncodable {
    let sessionId: String
    let sessionName: String
    let sourceRef: String
    let sourceDisplayName: String?
    let sourceOrigin: SourceOrigin?
    let sourceAvailability: SourceAvailability
    let profile: SessionProfile
    let detectorId: DetectorId
    let status: SessionStatus
    let candidateCount: Int?
    let keptCount: Int?
    let rejectCount: Int?
    let unsureCount: Int?
    let exportCount: Int?
    let createdAt: String
    let updatedAt: String
    let lastOpenedAt: String?
}

private struct StoredTimestampRange: Codable, JSONObjectEncodable {
    let first: Double
    let last: Double
}

private struct StoredTelemetry: Codable, JSONObjectEncodable {
    let detector_seconds: Double
    let extract_seconds: Double
    let total_runtime_seconds: Double
    let peak_rss_kb: Int
}

private struct StoredSessionSummary: Codable, JSONObjectEncodable {
    let id: String
    let title: String
    let session_name: String?
    let profile: SessionProfile
    let detector_id: DetectorId?
    var status: SessionStatus
    let created_at: String?
    var updated_at: String?
    var session_duration_seconds: Double?
    var source_ref: String
    var source_origin: SourceOrigin?
    var source_display_name: String?
    var source_availability: SourceAvailability?
    var candidate_count: Int
    var extracted_count: Int
    let timestamp_range: StoredTimestampRange
    let telemetry: StoredTelemetry
}

private struct StoredDetectionScores: Codable, JSONObjectEncodable {
    let audio: Double
    let video: Double
    let combined: Double
    let audio_model_probability: Double
    let audio_clip_probability: Double
}

private struct StoredDetection: Codable, JSONObjectEncodable {
    let id: String
    let index: Int
    let timestamp_seconds: Double
    let start_time_seconds: Double
    let end_time_seconds: Double
    let duration_seconds: Double
    let review_start_seconds: Double?
    let review_end_seconds: Double?
    let review_duration_seconds: Double?
    let confidence: ConfidenceLevel
    let scores: StoredDetectionScores
    let features: [String: Double?]
    let export_ref: [String: String?]?
}

private struct StoredSessionManifest: Codable, JSONObjectEncodable {
    let schema_version: String
    let kind: String
    var generated_at: String
    var session: StoredSessionSummary
    let artifacts: [String: String]
    let detections: [StoredDetection]
}

private struct StoredDecisionRecord: Codable, JSONObjectEncodable {
    let id: String
    let sessionId: String
    let detectionId: String
    let label: ReviewDecisionLabel
    let notes: String
    let createdAt: String
    let updatedAt: String
}

private struct RepairSourceResponsePayload: Codable, JSONObjectEncodable {
    let repaired: Bool
    let source: SourceSummaryPayload?
    let availability: SourceAvailability
    let updatedAt: String
}

private struct ReviewProxyRecordPayload: Codable, JSONObjectEncodable {
    let sessionId: String
    let status: ProxyStatus
    let url: String?
    let durationSeconds: Double?
    let updatedAt: String
    let playerBackend: PlayerBackend
}

private struct StoredJobRecord: Codable, JSONObjectEncodable {
    let jobId: String
    let sessionId: String
    let kind: JobKind
    let phase: JobPhase
    let status: JobStatus
    let progress: Double?
    let message: String?
    let errorCode: String?
    let errorMessage: String?
    let startedAt: String?
    let updatedAt: String
    let finishedAt: String?
}

private enum DateTimestamp {
    static let formatter: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    static func isoNow() -> String {
        formatter.string(from: Date())
    }
}

private final class MobileJobStore {
    private var jobs: [String: StoredJobRecord] = [:]
    private var counter = 0

    func createAnalysisJob(sessionId: String, startedAt: String) -> StoredJobRecord {
        counter += 1
        let jobId = String(format: "job_%04d", counter)
        let job = StoredJobRecord(
            jobId: jobId,
            sessionId: sessionId,
            kind: .analysis,
            phase: .detecting,
            status: .running,
            progress: 0.05,
            message: "Placeholder native analysis started.",
            errorCode: nil,
            errorMessage: nil,
            startedAt: startedAt,
            updatedAt: startedAt,
            finishedAt: nil
        )
        jobs[jobId] = job
        return job
    }

    func get(jobId: String) -> StoredJobRecord? {
        jobs[jobId]
    }

    @discardableResult
    func completeJob(
        jobId: String,
        phase: JobPhase,
        status: JobStatus,
        progress: Double,
        message: String,
        finishedAt: String
    ) -> StoredJobRecord {
        let previous = jobs[jobId] ?? StoredJobRecord(
            jobId: jobId,
            sessionId: "",
            kind: .analysis,
            phase: phase,
            status: status,
            progress: progress,
            message: message,
            errorCode: nil,
            errorMessage: nil,
            startedAt: finishedAt,
            updatedAt: finishedAt,
            finishedAt: finishedAt
        )
        let updated = StoredJobRecord(
            jobId: previous.jobId,
            sessionId: previous.sessionId,
            kind: previous.kind,
            phase: phase,
            status: status,
            progress: progress,
            message: message,
            errorCode: previous.errorCode,
            errorMessage: previous.errorMessage,
            startedAt: previous.startedAt,
            updatedAt: finishedAt,
            finishedAt: finishedAt
        )
        jobs[jobId] = updated
        return updated
    }

    func cancel(jobId: String) -> Bool {
        guard let existing = jobs[jobId] else {
            return false
        }
        let updatedAt = DateTimestamp.isoNow()
        jobs[jobId] = StoredJobRecord(
            jobId: existing.jobId,
            sessionId: existing.sessionId,
            kind: existing.kind,
            phase: .cancelled,
            status: .cancelled,
            progress: existing.progress,
            message: existing.message,
            errorCode: nil,
            errorMessage: nil,
            startedAt: existing.startedAt,
            updatedAt: updatedAt,
            finishedAt: updatedAt
        )
        return true
    }
}

private final class MobilePersistenceStore {
    private let fileManager = FileManager.default
    private let rootURL: URL
    private let manifestsURL: URL
    private let decisionsURL: URL
    private let reviewProxiesURL: URL
    private let sourcesURL: URL
    private let sessionsURL: URL

    init() throws {
        guard let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first else {
            throw DiveSenseiPluginError(code: "storage_unavailable", message: "Application Support directory is unavailable.")
        }

        rootURL = appSupport.appendingPathComponent("DiveSenseiMobile", isDirectory: true)
        manifestsURL = rootURL.appendingPathComponent("manifests", isDirectory: true)
        decisionsURL = rootURL.appendingPathComponent("decisions", isDirectory: true)
        reviewProxiesURL = rootURL.appendingPathComponent("review-proxies", isDirectory: true)
        sourcesURL = rootURL.appendingPathComponent("sources.json")
        sessionsURL = rootURL.appendingPathComponent("sessions.json")

        try fileManager.createDirectory(at: rootURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: manifestsURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: decisionsURL, withIntermediateDirectories: true)
        try fileManager.createDirectory(at: reviewProxiesURL, withIntermediateDirectories: true)

        if !fileManager.fileExists(atPath: sourcesURL.path) {
            try write([String: StoredSourceRecord](), to: sourcesURL)
        }
        if !fileManager.fileExists(atPath: sessionsURL.path) {
            try write([String: StoredSessionRecord](), to: sessionsURL)
        }
    }

    func savePickedPhotoSource(assetIdentifier: String) throws -> StoredSourceRecord {
        guard photoAuthorizationAvailability() != .permissionDenied else {
            throw DiveSenseiPluginError(code: "permission_denied", message: "Photos access is denied.")
        }

        guard let asset = fetchAsset(localIdentifier: assetIdentifier) else {
            throw DiveSenseiPluginError(code: "asset_not_found", message: "Selected Photos asset could not be resolved.")
        }

        let now = DateTimestamp.isoNow()
        let resource = PHAssetResource.assetResources(for: asset).first
        let displayName = resource?.originalFilename ?? "Picked Video"
        let fileSize = resource.flatMap(Self.assetFileSize)
        let source = StoredSourceRecord(
            sourceRef: "src_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())",
            origin: .photos,
            displayName: displayName,
            durationSeconds: asset.duration,
            fileSizeBytes: fileSize,
            canPersist: true,
            assetLocalIdentifier: asset.localIdentifier,
            createdAt: now,
            lastResolvedAt: now
        )

        var sources = try loadSources()
        sources[source.sourceRef] = source
        try write(sources, to: sourcesURL)
        return source
    }

    func sourceAvailability(sourceRef: String) throws -> SourceAvailabilityResponsePayload {
        guard let source = try loadSources()[sourceRef] else {
            throw DiveSenseiPluginError(code: "source_not_found", message: "Source was not found.")
        }

        return SourceAvailabilityResponsePayload(
            sourceRef: source.sourceRef,
            availability: availability(for: source),
            origin: source.origin,
            displayName: source.displayName,
            durationSeconds: source.durationSeconds,
            fileSizeBytes: source.fileSizeBytes,
            lastResolvedAt: source.lastResolvedAt
        )
    }

    func createSession(sourceRef: String, sessionName: String?, profile: SessionProfile, detectorId: DetectorId) throws -> CreateSessionResponsePayload {
        let sources = try loadSources()
        guard let source = sources[sourceRef] else {
            throw DiveSenseiPluginError(code: "source_not_found", message: "Source was not found.")
        }

        let now = DateTimestamp.isoNow()
        let sessionId = "sess_\(UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased())"
        let title = sessionName?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
            ? sessionName!.trimmingCharacters(in: .whitespacesAndNewlines)
            : source.displayName.replacingOccurrences(of: #"\.[^.]+$"#, with: "", options: .regularExpression)
        let detections = placeholderDetections(duration: max(source.durationSeconds ?? 60, 18))
        let manifestFileName = "\(sessionId).json"
        let session = StoredSessionRecord(
            sessionId: sessionId,
            sessionName: title,
            sourceRef: sourceRef,
            profile: profile,
            detectorId: detectorId,
            status: .created,
            candidateCount: detections.count,
            keptCount: 0,
            rejectCount: 0,
            unsureCount: 0,
            exportCount: 0,
            createdAt: now,
            updatedAt: now,
            lastOpenedAt: nil,
            manifestFileName: manifestFileName,
            reviewProxyFileName: nil
        )

        var sessions = try loadSessions()
        sessions[sessionId] = session
        try write(sessions, to: sessionsURL)
        try write([StoredDecisionRecord](), to: decisionFileURL(sessionId: sessionId))

        let manifest = buildManifest(session: session, source: source, detections: detections, availability: availability(for: source))
        try write(manifest, to: manifestFileURL(fileName: manifestFileName))
        return CreateSessionResponsePayload(sessionId: sessionId, status: .created, createdAt: now)
    }

    func listSessions() throws -> [SessionLibraryItemPayload] {
        let sessions = try loadSessions().values.sorted { $0.updatedAt > $1.updatedAt }
        let sources = try loadSources()

        return sessions.map { session in
            let source = sources[session.sourceRef]
            let availability = source.map { self.availability(for: $0) } ?? .missing
            return SessionLibraryItemPayload(
                sessionId: session.sessionId,
                sessionName: session.sessionName,
                sourceRef: session.sourceRef,
                sourceDisplayName: source?.displayName,
                sourceOrigin: source?.origin,
                sourceAvailability: availability,
                profile: session.profile,
                detectorId: session.detectorId,
                status: session.status,
                candidateCount: session.candidateCount,
                keptCount: session.keptCount,
                rejectCount: session.rejectCount,
                unsureCount: session.unsureCount,
                exportCount: session.exportCount,
                createdAt: session.createdAt,
                updatedAt: session.updatedAt,
                lastOpenedAt: session.lastOpenedAt
            )
        }
    }

    func requireSession(sessionId: String) throws -> StoredSessionRecord {
        guard let session = try loadSessions()[sessionId] else {
            throw DiveSenseiPluginError(code: "session_not_found", message: "Session was not found.")
        }
        return session
    }

    @discardableResult
    func updateSessionStatus(sessionId: String, status: SessionStatus, updatedAt: String) throws -> StoredSessionRecord {
        var sessions = try loadSessions()
        guard var session = sessions[sessionId] else {
            throw DiveSenseiPluginError(code: "session_not_found", message: "Session was not found.")
        }

        session.status = status
        session.updatedAt = updatedAt
        sessions[sessionId] = session
        try write(sessions, to: sessionsURL)
        try refreshManifest(for: session)
        return session
    }

    @discardableResult
    func promoteSessionToReviewReady(sessionId: String, updatedAt: String) throws -> StoredSessionRecord {
        var sessions = try loadSessions()
        guard var session = sessions[sessionId] else {
            throw DiveSenseiPluginError(code: "session_not_found", message: "Session was not found.")
        }
        session.status = .reviewReady
        session.updatedAt = updatedAt
        sessions[sessionId] = session
        try write(sessions, to: sessionsURL)
        try refreshManifest(for: session)
        return session
    }

    func repairSessionSource(sessionId: String, source: StoredSourceRecord) throws -> RepairSourceResponsePayload {
        var sessions = try loadSessions()
        guard var session = sessions[sessionId] else {
            throw DiveSenseiPluginError(code: "session_not_found", message: "Session was not found.")
        }

        session.sourceRef = source.sourceRef
        session.status = .created
        session.updatedAt = DateTimestamp.isoNow()
        session.reviewProxyFileName = nil
        sessions[sessionId] = session
        try write(sessions, to: sessionsURL)

        let oldProxyURL = reviewProxiesURL.appendingPathComponent("\(sessionId).mov")
        if fileManager.fileExists(atPath: oldProxyURL.path) {
            try? fileManager.removeItem(at: oldProxyURL)
        }

        try refreshManifest(for: session)
        return RepairSourceResponsePayload(
            repaired: true,
            source: source.asSummaryPayload(availability: availability(for: source)),
            availability: availability(for: source),
            updatedAt: session.updatedAt
        )
    }

    func getSessionManifest(sessionId: String) throws -> StoredSessionManifest {
        var sessions = try loadSessions()
        guard var session = sessions[sessionId] else {
            throw DiveSenseiPluginError(code: "session_not_found", message: "Session was not found.")
        }

        session.lastOpenedAt = DateTimestamp.isoNow()
        sessions[sessionId] = session
        try write(sessions, to: sessionsURL)
        try refreshManifest(for: session)
        return try read(StoredSessionManifest.self, from: manifestFileURL(fileName: session.manifestFileName))
    }

    func saveDecision(sessionId: String, detectionId: String, label: ReviewDecisionLabel, notes: String) throws -> StoredDecisionRecord {
        var sessions = try loadSessions()
        guard var session = sessions[sessionId] else {
            throw DiveSenseiPluginError(code: "session_not_found", message: "Session was not found.")
        }

        let now = DateTimestamp.isoNow()
        var decisions = try listDecisions(sessionId: sessionId)
        let existing = decisions.first { $0.detectionId == detectionId }
        let decision = StoredDecisionRecord(
            id: existing?.id ?? "\(sessionId):\(detectionId)",
            sessionId: sessionId,
            detectionId: detectionId,
            label: label,
            notes: notes,
            createdAt: existing?.createdAt ?? now,
            updatedAt: now
        )

        decisions.removeAll { $0.detectionId == detectionId }
        decisions.append(decision)
        decisions.sort { $0.detectionId < $1.detectionId }
        try write(decisions, to: decisionFileURL(sessionId: sessionId))

        session.keptCount = decisions.filter { $0.label == .keep }.count
        session.rejectCount = decisions.filter { $0.label == .reject }.count
        session.unsureCount = decisions.filter { $0.label == .unsure }.count
        session.updatedAt = now
        sessions[sessionId] = session
        try write(sessions, to: sessionsURL)
        try refreshManifest(for: session)
        return decision
    }

    func listDecisions(sessionId: String) throws -> [StoredDecisionRecord] {
        _ = try requireSession(sessionId: sessionId)
        let url = decisionFileURL(sessionId: sessionId)
        if !fileManager.fileExists(atPath: url.path) {
            return []
        }
        return try read([StoredDecisionRecord].self, from: url)
    }

    func getOrCreateReviewProxy(
        sessionId: String,
        bridge: CAPBridgeProtocol?,
        completion: @escaping (Result<ReviewProxyRecordPayload, Error>) -> Void
    ) {
        do {
            let session = try requireSession(sessionId: sessionId)
            let sources = try loadSources()
            guard let source = sources[session.sourceRef] else {
                throw DiveSenseiPluginError(code: "source_not_found", message: "Source was not found.")
            }

            let availability = self.availability(for: source)
            guard availability == .available else {
                throw DiveSenseiPluginError(code: availability.rawValue, message: "Source is not currently playable.")
            }

            let proxyURL = reviewProxiesURL.appendingPathComponent("\(sessionId).mov")
            if fileManager.fileExists(atPath: proxyURL.path) {
                let payload = try reviewProxyPayload(session: session, source: source, proxyURL: proxyURL, bridge: bridge)
                completion(.success(payload))
                return
            }

            guard source.origin == .photos, let assetIdentifier = source.assetLocalIdentifier, let asset = fetchAsset(localIdentifier: assetIdentifier) else {
                throw DiveSenseiPluginError(code: "review_proxy_unavailable", message: "Only Photos-backed sources are supported for review proxy generation right now.")
            }

            let options = PHVideoRequestOptions()
            options.deliveryMode = .automatic
            options.isNetworkAccessAllowed = true
            options.version = .original

            PHImageManager.default().requestAVAsset(forVideo: asset, options: options) { avAsset, _, _ in
                do {
                    guard let avAsset else {
                        throw DiveSenseiPluginError(code: "asset_unavailable", message: "Photos could not resolve the selected video asset.")
                    }

                    if let urlAsset = avAsset as? AVURLAsset {
                        if self.fileManager.fileExists(atPath: proxyURL.path) {
                            try? self.fileManager.removeItem(at: proxyURL)
                        }
                        try self.fileManager.copyItem(at: urlAsset.url, to: proxyURL)
                        let payload = try self.reviewProxyPayload(session: session, source: source, proxyURL: proxyURL, bridge: bridge)
                        completion(.success(payload))
                        return
                    }

                    guard let exportSession = AVAssetExportSession(asset: avAsset, presetName: AVAssetExportPresetPassthrough) else {
                        throw DiveSenseiPluginError(code: "proxy_export_failed", message: "Unable to create a review proxy export session.")
                    }

                    if self.fileManager.fileExists(atPath: proxyURL.path) {
                        try? self.fileManager.removeItem(at: proxyURL)
                    }

                    exportSession.outputURL = proxyURL
                    exportSession.outputFileType = .mov
                    exportSession.shouldOptimizeForNetworkUse = true
                    exportSession.exportAsynchronously {
                        switch exportSession.status {
                        case .completed:
                            do {
                                let payload = try self.reviewProxyPayload(session: session, source: source, proxyURL: proxyURL, bridge: bridge)
                                completion(.success(payload))
                            } catch {
                                completion(.failure(error))
                            }
                        case .failed:
                            completion(.failure(exportSession.error ?? DiveSenseiPluginError(code: "proxy_export_failed", message: "Review proxy export failed.")))
                        case .cancelled:
                            completion(.failure(DiveSenseiPluginError(code: "proxy_export_cancelled", message: "Review proxy export was cancelled.")))
                        default:
                            completion(.failure(DiveSenseiPluginError(code: "proxy_export_failed", message: "Review proxy export did not complete.")))
                        }
                    }
                } catch {
                    completion(.failure(error))
                }
            }
        } catch {
            completion(.failure(error))
        }
    }

    func deleteSession(sessionId: String, deleteExports: Bool) throws -> Bool {
        var sessions = try loadSessions()
        guard let session = sessions.removeValue(forKey: sessionId) else {
            return false
        }

        try write(sessions, to: sessionsURL)
        try? fileManager.removeItem(at: manifestFileURL(fileName: session.manifestFileName))
        try? fileManager.removeItem(at: decisionFileURL(sessionId: sessionId))
        try? fileManager.removeItem(at: reviewProxiesURL.appendingPathComponent("\(sessionId).mov"))
        if deleteExports {
            // No-op for now.
        }
        return true
    }

    private func refreshManifest(for session: StoredSessionRecord) throws {
        let sources = try loadSources()
        guard let source = sources[session.sourceRef] else {
            throw DiveSenseiPluginError(code: "source_not_found", message: "Source was not found.")
        }

        let manifestURL = manifestFileURL(fileName: session.manifestFileName)
        var manifest = try read(StoredSessionManifest.self, from: manifestURL)
        let availability = self.availability(for: source)
        manifest.generated_at = DateTimestamp.isoNow()
        manifest.session.status = session.status
        manifest.session.updated_at = session.updatedAt
        manifest.session.session_duration_seconds = source.durationSeconds
        manifest.session.source_ref = source.sourceRef
        manifest.session.source_origin = source.origin
        manifest.session.source_display_name = source.displayName
        manifest.session.source_availability = availability
        manifest.session.candidate_count = session.candidateCount
        manifest.session.extracted_count = session.keptCount
        try write(manifest, to: manifestURL)
    }

    private func buildManifest(
        session: StoredSessionRecord,
        source: StoredSourceRecord,
        detections: [StoredDetection],
        availability: SourceAvailability
    ) -> StoredSessionManifest {
        StoredSessionManifest(
            schema_version: "1.0.0",
            kind: "divesensei.ui-session",
            generated_at: session.createdAt,
            session: StoredSessionSummary(
                id: session.sessionId,
                title: session.sessionName,
                session_name: session.sessionName,
                profile: session.profile,
                detector_id: session.detectorId,
                status: session.status,
                created_at: session.createdAt,
                updated_at: session.updatedAt,
                session_duration_seconds: source.durationSeconds,
                source_ref: source.sourceRef,
                source_origin: source.origin,
                source_display_name: source.displayName,
                source_availability: availability,
                candidate_count: detections.count,
                extracted_count: 0,
                timestamp_range: StoredTimestampRange(
                    first: detections.first?.timestamp_seconds ?? 0,
                    last: detections.last?.timestamp_seconds ?? 0
                ),
                telemetry: StoredTelemetry(
                    detector_seconds: 0,
                    extract_seconds: 0,
                    total_runtime_seconds: 0,
                    peak_rss_kb: 0
                )
            ),
            artifacts: [
                "review_mode": "placeholder",
                "manifest_storage": "native_persisted"
            ],
            detections: detections
        )
    }

    private func placeholderDetections(duration: Double) -> [StoredDetection] {
        let count: Int
        switch duration {
        case ..<35:
            count = 1
        case ..<80:
            count = 2
        default:
            count = 3
        }

        let normalizedPositions = [0.22, 0.51, 0.77]
        return Array(normalizedPositions.prefix(count).enumerated()).map { index, position in
            let timestamp = max(1.0, min(duration - 1.0, duration * position))
            let start = max(0.0, timestamp - 3.0)
            let end = min(duration, timestamp + 4.0)
            let reviewStart = max(0.0, timestamp - 2.0)
            let reviewEnd = min(duration, timestamp + 2.0)
            return StoredDetection(
                id: String(format: "det-%04d", index + 1),
                index: index + 1,
                timestamp_seconds: timestamp,
                start_time_seconds: start,
                end_time_seconds: end,
                duration_seconds: end - start,
                review_start_seconds: reviewStart,
                review_end_seconds: reviewEnd,
                review_duration_seconds: reviewEnd - reviewStart,
                confidence: index == 0 ? .high : (index == 1 ? .medium : .low),
                scores: StoredDetectionScores(
                    audio: 8.1 - Double(index) * 1.7,
                    video: 0,
                    combined: 8.1 - Double(index) * 1.7,
                    audio_model_probability: 0.9 - Double(index) * 0.16,
                    audio_clip_probability: 0.86 - Double(index) * 0.15
                ),
                features: [
                    "spectral_flux": 1.0 - Double(index) * 0.17,
                    "rms": 0.73 - Double(index) * 0.11
                ],
                export_ref: nil
            )
        }
    }

    private func availability(for source: StoredSourceRecord) -> SourceAvailability {
        switch source.origin {
        case .photos:
            return photoAuthorizationAvailability(source: source)
        case .files:
            return .unsupported
        }
    }

    private func photoAuthorizationAvailability(source: StoredSourceRecord? = nil) -> SourceAvailability {
        let status = PHPhotoLibrary.authorizationStatus(for: .readWrite)
        switch status {
        case .denied, .restricted:
            return .permissionDenied
        case .authorized, .limited:
            if let source, let assetIdentifier = source.assetLocalIdentifier {
                return fetchAsset(localIdentifier: assetIdentifier) == nil ? .missing : .available
            }
            return .available
        case .notDetermined:
            return .available
        @unknown default:
            return .unsupported
        }
    }

    private func reviewProxyPayload(
        session: StoredSessionRecord,
        source: StoredSourceRecord,
        proxyURL: URL,
        bridge: CAPBridgeProtocol?
    ) throws -> ReviewProxyRecordPayload {
        guard let portableURL = bridge?.portablePath(fromLocalURL: proxyURL)?.absoluteString else {
            throw DiveSenseiPluginError(code: "bridge_unavailable", message: "Unable to create a web-playable proxy URL.")
        }
        return ReviewProxyRecordPayload(
            sessionId: session.sessionId,
            status: .ready,
            url: portableURL,
            durationSeconds: source.durationSeconds,
            updatedAt: DateTimestamp.isoNow(),
            playerBackend: .htmlVideo
        )
    }

    private func fetchAsset(localIdentifier: String) -> PHAsset? {
        let result = PHAsset.fetchAssets(withLocalIdentifiers: [localIdentifier], options: nil)
        return result.firstObject
    }

    private func loadSources() throws -> [String: StoredSourceRecord] {
        try read([String: StoredSourceRecord].self, from: sourcesURL)
    }

    private func loadSessions() throws -> [String: StoredSessionRecord] {
        try read([String: StoredSessionRecord].self, from: sessionsURL)
    }

    private func manifestFileURL(fileName: String) -> URL {
        manifestsURL.appendingPathComponent(fileName)
    }

    private func decisionFileURL(sessionId: String) -> URL {
        decisionsURL.appendingPathComponent("\(sessionId).json")
    }

    private func write<T: Encodable>(_ value: T, to url: URL) throws {
        let data = try JSONEncoder.divesensei.encode(value)
        try data.write(to: url, options: [.atomic])
    }

    private func read<T: Decodable>(_ type: T.Type, from url: URL) throws -> T {
        let data = try Data(contentsOf: url)
        return try JSONDecoder.divesensei.decode(type, from: data)
    }

    private static func assetFileSize(resource: PHAssetResource) -> Int64? {
        if let size = resource.value(forKey: "fileSize") as? CLong {
            return Int64(size)
        }
        if let size = resource.value(forKey: "fileSize") as? Int64 {
            return size
        }
        if let size = resource.value(forKey: "fileSize") as? NSNumber {
            return size.int64Value
        }
        return nil
    }
}

private final class SourceVideoPickerController: NSObject, PHPickerViewControllerDelegate {
    private let store: MobilePersistenceStore
    private let completion: (Result<StoredSourceRecord?, Error>) -> Void
    private var retainedPicker: PHPickerViewController?

    init(
        store: MobilePersistenceStore,
        completion: @escaping (Result<StoredSourceRecord?, Error>) -> Void
    ) {
        self.store = store
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

        guard let assetIdentifier = result.assetIdentifier else {
            completion(.failure(DiveSenseiPluginError(code: "missing_asset_identifier", message: "Selected video is missing a Photos asset identifier.")))
            return
        }

        do {
            let source = try store.savePickedPhotoSource(assetIdentifier: assetIdentifier)
            completion(.success(source))
        } catch {
            completion(.failure(error))
        }
    }
}
