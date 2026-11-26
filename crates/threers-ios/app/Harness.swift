// The shell every threers-ios app shares, iOS and visionOS: a CAMetalLayer
// handed to Rust (which draws it with Metal), a display link asking it for
// frames, fingers forwarded.
// On visionOS: a volume, where RealityKit draws the app's meshes in the
// shared space (the Metal window instead with `-D THREERS_WINDOW`), and the
// immersive space, opened and closed when the app asks
// (`threers_ios::request`). On iOS with `-D THREERS_REALITY`, RealityKit
// draws the scene instead (Reality.swift's `RealityStage`), on the screen or
// over the camera. Nothing here is particular to an app; the app is the Rust
// crate (`threers_ios::app!`). Built by app/build.sh with swiftc, with
// Reality.swift beside it.

import SwiftUI
import UIKit
import AVFoundation
import UniformTypeIdentifiers
import QuickLook
import RealityKit
import Metal
#if os(visionOS)
import CompositorServices
#else
import ARKit
#endif

@_silgen_name("threers_view_create")
func threers_view_create(_ layer: UnsafeMutableRawPointer, _ width: UInt32, _ height: UInt32, _ mode: UInt32) -> UnsafeMutableRawPointer?
@_silgen_name("threers_view_set")
func threers_view_set(_ view: UnsafeMutableRawPointer?, _ key: UInt32, _ a: Float, _ b: Float, _ c: Float)
@_silgen_name("threers_view_frame")
func threers_view_frame(_ view: UnsafeMutableRawPointer?) -> Bool
@_silgen_name("threers_view_destroy")
func threers_view_destroy(_ view: UnsafeMutableRawPointer?)
@_silgen_name("threers_ios_take_request")
func threers_ios_take_request() -> UInt32
@_silgen_name("threers_ios_take_url")
func threers_ios_take_url(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int

@_silgen_name("threers_ios_rec_peek")
func threers_ios_rec_peek(_ w: UnsafeMutablePointer<UInt32>, _ h: UnsafeMutablePointer<UInt32>, _ index: UnsafeMutablePointer<UInt32>, _ fps: UnsafeMutablePointer<UInt32>, _ last: UnsafeMutablePointer<Bool>) -> Bool
@_silgen_name("threers_ios_rec_take")
func threers_ios_rec_take(_ out: UnsafeMutableRawPointer, _ cap: Int) -> Bool

@_silgen_name("threers_ios_take_pick")
func threers_ios_take_pick() -> Bool
@_silgen_name("threers_ios_picked")
func threers_ios_picked(_ path: UnsafePointer<UInt8>, _ len: Int)
@_silgen_name("threers_ios_bookmark")
func threers_ios_bookmark(_ path: UnsafePointer<UInt8>, _ len: Int, _ data: UnsafePointer<UInt8>, _ dataLen: Int)
@_silgen_name("threers_ios_take_bookmark")
func threers_ios_take_bookmark(_ out: UnsafeMutablePointer<UInt8>?, _ cap: Int) -> Int
@_silgen_name("threers_ios_bookmark_opened")
func threers_ios_bookmark_opened(_ data: UnsafePointer<UInt8>, _ dataLen: Int, _ path: UnsafePointer<UInt8>, _ len: Int, _ gone: Bool)
@_silgen_name("threers_ios_take_keyboard")
func threers_ios_take_keyboard() -> Int32
@_silgen_name("threers_ios_keyboard")
func threers_ios_keyboard(_ height: Float)
@_silgen_name("threers_ios_chords_kept")
func threers_ios_chords_kept() -> Bool

@_silgen_name("threers_ios_take_copy")
func threers_ios_take_copy(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int

@_silgen_name("threers_ios_copy_len")
func threers_ios_copy_len() -> Int

@_silgen_name("threers_ios_take_paste")
func threers_ios_take_paste() -> Bool
@_silgen_name("threers_ios_pasted")
func threers_ios_pasted(_ text: UnsafePointer<UInt8>, _ len: Int)

@_silgen_name("threers_ios_take_look_up")
func threers_ios_take_look_up(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int

@_silgen_name("threers_ios_take_share")
func threers_ios_take_share(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int

@_silgen_name("threers_ios_take_audio")
func threers_ios_take_audio(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int
@_silgen_name("threers_ios_audio_time")
func threers_ios_audio_time(_ t: Double, _ playing: Bool)

/// Each web page the app asked to open, opened with the system; and the
/// sound it asked to play, played (and where it is, told back).
@MainActor func openRequestedURLs() {
    var buf = [UInt8](repeating: 0, count: 4096)
    while true {
        let n = threers_ios_take_url(&buf, buf.count)
        guard n > 0, let s = String(bytes: buf[0..<n], encoding: .utf8), let url = URL(string: s) else { break }
        UIApplication.shared.open(url)
    }
    pollShares()
    pollCopy()
    pollPaste()
    pollLookUp()
    pollPick()
    pollRecording()
    pollAudio()
    pollQuickLook()
}

/// The controller to present a sheet from: the front window's top one.
///
/// On visionOS only a flat window's. A volume cannot hold a popover, and
/// asking one to present a share sheet ends the app; with no flat window open
/// there is nothing to present over, and the caller does without.
@MainActor private func frontController() -> UIViewController? {
    let active = UIApplication.shared.connectedScenes.compactMap { $0 as? UIWindowScene }.filter { $0.activationState == .foregroundActive }
    #if os(visionOS)
    let scene = active.first { $0.session.role == .windowApplication }
    #else
    let scene = active.first
    #endif
    guard var top = scene?.windows.first(where: \.isKeyWindow)?.rootViewController ?? scene?.windows.first?.rootViewController else { return nil }
    while let p = top.presentedViewController { top = p }
    return top
}

/// A `.usdz` the app asked to show (`threers_ios::quick_look`): AR Quick
/// Look on iOS (RealityKit drawing it, placed in the room), its own window
/// in the shared space on visionOS.
@MainActor private func pollQuickLook() {
    guard let url = takeQuickLook() else { return }
    #if os(visionOS)
    _ = PreviewApplication.open(urls: [url])
    #else
    guard let top = frontController() else { return }
    QuickLookSource.shared.url = url
    let preview = QLPreviewController()
    preview.dataSource = QuickLookSource.shared
    top.present(preview, animated: true)
    #endif
}

#if !os(visionOS)
/// The one file AR Quick Look shows.
@MainActor private final class QuickLookSource: NSObject, QLPreviewControllerDataSource {
    static let shared = QuickLookSource()
    var url: URL?
    nonisolated func numberOfPreviewItems(in controller: QLPreviewController) -> Int { 1 }
    nonisolated func previewController(_ controller: QLPreviewController, previewItemAt index: Int) -> any QLPreviewItem {
        MainActor.assumeIsolated {
            let item = ARQuickLookPreviewItem(fileAt: url ?? URL(fileURLWithPath: "/"))
            item.allowsContentScaling = true
            return item
        }
    }
}
#endif

/// The document picker the app asked for (`threers_ios::pick_file`): a
/// file or a folder, read where it is (its access kept while the app runs).
@MainActor private final class Picker: NSObject, UIDocumentPickerDelegate {
    static let shared = Picker()
    private var held: Set<URL> = []

    /// `url` kept open while the app runs, its bookmark given to the app;
    /// its path.
    private func hold(_ url: URL) -> [UInt8] {
        if !held.contains(url), url.startAccessingSecurityScopedResource() { held.insert(url) }
        let path = Array(url.path.utf8)
        if let data = try? url.bookmarkData() {
            let bytes = Array(data)
            threers_ios_bookmark(path, path.count, bytes, bytes.count)
        }
        return path
    }

    func documentPicker(_ controller: UIDocumentPickerViewController, didPickDocumentsAt urls: [URL]) {
        for url in urls {
            let path = hold(url)
            threers_ios_picked(path, path.count)
        }
    }

    /// A file picked at an earlier launch, found by its bookmark (off the
    /// main thread: its provider may be slow).
    func open(_ bytes: [UInt8]) {
        Task {
            let found = await Task.detached { () -> Result<URL, Error> in
                var stale = false
                return Result { try URL(resolvingBookmarkData: Data(bytes), bookmarkDataIsStale: &stale) }
            }.value
            switch found {
            case .success(let url):
                let path = hold(url)
                threers_ios_bookmark_opened(bytes, bytes.count, path, path.count, false)
            case .failure(let e):
                let why = e as NSError
                let gone = why.domain == NSCocoaErrorDomain && (why.code == NSFileNoSuchFileError || why.code == NSFileReadNoSuchFileError)
                threers_ios_bookmark_opened(bytes, bytes.count, [], 0, gone)
            }
        }
    }
}

@MainActor private func pollPick() {
    let n = threers_ios_take_bookmark(nil, 0)
    if n > 0 {
        var bytes = [UInt8](repeating: 0, count: n)
        if threers_ios_take_bookmark(&bytes, n) == n { Picker.shared.open(bytes) }
    }
    guard threers_ios_take_pick() else { return }
    guard let top = frontController() else { return }
    let picker = UIDocumentPickerViewController(forOpeningContentTypes: [.data, .folder, .zip], asCopy: false)
    picker.delegate = Picker.shared
    picker.allowsMultipleSelection = false
    top.present(picker, animated: true)
}

/// A turn of the view being written as an MP4 (`threers_ios::record_turn`):
/// each frame the app drew taken (BGRA) and appended at its index / rate;
/// after the last, the file finished and offered to share.
@MainActor private final class TurnRecorder {
    static var current: TurnRecorder?
    let writer: AVAssetWriter
    let input: AVAssetWriterInput
    let adaptor: AVAssetWriterInputPixelBufferAdaptor
    let url: URL

    init?(width: Int, height: Int) {
        url = FileManager.default.temporaryDirectory.appendingPathComponent("brain-turn-\(Int(Date().timeIntervalSince1970)).mp4")
        guard let w = try? AVAssetWriter(outputURL: url, fileType: .mp4) else { return nil }
        writer = w
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264,
            AVVideoWidthKey: width,
            AVVideoHeightKey: height,
        ])
        input.expectsMediaDataInRealTime = false
        adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input, sourcePixelBufferAttributes: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
            kCVPixelBufferWidthKey as String: width,
            kCVPixelBufferHeightKey as String: height,
        ])
        guard writer.canAdd(input) else { return nil }
        writer.add(input)
        guard writer.startWriting() else { return nil }
        writer.startSession(atSourceTime: .zero)
    }
}

@MainActor private func pollRecording() {
    var w: UInt32 = 0, h: UInt32 = 0, index: UInt32 = 0, fps: UInt32 = 30
    var last = false
    guard threers_ios_rec_peek(&w, &h, &index, &fps, &last) else { return }
    if TurnRecorder.current == nil {
        TurnRecorder.current = TurnRecorder(width: Int(w), height: Int(h))
    }
    guard let rec = TurnRecorder.current else {
        // no writer: the frame let go, so the recording ends
        var sink = [UInt8](repeating: 0, count: Int(w * h * 4))
        _ = threers_ios_rec_take(&sink, sink.count)
        return
    }
    guard rec.input.isReadyForMoreMediaData, let pool = rec.adaptor.pixelBufferPool else { return }
    var made: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, pool, &made)
    guard let pb = made else { return }
    CVPixelBufferLockBaseAddress(pb, [])
    let rowBytes = CVPixelBufferGetBytesPerRow(pb)
    let width = Int(w) * 4
    if let base = CVPixelBufferGetBaseAddress(pb) {
        if rowBytes == width {
            _ = threers_ios_rec_take(base, rowBytes * Int(h))
        } else {
            var tight = [UInt8](repeating: 0, count: width * Int(h))
            _ = threers_ios_rec_take(&tight, tight.count)
            tight.withUnsafeBytes { src in
                for y in 0..<Int(h) {
                    memcpy(base.advanced(by: y * rowBytes), src.baseAddress!.advanced(by: y * width), width)
                }
            }
        }
    }
    CVPixelBufferUnlockBaseAddress(pb, [])
    rec.adaptor.append(pb, withPresentationTime: CMTime(value: Int64(index), timescale: Int32(fps)))
    if last {
        TurnRecorder.current = nil
        rec.input.markAsFinished()
        let url = rec.url
        rec.writer.finishWriting {
            DispatchQueue.main.async { presentShare(url.path) }
        }
    }
}

/// Text the app copied (`threers_ios::copy_text`), on the clipboard.
@MainActor private func pollCopy() {
    // (as much room as it needs: a chapter copied is copied)
    let need = threers_ios_copy_len()
    guard need > 0 else { return }
    var buf = [UInt8](repeating: 0, count: need)
    let n = threers_ios_take_copy(&buf, buf.count)
    if n > 0, let s = String(bytes: buf[0..<n], encoding: .utf8) {
        UIPasteboard.general.string = s
    }
}

/// The clipboard's text, where the app asked for it (`threers_ios::ask_paste`).
@MainActor private func pollPaste() {
    guard threers_ios_take_paste() else { return }
    let text = Array((UIPasteboard.general.string ?? "").utf8)
    threers_ios_pasted(text, text.count)
}

/// A word the app asked to look up (`threers_ios::look_up`), in the system's
/// dictionaries.
@MainActor private func pollLookUp() {
    var buf = [UInt8](repeating: 0, count: 1024)
    let n = threers_ios_take_look_up(&buf, buf.count)
    guard n > 0, let word = String(bytes: buf[0..<n], encoding: .utf8) else { return }
    #if os(iOS)
    guard let top = frontController() else { return }
    let entry = UIReferenceLibraryViewController(term: word)
    // (an iPad's a sheet of its own)
    entry.modalPresentationStyle = .formSheet
    top.present(entry, animated: true)
    #endif
}

/// Each file the app offered (`threers_ios::share`), in the system's share
/// sheet over the front window.
@MainActor private func pollShares() {
    var buf = [UInt8](repeating: 0, count: 4096)
    let n = threers_ios_take_share(&buf, buf.count)
    guard n > 0, let path = String(bytes: buf[0..<n], encoding: .utf8) else { return }
    presentShare(path)
}

/// The share sheet for the file at `path`, over the front window.
@MainActor private func presentShare(_ path: String) {
    guard let top = frontController() else { return }
    let sheet = UIActivityViewController(activityItems: [URL(fileURLWithPath: path)], applicationActivities: nil)
    // an iPad's (and visionOS's) sheet is a popover: from the window's middle
    if let pop = sheet.popoverPresentationController {
        pop.sourceView = top.view
        pop.sourceRect = CGRect(x: top.view.bounds.midX, y: top.view.bounds.midY, width: 1, height: 1)
        pop.permittedArrowDirections = []
    }
    top.present(sheet, animated: true)
}

/// The sound the app plays (`threers_ios::audio`), and what it was loaded from.
@MainActor private var audioPlayer: AVPlayer?
@MainActor private var audioURL = ""

@MainActor private func pollAudio() {
    var buf = [UInt8](repeating: 0, count: 4096)
    let n = threers_ios_take_audio(&buf, buf.count)
    if n > 0, let s = String(bytes: buf[0..<n], encoding: .utf8) {
        let parts = s.split(separator: "\t", maxSplits: 2).map(String.init)
        if parts[0] == "pause" {
            audioPlayer?.pause()
        } else if parts.count == 3, let url = URL(string: parts[2]) {
            #if os(iOS) || os(visionOS)
            // heard with the silent switch on, as spoken audio (the
            // default, ambient, is muted by it: no sound at all)
            try? AVAudioSession.sharedInstance().setCategory(.playback, mode: .spokenAudio)
            try? AVAudioSession.sharedInstance().setActive(true)
            #endif
            if audioPlayer == nil || audioURL != parts[2] {
                audioPlayer = AVPlayer(url: url)
                audioURL = parts[2]
            }
            if let at = Double(parts[1]), at >= 0 {
                audioPlayer?.seek(to: CMTime(seconds: at, preferredTimescale: 600), toleranceBefore: .zero, toleranceAfter: .zero)
            }
            audioPlayer?.play()
        }
    }
    if let p = audioPlayer {
        let t = p.currentTime().seconds
        threers_ios_audio_time(t.isFinite ? t : 0, p.rate > 0)
    }
}
@_silgen_name("threers_ios_immersive")
func threers_ios_immersive(_ open: Bool)
@_silgen_name("threers_ios_backdrop")
func threers_ios_backdrop(_ open: Bool)
@_silgen_name("threers_ios_backdrop_open")
func threers_ios_backdrop_open() -> Bool
@_silgen_name("threers_view_requests")
func threers_view_requests(_ view: UnsafeMutableRawPointer?)
@_silgen_name("threers_view_controls")
func threers_view_controls(_ view: UnsafeMutableRawPointer?, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_ios_press")
func threersPressControl(_ key: UInt64)
@_silgen_name("threers_ios_set_dark")
func threers_ios_set_dark(_ dark: Bool)
@_silgen_name("threers_ios_legibility")
func threers_ios_legibility(_ bold: Bool, _ contrast: Bool, _ motion: Bool, _ transparency: Bool, _ textScale: Float)

/// The body text's size at each of the system's text sizes, over the
/// default's (17 points).
private let bodySizes: [UIContentSizeCategory: Float] = [
    .extraSmall: 14, .small: 15, .medium: 16, .large: 17, .extraLarge: 19, .extraExtraLarge: 21, .extraExtraExtraLarge: 23,
    .accessibilityMedium: 28, .accessibilityLarge: 33, .accessibilityExtraLarge: 40, .accessibilityExtraExtraLarge: 47, .accessibilityExtraExtraExtraLarge: 53,
]
private var legibilitySent: [Float] = []

/// Tells the app the system's accessibility settings, when they change
/// (cheap: once a frame).
@MainActor func sendLegibility() {
    let scale = (bodySizes[UIApplication.shared.preferredContentSizeCategory] ?? 17) / 17
    let now: [Float] = [UIAccessibility.isBoldTextEnabled, UIAccessibility.isDarkerSystemColorsEnabled, UIAccessibility.isReduceMotionEnabled, UIAccessibility.isReduceTransparencyEnabled].map { $0 ? 1 : 0 } + [scale]
    guard now != legibilitySent else { return }
    legibilitySent = now
    threers_ios_legibility(now[0] == 1, now[1] == 1, now[2] == 1, now[3] == 1, scale)
}

/// An element assistive technology finds over a control of the UI the
/// layer draws (VoiceOver, Voice Control, Switch Control): its name, what
/// it is, where; activated, it presses the control.
final class UiControlElement: UIAccessibilityElement {
    let key: UInt64

    init(container: Any, control c: UiControl) {
        key = UInt64(c.key) ?? 0
        super.init(accessibilityContainer: container)
        accessibilityLabel = c.label
        accessibilityValue = c.value
        accessibilityFrameInContainerSpace = CGRect(x: c.x, y: c.y, width: c.w, height: c.h)
        var traits: UIAccessibilityTraits = []
        switch c.role {
        case "link": traits.insert(.link)
        case "searchbox", "textbox": traits.insert(.searchField)
        case "slider": traits.insert(.adjustable)
        case "tab": traits.insert(.button)
        default: traits.insert(.button)
        }
        if c.toggle && c.on { traits.insert(.selected) }
        if c.disabled { traits.insert(.notEnabled) }
        accessibilityTraits = traits
    }

    override func accessibilityActivate() -> Bool {
        threersPressControl(key)
        return true
    }
}

/// The layer Rust draws in.
final class LayerView: UIView {
    override class var layerClass: AnyClass { CAMetalLayer.self }
}

/// The layer's view in a container (ARKit's camera view goes in beside it,
/// beneath), the display link, and the fingers.
final class Host: UIView {
    private let surface = LayerView()
    private var view: UnsafeMutableRawPointer?
    private var finger: UITouch?
    /// Frames are drawn on a thread of their own (`renderFrame`), so the
    /// main thread — UIKit's, the keyboard's — never waits on one; every
    /// call into the Rust view, from either thread, holds `lock`.
    private let lock = NSLock()
    private var rendering = false
    /// the render thread's generation: one let go stops at its next frame
    private var renderGen = 0
    /// what the render thread left for the main one: the UI's controls
    /// (as JSON), when they changed
    private var pendingControls: Data?
    /// Touches, pinches and pans as they came, for the render thread to put
    /// in at its next frame: held by `inputLock` alone, for as long as an
    /// append takes, so a gesture never waits on a frame being drawn (which
    /// held up every event, and UIKit's delivery of the next, by up to a
    /// frame: rotating and pinching lagged the fingers).
    private let inputLock = NSLock()
    private var pendingInput: [(UInt32, Float, Float, Float)] = []
    /// the main thread's own light tick: the controls and the app's
    /// requests (sheets, URLs, the room), which are UIKit's to act on
    private var poll: CADisplayLink?
    /// What the app asked of the shell: 3 enter the room, 4 leave it.
    var onRequest: (UInt32) -> Void = { _ in }
    /// The UI's controls, for assistive technology (the layer is one
    /// picture to it otherwise).
    private var controlElements: [UiControlElement] = []
    override var accessibilityElements: [Any]? {
        get { controlElements }
        set {}
    }

    init() {
        super.init(frame: .zero)
        backgroundColor = .black
        isMultipleTouchEnabled = true
        for (name, at) in [(UIApplication.willResignActiveNotification, 0), (UIApplication.didBecomeActiveNotification, 1)] as [(Notification.Name, Float)] {
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { [weak self] _ in
                MainActor.assumeIsolated { self?.queue(7, at, 0, 0) }
            }
        }
        #if os(iOS)
        // how much of the view the on-screen keyboard covers
        NotificationCenter.default.addObserver(forName: UIResponder.keyboardWillChangeFrameNotification, object: nil, queue: .main) { [weak self] n in
            let end = (n.userInfo?[UIResponder.keyboardFrameEndUserInfoKey] as? NSValue)?.cgRectValue ?? .null
            MainActor.assumeIsolated {
                guard let self, let window = self.window else { return }
                let r = self.convert(window.convert(end, from: nil), from: window)
                // (one floating, or split, covers no edge)
                let docked = !r.isNull && r.maxY >= self.bounds.maxY - 1
                threers_ios_keyboard(docked ? Float(max(0, self.bounds.maxY - r.minY)) : 0)
            }
        }
        NotificationCenter.default.addObserver(forName: UIResponder.keyboardWillHideNotification, object: nil, queue: .main) { _ in
            threers_ios_keyboard(0)
        }
        #endif
        surface.frame = bounds
        surface.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        surface.isUserInteractionEnabled = false
        addSubview(surface)
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(pinched(_:)))
        let pan = UIPanGestureRecognizer(target: self, action: #selector(panned(_:)))
        pan.minimumNumberOfTouches = 2
        for g in [pinch, pan] as [UIGestureRecognizer] {
            g.cancelsTouchesInView = false
            // both at once: two fingers pinch and move together (UIKit lets
            // one recognizer of a view run alone, and a two-finger drag is
            // nearly always a pinch first: the pan never came)
            g.delegate = self
            addGestureRecognizer(g)
        }
    }

    required init?(coder: NSCoder) { fatalError("made in code") }

    private var scale: CGFloat { traitCollection.displayScale }
    private var pixels: (UInt32, UInt32) {
        (UInt32(max(1, bounds.width * scale)), UInt32(max(1, bounds.height * scale)))
    }

    override func didMoveToWindow() {
        super.didMoveToWindow()
        guard window != nil else {
            poll?.invalidate()
            poll = nil
            locked { rendering = false }
            return
        }
        surface.layer.contentsScale = scale
        threers_ios_set_dark(traitCollection.userInterfaceStyle == .dark)
        locked {
            (surface.layer as! CAMetalLayer).drawableSize = CGSize(width: Double(pixels.0), height: Double(pixels.1))
            if view == nil {
                view = threers_view_create(Unmanaged.passUnretained(surface.layer).toOpaque(), pixels.0, pixels.1, 0)
            }
        }
        sendInsets()
        if poll == nil {
            let l = CADisplayLink(target: self, selector: #selector(tick))
            l.preferredFrameRateRange = CAFrameRateRange(minimum: 10, maximum: 30, preferred: 30)
            l.add(to: .main, forMode: .common)
            poll = l
        }
        startRendering()
        // a hardware keyboard's keys come here (a text field's, to it)
        becomeFirstResponder()
    }

    override var canBecomeFirstResponder: Bool { true }

    /// A hardware keyboard's key: its character (the special keys as the
    /// function-key code points, Escape as 27) and the modifiers, to the app.
    override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        var took = false
        for p in presses {
            // (a text field's come on to here as well: typed there already)
            if isFirstResponder, let key = p.key, press(key) {
                took = true
                heldKey = key.keyCode
            }
        }
        if !took { super.pressesBegan(presses, with: event) }
    }

    /// The key down, which repeats until it is let go.
    private var heldKey: UIKeyboardHIDUsage?

    private func letGo(_ presses: Set<UIPress>?) {
        guard let k = heldKey, presses?.contains(where: { $0.key?.keyCode == k }) ?? true else { return }
        heldKey = nil
        queue(8, 0, 0, 0)
    }

    override func pressesEnded(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        letGo(presses)
        super.pressesEnded(presses, with: event)
    }

    override func pressesCancelled(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        letGo(presses)
        super.pressesCancelled(presses, with: event)
    }

    override func resignFirstResponder() -> Bool {
        letGo(nil)
        return super.resignFirstResponder()
    }

    /// The shortcuts, ahead of the system's menus' own (⌘B and ⌘E never
    /// come as keys): ⌘ and a key is the app's, but a text field's editing
    /// ones where the app leaves them to it (`threers_ios::keep_chords`),
    /// and those that close or hide the window. Escape too.
    override var keyCommands: [UIKeyCommand]? {
        let kept = isFirstResponder || threers_ios_chords_kept()
        return Host.chords.filter { kept || !["a", "c", "v", "x", "z"].contains($0.input) }
    }

    private static let chords: [UIKeyCommand] = {
        var keys = "abcdefgijklnoprstuvxyz0123456789,.;'/\\[]-=`".map { String($0) }
        keys.append("\r")
        let held: [UIKeyModifierFlags] = [.command, [.command, .shift], [.command, .alternate]]
        var out = keys.flatMap { key in held.map { UIKeyCommand(input: key, modifierFlags: $0, action: #selector(chord(_:))) } }
        out.forEach { $0.wantsPriorityOverSystemBehavior = true }
        out.append(UIKeyCommand(input: UIKeyCommand.inputEscape, modifierFlags: [], action: #selector(chord(_:))))
        return out
    }()

    @objc private func chord(_ c: UIKeyCommand) {
        let code = c.input == UIKeyCommand.inputEscape ? 27 : c.input?.unicodeScalars.first?.value
        if let code { send(code, c.modifierFlags) }
    }

    private func send(_ code: UInt32, _ m: UIKeyModifierFlags) {
        let held: [(UIKeyModifierFlags, Int)] = [(.control, 1), (.alternate, 2), (.command, 4)]
        let bits = held.reduce(0) { m.contains($1.0) ? $0 | $1.1 : $0 }
        queue(6, Float(code), m.contains(.shift) ? 1 : 0, Float(bits))
    }

    /// `key` to the app, the modifiers with it; false for one that says
    /// nothing (a modifier alone).
    fileprivate func pressBackspace() { send(8, []) }

    fileprivate func press(_ key: UIKey) -> Bool {
        let s = key.charactersIgnoringModifiers
        let m = key.modifierFlags
        let code: UInt32
        switch s {
        case UIKeyCommand.inputEscape: code = 27
        case UIKeyCommand.inputUpArrow: code = 0xF700
        case UIKeyCommand.inputDownArrow: code = 0xF701
        case UIKeyCommand.inputLeftArrow: code = 0xF702
        case UIKeyCommand.inputRightArrow: code = 0xF703
        case UIKeyCommand.inputPageUp: code = 0xF72C
        case UIKeyCommand.inputPageDown: code = 0xF72D
        case UIKeyCommand.inputHome: code = 0xF729
        case UIKeyCommand.inputEnd: code = 0xF72B
        default:
            switch key.keyCode {
            case .keyboardDeleteOrBackspace: code = 8
            case .keyboardDeleteForward: code = 0xF728
            case .keyboardTab: code = 9
            default:
                // the character as typed (Shift's: `?`, `+`, `]`), not as
                // unshifted; a shortcut's, the key's own
                let typed = key.characters.isEmpty || !m.isDisjoint(with: [.command, .control]) ? s : key.characters
                // (a key with no character is named: `UIKeyInputF1`)
                guard !typed.hasPrefix("UIKeyInput"), let c = typed.unicodeScalars.first else { return false }
                code = c.value
            }
        }
        send(code, m)
        return true
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        locked {
            (surface.layer as! CAMetalLayer).drawableSize = CGSize(width: Double(pixels.0), height: Double(pixels.1))
            threers_view_set(view, 0, Float(pixels.0), Float(pixels.1), 0)
        }
        sendInsets()
    }

    /// The system's light or dark, for the app's UI (its System theme).
    override func traitCollectionDidChange(_ previous: UITraitCollection?) {
        super.traitCollectionDidChange(previous)
        threers_ios_set_dark(traitCollection.userInterfaceStyle == .dark)
    }

    override func safeAreaInsetsDidChange() {
        super.safeAreaInsetsDidChange()
        sendInsets()
    }

    /// The UI's points and the safe area, in pixels.
    private func sendInsets() {
        let i = safeAreaInsets
        let k = Float(scale)
        locked {
            threers_view_set(view, 4, k, 0, 0)
            threers_view_set(view, 5, Float(i.top) * k, Float(i.bottom) * k, Float(max(i.left, i.right)) * k)
        }
    }

    /// A native text field over each of the UI's text controls (a search
    /// box, a text area), by its key: a tap on it brings up the on-screen
    /// keyboard, and what it holds goes to the control. Clear itself — the
    /// UI draws the text, the caret and the focus.
    private var fields: [String: ControlField] = [:]

    /// The render thread's frame rate: up to 60.
    static let frames = CAFrameRateRange(minimum: 30, maximum: 60, preferred: 60)

    private func placeFields(_ list: [UiControl]) {
        let edits = list.filter { $0.edit }
        for c in edits {
            let f = fields[c.key] ?? {
                let f = ControlField(key: c.key, multiline: c.role == "textbox" && !c.secret)
                f.addTarget(self, action: #selector(typed(_:)), for: .editingChanged)
                f.delegate = self
                addSubview(f)
                fields[c.key] = f
                return f
            }()
            // short of a button inside it (a search box's clear ✕): that
            // stays the UI's to press
            var right = c.x + c.w
            for o in list where !o.edit && o.y >= c.y && o.y + o.h <= c.y + c.h && o.x > c.x && o.x < right {
                right = o.x
            }
            f.frame = CGRect(x: c.x, y: c.y, width: max(1, right - c.x), height: c.h)
            f.accessibilityLabel = c.label
            f.keepSecret(c.secret)
            // a text the UI changed itself (its clear button): the field follows
            // (a password there is stays as it was typed here)
            if !f.isFirstResponder, !c.secret || c.value != nil { f.whole = c.value ?? "" }
        }
        let keep = Set(edits.map { $0.key })
        for (k, f) in fields where !keep.contains(k) {
            f.removeFromSuperview()
            fields[k] = nil
        }
    }

    @objc fileprivate func typed(_ f: ControlField) {
        guard let k = UInt64(f.key) else { return }
        var bytes = Array(f.whole.utf8)
        threers_ios_text(k, &bytes, bytes.count)
    }

    /// `f` holding the lock every call into the Rust view holds.
    private func locked<T>(_ f: () -> T) -> T {
        lock.lock()
        defer { lock.unlock() }
        return f()
    }

    /// The render thread, its display link on its own run loop, once.
    private func startRendering() {
        let gen = locked { () -> Int? in
            if rendering { return nil }
            rendering = true
            renderGen += 1
            return renderGen
        }
        guard let gen else { return }
        let target = RenderTarget(host: self, gen: gen)
        let thread = Thread {
            let l = CADisplayLink(target: target, selector: #selector(RenderTarget.tick(_:)))
            l.preferredFrameRateRange = Host.frames
            l.add(to: .current, forMode: .common)
            RunLoop.current.run()
        }
        thread.name = "threers render"
        thread.qualityOfService = .userInteractive
        thread.start()
    }

    /// One frame, on the render thread; the link let go once the view is.
    fileprivate func renderFrame(_ l: CADisplayLink, _ gen: Int) {
        lock.lock()
        guard rendering, gen == renderGen, view != nil else {
            lock.unlock()
            l.invalidate()
            CFRunLoopStop(CFRunLoopGetCurrent())
            return
        }
        inputLock.lock()
        let input = pendingInput
        pendingInput.removeAll(keepingCapacity: true)
        inputLock.unlock()
        for (key, a, b, c) in input {
            threers_view_set(view, key, a, b, c)
        }
        _ = threers_view_frame(view)
        var len = 0
        if let p = threers_view_controls(view, &len) {
            pendingControls = Data(bytes: p, count: len)
        }
        lock.unlock()
    }

    /// The main thread's tick: the UI's controls as native elements and
    /// fields, and the app's requests.
    @objc private func tick() {
        // a frame being drawn: next tick (the main thread does not wait)
        guard lock.try() else { return }
        lock.unlock()
        let json = locked { () -> Data? in
            // a change to AR or back: UIKit's, so here on the main thread
            threers_view_requests(view)
            defer { pendingControls = nil }
            return pendingControls
        }
        sendLegibility()
        // the keyboard, where the app asked for it or is done with it
        switch threers_ios_take_keyboard() {
        case 1: fields.values.min { ($0.frame.minY, $0.frame.minX) < ($1.frame.minY, $1.frame.minX) }?.becomeFirstResponder()
        case 2: fields.values.first { $0.isFirstResponder }?.resignFirstResponder()
        default: break
        }
        if let json, let list = try? JSONDecoder().decode([UiControl].self, from: json) {
            controlElements = list.map { UiControlElement(container: self, control: $0) }
            placeFields(list)
            UIAccessibility.post(notification: .layoutChanged, argument: nil)
        }
        let r = threers_ios_take_request()
        if r != 0 { onRequest(r) }
        openRequestedURLs()
    }

    private func send(_ touch: UITouch, _ phase: Float) {
        let p = touch.location(in: self)
        let k = scale
        queue(1, phase, Float(p.x * k), Float(p.y * k))
    }

    /// An input for the next frame, as it came: the crate merges a pinch's
    /// steps and a pan's moves (`gestures.rs`).
    private func queue(_ key: UInt32, _ a: Float, _ b: Float, _ c: Float) {
        inputLock.lock()
        defer { inputLock.unlock() }
        pendingInput.append((key, a, b, c))
    }

    /// The fingers on the view now (not lifted).
    private func down(_ event: UIEvent?, _ touches: Set<UITouch>) -> Int {
        event?.allTouches?.filter { $0.phase != .ended && $0.phase != .cancelled }.count ?? touches.count
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        // one finger alone turns the brain (or works the UI); two — landing
        // together, in one event, or the second after — are the pinch's and
        // the pan's, and the one turning stops (it went on turning under a
        // pinch whose fingers came down together)
        if finger == nil, down(event, touches) == 1, let t = touches.first {
            finger = t
            send(t, 0)
        } else if let f = finger {
            send(f, 3)
            finger = nil
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard let f = finger, touches.contains(f) else { return }
        if down(event, touches) > 1 {
            send(f, 3)
            finger = nil
            return
        }
        send(f, 1)
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { send(f, 2); finger = nil }
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { send(f, 3); finger = nil }
    }

    /// THREERS_TOUCH_LOG: the gestures as UIKit reports them
    private static let touchLog = ProcessInfo.processInfo.environment["THREERS_TOUCH_LOG"] != nil

    @objc private func pinched(_ g: UIPinchGestureRecognizer) {
        if Host.touchLog { print("touchlog swift pinch state \(g.state.rawValue) scale \(g.scale) touches \(g.numberOfTouches)") }
        let z = Float(g.scale)
        // where the fingers are: the zoom goes toward what is under them
        let at = g.location(in: self)
        // (lifted to one finger, the place jumps to it: not said)
        if g.numberOfTouches >= 2, z > 0 { queue(2, z, Float(at.x * scale), Float(at.y * scale)) }
        g.scale = 1
        // the fingers lifted: the pinch over
        if g.state == .ended || g.state == .cancelled || g.state == .failed { queue(2, 0, 0, 0) }
    }

    @objc private func panned(_ g: UIPanGestureRecognizer) {
        if Host.touchLog { print("touchlog swift pan state \(g.state.rawValue) \(g.translation(in: self)) touches \(g.numberOfTouches)") }
        let d = g.translation(in: self)
        let k = scale
        queue(3, Float(d.x * k), Float(d.y * k), 0)
        g.setTranslation(.zero, in: self)
    }

    deinit {
        poll?.invalidate()
        // the render thread stops at its next frame (no view)
        locked {
            rendering = false
            threers_view_destroy(view)
            view = nil
        }
    }
}

struct HostView: UIViewRepresentable {
    var onRequest: (UInt32) -> Void = { _ in }
    func makeUIView(context: Context) -> Host {
        let h = Host()
        h.onRequest = onRequest
        return h
    }
    func updateUIView(_ uiView: Host, context: Context) { uiView.onRequest = onRequest }
}

#if os(visionOS)
@_silgen_name("threers_app_run")
func threers_app_run(_ layerRenderer: UnsafeMutableRawPointer)
@_silgen_name("threers_ios_spatial_event")
func threers_ios_spatial_event(_ id: UInt64, _ phase: UInt32, _ ox: Float, _ oy: Float, _ oz: Float, _ dx: Float, _ dy: Float, _ dz: Float)

/// Both eyes in one texture array, a slice each, drawn in one pass where
/// the device can (threers' Metal renderer draws either); else a texture
/// with a viewport each.
struct EyesLayout: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {
        let layouts = capabilities.supportedLayouts(options: [])
        configuration.layout = layouts.contains(.layered) ? .layered : .shared
        configuration.colorFormat = .bgra8Unorm_srgb
        configuration.depthFormat = .depth32Float
        configuration.isFoveationEnabled = false
    }
}

/// The flat window (`THREERS_WINDOW` builds): the Metal view, as on iOS.
struct FlatWindow: View {
    @Environment(\.openImmersiveSpace) private var open
    @Environment(\.dismissImmersiveSpace) private var dismiss
    var body: some View {
        HostView { r in Task { @MainActor in await shellRequest(r, open: open, dismiss: dismiss) } }
    }
}

/// What the app asked of the shell: into the room (3) or out of it (4).
@MainActor func shellRequest(_ r: UInt32, open: OpenImmersiveSpaceAction, dismiss: DismissImmersiveSpaceAction) async {
    // one immersive space at a time: the room, the backdrop or the pieces
    func close() async {
        await dismiss()
        threers_ios_immersive(false)
        threers_ios_backdrop(false)
        threers_ios_space(false)
    }
    switch r {
    case 3, 5, 7:
        #if THREERS_WINDOW
        // the pieces are the volume's app's, which this build has not
        if r == 7 { return }
        #endif
        if threers_ios_immersive_open() || threers_ios_backdrop_open() || threers_ios_space_open() { await close() }
        let id = r == 3 ? "immersive" : r == 5 ? "backdrop" : "pieces"
        let result = await open(id: id)
        print("threers: \(id) space: \(result)")
        guard case .opened = result else { return }
        switch r {
        case 3: threers_ios_immersive(true)
        case 5: threers_ios_backdrop(true)
        default: threers_ios_space(true)
        }
    case 4, 6, 8:
        await close()
    default:
        break
    }
}

/// The backdrop: the system's own darkening of your surroundings
/// (`preferredSurroundingsEffect`, which takes an open immersive space)
/// behind the app's volume and windows — what is small and bright (a
/// neuron's branches) reads against it. Nothing of the app's own in it;
/// an environment you chose yourself (Control Center) stays.
struct Backdrop: View {
    var body: some View {
        RealityView { _ in }
            .preferredSurroundingsEffect(.ultraDark)
            .onDisappear { threers_ios_backdrop(false) }
    }
}

// --- the volume: RealityKit draws the app's meshes in the shared space -------

// (the volume's C functions: Reality.swift)
@_silgen_name("threers_ios_immersive_open") func threers_ios_immersive_open() -> Bool
@_silgen_name("threers_ios_volume_shown") func threers_ios_volume_shown(_ shown: Bool)

/// One of the UI's controls (`threers_ui::native`), in the image's points:
/// a native view stands over it, for the system's look highlight,
/// VoiceOver, and presses that are that control's.
struct NativeControl: Decodable, Identifiable {
    let key: String
    let role: String
    let label: String
    let x, y, w, h, r: CGFloat
    let on, toggle, disabled: Bool
    /// Typed into (a search box): a native text field stands over it.
    let edit: Bool
    /// It holds a password: the field over it is a secure one. Its value is
    /// not given: nil while there is one, empty when there is none.
    let secret: Bool
    let value: String?
    /// Slid: a view over it shows the look's highlight and passes the drag on.
    let slide: Bool
    /// The UI's accent, 0–1 sRGB: the highlight's colour.
    let tint: [Double]
    /// What it is painted on, and a slider's thumb (Pieces.swift).
    let fill: [Double]?
    let thumb: [CGFloat]?
    /// What of it is raised: its box and corners.
    let solid: [CGFloat]?
    /// A slider's rail: its bars, `[x, y, w, h, r, g, b, a]`, track first.
    let rails: [[CGFloat]]?
    /// A knob's turn, radians right of up.
    let dial: Double?
    var id: String { key }
    var color: Color { Color(.sRGB, red: tint[0], green: tint[1], blue: tint[2]) }
}

/// What the look lights on a control: its shape in the UI's accent, a
/// faint fill and an edge, as the web app's hover does. The system shows it
/// only while the control is looked at; the app never learns where the
/// eyes are.
struct Gaze: View {
    let c: NativeControl
    let k: CGFloat
    var body: some View {
        let shape = RoundedRectangle(cornerRadius: max(c.r * k, 4))
        ZStack {
            shape.fill(c.color.opacity(0.12))
            shape.strokeBorder(c.color.opacity(0.85), lineWidth: 1.5)
        }
        .frame(width: c.w * k, height: c.h * k)
        .hoverEffect { effect, isActive, _ in effect.opacity(isActive ? 1 : 0) }
    }
}

/// The volume's side, metres; the model fills most of it.
let volumeSide: Float = 0.6
/// The app's UI surfaces beside the volume (`threers_ios::volume::SURFACES`):
/// their images' pixels, shown at half (points), and image pixels a UI
/// pixel (2.8: a UI pixel is 1.4 points, so a web layout's 13-pixel type
/// reads at a volume's distance). 0: the card, a sidebar beside the
/// volume, 476 × 1008 points; 1: the toolbar under it, 700 × 90. Their
/// windows open at that and resize: the surfaces follow
/// (`VolumeModel.uiSizes`).
let uiDefaultSizes: [(w: Int, h: Int)] = [(952, 2016), (1400, 180)]
let uiScales: [Float] = [2.8, 2.8]
/// Each press and drag on the UI surfaces, to the console (`THREERS_UI_LOG=1`).
let uiLog = ProcessInfo.processInfo.environment["THREERS_UI_LOG"] != nil


/// The app's meshes as entities, kept in step with its scene each frame:
/// a mesh rebuilt when its geometry changes, a material when its material does.
@MainActor final class VolumeModel: ObservableObject {
    /// One: the volume and the windows of its UI share it
    static let shared = VolumeModel()
    let vol = threers_vol_create()
    /// fits the model to the volume
    let root = Entity()
    /// the model's frame: its focus at the centre, turned by drags
    let model = Entity()
    /// what drags land on: the volume's own box, unscaled (under `root`,
    /// scaled to the model, it grew past the volume's walls, where the
    /// system clips content and hits nothing, so drags never turned it)
    let hit = Entity()
    /// The whole scene as threers draws it (its own Metal renderer: glass,
    /// clip planes, shaders, all as in a window), on one quad facing the
    /// viewer: what RealityKit shows instead of rebuilding each mesh
    let quad = ModelEntity()
    private var quadTexture: LowLevelTexture?
    private let queue = MTLCreateSystemDefaultDevice()?.makeCommandQueue()
    /// sharp at arm's length, mipmapped for further off
    /// The quad's picture's side in pixels: about what it covers on the
    /// display (a 0.6 m volume, a metre or so away: a thousand), so what is
    /// drawn in pixels — a line's one, a point's few — shows that big, not
    /// halved and washed out by the mipmaps; a bigger volume, more.
    private var quadPixels = 1024
    private static func pixels(for side: Float) -> Int { side > 0.8 ? 2048 : 1024 }
    /// The app's meshes, lights and environment as entities (Reality.swift).
    private(set) lazy var scene = RealityScene(vol: vol, model: model)
    /// The parts of the UI that stand by themselves (Pieces.swift).
    let pieces = Pieces()
    /// What each text control's native field holds, by key.
    @Published var texts: [String: String] = [:]
    /// Each surface's image (nil: not shown) and its controls.
    @Published var ui: [UIImage?] = [nil, nil]
    @Published var controls: [[NativeControl]] = [[], []]
    /// Each surface's size now (image pixels): its window's.
    @Published var uiSizes = uiDefaultSizes
    /// The drag box is a sheet at the back.
    private var thinHit = false
    /// The volume's side now (metres): its window's, resized.
    private(set) var side = volumeSide
    var onRequest: (UInt32) -> Void = { _ in }

    init() {
        root.addChild(model)
        // drags land anywhere in the volume, a little inside its walls
        hit.components.set(CollisionComponent(shapes: [.generateBox(size: SIMD3(repeating: volumeSide * 0.95))]))
        hit.components.set(InputTargetComponent())
    }

    /// The volume's window resized to `side` metres (its shortest edge):
    /// the picture, the drag box and the model scale with it.
    func resize(_ side: Float) {
        guard side.isFinite, side > 0.05, abs(side - self.side) > 1e-4 else { return }
        self.side = side
        quad.scale = SIMD3(repeating: side / volumeSide)
        hit.scale = quad.scale
        let n = Self.pixels(for: side)
        if quadTexture != nil, n != quadPixels { Task { await makeQuad(n) } }
    }

    /// Surface `which`'s window is `size` points now: its image follows,
    /// in whole pixels (two a point).
    func resizeUI(_ which: Int, _ size: CGSize) {
        let px = (w: Int(size.width.rounded()) * 2, h: Int(size.height.rounded()) * 2)
        guard px.w >= 64, px.h >= 64, px != uiSizes[which] else { return }
        uiSizes[which] = px
    }

    private var loaded = false
    func load() async {
        // (the volume's view, and the space the pieces stand in)
        if loaded { return }
        loaded = true
        await scene.load()
        // threers' picture on a quad, unless the app asks for entities
        // (`VolumeDrawing::Entities`): RealityKit then draws each mesh
        if threers_vol_drawing(vol) == 0 { await makeQuad(Self.pixels(for: side)) }
    }

    /// The quad, its picture `n` pixels square.
    private func makeQuad(_ n: Int) async {
        let desc = LowLevelTexture.Descriptor(textureType: .type2D, pixelFormat: .rgba8Unorm_srgb, width: n, height: n, depth: 1, mipmapLevelCount: Int(log2(Double(n))) + 1, textureUsage: [.shaderRead, .shaderWrite])
        if queue != nil, let texture = try? LowLevelTexture(descriptor: desc), let resource = try? await TextureResource(from: texture) {
            var m = UnlitMaterial()
            m.color = .init(tint: .white, texture: .init(resource))
            m.blending = .transparent(opacity: .init(floatLiteral: 1))
            // a little inside the walls: turned to face you, its corners
            // must not reach past them, where the system clips
            let side = volumeSide * 0.9
            quad.model = ModelComponent(mesh: .generatePlane(width: side, height: side), materials: [m])
            quad.components.set(BillboardComponent())
            quadTexture = texture
            quadPixels = n
        }
    }

    /// The scene into the quad's texture, by threers, this frame.
    private func drawQuad() -> Bool {
        guard let texture = quadTexture, let cb = queue?.makeCommandBuffer() else { return false }
        let target = texture.replace(using: cb)
        let ok = threers_vol_draw(vol, 0, 0, 0, 1, UInt32(quadPixels),
                                  Unmanaged.passUnretained(target as AnyObject).toOpaque(),
                                  Unmanaged.passUnretained(cb as AnyObject).toOpaque())
        if ok, let blit = cb.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: target)
            blit.endEncoding()
        }
        cb.commit()
        return ok
    }

    func turn(_ radians: Float) { threers_vol_turn(vol, radians) }

    /// A tap in the volume at `p` (the scene's metres, its centre at the
    /// quad's): where on the quad's picture, for the app to pick there.
    func tap(_ p: SIMD3<Float>) {
        let side = self.side * 0.9
        threers_vol_tap(vol, p.x / side + 0.5, p.y / side + 0.5)
    }

    /// The native field over text control `c` holds `text` now: to the UI.
    func type(_ c: NativeControl, _ text: String) {
        texts[c.key] = text
        guard let k = UInt64(c.key) else { return }
        var bytes = Array(text.utf8)
        threers_ios_text(k, &bytes, bytes.count)
    }

    func press(_ c: NativeControl) {
        // the release that ends a drag (a list scrolled from a row) is not a press
        if dragging || Date().timeIntervalSince(dragEnded) < 0.3 {
            if uiLog { print("ui: press \"\(c.label)\" dropped: the end of a drag") }
            return
        }
        if uiLog { print("ui: press \"\(c.label)\" (\(c.role)) at \(Int(c.x)),\(Int(c.y))") }
        if let k = UInt64(c.key) { threers_ios_press(k) }
    }

    /// A finger on surface `which` (a pinch held, moved): to the UI as a
    /// touch, in the image's pixels, so sliders drag and lists scroll.
    private var touching = false
    private var dragging = false
    private var dragEnded = Date.distantPast
    func drag(_ which: UInt32, _ v: DragGesture.Value) {
        let k = Float(2)
        if !touching {
            touching = true
            if uiLog { print("ui: drag on \(which) from \(Int(v.startLocation.x)),\(Int(v.startLocation.y)) pt") }
            threers_ios_touch(which, 0, Float(v.startLocation.x) * k, Float(v.startLocation.y) * k)
        }
        if hypot(v.translation.width, v.translation.height) > 6 { dragging = true }
        threers_ios_touch(which, 1, Float(v.location.x) * k, Float(v.location.y) * k)
    }
    func dragEnd(_ which: UInt32, _ v: DragGesture.Value) {
        let k = Float(2)
        if !touching { threers_ios_touch(which, 0, Float(v.startLocation.x) * k, Float(v.startLocation.y) * k) }
        threers_ios_touch(which, 2, Float(v.location.x) * k, Float(v.location.y) * k)
        if uiLog { print("ui: drag on \(which) ended, moved \(Int(v.translation.width)),\(Int(v.translation.height)) pt") }
        if dragging { dragEnded = Date() }
        touching = false
        dragging = false
    }

    private var ticked = 0.0
    func tick() {
        // once a frame, though the volume and the pieces' space both ask
        let now = CACurrentMediaTime()
        guard now - ticked > 0.004 else { return }
        ticked = now
        sendLegibility()
        let n = threers_vol_frame(vol)
        pieces.sync(scene)
        // what drags the model lands behind the pieces among it, not before
        if pieces.inVolume != thinHit {
            thinHit = pieces.inVolume
            let s = volumeSide * 0.95
            hit.components.set(CollisionComponent(shapes: [thinHit
                ? .generateBox(size: SIMD3(s, s, 0.01)).offsetBy(translation: SIMD3(0, 0, -s / 2))
                : .generateBox(size: SIMD3(repeating: s))]))
        }
        var r = [Float](repeating: 0, count: 17)
        threers_vol_root(vol, &r)
        r.withUnsafeBufferPointer { model.transform = Transform(matrix: rkMatrix($0.baseAddress!)) }
        root.scale = SIMD3(repeating: side * 0.45 / max(r[16], 1e-4))
        // drawn whole by threers: no meshes of RealityKit's to keep in step
        if drawQuad() {
            root.isEnabled = false
            quad.isEnabled = !threers_ios_immersive_open()
            uiAndRequests()
            return
        }
        // the model at the volume's scale: a light's reach and power with it
        scene.metres = root.scale.x
        scene.sync(n)
        // the room shows the model while it is open: one brain, not two
        model.isEnabled = !threers_ios_immersive_open()
        uiAndRequests()
    }

    /// The app's UI surfaces, each when it changed; what it asked of the shell.
    private func uiAndRequests() {
        for i in uiSizes.indices {
            switch uiImage(i) {
            case .some(.some(let image)): ui[i] = image
            case .some(.none): ui[i] = nil
            case .none: break
            }
            var len = 0
            if let p = threers_vol_controls(vol, UInt32(i), &len),
               let list = try? JSONDecoder().decode([NativeControl].self, from: Data(bytes: p, count: len)) {
                controls[i] = list
                // a text the UI changed itself (its clear button): the field follows
                // (a password there is stays as it was typed)
                for c in list where c.edit && (!c.secret || c.value != nil) { texts[c.key] = c.value ?? "" }
            }
        }
        let req = threers_ios_take_request()
        if req != 0 { onRequest(req) }
        openRequestedURLs()
    }

    /// Surface `i`'s new image: nil when unchanged, .some(nil) when it is
    /// not shown now.
    private func uiImage(_ i: Int) -> UIImage?? {
        let size = uiSizes[i]
        return scene.uiImage(UInt32(i), size.w, size.h, uiScales[i]).map { $0.map { UIImage(cgImage: $0) } }
    }
}

/// One of the app's threers-ui surfaces: its image (the UI draws its own
/// background, as the web app's does), a native button
/// over each of its controls — the picture is threers-ui's; the look
/// highlight, VoiceOver and the press the system's — and drags on it sent
/// back as touches.
struct SurfaceImage: View {
    @ObservedObject var model: VolumeModel
    let which: Int
    let image: UIImage
    let radius: CGFloat

    var body: some View {
        let size = model.uiSizes[which]
        // the controls' rectangles are in UI pixels: points, so
        let k = CGFloat(uiScales[which]) / 2
        let space = "surface\(which)"
        ZStack(alignment: .topLeading) {
            Image(uiImage: image).resizable()
                .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named(space))
                    .onChanged { model.drag(UInt32(which), $0) }.onEnded { model.dragEnd(UInt32(which), $0) })
                .accessibilityHidden(true)
            // a native text field over each text control: the platform's
            // keyboard, the text sent back; the UI draws it (clear here)
            ForEach(model.controls[which].filter { $0.edit }) { c in
                // a text area takes line breaks; a search box, one line
                // (a password's is a secure one: hidden, not copied nor learnt)
                let text = Binding(get: { model.texts[c.key] ?? "" }, set: { model.type(c, $0) })
                Group {
                    if c.secret {
                        SecureField(c.label, text: text, prompt: Text(verbatim: "")).textContentType(.password)
                    } else {
                        TextField(c.label, text: text, prompt: Text(verbatim: ""), axis: c.role == "textbox" ? .vertical : .horizontal)
                    }
                }
                    .foregroundStyle(.clear)
                    .tint(.clear)
                    .textFieldStyle(.plain)
                    .autocorrectionDisabled()
                    .textInputAutocapitalization(.never)
                    .frame(width: c.w * k, height: c.h * k)
                    .background(Gaze(c: c, k: k))
                    .accessibilityLabel(c.label)
                    .offset(x: c.x * k, y: c.y * k)
            }
            // a slider: lit while looked at; the pinch held and moved slides it
            ForEach(model.controls[which].filter { $0.slide }) { c in
                Gaze(c: c, k: k)
                    .contentShape(RoundedRectangle(cornerRadius: c.r * k))
                    .gesture(DragGesture(minimumDistance: 0, coordinateSpace: .named(space))
                        .onChanged { model.drag(UInt32(which), $0) }.onEnded { model.dragEnd(UInt32(which), $0) })
                    .accessibilityElement()
                    .accessibilityLabel(c.label)
                    .offset(x: c.x * k, y: c.y * k)
            }
            ForEach(model.controls[which].filter { !$0.edit && !$0.slide }) { c in
                Button { model.press(c) } label: {
                    Gaze(c: c, k: k)
                }
                .buttonStyle(.plain)
                .contentShape(RoundedRectangle(cornerRadius: c.r * k))
                .help(c.label)
                .disabled(c.disabled)
                .accessibilityLabel(c.label)
                .accessibilityValue(c.toggle ? (c.on ? "On" : "Off") : "")
                .accessibilityAddTraits(c.toggle && c.on ? .isSelected : [])
                // a drag from a control (a list's row) scrolls
                .simultaneousGesture(DragGesture(minimumDistance: 8, coordinateSpace: .named(space))
                    .onChanged { model.drag(UInt32(which), $0) }.onEnded { model.dragEnd(UInt32(which), $0) })
                .offset(x: c.x * k, y: c.y * k)
            }
        }
        .frame(width: CGFloat(size.w) / 2, height: CGFloat(size.h) / 2, alignment: .topLeading)
        .coordinateSpace(name: space)
    }
}

/// One of the volume's UI surfaces in a window of its own, beside the
/// volume (the sidebar) or under it (the toolbar): moved where you like.
struct SurfaceWindow: View {
    @ObservedObject var model = VolumeModel.shared
    let which: Int
    let radius: CGFloat

    var body: some View {
        // as big as its window, resized as you like: the UI lays itself
        // out again at that size
        GeometryReader { g in
            Group {
                if let image = model.ui[which] {
                    SurfaceImage(model: model, which: which, image: image, radius: radius)
                        .clipShape(RoundedRectangle(cornerRadius: radius))
                } else {
                    Color.clear
                }
            }
            .frame(width: g.size.width, height: g.size.height, alignment: .topLeading)
            .onChange(of: g.size, initial: true) { _, s in model.resizeUI(which, s) }
        }
        .frame(minWidth: 240, minHeight: which == 1 ? 60 : 320)
    }
}

struct Volume: View {
    @Environment(\.openImmersiveSpace) private var open
    @Environment(\.dismissImmersiveSpace) private var dismiss
    @Environment(\.openWindow) private var openWindow
    @Environment(\.dismissWindow) private var dismissWindow
    @ObservedObject private var model = VolumeModel.shared
    @State private var updates: EventSubscription?
    @State private var dragged: CGFloat = 0
    @Environment(\.physicalMetrics) private var metrics

    var body: some View {
        // as big as its window, resized as you like
        GeometryReader3D { g in reality
            .onChange(of: g.size, initial: true) { _, s in
                model.resize(Float(metrics.convert(min(s.width, s.height, s.depth), to: .meters)))
            }
        }
        // no smaller than this; as big as you drag it
        .frame(minWidth: points(0.3), minHeight: points(0.3))
        .frame(minDepth: points(0.3))
    }

    /// `metres` in the volume's points.
    private func points(_ metres: Double) -> CGFloat {
        metrics.convert(CGFloat(metres), from: .meters)
    }

    private var reality: some View {
        RealityView { content in
            await model.load()
            content.add(model.root)
            content.add(model.quad)
            content.add(model.hit)
            content.add(model.pieces.volume)
            updates = content.subscribe(to: SceneEvents.Update.self) { _ in model.tick() }
        }
        // a tap: what is there (a soma, a region); two quickly, fly to it
        .simultaneousGesture(SpatialTapGesture().targetedToAnyEntity().onEnded { e in
            model.tap(e.convert(e.location3D, from: .local, to: .scene))
        })
        .gesture(DragGesture().targetedToAnyEntity()
            .onChanged { v in
                model.turn(Float(v.translation.width - dragged) * 0.01)
                dragged = v.translation.width
            }
            .onEnded { _ in dragged = 0 })
        // the app's UI, all threers-ui, in windows of its own (the sidebar
        // beside the volume, the toolbar under it), each shown while the
        // UI shows it
        .onChange(of: model.ui[0] == nil, initial: true) { _, hidden in
            if hidden { dismissWindow(id: "sidebar") } else { openWindow(id: "sidebar") }
        }
        .onChange(of: model.ui[1] == nil, initial: true) { _, hidden in
            if hidden { dismissWindow(id: "toolbar") } else { openWindow(id: "toolbar") }
        }
        .onAppear {
            model.onRequest = { r in Task { @MainActor in await shellRequest(r, open: open, dismiss: dismiss) } }
            // its controls are native: the room leaves its own panel out
            threers_ios_volume_shown(true)
        }
        .onDisappear { threers_ios_volume_shown(false) }
    }
}

@main
struct ThreersApp: App {
    @State private var style: ImmersionStyle = .mixed
    var body: some SwiftUI.Scene {
        #if THREERS_WINDOW
        WindowGroup { FlatWindow() }
        #else
        // the volume is what opens: launched from Home or restored, never a
        // window in its place (a window scene granted at launch had SwiftUI
        // build the UI's windows as the first scene)
        WindowGroup(id: "volume") { Volume() }
            .windowStyle(.volumetric)
            .defaultLaunchBehavior(.presented)
            .defaultSize(width: Double(volumeSide), height: Double(volumeSide), depth: Double(volumeSide), in: .meters)
            // resizable (its corner's handle): from its content's least up
            .windowResizability(.contentMinSize)
        // the UI's windows open with the volume (its `openWindow`), not on
        // their own: never launched or restored without it
        WindowGroup(id: "sidebar") { SurfaceWindow(which: 0, radius: 16) }
            .windowStyle(.plain)
            .defaultSize(width: 476, height: 1008)
            .defaultLaunchBehavior(.suppressed)
            .restorationBehavior(.disabled)
            .defaultWindowPlacement { _, context in
                if let volume = context.windows.first(where: { $0.id == "volume" }) { WindowPlacement(.leading(volume)) } else { WindowPlacement() }
            }
        WindowGroup(id: "toolbar") { SurfaceWindow(which: 1, radius: 38) }
            .windowStyle(.plain)
            .defaultSize(width: 700, height: 90)
            .defaultLaunchBehavior(.suppressed)
            .restorationBehavior(.disabled)
            .defaultWindowPlacement { _, context in
                if let volume = context.windows.first(where: { $0.id == "volume" }) { WindowPlacement(.below(volume)) } else { WindowPlacement() }
            }
        #endif
        ImmersiveSpace(id: "immersive") {
            CompositorLayer(configuration: EyesLayout()) { layerRenderer in
                layerRenderer.onSpatialEvent = { events in
                    for e in events {
                        guard let ray = e.selectionRay else { continue }
                        let phase: UInt32 = e.phase == .active ? 1 : (e.phase == .ended ? 2 : 3)
                        threers_ios_spatial_event(UInt64(bitPattern: Int64(e.id.hashValue)), phase,
                            Float(ray.origin.x), Float(ray.origin.y), Float(ray.origin.z),
                            Float(ray.direction.x), Float(ray.direction.y), Float(ray.direction.z))
                    }
                }
                let thread = Thread {
                    threers_app_run(Unmanaged.passUnretained(layerRenderer).toOpaque())
                    threers_ios_immersive(false)
                }
                thread.name = "threers immersive"
                thread.qualityOfService = .userInteractive
                thread.start()
            }
        }
        .immersionStyle(selection: $style, in: .mixed, .full)
        #if !THREERS_WINDOW
        ImmersiveSpace(id: "pieces") { PieceSpace() }
            .immersionStyle(selection: .constant(.mixed), in: .mixed)
        #endif
        ImmersiveSpace(id: "backdrop") { Backdrop() }
            .immersionStyle(selection: .constant(.mixed), in: .mixed)
            .immersiveEnvironmentBehavior(.coexist)
    }
}
#else
@main
struct ThreersApp: App {
    var body: some SwiftUI.Scene {
        #if THREERS_REALITY
        WindowGroup { RealityStage() }
        #else
        WindowGroup { HostView().ignoresSafeArea() }
        #endif
    }
}

/// What a RealityKit stage asks of the shell each frame (Reality.swift).
@MainActor func pollShell() {
    // nothing to enter or leave on iOS: the request is dropped
    _ = threers_ios_take_request()
    openRequestedURLs()
}

/// The app's UI over RealityKit's view (`Place::Screen`): its picture, the
/// fingers that land on it (the rest pass through to the scene: the orbit
/// camera, a tap), and its controls for assistive technology.
@available(iOS 18.0, *)
final class OverlayView: UIView {
    var model: StageModel?
    private var finger: UITouch?
    private var elements: [UIAccessibilityElement] = []
    override var accessibilityElements: [Any]? {
        get { elements }
        set {}
    }

    override init(frame: CGRect) {
        super.init(frame: frame)
        backgroundColor = .clear
        isOpaque = false
        layer.contentsGravity = .resize
    }

    required init?(coder: NSCoder) { fatalError("made in code") }

    func show(_ image: CGImage?, _ controls: [UiControl]) {
        layer.contents = image
        elements = controls.map { UiControlElement(container: self, control: $0) }
    }

    /// Only where the UI has something: the rest is the scene's.
    override func point(inside p: CGPoint, with event: UIEvent?) -> Bool {
        model?.onUi(p) ?? false
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        guard finger == nil, let t = touches.first else { return }
        finger = t
        model?.touch(0, t.location(in: self))
    }
    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { model?.touch(1, f.location(in: self)) }
    }
    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { model?.touch(2, f.location(in: self)); finger = nil }
    }
    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { model?.touch(3, f.location(in: self)); finger = nil }
    }
}

@available(iOS 18.0, *)
struct ScreenOverlay: UIViewRepresentable {
    @ObservedObject var model: StageModel
    func makeUIView(context: Context) -> OverlayView {
        let v = OverlayView(frame: .zero)
        v.model = model
        return v
    }
    func updateUIView(_ v: OverlayView, context: Context) {
        v.model = model
        v.show(model.ui, model.controls)
    }
}
#endif


/// A native text field over one of the UI's text controls (`Host`'s).
final class ControlField: UITextField {
    let key: String
    /// a text area's: Return is a line break, not the end
    let multiline: Bool

    init(key: String, multiline: Bool) {
        self.key = key
        self.multiline = multiline
        super.init(frame: .zero)
        textColor = .clear
        tintColor = .clear
        backgroundColor = .clear
        borderStyle = .none
        autocorrectionType = .no
        autocapitalizationType = .none
        spellCheckingType = .no
        returnKeyType = multiline ? .default : .done
    }

    required init?(coder: NSCoder) { fatalError("made in code") }

    /// The lines before the one in the field (which holds only one: a line
    /// break put in it is a space).
    private var before = ""

    /// All of the control's text.
    var whole: String {
        get { before + (text ?? "") }
        set {
            guard newValue != whole else { return }
            let line = multiline ? newValue.lastIndex(of: "\n").map { newValue.index(after: $0) } : nil
            before = String(newValue[..<(line ?? newValue.startIndex)])
            text = String(newValue[(line ?? newValue.startIndex)...])
        }
    }

    /// Nothing left in the field: into the line before; nothing there
    /// either, the app's (what its control holds may be more).
    override func deleteBackward() {
        guard multiline, (text ?? "").isEmpty, let app = superview as? Host else { return super.deleteBackward() }
        if before.isEmpty {
            app.pressBackspace()
        } else {
            whole = String(before.dropLast())
            app.typed(self)
        }
    }

    /// A password's: the system hides it, and does not copy, suggest or learn
    /// it. Emptied before it stops being one.
    func keepSecret(_ secret: Bool) {
        guard isSecureTextEntry != secret else { return }
        if !secret { text = "" }
        isSecureTextEntry = secret
        textContentType = secret ? .password : nil
    }

    /// Its editing's shortcuts, the app's where it keeps them (the rest are
    /// `Host.keyCommands`).
    override func pressesBegan(_ presses: Set<UIPress>, with event: UIPressesEvent?) {
        let app = superview as? Host
        let taken = presses.filter { p in
            guard let key = p.key, key.modifierFlags.contains(.command), threers_ios_chords_kept() else { return false }
            return ["a", "c", "v", "x", "z"].contains(key.charactersIgnoringModifiers.lowercased()) && app?.press(key) == true
        }
        if taken.isEmpty { super.pressesBegan(presses, with: event) }
    }
}

extension Host: UITextFieldDelegate {
    func textFieldShouldReturn(_ f: UITextField) -> Bool {
        if let c = f as? ControlField, c.multiline {
            c.whole += "\n"
            typed(c)
            return false
        }
        f.resignFirstResponder()
        return true
    }

    /// The keyboard put away: the view takes a hardware keyboard's keys again.
    func textFieldDidEndEditing(_ f: UITextField) {
        becomeFirstResponder()
    }
}


/// The render thread's display link's target: the host, weakly (a display
/// link holds its target), its frame drawn there.
final class RenderTarget: NSObject {
    weak var host: Host?
    let gen: Int
    init(host: Host, gen: Int) {
        self.host = host
        self.gen = gen
    }
    @objc func tick(_ l: CADisplayLink) {
        guard let host else {
            l.invalidate()
            CFRunLoopStop(CFRunLoopGetCurrent())
            return
        }
        host.renderFrame(l, gen)
    }
}

extension Host: UIGestureRecognizerDelegate {
    func gestureRecognizer(
        _ g: UIGestureRecognizer,
        shouldRecognizeSimultaneouslyWith other: UIGestureRecognizer
    ) -> Bool {
        true
    }
}
