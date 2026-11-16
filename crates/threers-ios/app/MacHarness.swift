// The shell for a threers-ios app on the Mac: RealityKit draws the app's
// scene in a window (Reality.swift's `RealityStage`: an orbit camera, the
// app's lights and environment), its UI over it. The same Rust crate as on
// iOS and visionOS (`threers_ios::app!`); built by app/build.sh `mac`.

import AppKit
import RealityKit
import SwiftUI
import UniformTypeIdentifiers

@_silgen_name("threers_ios_take_request") func threers_ios_take_request() -> UInt32
@_silgen_name("threers_ios_take_url") func threers_ios_take_url(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int
@_silgen_name("threers_ios_take_copy") func threers_ios_take_copy(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int
@_silgen_name("threers_ios_take_share") func threers_ios_take_share(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int
@_silgen_name("threers_ios_take_pick") func threers_ios_take_pick() -> Bool
@_silgen_name("threers_ios_picked") func threers_ios_picked(_ path: UnsafePointer<UInt8>, _ len: Int)
@_silgen_name("threers_ios_set_dark") func threers_ios_set_dark(_ dark: Bool)

/// What the app asked of the shell since the last frame: pages to open,
/// text to copy, files to show or keep, a file to pick, a model to preview.
@MainActor func pollShell() {
    _ = threers_ios_take_request()
    var buf = [UInt8](repeating: 0, count: 16384)
    while true {
        let n = threers_ios_take_url(&buf, buf.count)
        guard n > 0, let s = String(bytes: buf[0..<n], encoding: .utf8), let url = URL(string: s) else { break }
        NSWorkspace.shared.open(url)
    }
    let copied = threers_ios_take_copy(&buf, buf.count)
    if copied > 0, let s = String(bytes: buf[0..<copied], encoding: .utf8) {
        NSPasteboard.general.clearContents()
        NSPasteboard.general.setString(s, forType: .string)
    }
    // a file the app made, shown in the Finder to keep or send
    let shared = threers_ios_take_share(&buf, buf.count)
    if shared > 0, let path = String(bytes: buf[0..<shared], encoding: .utf8) {
        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: path)])
    }
    if threers_ios_take_pick() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            let path = Array(url.path.utf8)
            path.withUnsafeBufferPointer { threers_ios_picked($0.baseAddress!, path.count) }
        }
    }
    // a .usdz in the system's viewer (Preview: RealityKit's Quick Look)
    if let url = takeQuickLook() { NSWorkspace.shared.open(url) }
}

/// The app's UI over RealityKit's view (`Place::Screen`): its picture, the
/// mouse where it lands on the UI (elsewhere it passes through to the
/// orbit camera and the scene), its controls for VoiceOver.
final class OverlayView: NSView {
    var model: StageModel?
    private var image: CGImage?
    private var controls: [UiControl] = []

    override var isFlipped: Bool { true }
    override var wantsUpdateLayer: Bool { true }

    override init(frame: NSRect) {
        super.init(frame: frame)
        wantsLayer = true
        layer?.contentsGravity = .resize
    }

    required init?(coder: NSCoder) { fatalError("made in code") }

    func show(_ image: CGImage?, _ controls: [UiControl]) {
        self.image = image
        layer?.contents = image
        if controls.map(\.id) != self.controls.map(\.id) {
            self.controls = controls
            setAccessibilityChildren(controls.map { c in
                let e = NSAccessibilityElement()
                e.setAccessibilityRole(c.role == "slider" ? .slider : .button)
                e.setAccessibilityLabel(c.label)
                e.setAccessibilityValue(c.value)
                e.setAccessibilityFrameInParentSpace(NSRect(x: c.x, y: bounds.height - c.y - c.h, width: c.w, height: c.h))
                e.setAccessibilityParent(self)
                return e
            })
        }
    }

    /// Only where the UI has something: the rest is the scene's.
    override func hitTest(_ point: NSPoint) -> NSView? {
        let p = convert(point, from: superview)
        return model?.onUi(p) == true ? self : nil
    }

    private func at(_ e: NSEvent) -> CGPoint { convert(e.locationInWindow, from: nil) }
    override func mouseDown(with e: NSEvent) { model?.touch(0, at(e)) }
    override func mouseDragged(with e: NSEvent) { model?.touch(1, at(e)) }
    override func mouseUp(with e: NSEvent) { model?.touch(2, at(e)) }
    override func scrollWheel(with e: NSEvent) {
        // a wheel over a list: a drag of the same length
        let p = at(e)
        model?.touch(0, p)
        model?.touch(1, CGPoint(x: p.x + e.scrollingDeltaX, y: p.y + e.scrollingDeltaY))
        model?.touch(2, CGPoint(x: p.x + e.scrollingDeltaX, y: p.y + e.scrollingDeltaY))
    }
}

struct ScreenOverlay: NSViewRepresentable {
    @ObservedObject var model: StageModel
    func makeNSView(context: Context) -> OverlayView {
        let v = OverlayView(frame: .zero)
        v.model = model
        return v
    }
    func updateNSView(_ v: OverlayView, context: Context) {
        v.model = model
        v.show(model.ui, model.controls)
    }
}

@main
struct ThreersApp: App {
    @Environment(\.colorScheme) private var scheme
    var body: some SwiftUI.Scene {
        WindowGroup {
            RealityStage()
                .frame(minWidth: 480, minHeight: 360)
                .onChange(of: scheme, initial: true) { _, s in threers_ios_set_dark(s == .dark) }
        }
        .defaultSize(width: 1100, height: 760)
    }
}
