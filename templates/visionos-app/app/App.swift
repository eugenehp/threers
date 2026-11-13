// The SwiftUI shell of a threers Vision Pro app: a full immersive space
// whose CompositorLayer is handed to Rust (`threers_app_run`, exported by
// `threers::visionos_app!`), and look-and-pinch forwarded to it, with the
// system's accessibility settings and a way to have VoiceOver speak.
// Nothing here is particular to an app; the app is the Rust crate.

import CompositorServices
import SwiftUI
import UIKit

@_silgen_name("threers_app_run")
func threers_app_run(_ layerRenderer: UnsafeMutableRawPointer)

@_silgen_name("threers_xr_spatial_event")
func threers_xr_spatial_event(_ id: UInt64, _ phase: UInt32, _ ox: Float, _ oy: Float, _ oz: Float, _ dx: Float, _ dy: Float, _ dz: Float)

@_silgen_name("threers_xr_set_accessibility")
func threers_xr_set_accessibility(_ flags: UInt32, _ textScale: Float)

@_silgen_name("threers_xr_set_announcer")
func threers_xr_set_announcer(_ announcer: (@convention(c) (UnsafePointer<CChar>?, Bool) -> Void)?)

/// Both eyes in one texture, a viewport each, as threers draws them.
struct SharedLayout: CompositorLayerConfiguration {
    func makeConfiguration(capabilities: LayerRenderer.Capabilities, configuration: inout LayerRenderer.Configuration) {
        configuration.layout = .shared
        configuration.colorFormat = .bgra8Unorm_srgb
        configuration.depthFormat = .depth32Float
        configuration.isFoveationEnabled = false
        // The system's highlight of what is looked at: threers draws the
        // areas it may light up into a texture of their own.
        let formats = capabilities.supportedTrackingAreasFormats
        if let format = [MTLPixelFormat.r16Uint, .r8Uint].first(where: { formats.contains($0) }) {
            configuration.trackingAreasFormat = format
            // (The compositor reads it, so shaderRead too.)
            configuration.trackingAreasUsage = [.renderTarget, .shaderRead]
        }
    }
}

/// The system's accessibility settings, to Rust, now and whenever one
/// changes; and VoiceOver's voice, for the UI's announcements.
enum Accessibility {
    private static var started = false

    static func start() {
        guard !started else { return }
        started = true
        report()
        let changes: [Notification.Name] = [
            UIAccessibility.reduceMotionStatusDidChangeNotification,
            UIAccessibility.boldTextStatusDidChangeNotification,
            UIAccessibility.darkerSystemColorsStatusDidChangeNotification,
            UIAccessibility.reduceTransparencyStatusDidChangeNotification,
            UIAccessibility.voiceOverStatusDidChangeNotification,
            UIContentSizeCategory.didChangeNotification,
        ]
        for name in changes {
            NotificationCenter.default.addObserver(forName: name, object: nil, queue: .main) { _ in report() }
        }
        threers_xr_set_announcer { text, assertive in
            guard let text else { return }
            let message = String(cString: text)
            DispatchQueue.main.async {
                let priority: UIAccessibilityPriority = assertive ? .high : .default
                let spoken = NSAttributedString(string: message, attributes: [.accessibilitySpeechAnnouncementPriority: priority])
                UIAccessibility.post(notification: .announcement, argument: spoken)
            }
        }
    }

    static func report() {
        var flags: UInt32 = 0
        if UIAccessibility.isReduceMotionEnabled { flags |= 1 }
        if UIAccessibility.isBoldTextEnabled { flags |= 2 }
        if UIAccessibility.isDarkerSystemColorsEnabled { flags |= 4 }
        if UIAccessibility.isReduceTransparencyEnabled { flags |= 8 }
        if UIAccessibility.isVoiceOverRunning { flags |= 16 }
        if UITraitCollection.current.userInterfaceStyle == .dark { flags |= 32 }
        // The text size chosen, as body text scales from its default.
        let scale = UIFontMetrics(forTextStyle: .body).scaledValue(for: 17, compatibleWith: UITraitCollection(preferredContentSizeCategory: UIApplication.shared.preferredContentSizeCategory)) / 17
        threers_xr_set_accessibility(flags, Float(scale))
    }
}

@main
struct ThreersApp: App {
    @State private var style: ImmersionStyle = .full

    var body: some Scene {
        ImmersiveSpace(id: "threers") {
            CompositorLayer(configuration: SharedLayout()) { layerRenderer in
                Accessibility.start()
                layerRenderer.onSpatialEvent = { events in
                    for event in events {
                        guard let ray = event.selectionRay else { continue }
                        let phase: UInt32 = event.phase == .active ? 1 : (event.phase == .ended ? 2 : 3)
                        threers_xr_spatial_event(
                            UInt64(bitPattern: Int64(event.id.hashValue)), phase,
                            Float(ray.origin.x), Float(ray.origin.y), Float(ray.origin.z),
                            Float(ray.direction.x), Float(ray.direction.y), Float(ray.direction.z))
                    }
                }
                let thread = Thread { threers_app_run(Unmanaged.passUnretained(layerRenderer).toOpaque()) }
                thread.name = "threers render"
                thread.qualityOfService = .userInteractive
                thread.start()
            }
        }
        .immersionStyle(selection: $style, in: .full)
    }
}
