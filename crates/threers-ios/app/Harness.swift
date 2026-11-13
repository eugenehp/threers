// The shell every threers-ios app shares, iOS and visionOS: a CAMetalLayer
// handed to Rust (which draws it with Metal), a display link asking it for
// frames, fingers forwarded.
// On visionOS: a volume, where RealityKit draws the app's meshes in the
// shared space (the Metal window instead with `-D THREERS_WINDOW`), and the
// immersive space, opened and closed when the app asks
// (`threers_ios::request`). Nothing here is particular to an app; the app is
// the Rust crate (`threers_ios::app!`). Built by app/build.sh with swiftc.

import SwiftUI
import UIKit
#if os(visionOS)
import CompositorServices
import RealityKit
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

/// Each web page the app asked to open, opened with the system.
@MainActor func openRequestedURLs() {
    var buf = [UInt8](repeating: 0, count: 4096)
    while true {
        let n = threers_ios_take_url(&buf, buf.count)
        guard n > 0, let s = String(bytes: buf[0..<n], encoding: .utf8), let url = URL(string: s) else { return }
        UIApplication.shared.open(url)
    }
}
@_silgen_name("threers_ios_immersive")
func threers_ios_immersive(_ open: Bool)
@_silgen_name("threers_view_controls")
func threers_view_controls(_ view: UnsafeMutableRawPointer?, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_ios_press")
func threersPressControl(_ key: UInt64)
@_silgen_name("threers_ios_set_dark")
func threers_ios_set_dark(_ dark: Bool)

/// One of the window UI's controls (`threers_ui::native`), in its points.
struct UiControl: Decodable {
    let key: String
    let role: String
    let label: String
    let x, y, w, h: CGFloat
    let on, toggle, disabled: Bool
    let value: String?
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
    private var link: CADisplayLink?
    private var finger: UITouch?
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
        surface.frame = bounds
        surface.autoresizingMask = [.flexibleWidth, .flexibleHeight]
        surface.isUserInteractionEnabled = false
        addSubview(surface)
        let pinch = UIPinchGestureRecognizer(target: self, action: #selector(pinched(_:)))
        let pan = UIPanGestureRecognizer(target: self, action: #selector(panned(_:)))
        pan.minimumNumberOfTouches = 2
        for g in [pinch, pan] as [UIGestureRecognizer] {
            g.cancelsTouchesInView = false
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
            link?.invalidate()
            link = nil
            return
        }
        surface.layer.contentsScale = scale
        (surface.layer as! CAMetalLayer).drawableSize = CGSize(width: Double(pixels.0), height: Double(pixels.1))
        threers_ios_set_dark(traitCollection.userInterfaceStyle == .dark)
        if view == nil {
            view = threers_view_create(Unmanaged.passUnretained(surface.layer).toOpaque(), pixels.0, pixels.1, 0)
            sendInsets()
        }
        if link == nil {
            let l = CADisplayLink(target: self, selector: #selector(tick))
            l.add(to: .main, forMode: .common)
            link = l
        }
    }

    override func layoutSubviews() {
        super.layoutSubviews()
        (surface.layer as! CAMetalLayer).drawableSize = CGSize(width: Double(pixels.0), height: Double(pixels.1))
        threers_view_set(view, 0, Float(pixels.0), Float(pixels.1), 0)
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
        threers_view_set(view, 4, Float(scale), 0, 0)
        threers_view_set(view, 5, Float(i.top * scale), Float(i.bottom * scale), Float(max(i.left, i.right) * scale))
    }

    @objc private func tick() {
        _ = threers_view_frame(view)
        var len = 0
        if let p = threers_view_controls(view, &len),
           let list = try? JSONDecoder().decode([UiControl].self, from: Data(bytes: p, count: len)) {
            controlElements = list.map { UiControlElement(container: self, control: $0) }
            UIAccessibility.post(notification: .layoutChanged, argument: nil)
        }
        let r = threers_ios_take_request()
        if r != 0 { onRequest(r) }
        openRequestedURLs()
    }

    private func send(_ touch: UITouch, _ phase: Float) {
        let p = touch.location(in: self)
        threers_view_set(view, 1, phase, Float(p.x * scale), Float(p.y * scale))
    }

    override func touchesBegan(_ touches: Set<UITouch>, with event: UIEvent?) {
        if finger == nil, let t = touches.first {
            finger = t
            send(t, 0)
        } else if let f = finger {
            send(f, 3)  // a second finger: the gestures take over
            finger = nil
        }
    }

    override func touchesMoved(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { send(f, 1) }
    }

    override func touchesEnded(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { send(f, 2); finger = nil }
    }

    override func touchesCancelled(_ touches: Set<UITouch>, with event: UIEvent?) {
        if let f = finger, touches.contains(f) { send(f, 3); finger = nil }
    }

    @objc private func pinched(_ g: UIPinchGestureRecognizer) {
        threers_view_set(view, 2, Float(g.scale), 0, 0)
        g.scale = 1
    }

    @objc private func panned(_ g: UIPanGestureRecognizer) {
        let d = g.translation(in: self)
        threers_view_set(view, 3, Float(d.x * scale), Float(d.y * scale), 0)
        g.setTranslation(.zero, in: self)
    }

    deinit {
        link?.invalidate()
        threers_view_destroy(view)
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
    if r == 3 {
        let result = await open(id: "immersive")
        print("threers: immersive space: \(result)")
        guard case .opened = result else { return }
        threers_ios_immersive(true)
    } else if r == 4 {
        await dismiss()
        threers_ios_immersive(false)
    }
}

// --- the volume: RealityKit draws the app's meshes in the shared space -------

@_silgen_name("threers_vol_create") func threers_vol_create() -> UnsafeMutableRawPointer?
@_silgen_name("threers_vol_frame") func threers_vol_frame(_ v: UnsafeMutableRawPointer?) -> UInt32
@_silgen_name("threers_vol_object") func threers_vol_object(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ keys: UnsafeMutablePointer<UInt64>, _ floats: UnsafeMutablePointer<Float>) -> Bool
@_silgen_name("threers_vol_geometry") func threers_vol_geometry(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ ptrs: UnsafeMutablePointer<UnsafeRawPointer?>, _ counts: UnsafeMutablePointer<UInt32>) -> Bool
@_silgen_name("threers_vol_texture") func threers_vol_texture(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ w: UnsafeMutablePointer<UInt32>, _ h: UnsafeMutablePointer<UInt32>, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_vol_uvs") func threers_vol_uvs(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<Float>?
@_silgen_name("threers_vol_root") func threers_vol_root(_ v: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<Float>)
@_silgen_name("threers_vol_turn") func threers_vol_turn(_ v: UnsafeMutableRawPointer?, _ radians: Float)
@_silgen_name("threers_vol_ui") func threers_vol_ui(_ v: UnsafeMutableRawPointer?, _ which: UInt32, _ w: UInt32, _ h: UInt32, _ scale: Float, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_ios_tap") func threers_ios_tap(_ x: Float, _ y: Float)
@_silgen_name("threers_ios_immersive_open") func threers_ios_immersive_open() -> Bool
@_silgen_name("threers_vol_controls") func threers_vol_controls(_ v: UnsafeMutableRawPointer?, _ which: UInt32, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_ios_press") func threers_ios_press(_ key: UInt64)
@_silgen_name("threers_ios_text") func threers_ios_text(_ key: UInt64, _ text: UnsafePointer<UInt8>?, _ len: Int)
@_silgen_name("threers_ios_touch") func threers_ios_touch(_ which: UInt32, _ phase: UInt32, _ x: Float, _ y: Float)
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
    let value: String?
    /// Slid: a view over it shows the look's highlight and passes the drag on.
    let slide: Bool
    /// The UI's accent, 0–1 sRGB: the highlight's colour.
    let tint: [Double]
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
/// volume, 476 × 1008 points; 1: the toolbar under it, 700 × 90.
let uiSizes: [(w: Int, h: Int)] = [(952, 2016), (1400, 180)]
let uiScales: [Float] = [2.8, 2.8]
/// Each press and drag on the UI surfaces, to the console (`THREERS_UI_LOG=1`).
let uiLog = ProcessInfo.processInfo.environment["THREERS_UI_LOG"] != nil

private func matrix(_ f: [Float]) -> simd_float4x4 {
    simd_float4x4(SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]), SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15]))
}

/// The app's meshes as entities, kept in step with its scene each frame:
/// a mesh rebuilt when its geometry changes, a material when its material does.
@MainActor final class VolumeModel: ObservableObject {
    let vol = threers_vol_create()
    /// fits the model to the volume
    let root = Entity()
    /// the model's frame: its focus at the centre, turned by drags
    let model = Entity()
    /// what drags land on: the volume's own box, unscaled (under `root`,
    /// scaled to the model, it grew past the volume's walls, where the
    /// system clips content and hits nothing, so drags never turned it)
    let hit = Entity()
    private var entities: [UInt64: ModelEntity] = [:]
    private var built: [UInt64: (UInt64, UInt64)] = [:]
    private var vertexColor: ShaderGraphMaterial?
    /// What each text control's native field holds, by key.
    @Published var texts: [String: String] = [:]
    /// Each surface's image (nil: not shown) and its controls.
    @Published var ui: [UIImage?] = [nil, nil]
    @Published var controls: [[NativeControl]] = [[], []]
    var onRequest: (UInt32) -> Void = { _ in }

    init() {
        root.addChild(model)
        // drags land anywhere in the volume, a little inside its walls
        hit.components.set(CollisionComponent(shapes: [.generateBox(size: SIMD3(repeating: volumeSide * 0.95))]))
        hit.components.set(InputTargetComponent())
    }

    func load() async {
        vertexColor = try? await ShaderGraphMaterial(named: "/Root/VertexColor", from: "VertexColor.usda", in: .main)
    }

    func turn(_ radians: Float) { threers_vol_turn(vol, radians) }

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

    func tick() {
        let n = threers_vol_frame(vol)
        var r = [Float](repeating: 0, count: 17)
        threers_vol_root(vol, &r)
        model.transform = Transform(matrix: matrix(r))
        root.scale = SIMD3(repeating: volumeSide * 0.45 / max(r[16], 1e-4))
        var seen = Set<UInt64>()
        var keys = [UInt64](repeating: 0, count: 3)
        var f = [Float](repeating: 0, count: 32)
        for i in 0..<n {
            guard threers_vol_object(vol, i, &keys, &f) else { continue }
            let key = keys[0]
            seen.insert(key)
            let e = entities[key] ?? {
                let e = ModelEntity()
                model.addChild(e)
                entities[key] = e
                return e
            }()
            let flags = UInt32(f[22])
            e.isEnabled = flags & 1 != 0
            e.transform = Transform(matrix: matrix(f))
            let was = built[key]
            if was?.0 != keys[1], let mesh = mesh(i, lines: flags & 16 != 0, points: flags & 64 != 0) {
                e.model = ModelComponent(mesh: mesh, materials: [])
            }
            if was?.0 != keys[1] || was?.1 != keys[2] {
                e.model?.materials = [flags & (32 | 128) != 0 ? spriteMaterial(i, f, twoSided: flags & 8 != 0, cut: flags & 128 != 0) ?? material(f, flags) : material(f, flags)]
                // a sprite (a label) keeps facing whoever looks at it
                if flags & 32 != 0 {
                    e.components.set(BillboardComponent())
                }
                // see-through: depth first, then colour, so only the nearest
                // surface blends (single-layer glass; blended alone, a folded
                // cortex drew its folds and triangles through each other)
                if flags & 4 != 0 {
                    e.components.set(ModelSortGroupComponent(group: ModelSortGroup(depthPass: .prePass), order: 0))
                } else {
                    e.components.remove(ModelSortGroupComponent.self)
                }
            }
            built[key] = (keys[1], keys[2])
        }
        for (key, e) in entities where !seen.contains(key) {
            e.removeFromParent()
            entities[key] = nil
            built[key] = nil
        }
        // the room shows the model while it is open: one brain, not two
        model.isEnabled = !threers_ios_immersive_open()
        // the app's UI surfaces, each when it changed
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
                for c in list where c.edit { texts[c.key] = c.value ?? "" }
            }
        }
        let req = threers_ios_take_request()
        if req != 0 { onRequest(req) }
        openRequestedURLs()
    }

    /// Sprite `i`'s texture on an unlit, see-through quad.
    private func spriteMaterial(_ i: UInt32, _ f: [Float], twoSided: Bool = false, cut: Bool = false) -> RealityKit.Material? {
        var w: UInt32 = 0, h: UInt32 = 0, len = 0
        guard let p = threers_vol_texture(vol, i, &w, &h, &len), len == Int(w * h * 4),
              let provider = CGDataProvider(data: Data(bytes: p, count: len) as CFData),
              let image = CGImage(width: Int(w), height: Int(h), bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: Int(w) * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent),
              let texture = try? TextureResource(image: image, options: .init(semantic: .color))
        else { return nil }
        var m = UnlitMaterial()
        m.color = .init(tint: .white, texture: .init(texture))
        m.blending = .transparent(opacity: .init(floatLiteral: f[19]))
        // a slice is seen from both sides
        if twoSided { m.faceCulling = .none }
        // a textured mesh (a volume's slice) is cut where it is clear, as
        // its shader discards there: in the depth prepass too, or its clear
        // pixels hide what is behind them (the other slices)
        if cut { m.opacityThreshold = 0.02 }
        return m
    }

    private func material(_ f: [Float], _ flags: UInt32) -> RealityKit.Material {
        let blended = flags & 4 != 0
        // a line set: unlit, in its colour (in its vertices' colours, the
        // vertex-colour material's)
        if flags & (16 | 64) != 0 && flags & 2 == 0 {
            var m = UnlitMaterial(color: UIColor(red: CGFloat(f[16]), green: CGFloat(f[17]), blue: CGFloat(f[18]), alpha: 1))
            if blended { m.blending = .transparent(opacity: .init(floatLiteral: f[19])) }
            return m
        }
        let cull: MaterialParameterTypes.FaceCulling = flags & 8 != 0 ? .none : .back
        if flags & 2 != 0, var m = vertexColor {
            try? m.setParameter(name: "Opacity", value: .float(blended ? f[19] : 1))
            try? m.setParameter(name: "Roughness", value: .float(f[20]))
            try? m.setParameter(name: "Metallic", value: .float(f[21]))
            m.faceCulling = cull
            return m
        }
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: UIColor(red: CGFloat(f[16]), green: CGFloat(f[17]), blue: CGFloat(f[18]), alpha: 1))
        m.roughness = .init(floatLiteral: f[20])
        m.metallic = .init(floatLiteral: f[21])
        // its glow (an electrode's), as the window draws it
        if f[26] > 0 {
            m.emissiveColor = .init(color: UIColor(red: CGFloat(f[23]), green: CGFloat(f[24]), blue: CGFloat(f[25]), alpha: 1))
            m.emissiveIntensity = f[26]
        }
        if blended { m.blending = .transparent(opacity: .init(floatLiteral: f[19])) }
        m.faceCulling = cull
        return m
    }

    /// Mesh `i` as a RealityKit low-level mesh: positions, normals and, when
    /// it has them, vertex colours, interleaved.
    private func mesh(_ i: UInt32, lines: Bool = false, points: Bool = false) -> MeshResource? {
        var ptrs = [UnsafeRawPointer?](repeating: nil, count: 4)
        var counts = [UInt32](repeating: 0, count: 3)
        guard threers_vol_geometry(vol, i, &ptrs, &counts), let pp = ptrs[0], let np = ptrs[1], let ip = ptrs[3] else { return nil }
        let (n, cs, m) = (Int(counts[0]), Int(counts[1]), Int(counts[2]))
        guard n > 0, m > 0 else { return nil }
        let pos = pp.assumingMemoryBound(to: Float.self)
        let nor = np.assumingMemoryBound(to: Float.self)
        let col = ptrs[2]?.assumingMemoryBound(to: Float.self)
        // texture coordinates (a textured mesh: a volume's slice)
        var uvLen = 0
        let uvp = threers_vol_uvs(vol, i, &uvLen)
        let uv = uvLen == n * 2 ? uvp : nil
        let stride = (col != nil ? 40 : 24) + (uv != nil ? 8 : 0)
        let uvOffset = col != nil ? 40 : 24
        var attributes: [LowLevelMesh.Attribute] = [
            .init(semantic: .position, format: .float3, offset: 0),
            .init(semantic: .normal, format: .float3, offset: 12),
        ]
        if col != nil { attributes.append(.init(semantic: .color, format: .float4, offset: 24)) }
        if uv != nil { attributes.append(.init(semantic: .uv0, format: .float2, offset: uvOffset)) }
        let descriptor = LowLevelMesh.Descriptor(
            vertexCapacity: n, vertexAttributes: attributes,
            vertexLayouts: [.init(bufferIndex: 0, bufferStride: stride)],
            indexCapacity: m, indexType: .uint32)
        guard let mesh = try? LowLevelMesh(descriptor: descriptor) else { return nil }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        mesh.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let out = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            let per = stride / 4
            for v in 0..<n {
                let o = out + v * per
                let p = SIMD3(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2])
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
                o[0] = p.x; o[1] = p.y; o[2] = p.z
                o[3] = nor[v * 3]; o[4] = nor[v * 3 + 1]; o[5] = nor[v * 3 + 2]
                if let c = col {
                    o[6] = c[v * cs]; o[7] = c[v * cs + 1]; o[8] = c[v * cs + 2]
                    o[9] = cs == 4 ? c[v * cs + 3] : 1
                }
                if let t = uv {
                    // RealityKit's v runs up the image; the slice's rows down
                    o[uvOffset / 4] = t[v * 2]; o[uvOffset / 4 + 1] = 1 - t[v * 2 + 1]
                }
            }
        }
        mesh.withUnsafeMutableIndices { raw in
            raw.baseAddress!.copyMemory(from: ip, byteCount: m * 4)
        }
        // a line set: its segments as lines (RealityKit's line topology)
        mesh.parts.replaceAll([LowLevelMesh.Part(indexCount: m, topology: points ? .point : lines ? .line : .triangle, bounds: BoundingBox(min: lo, max: hi))])
        return try? MeshResource(from: mesh)
    }

    /// Surface `i`'s new image: nil when unchanged, .some(nil) when it is
    /// not shown now.
    private func uiImage(_ i: Int) -> UIImage?? {
        let size = uiSizes[i]
        var len = 0
        guard let p = threers_vol_ui(vol, UInt32(i), UInt32(size.w), UInt32(size.h), uiScales[i], &len) else { return nil }
        if len == 0 { return .some(nil) }
        guard len == size.w * size.h * 4 else { return nil }
        let data = Data(bytes: p, count: len)
        guard let provider = CGDataProvider(data: data as CFData),
              let image = CGImage(width: size.w, height: size.h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: size.w * 4,
                                  space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
        else { return nil }
        return .some(UIImage(cgImage: image))
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
        let size = uiSizes[which]
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
                TextField(c.label, text: Binding(get: { model.texts[c.key] ?? "" }, set: { model.type(c, $0) }), axis: c.role == "textbox" ? .vertical : .horizontal)
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

struct Volume: View {
    @Environment(\.openImmersiveSpace) private var open
    @Environment(\.dismissImmersiveSpace) private var dismiss
    @StateObject private var model = VolumeModel()
    @State private var updates: EventSubscription?
    @State private var dragged: CGFloat = 0

    var body: some View {
        RealityView { content in
            await model.load()
            content.add(model.root)
            content.add(model.hit)
            updates = content.subscribe(to: SceneEvents.Update.self) { _ in model.tick() }
        }
        .gesture(DragGesture().targetedToAnyEntity()
            .onChanged { v in
                model.turn(Float(v.translation.width - dragged) * 0.01)
                dragged = v.translation.width
            }
            .onEnded { _ in dragged = 0 })
        // the app's UI, all threers-ui: the sidebar beside the volume, on its
        // left as the web docks it; the toolbar under it, clear of the
        // window bar
        .ornament(attachmentAnchor: .scene(.leading), contentAlignment: .trailing) {
            if let image = model.ui[0] {
                SurfaceImage(model: model, which: 0, image: image, radius: 16).padding(.trailing, 40)
            }
        }
        .ornament(attachmentAnchor: .scene(.bottom), contentAlignment: .top) {
            if let image = model.ui[1] {
                SurfaceImage(model: model, which: 1, image: image, radius: 38).padding(.top, 180)
            }
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
        WindowGroup { Volume() }
            .windowStyle(.volumetric)
            .defaultSize(width: Double(volumeSide), height: Double(volumeSide), depth: Double(volumeSide), in: .meters)
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
    }
}
#else
@main
struct ThreersApp: App {
    var body: some SwiftUI.Scene {
        WindowGroup { HostView().ignoresSafeArea() }
    }
}
#endif
