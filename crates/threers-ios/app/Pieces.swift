// The app's pieces (`App::pieces`): parts of its UI standing by themselves
// as RealityKit entities, in the volume or in the room, in no window. Each
// is threers-ui's picture with the system's own controls over it (the look's
// highlight, VoiceOver, text fields), as the UI in the volume's windows is.

#if os(visionOS)
import ARKit
import RealityKit
import SwiftUI

@_silgen_name("threers_vol_gaze") func threers_vol_gaze(_ v: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<Float>)
@_silgen_name("threers_ios_piece_pose") func threers_ios_piece_pose(_ id: UInt32, _ x: Float, _ y: Float, _ z: Float, _ qx: Float, _ qy: Float, _ qz: Float, _ qw: Float, _ scale: Float)
@_silgen_name("threers_ios_head") func threers_ios_head(_ x: Float, _ y: Float, _ z: Float, _ qx: Float, _ qy: Float, _ qz: Float, _ qw: Float)
@_silgen_name("threers_vol_pieces") func threers_vol_pieces(_ v: UnsafeMutableRawPointer?, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<Float>?
@_silgen_name("threers_ios_press_on") func threers_ios_press_on(_ which: UInt32, _ key: UInt64)
@_silgen_name("threers_ios_text_on") func threers_ios_text_on(_ which: UInt32, _ key: UInt64, _ text: UnsafePointer<UInt8>?, _ len: Int)
@_silgen_name("threers_ios_space") func threers_ios_space(_ open: Bool)
@_silgen_name("threers_ios_space_open") func threers_ios_space_open() -> Bool

/// Floats a piece (`threers_ios::volume::PIECE_FLOATS`), and the surface the
/// first is (`PIECES`).
let pieceFloats = 24
let pieceSurfaces: UInt32 = 256

/// One piece's UI: its picture and controls, as the view over it shows them.
@MainActor final class PieceState: ObservableObject {
    let which: UInt32
    @Published var image: CGImage?
    @Published var controls: [NativeControl] = []
    @Published var texts: [String: String] = [:]
    /// UI pixels: the view's points.
    @Published var size = CGSize(width: 1, height: 1)
    /// Its glass, its corners, and how soft a feathered one's edge is.
    @Published var glass = 0
    @Published var radius: CGFloat = 0
    @Published var soft: CGFloat = 0
    /// Liquid glass's colour, and how strong.
    @Published var tint = Color.clear
    /// Light, for dark text (`Glass::is_pale`).
    @Published var pale = false
    /// How much thinner than its kind it is, 0 to 0.9.
    @Published var seeThrough: CGFloat = 0
    /// How far its controls stand out, in the view's points (0: flat), and
    /// those pushed in now, by key: how far, 0 to 1.
    @Published var raised: CGFloat = 0
    @Published var pushed: [String: CGFloat] = [:]
    /// Those held down.
    var held = Set<String>()

    /// Moves what is pushed toward where it is going.
    func settle() {
        guard raised > 0 else { return }
        for c in controls where c.stands {
            let to: CGFloat = held.contains(c.key) ? 0.75 : 0
            let at = pushed[c.key] ?? 0
            if abs(to - at) > 0.01 { pushed[c.key] = at + (to - at) * 0.35 } else if at != to { pushed[c.key] = to }
        }
    }

    /// How far out `c` stands, in points.
    func out(_ c: NativeControl) -> CGFloat { raised * (1 - (pushed[c.key] ?? 0)) }

    init(_ which: UInt32) { self.which = which }

    /// Image pixels a UI pixel.
    static let scale: Float = 3

    func sync(_ scene: RealityScene) {
        let (w, h) = (Int((Float(size.width) * Self.scale).rounded()), Int((Float(size.height) * Self.scale).rounded()))
        switch scene.uiImage(which, w, h, Self.scale) {
        case .some(.some(let i)): image = i
        case .some(.none): image = nil
        case .none: break
        }
        var len = 0
        if let p = threers_vol_controls(scene.vol, which, &len),
           let list = try? JSONDecoder().decode([NativeControl].self, from: Data(bytes: p, count: len)) {
            controls = list
            for c in list where c.edit && (!c.secret || c.value != nil) { texts[c.key] = c.value ?? "" }
        }
    }

    func press(_ c: NativeControl) {
        // the release that ends a drag is not a press
        if dragging || Date().timeIntervalSince(dragEnded) < 0.3 { return }
        if let k = UInt64(c.key) { threers_ios_press_on(which, k) }
    }

    func type(_ c: NativeControl, _ text: String) {
        texts[c.key] = text
        guard let k = UInt64(c.key) else { return }
        var bytes = Array(text.utf8)
        threers_ios_text_on(which, k, &bytes, bytes.count)
    }

    private var touching = false
    private var dragging = false
    private var dragEnded = Date.distantPast

    func drag(_ v: DragGesture.Value, ended: Bool) {
        touch(from: v.startLocation, at: v.location, ended: ended)
    }

    /// A slider moved a step, its thumb dragged a twentieth of its rail
    /// (as assistive technology adjusts it).
    func nudge(_ c: NativeControl, up: Bool) {
        let rail = c.rails?.first.map { $0[2] } ?? c.w
        let step = max(4, rail / 20) * (up ? 1 : -1)
        let from = c.thumb.map { CGPoint(x: $0[0] + $0[2] / 2, y: $0[1] + $0[3] / 2) } ?? CGPoint(x: c.x + c.w / 2, y: c.y + c.h / 2)
        touch(from: from, at: CGPoint(x: from.x + step / 2, y: from.y), ended: false)
        touch(from: from, at: CGPoint(x: from.x + step, y: from.y), ended: true)
        // (not a press: a drag)
        dragEnded = .distantPast
    }

    /// A finger (a pinch) down at `from`, now at `at`, in the view's points.
    func touch(from: CGPoint, at: CGPoint, ended: Bool) {
        let k = Self.scale
        if !touching {
            touching = true
            threers_ios_touch(which, 0, Float(from.x) * k, Float(from.y) * k)
        }
        if hypot(at.x - from.x, at.y - from.y) > 6 { dragging = true }
        threers_ios_touch(which, ended ? 2 : 1, Float(at.x) * k, Float(at.y) * k)
        if ended {
            if dragging { dragEnded = Date() }
            (touching, dragging) = (false, false)
        }
    }
}

/// The view on a piece.
struct PieceView: View {
    @ObservedObject var state: PieceState

    var body: some View {
        content
            .modifier(PieceGlass(kind: state.glass, radius: state.radius, soft: state.soft, tint: state.tint, pale: state.pale, seeThrough: state.seeThrough))
            .opacity(state.image == nil ? 0 : 1)
    }

    private var content: some View {
        let drag = DragGesture(minimumDistance: 0, coordinateSpace: .named("piece"))
            .onChanged { state.drag($0, ended: false) }.onEnded { state.drag($0, ended: true) }
        return ZStack(alignment: .topLeading) {
            if let image = state.image {
                Image(decorative: image, scale: CGFloat(PieceState.scale)).resizable()
                    // what stands out is not on the sheet too
                    .mask { Holes(rects: state.raised > 0 ? state.controls.filter { $0.stands } : []).fill(style: FillStyle(eoFill: true)) }
                    .modifier(Halo(on: (state.seeThrough > 0 && state.glass > 0) || state.glass == 8, strong: state.glass == 8, color: state.pale ? .white : .black))
                    .gesture(drag)
                    .accessibilityHidden(true)
            }
            if state.raised > 0, let image = state.image {
                ForEach(state.controls.filter { $0.stands }) { c in
                    Image(decorative: image, scale: CGFloat(PieceState.scale)).resizable()
                        .frame(width: state.size.width, height: state.size.height)
                        .offset(x: -c.block.minX, y: -c.block.minY)
                        .frame(width: c.block.width, height: c.block.height, alignment: .topLeading)
                        .clipShape(RoundedRectangle(cornerRadius: c.corners))
                        .offset(x: c.block.minX, y: c.block.minY)
                        .offset(z: state.out(c) + 0.5)
                        .allowsHitTesting(false)
                        .accessibilityHidden(true)
                }
            }
            ForEach(state.controls.filter { $0.edit }) { c in
                let text = Binding(get: { state.texts[c.key] ?? "" }, set: { state.type(c, $0) })
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
                .frame(width: c.w, height: c.h)
                .accessibilityLabel(c.label)
                .offset(x: c.x, y: c.y)
            }
            ForEach(state.controls.filter { $0.slide }) { c in
                Gaze(c: c, k: 1)
                    .contentShape(RoundedRectangle(cornerRadius: c.r))
                    .gesture(drag)
                    .accessibilityElement()
                    .accessibilityLabel(c.label)
                    .accessibilityValue(c.value ?? "")
                    // VoiceOver, Switch Control, Dwell Control: a step along
                    .accessibilityAdjustableAction { way in state.nudge(c, up: way == .increment) }
                    .offset(x: c.x, y: c.y)
            }
            ForEach(state.controls.filter { !$0.edit && !$0.slide }) { c in
                Button { state.press(c) } label: { Gaze(c: c, k: 1) }
                    .buttonStyle(PushStyle(state: state, key: c.key))
                    .offset(z: state.raised > 0 && c.stands ? state.out(c) + 1 : 0)
                    .contentShape(RoundedRectangle(cornerRadius: c.r))
                    .help(c.label)
                    .disabled(c.disabled)
                    .accessibilityLabel(c.label)
                    .accessibilityValue(c.toggle ? (c.on ? "On" : "Off") : "")
                    .accessibilityAddTraits(c.toggle && c.on ? .isSelected : [])
                    .simultaneousGesture(DragGesture(minimumDistance: 8, coordinateSpace: .named("piece"))
                        .onChanged { state.drag($0, ended: false) }.onEnded { state.drag($0, ended: true) })
                    .offset(x: c.x, y: c.y)
            }
        }
        .frame(width: state.size.width, height: state.size.height, alignment: .topLeading)
        .coordinateSpace(name: "piece")
    }
}

extension NativeControl {
    /// Pressed, so it stands out where controls do (a row of a table or a
    /// tree lies in it with the rest).
    var stands: Bool { !edit && !slide && !["row", "columnheader", "treeitem"].contains(role) && !typedIn }
    /// A text control, typed into by a native field or the UI's own keyboard.
    var typedIn: Bool { edit || ["textbox", "searchbox", "textarea"].contains(role) }
    /// What of it does, and its corners (none quite square).
    var block: CGRect {
        guard let s = solid, s.count == 5 else { return CGRect(x: x, y: y, width: w, height: h) }
        return CGRect(x: s[0], y: s[1], width: s[2], height: s[3])
    }
    var corners: CGFloat {
        let b = block
        let r = solid?.count == 5 ? solid![4] : self.r
        return max(r, min(6, min(b.width, b.height) / 4))
    }
}

/// All of a sheet but what stands out of it.
private struct Holes: Shape {
    let rects: [NativeControl]
    func path(in r: CGRect) -> Path {
        var p = Path(r)
        for c in rects {
            p.addRoundedRect(in: c.block, cornerSize: CGSize(width: c.corners, height: c.corners))
        }
        return p
    }
}

/// A soft edge round words, for glass too thin to read on by itself.
private struct Halo: ViewModifier {
    let on: Bool
    /// With nothing behind at all: wider, a shade to read in.
    var strong = false
    let color: Color
    func body(content: Content) -> some View {
        if on && strong {
            // a rim: each tight shadow spreads the last, as a stroke would
            content.shadow(color: color, radius: 0.8).shadow(color: color, radius: 0.8).shadow(color: color, radius: 1).shadow(color: color, radius: 1.2)
                .shadow(color: color.opacity(0.85), radius: 3).shadow(color: color.opacity(0.6), radius: 9)
        } else if on {
            content.shadow(color: color.opacity(0.9), radius: 1.5).shadow(color: color.opacity(0.6), radius: 5)
        } else {
            content
        }
    }
}

/// A plain button that says when it is held.
private struct PushStyle: ButtonStyle {
    let state: PieceState
    let key: String
    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .onChange(of: configuration.isPressed) { _, down in
                if down { state.held.insert(key) } else { state.held.remove(key) }
            }
    }
}

/// The glass behind a piece's UI (a slab's an entity, behind it). Kinds as
/// `threers_ios::volume::PIECE_FLOATS` says.
struct PieceGlass: ViewModifier {
    let kind: Int
    let radius: CGFloat
    let soft: CGFloat
    let tint: Color
    let pale: Bool
    let seeThrough: CGFloat

    /// Darkens the glass, so white text reads over a bright room
    /// (`Tokens::GLASS_GROUND`).
    static let scrim = Color.black.opacity(0.5)

    private var shape: RoundedRectangle { RoundedRectangle(cornerRadius: radius, style: .continuous) }

    func body(content: Content) -> some View {
        switch kind {
        case 8:
            // nothing behind: the words' shade and the solid controls alone
            content
        case 1...5 where seeThrough > 0, 7 where seeThrough > 0:
            thinned(content)
        case 1:
            content.background { shape.fill(Self.scrim) }.glassBackgroundEffect(in: shape)
        case 2:
            lit(content, rim: Self.white).glassBackgroundEffect(in: shape).modifier(LookedAt(shape: shape))
        case 3:
            content.background { shape.fill(Self.scrim) }.glassBackgroundEffect(.plate, in: shape)
        case 4:
            content
                .background { shape.fill(Self.scrim).blur(radius: soft / 2) }
                .glassBackgroundEffect(.feathered(padding: soft, softEdgeRadius: soft), in: shape)
        case 5:
            content
                .background { shape.fill(Color.black.opacity(0.3)) }
                .background(.ultraThickMaterial, in: shape)
                .glassBackgroundEffect(in: shape)
        case 6:
            // nothing blurred behind: darker, for the text to stand off
            lit(content.background { shape.fill(Color.black.opacity(seeThrough > 0 ? 0 : 0.62)) }, rim: Self.white).modifier(LookedAt(shape: shape))
        case 7:
            lit(content, rim: Self.colours).glassBackgroundEffect(in: shape).modifier(LookedAt(shape: shape))
        default:
            content
        }
    }

    /// What darkens it: thinned, the blur goes and most of this stays, for
    /// the words to stand off.
    private var shade: Color { seeThrough > 0 ? .black.opacity(0.88 * (1 - 0.15 * seeThrough)) : Self.scrim }

    /// Without the system's glass, which is all there or not: a blur, as
    /// much of it as is left, and what its kind has over it.
    @ViewBuilder private func thinned(_ content: Content) -> some View {
        let left = 1 - seeThrough
        if kind == 2 || kind == 7 {
            lit(content, rim: kind == 7 ? Self.colours : Self.white)
                .background { shape.fill(.ultraThinMaterial).opacity(left * 0.6) }
                .modifier(LookedAt(shape: shape))
        } else {
            content
                .background { shape.fill(shade).blur(radius: soft / 2) }
                .background {
                    if kind == 5 {
                        shape.fill(.ultraThickMaterial).opacity(left)
                    } else {
                        shape.fill(.ultraThinMaterial).opacity(left).mask { shape.fill(.black).padding(soft / 2).blur(radius: soft / 2) }
                    }
                }
        }
    }

    /// Lit as a drop is: its colour, light gathered at its top and foot, a
    /// rim bright where light comes over its upper edge and where it leaves
    /// by the lower.
    private func lit(_ content: some View, rim: AnyShapeStyle) -> some View {
        content
            .background {
                // (a pale one is lit by its own colour: no scrim)
                if !pale { shape.fill(shade) }
                shape.fill(tint).opacity(pale ? 1 - seeThrough / 2 : 1 - seeThrough)
                shape.fill(LinearGradient(stops: [
                    .init(color: .white.opacity(0.1), location: 0),
                    .init(color: .white.opacity(0), location: 0.4),
                    .init(color: .white.opacity(0), location: 0.75),
                    .init(color: .white.opacity(0.08), location: 1),
                ], startPoint: .top, endPoint: .bottom))
            }
            .overlay { shape.strokeBorder(rim, lineWidth: 1.5).allowsHitTesting(false) }
    }

    private static let white = AnyShapeStyle(LinearGradient(stops: [
        .init(color: .white.opacity(0.85), location: 0),
        .init(color: .white.opacity(0.1), location: 0.35),
        .init(color: .white.opacity(0.05), location: 0.65),
        .init(color: .white.opacity(0.5), location: 1),
    ], startPoint: .topLeading, endPoint: .bottomTrailing))

    private static let colours = AnyShapeStyle(AngularGradient(
        colors: [Color.red, .orange, .yellow, .green, .cyan, .blue, .purple, .red].map { $0.opacity(0.9) },
        center: .center))
}

/// Lit where you look.
private struct LookedAt: ViewModifier {
    let shape: RoundedRectangle
    func body(content: Content) -> some View {
        content.contentShape(.hoverEffect, shape).hoverEffect(.highlight)
    }
}

/// The pieces as entities: those in the volume under `volume`, those in the
/// room under `room`, kept in step with the app's each frame.
@MainActor final class Pieces {
    let volume = Entity()
    let room = Entity()
    /// Where the head points, on pieces pressed by looking: a ring, filled
    /// as the dwell runs (`Piece::dwell`).
    let reticle = Entity()
    private let ring = ModelEntity()
    private let fill = ModelEntity()
    let head = HeadTracker()
    /// (THREERS_POSE_LOG: each piece's place in the room, once)
    private var posed = Set<UInt32>()
    /// Whether any stands in the volume: what drags the model then lands
    /// behind them.
    private(set) var inVolume = false

    @MainActor private final class Piece {
        let state: PieceState
        /// Where the app puts it.
        let entity = Entity()
        /// Where it has been taken from there.
        let taken = Entity()
        /// What it is taken by, under it.
        let bar = ModelEntity()
        /// Its solid controls, by key, and its sliders' balls.
        var solids: [String: ModelEntity] = [:]
        var solidMade: [String: [Float]] = [:]
        /// The view, scaled to its metres.
        let face = Entity()
        let slab = ModelEntity()
        /// The slab as made: its size, corners, depth, tint.
        var made: [Float] = []

        init(_ id: UInt32) {
            state = PieceState(pieceSurfaces + id)
            face.components.set(ViewAttachmentComponent(rootView: PieceView(state: state)))
            entity.addChild(taken)
            for e in [slab, face, bar] { taken.addChild(e) }
            bar.isEnabled = false
        }
    }
    private var all: [UInt32: Piece] = [:]

    /// THREERS_PRESS_TEST: once, 20 s in, every control of every piece used
    /// as its view would (a press on each, a slider slid a quarter along,
    /// "typed" typed in each field), to the console.
    /// `all`: one control a frame, each new one as it shows (a menu's items,
    /// a keyboard's keys), up to 60 a piece.
    private var tested = Set<UInt32>()
    private var used: [UInt32: Set<String>] = [:]
    private let started = Date()
    private func pressTest(_ s: PieceState) {
        let mode = ProcessInfo.processInfo.environment["THREERS_PRESS_TEST"] ?? ""
        guard !mode.isEmpty, Date().timeIntervalSince(started) > 20, !s.controls.isEmpty else { return }
        var list = s.controls
        if mode == "all" {
            var done = used[s.which] ?? []
            guard done.count < 60, let next = s.controls.first(where: { !done.contains("\($0.role) \($0.label)") }) else { return }
            done.insert("\(next.role) \(next.label)")
            used[s.which] = done
            list = [next]
        } else if !tested.insert(s.which).inserted {
            return
        }
        for c in list {
            if c.edit {
                s.type(c, "typed")
                print("press test: piece \(s.which) typed in \"\(c.label)\"")
            } else if c.slide {
                let y = c.y + c.h / 2
                for x in stride(from: c.x + 4, through: c.x + c.w * 0.25, by: max(1, c.w / 40)) {
                    s.touch(from: CGPoint(x: c.x + 4, y: y), at: CGPoint(x: x, y: y), ended: false)
                }
                s.touch(from: CGPoint(x: c.x + 4, y: y), at: CGPoint(x: c.x + c.w * 0.25, y: y), ended: true)
                print("press test: piece \(s.which) slid \"\(c.label)\"")
            } else {
                s.press(c)
                print("press test: piece \(s.which) pressed \"\(c.label)\" (\(c.role))")
            }
        }
    }

    /// Points a metre, till the system says how large a view it made is.
    private static let pointsPerMetre: Float = 1360

    func sync(_ scene: RealityScene) {
        var len = 0
        let p = threers_vol_pieces(scene.vol, &len)
        // (copied: drawing a piece calls the app again)
        let f = p.map { Array(UnsafeBufferPointer(start: $0, count: len)) } ?? []
        var seen = Set<UInt32>()
        inVolume = false
        for at in stride(from: 0, to: f.count - f.count % pieceFloats, by: pieceFloats) {
            let v = Array(f[at..<at + pieceFloats])
            let id = UInt32(v[0])
            seen.insert(id)
            let piece = all[id] ?? Piece(id)
            all[id] = piece
            let home = v[1] == 1 ? room : volume
            if piece.entity.parent !== home { home.addChild(piece.entity) }
            if home === volume { inVolume = true }
            update(piece, v)
            // where it is shown in the room's coordinates (moved, turned, in
            // its volume): for pointing at it, while the room is open
            if let m = piece.taken.transformMatrix(relativeTo: .immersiveSpace) {
                let scale = simd_length(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z))
                let q = simd_quatf(float3x3(SIMD3(m.columns.0.x, m.columns.0.y, m.columns.0.z) / scale,
                                            SIMD3(m.columns.1.x, m.columns.1.y, m.columns.1.z) / scale,
                                            SIMD3(m.columns.2.x, m.columns.2.y, m.columns.2.z) / scale))
                threers_ios_piece_pose(id, m.columns.3.x, m.columns.3.y, m.columns.3.z, q.imag.x, q.imag.y, q.imag.z, q.real, scale)
                if ProcessInfo.processInfo.environment["THREERS_POSE_LOG"] != nil, posed.insert(id).inserted {
                    NSLog("pose: piece %u at %.3f %.3f %.3f scale %.3f", id, m.columns.3.x, m.columns.3.y, m.columns.3.z, scale)
                }
            }
            piece.state.sync(scene)
            piece.state.settle()
            pressTest(piece.state)
            solids(piece, v)
            // (a size the UI has not been drawn at yet, or none: not shown)
            piece.entity.isEnabled = piece.state.image != nil
        }
        for id in all.keys where !seen.contains(id) {
            all.removeValue(forKey: id)?.entity.removeFromParent()
        }
        head.send()
        aim(scene)
    }

    /// A flat ring in the `xz` plane, facing up.
    private static func ringMesh(outer: Float, inner: Float) -> MeshResource {
        let n = 48
        var points: [SIMD3<Float>] = []
        var indices: [UInt32] = []
        for i in 0..<n {
            let a = Float(i) / Float(n) * 2 * .pi
            points.append(SIMD3(cos(a) * outer, 0, sin(a) * outer))
            points.append(SIMD3(cos(a) * inner, 0, sin(a) * inner))
            let (o, i0, o1, i1) = (UInt32(2 * i), UInt32(2 * i + 1), UInt32(2 * ((i + 1) % n)), UInt32(2 * ((i + 1) % n) + 1))
            // both faces: seen from either side
            indices += [o, i0, o1, o1, i0, i1, o, o1, i0, o1, i1, i0]
        }
        var d = MeshDescriptor()
        d.positions = MeshBuffers.Positions(points)
        d.normals = MeshBuffers.Normals([SIMD3<Float>](repeating: SIMD3(0, 1, 0), count: points.count))
        d.primitives = .triangles(indices)
        return (try? MeshResource.generate(from: [d])) ?? .generateCylinder(height: 0.0005, radius: outer)
    }

    /// The reticle where the head points, as the volume says.
    private func aim(_ scene: RealityScene) {
        var g = [Float](repeating: 0, count: 8)
        threers_vol_gaze(scene.vol, &g)
        guard g[0] > 0 else {
            reticle.isEnabled = false
            return
        }
        if reticle.parent == nil {
            var m = UnlitMaterial(color: .white)
            m.blending = .transparent(opacity: 0.9)
            // a ring: a disc with a hole (two flat cylinders would sort badly)
            ring.model = ModelComponent(mesh: Self.ringMesh(outer: 0.007, inner: 0.0055), materials: [m])
            var f = UnlitMaterial(color: UIColor(red: 0.4, green: 0.7, blue: 1, alpha: 1))
            f.blending = .transparent(opacity: 0.95)
            fill.model = ModelComponent(mesh: .generateCylinder(height: 0.0006, radius: 0.0055), materials: [f])
            reticle.addChild(ring)
            reticle.addChild(fill)
            room.addChild(reticle)
        }
        reticle.isEnabled = true
        let normal = simd_normalize(SIMD3(g[4], g[5], g[6]))
        // (in front of what stands out of the piece: `raised` is 5 cm at most)
        reticle.position = SIMD3(g[1], g[2], g[3]) + normal * 0.052
        reticle.orientation = simd_quatf(from: SIMD3(0, 1, 0), to: normal)
        // on a control: the fill grows as the dwell runs, and goes once it
        // has pressed (the words under it showing); on the rest, a dot
        let k: Float = g[0] < 1 ? 0.3 : (g[7] >= 1 ? 0 : max(0.05, g[7]))
        fill.scale = SIMD3(k, 1, k)
        fill.isEnabled = k > 0
        ring.isEnabled = g[0] == 1
    }

    private func update(_ piece: Piece, _ v: [Float]) {
        let (w, h, metres) = (v[2], v[3], v[4])
        let state = piece.state
        let size = CGSize(width: CGFloat(w), height: CGFloat(h))
        if state.size != size { state.size = size }
        if state.glass != Int(v[13]) { state.glass = Int(v[13]) }
        if state.radius != CGFloat(v[14]) { state.radius = CGFloat(v[14]) }
        let soft: CGFloat = v[13] == 4 ? CGFloat(v[15]) : 0
        if state.soft != soft { state.soft = soft }
        // (no brighter than white words can be read on, as strong as it is)
        let bright = v[16] * 0.2126 + v[17] * 0.7152 + v[18] * 0.0722
        let dim = Double(min(1, 0.15 / max(bright, 1e-3)))
        let pale = v[22] == 1
        if state.pale != pale { state.pale = pale }
        let seeThrough = CGFloat(v[23])
        if state.seeThrough != seeThrough { state.seeThrough = seeThrough }
        // (pale: as light and as strong as dark words can be read on over a
        // dark room, `Tokens::PALE_GLASS_GROUND`)
        let light = paleTint(v)
        let tint = pale
            ? Color(.sRGBLinear, red: Double(light.x), green: Double(light.y), blue: Double(light.z), opacity: 0.8)
            : Color(.sRGBLinear, red: Double(v[16]) * dim, green: Double(v[17]) * dim, blue: Double(v[18]) * dim, opacity: Double(v[19]))
        if state.tint != tint { state.tint = tint }
        piece.entity.position = SIMD3(v[5], v[6], v[7])
        if v[12] == 1 {
            if piece.entity.components[BillboardComponent.self] == nil { piece.entity.components.set(BillboardComponent()) }
        } else {
            piece.entity.components.remove(BillboardComponent.self)
            piece.entity.orientation = simd_quatf(ix: v[8], iy: v[9], iz: v[10], r: v[11]).normalized
        }
        // the view's own size in metres, as the system made it
        let made = piece.face.components[ViewAttachmentComponent.self]?.bounds.extents.x ?? 0
        let natural = made > 1e-4 ? made : w / Self.pointsPerMetre
        piece.face.scale = SIMD3(repeating: w * metres / natural)
        // (points: as the view's own)
        let raised = CGFloat(v[20] / metres)
        if state.raised != raised { state.raised = raised }
        slab(piece, v, w * metres, h * metres, metres)
        bar(piece, v[21] == 1, w * metres, h * metres)
    }

    /// The bar it is taken by.
    private func bar(_ piece: Piece, _ on: Bool, _ w: Float, _ h: Float) {
        piece.bar.isEnabled = on
        guard on else {
            if piece.taken.components[ManipulationComponent.self] != nil {
                piece.taken.components.remove(ManipulationComponent.self)
                piece.taken.transform = .identity
            }
            return
        }
        piece.bar.position = SIMD3(0, -h / 2 - 0.022, 0.004)
        if piece.taken.components[ManipulationComponent.self] != nil { return }
        let size = SIMD3<Float>(max(0.05, min(w * 0.35, 0.12)), 0.009, 0.009)
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: .white)
        m.roughness = 0.35
        piece.bar.model = ModelComponent(mesh: .generateBox(size: size, cornerRadius: size.y / 2), materials: [m])
        // easier to take than it looks
        piece.bar.components.set(CollisionComponent(shapes: [.generateBox(size: size + SIMD3(0.02, 0.016, 0.016))]))
        piece.bar.components.set(InputTargetComponent())
        piece.bar.components.set(HoverEffectComponent(.highlight(.default)))
        piece.bar.components.set(ManipulationComponent.HitTarget(redirectedEntity: piece.taken))
        var taking = ManipulationComponent()
        taking.releaseBehavior = .stay
        piece.taken.components.set(taking)
    }

    /// Its solid controls: a block under each, the ball of each slider.
    private func solids(_ piece: Piece, _ v: [Float]) {
        let (w, h, metres, raised) = (v[2], v[3], v[4], v[20])
        var seen = Set<String>()
        if raised > 0 {
            for c in piece.state.controls {
                if c.stands {
                    let out = Float(piece.state.out(c) / max(piece.state.raised, 1e-3)) * raised
                    let b = c.block
                    block(piece, c, SIMD3(Float(b.midX) - w / 2, h / 2 - Float(b.midY), 0) * metres,
                          SIMD2(Float(b.width), Float(b.height)) * metres, Float(c.corners) * metres, raised, out, ground(v), v[22] == 1)
                    seen.insert(c.key)
                } else if c.typedIn {
                    // a field sunk in a rim
                    rim(piece, c, w, h, metres, raised, ground(v))
                    seen.formUnion((0..<4).map { "\(c.key).rim\($0)" })
                } else if let turn = c.dial {
                    // a knob: a cylinder standing out, its nub where it points
                    knob(piece, c, SIMD3(Float(c.x + c.w / 2) - w / 2, h / 2 - Float(c.y + c.h / 2), 0) * metres,
                         Float(c.w) / 2 * metres, raised, Float(turn), ground(v))
                    seen.insert(c.key)
                } else if let t = c.thumb, t.count == 4 {
                    // its rail a tube under the ball: the track, its fill over it
                    for (i, b) in (c.rails ?? []).enumerated() where b.count == 8 {
                        let thick = max(Float(b[3]), 3) * metres * (i == 0 ? 1 : 1.25)
                        let at = SIMD3((Float(b[0] + b[2] / 2)) - w / 2, h / 2 - Float(b[1] + b[3] / 2), 0) * metres
                        let colour = UIColor(red: b[4], green: b[5], blue: b[6], alpha: 1)
                        bar(piece, "\(c.key).rail\(i)", at + SIMD3(0, 0, thick / 2 + Float(i) * 0.0002), SIMD3(Float(b[2]) * metres, thick, thick), thick / 2, colour, shiny: true)
                        seen.insert("\(c.key).rail\(i)")
                    }
                    let r = Float(t[2]) * metres / 2
                    ball(piece, c.key, SIMD3((Float(t[0] + t[2] / 2) - w / 2) * metres, (h / 2 - Float(t[1] + t[3] / 2)) * metres, r * 0.7), r * 1.05,
                         // (white is lost on pale glass)
                         v[22] == 1 ? UIColor(red: c.tint[0], green: c.tint[1], blue: c.tint[2], alpha: 1) : .white)
                    seen.insert(c.key)
                }
            }
        }
        for key in piece.solids.keys where !seen.contains(key) {
            piece.solids.removeValue(forKey: key)?.removeFromParent()
            piece.solidMade.removeValue(forKey: key)
        }
    }

    private func solid(_ piece: Piece, _ key: String) -> ModelEntity {
        if let e = piece.solids[key] { return e }
        let e = ModelEntity()
        piece.taken.addChild(e)
        piece.solids[key] = e
        return e
    }

    private func block(_ piece: Piece, _ c: NativeControl, _ at: SIMD3<Float>, _ size: SIMD2<Float>, _ radius: Float, _ raised: Float, _ out: Float, _ ground: SIMD3<Float>, _ pale: Bool) {
        let e = solid(piece, c.key)
        e.position = at + SIMD3(0, 0, out / 2)
        // pushed in, it is as deep as is left of it
        e.scale = SIMD3(1, 1, max(out / raised, 0.05))
        let fill = (c.fill ?? []).map(Float.init)
        let made = [size.x, size.y, radius, raised, ground.x, ground.y, ground.z, pale ? 1 : 0] + fill
        guard piece.solidMade[c.key] != made else { return }
        piece.solidMade[c.key] = made
        let round = min(radius, min(size.x, size.y) / 2)
        let mesh = MeshResource.generateBox(size: SIMD3(size.x, size.y, raised), majorCornerRadius: max(round, 0.0005), minorCornerRadius: min(raised * 0.35, max(round, 0.0005)))
        var m = PhysicallyBasedMaterial()
        // what it is painted on, over the glass's own grey: solid, for
        // see-through things do not sort against the glass
        let a = fill.count < 4 ? 0 : fill[3]
        let rgb = (0..<3).map { (i: Int) -> Float in ground[i] + ((fill.count < 4 ? ground[i] : fill[i]) - ground[i]) * a }
        m.baseColor = .init(tint: UIColor(red: CGFloat(rgb[0]), green: CGFloat(rgb[1]), blue: CGFloat(rgb[2]), alpha: 1))
        if pale {
            // the room's light alone leaves a pale block grey
            m.emissiveColor = .init(color: UIColor(red: CGFloat(rgb[0]), green: CGFloat(rgb[1]), blue: CGFloat(rgb[2]), alpha: 1))
            m.emissiveIntensity = 0.5
        }
        m.roughness = 0.3
        m.clearcoat = 0.6
        e.model = ModelComponent(mesh: mesh, materials: [m])
        e.components.set(GroundingShadowComponent(castsShadow: true))
    }

    /// The glass's own colour, as a block with none is (sRGB): its grey, with
    /// the glass's tint, dark enough for white words.
    private func ground(_ v: [Float]) -> SIMD3<Float> {
        if v[22] == 1 {
            let c = paleTint(v)
            return SIMD3(srgb(c.x), srgb(c.y), srgb(c.z)) * 0.95
        }
        guard v[13] == 2 else { return SIMD3(repeating: 0.3) }
        let tint = SIMD3(srgb(v[16]), srgb(v[17]), srgb(v[18])) * 0.6
        let c = simd_mix(SIMD3<Float>(repeating: 0.3), tint, SIMD3(repeating: min(1, v[19] * 2)))
        // (as bright as 4.5:1 under white, no more)
        let bright = simd_dot(c * c, SIMD3(0.2126, 0.7152, 0.0722))
        return bright > 0.16 ? c * (0.16 / bright).squareRoot() : c
    }

    /// A pale glass's colour (linear), a luminance of 0.75 or more.
    private func paleTint(_ v: [Float]) -> SIMD3<Float> {
        let c = SIMD3(v[16], v[17], v[18])
        let bright = simd_dot(c, SIMD3(0.2126, 0.7152, 0.0722))
        return bright >= 0.75 ? c : simd_mix(c, SIMD3(repeating: 1), SIMD3(repeating: (0.75 - bright) / (1 - bright)))
    }

    /// A knob `radius` metres round standing `raised` out at `at`, turned
    /// `turn` radians right of up.
    private func knob(_ piece: Piece, _ c: NativeControl, _ at: SIMD3<Float>, _ radius: Float, _ raised: Float, _ turn: Float, _ ground: SIMD3<Float>) {
        let e = solid(piece, c.key)
        let tall = raised * 1.3
        e.position = at + SIMD3(0, 0, tall / 2)
        // its face to you (a cylinder stands along y), turned clockwise
        e.orientation = simd_quatf(angle: -turn, axis: SIMD3(0, 0, 1)) * simd_quatf(angle: .pi / 2, axis: SIMD3(1, 0, 0))
        let made = [radius, tall, ground.x, ground.y, ground.z] + c.tint.map(Float.init)
        guard piece.solidMade[c.key] != made else { return }
        piece.solidMade[c.key] = made
        let body = simd_mix(ground, SIMD3(repeating: 1), SIMD3(repeating: 0.2))
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: UIColor(red: CGFloat(body.x), green: CGFloat(body.y), blue: CGFloat(body.z), alpha: 1))
        m.roughness = 0.35
        m.metallic = 0.3
        m.clearcoat = 0.5
        e.model = ModelComponent(mesh: .generateCylinder(height: tall, radius: radius * 0.92), materials: [m])
        e.components.set(GroundingShadowComponent(castsShadow: true))
        // the nub: on its face (the cylinder's +y), near the rim, towards up
        e.children.removeAll()
        var n = PhysicallyBasedMaterial()
        n.baseColor = .init(tint: UIColor(red: c.tint[0], green: c.tint[1], blue: c.tint[2], alpha: 1))
        n.roughness = 0.2
        let nub = ModelEntity(mesh: .generateBox(size: SIMD3(radius * 0.18, radius * 0.12, radius * 0.5), cornerRadius: radius * 0.05), materials: [n])
        // (+y of the cylinder faces you; its -z is up once stood)
        nub.position = SIMD3(0, tall / 2 + radius * 0.05, -radius * 0.6)
        e.addChild(nub)
    }

    /// A rounded bar, `size` metres, its middle at `at`.
    private func bar(_ piece: Piece, _ key: String, _ at: SIMD3<Float>, _ size: SIMD3<Float>, _ radius: Float, _ color: UIColor, shiny: Bool = false) {
        let e = solid(piece, key)
        e.position = at
        var (r, g, b): (CGFloat, CGFloat, CGFloat) = (1, 1, 1)
        color.getRed(&r, green: &g, blue: &b, alpha: nil)
        let made = [size.x, size.y, size.z, radius, Float(r), Float(g), Float(b)]
        guard piece.solidMade[key] != made else { return }
        piece.solidMade[key] = made
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: color)
        m.roughness = shiny ? 0.2 : 0.35
        m.clearcoat = shiny ? 0.8 : 0.3
        let round = max(0.0002, min(radius, min(size.x, size.y) / 2))
        e.model = ModelComponent(mesh: .generateBox(size: size, majorCornerRadius: round, minorCornerRadius: min(round, size.z / 2)), materials: [m])
        e.components.set(GroundingShadowComponent(castsShadow: true))
    }

    /// A rim round a field, standing a little out: four bars.
    private func rim(_ piece: Piece, _ c: NativeControl, _ w: Float, _ h: Float, _ metres: Float, _ raised: Float, _ ground: SIMD3<Float>) {
        let (x, y, fw, fh) = (Float(c.x), Float(c.y), Float(c.w), Float(c.h))
        let t: Float = 2.5
        let d = raised * 0.35
        let lift = simd_mix(ground, SIMD3(repeating: 1), SIMD3(repeating: 0.25))
        let colour = UIColor(red: CGFloat(lift.x), green: CGFloat(lift.y), blue: CGFloat(lift.z), alpha: 1)
        // top, bottom, left, right: middles and sizes, UI points
        let sides: [(Float, Float, Float, Float)] = [
            (x + fw / 2, y - t / 2, fw + 2 * t, t), (x + fw / 2, y + fh + t / 2, fw + 2 * t, t),
            (x - t / 2, y + fh / 2, t, fh), (x + fw + t / 2, y + fh / 2, t, fh),
        ]
        for (i, s) in sides.enumerated() {
            let at = SIMD3(s.0 - w / 2, h / 2 - s.1, 0) * metres + SIMD3(0, 0, d / 2)
            bar(piece, "\(c.key).rim\(i)", at, SIMD3(s.2 * metres, s.3 * metres, d), t * metres / 2, colour)
        }
    }

    private func ball(_ piece: Piece, _ key: String, _ at: SIMD3<Float>, _ radius: Float, _ color: UIColor) {
        let e = solid(piece, key)
        e.position = at
        var (r, g, b): (CGFloat, CGFloat, CGFloat) = (1, 1, 1)
        color.getRed(&r, green: &g, blue: &b, alpha: nil)
        let made = [radius, Float(r), Float(g), Float(b)]
        guard piece.solidMade[key] != made else { return }
        piece.solidMade[key] = made
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: color)
        m.roughness = 0.15
        m.clearcoat = 1.0
        e.model = ModelComponent(mesh: .generateSphere(radius: radius), materials: [m])
        e.components.set(GroundingShadowComponent(castsShadow: true))
    }

    /// The slab of a liquid, clear or prism glass, behind its view.
    private func slab(_ piece: Piece, _ v: [Float], _ w: Float, _ h: Float, _ metres: Float) {
        guard v[13] == 2 || v[13] == 6 || v[13] == 7, v[15] > 0 else {
            piece.slab.isEnabled = false
            return
        }
        piece.slab.isEnabled = true
        let (radius, depth) = (v[14] * metres, v[15])
        let key = [w, h, radius, depth] + v[16..<20] + [v[23]]
        guard key != piece.made else { return }
        piece.made = key
        // a lip round the view, its edges rounded over as far as it is deep:
        // what is seen of its depth from in front
        let lip = depth * 0.35
        let mesh = MeshResource.generateBox(size: SIMD3(w + 2 * lip, h + 2 * lip, depth), majorCornerRadius: radius + lip, minorCornerRadius: min(depth * 0.45, radius + lip))
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: UIColor(red: CGFloat(srgb(v[16])), green: CGFloat(srgb(v[17])), blue: CGFloat(srgb(v[18])), alpha: 1))
        m.roughness = 0.04
        m.metallic = 0.0
        m.specular = 1.0
        m.clearcoat = 1.0
        m.clearcoatRoughness = 0.02
        m.blending = .transparent(opacity: .init(floatLiteral: v[19] * (1 - v[23])))
        piece.slab.model = ModelComponent(mesh: mesh, materials: [m])
        piece.slab.position.z = -depth / 2 - 0.0005
        piece.slab.components.set(GroundingShadowComponent(castsShadow: true))
    }

    private func srgb(_ c: Float) -> Float {
        c <= 0.0031308 ? c * 12.92 : 1.055 * pow(c, 1 / 2.4) - 0.055
    }
}

/// The head's place and turn in the room, for pieces pressed by looking:
/// ARKit's device anchor, else a fixed head (`THREERS_HEAD="x y z yaw
/// pitch"`, metres and degrees: where no tracking runs, the simulator).
@MainActor final class HeadTracker {
    private let session = ARKitSession()
    private let world = WorldTrackingProvider()
    private var running = false
    private var told = false
    private let fixed: [Float]? = ProcessInfo.processInfo.environment["THREERS_HEAD"].map { $0.split(separator: " ").compactMap { Float($0) } }.flatMap { $0.count == 5 ? $0 : nil }

    func start() async {
        guard fixed == nil, !running else { return }
        guard WorldTrackingProvider.isSupported else {
            NSLog("head: world tracking is not supported here")
            return
        }
        running = true
        do { try await session.run([world]) } catch {
            NSLog("head: %@", "\(error)")
            running = false
        }
    }

    func send() {
        if let f = fixed {
            let yaw = simd_quatf(angle: f[3] * .pi / 180, axis: SIMD3(0, 1, 0))
            let q = yaw * simd_quatf(angle: f[4] * .pi / 180, axis: SIMD3(1, 0, 0))
            threers_ios_head(f[0], f[1], f[2], q.imag.x, q.imag.y, q.imag.z, q.real)
            return
        }
        guard running, world.state == .running,
              let a = world.queryDeviceAnchor(atTimestamp: CACurrentMediaTime()) else { return }
        let m = a.originFromAnchorTransform
        let q = simd_quatf(m)
        if !told {
            told = true
            NSLog("head: tracked at %.2f %.2f %.2f", m.columns.3.x, m.columns.3.y, m.columns.3.z)
        }
        threers_ios_head(m.columns.3.x, m.columns.3.y, m.columns.3.z, q.imag.x, q.imag.y, q.imag.z, q.real)
    }
}

/// The room the pieces stand in: an open space, your surroundings showing.
struct PieceSpace: View {
    @State private var updates: EventSubscription?

    var body: some View {
        RealityView { content in
            let model = VolumeModel.shared
            await model.load()
            content.add(model.pieces.room)
            await model.pieces.head.start()
            // the app goes on though its volume's window is closed
            updates = content.subscribe(to: SceneEvents.Update.self) { _ in model.tick() }
        }
        .onDisappear { threers_ios_space(false) }
    }
}
#endif
