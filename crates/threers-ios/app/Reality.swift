// RealityKit drawing a threers app's scene: its meshes, materials, maps,
// lights and the light round it, kept in step with the app each frame. The
// same on visionOS (a volume), iOS and iPadOS (a RealityView, on the screen
// or over the camera) and the Mac (a RealityView in a window).
//
// RealityKit has no C API: the app runs in Rust (`threers_ios::volume`), its
// scene read out after each frame, and this builds the entities, rebuilding
// only what changed and rewriting a posed mesh's vertices in place.

import Foundation
import Metal
import RealityKit
import SwiftUI
#if os(macOS)
import AppKit
typealias RKColor = NSColor
#else
import UIKit
typealias RKColor = UIColor
#endif

@_silgen_name("threers_vol_create") func threers_vol_create() -> UnsafeMutableRawPointer?
@_silgen_name("threers_vol_frame") func threers_vol_frame(_ v: UnsafeMutableRawPointer?) -> UInt32
@_silgen_name("threers_vol_object") func threers_vol_object(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ keys: UnsafeMutablePointer<UInt64>, _ floats: UnsafeMutablePointer<Float>) -> Bool
@_silgen_name("threers_vol_geometry") func threers_vol_geometry(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ ptrs: UnsafeMutablePointer<UnsafeRawPointer?>, _ counts: UnsafeMutablePointer<UInt32>) -> Bool
@_silgen_name("threers_vol_texture") func threers_vol_texture(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ w: UnsafeMutablePointer<UInt32>, _ h: UnsafeMutablePointer<UInt32>, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_vol_map") func threers_vol_map(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ slot: UInt32, _ w: UnsafeMutablePointer<UInt32>, _ h: UnsafeMutablePointer<UInt32>, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_vol_uvs") func threers_vol_uvs(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<Float>?
@_silgen_name("threers_vol_instances") func threers_vol_instances(_ v: UnsafeMutableRawPointer?, _ i: UInt32, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<Float>?
@_silgen_name("threers_vol_lights") func threers_vol_lights(_ v: UnsafeMutableRawPointer?, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<Float>?
@_silgen_name("threers_vol_lighting") func threers_vol_lighting(_ v: UnsafeMutableRawPointer?) -> UInt32
@_silgen_name("threers_vol_drawing") func threers_vol_drawing(_ v: UnsafeMutableRawPointer?) -> UInt32
@_silgen_name("threers_vol_background") func threers_vol_background(_ v: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<Float>)
@_silgen_name("threers_vol_environment") func threers_vol_environment(_ v: UnsafeMutableRawPointer?, _ width: UInt32, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<Float>?
@_silgen_name("threers_vol_root") func threers_vol_root(_ v: UnsafeMutableRawPointer?, _ out: UnsafeMutablePointer<Float>)
@_silgen_name("threers_vol_turn") func threers_vol_turn(_ v: UnsafeMutableRawPointer?, _ radians: Float)
@_silgen_name("threers_vol_tap") func threers_vol_tap(_ v: UnsafeMutableRawPointer?, _ u: Float, _ v2: Float)
@_silgen_name("threers_vol_ray") func threers_vol_ray(_ v: UnsafeMutableRawPointer?, _ ray: UnsafePointer<Float>)
@_silgen_name("threers_vol_draw") func threers_vol_draw(_ v: UnsafeMutableRawPointer?, _ qx: Float, _ qy: Float, _ qz: Float, _ qw: Float, _ size: UInt32, _ texture: UnsafeMutableRawPointer, _ commandBuffer: UnsafeMutableRawPointer) -> Bool
@_silgen_name("threers_vol_ui") func threers_vol_ui(_ v: UnsafeMutableRawPointer?, _ which: UInt32, _ w: UInt32, _ h: UInt32, _ scale: Float, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_vol_ui_hit") func threers_vol_ui_hit(_ v: UnsafeMutableRawPointer?, _ which: UInt32, _ x: Float, _ y: Float, _ scale: Float) -> Bool
@_silgen_name("threers_vol_insets") func threers_vol_insets(_ v: UnsafeMutableRawPointer?, _ top: Float, _ bottom: Float, _ left: Float, _ right: Float)
@_silgen_name("threers_vol_controls") func threers_vol_controls(_ v: UnsafeMutableRawPointer?, _ which: UInt32, _ len: UnsafeMutablePointer<Int>) -> UnsafePointer<UInt8>?
@_silgen_name("threers_ios_tap") func threers_ios_tap(_ x: Float, _ y: Float)
@_silgen_name("threers_ios_press") func threers_ios_press(_ key: UInt64)
@_silgen_name("threers_ios_text") func threers_ios_text(_ key: UInt64, _ text: UnsafePointer<UInt8>?, _ len: Int)
@_silgen_name("threers_ios_touch") func threers_ios_touch(_ which: UInt32, _ phase: UInt32, _ x: Float, _ y: Float)
@_silgen_name("threers_ios_take_view_request") func threers_ios_take_view_request() -> UInt32
@_silgen_name("threers_ios_reality_mode") func threers_ios_reality_mode(_ ar: Bool)
@_silgen_name("threers_ios_take_quick_look") func threers_ios_take_quick_look(_ out: UnsafeMutablePointer<UInt8>, _ cap: Int) -> Int

/// Floats a mesh (`threers_ios::volume::OBJECT_FLOATS`), keys a mesh, floats a light.
let objectFloats = 48
let objectKeys = 4
let lightFloats = 16

/// Sixteen floats, column-major.
func rkMatrix(_ f: UnsafePointer<Float>) -> simd_float4x4 {
    simd_float4x4(SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]), SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15]))
}

/// RGBA8, straight alpha, sRGB, as a picture.
func rgbaImage(_ p: UnsafePointer<UInt8>, _ len: Int, _ w: Int, _ h: Int) -> CGImage? {
    guard len == w * h * 4, let provider = CGDataProvider(data: Data(bytes: p, count: len) as CFData) else { return nil }
    return CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                   space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue),
                   provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
}

/// Linear RGBA floats as an HDR picture (an environment's light).
func floatImage(_ p: UnsafePointer<Float>, _ count: Int, _ w: Int, _ h: Int) -> CGImage? {
    guard count == w * h * 4, let provider = CGDataProvider(data: Data(bytes: p, count: count * 4) as CFData) else { return nil }
    let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.floatComponents.rawValue | CGBitmapInfo.byteOrder32Little.rawValue)
    return CGImage(width: w, height: h, bitsPerComponent: 32, bitsPerPixel: 128, bytesPerRow: w * 16,
                   space: CGColorSpace(name: CGColorSpace.extendedLinearSRGB)!, bitmapInfo: info,
                   provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
}

private func rkColor(_ f: [Float], _ at: Int) -> RKColor {
    RKColor(red: CGFloat(f[at]), green: CGFloat(f[at + 1]), blue: CGFloat(f[at + 2]), alpha: 1)
}

/// The app's meshes, lights and environment as entities under `model`
/// (whose frame is the scene's), kept in step with its scene each frame: a
/// mesh rebuilt when its geometry changes, rewritten in place when its pose
/// does, a material rebuilt when its material does.
@available(iOS 18.0, macOS 15.0, *)
@MainActor final class RealityScene {
    let vol: UnsafeMutableRawPointer?
    /// The scene's frame: everything of the app's is under it.
    let model: Entity
    /// Whether a room surrounds the scene (a volume, AR): with
    /// `RealityLighting::Auto` it is lit by the room's light, else by its own.
    var room = true
    /// Metres a unit of the scene measures as shown: a light's reach and
    /// power scale with it, so the scene is lit as at its own size.
    var metres: Float = 1
    private struct Built { var geometry: UInt64; var material: UInt64; var motion: UInt64 }
    /// A low-level mesh as built, and what it was built for: rewritten in
    /// place while the counts and the layout hold.
    private struct Buffers { let mesh: LowLevelMesh; let layout: [Int] }
    private var entities: [UInt64: ModelEntity] = [:]
    private var built: [UInt64: Built] = [:]
    private var buffers: [UInt64: Buffers] = [:]
    /// An instanced mesh drawn by copies where the system has no instancing
    /// (before iOS, macOS and visionOS 26): its model, its copies.
    private var prototypes: [UInt64: ModelComponent] = [:]
    private var copies: [UInt64: [ModelEntity]] = [:]
    private var instanceData: [UInt64: AnyObject] = [:]
    private var lights: [Entity] = []
    /// The environment's light, when the app's lights the scene.
    private let ibl = Entity()
    private var iblReady = false
    /// Whether the entities were last told to take their light from `ibl`.
    private var receiving = false
    private var vertexColor: ShaderGraphMaterial?

    init(vol: UnsafeMutableRawPointer?, model: Entity) {
        self.vol = vol
        self.model = model
        model.addChild(ibl)
    }

    func load() async {
        vertexColor = try? await ShaderGraphMaterial(named: "/Root/VertexColor", from: "VertexColor.usda", in: .main)
    }

    /// `RealityLighting` resolved: 1 the room's, 2 the app's, 3 both.
    private var lighting: UInt32 {
        let m = threers_vol_lighting(vol)
        return m == 0 ? (room ? 1 : 2) : m
    }

    /// The scene's `n` meshes (`threers_vol_frame`), its lights and its
    /// environment, as of the app's last frame.
    func sync(_ n: UInt32) {
        let lighting = self.lighting
        syncEnvironment(lighting == 2)
        let receive = lighting == 2 && iblReady
        // told once, and again to an entity rebuilt
        let retell = receive != receiving
        receiving = receive
        var seen = Set<UInt64>()
        var keys = [UInt64](repeating: 0, count: objectKeys)
        var f = [Float](repeating: 0, count: objectFloats)
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
            f.withUnsafeBufferPointer { e.transform = Transform(matrix: rkMatrix($0.baseAddress!)) }
            let was = built[key]
            let instanced = flags & 1024 != 0
            var mc = instanced ? (prototypes[key] ?? e.model) : e.model
            var changed = false
            // its geometry: built anew, or its vertices rewritten (a pose)
            let posed = flags & 2048 != 0 && was?.motion != keys[3]
            if was?.geometry != keys[1] || posed,
               let mesh = mesh(key, i, lines: flags & 16 != 0, points: flags & 64 != 0, tangents: flags & 4096 != 0) {
                mc = ModelComponent(mesh: mesh, materials: mc?.materials ?? [])
                changed = true
            }
            if was?.geometry != keys[1] || was?.material != keys[2] {
                mc?.materials = [flags & (32 | 128) != 0
                    ? spriteMaterial(i, f, twoSided: flags & 8 != 0, cut: flags & 128 != 0, depth: flags & 256 == 0) ?? material(i, f, flags)
                    : material(i, f, flags)]
                changed = true
                // a sprite (a label) keeps facing whoever looks at it
                if flags & 32 != 0 {
                    e.components.set(BillboardComponent())
                }
                // see-through: depth first, then colour, so only the nearest
                // surface blends (single-layer glass; blended alone, a folded
                // cortex drew its folds and triangles through each other)
                if flags & 4 != 0 && flags & 256 == 0 {
                    e.components.set(ModelSortGroupComponent(group: ModelSortGroup(depthPass: .prePass), order: 0))
                } else {
                    e.components.remove(ModelSortGroupComponent.self)
                }
                // a shadow on what is under it (the floor, the volume's foot)
                if flags & 512 != 0 {
                    e.components.set(GroundingShadowComponent(castsShadow: true))
                } else {
                    e.components.remove(GroundingShadowComponent.self)
                }
            }
            if retell || was == nil {
                if receive {
                    e.components.set(ImageBasedLightReceiverComponent(imageBasedLight: ibl))
                } else {
                    e.components.remove(ImageBasedLightReceiverComponent.self)
                }
            }
            if instanced {
                if changed || was?.motion != keys[3], let mc {
                    instances(e, key, i, mc)
                }
            } else if changed {
                e.model = mc
            }
            built[key] = Built(geometry: keys[1], material: keys[2], motion: keys[3])
        }
        for (key, e) in entities where !seen.contains(key) {
            e.removeFromParent()
            entities[key] = nil
            built[key] = nil
            buffers[key] = nil
            prototypes[key] = nil
            copies[key] = nil
            instanceData[key] = nil
        }
        syncLights(lighting >= 2)
    }

    /// Instanced mesh `i`'s copies, at its instances' matrices: the system's
    /// instancing where it has it (26 on), else a child entity each.
    private func instances(_ e: ModelEntity, _ key: UInt64, _ i: UInt32, _ mc: ModelComponent) {
        var len = 0
        guard let p = threers_vol_instances(vol, i, &len) else { return }
        let count = len / 16
        if #available(iOS 26.0, macOS 26.0, visionOS 26.0, *) {
            e.model = mc
            var data = instanceData[key] as? LowLevelInstanceData
            if data == nil || data!.instanceCapacity < count {
                data = try? LowLevelInstanceData(instanceCount: count)
                instanceData[key] = data
            }
            guard let data else { return }
            data.instanceCount = count
            data.withMutableTransforms { out in
                for k in 0..<count { out[k] = rkMatrix(p + k * 16) }
            }
            var c = MeshInstancesComponent()
            c[partIndex: 0] = .init(data: data)
            e.components.set(c)
        } else {
            // the entity draws nothing itself; a copy of its model per instance
            prototypes[key] = mc
            e.model = nil
            var list = copies[key] ?? []
            while list.count > count { list.removeLast().removeFromParent() }
            while list.count < count {
                let c = ModelEntity()
                e.addChild(c)
                list.append(c)
            }
            for (k, c) in list.enumerated() {
                c.model = mc
                c.transform = Transform(matrix: rkMatrix(p + k * 16))
                if receiving { c.components.set(ImageBasedLightReceiverComponent(imageBasedLight: ibl)) }
            }
            copies[key] = list
        }
    }

    /// The app's directional, point and spot lights, as RealityKit's: in
    /// lux and lumens (`threers_ios::reality`), a point's power scaled with
    /// the scene's size so it lights what it lit at its own.
    private func syncLights(_ on: Bool) {
        var len = 0
        let p = on ? threers_vol_lights(vol, &len) : nil
        let count = p == nil ? 0 : len / lightFloats
        while lights.count > count { lights.removeLast().removeFromParent() }
        while lights.count < count {
            let e = Entity()
            model.addChild(e)
            lights.append(e)
        }
        guard let p else { return }
        for (k, e) in lights.enumerated() {
            let f = p + k * lightFloats
            let color = RKColor(red: CGFloat(f[1]), green: CGFloat(f[2]), blue: CGFloat(f[3]), alpha: 1)
            e.position = SIMD3(f[5], f[6], f[7])
            // RealityKit's lights shine along -z
            let way = simd_normalize(SIMD3(f[8], f[9], f[10]))
            let back = SIMD3<Float>(0, 0, -1)
            e.orientation = simd_dot(way, back) < -0.9999 ? simd_quatf(angle: .pi, axis: SIMD3(0, 1, 0)) : simd_quatf(from: back, to: way)
            let shadow = f[15] > 0
            // the reach in the scene's units, as shown; no limit: far past the scene
            let reach = f[11] > 0 ? f[11] * metres : 100
            let power = f[4] * metres * metres
            switch Int(f[0]) {
            case 1:
                e.components.remove(PointLightComponent.self)
                e.components.remove(SpotLightComponent.self)
                e.components.set(DirectionalLightComponent(color: color, intensity: f[4]))
                if shadow { e.components.set(DirectionalLightComponent.Shadow()) } else { e.components.remove(DirectionalLightComponent.Shadow.self) }
            case 2:
                e.components.remove(DirectionalLightComponent.self)
                e.components.remove(SpotLightComponent.self)
                e.components.set(PointLightComponent(color: color, intensity: power, attenuationRadius: reach, attenuationFalloffExponent: f[12]))
            default:
                e.components.remove(DirectionalLightComponent.self)
                e.components.remove(PointLightComponent.self)
                e.components.set(SpotLightComponent(color: color, intensity: power, innerAngleInDegrees: f[14], outerAngleInDegrees: f[13],
                                                    attenuationRadius: reach, attenuationFalloffExponent: f[12]))
                if shadow { e.components.set(SpotLightComponent.Shadow()) } else { e.components.remove(SpotLightComponent.Shadow.self) }
            }
        }
    }

    /// The light round the scene (its environment map, its ambient and
    /// hemisphere lights) as an image-based light, when the app lights it.
    private func syncEnvironment(_ on: Bool) {
        guard on else { return }
        var len = 0
        let width = 256
        guard let p = threers_vol_environment(vol, UInt32(width), &len) else { return }
        guard len > 0, let image = floatImage(p, len, width, width / 2) else {
            ibl.components.remove(ImageBasedLightComponent.self)
            iblReady = false
            return
        }
        Task { @MainActor in
            guard let env = try? await EnvironmentResource(equirectangular: image) else { return }
            self.ibl.components.set(ImageBasedLightComponent(source: .single(env)))
            self.iblReady = true
        }
    }

    /// Sprite `i`'s texture on an unlit, see-through quad.
    private func spriteMaterial(_ i: UInt32, _ f: [Float], twoSided: Bool = false, cut: Bool = false, depth: Bool = true) -> RealityKit.Material? {
        var w: UInt32 = 0, h: UInt32 = 0, len = 0
        guard let p = threers_vol_texture(vol, i, &w, &h, &len), let image = rgbaImage(p, len, Int(w), Int(h)),
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
        // a volume's rays: blended over the brain, hiding none of it
        if !depth { m.writesDepth = false }
        return m
    }

    /// Mesh `i`'s map in `slot` (`threers_ios::reality::MAP_SLOTS`), sampled
    /// repeating when its maps repeat.
    private func map(_ i: UInt32, _ slot: UInt32, _ semantic: TextureResource.Semantic, repeats: Bool) -> MaterialParameters.Texture? {
        var w: UInt32 = 0, h: UInt32 = 0, len = 0
        guard let p = threers_vol_map(vol, i, slot, &w, &h, &len), let image = rgbaImage(p, len, Int(w), Int(h)),
              let resource = try? TextureResource(image: image, options: .init(semantic: semantic))
        else { return nil }
        guard repeats else { return .init(resource) }
        let d = MTLSamplerDescriptor()
        d.sAddressMode = .repeat
        d.tAddressMode = .repeat
        d.minFilter = .linear
        d.magFilter = .linear
        d.mipFilter = .linear
        return .init(resource, sampler: .init(d))
    }

    private func material(_ i: UInt32, _ f: [Float], _ flags: UInt32) -> RealityKit.Material {
        let blended = flags & 4 != 0
        // a line set: unlit, in its colour (in its vertices' colours, the
        // vertex-colour material's)
        if flags & (16 | 64) != 0 && flags & 2 == 0 {
            var m = UnlitMaterial(color: rkColor(f, 16))
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
        let maps = UInt32(f[29])
        let repeats = f[37] > 0
        func slot(_ s: UInt32, _ semantic: TextureResource.Semantic) -> MaterialParameters.Texture? {
            maps & (1 << s) != 0 ? map(i, s, semantic, repeats: repeats) : nil
        }
        var m = PhysicallyBasedMaterial()
        m.baseColor = .init(tint: rkColor(f, 16), texture: slot(0, .color))
        if let t = slot(1, .normal) { m.normal = .init(texture: t) }
        m.roughness = .init(scale: f[20], texture: slot(2, .raw))
        m.metallic = .init(scale: f[21], texture: slot(3, .raw))
        if let t = slot(5, .raw) { m.ambientOcclusion = .init(texture: t) }
        // its glow (an electrode's), as the window draws it
        let glow = slot(4, .color)
        if f[26] > 0 || glow != nil {
            m.emissiveColor = .init(color: rkColor(f, 23), texture: glow)
            m.emissiveIntensity = f[26]
        }
        if f[27] > 0 {
            m.clearcoat = .init(floatLiteral: f[27])
            m.clearcoatRoughness = .init(floatLiteral: f[28])
        }
        if f[32] > 0 || f[33] > 0 || f[34] > 0 {
            m.sheen = .init(tint: rkColor(f, 32))
        }
        let alpha = slot(6, .raw)
        if blended || alpha != nil {
            m.blending = .transparent(opacity: .init(scale: f[19], texture: alpha))
        }
        if f[35] > 0 { m.opacityThreshold = f[35] }
        m.faceCulling = cull
        return m
    }

    /// Mesh `i` as a RealityKit low-level mesh: positions, normals and,
    /// when it has them, vertex colours, texture coordinates and tangents,
    /// interleaved. Rewritten in place (nil: the resource it had still
    /// shows it) while its counts and layout hold: a pose moves, the mesh
    /// stays.
    private func mesh(_ key: UInt64, _ i: UInt32, lines: Bool, points: Bool, tangents: Bool) -> MeshResource? {
        var ptrs = [UnsafeRawPointer?](repeating: nil, count: 5)
        var counts = [UInt32](repeating: 0, count: 3)
        guard threers_vol_geometry(vol, i, &ptrs, &counts), let pp = ptrs[0], let np = ptrs[1], let ip = ptrs[3] else { return nil }
        let (n, cs, m) = (Int(counts[0]), Int(counts[1]), Int(counts[2]))
        guard n > 0, m > 0 else { return nil }
        let pos = pp.assumingMemoryBound(to: Float.self)
        let nor = np.assumingMemoryBound(to: Float.self)
        let col = ptrs[2]?.assumingMemoryBound(to: Float.self)
        let tan = tangents ? ptrs[4]?.assumingMemoryBound(to: Float.self) : nil
        // texture coordinates (a textured mesh: a volume's slice, a map)
        var uvLen = 0
        let uvp = threers_vol_uvs(vol, i, &uvLen)
        let uv = uvLen == n * 2 ? uvp : nil
        // position 0, normal 12, colour 24 (16), uv (8), tangent and bitangent (12 each)
        let colorOffset = 24
        let uvOffset = colorOffset + (col != nil ? 16 : 0)
        let tanOffset = uvOffset + (uv != nil ? 8 : 0)
        let stride = tanOffset + (tan != nil ? 24 : 0)
        let topology: MTLPrimitiveType = points ? .point : lines ? .line : .triangle
        let layout = [n, m, stride, col != nil ? 1 : 0, uv != nil ? 1 : 0, tan != nil ? 1 : 0, Int(topology.rawValue)]
        var fresh: LowLevelMesh?
        let target: LowLevelMesh
        if let b = buffers[key], b.layout == layout {
            target = b.mesh
        } else {
            var attributes: [LowLevelMesh.Attribute] = [
                .init(semantic: .position, format: .float3, offset: 0),
                .init(semantic: .normal, format: .float3, offset: 12),
            ]
            if col != nil { attributes.append(.init(semantic: .color, format: .float4, offset: colorOffset)) }
            if uv != nil { attributes.append(.init(semantic: .uv0, format: .float2, offset: uvOffset)) }
            if tan != nil {
                attributes.append(.init(semantic: .tangent, format: .float3, offset: tanOffset))
                attributes.append(.init(semantic: .bitangent, format: .float3, offset: tanOffset + 12))
            }
            let descriptor = LowLevelMesh.Descriptor(
                vertexCapacity: n, vertexAttributes: attributes,
                vertexLayouts: [.init(bufferIndex: 0, bufferStride: stride)],
                indexCapacity: m, indexType: .uint32)
            guard let made = try? LowLevelMesh(descriptor: descriptor) else { return nil }
            buffers[key] = Buffers(mesh: made, layout: layout)
            fresh = made
            target = made
        }
        var lo = SIMD3<Float>(repeating: .infinity), hi = SIMD3<Float>(repeating: -.infinity)
        target.withUnsafeMutableBytes(bufferIndex: 0) { raw in
            let out = raw.baseAddress!.assumingMemoryBound(to: Float.self)
            let per = stride / 4
            for v in 0..<n {
                let o = out + v * per
                let p = SIMD3(pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2])
                lo = simd_min(lo, p)
                hi = simd_max(hi, p)
                o[0] = p.x; o[1] = p.y; o[2] = p.z
                let nv = SIMD3(nor[v * 3], nor[v * 3 + 1], nor[v * 3 + 2])
                o[3] = nv.x; o[4] = nv.y; o[5] = nv.z
                if let c = col {
                    o[6] = c[v * cs]; o[7] = c[v * cs + 1]; o[8] = c[v * cs + 2]
                    o[9] = cs == 4 ? c[v * cs + 3] : 1
                }
                if let t = uv {
                    // RealityKit's v runs up the image; the slice's rows down
                    o[uvOffset / 4] = t[v * 2]; o[uvOffset / 4 + 1] = 1 - t[v * 2 + 1]
                }
                if let t = tan {
                    // the bitangent from the normal and the tangent's sign;
                    // both turned with v, which is turned over above
                    let tv = SIMD3(t[v * 4], t[v * 4 + 1], t[v * 4 + 2])
                    let b = simd_cross(nv, tv) * t[v * 4 + 3]
                    let q = tanOffset / 4
                    o[q] = tv.x; o[q + 1] = tv.y; o[q + 2] = tv.z
                    o[q + 3] = -b.x; o[q + 4] = -b.y; o[q + 5] = -b.z
                }
            }
        }
        target.withUnsafeMutableIndices { raw in
            raw.baseAddress!.copyMemory(from: ip, byteCount: m * 4)
        }
        // a line set: its segments as lines (RealityKit's line topology)
        target.parts.replaceAll([LowLevelMesh.Part(indexCount: m, topology: topology, bounds: BoundingBox(min: lo, max: hi))])
        guard let fresh else { return nil }
        return try? MeshResource(from: fresh)
    }

    /// UI surface `which`'s new picture at `w` × `h` pixels: nil when
    /// unchanged, .some(nil) when it is not shown now.
    func uiImage(_ which: UInt32, _ w: Int, _ h: Int, _ scale: Float) -> CGImage?? {
        var len = 0
        guard let p = threers_vol_ui(vol, which, UInt32(w), UInt32(h), scale, &len) else { return nil }
        if len == 0 { return .some(nil) }
        guard let image = rgbaImage(p, len, w, h) else { return nil }
        return .some(image)
    }
}

/// One of a UI's controls (`threers_ui::native`), in its points: for
/// assistive technology over a UI drawn as one picture.
struct UiControl: Decodable, Identifiable {
    let key: String
    let role: String
    let label: String
    let x, y, w, h: CGFloat
    let on, toggle, disabled: Bool
    let value: String?
    var id: String { key }
}

/// A file the app asked to show in AR Quick Look (`threers_ios::quick_look`).
@MainActor func takeQuickLook() -> URL? {
    var buf = [UInt8](repeating: 0, count: 4096)
    let n = threers_ios_take_quick_look(&buf, buf.count)
    guard n > 0, let path = String(bytes: buf[0..<n], encoding: .utf8) else { return nil }
    return URL(fileURLWithPath: path)
}

#if !os(visionOS)
/// RealityKit on a screen (an iPhone's, an iPad's, a Mac's): the app's
/// scene in a `RealityView`, turned with RealityKit's orbit camera, or,
/// on an iPhone or iPad, over the camera on the floor in front of you (the
/// app's `Request::Ar`). Its UI over it: `Place::Screen`.
@available(iOS 18.0, macOS 15.0, *)
@MainActor final class StageModel: ObservableObject {
    static let shared = StageModel()
    let vol = threers_vol_create()
    /// fits the model to a metre across, or to the floor in AR
    let root = Entity()
    /// the scene's frame: its focus at the middle
    let model = Entity()
    lazy var scene = RealityScene(vol: vol, model: model)
    /// The screen's UI and its controls, its size in points and pixels a point.
    @Published var ui: CGImage?
    @Published var controls: [UiControl] = []
    var uiSize = CGSize(width: 1, height: 1)
    var uiScale: CGFloat = 2
    /// Over the camera (iOS): the app asked.
    @Published var ar = false
    @Published var background = Color.black
    /// The scene's side, as shown: a metre in a window, a table top in AR.
    var side: Float { ar ? 0.3 : 1 }
    /// A ray through the screen, into the scene's frame (the RealityView's).
    var cast: ((CGPoint) -> (origin: SIMD3<Float>, direction: SIMD3<Float>)?)?

    init() {
        root.addChild(model)
        scene.room = false
    }

    func tick() {
        let n = threers_vol_frame(vol)
        var r = [Float](repeating: 0, count: 17)
        threers_vol_root(vol, &r)
        r.withUnsafeBufferPointer { model.transform = Transform(matrix: rkMatrix($0.baseAddress!)) }
        let k = side * 0.5 / max(r[16], 1e-4)
        root.scale = SIMD3(repeating: k)
        scene.metres = k
        scene.room = ar
        scene.sync(n)
        var bg = [Float](repeating: 0, count: 4)
        threers_vol_background(vol, &bg)
        let colour = Color(.sRGBLinear, red: Double(bg[0]), green: Double(bg[1]), blue: Double(bg[2]), opacity: Double(bg[3]))
        if colour != background { background = colour }
        let w = Int(uiSize.width * uiScale), h = Int(uiSize.height * uiScale)
        if w > 0, h > 0 {
            switch scene.uiImage(2, w, h, Float(uiScale)) {
            case .some(.some(let image)): ui = image
            case .some(.none): ui = nil
            case .none: break
            }
        }
        var len = 0
        if let p = threers_vol_controls(vol, 2, &len),
           let list = try? JSONDecoder().decode([UiControl].self, from: Data(bytes: p, count: len)) {
            controls = list
        }
        switch threers_ios_take_view_request() {
        case 1: setAr(false)
        case 2: setAr(true)
        default: break
        }
        pollShell()
    }

    func setAr(_ on: Bool) {
        #if os(iOS)
        guard on != ar else { return }
        ar = on
        threers_ios_reality_mode(on)
        #endif
    }

    /// A tap on the scene at `p` (the view's points): a select along the
    /// ray through it.
    func tap(_ p: CGPoint) {
        guard let ray = cast?(p) else { return }
        // the ray in the RealityView's frame, into the scene's
        let inv = model.transformMatrix(relativeTo: nil).inverse
        let o = inv * SIMD4(ray.origin, 1), d = inv * SIMD4(ray.direction, 0)
        var f: [Float] = [o.x, o.y, o.z, d.x, d.y, d.z]
        threers_vol_ray(vol, &f)
    }

    /// A finger on the UI: `phase` 0 down, 1 moved, 2 up, 3 cancelled; at
    /// `p` in its points.
    func touch(_ phase: UInt32, _ p: CGPoint) {
        threers_ios_touch(2, phase, Float(p.x * uiScale), Float(p.y * uiScale))
    }

    /// Whether the UI has something at `p` (its points).
    func onUi(_ p: CGPoint) -> Bool {
        ui != nil && threers_vol_ui_hit(vol, 2, Float(p.x * uiScale), Float(p.y * uiScale), Float(uiScale))
    }

    func resize(_ size: CGSize, scale: CGFloat, insets: (top: CGFloat, bottom: CGFloat, left: CGFloat, right: CGFloat)) {
        uiSize = size
        uiScale = scale
        threers_vol_insets(vol, Float(insets.top), Float(insets.bottom), Float(insets.left), Float(insets.right))
    }
}


@available(iOS 18.0, macOS 15.0, *)
struct RealityStage: View {
    @ObservedObject var model = StageModel.shared
    @State private var updates: EventSubscription?

    var body: some View {
        GeometryReader { g in
            ZStack {
                reality
                ScreenOverlay(model: model)
            }
            .onChange(of: g.size, initial: true) { _, s in
                model.resize(s, scale: displayScale, insets: (g.safeAreaInsets.top, g.safeAreaInsets.bottom, g.safeAreaInsets.leading, g.safeAreaInsets.trailing))
            }
        }
        .background(model.ar ? Color.clear : model.background)
        .ignoresSafeArea()
    }

    @Environment(\.displayScale) private var displayScale

    private var reality: some View {
        RealityView { content in
            await model.scene.load()
            content.add(model.root)
            content.camera = .virtual
            let c = content
            model.cast = { p in c.ray(through: p, in: .local, to: .scene) }
            updates = content.subscribe(to: SceneEvents.Update.self) { _ in model.tick() }
        } update: { content in
            #if os(iOS)
            if model.ar {
                content.camera = .spatialTracking
                // on the floor or a table in front of you
                if model.root.parent?.components[AnchoringComponent.self] == nil {
                    let anchor = AnchorEntity(.plane(.horizontal, classification: .any, minimumBounds: SIMD2(0.2, 0.2)))
                    anchor.addChild(model.root)
                    content.add(anchor)
                    model.root.position = SIMD3(0, model.side * 0.5, 0)
                }
            } else {
                content.camera = .virtual
                if let anchor = model.root.parent, anchor.components[AnchoringComponent.self] != nil {
                    content.remove(anchor)
                    model.root.removeFromParent()
                    model.root.position = .zero
                    content.add(model.root)
                }
            }
            #endif
        }
        .realityViewCameraControls(model.ar ? .none : .orbit)
        .gesture(SpatialTapGesture().onEnded { e in model.tap(e.location) })
    }
}
#endif
