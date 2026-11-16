// Opens a .usdz with RealityKit, as AR Quick Look and `Entity(contentsOf:)`
// do, and prints what it found: a line per model entity — its triangles,
// then per material its kind and which maps it has. Run by
// tests/realitykit_usdz.rs.
//
//   usdz_check file.usdz

import Foundation
import RealityKit

@main
struct Check {
    @MainActor static func main() async {
        let url = URL(fileURLWithPath: CommandLine.arguments[1])
        let root: Entity
        do {
            root = try await Entity(contentsOf: url)
        } catch {
            print("error \(error)")
            exit(1)
        }
        func walk(_ e: Entity) {
            if let m = e.components[ModelComponent.self] {
                var triangles = 0
                for model in m.mesh.contents.models {
                    for part in model.parts { triangles += (part.triangleIndices?.count ?? 0) / 3 }
                }
                var line = "model \(e.name) triangles=\(triangles)"
                for material in m.materials {
                    if let p = material as? PhysicallyBasedMaterial {
                        line += " pbr"
                        if p.baseColor.texture != nil { line += "+color" }
                        if p.normal.texture != nil { line += "+normal" }
                        if p.roughness.texture != nil { line += "+roughness" }
                        if p.emissiveColor.texture != nil { line += "+emissive" }
                        line += String(format: ":clearcoat=%.2f", p.clearcoat.scale)
                    } else {
                        line += " \(type(of: material))"
                    }
                }
                print(line)
            }
            e.children.forEach(walk)
        }
        walk(root)
    }
}
