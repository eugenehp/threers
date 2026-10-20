//! `InstancedMesh` with material arrays and morph targets (shared and per
//! instance, three.js r159+'s `setMorphAt`), morphed 4-component vertex
//! colours, and morph targets on a `ShaderMaterial`, on the wgpu backend —
//! against three.js's definitions (see `tests/support/instanced_truth.rs`).
//! Skipped when no GPU adapter is available.

#[path = "support/instanced_truth.rs"]
mod truth;

use threers::HeadlessRenderer;

fn renderer() -> Option<HeadlessRenderer> {
    let mut r = HeadlessRenderer::builder().size(truth::N, truth::N).build().ok()?;
    r.set_msaa(1);
    Some(r)
}

#[test]
fn instanced_meshes_draw_groups_and_morphs_as_three_js_does() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = truth::check("wgpu", &mut |scene, cam| r.render_to_rgba(scene, cam));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// three.js `instanceColor` multiplies the diffuse colour (and any vertex
/// colour) per instance.
#[test]
fn instance_colours_tint_each_instance_as_three_js_does() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = truth::check_instance_colours("wgpu", &mut |scene, cam| r.render_to_rgba(scene, cam));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn four_component_colours_morph_with_their_alpha() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = truth::check_colour_morph("wgpu", &mut |scene, cam| r.render_to_rgba(scene, cam));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// three.js applies morph targets to a `ShaderMaterial` through
/// `#include <morphtarget_vertex>`; here the default vertex stage does it, and
/// a custom one calls the preamble's `morph_position`. Either way the shape
/// drawn is the morphed one — the same pixels as a `MeshBasicMaterial` on the
/// same morphed mesh, which `tests/wgpu_groups_morph.rs` holds to the
/// raycaster.
#[test]
fn shader_materials_draw_morph_targets() {
    use std::sync::Arc;
    use threers::core::{BufferAttribute, Mesh, Object3D};
    use threers::geometries::PlaneGeometry;
    use threers::materials::{BasicMaterial, Material, ShaderMaterial};
    use threers::math::Color;
    use threers::scene::Scene;
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    const FRAG: &str = "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n    return vec4<f32>(1.0, 0.0, 0.0, 1.0);\n}\n";
    const VERT: &str = "@vertex\nfn vs_custom(in: VsIn, @builtin(vertex_index) vi: u32) -> VsOut {\n    var out = standard_vertex(in);\n    let world = mesh.model * vec4<f32>(morph_position(in.position, vi), 1.0);\n    out.world_pos = world.xyz;\n    out.clip_pos = frame.view_proj * world;\n    out.clip_zw = out.clip_pos.zw;\n    return out;\n}\n";
    let geometry = {
        let mut g = PlaneGeometry::new(0.8, 0.8);
        let base = g.get_attribute("position").unwrap().array.clone();
        let t: Vec<f32> = base
            .chunks_exact(3)
            .flat_map(|p| [p[0] * 2.0, p[1] + 0.5, p[2]])
            .collect();
        g.set_morph_attribute("position", vec![BufferAttribute::new(t, 3)]);
        Arc::new(g)
    };
    let render = |r: &mut threers::HeadlessRenderer, material: Material| {
        let mut mesh = Mesh::from_arc(geometry.clone(), Arc::new(material));
        mesh.morph_influences = vec![0.75];
        let mut scene = Scene::new();
        scene.background = Color::from_hex(0x000000);
        scene.add(Object3D::mesh(mesh));
        r.render_to_rgba(&mut scene, &truth::camera())
    };
    let coverage = |img: &[u8]| -> Vec<bool> { img.chunks(4).map(|p| p[0] > 128).collect() };
    let reference = coverage(&render(&mut r, BasicMaterial::new(Color::from_hex(0xff0000)).into()));
    let lit = reference.iter().filter(|&&b| b).count();
    for (name, material) in [
        ("default vertex", ShaderMaterial::new(FRAG)),
        ("custom vertex", ShaderMaterial::new(FRAG).with_vertex(VERT)),
    ] {
        let got = coverage(&render(&mut r, Material::Shader(material)));
        let differ = got.iter().zip(&reference).filter(|(a, b)| a != b).count();
        eprintln!("shader material, {name}: {lit} lit in the reference, {differ} differ");
        assert!(lit > 2000, "the morphed reference covers {lit} pixels");
        // Edge pixels may round either way; the unmorphed square would differ
        // by thousands.
        assert!(differ < 40, "{name}: {differ} pixels differ from the morphed shape");
    }
}
