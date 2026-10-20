//! The wgpu backend's material models against three.js r165, computed
//! analytically (see `tests/support/lighting_truth.rs`): one sphere per
//! material under ambient, hemisphere, directional, point and spot lights,
//! every sphere pixel compared with the bytes three.js's formulas give.
//! Skipped when no GPU adapter is available.

#[path = "support/lighting_truth.rs"]
mod truth;

use threers::HeadlessRenderer;

fn renderer() -> Option<HeadlessRenderer> {
    let mut r = HeadlessRenderer::builder().size(truth::N, truth::N).build().ok()?;
    r.set_msaa(1);
    Some(r)
}

#[test]
fn every_material_model_matches_three_js() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let cam = truth::camera();
    let mut failures = Vec::new();
    for (name, material, model) in truth::cases() {
        let mut scene = truth::sphere_scene(material);
        let rgba = r.render_to_rgba(&mut scene, &cam);
        let (n, err, at, got, want) = truth::compare(model, &rgba);
        eprintln!("wgpu {name:>15}: {n} px, worst {err} at {at:?}: got {got:?} want {want:?}");
        assert!(n > 5000, "{name}: only {n} pixels compared");
        if err > 2 {
            failures.push(format!("{name}: {err} levels off at {at:?} (got {got:?}, three.js {want:?})"));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Shadow receipt per material model and caster type, against three.js.
#[test]
fn every_material_receives_shadows_as_three_js_does() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = truth::check_shadows("wgpu", &mut |scene, cam| r.render_to_rgba(scene, cam));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// An `InstancedMesh` casts its shadow — instance by instance, morphed per
/// instance — under every light type, where three.js puts it.
#[test]
fn instanced_meshes_cast_shadows_as_three_js_does() {
    let Some(mut r) = renderer() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = truth::check_instanced_shadows("wgpu", &mut |scene, cam| r.render_to_rgba(scene, cam));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[path = "support/blending.rs"]
mod blending;

/// Transparent surfaces blend as three.js's default renderer blends them —
/// on the encoded bytes — on the default (sRGB) target and on a plain one.
#[test]
fn transparent_surfaces_blend_as_three_js_does() {
    let mut failures = Vec::new();
    for format in [wgpu::TextureFormat::Rgba8UnormSrgb, wgpu::TextureFormat::Rgba8Unorm] {
        for msaa in [1, 4] {
            let Ok(mut r) = threers::HeadlessRenderer::builder()
                .size(blending::N, blending::N)
                .color_format(format)
                .build()
            else {
                eprintln!("skipping: no GPU adapter");
                return;
            };
            r.set_msaa(msaa);
            failures.extend(blending::check(&format!("wgpu {format:?} msaa {msaa}"), &mut |s, c| {
                r.render_to_rgba(s, c)
            }));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// A transparent `ShaderMaterial` blends on the encoded bytes too: three.js
/// mixes whatever the shader wrote (after its `colorspace_fragment`) with the
/// canvas's bytes. The shader here ends the documented way, with
/// `framebuffer_encode`, and returns the material's opacity.
#[test]
fn transparent_shader_materials_blend_as_three_js_does() {
    use threers::core::{Mesh, Object3D};
    use threers::geometries::PlaneGeometry;
    use threers::materials::{Material, ShaderMaterial};
    use threers::math::Vector3;
    const FRAG: &str = "@fragment\nfn fs_main(in: VsOut) -> @location(0) vec4<f32> {\n    let c = u_data.data[0];\n    return vec4<f32>(framebuffer_encode(c.rgb), c.a);\n}\n";
    let mut failures = Vec::new();
    for msaa in [1, 4] {
        let Ok(mut r) = threers::HeadlessRenderer::builder().size(blending::N, blending::N).build() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        r.set_msaa(msaa);
        for (back, front, a) in blending::CASES {
            let mut scene = blending::back_scene(back);
            let lin = threers::math::Color::from_hex(front);
            let mut m = ShaderMaterial::new(FRAG).with_data(vec![[lin.r, lin.g, lin.b, a]]);
            m.transparent = true;
            let mut f = Object3D::mesh(Mesh::new(PlaneGeometry::new(2.5, 2.5), Material::Shader(m)));
            f.position = Vector3::new(0.0, 0.0, 0.5);
            scene.add(f);
            let rgba = r.render_to_rgba(&mut scene, &blending::camera());
            failures.extend(blending::judge(&format!("wgpu msaa {msaa}"), "shader", &rgba, back, front, a));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The frames that take other routes through the renderer blend the same
/// way: a pane with a partial clear under MSAA (drawn into a pane target of
/// its own and composited back) and a frame with screen-space refraction in
/// it (the opaque pass goes into the capture first).
#[test]
fn transparent_surfaces_blend_as_three_js_does_in_panes_and_refraction_frames() {
    use threers::core::{Mesh, Object3D};
    use threers::geometries::SphereGeometry;
    use threers::materials::{PhysicalMaterial, TransparencyMode};
    use threers::math::{Color, Vector3};
    let n = blending::N as f32;
    let mut failures = Vec::new();
    for msaa in [1, 4] {
        let Ok(mut r) = threers::HeadlessRenderer::builder().size(blending::N, blending::N).build() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        r.set_msaa(msaa);
        // A centred pane, scissored: a partial clear.
        r.renderer().set_viewport(n / 4.0, n / 4.0, n / 2.0, n / 2.0);
        r.renderer().set_scissor(n / 4.0, n / 4.0, n / 2.0, n / 2.0);
        r.renderer().set_scissor_test(true);
        failures.extend(blending::check_kinds(&format!("wgpu pane msaa {msaa}"), &blending::FRONTS, &mut |s, c| {
            r.render_to_rgba(s, c)
        }));
        r.renderer().set_scissor_test(false);
        r.renderer().reset_viewport();
        r.renderer().reset_scissor();
        // Screen-space refraction glass, off in a corner.
        for kind in blending::FRONTS {
            for (back, front, a) in blending::CASES {
                let mut scene = blending::scene_with(back, front, a, kind);
                let mut glass = PhysicalMaterial::new(Color::WHITE);
                glass.transmission = 1.0;
                glass.transparency = TransparencyMode::Refract;
                let mut g = Object3D::mesh(Mesh::new(SphereGeometry::new(0.1, 12, 8), glass.into()));
                g.position = Vector3::new(-0.85, 0.85, 0.8);
                scene.add(g);
                let rgba = r.render_to_rgba(&mut scene, &blending::camera());
                failures.extend(blending::judge(
                    &format!("wgpu refraction msaa {msaa}"),
                    &format!("{kind:?}"),
                    &rgba,
                    back,
                    front,
                    a,
                ));
            }
        }
    }
    assert!(failures.is_empty(), "{} failures:\n{}", failures.len(), failures.join("\n"));
}
