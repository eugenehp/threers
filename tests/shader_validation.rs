//! Static WGSL validation for the built-in shader sources.
//!
//! WGSL is compiled by naga at device time, so `cargo build` says nothing about
//! whether these shaders parse — a typo surfaces only when a scene using that
//! code path is first rendered, on whichever machine runs it. Front-loading the
//! parse + validate here catches it at `cargo test` instead.
//!
//! **Known gap:** naga's uniformity analysis is more permissive than Chrome's
//! (Tint). A `dpdx`/`dpdy` reachable from non-uniform control flow passes here
//! and is rejected in the browser with
//! `'dpdx' must only be called from uniform control flow`. If you touch
//! derivative-using code, re-check it in the browser — `cargo test` alone will
//! not catch that class of error.

use wgpu::naga;

fn validate(name: &str, source: &str) {
    let module = match naga::front::wgsl::parse_str(source) {
        Ok(m) => m,
        Err(e) => panic!("{name}: WGSL parse failed:\n{}", e.emit_to_string(source)),
    };
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    if let Err(e) = validator.validate(&module) {
        panic!("{name}: WGSL validation failed:\n{e:?}");
    }
}

#[test]
fn main_mesh_shader_is_valid() {
    validate("SHADER_SOURCE", threers::renderer::shader::legacy_shader_source());
}

#[test]
fn auxiliary_shaders_are_valid() {
    use threers::renderer::shader as s;
    validate("POSTFX_SHADER", s::POSTFX_SHADER);
    validate("SS_BLIT_SHADER", s::SS_BLIT_SHADER);
    validate("TAA_SHADER", s::TAA_SHADER);
    validate("OIT_RESOLVE_SHADER", s::OIT_RESOLVE_SHADER);
}

/// Every material variant the main shader can be compiled with: each map role
/// alone, all of them together, none, and a few mixes. A variant declares only
/// its own maps, so an accessor or `HAS_*` constant the shader relies on and a
/// variant forgets fails here rather than on the first such material.
#[test]
fn material_map_variants_are_valid() {
    use threers::renderer::material_maps::{prelude, MapRole, MapVariant};
    use threers::renderer::shader::shader_source_with_maps;
    let mut variants = vec![MapVariant::default()];
    let mut all = MapVariant::default();
    for role in MapRole::ALL {
        variants.push(MapVariant::default().with(role));
        all = all.with(role);
    }
    variants.push(all);
    variants.push(MapVariant::default().with(MapRole::Map).with(MapRole::AlphaMap).with(MapRole::BumpMap));
    variants.push(MapVariant::default().with(MapRole::NormalMap).with(MapRole::LightMap).with(MapRole::DisplacementMap));
    variants.push(MapVariant::default().with_env());
    variants.push(all.with_env());
    for v in variants {
        validate(&format!("SHADER_SOURCE {v:?}"), &shader_source_with_maps(&prelude(v)));
    }
}

/// Every material variant translated to GLSL ES 3.0, as wgpu's WebGL2 backend
/// translates it, with no name ANGLE rejects. GLSL reserves the `gl_` prefix,
/// and naga keeps a WGSL name and suffixes a repeat of it: two functions each
/// with a local `gl` become `gl` and `gl_1`, the second of which no WebGL2
/// driver compiles — while naga's own validation, and so every test above,
/// accepts it. Real built-ins (`gl_Position`, `gl_FragCoord`, …) continue
/// with a capital letter.
#[test]
fn material_map_variants_translate_for_webgl2() {
    use threers::renderer::material_maps::{prelude, MapRole, MapVariant};
    use threers::renderer::shader::shader_source_with_maps;
    let mut variants = vec![MapVariant::default()];
    let mut all = MapVariant::default();
    for role in MapRole::ALL {
        variants.push(MapVariant::default().with(role));
        all = all.with(role);
    }
    variants.push(all);
    for v in variants {
        let source = shader_source_with_maps(&prelude(v));
        let module = naga::front::wgsl::parse_str(&source).expect("parses");
        let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), naga::valid::Capabilities::all())
            .validate(&module)
            .expect("validates");
        for (stage, entry) in [(naga::ShaderStage::Vertex, "vs_main"), (naga::ShaderStage::Fragment, "fs_main")] {
            // Pipeline-overridable constants at their defaults, resolved
            // first as wgpu resolves them: the GLSL writer takes none.
            let (module, info) = naga::back::pipeline_constants::process_overrides(
                &module,
                &info,
                Some((stage, entry)),
                &Default::default(),
            )
            .unwrap_or_else(|e| panic!("{v:?} {entry}: {e}"));
            let (module, info) = (&*module, &*info);
            let options = naga::back::glsl::Options {
                version: naga::back::glsl::Version::Embedded { version: 300, is_webgl: true },
                ..Default::default()
            };
            let pipeline = naga::back::glsl::PipelineOptions {
                shader_stage: stage,
                entry_point: entry.to_string(),
                multiview: None,
            };
            let mut out = String::new();
            let mut writer = naga::back::glsl::Writer::new(
                &mut out,
                module,
                info,
                &options,
                &pipeline,
                naga::proc::BoundsCheckPolicies::default(),
            )
            .unwrap_or_else(|e| panic!("{v:?} {entry}: {e}"));
            writer.write().unwrap_or_else(|e| panic!("{v:?} {entry}: {e}"));
            let bad: Vec<&str> = out
                .match_indices("gl_")
                .filter(|(i, _)| *i == 0 || !out.as_bytes()[i - 1].is_ascii_alphanumeric() && out.as_bytes()[i - 1] != b'_')
                .map(|(i, _)| {
                    let end = out[i..].find(|c: char| !(c.is_ascii_alphanumeric() || c == '_')).map_or(out.len(), |e| i + e);
                    &out[i..end]
                })
                .filter(|name| !name.as_bytes().get(3).is_some_and(|c| c.is_ascii_uppercase()))
                .collect();
            assert!(bad.is_empty(), "{v:?} {entry}: GLSL names ANGLE rejects: {bad:?}");
        }
    }
}
