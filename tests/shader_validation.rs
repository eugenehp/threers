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
    validate("SHADER_SOURCE", threers::renderer::shader::SHADER_SOURCE);
}

#[test]
fn auxiliary_shaders_are_valid() {
    use threers::renderer::shader as s;
    validate("POSTFX_SHADER", s::POSTFX_SHADER);
    validate("SS_BLIT_SHADER", s::SS_BLIT_SHADER);
    validate("TAA_SHADER", s::TAA_SHADER);
    validate("OIT_RESOLVE_SHADER", s::OIT_RESOLVE_SHADER);
}
