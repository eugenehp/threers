//! Each texture keeps its own sampler state and uv transform on wgpu — a
//! material whose maps disagree gets a shader variant
//! (`renderer::material_maps`). See `tests/support/material_variants.rs`.
//! Skipped without a GPU adapter.

#[path = "support/material_variants.rs"]
mod support;

#[test]
fn each_texture_keeps_its_own_sampler_and_transform() {
    let Ok(mut r) = threers::HeadlessRenderer::builder()
        .size(support::N, support::N)
        .build()
    else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let failures = support::check("wgpu", &mut |s, c| r.render_to_rgba(s, c));
    assert!(
        failures.is_empty(),
        "{} failures:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// Every map a material binds is uploaded first: an `iridescenceThicknessMap`
/// was bound but never uploaded, and the first frame drawing one panicked.
#[test]
fn every_bound_map_is_uploaded() {
    use std::sync::Arc;
    use threers::materials::PhysicalMaterial;
    use threers::math::Color;
    use threers::textures::{Texture, TextureFormat};
    let Ok(mut r) = threers::HeadlessRenderer::builder().size(16, 16).build() else {
        eprintln!("skipping: no GPU adapter");
        return;
    };
    let mut m = PhysicalMaterial::new(Color::WHITE);
    m.iridescence = 1.0;
    m.iridescence_thickness_map = Some(Arc::new(Texture::new(
        1,
        1,
        TextureFormat::Rgba8Unorm,
        vec![128; 4],
    )));
    let mut scene = threers::scene::Scene::new();
    scene.add(threers::core::Object3D::mesh(threers::core::Mesh::new(
        threers::geometries::PlaneGeometry::new(1.0, 1.0),
        m.into(),
    )));
    let mut cam = threers::cameras::PerspectiveCamera::new(50.0, 1.0, 0.1, 10.0);
    cam.position = threers::math::Vector3::new(0.0, 0.0, 3.0);
    r.render_to_rgba(&mut scene, &cam);
}
