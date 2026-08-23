//! Height maps have to move geometry, not shading.
//!
//! The give-away for a displacement map that is wired up everywhere except the
//! vertex stage is that the picture changes not at all — the silhouette is the
//! test, because that is the one thing normal mapping cannot fake.
//!
//! Skipped when no GPU adapter is available.

use std::sync::Arc;

use threers::{
    AmbientLight, Color, HeadlessRenderer, Material, Mesh, Object3D, PerspectiveCamera, Scene,
    SphereGeometry, StandardMaterial, Texture, TextureFormat, Vector3,
};

const W: u32 = 240;
const H: u32 = 240;

fn renderer() -> Option<HeadlessRenderer> {
    HeadlessRenderer::builder()
        .size(W, H)
        .color_format(wgpu::TextureFormat::Rgba8Unorm)
        .build()
        .ok()
}

/// A height map that is white over the northern half and black over the
/// southern — a hemisphere-wide step, so the silhouette change is unmissable.
fn north_step() -> Arc<Texture> {
    let (w, h) = (64u32, 32u32);
    let mut data = vec![255u8; (w * h * 4) as usize];
    for y in 0..h {
        for x in 0..w {
            let o = ((y * w + x) * 4) as usize;
            // Row 0 is the north pole, and `flip_y` is on by default, so the
            // white half lands on the north.
            let v = if y < h / 2 { 255 } else { 0 };
            data[o..o + 3].copy_from_slice(&[v, v, v]);
        }
    }
    Arc::new(Texture::new(w, h, TextureFormat::Rgba8Unorm, data))
}

fn render(r: &mut HeadlessRenderer, map: Option<Arc<Texture>>, scale: f32) -> Vec<u8> {
    let mut m = StandardMaterial::new(Color::from_hex(0xcc4422));
    m.displacement_map = map;
    m.displacement_scale = scale;
    let mut scene = Scene::new();
    scene.background = Color::BLACK;
    scene.add_light(AmbientLight::new(Color::WHITE, 3.0));
    scene.add(Object3D::mesh(Mesh::new(
        SphereGeometry::new(1.0, 96, 48),
        Material::Standard(m),
    )));
    let mut c = PerspectiveCamera::new(50.0, 1.0, 0.1, 100.0);
    c.position = Vector3::new(0.0, 0.0, 4.0);
    c.look_at(Vector3::ZERO);
    r.render_to_rgba(&mut scene, &c)
}

/// How many pixels in a column are the material's colour.
fn column_coverage(img: &[u8], x: u32) -> u32 {
    (0..H)
        .filter(|y| {
            let o = ((y * W + x) * 4) as usize;
            img[o] > 40
        })
        .count() as u32
}

#[test]
fn a_displacement_map_changes_the_silhouette() {
    let Some(mut r) = renderer() else { return };
    // Both maps live for the whole test: the renderer's texture cache is keyed
    // by the Arc's address, so a dropped one can be aliased by the next.
    let map = north_step();

    let flat = render(&mut r, None, 0.0);
    let bumped = render(&mut r, Some(map.clone()), 0.35);

    let mid = W / 2;
    let flat_h = column_coverage(&flat, mid);
    let bumped_h = column_coverage(&bumped, mid);
    assert!(flat_h > 50, "the sphere is not being drawn: {flat_h}");
    assert!(
        bumped_h > flat_h + 4,
        "the silhouette did not grow — the height map never reached the vertex \
         stage: flat {flat_h}px vs displaced {bumped_h}px"
    );
}

#[test]
fn displacement_follows_the_map_not_the_whole_mesh() {
    let Some(mut r) = renderer() else { return };
    let map = north_step();
    let flat = render(&mut r, None, 0.0);
    let bumped = render(&mut r, Some(map.clone()), 0.35);

    // The map is white north, black south, so only the top of the silhouette
    // should move. A uniform swell would mean the sample is being ignored and
    // some constant applied instead.
    let top_grew = |img: &[u8]| {
        (0..H).find(|&y| {
            let o = ((y * W + W / 2) * 4) as usize;
            img[o] > 40
        })
    };
    let bottom = |img: &[u8]| {
        (0..H).rev().find(|&y| {
            let o = ((y * W + W / 2) * 4) as usize;
            img[o] > 40
        })
    };
    let (ft, bt) = (top_grew(&flat).unwrap(), top_grew(&bumped).unwrap());
    let (fb, bb) = (bottom(&flat).unwrap(), bottom(&bumped).unwrap());
    assert!(bt + 3 < ft, "the north edge should rise: {ft} → {bt}");
    assert!(
        bb.abs_diff(fb) <= 2,
        "the south edge should stay put: {fb} → {bb}"
    );
}

#[test]
fn scale_zero_and_no_map_render_identically() {
    let Some(mut r) = renderer() else { return };
    let map = north_step();
    let none = render(&mut r, None, 0.0);
    let zero = render(&mut r, Some(map.clone()), 0.0);
    let differing = none
        .chunks_exact(4)
        .zip(zero.chunks_exact(4))
        .filter(|(a, b)| a[0].abs_diff(b[0]) > 2)
        .count();
    assert_eq!(differing, 0, "a zero scale should be a no-op");
}
